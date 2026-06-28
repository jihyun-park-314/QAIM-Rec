"""Stage2: F7 hybrid training skeleton.

Architecture (plan.md v0.4.5 §3 M3/M4):
  Frozen F6a SASRec + trainable text_encoder (bge-base) + trainable Projector
  Loss: L = L_retrieval + α · L_align
    L_retrieval = BPR(pos_logits, neg_logits)   (from SASRec.forward)
    L_align = 1 - cos_sim(prefix[:,0,:], item_emb(pos_target).detach())

  α grid (plan.md §7 #9): {0.01, 0.1, 0.5} — start low to avoid alignment
  swamping retrieval signal. Both raw loss sizes + grad norm logged per step.

On-the-fly re-encoding (plan.md §7 #11):
  h_query and h_memory re-encoded every step via current text_encoder
  (no cached vectors) because text_encoder is being fine-tuned.

Usage — CPU smoke (20-user mock):
    docker exec -e PYTHONPATH=/qaim-rec qaim-rec python3 \\
        src/training/train_hybrid.py --smoke

Usage — full training (after bank build, GPU):
    docker exec -e PYTHONPATH=/qaim-rec qaim-rec python3 \\
        src/training/train_hybrid.py \\
        --bank_jsonl data/memory_full/memory_full.jsonl \\
        --checkpoint checkpoints/Books/sasrec_pretrain.pt \\
        --category Books --device cuda:0 --alpha 0.1 --num_epochs 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.sasrec import SASRec
from src.models.projector import IntentProjector
from src.models.dataloader import load_data, WarpSampler, random_neq

# BGE query prefix (matches embed.py convention)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Text encoder wrapper (gradient-enabled for Stage2)

class TextEncoder(nn.Module):
    """Thin wrapper around bge-base transformer for Stage2 fine-tuning.

    Unlike EmbeddingModel (inference, numpy output), this module keeps
    grad enabled so the transformer weights update with the projector.
    """

    def __init__(self, model_id: str = "BAAI/bge-base-en-v1.5", device: str = "cpu") -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self._model = AutoModel.from_pretrained(model_id, local_files_only=True)
        self.device = device
        self._model.to(device)

    @property
    def dim(self) -> int:
        return self._model.config.hidden_size

    def param_groups(self, lr_encoder: float) -> list[dict]:
        """Single param group for asymmetric-LR optimizer (LLaVA convention).

        Projector gets its own group externally with a higher LR.
        """
        return [{"params": self._model.parameters(), "lr": lr_encoder}]

    def encode(self, texts: list[str], is_query: bool = False) -> torch.Tensor:
        """Encode texts → [N, d_text] float32 tensor (gradient-enabled).

        Returns L2-normalized vectors, same convention as EmbeddingModel.
        """
        if is_query:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]
        enc = self._tok(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        out = self._model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(vecs, p=2, dim=-1)  # [N, d_text]


# ---------------------------------------------------------------------------
# Stage1 encoder loader


def load_stage1_weights(ckpt_path: str, text_encoder: TextEncoder) -> None:
    """Load Stage1 fine-tuned BGE weights into Stage2 TextEncoder.

    Stage1 TextEncoderWithHead stores encoder as _enc.* keys.
    Stage2 TextEncoder stores it as _model.* — strip prefix and load directly
    into text_encoder._model (the AutoModel).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    stage1_sd = ckpt["model_state_dict"]
    enc_sd = {k[len("_enc."):]: v
              for k, v in stage1_sd.items() if k.startswith("_enc.")}
    missing, unexpected = text_encoder._model.load_state_dict(enc_sd, strict=True)
    if missing or unexpected:
        print(f"[stage1] WARNING missing={missing} unexpected={unexpected}")
    print(f"[stage1] encoder loaded from {ckpt_path}  ({len(enc_sd)} keys)")


# ---------------------------------------------------------------------------
# Bank mock builder

def _load_mock_from_jsonl(jsonl_path: str, n: int = 5) -> list[dict]:
    """Load first n users from bank JSONL as mock records.

    Supports two formats:
      (a) per-user with cluster_summaries (memory_full format)
      (b) per-memory f3_bank format with intent_description (one record per memory)

    Returns list of {user_id, k_personal, source_texts (flat list)}.
    """
    user_map: dict[str, dict] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = str(rec.get("user_id", ""))

            if "cluster_summaries" in rec:
                # Format (a): per-user record
                texts = []
                for cs in rec.get("cluster_summaries", []):
                    texts.extend(cs.get("source_texts", []))
                if texts and uid not in user_map:
                    user_map[uid] = {
                        "user_id": uid,
                        "k_personal": rec.get("k_personal", 1),
                        "source_texts": texts,
                    }
            elif "intent_description" in rec:
                # Format (b): per-memory f3_bank record
                txt = rec["intent_description"]
                if uid not in user_map:
                    user_map[uid] = {
                        "user_id": uid,
                        "k_personal": int(rec.get("meta", {}).get("k_personal", 1))
                        if isinstance(rec.get("meta"), dict) else 1,
                        "source_texts": [],
                    }
                user_map[uid]["source_texts"].append(txt)

            if len(user_map) >= n:
                break

    return list(user_map.values())[:n]


# ---------------------------------------------------------------------------
# Controllability measurement


def compute_controllability_jaccard(
    model: SASRec,
    text_encoder: TextEncoder,
    projector: "IntentProjector",
    query_texts: list[str],
    correct_texts: list[str],
    wrong_texts: list[str],
    seq_np: np.ndarray,
    k: int = 10,
) -> dict:
    """Measure prefix steering strength via top-k Jaccard.

    Returns:
        J_correct_wrong:   Jaccard(correct prefix top-k, wrong prefix top-k)
                           Lower → prefix is steering output meaningfully.
        J_correct_vanilla: Jaccard(correct prefix top-k, default_intent top-k)
                           Lower → prefix actually changes output vs no prefix.

    Both are averaged across the batch.
    """
    model.eval()
    B = seq_np.shape[0]

    with torch.no_grad():
        h_q = text_encoder.encode(query_texts, is_query=True)
        h_correct = text_encoder.encode(correct_texts, is_query=False)
        h_wrong = text_encoder.encode(wrong_texts, is_query=False)

        topk_per_cond: dict[str, list[set]] = {}
        for cond, h_mem in [("correct", h_correct), ("wrong", h_wrong)]:
            pfx = projector(h_q, h_mem)  # [B, 1, d_sasrec]
            log_feats, _ = model.log2feats(seq_np, prefix_embeds=pfx)
            last = log_feats[:, -1, :]  # [B, d]
            scores = model.item_emb.weight[1:] @ last.T  # [N_items, B]
            topk = scores.topk(k, dim=0).indices + 1  # [k, B], 1-indexed
            topk_per_cond[cond] = [set(topk[:, b].tolist()) for b in range(B)]

        # vanilla: default_intent prefix
        vanilla_pfx = model.default_intent.unsqueeze(0).expand(B, 1, -1)
        log_feats_v, _ = model.log2feats(seq_np, prefix_embeds=vanilla_pfx)
        last_v = log_feats_v[:, -1, :]
        scores_v = model.item_emb.weight[1:] @ last_v.T
        topk_v = scores_v.topk(k, dim=0).indices + 1
        topk_per_cond["vanilla"] = [set(topk_v[:, b].tolist()) for b in range(B)]

    def mean_jaccard(sets_a: list[set], sets_b: list[set]) -> float:
        jaccards = [
            len(a & b) / len(a | b) if a | b else 0.0
            for a, b in zip(sets_a, sets_b)
        ]
        return round(sum(jaccards) / len(jaccards), 4)

    model.train()
    return {
        "J_correct_wrong": mean_jaccard(
            topk_per_cond["correct"], topk_per_cond["wrong"]
        ),
        "J_correct_vanilla": mean_jaccard(
            topk_per_cond["correct"], topk_per_cond["vanilla"]
        ),
    }


# ---------------------------------------------------------------------------
# BPR loss

def bpr_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    """BPR loss over non-padding positions (pos != 0)."""
    mask = (pos_logits != 0).float()
    loss = -F.logsigmoid(pos_logits - neg_logits)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Alignment loss

def align_loss(
    prefix_embeds: torch.Tensor,   # [B, 1, d_sasrec]
    item_emb: nn.Embedding,        # SASRec item embedding (frozen)
    pos_target_ids: torch.Tensor,  # [B] — last positive item per user
) -> torch.Tensor:
    """1 - cos_sim(prefix[:,0,:], item_emb(pos_target).detach()).

    Pushes the prefix toward the target item in SASRec space.
    item_emb is detached so gradients only flow to prefix / projector.
    """
    prefix_vec = prefix_embeds[:, 0, :]                      # [B, d_sasrec]
    target_vec = item_emb(pos_target_ids).detach()            # [B, d_sasrec]
    cos = F.cosine_similarity(prefix_vec, target_vec, dim=-1) # [B]
    return (1.0 - cos).mean()


# ---------------------------------------------------------------------------
# One training step

def train_step(
    model: SASRec,
    text_encoder: TextEncoder,
    projector: IntentProjector,
    optimizer: torch.optim.Optimizer,
    query_texts: list[str],         # [B] query stand-ins
    memory_texts: list[str],        # [B] memory source texts
    wrong_memory_texts: list[str],  # [B] swapped memory source texts
    seq_np: np.ndarray,             # [B, maxlen]
    pos_np: np.ndarray,             # [B, maxlen]
    neg_np: np.ndarray,             # [B, maxlen]
    alpha: float,
    device: str,
    use_contrastive: bool = False,
) -> dict:
    """Single Stage2 training step. Returns {L, L_retrieval, L_align, grad_norm}."""
    optimizer.zero_grad()

    # On-the-fly encode (§7 #11) — text_encoder is in training mode
    h_query = text_encoder.encode(query_texts, is_query=True)    # [B, d_text]
    h_memory = text_encoder.encode(memory_texts, is_query=False)  # [B, d_text]

    # Projector → prefix (correct)
    prefix_embeds = projector(h_query, h_memory)  # [B, 1, d_sasrec]

    if use_contrastive:
        h_wrong = text_encoder.encode(wrong_memory_texts, is_query=False)  # [B, d_text]
        prefix_wrong = projector(h_query, h_wrong)  # [B, 1, d_sasrec]

    # Frozen SASRec forward
    B = seq_np.shape[0]
    uid_dummy = np.zeros(B, dtype=np.int32)
    pos_logits, neg_logits = model(
        uid_dummy, seq_np, pos_np, neg_np,
        prefix_embeds=prefix_embeds,
    )

    # L_retrieval: BPR
    pos_t = torch.LongTensor(pos_np).to(device)
    neg_t = torch.LongTensor(neg_np).to(device)
    l_retrieval = bpr_loss(pos_logits, neg_logits)

    # L_align: cosine distance between prefix and last positive item
    last_pos = torch.LongTensor(pos_np[:, -1]).to(device)  # [B]
    l_align = align_loss(prefix_embeds, model.item_emb, last_pos)

    # SPEC loss: L = L_retrieval + α·L_align
    loss = l_retrieval + alpha * l_align
    metrics_extra: dict = {}
    if use_contrastive:
        # Ablation: correct prefix closer to target than wrong prefix (margin=0.1)
        target_vec = model.item_emb(last_pos).detach()  # [B, d_sasrec]
        sim_correct = F.cosine_similarity(prefix_embeds[:, 0, :], target_vec)  # [B]
        sim_wrong   = F.cosine_similarity(prefix_wrong[:, 0, :],  target_vec)  # [B]
        l_contrastive = F.relu(0.1 - sim_correct + sim_wrong).mean()
        loss = loss + alpha * l_contrastive
        metrics_extra["L_contrastive"] = round(l_contrastive.item(), 6)

    loss.backward()

    grad_norm = sum(
        p.grad.norm().item() ** 2
        for p in list(text_encoder.parameters()) + list(projector.parameters())
        if p.grad is not None
    ) ** 0.5

    optimizer.step()

    return {
        "L": round(loss.item(), 6),
        "L_retrieval": round(l_retrieval.item(), 6),
        "L_align": round(l_align.item(), 6),
        **metrics_extra,
        "grad_norm": round(grad_norm, 6),
    }


# ---------------------------------------------------------------------------
# Smoke test

def run_smoke(args: SimpleNamespace) -> None:
    """CPU smoke: mock bank → 1 step × 3 α values.

    Verifies: (a) prefix forward with correct P offset,
              (b) 4 conditions produce different outputs,
              (c) loss/backprop works,
              (d) α-wise loss size logging.
    """
    print("[smoke] Loading mock bank ...")
    mock_bank_path = getattr(args, "mock_bank", None) or "data/memory/Books/f3_bank.jsonl"
    mock_users = _load_mock_from_jsonl(mock_bank_path, n=5)
    assert len(mock_users) >= 2, "Need at least 2 users for wrong-condition swap"
    print(f"[smoke] {len(mock_users)} mock users loaded")

    print("[smoke] Loading text encoder (bge-base, CPU) ...")
    text_enc = TextEncoder(device="cpu")
    if getattr(args, "stage1_ckpt", None):
        load_stage1_weights(args.stage1_ckpt, text_enc)
    else:
        print("[smoke] stage1_ckpt not set — using raw pretrained BGE (baseline mode)")
    print(f"[smoke] text_encoder.dim = {text_enc.dim}")

    # Load F6a checkpoint
    print("[smoke] Loading F6a checkpoint ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved = ckpt["args"]
    if isinstance(saved, dict):
        saved = SimpleNamespace(**saved)
    eval_args = SimpleNamespace(
        maxlen=saved.maxlen, hidden_units=saved.hidden_units,
        num_blocks=saved.num_blocks, num_heads=saved.num_heads,
        dropout_rate=saved.dropout_rate, norm_first=saved.norm_first,
        device="cpu",
    )
    dataset = load_data(args.category, args.data_dir)
    model = SASRec(dataset["usernum"], dataset["itemnum"], eval_args)
    model.load_state_dict(ckpt["model_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)  # frozen
    model.train()  # train mode for dropout consistency
    print("[smoke] SASRec frozen.")

    d_sasrec = saved.hidden_units
    d_text = text_enc.dim
    projector = IntentProjector(d_text=d_text, d_sasrec=d_sasrec)
    print(f"[smoke] Projector: [{2*d_text}→256→{d_sasrec}]  params={sum(p.numel() for p in projector.parameters()):,}")

    # Build mock batch (B=2)
    B = 2
    query_texts = [mock_users[i]["source_texts"][0] for i in range(B)]
    memory_texts = [mock_users[i]["source_texts"][0] for i in range(B)]
    wrong_texts = [mock_users[(i + 1) % B]["source_texts"][0] for i in range(B)]

    # Fake sequences — random items within valid range
    rng = random.Random(42)
    itemnum = dataset["itemnum"]
    maxlen = saved.maxlen
    seq_np = np.zeros((B, maxlen), dtype=np.int32)
    pos_np = np.zeros((B, maxlen), dtype=np.int32)
    neg_np = np.zeros((B, maxlen), dtype=np.int32)
    for b in range(B):
        # Fill last 5 positions with random items
        for j in range(maxlen - 5, maxlen):
            item = rng.randint(1, itemnum)
            seq_np[b, j] = item
            pos_np[b, j] = rng.randint(1, itemnum)
            neg_np[b, j] = rng.randint(1, itemnum)

    # ── (a) Prefix forward with P offset ────────────────────────────────────
    print("\n[smoke] (a) Testing prefix P offset ...")
    with torch.no_grad():
        h_q = text_enc.encode(query_texts, is_query=True)
        h_m = text_enc.encode(memory_texts, is_query=False)
    prefix = projector(h_q, h_m)  # [B, 1, d_sasrec]
    assert prefix.shape == (B, 1, d_sasrec), f"Wrong prefix shape: {prefix.shape}"
    with torch.no_grad():
        log_feats, P = model.log2feats(seq_np, prefix_embeds=prefix)
    assert P == 1, f"Expected P=1, got P={P}"
    assert log_feats.shape[1] == maxlen + 1, \
        f"Expected seq_len={maxlen+1}, got {log_feats.shape[1]}"
    print(f"  prefix.shape={prefix.shape}  P={P}  log_feats.shape={log_feats.shape}  OK")

    # ── (b) 4 conditions produce different outputs ───────────────────────────
    print("\n[smoke] (b) Testing 4 conditions produce different outputs ...")
    from src.eval.conditions import make_prefix

    condition_outputs = {}
    with torch.no_grad():
        for cond in ("vanilla", "correct", "wrong", "default"):
            pfx = make_prefix(
                condition=cond,
                projector=projector,
                default_intent=model.default_intent,
                h_query=h_q,
                h_memory_correct=h_m,
                h_memory_wrong=text_enc.encode(wrong_texts, is_query=False),
                device="cpu",
            )
            lf, _ = model.log2feats(seq_np, prefix_embeds=pfx)
            last = lf[0, -1, :].clone()
            condition_outputs[cond] = last

    # All 4 must differ
    cond_list = list(condition_outputs.keys())
    all_different = all(
        not torch.allclose(condition_outputs[a], condition_outputs[b], atol=1e-5)
        for i, a in enumerate(cond_list) for b in cond_list[i+1:]
    )
    for c, v in condition_outputs.items():
        print(f"  {c:8s}: final_feat norm={v.norm().item():.4f}")
    assert all_different, "Some conditions produced identical outputs!"
    print("  All 4 conditions differ: OK")

    # ── (c) + (d) Loss/backprop and α logging ───────────────────────────────
    print("\n[smoke] (c)+(d) Loss/backprop + α logging ...")
    lr_encoder = getattr(args, "lr_encoder", 2e-6)
    lr_projector = getattr(args, "lr_projector", 1e-3)
    print(f"  LLaVA asymmetric LR — encoder={lr_encoder:.1e}  projector={lr_projector:.1e}")
    alpha_grid = [0.01, 0.1, 0.5]
    for alpha in alpha_grid:
        proj_fresh = IntentProjector(d_text=d_text, d_sasrec=d_sasrec)
        # LLaVA convention: encoder frozen/low-LR, projector high-LR
        opt = torch.optim.Adam(
            text_enc.param_groups(lr_encoder) + [
                {"params": proj_fresh.parameters(), "lr": lr_projector}
            ]
        )
        metrics = train_step(
            model=model,
            text_encoder=text_enc,
            projector=proj_fresh,
            optimizer=opt,
            query_texts=query_texts,
            memory_texts=memory_texts,
            wrong_memory_texts=wrong_texts,
            seq_np=seq_np,
            pos_np=pos_np,
            neg_np=neg_np,
            alpha=alpha,
            device="cpu",
            use_contrastive=False,
        )
        print(f"  α={alpha:<4}  L={metrics['L']:.5f}  "
              f"L_ret={metrics['L_retrieval']:.5f}  "
              f"L_align={metrics['L_align']:.5f}  "
              f"grad_norm={metrics['grad_norm']:.5f}")
        assert metrics["L"] > 0, "Loss should be positive"
        assert metrics["grad_norm"] > 0, "Grad norm should be non-zero"
    print("  Loss/backprop OK, α scaling reflected in total L.")

    # ── Controllability initial measurement ─────────────────────────────────
    print("\n[smoke] Controllability initial values (untrained projector) ...")
    proj_ctrl = IntentProjector(d_text=d_text, d_sasrec=d_sasrec)
    ctrl = compute_controllability_jaccard(
        model=model,
        text_encoder=text_enc,
        projector=proj_ctrl,
        query_texts=query_texts,
        correct_texts=memory_texts,
        wrong_texts=wrong_texts,
        seq_np=seq_np,
        k=10,
    )
    print(f"  J(correct,wrong)   = {ctrl['J_correct_wrong']:.4f}"
          f"  (lower after training → prefix steers items)")
    print(f"  J(correct,vanilla) = {ctrl['J_correct_vanilla']:.4f}"
          f"  (lower → prefix changes output vs no prefix)")
    print("  [baseline] random projector → values are noise-driven, not preference-driven.")
    print("             After training: J(c,w) should stay low but reflect real preference gap;"
          " J(c,v) should drop if prefix meaningfully steers vs no-prefix.")

    print("\n[smoke] ALL CHECKS PASSED (CPU, mock data — not real results)")


# ---------------------------------------------------------------------------
# Full training helpers


def _load_bank_full(bank_jsonl: str) -> dict[str, list[dict]]:
    """Load bank JSONL → {uid_str: [{mid, text}, ...]}

    Supports f3_bank format (per-memory, intent_description) and
    memory_full format (per-user, cluster_summaries).
    """
    user_memories: dict[str, list[dict]] = {}
    with open(bank_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = str(rec.get("user_id", ""))
            if not uid:
                continue
            if "intent_description" in rec:
                mid = str(rec.get("memory_id", ""))
                user_memories.setdefault(uid, []).append(
                    {"mid": mid, "text": rec["intent_description"]}
                )
            elif "cluster_summaries" in rec:
                for cs in rec.get("cluster_summaries", []):
                    mid = str(cs.get("cluster_id", ""))
                    for txt in cs.get("source_texts", []):
                        user_memories.setdefault(uid, []).append({"mid": mid, "text": txt})
    return user_memories


def _load_pseudo_queries(
    pairs_jsonl: str,
    mid_to_uid: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Load align_pairs.jsonl → {uid_str: [query_text, ...]}

    align_pairs rows may omit user_id (only positive_memory_id present).
    Pass mid_to_uid (built from the bank) to resolve uid from positive_memory_id.
    """
    user_queries: dict[str, list[str]] = {}
    with open(pairs_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = str(rec.get("user_id", ""))
            if not uid and mid_to_uid:
                uid = mid_to_uid.get(rec.get("positive_memory_id", ""), "")
            q = rec.get("query", "")
            if uid and q:
                user_queries.setdefault(uid, []).append(q)
    return user_queries


def _route_batch(
    uid_arr: np.ndarray,
    user_queries: dict,
    user_memories: dict,
    text_enc: TextEncoder,
    rng: random.Random,
    query_seed_offset: int = 0,
) -> tuple[list[str], list[str], list[str], list[int]]:
    """Route each uid to its top-1 memory via cosine sim.

    Caller is responsible for no_grad + eval mode.
    Returns (query_texts, correct_mem_texts, wrong_mem_texts, valid_idx).
    valid_idx: positions in uid_arr that were successfully routed.
    wrong_mem_texts: cyclic-shifted correct_mem_texts for ablation/L_align.
    """
    infos: list[tuple[int, str, list[str]]] = []
    for i, uid in enumerate(uid_arr):
        uid_s = str(int(uid))
        queries = user_queries.get(uid_s)
        mems = user_memories.get(uid_s)
        if not queries or not mems:
            continue
        q_idx = (rng.randint(0, len(queries) - 1) + query_seed_offset) % len(queries)
        infos.append((i, queries[q_idx], [m["text"] for m in mems]))

    if not infos:
        return [], [], [], []

    # Batch-encode all queries at once to avoid per-user round trips
    h_queries = text_enc.encode([info[1] for info in infos], is_query=True)  # [n, d]

    selected_queries, selected_mems, valid_idx = [], [], []
    for j, (orig_i, q_text, mem_texts) in enumerate(infos):
        if len(mem_texts) == 1:
            best_mem = mem_texts[0]
        else:
            h_mems = text_enc.encode(mem_texts, is_query=False)  # [k, d]
            sims = (h_mems @ h_queries[j : j + 1].T).squeeze(-1)  # [k]
            best_mem = mem_texts[sims.argmax().item()]
        selected_queries.append(q_text)
        selected_mems.append(best_mem)
        valid_idx.append(orig_i)

    n = len(selected_queries)
    wrong_mems = [selected_mems[(j + 1) % n] for j in range(n)]
    return selected_queries, selected_mems, wrong_mems, valid_idx


@torch.no_grad()
def _check_baseline_routing(
    text_enc: TextEncoder,
    user_memories: dict,
    pairs_jsonl: str,
    n: int = 200,
    seed: int = 42,
    mid_to_uid: dict[str, str] | None = None,
) -> None:
    """Verify pre-training routing accuracy using align_pairs GT.

    Expects ~0.94 for raw-BGE baseline, ~0.96 for Stage1 encoder.
    Non-fatal: prints a warning if pairs_jsonl not found or no matchable pairs.
    """
    rng = random.Random(seed)
    all_pairs: list[dict] = []
    try:
        with open(pairs_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec.get("positive_memory_id") or not rec.get("query"):
                    continue
                if not rec.get("user_id") and mid_to_uid:
                    rec = dict(rec, user_id=mid_to_uid.get(rec["positive_memory_id"], ""))
                if rec.get("user_id"):
                    all_pairs.append(rec)
    except FileNotFoundError:
        print(f"[baseline] {pairs_jsonl} not found — skipping routing accuracy check")
        return

    sample = rng.sample(all_pairs, min(n, len(all_pairs)))
    was_training = text_enc.training
    text_enc.eval()

    hits = total = 0
    for rec in sample:
        uid_s = str(rec["user_id"])
        pos_mid = rec["positive_memory_id"]
        mems = user_memories.get(uid_s, [])
        if not mems:
            continue
        if pos_mid not in {m["mid"] for m in mems}:
            continue
        if len(mems) == 1:
            hits += 1
            total += 1
            continue
        h_q = text_enc.encode([rec["query"]], is_query=True)
        h_mems = text_enc.encode([m["text"] for m in mems], is_query=False)
        sims = (h_mems @ h_q.T).squeeze(-1)
        if mems[sims.argmax().item()]["mid"] == pos_mid:
            hits += 1
        total += 1

    if was_training:
        text_enc.train()
    if total > 0:
        acc = hits / total
        note = "~0.94 expected for raw-BGE, ~0.96 for Stage1"
        print(f"[baseline] pre-training routing accuracy (n={total}): {acc:.4f}  ({note})")
    else:
        print("[baseline] routing check: no matchable pairs (positive_memory_id not in bank?)")


@torch.no_grad()
def _eval_val_loss(
    model: SASRec,
    text_enc: TextEncoder,
    projector: IntentProjector,
    dataset: dict,
    user_queries: dict,
    user_memories: dict,
    alpha: float,
    device: str,
    maxlen: int,
    batch_size: int,
    rng: random.Random,
    n_val: int = 512,
) -> float:
    """BPR loss on val-split items. Used for best-checkpoint selection and early stopping."""
    model.eval(); text_enc.eval(); projector.eval()

    val_uids = [
        uid for uid in dataset["user_valid"]
        if str(uid) in user_queries
        and str(uid) in user_memories
        and len(dataset["user_train"].get(uid, [])) >= 1
    ]
    if not val_uids:
        model.train(); text_enc.train(); projector.train()
        return float("inf")

    sample_uids = rng.sample(val_uids, min(n_val, len(val_uids)))
    itemnum = dataset["itemnum"]
    total_l_ret = 0.0
    n_batches = 0

    for b_start in range(0, len(sample_uids), batch_size):
        batch_uids = sample_uids[b_start : b_start + batch_size]
        seqs, poss, negs, uid_list = [], [], [], []

        for uid in batch_uids:
            train_items = dataset["user_train"].get(uid, [])
            val_items = dataset["user_valid"].get(uid, [])
            if not train_items or not val_items:
                continue
            seq = np.zeros(maxlen, dtype=np.int32)
            pos = np.zeros(maxlen, dtype=np.int32)
            neg = np.zeros(maxlen, dtype=np.int32)
            items = train_items[-maxlen:]
            for j, it in enumerate(items):
                seq[maxlen - len(items) + j] = it
            pos[maxlen - 1] = val_items[0]
            neg[maxlen - 1] = random_neq(1, itemnum + 1, set(train_items + [val_items[0]]))
            seqs.append(seq); poss.append(pos); negs.append(neg)
            uid_list.append(uid)

        if not seqs:
            continue

        uid_np = np.array(uid_list, dtype=np.int32)
        seq_np = np.stack(seqs)
        pos_np = np.stack(poss)
        neg_np = np.stack(negs)

        q_texts, m_texts, _, valid_idx = _route_batch(
            uid_np, user_queries, user_memories, text_enc, rng
        )
        if not valid_idx:
            continue

        seq_np = seq_np[valid_idx]
        pos_np = pos_np[valid_idx]
        neg_np = neg_np[valid_idx]

        h_q = text_enc.encode(q_texts, is_query=True)
        h_m = text_enc.encode(m_texts, is_query=False)
        pfx = projector(h_q, h_m)

        uid_dummy = np.zeros(len(valid_idx), dtype=np.int32)
        pos_logits, neg_logits = model(uid_dummy, seq_np, pos_np, neg_np, prefix_embeds=pfx)
        total_l_ret += bpr_loss(pos_logits, neg_logits).item()
        n_batches += 1

    model.train(); text_enc.train(); projector.train()
    return total_l_ret / max(n_batches, 1)


def _sample_ctrl_users(
    user_queries: dict,
    user_memories: dict,
    n: int,
    seed: int,
) -> list[str]:
    """Pick n users with ≥1 query and ≥1 memory for per-epoch controllability monitoring."""
    rng = random.Random(seed)
    valid = [
        u for u in set(user_queries) & set(user_memories)
        if user_queries[u] and user_memories[u]
    ]
    return rng.sample(valid, min(n, len(valid)))


@torch.no_grad()
def _eval_controllability(
    model: SASRec,
    text_enc: TextEncoder,
    projector: IntentProjector,
    ctrl_uid_strs: list[str],
    user_queries: dict,
    user_memories: dict,
    dataset: dict,
    device: str,
    maxlen: int,
    k: int = 10,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Compute J(c,w), J(c,v), J(c,cs2) on fixed ctrl users.

    J_cw:  Jaccard(correct, wrong)         — lower → prefix steers by user identity
    J_cv:  Jaccard(correct, vanilla)       — lower → prefix changes output vs no-prefix
    J_cs2: Jaccard(correct, correct-seed2) — upper noise bound (same user, diff query)
    Training signal: J_cw < J_cs2 means prefix distinguishes users more than query noise.
    """
    if rng is None:
        rng = random.Random(42)
    model.eval(); text_enc.eval(); projector.eval()

    q_texts, q2_texts, c_mems, w_mems, seqs = [], [], [], [], []
    for uid_s in ctrl_uid_strs:
        queries = user_queries.get(uid_s, [])
        mems = user_memories.get(uid_s, [])
        if not queries or not mems:
            continue
        q_texts.append(queries[0])
        q2_texts.append(queries[1 % len(queries)])  # same as q1 if user has only 1 query
        c_mems.append(mems[0]["text"])
        others = [u for u in ctrl_uid_strs if u != uid_s and user_memories.get(u)]
        w_uid = rng.choice(others) if others else uid_s
        w_mems.append(user_memories[w_uid][0]["text"])
        # Use real training sequence — all-zero sequences cause item_target_norm=0,
        # which zeroes out every prefix via the scale fix, making all conditions identical.
        train_items = dataset["user_train"].get(int(uid_s), [])
        seq = np.zeros(maxlen, dtype=np.int32)
        chunk = train_items[-maxlen:]
        offset = maxlen - len(chunk)
        for j, it in enumerate(chunk):
            seq[offset + j] = it
        seqs.append(seq)

    if not q_texts:
        model.train(); text_enc.train(); projector.train()
        return {"J_cw": 0.0, "J_cv": 0.0, "J_cs2": 0.0}

    seq_np = np.stack(seqs)
    B = len(q_texts)

    h_q = text_enc.encode(q_texts, is_query=True)
    h_q2 = text_enc.encode(q2_texts, is_query=True)
    h_c = text_enc.encode(c_mems, is_query=False)
    h_w = text_enc.encode(w_mems, is_query=False)

    pfx_c = projector(h_q, h_c)
    pfx_w = projector(h_q, h_w)
    pfx_cs2 = projector(h_q2, h_c)
    pfx_vanilla = model.default_intent.unsqueeze(0).expand(B, 1, -1).to(device)

    def topk_sets(pfx: torch.Tensor) -> list[set]:
        lf, _ = model.log2feats(seq_np, prefix_embeds=pfx)
        last = lf[:, -1, :]  # [B, d] — prefix positions already skipped by P offset
        scores = model.item_emb.weight[1:] @ last.T  # [N_items, B]
        topk = scores.topk(k, dim=0).indices + 1  # [k, B], 1-indexed
        return [set(topk[:, b].tolist()) for b in range(B)]

    tk_c = topk_sets(pfx_c)
    tk_w = topk_sets(pfx_w)
    tk_cs2 = topk_sets(pfx_cs2)
    tk_v = topk_sets(pfx_vanilla)

    def mean_j(a_sets: list[set], b_sets: list[set]) -> float:
        return round(sum(
            len(a & b) / len(a | b) if (a | b) else 0.0
            for a, b in zip(a_sets, b_sets)
        ) / max(len(a_sets), 1), 4)

    model.train(); text_enc.train(); projector.train()
    return {
        "J_cw":  mean_j(tk_c, tk_w),
        "J_cv":  mean_j(tk_c, tk_v),
        "J_cs2": mean_j(tk_c, tk_cs2),
    }


# ---------------------------------------------------------------------------
# Per-epoch diagnostics


@torch.no_grad()
def _measure_prefix_scale(
    model: SASRec,
    text_enc: TextEncoder,
    projector: IntentProjector,
    ctrl_uid_strs: list[str],
    user_queries: dict,
    user_memories: dict,
    dataset: dict,
    device: str,
    maxlen: int,
) -> dict:
    """Verify scale fix is holding each epoch.

    item_target_norm should be ~1.72 (non-pad mean); if it drops to ~0.26 the
    padding-inclusive bug has regressed. share_est = 1/(1+L_nonpad_median).
    """
    model.eval(); text_enc.eval(); projector.eval()

    seqs, q_texts, m_texts = [], [], []
    for uid_s in ctrl_uid_strs:
        queries = user_queries.get(uid_s, [])
        mems = user_memories.get(uid_s, [])
        if not queries or not mems:
            continue
        q_texts.append(queries[0])
        m_texts.append(mems[0]["text"])
        train_items = dataset["user_train"].get(int(uid_s), [])
        seq = np.zeros(maxlen, dtype=np.int32)
        chunk = train_items[-maxlen:]
        offset = maxlen - len(chunk)
        for j, it in enumerate(chunk):
            seq[offset + j] = it
        seqs.append(seq)

    if not seqs:
        model.train(); text_enc.train(); projector.train()
        return {}

    seq_np = np.stack(seqs)
    seqs_t = model.item_emb(torch.LongTensor(seq_np).to(device))
    seqs_t = seqs_t * (model.item_emb.embedding_dim ** 0.5)
    poss = np.tile(np.arange(1, seq_np.shape[1] + 1), [seq_np.shape[0], 1])
    poss = poss * (seq_np != 0)
    seqs_t = seqs_t + model.pos_emb(torch.LongTensor(poss).to(device))

    item_norms = seqs_t.norm(dim=-1)  # [B, L]
    nonpad_mask = item_norms > 1e-6
    item_target_norm = (
        item_norms[nonpad_mask].mean().item() if nonpad_mask.any() else item_norms.mean().item()
    )
    L_nonpad_med = nonpad_mask.float().sum(dim=1).median().item()
    share_est = 100.0 / (1.0 + L_nonpad_med) if L_nonpad_med > 0 else 100.0

    n = min(len(seqs), len(q_texts))
    h_q = text_enc.encode(q_texts[:n], is_query=True)
    h_m = text_enc.encode(m_texts[:n], is_query=False)
    pfx = projector(h_q, h_m)
    pfx_raw_norm = pfx.norm(dim=-1).mean().item()

    model.train(); text_enc.train(); projector.train()
    return {
        "item_target_norm": round(item_target_norm, 4),
        "L_nonpad_med": round(L_nonpad_med, 1),
        "share_est_pct": round(share_est, 1),
        "pfx_raw_norm": round(pfx_raw_norm, 4),
    }


# ---------------------------------------------------------------------------
# Full training loop


def run_training(args: SimpleNamespace) -> None:
    """Stage2 full training: frozen SASRec + text_encoder + projector.

    L = L_retrieval (BPR, full-catalog next-item) + α · L_align (cos-distance prefix→item).
    LLaVA asymmetric LR: SASRec frozen / encoder args.lr_encoder / projector args.lr_projector.
    Works for both STEP2a (--stage1_ckpt provided) and STEP2b (raw-BGE baseline).
    """
    seed = getattr(args, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)

    device = args.device
    alpha = args.alpha
    lr_encoder = getattr(args, "lr_encoder", 2e-6)
    lr_projector = getattr(args, "lr_projector", 1e-3)
    batch_size = getattr(args, "batch_size", 64)
    patience = getattr(args, "patience", 3)

    # ── Frozen SASRec ───────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved = ckpt["args"]
    if isinstance(saved, dict):
        saved = SimpleNamespace(**saved)
    model_args = SimpleNamespace(
        maxlen=saved.maxlen, hidden_units=saved.hidden_units,
        num_blocks=saved.num_blocks, num_heads=saved.num_heads,
        dropout_rate=saved.dropout_rate, norm_first=saved.norm_first,
        device=device,
    )
    dataset = load_data(args.category, args.data_dir)
    model = SASRec(dataset["usernum"], dataset["itemnum"], model_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.train()
    maxlen = saved.maxlen
    d_sasrec = saved.hidden_units
    print(f"[model] SASRec frozen  maxlen={maxlen}  d={d_sasrec}  "
          f"users={dataset['usernum']}  items={dataset['itemnum']}")

    # ── Text encoder: STEP2a (stage1 weights) or STEP2b (raw BGE) ──────────
    text_enc = TextEncoder(model_id="BAAI/bge-base-en-v1.5", device=device)
    if getattr(args, "stage1_ckpt", None):
        load_stage1_weights(args.stage1_ckpt, text_enc)
        enc_tag = "STEP2a (stage1 encoder)"
    else:
        enc_tag = "STEP2b (raw-BGE baseline)"
    print(f"[model] encoder: {enc_tag}")
    text_enc.train()
    d_text = text_enc.dim

    # ── Projector + LLaVA asymmetric optimizer ──────────────────────────────
    projector = IntentProjector(d_text=d_text, d_sasrec=d_sasrec).to(device)
    if getattr(args, "proj_ckpt", None):
        _p = torch.load(args.proj_ckpt, map_location="cpu")
        _proj_state = _p.get("projector_state", _p)
        projector.load_state_dict(_proj_state)
        print(f"[projector] warm-start from {args.proj_ckpt}")
    optimizer = torch.optim.Adam(
        text_enc.param_groups(lr_encoder) + [
            {"params": projector.parameters(), "lr": lr_projector}
        ]
    )
    print(f"[opt] lr_enc={lr_encoder:.1e}  lr_proj={lr_projector:.1e}  alpha={alpha}")

    # ── Bank + pseudo-queries ───────────────────────────────────────────────
    pairs_jsonl = (
        getattr(args, "pairs_jsonl", None)
        or f"data/processed/{args.category}/align_pairs.jsonl"
    )
    print(f"[data] bank:    {args.bank_jsonl}")
    user_memories = _load_bank_full(args.bank_jsonl)
    print(f"[data] {len(user_memories)} users in bank")
    mid_to_uid: dict[str, str] = {
        m["mid"]: uid
        for uid, mems in user_memories.items()
        for m in mems
    }
    print(f"[data] queries: {pairs_jsonl}")
    user_queries = _load_pseudo_queries(pairs_jsonl, mid_to_uid=mid_to_uid)
    print(f"[data] {len(user_queries)} users with pseudo-queries")

    # Pre-training routing accuracy check (confirms ~0.94 for STEP2b, ~0.96 for STEP2a)
    _check_baseline_routing(text_enc, user_memories, pairs_jsonl, n=200, seed=seed,
                            mid_to_uid=mid_to_uid)

    # ── WarpSampler ─────────────────────────────────────────────────────────
    n_workers = getattr(args, "n_workers", 3)
    sampler = WarpSampler(
        dataset["user_train"], dataset["usernum"], dataset["itemnum"],
        batch_size=batch_size, maxlen=maxlen, n_workers=n_workers,
    )
    steps_per_epoch = (dataset["usernum"] + batch_size - 1) // batch_size
    print(f"[train] {steps_per_epoch} steps/epoch  "
          f"{steps_per_epoch * args.num_epochs} total steps  {args.num_epochs} epochs")

    # ── Fixed ctrl users for controllability monitoring ─────────────────────
    ctrl_n = getattr(args, "ctrl_users", 16)
    ctrl_uid_strs = _sample_ctrl_users(user_queries, user_memories, ctrl_n, seed)
    print(f"[ctrl] {len(ctrl_uid_strs)} users for controllability monitoring")

    # ── Output dir + checkpoint path ────────────────────────────────────────
    out_dir = Path(getattr(args, "output_dir", None) or f"checkpoints/{args.category}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_tag = "stage2_stage1enc" if getattr(args, "stage1_ckpt", None) else "stage2_rawbge"
    best_ckpt_path = out_dir / f"{ckpt_tag}_best.pt"

    # ── Encoder drift baseline — flat snapshot for true L2 distance ─────────
    enc_params0 = torch.cat([p.data.flatten() for p in text_enc._model.parameters()]).clone()
    enc_norm0 = enc_params0.norm().item()

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train(); text_enc.train(); projector.train()
        epoch_l_ret = epoch_l_align = epoch_l_contr = 0.0
        epoch_steps = routed_total = batch_total = 0

        for step in range(1, steps_per_epoch + 1):
            # Unpack WarpSampler zip
            u_it, s_it, p_it, n_it = sampler.next_batch()
            uid_arr = np.array(list(u_it), dtype=np.int32)
            seq_np = np.array(list(s_it), dtype=np.int32)
            pos_np = np.array(list(p_it), dtype=np.int32)
            neg_np = np.array(list(n_it), dtype=np.int32)

            batch_total += uid_arr.shape[0]

            # Routing: eval + no_grad → top-1 memory selection per user
            text_enc.eval()
            with torch.no_grad():
                q_texts, m_texts, w_texts, valid_idx = _route_batch(
                    uid_arr, user_queries, user_memories, text_enc, rng
                )
            text_enc.train()

            if not valid_idx:
                continue
            routed_total += len(valid_idx)

            # Subset batch to routed users; train_step re-encodes with grad
            seq_np = seq_np[valid_idx]
            pos_np = pos_np[valid_idx]
            neg_np = neg_np[valid_idx]

            metrics = train_step(
                model=model,
                text_encoder=text_enc,
                projector=projector,
                optimizer=optimizer,
                query_texts=q_texts,
                memory_texts=m_texts,
                wrong_memory_texts=w_texts,
                seq_np=seq_np,
                pos_np=pos_np,
                neg_np=neg_np,
                alpha=alpha,
                device=device,
                use_contrastive=args.use_contrastive,
            )

            epoch_l_ret += metrics["L_retrieval"]
            epoch_l_align += metrics["L_align"]
            epoch_l_contr += metrics.get("L_contrastive", 0.0)
            epoch_steps += 1

            if step % 200 == 0:
                print(f"  step {step}/{steps_per_epoch}  "
                      f"L_ret={metrics['L_retrieval']:.4f}  "
                      f"L_align={metrics['L_align']:.4f}  "
                      f"L_contr={metrics.get('L_contrastive', 0):.4f}  "
                      f"L={metrics['L']:.4f}  grad={metrics['grad_norm']:.4f}")

        avg_l_ret   = epoch_l_ret   / max(epoch_steps, 1)
        avg_l_align = epoch_l_align / max(epoch_steps, 1)
        avg_l_contr = epoch_l_contr / max(epoch_steps, 1)
        routing_cov = routed_total / max(batch_total, 1)

        val_loss = _eval_val_loss(
            model, text_enc, projector, dataset,
            user_queries, user_memories, alpha, device, maxlen, batch_size,
            rng=random.Random(seed + epoch),
        )
        ctrl = _eval_controllability(
            model, text_enc, projector, ctrl_uid_strs,
            user_queries, user_memories, dataset, device, maxlen,
            k=10, rng=random.Random(seed + epoch),
        )

        # J_cw < J_cs2 → prefix differentiates users more than query noise → signal
        gap = ctrl["J_cw"] - ctrl["J_cs2"]
        gap_label = "SIGNAL" if gap < 0 else "noise"

        # Encoder drift: actual L2 distance from initial params (not norm change)
        enc_params_now = torch.cat([p.data.flatten() for p in text_enc._model.parameters()])
        enc_drift_pct = (enc_params_now - enc_params0).norm().item() / max(enc_norm0, 1e-9) * 100

        # Prefix scale health check (item_target_norm ~1.72 → fix holding; ~0.26 → regression)
        scale_info = _measure_prefix_scale(
            model, text_enc, projector, ctrl_uid_strs,
            user_queries, user_memories, dataset, device, maxlen,
        )

        print(f"\nEpoch {epoch}/{args.num_epochs}  "
              f"L_ret={avg_l_ret:.4f}  L_align={avg_l_align:.4f}  L_contr={avg_l_contr:.4f}  "
              f"val_L_ret={val_loss:.4f}  routing_cov={routing_cov:.3f}")
        print(f"  Ctrl: J(c,w)={ctrl['J_cw']:.4f}  J(c,v)={ctrl['J_cv']:.4f}  "
              f"J(c,cs2)={ctrl['J_cs2']:.4f}  gap={gap:+.4f} [{gap_label}]")
        if scale_info:
            print(f"  Scale: item_norm={scale_info['item_target_norm']:.4f}  "
                  f"L_nonpad={scale_info['L_nonpad_med']}  "
                  f"share~{scale_info['share_est_pct']:.1f}%  "
                  f"pfx_raw_norm={scale_info['pfx_raw_norm']:.4f}")
        print(f"  Encoder drift: {enc_drift_pct:.3f}%  "
              f"(>0.1% = updating; <0.01% = lr too low or frozen)")

        # Routing accuracy check every 3 epochs (detect encoder collapse)
        if epoch % 3 == 0:
            text_enc.eval()
            _check_baseline_routing(text_enc, user_memories, pairs_jsonl, n=200,
                                    seed=seed, mid_to_uid=mid_to_uid)
            text_enc.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "text_encoder_state": text_enc.state_dict(),
                    "projector_state": projector.state_dict(),
                    "ctrl": ctrl,
                    "config": {
                        "d_text": d_text, "d_sasrec": d_sasrec,
                        "alpha": alpha, "encoder": "BAAI/bge-base-en-v1.5",
                        "lr_encoder": lr_encoder, "lr_projector": lr_projector,
                        "stage1_ckpt": getattr(args, "stage1_ckpt", None),
                    },
                },
                best_ckpt_path,
            )
            print(f"  [ckpt] best → {best_ckpt_path}  (val_loss={val_loss:.4f})")
        else:
            no_improve += 1
            print(f"  [patience] {no_improve}/{patience}  best={best_val_loss:.4f}")
            if no_improve >= patience:
                print(f"[early stop] epoch {epoch}  best_val_loss={best_val_loss:.4f}")
                break

    sampler.close()
    print(f"\nStage2 training complete.  Best val_L_ret: {best_val_loss:.4f}")
    print(f"Checkpoint: {best_ckpt_path}")


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run CPU smoke test")
    parser.add_argument("--bank_jsonl", type=str, default=None)
    parser.add_argument("--mock_bank", type=str, default="data/memory/Books/f3_bank.jsonl",
                        help="Bank JSONL for smoke mock users (f3_bank or memory_full format)")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/Books/sasrec_pretrain.pt")
    parser.add_argument("--stage1_ckpt", type=str, default=None,
                        help="Stage1 checkpoint (stage1_align_best.pt) to init encoder. "
                             "Omit for baseline ablation (raw BGE).")
    parser.add_argument("--category", type=str, default="Books")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=10)
    # LLaVA asymmetric LR
    parser.add_argument("--lr_encoder", type=float, default=1e-5,
                        help="Encoder LR (low, preserves Stage1 alignment)")
    parser.add_argument("--lr_projector", type=float, default=1e-3,
                        help="Projector LR (high, bridges modality gap)")
    parser.add_argument("--use_contrastive", action="store_true", default=False,
                        help="Ablation: add L_contrastive to loss (default: off, SPEC loss only)")
    # Full training
    parser.add_argument("--pairs_jsonl", type=str, default=None,
                        help="Align pairs JSONL for pseudo-queries. "
                             "Defaults to data/processed/{category}/align_pairs.jsonl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early-stopping patience (epochs without val improvement)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Checkpoint output dir. Defaults to checkpoints/{category}")
    parser.add_argument("--ctrl_users", type=int, default=16,
                        help="Number of users for per-epoch controllability monitoring")
    parser.add_argument("--n_workers", type=int, default=3,
                        help="WarpSampler worker threads")
    parser.add_argument("--proj_ckpt", type=str, default=None,
                        help="Warm-start projector from a stage2 checkpoint "
                             "(loads projector_state). Default: cold random init.")
    cli = parser.parse_args()

    if cli.smoke or cli.bank_jsonl is None:
        run_smoke(cli)
    else:
        run_training(cli)


if __name__ == "__main__":
    main()
