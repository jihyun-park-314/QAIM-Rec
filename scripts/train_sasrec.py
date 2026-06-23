"""F6a: SASRec training script — pmixer-adapted with full-ranking eval.

Usage:
    # Smoke (200-user preprocessed data):
    python scripts/train_sasrec.py --category Books --data_dir data/processed \\
        --maxlen 50 --hidden_units 64 --num_blocks 2 --num_heads 1 \\
        --dropout_rate 0.2 --batch_size 128 --num_epochs 10 --device cpu

    # Full run (Beauty):
    python scripts/train_sasrec.py --category Beauty_and_Personal_Care \\
        --data_dir data/processed --maxlen 50 --hidden_units 256 --num_blocks 2 \\
        --num_heads 1 --dropout_rate 0.2 --l2_emb 0.0 --batch_size 128 \\
        --num_epochs 200 --device cuda:0

    # Full run (Books):
    python scripts/train_sasrec.py --category Books --data_dir data/processed \\
        --maxlen 50 --hidden_units 256 --num_blocks 2 --num_heads 1 \\
        --dropout_rate 0.2 --l2_emb 0.0 --batch_size 128 --num_epochs 200 \\
        --device cuda:0
"""
import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sasrec import SASRec
from src.models.dataloader import load_data, WarpSampler
from src.eval.full_ranking import evaluate_full, evaluate_full_stratified, print_metrics


def parse_args():
    p = argparse.ArgumentParser(description="F6a SASRec training")
    p.add_argument("--category", required=True)
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--checkpoint_dir", default="checkpoints")

    # Model
    p.add_argument("--maxlen", type=int, default=50)
    p.add_argument("--hidden_units", type=int, default=256)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=1)
    p.add_argument("--dropout_rate", type=float, default=0.2)
    p.add_argument("--norm_first", action="store_true", default=False,
                   help="Pre-LN (True) vs Post-LN (False, default, pmixer default)")

    # Training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2_emb", type=float, default=0.0)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=20,
                   help="Evaluate on val set every N epochs")

    # Infra
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sequences_jsonl", default=None,
                   help="Path to sequences.jsonl for warm/cold stratified eval. "
                        "If omitted, uses data_dir/{category}/sequences.jsonl")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[train] category={args.category}, device={args.device}")
    print(f"[train] loading data from {args.data_dir}/{args.category}/")

    dataset = load_data(args.category, args.data_dir)
    usernum = dataset["usernum"]
    itemnum = dataset["itemnum"]
    print(f"[train] usernum={usernum}, itemnum={itemnum}")

    # Number of training steps per epoch
    num_batch = max(1, usernum // args.batch_size)
    t0 = time.time()

    sampler = WarpSampler(
        dataset["user_train"], usernum, itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=args.num_workers,
    )

    model = SASRec(usernum, itemnum, args).to(args.device)

    # Xavier init — matches pmixer exactly
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    model.item_emb.weight.data[0, :] = 0  # padding index
    model.pos_emb.weight.data[0, :] = 0   # padding index

    # No weight_decay — L2 is applied manually to item_emb only (matching pmixer exactly)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    bce_criterion = torch.nn.BCEWithLogitsLoss()

    ckpt_dir = os.path.join(args.checkpoint_dir, args.category)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_ndcg10 = 0.0
    best_epoch = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(range(num_batch), desc=f"Epoch {epoch:3d}/{args.num_epochs}", ncols=90, leave=False,
                    disable=not sys.stdout.isatty())
        for step in pbar:
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = (np.array(x) for x in [u, seq, pos, neg])

            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)

            # Mask padding positions (where pos==0)
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            # L2 regularization on item embeddings only — matches pmixer: sum(param**2)
            if args.l2_emb > 0:
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.sum(param ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / num_batch
        elapsed = time.time() - t0

        print(f"[epoch {epoch:3d}/{args.num_epochs}] loss={avg_loss:.4f}  elapsed={elapsed:.0f}s")

        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            print(f"  [eval val] epoch={epoch}")
            model.eval()
            val_metrics = evaluate_full(model, dataset, args, split="val", max_users=usernum)
            print_metrics(val_metrics, prefix="val")

            val_ndcg10 = val_metrics.get("NDCG@10", 0.0)
            if val_ndcg10 >= best_val_ndcg10:
                best_val_ndcg10 = val_ndcg10
                best_epoch = epoch
                ckpt_path = os.path.join(ckpt_dir, "sasrec_pretrain.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "usernum": usernum,
                    "itemnum": itemnum,
                }, ckpt_path)
                print(f"  [ckpt] saved → {ckpt_path} (NDCG@10={val_ndcg10:.4f})")

    sampler.close()

    print(f"\n[train] Done. Best val NDCG@10={best_val_ndcg10:.4f} at epoch {best_epoch}")
    print(f"[train] Checkpoint: {os.path.join(ckpt_dir, 'sasrec_pretrain.pt')}")

    # Final test evaluation using best checkpoint
    print("\n[eval test] Loading best checkpoint...")
    ckpt = torch.load(os.path.join(ckpt_dir, "sasrec_pretrain.pt"), map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Resolve sequences.jsonl path for warm/cold stratification
    seq_jsonl = args.sequences_jsonl or os.path.join(
        args.data_dir, args.category, "sequences.jsonl"
    )

    strat = evaluate_full_stratified(
        model, dataset, args, split="test", max_users=usernum,
        sequences_jsonl_path=seq_jsonl,
    )
    test_metrics = strat["overall"]

    print("  [test metrics — overall]")
    print_metrics(test_metrics, prefix="test/overall")
    print(f"  [warm n={strat['counts']['warm']}  cold n={strat['counts']['cold']}]")
    if strat["counts"]["warm"] > 0:
        print_metrics(strat["warm"], prefix="test/warm  ")
    if strat["counts"]["cold"] > 0:
        print_metrics(strat["cold"], prefix="test/cold  ")

    # Save test metrics alongside checkpoint
    import json
    results = {
        "category": args.category,
        "best_epoch": best_epoch,
        "best_val_ndcg10": best_val_ndcg10,
        "val_metrics": ckpt.get("val_metrics", {}),
        "test_metrics": test_metrics,
        "test_metrics_stratified": strat,
        "args": vars(args),
    }
    results_path = os.path.join(ckpt_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[train] Results saved → {results_path}")


if __name__ == "__main__":
    main()
