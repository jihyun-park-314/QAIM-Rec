# QAIM-Rec Code Audit v0.4.22

Audit date: 2026-06-30

Scope: read-only audit of current code and artifacts against `plan.md`, `decisions.md`, and `research_idea.pdf`. No code, data, split, bank, or training behavior was modified.

## 1. Executive Summary

Full training currently allowed: NO.

Top 5 blockers:

1. `L_mem_X` self-reconstruction risk is confirmed in current artifacts: 1,005 / 1,146 valid samples have `X_in_seq=True` (87.7%), above the 30% gate.
2. The code still allows `run_training()` with `--beta_mem_x > 0`; there is no hard runtime gate enforcing the documented "FULL TRAINING: FORBIDDEN" state.
3. `data/processed/Books/align_pairs.jsonl` has a broken manifest contract: actual rows 52,236 and md5 `2a8d11f...`, manifest rows 45 and md5 `b789b013...`.
4. Stage2/eval memory text uses only `intent_description`, while the bank/spec define `embedding.source_text` as the routing/steering text.
5. Stale generation/evaluation paths remain and can silently produce old invalid results (`generate_pseudo_queries.py` direct align-pairs path; `eval_steered.py` summary fields for unimplemented R@50/R@100).

Top 5 major risks:

1. `--use_contrastive` retains a spec-outside loss path, default off but still runnable.
2. Prefix creation/routing is duplicated across train/eval/diagnostics instead of a single shared function.
3. `SASRec.predict()` still hardcodes `prefix_embeds=None`.
4. `WarpSampler` uses multiple worker processes with queue race ordering, so fixed seeds do not guarantee exact repeatability when `n_workers > 1`.
5. Stage2/eval do not verify md5 manifests before consuming frozen artifacts.

Top 5 refactor candidates:

1. Extract L_mem construction and masking from `train_step()`.
2. Replace `_route_batch()` positional tuple contract with a typed batch object.
3. Unify train/eval prefix construction and routing.
4. Centralize bank loading and memory text selection.
5. Add artifact manifest validation to training/eval entrypoints.

## 2. Severity Table

| ID | Severity | Area | Finding | Required before full training |
|---|---|---|---|---|
| F-01 | P0 | L_mem_X | X target is usually already in the input sequence | YES |
| F-02 | P0 | Training gate | Code does not prevent forbidden beta_mem full training | YES |
| F-03 | P1 | Reproducibility | Current align_pairs manifest is stale/mismatched | YES |
| F-04 | P1 | Memory semantics | Stage2 uses `intent_description`, not `embedding.source_text` | YES |
| F-05 | P1 | Loss surface | Spec-outside `L_contrastive` remains behind CLI flag | NO, if enforced off |
| F-06 | P1 | Data generation | Stale P2 script can write invalid align_pairs directly | YES |
| F-07 | P1 | Evaluation | Eval summary prints metrics not computed by full-ranking path | NO for training, YES for evaluation claims |
| F-08 | P2 | Determinism | Multiprocess sampler can reorder fixed-seed batches | SHOULD |
| F-09 | P2 | Path consistency | Prefix/routing duplicated across files | SHOULD |
| F-10 | P2 | API safety | `predict()` ignores prefix | CAN WAIT if unused |

P0 = invalidates results / leakage / full training blocker. P1 = major correctness risk. P2 = maintainability or reproducibility risk. P3 = nice-to-have.

## 3. Spec Compliance Matrix

| Component | Expected by plan/decisions | Actual code behavior | Status | Evidence |
|---|---|---|---|---|
| PDF core method | Offline memory, query-memory alignment, MLP(h_query,h_memory), prefix steering, online LLM 0 | Projector concatenates query/memory and returns P=1 prefix | PASS | `src/models/projector.py:1` |
| Frozen SASRec in Stage2 | SASRec frozen unless ablated | All SASRec params set `requires_grad_(False)` | PASS | `src/training/train_hybrid.py:1174` |
| L_mem_X | `softplus(-(zX_pos-zX_neg))`, target X, m+ provenance, m- same-user diff memory | Implemented with source_item_id target and two prefix forwards | PASS with P0 gate failure | `src/training/train_hybrid.py:360`, `src/training/train_hybrid.py:379`, `src/training/train_hybrid.py:388` |
| L_mem_Wtrain_aligned | Only A+B: W==X or W in X memory; C/D/K=1 masked | Implemented via `w == x or w in item_ids` under `lmem_x_mask` | PASS | `src/training/train_hybrid.py:391`, `src/training/train_hybrid.py:397`, `src/training/train_hybrid.py:405` |
| Wtrain_all | Forbidden | No all-sample Wtrain branch found | PASS | `src/training/train_hybrid.py:394` |
| L_mem m+ | Must be `align_pairs.positive_memory_id`, not router top-1 | `_route_batch()` keeps routed_mem and prov_pos_mem separate | PASS | `src/training/train_hybrid.py:730`, `src/training/train_hybrid.py:780` |
| L_mem m- | Same-user different-cluster, no cross-user fallback | Neg candidates are same user's memories with different mid | PASS | `src/training/train_hybrid.py:782` |
| K=1 mask | Mask out for L_mem | `valid = k_personal >= 2 ...` | PASS | `src/training/train_hybrid.py:785` |
| beta defaults | `beta_mem_x=0`, `beta_mem_wtrain=0` | Parser defaults both to 0.0 | PASS | `src/training/train_hybrid.py:1697` |
| beta=0 no-op | No L_mem contribution when both beta are 0 | Smoke: base `L_mem_X=0`, `L_mem_Wtrain=0`; code skips branches | PASS for L_mem terms; historical equivalence UNKNOWN | `src/training/train_hybrid.py:363`, `src/training/train_hybrid.py:394` |
| Hidden losses | No hidden/spec-outside loss | `--use_contrastive` still adds `alpha * l_contrastive`, default false | FAIL if enabled | `src/training/train_hybrid.py:351`, `src/training/train_hybrid.py:1678` |
| Split parser | Use `splits["users"]`, not top-level keys | Loader and bank use `splits["users"]` | PASS | `src/models/dataloader.py:44`, `src/memory/bank.py:193` |
| Train sampler | Train split only | `WarpSampler(dataset["user_train"], ...)` | PASS | `src/training/train_hybrid.py:1233` |
| W_train source | `pos_np[:,-1]` from user_train only | WarpSampler sets `nxt=user_train[uid][-1]`; Stage2 captures before X replacement | PASS | `src/models/dataloader.py:84`, `src/training/train_hybrid.py:1298` |
| Full ranking eval | Full catalog, seen masking, Recall/NDCG/MRR | Implemented for K=[5,10,20] | PASS | `src/eval/full_ranking.py:17`, `src/eval/full_ranking.py:68`, `src/eval/full_ranking.py:103` |
| Steered eval prefix | Prefix must be injected | `evaluate_full_with_ranks(... prefix_fn=prefix_fn)` calls `log2feats(...prefix...)` | PASS | `scripts/eval_steered.py:448`, `src/eval/full_ranking.py:189` |
| Single prefix path | Train/eval/diagnostic should share construction | `conditions.make_prefix` exists but `eval_steered.py` and train route manually | FAIL | `src/eval/conditions.py:24`, `scripts/eval_steered.py:58`, `src/training/train_hybrid.py:706` |
| Prefix scale | Non-padding item norm | Nonpad mask used before mean | PASS | `src/models/sasrec.py:99` |
| `predict()` | Should not silently ignore prefix if used for steering | Still calls `log2feats(log_seqs)` with prefix None | FAIL if used | `src/models/sasrec.py:145` |
| Artifact md5 | Frozen artifacts should be checked before use | Some build scripts check md5, Stage2/eval do not; align_pairs manifest mismatch | FAIL | `scripts/build_candidates.py:50`, `src/training/train_hybrid.py:1215` |

## 4. Detailed Findings

### F-01. X_in_seq self-reconstruction risk

Severity: P0

Symptom: `L_mem_X` valid samples usually contain target X inside the SASRec input sequence.

Why it matters: `L_mem_X` can teach score-space discrimination for an item the sequence already exposes, so a full beta_mem_x training run may measure self-reconstruction rather than memory-conditioned steering.

Code evidence: the audit script defines `x_in_seq = (x in seq_items_set)` and classifies `B_in_seq_before_wtrain` when X is in `user_train[:-1]` (`scripts/step2g_xinseq_audit.py:209`, `scripts/step2g_xinseq_audit.py:225`, `scripts/step2g_xinseq_audit.py:228`). Current rerun: total 2,000, L_mem_X valid 1,146, X_in_seq 1,005 (87.7%), A_eq_wtrain 141, C_not_found 0.

Suggested fix: choose one of the documented designs: restrict `L_mem_X` to X==W_train, build `seq_before_X`, or demote `L_mem_X` and promote `L_mem_Wtrain_aligned`; update spec before retraining.

Required before full training: YES.

### F-02. Forbidden full training is not enforced in code

Severity: P0

Symptom: Documentation says full training is forbidden, but the CLI still accepts beta values and calls `run_training()` without an abort.

Why it matters: A single command with `--beta_mem_x 1` can produce invalid full-training checkpoints despite the current gate.

Code evidence: beta args are accepted (`src/training/train_hybrid.py:1697`), `use_lmem` is derived from beta values (`src/training/train_hybrid.py:1257`), and `run_training()` proceeds into the loop (`src/training/train_hybrid.py:1267`) without checking the X_in_seq gate.

Suggested fix: add an explicit runtime guard unless an updated spec flag or approved resolved mode is supplied.

Required before full training: YES.

### F-03. `align_pairs.jsonl` manifest mismatch

Severity: P1

Symptom: Actual `data/processed/Books/align_pairs.jsonl` has 52,236 rows and md5 `2a8d11f38864498f8b198cd74c04520c`, but `align_pairs.manifest.json` claims 45 rows and md5 `b789b0131eddb83a9b9050127543a383`.

Why it matters: The current training input may be semantically valid, but reproducibility gates cannot prove it. Stage2 also does not verify the manifest before use.

Code evidence: `train_hybrid.py` loads the path directly (`src/training/train_hybrid.py:1211`); manifest writer exists in the regen script (`scripts/regen_align_pairs_v0417.py:219`). The current manifest content was inspected during this audit.

Suggested fix: regenerate or repair the manifest for the current merged align_pairs artifact and add manifest validation to Stage1/Stage2/eval entrypoints.

Required before full training: YES.

### F-04. Stage2 memory text does not match `embedding.source_text`

Severity: P1

Symptom: The f3 bank contains `embedding.source_text`, but `_load_bank_full()` stores `rec["intent_description"]` as the memory text.

Why it matters: The spec defines `source_text` as the routing and steering text. Stage1 bank vectors were generated from source_text-like content, while Stage2/eval re-encode only intent_description, changing the memory representation.

Code evidence: bank loader uses `rec["intent_description"]` (`src/training/train_hybrid.py:655`). The bank schema includes `embedding.source_text` in artifact rows and plan lines define source_text as the memory embedding text (`plan.md:179`).

Suggested fix: load `rec["embedding"]["source_text"]` when present, fallback to intent_description only for legacy artifacts, and rerun routing/prefix smoke.

Required before full training: YES.

### F-05. Spec-outside contrastive loss remains runnable

Severity: P1

Symptom: `--use_contrastive` adds an extra margin loss to Stage2.

Why it matters: v0.4.22 loss is explicitly `L_retrieval + alpha*L_align + beta_x*L_mem_X + beta_w*L_mem_Wtrain_aligned`. A runnable extra loss can silently invalidate comparisons if enabled.

Code evidence: contrastive branch adds `loss = loss + alpha * l_contrastive` (`src/training/train_hybrid.py:351`), CLI flag exists with default false (`src/training/train_hybrid.py:1678`).

Suggested fix: remove it, rename it as a clearly documented historical ablation, or make training abort if enabled without an explicit ablation mode.

Required before full training: NO if strictly off, but YES for clean release.

### F-06. Stale P2 generator can emit invalid align_pairs

Severity: P1

Symptom: `generate_pseudo_queries.py` directly writes `align_pairs` without `source_item_id` and with same-user plus cross-user hard negatives. `regen_align_pairs_v0417.py` later fixes this, but the stale path still exists.

Why it matters: Running the old documented P2 train command can silently recreate an align_pairs file that does not satisfy v0.4.22.

Code evidence: old writer emits only query, positive_memory_id, hard_negative_memory_ids (`scripts/generate_pseudo_queries.py:394`); regen adds `source_item_id` and K-specific negatives (`scripts/regen_align_pairs_v0417.py:150`).

Suggested fix: disable direct align_pairs writing in `generate_pseudo_queries.py` or update it to v0.4.22 semantics.

Required before full training: YES.

### F-07. Evaluation summary includes metrics not computed by full-ranking path

Severity: P1

Symptom: `eval_steered.py` summary prints R@50/R@100 for all conditions, but `evaluate_full_with_ranks()` computes only @5/@10/@20.

Why it matters: Summary rows for vanilla/steered W-target conditions can show zeros for R@50/R@100 that are not real measurements.

Code evidence: full-ranking `_KS = [5, 10, 20]` (`src/eval/full_ranking.py:17`); eval summary prints R@50/R@100 via `.get(..., 0)` (`scripts/eval_steered.py:530`).

Suggested fix: either compute @50/@100 in `full_ranking.py` or remove those columns for conditions lacking them.

Required before full training: NO, but required before evaluation claims.

### F-08. Fixed seeds do not fully determine sampler order

Severity: P2

Symptom: `WarpSampler` starts multiple worker processes and consumes a shared queue.

Why it matters: Per-worker seeds are fixed, but queue arrival order can vary with process scheduling, changing batches and `rng` consumption in routing/negative selection.

Code evidence: workers are created in `WarpSampler.__init__` (`src/models/dataloader.py:119`), each gets a seed from `np.random.randint` (`src/models/dataloader.py:123`), and batches are read from a multiprocessing queue (`src/models/dataloader.py:129`). Stage2 default `n_workers=3` (`src/training/train_hybrid.py:1232`).

Suggested fix: use `n_workers=1` for audited runs or implement deterministic batch construction in the main process.

Required before full training: SHOULD.

### F-09. Prefix/routing logic is duplicated

Severity: P2

Symptom: `conditions.make_prefix()` exists but train and eval use separate custom prefix/routing code.

Why it matters: This project already had train/eval prefix mismatch bugs. Duplication is a recurrence risk.

Code evidence: helper exists (`src/eval/conditions.py:24`); eval defines `make_steered_prefix_fn()` (`scripts/eval_steered.py:58`); train uses `_route_batch()` and direct projector calls (`src/training/train_hybrid.py:706`).

Suggested fix: create one shared route-and-prefix builder used by training, eval, and diagnostics.

Required before full training: SHOULD.

### F-10. `SASRec.predict()` still ignores prefix

Severity: P2

Symptom: `predict()` calls `log2feats(log_seqs)` without a prefix argument.

Why it matters: Any future evaluation or serving code using `predict()` will silently run vanilla SASRec.

Code evidence: `src/models/sasrec.py:145`.

Suggested fix: add optional `prefix_embeds=None` to `predict()` and thread it to `log2feats`.

Required before full training: NO if all eval continues through `log2feats`, but should be fixed before serving or broader eval.

## 5. L_mem-Specific Audit

- `L_mem_X` status: implemented as intended mathematically, but full training is blocked by X_in_seq risk. Evidence: `src/training/train_hybrid.py:360`, `src/training/train_hybrid.py:388`.
- `L_mem_Wtrain_aligned` status: implemented as A+B-only ablation. Evidence: `src/training/train_hybrid.py:397`, `src/training/train_hybrid.py:405`.
- `Wtrain_all` absence: no all-sample Wtrain loss branch found.
- m+ provenance status: PASS. Positive memory is looked up by `positive_memory_id`, separate from router top-1. Evidence: `src/training/train_hybrid.py:780`.
- m- intra-user status: PASS. Negatives are same-user memories with different mid. Evidence: `src/training/train_hybrid.py:782`.
- K=1 mask status: PASS. Evidence: `src/training/train_hybrid.py:785`.
- X_in_seq risk: FAIL/P0. Current rerun confirms 87.7%.
- beta=0 compatibility: PASS for L_mem terms. `--smoke_lmem --batch_size 32` base mode reported `L_mem_X=0.000000`, `L_mem_Wtrain=0.000000`. UNKNOWN for historical equivalence beyond current X-target Stage2 behavior because no legacy no-source-item comparison test exists.

## 6. Train/Eval Path Consistency

PASS:

- SASRec `log2feats()` accepts prefix and returns prefix offset P (`src/models/sasrec.py:77`).
- BPR scoring excludes prefix positions with `item_feats = log_feats[:, P:, :]` (`src/models/sasrec.py:134`).
- Causal mask over P+L lets later item positions attend to prefix (`src/models/sasrec.py:105`, `src/models/sasrec.py:110`).
- Prefix scale uses non-padding item norm (`src/models/sasrec.py:99`).
- Full-ranking eval injects prefix through `prefix_fn` (`src/eval/full_ranking.py:96`, `src/eval/full_ranking.py:189`).

FAIL/RISK:

- Prefix construction is not unified across train/eval/diagnostics.
- `predict()` remains vanilla-only.
- `eval_steered.py` summary has stale metric columns for @50/@100.

## 7. Leakage Section

Split/data findings:

- `load_data()` correctly parses `splits["users"]` (`src/models/dataloader.py:44`).
- `user_train`, `user_valid`, and `user_test` are separated (`src/models/dataloader.py:46`).
- Stage2 sampler receives only `dataset["user_train"]` (`src/training/train_hybrid.py:1233`).
- `W_train = pos_np[:,-1]` comes from the last train item produced by WarpSampler (`src/models/dataloader.py:84`).
- Val/test are not used in Stage2 training loss. Eval validation loss uses val only for checkpoint selection (`src/training/train_hybrid.py:893`), not training gradients.
- Memory bank split schema bug is fixed in current bank assembly code (`src/memory/bank.py:193`).
- f3 bank manifest asserts `val_test_leakage: 0`, `evidence_empty_item_ids: 0`, and matching md5 for `data/memory/Books/f3_bank.jsonl`.
- Existing lightweight tests passed: `python3 -m pytest tests/test_split_consistency.py tests/test_no_leakage.py tests/test_leakage_v2.py -q` -> 39 passed.

Leakage verdict: no direct val/test leakage found in current split, sampler, or memory bank evidence. The current blocker is self-reconstruction via train-history X, not val/test leakage.

## 8. Refactor Plan

Must-fix before training:

1. Add runtime guard for forbidden beta_mem full training until the X_in_seq design is resolved.
2. Repair and validate `align_pairs` manifest for the current 52,236-row artifact.
3. Replace Stage2/eval memory text with `embedding.source_text`.
4. Disable or update stale direct align_pairs generation in `generate_pseudo_queries.py`.

Should-fix soon:

1. Extract L_mem sample construction and loss computation into a helper module.
2. Create a typed `RouteBatch` object instead of returning a 6-tuple plus `prov_data`.
3. Use one shared route/prefix function across train, eval, and diagnostics.
4. Make deterministic audited runs use `n_workers=1` or deterministic main-process sampling.
5. Remove stale @50/@100 columns unless full-ranking computes them.

Can wait:

1. Add optional prefix to `SASRec.predict()`.
2. Split per-epoch diagnostics out of `train_hybrid.py`.
3. Add richer artifact manifest reporting to all scripts.

## 9. Existing Tests and Suggested Tests

Existing tests run:

- `python3 -m pytest tests/test_split_consistency.py tests/test_no_leakage.py tests/test_leakage_v2.py -q`
- Result: 39 passed in 4.42s.

Smoke/read-only commands run:

- `python3 scripts/step2g_xinseq_audit.py --n_samples 2000 --seed 42 --device cpu`
- `python3 src/training/train_hybrid.py --smoke_lmem --device cpu --category Books --bank_jsonl data/memory/Books/f3_bank.jsonl --pairs_jsonl data/processed/Books/align_pairs.jsonl --checkpoint checkpoints/Books/sasrec_pretrain.pt --batch_size 32`
- Artifact md5 audit for splits, bank, align_pairs, candidates, p1_extractions.

Suggested tests to add:

1. `test_beta_zero_noop`: compare beta=0 against an explicit no-Lmem baseline, not only zero L_mem metrics.
2. `test_lmem_x_target_is_source_item`: assert `x_targets` comes from `align_pairs.source_item_id`.
3. `test_lmem_wtrain_mask_ab_only`: assert C/D samples produce zero Wtrain valid count.
4. `test_no_val_test_in_train_loss`: assert Stage2 train_step inputs never include valid/test targets.
5. `test_mpos_is_provenance_not_router`: force router miss and assert L_mem m+ is still provenance.
6. `test_mneg_same_user_different_cluster`: assert all L_mem m- are same-user and different mid.
7. `test_k1_mask`: K=1 users must produce zero L_mem valid samples.
8. `test_x_in_seq_audit`: CI-friendly sample gate fails above threshold.
9. `test_prefix_eval_injection`: spy on `log2feats(prefix_embeds=...)`.
10. `test_no_hidden_loss`: fail if `--use_contrastive` is enabled in non-ablation mode.
11. `test_manifest_md5`: Stage1/Stage2 must abort on stale align_pairs/bank manifests.
12. `test_memory_text_source_text`: assert Stage2 uses `embedding.source_text` when present.

## 10. Recommended Next Action

Recommended next implementer prompt:

```text
Implement the v0.4.22 pre-training blockers only. Do not train.
1. Add a hard guard in train_hybrid.py that aborts any full run with beta_mem_x>0 or beta_mem_wtrain>0 while the current X_in_seq gate is unresolved, unless an explicit --allow_lmem_after_xinseq_resolution flag is provided.
2. Change _load_bank_full() and all audit/eval loaders that consume f3_bank.jsonl to use embedding.source_text when present, with intent_description as legacy fallback.
3. Disable or update generate_pseudo_queries.py so it cannot write stale align_pairs lacking source_item_id.
4. Add manifest md5 validation for align_pairs and f3_bank before Stage1/Stage2/eval.
5. Add focused tests for the above. Do not run full training.
```

## 11. Final Training Status

FULL TRAINING STATUS: BLOCKED

Exact blockers:

1. X_in_seq ratio is 87.7% for L_mem_X valid samples, above the 30% gate.
2. Code lacks a runtime guard against forbidden beta_mem full training.
3. Current align_pairs manifest is stale/mismatched.
4. Stage2 memory text is not the spec-defined `embedding.source_text`.
5. Stale align_pairs generation path can recreate invalid inputs.

Minimum commands/checks needed to unblock after fixes:

```bash
python3 -m pytest tests/test_split_consistency.py tests/test_no_leakage.py tests/test_leakage_v2.py -q
python3 scripts/step2g_xinseq_audit.py --n_samples 2000 --seed 42 --device cpu
python3 src/training/train_hybrid.py --smoke_lmem --device cpu --category Books --bank_jsonl data/memory/Books/f3_bank.jsonl --pairs_jsonl data/processed/Books/align_pairs.jsonl --checkpoint checkpoints/Books/sasrec_pretrain.pt --batch_size 32
```

Additional required checks after implementation:

- New manifest md5 test must pass for `align_pairs.jsonl` and `f3_bank.jsonl`.
- New memory-text test must prove `embedding.source_text` is used.
- New beta-zero/no-hidden-loss tests must pass.
