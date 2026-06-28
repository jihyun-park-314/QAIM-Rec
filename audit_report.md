# QAIM-Rec 정적 감사 보고서

**감사 기준**: plan.md v0.4.15, decisions.md  
**감사 방법**: 코드 읽기 전용 (정적 분석). 실행 없음.  
**감사 시점**: 2026-06-28  
**감사 대상**: 7개 영역 A–G (전 파이프라인)

---

## 판정 범례

| 기호 | 의미 |
|------|------|
| ✓ MATCH | 코드 줄로 입증 |
| ✗ MISMATCH | SPEC↔코드 불일치. severity 명시 |
| ? UNKNOWN | 해당 파일 미확인 |

심각도: **CRITICAL** (실행 즉시 실패/훈련 무효) / **MAJOR** (결과를 조용히 오염) / **MINOR** (영향 제한적)

---

## A. 전처리 / 데이터 (M0) — src/data/preprocess.py

### A-1: 5-core 필터
- **SPEC** (plan.md §2 M0): user ≥5, item ≥5, 수렴까지 반복
- **코드** (preprocess.py:88-112): `five_core()` — user_counts/item_counts 동시 필터, `len(filtered)==len(data)` 조건으로 수렴 판정
- **판정**: ✓ MATCH

### A-2: LOO split
- **SPEC**: test=마지막, val=끝에서 두 번째, train=나머지
- **코드** (preprocess.py:231-240):
  ```python
  "train": item_ids[:-2],
  "val": item_ids[-2],
  "test": item_ids[-1],
  ```
  timestamp 정렬 후 슬라이스 (line 223)
- **판정**: ✓ MATCH

### A-3: 시간 sanity 플래그
- **SPEC**: 95th percentile 미만 test_ts → 플래그 (제외 아님)
- **코드** (preprocess.py:249-255): `t_cut = test_timestamps_sorted[pct95_idx]`, `temporal_flag=True` 설정만
- **판정**: ✓ MATCH (제외 없음 확인)

### A-4: ID 공간
- **SPEC**: 1-indexed, 0=padding
- **코드** (preprocess.py:187-190): `user2id = {u: i+1 for i, u in enumerate(users)}`
- **splits.json 키 형식**: `str(uid)` → 문자열 정수 "1", "2", ... (line 234)
- **판정**: ✓ MATCH

### A-5: memory eligibility 태깅 — is_discriminative 누락
- **SPEC** (decisions.md): 메모리 대상 조건 = rating≥4 AND ≥10 words AND is_discriminative
- **코드** (preprocess.py:341-345):
  ```python
  is_eligible = (
      r["rating"] >= _MIN_RATING      # ≥4
      and n_words >= _MIN_WORDS        # ≥10
      and in_train                     # train 포함
  )
  ```
  `is_discriminative` 조건 없음. 해당 필드 기록 자체도 없음.
- **판정**: ✗ MISMATCH — **MINOR**  
  **Impact**: preprocess.py의 `is_eligible` ≠ SPEC 메모리 대상 정의. is_discriminative는 합성 파이프라인(F2)이 담당한다면 문제 없으나, sequences.jsonl 소비자가 `is_eligible`을 메모리 대상으로 오용할 경우 is_discriminative 필터 우회. 합성 코드 미확인 — 불명(코드 미확인).

---

## B. 메모리 뱅크 구축 (M2) — src/memory/bank.py

### B-1: splits.json 스키마 불일치 — leakage 검사 무효화
- **SPEC**: evidence.timestamps ⊂ user train timestamps 보장
- **실제 splits.json 형식** (preprocess.py:263):
  ```json
  { "users": {"1": {"train": [int, ...], ...}, ...}, "meta": {...} }
  ```
- **bank.py 코드** (bank.py:193-197):
  ```python
  # 주석: splits.json format: {user_id_str: {"train": [...], ...}}  ← 틀린 가정
  train_user_set = set(splits.keys())           # → {"users", "meta"}
  for uid_str, split in splits.items():
      ts_list = [item.get("timestamp") for item in split.get("train", [])
                 if item.get("timestamp")]
  ```
  `splits.keys()` = `{"users", "meta"}` — 실제 유저 ID가 아님.  
  내부 루프에서 `split.get("train", [])` = `[]` (그 레벨에 "train" 키 없음) → `ts_list = []`  
  결과: `train_timestamps_by_user = {"users": set(), "meta": set()}`
- **assemble_user_bank 영향** (bank.py:207): `train_ts = train_timestamps_by_user.get(uid_str)` → 항상 `None` → leakage 검사 `if train_timestamps is not None:` 조건 미통과 → **항상 스킵**
- **user set validation 영향** (bank.py:235-243): `train_user_set = {"users", "meta"}` → bank의 모든 유저가 "extra_in_bank"로 오표기
- **판정**: ✗ MISMATCH — **MAJOR**  
  **Impact**: splits_path가 전달될 때 leakage 검사가 완전히 우회됨. 훈련 세트 바깥 evidence가 메모리에 들어가도 감지 안 됨. (splits_path=None이면 이 경로 실행 안 됨 — 호출자 확인 필요)

### B-2: K 분기 로직
- **SPEC** (plan.md §7 #1): K≥1 personal only, K=0 prototype fallback
- **코드** (bank.py:108-140): `if k >= 1: personal only`, else nearest_prototype or DEFAULT_INTENT
- **판정**: ✓ MATCH

### B-3: leakage check 함수 자체
- **코드** (bank.py:73-80): `all(ts in train_timestamps for ts in evidence_timestamps)` — 로직 자체는 정확
- **판정**: ✓ MATCH (단 B-1 이슈로 실제로 호출되지 않음)

### B-4: is_discriminative 필터 — 불명
- bank.py는 synth_units_by_user를 그대로 사용; is_discriminative 검사 없음
- 합성 파이프라인(synthesize_user 등) 미확인
- **판정**: ? UNKNOWN (코드 미확인)

---

## C. Query 생성 (P2) — align_pairs.jsonl

### C-1: align_pairs 생성 스크립트
- 생성 스크립트 미확인 (scripts/generate_pairs.py 등)
- **판정**: ? UNKNOWN (코드 미확인)

### C-2: align_pairs 소비 — train_align.py
- **SPEC**: query → positive_memory_id → provenance label
- **코드** (train_align.py: align_pairs 로딩): `positive_memory_id` 필드를 GT 레이블로 사용, routing accuracy도 이로 계산
- **판정**: ✓ MATCH (소비 측)

### C-3: eval에서의 query 출처
- eval_steered.py line 192: `_load_pseudo_queries(pairs_jsonl, mid_to_uid=mid_to_uid)` → align_pairs.jsonl에서 로드
- 훈련/평가 같은 파일에서 query 획득 → train query가 eval에도 사용됨
- **SPEC**: eval은 held-out item에 대한 메트릭 — eval의 prefix query 출처가 train pairs에서 온다면 train/eval 쿼리 구분 없음
- **판정**: ✓ MATCH (SPEC에 eval query 출처 별도 지정 없음 — 현행 방식 허용)

---

## D. Stage1 학습 (train_align.py) — src/training/losses.py

### D-1: InfoNCE 구현
- **SPEC** (plan.md §3 M4): `L_align = -log[exp(sim(hq,h+)/τ) / (exp(sim(hq,h+)/τ) + Σ_neg exp(sim(hq,h-)/τ))]`, in-batch + hard negative
- **코드** (losses.py:16-42):
  ```python
  inbatch_logits = (h_q @ h_pos.T) / tau   # [B, B]
  hard_logits = (h_q.unsqueeze(1) * h_hard_neg).sum(-1) / tau  # [B, K]
  all_logits = torch.cat([inbatch_logits, hard_logits], dim=1)  # [B, B+K]
  labels = torch.arange(B, device=h_q.device)
  return F.cross_entropy(all_logits, labels)
  ```
  대각선(row i, col i) = positive pair; 나머지 in-batch + hard_neg = negative
- **판정**: ✓ MATCH

### D-2: Stage1 encoder LR
- **코드** (train_align.py:603-605): `lr_encoder=2e-5`, `lr_head=1e-4`
- **SPEC**: (decisions.md) Stage1 lr_encoder 별도 지정 미확인 — 2e-5는 통상 합리적 값
- **판정**: ? UNKNOWN (decisions.md Stage1 LR 지정 재확인 필요)

### D-3: h_memory 소스 — frozen bank vectors
- **코드** (train_align.py): bank JSON에서 numpy 배열 로드, frozen (no_grad)
- SPEC §7 #11 on-the-fly re-encoding은 Stage2에만 해당
- **판정**: ✓ MATCH (Stage1은 frozen bank 사용 정상)

---

## E. Stage2 학습 (train_hybrid.py)

### E-1: Bug1 미전파 — lr_encoder default ★CRITICAL★
- **decisions.md**: Bug1 fix = `lr_enc` 2e-6 → 1e-5
- **코드** (train_hybrid.py:1235):
  ```python
  parser.add_argument("--lr_encoder", type=float, default=2e-6)
  ```
  argparse default가 여전히 `2e-6`
- **판정**: ✗ MISMATCH — **CRITICAL**  
  **Impact**: `--lr_encoder` 명시 없이 실행하면 Bug1 재현. text_encoder가 사실상 frozen → Stage2 text embedding이 업데이트 안 됨 → projector가 고정된 embedding으로 수렴 → 의도한 modality bridging 미작동. decisions.md에는 "fixed"로 기록되어 있어 운영자가 fix됐다고 착각하기 쉬움.

### E-2: Loss formula 불일치 — 미문서화 L_contrastive ★MAJOR★
- **SPEC** (plan.md): `L = L_retrieval + α·L_align`, α=0.1
- **코드** (train_hybrid.py:320):
  ```python
  loss = l_retrieval + alpha * (l_align + l_contrastive)
  ```
- **L_contrastive 정의** (train_hybrid.py:312-317): margin hinge loss
  ```python
  F.relu(0.1 - sim_correct + sim_wrong).mean()
  ```
  SPEC에 없는 항.
- **판정**: ✗ MISMATCH — **MAJOR**  
  **Impact**: 실제 학습 손실 = SPEC의 것과 다름. L_contrastive가 projector를 추가적으로 당기는 방향 → routing 정확도에 영향. decisions.md / plan.md 어디에도 이 항이 기재되지 않음 → 의도적 추가인지 실수인지 판단 불가.

### E-3: on-the-fly re-encoding
- **SPEC** (§7 #11): Stage2 각 스텝에서 현재 encoder로 h_query, h_memory 재인코딩
- **코드** (train_hybrid.py:287-289):
  ```python
  h_query  = text_encoder.encode(query_texts, is_query=True)
  h_memory = text_encoder.encode(memory_texts, is_query=False)
  ```
  cached embedding.vector 미사용, gradient 흐름 있음
- **판정**: ✓ MATCH

### E-4: SASRec frozen
- **코드** (train_hybrid.py:992): `p.requires_grad_(False)` for all SASRec params
- **판정**: ✓ MATCH

### E-5: LLaVA asymmetric LR 구조
- **SPEC**: SASRec frozen / text_encoder low-LR / projector high-LR
- **코드** (train_hybrid.py:~1230): SASRec frozen ✓, lr_encoder < lr_projector 구조 ✓ (E-1 이슈 별도)
- **판정**: ✓ MATCH (구조; 실제 LR 값은 E-1 이슈)

### E-6: Checkpoint 저장 키
- **코드** (train_hybrid.py:1186-1200): `text_encoder_state`, `projector_state`, `config` 저장
- **판정**: ✓ MATCH

---

## F. 평가 (M5) — src/eval/full_ranking.py, scripts/eval_steered.py

### F-1: Bug2 fix — prefix_fn 파라미터
- **코드** (full_ranking.py:22): `evaluate_full(... prefix_fn: Callable | None = None ...)`
- prefix_fn 전달 경로 (full_ranking.py:96-97):
  ```python
  pfx = prefix_fn(u, seq_np) if prefix_fn is not None else None
  log_feats, _ = model.log2feats(seq_np, prefix_embeds=pfx)
  ```
- **판정**: ✓ MATCH (Bug2 fix 확인)

### F-2: eval target — LOO held-out item
- **코드** (full_ranking.py): `target_dict[u][0]` = splits["users"][uid]["test"] or ["val"]
- SPEC의 provenance target이 아닌 LOO item으로 평가
- **판정**: ✓ MATCH

### F-3: prefix 경로 일관성 — train ↔ eval
- **훈련** (train_hybrid.py:287-289): `text_enc.encode() → h_q, h_m → projector → prefix`
- **eval** (eval_steered.py:82-93): `text_enc.encode(is_query=True) → h_q`, `text_enc.encode(is_query=False) → h_m → projector → prefix`
- **판정**: ✓ MATCH

### F-4: eval_steered.py checkpoint 로딩
- **코드** (eval_steered.py:120-122): `text_enc_state = ckpt.get("text_encoder_state")` → E-6 저장 키와 일치
- **판정**: ✓ MATCH

### F-5: sasrec.py predict() — prefix 미지원
- **코드** (sasrec.py:147): `self.log2feats(log_seqs)` — prefix_embeds=None 하드코딩
- eval_steered.py는 `log2feats` 직접 호출하므로 현행 eval path는 무관
- **판정**: ✗ MISMATCH — **MINOR**  
  **Impact**: `predict()` API로 steered inference를 시도하면 prefix 무시됨. 현재 eval 경로는 log2feats를 직접 쓰므로 실제 평가에는 영향 없으나, predict()를 사용하는 코드가 생기면 silent failure.

---

## G. 교차 영역 경로 일관성

### G-1: user_id 키 형식
- splits.json: `str(int_uid)` = "1", "2", ... (preprocess.py:234)
- bank.py: `uid_str = str(uid)` (line 202) — 동일 형식
- train_hybrid.py / eval_steered.py: str user_id 사용
- **판정**: ✓ MATCH (단, B-1 이슈로 bank.py가 splits를 잘못 읽음)

### G-2: item ID 공간
- preprocess.py: 1-indexed int item IDs
- SASRec padding = 0 (sasrec.py 내 item_emb 0번 = padding)
- splits.json train/val/test: int item IDs
- **판정**: ✓ MATCH

### G-3: embedding 공간 일관성
- Stage1 TextEncoderWithHead: bge-base → 768→768 projection (동일 공간)
- Stage2 TextEncoder: bge-base → 768 raw
- 두 stage가 같은 base 모델 사용하되, head 유무 차이 → bank 구축 시 어떤 encoder 사용했는지에 따라 Stage2 routing 품질에 영향
- bank에 저장된 embedding.vector의 source_model vs 학습에서 사용하는 encoder 일치 여부: **불명(코드 미확인)** — bank 생성 스크립트 미확인

### G-4: checkpoint 키 일관성
- train_hybrid.py 저장: `text_encoder_state`, `projector_state`, `config` ✓
- eval_steered.py 로딩: 동일 키 ✓
- **판정**: ✓ MATCH

---

## 종합 불일치 목록

| ID | 영역 | 심각도 | 파일:라인 | SPEC | 코드 | 영향 |
|----|------|--------|-----------|------|------|------|
| **E-1** | Stage2 | **CRITICAL** | train_hybrid.py:1235 | lr_enc=1e-5 (Bug1 fix) | default=2e-6 | --lr_encoder 미지정 시 encoder 사실상 frozen, Bug1 재현 |
| **E-2** | Stage2 | **MAJOR** | train_hybrid.py:312-320 | L=L_retr+α·L_align | L=L_retr+α·(L_align+L_contra) | 미문서화 L_contrastive 항이 손실에 포함, SPEC과 다른 gradient |
| **B-1** | Memory | **MAJOR** | bank.py:193-197 | leakage check 수행 | splits.json 스키마 오독 → 항상 skip | evidence leakage 무감지, user set validation 오동작 |
| **A-5** | Preprocess | **MINOR** | preprocess.py:341-345 | is_eligible에 is_discriminative 포함 | is_discriminative 조건 없음 | sequences.jsonl is_eligible ≠ SPEC 메모리 대상 정의 |
| **F-5** | Eval | **MINOR** | sasrec.py:147 | predict() → prefix 지원 | prefix_embeds=None 하드코딩 | predict() 사용 시 silent prefix 무시 (현행 eval path는 무관) |

### 확인된 매치 (코드 줄 입증)

| 항목 | 확인 코드 위치 |
|------|---------------|
| Bug3 fix (item_target_norm non-pad) | sasrec.py:99-101 |
| Bug2 fix (prefix_fn 파라미터) | full_ranking.py:22, 96-97 |
| InfoNCE 구현 | losses.py:32-42 |
| LOO split (test=last, val=penultimate) | preprocess.py:235-237 |
| 5-core 반복 수렴 | preprocess.py:88-112 |
| SASRec frozen in Stage2 | train_hybrid.py:992 |
| on-the-fly re-encoding | train_hybrid.py:287-289 |
| prefix injection position (P prepended, score from P: onward) | sasrec.py:105, 135 |
| projector 아키텍처 [2·d_text→256→GELU→d_sasrec, unsqueeze] | projector.py:24-40 |
| eval target = LOO held-out | full_ranking.py (target_dict[u][0]) |
| train/eval prefix path 동일 | eval_steered.py:82-93 vs train_hybrid.py:287-289 |
| checkpoint 키 일치 | train_hybrid.py:1186-1200, eval_steered.py:120-122 |
| bank K 분기 | bank.py:108-140 |

### 불명 (코드 미확인)

| 항목 | 이유 |
|------|------|
| align_pairs.jsonl 생성 로직 | 생성 스크립트 미확인 |
| is_discriminative 판정 로직 | synthesis pipeline 미확인 |
| bank embedding 생성 시 사용한 encoder | bank 생성 스크립트 미확인 |
| Stage1 lr_encoder 2e-5 SPEC 근거 | decisions.md 해당 항목 재확인 필요 |
