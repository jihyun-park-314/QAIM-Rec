# Phase A + B 구현 계획

## 작업 범위
- Phase A: 전처리 파이프라인 (M0 / F1) — Books, Beauty_and_Personal_Care
- Phase B: SASRec 백본 (F6a) — pmixer 통합 + prefix 훅 + full-ranking eval

---

## Phase A: 전처리

### 신규 파일

| 파일 | 역할 |
|---|---|
| `src/data/preprocess.py` | 핵심 전처리 로직 (A1~A6 전체) |
| `scripts/preprocess.py` | CLI 진입점 (thin wrapper) |
| `tests/test_split_consistency.py` | 모든 모듈이 동일 split 사용 assert |
| `tests/test_no_leakage.py` | val/test∉train, eligible⊂train-history assert |

### A1 — 로드·정제
- 필드명: `user_id`, `parent_asin`, `timestamp`, `rating`, `text` (sample.py 기존 헬퍼 재사용)
- (user, item) 중복 → 최초 timestamp 1개 보존
- `parent_asin` 또는 `timestamp` 없으면 skip (fail-loud는 파일 자체 없을 때만)

### A2 — 5-core (백본 표준)
- 수렴까지 반복: user 인터랙션 수 ≥5 AND item 인터랙션 수 ≥5
- rating·텍스트 무관 (SASRec 표준)

### A3 — 서브샘플 + 재 5-core
- 제안 N=20,000 (근거: 문헌 SASRec/BERT4Rec 범위 10K~50K, eligible ~5%이면 ~1K명 → LLM 호출 ~7K건)
- random.sample(users, N, seed=42)
- 서브샘플 후 5-core 재적용 → 최종 U < N 가능, 실제 볼륨 리포트

### A4 — ID remap
- user→1..U, item→1..I (0=padding)
- id_maps.json 양방향: `{"user2id": {...}, "id2user": {...}, "item2id": {...}, "id2item": {...}}`

### A5 — Global Split (source of truth)
- leave-one-out: test=마지막 item_id, val=마지막-1, train=나머지
- global temporal sanity: 전체 test target ts의 95th percentile = T_cut; test ts < T_cut이면 `temporal_flag: true`
- splits.json 스키마:
  ```json
  {
    "users": {"1": {"train": [1,2,3], "val": 4, "test": 5, "temporal_flag": false}},
    "meta": {"n_users": U, "n_items": I, "n_interactions": N,
              "global_temporal_cutoff_ts": T, "temporal_cutoff_pct": 0.95}
  }
  ```

### A6 — 산출물
1. **sasrec.txt**: `user_id item_id\n` per interaction, user 오름차순·각 user 내 timestamp 오름차순  
   (pmixer `data_partition()` 호환 포맷: `u, i = line.rstrip().split(' ')`)
2. **sequences.jsonl**: 유저당 1줄  
   ```json
   {"user_id": 1, "orig_user_id": "A1B2C3", "items": [
     {"item_id": 1, "orig_item_id": "B001", "ts": 123456, "rating": 4.0,
      "has_review": true, "n_words": 25, "is_eligible": true}
   ]}
   ```
3. **splits.json**: 위 A5 스키마
4. **id_maps.json**: 위 A4 스키마
5. **volume_report.json**: n_users/items/interactions raw→5core→subsample→final, avg/median seq len, density, eligible user ratio·interaction ratio

### Memory eligibility 태그 (추출 아님)
- `is_eligible`: rating≥4 AND len(text.split())≥10 AND 해당 interaction이 train history에 속함 (val/test target 제외)

### A7 — 테스트
- `test_split_consistency.py`: splits.json 로드 → train+val+test = 전체 시퀀스 assert; val/test 아이템이 train에 없음 assert
- `test_no_leakage.py`: 모든 eligible interaction의 item_id ∈ train (val_target, test_target 제외) assert

### CLI
```bash
python scripts/preprocess.py --category Books --n_users 200 --seed 42 --data_dir data/raw --out_dir data/processed
python scripts/preprocess.py --category Beauty_and_Personal_Care --n_users 200 --seed 42 ...
```
smoke는 `--n_users 200`으로 실행 (내가 실행).

---

## Phase B: SASRec 통합

### B1 — pmixer clone
```bash
git clone https://github.com/pmixer/SASRec.pytorch third_party/SASRec.pytorch
```
커밋 해시·의존성 기록: `third_party/SASRec.pytorch/INTEGRATION_NOTES.md`

### 신규 파일

| 파일 | 역할 |
|---|---|
| `src/models/sasrec.py` | pmixer 기반 + prefix 훅 + default_intent |
| `src/models/dataloader.py` | sasrec.txt + splits.json → DataLoader |
| `src/eval/full_ranking.py` | Recall/NDCG/MRR@{5,10,20} 전체 카탈로그 |
| `scripts/train_sasrec.py` | F6a 학습 CLI |

### B2 — 최소 변경 원칙

**Core 불변 (pmixer model.py 그대로):**
- `item_emb`, `pos_emb`, `emb_dropout`
- `attention_layernorms`, `attention_layers` (MultiheadAttention)
- `forward_layernorms`, `forward_layers` (PointWiseFeedForward)
- `last_layernorm`
- BPR loss 공식 (`pos_logits - neg_logits`)

**우리가 추가하는 것 (4가지뿐):**

1. **데이터 로더**: `src/models/dataloader.py` — sasrec.txt + splits.json 읽기, pmixer 원본 데이터 로더 대체
2. **full-ranking eval**: `src/eval/full_ranking.py` — 100-neg 샘플링 제거 → 전체 카탈로그 Recall/NDCG/MRR@{5,10,20}
3. **prefix 훅** (`forward(seq, prefix_embeds=None)`): None이면 완전 무동작 (F6a baseline은 None)
4. **learnable default_intent**: `self.default_intent = nn.Parameter(torch.zeros(1, hidden_units))` — F6a 학습 시 미사용 (구조만)

### Prefix 훅 아키텍처 (PPR 방식, right-aligned 보존)

```
[기존 log2feats 변경 없음까지]
item_embs = item_emb(log_seqs)       # [B, L, d]
item_embs *= sqrt(d)
item_embs += pos_emb(0..L-1)        # 표준 위치임베딩 (변경 없음)
item_embs = emb_dropout(item_embs)
item_embs *= ~padding_mask

[prefix 삽입 — prefix_embeds가 None이 아닐 때만]
seqs = cat([prefix_embeds, item_embs], dim=1)  # [B, P+L, d]
# 위치임베딩 없는 prefix를 앞에 concat (right-aligned 보존)

# 어텐션 마스크: [P+L, P+L] 인과 마스크 (표준 upper-triangular 그대로)
# → prefix에 item이 자유롭게 attention 가능, item 간 인과성 유지
attention_mask = ~tril(ones(P+L, P+L))  # 기존 생성 방식 동일, 크기만 확장

# 패딩 마스크 확장
extended_padding_mask = cat([zeros(B,P,bool), item_padding_mask], dim=1)
seqs *= ~extended_padding_mask.unsqueeze(-1)

# → 각 attention layer 통과 (코드 변경 없음)

# 손실 계산: log_feats[:, P:, :] (prefix 위치 제외)
# 평가 scoring: log_feats[:, -1, :] (마지막 = 마지막 item, 변경 없음)
```

**핵심 보존 사항**: 아이템 위치임베딩은 prefix 삽입 전에 적용 → SASRec의 right-aligned semantics 유지. prefix가 없으면(None) 코드 경로 완전히 동일.

### B3 — 동일성 점검 (필수 산출물)
- `diff src/models/sasrec.py third_party/SASRec.pytorch/model.py`로 core 변경 없음 입증
- 변경 표 (core 불변 + 우리 추가 4종) 정리
- smoke 시 prefix=None 조건에서 원본과 동일 loss 곡선 확인

### B4 — smoke 학습
- smoke 데이터 (200 유저 전처리 결과)로 few epoch 학습
- metric sane 여부 확인: Recall@10 > 0, loss 수렴 방향
- `checkpoints/{cat}/sasrec_pretrain.pt` 저장

### B5 — 전체 학습 명령어 (내가 실행)
```bash
# Beauty_and_Personal_Care 먼저
python scripts/train_sasrec.py \
  --category Beauty_and_Personal_Care \
  --data_dir data/processed \
  --maxlen 50 --hidden_units 256 --num_blocks 2 --num_heads 1 \
  --dropout_rate 0.2 --l2_emb 0.0 --batch_size 128 --num_epochs 200 \
  --device cuda:0

# Books
python scripts/train_sasrec.py \
  --category Books \
  --data_dir data/processed \
  --maxlen 50 --hidden_units 256 --num_blocks 2 --num_heads 1 \
  --dropout_rate 0.2 --l2_emb 0.0 --batch_size 128 --num_epochs 200 \
  --device cuda:0
```
예상 시간: Beauty ~2h, Books ~4-6h (4×2080Ti, 단일 GPU 기준; 추후 분산 옵션 추가 가능)

---

## 체크리스트

### Phase A
- [ ] `src/data/preprocess.py` 작성
- [ ] `scripts/preprocess.py` 작성
- [ ] `tests/test_split_consistency.py` 작성
- [ ] `tests/test_no_leakage.py` 작성
- [ ] smoke 실행 (200 유저) — 내가 실행
- [ ] 산출물 sane 확인

### Phase B
- [ ] `git clone` pmixer repo → `third_party/SASRec.pytorch`
- [ ] `src/models/sasrec.py` 작성
- [ ] `src/models/dataloader.py` 작성
- [ ] `src/eval/full_ranking.py` 작성
- [ ] `scripts/train_sasrec.py` 작성
- [ ] diff 동일성 점검 + 변경 표 작성
- [ ] smoke 학습 (200 유저, few epoch) — 내가 실행

---

## 의존성 (pip 명령어, 내가 설치)
```bash
pip install torch torchvision torchaudio  # PyTorch (CUDA 12.x)
pip install numpy scipy tqdm pyyaml
```
(임베딩 모델 자동 다운로드 없음 — B 단계에서 불필요)
