# QAIM-Rec: Intent-Memory-Steered SASRec — 설계 문서 (plan.md, 현재 확정 상태 · v0.4.22)

> 이 파일은 **현재 확정된 설계·구조·실행 계획**만 담는다. override 없이 그대로 읽으면 된다. **변경 이력·기각된 대안·버전별 사유는 `decisions.md`** 참조.
> 핵심 불변항: projector=`MLP(h_query,h_memory)` / 선택(selection)이지 fusion 아님 / 추론 시 온라인 LLM 0회 / 단일 global splits.json / leave-one-out / learnable `[DEFAULT_INTENT]`(진짜 cold) / 백본=전체 상호작용 5-core(rating 무관) / 메모리 eligibility=rating≥4 ∧ ≥10단어 ∧ is_discriminative(백본과 분리된 knob).

## 현재 상태 한눈에 (진행 위치)

**완료**:
- **메모리 구축(M2) 종결** — provenance 사태 해소. 재추출(53,707 레코드/9,807 유저, eligible_min=1, distinctiveness leakage 0.1%) → 클러스터링 → synth. f3_bank.jsonl(15,880 personal 유닛/9,705 유저, md5 8cfb1177…) **evidence.item_ids 빈 [] 0건**. prototype 15개(complete linkage) + K=0 fallback 861명. finalized bank md5 90a6ab1b…. K 분포: K=0 8.1% / K=1 49.9% / **K≥2 41.9%(4,428명, 헤드라인 모집단 충분)**.
- **F6a SASRec 사전학습** checkpoints/Books/sasrec_pretrain.pt(test NDCG@10=0.0247) — *재생성 금지*. splits.json md5 d2762a0…(10,566 유저/9,041 아이템) — *재생성 금지*.
- **P2 train 쿼리 생성 + 측정1** — C4 원문 프롬프트(입력=리뷰만, null 제거, title/author 비-leakage). provenance routing **floor 0.90~0.95**(분해로 검증: leak·lexical·길이 무관 — *동일인코더 일관성*이지 일반화 아님; P2 재료가 깨끗 + Stage1 좋은 초기화의 신호).

**핵심 방법론 결정(현재)**:
- **학습 = 2-stage**: **Stage1**(`train_align.py`)=text encoder를 InfoNCE contrastive로 학습(provenance positive + wrong-memory hard-neg, **LLM-judge 제거**). **Stage2**(`train_hybrid.py`)=projector(MLP) 학습+SASRec steering 동시, **LLaVA 비대칭 튜닝**(SASRec frozen / encoder 저-LR / **projector 강-LR** = modality gap 해소).
- **prefix P=1 유지**(PPR이 P=1으로 frozen SASRec steering 성공 입증). Pilot4 mini NO-GO는 메커니즘 결함 아니라 *미학습 projector*가 원인 → **controllability는 Stage2 *후* 측정**(게이트 아님).
- **routing accuracy GT = provenance self-consistent**(P3-judge·C4 아님). **circularity**: F8 절대성능은 *real-query 상한*으로 선제 고지, recovery@N(correct vs wrong)은 circularity-robust 헤드라인.
- **contribution(측정 후 확정)**: C(query-activated 통합 시스템) 1차 + B(wrong-memory hard-neg alignment, *분포 밖* query에서 frozen-bge 대비 증명) 종속 + A(recovery) 대표분석. 메커니즘 novelty 주장 금지(PPR로 crowded), 방어 novelty=combinatorial.

**Stage2 실행 + 두 측정버그 수정 (v0.4.15)**:
- Stage1(`stage1_align_best.pt`, routing 0.958, sim gap +47%) ✓, Stage2 학습 2회(2a=stage1-enc / 2b=raw-bge) ✓ — 파이프라인 공학 완성.
- **버그1 — Stage2 encoder 사실상 frozen**: lr_enc=2e-6(projector의 1/500) → encoder norm diff 0.07% → Stage2가 "projector만 학습한 반쪽". routing_cov(0.918)는 routing *품질*이 아니라 bank 커버리지 상수(9705/10566)였음(학습과 무관). → lr_enc 1e-5로 상향.
- **버그2 — 평가에 prefix 미주입**: `full_ranking.py evaluate_full()`이 `prefix_embeds=None` → 우리가 "steered"라 본 모든 Recall이 실은 vanilla. "C 미성립"·"J≈0.03 vs Recall 무변화 모순"은 *측정 불일치*였음(학습루프는 prefix 주입, 평가는 미주입 = 다른 경로). → `eval_steered.py`로 prefix 주입 평가 추가, 학습·평가 단일 경로 통일.
- **버그3 — prefix 스케일**: `sasrec.py` prefix rescale의 `item_target_norm`이 *padding 포함* 평균(0.2573, 85% padding) → prefix가 0.44% attention share(invisible)로 들어감. non-padding 평균(1.7236)으로 정정 → 16.7% share(의도 ~9% 범위). **저장된 모든 ckpt가 0.44% prefix로 학습됨 → projector가 방향 못 잡은 근본 원인.**
- **현재 측정(올바른 평가, 폭주 아닌 invisible 학습 ckpt)**: steered≈vanilla(R@10 0.048), per-user rank delta mean +1.97~+2.65이나 std 163/125로 **통계 유의 안 됨**(상쇄, improve≈degrade). = "강하게 흔들되 방향 랜덤" — 0.44% 학습의 예상 결과. 2b(raw-bge)≥2a(stage1)이나 *둘 다 버그 하 학습*이라 B 판정 불가.
- **표준 측정 규율**: steered 평가는 Recall delta 평균뿐 아니라 **improve/degrade/same 분포 + rank delta std/SE**를 항상 보고(평균만으론 "상쇄"와 "무효과" 구분 불가). prefix 들어가는 모든 측정은 단일 함수 경유(버그2 교훈).

**★근본병목 확정 (v0.4.21) — 원인 A(prefix→score 방향 gradient 부재), L_mem 제안**:
- 재포지셔닝(candidate recovery) 사망 폐기: Cross@100 0.69%(73명), augmentation gain 음수, new_slots 0(steered top-10이 vanilla top-90에 이미 다 있음).
- **근본병목**: prefix→score 전달 진단 — attenuation 1.44~1.78(>1)=prefix가 h_last에 *증폭 전달*(구조 정상). 단 **cos(Δh, item_emb_W) pos_frac=50%**(동전던지기)=prefix가 정답 W 방향과 무상관. **원인 A 확정**: 목적함수(L_retrieval+α·L_align)에 "correct memory가 wrong보다 정답 score 더 올려라" gradient 경로 없음. L_retrieval은 W가 LOO라 안 닿고, L_align은 prefix-space cosine이지 h_last→score 아님. → top-K 0 + B 죽음 + 방향 50% = 단일 원인.
- **다음: L_mem (score-space memory contrast)** — `L_mem=-log[exp(z_y(q,m+))/(exp(z_y(q,m+))+exp(z_y(q,m-)))]`, z_y=item_emb(W)·h_last(seq,prefix). SASRec score 기반(L_align의 cosine과 다름). 빠진 gradient 경로 복원. ★E-2 반면교사: 명문화+β 분리(--beta_mem)+ablation+W 누설 점검 필수.
- **판정**: 학습 후 cos(Δh,W) pos_frac 50%→>50%(방향) → Δz_y↑ → top-K↑. 단 방향 잡혀도 바닥효과면 top-K 0 가능(그땐 backbone 통상치 재검토). L_mem = 분기점.


**★v0.4.23 정정 — v0.4.22 self-reconstruction FORBIDDEN 철회 · main은 β 경험비교로 이연 · 진짜 게이트=code blocker**:
- **self-reconstruction FORBIDDEN 정정**: v0.4.22의 "`X_in_seq` 87.7% → self-reconstruction → FULL TRAINING FORBIDDEN"은 틀렸다. `L_mem`은 `m+`/`m-` 대조이고 `seq`가 둘 사이에 고정이라 in-seq baseline이 `Δz`에서 *상쇄*되고, Stage2는 **SASRec frozen**이라 seq 경로 학습 파라미터가 0개 → self-reconstruction *기계적 불가능*. 데이터 확증: in-seq `pos_frac` 0.584 ≈ out-seq 0.567(self-recon이면 in-seq가 훨씬 높아야). → FORBIDDEN의 self-reconstruction 근거 철회. (단 `L_mem` 경로 SASRec `requires_grad=False`·gradient가 prefix로만 흐름은 코드 재확인 전 "불명".)
- **진짜 게이트 = code-audit blocker**: (i) align_pairs manifest mismatch(52,236행 vs 45행), (ii) `source_text` vs `intent_description`(로더가 후자만 쓰면 `h_memory`가 달라져 모든 Δz 측정 오염 — *코드+데이터로 실태 확인 후* blocker/non-issue 판정), (iii) runtime guard 부재·stale P2·`--use_contrastive` 잔존. 학습은 이 blocker 해결 후.
- **main 변수 이연(사전 anoint 금지)**: `L_mem_X`(circular/retrieval, X-target SE14 정합=상한)도, `L_mem_Wtrain_aligned`(honest recommendation, A+B in-cluster ~49%·K≥2)도 *둘 다 측정-양수*(STEP1: X pos_frac 0.58, Wtrain A+B +0.018~0.029). 어느 쪽도 사전 dominant 아님 → `β_x`/`β_w` 별도 ablation으로 학습 후 비교해 확정. 논문 헤드라인 후보는 honest(`Wtrain_aligned`); `X`는 circularity로 mechanism/상한.
- **판정기준**: backbone NDCG@10 0.0247 = sparse-Amazon full-ranking 문헌밴드(0.017~0.037) 내 → floor 진짜. 성공 = honest-query 방향 획득(top-K 아님, floor 진단치와 부차보고).

**★L_mem 타깃 정합성 측정 (v0.4.22, ★main 판정은 v0.4.23에서 β 경험비교로 이연)** — 아래 "main=`L_mem_X`"는 v0.4.23에서 *보류*됨:
- **STEP1 사전측정(재학습 0)**: K≥2 샘플 2,000개에서 W_train 정합성 taxonomy 확인. A+B(`W_train`이 query source item `X`와 동일하거나 같은 provenance cluster) = 982개(49.1%), C+D(`W_train`이 다른 cluster이거나 memory provenance 없음) = 1,018개(50.9%). `W_train`-target contrast는 A+B에서 양수(2a +0.018 / 2b +0.029)였으나 C+D에서 음수(2a -0.022 / 2b -0.031). → **`W_train_all`은 gradient poison 위험으로 폐기**.
- **main 손실 = `L_mem_X`**: 원래 학습 정의를 유지한다. train query `q`는 source review item `X`에서 생성되고, `m+`는 `align_pairs.positive_memory_id` 기반 provenance memory, `m-`는 same-user different-cluster memory다. `L_mem_X = softplus(-(z_X(q,m+) - z_X(q,m-)))`, `z_X(q,m)=item_emb(X)·h_last(seq,prefix(q,m))`. **router top-1을 positive로 쓰지 않는다.** K=1은 intra-user negative가 없으므로 mask한다.
- **ablation 손실 = `L_mem_Wtrain_aligned`**: 추천 next-item target과 직접 연결해보는 보조 실험이다. `W_train=user_train[uid][-1]=pos_np[:,-1]`을 target으로 쓰되, **A+B(`mid(W_train)==mid(X)` 또는 `W_train==X`)에서만** 계산한다. C/D와 K=1은 mask한다. `W_train_all`은 구현·학습 금지.
- **loss/argparse 규율**: `L = L_retrieval + α·L_align + β_x·L_mem_X + β_w·L_mem_Wtrain_aligned`. `--beta_mem_x default 0.0`, `--beta_mem_wtrain default 0.0`. β=0일 때 기존 SPEC과 동일해야 하며, full training은 `β_x>0,β_w=0`(main)과 `β_x=0,β_w>0`(ablation)을 분리해 비교한다. combined는 두 단독 효과 확인 전 full training 금지.
- **★STEP2G 감사 완료 — FULL TRAINING: FORBIDDEN**: X_in_seq 감사(N=2,000 seed=42, `scripts/step2g_xinseq_audit.py`)와 β=0 backward-compatibility smoke 모두 완료. 결과: L_mem_X valid 1,146개 중 **87.7%(1,005개)가 X_in_seq=True**(B: X가 seq에 있지만 W_train과 다름). A(X==W_train)=12.3%(141개), C(not found)=0%. X_in_seq=True 87.7%는 self-reconstruction threshold 30%를 크게 초과 → **FULL TRAINING: FORBIDDEN**. β=0 compat OK(base mode L_mem_X=L_mem_Wtr=0.000000 확인). 해결안 A(X==W_train 한정)/B(seq_before_X)/C(L_mem_Wtrain_aligned 격상) 중 하나 결정 후 SPEC 갱신 시까지 재학습 금지.

**이전 (v0.4.20) 3층 평가**: 메커니즘 작동(X-target SE14)하나 top-K 효과≈0(Books 바닥효과 70.9% rank200+). B 죽음(2a≈2b). 저자명 leakage 아님(4겹). candidate-recovery 재포지셔닝 시도 → v0.4.21서 측정으로 사망.

**이전 (v0.4.16) — 전 과정 정적 감사**: E-1(lr_enc default 2e-6, CLI로 1e-5 명시해 직전 학습은 무영향), E-2(SPEC외 L_contrastive, 제거됨), B-1(leakage 검사 ts→item_id 재설계, 뱅크 clean 확정: 개인유닛 leakage 0, prototype-K0 우연충돌 10건은 추론무영향). 학습 버그 전부 수정 후 첫 공정 학습 완료(encoder drift 0.68%, L_contr=0, share 16.7%) — 단 그 위에서 v0.4.17의 설계 문제가 드러남.

---

## 0. Context

**문제 정의.** 순차추천(SASRec)은 사용자의 클릭/구매 시퀀스만으로 다음 아이템을 예측한다. 그러나 한 사용자는 시점·맥락에 따라 서로 다른 **"현재 질의로 활성화 가능한 맥락적 선호 단위(query-activatable contextual preference)"**를 갖는다(예: "캠핑용 장비를 찾는 나"와 "홈오피스를 꾸미는 나"는 같은 사람이지만 선호 축이 다름). 이 단위는 구매 목적(purchase purpose)만을 가리키지 않으며 — 기능적 용도, 취향/스타일, 콘텐츠 소비 맥락 등 질의가 활성화하는 모든 선호 차원을 포함한다. 시퀀스 전체를 평균화한 단일 사용자 표현은 이 맥락 전환을 표현하지 못하고, 매 질의마다 LLM을 호출해 재정렬하는 방식은 온라인 비용·지연이 크다.

**제안하는 해법.** 사용자의 과거 리뷰를 오프라인에서 LLM으로 분석해 "이 사람이 무엇을 위해 샀는가 + 그 목적에서 무엇을 선호했는가"를 단위로 갖는 **Intent Memory Bank**를 사용자별로 구축해 둔다. 실시간 질의가 들어오면 (1) 온라인 LLM 호출 없이) 텍스트 인코더로 질의를 임베딩(`h_query`)하고, (2) 메모리 뱅크와의 유사도로 가장 관련된 의도(top-1 또는 soft top-k)를 **선택(selection)**하고, (3) `h_query`와 선택된 의도의 메모리 임베딩(`h_memory`)을 함께 MLP projector에 입력해 `h_intent = MLP(h_query, h_memory)`를 SASRec의 표현 공간에 투영한 **prefix 토큰**으로 만들어 시퀀스 인코딩 앞에 붙인다. 이 prefix가 SASRec의 후보 생성(전체 카탈로그에 대한 스코어링) 단계 자체를 조건화(steering)한다 — reranking이나 feature-concat이 아니라 생성 단계 개입.

**왜 이 구조인가 (대안과의 차이).**
- vs. 단일 사용자 임베딩: 의도별로 분리된 메모리는 맥락 전환을 표현 가능 + per-intent 해석/평가 가능.
- vs. 잠재 multi-interest(ComiRec/MIND): 이들은 *아이템 ID 시퀀스*에서 캡슐/어텐션으로 잠재 다중의도 벡터를 뽑을 뿐 — 텍스트 근거·해석·*질의 활성화*가 없다. 본 연구는 리뷰 근거의 *해석 가능한* 의도를 *실시간 질의로 선택*한다.
- vs. 온라인 LLM 재정렬: 메모리는 정적 저장소(추론 시 LLM 호출 0회) — 지연/비용 없음.
- vs. **Persona4Rec(최근접 경쟁)**: Persona4Rec는 *아이템-side* 페르소나를 *사용자 이력으로 수동 선택*해 *reranking*한다. 본 연구는 (i) *유저-side* 맥락 메모리, (ii) *실시간 NL 질의 활성화*(수동 이력 선택이 아님 — PDF가 지적한 "다맥락이나 질의 활성화 없음" 공백을 정조준), (iii) reranking이 아닌 *생성 단계 steering*(reachability 실험으로 차별 입증)으로 구분된다.
- vs. feature-concat: concat은 모델이 "무시"하기 쉬운 추가 피처로 취급될 수 있는 반면, prefix는 self-attention 입력 자체를 바꿔 후보 생성 분포를 직접 이동시킨다 (가설 — Pilot 4에서 검증).

**현재 상태.** `QAIM-Rec/` 리포지토리에는 M0~M6 모듈 골격, 외부화된 프롬프트(`config/prompts/`), LLM 클라이언트+캐시, Pilot 1/2 스크립트가 구현되어 있으며, **Pilot 1(intent 추출 A/B)이 실행 완료**된 상태다(3개 도메인 parse 100%, discriminative 50~65%). 이 문서(v0.4)를 최종 설계 기준으로 하여 이후 구현·정정은 본 문서를 따른다.

**의도한 결과.** (a) Feasibility Pilot 4종의 결과로 K(사용자당 메모리 수), 데이터 카테고리, leakage 방지 강도, prefix 메커니즘의 타당성을 데이터로 확정하고, (b) go 판정 시 전체 파이프라인(LLM 추출 → 메모리 뱅크 → encoder/projector/SASRec 학습 → full-ranking 평가)을 실행할 수 있는 모듈 구조와 인터페이스를 마련한다.

## 0.1 PDF 대조 노트 (v0.3.2 신규 — `research_idea.pdf` 원문 대조)

이전 버전의 PDF 대조는 원문을 직접 읽지 않은 채 수행되었다. v0.3.2에서 `research_idea.pdf`(8슬라이드) 전체를 읽고 본 문서와 대조한 결과는 다음과 같다 — **§7의 확정 불변항을 뒤집을 결함은 없었음**.

**(A) Intent Memory 구축 메커니즘 — 의도적 분기(이미 승인됨)**
- PDF(p.4, 모듈1)는 "사용자의 구매 리뷰+상품 메타데이터를 LLM이 **한꺼번에** 읽고 귀납적으로 K개 메모리를 생성, K∈[2,k] 자동결정"(유저당 LLM 호출 1회)을 제안한다.
- 본 문서(§2.2)는 인터랙션별 P1 호출 → discriminative 필터 → 임베딩 기반 agglomerative clustering(τ_personal) → 클러스터별 synthesis(§7 #1, 회신으로 이미 승인)로 분기되어 있다.
- **판단**: 본 문서의 분기를 유지한다. 이유: (i) PDF의 "한꺼번에" 방식은 이력이 긴 유저에서 LLM 컨텍스트 한계에 부딫힐 수 있음, (ii) leakage 경계(history-only) 강제가 클러스터 단위가 단일 거대 프롬프트보다 쉬움, (iii) Pilot4의 필드 단위 wrong-intent swap(§7 #4)은 분해 구조 필요, (iv) threshold 기반 클러스터링은 Pilot2(§4)에서 실측·재현 가능하나 "LLM 자동결정 K"는 감사 불가.

**(B) "0. SASRec 학습" 단계 — 누락 발견, F6a로 추가(아래 §4)**
- PDF(p.3, p.6)는 파이프라인 최상단에 "0. SASRec 학습"(상품ID+구매log만으로 사전학습)을 별도 offline 단계로 명시한다. p.6의 "인사이트2(비대칭 튜닝, LLaVA 차용)"는 이 사전학습된 SASRec을 Stage2에서 frozen/극소-LR로 **보존**하는 것이 핵심 — 이 전제 없이는 freeze/low-LR 설계가 의미를 잃는다.
- 본 문서 §4 Stage F에는 F6(Stage1)/F7(Stage2)만 있고 이 사전학습 단계가 빠져 있었다(Pilot4는 자체 미니 버전을 갖고 있었으나 전체 파이프라인엔 대응 단계 없음).
- **반영**: §4 Stage F에 **F6a(SASRec vanilla pretraining)**를 추가 — F6a 체크포인트를 F7(steering base)과 F8(vanilla baseline)이 동일하게 공유하도록 하여 prefix 효과의 인과 분리를 보장한다(아래 §4).

**(C) 그 외 정합 확인 항목** — h_intent = MLP(h_query, h_memory)(PDF p.4/5 ↔ §2.1/§3 M3), top-1 selection·추론 시 LLM 0회(PDF p.5/p.2 ↔ §7 #5), L = L_retrieval + α·L_align(PDF p.4 ↔ M4 losses.py), "잘못된 메모리를 negative로"(PDF p.7 ↔ hard negative), "reranking 아닌 탐색 단계 개입"(PDF p.7 ↔ §0) — 모두 일치. `persona`(§2.1, v0.3)는 PDF엔 명시적 스키마가 없으나 PDF의 "K개 해석 가능한 맥락 기억"(Contribution2) 목표를 구체화한 것으로 모순 없음. Amazon-C4(PDF p.8 ↔ §6.5) 방향 일치 — 단 PDF의 "aggregation 가능 여부 ✓" 체크는 수치 근거가 없어 F1b에서 독립적으로 재확인 필요.

## 1. Repo 구조

```
QAIM-Rec/
├── config/
│   └── prompts/{dataset}/{domain}/{prompt}.txt  # 외부화된 LLM 프롬프트
│       ├── _default/              # 공통 기본값 (도메인 오버라이드 없으면 이 파일 사용)
│       │   ├── p1_base.txt
│       │   ├── p1_aspect.txt
│       │   ├── p2_pseudo_query_train.txt   # (v0.4.6) history 리뷰 → 1인칭 쿼리 (입장1, 누설0)
│       │   ├── p2_pseudo_query_eval.txt    # (v0.4.6) val/test 타깃 리뷰 → 1인칭 쿼리 (입장2, eval 전용)
│       │   ├── p3_align_judge.txt  # (v0.4.11 ablation 전용 — 핵심 파이프라인 미사용)
│       │   ├── synth_intent_attr.txt
│       │   └── synth_persona.txt
│       └── {domain}/              # 도메인별 오버라이드 (있으면 _default 대신 사용)
│
├── configs/                     # 모든 단계의 실험 설정 (yaml), 시드 포함
│   ├── data/{category}.yaml
│   ├── llm/{p1,p1_aspect,p2,p3}.yaml  # 모델명, temperature, self-consistency T 등
│   ├── memory/bank.yaml          # 클러스터링/K 설정
│   ├── model/{sasrec,encoder,projector}.yaml
│   ├── train/{align,hybrid}.yaml
│   └── eval/{full_ranking,ablations}.yaml
│
├── data/
│   ├── raw/{category}/           # 로컬 직접 배치 JSONL (reviews/meta, gitignore; HF 자동 다운로드 안 함)
│   ├── processed/{category}/
│   │   ├── sequences.jsonl        # 유저별 (item_id, ts, rating, review) 시퀀스
│   │   ├── splits.json            # train/val/test 인덱스 (leave-one-out + global temporal sanity)
│   │   ├── candidates.jsonl       # (v0.4.6) FROZEN: pre-P1 필터 통과 후보 리뷰(md5 고정·재생성 금지)
│   │   ├── p1_extractions.jsonl   # (v0.4.6) 단일 정본 P1 산출물 — item_id/ts/rating 1급 필드 + P1 출력
│   │   ├── p1_extractions.manifest.json # (v0.4.6) 행수·유저수·md5·prompt_version
│   │   ├── memory_bank/{user_id}.json  # Intent Memory Bank (유저별 샤딩; evidence.item_ids 채워짐)
│   │   ├── pseudo_queries.jsonl   # P2 출력 (train: history-grounded + memory_id 조회 / eval: target-grounded)
│   │   └── align_pairs.jsonl      # (v0.4.11) provenance 라벨: query, positive_memory_id, hard_negatives
│   └── llm_cache/llm_cache.sqlite # (v0.4.7 경로정정) (prompt_hash → response) 캐시, 모든 LLM 호출 경유 (구 data/cache/는 오기)
│
├── src/
│   ├── data/          # M0: (로컬)수급/필터/시퀀스/split
│   ├── llm/           # M1: 프롬프트 템플릿, LLM 클라이언트+캐시
│   ├── memory/        # M2: 클러스터링, 메모리 합성, 뱅크 저장/라우팅
│   ├── models/        # M3: SASRec, text encoder+router, MLP projector
│   ├── training/      # M4: loss, stage1(align)/stage2(hybrid) 학습 루프
│   ├── eval/          # M5: full-ranking 평가, 비교군, stratification
│   └── pilot/         # M6: Pilot 1~4 스크립트 + 리포트
│
├── scripts/            # CLI 진입점 (각 모듈을 config로 실행하는 thin wrapper)
├── results/{exp_name}/ # 메트릭 json, 리포트, 그림
├── checkpoints/{exp_name}/
└── tests/              # leakage probe, schema validation, 단위테스트
```

**설계 원칙.** 모든 단계는 `config -> 입력 경로 -> 출력 경로`의 순수 함수형 CLI로 분리(재현성: config에 시드 고정 + 버전 기록). LLM 호출은 전부 `src/llm/client.py`를 경유하며 캐시 키 = `hash(prompt_template_version, model_id, input_payload)` — 동일 입력 재실행 시 비용 0. Pilot은 `src/pilot/`에서 동일 모듈(M0~M2)을 소규모 config로 재사용 — 별도 구현 금지(파일럿용 임시 코드가 본 파이프라인과 분기되는 것을 방지).

## 2. 핵심 데이터 구조: Intent Memory Bank

### 2.1 단일 메모리 단위 — `IntentMemoryUnit`

**(v0.2)** PDF 정합을 위해 `embedding`을 라우팅(sim top-1)과 projector 입력(`h_intent = MLP(h_query, h_memory)`)에 공유하는 단일 표현 `h_memory`로 정의한다. **(v0.3)** `persona`를 `intent_description`/`preference_signal`과 형제 레벨의 코어 필드로 추가 — v0.1의 규칙 기반 persona_tag가 아니라 **LLM이 추론한 per-memory 맥락적 구매자 디스포지션**이며, `h_memory.source_text`에 포함되어 routing과 steering 양쪽에 쓰인다(§7 결정 #4).

```python
IntentMemoryUnit = {
  "memory_id": str,              # f"{user_id}::m{idx}" (prototype: "PROTOTYPE::{category}::p{idx}")
  "user_id": str,                # prototype memory는 "GLOBAL"

  # --- 사람이 읽는 의도 설명 (해석 / Pilot4 wrong-intent 합성 / stratify용) ---
  "intent_description": str,     # "이 묶음의 공통 목적·용도"를 1문장으로.

  # --- (v0.3 신규) per-memory 맥락적 구매자 디스포지션 — routing(h_memory) + steering(payload) 공용 ---
  "persona": {
    "tag": str | None,           # LLM이 생성하는 2~4단어 짧은 라벨 (고정 taxonomy 아님).
                                  # (v0.3.1) 클러스터 내 disposition_note(P1, §5)가 전부 null인
                                  # (=비변별적 디스포지션) 경우 "typical" 또는 None.
    "description": str,          # "이 구매 맥락에서 평소와 다른/특이한 구매 성향" 등을 담은 LLM 추론 1~2문장.
                                  # preference_signal.attributes로 환원되지 않는 상위 신호
                                  # (예: "이 맥락에선 가격보다 내구성 우선", "검증된 것만 vs 탐색적").
                                  # (v0.3.1) disposition_note가 전부 null인 클러스터는 ""(빈 문자열) —
                                  # "특이사항 없음" 류의 boilerplate를 채우지 않는다(아래 source_text 참조).
  },

  # --- steering payload ---
  "preference_signal": {
    "attributes": {
      "price_band": str,           # "budget" | "mid-range" | "premium" | "unknown"
      "feature_priorities": [str], # 최대 3개, 강조 순
      "brand_tendency": str,       # 구체 브랜드명 또는 "agnostic"
      "style": str | None,         # 디자인/미감 언급 시
      "avoid": [str] | None,       # (v0.3 선택) 이 의도에서 명시적으로 피하는 가격대/브랜드/특성 (negative steering). 없으면 None.
    },
    "summary": str,                # 위 속성들을 자연어 1~2문장으로 종합
  },

  # --- 근거 (해석/디버깅/leakage 검사용; prototype은 다수 유저의 대표 샘플) ---
  "evidence": {
    "item_ids": [str],             # (v0.4.6) 클러스터 멤버 item_id — p1_extractions의 1급 필드에서 채움(빈 [] 금지)
    "review_snippets": [str],      # 클러스터 대표 스니펫 (최대 2~3개)
    "timestamps": [int],           # 전부 train-history 구간 이내여야 함 (leakage 검사 대상). prototype은 빈 리스트 허용.
  },

  # --- h_memory: 라우팅(sim) + projector(h_intent = MLP(h_query, h_memory)) 공용 임베딩 ---
  "embedding": {
    "vector": list[float],
    "source_text": str,            # (v0.3.1) = " ".join(x for x in [intent_description, persona.description,
                                    #   preference_signal.summary] if x)  — persona.description==""이면
                                    #   해당 조각을 생략(빈 boilerplate가 routing 임베딩에서 intent_description의
                                    #   변별 신호를 희석하는 것을 방지, §7 #4).
    "model_id": str,
  },

  # --- 메타 ---
  "meta": {
    "category": str,
    "cluster_size": int,           # 이 메모리로 합쳐진 원본 로그 수 (prototype: 합산된 유저 수)
    "is_prototype": bool,          # population-level 공유 prototype 여부 (§2.2 fallback)
    "prompt_versions": {"p1": str, "synth": str},
  },
}
```

`IntentMemoryBank` (사용자당 1개 파일, `data/processed/{category}/memory_bank/{user_id}.json`):

```python
IntentMemoryBank = {
  "user_id": str,
  "k_personal": int,                # 사용자 자신의 클러스터에서 생성된 메모리 수
  "k_prototype": int,                # fallback으로 보완된 prototype 메모리 수 (정상 케이스 0)
  "history_length": int,             # 메모리 구축에 사용된 train-history 길이
  "memories": [IntentMemoryUnit, ...],  # personal + (필요시) prototype, 합쳐서 라우팅 후보
}
```

**persona (v0.3, 코어 필드).** §2.2의 per-cluster synthesis에서 `intent_description`/`preference_signal`과 동시에 LLM이 생성한다. stratified 분석(`src/eval/stratify.py`)은 `persona.tag`를 직접 그룹 키로 사용 — v0.2의 사후 규칙 기반 계산은 더 이상 필요하지 않다(§2.3/§3 M5).

### 2.2 구축 파이프라인 (구매로그 N개 → 메모리 K개)

**(v0.2 변경)** 권장#6 반영으로 (A) 개인별 메모리 구축, (B) population-level prototype 구축, (C) 유저별 뱅크 조립+fallback 3단계로 구성한다.

#### (A) 개인별 메모리 구축 (per-user personal memory)

```
[per-interaction]                    [grouping]                  [per-cluster synthesis]
(item_meta, review)                                                
   │  P1 (LLM, 1회/로그)              
   ▼                                  
{purpose, is_discriminative,         
 preference_attrs}  ──┐               
                      │  is_discriminative==False  → 클러스터링에서 제외
                      │  (모이면 별도 "generic 메모리" 1개로 유지할지는 §7 결정 #1과 연동)
                      ▼
            임베딩(purpose 텍스트) → 사용자 내 agglomerative clustering
            (distance threshold τ_personal, k_min=2 ≤ K_personal ≤ k_max=5)
                      │
                      ▼
            클러스터 1개 ──► IntentMemoryUnit 1개 (user_id=해당 유저, is_prototype=False)
              - preference_signal.attributes: 알고리즘적 집계 (LLM 호출 없음, 결합 호출의 입력으로 사용)
                  · price_band: 최빈값
                  · feature_priorities: union 후 빈도 상위 3개
                  · brand_tendency: 최빈 브랜드 (없으면 "agnostic")
                  · avoid: union (P1의 preference_attrs.avoid가 있는 경우만)
              - intent_description + preference_signal.summary (v0.3.2 변경: 결합 1회 호출):
                  cluster 내 purpose들 + 위 attributes를 입력으로, LLM이
                  {"intent_description": "...", "preference_signal_summary": "..."}를
                  strict-JSON으로 동시 생성(§3 M1 `synth_intent_attr_summary.py`).
                  두 필드는 같은 클러스터 컨텍스트에서 함께 생성되어야 서로 정합적이므로
                  분리 호출하지 않는다(이전 2회 분리 호출 → 1회로 통합, 비용식은 아래에서 갱신).
              - persona (v0.3 신규): cluster 내 disposition_note(P1, §5)들이 전부 null이면
                  LLM 호출 없이 {tag: "typical" 또는 None, description: ""}로 설정(비변별적 디스포지션,
                  v0.3.1). 하나 이상 non-null이면 해당 disposition_note들 + purpose/attributes를
                  LLM 1회로 종합 → {tag, description}(2~4단어 tag, 1~2문장 description),
                  §3 M1 `synth_persona.py`.
              - evidence: cluster item_ids + 대표 스니펫(중심에 가장 가까운 2개), timestamps
              - embedding: source_text = " ".join(non-empty among [intent_description, persona.description,
                  summary]) → h_memory (§2.1, v0.3.1)
```

#### (B) Population-level prototype 구축 (v0.2 신규, §3 M2 `prototypes.py`)

```
모든 유저의 discriminative purpose 임베딩 전체 풀(pool)
                      │
                      ▼
            population-level agglomerative clustering
            (distance threshold τ_global, 별도 후보 산정)
                      │
                      ▼
            상위 P≈8~15개 클러스터(크기 기준) ──► IntentMemoryUnit 1개씩
              - memory_id: "PROTOTYPE::{category}::p{idx}"
              - user_id: "GLOBAL", is_prototype=True
              - (A)와 동일한 synth.py 로직 재사용(intent_description/persona/attributes/summary/embedding)
              - evidence.item_ids/review_snippets: 클러스터 대표 샘플(여러 유저에서 추출), timestamps=[] 허용
              - meta.cluster_size: 클러스터에 포함된 (유저, 로그) 수
                      │
                      ▼
            data/processed/{category}/memory_bank/_prototypes.json (카테고리당 1개 파일)
```

#### (C) 유저별 뱅크 조립 + fallback (`bank.assemble`, §3 M2 `bank.py`)

```
IntentMemoryBank.memories = personal 메모리 (K_personal개, (A)의 출력)

if K_personal >= k_min(=2):
    k_prototype = 0  (보충 불필요)
else:
    anchor = K_personal > 0 인 경우: 해당 유저 discriminative purpose 임베딩의 평균
             K_personal == 0 인 경우: 전체 구매이력(purchase-history) 임베딩의 centroid
    _prototypes.json에서 anchor와 cosine sim 상위 (k_min - K_personal)개 prototype 추가
    k_prototype = 추가된 prototype 개수

memories = personal ++ 추가된 prototype  → IntentMemoryBank.memories (라우팅 후보 전체)
```

**LLM 호출 비용 (v0.3.2: intent_description+preference_signal.summary 결합 호출로 갱신)**
- (A) 사용자 1명, 히스토리 N개, 클러스터 K_personal개 기준: P1 N회(v0.3: disposition_note 포함, 추가 호출 없음) + (intent_description+preference_signal.summary 결합) K_personal회 + (persona synthesis) K'_personal회(K'_personal ≤ K_personal — v0.3.1: disposition_note가 전부 null인 클러스터는 LLM 호출 없이 {tag:"typical"/None, description:""} 처리) ≈ N + K_personal + K'_personal회 (상한 N + 2·K_personal회).
- (B) 카테고리당 1회: prototype P개에 대해 (결합 호출) P회 + (persona synthesis) P'회(P' ≤ P, 동일 조건) ≈ P + P'회 (상한 2P회) — 전체 유저 풀에 대해 한 번만 수행.
- (C)는 임베딩 유사도 계산만(로컬, LLM 호출 없음).
- 캐시로 재실행 비용 0.

**중요 — leakage 방지.**
- (A): 메모리 구축은 **train-history 구간(=val/test 타깃 제외)**의 인터랙션만 사용한다. `evidence.timestamps`는 전부 해당 유저의 split cutoff 이전이어야 하며, 이는 §6 split 설계 및 테스트(`tests/test_no_leakage.py`)로 강제.
- (B): prototype은 (A)와 동일하게 **각 유저의 train-history 구간에서 이미 필터링된** discriminative purpose 텍스트만을 풀(pool)로 사용 — 개별 유저 단위 cutoff가 이미 적용된 입력을 합치는 것이므로 별도의 global cutoff가 추가로 필요하지 않음. prototype의 `evidence.timestamps=[]`는 "다수 유저의 대표 샘플이라 단일 시점으로 표현 불가"를 의미하며 leakage 검사 대상에서 제외(빈 리스트 허용, §2.1).

### 2.3 구조화 vs 자유 텍스트 — 둘 다 보존하는 이유

- `attributes`(구조화): (a) Pilot 4의 "올바른/틀린 intent" 합성에 필드 단위 swap 가능, (b) per-attribute stratified 분석(§3 M5 `stratify.py`)의 입력, (c) projector 없이도 빠른 rule-based ablation 가능.
- `summary`(자유 텍스트): `intent_description`, `persona.description`과 결합되어 `embedding.source_text`(=h_memory)를 구성 — LLM이 자연스럽게 종합한 문장이 구조화 필드 나열보다 임베딩 품질이 좋을 가능성이 높음(가설, description-only와의 비교는 §7 결정 #3에서 ablation).
- 두 표현은 **항상 동일한 출처(같은 클러스터)에서 동시에 생성**되어 서로 다른 합성 호출 간 불일치가 없도록 한다.
- **(v0.2)** 이렇게 결합된 `h_memory`는 (i) 라우팅(query `h_query`와의 sim top-1)과 (ii) projector 입력(`h_intent = MLP(h_query, h_memory)`, §2.1/§3 M3)을 **동시에** 담당하는 단일 표현이다 — 별도의 라우팅용/projector용 임베딩을 두지 않음(재현성·일관성).
- **(v0.3 신규) `persona`**: `intent_description`(이 묶음의 목적)·`preference_signal`(속성/선호)과는 별도로, **"이 맥락에서 이 사람이 어떤 종류의 구매자가 되는가"**라는 상위 신호를 LLM이 직접 추론한다 — attributes의 재인코딩이 아니라 P1의 `disposition_note`(§5, "평소와 다른/특이한 구매 성향" 질문)를 클러스터 단위로 종합한 결과. Pilot 4의 wrong-intent 합성에서는 `intent_description`/`attributes`와 함께 `persona`도 swap 대상이다(다른 클러스터의 persona로 교체해야 "잘못된 디스포지션"이 합성됨, §4 Stage P4).

## 3. 모듈 분해와 인터페이스

각 모듈은 `config(yaml) + 입력경로 → 출력경로`의 CLI. 의존성은 화살표(M0 → M1 → M2 → M4, M3는 M4가 사용).

### M0 — Data (`src/data/`)
- (데이터 수급): Amazon Reviews 2023의 선택 카테고리 review+meta를 **사용자가 직접 로컬에 배치**(`data/raw/{category}/{reviews,meta}.jsonl`). 코드의 HF 자동 다운로드는 하지 않으며, 파일이 없으면 즉시 실패(fail-loud).
- `filter.py`: k-core 필터(§6) → `processed/{category}/sequences_raw.jsonl`
- `sequence.py`: 유저별 (item_id, ts, rating, review_text, item_meta) 시간순 시퀀스 생성
- `split.py`: leave-one-out + global temporal sanity → `processed/{category}/splits.json`
  - **출력**: 유저별 `{history: [...], val_target, test_target}` 인덱스
- **입력**: 로컬 `data/raw/{category}/{reviews,meta}.jsonl`, category명, 필터 파라미터 / **출력**: `sequences.jsonl`, `splits.json`

### M1 — LLM Pipeline (`src/llm/`)
- `client.py`: **Ollama**(`http://localhost:11434`) wrapper — JSON-schema constrained decoding + `think:false`(top-level), sqlite 캐시(`prompt_hash -> response`). (§7 #7)
- `prompts/p1_intent.py`, `p2_pseudo_query.py`, `p3_align_judge.py`: 템플릿 + JSON 파싱/검증/재시도(파싱 실패시 최대 2회 재시도, 최종 실패는 별도 로그). **(v0.3)** `p1_intent.py`는 `disposition_note`(§5) 필드를 추가로 출력 — M2 `synth.py`의 persona synthesis 입력으로 사용. **(v0.4.6)** P1 출력 레코드는 `item_id`/`timestamp`/`rating`을 1급 필드로 함께 기록하고, 병렬 워커 shard는 즉시 단일 정본 `p1_extractions.jsonl`(+manifest md5)로 머지 — 다운스트림은 이 정본만 읽음. `p2_pseudo_query.py`는 **단일 프롬프트**(`p2_pseudo_query.txt`, C4 원문+JSON, v0.4.12)를 2-모드(train=history 리뷰+provenance positive / eval=타깃 리뷰+pair 없음)로 로드. 입력=리뷰 본문(title/author 비-leakage, v0.4.12 — G3는 info 보고).
- `prompts/synth_intent_attr_summary.py` **(v0.3.2 신규)**: 클러스터 내 purpose들 + 알고리즘 집계된 attributes를 입력으로 `{"intent_description": str, "preference_signal_summary": str}`를 1회 결합 호출로 strict-JSON 생성(§2.2 (A), M2 `synth.py`에서 사용).
- `prompts/synth_persona.py` **(v0.3 신규)**: 클러스터 내 non-null `disposition_note`(들) + purpose/attributes를 입력으로 `{"tag": str, "description": str}`를 생성(§2.2 (A)/§7 결정 #4, M2 `synth.py`에서 사용. disposition_note 전부 null인 클러스터는 호출 생략).
- **입력**: (item_meta, review) 배치 또는 (query, memory) 배치 / **출력**: 파싱된 dict 리스트 + 실패율 리포트
- M0 → M1 의존 (item_meta, review 필요)

### M2 — Memory (`src/memory/`)
- `cluster.py`: 유저별 P1 결과(discriminative만) → agglomerative clustering(τ_personal) → K_personal개 그룹. **(v0.4.6)** 입력 = 단일 정본 `p1_extractions.jsonl`(item_id 1급 필드). 각 클러스터는 멤버 레코드의 `item_id`/`timestamp`를 **보존**(cluster_summaries에 source_texts만 남기지 않음) → `synth.py`가 `evidence.item_ids`/`timestamps`를 채움. `item_id ↔ cluster_label` 매핑을 저장해 P2 train query의 memory_id 조회·leakage timestamp 검사에 사용.
- `synth.py`: 그룹 → `IntentMemoryUnit` (§2.2 (A), v0.3.2): preference_signal.attributes는 알고리즘적 집계(LLM 호출 없음) → `prompts/synth_intent_attr_summary.py`로 intent_description/preference_signal.summary를 1회 결합 호출로 생성 → (disposition_note 비-null 시) `prompts/synth_persona.py`로 persona 1회 추가 호출(전부 null이면 호출 없이 `{tag:"typical"/None, description:""}`) → 클러스터당 최대 2회(LLM). `embedding.source_text = " ".join(non-empty among [intent_description, persona.description, summary])`
- `prototypes.py` **(v0.2 신규)**: 전체 유저의 discriminative `purpose` 임베딩을 모아 population-level agglomerative clustering(별도 threshold τ_global) → 상위 P≈8~15개 클러스터를 `is_prototype=True` `IntentMemoryUnit`(`memory_id="PROTOTYPE::{category}::p{idx}"`, `user_id="GLOBAL"`)으로 합성 → `processed/{category}/memory_bank/_prototypes.json`. `synth.py`와 동일한 합성 로직 재사용.
- `embed.py`: sentence-embedding 모델 wrapper (h_memory 생성 — routing + projector 공용). `encode(text) -> vector` 외에 **`embed_purchase_history(user_history) -> vector`** (v0.3.2 신규, §2.2 (C)/§7 결정 #1): 유저 train-history 아이템들의 `(title+category)` 텍스트를 동일 인코더(§7 결정 #8, bge-base)로 임베딩 후 평균 — purpose 임베딩과 동일 공간이어야 prototype centroid와의 cosine sim 비교가 성립하므로 동일 인코더 사용이 필수. K_personal==0 유저의 prototype fallback anchor로 사용(§2.2 (C)).
- `bank.py`: `IntentMemoryBank.load(user_id)`, `.save()`, `.route(query_emb, top_k=1) -> List[IntentMemoryUnit]` (cosine sim, personal+prototype 후보 전체 대상). **`assemble(user_id, k_min=2)`**: K_personal < k_min이면 `_prototypes.json`에서 해당 유저의 (discriminative purpose 평균 임베딩, K_personal==0이면 전체 구매이력 임베딩 centroid)과 가장 가까운 prototype (k_min - K_personal)개를 추가해 `k_prototype` 채움.
- **입력**: M1 출력(P1 결과) + M0 split(`history`만) / **출력**: `processed/{category}/memory_bank/{user_id}.json`, `processed/{category}/memory_bank/_prototypes.json`
- M0, M1 → M2 의존

### M3 — Models (`src/models/`)
- `sasrec.py`: 표준 SASRec (item embedding table, causal self-attention stack, 마지막 hidden state로 전체 카탈로그 dot-product scoring). **prefix injection point 노출**: `forward(seq_item_ids, prefix_embeds=None)` — `prefix_embeds`(P×d)가 주어지면 입력 시퀀스 앞에 concat 후 self-attention(단, 출력 스코어링/loss 위치에는 포함 안 함).
- `text_encoder.py`: 사전학습 sentence encoder + 학습가능 projection head (L_align 대상) — `encode(text) -> emb_q`
- `router.py`: `encode(query) -> emb_q = h_query`, `bank.route(h_query, top_k)` 호출(= `h_memory` 후보들과 cosine sim), top-1 또는 soft top-k 가중합 결정 (selection 정책, §7 결정은 top-1 고정/soft 비교)
- `projector.py` **(v0.2 변경)**: MLP, `forward(h_query, h_memory) -> h_intent`(= `prefix_embeds`, P×d_sasrec). `h_query`=router의 `encode(query)` 출력, `h_memory`=routed `IntentMemoryUnit.embedding.vector`(라우팅에 쓴 것과 동일 벡터). 두 임베딩은 concat 후 MLP 입력으로 사용(차원: `2*d_emb -> ... -> P*d_sasrec`). **(v0.3)** `h_memory`의 `source_text`에 `persona.description`이 포함되므로(§2.1), `h_intent`에는 persona 정보가 자연히 반영된다 — projector 구조 자체의 변경은 없음.
- **의존성**: M3는 독립 구현 가능(메모리 뱅크는 인터페이스 mock으로 테스트), M4 학습 시 M2 산출물 필요

### M4 — Training (`src/training/`)
- `losses.py`: `info_nce(emb_q, emb_pos, emb_neg)` (L_align), `retrieval_loss(scores, target_item, full_catalog)` (L_retrieval — full-softmax 또는 sampled-softmax+all-item 정규화, §7에서 확정), **`score_space_memory_contrast`(v0.4.22, L_mem 계열)**.
- `train_align.py` (Stage 1): **(v0.4.11)** provenance 결정론 라벨 `align_pairs.jsonl`(query, positive_memory_id=provenance 조회, hard_negatives)로 text_encoder(+router 임베딩 공간) supervised contrastive(InfoNCE) 학습. SASRec/projector 미사용. **`h_memory`는 고정(frozen encoder 생성값)**. L_align 배치: 1 positive당 (i) same-user 다른 메모리 전부 hard negative(K-1≤4) + (ii) in-batch negative. (P3 LLM-judge 라벨 미사용 — judge 제거.)
- `train_hybrid.py` (Stage 2): F6a 체크포인트(`checkpoints/{exp_name}/sasrec_pretrain.pt`)에서 SASRec을 로드(**기본값: frozen**, 1e-2x LR은 ablation) + text_encoder(저-LR, F6 체크포인트에서 초기화) + projector(**강-LR — v0.4.14 LLaVA 비대칭 튜닝: 인코더 저-LR / projector 강-LR로 modality gap 해소 주력, catastrophic forgetting 방지**) 동시 학습. 기본 loss는 `L = L_retrieval + α·L_align`이며, v0.4.22 이후 **선택적 score-space memory contrast**를 별도 β로 추가한다: `L = L_retrieval + α·L_align + β_x·L_mem_X + β_w·L_mem_Wtrain_aligned`. 기본값은 `β_x=β_w=0`으로 기존 SPEC과 동일해야 한다. `L_mem_X`(`q`의 source item `X` + provenance `m+`)와 `L_mem_Wtrain_aligned`(A+B에서 `W_train`이 `m+`와 정합)는 **(v0.4.23) 둘 다 측정-양수인 ablation 후보로, main은 β 학습 비교 후 확정**(사전 anoint 금지). M2 메모리 뱅크에서 매 스텝 유저의 top-1 메모리를 router로 선택해 inference path를 유지하되, **L_mem의 positive는 router top-1이 아니라 provenance `positive_memory_id`**를 사용한다. `m-`=same-user different-cluster, K=1 mask(L_mem은 사실상 K≥2 목적함수). `h_query`, `h_memory` 둘 다 **on-the-fly로 현재 text_encoder를 통해 재인코딩**(§7 결정 #11) → `projector(h_query, h_memory)` → prefix(P=1) → SASRec. **(v0.4.23 gate)** full training 전 통과해야 할 것은 ~~`X_in_seq` self-reconstruction 감사~~(v0.4.23 철회 — 대조+frozen에서 void)가 아니라 **code-audit blocker**(align_pairs manifest 일치 · `embedding.source_text` 사용 · forbidden-β runtime guard · stale P2 차단 · hidden `--use_contrastive` guard)와 β=0 backward-compatibility다.
- **입력**: M0 splits(train), M2 memory_bank, M1 align_pairs / **출력**: `checkpoints/{exp_name}/{stage}_best.pt`

### M5 — Eval (`src/eval/`)
- `full_ranking.py`: 전체 카탈로그에 대해 Recall@{5,10,20}/NDCG@{5,10,20}/MRR 계산 (sampled-negative 금지)
- `conditions.py`: 비교군 정의 — `{none, avg_intent, selected_intent} × {prefix, concat}` + `{correct_intent, wrong_intent}` + `query_only(leakage baseline)`. 각 condition은 "prefix_embeds를 어떻게 만드는가"의 함수로 통일 구현(코드 중복 방지). **(v0.3.2)** `vanilla_finetuned` **(조건부)**: F7에서 SASRec frozen(기본값)이면 `none`(=F6a 체크포인트, 무prefix)이 곧 공정한 비교 기준이므로 불필요. F7 ablation(1e-2x LR, SASRec 갱신)을 실행하는 경우에만, F7과 동일한 step/LR로 SASRec을 prefix 없이 추가 학습한 `vanilla_finetuned`을 F8에 추가 — steering 효과와 추가 finetuning 효과를 분리.
- `recovery_analysis.py` **(v0.2 신규, 권장#7 headline 실험)**: `correct_intent` / `vanilla(no-prefix)` / `wrong_intent` 3-조건의 top-N 추천 집합을 비교해 `Recovery@N = P(target ∈ TopN_steered(correct) AND target ∉ TopN_vanilla)`를 계산. `conditions.py`의 동일 prefix-생성 함수를 재사용.
- `stratify.py`: history-length bucket별, `persona.tag`별(v0.3: 코어 필드를 직접 그룹 키로 사용, 사후계산 불필요), per-memory(per-intent) recall, **routing accuracy(v0.4.10: provenance self-consistent GT — P2 train 쿼리 q[리뷰 r에서 생성] → r의 item_id → provenance cluster_label = 정답 memory. router top-1 == 그 cluster 비율. P3 라벨·C4 정합 *아님*. P3는 학습 신호 전용)**, diversity(추천 리스트 내 카테고리/브랜드 엔트로피), latency(prefix 계산 ~ scoring 전체).
- **Amazon-C4 (v0.4.10 정정 — routing GT 아님, 선택적 OOD 참고)**: ~~C4를 정량 routing 평가로 격상~~ 폐기. C4는 (i) query→*item* 검색이지 query→*memory* 라우팅이 아니고 (ii) C4 유저가 우리 splits 유저와 거의 안 겹쳐 routing GT 부적합. **routing accuracy의 GT = provenance self-consistent**(위 `stratify.py`). C4는 *프롬프트 스타일 참조*(P2가 이미 차용) + *선택적* OOD 쿼리 robustness 질적 비교로만 사용(주 지표 아님).
- **입력**: checkpoint + memory_bank + splits(test) + (Amazon-C4 서브셋) / **출력**: `results/{exp_name}/metrics.json`, `results/{exp_name}/stratified.csv`, `results/{exp_name}/recovery.json`, `results/{exp_name}/c4_routing_eval.json`

### M6 — Pilot (`src/pilot/`)
- `pilot1_intent_extraction.py`, `pilot2_facet_distribution.py`, `pilot3_leakage_probe.py`, `pilot4_steering_sanity.py`
- 각 스크립트는 M0~M5의 함수를 소규모 config로 재호출(별도 구현 금지) + go/no-go 판정 출력(`results/pilot/pilotN_report.json`)

## 4. 단계별 마일스톤 (Pilot → Gate → Full Pipeline → Eval)

### Stage P1 — Intent 추출 파일럿 (A/B 비교)

**핵심 프레이밍.** intent = "현재 질의로 활성화 가능한 사용자의 맥락적 선호 단위(query-activatable contextual preference)". 구매 목적(purchase purpose)은 여러 aspect 중 하나일 뿐 전체 정의가 아니다. 헤드라인: query-activation + interpretable + generation-steering.

**A/B 구조.** 동일 샘플·동일 LLM, 프롬프트만 변수.
- **p1_base**: 기존 §5 purpose 중심 스키마 = narrow baseline. 변경 없음.
- **p1_aspect**: 도메인 일반 schema (§5 p1_aspect 참조). 새로운 스키마로 멀티 aspect 포착.

**도메인 비교.** 실제 보유한 3개로 A/B를 돌린다(기능형 미보유, §6.1):
- 라이프스타일: `Amazon_Fashion`, `Beauty_and_Personal_Care`
- 콘텐츠: `Books`
목표: "어느 도메인이 변별·aspect·멀티맥락(및 per-user 커버리지)이 풍부한지" 실측.

**입력**: 각 카테고리 `data/raw/{category}/reviews.jsonl` + `meta.jsonl` (로컬만, HF 다운로드 금지). 파일 없으면 즉시 실패.

**실행 순서**:
1. N=3 smoke → 파이프라인 동작 확인 (`--n_samples 3`)
2. 동일 config 2회차 → cache hit 확인
3. N=20 small comparison → 방향성 점검 (승인 없이 가능)
4. **N=200은 승인 전 절대 실행 금지**

**실행**: M1/`p1_intent.py` 또는 `p1_aspect.py`만 실행 (M2 클러스터링 없음)

**출력**: `results/pilot/pilot1_{base|aspect}_{category}.json` — 아래 per-cell 리포트 지표.

**리포트 지표 (카테고리 × 프롬프트 셀별)**:
- `parse_success_rate`, `discriminative_ratio`
- `aspect_coverage_ratio` (p1_aspect 전용: aspect_coverage 4필드 중 ≥2개 non-null/non-empty 비율)
- `discriminative_and_aspect_valid_ratio`
- `disposition_note_nonnull_ratio`
- `intent_length_stats` (purpose/contextual_intent[0] 단어 수 분포)
- `mean/median cache-miss latency`, `mean cache-hit latency`
- `estimated_n200_time` = cache-miss mean × 200 (cache-hit latency 사용 금지)
- `failed_samples` (실패 인덱스)
- 원시 LLM 출력 2~3개 (`raw_outputs`) — 스키마가 실제로 멀티맥락·taste를 잡는지 육안 확인용

**성공 기준 (go)**: 파싱 성공률 ≥ 95% **and** discriminative 비율 ≥ 40% (N=20은 방향성 점검; 확정 go/no-go는 N=200에서)

**중단/조정 기준**: discriminative 비율 < 40% → ① 카테고리 교체 또는 ② 프롬프트 재시도(최대 2회)

**예상 비용**: LLM 호출 20회/셀 (smoke=3회), 전체 3도메인×2프롬프트 = 최대 120회 (N=20 기준)

### Stage P2 — 사용자 facet 분포 파일럿
- **입력**: Pilot 1의 P1 결과를 이력 ≥L(예: L=8)인 사용자 N≈300~500명으로 확장 실행 (M0 필터 + M1 P1)
- **실행**: discriminative 로그만으로 M2/`cluster.py` 실행(임베딩+agglomerative, threshold τ_personal 후보 2~3개로 스윕). **(v0.2 추가)** 동시에 M2/`prototypes.py`를 같은 후보군 전체에 대해 1회 실행(τ_global 1개 후보로 우선 산출)하여 population-level prototype 클러스터 수/크기 분포도 함께 산출.
- **출력**: `results/pilot/pilot2_report.json` — τ_personal별 "사용자당 클러스터 수 K_personal" 분포, "K_personal≥2인 사용자 비율(=fallback 불필요 비율)", **prototype 클러스터 수 및 "fallback이 필요한 사용자(K_personal < k_min) 비율"**
- **성공 기준 (go)**: 적절한 τ_personal에서 이력≥L 사용자 중 K_personal≥2 비율 ≥ 30% **이거나**, fallback 비율이 높더라도 prototype 클러스터가 의미 있게 분리되는 경우(=fallback이 실제로 사용 가능한 형태로 동작) — 즉 K_personal≥2 OR (K_personal<2 AND 적합한 prototype 존재) 비율 ≥ 30%
- **결정에 미치는 영향**: 이 결과로 §7 결정 #1의 (a)+(c) 조합(agglomerative + prototype fallback, 결정 회신으로 이미 확정)에서 fallback 발동 빈도를 실측 — F3 LLM 호출량(F3 비용 추정)에 직접 반영
- **예상 비용**: 추가 LLM 호출 ≈ (N × 평균 이력) + 2·K_personal(요약, v0.3.2: intent_description+preference_signal.summary 결합 1회 + persona 1회 = 최대 2회/클러스터) + 2P(prototype, P≈8~15, 동일), 임베딩은 로컬 모델로 GPU-경량. 실제 호출 수는 disposition_note non-null 비율에 따라 감소(§2.2)

### Stage P3 — Leakage Probe (v0.2: leakage floor로 재정의)
- **입력**: Pilot 2 대상 사용자 일부(N≈100)에 대해 M1/`p2_pseudo_query.py`로 **decontaminated pseudo-query**(브랜드/모델명/정확 스펙 마스킹 완료, P2 검증 통과분만) 생성. P1의 `purpose` 텍스트는 식별정보 마스킹이 보장되지 않으므로 **사용하지 않음**(요구사항 [필수]#3).
- **실행**: pseudo-query 임베딩으로 **전체 카탈로그**에서 해당 유저의 test target item을 직접 검색 → `R_query_only`(Recall@10/50)
- **출력**: `results/pilot/pilot3_report.json` — `R_query_only` 값 + P2 마스킹 통과율(파싱/검증 실패로 제외된 비율)
- **재정의된 성공 기준 (go)**: 절대 임계값(예: Recall<5%) 폐기. `R_query_only`는 **"leakage floor"**로 기록되어 F8의 `query_only` 비교군의 기준선이 된다 — Pilot 자체의 pass/fail이 아니라 **F8에서 steered 모델이 `R_query_only` 대비 추가로 만들어내는 lift를 해석하기 위한 사전 측정값**. Pilot 3는 "측정이 가능하고 마스킹 통과율이 합리적(예: ≥70%)인지"만 확인하면 go.
- **중단/조정 기준**: P2 마스킹 통과율이 낮음(<70%) → P2 프롬프트의 식별정보 제거 규칙 강화 후 재측정(최대 2회). `R_query_only` 자체가 높게 나오는 것은 **중단 기준이 아니며**, F8 해석 시 "이 카테고리는 query만으로도 답이 잘 드러남 → steering의 추가 기여를 더 엄격히(R_query_only 초과분만 의미있는 것으로) 평가" 식으로 반영.
- **예상 비용**: pseudo-query 생성 ≈ N회 LLM 호출 + 임베딩/검색(로컬)

### Stage P4 — Steering Controllability (v0.4.14: *학습 후* 측정으로 재배치 — 게이트 아님)
> **v0.4.14 재해석**: Pilot4 mini(미학습 projector, few-epoch)는 NO-GO였으나 — 이는 메커니즘 결함이 아니라 **bge→SASRec modality gap을 메우는 projector를 학습하지 않은 것**이 원인(PPR이 P=1 prefix로 frozen SASRec steering 성공 = P=1은 죄 없음; LLaVA = projector를 강하게 학습해야 gap 해소; plan §7#2 = 미학습 prefix로 steering 판정 금지). 따라서 **controllability는 *전체 Stage2 학습 후* 측정**하며, 이는 *게이트가 아니라* 학습 결과 검증이다.
- **무엇**: 학습된 projector로 같은 쿼리에 correct-intent vs wrong-intent(타클러스터/타유저 메모리 swap) prefix를 줬을 때 추천 top-k가 유의하게 갈리는가(controllability) + correct prefix Recall@10 > concat baseline(방향성).
- **언제**: 전체 Stage1(F6) → 전체 Stage2(F7, LLaVA 비대칭: SASRec frozen / encoder 저-LR / projector 강-LR) *완료 후*. Stage2 학습 *중*에도 모니터링.
- **판정**: 학습 후에도 correct/wrong 분리가 없으면 *그때* prefix 주입 메커니즘(cross-attention 등) 의심(§7 결정 #5). 잔여 리스크: PPR prefix는 안정적 user-profile에서, 우리는 *가변 query*에서 — controllability가 PPR엔 없던 추가 요구라 "PPR 되니 우리도 무조건"은 아님(학습 후 측정으로 확정).
- **출력**: `results/pilot/pilot4_report.json`(과거 mini NO-GO 기록 보존) + F8의 controllability 표.

### Gate — Go/No-Go 리뷰
- **P1, P2는 hard gate**(미충족 시 카테고리/프롬프트 재설계까지 재귀). **P3(leakage floor)는 측정값 기록**(pass/fail 아님, §Stage P3). **P4(controllability)는 게이트가 아니라 Stage2 *학습 후* 검증으로 재배치**(v0.4.14). **이 리뷰 시점에 사용자 확인 필수** — 전체 파이프라인 착수 전 중단점.

### Stage F — 전체 파이프라인 (Gate 통과 후)
| Stage | 내용 | 입력 | 출력 | 비용 추정 |
|---|---|---|---|---|
| F1a | M0 실행 + **데이터 볼륨 리포트** — 후보 2~3 카테고리(§7 결정#6) 각각의 유저 수/아이템 수/평균 이력 길이/k-core 필터 후 잔존율 산출 | 로컬 raw JSONL | volume_report.json (카테고리별 표) | 다운로드/전처리, GPU 불필요. **본 리포트 공유 후 카테고리 최종 확정에 대한 사용자 승인** (결정#6 보류 조건) |
| F1b | F1a에서 확정된 카테고리로 M0 전체 실행(sequences/splits 확정). **Amazon-C4 user/item ID 오버랩 비율 점검**(§6.5, §4 F8 입력용) | F1a 승인 | sequences.jsonl, splits.json, c4_overlap_report.json | 전처리, GPU 불필요 |
| F2 | M1 P1 전체 실행 (train-history 전체) | F1b 출력 | P1 결과 (캐시) | LLM 호출 ≈ Σ유저별 이력 길이 — 카테고리/서브샘플 크기에 비례, 사전 추정치 산출 후 **사용자 승인** |
| F3 | M2 메모리 뱅크 구축 (personal + prototype, P2의 fallback 비율 반영) | F1b+F2 | memory_bank/*.json, _prototypes.json | LLM 호출 ≈ Σ(2·K_personal/유저) + 2P(prototype) (v0.3.2: 결합 호출 1회 + persona 0~1회 = 최대 2회/클러스터, disposition_note 전부 null인 클러스터는 persona synthesis 생략) |
| F4 | M1 P2 (pseudo-query: **train**=history-grounded+memory_id 조회 / **eval**=target-grounded 격리, §6.6) | F1b+F3(provenance) | pseudo_queries.jsonl | LLM 호출 ≈ Σ(train eligible 리뷰)+Σ(eval 타깃 리뷰). P1과 같은 입력 — 배치 동시 실행 권장 |
| F5 | ~~M1 P3 (align judge, self-consistency T=3)~~ **(v0.4.11 제거 — provenance가 positive 라벨 결정론 제공; judge는 선택적 ablation으로만)** | F3+F4 | align_pairs.jsonl(provenance 라벨, LLM-free) | LLM 호출 0(judge 제거). align_pairs = provenance 조회로 생성 |
| F6 | M4 Stage1 (L_align) | F5 | encoder ckpt | GPU, 1 카테고리 기준 추정 후 승인 |
| F6a **(v0.3.2 신규)** | SASRec vanilla pretraining (item_id 시퀀스만, 텍스트/의도 미사용) — F6과 입력이 겹치지 않아 **병렬 실행 가능** | F1b (sequences/splits, train) | checkpoints/{exp_name}/sasrec_pretrain.pt | GPU, F6과 동시 추정·승인 |
| F7 | M4 Stage2 (hybrid, 기본 L=L_retrieval+α·L_align; v0.4.22 L_mem 실험은 `β_x·L_mem_X` main 후보와 `β_w·L_mem_Wtrain_aligned` ablation을 **분리**해 실행, `h_query`/`h_memory` on-the-fly 재인코딩, §7#11). SASRec은 F6a 체크포인트에서 로드(기본값 frozen). **L_mem full training 전 X_in_seq·β=0 감사 통과 필요** | F1b+F3+F6+F6a | 최종 ckpt | GPU, 추정 후 승인 |
| F8 | M5 전체 평가 — 전 비교군 × 전 카테고리 + `recovery_analysis.py` headline 실험(correct/vanilla/wrong-intent 3-way, circularity-robust) + **provenance routing accuracy**(v0.4.10, C4 아님) + **controllability 표**(v0.4.14, Stage2 후) + `query_only` 비교군을 Pilot 3 leakage floor와 함께 보고. 절대 성능은 *review-derived = real-query 상한*으로 선제 고지(v0.4.8). vanilla 베이스라인은 F6a 체크포인트를 F7과 공유(조건부 `vanilla_finetuned`, §3 M5) | F1b+F3+F6a+F7 | results/metrics.json, recovery.json, routing_acc.json | 추론만, 경량 |

**원칙**: F2, F5(LLM 대량 호출)와 F6/F6a/F7(전체 학습) 착수 전 각각 예상 호출수/비용/시간 추정치를 공유하고 **개별 승인**을 받는다(스펙의 "큰 연산 전 확인" 요구사항). F1a→F1b 사이에도 카테고리 확정에 대한 승인이 필요(§7 결정#6).

## 5. LLM 프롬프트 초안 — (이동됨) `decisions.md` 참조

> 프롬프트 *초안*(P1-base/P1-aspect/P2/P3-judge)은 `decisions.md`의 "(구) LLM 프롬프트 초안"으로 이동. **실제 운영 프롬프트는 `config/prompts/{dataset}/{domain}/*.txt`** (P2 = `p2_pseudo_query.txt`, C4 원문 + JSON, 입력=리뷰만; P3-judge는 핵심 파이프라인에서 제거되어 선택적 ablation 전용).

## 6. 데이터 설계 (획득/필터/temporal split/시퀀스 구성)

### 6.1 카테고리 선택 (v0.4 — 실제 보유 카테고리 반영)
**실제 보유(로컬 다운로드 완료)**: `Amazon_Fashion`(라이프스타일), `Beauty_and_Personal_Care`(라이프스타일), `Books`(콘텐츠). 기능형(Office_Products/Tools/Pet_Supplies)은 *미보유* — 필요 시 추후 추가(라이프스타일 vs 기능형 대조용, 필수 아님).

P1 A/B(§4 Stage P1)의 목적은 어느 도메인에서 discriminative·aspect_coverage·멀티맥락이 풍부한지 실측하는 것이다. **Pilot 2-lite 결과로 갱신된 관점** (eligibility = count≥5 + ≥10단어 + rating≥4 기준):
- **eligibility(=personal 경로 커버리지)**: Books 4.7%(483,835명) > Beauty 1.9%(219,214명) ≫ **Fashion 0.1%(2,408명)**. Fashion은 충동구매·낮은 재구매라는 *구조적 성질*이라 필터를 풀어도 회복되지 않음 — personal path 사실상 0.
- **추출 품질(장문 N=20)**: 3개 모두 양호(parse 100%, disc 50~65%, 환각 없이 review-grounded). Books aspect_coverage 95%는 *sub-genre 의존*: 아동서·성인 **논픽션**(self-help/business/health)은 풍부하나 **문학 *소설*은 약함**("immersive reading" 류 → non-discriminative). N=20 Books 표본은 약 2/3이 아동서.
- **단문(3~9단어) ablation**: discriminative 10~15%뿐, 50~75%가 빈 ci — 모델이 감정만 있는 리뷰엔 의도 생성을 *거부*(환각 없음). 성공 케이스는 명시적 맥락 또는 메타데이터 grounding(예: "absolute beginner book" + 아이템 "piano classics" → 피아노 학습)일 때만.
- **결정**: Books는 ComiRec/MIND/Persona4Rec의 표준 카테고리이자 커버리지 최상 → **주전장 후보 1순위. Beauty 보조(2순위). Fashion은 주전장에서 제외**(sparse·prototype-dominated 대조군으로만 선택적 활용 — extraction은 멀쩡하나 *coverage*가 구조적으로 부족, 필터·프롬프트로 해결 불가). 최종 확정은 Pilot 2 클러스터링(per-user K≥2)으로.
- **데이터 위생(전체 실행 전 정리)**: Books 카테고리에 비-도서 혼입(CD/공구 등) 확인 — full study에서는 meta category로 실제 도서만 필터. domain_type 라벨 오류(Beauty="functional") 수정.
- 각 카테고리 볼륨/k-core 잔존율은 F1a(§4)에서 확인.
- Amazon-C4: §6.5에서 정식 정량 평가 입력(query↔memory↔item routing).

### 6.2 필터링 (v0.4 — 백본/메모리 필터 분리)
**백본(SASRec) 데이터와 메모리 데이터의 필터를 분리한다**(혼동 방지). 문헌 표준은 *상호작용 5-core*(리뷰 텍스트·평점 무관)이며, 백본은 이 표준을 따르고 메모리만 텍스트/평점 조건을 둔다.

- **백본(SASRec) = 전체 상호작용, 5-core**: 사용자 ≥5·아이템 ≥5 인터랙션(반복 적용). 리뷰/평점으로 거르지 않음(BERT4Rec/SASRec 표준과 정합 — 리뷰의 *존재*를 implicit feedback으로 취급). max sequence length 20, leave-one-out(§6.4). 백본을 5점·리뷰보유로 거르면 데이터가 줄고 baseline이 비표준이 되므로 *금지*.
- **메모리(Module 1/2) = positive 리뷰 기반**: `rating≥4`(positive) + 리뷰 텍스트가 있는 인터랙션만 P1 입력. positive를 기본값으로 하는 이유는 메모리가 "이 맥락에서 *만족·욕구한* 선호"를 담고 routing이 그쪽으로 steer하기 때문(`=5`는 더 엄격한 옵션 — funnel로 비용 보고 후 ≥4/=5 확정). `avoid`/disposition은 그 리뷰의 비판적 뉘앙스에서 여전히 추출.
- **메모리 eligibility (v0.4.1 — ablation으로 확정)**: `rating≥4` + **리뷰 ≥10단어(*비용 프리필터*)** + `is_discriminative==true`(*진짜 필터*). 단문(3~9단어) ablation 결과 discriminative 10~15%·빈 ci 50~75%로 단문은 추출 부적합이 실측됨 → 10단어 프리필터 정당화(그 아래로 내리지 않음). 단 "단어수<구체성"이므로 길이는 *프리필터*일 뿐, eligibility의 본질은 *discriminative 산출 여부*로 정의한다. (count 임계는 Pilot 2 클러스터링 결과로 확정, 현재 working value ≥5.)
- **(v0.4.6) 후보 선택 = 결정론 + provenance ("첫 12개" 폐기)**: P1에 보낼 리뷰는 다음 순서로 *결정론적으로* 고른다 — 입력 파일 순서에 **불변**이어야 한다(GATE FAIL 재발 방지).
  1. **pre-P1 필터**(LLM 불요): `train-history 구간` ∩ `rating≥4` ∩ `≥10단어`. (is_discriminative는 P1 후 적용 — pre-P1엔 못 씀.)
  2. **frozen candidates**: 1을 통과한 (user_id, item_id, timestamp, rating, review_text, item_title) 행을 `candidates.jsonl`로 **1회 생성·md5 고정**. 이후 *재생성 금지*, 빌드는 md5 검증 후 사용.
  3. **cap 선택**: 유저별로 cap N(config, 현 12)을 넘으면 **`timestamp` 내림차순 → 동률은 `item_id` 사전순**의 안정정렬로 상위 N개 선택(파일순 아님). 선택된 item_id를 `selected_review_ids`로 저장.
  - **(v0.4.7) `eligible_min`=1**: 유저가 selection 대상이 되는 최소 eligible 리뷰 수 = **1**(2 아님). eligible 1개 유저도 추출 대상 → is_discriminative면 K_personal=1(k_min=1 정합), 아니면 K=0→fallback. `[DEFAULT_INTENT]`는 eligible 0 유저에게만. (eligible_min=2는 K=1 층화 모집단을 인위 축소했던 버그 — v0.4.7 수정.)
  - 이렇게 하면 candidates가 어떤 순서로 저장돼도 **동일 N개가 결정론적으로 선택**되고, 캐시 키도 안정적이며, 사후 복원이 애초에 불필요하다. cap이 multi-intent를 자르는지(=cap 적용 시 K_personal 분포 변화)는 F3에서 점검해 N 상향을 검토.
- **(v0.4.6) provenance 보존 — evidence.item_ids 영구 채움**: P1 산출물(`p1_extractions.jsonl`)은 각 레코드에 `item_id`/`timestamp`/`rating`을 **1급 필드로** 유지하고, 클러스터링은 멤버 `item_id ↔ cluster_label` 매핑을 보존한다 → `synth.py`가 `evidence.item_ids`/`evidence.timestamps`를 채운다(빈 `[]` 금지). cluster_summaries에 합성 `source_texts`만 남기던 방식 폐기. (이로써 P2 train query의 `memory_id`도 source-review→cluster *조회*로 확정 — §6.5.)
- **감정만 있는 리뷰("I love it", "My favorite dress!") → 메모리 만들지 않음**: 모델이 빈 ci를 반환하는 것이 정답. 아이템 정보로 억지 의도를 만들면 *아이템 정체성의 재진술*(=SASRec 백본이 이미 가진 정보)이라 메모리 가치가 없고 routing을 오염시킨다. 그런 인터랙션의 아이템/행동 신호는 **백본 + prototype fallback**으로 들어가며, 시퀀스에는 남되 메모리에는 안 들어간다.
- **LLM 추출 규모 (v0.4.2 — 현실적 스케일)**: 전체를 다 돌리지 않고 **유저 ~10K–50K로 서브샘플**(SIGIR/KDD LLM-rec 관행; SASRec+projector 학습·통계 유의성에 충분)하며, *그 서브샘플과 동일 유저셋*을 백본·메모리·평가가 공유한다(§6.4 single split). **LLM 추출은 *eligible 유저(~5%)*에게만** 수행하므로(나머지는 prototype/`default_intent`) 호출 수 ≈ (서브샘플 내 eligible 유저)×(평균 eligible 리뷰) — 예: 20K 유저면 Books eligible ~940명×~7 ≈ 7K 호출, Beauty ~380명×~7 ≈ 3K 호출 수준(전체 유저 추출이 아님). 16s/콜이면 7B+4-way 병렬로 수 시간. 정확한 크기는 F1a 볼륨 리포트로 확정.
- 리뷰 없는/짧은 인터랙션도 **시퀀스(추천 대상)에는 남는다** — 메모리 evidence에만 못 들어갈 뿐, 백본 학습·평가에서 제외하지 않음(짧은 이력 유저는 prototype fallback + stratified 비교).

### 6.3 시퀀스 구성
유저별 `(item_id, timestamp, rating, review_text, item_meta)`를 timestamp 오름차순으로 정렬한 리스트.

### 6.4 Split — leave-one-out + global temporal sanity (하이브리드)
**(v0.4.2) 단일 Global Split = single source of truth.** 파이프라인 *최초*(M0 학습 이전)에 `splits.json`(=`train_test_split.json`)을 한 번 생성·저장하고, **M0(vanilla SASRec)·M1(LLM 추출)·M4(hybrid)·M5(eval) 전부가 *바이트 단위로 동일한* 이 파일을 소비**한다(baseline과 제안 모델이 정확히 같은 history를 보고 같은 target을 맞추도록 강제). `tests/test_split_consistency.py`로 모든 모듈이 동일 split 인덱스를 쓰는지 assert.
1. **Leave-one-out** (SASRec 표준): 유저별 마지막 아이템 = test target, 마지막-1 = val target, 나머지 = history
2. **Global temporal sanity**: 전체 데이터셋의 timestamp 분포에서 상위 X%(예: 95th percentile) 시점 `T_cut`을 계산. test target의 timestamp가 `T_cut`보다 현저히 이른(= 데이터 후반부 트렌드를 전혀 반영 못하는) 유저는 평가에서 제외하거나 별도 세그먼트로 표시 — population-level future leakage 점검
3. **메모리 구축 = history 구간만** (val/test target 제외) — §2.2의 leakage 방지와 동일 제약을 split 레벨에서도 명시적으로 강제

### 6.5 Amazon-C4 활용 (v0.4.10 정정 — routing GT 폐기, 프롬프트 스타일 참조 전용)
> **v0.4.10**: C4를 정량 routing 평가로 격상했던 v0.2 결정을 *정정*. C4는 우리 routing GT가 될 수 없다 — (i) C4는 query→*item*(parent_asin) 검색이지 query→*memory* 라우팅이 아니고, (ii) C4 유저는 전체 Amazon 테스트셋에서 추출되어 우리 5-core Books splits 유저와 거의 안 겹쳐(memory_bank 조인 ≈ 0), (iii) C4 쿼리는 *5점 리뷰 1건*에서 만든 single-review라 우리 *multi-review 클러스터* 중 "정답"이 정의 불가.
- **routing accuracy GT = provenance self-consistent**(§3 M5 `stratify.py`): train 쿼리 q→리뷰 item_id→cluster_label=정답 memory. *데이터 구조 내재 결정론*. C4·P3-judge 아님.
- **C4의 실제 용도**: (i) *프롬프트 스타일 참조* — P2가 C4 원문 프롬프트(첫1인칭 환언+제품명 숨김)를 이미 차용(§6.6, v0.4.12). (ii) *선택적* OOD 쿼리 robustness 질적 비교(원하면 다운로드, 주 지표 아님). BLAIR 코드를 Books로 직접 돌릴 필요 없음.
- end-to-end **학습·평가**는 Amazon Reviews 2023 하나로 닫는다.

### 6.6 P2 Pseudo-query 생성 — 리뷰원문 기반 + temporal train/eval 분리 (v0.4.6)

쿼리는 **리뷰 원문**에서 생성한다(합성 메모리 텍스트가 아님 — C4/BLAIR 계보). 리뷰가 구매 목적·동기·맥락을 가장 잘 담고, 이 쿼리 데이터셋 자체가 \"실제 질의 시 성능\"을 측정 가능하게 하는 연구 산출물이다. 단 next-item 세팅의 누설을 막기 위해 **시간으로 train/eval을 분리**한다(BLAIR도 timestamp로 train/eval 분리).

- **train query (입장1, 누설 0)** — `p2_pseudo_query_train.txt`
  - 입력: 유저의 **train-history 리뷰**(timestamp < val cutoff)의 원문 + item 메타(category·속성).
  - 출력: 그 리뷰의 구매 동기를 1인칭 검색 질의로 환언("이 사람이 사기 전 검색창에 뭐라 쳤을까").
  - **positive 라벨**: 그 source 리뷰의 `item_id` → (v0.4.6 provenance) **cluster_label 조회** → 해당 memory가 positive(코사인 Path B 불필요). same-user 다른 cluster / 타유저 memory = hard negative(§5).
  - 용도: align-pair → Stage1/2 학습.
- **eval query (입장2, eval 전용 격리)** — `p2_pseudo_query_eval.txt`
  - 입력: 유저의 **val/test 타깃 아이템 리뷰**(held-out future) 원문 + 메타.
  - 출력: BLAIR식 — 타깃 리뷰를 1인칭으로 환언하되 **제품명을 숨김**(찾는 사람 톤). C4 \"real query\" 정신.
  - **학습에 절대 미혼입** — F8 \"real query 성능\" 측정 전용. target item이 곧 정답이라 train에 쓰면 누설.
- **공통 leakage/자명성 가드 (v0.4.7 — distinctiveness 디텍터)**: 입력에 title·brand·저자를 *맥락*으로 줄 수 있으나 **출력 쿼리엔 금지**. 생성 시 reject+retry(≤2) 후 **사후 디텍터**로 재검. 디텍터는 *길이(unigram/bigram) 기준이 아니라 고유성(distinctiveness) 기준*: ① 제목 토큰에서 stopword + 장르/카테고리 공통어 사전(mystery/romance/thriller/piano/history/love/fiction…) + corpus document-frequency 상위(전체 books 제목 DF 높은 토큰=공통어, 임계 config) 제거 → ② 남은 *distinctive n-gram(연속)* 또는 **저자명(meta author 정확 매칭)** 이 query/source_text에 연속 매칭될 때만 leakage=true. 단일 장르어 1개 겹침은 누설 아님; 저자명은 1단어여도 누설. train은 history-only라 구조적으로 누설 0, eval은 격리로 누설 차단. **이 디텍터는 P1 메모리 source_text의 `leakage_detected` 마커 재계산과 P2 eval query 게이트에 *동일하게* 재사용**(unigram-substring 구버전은 22% 과발화로 leakage rate 지표를 오염시켜 폐기, v0.4.7). leakage rate는 *논문 보고 지표*이므로 distinctiveness 기준으로 산정한 값만 보고한다.
- **호출 규모**: train ≈ Σ유저 (선택된 train-history eligible 리뷰 수, cap 적용), eval ≈ Σ유저 (val+test 타깃 중 리뷰 보유분). P1과 동일 입력이라 배치 동시 실행 권장(F4).

## 7. 설계 결정 목록 & 확인 필요 사항

각 항목: 결정 내용 / 옵션별 장단점 / 현재 제안 / 확인 필요 여부.

**#1. 메모리 K 결정 방식 — ✅ 확정 (회신: agglomerative + 공유 prototype fallback 필수)**
- (a) 사용자별 자동(agglomerative + distance threshold, k_min~k_max): 실제 의도 다양성 반영, 그러나 threshold τ가 카테고리마다 다를 수 있음
- (b) 고정 K (예: 항상 3): 구현 단순, 그러나 의도가 1개뿐인 유저에 억지로 3개를 만들면 노이즈 메모리 생성
- (c) 공유 prototype: cold-start/짧은 이력에 강건, 그러나 유저 특화 brand_tendency 등 손실
- **확정 (v0.4.3 갱신)**: (a)+(c) 결합 — personal agglomerative(τ_personal, **k_min=1**·k_max=5)로 K_personal 산출. **K_personal을 0/1/≥2로 정직히 보고하고 억지로 K≥2를 만들지 않는다**(리뷰가 동질이면 K=1, 변별 리뷰가 없으면 K=0). K_personal==0(또는 K==1인데 query-memory sim이 낮은 경우)에는 population-level prototype(§2.2 (B), τ_global, P≈8-15개)을 *명시적 fallback*으로만 부여(`is_prototype=true`, 평가 분리), 신호 자체가 없으면 learnable `[DEFAULT_INTENT]`(v0.4.2). prototype을 personal 메모리처럼 "보충"하지 않는다. τ_personal/τ_global과 K 분포는 Pilot 2/3에서 실측.
- **(v0.4.2) cold-start prefix — zero/global-average 금지, learnable `[DEFAULT_INTENT]` 토큰 사용**: 메모리/prototype 매칭이 의미 없는 진짜 cold-start(구매이력 신호도 빈약) 또는 eval의 "no-intent" baseline에는, 평균 텐서(특징 뭉개짐)나 zero-vector(dead neuron·attention 왜곡) 대신 **학습 가능한 d_sasrec 차원 prefix 파라미터** `default_intent` 하나를 두고 Stage 2에서 함께 학습한다 — 모델이 "특정 의도 없을 때의 최적 중립 방향"을 스스로 학습. 구매이력 신호가 있는 유저는 위 centroid→prototype 경로 유지, 신호 없는 유저만 `default_intent`. (§3 M3 projector/sasrec, §3 M5 `none` 조건의 학습형 baseline.)

**#2. preference_signal 저장 형식**
- 구조화 필드만: ablation/분석 용이하나 임베딩 입력으로는 부자연스러움
- 자유 텍스트만: 임베딩 품질 좋을 가능성, 그러나 controllability 실험(필드 swap)에 불리
- **제안**: §2.1처럼 **둘 다 보존**(attributes + summary, 동일 클러스터에서 동시 생성). 추가 비용은 클러스터당 LLM 1회 — 미미. **확인 필요 없음(제안대로 진행 가능), 단 이견 있으면 표시**

**#3. 라우팅/projector 임베딩 입력 — (v0.3.1 정합: intent+persona.description+summary)**
- **(v0.3.1 확정)** §2.1에서 `embedding.source_text = " ".join(x for x in [intent_description, persona.description, preference_signal.summary] if x)`를 `h_memory`의 기본값으로 한다 — `persona.description`이 ""이면 생략(boilerplate 희석 방지, §7 #4). description-only 등 변형은 §4 Stage F ablation으로 비교(저비용). **확인 필요 없음**

**#4. persona — ✅ 확정 (v0.3: 코어 필드로 복원/업그레이드 — v0.2 결정 #4 번복)**
- v0.1: 고정 taxonomy(5~6개, 예: `budget-conscious`, `premium/brand-loyal`, `feature-maximizer`, `convenience-first`, `quality/durability-focused`, `aesthetics-driven`)에서 규칙 기반으로 매핑하는 `persona_tag`. v0.2: 코어에서 제거(분석 전용 사후계산).
- **v0.3 번복 사유**: persona_tag를 attributes(price_band/feature_priorities/brand_tendency)의 규칙 기반 재인코딩으로 두면, attributes로 환원되지 않는 상위 신호("이 맥락에선 가격보다 내구성 우선", "검증된 것만 vs 탐색적 구매")를 표현할 수 없다. 이런 신호는 사람이 작성한 리뷰에서 LLM만이 포착 가능 — 규칙으로는 불가능.
- **확정 (v0.3)**: `persona: {tag: str, description: str}`을 `IntentMemoryUnit`(§2.1)에 `intent_description`/`preference_signal`과 형제 레벨의 코어 필드로 추가.
  - **per-memory(intent별)**: 같은 유저라도 메모리(클러스터)마다 다른 persona를 가질 수 있다 — "이 사람은 캠핑 장비를 살 때는 가성비 위주이지만, 홈오피스 장비를 살 때는 평소와 달리 내구성/품질을 최우선한다"와 같은 맥락별 전환을 포착.
  - **생성**: §2.2 (A)/(B) per-cluster synthesis에서 `intent_description`/`preference_signal.summary`(v0.3.2: 결합 1회 호출, `synth_intent_attr_summary.py`)와 별도로, P1(§5)의 `disposition_note`들을 입력으로 LLM 1회 추가 호출(`synth_persona.py`, cost: N+K → N+2K, P → 2P, **단 v0.3.1: 클러스터 내 disposition_note가 전부 null이면 이 호출을 생략**하므로 N+2K/2P는 상한이며 실제로는 N+K~N+2K/P~2P 사이, §2.2).
  - **non-redundancy**: `disposition_note`(P1, §5)는 "이 구매 맥락에서 평소와 다른/특이한 구매 성향"을 명시적으로 묻는 질문으로, persona가 preference_signal.attributes의 단순 재서술이 되지 않도록 한다. **(v0.3.1)** disposition_note가 전부 null인 클러스터(=비변별적 디스포지션)는 LLM 호출 없이 `persona = {tag: "typical" 또는 None, description: ""}`로 설정한다 — "특이사항 없음" 류의 boilerplate 문장을 채우지 않는다. 이유: 그런 boilerplate가 다수 메모리에 반복되면 routing 임베딩에서 `intent_description`의 변별 신호를 희석한다(아래 이중 역할 참조).
  - **설계 결정 — persona의 이중 역할**: persona는 (i) **routing**: `h_memory.source_text = " ".join(x for x in [intent_description, persona.description, preference_signal.summary] if x)`(§2.1, v0.3.1 — 빈 `persona.description`은 생략)에 포함되어 `h_query`와의 sim 계산에 기여 — 질의가 "특정 디스포지션"을 암시하면(예: "내구성이 중요해요") 해당 persona를 가진 메모리가 더 잘 라우팅되며, persona가 routing에 실질적으로 기여하는 것은 **disposition이 실제로 변별적일 때만**(persona.description ≠ "")이다. (ii) **steering**: projector(`h_intent = MLP(h_query, h_memory)`)의 입력인 `h_memory`에 이미 persona가 인코딩되어 있으므로, 별도 입력 채널 추가 없이 steering payload에 자연히 반영됨(§3 M3 projector 노트).
  - **Pilot 4 wrong-intent 합성**: `intent_description`/`attributes`와 함께 `persona`도 함께 swap — "틀린 의도"가 디스포지션 차원에서도 일관되게 틀리도록 함(§4 Stage P4).
  - **(선택, v0.3)** `preference_signal.attributes.avoid: [str] | None` 추가 — negative steering 여지. P1의 `preference_attrs.avoid`(per-log)를 클러스터 단위 union으로 집계(추가 LLM 호출 없음).

**#5. Prefix 메커니즘 세부 — ✅ 확정 (회신: P=1로 시작, projector 입출력은 PDF 정합 형태로 수정)**
- prefix 길이 P: 1 vs 여러 개(예: 4) — P=1은 "하나의 의도 벡터", P>1은 preference의 여러 측면(가격/기능/브랜드)을 분리된 슬롯으로 표현 가능하나 파라미터/복잡도 증가
- 주입 지점: (i) 입력 시퀀스 앞단에 virtual token으로 concat 후 self-attention 통과(스코어링 위치에서는 제외) vs (ii) 모든 레이어에 additive bias(true prefix-tuning, P×L×d 파라미터)
- **확정**: (i) + P=1(단일 벡터)로 시작. **(v0.2 변경, 필수#1)** projector 입력은 `h_memory`(메모리 summary 임베딩) 단독이 아니라 PDF Module 2.3/3.2의 `h_intent = MLP(h_query, h_memory)`로 — `h_query`(router의 query 임베딩)와 `h_memory`(routed 메모리의 `embedding.vector`, §2.1)를 함께 입력받아 `prefix_embeds`(1×d_sasrec)를 생성(§3 M3 projector.py). P>1/방식(ii) 확장은 전체 파이프라인 단계의 ablation으로 이연.

**#6. 카테고리 선택 — ✅ Books 주전장(보조 Beauty), Fashion 제외 (v0.4.1, §6.1)**
- Pilot 2-lite funnel(count≥5+≥10단어+rating≥4): Books 4.7%(484k) > Beauty 1.9%(219k) ≫ Fashion 0.1%(2.4k). **Fashion은 coverage가 구조적으로 부족(필터·프롬프트로 해결 불가)해 주전장에서 제외**, sparse·prototype 대조군으로만 선택. **주전장 Books, 보조 Beauty.** per-user K≥2 최종 확정은 Pilot 2 클러스터링. 볼륨/잔존율은 F1a(§4)로 확인. (Books는 비-도서 혼입 정리 + 아동/논픽션/소설 sub-genre 구성 보고 필요, §6.1.)

**#7. LLM 선택 (P1~P3) — ✅ 확정 (실제 서빙 환경 반영, v0.4 갱신)**
- 보유 자원: 4× RTX 2080 Ti (11GB VRAM/장, CUDA 12.2, driver 535.274.02, Turing/sm75). 4장 합 44GB — 26B 모델도 멀티-GPU 분할로 GPU 적재 가능(CPU 오프로드 아님).
- **확정**: `gemma4:26b` via **Ollama** (http://localhost:11434). 당초 계획(Qwen2.5-7B-Instruct bnb-nf4) 대체. 출력 안정화: **JSON-schema constrained decoding + 스키마 maxLength/maxItems/additionalProperties:false + `think:false`(top-level)** — 초기 truncation/지연의 원인은 모델 크기가 아니라 *무한정 긴 출력*이었고 이 패치로 해소(parse 100%, latency ~16s/콜).
- **A/B 동일 모델 보장**: p1_base·p1_aspect 두 config 모두 `model_id: gemma4:26b`, `api_url: http://localhost:11434/api/...` — prompt만 변수.
- **처리량 (미결)**: ~16s/콜은 Pilot엔 충분하나 전체 F2(수만 호출)엔 부적합. F2 착수 전 (i) 7~14B로 교체 + GPU당 1카피 4-way 병렬, 또는 (ii) 서브샘플(§6.2) 중 하나를 재측정으로 확정.
- **N=200 latency 추정**: `estimated_n200_time_s = cache-miss mean latency × 200` — Pilot 1 리포트에 포함.

**#8. 임베딩 모델 (routing/text encoder 초기화) — ✅ bge-base, 로컬 캐시 전용**
- **확정**: `BAAI/bge-base-en-v1.5`(검색/쿼리-문서 매칭 강점, query↔memory task 정합).
- **수급 정책(v0.4)**: 임베딩 모델 **자동 다운로드 금지**. 코드는 (1) Docker 내 sentence-transformers 설치 여부, (2) 모델이 로컬/프로젝트 캐시(`HF_HOME=TRANSFORMERS_CACHE=/qaim-rec/.cache/huggingface`)에 있는지만 확인하고, 없으면 *다운로드 명령어만 출력하고 멈춤* — 사용자가 Docker에서 직접 받는다.

**#9. P3 후보쌍 생성 / L_retrieval 정의**
- 후보쌍 폭발 방지: 유저 자신의 K개 메모리 + 무작위 타유저 메모리 M개(§5)
- L_retrieval: full-softmax(전체 카탈로그, 카탈로그 크기에 따라 비용 큼) vs in-batch negative + 주기적 full-ranking 검증
- **제안**: 학습은 in-batch negative(효율) + L_align과의 균형(α) 탐색. **(v0.4.2)** InfoNCE(L_align)와 next-item CE(L_retrieval)는 스케일이 크게 달라 α=1.0이면 alignment가 추천 신호를 덮을 수 있으므로, **α는 {0.01, 0.1, 0.5}처럼 *낮은 값에서 시작*하고, 두 loss의 raw 크기·gradient norm을 매 스텝 로깅해 스케일을 맞춘다**(§ Troubleshooting). **평가만** full-ranking(스펙 원칙 유지). **확인 필요 없음**

**#12. Score-space memory contrast (L_mem) — ✅ v0.4.22 현재 설계, full training 전 gate 필요**
- **문제**: v0.4.21 진단에서 prefix는 h_last에 전달되지만 정답 item embedding 방향과 무상관(pos_frac≈50%)이었다. 원인은 `L_retrieval+α·L_align`에 "correct memory가 wrong memory보다 SASRec score를 더 올려라"는 score-space gradient가 없기 때문으로 정리했다.
- **사전측정 결과**: `W_train_all`은 폐기한다. K≥2 샘플 2,000개 중 A+B(`W_train`이 `X`와 동일하거나 같은 cluster) 49.1%, C+D(다른 cluster/no provenance) 50.9%. `W_train`-target Δz는 A+B에서 양수, C+D에서 음수였다. 모든 샘플에 `W_train`을 target으로 쓰면 절반의 gradient가 memory 의미와 충돌한다.
- **main 후보 = `L_mem_X`**: `q`는 source item `X`의 리뷰에서 생성, `m+`는 `positive_memory_id` provenance memory, `m-`는 same-user different-cluster memory. `L_mem_X=softplus(-(z_X(q,m+)-z_X(q,m-)))`, `z_X=item_emb(X)·h_last(seq,prefix(q,m))`. K=1은 mask. cross-user fallback 금지. router top-1은 inference/routing용이며, L_mem positive로 쓰지 않는다.
- **ablation 후보 = `L_mem_Wtrain_aligned`**: `W_train=user_train[uid][-1]`을 target으로 쓰되 A+B(`mid(W_train)==mid(X)` 또는 `W_train==X`)에서만 계산. C/D/K=1은 mask. `W_train_all`은 금지.
- **실행 규율**: `--beta_mem_x default 0.0`, `--beta_mem_wtrain default 0.0`. β=0 backward-compatibility를 코드+로그로 확인한다. full training은 `x-only`와 `wtrain-aligned-only`를 분리해 실행하고, combined는 두 단독 효과를 확인하기 전 금지.
- **남은 gate**: `L_mem_X`는 X-target score-space steering을 가르치는 가장 clean한 정의지만, `X`가 input `seq`에 이미 포함될 경우 self-reconstruction 위험이 있다. 따라서 full training 전 `X_in_seq` 비율, β=0 route/loss 동일성, val/test target 비참조를 추가 감사한다.

**#10. Split: leave-one-out vs 순수 global temporal**
- §6.4에서 하이브리드로 제안(LOO 골격 + global temporal sanity 필터). 순수 global temporal(전 유저 동일 cutoff)은 유저별 시퀀스 길이가 불균등해져 짧은 이력 유저가 과도하게 배제될 위험
- **확인 필요 없음(제안대로 진행), 단 평가 결과 해석 시 이 선택을 명시**

**#11. (v0.3.2 신규, ✅ 확정) Stage2 `h_memory`/`h_query` staleness — on-the-fly 재인코딩**
- **문제**: M4 `train_hybrid.py`는 text_encoder를 저-LR로 계속 갱신하므로, M2에서 캐시된 `memory_bank`의 `embedding.vector`를 그대로 쓰면 Stage1/M2 시점 기준으로 stale해짐.
- **분석**: 유저당 라우팅 후보(personal+prototype)는 ≤~7개로 매우 적음 → 매 학습 스텝마다 해당 후보 전체 + `h_query`를 현재 text_encoder로 재인코딩하는 비용은 무시 가능(짧은 텍스트, 소수 forward pass).
- **확정**: Stage2는 **항상 on-the-fly 재인코딩**(라우팅 후보 재인코딩 → top-1 재라우팅 → projector 입력)을 기본값으로 한다. `memory_bank`의 캐시된 `embedding.vector`는 M2 산출 시점의 클러스터링/prototype 매칭/Stage1 학습 타깃으로만 사용되고 Stage2 정합성에는 의존하지 않음 — 별도의 주기적 재캐싱 전략 불필요. **확인 필요 없음**.


