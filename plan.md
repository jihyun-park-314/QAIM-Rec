# QAIM-Rec: Intent-Memory-Steered SASRec — 설계 문서 (v0.3.1, FINAL)

> **v0.3.1 (persona 희석 방지 패치)**: v0.3의 persona 도입을 유지하되, persona가 routing 임베딩을 희석하지 않도록 다음을 확정한다.
> - 클러스터 내 `disposition_note`(P1, §5)가 전부 null인(=비변별적 디스포지션) 경우, `persona.description`을 "특이사항 없음" 류의 boilerplate 문장으로 채우지 않는다 → **빈 문자열("")**로 두고, `persona.tag`도 `"typical"` 또는 `None`으로 둔다. 이 경우 LLM 호출도 생략한다(§2.2 비용 절감).
> - `h_memory.source_text`는 **비어 있지 않은 조각만 이어 붙인다**: `source_text = " ".join(x for x in [intent_description, persona.description, preference_signal.summary] if x)`.
> - 이유: 다수 메모리에 동일 boilerplate가 들어가면 routing 임베딩에서 `intent_description`의 변별 신호가 희석된다. persona가 routing에 기여하는 것은 *실제로 변별적 disposition이 있을 때만*이어야 한다.
> - 반영 위치: §2.1(source_text 정의, persona 필드 주석), §2.2 (A)/(B) per-cluster synthesis(persona 생성 분기 및 비용 공식), §7 #4(이중 역할 서술).
>
> **v0.3 (persona 복원/업그레이드)**: v0.2 결정 #4(persona_tag 코어 스키마 제거)를 번복한다. v0.1의 규칙 기반 persona_tag가 아니라, **LLM이 추론하는 per-memory(intent별) 맥락적 구매자 디스포지션** `persona: {tag, description}`을 `intent_description`/`preference_signal`과 형제 레벨의 코어 필드로 재도입한다. `h_memory.source_text`에 포함되어 routing(sim top-1)과 steering(projector payload) 양쪽에 쓰인다. (선택) `preference_signal.avoid`(negative steering)도 함께 추가. 상세는 §2.1/§2.2/§2.3/§3/§7 #4. 아래 v0.2 변경 목록 중 **4번 항목은 본 v0.3로 대체**되며, 1~3·5~7은 그대로 유지한다.
>
> v0.2: `연구아이디어.pdf` 대조 완료 + 사용자 피드백 반영. 주요 변경:
> 1. **(필수)** Projector를 `h_intent = MLP(h_query, h_memory)`로 수정 — 메모리 payload만이 아니라 **질의 임베딩과 메모리 임베딩을 함께 입력**받아 융합(PDF Module 2.3/3.1-3.2 정합). 메모리 임베딩 1개가 routing(sim top-1)과 projector 입력을 겸함.
> 2. **(필수)** Pilot 4: 미학습 projector로 (b)를 판정하지 않음. 소형 슬라이스에서 encoder+projector **최소 학습** 후 비교, 불가 시 (b)는 F7/F8로 이연되는 한계로 명시.
> 3. **(필수)** Pilot 3: P2 decontaminated pseudo-query로 측정 + "Recall<5%" 절대기준 폐기 → **leakage floor**로 재정의(F8의 query-only 조건 기준선으로 이행).
> 4. ~~**(권장)** `persona_tag` 코어 스키마에서 제거(분석 전용 optional, 필요시 사후 계산).~~ → **v0.3에서 번복**: persona를 LLM 추론 기반 코어 필드로 복원(위 v0.3 항목, §7 #4).
> 5. **(권장)** Amazon-C4를 routing(query↔memory↔item) **정량 평가**에 정식 투입(F8).
> 6. **(권장)** Pilot 2 fallback: "단일 메모리"가 아닌 **population-level 공유 prototype**으로 격상(agglomerative 유지 + prototype 보완).
> 7. **(권장)** "고정 후보집합 밖 회수" → correct/wrong/vanilla 3-way 비교의 **headline 실험**으로 격상.
>
> 결정 회신 반영: #1 agglomerative+prototype fallback 확정 / #4 persona_tag 제거 → **v0.3에서 persona 코어 필드로 복원(업그레이드)**, §7 #4 참조 / #5 prefix P=1(단일 h_intent) 확정 / #6 카테고리 제안 수용, 단 F1 이전에 데이터 볼륨 리포트로 사전 보고 / #7 LLM 선택은 GPU 자원 확인 후 보류.

## 0. Context

**문제 정의.** 순차추천(SASRec)은 사용자의 클릭/구매 시퀀스만으로 다음 아이템을 예측한다. 그러나 한 사용자는 시점·맥락에 따라 서로 다른 **"현재 질의로 활성화 가능한 맥락적 선호 단위(query-activatable contextual preference)"**를 갖는다(예: "캠핑용 장비를 찾는 나"와 "홈오피스를 꾸미는 나"는 같은 사람이지만 선호 축이 다름). 이 단위는 구매 목적(purchase purpose)만을 가리키지 않으며 — 기능적 용도, 취향/스타일, 콘텐츠 소비 맥락 등 질의가 활성화하는 모든 선호 차원을 포함한다. 시퀀스 전체를 평균화한 단일 사용자 표현은 이 맥락 전환을 표현하지 못하고, 매 질의마다 LLM을 호출해 재정렬하는 방식은 온라인 비용·지연이 크다.

**제안하는 해법.** 사용자의 과거 리뷰를 오프라인에서 LLM으로 분석해 "이 사람이 무엇을 위해 샀는가 + 그 목적에서 무엇을 선호했는가"를 단위로 갖는 **Intent Memory Bank**를 사용자별로 구축해 둔다. 실시간 질의가 들어오면 (1) 온라인 LLM 호출 없이) 텍스트 인코더로 질의를 임베딩(`h_query`)하고, (2) 메모리 뱅크와의 유사도로 가장 관련된 의도(top-1 또는 soft top-k)를 **선택(selection)**하고, (3) `h_query`와 선택된 의도의 메모리 임베딩(`h_memory`)을 함께 MLP projector에 입력해 `h_intent = MLP(h_query, h_memory)`를 SASRec의 표현 공간에 투영한 **prefix 토큰**으로 만들어 시퀀스 인코딩 앞에 붙인다. 이 prefix가 SASRec의 후보 생성(전체 카탈로그에 대한 스코어링) 단계 자체를 조건화(steering)한다 — reranking이나 feature-concat이 아니라 생성 단계 개입.

**왜 이 구조인가 (대안과의 차이).**
- vs. 단일 사용자 임베딩: 의도별로 분리된 메모리는 맥락 전환을 표현 가능 + per-intent 해석/평가 가능.
- vs. 온라인 LLM 재정렬: 메모리는 정적 저장소(추론 시 LLM 호출 0회) — 지연/비용 없음.
- vs. feature-concat: concat은 모델이 "무시"하기 쉬운 추가 피처로 취급될 수 있는 반면, prefix는 self-attention 입력 자체를 바꿔 후보 생성 분포를 직접 이동시킨다 (가설 — Pilot 4에서 검증).

**현재 상태.** `QAIM-Rec/` 리포지토리는 빈 디렉토리(`src/`, `data/`, `docker/`, 빈 `docker-compose.yml`)뿐이며 기존 코드 자산 없음. 이 문서는 처음부터의 설계이며, 이 문서(v0.3.1)를 최종 설계로 확정하여 이후 모든 구현은 본 문서를 기준으로 진행한다.

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
├── configs/                     # 모든 단계의 실험 설정 (yaml), 시드 포함
│   ├── data/{category}.yaml
│   ├── llm/{p1,p2,p3}.yaml       # 모델명, temperature, self-consistency T 등
│   ├── memory/bank.yaml          # 클러스터링/K 설정
│   ├── model/{sasrec,encoder,projector}.yaml
│   ├── train/{align,hybrid}.yaml
│   └── eval/{full_ranking,ablations}.yaml
│
├── data/
│   ├── raw/{category}/           # HF에서 받은 원본 (gitignore)
│   ├── processed/{category}/
│   │   ├── sequences.jsonl        # 유저별 (item_id, ts, rating, review) 시퀀스
│   │   ├── splits.json            # train/val/test 인덱스 (leave-one-out + global temporal sanity)
│   │   ├── memory_bank/{user_id}.json  # Intent Memory Bank (유저별 샤딩)
│   │   ├── pseudo_queries.jsonl   # P2 출력 (train pair 구축용 + eval query)
│   │   └── align_pairs.jsonl      # P3 출력 (query, memory_id, label)
│   └── cache/llm_cache.sqlite     # (prompt_hash → response) 캐시, 모든 LLM 호출 경유
│
├── src/
│   ├── data/          # M0: 다운로드/필터/시퀀스/split
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
    "item_ids": [str],
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
- `download.py`: HF `McAuley-Lab/Amazon-Reviews-2023`에서 선택 카테고리 review+meta 다운로드 → `data/raw/{category}/`
- `filter.py`: k-core 필터(§6) → `processed/{category}/sequences_raw.jsonl`
- `sequence.py`: 유저별 (item_id, ts, rating, review_text, item_meta) 시간순 시퀀스 생성
- `split.py`: leave-one-out + global temporal sanity → `processed/{category}/splits.json`
  - **출력**: 유저별 `{history: [...], val_target, test_target}` 인덱스
- **입력**: HF dataset id, category명, 필터 파라미터 / **출력**: `sequences.jsonl`, `splits.json`

### M1 — LLM Pipeline (`src/llm/`)
- `client.py`: open-weight 모델 wrapper(vLLM/HF) + sqlite 캐시(`prompt_hash -> response`)
- `prompts/p1_intent.py`, `p2_pseudo_query.py`, `p3_align_judge.py`: 템플릿 + JSON 파싱/검증/재시도(파싱 실패시 최대 2회 재시도, 최종 실패는 별도 로그). **(v0.3)** `p1_intent.py`는 `disposition_note`(§5) 필드를 추가로 출력 — M2 `synth.py`의 persona synthesis 입력으로 사용.
- `prompts/synth_intent_attr_summary.py` **(v0.3.2 신규)**: 클러스터 내 purpose들 + 알고리즘 집계된 attributes를 입력으로 `{"intent_description": str, "preference_signal_summary": str}`를 1회 결합 호출로 strict-JSON 생성(§2.2 (A), M2 `synth.py`에서 사용).
- `prompts/synth_persona.py` **(v0.3 신규)**: 클러스터 내 non-null `disposition_note`(들) + purpose/attributes를 입력으로 `{"tag": str, "description": str}`를 생성(§2.2 (A)/§7 결정 #4, M2 `synth.py`에서 사용. disposition_note 전부 null인 클러스터는 호출 생략).
- **입력**: (item_meta, review) 배치 또는 (query, memory) 배치 / **출력**: 파싱된 dict 리스트 + 실패율 리포트
- M0 → M1 의존 (item_meta, review 필요)

### M2 — Memory (`src/memory/`)
- `cluster.py`: 유저별 P1 결과(discriminative만) → agglomerative clustering(τ_personal) → K_personal개 그룹
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
- `losses.py`: `info_nce(emb_q, emb_pos, emb_neg)` (L_align), `retrieval_loss(scores, target_item, full_catalog)` (L_retrieval — full-softmax 또는 sampled-softmax+all-item 정규화, §7에서 확정)
- `train_align.py` (Stage 1): M1-P3 산출 (query, memory, label) pairs로 text_encoder(+router 임베딩 공간) contrastive 학습. SASRec/projector 미사용. **이 단계에서 `h_memory`(=memory_bank의 `embedding.vector`)는 고정(frozen encoder로 생성된 값)**. **(v0.3.2)** L_align 배치 구성(negative, §2.4): 1 positive당 (i) same-user 다른 메모리 전부를 hard negative로(보통 K-1≤4개) + (ii) in-batch negative(다른 샘플의 positive를 negative pool로 재사용)를 기본값으로 한다(저비용·표준 관행, 별도 승인 불필요).
- `train_hybrid.py` (Stage 2): F6a 체크포인트(`checkpoints/{exp_name}/sasrec_pretrain.pt`)에서 SASRec을 로드(**기본값: frozen**, 1e-2x LR은 ablation) + text_encoder(저-LR, F6 체크포인트에서 초기화) + projector(기본 LR) 동시 학습. `L = L_retrieval + α·L_align`. M2 메모리 뱅크에서 매 스텝 유저의 top-1 메모리를 router로 선택 → `h_query`, `h_memory` 둘 다 **on-the-fly로 현재 text_encoder를 통해 재인코딩**(§7 결정 #11 — 라우팅 후보 ≤~7개라 비용 무시 가능, memory_bank에 캐시된 `embedding.vector`는 Stage2 정합성에 의존하지 않음) → `projector(h_query, h_memory)` → prefix → SASRec.
- **입력**: M0 splits(train), M2 memory_bank, M1 align_pairs / **출력**: `checkpoints/{exp_name}/{stage}_best.pt`

### M5 — Eval (`src/eval/`)
- `full_ranking.py`: 전체 카탈로그에 대해 Recall@{5,10,20}/NDCG@{5,10,20}/MRR 계산 (sampled-negative 금지)
- `conditions.py`: 비교군 정의 — `{none, avg_intent, selected_intent} × {prefix, concat}` + `{correct_intent, wrong_intent}` + `query_only(leakage baseline)`. 각 condition은 "prefix_embeds를 어떻게 만드는가"의 함수로 통일 구현(코드 중복 방지). **(v0.3.2)** `vanilla_finetuned` **(조건부)**: F7에서 SASRec frozen(기본값)이면 `none`(=F6a 체크포인트, 무prefix)이 곧 공정한 비교 기준이므로 불필요. F7 ablation(1e-2x LR, SASRec 갱신)을 실행하는 경우에만, F7과 동일한 step/LR로 SASRec을 prefix 없이 추가 학습한 `vanilla_finetuned`을 F8에 추가 — steering 효과와 추가 finetuning 효과를 분리.
- `recovery_analysis.py` **(v0.2 신규, 권장#7 headline 실험)**: `correct_intent` / `vanilla(no-prefix)` / `wrong_intent` 3-조건의 top-N 추천 집합을 비교해 `Recovery@N = P(target ∈ TopN_steered(correct) AND target ∉ TopN_vanilla)`를 계산. `conditions.py`의 동일 prefix-생성 함수를 재사용.
- `stratify.py`: history-length bucket별, `persona.tag`별(v0.3: 코어 필드를 직접 그룹 키로 사용, 사후계산 불필요), per-memory(per-intent) recall, routing accuracy(= LLM(P3) 라벨과 router top-1 일치율), diversity(추천 리스트 내 카테고리/브랜드 엔트로피), latency(prefix 계산 ~ scoring 전체).
- **Amazon-C4 정량 평가 (v0.2 신규, 권장#5)**: `query↔memory↔item` routing을 정식 평가 — C4의 (query, item_id, user_id) 중 user_id가 본 카테고리 memory_bank에 존재하는 서브셋에 대해, router top-1 메모리로 만든 prefix로 steered ranking에서의 target item Recall/NDCG를 측정. user/item ID 오버랩 비율은 F1에서 사전 점검(§4 F1).
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

**도메인 비교.** `data/raw/`에 있는 카테고리로 A/B를 돌리되, 도메인 유형별 1개씩:
- 기능형: `Office_Products` 또는 `Pet_Supplies`
- 라이프스타일: `Amazon_Fashion` 또는 `Beauty_and_Personal_Care`
- 콘텐츠: `Books`
없는 카테고리는 건너뛰고 명시. 목표: "어느 도메인이 변별·aspect·멀티맥락이 풍부한지" 실측.

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

### Stage P4 — Minimal Steering Sanity (v0.2: 최소 학습 projector)
- **입력**: 1개 카테고리, Pilot 2/3 대상 사용자 중 K_personal≥2 확인된 소수(N≈20~50명)에 대해 "올바른 intent" 메모리(M2 실제 산출) + "틀린 intent" 메모리(다른 무관 클러스터 또는 다른 유저의 메모리를 swap)
- **실행**: 처음부터 학습한 소형 SASRec(M3, 1개 카테고리 데이터로 빠르게 사전학습) + **(v0.2 변경)** text_encoder(소형, 일부 레이어만) + projector를 P3의 align pair 소형 서브셋으로 **최소 학습**(few epoch, L_align만 또는 L_align+소량 L_retrieval) 후 `h_intent = MLP(h_query, h_memory)`로 prefix 주입.
- **측정**:
  - (a) controllability: top-k(올바른 intent) vs top-k(틀린 intent) 의 Jaccard overlap이 top-k(올바른) vs top-k(올바른, 다른 seed) 보다 유의하게 낮은가 → prefix가 출력에 실제 영향을 주는가
  - (b) 방향성: top-k(올바른 intent, 최소학습 projector) 의 Recall@10(held-out 상호작용 기준) > top-k(feature-concat baseline, 동일 h_query/h_memory를 concat하여 SASRec 입력에 추가) 의 Recall@10
- **출력**: `results/pilot/pilot4_report.json`
- **성공 기준 (go)**: (a) 성립 (prefix가 controllable) **and** (b)가 최소 50% 이상의 샘플 유저에서 성립(작고 노이즈 있어도 방향성 확인)
- **중단/조정 기준**: (a) 불성립 → prefix injection 지점/방식 재검토(§7 결정 #5). (b) 불성립이지만 (a) 성립 → 최소 학습 규모의 한계로 보고 전체 파이프라인 단계(F7/F8)에서 재검증(hard gate 아님)
- **명시적 한계 (요구사항 [필수]#2)**: Pilot 4의 최소 학습은 few-epoch/소형 슬라이스이므로, "prefix가 feature-concat을 능가하는가"에 대한 **확정적 검증이 아니다**. 만약 (b) 측정 자체가 자원/시간상 불가능하면, 본 항목을 (a)만 측정하는 것으로 축소하고 "(b)에 대한 실질적 검증은 F7(학습)+F8(평가)로 이연됨"을 본 리포트에 명시한다 — 미학습 prefix로 (b)를 판정하지 않는다.
- **예상 비용**: SASRec 소형 사전학습 + text_encoder/projector 최소 학습(1 카테고리, few epoch, 단일 GPU 1~2시간 내외 추정 — 카테고리 규모/GPU에 따라 §7 결정#7 확정 후 재추정)

### Gate — Go/No-Go 리뷰
- 4개 파일럿 리포트를 함께 검토. P1, P2는 hard gate(미충족 시 카테고리/프롬프트 재설계까지 재귀), P3/P4는 soft gate(방향 수정 후 진행 가능). **이 리뷰 시점에 사용자 확인 필수** — 전체 파이프라인(아래) 착수 전 중단점.

### Stage F — 전체 파이프라인 (Gate 통과 후)
| Stage | 내용 | 입력 | 출력 | 비용 추정 |
|---|---|---|---|---|
| F1a | M0 실행 + **데이터 볼륨 리포트** — 후보 2~3 카테고리(§7 결정#6) 각각의 유저 수/아이템 수/평균 이력 길이/k-core 필터 후 잔존율 산출 | HF dataset | volume_report.json (카테고리별 표) | 다운로드/전처리, GPU 불필요. **본 리포트 공유 후 카테고리 최종 확정에 대한 사용자 승인** (결정#6 보류 조건) |
| F1b | F1a에서 확정된 카테고리로 M0 전체 실행(sequences/splits 확정). **Amazon-C4 user/item ID 오버랩 비율 점검**(§6.5, §4 F8 입력용) | F1a 승인 | sequences.jsonl, splits.json, c4_overlap_report.json | 전처리, GPU 불필요 |
| F2 | M1 P1 전체 실행 (train-history 전체) | F1b 출력 | P1 결과 (캐시) | LLM 호출 ≈ Σ유저별 이력 길이 — 카테고리/서브샘플 크기에 비례, 사전 추정치 산출 후 **사용자 승인** |
| F3 | M2 메모리 뱅크 구축 (personal + prototype, P2의 fallback 비율 반영) | F1b+F2 | memory_bank/*.json, _prototypes.json | LLM 호출 ≈ Σ(2·K_personal/유저) + 2P(prototype) (v0.3.2: 결합 호출 1회 + persona 0~1회 = 최대 2회/클러스터, disposition_note 전부 null인 클러스터는 persona synthesis 생략) |
| F4 | M1 P2 (pseudo-query: 학습용 + eval query) | F1b | pseudo_queries.jsonl | LLM 호출 ≈ Σ유저별 이력 길이 (P1과 별도 호출이나 같은 입력 — 배치 동시 실행 권장) |
| F5 | M1 P3 (align judge, self-consistency T=3) | F3+F4 | align_pairs.jsonl | LLM 호출 ≈ (query, memory 후보쌍 수) × T — 후보쌍 수 제한 전략 필요(§7) |
| F6 | M4 Stage1 (L_align) | F5 | encoder ckpt | GPU, 1 카테고리 기준 추정 후 승인 |
| F6a **(v0.3.2 신규)** | SASRec vanilla pretraining (item_id 시퀀스만, 텍스트/의도 미사용) — F6과 입력이 겹치지 않아 **병렬 실행 가능** | F1b (sequences/splits, train) | checkpoints/{exp_name}/sasrec_pretrain.pt | GPU, F6과 동시 추정·승인 |
| F7 | M4 Stage2 (hybrid, L=L_retrieval+α·L_align, `h_query`/`h_memory` on-the-fly 재인코딩, §7#11). SASRec은 F6a 체크포인트에서 로드(기본값 frozen) | F1b+F3+F6+F6a | 최종 ckpt | GPU, 추정 후 승인 |
| F8 | M5 전체 평가 — 전 비교군 × 전 카테고리 + **(v0.2 신규)** `recovery_analysis.py` headline 실험(correct/vanilla/wrong-intent 3-way, 권장#7) + **Amazon-C4 정량 routing 평가**(§6.5, 권장#5) + `query_only` 비교군을 Pilot 3의 leakage floor와 함께 보고. vanilla 베이스라인은 F6a 체크포인트를 F7과 공유(조건부 `vanilla_finetuned`, §3 M5) | F1b+F3+F6a+F7 | results/metrics.json, recovery.json, c4_routing_eval.json | 추론만, 상대적으로 경량 (C4 서브셋 크기는 F1b 오버랩 점검 결과에 따름) |

**원칙**: F2, F5(LLM 대량 호출)와 F6/F6a/F7(전체 학습) 착수 전 각각 예상 호출수/비용/시간 추정치를 공유하고 **개별 승인**을 받는다(스펙의 "큰 연산 전 확인" 요구사항). F1a→F1b 사이에도 카테고리 확정에 대한 승인이 필요(§7 결정#6).

## 5. LLM 프롬프트 초안 (P1~P3)

모든 프롬프트는 strict-JSON 출력 + few-shot 동봉을 기본으로 한다. 아래는 system+task 초안.

### P1-base — Intent 추출, purpose 중심 (narrow baseline, 변경 없음)

구현: `src/llm/prompts/p1_intent.py` (PROMPT_VERSION=`p1_base_v1`)

```text
[SYSTEM]
You analyze a single e-commerce purchase (item metadata + the buyer's review) to infer
WHY they bought it (situational purpose/use-case) and WHAT they prioritized when choosing it.

Output strict JSON only, matching this schema:
{
  "purpose": string,            // one sentence: the situational need/use-case this purchase
                                 // serves (e.g. "setting up a quiet workspace for late-night study"),
                                 // NOT a restatement of the product category
                                 // (e.g. NOT "wanted a good desk lamp")
  "is_discriminative": boolean, // false if "purpose" is so generic it would fit almost any
                                 // purchase in this category (e.g. "for daily use",
                                 // "good quality product", "as a gift")
  "disposition_note": string | null,  // ONE sentence describing any UNUSUAL or ATYPICAL buyer
                                 // disposition — null if ordinary.
  "preference_attrs": {
    "price_band": "budget" | "mid-range" | "premium" | "unknown",
    "feature_priorities": [string, ...max 3, ordered by emphasis in the review],
    "brand_tendency": string,
    "style": string | null,
    "avoid": [string] | null
  }
}

[USER]
Item title: {title}
Category: {category}
Brand: {brand}
Price: {price}
Rating: {rating}/5
Review: """{review_text}"""
```

**파싱/검증**: `purpose` 비어있음/2문장 이상/제목 그대로 복사 → 재시도(최대 2회). `is_discriminative=false`인 항목은 클러스터링에서 제외(§2.2). `disposition_note`는 null 허용.

---

### P1-aspect — 도메인 일반 contextual preference 추출 (신규, A/B 비교용)

구현: `src/llm/prompts/p1_aspect.py` (PROMPT_VERSION=`p1_aspect_v1`)

**프레이밍**: intent = "현재 질의로 활성화 가능한 맥락적 선호 단위(query-activatable contextual preference)". 기능형 용도뿐 아니라 취향/스타일/장르, 소비 맥락, 선택 기준 등 모든 선호 차원을 포괄.

```text
[SYSTEM]
You analyze a single e-commerce interaction (item metadata + buyer review) to extract
the user's CONTEXTUAL PREFERENCE — the specific slice of this person's preferences
activated by this particular interaction.

Output strict JSON only:
{
  "contextual_intent": [string],   // 1 sentence normally. Emit up to 2 ONLY if the review
                                    // CLEARLY shows two DISTINCT contexts (e.g. commuting AND travel).
                                    // When ambiguous, emit exactly 1.
  "is_discriminative": boolean,    // true if specific enough to distinguish this person
                                    // from someone with a different preference context.
                                    // NOT about whether a purchase purpose exists.
  "aspect_coverage": {
    "usage_context": string | null,           // functional use-case or consumption occasion/mood
    "taste_or_style_preference": string | null, // aesthetic/style/genre/mood preference
    "selection_criteria": [string, max 3],    // what the user evaluated when choosing
    "preference_tradeoff": string | null      // what they prioritized OVER something else
  },
  "disposition_note": string | null,
  "preference_attrs": {
    "price_band": "budget" | "mid-range" | "premium" | "unknown",
    "feature_priorities": [string, max 3],
    "brand_or_creator_tendency": "brand|author|creator|designer|agnostic",
    "style": string | null,
    "avoid": [string] | null
  }
}

[USER]
Item title: {title}
Category: {category}
Brand/Creator: {brand}
Price: {price}
Rating: {rating}/5
Review: """{review_text}"""
```

**파생 지표**: `aspect_coverage_valid` = aspect_coverage 4필드 중 ≥2개 non-null/non-empty (`is_discriminative`와 별개). `contextual_intent` 길이 ≥2인 샘플 = 멀티맥락 포착 확인.

**파싱/검증**: `contextual_intent` 비어있음 → 재시도. `contextual_intent` 길이 > 2이면 앞 2개만 보존. clustering은 이번 단계 실행 안 함.

### P2 — Pseudo-query 생성 (per 구매로그)

```text
[SYSTEM]
Convert this review into the search query the buyer likely typed BEFORE purchasing —
phrased as a general need, NOT identifying this specific product.

Output strict JSON:
{
  "query": string | null   // 3-12 word natural search query, or null if no clear
                            // pre-purchase need is expressed in the review
}

STRICT RULES:
- MUST NOT contain: brand names, model numbers/names, or exact unique specs
  (exact capacity/size numbers, SKU-like tokens) that would identify this specific item.
- MUST express the underlying need/use-case in generalized terms
  (e.g. "quiet keyboard for late night typing", NOT "Logitech MX Keys Mini").
- If the review only describes satisfaction without revealing a need, output {"query": null}.

[USER]
Item title: {title}   <-- for context only, do NOT copy into output
Category: {category}
Review: """{review_text}"""
```

**파싱/검증**: 출력 `query`에 `title`의 토큰(브랜드/모델명 등)이 포함되면 식별정보 누출로 간주, 재생성(최대 2회) 또는 해당 로그 제외. 이 검증 로직은 Pilot 3 결과에 따라 강화(§4 Stage P3).

### P3 — 질의-메모리 정렬 Judge (학습쌍 구축)

```text
[SYSTEM]
You judge whether a MEMORY (a customer's inferred purchase intent and preferences)
is RELEVANT to a search QUERY — i.e., would a product matching this memory's intent
and preferences satisfy what the query is looking for?

Output strict JSON:
{
  "label": "positive" | "negative",
  "rationale": string   // one sentence
}

[USER]
QUERY: "{pseudo_query}"

MEMORY:
- intent: "{intent_description}"
- preferences: "{preference_signal.summary}"
```

**Self-consistency**: temperature=0.7, T=3회 독립 호출 → 다수결. 2:1도 다수결 채택, **동률 없음(T=3 홀수)**이므로 ambiguous 케이스는 발생하지 않으나, 3회 중 JSON 파싱 실패가 1회 이상이면 해당 쌍은 제외(노이즈 억제). Positive/negative 쌍 구축 시 **같은 유저의 다른 메모리를 hard negative**로 우선 사용(랜덤 negative보다 informative).

**후보쌍 생성 전략 (비용 통제, §7 결정 #9 연동)**: 모든 (query × memory) 조합은 비용 폭발 → 유저 자신의 K개 메모리 + 무작위 다른 유저 메모리 M개(M=2~3)만 후보로 구성, 총 후보쌍 ≈ Σ유저 (이력 길이 × (K+M)).

## 6. 데이터 설계 (획득/필터/temporal split/시퀀스 구성)

### 6.1 카테고리 선택 (§7 결정 #6 — 제안 수용, F1a 데이터 볼륨 리포트로 최종 확정)
"query-activatable contextual preference"가 풍부하고 리뷰 텍스트가 충실한 카테고리를 우선한다. P1 A/B(§4 Stage P1)에서는 도메인 유형별 실측을 위해 3종류를 대상으로 한다:
- **기능형**: `Office_Products`, `Pet_Supplies` (또는 `Tools_and_Home_Improvement`) — 기능적 용도·목적 신호 풍부
- **라이프스타일**: `Amazon_Fashion`, `Beauty_and_Personal_Care` — 취향/스타일 신호 풍부 (가설: 변별·aspect가 기능형보다 다양할 것)
- **콘텐츠**: `Books` — 장르/분위기/소비 맥락 신호 (p1_base의 purpose 추출이 어렵지만 p1_aspect의 taste 필드는 풍부할 것으로 가설)
P1 A/B의 목적: 어느 도메인에서 discriminative·aspect_coverage·멀티맥락이 풍부한지 실측. `data/raw/`에 없는 카테고리는 건너뜀.
- **(v0.2 추가, 결정#6 회신 반영)** F1b 착수 전 **F1a에서 각 후보 카테고리의 유저 수/아이템 수/평균 이력 길이/k-core 필터 후 잔존율을 데이터 볼륨 리포트로 먼저 보고**하고, 이를 바탕으로 최종 카테고리(2~3개)를 확정한다(§4 Stage F1a).
- Amazon-C4: 종전 "별도 검증용/정성"에서 **(v0.2 변경)** §6.5에서 정식 정량 평가 입력으로 격상.

### 6.2 필터링
- k-core 필터(반복 적용): 사용자 ≥5 인터랙션, 아이템 ≥5 인터랙션 (표준 SASRec 전처리와 동일선상)
- 리뷰 텍스트 길이 ≥ 10 토큰인 인터랙션만 P1 입력 대상(짧은 리뷰는 의도 추출 신뢰도 낮음) — 단, 시퀀스 자체(추천 대상)에서는 제외하지 않음(리뷰 없는 인터랙션도 시퀀스엔 남되, 메모리 evidence에는 못 들어감)
- 메모리 구축 대상 = 이력 길이 ≥ L (Pilot 2에서 L 확정, 1차 가설 L=8) 사용자. L 미만 사용자는 **메모리 없이 평가**(콜드/숏 히스토리 세그먼트로 stratified 비교에 포함 — 제외하지 않음)
- **(v0.3.2, 선택)** M1 P2(pseudo-query) 입력을 `rating≥4` 리뷰로 제한하는 안 — Amazon-C4(§6.5)가 5점 리뷰 기반으로 생성되어 분포 비교에 유리하고 "구매 전 욕구" 프레이밍에 더 부합하나, **하드 게이트 아님**: 기본값은 전체 평점 사용이며, `rating≥4` 제한은 §4 Stage F의 **ablation 옵션**으로만 고려(기본 파이프라인에 영향 없음).

### 6.3 시퀀스 구성
유저별 `(item_id, timestamp, rating, review_text, item_meta)`를 timestamp 오름차순으로 정렬한 리스트.

### 6.4 Split — leave-one-out + global temporal sanity (하이브리드)
1. **Leave-one-out** (SASRec 표준): 유저별 마지막 아이템 = test target, 마지막-1 = val target, 나머지 = history
2. **Global temporal sanity**: 전체 데이터셋의 timestamp 분포에서 상위 X%(예: 95th percentile) 시점 `T_cut`을 계산. test target의 timestamp가 `T_cut`보다 현저히 이른(= 데이터 후반부 트렌드를 전혀 반영 못하는) 유저는 평가에서 제외하거나 별도 세그먼트로 표시 — population-level future leakage 점검
3. **메모리 구축 = history 구간만** (val/test target 제외) — §2.2의 leakage 방지와 동일 제약을 split 레벨에서도 명시적으로 강제

### 6.5 Amazon-C4 활용 (v0.2 변경 — 권장#5: 정량 routing 평가로 격상)
- **정량 평가 (신규, F8)**: 스키마(`query, item_id, user_id`)에서 `user_id`가 본 카테고리의 memory_bank(F3)에 존재하는 서브셋을 추출 → 해당 `query`로 router top-1 메모리를 선택 → `h_intent = MLP(h_query, h_memory)` prefix로 steered ranking 수행 → `item_id`(C4가 명시한 target)에 대한 Recall@{10,20}/NDCG@{10,20} 측정. 이는 query↔memory↔item 연쇄가 실제 사용자 질의로도 작동하는지의 **직접적 정량 증거**.
- **오버랩 caveat (F1b에서 사전 점검)**: Amazon-C4와 Amazon Reviews 2023의 `user_id`/`item_id` 네임스페이스가 동일 스냅샷 기준이 아닐 수 있음 — F1b에서 오버랩 비율을 먼저 측정(`c4_overlap_report.json`)하고, 오버랩이 낮으면 (i) 해당 서브셋만으로 평가 규모를 작게 보고하거나 (ii) `item_id`만 카탈로그에 존재하면 `user_id` 없이 "query→item" zero-shot routing(메모리는 prototype만 사용)으로 축소 평가하는 대안을 명시.
- **분포 검증 (기존 유지)**: P2 pseudo-query 분포와 C4 실제 질의 분포 비교(정성/분포 분석 보조 자료로 유지).
- end-to-end **학습**은 Amazon Reviews 2023 하나로 닫는다(스펙 원칙 유지) — C4는 평가 전용 입력.

## 7. 설계 결정 목록 & 확인 필요 사항

각 항목: 결정 내용 / 옵션별 장단점 / 현재 제안 / 확인 필요 여부.

**#1. 메모리 K 결정 방식 — ✅ 확정 (회신: agglomerative + 공유 prototype fallback 필수)**
- (a) 사용자별 자동(agglomerative + distance threshold, k_min~k_max): 실제 의도 다양성 반영, 그러나 threshold τ가 카테고리마다 다를 수 있음
- (b) 고정 K (예: 항상 3): 구현 단순, 그러나 의도가 1개뿐인 유저에 억지로 3개를 만들면 노이즈 메모리 생성
- (c) 공유 prototype: cold-start/짧은 이력에 강건, 그러나 유저 특화 brand_tendency 등 손실
- **확정**: (a)+(c) 결합 — personal agglomerative(τ_personal, k_min=2·k_max=5)로 K_personal 산출, K_personal < k_min이면 population-level prototype(§2.2 (B), τ_global, P≈8-15개)에서 부족분을 보충(`assemble`, §3 M2). K_personal==0인 유저는 전체 구매이력 임베딩 centroid로 prototype 매칭. τ_personal/τ_global 실제 값과 fallback 발동 비율은 Pilot 2(§4 Stage P2)에서 실측해 F3 비용 추정에 반영.

**#2. preference_signal 저장 형식**
- 구조화 필드만: ablation/분석 용이하나 임베딩 입력으로는 부자연스러움
- 자유 텍스트만: 임베딩 품질 좋을 가능성, 그러나 controllability 실험(필드 swap)에 불리
- **제안**: §2.1처럼 **둘 다 보존**(attributes + summary, 동일 클러스터에서 동시 생성). 추가 비용은 클러스터당 LLM 1회 — 미미. **확인 필요 없음(제안대로 진행 가능), 단 이견 있으면 표시**

**#3. 라우팅/projector 임베딩 입력 — (v0.2 변경: 기본값 description+summary로 확정)**
- description만 vs description+summary 결합
- **(v0.2 변경)** §2.1에서 `embedding.source_text = f"{intent_description} {summary}"`(description+summary 결합)를 `h_memory`의 기본값으로 확정 — preference_signal까지 포함해야 projector(`h_intent = MLP(h_query, h_memory)`)가 선호 정보를 충분히 반영할 수 있음(필수#1과 정합). description만 사용하는 버전은 §4 Stage F의 ablation으로 비교(추가 임베딩 계산만 필요, 저비용). **확인 필요 없음**

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

**#6. 카테고리 선택 — ✅ 제안 수용 (단, F1a 데이터 볼륨 리포트로 최종 확정, §6.1)**
- §6.1 제안(Office_Products, Tools_and_Home_Improvement, Pet_Supplies/Electronics) 수용. 착수 전 F1a(§4 Stage F)에서 각 카테고리의 유저/아이템 수, 평균 이력 길이, k-core 필터 후 잔존율을 보고하고 그 결과로 2~3개 최종 확정 — **F1a 리포트에 대한 승인 필요**.

**#7. LLM 선택 (P1~P3) — ✅ 확정 (GPU 정보 확보 후 unblock)**
- 보유 자원: 4× RTX 2080 Ti (11GB VRAM/장, CUDA 12.2, driver 535.274.02, Turing/sm75).
- **확정**: `Qwen2.5-7B-Instruct`, **양자화 1차값 = bnb-nf4(4-bit)** — Turing(sm75)에서 AWQ 커널 지원이 불안정할 수 있어, AWQ는 향후 Docker 환경 셋업 단계에서 검증 후 대안으로 고려(이 plan 범위 밖). 단일 GPU에 적재(~6-8GB), Pilot1(N=200)은 단일 GPU로 충분.
- **서빙 비고**: Turing은 FlashAttention-2 미지원(Ampere+ 전용) → vLLM/HF 서빙 시 `xformers` 또는 `eager` 백엔드로 폴백.
- Pilot1 실행 전 `configs/llm/p1.yaml`의 `model_id` 플레이스홀더(`<TBD-pending-GPU-info>`)를 `Qwen/Qwen2.5-7B-Instruct`(bnb-nf4)로 교체.

**#8. 임베딩 모델 (routing/text encoder 초기화)**
- 후보: `BAAI/bge-base-en-v1.5`, `sentence-transformers/all-mpnet-base-v2`, `Alibaba-NLP/gte-base-en-v1.5`
- **제안**: bge-base 계열(검색/쿼리-문서 매칭에 강함, 본 연구의 query↔memory 매칭과 task 정합). **확인 필요 없음(제안대로 진행 가능)**

**#9. P3 후보쌍 생성 / L_retrieval 정의**
- 후보쌍 폭발 방지: 유저 자신의 K개 메모리 + 무작위 타유저 메모리 M개(§5)
- L_retrieval: full-softmax(전체 카탈로그, 카탈로그 크기에 따라 비용 큼) vs in-batch negative + 주기적 full-ranking 검증
- **제안**: 학습은 in-batch negative(효율) + L_align과의 균형(α)은 작은 그리드(예: {0.1, 0.5, 1.0})로 탐색, **평가만** full-ranking(스펙 원칙 유지). **확인 필요 없음**

**#10. Split: leave-one-out vs 순수 global temporal**
- §6.4에서 하이브리드로 제안(LOO 골격 + global temporal sanity 필터). 순수 global temporal(전 유저 동일 cutoff)은 유저별 시퀀스 길이가 불균등해져 짧은 이력 유저가 과도하게 배제될 위험
- **확인 필요 없음(제안대로 진행), 단 평가 결과 해석 시 이 선택을 명시**

**#11. (v0.3.2 신규, ✅ 확정) Stage2 `h_memory`/`h_query` staleness — on-the-fly 재인코딩**
- **문제**: M4 `train_hybrid.py`는 text_encoder를 저-LR로 계속 갱신하므로, M2에서 캐시된 `memory_bank`의 `embedding.vector`를 그대로 쓰면 Stage1/M2 시점 기준으로 stale해짐.
- **분석**: 유저당 라우팅 후보(personal+prototype)는 ≤~7개로 매우 적음 → 매 학습 스텝마다 해당 후보 전체 + `h_query`를 현재 text_encoder로 재인코딩하는 비용은 무시 가능(짧은 텍스트, 소수 forward pass).
- **확정**: Stage2는 **항상 on-the-fly 재인코딩**(라우팅 후보 재인코딩 → top-1 재라우팅 → projector 입력)을 기본값으로 한다. `memory_bank`의 캐시된 `embedding.vector`는 M2 산출 시점의 클러스터링/prototype 매칭/Stage1 학습 타깃으로만 사용되고 Stage2 정합성에는 의존하지 않음 — 별도의 주기적 재캐싱 전략 불필요. **확인 필요 없음**.

---
## 8. 다음 단계

1. ~~`연구아이디어.pdf` 대조~~ — v0.2에서 완료, v0.3.2에서 재대조(§0.1) 완료.
2. ~~#1, #4, #5, #6, #7, #11에 대한 회신~~ — v0.2~v0.3.2에서 반영·확정. **모든 §7 항목 확정 완료** — Pilot 1 골격/config의 `model_id`를 `Qwen/Qwen2.5-7B-Instruct`(bnb-nf4)로 확정 적용.
3. `src/pilot/` 골격(함수 시그니처 + config 템플릿) 작성 — 구현/실행 없음.
4. 골격 작성 후 Stage P1(§4)의 LLM 호출 수(N=200) 기준 예상 비용(시간/리소스) 추정치를 공유하고, Pilot 1 실행 전 승인을 받는다.
