eady to code?

 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Pilot 1 실행 가능하게 구현 + N=200 비용 추정 후 승인 요청

 Context
                                                  
 plan.md §8 "다음 단계" 4번: "골격 작성 후 Stage P1(§4)의 LLM 호출 수(N=200) 기준 예상
 비용(시간/리소스) 추정치를 공유하고, Pilot 1 실행 전 승인을 받는다." 현재 src/pilot/ 골격은
 작성되었고 Docker/Ollama(gemma4:26b) 환경도 검증 완료(GPU 4개 인식, /qaim-rec
 워크디렉터리, Ollama 연결 확인). 그러나 load_config/run_pilot1은 아직
 NotImplementedError이고, src/llm/(M1)·src/data/(M0)는 아직 존재하지 않는다.

 §1 설계 원칙("Pilot은 M0~M2를 소규모 config로 재사용, 별도 구현 금지")에 따라, Pilot 1을
 실제로 돌리려면 M0(샘플링)·M1(LLM 클라이언트+P1 프롬프트)의 최소 구현이 필요하다. 이번
 작업의 목표는:
 1. Pilot 1을 end-to-end로 실행 가능하게 만드는 최소 구현 (M0 샘플링, M1 client+p1_intent,
 common.py, run_pilot1)
 2. 작은 smoke test(N=3 정도)로 gemma4:26b 1회 호출당 실측 시간을 측정
 3. 측정값으로 N=200 비용/시간 추정치를 산출해 사용자에게 공유하고, 전체 N=200 실행은 사용자
 승인 후에 진행 (이 작업에서는 실행하지 않음)

 1. src/pilot/common.py — load_config/write_report 구현

 - load_config(path, cls): yaml.safe_load(open(path)) → cls(**data). 누락된 키는 dataclass
 기본값으로 채워짐(모든 Pilot Config 필드에 기본값이 있음).
 - write_report(report, path): os.makedirs(os.path.dirname(path), exist_ok=True) 후
 json.dump(dataclasses.asdict(report), f, indent=2, ensure_ascii=False).

 2. src/llm/ (M1, 신규) — Ollama 클라이언트 + sqlite 캐시

 src/llm/__init__.py — 빈 패키지 마커

 src/llm/client.py

 - LLMConfig dataclass: model_id, api_url, temperature, max_new_tokens, prompt_version,
 cache_db, retry_max=2 (configs/llm/p1.yaml 필드와 1:1 대응; self_consistency_t는
 p3.yaml에만 있으므로 Optional[int]=None)
 - load_llm_config(path) -> LLMConfig: yaml 로드 (common.load_config 재사용)
 - LLMClient:
   - __init__(config): cache_db 부모 디렉터리 생성, sqlite 연결, CREATE TABLE IF NOT EXISTS
 cache (key TEXT PRIMARY KEY, response TEXT)
   - generate(prompt: str) -> str: 캐시 키 =
 sha256(f"{prompt_version}|{model_id}|{temperature}|{prompt}"). 캐시 hit이면 즉시 반환(0
 비용). miss면 requests.post(api_url, json={"model": model_id, "prompt": prompt, "stream":
 False, "options": {"temperature": temperature, "num_predict": max_new_tokens}}) →
 response.json()["response"] → 캐시 저장 후 반환.
 - 캐시 키에 prompt_template_version(=prompt_version)·model_id·input_payload(=prompt 전체,
 system+user 포함) 포함 → plan.md §1 캐시 설계와 일치.

 3. src/llm/prompts/p1_intent.py (M1)

 - src/llm/prompts/__init__.py — 빈 패키지 마커
 - SYSTEM_PROMPT: plan.md §5 P1의 [SYSTEM] 블록 그대로 (strict-JSON 스키마: purpose,
 is_discriminative, disposition_note,
 preference_attrs.{price_band,feature_priorities,brand_tendency,style,avoid})
 - build_prompt(item: dict) -> str: [USER] 블록을
 title/category/brand/price/rating/review_text로 포맷, SYSTEM_PROMPT와 결합한 전체 prompt
 문자열 반환
 - parse_response(raw_text: str) -> dict | None:
   - 마크다운 코드펜스(json ... ) 제거 후 json.loads
   - 필수 키 존재 확인 (purpose, is_discriminative, disposition_note, preference_attrs)
   - purpose 검증(plan.md §5 파싱/검증): 비어있음 / 2문장 이상(./!/? 개수로 근사) / item
 title과 동일(대소문자 무시 strip 비교) → 검증 실패시 None 반환(재시도 트리거)
 - run_p1(client: LLMClient, item: dict, retry_max: int) -> dict | None: build_prompt →
 client.generate → parse_response, 실패시 최대 retry_max회 재시도(plan.md §5), 모두 실패하면
 None

 4. src/data/sample.py (M0 최소 구현, Pilot 1용)

 - src/data/__init__.py — 빈 패키지 마커
 - sample_review_pairs(category: str, n_samples: int, seed: int, min_review_tokens: int =
 10) -> list[dict]:
   a. datasets.load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}",
 split="full", streaming=True, trust_remote_code=True)를 스트리밍하며 text 토큰
 수(len(text.split())) ≥ min_review_tokens인 리뷰만 대상으로 reservoir sampling(seed 고정,
 random.Random(seed))으로 정확히 n_samples개 선택 — 전체 다운로드 없이 결정론적 샘플 확보
   b. 선택된 리뷰들의 parent_asin 집합 수집
   c. f"raw_meta_{category}" 설정을 동일하게 스트리밍하며 parent_asin이 위 집합에 속하는
 항목만 dict로 수집({parent_asin: meta})
   d. 각 리뷰에 대응 meta를 join하여 {"title", "category", "brand", "price", "rating",
 "review_text", "parent_asin"} 형태의 dict 리스트로 반환 — p1_intent.build_prompt의 입력
 스키마와 1:1 대응
   - 카테고리 이름/raw_review_*·raw_meta_* config 이름은 HF 데이터셋 카드 기준(실제 다운로드
 시 1차 검증 — 이름이 다르면 즉시 예외로 드러남, 별도 방어코드 불필요)

 5. src/pilot/pilot1_intent_extraction.py::run_pilot1 구현

 1. sample_review_pairs(config.category, config.n_samples, config.seed) → N개 (item_meta,
 review) dict
 2. load_llm_config(config.llm_config_path) → LLMClient 생성
 3. 각 샘플에 run_p1(client, item, config.retry_max) 적용 (tqdm 진행바)
 4. 집계:
   - n_total = N, n_parsed = None이 아닌 결과 수, parse_success_rate = n_parsed/n_total
   - n_discriminative = is_discriminative == True인 파싱 결과 수, discriminative_ratio =
 n_discriminative/n_parsed
   - disposition_note_nonnull_ratio = disposition_note is not None인 비율 (파싱된 것 중)
   - purpose_length_stats: 파싱된 purpose들의 단어 수 분포(min/max/mean/median)
   - go_nogo = parse_success_rate >= 0.95 and discriminative_ratio >= 0.40
   - notes: go/no-go 사유 + 카테고리/N 요약
 5. Pilot1Report(...) 반환

 6. Smoke test + N=200 비용 추정 (이번 작업에서 실행, 결과만 보고)

 - docker exec qaim-rec python -m src.pilot.pilot1_intent_extraction --config <임시로
 n_samples=3인 테스트 config> 실행해 파이프라인 동작 확인 (다운로드 포함 첫 실행 시간은
 별도로 보고)
 - 캐시 워밍 후(2회차 실행) gemma4:26b 1회 호출당 평균 latency 측정
 - 평균 latency × 200으로 N=200 전체 LLM 호출 시간 추정, 데이터 다운로드/HF 캐시 시간은 별도
 항목으로 분리해 보고
 - N=200 풀 실행(configs/pilot/pilot1.yaml 그대로)은 추정치 공유 후 사용자 승인을 받고 진행
 — 이번 턴에서는 실행하지 않음

 검증

 - python -m src.pilot.pilot1_intent_extraction --config <smoke-config> (N=3)이
 results/pilot/에 JSON 리포트를 정상 생성하는지 확인
 - 리포트의 purpose_length_stats/parse_success_rate/discriminative_ratio 값이 합리적
 범위인지 수동 확인