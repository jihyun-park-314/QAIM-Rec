# QAIM-Rec — 결정 기록 (decisions.md)

> 이 파일은 **변경 이력·기각된 대안·각 버전의 결정 사유**를 담는다. *현재 확정 설계*는 `plan.md`를 보라(여기엔 override가 없다).
> - 아래 **변경 이력(Changelog)** 은 최신(v0.4.14)→과거(v0.3) 순. 각 블록은 "왜 그 결정을 했는가"의 기록이며, 그 결정의 *적용 결과*는 plan.md 본문에 이미 반영돼 있다.
> - 맨 아래 **(구) LLM 프롬프트 초안**은 설계 초기의 초안이다. **실제 운영 프롬프트는 `config/prompts/{dataset}/{domain}/*.txt`에 외부화**돼 있으므로, 이 초안은 역사적 참고용이다.

---

## 변경 이력 (Changelog) — 최신순

> **v0.4.19 (1순위 작업 설계 — 타겟-리뷰 정합쿼리 평가 프로토콜: 목적·함정·검증게이트)** — C(메커니즘)가 X-target SE 14로 증명됐으나 *circularity 상한*. 이를 *circularity-robust 실성능*으로 끌어내리는 핵심 측정의 설계. 7회 측정함정 교훈을 *선제* 반영. **이 블록이 본문·이전 changelog의 상충 기술을 우선한다.**
>
> - **작업 목적 (무엇을 증명하나)**: ① circularity 제거(식별자 가림) 후에도 steering이 작동하나(누설 아닌 진짜 추천), ② 절대성능 상한 = "이상적 쿼리(타깃리뷰)"에서 vanilla 대비 lift, ③ train쿼리(미정렬)↔eval쿼리(정렬) 분포차가 성능에 주는 영향(v0.4.13 "Stage1=gap 축소" 첫 측정). reviewer 첫 질문("X로 X맞추면 누설 아니냐")에 답하는 측정.
> - **★선제 식별 함정 5 (설계에 박음)**: (1)타깃리뷰→쿼리면 쿼리가 타깃 묘사=누설 → P2 eval 프롬프트가 *식별자(제목/저자/캐릭터) 가리고 의도축(장르/톤/구조)만*, leakage rate 측정 후 평가. (2)타깃 W 리뷰가 메모리에 있으면 라우팅 자명 → W는 미래라 train-history 메모리에 없음 확인(v0.4.17 W∈memory=0%), eval쿼리가 *기존 train-history 메모리*로 라우팅됨을 확인. (3)eval쿼리 생성=새 LLM노이즈 → train과 동일 프롬프트·동일 검증(provenance/leakage). (4)타깃에 리뷰 없는 유저는 eval쿼리 불가 → test타깃 리뷰(≥10단어) 보유 유저 % = eval 모집단 확정. (5)평가경로 어긋남(7회 교훈) → eval쿼리 평가가 X-target/학습과 *동일 prefix 함수* 재사용, 새 경로 금지.
> - **★검증 게이트 (생성 후)**: eval쿼리 leakage rate(식별자 남은 %, 낮아야), null rate, 모집단 크기(타깃리뷰 보유 %), train쿼리 대비 길이/분포. 라우팅이 train-history 메모리로 가는 비율.
> - **★핵심 측정 (3층 비교)**: 동일 test 유저·동일 타깃 W에 대해 — (i) vanilla(prefix 없음), (ii) steered+train쿼리(과거 의도, 미정렬=하한), (iii) steered+eval쿼리(타깃 의도, 정렬=상한). Recall@10/20 + SE_ratio + improve/degrade. **(iii)>(i) = circularity-robust 작동(식별자 가렸으니), (iii)>(ii) = 쿼리 정합의 가치(=Stage1이 좁혀야 할 gap의 크기).**
> - **순서 (생성 전 준비 필수)**: STEP0(모집단·프롬프트·평가경로 준비, LLM 0) → STEP1(eval쿼리 생성, 사용자 LLM 실행) → STEP2(쿼리 품질 검증) → STEP3(3층 평가) → STEP4(해석: C 실성능 확정/B). 생성 전에 모집단 크기·평가코드 준비해 *맹목 생성 방지*.
> - **결과 위치**: (iii)가 (i) 대비 유의(SE≥2)하고 leakage 낮으면 → C가 "상한"에서 "circularity-robust 실성능"으로 확정 = 논문 헤드라인. (iii)≈(i)면 → X-target SE14는 circularity 덕이었고 실성능은 약함, 재검토. 정직하게 데이터로.


> **v0.4.18 (★첫 작동 증명 — C 성립[X-target SE_ratio 14] · B 전달 초기신호 · W-target 음수=방향특이성 통제)** — 재설계(v0.4.17: 타겟 X + intra-user hard-neg) 후 *인과 닫힌* 평가에서 처음으로 "방법 작동"이 데이터로 섬. 7회 디버깅 종료, 진짜 결과 단계 진입. 단 C=상한·B=초기신호로 정확히 위치 유지. **이 블록이 본문·이전 changelog의 상충 기술을 우선한다.**
>
> - **★평가 측정버그 3개 자가수정 (이번 결과 신뢰성의 근거)**: (i) stale 결과(옛 W-target ckpt 11:55 ≠ 새 17:58) 발견·재평가, (ii) recovery routing이 "user의 모든 pos_mid 집합"으로 판정해 *항상 correct=1.0* → "첫 쿼리의 specific mid"로 정정(`_load_user_first_pos_mid`), (iii) Headline1이 여전히 W 타깃 → `evaluate_x_target()` 신작(X를 seen 마스킹에서 제외). **셋 다 안 잡았으면 이번도 가짜. 측정 정직성이 결과 신뢰의 전제.**
> - **★C 성립 (Headline1, X-target 인과닫힌 평가)**: 2a R@10 delta +0.0048 **SE_ratio +14.57** PASS / 2b +0.0039 SE_ratio +12.98 PASS. SE_ratio 14 = 노이즈 확률 사실상 0(이전 7회의 ±노이즈 SE 1.3과 질적 차이). **"prefix가 X(쿼리 의도 아이템) 방향으로 추천을 이동"는 인과 가설 검증.** 2a>2b 전 지표(delta·SE·improve count). ★위치 정직: 이는 *메커니즘 작동 증명(상한)*이지 실용 추천성능 아님 — X-target은 circularity 내재(쿼리가 X 묘사→X 맞춤). circularity-robust 진짜 헤드라인 = recovery@N + (미생성)타겟-리뷰 정합쿼리. reviewer "X로 X맞추면 당연" 질문에 "상한 + recovery가 robust 헤드라인"으로 방어(v0.4.8).
> - **B 전달 초기신호 (Headline2, recovery@N)**: 2a routing 0.9531 > 2b 0.9337(+1.9%p, 전체 9171명 견고), correct-wrong R@10 gap 2a +0.0288 > 2b +0.0193. Stage1(intra-user)→routing↑→추천 gap↑ 전달경로 보임. ★단 **wrong group 430/608명으로 작아 추천gap 유의성은 미확정 — "초기신호"지 "확정" 아님**(routing 우위는 견고, 추천 전달의 통계적 유의성은 더 다져야). 4명/10000 노이즈 교훈 유지.
> - **W-target 음수 = 방향특이성 통제군(강점)**: X로 steering 시 W ranking *나빠짐*(2a SE -2.13). steering이 *아무 방향이나* 올리는 게 아니라 *X 방향 특이적*임을 증명. X↑ ∧ W↓ → "direction-specific, not generic boost" — 논문 강력 통제.
> - **현재 contribution 위치**: C(query-activated steering 시스템) = 메커니즘 작동 증명됨(상한). B(Stage1 intra-user alignment) = 전달 초기신호(routing 견고, 추천 유의성 미확정). A(recovery@N) = 대표분석. 다음: (1)circularity-robust 실성능 = 타겟-리뷰 정합쿼리 생성(미뤄둔 작업)으로 절대 Recall 상한 + recovery 헤드라인, (2)B 유의성 = wrong sample 키우거나 통계 보강, (3)Stage1 미포화(delta 단조증가)라 epoch↑ 여지.


> **v0.4.17 (설계 재정합 — L_align 타겟 W→X[선택지 A] · intra-user hard-neg · 학습/추론 층위 구분)** — 구현 버그(v0.4.15-16) 제거 후 드러난 *설계 정합성* 2대 문제를 진단으로 확정하고 재설계. 방법의 정체성을 명확히: 학습=A(쿼리-의도-아이템 인과 닫기), 추론="다음 추천 steering". **이 블록이 본문·이전 changelog의 상충 기술을 우선한다.**
>
> - **★문제1 확정 — L_align 타겟이 구조적으로 틀림 (W ∈ memory = 0.0%)**: 현재 L_align은 prefix를 LOO 타겟(미래 item W) 방향으로 미는데, 진단 결과 **W ∈ any memory = 0/52,239 = 0.0%, 100% 유저 미커버**. W는 정의상 미래 아이템이라 메모리(train-history에서만 생성)에 *구조적으로* 못 들어감. → L_align이 쿼리·메모리 어디에도 없는 방향을 가르침 = *틀린 신호*. B(LOO 유지) 옹호 근거 0.
> - **★선택지 A 채택 — L_align 타겟 = 쿼리 출처 item X**: prefix를 *쿼리가 생성된 그 리뷰의 item X* 방향으로. "query_about_X → cluster_containing_X → X"로 인과 사슬을 닫음. **A는 *학습 방식*이고 "다음 추천 steering"은 *추론 목적* — 다른 층위라 충돌 없음**: 학습 시 쿼리 의도를 아는 상태로 의도→메모리→X steering을 *가르치고*, 추론 시 실시간 쿼리→의도→다음추천 steering으로 *발휘*. W로 학습하면 추론 때 steering이 쿼리를 안 따르므로, A로 학습해야 추론 목적이 성립(사용자 지적). circularity는 recovery@N(상대 gap)로 robust 관리(v0.4.8 유지).
> - **★문제2 확정 — Stage1이 진짜 과제를 못 배움 (K≥2 routing delta ≈ 0)**: Stage1 vs frozen-bge routing: K=1 trivial 1.0(과대평가 주범), **K≥2 delta -0.0003(개선 0), K≥5 -0.018(오히려 나쁨)**. 즉 Stage1 대조학습이 "많은 메모리 중 구별"이라는 진짜 과제에 기여 0 → **현재 contribution B(Stage1 alignment)가 데이터상 죽어있음.** 원인: hard-neg 91.6% cross-user(쉬움, 취향 자체가 달라 자명) / intra-user 8.4%(어려움, 실제 필요 능력) → encoder가 "같은 유저 내 다른 클러스터 구별"을 배울 기회 없음.
> - **★hard-neg 재설계 — intra-user 1순위**: K≥2 유저는 *같은 유저 다른 클러스터*를 primary hard-neg로, cross-user는 보조(in-batch)로만. 그래야 Stage1이 K≥2 구별(routing의 진짜 과제)을 학습 → B 부활 가능성. routing accuracy 보고는 **K=1 제외, K≥2 intra-user 기준**으로 정정(0.958은 K=1 포함 과대평가).
> - **재설계 순서**: align_pairs 재생성(L_align 타겟 X + intra-user hard-neg) → Stage1 재학습 → Stage2 재학습 → 평가. ★메모리 뱅크·splits·F6a SASRec은 *불변*(재생성 대상은 align_pairs와 학습뿐). 
> - **평가 프로토콜 연쇄 변경**: A 채택으로 평가도 정합 — (i) recovery@N(correct vs wrong, train 쿼리, circularity-robust 헤드라인), (ii) ★학습 정합 평가: 쿼리 출처 item X를 타겟으로 한 steered vs vanilla(인과 닫힘, A의 직접 검증), (iii) LOO 미래 item 평가는 *부차적*(W=0% 미정렬이라 약한 하한). 헤드라인=recovery@N + X-타겟 정합 평가.
> - **contribution 재정렬(데이터 후 확정)**: 문제2 때문에 B(Stage1 alignment)는 *intra-user 재학습으로 K≥2 delta>0 입증해야* 살아남음. 못 살리면 C(query-activated 시스템) 단독. A(recovery@N)는 대표분석.
> - **메타**: 7회 구현버그 제거 후 *처음으로* 순수 설계 문제 직면. 이번은 "버그 수정"이 아니라 "방법 정체성 확정"(A=학습 인과 닫기). W=0.0%·K≥2 delta≈0이 추측 아닌 *측정*으로 방향을 정함.


> **v0.4.16 (전 과정 정적 감사 — E-1 거짓 기록 정정 · E-2 미문서화 L_contrastive · B-1 leakage 검사 무효 · 평가 정체=B 미정렬)** — 7영역(전처리→생성→학습→평가) 정적 감사로 *지금까지 모든 Stage2 결과가 무효였던* 구조적 원인 확정. 개별 버그 추적을 멈추고 SPEC↔코드 전면 대조. **이 블록이 본문·이전 changelog의 상충 기술을 우선한다.**
>
> - **★E-1 (CRITICAL) — v0.4.15 "버그1 fix"가 거짓 기록**: decisions.md/plan.md에 "lr_enc 2e-6→1e-5 상향(fixed)"이라 적었으나 `train_hybrid.py:1235`의 argparse default가 **여전히 2e-6**. `--lr_encoder` CLI 미지정 시 *또 2e-6으로 frozen 학습*. → 직전 재학습들이 실제로 어떤 lr_enc로 돌았는지 *실행 로그에서 확인 필요*. default를 1e-5로 코드 수정 + decisions의 "fixed" 기록은 *코드 미반영이었음*을 명시(기록과 코드의 괴리 = 이번 감사의 핵심 교훈: "문서에 fixed"가 "코드에 fixed"를 보장 안 함).
> - **★E-2 (MAJOR) — SPEC 외 L_contrastive가 손실에 포함**: `train_hybrid.py:312-320` 실제 손실 = `L_retrieval + α·(L_align + L_contrastive)`, L_contrastive=`relu(0.1 - sim_correct + sim_wrong)`(margin hinge). **plan/decisions 어디에도 없는 항.** 직전 "contrastive 추가하니 2a +2.59→-2.08 퇴보"의 그 항 — 즉 *SPEC에 없는 손실로 학습한 모델*을 평가해 옴. **결정**: 핵심 손실은 SPEC대로 `L_retrieval + α·L_align` 유지. L_contrastive는 *선택적 ablation 항*으로만(plan에 명문화하거나 제거). 기본 학습에서 제외하고, 넣을 거면 별도 가중치·문서화 후 ablation으로 비교.
> - **★B-1 (MAJOR) — leakage 검사가 항상 skip**: `bank.py:193-197`이 splits.json 스키마를 `{uid:{...}}`로 오독(실제 `{"users":{...},"meta":{...}}`) → `splits.keys()`={"users","meta"} → `train_timestamps_by_user`가 빈 dict → **evidence.timestamps train-history 검사가 통째 우회**(splits_path 전달 시). 즉 "evidence leakage 0 검증"이 *검사를 안 한 것*. test/val 타깃이 메모리에 새어들어가도 미감지. user-set validation도 모든 유저를 "extra"로 오표기. → 스키마 정정(`splits["users"].items()`) 후 **leakage 재검증 필수**(메모리 뱅크가 실제로 clean한지 지금 모름).
> - **평가 정체 확정 = B (쿼리-타겟 미정렬)**: `full_ranking.py`/`eval_steered.py` 코드로 확인 — 타겟=splits test held-out *미래* 아이템(LOO), 쿼리=align_pairs(train-history 리뷰). 메모리 evidence는 train-history만이라 **test 타겟은 어떤 메모리 cluster에도 구조적으로 못 들어감**. → 쿼리(과거 의도)와 타겟(미래 구매) 사이 선험적 정렬 없음. **Recall delta≈0은 "메커니즘 사망"이 아니라 "쿼리-타겟 미정렬이 신호를 지움"**. 단 — eval 쿼리를 *타겟 리뷰*에서 생성하면 정렬되지만 그건 *circularity 누설*(C4 방식; ACL 수용되나 우리는 절대성능=상한으로 고지, 헤드라인=recovery@N 상대gap, v0.4.8). **평가 프로토콜은 세 버그 수정 후 확정**(다른 경로 버그 위에서 평가 고치면 무의미).
> - **MINOR**: A-5(preprocess `is_eligible`에 is_discriminative 누락 — synth가 담당하면 무해, sequences 소비자가 오용 시 우회), F-5(`sasrec.py predict()`가 prefix 하드코딩 None — 현 eval은 log2feats 직접 호출이라 무관하나 predict() 사용 시 silent). 불명(미확인): align_pairs 생성 로직, is_discriminative 판정 위치, bank embedding 생성 encoder.
> - **확인된 MATCH(코드 입증)**: InfoNCE(losses.py), LOO split, 5-core, on-the-fly 재인코딩(E-3, train_hybrid.py:287-289 — *Stage2 encoder는 캐시 아닌 재인코딩*, gradient 흐름 정상), SASRec frozen, prefix injection 위치(스코어링에서 prefix 제외), train↔eval prefix 경로 동일, checkpoint 키, Bug2(prefix_fn)·Bug3(non-pad norm) fix. → **방법의 *구조*는 SPEC과 일치. 어긋난 건 (E-1)lr default·(E-2)손실 항·(B-1)leakage 검사·평가 프로토콜.**
> - **메타 교훈**: 7회째 "방법 실패처럼 보인 것"이 또 구현/측정 결함. 특히 **E-1은 "문서에 fixed라 적은 것"과 "코드가 실제 fixed인 것"의 괴리** — 앞으로 fix는 *코드 줄 + 실행 로그*로만 확정, 문서 기록을 신뢰하지 않음. **지금까지 모든 Stage2 결과는 (E-1)frozen-가능 +(E-2)SPEC외 손실 +(B-1)leakage 미검사 상태 산출 → 전부 무효. 세 버그 수정 후 *처음으로* 공정한 시험.**


> **v0.4.15 (Stage1/2 실행 — 세 측정버그 발견·수정: encoder frozen · 평가 prefix 미주입 · prefix 스케일 invisible)** — Stage2 학습 후 "성능 안 남"의 근본 원인이 *방법*이 아니라 *세 측정/스케일 버그*였음을 단계적으로 규명. contribution 판정은 올바른 스케일 재학습 후로 보류.
>
> - **경과 요약**: Stage1(`stage1_align_best.pt`, train routing 0.958, correct-wrong sim gap +47%) 정상. Stage2 2회(2a=stage1-enc / 2b=raw-bge) 학습, val_L_ret 동률(0.2804/0.2774). 이 "동률"을 처음엔 "B 약함"으로 읽었으나 — 세 버그가 드러나며 *그 판정 자체가 무효*였음이 확인됨.
> - **버그1 — Stage2 encoder 사실상 frozen**: lr_enc=2e-6은 projector lr_proj=1e-3의 1/500. encoder norm diff = 0.07%(판정 <0.1% → frozen). 10 epoch 동안 projector만 500배 빠르게 학습 = "반쪽 학습". `routing_cov`(0.918 flat)는 routing 품질이 아니라 **bank 커버리지 상수**(9705 bank유저/10566 전체)였음 — 학습과 무관, flat이 당연. → **lr_enc 1e-5로 상향**(LLaVA "저-LR 보존"은 frozen이 아니라 projector보다 낮을 뿐).
> - **버그2 — 평가에 prefix 미주입 (가장 치명적)**: `full_ranking.py evaluate_full()`이 `log2feats(seq_np)` = `prefix_embeds=None`. → 우리가 "Steered Recall"이라 측정한 모든 값이 *실은 vanilla*. "steered=vanilla=0.042 → C 미성립" 결론은 **두 vanilla를 비교한 측정버그**. "controllability J(c,v)≈0.03(prefix가 top-10 바꿈) vs Recall 무변화"의 *모순*도 허구 — J는 학습루프 내 prefix 주입 경로, Recall은 prefix 없는 평가 경로 = **서로 다른 코드 경로**였음. plan §3 M5가 "conditions.py로 평가 통일"하라 설계한 이유가 정확히 이것. → `eval_steered.py`로 prefix 주입 평가 추가, 학습·평가 단일 log2feats 경로 통일.
> - **버그3 — prefix 스케일 invisible**: `sasrec.py` prefix rescale의 `item_target_norm`을 `seqs.norm(dim=-1).mean()`으로 계산 → **padding position(norm=0) 포함** 평균. Books 패딩 비율 85.1%(non-pad/유저 mean 7.5) → OLD norm 0.2573 vs NEW(non-pad) 1.7236, **6.7× 차이**. OLD에서 prefix가 0.2573으로 rescale되어 실제 item norm 1.7236 대비 **attention share ≈0.44%(invisible)**. non-padding 평균으로 정정 → **16.7%**(의도 ~9% 범위, causal-mask 구조상 마지막 위치 기준). **결정적 함의: 저장된 모든 ckpt(Stage1·Stage2)가 0.44% invisible prefix로 학습됨 → projector가 유용한 방향을 학습할 수 없었던 근본 원인.** (이전에 "스케일 폭주 97%"로 본 것은 학습루프 *순간*의 LayerNorm 효과였고, 저장 ckpt의 실효 학습은 invisible 0.44%였음 — 두 측정이 다른 지점.)
> - **올바른 평가 후 첫 측정 (invisible-학습 ckpt 기준)**: steered≈vanilla(R@10 0.048), per-user rank delta n=10000에서 mean +1.97(2a)/+2.65(2b)이나 **std 163/125, SE≈1.6 → 유의하지 않음**(improve 4333 vs degrade 4180 ≈ 상쇄). = "강하게 흔들되 방향 랜덤" — 0.44% 학습의 예상 결과. **2b(raw-bge)≥2a(stage1)**이나 둘 다 버그 하 학습이라 B 판정 보류(Stage1도 invisible prefix로 학습되어 우위가 사라졌을 수 있음).
> - **표준 측정 규율 추가**: steered 평가는 Recall delta 평균뿐 아니라 **improve/degrade/same + rank delta std/SE**를 항상 보고(평균만으론 "상쇄"≠"무효과" 구분 불가 — 이번에 평균 0이 사실은 강한 양방향 상쇄였음). prefix가 들어가는 모든 측정은 단일 함수 경유(버그2 재발 방지).
> - **재학습 변수 확정**: 올바른 스케일(non-pad norm 1.72) + lr_enc=1e-5 + α=0.1(plan §7#9 "낮은 값에서 시작" — 중간에 0.3 권고가 나왔으나 plan 기조로 정정) + **cold projector**(기존 ckpt 방향이 랜덤이라 warm-start 이점 없음). 재학습 후 `eval_steered.py`(prefix 주입+분포)로 steered vs vanilla(C) + improve/degrade 쏠림 + 2a vs 2b(B) **첫 유효 판정**.
> - **메타 교훈**: Pilot4(prefix 너무 약함=묻힘) → 반쪽학습(encoder frozen) → 평가버그(prefix 미주입) → 스케일 invisible(0.44%) — 네 번 모두 "방법 실패"가 아니라 *측정/설정 결함*이었음. "성능 안 남"을 방법 탓하기 전에 측정 경로부터 검증하는 규율이 매번 옳았음.


> **v0.4.14 (Pilot4 NO-GO 재해석 — 메커니즘 결함 아님·미학습 projector가 원인 · LLaVA 비대칭 튜닝 · controllability는 Stage2 *후*)** — Pilot4 mini NO-GO를 PPR·LLaVA·plan 자신의 §7#2 경고에 비춰 재해석. prefix 메커니즘(P=1)은 결함이 아니며, modality gap을 메우는 projector를 *제대로 학습하지 않은 것*이 원인. "메커니즘 변경(cross-attn/final-feature)" 제안 철회. **이 블록이 본문·이전 changelog의 상충 기술을 우선한다.**
>
> - **Pilot4 NO-GO 원인 = 미학습 projector (메커니즘 결함 아님)**: Pilot4 mini(J_correct-vs-vanilla=0.866, controllability 16.7%)는 P=1 prefix가 묻힌 게 아니라, **bge 텍스트 공간(차원 A)→SASRec item 공간(차원 B)을 잇는 projector(MLP)가 few-epoch라 *modality gap을 못 메운* 상태**였기 때문. 학습 안 된 projector의 h_intent는 SASRec 공간에서 *의미 없는 좌표*라 무시됨(당연). 3대 근거: ① **PPR(TKDE 2024)** §IV-C — prompt length=1로 frozen SASRec(PPR-light) prompt-tuning이 fine-tuning *능가*("longer prompts no improvement"). **P=1은 죄 없음 — 단 *학습된* P=1.** ② **LLaVA 인사이트(PDF p.6)** — 다른 공간 연결은 중간 projector(MLP)를 *강하게 학습*해야 modality gap 해소(비대칭 튜닝). ③ **plan 자신 §7#2·Pilot4 한계노트** — "미학습 projector로 steering 판정 금지, 불가 시 F7/F8 이연". → **Pilot4를 *학습 전*에 게이트로 둔 것이 설계 오류. 미학습 측정은 실패가 예정됨.**
>
> - **"메커니즘 변경" 철회 (cross-attn/final-feature 보류)**: 직전 제안한 prefix→cross-attention/final-feature 교체는 *증상(안 먹힘)*을 보고 *원인(미학습)*을 건너뛴 성급한 처방. PPR이 P=1 prefix로 성공했으므로 메커니즘은 검증됨. **prefix 주입(P=1) 유지. cross-attn은 "전체 학습 후에도 controllability 분리가 안 될 때"의 최후 수단으로만 보류.**
>
> - **Stage2 = LLaVA 비대칭 튜닝 명시 (projector 강하게)**: PDF p.6 인사이트2 = frozen backbone 보존 + **MLP(projector)만 *강한* LR로 학습**해 catastrophic forgetting 없이 modality bridge. Stage2 LR 위계 명확화: **SASRec frozen(기본) / text_encoder *저*-LR(Stage1 보존 미세조정) / projector *강*-LR(modality gap 해소 주력)**. ("기본 LR"이 아니라 인코더보다 *확실히 높은* LR — LLaVA 차용의 핵심.)
>
> - **controllability 측정 = Stage2 *후*로 이동 (게이트 재배치)**: Pilot4(미학습)를 게이트로 쓰지 않는다. 순서: **전체 Stage1(인코더 정렬) → 전체 Stage2(LLaVA 비대칭, projector 강학습) → 그 후 controllability 재측정**(학습된 prefix가 correct vs wrong 분리하나). Stage2 학습 *중* controllability 모니터링; **전체 학습 후에도 분리 안 되면 *그때* 메커니즘(cross-attn) 의심.** 잔여 리스크: PPR prefix는 안정적 user-profile에서, 우리는 *가변 query*에서 — controllability가 PPR엔 없던 *추가* 요구라 "PPR 되니 우리도 무조건"은 아니고 "학습하면 될 가능성 높다"까지(증명은 Stage2 후 측정).
>
> **v0.4.13 (측정1 통과 — routing floor 0.90은 *동일인코더 일관성* · Stage1=train↔real gap 축소 · 학습=2-stage · ~~Pilot4 게이트~~ → v0.4.14에서 controllability를 Stage2 후로 이동)** — P2 train 생성 후 provenance routing floor 실측·분해 완료. **이 블록이 본문·이전 changelog의 상충 기술을 우선한다.**
>
> - **측정1 통과 — 단 "동일 인코더 내부 일관성"으로 정확히 해석 (frozen으로 충분 ≠)**: 학습 *전* frozen bge-base만으로 K≥2 routing floor **0.8986~0.9537**. ★중요: 쿼리도 memory bank 벡터도 *같은 frozen bge*로 만들어졌고 *train pseudo-query 분포 안*에서 잰 값 — 즉 **동일 인코더 내부의 일관성 측정이지 학습 후 일반화 능력이 아니다.** 3-way 분해로 "부풀림" 기각: (1) leak 쿼리(23%)가 *오히려 낮음*(0.921 vs 0.964) — 식별정보는 노이즈, (2) Jaccard 최하위(단어공유≈0)에서도 **0.9324** — lexical 복사 아닌 *의미 정렬*, (3) 길이 무관. single-review K=1은 메모리 1개라 trivially 1.0 → 보고 *제외*. **이 floor의 진짜 의미 = (i) P2 생성 품질이 깨끗(Q1=0.93이면 LLM이 리뷰 복사 아닌 intent 추출; 나빴으면 Q1이 먼저 붕괴) + (ii) Stage1이 좋은 초기화에서 출발(gradient가 처음부터 맞는 방향). "학습 불필요"가 아님.**
>
> - **Stage1의 역할 = train↔real query 분포 gap을 좁힌다 (v0.4.13 자체정정)**: 이전 초안의 "routing 헤드룸 작다(0.93→0.95)" 프레임은 *틀렸다* — 그 0.93은 *train 분포 내 동일인코더 일관성*이라, Stage1이 좁히는 축(train pseudo-query ↔ inference 시 real user query gap)을 *재지 않았다*. frozen bge는 그 gap에 대해 한 일이 없다. Stage1은 여전히 필요하며, 그 가치 = ① **train↔real 분포 gap 축소**(일반화) + ② **wrong-memory hard-neg margin**(recovery@N correct-wrong gap 확대 = contribution B). **증명 축 정정: train 분포 내 routing(0.93→0.95)이 아니라 *분포 밖* query(held-out 손작성 50~100개 또는 eval 타깃 쿼리)에서 frozen-bge vs Stage1 비교 — 거기서 Stage1이 robust하면 gap 축소가 데이터로 증명됨.** train 내 0.90으로는 증명 불가.
>
> - **학습 = 2-stage 명확화 (4 step 아님)**: 원 아이디어의 "text encoder 학습 / contrastive / mlp / steering"은 *2 stage*로 묶임. **Stage 1(`train_align.py`) = text encoder를 contrastive(InfoNCE)로 학습**(= "인코더 학습"과 "contrastive"는 동일 행위, 별개 step 아님). 입력 `align_pairs.jsonl`(provenance positive + wrong-memory hard-neg), SASRec/projector 미사용. **Stage 2(`train_hybrid.py`) = projector(MLP) 학습 + SASRec steering 동시**(frozen F6a + Stage1 인코더 저-LR + projector 기본LR, L=L_retrieval+α·L_align). LLM-judge 제거(v0.4.11)는 Stage1의 *라벨 출처*만 provenance로 바꿈.
>
> - **다음 게이트 = Pilot 4 (steering controllability, Stage1 전체학습 *전*)**: floor 0.93 + 외부평가 경고("Pilot4 미루지 말 것")에 따라, 전체 Stage1 투자 전에 **메커니즘 de-risk**: P=1 prefix가 frozen SASRec 출력을 실제로 perturb하나. correct-intent vs wrong-intent prefix → top-k 유의 차이(controllability) + correct prefix Recall > concat baseline(방향성). K≥2 소수(N≈20~50) mini Stage1+Stage2. **빨간불(prefix가 20-item 히스토리에 씻겨나감/over-smoothing)이면 학습이 아니라 prefix 주입 메커니즘 재검토.** 통과 시에만 전체 Stage1→Stage2.
>
> **v0.4.12 (P2 프롬프트 확정 — C4 원문 채택 · null 분기 제거 · title/author 비-leakage)** — 16-cell ablation + C4 실증 검토로 P2 쿼리 프롬프트 확정. **이 블록이 본문·이전 changelog의 상충 기술을 우선한다.**
>
> - **P2 프롬프트 = C4 원문 + strict JSON (입력=리뷰만)**: `config/prompts/amazon/Books/p2_pseudo_query.txt` = Amazon-C4 원 프롬프트("rephrase in first-person... the name of the product must not show... ignore irrelevant info") + `Output strict JSON {"query": str}`. 입력은 `{review_text}`뿐(title 등 메타 *미입력*). train·eval 공통 1파일(2모드: train=history 리뷰+provenance positive, eval=타깃 리뷰+pair 없음).
>
> - **null 분기 제거 (검증 완료)**: ablation 16-cell이 전부 null 분기를 달아 OOOO에서 null 42.2% 발생 — *리뷰 내용 문제가 아니라 프롬프트가 빠져나갈 문을 준 것*. 메모리에 들어간 리뷰(is_discriminative 통과)만 쿼리화하므로 *이미 의도가 있는 리뷰*다. null 조건 제거(C4J variant) → n=45 null=0%, K≥2 routing Hits@1=1.000 확인. **null 분기 제거 확정.**
>
> - **title/author는 leakage로 보지 않음 (C4 준수 + 실증)**: C4 실증 검토 — C4는 movies에서도 *제품명(title)만* 금지하고 *감독·배우·장르는 미제한*하고 ACL accept(query→item retrieval, 50-item 동도메인 풀). 우리 세팅은 *더 안전*: routing 타깃 = 메모리 `source_text`(P1 intent)이고 **저자명은 source_text에 없어 cosine sim에 무영향**(author guard 없는 C4J도 routing Hits@1=1.000 실측). SASRec scoring도 item-ID 시퀀스 기반이라 쿼리 텍스트가 직접 닿지 않음(쿼리는 메모리→prefix 경유, 메모리에 저자 없음). → **title/author 비-leakage 결정. G3 게이트 hard→info(보고만, 배제 안 함).** (이전 v0.4.7~v0.4.11의 "저자 숨김/distinctiveness 디텍터" 강조는 P2 쿼리에 대해선 *완화* — P1 메모리 source_text의 leakage 디텍터[논문 leakage rate 지표]와는 별개로 유지.)
>
> - **F8 선제 분석 한 줄 (reviewer 보험, 강제 아님)**: author/title이 routing엔 무영향이나, end-to-end NDCG에서 "author-mention 쿼리 vs not"의 차이를 F8에서 *측정·보고*해 "저자 노출이 부당한 uplift를 주지 않음"을 선제 입증. 차이 무시 가능이면 author 허용이 데이터로 정당화(차이 크면 그때 가드). *추측으로 막지 말고 측정으로 결정* — 프로젝트 게이트 철학 일치.
>
> **v0.4.11 (P3 LLM-judge 제거 — provenance가 정답셋 대체 · P2 단일 프롬프트 2-모드)** — provenance 복원의 두 번째 배당금. **이 블록이 본문·이전 changelog의 상충 기술을 우선한다.**
>
> - **P3 LLM-judge 제거 (핵심 파이프라인)**: 기존 P3(`p3_align_judge.txt`, self-consistency T=3)는 (query, memory)를 LLM이 positive/negative 판정 — *provenance가 없던 시절의 우회책*. 이제: train 쿼리 q는 리뷰 r에서 생성되고 r.item_id는 정확히 하나의 메모리 evidence.item_ids에 속함 → **positive = provenance 조회(결정론, judge 불요)**. hard negative = r의 클러스터가 아닌 모든 메모리(same-user 다른 클러스터/타유저 — 클러스터링이 τ로 이미 분리, *구성*이지 판정 아님). recovery@N의 wrong memory도 provenance 기반. → **P3 제거**. 효과: (i) LLM 노이즈 곱 P1×P3×P2 → **P1×P2**(외부평가 약점#4 완화), (ii) F5(judge, 쿼리×메모리쌍×T=3) LLM 비용 0, (iii) 라벨이 결정론·재현 가능. **P3는 선택적 ablation으로만 잔존**("provenance 라벨 ≥ LLM-judge 라벨" robustness — 핵심 아님). 본문 §5 P3 judge 절·F5 stage·prompts/p3_align_judge는 *ablation 표기*로 강등.
>
> - **align-pair 구성 (provenance 결정론)**: `align_pairs.jsonl` = {query, positive_memory_id(provenance 조회), hard_negatives(same-user 타클러스터 전부 + 타유저 샘플)}. Stage1(`train_align.py`)은 이 결정론 라벨로 supervised contrastive(InfoNCE) — judge 산출 라벨 미사용. (single-review K=1 클러스터는 q↔positive가 *같은 소스 리뷰 공유*라 학습신호가 쉬움 — leakage 아님[둘 다 train-history], train<eval 분포갭으로 F8에서 정직히 드러남. 사전폐기 금지, single-review K=1 기여 별도 측정.)
>
> - **P2 단일 프롬프트 2-모드 (C4 스타일 + 저자 숨김)**: 프롬프트 텍스트는 train·eval *공통 1개* `p2_pseudo_query.txt`(C4 원 프롬프트 차용 + "제품명 *및 저자명* 출력 금지" — Books는 저자로 타깃 특정 가능하므로 author 추가). 생성 스크립트는 2-모드: **train-mode**(입력=history 리뷰, 출력=쿼리+provenance positive 라벨, pair 생성) / **eval-mode**(입력=val/test 타깃 리뷰, 출력=쿼리만, **pair 없음** — 정답 메모리 주입은 leakage). C4는 *프롬프트 스타일 참조*일 뿐 데이터셋·GT 아님(v0.4.10). 기존 `p2_pseudo_query_train.txt`/`_eval.txt` 2파일 → 1파일 2모드로 통합.
>
> **v0.4.10 (routing GT 정정 — C4 폐기, provenance self-GT 채택 · P2 train-first)** — v0.4.8이 routing accuracy의 ground truth를 Amazon-C4 정합으로 잡은 것을 *정정*. **이 블록이 본문·이전 changelog의 상충 기술을 우선한다.**
>
> - **C4를 routing ground truth로 쓰지 않는다 (v0.4.8 정정)**: C4 README 확인 — C4는 (i) **query→*item*(parent_asin) 검색**이지 query→*memory* 라우팅이 아니고, (ii) C4 유저는 *전체 Amazon 테스트셋*에서 추출되어 우리 **5-core Books splits 유저와 거의 안 겹침**(메모리 뱅크 조인 ≈ 0), (iii) C4 쿼리는 *5점 리뷰 1건*에서 ChatGPT가 만든 single-review 쿼리라 우리 *multi-review 클러스터* 중 "정답"이 정의 불가. → C4는 routing GT 부적합. (C4의 진짜 가치 = "5점 리뷰→1인칭 복합쿼리" *프롬프트 스타일* 참조 + 선택적 OOD robustness 질적 비교. 주 routing 지표 아님. BLAIR 코드를 Books로 직접 돌릴 필요 없음 — P2가 BLAIR 스타일 이미 구현.)
>
> - **routing accuracy GT = provenance self-consistent (진짜 대안)**: 이미 결정론적 GT 보유 — provenance lock 덕분. **P2 train 쿼리** q는 *history 리뷰 r*에서 생성되고, r의 `item_id`는 이미 **어느 `cluster_label`에 속하는지 provenance로 확정**(재추출의 전체 목적). 따라서 **routing accuracy = (router가 q로 top-1 선택한 memory == r의 provenance cluster) 비율** — *LLM 판단 아님, 데이터 구조 내재 결정론 정답*. P3 LLM-judge도 C4도 아닌 self-consistent GT. `stratify.py`는 이 정의로 측정. (P3 라벨은 Stage1/2 *학습 신호* 전용, 평가 지표 아님 — v0.4.8의 이 부분 유지.)
>
> - **P2 train-first (eval 지연)**: routing 측정1의 GT가 *train* 쿼리(provenance 확정)에서 나오므로, **train 쿼리만 먼저 생성하면 측정1 가능**. eval 쿼리(target-review, BLAIR식, 학습 격리)는 *학습 비용 2배인데 아직 안 씀* → P3/Stage 직전 생성. **순서: P2 train 생성 → 측정1(provenance routing acc)+측정2(wrong-mem-neg ablation gap) → contribution 게이트 → eval 생성 → P3 → Stage.** 프롬프트 2종(`p2_pseudo_query_train.txt`/`_eval.txt`)은 지금 둘 다 *작성*, *실행*은 train 먼저.
>
> **v0.4.9 (prototype linkage 수정 — ward로 mega-cluster 해소)** — (B) prototype 구축에서 average-linkage가 단일 mega-cluster(99.9%)로 붕괴한 문제를 수정. **이 블록이 본문·이전 changelog의 상충 기술을 우선한다.**
>
> - **prototype linkage average→complete (mega-cluster 해소)**: 첫 prototype 빌드에서 전체 15,880 personal-unit의 99.9%가 단일 클러스터(p0)로 병합 → K=0 fallback이 prototype 1개로 집중. **근본 원인**: bge-base 임베딩 anisotropy(무작위 쌍 평균 코사인 **0.759±0.092**)로 기본 유사도가 높은데 average-linkage가 *클러스터 간 평균 거리*를 쓰면 τ=0.35(sim0.65)에서 전부 chaining 병합. *personal 클러스터링(유저당 2~20 포인트)은 chaining 안 생겨 정상*이었고, prototype(cross-user 15,880 포인트)에서만 터짐. **수정·실제 적용**: `build_prototypes.py`는 **complete linkage + TAU_FALLBACKS=[0.20,0.15,0.10](sim 0.80/0.85/0.90)**로 구현. sweep 실측 — ward·complete 모든 조합에서 mega-cluster 소멸. **채택 = complete sim0.80**(첫 τ=0.20에서 통과). 실측: **P=15, 최대 prototype 17.3%(2746/15880), mega 재발 0**, cluster_size [2746,2359,…,192] 큰·작은 혼합(K=0 fallback 적절). ward sim0.75도 동등 — *재론 안 함*.
>
> - **이 선택은 헤드라인 무관 (descope 판단 기록)**: prototype은 **K=0 유저(861명, 8.1%) fallback 전용** — 헤드라인(K≥2 recovery)은 personal 메모리만 쓰므로 prototype 분포는 *무관*. K=0는 애초에 personal intent 신호가 없는 군이라, prototype taxonomy의 미세 품질차(ward vs complete)는 F8에 영향 0. 따라서 prototype linkage·τ 미세조정은 *rabbit-hole*이며 balanced+결정론이면 충분(완벽 추구 금지). "population prototype taxonomy 해석가능성"은 *minor 주장*이지 contribution 아님.
>
> - **personal 뱅크 불변 확인 + finalize md5**: 이번 수정은 `_prototypes.json` + K=0 유저 파일만 변경. personal 뱅크(`f3_bank.jsonl`, md5 8cfb1177…, 15,880 유닛 evidence.item_ids 빈 [] 0건)는 바이트 불변. **finalize 산출**: `_prototypes.json` md5 377b7322…(P=15), finalized bank md5 90a6ab1b…(K≥1 padding 0, K=0 861명 prototype 1개씩 no_centroid_fallback=0, 2회 빌드 동일 = 결정론). **메모리 구축(M2) 전체 종결.**
>
> **v0.4.8 (provenance 종결 · 실측 K 분포 · routing 지표 수정 · circularity 선제고지 · contribution 게이트)** — 재추출→클러스터링→synth 완료 후 provenance 사태 종결 확인 + 외부 평가에서 도출된 *진짜* 방법론 결정 반영. **이 블록이 본문·이전 changelog의 상충 기술을 우선한다.**
>
> - **provenance 사태 종결 (검증 완료)**: synth 산출 15,880 personal 유닛 전수 검증 — `evidence.item_ids` 빈 `[]` **0건**, evidence 멤버 수 == cluster_size(불일치 0), item_ids↔timestamps 길이 일치, val/test 누설 0(splits 전수 대조). v0.4.6 재추출(provenance lock)의 1차 목적 달성 확정. 정본 = `data/memory/Books/f3_bank.jsonl`(15,880 유닛 / 9,705 유저, s0+s1 disjoint 머지, md5 8cfb1177…, manifest 저장). per-user `memory_bank/{user_id}.json` 분해는 (B)prototype 이후 bank finalize 시점에 수행.
>
> - **실측 K_personal 분포 (외부평가 약점#3 *기각*)**: 전체 10,566 유저 기준 K=0 861(8.1%) / K=1 5,277(49.9%) / **K≥2 4,428(41.9%)**. K≥1 통계 mean=1.64·median=1·p90=3·max=5. 외부 평가가 "multi-intent 헤드라인이 *작은 eligible의 또 일부(상위10%)*에만 적용"이라 우려했으나 — **실측 K≥2가 41.9%·절대 4,428명으로 헤드라인 모집단 충분**(그 우려는 eligible 분포 p90=12 추정인데, 실제 클러스터링은 eligible 적은 유저도 K≥2 형성). **Beauty 보조 조기편입 불필요**. K=1 중 single-review(cluster_size=1) 기원 18.1%(955명) = v0.4.7 eligible_min=1 신규군 — 사전폐기 없이 F8 K=1 층화에서 기여 측정.
>
> - **routing accuracy 지표 수정 (P3 라벨 함정 제거 — 방법론 필수)**: 기존 `stratify.py`가 routing accuracy를 *P3(LLM-judge) 라벨과 router top-1 일치율*로 정의. **이는 라우팅 정확도가 아니라 *두 LLM의 일치도*** — P3가 ground truth가 아니므로 논문에 "routing accuracy"로 보고하면 위험. **수정**: routing accuracy는 **Amazon-C4 정합**(query→실제 item→그 item이 속한 유저 메모리)을 ground truth로 측정. P3 라벨은 *Stage1/2 학습 신호*로만, *평가 지표로는 쓰지 않음*. C4 정합 없는 슬라이스는 routing accuracy 미보고(recovery@N으로 인과효과 측정).
>
> - **circularity 선제 고지 (review-derived = real-query 성능의 *상한*)**: P2 쿼리(train=history 리뷰, eval=target 리뷰 환언)는 모두 *답에서 역생성*되어 실제 underspecified 쿼리보다 타깃과 과정렬 → F8 절대 성능은 **실제 질의 성능의 상한**으로 *선제 명시 보고*. **단 헤드라인(recovery@N: correct vs wrong)은 circularity-robust** — *같은 쿼리*에 *다른 메모리*를 넣는 within-query 대조라 쿼리 분포 낙관성과 무관하게 "활성화의 인과효과"를 분리(절대성능 과대평가 ≠ 인과주장 붕괴). 완화책: held-out 유저에 손으로 쓴 underspecified 쿼리 50~100개 *독립* 셋 확보(가능 시). (leakage 게이트=식별정보 문제, circularity=분포 문제 — 별개.)
>
> - **discovery vs description 정직 분리 (서술 규율)**: intent **discovery**는 unsupervised clustering(τ_personal), intent **description**만 LLM. 논문에서 "LLM이 의도를 귀납 발견"으로 흐리지 말 것(안 그러면 "리뷰 임베딩 K-means + GPT 캡션"으로 깎임). *해석가능성*(description)은 살되 *discovery*는 clustering에 정직히 귀속.
>
> - **contribution 포지셔닝 = 측정 후 확정**: 방향은 **C(query-activated interpretable 통합 시스템) 1차 + B(wrong-memory hard-neg alignment) 종속 + A(correct/wrong recovery) 대표분석**. 단 비싼 F-stage(F5 대량 P3·F7 학습) *전에* 저비용 2측정으로 방어선 확정: ① real-ish 쿼리(C4 소량) top-1 라우팅 correct 선택률(↓이면 A·C 붕괴) ② wrong-memory-neg ablation gap 소슬라이스(↓이면 B 붕괴). **메커니즘 novelty(frozen-backbone prefix-steering)는 PPR 등으로 crowded — 주장 금지, 방어 novelty=combinatorial.** (외부평가의 IGR-SR/MemGuide/AMA 인용은 실물 확인 후에만 반영.)
>
> **v0.4.7 (재추출 실행 확정 — eligible_min=1 · distinctiveness leakage 디텍터 · 4-GPU 추출 · 캐시 경로 정정)** — v0.4.6 재추출 직전 smoke 검증·3대 확인에서 확정된 수정. **이 블록이 본문·이전 changelog의 상충 기술을 우선한다.**
>
> - **eligible_min 2→1 (k_min=1 정합 — 층화 모집단 정상화)**: 메모리 selection의 `eligible_min`을 2→**1**로 낮춘다. 기존 `=2`는 eligible 리뷰 1개 유저(Books 883명)를 *추출 user selection 단계에서* 배제해, plan §7 #1(v0.4.3)의 **k_min=1**(eligible 1개여도 K_personal=1 personal memory 구성) 취지를 박탈하고 이들을 완전 cold-start([DEFAULT_INTENT])로 강등시켰다 — K_personal=1 층화 모집단이 인위적으로 축소되어 *헤드라인(K별 lift)* 분석이 왜곡됨. `=1`로 낮추면 eligible 1개 유저도 P1 추출 → is_discriminative면 K_personal=1, 아니면 자연히 K=0 → prototype/default(정상). `[DEFAULT_INTENT]`는 *진짜* 신호 없는(eligible 0) 유저에게만 남는다. **백본/평가 무영향 재확인**: 883명은 splits.json(5-core+LOO, `eligible_min` 미참조)에 그대로 남아 SASRec 학습·평가 대상 — 메모리 경로만 바뀜. 비용 +883콜(무시). F3에서 *single-review로 만든 K=1 비율*을 보고(품질은 F7/F8이 측정, 사전 폐기 금지).
>
> - **leakage 디텍터 = distinctiveness 기반으로 교체 (논문 leakage rate 지표 정상화)**: 기존 `check_leakage()`는 *제목 unigram(≥4자) substring* 매칭이라 장르/공통어("mystery","piano","dark romance")에 **22% 과발화**하고, 다중토큰 실제 제목 누설은 누락한다. `leakage_detected`는 *게이트가 아니라 마커*라 파이프라인엔 무해하나(데이터 미배제), **논문에 보고되는 leakage rate 지표를 오염**시킨다(리뷰어 reject 리스크) — 22%는 garbage 수치이고, 그 레코드를 버리면 정상 intent 신호를 대량 폐기하는 자해다. **교체 규칙(길이 아닌 *고유성*)**: ① 제목 토큰에서 stopword + **장르/카테고리 공통어 사전** + **corpus document-frequency 상위**(전체 books 제목 DF 높은 토큰=공통어) 제거 → 남은 *distinctive n-gram(연속)* 또는 **저자명(meta author 정확 매칭)** 이 source_text/query에 연속 매칭될 때만 leakage=true. ② 단일 장르어 1개 겹침은 누설 아님. 저자명은 1단어여도 누설. *근거*: "dark romance"(연속 장르어)도 FP 안 나고, "The Silent Patient"(고유 trigram)·"Rowling"(저자)은 잡힌다. **추출 본체 불변·사후 재계산** — source_text·title·author로 leakage_detected만 재계산하므로 *재추출 불필요·추출과 병렬*. P2 eval 가드(§6.6)를 **게이트로 승격할 때 동일 디텍터 재사용**(BLAIR식 제품명 숨김 검증).
>
> - **4-GPU disjoint 추출 (품질 불변·속도 2×)**: 워커는 LLM-only(임베딩 없음)라 병렬 레버 = Ollama 인스턴스 수. gemma4:26b Q4_K_M(~17.9GB)는 2×2080Ti(22GB)에 적재 → **2 인스턴스(GPU0,1@11434 / GPU2,3@11435)** 로 처리량 2×. `parallel_memory_build.py`에 `--api_url` + `--user_shard_idx N_OF_M`(유저 disjoint 분할: `sorted(user_ids)` 안정정렬 후 `idx%M==N`, 한 유저=한 샤드) 추가. **품질 동일**: 동일 모델·temperature=0.0·prompt_version=p1_v2 → 같은 입력=같은 출력(temp=0 결정론). 머지는 *두 output dir 모두* 스캔→단일 정본 `p1_extractions.jsonl`(유저 disjoint assert). sqlite 캐시 동시 write 락은 WAL 모드 또는 샤드별 캐시 후 머지로 회피.
>
> - **prompt_version 명확화**: `p1_v2`는 *캐시 키 분리 라벨*일 뿐, **프롬프트 텍스트는 v0.4.5 lock본(`p1_books_b`)과 바이트 동일**. 옛 실패 빌드의 캐시(50,358건)에 stale-hit하는 것을 막아 옛/새 추출 혼입을 차단하는 용도. smoke `cache_hit=0`으로 분리 확인됨.
>
> - **캐시 경로 정정**: 보존 대상 메인 LLM 캐시의 실경로는 **`data/llm_cache/llm_cache.sqlite`**(68M, 운용 캐시)다. 이전 문서가 적은 `data/cache/llm_cache.sqlite`(24K, 구 미사용)는 오기 — 절대 보존 목록·본문 §1 폴더 레이아웃을 실경로로 정정.
>
> - **정본 완결성 재확인(불변)**: `p1_extractions.jsonl`은 is_discriminative=false 레코드도 *플래그 달고 전량 저장*(워커에 분기 없음), eligibility 필터는 downstream `build_f3_bank.py`에서 처음 적용 — 정본이 완전한 추출 아티팩트라 disc 기준 변경 시 재추출 불요(smoke 63/63 all-true는 sampling coincidence).
>
> **v0.4.6 (provenance lock · 단일 추출 아티팩트 · 결정론적 선택 · P2 temporal split)** — full 추출 후 발견한 *evidence.item_ids 공란* + *캐시 복원 GATE FAIL*을 영구 차단하고, P2를 시간순 train/eval로 명문화한다. **이 블록이 본문·이전 changelog의 상충 기술을 우선한다.**
>
> - **provenance lock (item_id 전 파이프라인 보존 — 재추출의 1차 목적)**: evidence.item_ids가 항상 `[]`였던 근본 원인 = (i) cluster_summaries에 `source_texts`만 저장하고 `item_id`를 버림, (ii) `build_f3_bank.py`가 `make_intent_memory_unit`에 evidence_map 미전달. **수정**: P1 추출 산출물에 `item_id`/`timestamp`/`rating`을 **1급 필드**로 포함하고, 클러스터링 시 멤버십(`item_id ↔ cluster_label`)을 **저장**한다 → `evidence.item_ids`/`evidence.timestamps`가 *본질적으로* 채워지고, 사후 복원(캐시 재조립·코사인 Path B)이 불필요해진다. 부수 효과: **P2 train query의 `memory_id`도 코사인 근사 없이 source-review→cluster *조회*로 확정**(§6.5, §4 F4).
>
> - **단일 추출 아티팩트 (chunk 분산 저장 폐기)**: 기존 `parallel_memory_build.py → chunk_XXXX.jsonl → 머지 → memory_b_uNNN.jsonl → per-user JSON 재분할` 경로에서 provenance가 유실되었다. **수정**: 병렬 워커는 shard를 쓰되, 즉시 **단일 정본 `data/processed/{category}/p1_extractions.jsonl` 1개로 머지**(+`p1_extractions.manifest.json`에 행수·유저수·md5). 다운스트림(필터/클러스터/synth/bank)은 **이 정본 파일만** 읽는다. chunk는 워커 로그로만 두고 정본화 금지(다중 source-of-truth 제거).
>
> - **결정론적 선택 (max_reviews_per_user 재정의 — "첫 12개" 폐기)**: GATE FAIL의 직접 원인 = `max_reviews_per_user=12`가 *파일 순서상 첫 12개*를 골랐는데 `books_memory_candidates.jsonl`이 재생성/재정렬되어 어느 12개인지 바뀜(source_text 매칭 63.6%, 유저 49.6%만 100% 복원). **수정 3종**: ① **candidates.jsonl을 frozen 아티팩트로**(1회 생성·md5 고정·재생성 금지, 빌드 입력으로 hash 검증). ② cap은 *raw 리뷰*가 아니라 **pre-P1 필터(rating≥4 & ≥10단어 & train-history) 통과분**에만 적용(cap 예산을 비-eligible에 낭비 방지). ③ 선택 기준은 파일순이 아니라 **most-recent N**(`timestamp` 내림차순, 동률은 `item_id` 안정 tiebreak) — 파일 순서와 무관하게 결정론. **선택된 item_id 리스트를 유저별로 저장**(`selected_review_ids`). cap 값(현 12)은 config 노출 — multi-intent 유저를 자르는지(=cap 적용 시 K_personal 감소) F3에서 점검해 상향 검토. *근거*: 최근 리뷰가 현재 intent를 더 잘 반영하고, timestamp+item_id 안정정렬은 입력 파일 순서에 불변.
>
> - **재추출 결정**: 캐시 복원(STEP 1A′) GATE FAIL → 부분 복원(49.6% 유저) 짜깁기는 align-pair 라벨 일관성을 깨므로 금지. 코드를 위 3종으로 고친 뒤 **전체 재추출**(4×2080Ti 병렬). **캐시 클리어로 옛/새 추출 혼입 방지**(50,358 히트 + 11,226 신생성 섞임 금지 — 전부 동일 prompt_version으로 새로). 재추출 후 **K_personal(0/1/≥2) 분포·clustering 재현성 재검증** 필수(기존 검증 자산은 새 뱅크로 무효화되므로 재확립).
>
> - **P2 temporal split (train/eval 분리 — 시간순)**: 리뷰원문 기반 쿼리 생성(C4/BLAIR 계보)을 유지하되 **시간으로 분리**한다. **train query** = train-history 리뷰(입장1, 누설 0) → provenance로 source-review→cluster *조회*해 positive 라벨 확정(코사인 Path B 불필요). **eval query** = val/test 타깃 아이템 리뷰(입장2, C4식 \"real query\") → **학습에 절대 미혼입·eval 전용 격리**. 둘 다 입력에 item 메타(category·속성)를 *맥락*으로 포함하되 **title·brand·저자·고유명사는 출력 쿼리에서 금지**(누설+자명성 방지: 생성 시 reject+retry≤2 + 사후 디텍터). 프롬프트 2종 분리: `p2_pseudo_query_train.txt`(history 리뷰 → 1인칭 쿼리), `p2_pseudo_query_eval.txt`(타깃 리뷰 → 1인칭 쿼리, BLAIR식 제품명 숨김). (§6.5, §4 F4, §3 M1.)
>
> - **5-core / rating≥4 재확인 (불변)**: 백본(SASRec) = 전체 상호작용 **5-core**(표준, 리뷰·평점 무관)와 메모리 eligibility = **rating≥4 + ≥10단어 + is_discriminative**는 *분리된 knob*임을 재확인(§6.2). cap·선택 정책은 **메모리 pre-P1 필터 통과분에만** 적용되고 백본 5-core·splits.json에는 영향 없음. step-by-step 구축(per-review P1 → filter → embed → cluster → synth → bank → prototypes)은 건전하므로 단계 구조 유지 — 이번 수정은 *provenance 보존·선택 결정론·단일 아티팩트*에 한정.
>
> **v0.4.5 (Books P1 검증 완료 — 정의·프롬프트·코드·throughput 확정)**:
> - **정의/프롬프트 lock**: 18건 라벨드 미니셋에서 과보정 케이스(pacing/emotion/series) **10/10 복원**, 경계어("interesting") TRUE 정상, pure-approval/metadata false 정상(조정 17/18). is_discriminative 정의(§v0.4.4)와 `p1_books_b`(variant B, compact 5필드) **확정**.
> - **코드 수정**: `parse_response()`가 is_discriminative=false & contextual_intent="" 정상응답을 schema_error로 폐기하던 버그 수정.
> - **throughput 현실**: concurrency는 GPU-saturated라 무의미(c=1 264 vs c=8 310 calls/hr; c=16은 timeout으로 열화). 12b도 26b보다 느림(기검증) → **모델 크기·concurrency 둘 다 throughput 레버 아님**. 병목 = Ollama+26B 단일 인스턴스 GPU 포화.
> - **추출 규모**: cap=12 → Σ min(eligible_u,12) = **48,243 calls**. c=1(264/hr) 기준 **~7.6일**. c=8 별도 스크립트는 +15%뿐이라 비권장.
> - **실행 권고**: full 추출을 백그라운드로 띄우고 *그 사이 F3(cluster/synth/prototypes/bank)를 빌드*(추출 비차단). 규모 vs 시간 옵션: (a) 전체 9,807 ~7.6일, (b) ~5k eligible 서브샘플 ~3일, (c) Beauty/ablation 대비 **vLLM tensor-parallel + guided decoding/continuous batching**으로 전환(실질 가속, 인프라 투자).
>
> **v0.4.4 (intent 정의 명문화 — is_discriminative 기준 교정)**: Books P1 A/B 분기에서 드러난 *정의 충돌*을 해소(b.txt가 페이싱·감정을 false로 깎던 문제). **이 블록이 본문·이전 changelog의 상충 기술을 우선한다.**
> - **Intent의 정의 (QAIM)**: intent = *query로 활성화 가능한 사용자의 독서 모드/선호 축*. 판정 기준은 "이 리뷰가 *질의로 부를 수 있고 아이템 군집과 상관되는 선호 차원*을 드러내는가"이다 — **"이 독자를 같은 장르의 다른 독자와 구별하는가"가 아니다.** QAIM의 대비 축은 *유저 내 intent 간*(예: 같은 유저의 "page-turner 스릴러" vs "느린 문예소설")이지 *유저 간*이 아니다.
> - **is_discriminative = true**: 페이싱(page-turner/완급), 감정 톤(몰입/위안/감동), 인물 깊이, reader level, 문체, 구조, 상황 맥락, 시리즈 연속성 등 *명명 가능한 선호 차원*이 하나라도 있을 때. **페이싱·감정도 valid intent** — generic하다는 이유로 버리지 않는다.
> - **is_discriminative = false**: 리뷰가 *대비 차원을 명명하지 못하고* 순수 valence/승인만 담을 때 — 명백 false는 **"loved it","great book","amazing","highly recommend"** 같은 *가치판단어*(무엇을 원했는지가 0), 또는 신호가 metadata에서만 올 때(`grounding_level=metadata_dominant`). **판정 테스트**: "이 표현이 *책들이 갈리는 축*(완급·감정·몰입·인물·레벨·문체·구조·상황맥락·시리즈)을 가리키는가?" → 그렇다면 true. 'engrossing/immersive/interesting' 같은 강도어는 *하드 false 아님* — 몰입(immersion) 축을 함의하면 true, 단순 호감이면 false로 *문맥 판정*. **근거**: 순수 승인의 유일한 신호("이 아이템/장르를 좋아함")는 *이미 백본 SASRec co-occurrence가 포착* — 메모리는 백본이 못 잡는 *맥락 선호 축*을 위한 것이라 승인 재진술은 redundant. false 케이스의 `contextual_intent`는 빈 문자열 `""`로 둔다(코드 §parse_response와 일치).
> - **steering 효과(선호 축이 얼마나 tight해서 아이템 군집을 좁히는가)는 필터가 아니라 F7/F8 실험·stratification이 측정**한다. generic intent를 *사전에 폐기하지 않는다*(유저 내 대비 + 풍부한 `preference_summary`가 유용성 보존). 비율(0.93/0.97 등)을 타겟하지 않는다 — *정의*가 비율을 정한다.
> - **프롬프트**: B의 `head+tail` truncation·`metadata-disambiguation-only` 정책은 유지. b.txt의 페이싱/감정 BAD 예시 3개는 *제거*하고 *positive(=true) 예시로 전환*. pure-approval BAD + metadata-driven BAD만 유지.
>
> **v0.4.3 (메모리 구축 재설계 — compact-first / K 정직성 / synth 경량화)**: 메모리의 소비자는 routing 임베딩 + steering MLP뿐이라는 원칙에 따라 단순화. **이 블록이 아래 본문의 상충 기술을 우선한다.**
> - **K 정직성 (k_min 2→1)**: agglomerative **k_min=1**(억지 K≥2 금지). K_personal을 **0(메모리 없음)/1(single-intent)/≥2(multi-intent)** 로 정직히 보고. 본문의 모든 `k_min=2`는 `k_min=1`로 대체. prototype은 부족분 "보충"이 아니라 §2.2(B)·M5의 *명시적 fallback*으로만(`is_prototype=true`) — personal인 척하지 않음. 진짜 cold(신호 없음)는 learnable `[DEFAULT_INTENT]`(v0.4.2).
> - **P1 schema compact-first (확정 아님 — pilot이 검증)**: 기본 추출은 5필드 `{contextual_intent, preference_summary, evidence_span[], is_discriminative, grounding_level}`로 *축소해 먼저 테스트*. 기존 `aspect_coverage`(4)/`preference_attrs`(5)/`disposition_note`/`persona`/confidence 등급은 **삭제가 아니라**, pilot에서 cluster coherence·query 변별성이 부족할 때 *근거를 갖고 복원*하는 ablation으로 강등(처음부터 12필드로 시작하지 않음). source_text엔 제목·작가 절대 미포함(Books leakage).
> - **synth 경량화 (deterministic-first)**: cluster→memory는 LLM 재호출 대신 **medoid의 contextual_intent + preference keyword 합집합 → source_text, embedding=centroid**가 기본. LLM synth(intent/persona)는 ablation으로만 — 추출 외 LLM 호출 최소화.
> - **eval 층화 = K_personal**: results.json의 warm/cold(="eligible 리뷰 ≥1개")는 우리 contribution 층화가 아님 → 폐기하고 **K_personal(0/1/≥2)** 로 재정의. 핵심 분석 = K≥2에서 routing 성공 시 lift(§3 M5 `stratify.py`).
>
> **v0.4.2 (외부 리뷰 반영 — 스케일/cold-start/loss/split 강화)**: 4개 개선 수용.
> - **현실적 스케일**: LLM 전수 추출 대신 **유저 ~10K–50K 서브샘플**, 추출은 *eligible(~5%)에게만* → 호출 ~수천(전체 유저 추출 아님; "비용 폭발"은 전체 추출 가정에서 온 과장)(§6.2).
> - **cold-start prefix**: zero/global-average 금지 → **학습 가능한 `[DEFAULT_INTENT]` prefix 파라미터**(Stage 2 학습)로 신호 없는 유저·"no-intent" baseline 처리(§7 #1).
> - **loss 스케일**: α는 {0.01, 0.1, 0.5}처럼 낮게 시작 + 두 loss raw 크기/gradient norm 로깅·밸런싱(§7 #9).
> - **single global split**: 최초에 `splits.json` 1회 생성, M0/M1/M4/M5가 *바이트 동일*하게 소비 + consistency 테스트(§6.4).
>
> **v0.4.1 (Pilot 2-lite 실측 반영 — funnel/ablation)**: 실제 결과 파일을 직접 검증하고 반영.
> - **카테고리 결정**: funnel(count≥5+≥10단어+rating≥4) Books 4.7%(484k)>Beauty 1.9%(219k)≫Fashion 0.1%(2.4k). **Fashion은 coverage 구조적 부족으로 주전장 제외**(extraction은 정상; 필터·프롬프트로 해결 불가). **주전장 Books, 보조 Beauty** — K≥2 최종 확정은 Pilot 2 클러스터링(§6.1, §7 #6).
> - **메모리 eligibility 확정**: 단문(3~9단어) ablation에서 discriminative 10~15%·빈 ci 50~75% → `rating≥4` + `≥10단어`(프리필터) + `is_discriminative`(실필터). 감정만 있는 리뷰는 메모리 없음(아이템 신호는 백본/prototype으로) — 억지 아이템-재진술 금지(§6.2).
> - **Books 주의**: 카테고리에 비-도서 혼입(CD/공구) + aspect 풍부함이 sub-genre 의존(아동서·논픽션 강, 문학소설 약). full study 전 실제 도서 필터 + 구성 보고(§6.1).
>
> **v0.4 (구현 정합성 패치 — 실제 서빙/데이터/필터 결정 반영)**: v0.3.2까지의 설계를 유지하되, 구현·파일럿 진행 중 확정된 사항과 어긋난 서술을 정정한다.
> - **LLM 서빙**: 문서 전반의 `Qwen2.5-7B-Instruct(bnb-nf4)` 잔존 표기를 제거하고 **`gemma4:26b` via Ollama**(JSON-schema constrained decoding, `think:false`)로 통일(§3 M1, §7 #7, §8). Pilot엔 충분하나 전체 F2 처리량(병렬/소형모델/서브샘플)은 미결로 명시.
> - **데이터 수급**: HF 자동 다운로드 → **로컬 JSONL 전용**(없으면 즉시 실패)로 정정(§1, §3 M0, §4 P1).
> - **rating/필터 분리**: 백본(SASRec)은 전체 상호작용 5-core(표준), **메모리는 positive(rating≥4) 리뷰 기반**으로 분리. 메모리 eligibility의 길이/개수 임계(기존 ≥10단어·≥8)는 *고정값이 아니라 Pilot 2에서 실측*해 정함 — 문헌 표준(5-core / 텍스트연구 ≥3) 대비 ≥8은 과도함을 명시. LLM 추출은 ~100K–300K 상호작용 서브샘플(§6.1/§6.2).
> - **카테고리**: 실제 보유 = `Amazon_Fashion`/`Beauty_and_Personal_Care`/`Books`(기능형 미보유). per-user 다중의도 커버리지는 Books 최상(문헌의 multi-interest 표준 카테고리), Fashion 최하 — 주전장은 Pilot 2로 확정(§6.1, §7 #6).
> - **관련연구**: 최근접 경쟁 Persona4Rec와 잠재 multi-interest(ComiRec/MIND)를 §0 차별점에 명시.
> - **내부 정합**: `embedding.source_text`에 `persona.description` 포함을 §7 #3에도 반영(§2.1 v0.3.1과 일치).
>
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

---

## (구) LLM 프롬프트 초안 — 역사적 참고용 (운영 프롬프트는 config/prompts/ 참조)


모든 프롬프트는 strict-JSON 출력을 기본으로 한다. **아래는 초기 *초안*이며, 실제 운영 프롬프트는 `config/prompts/{dataset}/{domain}/*.txt`로 외부화·간결화(Persona4Rec 수준)되어 있고 `p1_aspect`는 v4로 안정화(parse 100%, ~16s/콜)됨.** 운영 스키마는 string 필드 maxLength/배열 maxItems 캡과 "intent는 *리뷰*에서 추출·메타데이터는 보조 맥락(카테고리 재진술 금지)" 가드를 포함한다.

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

### P3 — 질의-메모리 정렬 Judge (학습쌍 구축) — **(v0.4.11 핵심 파이프라인에서 제거; 선택적 ablation 전용)**

> **(v0.4.11)** provenance 복원으로 align-pair positive 라벨은 *결정론적 조회*로 얻으므로 이 LLM-judge는 핵심 학습에 불필요. 아래 프롬프트는 "provenance 라벨 vs LLM-judge 라벨" robustness ablation을 *원할 때만* 사용. 기본 파이프라인은 provenance 라벨(judge 호출 0).

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
