🔥 전체 실험 구조 (한눈에)
Stage 1: In-domain (site별)
Stage 2: Joint training (multi-site)
Stage 3: Cross-site transfer (pairwise)
Stage 4: Leave-one-site-out (multi→single)
Stage 5: Label efficiency (few-shot)
Stage 6: Distribution-aware (네 방법)

👉 이 6단 구조면 논문 그대로 나옴

0️⃣ 데이터 전제 (지금 상태)
sites:
  - pohang
  - utah_2019
  - utah_2023

labels:
  - event
  - noise
  - unlabel

unit:
  - 2s segment

👉 아주 이상적인 세팅

1️⃣ Stage 1: In-domain baseline
목적
각 site 난이도 파악
baseline 성능 확보
pretrain 효과 확인
실험
stage1_pohang_only
stage1_utah2019_only
stage1_utah2023_only
구성
train / val / test (site 내부)
group_id 기준 split
비교
setting	설명
no pretrain	pure supervised
pretrain	unlabeled 활용
+ dynamic center	네 방법
2️⃣ Stage 2: Joint training
목적
multi-site data가 도움이 되는가?
domain diversity 효과 확인
실험
stage2_joint_all
stage2_joint_pohang_utah2019
stage2_joint_pohang_utah2023
stage2_joint_utah2019_utah2023
핵심 질문

👉 "데이터 많이 쓰면 항상 좋은가?"

→ 아니면 negative transfer 있는가?

3️⃣ Stage 3: Cross-site transfer (핵심)
목적

👉 논문 핵심

domain shift 정량화
asymmetric transfer 분석
실험 (6개)
pohang → utah2019
pohang → utah2023
utah2019 → pohang
utah2019 → utah2023
utah2023 → pohang
utah2023 → utah2019
설정
train: source site
test: target site
추가 실험
source only
source + pretrain
source + dynamic center
4️⃣ Stage 4: Leave-one-site-out
목적

👉 진짜 generalization 테스트

실험
train: utah2019 + utah2023 → test: pohang
train: pohang + utah2023 → test: utah2019
train: pohang + utah2019 → test: utah2023
의미

👉 “여러 데이터 봤는데도 안 되냐?”

→ distribution gap 증명

5️⃣ Stage 5: Label efficiency
목적

👉 너 연구 핵심 중 하나

실험
fractions:
  1%, 5%, 10%, 25%, 50%, 100%
적용 위치
우선순위 1
stage1 (site별)
우선순위 2
stage3 (cross-site)
비교
방법	의미
no pretrain	baseline
pretrain	representation 효과
dynamic center	distribution 대응
6️⃣ Stage 6: Distribution-aware (핵심 contribution)
네 방법
dynamic center
bridge detection
unlabeled 활용
넣는 위치

👉 아래에만 넣어도 충분히 강력함

stage3 (cross-site)
stage4 (leave-one-out)
stage5 (few-shot)
🔑 Unlabeled 전략 (중요)
기본 원칙

👉 test domain은 절대 pretrain에 넣지 않음

예시
pohang → utah2023
train: pohang (labeled)
test: utah2023

pretrain:
  ✔ pohang
  ✔ utah2019 (optional)
  ❌ utah2023
확장 실험 (논문용)
+ utah2023 unlabeled 포함

👉 transductive setting (추가 실험용)

📊 결과 표 구성 (논문용)
Table 1: In-domain
site	no pretrain	pretrain	ours
Table 2: Cross-site
source → target	baseline	pretrain	ours
Table 3: Joint vs Single
setting	F1
single	
joint	
Table 4: Label efficiency
fraction	baseline	pretrain	ours
📈 핵심 분석 포인트

이거 꼭 논문에 써야 하는 부분

1. Utah2019 ↔ Utah2023 vs Pohang

👉 같은 지역 vs 다른 지역

2. Joint training 효과

👉 데이터 많아도 항상 좋은가?

3. Asymmetric transfer
A → B ≠ B → A

👉 매우 중요

4. Few-shot 성능

👉 pretrain vs distribution-aware 비교

5. Unlabeled의 역할

👉 진짜 도움 되는지

🚀 실행 순서 (현실적인)
1단계
stage1 3개
2단계
stage2_joint_all
3단계
stage3 6개
4단계
label efficiency
5단계
distribution-aware