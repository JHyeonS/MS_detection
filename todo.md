# TODO.md

## 📌 Current Status

- Stage1 HPO completed (contrast vs reconst)
- Fixed-threshold evaluation pipeline 구축 완료
- Ranking script updated (`test_metrics_fixed_threshold.json` 기반)
- 결과 확인:
  - reconst > contrast (명확한 성능 차이 존재)
- 기존 ranking 오류 원인 해결 완료 (summary.json → fixed threshold 전환)

---

## 🔥 Immediate Tasks (우선순위 높음)

### 1. Stage1 추가 HPO (hpo3)
- [ ] num_layers: 5, 6
- [ ] latent_dim: 128, 256, 512
- [ ] bandpass: (3,50), (5,60), (5,80)
- [ ] method: contrast, reconst
- [ ] queue script로 전체 실행
- [ ] 결과 ranking (fixed-threshold 기준)

---

### 2. Top Architecture 선정
- [ ] hpo3 결과 기반 top-k 선정
- [ ] contrast vs reconst 비교 정리
- [ ] 최종 encoder 구조 결정

---

## 🚀 Next Stage

### 3. Stage2 HPO (finetune 단계)
- [ ] freeze_encoder: True / False
- [ ] learning rate tuning
- [ ] anomaly loss weight tuning
- [ ] top architecture 기준으로 진행
- [ ] pretrain checkpoint 연결 확인

---

### 4. Evaluation 구조 변경 (중요)
현재:
- test에서 threshold sweep

목표:
- [ ] val에서 threshold sweep
- [ ] test에서는 fixed threshold 적용

구조:
train → val(threshold tuning) → test(fixed)

---

## 🧪 Optional but Important (논문 강화)

### 5. Representation / Pretrain 영향 검증
- [ ] freeze encoder 실험
- [ ] low LR finetune
- [ ] contrast vs reconst 차이 분석
- [ ] representation quality 확인

---

## 📊 Analysis & Automation

### 6. Ranking / 결과 분석 개선
- [ ] contrast + reconst 통합 ranking
- [ ] seed 평균 / std 분석
- [ ] top-k 자동 summary

---

## ⚙️ Experiment Management

### 7. 체크 필수 사항
- [ ] pretrained encoder 제대로 load 확인
- [ ] run_root / experiment 충돌 방지
- [ ] 로그에서 weight load 확인
  - "loaded encoder weights from ..."

---

## 🎯 Final Goal

- Stage1 → Stage2 최적 구조 확보
- evaluation pipeline 정리 (val → test)
- contrast vs reconst 연구 결론 도출
- 논문용 실험 결과 확보

---

## 🧠 One-line Summary

> 지금은 **Stage1 보강 HPO → Stage2 HPO → evaluation 구조 정리** 이 3개가 핵심이다.
