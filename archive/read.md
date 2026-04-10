# MS_Detection 프로젝트 구조 요약

## 목표
DAS 기반 microseismic detection을 위한 연구 코드 구조이다.

전체 흐름은 아래와 같다.

1. **Pretrain**
   - CAE 기반 self-supervised pretraining
   - latent embedding으로 hypersphere center `c` 계산
   - `best.pt` 저장

2. **Finetune**
   - pretrained encoder 로드
   - `loss_fcl + lambda_anomaly * loss_anomaly` 학습
   - Deep SAD 스타일 anomaly loss 사용

3. **Test**
   - finetuned checkpoint 로드
   - classifier output + anomaly score(`||z-c||^2`) 평가
   - predictions / metrics 저장

---

## 디렉토리 구조

```text
MS_Detection/
├─ configs/train/
│  ├─ base.yaml
│  ├─ pretrain.yaml
│  ├─ train.yaml
│  └─ test.yaml
│
├─ scripts/
│  ├─ pretrain.sh
│  ├─ train.sh
│  └─ test.sh
│
├─ src/
│  ├─ dataset/
│  │  └─ finetune_dataset.py
│  │
│  ├─ dataloader/
│  │  ├─ pretrain_dataloader.py
│  │  ├─ finetune_dataloader.py
│  │  └─ test_dataloader.py
│  │
│  ├─ models/
│  │  ├─ cnn_encoder.py
│  │  └─ pretrain_reconstruction.py
│  │
│  ├─ training/
│  │  ├─ trainer_pretrain.py
│  │  ├─ trainer_finetune.py
│  │  └─ trainer_test.py
│  │
│  └─ utils/
│     ├─ device.py
│     └─ config_io.py
│
├─ data/
│  └─ 0312/
│     └─ metadata/
│        └─ experiments/
│           └─ stage1_pohang_only/
│              ├─ pretrain.csv
│              ├─ train.csv
│              ├─ val.csv
│              └─ test.csv
│
└─ runs/
   └─ 0313/
      ├─ pretrain/
      │  └─ stage1_pohang_only/
      ├─ finetune/
      │  └─ stage1_pohang_only/
      └─ test/
         └─ stage1_pohang_only/
```

---

## Config 역할

### `base.yaml`
공통 설정을 관리한다.
- seed
- device
- run_root
- experiment name
- split_dir
- model 구조

### `pretrain.yaml`
pretraining 전용 설정
- pretrain mode
- epoch
- lr
- batch size
- pretrain용 csv (`pretrain.csv`)

### `train.yaml`
fine-tuning 전용 설정
- train / val csv
- batch size
- optimizer
- `lambda_anomaly`, `eta`
- freeze 여부 등

### `test.yaml`
test 전용 설정
- test csv
- batch size
- optional checkpoint 경로

---

## 데이터 split 방식

현재 메타데이터는 `split_dir` 아래에 모아두는 구조이다.

예시:

```text
data/0312/metadata/experiments/stage1_pohang_only/
├─ pretrain.csv
├─ train.csv
├─ val.csv
└─ test.csv
```

즉 config에서는 전체 절대경로를 반복해서 쓰기보다,
파일명만 쓰고 `split_dir`와 합쳐서 해석하는 구조를 사용한다.

예:
- `pretrain_csv: "pretrain.csv"`
- `train_csv: "train.csv"`
- `val_csv: "val.csv"`
- `test_csv: "test.csv"`

---

## Run 저장 구조

실험 결과는 `run_root` 아래에 stage별로 저장한다.

```text
runs/0313/
├─ pretrain/
├─ finetune/
└─ test/
```

각 stage는 다시 `experiment` 이름으로 분리된다.

예:
```text
runs/0313/pretrain/stage1_pohang_only/
runs/0313/finetune/stage1_pohang_only/
runs/0313/test/stage1_pohang_only/
```

---

## 주요 체크포인트

### Pretrain 결과
저장 위치:
```text
runs/0313/pretrain/stage1_pohang_only/best.pt
```

포함 정보:
- model state dict
- optimizer state dict
- best loss
- pretrain mode
- `center_c`

### Finetune 결과
저장 위치:
```text
runs/0313/finetune/stage1_pohang_only/best.pt
```

포함 정보:
- finetuned model state dict
- optimizer state dict
- best metric
- `center_c`

### Test 결과
저장 위치:
```text
runs/0313/test/stage1_pohang_only/
```

예상 파일:
- `predictions.csv`
- `metrics.json`
- `merged_config.yaml`
- `base_config.yaml`
- `stage_config.yaml`
- `run_metadata.json`

---

## Loss 구조

### Pretrain
- reconstruction loss

### Finetune
- `loss_fcl`
- `loss_anomaly`

최종:
```text
loss_total = loss_fcl + lambda_anomaly * loss_anomaly
```

### Anomaly loss
Deep SAD 스타일로 latent vector `z`와 center `c`의 거리 사용

```text
dist = ||z - c||^2
```

- normal: center에 가깝게
- anomaly: center에서 멀어지게

---

## 실행 순서

### 1. Pretrain
```bash
bash scripts/pretrain.sh
```

### 2. Finetune
```bash
bash scripts/train.sh
```

### 3. Test
```bash
bash scripts/test.sh
```

---

## 핵심 설계 원칙

1. **run_root 아래에 모든 실험 결과 저장**
2. **experiment 이름으로 stage별 폴더 분리**
3. **split_dir 아래에 pretrain/train/val/test csv 정리**
4. **device 설정은 yaml에서 관리**
5. **config snapshot을 각 run 폴더에 저장**
6. **pretrain ckpt의 center_c를 finetune / test에서 재사용**

---

## 현재 구조의 장점

- 실험 재현성 확보
- pretrain / finetune / test 경로 일관성
- metadata split 관리 용이
- DAS 연구용으로 확장 가능
- 추후 Utah / Pohang 통합 실험에도 유리

---

## 앞으로 추가 가능 항목

- ROC-AUC / PR-AUC 계산
- confusion matrix 저장
- anomaly score threshold tuning
- multi-experiment 자동 반복 실행
- slurm batch script 통합
