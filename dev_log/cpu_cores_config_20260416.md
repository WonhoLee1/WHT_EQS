# Implementation Plan - CPU Core Configuration (2026-04-16)

## 1. 개요
사용자가 JAX/XLA 연산에 사용할 CPU 코어(스레드) 수를 제어할 수 있도록 CLI 인수를 추가함. 하드코딩된 값을 제거하여 하드웨어 환경에 최적화된 연산이 가능하도록 개선.

## 2. 변경 사항
### 2.1. `main_shell_opt.py`
- `argparse` 섹션에 `--cores` 인수 추가 (기본값: 6).
- `XLA_FLAGS` 환경 변수 설정 시 `intra_op_parallelism_threads` 값을 `args.cores`로 동적 할당하도록 수정.

## 3. 사용 방법
터미널에서 다음과 같이 실행하여 코어 수를 조정할 수 있음:
```powershell
# 12개의 코어를 사용하여 실행
python main_shell_opt.py --run --cores 12
```

## 4. 기대 효과
- 멀티코어 환경에서 병렬 연산 효율 극대화.
- 사용 중인 PC 사양에 맞는 유연한 자원 할당 가능.
