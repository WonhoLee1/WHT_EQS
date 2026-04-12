# ShellFEM 최적화 속도 개선 및 Adjoint Method 적용 계획

현재 JAX의 `cg` 솔버를 직접 미분(Forward-mode/Reverse-mode through iterations)하는 방식은 반복 횟수가 많아질수록 기하급수적으로 속도가 느려집니다. 이를 해결하기 위해 **Adjoint Method (수반 행렬법)**를 도입하여 최적화 속도를 획기적으로 개선합니다.

## User Review Required

> [!IMPORTANT]
> **Adjoint Method 도입 효과**
> - 기존: CG 반복 횟수만큼 미분 그래프 생성 (매우 느림)
> - 변경: 단 한 번의 추가 선형 해석(Adjoint solve)으로 그래디언트 산출 (매우 빠름)
> - 기대 효과: 스테틱 해석이 포함된 최적화 루틴 속도 10배 이상 향상 예상

## Proposed Changes

### 1. ShellFEM Solver 고도화 (`shell_solver.py`)
- [MODIFY] `solve_static_partitioned`를 `jax.custom_vjp`로 래핑.
- [NEW] `_solve_static_fwd`: 순방향 해석 및 체크포인트 저장.
- [NEW] `_solve_static_bwd`: 수반 방정식($K \lambda = \text{grad}_u L$)을 풀고 파라미터($t, pz, E$)에 대한 그래디언트 산출.

### 2. 최적화 루틴 재가동
- [RUN] `python main_shell_verification.py`
- 수렴 속도 모니터링 및 그래디언트 정확도(Initial Loss 변화량) 검증.

## Verification Plan

### Automated Tests
- 최적화 루틴 실행 시 Iteration당 소요 시간 비교 (목표: 10초 이내/Iter).
- 동일 조건에서 `jax.grad` 결과와 Adjoint 결과의 일치 여부 확인 (소규모 mesh 기준).

### Manual Verification
- `optimization_run_v9.log`에서 Loss 값이 매 이터레이션마다 빠르게 갱신되는지 확인.
