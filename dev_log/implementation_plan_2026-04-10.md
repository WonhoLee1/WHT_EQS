# [Implementation Plan] High-Performance JAX-Native Assembly & Solver 도입

현재 `ShellFEM` 솔버의 조립 로직은 `sparse=True` 옵션 사용 시 NumPy와 SciPy를 혼용하고 있어, JAX의 핵심 장점인 **JIT 컴파일** 및 **VJP(Vector-Jacobian Product)를 통한 자동 미분** 성능을 100% 활용하지 못하고 있습니다. 

이를 `jax-fem` 및 `JaxSSO`에서 영감을 얻은 **순수 JAX 기반 최적화 엔진**으로 업그레이드하고자 합니다.

## User Review Required

> [!IMPORTANT]
> - **Sparse Matrix 처리**: `scipy.sparse` 대신 `jax.experimental.sparse`를 사용하여 조립과 해석을 수행합니다. 이는 GPU 가속 및 미분 가능성을 극대화하지만, 특정 희소 행렬 연산에서 JAX 버전에 따른 제약이 있을 수 있습니다.
> - **조립 방식 변경**: 반복적인 `at[].add()` 대신 `jax.ops.segment_sum`을 사용하여 GPU에서의 병렬 처리 성능을 높일 예정입니다.

## Proposed Changes

### [ShellFemSolver/shell_solver.py](file:///c:/Users/GOODMAN/code_sheet/ShellFemSolver/shell_solver.py)

#### [MODIFY] `ShellFEM.__init__`
- `segment_sum`에 최적화된 형태로 DOF 인덱스를 사전 계산합니다.
- 조립 시 중복 연산을 피하기 위한 Indexing Map을 구성합니다.

#### [MODIFY] `ShellFEM.assemble`
- `sparse=True` 시에도 SciPy를 호출하지 않고 `jax.experimental.sparse.BCOO` 등을 활용하여 순수 JAX Array를 반환하도록 리팩토링합니다.
- `at[].add()` 대신 `segment_sum`을 적용하여 전역 강성 행렬 조립 속도를 향상시킵니다.

#### [MODIFY] `ShellFEM.solve_static_cg`
- JAX Sparse Matrix를 직접 입력받아 해석할 수 있도록 최적화합니다.
- `jax.lax.while_loop`를 유지하면서 대규모 구조물의 수렴 가속화를 위한 간단한 Preconditioner 옵션을 검토합니다.

## Open Questions

> [!QUESTION]
> 현재 시스템에서 `jax.experimental.sparse` 기능이 안정적으로 작동하는 JAX 버전인지 확인이 필요합니다. (최신 버전 사용 권장)
> 또한, Sparse Solver에서 직접적인 `spsolve` (LU 분해) 대신 `CG`(Conjugate Gradient) 방식을 계속 주력으로 사용할지 여부를 확인해 주십시오. (대규모 모델 최적화에는 CG가 유리합니다.)

## Verification Plan

### Automated Tests
- `python ShellFemSolver/main_test.py`를 실행하여 기존 결과와 일치하는지 확인 (특히 Sparse 모드에서의 정합성).
- 대규모 격자 모델에서 조립 및 해석 시간 측정 (리팩토링 전후 비교).

### Manual Verification
- `main_shell_verification.py`를 통한 전체 파이프라인 JIT 컴파일 가능 여부 확인.
