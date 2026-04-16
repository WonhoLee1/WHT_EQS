# WHTOOLS Issue Tracker: Shell FEM Optimization

이 파일은 **WHTOOLS** Shell FEM 파이프라인의 개선 사항과 해결된 이슈들을 기록하여, 향후 세션에서 동일한 문제가 반복되지 않도록 관리합니다.

## 1. 해결된 이슈 (Resolved Issues)

| ID | 날짜 | 이슈 내용 | 해결 방안 |
| :--- | :--- | :--- | :--- |
| **ISSUE-001** | 2026-04-08 | JAX JIT 내 `len()` 호출 오류 | 모든 `len(tracer)` 호출을 `.shape[0]` 또는 정적 변수로 교체함. |
| **ISSUE-002** | 2026-04-08 | `loss_fn` 중복 연산 오버헤드 | `field_results` 캐싱 로직 도입으로 다중 케이스별 연산 횟수를 1/3로 단축함. |
| **ISSUE-003** | 2026-04-08 | 비드 패턴 경계부 형상 감쇄 | `reflect` 패딩을 적용하여 `convolve2d` 시 경계부 주저앉음 현상 해결. |
| **ISSUE-004** | 2026-04-08 | 응력 복원 정확도 부족 | 요소별 정확한 `t`, `E` 물성을 반영하여 Stress/Strain 계산식 정교화. |
| **ISSUE-005** | 2026-04-08 | GT 생성 루프 비효율성 | Eigen-solver 및 Plot 생성을 루프 밖으로 이동하여 초기화 속도 개선. |
| **ISSUE-006** | 2026-04-08 | SED 노드 맵핑 불일치 (`ValueError`) | 요소 단위 SED를 노드 평균으로 맵핑하는 로직을 추가하여 `griddata` 호환성 확보. |
| **ISSUE-007** | 2026-04-08 | 물성 브로드캐스팅 오류 (`ValueError`) | `isinstance` 대신 `jnp.ndim` 기반 배열 판별 로직을 적용하여 방송(Broadcasting) 오류 해결. |
| **ISSUE-008** | 2026-04-08 | 모멘트 계산 누락 (`KeyError: 'moments'`) | 리팩토링 중 누락된 모멘트 계산 및 노드 맵핑 로직을 복구하여 최종 리포트 생성 기능 정상화. |
| **ISSUE-009** | 2026-04-08 | 변수명 오타 및 차원 불일치 (`NameError`, `ValueError`) | `jnp.full` 및 `concatenate` 가독성 확보를 통한 수치 연산부 전수 점검 및 안정화 완료. |
| **ISSUE-010** | 2026-04-08 | 시각화 시 차원 불일치 (`ValueError`) | 전역(Global) 최적화 변수 시각화 시 단일 값을 노드 배열로 확장(`np.full`)하여 PyVista 호환성 해결. |
| **ISSUE-011** | 2026-04-08 | 고유진동수 최적화 정체 (3.2Hz) | `gt_init_scale` 영향(0.5) 및 `max_iterations`(200) 증설로 비드 형성을 강제 유도하여 강성 보강 효과를 확보함. |
| **ISSUE-012** | 2026-04-08 | JAX Tracer 제어 흐름 오류 (`TracerBoolConversionError`) | Solver의 `assemble` 및 `solve_eigen` 내 Python `if`문을 `jnp.where` 및 고정 인덱싱으로 리팩토링하여 JIT 호환성 확보. |
| **ISSUE-013** | 2026-04-08 | 굽힘 곡률 부호 불일치 (`Curvature Sign`) | `_B_bending_q4`와 `recover_curvature` 간의 부호를 일치시켜 물리적 정합성을 완성함. |
| **ISSUE-014** | 2026-04-09 | 최적화 모니터링 가시성 부족 | 루프 출력 형식을 2행으로 개편하고 타겟 값을 상단에 명시적으로 노출함. |
| **ISSUE-015** | 2026-04-09 | 초기 오차 부적합 (Freq mismatch) | 최적화 시작 전 GT와 초기 모델의 상태를 비교하는 진단 리포트 로직을 추가함. |
| **ISSUE-016** | 2026-04-09 | Tray 형상의 수치적 불안정성 | [결과] 안정화 성공 (ISSUE-017로 보완됨). |
| **ISSUE-017** | 2026-04-09 | 3D Tray (H=50mm) 메쉬 통합 | `generate_tray_mesh_quads` 복구 및 3D 응력 복원 로직 통합 보충 완료. |
| **ISSUE-018** | 2026-04-09 | JAX-Scipy `griddata` 데이터 타입 충돌 | `pts_h`/`pts_l`을 실제 노드 좌표로 맵핑하고 NumPy 캐스트를 강제하여 최적화 셋업 안정화. |
| **ISSUE-019** | 2026-04-09 | `effective_weights` 미정의 (`NameError`) | 존재하지 않는 변수 `effective_weights`를 `loss_weights`로 교체. |
| **ISSUE-020** | 2026-04-09 | `best_loss`/`best_params`/`wait` 미초기화 | 루프 진입 전 `best_loss=inf`, `best_params=params copy`, `wait=0`, `self.history=[]` 초기화 추가. |
| **ISSUE-021** | 2026-04-09 | `auto_scale` 파라미터 미정의 (`NameError`) | `optimize` 메서드 시그니처에 `auto_scale=True` 매개변수 추가. |
| **ISSUE-022** | 2026-04-09 | JIT 컴파일된 `loss_vg` 미사용 (매 반복 재컴파일) | 루프 내부에서 `jax.value_and_grad(loss_fn, ...)` 직접 호출을 사전 컴파일된 `loss_vg` 호출로 교체하여 성능 대폭 개선. |
| **ISSUE-023** | 2026-04-09 | `jit` 미임포트 (`NameError`) | `import jax`만 존재하는 상태에서 `jit(...)` 호출. `jax.jit(...)`로 수정. |
| **ISSUE-024** | 2026-04-09 | `verify()` 내 불일치 (`KeyError: max_surface_stress`) | 타겟 딕셔너리는 `max_stress`/`max_strain` 사용하나 verify에선 `max_surface_stress`/`max_surface_strain`으로 참조. 필드 이름 동일하게 수정. |
| **ISSUE-025** | 2026-04-09 | `solve_eigen_sparse` 미지원 인자 (`sigma`) | `verify()`에서 존재하지 않는 `sigma=100.0` 인자 전달. 제거하여 해결. |
| **ISSUE-026** | 2026-04-09 | 이중 주파수 변환 (`sqrt/2π` 중복 적용) | `solve_eigen_sparse`가 이미 Hz를 반환하는데 `generate_targets`와 `verify`에서 다시 `sqrt/2π` 변환 적용. 전체 파이프라인에서 중복 변환 제거하여 물리적 정합성 복구. |
| **ISSUE-027** | 2026-04-09 | `compute_moment` 미구현 (`AttributeError`) | `ShellFEM`에 존재하지 않는 `compute_moment` 메서드 호출. 반력 모멘트 성분(Mx, My)을 이용한 근사 모멘트 산출로 대체. |
| **ISSUE-028** | 2026-04-11 | COO Index Caching (연산 속도 최적화) | Sparse 행렬 조립 시 발생하는 대규모 인덱스 생성 및 결합 오버헤드를 제거하기 위해 __init__에서 전역 인덱스를 사전 계산 및 캐싱하여 JAX 실행 성능 극대화. |
| **ISSUE-029** | 2026-04-11 | 제작 제약 조건 표준화 (150mm) | 실제 가공성을 고려하여 최소 비드 폭(min_bead_width)을 150.0mm로 상향 조정하고 전 단계에 일관되게 적용. |
| **ISSUE-030** | 2026-04-11 | Stage 2 (Fine-tuning) 최적화 활성화 | Stage 1에서 잡힌 대략적인 형상을 바탕으로, 높은 MAC 가중치와 하이브리드 모드 추적 방식을 적용하여 고유진동수 및 모드 형상에 대한 정밀 정합을 수행. |
| **ISSUE-031** | 2026-04-16 | Visualization Porting | main_shell_verification.py의 고품질 시각화 및 검증 로직을 main_shell_opt.py로 이식 완료. |
| **ISSUE-032** | 2026-04-16 | Indentation & Logic Fixes | optimize_v2의 IndentationError 해결, missing 변수 복구, modal matching 차원 불일치(375 vs 2250) 해결 및 get_metrics 로컬 정의 추가. |
| **ISSUE-033** | 2026-04-16 | Tray Geometry Customization | `PlateFEM` 및 `EquivalentSheetModel`에 `wall_height` 인수를 추가하고 CLI 상에서 `--tray_h`로 조정 가능하도록 개편. |
| **ISSUE-034** | 2026-04-16 | JAX/XLA CPU Parallelism control | `--cores` 인수를 추가하여 `intra_op_parallelism_threads`를 동적으로 제어할 수 있도록 개선. |

## 2. 주의 사항 및 가이드 (Lessons Learned)

> [!IMPORTANT]
> **JAX JIT 제약 사항**: `ShellFEM` 클래스의 메서드들을 JIT 컴파일할 때 `len()` 대신 `.shape[0]`을 사용해야 하며, 특히 **Tracer가 포함된 `if` 문은 절대 금지**됩니다 (`jnp.where` 또는 고정 로직 사용 필요)

> [!TIP]
> **강성 보강 초기치**: 비드 형상이 복잡할 경우 `gt_init_scale`을 0.5 이상으로 설정하는 것이 초기 경사 정보 확보에 유리합니다.

> [!WARNING]
> **데이터 일관성**: 모든 연산은 요소별(Element-wise)로 수행되지만 `WHT_EQS_visualization.py`와의 호환성을 위해 최종 필드 결과는 반드시 노드 평균(Nodal average)으로 변환하여 반환해야 합니다.

> [!NOTE]
> **아키텍처 구조**: `PlateFEM`은 고수준 API 래퍼이며, 실제 수치 연산 및 JAX 최적화 코드는 `ShellFemSolver/shell_solver.py`의 `ShellFEM` 클래스에서 수행됩니다. 디버깅 및 알고리즘 수정 시 이 파일이 핵심입니다.

## 3. 향후 개선 백로그 (Future Backlog)

- [ ] Sparse Solver (CPU)와 JAX (GPU) 간의 데이터 전송 오버헤드 최소화 연구
- [ ] 대규모 메쉬 대응을 위한 `jax.experimental.sparse` 도입 검토
- [ ] 더 복잡한 비드 패턴 모델링을 위한 High-order shell element(Q8/Q9) 확장 가능성 타진
