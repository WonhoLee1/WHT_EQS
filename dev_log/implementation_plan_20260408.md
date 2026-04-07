# 📝 Implementation Plan: Shell FEM 성능 및 정밀도 고도화

본 계획은 이전 코드 리뷰에서 식별된 4가지 핵심 이슈(성능 중복, 루프 비효율, 두께 근사 오차, 경계 패딩 문제)를 해결하기 위한 구체적인 수정을 목표로 합니다.

## 1. 개요 (Goal)
- **최적화 속도 향상**: `loss_fn` 내 중복 연산을 제거하여 iteration당 계산 시간을 50% 이상 단축.
- **수치적 신뢰도 확보**: 국부 두께 변화를 고려한 응력 계산 로직 도입 및 형상 왜곡 방지.

## 2. 제안된 변경 사항 (Proposed Changes)

---

### [Component] ShellFemSolver (`shell_solver.py`)

#### [MODIFY] [shell_solver.py](file:///c:/Users/GOODMAN/code_sheet/ShellFemSolver/shell_solver.py)
- **응력 복원 로직 수정**: `compute_field_results`에서 `t_nodes.mean()` 대신 각 요소(T3, Q4)별 노드 평균 두께를 사용하도록 변경.
- **필드 연산 통합**: `compute_max_surface_stress`, `compute_max_surface_strain`, `compute_strain_energy_density`가 각각 `compute_field_results`를 호출하던 구조를 개선하여, 한 번의 연산 결과를 재사용하도록 유도.

---

### [Component] Optimization Engine (`main_shell_verification.py`)

#### [MODIFY] [main_shell_verification.py](file:///c:/Users/GOODMAN/code_sheet/main_shell_verification.py)
- **`generate_targets` 리팩토링**:
    - `for case in self.cases` 루프 내부에 존재하는 `Summary Plot` 생성 로직과 `eigen_sparse` 호출을 루프 외부(마지막)로 이동.
- **`loss_fn` 최적화**:
    - `p_actual`을 한 번 계산한 후, `compute_field_results`를 한 번만 호출하여 `stress`, `strain`, `sed` 값을 추출하도록 수정.
- **비드 스무딩 패딩 개선**:
    - `jax.scipy.signal.convolve2d` 적용 전, `jnp.pad`를 사용하여 `reflect` 모드 패딩을 선행 적용함으로써 경계부 감쇠 해결.

---

## 3. 검증 계획 (Verification Plan)

### Automated Tests
- `test_solvers.py`를 실행하여 개선된 응력 복원 로직이 기존 이론해와 일치하는지 확인.
- `main_shell_verification.py`를 실행하여 최적화 속도(Iteration per second)가 실제로 향상되었는지 측정.

### Manual Verification
- [STAGE 1~3] 시각화 창을 통해 판 가장자리의 비드 형상이 `reflect` 패딩 적용 후 왜곡 없이 생성되는지 육안 확인.
- 최종 검증 보고서의 Max Stress/Strain 값이 기존 버전 대비 합리적인 수준으로 변동되었는지 체크.

## 4. 사용자 검토 필요 (User Review Required)

> [!IMPORTANT]
> **응력/변형률 값의 변화**: 요소별 두께를 정확히 반영할 경우, 비드 부위의 응력이 기존(평균 두께 사용) 대비 높게 측정될 수 있습니다. 이는 정밀도가 향상된 결과이지만, 타겟 매칭 시 가중치 조정이 필요할 수 있습니다.

> [!WARNING]
> **Z-좌표 처리**: `assemble()` 내에서 Z-좌표를 `add`하는 방식이 트레이 메쉬의 기본 높이와 중첩되는 부분에 대해 명확한 주석과 가이드를 추가할 예정입니다.
