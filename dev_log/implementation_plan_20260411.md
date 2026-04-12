# Shell FEM Optimization Pipeline Enhancement Plan (v4)

본 계획서는 Shell FEM 최적화 파이프라인의 **연산 성능 고도화**와 **제작 제약 조건 반영**, 그리고 **Stage 2(Fine-tuning) 활성화**를 목표로 합니다.

## User Review Required

> [!IMPORTANT]
> **성능 최적화 (COO Index Caching)**: 매 반복마다 수행되던 Sparse 행렬 인덱스 계산 및 결합 과정을 `__init__` 단계에서 사전 계산(Caching)하도록 구조를 변경합니다. 이는 JIT 컴파일 효율을 높이고 루프 속도를 대폭 개선할 것입니다.
> 
> **제작 제약 조건 (150mm)**: 비드 패턴의 최소 폭(`min_bead_width`)을 기존 120mm에서 **150mm**로 상향 조정합니다. 이는 실제 프레스 금형 공정의 제작 가능성을 확보하기 위함입니다.
> 
> **Stage 2 활성화**: 1단계에서 잡힌 대략적인 형상을 바탕으로, 모드 형상(MAC) 매칭에 높은 가중치를 두어 정밀 최적화를 수행하는 2단계를 활성화합니다.

## Proposed Changes

### 1. [ShellFemSolver](file:///c:/Users/GOODMAN/code_sheet/ShellFemSolver) (Core Engine Optimization)

#### [MODIFY] [shell_solver.py](file:///c:/Users/GOODMAN/code_sheet/ShellFemSolver/shell_solver.py)
- **`__init__`**: `quad_global_i/j`, `tria_global_i/j`를 넘어, RBE2를 포함한 **전체 시스템의 `all_indices`를 사전 계산**.
- **`assemble`**:
  - `indices_i`, `indices_j` 생성을 제거하고 사전 계산된 캐시 사용.
  - `BCOO` 생성 시 인덱스 스택 연산을 생략하여 오버헤드 최소화.

---

### 2. [Root](file:///c:/Users/GOODMAN/code_sheet/) (Pipeline Activation & Constraints)

#### [MODIFY] [main_shell_verification.py](file:///c:/Users/GOODMAN/code_sheet/main_shell_verification.py)
- **Stage 1**: `min_bead_width`를 150.0으로 상향.
- **Stage 2**: 
  - 주석 해제하여 활성화.
  - `opt_config_stage2` (Stage 1 결과 계계) 및 `weights_stage2` (MAC 가중치 상향) 적용.
  - `learning_rate`를 0.1 이하로 설정하여 안정적 수렴 유도.

#### [MODIFY] [issue_tracker.md](file:///c:/Users/GOODMAN/code_sheet/issue_tracker.md)
- **ISSUE-028**: COO Index Caching을 통한 조립 성능 개선 기록.
- **ISSUE-029**: 제작 제약 조건(150mm) 반영 기록.
- **ISSUE-030**: Stage 2 활성화를 통한 모드 매칭 정밀도 향상 기록.

## Open Questions

> [!QUESTION]
> Stage 2의 `max_iterations`를 현재 주석된 350회 그대로 유지할까요, 아니면 초기 검증을 위해 100회 정도로 제한할까요?

## Verification Plan

### Automated Tests
- `python main_shell_verification.py` 실행을 통한 파이프라인 완주 확인.
- `verification_report.md` 점검.

### Manual Verification
- 루프 시간 단축 여부 확인.
