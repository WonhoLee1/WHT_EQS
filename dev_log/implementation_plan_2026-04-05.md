# T3 (DKT) Shell Element Fidelity Restoration Plan

DKT 요소의 해석 실패(Fidelity FAIL)와 코드 유실 문제를 해결하기 위해, 백업 파일로부터 `ShellFEM` 구조를 복구하고 검증된 Batoz(1980) 수단을 재적용합니다.

## User Review Required

> [!IMPORTANT]
> - `ShellFEM` 클래스 구조 복구: 앞선 작업 중 유실된 클래스 메서드(`assemble`, `solve_eigen` 등)를 백업 파일로부터 완전히 복원합니다.
> - `node_coords` 속성 추가: 가시화 코드(`WHT_EQS_visualization.py`)와의 호환성을 위해 `__init__` 시점에 필수 속성을 누락 없이 생성합니다.

## Proposed Changes

### [Component] ShellFemSolver/shell_solver.py

#### [MODIFY] [shell_solver.py](file:///c:/Users/GOODMAN/code_sheet/ShellFemSolver/shell_solver.py)
1. **[복구]** `ShellFEM` 클래스의 전체 구조를 백업(`shell_solver_backup_2026-04-05.py`)으로부터 대치.
2. **[수정]** `_B_bending_tria3` 및 `_K_bending_tria3`를 Batoz(1980) 명시적 B-Matrix 및 3점 가우스-해머 적분으로 교체.
3. **[수정]** `compute_tria3_local` 및 `compute_q4_local`에서 회전 관성(Rotary Inertia) 계산을 박판 이론($t^2/12$)에 맞게 정밀화하여 주파수 오차 해결.
4. **[수정]** `recover_curvature_tria_bending` 함수에 Kirchhoff 부호 규약($\kappa = -w_{,ii}$) 반영을 위한 부호 반전 추가.
5. **[수정]** `compute_field_results`에서 응력/변형률 산출 시 물리적 차원(Dimension) 일관성을 확보 (모멘트가 아닌 순수 곡률 기반 계산).

## Verification Plan

### Automated Tests
- `python main_shell_verification.py --test bending`: 정적 굽힘 해석 결과(처짐, 응력) PASS 여부 확인.
- `python main_shell_verification.py --test freq`: 고유진동수(Frequency) PASS 여부 확인.
- `python ShellFemSolverVerification/verification_runner.py`: 전체 모델에 대한 최종 Fidelity 리포트 생성 및 PASS 확인.

### Manual Verification
- `results/master_fidelity_report_final.md` 파일을 열어 모든 항목이 **PASS**로 표시되는지 육안 확인.
