# Implementation Plan: Restoring Shell FEM Optimization Performance (Frequency Matching)

현장 분석 결과, 최적화 모델의 고유진동수가 **3.2 Hz (평판 수준)**에 머물러 있는 것은 **지형 최적화(Topography/PZ)가 진행되지 않았거나 비드에 의한 강성 보강 효과(Arch effect)를 Solver가 포착하지 못하고 있음**을 의미합니다. 이를 해결하기 위해 Solver의 수치적 정합성을 맞추고 최적화 설정을 대폭 강화합니다.

## User Review Required

> [!IMPORTANT]
> **최적화 시간 증가**: 반복 횟수를 80회에서 200회 이상으로 늘릴 예정입니다. 이는 계산 시간을 증가시킬 수 있으나, 비드 형성(Bead growth)을 관찰하기 위해 필수적입니다.
> 
> **PZ 초기값 강제화**: `gt_init_scale`을 현재 0.3에서 0.5~0.6 정도로 높여 Optimizer가 비드의 존재를 명확히 인지하게 할 예정입니다.

## Proposed Changes

### [Core Solver]
#### [MODIFY] [shell_solver.py](file:///c:/Users/GOODMAN/code_sheet/ShellFemSolver/shell_solver.py)
- **Bending Sign Alignment**: `_B_bending_q4`에서의 $\kappa_x$ 공식과 `recover_curvature_quad_bending`에서의 $\kappa_x$ 부호가 반대(+, -)인 부분을 일치시켜 물리적 일관성 확보.
- **Geometric Coupling Verification**: 요소가 기울어졌을 때(Bead slope) 면내 강성이 면외 거동으로 전이되는 `Ts_q` 회전 행렬 연산부의 재검증.

---

### [Optimization Engine]
#### [MODIFY] [main_shell_verification.py](file:///c:/Users/GOODMAN/code_sheet/main_shell_verification.py)
- **Iteration 상향**: `Stage 1` Coarse 최적화 80 -> 200회로 증설.
- **Learning Rate 조정**: `pz` 파라미터의 변동성을 키우기 위해 LR을 0.1에서 0.5 수준으로 상향 검토.
- **Loss Logic 보강**: `l_freq` 계산 시 1차 모드 불일치에 대한 패널티 가중치를 강화.
- **초기화 누락 방지**: `target_z_low`가 `None`이거나 0이 아닌지 체크하는 로직 추가.

## Open Questions

- [ ] 현재 3.2 Hz 모드가 단순히 Rigid Body Mode(RBM)를 잘못 잡은 것인지, 아니면 실제 탄성 1차 모드인지 `freq_all`을 전수 조사할 필요가 있습니다. (현재 3.2는 평판 1차 모드와 일치함)

## Verification Plan

### Automated Tests
- `python main_shell_verification.py` 실행 후 [STAGE 3] 결과에서 **Mode 1의 Optimized 주파수가 20Hz 이상으로 상승**하는지 확인.
- 비드 형상이 평탄화되지 않고 Target의 패턴을 추종하는지 시각적 확인.

### Manual Verification
- `verification_report.md` 내의 MAC(Modal Assurance Criterion) 지수가 0.1에서 0.7 이상으로 개선되는지 확인.
