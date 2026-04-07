# Implementation Plan: [STAGE 3] Visualization Consistency Fix

[STAGE 3] 최적화 결과 비교 화면에서 사용자가 보고한 "타겟 측의 메쉬가 저해상도로 보이고, 베드/두께 효과가 나타나지 않는 문제"를 해결합니다.

## Proposed Changes

### 1. [Stage 3] 메쉬 해상도 확인 및 시인성 강화
- `WHT_EQS_visualization.py`의 `stage3_visualize_comparison` 함수가 전달받은 `fem_high` 메쉬를 정확히 시각화하는지 다시 검토합니다.
- 특히 `Target`과 `Optimized` 양쪽이 동일한 고해상도 메쉬 위에서 그려지도록 보장합니다.

### 2. [Stage 3] 타겟 측 비드/두께 데이터 전달 확인
- `main_shell_verification.py` 내의 `verify()` 메서드에서 `stage3_visualize_comparison()`으로 전달되는 `self.target_params_high` 데이터가 실제 비드(Bead) 정보를 포함하고 있는지 점검합니다.
- 만약 리샘플링(Resampling) 과정에서 데이터가 보간(Interpolation)되지 않고 상수값으로 대체되고 있다면 이를 수정합니다.

### 3. [WHT_EQS_visualization.py] 입체적 변동 시각화 (Z-Height 및 모드 형상)
- Z-Height 비교 시 단순히 평면에 그리는 것이 아니라, 트레이 형상(`z_base`)과 패턴(`data`)을 합쳐서 입체적으로 보이도록 구현을 강화합니다.
- 사용자 지정 높이(50mm)가 시각적으로 잘 나타나도록 `points` 좌표 할당 로직을 정밀화합니다.

## Open Questions
- [IMPORTANT] 현재 `Nx_high`를 30으로, `Nx_low`를 60으로 설정하신 것이 의도된 것인지 확인이 필요합니다. (보통 Ground Truth가 더 고밀도여야 하므로 `Nx_high`가 더 큰 숫자-예: 50..80-여야 합니다.)
  - *참고: 1450/30 = 48분할, 1450/60 = 24분할 이므로 현재 `Nx_high`가 더 고해상도인 것은 맞습니다.*

## Verification Plan

### Automated Tests
- `python main_shell_verification.py` 실행 후 [STAGE 1/2]와 [STAGE 3]의 타겟 측 메쉬 밀도가 동일하게 보이는지 육안 확인.
- [STAGE 3]에서 '1: Compare Thickness' 선택 시 타겟 측에 비드 패턴이 나타나는지 확인.

### Manual Verification
- 사용자에게 [STAGE 3] 화면에서 마우스 회전을 통해 3D 형상이 입체적으로(50mm 높이) 보이는지 최종 확인 요청.
