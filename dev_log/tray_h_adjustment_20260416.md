# Implementation Plan - Tray Height Adjustment (2026-04-16)

## 1. 개요
사용자가 Tray의 외곽 벽 높이(`tray h=50.0`)를 조정할 수 있도록 시스템을 개편함. 기존의 하드코딩된 값을 제거하고 인수를 통해 동적으로 제어 가능하도록 수정.

## 2. 변경 사항
### 2.1. `WHT_EQS_analysis.py`
- `PlateFEM.__init__`에 `wall_width` 및 `wall_height` 인수를 추가.
- `generate_tray_mesh_quads` 호출 시 해당 인수를 사용하도록 수정.

### 2.2. `main_shell_opt.py`
- `EquivalentSheetModel.__init__`이 `wall_height`를 받아 `PlateFEM`에 전달하도록 수정.
- `argparse` 섹션에 `--tray_h` 인수를 추가 (기본값 50.0).
- `model` 생성 시 `args.tray_h`를 사용하도록 업데이트.

## 3. 사용 방법
터미널에서 다음과 같이 실행하여 Tray 높이를 조정할 수 있음:
```powershell
python main_shell_opt.py --run --tray_h 30.0
```

## 4. 기대 효과
- 다양한 Tray 형상에 대한 최적화를 코드 수정 없이 수행 가능.
- CLI를 통한 실험 자동화 용이성 증대.
