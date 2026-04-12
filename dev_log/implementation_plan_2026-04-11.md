# Tray Topography Optimization 가동 및 검증 계획

솔버의 정밀도 검증이 완료되었으므로, 대형 Tray 모델을 대상으로 한 **등가 시트 모델링(Equivalent Sheet Modeling)** 최적화 루틴을 본격적으로 가동합니다.

## User Review Required

> [!IMPORTANT]
> **연산 시간 및 리소스**
> - 본 최적화 루틴은 80회 이상의 반복 계산을 수행하며, JAX의 JIT 컴파일 과정을 포함하여 약 5~10분 정도의 연산 시간이 소요될 수 있습니다.
> - 최적화 도중 'q' 키를 누르면 즉시 중단하고 그때까지의 최적 결과를 저장합니다.

## Proposed Changes

### 1. 최적화 엔진 설정 (`main_shell_verification.py`)
- [MODIFY] `Lx, Ly`: 1450.0, 850.0 mm (대형 Tray 규격)
- [MODIFY] `target_config`: TNYBV 복합 비드 패턴 적용
- [MODIFY] `opt_config`: `pz` (Topography) 설계 변수 활성화 및 `bead_smoothing` 필터(120mm) 적용

### 2. 최적화 가동 및 모니터링
- [RUN] `python main_shell_verification.py`
- 실시간 수렴 로그 분석: Displacement 점진적 감소 및 Frequency 정합성 확인.
- JIT 컴파일 최적화 확인 (XLA CPU Multi-threading 활성화).

### 3. 최종 검증 보고서 생성
- [NEW] `verification_report.md`: 최적화 전/후의 오차율 비교 테이블.
- [OUTPUT] `verify_3d_*.png`: 고유모드 형상 및 파라미터 진화 과정 시각화.

## Verification Plan

### Automated Tests
- `python main_shell_verification.py` 실행 후 `verification_report.md` 생성 여부 및 오차율 10% 이내 진입 확인.

### Manual Verification
- ParaView를 통해 `opt_modal_results.vtkhdf`를 열어 진동 모드 형상이 타겟 비드판과 유사한지 육안 검증.
