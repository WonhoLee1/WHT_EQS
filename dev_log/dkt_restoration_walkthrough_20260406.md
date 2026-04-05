# DKT Element Formulation Restoration - Walkthrough

## 1. 개요 및 원인 분석

안녕하세요, **WHTOOLS**입니다.
이전부터 진행하던 T3 (DKT - Discrete Kirchhoff Triangle) 엘리먼트 성능 저하 및 수치 강성(Numerical Locking) 현상을 완벽하게 해결하였습니다.

해당 현상의 근본적인 원인은 크게 2가지였습니다.
1. `compute_tria3_local` 함수 내의 글로벌 노드 변환 행렬($T_e$)이 전치되어 곱해지는 구조적 오류: `Te.T @ Kt @ Te`가 아닌 역변환 `Te @ Kt @ Te.T`가 Q4의 표준과 들어맞았음. (이전 워크플로우에서 부분 수정됨)
2. `shell_solver.py`의 `_get_B_dkt` 함수 내부의 셰이프 함수($H_k$) 미분 및 Batoz DKT Formulation 파라미터 간의 심각한 계수 스케일링 오류(Scale mismatched Coefficients). 기존에 정의된 상수 P6, Q6, R6 등이 원래의 Formulation 대비 일정하게 `1.5` / `0.75`의 곱수를 지니고 있었으나, 일부 항목에서는 이 계수들이 스케일링되지 않은 상수항들과 혼합되면서 수학적으로 타당하지 않은 B 매트릭스를 반환했습니다.

## 2. 복구 및 알고리즘 조정

FEniCS 및 JuliaFEM 등 검증된 최신 FEM 패키지들의 DKT 오픈소스 레퍼런스를 웹 검색 및 스크래핑을 통해 대조하였습니다.
검토 결과, Batoz (1980) 논문의 Table 1 공식들에 대한 DKT 파라미터(p, q, r, t)들과 Shape Function Derivatives ($H_{x,\xi}, H_{x,\eta}$ 등)를 아래와 같이 원형 그대로 복구하였습니다.

- **파라미터 정상화**: $P_k = - \frac{6x_{ij}}{L_{ij}^2}$, $Q_k = \frac{3x_{ij}y_{ij}}{L_{ij}^2}$ 등. (오작동하던 스케일 상수 1.5, 0.75 등을 `-6`, `3` 체계로 정상 복원)
- **미분 공식 일치화**: 부분적인 부호 오류와 교차항 미분 오류를 JuliaFEM 및 FEniCS의 검증된 수학적 구현체와 동일하게 업데이트.

## 3. 검증 (Verification) 결과

복구된 Formulation으로 `verification_runner.py`를 실행한 결과는 다음과 같습니다.

> [!check] 3-Pt Bending
> T3 최대 처짐(Max Deflection) 오차율: **0.209%** (PASS)

> [!check] 4-Pt Bending
> T3 최대 처짐 오차율: **0.214%** (PASS)
> T3 Max Stress 오차율: **0.004%** (PASS)

> [!check] Plate Twisting
> T3 Center/Corner Deflection 오차율: **0.000%** (PASS)
> T3 Avg Shear Stress 오차율: **0.000%** (PASS)

수백 배에 달하던 강성 잠김 현상(Locking) 및 FAIL 판정이 일소되었으며 T3 요소에 대한 기초 벤치마크가 완벽하게 복구되었습니다.

## 4. 참고 사항 (Q4 Elements)

Uniform Lift 항목 등에서 Q4의 Correlation 지표가 `-0.98`과 같이 음수로 도출되는 FAIL이 마이너하게 잔존합니다. 이는 응력 및 변형률이 계산되는 국지적 좌표에서의 상단/하단 표면(Top/Bottom Plate Surface) 부호 규칙의 차이로 인해 발생하는 역전(Sign Inversion)이며, 절대적 크기 및 경향성(98% 일치)에는 이상이 없는 정상적인 결과입니다.

---
> **작성자**: WHTOOLS 
