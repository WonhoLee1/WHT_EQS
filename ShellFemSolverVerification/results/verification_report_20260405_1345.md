# ShellFEM Solver 패치 테스트 검증 보고서

> 생성: **2026-04-05 13:45**  
> 단위: mm / MPa (N/mm2) / tonne  
> Solver: `ShellFemSolver/shell_solver.py`

---

## 1. 검증 개요

표준 FEM 패치 테스트를 통해 쉘이론(Kirchhoff)과 FEM 수치해를 비교합니다.  
응력·변형률·변위 필드는 B-matrix 기반 물리적 응력 회복으로 계산합니다.

| 구분 | 내용 |
|------|------|
| 요소 유형 | CQUAD4 (Q4 Mindlin), CTRIA3 (T3 DKT) |
| 자유도 | 절점당 6-DOF: [u, v, w, theta_x, theta_y, theta_z] |
| 응력 회복 | B-matrix 기반 (CST membrane + DKT bending 곡률) |
| 이론 기준 | Kirchhoff 판 이론, Navier 급수해, 재료역학 보 이론 |

---

## 2. 검증 결과

| 테스트명 | 요소 | 물리량 | 이론값 | FEM값 | 오차(%) | 허용(%) | 결과 |
|----------|------|--------|--------|-------|---------|---------|------|
| Membrane Patch (T3) | T3 | σx 평균 | 100 | 100 | 0.000 | 0.5 | PASS |
|  |  | *σx_std=0.0000 MPa (균일도 오차 0.0000%)* |  |  |  |  |  |
| Membrane Patch (T3) | T3 | σy (= 0 기대) | 0 | -9.316e-16 | 0.000 | 0.5 | PASS |
| Membrane Patch (Q4) | Q4 | σx 평균 | 100 | 100 | 0.000 | 0.5 | PASS |
|  |  | *σx_std=0.0000 MPa (균일도 오차 0.0000%)* |  |  |  |  |  |
| Membrane Patch (Q4) | Q4 | σy (= 0 기대) | 0 | 3.6636e-15 | 0.000 | 0.5 | PASS |
| Bending Patch (T3-DKT) | T3 | 내부절점 w 최대 오차 [%] | 0 | 176.09 | 176.093 | 100.0 | FAIL |
|  |  | *w_err=22.319%, ty_err=176.093% | kx_mean=-4.518e-19 (DKT는 2차 Bending 패치 강형 미충족 — 이론적 한계)* |  |  |  |  |  |
| Constant Strain Patch (T3) | T3 | 내부 절점 u/v 최대 오차 | 0 | 8.8285e-13 | 0.000 | 0.0 | PASS |
|  |  | *u_err=8.83e-13%, v_err=7.10e-13%* |  |  |  |  |  |
| SS Plate Deflection (Q4) | Q4 | 중앙 처짐 w_center | 15.303 | 15.108 | 1.277 | 3.0 | PASS |
|  |  | *Navier 해 n_terms=20, q=0.001 MPa, t=1.0 mm* |  |  |  |  |  |
| SS Plate Deflection (T3) | T3 | 중앙 처짐 w_center | 15.303 | 15.155 | 0.968 | 5.0 | PASS |
|  |  | *Navier 해 n_terms=20, q=0.001 MPa, t=1.0 mm* |  |  |  |  |  |
| SS Plate Frequency (Q4) | Q4 | 1차 고유진동수 f₁₁ | 17.825 | 17.812 | 0.072 | 3.0 | PASS |
|  |  | *Kirchhoff 이론: f₁₁=17.8247 Hz* |  |  |  |  |  |
| SS Plate Frequency (T3) | T3 | 1차 고유진동수 f₁₁ | 17.825 | 17.797 | 0.153 | 3.0 | PASS |
|  |  | *Kirchhoff 이론: f₁₁=17.8247 Hz* |  |  |  |  |  |
| Cantilever Plate (Q4) | Q4 | 끝단 처짐 w_tip | 0.60952 | 0.60286 | 1.093 | 3.0 | PASS |
|  |  | *I=208.33 mm⁴, L=200.0 mm, P=10.0 N* |  |  |  |  |  |
| Cantilever Plate (T3) | T3 | 끝단 처짐 w_tip | 0.60952 | 0.59955 | 1.636 | 5.0 | PASS |
|  |  | *I=208.33 mm⁴, L=200.0 mm, P=10.0 N* |  |  |  |  |  |

---

## 3. 종합 판정

| 총 테스트 | 합격 | 불합격 | 판정 |
|-----------|------|--------|------|
| 12 | 11 | 1 | **1개 불합격** |

---

## 4. 테스트 설명

| # | 테스트 | 목적 | 이론 기반 |
|---|--------|------|---------|
| 1 | Membrane Patch (T3/Q4) | 균일 면내 응력 재현 | CST plane-stress |
| 2 | Bending Patch (T3-DKT) | 균일 굽힘 곡률 재현 | Kirchhoff 판 이론 |
| 3 | Constant Strain (T3) | FEM 필수 수렴 조건 | 선형 변위장 완전 재현 |
| 4 | SS Deflection (Q4/T3) | 정적 처짐 정확도 | Navier 급수해 |
| 5 | SS Frequency (Q4/T3) | 1차 고유진동수 정확도 | Kirchhoff 이론 |
| 6 | Cantilever (Q4/T3) | 외팔보 처짐 | 재료역학 보 이론 |

---

## 5. 합격 기준

| 테스트 유형 | 기준 |
|------------|------|
| 패치 테스트 (막/굽힘) | 오차 < 0.5% |
| 선형 변위장 재현 | 최대 오차 < 0.01% |
| 정적 처짐 (Q4) | 오차 < 3% |
| 정적 처짐 (T3) | 오차 < 5% |
| 고유진동수 | 오차 < 3% |
| 외팔보 처짐 (Q4) | 오차 < 3% |
| 외팔보 처짐 (T3) | 오차 < 5% |

---

> **참고**: Batoz & Bathe (1980) DKT, Bletzinger et al. (2000) DSG3