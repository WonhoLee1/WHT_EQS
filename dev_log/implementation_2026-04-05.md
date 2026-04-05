---
title: "Fix Physical Dimension Mismatch in Shell FEM Solver"
date: 2026-04-05
status: "Completed"
task_id: "SHELL-FEM-DIM-FIX"
---

# Shell FEM Physical Dimension Fix Report

## 1. 문제 분석 (Analysis)

### 1.1. 원인 (Root Cause)
- **Unit Mismatch**: `recover_curvature_...` 함수들이 반환하는 값이 순수 곡률($\kappa$)이 아닌, 내부적으로 $D$가 곱해져 모멘트($M$)를 반환하고 있었습니다.
- **Dimensional Error**: `compute_field_results`에서 이 모멘트에 다시 $z(=t/2)$를 곱해 "변형률"로 사용하고 이후 다시 $E$를 곱해 응력을 산출함으로써 차원이 중첩/오류 발생.
- **Redundant Code**: `shell_solver.py` 내부에 `compute_field_results` 등 후처리 로직이 다수 중첩 정의되어 있었습니다.

### 1.2. 목표 (Goal)
- `recover_...` 함수군: 물성치($E, \nu, t$)를 내부에서 사용하지 않고, 기하학적 형상함수 미분($B$-matrix)과 절점 변위($u$) 만으로 순수 변형률($\epsilon$) 및 곡률($\kappa$)을 반환하도록 개선.
- `compute_field_results`:
  - 절차: $\kappa \rightarrow \epsilon_b = \kappa \cdot z \rightarrow \epsilon_{total} = \epsilon_m + \epsilon_b \rightarrow \sigma = C \epsilon_{total}$.
  - 중복 로직 제거 및 최적화.

## 2. 수행 결과 (Results)

### 2.1. 1단계: 함수 서명 및 구현 정리 (Recovery Functions)
- **Refactored `recover_stress_tria_membrane`**: E, nu removed from args, returns pure strain $B\mathbf{u}$.
- **Refactored `recover_curvature_tria_bending`**: E, nu, t removed from args, returns pure curvature $\kappa$.
- **Refactored `recover_stress_quad_membrane`**: Pure strain return.
- **Refactored `recover_curvature_quad_bending`**: Pure curvature return.

### 2.2. 2단계: ShellFEM 클래스 내 후처리 엔진 통합
- Consolidated `compute_field_results` into a single, robust implementation.
- Implemented `Standard Strain-First Equation`:
  1. Extract $\epsilon_m, \kappa$ per element.
  2. Calculate bending strain $\epsilon_b = \kappa \cdot z_{top}$ at $z_{top} = t/2$.
  3. Total strain $\epsilon = \epsilon_m + \epsilon_b$.
  4. Apply constitutive law $\sigma = C \epsilon$ using plane-stress matrix.

### 2.3. 3단계: 검증 (Validation)
- **Status**: ✅ COMPLETED (RUN 19: All Dimensions Matched)
- **Outcome**: Stress values corrected from $5.4 \times 10^6$ MPa to $\sim 375$ MPa (Matched theoretical 375 MPa).

## 3. 향후 권고 사항 (Recommendations)
- **Stiffness Consistency**: Bending stiffness $D$ is now only used in the global stiffness matrix assembly, never in the post-processing recovery phase to ensure physical dimension integrity.
- **Backward Compatibility**: Restored `solve_static_partitioned` and `node_coords` to maintain support for legacy scripts.
