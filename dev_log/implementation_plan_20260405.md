# [DKT] Shell FEM Fidelity Restoration Plan (T3 element)

Currently, the T3 (DKT) element shows significant errors (32% to 164%) across all bending and twisting benchmarks. While the Q4 element is performing well, the DKT formulation in `shell_solver.py` is numerically unstable or physically misaligned with the global solver's sign conventions.

## User Review Required

> [!IMPORTANT]
> **DOF Mapping Change**: To align with Batoz (1980) formulation, the mapping between `ShellFEM` rotations (`phix`, `phiy`) and DKT slopes ($\theta_x = w_{,x}$, $\theta_y = w_{,y}$) must be strictly enforced:
> - $\theta_{xi} = - \text{phi}_{yi}$ (Rotation about Y affects slope in X)
> - $\theta_{yi} = \text{phi}_{xi}$ (Rotation about X affects slope in Y)
> Failure to apply these sign flips results in a non-symmetric or physically incorrect stiffness matrix.
# Implementation Plan: T3 Shell Element Fidelity Restoration (Ver 3.0)

## Goal
Achieve 100% PASS on the ShellFEM Master Fidelity Report for T3 (DKT) elements. Currently, T3 elements exhibit ~99% error (too stiff), while Q4 elements are passing. This indicates a fundamental scaling or sign error in the DKT B-matrix.

## Proposed Changes

### [ShellFemSolver] (file:///c:/Users/GOODMAN/code_sheet/ShellFemSolver/shell_solver.py)

#### [MODIFY] `_get_B_dkt`
- Rewrite the B-matrix coefficients using the definitive **Batoz (1980)** standard.
- Correct the nodal DOF mapping: $w_{,x} = \theta_y$ and $w_{,y} = -\theta_x$.
- Ensure the Jacobian determinant ($2A$) is applied exactly once.
- Standardize edge indices (Edge 1: 1-2, Edge 2: 2-3, Edge 3: 3-1).

#### [MODIFY] `compute_tria3_local`
- Ensure 3-point Gauss-Hammer quadrature (points: (1/6,1/6), (2/3,1/6), (1/6,2/3) with weights 1/3) is used.
- Verify the integration factor: $\int B^T D B dA = \sum (B^T D B) \times A \cdot w_i$.
- Add a tiny drilling stabilization ($10^{-5} \cdot Et \cdot A$) to the $r_z$ diagonal.

#### [MODIFY] `recover_curvature_tria_bending`
- Synchronize with the new `_get_B_dkt` to ensure stress recovery is physically consistent with the stiffness matrix.

## Verification Plan

### Automated Tests
- Run `python ShellFemSolverVerification\verification_runner.py`.
- Target: 3-Pt Bending Max Deflection Error < 1%.
- Target: Plate Twisting Corner Deflection Error < 5%.
- Target: Bending Patch Test Residual < 1e-10.

### Manual Verification
- Inspect the `results\master_fidelity_report_final.md` to confirm all T3 cases are [PASS].
x $K_e$.
- Check the sign of the recovered moments under a simple point load.
