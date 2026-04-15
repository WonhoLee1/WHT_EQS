# Implementation Plan - Shell FEM Solver Upgrade (DKT to MITC3)
**Date**: 2026-04-16
**Author**: Antigravity (Assistant)

## 1. Summary
Replacing the legacy DKT (Discrete Kirchhoff Triangle) implementation with the modern MITC3 (Mixed Interpolation of Tensorial Components) element to ensure robustness against shear locking and support for both thin and thick shell analysis.

## 2. Proposed Changes

### 2.1. `ShellFemSolver/shell_solver.py`
- **Deprecate/Remove DKT**: Remove `_get_B_dkt` and associated logic from the main solver path.
- **Implement MITC3 Bending**: Use standard Mindlin bending matrices for the triangular element.
- **Refine MITC3 Shear**: Update `_get_B_mitc3` to correctly implement the edge-based shear strain interpolation according to Bathe's MITC3 formulation.
- **Update Assembly**: 
    - In `compute_mitc3_local`, integrate:
        - Membrane stiffness (CST).
        - Bending stiffness (Linear interpolation).
        - Shear stiffness (MITC3 interpolated).
    - Ensure correct transformation between global [thx, thy] and local [beta_x, beta_y].

### 2.2. Validation
- Update `compare_solvers.py` to verify the accuracy of the new MITC3 implementation against solid mesh models.

## 3. Implementation Details (MITC3 Shear)
MITC3 interpolates covariant shear strains $\gamma_{13}$ and $\gamma_{23}$ such that:
- The shear strain along each edge is constant.
- These constant edge shear strains are derived from the nodal displacements $w_i$ and rotations $\beta_i$.

## 4. Schedule
1. Update `shell_solver.py` with MITC3 logic.
2. Verify with existing test cases.
3. Update documentation and dev logs.
