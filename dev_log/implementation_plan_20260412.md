# Implementation Plan - Solver Standardization & Verification Fix

This plan addresses both the zero-bending displacement issue in the `ShellFEM` solver and the `'BCOO' object has no attribute 'dot'` crash in the verification pipeline.

## User Review Required

> [!IMPORTANT]
> - The **Shell Solver Refactor** will correctly couple rotations to out-of-plane displacements. This is a fundamental fix for the project's structural accuracy.
> - The **Verification Fix** removes legacy `.dot()` calls that are incompatible with JAX's sparse BCOO matrices.

## Proposed Changes

### 1. Shell FEM Solver (Core)

#### [MODIFY] [shell_solver.py](file:///c:/Users/GOODMAN/code_sheet/ShellFemSolver/shell_solver.py)
- **Standardize Mapping**: Implement `T_rot_single = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]` across all elements.
- **Synchronize B-Matrices**: Update MITC3, MITC4, DKT, and Standard Q4 B-matrices to use the local order `[w, beta_x, beta_y]`.
- **Refactor Recovery**: Update `recover_curvature` functions to match the standardized local coordinate system.

### 2. Verification Pipeline (Bug Fix)

#### [MODIFY] [main_shell_verification.py](file:///c:/Users/GOODMAN/code_sheet/main_shell_verification.py)
- **Fix BCOO Attribute Error**: Delete line 1751 (`R_full_opt = K_opt.dot(...)`) which is redundant and physically broken. Ensure line 1752 (using `@`) is the only active calculation.
- **Update Metrics Mapping**: Ensure `max_strain` in the report correctly maps to `strain_equiv`.

---

## Open Questions

- None. The bug in `main_shell_verification.py` is a clear syntax/version mismatch for JAX sparse arrays.

## Verification Plan

### Automated Tests
1. **Solver Test**: Run the manual bending script. Verify `Max W` > 10mm.
2. **Crash Test**: Run `python main_shell_opt.py` and ensure it passes the `model.verify()` stage without the `AttributeError`.

### Manual Verification
- Review the generated "Professional Structural Optimization Verification Report" and check if Bending (`bend_x/y`) now shows non-zero `Target Val`.
