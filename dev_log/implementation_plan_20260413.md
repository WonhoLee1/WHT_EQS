# Shell FEM Stability and MITC3 Unification Plan

This plan aims to resolve the bending locking issue and restore the fidelity of the Shell FEM solver by unifying all triangular elements to the MITC3 formulation and fixing the degree-of-freedom (DOF) mapping.

## User Review Required

> [!IMPORTANT]
> - **Unification**: All `T3` elements will now use the `MITC3` formulation. The `DKT` (Discrete Kirchhoff Triangle) logic will be removed or bypassed to ensure stability.
> - **DOF Mapping**: We will use the standardize rotation mapping: `beta_x = thy`, `beta_y = -thx`. This is critical for coupling global rotations to local bending curvature.
> - **Compatibility**: Essential internal methods for index caching will be restored to support the `main_shell_opt.py` pipeline.

## Proposed Changes

### Shell Solver Core (`ShellFemSolver/shell_solver.py`)

#### [MODIFY] [shell_solver.py](file:///c:/Users/GOODMAN/code_sheet/ShellFemSolver/shell_solver.py)
1. **Unify T3 Assembly**: Update `ShellFEM.assemble` to call `compute_mitc3_local` for all 3-node elements.
2. **Fix MITC3 Mapping**: Implement `T_rot` transformation within `compute_mitc3_local` to map `[w, thx, thy]` to local `[w, beta_x, beta_y]`.
3. **Restore Caching**: Add the `_prepare_assembly_cache` method to `ShellFEM` class to fix the `AttributeError` in the optimization script.
4. **Curvature Recovery**: Update `recover_curvature_tria_bending` to align with the `MITC3` B-matrix definition and check for sign consistency.

## Verification Plan

### Automated Tests
1. **Optimization Pipeline**: Run `$env:RUN_FULL_OPT="1"; python main_shell_opt.py` to verify that the initialization crash is resolved and `bend_x/y` target values are non-zero.
2. **Fidelity Suite**: Run `python ShellFemSolverVerification/verification_runner.py` to ensure the overall score returns to $\ge 24/36$ and T3 bending errors are minimized.

### Manual Verification
- Inspect the generated `master_fidelity_report_final.md` to confirm T3/Q4 performance parity in bending cases.
