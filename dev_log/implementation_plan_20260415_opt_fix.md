# Shell FEM Optimization Diagnosis & Fix (main_shell_opt.py)

This plan addresses the recurring MAC=0 issue in the optimization script and improves visibility into the global design variables (`t`, `rho`, `E`).

## User Review Required

> [!NOTE]
> **MAC=0 Root Cause:** I previously fixed the 3D interpolation in `main_shell_verification.py`, but `main_shell_opt.py` (the script you are currently running) still uses the old 2D projection logic, which causes matching failures on vertical walls.

> [!INFO]
> **Global Variable Monitoring:** `t`, `rho`, and `E` are being optimized as scalars. Because their gradients are summed over the entire mesh, they might move slowly compared to local `pz` nodes, or the mass penalty might be keeping them close to the initial state. I will add explicit value tracking to the logs.

## Proposed Changes

### 1. Fix MAC=0.0 & Interpolation

#### [MODIFY] [main_shell_opt.py](file:///c:/Users/GOODMAN/code_sheet/main_shell_opt.py)
*   **Coordinate Fix:** Change `pts_l` and `pts_h` to use all 3 dimensions (`[:, :3]`) to handle the 3D topography of the tray correctly during interpolation.
*   **Safe Interpolation update:** Ensure `safe_interp` uses the 3D points.

### 2. Design Variable Monitoring & Visibility

#### [MODIFY] [main_shell_opt.py](file:///c:/Users/GOODMAN/code_sheet/main_shell_opt.py)
*   **Enhanced Logging:** 
    *   Update the iteration printer to show the current physical values of `t`, `rho`, and `E` every 10 iterations.
    *   Increase precision for `dt` and `dpz` display to capture subtle movements.
    *   Explicitly print if a parameter is "Fixed" vs "Optimizing" at the start.

### 3. Structural Consistency

*   Ensure the `loss_fn` in `main_shell_opt.py` correctly handles the global vs local parameter broadcasting, matching the optimized logic in `main_shell_verification.py`.

## Verification Plan

### Automated Tests
*   Run `python main_shell_opt.py` with `RUN_FULL_OPT=1`.
*   **Check:**
    *   The `MAC` column in the evaluation report should show non-zero values (typically > 0.8) from Iteration 0 if `init_pz_from_gt=True` is used.
    *   The log should now show `t: 1.000 -> 1.005` or similar changes.

### Manual Verification
*   Verify that `dt` is no longer exactly `0.0000` after a few iterations if the target requires a thickness change.
