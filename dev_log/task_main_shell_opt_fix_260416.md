# TASK: Fix Optimization Pipeline Indentation & Logic (main_shell_opt.py)

## 1. Issue Description
- **Error:** `IndentationError: expected an indented block after function definition on line 264`
- **Root Cause:** Incomplete indentation during the porting of `optimize_v2` and `verify` methods from the verification script.
- **Secondary Issues:** 
    - Missing variable definitions (`cand_modes`, `norm_t`).
    - Dimension mismatch in modal matching (`dot_general` error between 375 nodes and 2250 DOFs).
    - `ImportError` for `get_metrics`.

## 2. Actions Taken
- **Indentation Repair:** Re-indented the entire `optimize_v2` and `verify` function bodies (lines 277-1250) to align with the `EquivalentSheetModel` class.
- **Variable Alignment:**
    - Explicitly defined `cand_modes = vecs_filtered[2::6, :]` to extract Z-displacements for modal matching.
    - Updated `target_n` normalization to use the pre-calculated `t_norms`.
- **Import Optimization:** Removed invalid `get_metrics` import and implemented it as a local helper within `verify()`.
- **Validation Run:** Executed `$env:RUN_FULL_OPT="1"; python main_shell_opt.py` and confirmed:
    - JAX graph compilation success.
    - Loss reduction from `2.7e4` to `5.4e3` in 3 iterations.
    - Beautifully formatted modal and optimization reports are functional.

## 3. Results
- [x] IndentationError resolved.
- [x] Modal matching dimension mismatch resolved.
- [x] Post-processing similarity metrics functional.
- [x] Pipeline stable and ready for full optimization runs.
