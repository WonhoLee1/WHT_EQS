# Implementation Plan: Porting Visualization and Verification Features

## 1. Objective
Port the high-fidelity post-processing and interactive visualization workflows from `main_shell_verification.py` to `main_shell_opt.py` to provide a seamless optimization and verification pipeline.

## 2. Key Features to Port
- **Stage 1 & 2: Ground Truth Visualization**
    - `generate_targets` with `stage1_visualize_patterns` and `stage2_visualize_ground_truth`.
    - Automated generation of `ground_truth_3d_loadcases.png`.
    - Modal analysis export to VTKHDF.
- **Stage 4: Comprehensive Verification**
    - `verify` method with high-resolution interpolation and sparse solving.
    - Detailed 3x3 per-case performance plots (`verify_3d_*.png`).
    - Parameter evolution and consistency check plots.
    - Automated Markdown report generation (`verification_report.md`).
    - Final interactive PyVista comparison stage.

## 3. Implementation Steps
1. **Dependency Consolidation**:
    - Import all mandatory modules (`pickle`, `datetime`, `koreanize_matplotlib`).
    - Import specialized utility modules (`WHTable`, `wh_print_banner`, `safe_eigh`).
2. **Class-Local Definition**:
    - Move `EquivalentSheetModel` class definition into `main_shell_opt.py`.
    - Integrate `optimize_v2` as a standard method of the class.
    - Integrate `generate_targets` and `verify` with full rich logic.
3. **Refactoring Monkey-Patching**:
    - Remove external monkey-patching and redundant imports from `main_shell_verification.py`.
4. **Validation**:
    - Ensure all file paths and mesh settings (Lx, Ly, Nx) are synchronized.
    - Test the full pipeline: Target Gen -> Optimization -> Verification.

## 4. Expected Deliverables
- `main_shell_opt.py`: Updated self-contained script.
- `ground_truth_3d_loadcases.png`: Initial pattern check.
- `verify_3d_*.png`: Final performance assessment.
- `verification_report.md`: Quantitative accuracy summary.
