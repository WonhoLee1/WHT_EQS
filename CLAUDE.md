# CLAUDE.md
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.



## Commands

### Install dependencies
```bash
pip install jax jaxlib numpy matplotlib pyvista optax scipy h5py
```

### Run unit tests
```bash
# From the repo root
python -m pytest tests/test_opt_targets.py -v

# Run a single test method
python -m pytest tests/test_opt_targets.py::TestOptTargets::test_modes_mac -v
```

### Run the FEM solver verification suite
```bash
python ShellFemSolverVerification/verification_runner.py
python ShellFemSolverVerification/patch_tests.py
```

### Run the full optimization pipeline
```bash
# Full interactive pipeline: generate GT → optimize → visualize
python main_shell_verification.py

# Non-interactive (suppresses PyVista windows)
set NON_INTERACTIVE=1 && python main_shell_verification.py

# Optimization-only (requires pre-cached target_cache.pkl)
set RUN_FULL_OPT=1 && python main_shell_opt.py
```

---

## Architecture Overview

### 3-Stage Pipeline (`main_shell_verification.py`)

The top-level workflow, orchestrated by `EquivalentSheetModel`, runs three stages:

1. **Stage 1 – Generate Ground Truth**: A high-resolution FEM model (`fem_high`, e.g. 120×60 elements) is created with a bead pattern (alphanumeric font rendered via `WHT_EQS_pattern_generator.py`). Static and modal analyses are run for every `LoadCase`. Results are cached to `target_cache.pkl` and as `.vtkhdf` files.

2. **Stage 2 – Optimize**: A low-resolution FEM model (`fem`, e.g. 30×15 elements) optimizes parameters `{t, rho, E, pz}` to reproduce the high-res GT behavior. The `optimize_v2()` method (in `main_shell_opt.py`, monkey-patched onto `EquivalentSheetModel`) drives this using JAX's `value_and_grad` + Optax Adam.

3. **Stage 3 – Verify**: Side-by-side comparison of optimized vs. GT results via PyVista.

### Key Design Patterns

**Two-Track JIT** (`main_shell_opt.py`):
Two separate JIT-compiled functions — `loss_vg_with_eigen` and `loss_vg_no_eigen` — alternate every `eigen_freq` steps. The heavy `eigh` call is skipped on non-eigen steps and replaced by a Rayleigh quotient approximation that keeps the gradient chain alive.

**AOT Pruning**:
The `case_needs` dict pre-analyzes which physical quantities (stress, strain, reaction, displacement) are actually required by the configured `OptTarget` list. Python `if` branches before JIT compilation eliminate unused FEM computations from the XLA graph entirely.

**Safe Eigenvalue Gradients** (`solver.py`):
A custom VJP (`safe_eigh`) handles repeated eigenvalues that cause NaN gradients in JAX's default `eigh`. Micro-perturbation (`eps_unique = jnp.logspace(-9, -4, N)`) is added to diagonals before calling `safe_eigh`.

**Parameter Scaling**:
All optimization parameters are internally divided by their initial magnitude (`self.scaling[k]`). This normalizes Adam's gradient scale across variables with very different physical units (e.g. `E=210000 MPa` vs `t=1 mm`).

**2D Parametric Mode Mapping**:
When the optimization grid resolution differs from the GT resolution, mode shapes are mapped through 2D parametric space `(x/Lx, y/Ly)` rather than 3D coordinates. This prevents mapping failures on non-flat geometry (e.g. tray walls).

### Module Responsibilities

| Module | Class / Entry | Role |
|--------|--------------|------|
| `main_shell_verification.py` | `EquivalentSheetModel` | Top-level pipeline: adds cases, generates GT, calls optimize, verify |
| `main_shell_opt.py` | `optimize_v2()` | JAX optimization loop; monkey-patched onto `EquivalentSheetModel` |
| `ShellFemSolver/shell_solver.py` | `ShellFEM` | 6-DOF shell element (MITC3/MITC4); `assemble()`, `solve_static_partitioned()`, stress/strain computation |
| `solver.py` | `safe_eigh` | Custom VJP eigensolver; used only for modal gradient stability |
| `opt_targets.py` | `OptTarget`, `ResultBundle` | Target type definitions (FIELD_STAT, MASS, MODES, RBE_REACTION); `compute_error()` |
| `WHT_EQS_load_cases.py` | `TwistCase`, `PureBendingCase`, etc. | `get_bcs(fem)` returns `(fixed_dofs, fixed_vals, force_vector)` |
| `WHT_EQS_pattern_generator.py` | — | Stroke-based renderer generating thickness/topography/material fields from alphanumeric strings |
| `WHT_EQS_visualization.py` | — | PyVista 3-stage visualizer; controlled by `NON_INTERACTIVE` env var |
| `wh_utils.py` | `WHTable`, `wh_print_banner` | Terminal formatted output only |

### OptTarget Configuration Schema

`opt_target_config` is a dict keyed by case name plus a special `"global_targets"` key:
```python
{
    "case_name": {
        "opt_targets": [
            {"target_type": "field_stat", "field": "u_static",
             "reduction": "mse", "compare_mode": "relative", "weight": 2.0},
            {"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 1.0},
        ]
    },
    "global_targets": [
        {"target_type": "modes", "compare_mode": "mac",
         "num_modes": 3, "freq_weight": 0.5, "weight": 5.0}
    ]
}
```
`apply_case_targets_from_spec()` in `opt_targets.py` parses this and binds targets to cases. Global targets (mass, modes) must use the `"global_targets"` key — not `"global"`.

### JAX Constraints to Keep in Mind

- All FEM assembly and solve operations inside `loss_fn` must be JAX-traceable (no Python-side conditionals on JAX arrays).
- `case_needs` pruning uses **Python** booleans (evaluated before JIT trace) — this is intentional.
- `jax.debug.print` inside JIT runs on every trace iteration; remove diagnostic prints before production runs.
- Parameters are stored as `(Nx+1, Ny+1)` shaped arrays for local optimization. Global parameters are stored as `(1,)` scalars.
- `msvcrt.kbhit()` keyboard interrupt detection is Windows-only; guarded by the Windows-only import.