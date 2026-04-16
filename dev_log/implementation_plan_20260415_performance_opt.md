# Implementation Plan - Performance Optimization for Local Parameters

The optimization pipeline has slowed down significantly (~45s/iter) due to the increased complexity of the computation graph when using `type: local` for parameters and topography (`pz`). This plan aims to optimize the assembly process and JAX graph complexity to restore performance.

## User Review Required

> [!IMPORTANT]
> The slowdown is primarily caused by JAX differentiating through the coordinate transformation and assembly process for hundreds of unique local parameters. We propose several structural optimizations to speed this up.

## Proposed Changes

### [Component] ShellFemSolver (`shell_solver.py`)

#### [MODIFY] [shell_solver.py](file:///c:/Users/GOODMAN/code_sheet/ShellFemSolver/shell_solver.py)
- **Vectorized Assembly**: Refactor `assemble` to use pre-cached indices (`_cached_indices`) and a single `at[].add()` operation for all elements, reducing the number of nodes in the JAX graph.
- **Coordinate Transformation Optimization**: Cache the base coordinates and only apply the `pz` perturbation once, rather than re-indexing into the coordinate array multiple times.

### [Component] Optimization Loop (`main_shell_opt.py`)

#### [MODIFY] [main_shell_opt.py](file:///c:/Users/GOODMAN/code_sheet/main_shell_opt.py)
- **Modal Track Sparsity**: Increase the interval for the heavy `with_eigen` track (e.g., from 5 to 10 iterations) and rely more on the efficient `no_eigen` (Rayleigh Quotient) track.
- **Reporting Sync Optimization**: Use `jax.tree_util.tree_map(lambda x: x.block_until_ready(), ...)` to manage synchronization points more efficiently.

## Open Questions
- 사용자가 체감하는 "멈춤" 현상이 **컴파일 시간**입니까, 아니면 **반복당 실행 속도**입니까? (제 로그상으로는 반복당 약 40~50초가 소요되고 있습니다.)
- 고유진동수 해석의 정확도가 매 프레임 중요합니까? 아니면 10회당 1회 정도로 업데이트 주기를 늘려도 보상(Rayleigh Quotient)으로 충분합니까?

## Verification Plan

### Automated Tests
- Run `python main_shell_opt.py` and measure the time per iteration.
- Target: Reduce iteration time to < 10s for the same resolution.
