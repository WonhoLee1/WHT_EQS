# Optimized Shell FEM Solver & Tray Geometry Integration (2026-04-08)

This plan outlines the stabilization of the JAX Shell FEM solver and its integration into the 3D Tray verification pipeline.

## 1. Solver Stabilization
- Implement `compute_field_results` for nodal-averaged outputs.
- Fix shape mismatches in stress/strain fields.
- Hybrid solver: Sparse Scipy (GT) + Dense JAX (Opt).

## 2. Tray Geometry Integration
- Restore `generate_tray_mesh_quads`.
- Configure 1450x850x50mm tray dimensions.
- Map high-res GT load cases to low-res optimized models.

## 3. Modal Analysis
- Skip first 6 rigid body modes.
- Track top 5 structural frequencies.
- Implement MAC-based mode matching.

## 4. Optimization Setup
- Bridge JAX/Scipy data with explicit NumPy casting.
- Use actual nodal coordinates for interpolation.
- Finalize differentiable loss function closure.
