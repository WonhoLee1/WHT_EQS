# Equivalent Sheet Model Optimization & Verification

A JAX-based differentiable FEM framework for optimizing equivalent sheet properties of complex bead-welded structures.

## 🚀 Overview
This repository provides a modular pipeline to:
1.  **Generate Ground Truth**: Create high-resolution target responses for complex bead patterns (thickness and topography).
2.  **Interactive Visualization**: Inspect patterns and FEM results using an interactive PyVista-based 3D visualizer at multiple stages.
3.  **Optimize**: Use gradient-based optimization (Adam via Optax) to find equivalent uniform or varying sheet properties (Thickness, Density, Young's Modulus) that match the complex targets.
4.  **Verify**: Compare optimized results against targets using side-by-side 3D visualization.

## 📦 Project Structure
- `main_verification.py`: The main entry point orchestrating the workflow.
- `solver.py`: 12-DOF Bending Plate FEM solver (differentiable with JAX).
- `WHT_EQS_pattern_generator.py`: Stroke-based font rendering for bead patterns.
- `WHT_EQS_load_cases.py`: Boundary condition and force definitions for various mechanical tests (Twist, Bending, Corner Lift).
- `WHT_EQS_visualization.py`: 3D interactive visualization tools using PyVista.

## 🛠 Features
- **Alphanumeric Bead Patterns**: Supports strings like "ABC", "TNY" with customizable thickness/height.
- **Topography Support**: Optimizes and visualizes both thickness-based and topography-based reinforcements.
- **Side-by-side Comparison**: Synchronized camera views for target vs. optimized assessment.

## 🚦 Getting Started
Ensure you have the following dependencies installed:
```bash
pip install jax jaxlib numpy matplotlib pyvista optax scipy
```

Run the main verification script:
```bash
python main_verification.py
```

## 📄 License
MIT License
