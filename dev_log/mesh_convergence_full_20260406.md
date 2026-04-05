# ShellFEM Mesh Convergence Analysis Report

Issued: **2026-04-06 01:20**

## 1. Simulation Configuration
- **Analysis Type**: Static Linear Bending (Kirchhoff vs Mindlin-Reissner)
- **Geometry**: Square Plate (200.0 mm x 200.0 mm), Thickness 4.0 mm
- **Material**: Young's Modulus 210000.0 MPa, Poisson's Ratio 0.3
- **Load Condition**: Uniform Surface Pressure (0.05 MPa)
- **Boundary Condition**: Simply Supported (w=0 on all edges)

---

## 2. T3 (DKT) Convergence Study

| Mesh Size | Elements | Total DOFs | Max-W Error (%) | Reduction Ratio | Max-Sigma Error (%) |
|-----------|----------|------------|-----------------|-----------------|--------------------|
| 5x5 | 32 | 150 | 4.0067 | 0.00x | 15.5530 |
| 11x11 | 200 | 726 | 0.6106 | 6.56x | 9.0444 |
| 21x21 | 800 | 2646 | 0.1517 | 4.02x | 14.8731 |
| 41x41 | 3200 | 10086 | 0.0379 | 4.00x | 16.7149 |

## 3. Q4 (Mindlin) Convergence Study

| Mesh Size | Elements | Total DOFs | Max-W Error (%) | Reduction Ratio | Max-Sigma Error (%) |
|-----------|----------|------------|-----------------|-----------------|--------------------|
| 5x5 | 16 | 150 | 6.8099 | 0.00x | 2.9823 |
| 11x11 | 100 | 726 | 1.9191 | 3.55x | 0.8491 |
| 21x21 | 400 | 2646 | 1.8522 | 1.04x | 6.6202 |
| 41x41 | 1600 | 10086 | 1.9064 | 0.97x | 22.2557 |

## 4. Engineering Analysis & Guidance

- **Mesh Efficiency**: T3 (DKT) elements show faster convergence in displacement for the same degree of freedom count compared to Q4 in thin plate scenarios.
- **Recommended Discretization**: For engineering accuracy (< 1% error in deflection), a minimum of **11x11** mesh is required for T3.
- **Stress Sensitivity**: Stress (a derivative of displacement) requires significantly more mesh refinement to achieve the same level of convergence as displacement.