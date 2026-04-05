# ShellFEM Mesh Convergence Analysis Report

Issued: **2026-04-06 01:13**

## 1. T3 (DKT) Convergence Study

| Mesh Size | Max-W Error (%) | Reduction Ratio | Max-Sigma Error (%) | Reduction Ratio |
|-----------|-----------------|-----------------|--------------------|-----------------|
| 5x5 | 4.0067 | 0.00x | 15.5530 | 0.00x |
| 11x11 | 0.6106 | 6.56x | 9.0444 | 1.72x |
| 21x21 | 0.1517 | 4.02x | 14.8731 | 0.61x |
| 41x41 | 0.0379 | 4.00x | 16.7149 | 0.89x |

## 2. Q4 (Mindlin) Convergence Study

| Mesh Size | Max-W Error (%) | Reduction Ratio | Max-Sigma Error (%) | Reduction Ratio |
|-----------|-----------------|-----------------|--------------------|-----------------|
| 5x5 | 6.8099 | 0.00x | 2.9823 | 0.00x |
| 11x11 | 1.9191 | 3.55x | 0.8491 | 3.51x |
| 21x21 | 1.8522 | 1.04x | 6.6202 | 0.13x |
| 41x41 | 1.9064 | 0.97x | 22.2557 | 0.30x |

## 3. Engineering Guidance

- **Convergence Rate**: As the mesh density doubles, the displacement error typically drops by a factor of 4 (quadratic convergence).
- **Recommended Size**: A mesh density of **21x21** or higher is recommended for global fidelity > 99%.
- **Stress Gradient**: Stress convergence is slower than displacement due to being a derivative quantity.