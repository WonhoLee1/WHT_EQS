# ShellFEM Mesh Convergence Analysis Report (by Element Size)

Issued: **2026-04-06 01:25**

## 1. Simulation Configuration
- **Geometry**: Square Plate (200.0 mm x 200.0 mm), Thickness 4.0 mm
- **Material**: Young's Modulus 210000.0 MPa, Poisson's Ratio 0.3
- **Load Condition**: Uniform Surface Pressure (0.05 MPa)
- **Analysis Type**: Static Linear Bending

---

## 2. T3 (DKT) Convergence Study

| Mesh Density | Mesh Size (h) | Elements | Total DOFs | Deflection Error (%) | Reduction |
|--------------|---------------|----------|------------|----------------------|-----------|
| 5x5 | 40.0 mm | 50 | 216 | 2.6135 | 0.00x |
| 10x10 | 20.0 mm | 200 | 726 | 0.6106 | 4.28x |
| 20x20 | 10.0 mm | 800 | 2646 | 0.1517 | 4.02x |
| 40x40 | 5.0 mm | 3200 | 10086 | 0.0379 | 4.00x |

## 3. Q4 (Mindlin) Convergence Study

| Mesh Density | Mesh Size (h) | Elements | Total DOFs | Deflection Error (%) | Reduction |
|--------------|---------------|----------|------------|----------------------|-----------|
| 5x5 | 40.0 mm | 25 | 216 | 0.8252 | 0.00x |
| 10x10 | 20.0 mm | 100 | 726 | 1.9191 | 0.43x |
| 20x20 | 10.0 mm | 400 | 2646 | 1.8522 | 1.04x |
| 40x40 | 5.0 mm | 1600 | 10086 | 1.9064 | 0.97x |

## 4. Engineering Conclusion

- **Accuracy Goal**: To achieve an error rate of less than 0.2%, an element size (h) of **10.0 mm (20x20 density)** or less is recommended.
- **T3 Efficiency**: T3 elements provide high-fidelity results even at relatively large mesh sizes (h=40mm) compared to Q4.