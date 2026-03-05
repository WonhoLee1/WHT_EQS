"""
==============================================================================
 [BENCHMARK] Flat Plate Natural Frequency Comparison
 - Analytical (Simply-Supported Plate Theory)
 - PlateFEM   (Existing Solver - solver.py)
 - ShellFEM   (New Triangle Solver - shell_solver.py)
 
 Boundary: Simply-Supported (SSSS) on all 4 edges
 Geometry: 1000 x 400 mm, t = 5 mm
 Material: Steel (E=210000 MPa, nu=0.3, rho=7.85e-9 tonne/mm3)
==============================================================================
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

# ========================
# 1. PARAMETERS
# ========================
Lx, Ly = 1000.0, 400.0
t = 5.0
E = 210000.0 
nu = 0.3
rho = 7.85e-9

# ========================
# 2. ANALYTICAL SOLUTION (Simply-Supported Rectangular Plate)
# ========================
# f_mn = (pi/2) * [(m/a)^2 + (n/b)^2] * sqrt(D / (rho * t))
# D = E * t^3 / (12 * (1 - nu^2))

D = E * t**3 / (12.0 * (1.0 - nu**2))
print(f"Flexural Rigidity D = {D:.2f} N·mm")
print(f"Mass per unit area  = {rho * t:.6e} tonne/mm²")
print(f"sqrt(D/(rho*t))     = {np.sqrt(D / (rho * t)):.2f} mm²/s")

print("\n" + "="*70)
print(" ANALYTICAL: Simply-Supported Plate Natural Frequencies (SSSS)")
print("="*70)
analytical_freqs = []
for m in range(1, 5):
    for n in range(1, 4):
        f_mn = (np.pi / 2.0) * ((m / Lx)**2 + (n / Ly)**2) * np.sqrt(D / (rho * t))
        analytical_freqs.append((m, n, f_mn))
        
analytical_freqs.sort(key=lambda x: x[2])
print(f"{'Mode':>8s} | {'Freq (Hz)':>10s}")
print("-" * 25)
for m, n, f in analytical_freqs[:10]:
    print(f"  ({m},{n})   | {f:10.2f}")

# ========================
# 3. PlateFEM (Existing Solver)
# ========================
print("\n" + "="*70)
print(" PlateFEM (solver.py): Simply-Supported Eigenvalue Test")
print("="*70)

from solver import PlateFEM as OriginalPlateFEM

for nx, ny in [(25, 10), (50, 20)]:
    fem_p = OriginalPlateFEM(Lx, Ly, nx, ny)
    params_p = {
        't': jnp.full((nx+1, ny+1), t),
        'E': jnp.full((nx+1, ny+1), E),
        'rho': jnp.full((nx+1, ny+1), rho),
        'z': jnp.zeros((nx+1, ny+1))
    }
    K_p, M_p = fem_p.assemble(params_p)
    
    # Simply-Supported BC: Fix w=0 on all 4 edges (DOF index 2 for w in 6-DOF)
    tol = 1e-3
    edge_nodes = np.where(
        (fem_p.node_coords[:, 0] < tol) | (fem_p.node_coords[:, 0] > Lx - tol) |
        (fem_p.node_coords[:, 1] < tol) | (fem_p.node_coords[:, 1] > Ly - tol)
    )[0]
    
    fixed_dofs = []
    fixed_vals = []
    for node in edge_nodes:
        fixed_dofs.append(int(node * 6 + 2))  # w = 0
        fixed_vals.append(0.0)
    # Add rigid body constraints (prevent u,v,tz drift)
    fixed_dofs.extend([0, 1, 5])  # u, v, tz at node 0
    fixed_vals.extend([0.0, 0.0, 0.0])
    
    fixed_dofs = jnp.array(fixed_dofs)
    fixed_vals = jnp.array(fixed_vals)
    free_dofs = jnp.setdiff1d(jnp.arange(fem_p.total_dof), fixed_dofs)
    
    vals_p, vecs_p = fem_p.solve_eigen(K_p, M_p, num_modes=15)
    freqs_p = np.sqrt(np.maximum(np.array(vals_p), 0.0)) / (2 * np.pi)
    
    # Filter elastic modes
    elastic_p = freqs_p[freqs_p > 1.0]
    
    print(f"\n  Mesh: {nx}x{ny} ({fem_p.num_nodes} nodes)")
    print(f"  {'Mode':>8s} | {'Freq (Hz)':>10s}")
    print("  " + "-" * 25)
    for i, f in enumerate(elastic_p[:5]):
        print(f"    {i+1:>4d}   | {f:10.2f}")

# ========================
# 4. ShellFEM (New Solver)
# ========================
print("\n" + "="*70)
print(" ShellFEM (shell_solver.py): Simply-Supported Eigenvalue Test")
print("="*70)

from shell_solver import ShellFEM
from mesh_utils import generate_rect_mesh_triangles

for nx, ny in [(25, 10), (50, 20)]:
    nodes, elements = generate_rect_mesh_triangles(Lx, Ly, nx, ny)
    fem_s = ShellFEM(nodes, elements)
    
    params_s = {
        't': jnp.full(fem_s.num_nodes, t),
        'E': jnp.full(fem_s.num_nodes, E),
        'rho': jnp.full(fem_s.num_nodes, rho),
    }
    K_s, M_s = fem_s.assemble(params_s)
    
    # Simply-Supported BC
    tol = 1e-3
    edge_nodes_s = np.where(
        (np.array(fem_s.node_coords[:, 0]) < tol) | (np.array(fem_s.node_coords[:, 0]) > Lx - tol) |
        (np.array(fem_s.node_coords[:, 1]) < tol) | (np.array(fem_s.node_coords[:, 1]) > Ly - tol)
    )[0]

    fixed_dofs_s = []
    fixed_vals_s = []
    for node in edge_nodes_s:
        fixed_dofs_s.append(int(node * 6 + 2))  # w = 0
        fixed_vals_s.append(0.0)
    fixed_dofs_s.extend([0, 1, 5])
    fixed_vals_s.extend([0.0, 0.0, 0.0])
    
    fixed_dofs_s = jnp.array(fixed_dofs_s)
    fixed_vals_s = jnp.array(fixed_vals_s)
    
    vals_s, vecs_s = fem_s.solve_eigen(K_s, M_s, num_modes=15)
    freqs_s = np.sqrt(np.maximum(np.array(vals_s), 0.0)) / (2 * np.pi)
    
    elastic_s = freqs_s[freqs_s > 1.0]
    
    print(f"\n  Mesh: {nx}x{ny} ({fem_s.num_nodes} nodes, {fem_s.num_elements} triangles)")
    print(f"  {'Mode':>8s} | {'Freq (Hz)':>10s}")
    print("  " + "-" * 25)
    for i, f in enumerate(elastic_s[:5]):
        print(f"    {i+1:>4d}   | {f:10.2f}")

# ========================
# 5. SUMMARY TABLE
# ========================
print("\n" + "="*70)
print(" CROSS-COMPARISON SUMMARY (First 3 Elastic Modes)")
print("="*70)
print(f"{'Source':>20s} | {'Mode 1 (Hz)':>12s} | {'Mode 2 (Hz)':>12s} | {'Mode 3 (Hz)':>12s}")
print("-" * 70)
ana = [x[2] for x in analytical_freqs[:3]]
print(f"{'Analytical (SSSS)':>20s} | {ana[0]:>12.2f} | {ana[1]:>12.2f} | {ana[2]:>12.2f}")
