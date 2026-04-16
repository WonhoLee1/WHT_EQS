
import os
import sys
import numpy as np
import jax.numpy as jnp
from ShellFemSolver.shell_solver import ShellFEM
from ShellFemSolver.mesh_utils import generate_tray_mesh_quads


def diagnose():
    Lx, Ly = 1450.0, 850.0
    nx, ny = 24, 14
    
    params_base = {
        't': 1.0,
        'E': 210000.0,
        'rho': 7.85e-9,
    }
    
    # 1. Vertical Walls (50mm)
    print("\n[DIAGNOSIS] Case A: Vertical Walls (50mm)")
    n_v, q_v = generate_tray_mesh_quads(Lx, Ly, wall_width=50.0, wall_height=50.0, nx=nx, ny=ny, mode='vertical')
    fem_v = ShellFEM(n_v, quads=q_v)
    K_v, M_v = fem_v.assemble(params_base, sparse=True)
    vals_v, _ = fem_v.solve_eigen_sparse(K_v, M_v, num_modes=10)
    print(f" -> Frequencies (Vertical): {np.array(vals_v)[6:10]}")
    
    # 2. Sloped Walls (50mm)
    print("\n[DIAGNOSIS] Case B: Sloped Walls (50mm)")
    n_s, q_s = generate_tray_mesh_quads(Lx, Ly, wall_width=50.0, wall_height=50.0, nx=nx, ny=ny, mode='sloped')
    fem_s = ShellFEM(n_s, quads=q_s)
    K_s, M_s = fem_s.assemble(params_base, sparse=True)
    vals_s, _ = fem_s.solve_eigen_sparse(K_s, M_s, num_modes=10)
    print(f" -> Frequencies (Sloped): {np.array(vals_s)[6:10]}")

    # 3. High Resolution (120x60) - Sloped
    print("\n[DIAGNOSIS] Case C: High Res Sloped (120x60)")
    n_h, q_h = generate_tray_mesh_quads(Lx, Ly, wall_width=50.0, wall_height=50.0, nx=120, ny=60, mode='sloped')
    fem_h = ShellFEM(n_h, quads=q_h)
    K_h, M_h = fem_h.assemble(params_base, sparse=True)
    vals_h, _ = fem_h.solve_eigen_sparse(K_h, M_h, num_modes=10)
    print(f" -> Frequencies (High-Res Sloped): {np.array(vals_h)[6:10]}")


if __name__ == "__main__":
    diagnose()
