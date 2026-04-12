import jax
import jax.numpy as jnp
import numpy as np
from ShellFemSolver.shell_solver import ShellFEM
from ShellFemSolver.mesh_utils import generate_rect_mesh_quads

def test():
    Lx, Ly = 1000.0, 400.0
    nx, ny = 10, 4
    nodes, elements = generate_rect_mesh_quads(Lx, Ly, nx, ny)
    fem = ShellFEM(nodes, elements)
    
    params = {
        't': jnp.ones(len(elements)),
        'rho': jnp.full(len(elements), 7.85e-9),
        'nu': 0.3,
        'E': jnp.full(len(elements), 210000.0)
    }
    
    print("Assembling...")
    K, M = fem.assemble(params, sparse=True)
    
    print("Solving Sparse Eigen (sigma=None)...")
    freqs_s, _ = fem.solve_eigen_sparse(K, M, num_modes=10, sigma=None)
    print("Sparse Freqs:", freqs_s)
    
    print("Assembling Dense...")
    Kd, Md = fem.assemble(params, sparse=False)
    print("Solving Dense Eigen...")
    freqs_d, _ = fem.solve_eigen(Kd, Md, num_modes=10, num_skip=0)
    print("Dense Freqs:", freqs_d[:10])

if __name__ == "__main__":
    test()
