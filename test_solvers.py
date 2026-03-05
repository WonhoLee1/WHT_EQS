import jax
import jax.numpy as jnp
import numpy as np
import json

from solver import PlateFEM
from ShellFemSolver.shell_solver import ShellFEM
from ShellFemSolver.mesh_utils import generate_rect_mesh_quads

Lx, Ly = 1000.0, 400.0
nx, ny = 25, 10
E, t, nu, rho = 210000.0, 1.0, 0.3, 7.85e-9

out = {}

def test_PlateFEM():
    fem = PlateFEM(Lx, Ly, nx, ny)
    params = {
        't': jnp.full((nx+1, ny+1), t),
        'rho': jnp.full((nx+1, ny+1), rho),
        'E': jnp.full((nx+1, ny+1), E),
    }
    K, M = fem.assemble(params)

    edge_nodes = set()
    for i in range(nx + 1):
        for j in range(ny + 1):
            if i == 0 or i == nx or j == 0 or j == ny:
                edge_nodes.add(i * (ny + 1) + j)
    
    fixed_dofs = []
    for n in edge_nodes:
        fixed_dofs.extend([n*6+0, n*6+1, n*6+2])

    fixed_dofs = jnp.array(list(sorted(fixed_dofs)))
    mask = jnp.ones(fem.total_dof, dtype=bool)
    mask = mask.at[fixed_dofs].set(False)
    free_dofs = jnp.where(mask)[0]
    
    K_ff = K[jnp.ix_(free_dofs, free_dofs)]
    M_ff = M[jnp.ix_(free_dofs, free_dofs)]
    
    vals, vecs = fem.solve_eigen(K_ff, M_ff, num_modes=10)
    freqs = jnp.sqrt(jnp.maximum(vals, 0.0)) / (2 * jnp.pi)
    out['PlateFEM'] = np.array(freqs[:5]).tolist()
    
def test_ShellFEM():
    nodes, elements = generate_rect_mesh_quads(Lx, Ly, nx, ny)
    fem = ShellFEM(nodes, elements)
    params = {
        't': jnp.full(len(nodes), t),
        'rho': jnp.full(len(nodes), rho),
        'E': jnp.full(len(nodes), E),
    }
    K, M = fem.assemble(params)

    edge_nodes = set()
    for i, c in enumerate(nodes):
        if c[0] == 0 or c[0] == Lx or c[1] == 0 or c[1] == Ly:
            edge_nodes.add(i)

    fixed_dofs = []
    for n in edge_nodes:
        fixed_dofs.extend([n*6+0, n*6+1, n*6+2])

    fixed_dofs = jnp.array(list(sorted(fixed_dofs)))
    mask = jnp.ones(fem.total_dof, dtype=bool)
    mask = mask.at[fixed_dofs].set(False)
    free_dofs = jnp.where(mask)[0]
    
    K_ff = K[jnp.ix_(free_dofs, free_dofs)]
    M_ff = M[jnp.ix_(free_dofs, free_dofs)]
    
    vals, vecs = fem.solve_eigen(K_ff, M_ff, num_modes=10)
    freqs = jnp.sqrt(jnp.maximum(vals, 0.0)) / (2 * jnp.pi)
    out['ShellFEM'] = np.array(freqs[:5]).tolist()

if __name__ == '__main__':
    test_PlateFEM()
    test_ShellFEM()
    with open('test_results.json', 'w') as f:
        json.dump(out, f, indent=4)
