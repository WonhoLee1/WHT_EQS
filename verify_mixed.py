import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from ShellFemSolver.shell_solver import ShellFEM
from ShellFemSolver.mesh_utils import generate_rect_mesh_quads, generate_rect_mesh_triangles

Lx, Ly = 1000.0, 400.0
E, t, nu, rho = 210000.0, 1.0, 0.3, 7.85e-9

def run_fem(fem, nodes, label):
    params = {'t': jnp.full(len(nodes), t), 'rho': jnp.full(len(nodes), rho), 'E': jnp.full(len(nodes), E)}
    K, M = fem.assemble(params)
    edge = [i for i,c in enumerate(nodes) if c[0]==0 or c[0]==Lx or c[1]==0 or c[1]==Ly]
    fixed = jnp.array(sorted([n*6+2 for n in edge]))
    mask = jnp.ones(fem.total_dof, dtype=bool).at[fixed].set(False)
    free = jnp.where(mask)[0]
    vals,_ = fem.solve_eigen(K[jnp.ix_(free,free)], M[jnp.ix_(free,free)], 10)
    freqs = jnp.sqrt(vals)/(2*jnp.pi)
    print(f'{label}: Hz = {[round(float(v),4) for v in freqs[:5]]}')

nodes_q, quads = generate_rect_mesh_quads(Lx, Ly, 25, 10)
fem_q = ShellFEM(nodes_q, quads=quads)
run_fem(fem_q, nodes_q, 'CQUAD4')

nodes_t, trias = generate_rect_mesh_triangles(Lx, Ly, 25, 10)
fem_t = ShellFEM(nodes_t, trias=trias)
run_fem(fem_t, nodes_t, 'CTRIA3 DKT+DSG3')

D = E*t**3/(12*(1-nu**2))
f11 = (3.14159265/2)*((D/(rho*t))**0.5)*((1/Lx)**2+(1/Ly)**2)
print(f'Theory (1,1): {f11:.4f} Hz')
