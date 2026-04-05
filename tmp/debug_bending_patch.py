import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, r'C:\Users\GOODMAN\code_sheet\ShellFemSolver')
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from shell_solver import ShellFEM
sys.path.insert(0, r'C:\Users\GOODMAN\code_sheet')
from ShellFemSolverVerification.patch_tests import (
    make_tria_mesh, _node_ids_on_edge, _build_params, _solve_static, recover_bending_curvature_tria
)

E=210000.0; t=5.0; nu=0.3; M0=1000.0; nx=4; ny=4
Lx=100.0; Ly=60.0
nodes, trias = make_tria_mesh(Lx, Ly, nx, ny)
D = E*t**3/(12.0*(1.0-nu**2))
k = M0/D
print(f'D={D:.4f}  kappa_th={k:.6e}')

bnd = set()
for ax, val in [(0,0.),(0,Lx),(1,0.),(1,Ly)]:
    bnd.update(_node_ids_on_edge(nodes, ax, val).tolist())

fd, fv = [], []
for i in bnd:
    x, y = nodes[i,0], nodes[i,1]
    fd += [i*6+2, i*6+3, i*6+4, i*6+0, i*6+1, i*6+5]
    fv += [0.5*k*x**2, 0., -k*x, 0., 0., 0.]

total = len(nodes)*6
fd = np.array(fd); fv = np.array(fv)
fr = np.setdiff1d(np.arange(total), fd)
print(f'Free DOFs: {len(fr)}  |  boundary nodes: {len(bnd)}')

params = _build_params(len(nodes), E, t)
fem = ShellFEM(nodes, trias=trias)
F = np.zeros(total)
u = _solve_static(fem, params, F, fr, fd, fv)

interior = np.setdiff1d(np.arange(len(nodes)), list(bnd))
print(f'Interior nodes: {len(interior)}')
for i in interior[:5]:
    x = nodes[i,0]
    wf = u[i*6+2]; wt = 0.5*k*x**2
    tyf = u[i*6+4]; tyt = -k*x
    print(f'  n{i:3d} x={x:6.1f}: w_fem={wf:+.4e} w_th={wt:+.4e}  ty_fem={tyf:+.4e} ty_th={tyt:+.4e}')

# curvature recovery
kappa = np.array(recover_bending_curvature_tria(jnp.array(u), jnp.array(nodes), jnp.array(trias)))
print(f'kappa: kx_mean={np.mean(kappa[:,0]):.4e}  kx_std={np.std(kappa[:,0]):.4e}')
print(f'       ky_mean={np.mean(kappa[:,1]):.4e}')
print(f'  (theory kx={k:.4e})')
