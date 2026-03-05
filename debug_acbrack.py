import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from ShellFemSolver.bdf_reader import build_fem_from_bdf

fem, params, spc_dofs, num_modes = build_fem_from_bdf(r'D:\PythonCodeStudy\ACbrack.fem')
K, M = fem.assemble(params, sparse=True)

# Apply SPCs
fixed = sorted(set([ni * 6 + di for ni, di in spc_dofs]))
mask = np.ones(K.shape[0], dtype=bool)
mask[fixed] = False
free = np.where(mask)[0]

K_ff = K[free, :][:, free]
M_ff = M[free, :][:, free]

# Check diagonal
diag_k = K_ff.diagonal()
zero_k = np.where(diag_k == 0)[0]
print(f"Nodes with zero diagonal in K: {len(zero_k)}")
if len(zero_k) > 0:
    for idx in zero_k[:10]:
        global_idx = free[idx]
        node = global_idx // 6
        dof = global_idx % 6
        print(f"  Node {node} (Nastran ID?), DOF {dof+1}")

# Check M
diag_m = M_ff.diagonal()
zero_m = np.where(diag_m == 0)[0]
print(f"Nodes with zero diagonal in M: {len(zero_m)}")
