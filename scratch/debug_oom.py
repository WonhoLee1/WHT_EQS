import jax
import jax.numpy as jnp
from ShellFemSolver.shell_solver import ShellFEM

nodes = jnp.zeros((1421, 3))
quads = jnp.zeros((1344, 4), dtype=jnp.int32)
fem = ShellFEM(nodes, quads=quads)
params = {'E': 210000.0, 't': 1.0, 'rho': 7.85e-9}

print("Testing assembly...")
K, M = fem.assemble(params, sparse=True)
print(f"K shape: {K.shape}, type: {type(K)}")

print("Testing sparse slicing...")
free = jnp.arange(100, 8000)
try:
    Kff = K[free, :][:, free]
    print(f"Kff shape: {Kff.shape}")
except Exception as e:
    print(f"Error during sparse slicing: {e}")

print("Testing eigenvalue solve...")
# ...
