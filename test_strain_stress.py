import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"

import jax
import jax.numpy as jnp
from solver import PlateFEM

# Create simple FEM
fem = PlateFEM(100, 40, 10, 5)

# Create simple params
params = {
    't': jnp.full((11, 6), 2.0),
    'rho': jnp.full((11, 6), 7.5e-9),
    'E': jnp.full((11, 6), 200000.0)
}

# Assemble
K, M = fem.assemble(params)

# Create simple displacement
u = jnp.zeros(fem.total_dof)
u = u.at[::3].set(jnp.linspace(0, 1, fem.num_nodes))  # Set some z displacement

print(f"u shape: {u.shape}")
print(f"num_nodes: {fem.num_nodes}")

# Test compute_curvature
try:
    curv = fem.compute_curvature(u)
    print(f"✓ Curvature computed successfully: shape={curv.shape}")
except Exception as e:
    print(f"✗ Curvature failed: {e}")
    import traceback
    traceback.print_exc()

# Test compute_moment
try:
    mom = fem.compute_moment(u, params)
    print(f"✓ Moment computed successfully: shape={mom.shape}")
except Exception as e:
    print(f"✗ Moment failed: {e}")
    import traceback
    traceback.print_exc()
