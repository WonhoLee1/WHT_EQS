import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add relevant paths
sys.path.append('c:/Users/GOODMAN/code_sheet')
from ShellFemSolver.shell_solver import ShellFEM

def test_debug():
    # 1. Setup a dummy mesh
    nodes = jnp.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype=jnp.float32)
    quads = jnp.array([[0,1,2,3]], dtype=jnp.int32)
    fem = ShellFEM(nodes, quads)
    
    # 2. Setup tracers like in the real loss_fn
    @jax.jit
    def loss_fn(p_scaled, scales):
        combined = {k: v * scales[k] for k, v in p_scaled.items()}
        # Simulate broadcasting
        p_fem = {k: combined[k] if combined[k].ndim > 1 else jnp.full((4,), combined[k][0]) for k in combined}
        K, M = fem.assemble(p_fem)
        return jnp.sum(K)

    # 3. Simulate inputs
    p_scaled = {'E': jnp.array([1.0]), 't': jnp.array([1.0]), 'rho': jnp.array([1.0])}
    scales = {'E': 210000.0, 't': 1.0, 'rho': 7.85e-9}
    
    try:
        print("Tracing loss_fn...")
        res = loss_fn(p_scaled, scales)
        print("Success! Result:", res)
    except Exception as e:
        print("Caught Exception:", type(e).__name__)
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug()
