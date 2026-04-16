import jax
import jax.numpy as jnp
from ShellFemSolver.shell_solver import ShellFEM
import numpy as np

def test():
    nodes = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype=float)
    elements = [[0,1,2,3]]
    fem = ShellFEM(nodes, elements)
    params = {'E': 210000.0, 't': 1.0, 'rho': 7.85e-9}
    K, M = fem.assemble(params)
    print(f"K shape: {K.shape}")
    print(f"K mean: {jnp.mean(K)}")
    
    # Test gradient
    def loss(t_val):
        K, _ = fem.assemble({'t': t_val})
        return jnp.sum(K)
    
    grad_fn = jax.grad(loss)
    g = grad_fn(1.0)
    print(f"Grad of sum(K) w.r.t t: {g}")

if __name__ == "__main__":
    test()
