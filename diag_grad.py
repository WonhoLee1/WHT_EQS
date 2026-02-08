
import jax
import jax.numpy as jnp
from solver import PlateFEM

def test_grad():
    fem = PlateFEM(1.0, 0.5, 10, 5)
    num_nodes = (10+1)*(5+1)
    params = {
        't': jnp.ones(num_nodes),
        'rho': jnp.full(num_nodes, 7.85e-9),
        'E': jnp.full(num_nodes, 210000.0),
        'z': jnp.zeros(num_nodes)
    }
    
    def loss_fn(p):
        K, M = fem.assemble(p)
        # Solve a tiny static problem
        F = jnp.zeros(fem.total_dof).at[100].set(1.0) # Arbitrary node
        u = jax.scipy.linalg.solve(K + 1e-6*jnp.eye(fem.total_dof), F)
        return jnp.sum(u**2)

    try:
        val, grad = jax.value_and_grad(loss_fn)(params)
        print(f"Test Loss: {val}")
        print(f"Test Grad t sum: {jnp.sum(grad['t'])}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_grad()
