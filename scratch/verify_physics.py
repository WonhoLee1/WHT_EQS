import jax
import jax.numpy as jnp
from ShellFemSolver.shell_solver import ShellFEM
import numpy as np

def compare():
    # Single Quad Element Test
    nodes = np.array([[0,0,0], [10,0,0], [10,10,0], [0,10,0]], dtype=float)
    elements = [[0,1,2,3]]
    fem = ShellFEM(nodes, elements)
    params = {'E': 210000.0, 't': 1.0, 'rho': 7.85e-9}
    
    K, M = fem.assemble(params)
    
    # Calculate Theoretical Mass
    # Area = 100. t = 1.0. rho = 7.85e-9.
    # Total Mass = 7.85e-7. Per node = 1.9625e-7.
    m_node = np.diag(M)[0]
    print(f"Calculated Mass per node: {m_node:.10e}")
    print(f"Theoretical Mass per node: {7.85e-9 * 100 * 1.0 / 4.0:.10e}")
    
    # Eigenvalues
    freqs, _ = fem.solve_eigen(K, M, num_modes=10, num_skip=0)
    print(f"First 10 frequencies (Hz): {freqs}")

if __name__ == "__main__":
    compare()
