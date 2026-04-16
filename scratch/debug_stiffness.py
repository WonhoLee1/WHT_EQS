import jax
import jax.numpy as jnp
from ShellFemSolver.shell_solver import ShellFEM
import numpy as np

def debug_stiffness():
    # 10x10x1 plate. E=2.1e5, rho=7.85e-9
    L, t, E, rho = 1000.0, 10.0, 210000.0, 7.85e-9
    nodes = np.array([[0,0,0], [L,0,0], [L,L,0], [0,L,0]], dtype=float)
    elements = [[0,1,2,3]]
    fem = ShellFEM(nodes, elements)
    params = {'E': E, 't': t, 'rho': rho}
    
    K, M = fem.assemble(params)
    
    # 1. Check volume/mass integration
    theoretical_mass = L * L * t * rho
    fem_mass = jnp.sum(M[::6, ::6]) # Sum mass in x-translation
    print(f"Theory Mass: {theoretical_mass:.5e}")
    print(f"FEM Mass: {fem_mass:.5e}")
    
    # 2. Check Static Deflection of single element as cantilever
    # cantilever (1m x 1m x 0.01m). Load P=100 at end.
    # Fixed at nodes 0, 3. Load at 1, 2.
    fixed = [0*6+i for i in range(6)] + [3*6+i for i in range(6)]
    F = np.zeros(24); F[1*6+2] = -50; F[2*6+2] = -50
    free = np.setdiff1d(np.arange(24), fixed)
    
    u_vec = np.zeros(24)
    u_free = np.linalg.solve(K[free,:][:,free], F[free])
    u_vec[free] = u_free
    
    fe_w = -np.mean(u_vec[[1*6+2, 2*6+2]])
    # Beam theory: w = PL^3 / (3EI). I = w*t^3/12. (here w=L)
    I = L * t**3 / 12.0
    th_w = (100 * L**3) / (3 * E * I)
    print(f"Theory Cantilever Deflection: {th_w:.5e}")
    print(f"FEM Cantilever Deflection: {fe_w:.5e}")
    print(f"Deflection Ratio (FE/TH): {fe_w/th_w:.5f}")

if __name__ == "__main__":
    debug_stiffness()
