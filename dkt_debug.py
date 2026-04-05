import jax
import jax.numpy as jnp
from ShellFemSolver.shell_solver import compute_tria3_local

def debug_dkt():
    # 1. Single Element Props
    E = 210000.0
    t = 2.0
    nu = 0.3
    rho = 7.85e-9
    
    # Square element split diagonally: (0,0), (10,0), (0,10)
    # L = 10. Area = 50.
    x2d = jnp.array([0.0, 10.0, 0.0])
    y2d = jnp.array([0.0, 0.0, 10.0])
    
    K, M = compute_tria3_local(E, t, nu, rho, x2d, y2d)
    
    # 2. Extract [w1, phix1, phiy1, w2, phix2, phiy2, w3, phix3, phiy3]
    # In compute_tria3_local, bend_idx = [2,3,4, 8,9,10, 14,15,16]
    bend_idx = [2,3,4, 8,9,10, 14,15,16]
    Kb = K[jnp.ix_(bend_idx, bend_idx)]
    
    # 3. Analytical Reference (D)
    D_val = (E * t**3) / (12 * (1 - nu**2))
    print(f"Bending Rigidity (D): {D_val:.4f}")
    
    # 4. Check K magnitude
    # For a simple beam: K_ww ~ 12EI/L^3. 
    # For a plate element, K_rotation ~ D
    K_rot_avg = jnp.mean(jnp.abs(Kb[1::3, 1::3])) # Avg rotation stiffness
    print(f"FEM Avg Rotation Stiffness: {K_rot_avg:.4f}")
    print(f"Ratio (FEM / D): {K_rot_avg / D_val:.4f}")
    
    # 5. Check if K is singular (excluding rigid body)
    vals = jnp.linalg.eigvalsh(Kb)
    print(f"Bending Eigenvalues: {vals}")
    # 3 rigid body modes (1 translation, 2 rotations) should be near 0
    
if __name__ == "__main__":
    debug_dkt()
