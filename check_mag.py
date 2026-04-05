import jax
import jax.numpy as jnp
from ShellFemSolver.shell_solver import compute_tria3_local, compute_q4_local

def check_magnitude():
    E = 210000.0
    t = 2.0
    nu = 0.3
    rho = 7.85e-9
    
    # 5x2.5 quad vs two 5x2.5 triangles
    # Q4: (-2.5, -1.25) to (2.5, 1.25) => a=2.5, b=1.25
    Kq, Mq = compute_q4_local(E, t, nu, rho, 2.5, 1.25)
    
    # T3: (0,0), (5,0), (0, 2.5)
    x = jnp.array([0.0, 5.0, 0.0])
    y = jnp.array([0.0, 0.0, 2.5])
    Kt, Mt = compute_tria3_local(E, t, nu, rho, x, y)
    
    # Compare max element in bending part
    q_bend = jnp.array([2,3,4, 8,9,10, 14,15,16, 20,21,22])
    Kq_bend = Kq[jnp.ix_(q_bend, q_bend)]
    
    t_bend = jnp.array([2,3,4, 8,9,10, 14,15,16])
    Kt_bend = Kt[jnp.ix_(t_bend, t_bend)]
    
    print(f"Q4 Max Bending K: {jnp.max(jnp.abs(Kq_bend)):.2f}")
    print(f"T3 Max Bending K: {jnp.max(jnp.abs(Kt_bend)):.2f}")
    print(f"Ratio (T3 / Q4): {jnp.max(jnp.abs(Kt_bend)) / jnp.max(jnp.abs(Kq_bend)):.4f}")
    
    area_t = 0.5 * 5.0 * 2.5
    print(f"T3 Area: {area_t}")
    
    D = (E * t**3) / (12 * (1 - nu**2))
    print(f"Bending Rigidity D: {D:.2f}")

    # For T3, Kt_bend is 9x9. Does it have any zero eigenvalues?
    vals = jnp.linalg.eigvalsh(Kt_bend)
    print(f"T3 Bending Eigenvals: {vals}")

if __name__ == "__main__":
    check_magnitude()
