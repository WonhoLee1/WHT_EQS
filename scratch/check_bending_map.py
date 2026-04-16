import jax
import jax.numpy as jnp
import numpy as np

def check_mapping():
    # DOFs: [w, beta_x, beta_y]
    # convention: kappa_x = beta_{x,x}, kappa_y = beta_{y,y}, kappa_xy = beta_{x,y} + beta_{y,x}
    z = jnp.zeros(1)
    dN_dx, dN_dy = jnp.array([1.0]), jnp.array([2.0])
    
    # Old Row 0: jnp.stack([z, z, -dN_dx], axis=1).flatten() -> index 2
    # This was kappa_x = -beta_{y,x}. WRONG.
    
    # Correct Mapping:
    row0 = jnp.array([0.0, dN_dx[0], 0.0]) # kappa_x = beta_x,x
    row1 = jnp.array([0.0, 0.0, dN_dy[0]]) # kappa_y = beta_y,y
    row2 = jnp.array([0.0, dN_dy[0], dN_dx[0]]) # kappa_xy = beta_x,y + beta_y,x
    B = jnp.stack([row0, row1, row2])
    print("Correct B-bending (1 node, 3 DOFs):")
    print(B)

if __name__ == "__main__":
    check_mapping()
