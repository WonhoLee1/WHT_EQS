
import jax
import jax.numpy as jnp
import os

try:
    print("JAX version:", jax.__version__)
    
    # Check devices
    print("Devices:", jax.devices())
    
    x = jnp.ones((10, 10))
    y = x @ x
    print("Matmul result shape:", y.shape)
    print("JAX basic test passed.")
except Exception as e:
    print("JAX failed:", e)
