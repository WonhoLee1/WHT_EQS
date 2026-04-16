import jax.numpy as jnp
import jax

def test_mean():
    # Test 1: Empty 2D array
    a = jnp.zeros((0, 3))
    try:
        m1 = a.mean(axis=1)
        print(f"Empty 2D mean(1) shape: {m1.shape}")
    except Exception as e:
        print(f"Empty 2D mean(1) failed: {e}")

    # Test 2: 1D array with axis=1
    b = jnp.zeros(5)
    try:
        m2 = b.mean(axis=1)
        print(f"1D mean(1) shape: {m2.shape}")
    except Exception as e:
        print(f"1D mean(1) failed: {e}")

if __name__ == "__main__":
    test_mean()
