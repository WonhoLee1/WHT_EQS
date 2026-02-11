from main_verification import run_matching_example
import jax

# Force CPU
jax.config.update('jax_platform_name', 'cpu')

if __name__ == '__main__':
    run_matching_example()
