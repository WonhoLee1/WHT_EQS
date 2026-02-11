from unittest.mock import patch
import builtins
import jax
import sys

# Mock input to return '0' automatically
def mock_input(prompt=None):
    if prompt:
        print(prompt)
    print(">> AUTOMATED INPUT: 0")
    return "0"

builtins.input = mock_input

# Force CPU
jax.config.update('jax_platform_name', 'cpu')

from main_verification import run_matching_example

if __name__ == '__main__':
    run_matching_example()
