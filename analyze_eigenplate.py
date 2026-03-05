import jax
jax.config.update('jax_enable_x64', True)
from ShellFemSolver.bdf_reader import run_eigen_from_bdf

print("=== Analyzing D:\\PythonCodeStudy\\eigenplate.fem ===")
# Let's run with 5 modes.
freqs, fem = run_eigen_from_bdf(
    r'D:\PythonCodeStudy\eigenplate.fem',
    num_modes=5
)
