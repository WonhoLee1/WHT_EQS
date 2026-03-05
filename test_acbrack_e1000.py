import jax
jax.config.update('jax_enable_x64', True)
from ShellFemSolver.bdf_reader import run_eigen_from_bdf

print("=== Analyzing ACbrack.fem with E=1000 MPa Override ===")
freqs, fem = run_eigen_from_bdf(
    r'D:\PythonCodeStudy\ACbrack.fem',
    num_modes=5,
    E=1000.0 # Forced Override to test hypothesis
)
print("Results:", freqs)
