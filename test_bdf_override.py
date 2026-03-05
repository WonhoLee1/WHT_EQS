import jax
jax.config.update('jax_enable_x64', True)
from ShellFemSolver.bdf_reader import run_eigen_from_bdf

print("=== Overriding Parameters: t=2.0mm, E=70000MPa (Aluminum-like) ===")
freqs, fem = run_eigen_from_bdf(
    r'D:\PythonCodeStudy\oilpan_ERP_opti.fem',
    num_modes=5,
    t=2.0,      # Override from 6.0 -> 2.0
    E=70000.0,  # Override from 210000 -> 70000
    rho=2.7e-9, # Aluminum rho
    nu=0.33
)
