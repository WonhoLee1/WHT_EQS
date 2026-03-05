import jax
jax.config.update('jax_enable_x64', True)
from ShellFemSolver.bdf_reader import run_eigen_from_bdf

freqs, fem = run_eigen_from_bdf(
    r'D:\PythonCodeStudy\oilpan_ERP_opti.fem',
    num_modes=10
)
