import jax
jax.config.update('jax_enable_x64', True)
from ShellFemSolver.bdf_reader import read_fem_file, run_eigen_from_bdf

print("=== Analyzing ACbrack.fem ===")
# mesh = read_fem_file(r'D:\PythonCodeStudy\ACbrack.fem')
# for k in mesh:
#     if isinstance(mesh[k], (list, dict)):
#         print(f"{k}: {len(mesh[k])}")
#     elif hasattr(mesh[k], 'shape'):
#         print(f"{k}: {mesh[k].shape}")

freqs, fem = run_eigen_from_bdf(
    r'D:\PythonCodeStudy\ACbrack.fem',
    num_modes=5
)
print("Results:", freqs)
