import sys
import os
import jax.numpy as jnp
import numpy as np

# Add workspace to path
sys.path.append(r'c:\Users\GOODMAN\code_sheet')
from ShellFemSolver.shell_solver import ShellFEM
from ShellFemSolverVerification.patch_tests import PatchTests

def debug_t3():
    tester = PatchTests()
    # Mock params consistent with runner
    res = tester.test_3pt_bending("T3", nx=11, ny=3)
    for r in res:
        print(f"{r.test_name} | {r.metric} | Target: {r.theoretical:.4e} | Found: {r.actual:.4e} | Error: {r.error_pct:.2f}%")

if __name__ == "__main__":
    debug_t3()
