# -*- coding: utf-8 -*-
import os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))

from ShellFemSolver.shell_solver import ShellFEM

def main():
    # Benchmark problem setup: 51x51 Mesh (approx 15,606 DOFs)
    Lx, Ly, nx, ny = 1000.0, 1000.0, 51, 51
    E, t, rho, nu = 210000.0, 5.0, 7.85e-9, 0.3
    
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    xv, yv = np.meshgrid(x, y)
    nodes = np.stack([xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten())], axis=1)
    
    # Generate Quads
    els = []
    for j in range(ny-1):
        for i in range(nx-1):
            els.append([j*nx+i, (j+1)*nx+i, (j+1)*nx+i+1, j*nx+i+1])
    elements = np.array(els)
    
    # Initialize Solver
    fem = ShellFEM(nodes, trias=None, quads=elements)
    params = {
        'E': jnp.ones(len(nodes)) * E,
        't': jnp.ones(len(nodes)) * t,
        'rho': jnp.ones(len(nodes)) * rho
    }
    
    # BCs: Cantilever
    fixed_nodes = np.where(nodes[:, 0] < 1e-5)[0]
    fixed_dofs = []
    for node in fixed_nodes:
        fixed_dofs.extend([int(node*6 + i) for i in range(6)])
    
    F = jnp.zeros(len(nodes)*6)
    # Uniform pressure on free side
    tip_nodes = np.where(nodes[:, 0] > Lx - 1e-5)[0]
    F = F.at[tip_nodes * 6 + 2].set(-1.0)
    
    # Warm up (to exclude JIT time from performance analysis)
    # However, for a single run comparison, we just measure after the first call.
    # To see TRUE parallel execution speed, we should run twice and measure the second.
    
    # 1. First Run (JIT + Execution)
    t0 = time.perf_counter()
    u1 = fem.solve_static(params, F, jnp.array(fixed_dofs), jnp.zeros(len(fixed_dofs)))
    u1.block_until_ready()
    t_cold = (time.perf_counter() - t0) * 1000.0
    
    # 2. Second Run (Pure Execution)
    t1 = time.perf_counter()
    u2 = fem.solve_static(params, F, jnp.array(fixed_dofs), jnp.zeros(len(fixed_dofs)))
    u2.block_until_ready()
    t_warm = (time.perf_counter() - t1) * 1000.0
    
    print(f"JIT_TIME: {t_cold - t_warm:.2f} ms")
    print(f"EXEC_TIME: {t_warm:.2f} ms")
    print(f"TOTAL_TIME: {t_cold:.2f} ms")

if __name__ == "__main__":
    main()
