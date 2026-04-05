import jax
import jax.numpy as jnp
import numpy as np
from ShellFemSolver.shell_solver import ShellFEM

def test_twisting_debug():
    Lx, Ly, t, E, nu, F_c = 100.0, 100.0, 2.0, 210000.0, 0.3, 50.0
    nx, ny = 5, 5 # Small mesh for debugging
    x, y = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny)
    xv, yv = np.meshgrid(x, y)
    nodes = np.stack([xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten())], axis=1)
    
    # Quads
    els = []
    for j in range(ny-1):
        for i in range(nx-1):
            els.append([j*nx+i, (j+1)*nx+i+1, (j+1)*nx+i+1, j*nx+i+1]) # Oh wait, my mesh gen earlier had a mistake?
            # Re-gen correctly
    els = [[j*nx+i, j*nx+i+1, (j+1)*nx+i+1, (j+1)*nx+i] for j in range(ny-1) for i in range(nx-1)]
    els = np.array(els)
    
    fem = ShellFEM(nodes, quads=els)
    params = {'E': jnp.ones(len(nodes))*E, 't': jnp.ones(len(nodes))*t, 'rho': jnp.ones(len(nodes))*7.85e-9}
    
    n00, nL0, n0L, nLL = 0, nx-1, (ny-1)*nx, len(nodes)-1
    fixed_dofs = [n00*6+2, nL0*6+2, n0L*6+2]
    F = np.zeros(len(nodes)*6); F[nLL*6+2] = F_c
    u = fem.solve_static(params, F, np.array(fixed_dofs), np.zeros(3))
    
    field = fem.compute_field_results(u, params)
    
    print(f"Theory Max Tau_xy: {3*F_c/t**2}")
    print(f"Theory VM: {np.sqrt(3)*(3*F_c/t**2)}")
    
    # Check raw components if possible
    # Wait, I don't return raw components in results.
    # I'll check 'stress_vm'
    print(f"FEM Avg VM: {np.mean(field['stress_vm'])}")
    
    # Check max element VM
    print(f"FEM Max VM: {np.max(field['stress_vm_el'])}")

if __name__ == "__main__":
    test_twisting_debug()
