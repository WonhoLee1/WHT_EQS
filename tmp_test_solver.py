import os
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Add workspace to path
sys.path.append(r'c:\Users\GOODMAN\code_sheet')

from ShellFemSolver.shell_solver import ShellFEM

def test_solver_consistency():
    print("Testing Shell Solver Consistency...")
    Lx, Ly = 500.0, 300.0
    nx, ny = 10, 6
    
    # Create simple grid nodes
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    X, Y = np.meshgrid(x, y)
    nodes = np.stack([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())], axis=1)
    
    # Create quad elements
    quads = []
    for i in range(ny):
        for j in range(nx):
            n1 = i * (nx + 1) + j
            n2 = n1 + 1
            n4 = (i + 1) * (nx + 1) + j
            n3 = n4 + 1
            quads.append([n1, n2, n3, n4])
    quads = np.array(quads)

    # Correct Initialization
    fem = ShellFEM(nodes, quads=quads)
    
    params = {
        't': jnp.ones(len(nodes)),
        'E': 210000.0 * jnp.ones(len(nodes)),
        'rho': 7.85e-9 * jnp.ones(len(nodes)),
        'z': jnp.zeros(len(nodes))
    }
    
    # 1. Test Assembly
    K, M = fem.assemble(params)
    print(f"Assembly OK. K shape: {K.shape}")
    
    # 2. Dummy solution vector (e.g. uniform unit displacement)
    u = jnp.ones(fem.total_dof) * 0.01 
    
    # 3. Test Field Results
    res = fem.compute_field_results(u, params)
    
    expected_keys = ['stress_vm', 'strain_max_principal', 'sed_node', 'moments', 'u_mag', 'sig_el', 'eps_el', 'vm_el', 'sed_el']
    for k in expected_keys:
        if k not in res:
            print(f"MISSING KEY: {k}")
            return False
            
    # Check dimensionality: stress_vm, sed_node should be (num_nodes,)
    if res['stress_vm'].shape[0] != fem.num_nodes:
        print(f"Dimension Mismatch: stress_vm is {res['stress_vm'].shape}, expected {fem.num_nodes}")
        return False
    if res['sed_node'].shape[0] != fem.num_nodes:
        print(f"Dimension Mismatch: sed_node is {res['sed_node'].shape}, expected {fem.num_nodes}")
        return False
    # Check moments: (num_nodes, 3)
    if res['moments'].shape != (fem.num_nodes, 3):
        print(f"Dimension Mismatch: moments is {res['moments'].shape}, expected {(fem.num_nodes, 3)}")
        return False
    
    # Check element-wise density
    if res['sed_el'].shape[0] != (fem.trias.shape[0] + fem.quads.shape[0]):
        print(f"Dimension Mismatch: sed_el is {res['sed_el'].shape}, expected {fem.trias.shape[0] + fem.quads.shape[0]}")
        return False
        
    print("Consistency Test Passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_solver_consistency()
        if success:
            print("SUCCESS")
            sys.exit(0)
        else:
            print("FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
