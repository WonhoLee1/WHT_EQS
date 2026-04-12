# -*- coding: utf-8 -*-
import os
import sys
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# Add parent directories to sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))

from ShellFemSolver.shell_solver import ShellFEM

# --- Mock petsc4py for Windows compatibility BEFORE importing jax_fem ---
import types
mock_petsc = types.ModuleType("petsc4py")
mock_petsc.PETSc = types.SimpleNamespace()
mock_petsc.PETSc.IntType = np.int32
mock_petsc.PETSc.ScalarType = np.float64
class MockMat:
    def createAIJ(self, *args, **kwargs): return self
mock_petsc.PETSc.Mat = MockMat
sys.modules["petsc4py"] = mock_petsc

try:
    from jax_fem.problem import Problem
    from jax_fem.generate_mesh import box_mesh, Mesh
    from jax_fem.solver import assign_bc, apply_bc_vec
except ImportError as e:
    print(f"Failed to import jax_fem: {e}")

# --- 1. Common Material & Geometry ---
L, W, t = 200.0, 200.0, 4.0
E, nu = 210000.0, 0.3
Fc = 100.0  # Increased load for better visibility [N]

# --- 2. Custom ShellFEM Benchmark ---
def run_shellfem():
    nx, ny = 11, 11
    x = np.linspace(0, L, nx)
    y = np.linspace(0, W, ny)
    xv, yv = np.meshgrid(x, y)
    nodes = np.stack([xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten())], axis=1)
    elements = []
    for j in range(ny-1):
        for i in range(nx-1):
            n1, n2, n4, n3 = j*nx+i, j*nx+i+1, (j+1)*nx+i, (j+1)*nx+i+1
            elements.append([n1, n2, n3, n4])
    quads = jnp.array(elements)
    
    fem = ShellFEM(nodes, quads=quads)
    params = {'E': E, 't': t, 'rho': 7.85e-9, 'quad_type': 'mitc4'}
    
    n00, nL0, n0L = 0, nx-1, (ny-1)*nx
    fixed_dofs = jnp.array([n00*6+2, nL0*6+2, n0L*6+2])
    fixed_dofs = jnp.concatenate([fixed_dofs, jnp.array([n00*6+0, n00*6+1, n00*6+5])])
    fixed_vals = jnp.zeros(len(fixed_dofs))
    
    center_node = (ny//2)*nx + (nx//2)
    forces = jnp.zeros(len(nodes)*6)
    forces = forces.at[center_node*6+2].set(Fc)
    
    u = fem.solve_static(params, forces, fixed_dofs, fixed_vals)
    return jnp.abs(u[center_node*6+2]), 0.0

# --- 3. JAX-FEM Solid Benchmark (Custom Solver for Windows) ---
import scipy.sparse
import scipy.sparse.linalg

def jax_fem_pure_solve(problem):
    """A pure JAX/Scipy Newton solver to bypass PETSc dependency in jax_fem."""
    dofs = np.zeros(problem.num_total_dofs_all_vars)
    dofs = assign_bc(dofs, problem)
    
    sol_list = problem.unflatten_fn_sol_list(dofs)
    res_list = problem.newton_update(sol_list) 
    res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
    res_vec = apply_bc_vec(res_vec, dofs, problem)
    
    val = np.array(problem.V)
    row = np.array(problem.I)
    col = np.array(problem.J)
    
    N = problem.num_total_dofs_all_vars
    A_scipy = scipy.sparse.csr_matrix((val, (row, col)), shape=(N, N))
    
    fixed_dof_indices = []
    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            n_inds = np.array(fe.node_inds_list[i]).flatten()
            v_inds = np.array(fe.vec_inds_list[i]).flatten()
            dof_indices = n_inds * fe.vec + v_inds + problem.offset[ind]
            fixed_dof_indices.extend(dof_indices.tolist())
    fixed_dof_indices = np.unique(np.array(fixed_dof_indices, dtype=np.int32))
    
    all_dofs = np.arange(N)
    is_fixed = np.zeros(N, dtype=bool)
    is_fixed[fixed_dof_indices[fixed_dof_indices < N]] = True
    free_dof_indices = all_dofs[~is_fixed]
    
    b = -np.array(res_vec)
    b_free = b[free_dof_indices]
    A_free = A_scipy[free_dof_indices, :][:, free_dof_indices]
    
    delta_free = scipy.sparse.linalg.spsolve(A_free, b_free)
    delta = np.zeros(N)
    delta[free_dof_indices] = delta_free
    dofs = dofs + delta
    return problem.unflatten_fn_sol_list(dofs)

def run_jaxfem():
    nx, ny, nz = 10, 10, 4 # Increased Z elements for better bending accuracy
    out_mesh = box_mesh(nx, ny, nz, L, W, t)
    mesh = Mesh(out_mesh.points, out_mesh.cells_dict['hexahedron'], ele_type='HEX8')
    
    def corner_00(p): return jnp.logical_and(jnp.isclose(p[0], 0, atol=1.e-5), jnp.isclose(p[1], 0, atol=1.e-5))
    def corner_L0(p): return jnp.logical_and(jnp.isclose(p[0], L, atol=1.e-5), jnp.isclose(p[1], 0, atol=1.e-5))
    def corner_0W(p): return jnp.logical_and(jnp.isclose(p[0], 0, atol=1.e-5), jnp.isclose(p[1], W, atol=1.e-5))
    def zero_fn(p): return 0.0
    
    location_fns = [corner_00, corner_L0, corner_0W, corner_00, corner_00]
    vecs = [2, 2, 2, 0, 1] 
    value_fns = [zero_fn] * 5
    dirichlet_bc_info = [location_fns, vecs, value_fns]
    
    area_center = (L/nx) * (W/ny)
    pressure = Fc / area_center

    class PlateBending(Problem):
        def get_tensor_map(self):
            def stress(u_grad):
                mu = E / (2. * (1. + nu))
                lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
                epsilon = 0.5 * (u_grad + u_grad.T)
                sigma = lmbda * jnp.trace(epsilon) * jnp.eye(self.dim) + 2 * mu * epsilon
                return sigma
            return stress
            
        def get_mass_map(self):
            def point_force(u, x, *args):
                in_center_x = jnp.abs(x[0] - L/2) < (L/nx/2.01)
                in_center_y = jnp.abs(x[1] - W/2) < (W/ny/2.01)
                # Apply as volume force in the top layer of elements
                in_top_layer = x[2] > (t - (t/nz) - 1e-5)
                # JAX-FEM mass_map is force per unit volume.
                # Total force Fc = Integral(f_vol * dV)
                # Over one center element volume: V_el = (L/nx)*(W/ny)*(t/nz)
                # So f_vol = Fc / V_el
                val = jnp.where(jnp.logical_and(jnp.logical_and(in_center_x, in_center_y), in_top_layer), 
                                Fc / (area_center * (t/nz)), 0.0) 
                return jnp.array([0., 0., val])
            return point_force

    problem = PlateBending(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
    
    t0 = time.time()
    sol_list = jax_fem_pure_solve(problem)
    t1 = time.time()
    
    center_pt = jnp.array([L/2.0, W/2.0, t])
    nodes_3d = out_mesh.points
    dists = jnp.linalg.norm(nodes_3d - center_pt, axis=1)
    center_node_idx = jnp.argmin(dists)
    
    def_center = jnp.abs(sol_list[0][center_node_idx, 2])
    return def_center, t1 - t0

# --- 4. Main Execution ---
if __name__ == "__main__":
    print("="*60)
    print("SOLVER COMPARISON: ShellFEM (MITC4) vs JAX-FEM (HEX8 Solid)")
    print(f"Geometry: {L}x{W}x{t} mm, E={E}, nu={nu}")
    print(f"Loading: {Fc} N at center top")
    print("="*60)
    
    print("\n[1/2] Running ShellFEM (Custom)...")
    try:
        def_sf, _ = run_shellfem()
        print(f"  > Deflection: {def_sf:.6f} mm")
    except Exception as e:
        print(f"  ! ShellFEM failed: {e}")
        def_sf = 0.0
    
    print("\n[2/2] Running JAX-FEM (Solid 3D)...")
    try:
        def_jf, time_jf = run_jaxfem()
        print(f"  > Deflection: {def_jf:.6f} mm")
        print(f"  > Solve Time: {time_jf:.4f} sec")
        
        diff_pct = (def_sf - def_jf)/def_jf * 100 if def_jf != 0 else 0.0
        print("\n" + "="*60)
        print(f"COMPARISON RESULTS:")
        print(f"  - Shell Deflection: {def_sf:.6f} mm")
        print(f"  - Solid Deflection: {def_jf:.6f} mm")
        print(f"  - Difference: {diff_pct:.2f} %")
        print(" (Note: Solid model with 11x11x4 HEX8 elements is much 'stiffer' than MITC4 Shell)")
        print("="*60)
    except Exception as e:
        print(f"  ! JAX-FEM failed: {e}")
        import traceback
        traceback.print_exc()
