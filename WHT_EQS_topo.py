# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

"""
WHT_EQS_topo.py
Modular Topology Optimization and General Shell Solver for JAX.
Supports Isoparametric 6-DOF Shell Elements (Quad/Tri).
"""

# --- 1. SIMP & Filtering Logic ---

@jit
def apply_simp_penalization(x, p=3.0, x_min=1e-3):
    """
    Apply SIMP (Solid Isotropic Material with Penalization).
    E_eff = x^p * E_0
    """
    return x_min + (1.0 - x_min) * (x ** p)

@jit
def compute_mac(mode1, mode2):
    """
    Compute Modal Assurance Criterion (MAC) between two mode shapes.
    """
    num = jnp.abs(jnp.dot(mode1, mode2)) ** 2
    den = (jnp.dot(mode1, mode1) * jnp.dot(mode2, mode2)) + 1e-10
    return num / den

@jit
def density_filter(x, nx, ny, radius=1.5):
    """
    Simple 2D Density Filter for structured grids.
    For unstructured grids, we use a distance-based weight matrix.
    """
    # Assuming x is (nx+1, ny+1) or flattened
    x_grid = x.reshape((nx + 1, ny + 1))
    
    # Placeholder for a more robust JAX-compatible filtering
    # In practice, this would be a matrix-vector product: x_filt = W @ x
    return x

# --- 2. General 6-DOF Shell Formulation (Isoparametric) ---

class GeneralShellSolver:
    """
    General 6-DOF Isoparametric Shell Solver.
    Supports arbitrary node coordinates (u, v, w, rx, ry, rz).
    """
    def __init__(self, nodes, elements, elem_type='quad', Lx=None, Ly=None):
        self.node_coords = jnp.array(nodes)      # (num_nodes, 3)
        self.elements = jnp.array(elements)       # (num_elems, 4 for quad, 3 for tri)
        self.elem_type = elem_type
        self.num_nodes = len(nodes)
        self.num_elems = len(elements)
        self.dof_per_node = 6
        self.total_dof = self.num_nodes * self.dof_per_node
        self.Lx = Lx if Lx is not None else float(jnp.max(self.node_coords[:, 0]))
        self.Ly = Ly if Ly is not None else float(jnp.max(self.node_coords[:, 1]))
        
        # Pre-compute element connectivity for assembly
        self.num_nodes_per_elem = 4 if elem_type == 'quad' else 3
        self.dofs_per_elem = self.num_nodes_per_elem * self.dof_per_node
        
        # Element DOF indices (num_elems, 24 for quad)
        # Using a simple node_idx * 6 + [0,1,2,3,4,5] mapping
        base_offsets = jnp.arange(6)
        
        def get_elem_dofs(nodes):
            res = []
            for n in nodes:
                res.append(n * 6 + base_offsets)
            return jnp.concatenate(res)
            
        self.element_dofs = vmap(get_elem_dofs)(self.elements) # (num_elems, 24)

    @partial(jit, static_argnums=(0,))
    def _get_mitc4_K(self, nodes_xyz, t, E, nu, rho, area_approx):
        """
        Detailed MITC4 (Mixed Interpolation of Tensorial Components) Shell Element.
        """
        # Material matrices
        Db = (E * t**3 / (12*(1-nu**2))) * jnp.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1-nu)/2]
        ])
        Dm = (E * t / (1-nu**2)) * jnp.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1-nu)/2]
        ])
        kappa = 5.0/6.0
        Ds = (E * t * kappa / (2*(1+nu))) * jnp.array([
            [1, 0],
            [0, 1]
        ])

        # Gauss points (2x2)
        gp = 0.577350269189626
        weights = [1.0, 1.0]
        
        K = jnp.zeros((24, 24))
        
        # --- Pre-compute Transformation to Local Element Plane ---
        v12 = nodes_xyz[1] - nodes_xyz[0]
        v13 = nodes_xyz[2] - nodes_xyz[0]
        normal = jnp.cross(v12, v13)
        unit_normal = normal / (jnp.linalg.norm(normal) + 1e-10)
        x_local = v12 / (jnp.linalg.norm(v12) + 1e-10)
        y_local = jnp.cross(unit_normal, x_local)
        R = jnp.stack([x_local, y_local, unit_normal], axis=1) # (3, 3) Local to Global
        
        # Transform node coordinates to local 2D plane
        nodes_local = (nodes_xyz - nodes_xyz[0]) @ R # (4, 3)
        xy = nodes_local[:, :2] # (4, 2)
        
        # Integration
        for xi in [-gp, gp]:
            for eta in [-gp, gp]:
                # Shape functions and derivatives
                N = 0.25 * jnp.array([
                    (1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)
                ])
                dN_dxi = 0.25 * jnp.array([
                    [-(1-eta), (1-eta), (1+eta), -(1+eta)],
                    [-(1-xi), -(1+xi), (1+xi), (1-xi)]
                ])
                
                # Jacobian
                J = dN_dxi @ xy # (2, 2)
                detJ = jnp.linalg.det(J)
                invJ = jnp.linalg.inv(J)
                dN_dx = invJ @ dN_dxi # (2, 4)
                
                # Membrane B-matrix (8x24 -> Actually 3x24 in local DOFs)
                Bm = jnp.zeros((3, 24))
                for i in range(4):
                    # Local DOFs: [u, v, w, rx, ry, rz]
                    Bm = Bm.at[0, i*6].set(dN_dx[0, i])
                    Bm = Bm.at[1, i*6+1].set(dN_dx[1, i])
                    Bm = Bm.at[2, i*6].set(dN_dx[1, i])
                    Bm = Bm.at[2, i*6+1].set(dN_dx[0, i])
                
                # Bending B-matrix (3x24)
                Bb = jnp.zeros((3, 24))
                for i in range(4):
                    Bb = Bb.at[0, i*6+4].set(dN_dx[0, i]) # rx contributes to bending? 
                    # Note: Local rotation conventions vary. Use Mindlin-Reissner.
                    Bb = Bb.at[1, i*6+3].set(-dN_dx[1, i])
                    Bb = Bb.at[2, i*6+4].set(dN_dx[1, i])
                    Bb = Bb.at[2, i*6+3].set(-dN_dx[0, i])
                
                K += (Bm.T @ Dm @ Bm + Bb.T @ Db @ Bb) * detJ
                
        # --- Shear part (Simplified for now, MITC requires special interpolation) ---
        # Using 1-point integration for shear to avoid locking (Basic RM)
        Bs = jnp.zeros((2, 24))
        # Evaluate at (0,0)
        dN_dxi_0 = 0.25 * jnp.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])
        J0 = dN_dxi_0 @ xy
        invJ0 = jnp.linalg.inv(J0)
        dN_dx0 = invJ0 @ dN_dxi_0
        N0 = jnp.array([0.25, 0.25, 0.25, 0.25])
        
        for i in range(4):
            Bs = Bs.at[0, i*6+2].set(dN_dx0[0, i])
            Bs = Bs.at[0, i*6+4].set(N0[i])
            Bs = Bs.at[1, i*6+2].set(dN_dx0[1, i])
            Bs = Bs.at[1, i*6+3].set(-N0[i])
            
        K += (Bs.T @ Ds @ Bs) * (jnp.linalg.det(J0) * 4.0)
        
        # Add Drill stiffness (Numerical stabilization)
        drill_k = 1e-4 * E * t * area_approx
        for i in range(4):
            K = K.at[i*6+5, i*6+5].add(drill_k)
            
        # Transform K from Local Element Coordinate to Global
        T_3x3 = R # Local to Global
        z3 = jnp.zeros((3,3))
        T_6x6 = jnp.block([[T_3x3, z3], [z3, T_3x3]])
        T_elem = jax.scipy.linalg.block_diag(T_6x6, T_6x6, T_6x6, T_6x6)
        
        K_global = T_elem @ K @ T_elem.T
        
        # Mass matrix (Consistent)
        M = jnp.zeros((24, 24))
        # Simplified mass for now (Lumped)
        m_node = (rho * t * area_approx) / 4.0
        m_diag = jnp.repeat(jnp.array([m_node, m_node, m_node, 1e-6, 1e-6, 1e-6]), 4)
        M_global = jnp.diag(m_diag)
        
        return K_global, M_global

    @partial(jit, static_argnums=(0,))
    def compute_local_K(self, nodes_xyz, t, E, nu, rho):
        # Calculate approximate area
        v12 = nodes_xyz[1] - nodes_xyz[0]
        v13 = nodes_xyz[2] - nodes_xyz[0]
        area_approx = 0.5 * jnp.linalg.norm(jnp.cross(v12, v13))
        
        # Fallback to MITC4 implementation
        return self._get_mitc4_K(nodes_xyz, t, E, nu, rho, area_approx)

    def assemble(self, params):
        """
        Assembles global K and M matrices using Isoparametric theory.
        params: dict with 't', 'rho', 'E' (at nodes or elements)
        """
        # Node properties to Element properties (Average)
        t_elems = jnp.mean(params['t'][self.elements], axis=1)
        rho_elems = jnp.mean(params['rho'][self.elements], axis=1)
        E_elems = jnp.mean(params['E'][self.elements], axis=1)
        nu = 0.3
        
        # Batch compute all element matrices
        if 'z' in params:
            # Rebuild 3D coordinates using optimized Z
            curr_nodes_xyz = jnp.column_stack([self.node_coords[:, :2], params['z'].flatten()])
            elem_xyz = curr_nodes_xyz[self.elements]
        else:
            elem_xyz = self.node_coords[self.elements] # (num_elems, 4, 3)
        
        Ke, Me = vmap(lambda xyz, t, rho, E: self.compute_local_K(xyz, t, E, nu, rho))(
            elem_xyz, t_elems, rho_elems, E_elems
        )
        
        # Assemble into Global matrices
        K_global = jnp.zeros((self.total_dof, self.total_dof))
        M_global = jnp.zeros((self.total_dof, self.total_dof))
        
        # Indices for scattering
        I_local = jnp.repeat(jnp.arange(self.dofs_per_elem), self.dofs_per_elem)
        J_local = jnp.tile(jnp.arange(self.dofs_per_elem), self.dofs_per_elem)
        
        Global_I = self.element_dofs[:, I_local].flatten()
        Global_J = self.element_dofs[:, J_local].flatten()
        
        K_global = K_global.at[Global_I, Global_J].add(Ke.reshape(-1))
        M_global = M_global.at[Global_I, Global_J].add(Me.reshape(-1))
        
        return K_global, M_global

    def solve_static_partitioned(self, K, F, free_dofs, fixed_dofs, fixed_vals):
        """
        Standard partitioned solver (Same interface as PlateFEM).
        """
        K_ff = K[free_dofs, :][:, free_dofs]
        F_f = F[free_dofs]
        
        # Apply fixed values (enforced displacement)
        if jnp.any(fixed_vals != 0):
            F_f = F_f - K[free_dofs, :][:, fixed_dofs] @ fixed_vals
            
        u_free = jax.scipy.linalg.solve(K_ff, F_f)
        
        u = jnp.zeros(self.total_dof)
        u = u.at[free_dofs].set(u_free)
        u = u.at[fixed_dofs].set(fixed_vals)
        return u

    def solve_eigen(self, K, M, num_modes=5):
        """
        Eigen solver using lumped mass approach.
        """
        # Stability: Small epsilon on diagonal
        m_diag = jnp.diag(M)
        m_diag = jnp.maximum(m_diag, 1e-15)
        m_inv_sqrt = 1.0 / jnp.sqrt(m_diag)
        
        K_stable = (K + K.T) / 2.0
        A = K_stable * m_inv_sqrt[:, None] * m_inv_sqrt[None, :]
        
        vals, vecs_v = jnp.linalg.eigh(A)
        vecs_u = vecs_v * m_inv_sqrt[:, None]
        
        return jnp.maximum(vals, 0.0), vecs_u

    def compute_curvature(self, u):
        """
        Compute curvature field from displacement.
        """
        theta_x = u[3::6]  # rotation about x-axis
        theta_y = u[4::6]  # rotation about y-axis
        
        # Reshape to grid
        nx1, ny1 = self.nx + 1, self.ny + 1
        theta_x_grid = theta_x.reshape(nx1, ny1)
        theta_y_grid = theta_y.reshape(nx1, ny1)
        
        kappa_xx = jnp.zeros_like(theta_x_grid)
        kappa_yy = jnp.zeros_like(theta_y_grid)
        
        # Finite differences
        kappa_xx = kappa_xx.at[1:-1, :].set(-(theta_y_grid[2:, :] - theta_y_grid[:-2, :]) / (2 * self.dx))
        kappa_xx = kappa_xx.at[0, :].set(-(theta_y_grid[1, :] - theta_y_grid[0, :]) / self.dx)
        kappa_xx = kappa_xx.at[-1, :].set(-(theta_y_grid[-1, :] - theta_y_grid[-2, :]) / self.dx)
        
        kappa_yy = kappa_yy.at[:, 1:-1].set((theta_x_grid[:, 2:] - theta_x_grid[:, :-2]) / (2 * self.dy))
        kappa_yy = kappa_yy.at[:, 0].set((theta_x_grid[:, 1] - theta_x_grid[:, 0]) / self.dy)
        kappa_yy = kappa_yy.at[:, -1].set((theta_x_grid[:, -1] - theta_x_grid[:, -2]) / self.dy)
        
        dtheta_y_dy = jnp.zeros_like(theta_y_grid)
        dtheta_y_dy = dtheta_y_dy.at[:, 1:-1].set((theta_y_grid[:, 2:] - theta_y_grid[:, :-2]) / (2 * self.dy))
        dtheta_y_dy = dtheta_y_dy.at[:, 0].set((theta_y_grid[:, 1] - theta_y_grid[:, 0]) / self.dy)
        dtheta_y_dy = dtheta_y_dy.at[:, -1].set((theta_y_grid[:, -1] - theta_y_grid[:, -2]) / self.dy)
        
        dtheta_x_dx = jnp.zeros_like(theta_x_grid)
        dtheta_x_dx = dtheta_x_dx.at[1:-1, :].set((theta_x_grid[2:, :] - theta_x_grid[:-2, :]) / (2 * self.dx))
        dtheta_x_dx = dtheta_x_dx.at[0, :].set((theta_x_grid[1, :] - theta_x_grid[0, :]) / self.dx)
        dtheta_x_dx = dtheta_x_dx.at[-1, :].set((theta_x_grid[-1, :] - theta_x_grid[-2, :]) / self.dx)
        
        kappa_xy = (dtheta_y_dy - dtheta_x_dx) / 2.0
        
        return jnp.stack([kappa_xx.flatten(), kappa_yy.flatten(), kappa_xy.flatten()], axis=1)

    def compute_moment(self, u, params):
        nu = 0.3
        curvatures = self.compute_curvature(u)
        t = params['t'].flatten()
        E = params['E'].flatten()
        D0 = E * t**3 / (12 * (1 - nu**2))
        M_xx = D0 * (curvatures[:, 0] + nu * curvatures[:, 1])
        M_yy = D0 * (nu * curvatures[:, 0] + curvatures[:, 1])
        M_xy = D0 * ((1 - nu) / 2.0) * curvatures[:, 2]
        return jnp.stack([M_xx, M_yy, M_xy], axis=1)

    def compute_max_surface_stress(self, u, params):
        moments = self.compute_moment(u, params)
        t = params['t'].flatten()
        sigma_xx = 6.0 * moments[:, 0] / (t**2)
        sigma_yy = 6.0 * moments[:, 1] / (t**2)
        tau_xy = 6.0 * moments[:, 2] / (t**2)
        return jnp.sqrt(sigma_xx**2 + sigma_yy**2 - sigma_xx * sigma_yy + 3.0 * tau_xy**2)

    def compute_max_surface_strain(self, u, params):
        curvatures = self.compute_curvature(u)
        t = params['t'].flatten()
        z_max = t / 2.0
        eps_xx = z_max * curvatures[:, 0]
        eps_yy = z_max * curvatures[:, 1]
        gamma_xy = z_max * curvatures[:, 2]
        return jnp.sqrt(eps_xx**2 + eps_yy**2 - eps_xx * eps_yy + 3.0 * gamma_xy**2)

    def compute_strain_energy_density(self, u, params):
        curvatures = self.compute_curvature(u)
        moments = self.compute_moment(u, params)
        return 0.5 * (curvatures[:, 0] * moments[:, 0] + curvatures[:, 1] * moments[:, 1] + 2.0 * curvatures[:, 2] * moments[:, 2])

# --- 3. Performance Objective Functions ---

@jit
def compute_compliance(u, F, free_indices):
    """
    Compliance = F^T * u (Work done by external forces)
    """
    return jnp.dot(F[free_indices], u[free_indices])

@jit
def compute_frequency_loss(eigenvalues, target_idx=0):
    """
    Loss for frequency maximization.
    """
    return -eigenvalues[target_idx] # Minimize negative frequency

@jit
def compute_mass_constraint(curr_mass, limit_mass, penalty=1e6):
    """
    Inequality mass constraint: Penalty * max(0, curr_mass - limit_mass)
    """
    return penalty * jnp.maximum(0.0, curr_mass - limit_mass)**2
