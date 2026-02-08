# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from functools import partial

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

class PlateFEM:
    """
    Maximum Performance 3-DOF Plate FEM solver.
    Uses broadcasting to avoid O(N^3) matrix-matrix products.
    """
    def __init__(self, Lx, Ly, nx, ny):
        self.Lx, self.Ly, self.nx, self.ny = Lx, Ly, nx, ny
        self.dx, self.dy = Lx / nx, Ly / ny
        self.num_nodes = (nx + 1) * (ny + 1)
        self.dof_per_node = 3
        self.total_dof = self.num_nodes * self.dof_per_node
        self._generate_mesh_data()
        self._precompute_unit_matrices()

    def _generate_mesh_data(self):
        x, y = jnp.linspace(0, self.Lx, self.nx + 1), jnp.linspace(0, self.Ly, self.ny + 1)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        self.node_coords = jnp.stack([X.flatten(), Y.flatten()], axis=1)
        
        elems = []
        for i in range(self.nx):
            for j in range(self.ny):
                n = i * (self.ny + 1) + j
                elems.append([n, n + self.ny + 1, n + self.ny + 2, n + 1])
        self.elements = jnp.array(elems, dtype=jnp.int32)
        
        self.element_dof_indices = vmap(
            lambda elem: jnp.concatenate([jnp.array([n*3, n*3+1, n*3+2]) for n in elem])
        )(self.elements)
        
        self.assem_I = self.element_dof_indices[:, jnp.repeat(jnp.arange(12), 12)].flatten()
        self.assem_J = self.element_dof_indices[:, jnp.tile(jnp.arange(12), 12)].flatten()

    def _precompute_unit_matrices(self):
        a, b = self.dx / 2.0, self.dy / 2.0
        gp = 0.577350269189626
        nu = 0.3
        Db_const = jnp.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
        
        Kb, Ks = jnp.zeros((12, 12)), jnp.zeros((12, 12))
        for xi in [-gp, gp]:
            for eta in [-gp, gp]:
                Bb = self._get_B_bending(xi, eta, a, b)
                Kb += (Bb.T @ Db_const @ Bb) * (a * b)
        
        Bs = self._get_B_shear(0.0, 0.0, a, b)
        self.K_unit_bending, self.K_unit_shear = Kb, (Bs.T @ Bs) * (a * b * 4.0)

    def _get_B_bending(self, xi, eta, a, b):
        dN_dxi = 0.25 * jnp.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
        dN_deta= 0.25 * jnp.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
        dN_dx, dN_dy = dN_dxi / a, dN_deta / b
        B = jnp.zeros((3, 12))
        for i in range(4):
            idx = i * 3
            B = B.at[0, idx+2].set(dN_dx[i])
            B = B.at[1, idx+1].set(-dN_dy[i])
            B = B.at[2, idx+2].set(dN_dy[i])
            B = B.at[2, idx+1].set(-dN_dx[i])
        return B

    def _get_B_shear(self, xi, eta, a, b):
        dN_dxi = 0.25 * jnp.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
        dN_deta= 0.25 * jnp.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
        dN_dx, dN_dy = dN_dxi / a, dN_deta / b
        N = 0.25 * jnp.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])
        B = jnp.zeros((2, 12))
        for i in range(4):
            idx = i * 3
            B = B.at[0, idx].set(dN_dx[i])
            B = B.at[0, idx+2].set(N[i])
            B = B.at[1, idx].set(dN_dy[i])
            B = B.at[1, idx+1].set(-N[i])
        return B

    def assemble(self, params):
        t = params['t'].flatten()[self.elements].mean(axis=1)
        E = params['E'].flatten()[self.elements].mean(axis=1)
        rho = params['rho'].flatten()[self.elements].mean(axis=1)
        
        D_bend = (E * t**3 / (12 * (1 - 0.3**2)))
        D_shear = (E * t * (5/6) / (2 * (1 + 0.3)))
        
        Ke_all = D_bend[:, None, None] * self.K_unit_bending + D_shear[:, None, None] * self.K_unit_shear
        
        K = jnp.zeros((self.total_dof, self.total_dof))
        K = K.at[self.assem_I, self.assem_J].add(Ke_all.flatten())
        
        # Stability: direct diagonal addition is faster than K + eye
        K = K.at[jnp.diag_indices(self.total_dof)].add(1e-8)
        
        m_node = rho * (self.dx * self.dy * t) / 4.0
        # Use vector for mass to avoid O(N^3)
        M_diag = jnp.zeros(self.total_dof)
        node_idx = self.elements.flatten()
        M_diag = M_diag.at[node_idx * 3].add(jnp.repeat(m_node, 4))
        M_diag = M_diag.at[node_idx * 3 + 1].add(jnp.repeat(m_node * (t**2/12.0 + 1e-12), 4))
        M_diag = M_diag.at[node_idx * 3 + 2].add(jnp.repeat(m_node * (t**2/12.0 + 1e-12), 4))
        
        return K, M_diag + 1e-15

    def solve_static_partitioned(self, K, F, free_dofs, fixed_dofs, fixed_vals):
        K_ff = K[jnp.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs] - K[jnp.ix_(free_dofs, fixed_dofs)] @ fixed_vals
        # Use symmetric solver
        u_f = jax.scipy.linalg.solve(K_ff, F_f, assume_a='pos')
        return jnp.zeros(self.total_dof).at[free_dofs].set(u_f).at[fixed_dofs].set(fixed_vals)

    def solve_eigen(self, K, M_diag, num_modes=10):
        # FAST O(N^2) Diagonal scaling
        inv_sqrt_M = 1.0 / jnp.sqrt(M_diag)
        K_std = (inv_sqrt_M[:, None] * K) * inv_sqrt_M[None, :]
        vals, vecs_std = jnp.linalg.eigh(K_std)
        # Scale back modes: v = M^-1/2 * v_std
        vecs = inv_sqrt_M[:, None] * vecs_std
        return vals[:num_modes], vecs[:, :num_modes]

    def compute_curvature(self, u):
        tx, ty = u[1::3].reshape(self.nx + 1, self.ny + 1), u[2::3].reshape(self.nx + 1, self.ny + 1)
        dty_dx, dtx_dy = jnp.gradient(ty, self.dx, axis=0), jnp.gradient(tx, self.dy, axis=1)
        dty_dy, dtx_dx = jnp.gradient(ty, self.dy, axis=1), jnp.gradient(tx, self.dx, axis=0)
        return jnp.stack([dty_dx.flatten(), -dtx_dy.flatten(), (dty_dy - dtx_dx).flatten()], axis=1)

    def compute_moment(self, u, params):
        curv, t, E = self.compute_curvature(u), params['t'].flatten(), params['E'].flatten()
        Db = E * t**3 / (10.92) # 12*(1-0.3^2)
        return jnp.stack([Db*(curv[:,0]+0.3*curv[:,1]), Db*(curv[:,1]+0.3*curv[:,0]), Db*0.35*curv[:,2]], axis=1)

    def compute_max_surface_stress(self, u, params):
        mom, t = self.compute_moment(u, params), params['t'].flatten()
        sx, sy, sxy = 6*mom[:,0]/(t**2+1e-9), 6*mom[:,1]/(t**2+1e-9), 6*mom[:,2]/(t**2+1e-9)
        return jnp.sqrt(sx**2 - sx*sy + sy**2 + 3*sxy**2)

    def compute_max_surface_strain(self, u, params):
        curv, t = self.compute_curvature(u), params['t'].flatten()
        ex, ey, exy = curv[:,0]*t/2, curv[:,1]*t/2, curv[:,2]*t/2
        return jnp.sqrt(ex**2 - ex*ey + ey**2 + 3*exy**2)

    def compute_strain_energy_density(self, u, params):
        curv, mom = self.compute_curvature(u), self.compute_moment(u, params)
        return 0.5 * jnp.sum(curv * mom, axis=1)
