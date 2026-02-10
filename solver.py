# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from functools import partial
from jax import custom_vjp

# --- Safe Eigenvalue Solver with Gradient Stabilization ---
@custom_vjp
def safe_eigh(A):
    vals, vecs = jnp.linalg.eigh(A)
    return vals, vecs

def safe_eigh_fwd(A):
    vals, vecs = jnp.linalg.eigh(A)
    return (vals, vecs), (vals, vecs)

def safe_eigh_bwd(res, g):
    vals, vecs = res
    g_vals, g_vecs = g
    
    # Gradient of eigenvalues
    # dL/dA_vals = sum(g_val_i * v_i * v_i.T)
    grad_A_vals = jnp.einsum('k,ik,jk->ij', g_vals, vecs, vecs)
    
    # Gradient of eigenvectors
    # F_ij = 1 / (lambda_j - lambda_i) if i != j else 0
    # Safe division: mask out diagonal and close values
    diff = vals[:, None] - vals[None, :]
    
    # Create a mask for non-zero differences, and replace 0s with a large number
    # so that 1/large_number becomes 0.
    # This effectively sets F_ii = 0 and F_ij = 0 if vals_i == vals_j
    F = jnp.where(jnp.abs(diff) < 1e-9, jnp.inf, 1.0 / diff)
    F = jnp.where(jnp.isinf(F), 0.0, F) # Replace inf with 0
    F = jnp.where(jnp.isnan(F), 0.0, F) # Replace nan with 0 (shouldn't happen with inf handling)
    
    vt_gv = vecs.T @ g_vecs
    
    # Contribution from eigenvector derivatives
    P = F * vt_gv
    
    grad_A_vecs = vecs @ P @ vecs.T
    
    # Combine (for real symmetric A)
    # The gradient must be symmetric
    total_grad = grad_A_vals + grad_A_vecs
    total_grad = 0.5 * (total_grad + total_grad.T)
    
    return (total_grad,)

safe_eigh.defvjp(safe_eigh_fwd, safe_eigh_bwd)

# Enable 64-bit precision for FEM accuracy
jax.config.update("jax_enable_x64", True)

class PlateFEM:
    """
    Differentiable FEM solver for a rectangular plate.
    Uses a 12-DOF rectangular plate bending element (ACM/MZC).
    """
    def __init__(self, Lx, Ly, nx, ny):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.num_nodes = (nx + 1) * (ny + 1)
        self.num_elements = nx * ny
        self.dof_per_node = 6  # u, v, w, theta_x, theta_y, theta_z
        self.total_dof = self.num_nodes * self.dof_per_node
        
        self._generate_mesh_data()

    def _generate_mesh_data(self):
        # Node coordinates
        x = jnp.linspace(0, self.Lx, self.nx + 1)
        y = jnp.linspace(0, self.Ly, self.ny + 1)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        self.node_coords = jnp.stack([X.flatten(), Y.flatten()], axis=1)

        # Element connectivity (node indices for each element)
        # Element indices (i, j) -> Nodes:
        # (i, j)   -> n0
        # (i+1, j) -> n1
        # (i+1, j+1) -> n2
        # (i, j+1) -> n3
        # Global node index = i * (ny + 1) + j
        # But meshgrid('ij') makes x vary first? No:
        # indexing='ij': X[i,j] = x[i], Y[i,j] = y[j]
        # Flatten order (default C): changes last index (j) fastest.
        # So node index k corresponds to x[k // (ny+1)], y[k % (ny+1)]
        
        elems = []
        for i in range(self.nx):
            for j in range(self.ny):
                n_bl = i * (self.ny + 1) + j      # Bottom-Left
                n_br = (i + 1) * (self.ny + 1) + j # Bottom-Right
                n_tr = (i + 1) * (self.ny + 1) + (j + 1) # Top-Right
                n_tl = i * (self.ny + 1) + (j + 1) # Top-Left
                elems.append([n_bl, n_br, n_tr, n_tl])
        self.elements = jnp.array(elems, dtype=jnp.int32)
        
        # Build DOF indices for assembly
        # For each element, we have 4 nodes * 3 DOFs = 12 DOFs
        # We need a map from (elem_idx, local_dof) -> global_dof
        
        def get_dof_indices(node_idx):
            start = node_idx * 6
            return jnp.array([start, start+1, start+2, start+3, start+4, start+5])

        self.element_dof_indices = vmap(
            lambda elem: jnp.concatenate([get_dof_indices(n) for n in elem])
        )(self.elements)

    def _get_membrane_K(self, E, t, nu, rho):
        # Q4 Plane Stress Element Stiffness (2x2 Gauss Integration)
        # DOFs: u, v at each node.
        
        a = self.dx / 2.0
        b = self.dy / 2.0
        
        # Constitutive Matrix (Plane Stress)
        C = (E * t / (1 - nu**2)) * jnp.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])
        
        # Gauss Points (2x2)
        gp = 0.577350269189626
        
        K_m = jnp.zeros((8, 8)) # 4 nodes * 2 DOFs
        
        for xi in [-gp, gp]:
            for eta in [-gp, gp]:
                # B matrix for membrane (3x8)
                # Strain = [du/dx, dv/dy, du/dy + dv/dx]
                
                # Derivatives of shape functions
                # N1 = 0.25*(1-xi)*(1-eta)
                dN_dxi = 0.25 * jnp.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
                dN_deta= 0.25 * jnp.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
                
                dN_dx = dN_dxi / a
                dN_dy = dN_deta / b
                
                B = jnp.zeros((3, 8))
                for i in range(4):
                    idx = i * 2
                    # eps_x = du/dx
                    B = B.at[0, idx].set(dN_dx[i])
                    # eps_y = dv/dy
                    B = B.at[1, idx+1].set(dN_dy[i])
                    # gam_xy = du/dy + dv/dx
                    B = B.at[2, idx].set(dN_dy[i])
                    B = B.at[2, idx+1].set(dN_dx[i])
                    
                detJ = a * b
                K_m += (B.T @ C @ B) * detJ
                
        # Membrane Mass (Lumped)
        area = 4 * a * b
        volume = area * t
        m_node = rho * volume / 4.0
        M_m = jnp.eye(8) * m_node
        
        return K_m, M_m

    @partial(jit, static_argnums=(0,))
    def element_stiffness(self, E, t, nu):
        """
        Computes 12x12 stiffness matrix for a single rectangular element.
        Based on standard ACM/MZC element theory or similar.
        
        Simplification: We will use a standard rectangular element stiffness code.
        References: Przemieniecki, define variables a=dx/2, b=dy/2
        """
        a = self.dx / 2.0
        b = self.dy / 2.0
        
        # Flexural rigidity
        D = E * t**3 / (12.0 * (1.0 - nu**2))
        
        # This is a placeholder for the actual massive 12x12 matrix formulas.
        # For brevity/correctness in this plan, I'll use a simplified isotropic approximation 
        # or we might need to include the full coefficient matrix.
        # Let's use a very basic approximation or a library function if possible.
        # Since exact coefficients are long, I will assume a helper function `rect_plate_k` exists
        # In a real implementation we paste the coefficients.
        
        # For now, let's implement the core integration logic using Gaussian Quadrature to be safe and generic.
        # Bending strain-displacement matrix B.
        # D matrix.
        # K = int B.T D B dA
        
        # Shape functions for 12-DOF Rectangle (ACM)
        # w(x,y) polynomial terms: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3, x^3y, xy^3
        
        # Let's rely on numerical integration (2x2 or 3x3 Gauss)
        # It's slower but less error-prone to hardcode coefficients.
        
        # Gauss points (2x2 is enough for cubic)
        gp = jnp.array([-0.577350269189626, 0.577350269189626])
        gw = jnp.array([1.0, 1.0])
        
        # Constitutive matrix
        C = D * jnp.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0]
        ])
        
        # We need to construct K by summing contributions from Gauss points
        # But doing this inside JIT is fine.
        
        K = jnp.zeros((12, 12))
        
        # Helper to get B matrix at local (xi, eta) in [-1, 1]
        # x = xi * a, y = eta * b
        
        # Actually, simpler to just map parameters to a standard hardcoded K if possible.
        # But ok, let's stick to numerical integration for flexibility.
        
        def shape_func_derivatives(xi, eta):
            # Returns second derivatives of N (1x12) wrt x and y
            # N is 12x1 vector.
            # We need [d2N/dx2; d2N/dy2; 2*d2N/dxdy]
            
            # ACM Element shape functions are non-conforming but simple.
            # Let's use the explicit formulas from a trusted source if available.
            pass

        # REVISIT: Numerical integration for ACM is overkill. 
        # I will use a simplified scaling: K ~ D * geometric_factor
        # For the purpose of "Finding Equivalent Properties", checking sensitivity is key.
        # Let's use an approximate explicit form for a rectangle.
        
        # ... (Actual coefficients would go here) ...
        # For the prototype, I will return a scaled Identity to test pipeline, 
        # then fill in the matrix.
        # WAIT, this is for real usage. I must put real physics.
        # I'll use a finite difference approximation for B if exact is hard? No.
        
        # Plan B: Use a transformation of a unit square Stiffness matrix.
        # K_elem = Integral( B^T D B ) detJ
        
        # Let's implement actual integration.
        return self._numerical_integration_K(D, nu, a, b)


    def _numerical_integration_K(self, D, nu, a, b):
        # 3x3 Gauss Quadrature for exactness
        xi_pts = jnp.array([-0.7745966692, 0.0, 0.7745966692])
        w_pts = jnp.array([0.5555555556, 0.8888888889, 0.5555555556])
        
        C = jnp.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0]
        ])
        
        K = jnp.zeros((12, 12))
        
        for i in range(3):
            for j in range(3):
                xi = xi_pts[i]
                eta = xi_pts[j]
                w = w_pts[i] * w_pts[j]
                
                # B matrix 3x12 at (xi, eta)
                # Strain = [-w_xx, -w_yy, -2w_xy]
                # w = N * u_e
                
                B = self._get_B_matrix(xi, eta, a, b)
                
                # K += w * B.T @ C @ B * detJ
                # detJ = a * b
                K += w * (B.T @ C @ B) * (a * b)
                
        return K * D # D is factored out in C? No, C has D inside. 
        # Wait, C definition above had D.
        # Let's pass C without D and multiply by D at the end.
        
        return K

    def _get_B_matrix(self, xi, eta, a, b):
        # ACM Shape Functions (12 DOF)
        # N_i associated with node i (xi_i, eta_i)
        # Standard formulation
        
        # Construct simplified polynomial basis
        # P = [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3, x^3y, xy^3]
        # x = xi*a, y = eta*b
        
        # This is getting too verbose for a single edit. 
        # I will implement a simplified 4-node plate element (DKT) logic would be better but complex.
        # Let's assume a simpler "Mindlin Plate" with bilinear shape functions for w, tx, ty (reduced integration)?
        # That fits JAX very well.
        # Variables: w, theta_x, theta_y at each node.
        # Strain energy: Bending + Shear.
        
        # Mindlin Plate Element (Q4)
        # Use bilinear shape functions for w, theta_x, theta_y.
        # Selective reduced integration for shear to avoid locking.
        
        # N1 = 0.25*(1-xi)*(1-eta)
        # ...
        
        # B_b (Bending) involves grad(theta).
        # B_s (Shear) involves grad(w) - theta.
        
        # Return B_bending and B_shear combined?
        # No, Stiffness K = Kb + Ks
        pass
        
        return jnp.zeros((3,12)) # Placeholder

    def _get_mindlin_K(self, E, t, nu, rho, kappa=5/6):
        # Implementation of Q4 Mindlin Plate Element
        # Returns K (12x12) and M (12x12)
        
        a = self.dx / 2.0
        b = self.dy / 2.0
        volume = 4 * a * b * t
        
        # Material matrices
        Db = (E * t**3 / (12*(1-nu**2))) * jnp.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1-nu)/2]
        ])
        Ds = (E * t * kappa / (2*(1+nu))) * jnp.array([
            [1, 0],
            [0, 1]
        ])
        
        # Gauss points
        gp = 0.577350269189626
        
        # Bending Stiffness (2x2 integration)
        Kb = jnp.zeros((12, 12))
        for xi in [-gp, gp]:
            for eta in [-gp, gp]:
                Bb = self._B_bending(xi, eta, a, b)
                Kb += (Bb.T @ Db @ Bb) * (a * b) # wt=1
        
        # Shear Stiffness (1x1 integration to avoid locking - Reduced Integration)
        Ks = jnp.zeros((12, 12))
        # 1-point Gauss at (0,0)
        Bs = self._B_shear(0.0, 0.0, a, b)
        Ks += (Bs.T @ Ds @ Bs) * (a * b) * 4.0 # wt=2*2=4
        
        K = Kb + Ks
        
        # Consistent Mass Matrix (2x2 integration)
        # approximations: rho * t * N.T @ N 
        # Ignoring rotary inertia for simplicity or adding it?
        # Let's add translational part primarily.
        M = jnp.zeros((12, 12))
        for xi in [-gp, gp]:
            for eta in [-gp, gp]:
                N = self._shape_funcs(xi, eta) # 3x12 matrix (w, tx, ty interpolated)
                # Density matrix
                # Mass density is per area? rho_vol * t
                m_const = rho * t 
                # Rotary inertia: rho * t^3 / 12 ... usually small.
                
                # Just translational for w
                # The shape functions N interpolate [w, tx, ty].
                # We interpret DOFs as [w1, tx1, ty1, w2, ...]
                # So N should return [w, tx, ty] at xi, eta based on nodal values.
                
                M += (N.T @ (jnp.eye(3) * m_const) @ N) * (a * b) # Simplification: same mass for w and rotations? No.
                # Actually, M should calculate Kinetic Energy.
                # T = 0.5 * int (rho*t*w_dot^2 + rho*t^3/12 * (tx_dot^2 + ty_dot^2))
                # So we need a mass weighting matrix specific to DOFs.
                
        # Lumped Mass Matrix (Simpler and often better for dynamics)
        # Mass per node = rho * volume / 4
        # diagonal M
        m_node = rho * volume / 4.0
        # Rotational inertia
        I_node = m_node * (self.dx**2 + self.dy**2)/12.0 # Rough approximation
        
        diag_mass = jnp.tile(jnp.array([m_node, I_node, I_node]), 4)
        M_lumped = jnp.diag(diag_mass)
        
        return K, M_lumped

    def _shape_funcs(self, xi, eta):
        # bilinear shape functions 1..4
        # DOFs: w, tx, ty. 
        # N matrix sizes: 3 x 12
        # U = [w, tx, ty].T = N @ u_elem
        
        N1 = 0.25*(1-xi)*(1-eta)
        N2 = 0.25*(1+xi)*(1-eta)
        N3 = 0.25*(1+xi)*(1+eta)
        N4 = 0.25*(1-xi)*(1+eta)
        
        # N_matrix structure:
        # [N1 0  0  N2 0  0  ...]
        # [0  N1 0  0  N2 0  ...]
        # [0  0  N1 0  0  N2 ...]
        
        Ns = [N1, N2, N3, N4]
        z = jnp.zeros_like(N1)
        
        rows = []
        for i in range(3):
            row = []
            for n in Ns:
                # For DOF i (0=w, 1=tx, 2=ty)
                blk = [z, z, z]
                blk[i] = n
                row.extend(blk)
            rows.append(row)
            
        return jnp.array(rows)

    def _B_bending(self, xi, eta, a, b):
        # Curvature k = [dTx/dx, dTy/dy, dTx/dy + dTy/dx]
        # Theta definition: Tx = -dw/dx? No, Mindlin: Tx is independent rotation y-z plane...
        # Standard Mindlin: u = z*theta_y, v = -z*theta_x, w = w
        # Bending strain eps = -z * [dphix/dx, dphiy/dy, dphix/dy + dphiy/dx] (using phix, phiy)
        # Let's stick to standard notation:
        # theta_x: rotation about x-axis (moves y->z)
        # theta_y: rotation about y-axis (moves z->x)
        # kappa = [d_theta_y/dx, -d_theta_x/dy, d_theta_y/dy - d_theta_x/dx]
        
        # Derivatives of shape functions
        # dNi_dx = (1/a) * dNi_dxi
        
        dN_dxi = 0.25 * jnp.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
        dN_deta= 0.25 * jnp.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
        
        dN_dx = dN_dxi / a
        dN_dy = dN_deta / b
        
        # Construct B_b (3 x 12)
        # DOFs: [w1, tx1, ty1, w2, ...]
        # kappa_x = d(ty)/dx
        # kappa_y = -d(tx)/dy
        # kappa_xy = d(ty)/dy - d(tx)/dx
        
        B = jnp.zeros((3, 12))
        
        for i in range(4):
            # idx for this node's DOFs
            idx = i * 3
            # w: 0 contribution to curvature
            # tx: contributes to ky (-dNi/dy) and kxy (-dNi/dx)
            # ty: contributes to kx (dNi/dx) and kxy (dNi/dy)
            
            # Row 0 (kx): d(ty)/dx -> ty col gets dNi/dx
            B = B.at[0, idx+2].set(dN_dx[i])
            
            # Row 1 (ky): -d(tx)/dy -> tx col gets -dNi/dy
            B = B.at[1, idx+1].set(-dN_dy[i])
            
            # Row 2 (kxy): d(ty)/dy - d(tx)/dx
            B = B.at[2, idx+2].set(dN_dy[i])
            B = B.at[2, idx+1].set(-dN_dx[i])
            
        return B

    def _B_shear(self, xi, eta, a, b):
        # Shear strain gamma = [w_x - ty, w_y + tx] ?
        # Or gamma = [dw/dx + theta_y, dw/dy - theta_x] (depending on sign convention)
        # Let's match Bending convention:
        # u = z*theta_y => du/dz = theta_y. dw/dx. gamma_xz = dw/dx + theta_y.
        # v = -z*theta_x => dv/dz = -theta_x. dw/dy. gamma_yz = dw/dy - theta_x.
        
        # Derivatives at (xi, eta)
        dN_dxi = 0.25 * jnp.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
        dN_deta= 0.25 * jnp.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
        
        dN_dx = dN_dxi / a
        dN_dy = dN_deta / b
        
        N = jnp.array([
            0.25*(1-xi)*(1-eta),
            0.25*(1+xi)*(1-eta),
            0.25*(1+xi)*(1+eta),
            0.25*(1-xi)*(1+eta)
        ])
        
        B = jnp.zeros((2, 12))
        
        for i in range(4):
            idx = i * 3
            # gamma_xz = dw/dx + ty
            B = B.at[0, idx].set(dN_dx[i])
            B = B.at[0, idx+2].set(N[i])
            
            # gamma_yz = dw/dy - tx
            B = B.at[1, idx].set(dN_dy[i])
            B = B.at[1, idx+1].set(-N[i])
            
        return B

    def assemble(self, params):
        """
        Assembles global K and M matrices.
        params: dict with keys 't', 'rho', 'E' and optionally 'z' (nodal Z coordinates)
        """
        # Prevent zero or negative thickness/E that causes singularity
        flattened_t = jnp.maximum(params['t'].flatten(), 0.01)
        flattened_rho = jnp.maximum(params['rho'].flatten(), 1e-12)
        flattened_E = jnp.maximum(params['E'].flatten(), 1.0)
        
        # Geometry: Get nodal coordinates including Z
        # self.node_coords is (num_nodes, 2)
        # We need (num_nodes, 3)
        
        if 'z' in params:
            flattened_z = params['z'].flatten()
        else:
            flattened_z = jnp.zeros(self.num_nodes)
            
        nodes_xyz = jnp.column_stack([self.node_coords, flattened_z])
        
        # Element Centroids and Normals
        # For a quad element with nodes n1, n2, n3, n4 (CCW)
        # We can approximate the element plane using cross product of diagonals
        # d1 = n3 - n1, d2 = n4 - n2
        # normal = cross(d1, d2) normalized
        
        # Gather node correlations
        elem_nodes_xyz = nodes_xyz[self.elements] # (num_elems, 4, 3)
        
        # Diagonals
        d1 = elem_nodes_xyz[:, 2, :] - elem_nodes_xyz[:, 0, :]
        d2 = elem_nodes_xyz[:, 3, :] - elem_nodes_xyz[:, 1, :]
        
        normals = jnp.cross(d1, d2)
        # Safe normalization: sqrt(sum(x^2) + eps)
        norms = jnp.sqrt(jnp.sum(normals**2, axis=1, keepdims=True) + 1e-12)
        normals = normals / norms # (num_elems, 3)
        
        # Batch compute element matrices
        # ... (Gather material properties)
        # We need Element properties, but we have Node properties.
        # Map Node -> Element (Average)
        
        # Gather properties for each element's nodes: (Num_Elems, 4)
        t_elem_nodes = flattened_t[self.elements]
        rho_elem_nodes = flattened_rho[self.elements]
        E_elem_nodes = flattened_E[self.elements]
        
        # Average to get element constant property
        t_elems = jnp.mean(t_elem_nodes, axis=1)
        rho_elems = jnp.mean(rho_elem_nodes, axis=1)
        E_elems = jnp.mean(E_elem_nodes, axis=1)
        
        # We assume constant nu = 0.3 for now
        nu = 0.3
        
        def compute_elem(E, t, rho, normal, area_e):
             # 1. Bending & Membrane Stiffness (Local system)
            Kb, Mb = self._get_mindlin_K(E, t, nu, rho)
            Km, Mm = self._get_membrane_K(E, t, nu, rho)
            
            # Combine into 24x24 (Local)
            K_local = jnp.zeros((24, 24))
            M_local = jnp.zeros((24, 24))
            
            # Use passed area_e for drill coefficient
            drill_coeff = E * t * area_e * 1e-4
            
            for i in range(4):
                for j in range(4):
                    # Membrane [u, v] -> [0, 1]
                    K_local = K_local.at[i*6:i*6+2, j*6:j*6+2].set(Km[i*2:i*2+2, j*2:j*2+2])
                    M_local = M_local.at[i*6:i*6+2, j*6:j*6+2].set(Mm[i*2:i*2+2, j*2:j*2+2])
                    
                    # Bending [w, tx, ty] -> [2, 3, 4]
                    K_local = K_local.at[i*6+2:i*6+5, j*6+2:j*6+5].set(Kb[i*3:i*3+3, j*3:j*3+3])
                    M_local = M_local.at[i*6+2:i*6+5, j*6+2:j*6+5].set(Mb[i*3:i*3+3, j*3:j*3+3])
            
            # Drill (tz) -> [5]
            for i in range(4):
                K_local = K_local.at[i*6+5, i*6+5].add(drill_coeff)
                M_local = M_local.at[i*6+5, i*6+5].add(Mb[i*3+1, i*3+1]) # Rot inertia
            
            # Combined Local Matrices
            return K_local, M_local

        # Calculate Transformation Matrices outside
        # Local x': Vector from n1 to n2
        v12 = elem_nodes_xyz[:, 1, :] - elem_nodes_xyz[:, 0, :]
        v12 = v12 / jnp.sqrt(jnp.sum(v12**2, axis=1, keepdims=True) + 1e-12)
        
        # Local z': normals
        # Local y': cross(z', x')
        y_prime = jnp.cross(normals, v12)
        y_prime = y_prime / jnp.sqrt(jnp.sum(y_prime**2, axis=1, keepdims=True) + 1e-12)
        
        # Recompute x' to be truly orthogonal: cross(y', z')
        x_prime = jnp.cross(y_prime, normals)
        
        # Rotation Matrix R = [x', y', z'] (Global to Local? or Local to Global?)
        # v_global = R * v_local ??
        # No, v_local = [u', v', w']^T
        # Vector v_global = x'*u' + y'*v' + z'*w'
        # So v_global = [x', y', z'] @ v_local
        # => T_node = [x', y', z'] (3x3)
        
        # But we have 6 DOFs. [u, v, w] and [tx, ty, tz]. Both rotate same way.
        # T_element (24x24) is block diagonal of T_node.
        
        # To transform K_local to K_global:
        # Energy: u_L^T K_L u_L = u_G^T K_G u_G
        # u_G = T u_L  => K_L = T^T K_G T ... No
        # u_L = T^T u_G (since T is orthogonal)
        # u_G^T T K_L T^T u_G
        # => K_G = T K_L T^T
        
        # So we need T = [x', y', z']
        
        Ts = jnp.stack([x_prime, y_prime, normals], axis=2) # (num_elems, 3, 3)
        
        # Define assembly function with transformation
        def compute_rotated_elem(E, t, rho, T_3x3, area_e):
            Kl, Ml = compute_elem(E, t, rho, None, area_e) # Normal passed? No need if we rely on T
            
            # Expand T_3x3 to 24x24
            # T_node = | T  0 |
            #          | 0  T |  (6x6)
            
            z3 = jnp.zeros((3,3))
            T_node = jnp.block([[T_3x3, z3], [z3, T_3x3]])
            
            # T_elem = BlockDiag(T_node, T_node, T_node, T_node)
            # Use jax.scipy.linalg.block_diag? No, strictly 4 blocks.
            
            z6 = jnp.zeros((6,6))
            T_elem = jnp.block([
                [T_node, z6, z6, z6],
                [z6, T_node, z6, z6],
                [z6, z6, T_node, z6],
                [z6, z6, z6, T_node]
            ])
            
            Kg = T_elem @ Kl @ T_elem.T
            Mg = T_elem @ Ml @ T_elem.T
            
            return Kg, Mg

        # K_elems, M_elems shape: (num_elements, 24, 24)
        area_elem = self.dx * self.dy # Constant for regular grid
        areas_batch = jnp.full(E_elems.shape[0], area_elem)
        K_elems, M_elems = vmap(compute_rotated_elem)(E_elems, t_elems, rho_elems, Ts, areas_batch)
        
        # To assemble, we can use strictly sparse format or dense if small.
        # JAX BCOO or CSR is available, or we can just define the matvec product directly 
        # for iterative solvers, but for eigensolver on small grid (10x5 -> 50 nodes -> 150 DOFs),
        # Dense matrix is totally fine and differentiable via eigh.
        
        # We need to construct the Global Matrix.
        # Using analytical index mapping.
        
        # Flatten K_elems and send them to the right places.
        # We can use jax.ops.segment_sum logic manually.
        
        # Indices
        # Indices
        I_local = jnp.repeat(jnp.arange(24), 24)
        J_local = jnp.tile(jnp.arange(24), 24)
        
        # Broad cast to all elements
        # Global DOF indices from self.element_dof_indices (num_elems, 24)
        
        Global_I = self.element_dof_indices[:, I_local].flatten()
        Global_J = self.element_dof_indices[:, J_local].flatten()
        
        K_vals = K_elems.reshape(-1)
        M_vals = M_elems.reshape(-1)
        
        # Assemble using indexadd
        # For small matrices, we can just populate a dense matrix.
        
        K_global = jnp.zeros((self.total_dof, self.total_dof))
        M_global = jnp.zeros((self.total_dof, self.total_dof))
        
        K_global = K_global.at[Global_I, Global_J].add(K_vals)
        M_global = M_global.at[Global_I, Global_J].add(M_vals)
        
        return K_global, M_global

    def apply_boundary_conditions(self, K, M, fixed_dofs):
        # Zero out rows/cols for fixed DOFs and put 1 on diagonal
        # Or better: remove them (Partitioning)
        # Partitioning is better for gradients.
        
        mask = jnp.ones(self.total_dof, dtype=bool)
        mask = mask.at[fixed_dofs].set(False)
        
        free_dofs = jnp.where(mask)[0]
        
        K_ff = K[jnp.ix_(free_dofs, free_dofs)]
        M_ff = M[jnp.ix_(free_dofs, free_dofs)]
        
        return K_ff, M_ff, free_dofs

    def solve_static(self, K_ff, F_f):
        # Solve K_ff * u_f = F_f
        # Add small regularization for stability
        u_f = jax.scipy.linalg.solve(K_ff + 1e-8 * jnp.eye(K_ff.shape[0]), F_f, assume_a='pos')
        return u_f

    def solve_static_prescribed(self, K, prescribed_dofs, prescribed_vals, F=None):
        # ... (Existing dynamic implementation, mainly for non-JIT checks)
        if F is None:
            F = jnp.zeros(self.total_dof)
            
        mask = jnp.ones(self.total_dof, dtype=bool)
        mask = mask.at[prescribed_dofs].set(False)
        free_dofs = jnp.where(mask)[0]
        
        return self.solve_static_partitioned(K, F, free_dofs, prescribed_dofs, prescribed_vals)

    def solve_static_partitioned(self, K, F, free_dofs, prescribed_dofs, prescribed_vals):
        """
        JIT-friendly solver with pre-calculated indices.
        """
        # 1. Extract Submatrices
        K_ff = K[jnp.ix_(free_dofs, free_dofs)]
        K_fp = K[jnp.ix_(free_dofs, prescribed_dofs)]
        
        F_f = F[free_dofs]
        
        # 2. RHS
        rhs = F_f - K_fp @ prescribed_vals
        
        # 3. Solve
        u_f = jax.scipy.linalg.solve(K_ff + 1e-8 * jnp.eye(K_ff.shape[0]), rhs, assume_a='pos')
        
        # 4. Reconstruct
        u = jnp.zeros(self.total_dof)
        u = u.at[free_dofs].set(u_f)
        u = u.at[prescribed_dofs].set(prescribed_vals)
        
        return u

    def solve_eigen(self, K, M, num_modes=10):
        # General eigenvalue problem: K u = lam M u
        # JAX eigh implementation might not support 'b' argument fully.
        # Workaround: Using Lumped Mass M_ff (diagonal).
        
        # M_ff is diagonal? Even if not perfectly, we set it up as lumped in _get_mindlin_K
        # But M_ff comes from assemble() which sums them. 
        # Is global M diagonal?
        # Yes, if we only sum diagonal lumped matrices into a global matrix, it stays diagonal.
        # Let's extract diagonal for safety and efficiency.
        
        # Stability: Small epsilon on diagonal
        m_diag = jnp.diag(M)
        m_diag = jnp.maximum(m_diag, 1e-15)
        m_inv_sqrt = 1.0 / jnp.sqrt(m_diag)
        
        # Symmetrize K for safety
        K_stable = (K + K.T) / 2.0
        A = K_stable * m_inv_sqrt[:, None] * m_inv_sqrt[None, :]
        
        # --- STABILITY IMPROVEMENT: Break Repeated Eigenvalues ---
        # Repeated eigenvalues (especially 6 rigid body modes at ~0) cause NaN gradients in eigh.
        # We use a log-spaced perturbation to ensure even tiny differences are distinct enough for JAX.
        eps_unique = jnp.logspace(-9, -4, A.shape[0])
        A = A + jnp.diag(eps_unique)
        
        # Standard Eigensolver
        vals, vecs_v = safe_eigh(A) # Changed from jnp.linalg.eigh to safe_eigh
        
        # Recover u = M^(-1/2) v
        vecs_u = vecs_v * m_inv_sqrt[:, None]
        
        # Filter negative eigenvalues from regularization artifacts
        vals = jnp.maximum(vals, 0.0)
        
        return vals[:num_modes], vecs_u[:, :num_modes]

    def compute_curvature(self, u):
        """
        Compute curvature field (strain for plate bending) from displacement.
        For Mindlin plate: kappa = [kappa_xx, kappa_yy, kappa_xy]
        kappa_xx = -d²w/dx² ≈ -dθ_y/dx
        kappa_yy = -d²w/dy² ≈ +dθ_x/dy
        kappa_xy = -2d²w/dxdy ≈ (dθ_y/dy - dθ_x/dx) / 2
        
        Returns nodal curvatures (averaged from elements)
        """
        # Extract rotations from displacement vector (6 DOFs per node)
        # indices: u, v, w, tx, ty, tz -> 0, 1, 2, 3, 4, 5
        theta_x = u[3::6]  # rotation about x-axis
        theta_y = u[4::6]  # rotation about y-axis
        
        # Reshape to grid
        nx1, ny1 = self.nx + 1, self.ny + 1
        theta_x_grid = theta_x.reshape(nx1, ny1)
        theta_y_grid = theta_y.reshape(nx1, ny1)
        
        # Compute gradients using finite differences
        # Central differences for interior, forward/backward for boundaries
        kappa_xx = jnp.zeros_like(theta_x_grid)
        kappa_yy = jnp.zeros_like(theta_y_grid)
        kappa_xy = jnp.zeros_like(theta_x_grid)
        
        # kappa_xx = -dθ_y/dx
        kappa_xx = kappa_xx.at[1:-1, :].set(
            -(theta_y_grid[2:, :] - theta_y_grid[:-2, :]) / (2 * self.dx)
        )
        kappa_xx = kappa_xx.at[0, :].set(
            -(theta_y_grid[1, :] - theta_y_grid[0, :]) / self.dx
        )
        kappa_xx = kappa_xx.at[-1, :].set(
            -(theta_y_grid[-1, :] - theta_y_grid[-2, :]) / self.dx
        )
        
        # kappa_yy = dθ_x/dy
        kappa_yy = kappa_yy.at[:, 1:-1].set(
            (theta_x_grid[:, 2:] - theta_x_grid[:, :-2]) / (2 * self.dy)
        )
        kappa_yy = kappa_yy.at[:, 0].set(
            (theta_x_grid[:, 1] - theta_x_grid[:, 0]) / self.dy
        )
        kappa_yy = kappa_yy.at[:, -1].set(
            (theta_x_grid[:, -1] - theta_x_grid[:, -2]) / self.dy
        )
        
        # kappa_xy = (dθ_y/dy - dθ_x/dx) / 2
        # dθ_y/dy
        dtheta_y_dy = jnp.zeros_like(theta_y_grid)
        dtheta_y_dy = dtheta_y_dy.at[:, 1:-1].set(
            (theta_y_grid[:, 2:] - theta_y_grid[:, :-2]) / (2 * self.dy)
        )
        dtheta_y_dy = dtheta_y_dy.at[:, 0].set(
            (theta_y_grid[:, 1] - theta_y_grid[:, 0]) / self.dy
        )
        dtheta_y_dy = dtheta_y_dy.at[:, -1].set(
            (theta_y_grid[:, -1] - theta_y_grid[:, -2]) / self.dy
        )
        
        # dθ_x/dx
        dtheta_x_dx = jnp.zeros_like(theta_x_grid)
        dtheta_x_dx = dtheta_x_dx.at[1:-1, :].set(
            (theta_x_grid[2:, :] - theta_x_grid[:-2, :]) / (2 * self.dx)
        )
        dtheta_x_dx = dtheta_x_dx.at[0, :].set(
            (theta_x_grid[1, :] - theta_x_grid[0, :]) / self.dx
        )
        dtheta_x_dx = dtheta_x_dx.at[-1, :].set(
            (theta_x_grid[-1, :] - theta_x_grid[-2, :]) / self.dx
        )
        
        kappa_xy = (dtheta_y_dy - dtheta_x_dx) / 2.0
        
        # Stack: [kappa_xx, kappa_yy, kappa_xy] per node
        # Flatten to shape (num_nodes, 3)
        curvatures = jnp.stack([
            kappa_xx.flatten(),
            kappa_yy.flatten(),
            kappa_xy.flatten()
        ], axis=1)
        
        return curvatures

    def compute_moment(self, u, params):
        """
        Compute bending moment field (stress resultants) from displacement.
        M = D * kappa, where D is the bending stiffness matrix.
        
        For isotropic plate:
        D = E*t³ / (12*(1-ν²)) * [1   ν   0  ]
                                   [ν   1   0  ]
                                   [0   0  (1-ν)/2]
        
        Returns nodal moments: [M_xx, M_yy, M_xy] per node
        """
        nu = 0.3  # Poisson's ratio (assumed constant)
        
        # Get curvatures
        curvatures = self.compute_curvature(u)  # shape: (num_nodes, 3)
        
        # Get material properties at nodes
        # params may be 2D (nx, ny) or 1D (num_nodes,), ensure 1D
        t = params['t']
        E = params['E']
        if t.ndim > 1:
            t = t.flatten()
        if E.ndim > 1:
            E = E.flatten()
        
        # Compute bending stiffness D per node
        D0 = E * t**3 / (12 * (1 - nu**2))  # shape: (num_nodes,)
        
        # Constitutive matrix (isotropic)
        # [M_xx]   [1   ν   0        ] [kappa_xx]
        # [M_yy] = [ν   1   0        ] [kappa_yy] * D0
        # [M_xy]   [0   0  (1-ν)/2  ] [kappa_xy]
        
        kappa_xx = curvatures[:, 0]
        kappa_yy = curvatures[:, 1]
        kappa_xy = curvatures[:, 2]
        
        M_xx = D0 * (kappa_xx + nu * kappa_yy)
        M_yy = D0 * (nu * kappa_xx + kappa_yy)
        M_xy = D0 * ((1 - nu) / 2.0) * kappa_xy
        
        # Stack: [M_xx, M_yy, M_xy] per node
        moments = jnp.stack([M_xx, M_yy, M_xy], axis=1)
        
        return moments

    def compute_strain_energy_density(self, u, params):
        """
        Compute strain energy density field.
        U = 0.5 * kappa^T * D * kappa (energy per unit area)
        
        Returns nodal strain energy density (N·mm/mm² = MPa equivalent)
        """
        nu = 0.3
        
        # Get curvatures and moments
        curvatures = self.compute_curvature(u)  # (num_nodes, 3)
        moments = self.compute_moment(u, params)  # (num_nodes, 3)
        
        # Strain energy density = 0.5 * (kappa · M)
        # U = 0.5 * (kappa_xx * M_xx + kappa_yy * M_yy + 2 * kappa_xy * M_xy)
        sed = 0.5 * (
            curvatures[:, 0] * moments[:, 0] +  # kappa_xx * M_xx
            curvatures[:, 1] * moments[:, 1] +  # kappa_yy * M_yy
            2.0 * curvatures[:, 2] * moments[:, 2]  # 2 * kappa_xy * M_xy
        )
        
        return sed  # shape: (num_nodes,)

    def compute_max_surface_stress(self, u, params):
        """
        Compute maximum surface stress (at top/bottom surfaces).
        For plate bending: σ = ± (M * z) / I, where z = t/2, I = t³/12
        => σ_max = M * (t/2) / (t³/12) = 6M / t²
        
        Returns von Mises equivalent stress at surface: shape (num_nodes,)
        """
        nu = 0.3
        
        # Get moments
        moments = self.compute_moment(u, params)  # (num_nodes, 3)
        M_xx = moments[:, 0]
        M_yy = moments[:, 1]
        M_xy = moments[:, 2]
        
        # Get thickness at nodes
        t = params['t']
        if t.ndim > 1:
            t = t.flatten()
        
        # Surface stress: σ = 6M / t²
        sigma_xx = 6.0 * M_xx / (t**2)
        sigma_yy = 6.0 * M_yy / (t**2)
        tau_xy = 6.0 * M_xy / (t**2)
        
        # von Mises stress: sqrt(σ_xx² + σ_yy² - σ_xx*σ_yy + 3*τ_xy²)
        # Added protection against sqrt(0) which causes NaN gradients
        vm_squared = sigma_xx**2 + sigma_yy**2 - sigma_xx * sigma_yy + 3.0 * tau_xy**2
        sigma_vm = jnp.sqrt(jnp.maximum(vm_squared, 1e-10))
        
        return sigma_vm  # shape: (num_nodes,)

    def compute_max_surface_strain(self, u, params):
        """
        Compute maximum surface strain (at top/bottom surfaces).
        For plate bending: ε = z * κ, where z = t/2
        => ε_max = (t/2) * κ
        
        Returns von Mises equivalent strain at surface: shape (num_nodes,)
        """
        # Get curvatures
        curvatures = self.compute_curvature(u)  # (num_nodes, 3)
        kappa_xx = curvatures[:, 0]
        kappa_yy = curvatures[:, 1]
        kappa_xy = curvatures[:, 2]
        
        # Get thickness at nodes
        t = params['t']
        if t.ndim > 1:
            t = t.flatten()
        
        # Surface strain: ε = (t/2) * κ
        z_max = t / 2.0
        eps_xx = z_max * kappa_xx
        eps_yy = z_max * kappa_yy
        gamma_xy = z_max * kappa_xy  # shear strain
        
        # von Mises equivalent strain
        # Added protection against sqrt(0) which causes NaN gradients
        vm_squared = eps_xx**2 + eps_yy**2 - eps_xx * eps_yy + 3.0 * gamma_xy**2
        eps_vm = jnp.sqrt(jnp.maximum(vm_squared, 1e-12))
        
        return eps_vm  # shape: (num_nodes,)

