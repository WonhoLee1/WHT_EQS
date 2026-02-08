
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from solver import PlateFEM
import time
import os
import concurrent.futures

# --- Configuration ---
Lx = 1000.0  # mm
Ly = 400.0   # mm
Base_t = 1.0 # mm
Bead_t = 2.0 # mm
Base_E = 200000.0 # MPa
Base_rho = 7.5e-9 # tonne/mm^3

# Resolutions
Nx_high = 50 
Ny_high = 20
Nx_low = 20
Ny_low = 10

# --- Helper Functions ---
def get_thickness_field(X, Y):
    # Scale coordinates (mm)
    # Define "A" shape
    w_stroke = 80.0 # mm stroke width
    
    # 1. Left leg
    # Line from (200, 100) to (500, 350)
    # Dist from line
    p1 = jnp.array([200.0, 50.0])
    p2 = jnp.array([500.0, 350.0])
    
    vec = p2 - p1
    len_sq = jnp.sum(vec**2)
    # Project (X,Y) onto line
    # shape of X: (Nx, Ny)
    
    # Vectorized distance to segment logic
    # P = (X, Y)
    px = X - p1[0]
    py = Y - p1[1]
    
    t_proj = (px * vec[0] + py * vec[1]) / len_sq
    t_proj = jnp.clip(t_proj, 0.0, 1.0)
    
    # Closest point
    cx = p1[0] + t_proj * vec[0]
    cy = p1[1] + t_proj * vec[1]
    
    dist_sq = (X - cx)**2 + (Y - cy)**2
    mask1 = dist_sq < (w_stroke/2)**2
    
    # 2. Right leg (500, 350) to (800, 50)
    p3 = jnp.array([800.0, 50.0])
    vec2 = p3 - p2
    len_sq2 = jnp.sum(vec2**2)
    
    px2 = X - p2[0]
    py2 = Y - p2[1]
    
    t_proj2 = (px2 * vec2[0] + py2 * vec2[1]) / len_sq2
    t_proj2 = jnp.clip(t_proj2, 0.0, 1.0)
    
    cx2 = p2[0] + t_proj2 * vec2[0]
    cy2 = p2[1] + t_proj2 * vec2[1]
    
    dist_sq2 = (X - cx2)**2 + (Y - cy2)**2
    mask2 = dist_sq2 < (w_stroke/2)**2
    
    # 3. Horizontal bar (300, 180) to (700, 180)
    # Simple box
    mask3 = (Y > 180 - w_stroke/2) & (Y < 180 + w_stroke/2) & (X > 350) & (X < 650)
    
    is_bead = mask1 | mask2 | mask3
    return jnp.where(is_bead, Bead_t, Base_t)

def get_density_field(X, Y, seed=42):
    """
    Generate spatially varying density field with 3x2 grid regions.
    Each region has random density within ±30% of base value (7.5e-9).
    """
    np.random.seed(seed)
    
    # Define 3x2 grid boundaries
    x_boundaries = [0, Lx/3, 2*Lx/3, Lx]
    y_boundaries = [0, Ly/2, Ly]
    
    # Generate random density for each of 6 regions (within ±30%)
    base_rho = 7.5e-9
    rho_field = jnp.ones_like(X) * base_rho
    
    for i in range(3):  # X divisions
        for j in range(2):  # Y divisions
            # Random variation: ±30%
            variation = np.random.uniform(-0.3, 0.3)
            region_rho = base_rho * (1 + variation)
            
            # Define region mask
            mask = ((X >= x_boundaries[i]) & (X < x_boundaries[i+1]) &
                    (Y >= y_boundaries[j]) & (Y < y_boundaries[j+1]))
            
            rho_field = jnp.where(mask, region_rho, rho_field)
    
    return rho_field

def get_E_field(X, Y, seed=42):
    """
    Generate spatially varying Young's modulus field with 3x2 grid regions.
    Each region has random E within -50% to 0% of base value (200 GPa).
    Values are always ≤ 200 GPa.
    """
    np.random.seed(seed + 100)  # Different seed from density
    
    # Define 3x2 grid boundaries
    x_boundaries = [0, Lx/3, 2*Lx/3, Lx]
    y_boundaries = [0, Ly/2, Ly]
    
    # Generate random E for each of 6 regions (200 GPa * [0.5, 1.0])
    base_E = 200000.0  # MPa (200 GPa)
    E_field = jnp.ones_like(X) * base_E
    
    for i in range(3):  # X divisions
        for j in range(2):  # Y divisions
            # Random variation: -50% to 0% (always ≤ base_E)
            variation = np.random.uniform(-0.5, 0.0)
            region_E = base_E * (1 + variation)
            
            # Define region mask
            mask = ((X >= x_boundaries[i]) & (X < x_boundaries[i+1]) &
                    (Y >= y_boundaries[j]) & (Y < y_boundaries[j+1]))
            
            E_field = jnp.where(mask, region_E, E_field)
    
    return E_field


# --- Class Hierarchy ---

class LoadCase:
    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight

    def get_bcs(self, fem):
        # Returns (fixed_dofs, fixed_vals, F, [optional] target_u)
        raise NotImplementedError

class TwistCase(LoadCase):
    def __init__(self, name, axis=None, value=1.5, mode='angle', weight=1.0):
        super().__init__(name, weight)
        # Auto-detect axis from name if not provided
        if axis is None:
            if '_x' in name.lower():
                axis = 'x'
            elif '_y' in name.lower():
                axis = 'y'
            else:
                axis = 'x'  # default
        self.axis = axis # 'x' or 'y'
        self.value = value # degrees or Moment N*mm
        self.mode = mode # 'angle' or 'moment'
        
    def get_bcs(self, fem):
        tol = 1e-3
        Lx, Ly = fem.Lx, fem.Ly
        F = jnp.zeros(fem.total_dof)
        fixed_dofs = []
        fixed_vals = []
        
        # Center Fix (Prevent Rigid Body)
        # Fix z=0, ty=0 at center bottom?
        # Just fix one node (0,0) or center to prevent drift if needed.
        # But if prescribing w, we might constrain rigid body.
        
        if self.axis == 'x':
            # Twist about X axis
            # Ends are x=0 and x=L
            left_nodes = jnp.where(fem.node_coords[:, 0] < tol)[0]
            right_nodes = jnp.where(fem.node_coords[:, 0] > Lx - tol)[0]
            
            yc = Ly / 2.0
            
            if self.mode == 'angle':
                angle_rad = self.value * np.pi / 180.0
                
                # Left (x=0): Theta_x = -angle, w = (y-yc)*tan(-angle)
                y_left = fem.node_coords[left_nodes, 1]
                w_left = (y_left - yc) * np.tan(-angle_rad)
                
                for i, node in enumerate(left_nodes):
                    fixed_dofs.extend([node*3 + 0, node*3 + 1])
                    fixed_vals.extend([w_left[i], -angle_rad])

                # Right (x=L): Theta_x = +angle, w = (y-yc)*tan(angle)
                y_right = fem.node_coords[right_nodes, 1]
                w_right = (y_right - yc) * np.tan(angle_rad)
                
                for i, node in enumerate(right_nodes):
                    fixed_dofs.extend([node*3 + 0, node*3 + 1])
                    fixed_vals.extend([w_right[i], angle_rad])
                
                # Fix Ty at one node (0,0) to prevent y-drift
                n0 = 0 
                fixed_dofs.append(n0*3 + 2)
                fixed_vals.append(0.0)

            elif self.mode == 'moment':
                # Apply Moment Mx at ends
                # F vector idx: node*3 + 1 (Theta_x)
                # Distributed moment? Or sum?
                # Value is total moment? Distribute by edge length.
                m_node = self.value / len(right_nodes)
                
                # Left: -Moment? Or +? To twist opposite.
                # Right: +Moment
                F = F.at[right_nodes * 3 + 1].add(m_node)
                F = F.at[left_nodes * 3 + 1].add(-m_node)
                
                # Fix w=0 at center line y=Ly/2 to define axis?
                # Or fix center node (Lx/2, Ly/2) w=0, tx=free, ty=free
                center_node = jnp.argmin((fem.node_coords[:,0]-Lx/2)**2 + (fem.node_coords[:,1]-Ly/2)**2)
                fixed_dofs.extend([center_node*3+0, center_node*3+2]) # Fix w, ty
                fixed_vals.extend([0.0, 0.0])
                # Need to prevent x-rotation rigid body? No, moments cause it.
                # Actually if applying moments, we assume it's balanced?
                # Fix center tx? No that blocks twist.
                # Simply fix one end's rotation and apply moment at other?
                # Let's fix center w, and maybe one point y-rot?
                pass
                
        elif self.axis == 'y':
            # Twist about Y axis (Ends y=0, y=L)
            bot_nodes = jnp.where(fem.node_coords[:, 1] < tol)[0]
            top_nodes = jnp.where(fem.node_coords[:, 1] > Ly - tol)[0]
            xc = Lx / 2.0
            
            if self.mode == 'angle':
                angle_rad = self.value * np.pi / 180.0
                
                # Bot (y=0): Theta_y = -angle? Right-hand rule?
                # Theta_y is rotation about y-axis.
                # w = -(x-xc)*tan(angle)  (check sign)
                
                # Let's define positive twist: z goes up at x>xc
                # Bot: w = (x-xc)*tan(-angle)
                x_bot = fem.node_coords[bot_nodes, 0]
                w_bot = (x_bot - xc) * np.tan(-angle_rad)
                
                for i, node in enumerate(bot_nodes):
                    fixed_dofs.extend([node*3 + 0, node*3 + 2]) # w, ty
                    fixed_vals.extend([w_bot[i], -angle_rad])
                    
                # Top (y=L): w = (x-xc)*tan(angle)
                x_top = fem.node_coords[top_nodes, 0]
                w_top = (x_top - xc) * np.tan(angle_rad)
                
                for i, node in enumerate(top_nodes):
                    fixed_dofs.extend([node*3 + 0, node*3 + 2])
                    fixed_vals.extend([w_top[i], angle_rad])
                    
                # Fix Tx at center
                n0 = 0
                fixed_dofs.append(n0*3+1)
                fixed_vals.append(0.0)

        return jnp.array(fixed_dofs, dtype=jnp.int32), jnp.array(fixed_vals, dtype=jnp.float64), F

class PureBendingCase(LoadCase):
    def __init__(self, name, axis=None, value=3.0, mode='angle', weight=1.0):
        super().__init__(name, weight)
        # Auto-detect axis from name if not provided
        if axis is None:
            if '_x' in name.lower():
                axis = 'x'
            elif '_y' in name.lower():
                axis = 'y'
            else:
                axis = 'y'  # default
        self.axis = axis # bending axis (curvature about this axis)
        self.value = value
        self.mode = mode
        
    def get_bcs(self, fem):
        # Pure Bending about Y axis -> Cylindrical curvature along X
        # Curvature d2w/dx2
        tol = 1e-3
        Lx, Ly = fem.Lx, fem.Ly
        F = jnp.zeros(fem.total_dof)
        fixed_dofs = []
        fixed_vals = []
        
        if self.axis == 'y':
            # Bend Y: Curvature in XZ plane (rotate about Y)
            # Ends x=0, x=Lx
            left_nodes = jnp.where(fem.node_coords[:, 0] < tol)[0]
            right_nodes = jnp.where(fem.node_coords[:, 0] > Lx - tol)[0]
            
            if self.mode == 'angle':
                angle_rad = self.value * np.pi / 180.0
                
                # Smile shape: Left slopes down (Ty < 0?), Right slopes up (Ty > 0?)
                # Vector Ty is rotation about Y.
                # Z = x^2. dw/dx = 2x. 
                # At x=0 (left? no, center is better for symmetry)
                # Let's pivot at center.
                # Left: x=-L/2. Slope -A. Right: x=L/2. Slope +A.
                # Ty = -dw/dx (usually? check solver definition).
                # Solver: theta_y = -dw/dx (Kirchhoff)
                
                # Left: Ty = +angle (if slope is neg)
                # Right: Ty = -angle (if slope is pos)
                
                # Let's enforce:
                # Left (x=0): Ty = angle. w=0
                # Right (x=L): Ty = -angle. w=0
                # This creates a hump (negative curvature)? 
                # angle > 0. Left slope < 0 -> Ty > 0. Correct.
                
                for node in left_nodes:
                    fixed_dofs.extend([node*3+0, node*3+2]) # w, ty
                    fixed_vals.extend([0.0, angle_rad]) 
                    
                for node in right_nodes:
                    fixed_dofs.extend([node*3+0, node*3+2])
                    fixed_vals.extend([0.0, -angle_rad])
                
                # Fix Tx at one node
                fixed_dofs.append(left_nodes[0]*3 + 1)
                fixed_vals.append(0.0)
                
        elif self.axis == 'x':
            # Bend X: Curvature in YZ plane (rotate about X)
            # Ends y=0, y=Ly
            bot_nodes = jnp.where(fem.node_coords[:, 1] < tol)[0]
            top_nodes = jnp.where(fem.node_coords[:, 1] > Ly - tol)[0]
            
            if self.mode == 'angle':
                angle_rad = self.value * np.pi / 180.0
                
                # Bot: Tx = -angle (slope down w.r.t y)
                # Top: Tx = +angle
                
                for node in bot_nodes:
                     fixed_dofs.extend([node*3+0, node*3+1]) # w, tx
                     fixed_vals.extend([0.0, -angle_rad])
                     
                for node in top_nodes:
                     fixed_dofs.extend([node*3+0, node*3+1])
                     fixed_vals.extend([0.0, angle_rad])
                     
                # Fix Ty at one node
                fixed_dofs.append(bot_nodes[0]*3 + 2)
                fixed_vals.append(0.0)

        return jnp.array(fixed_dofs, dtype=jnp.int32), jnp.array(fixed_vals, dtype=jnp.float64), F

class CornerLiftCase(LoadCase):
    def __init__(self, name, corner='br', value=1.0, mode='disp', weight=1.0):
        super().__init__(name, weight)
        self.corner = corner # 'tl', 'tr', 'bl', 'br'
        self.value = value
        self.mode = mode
        
    def get_bcs(self, fem):
        # Fix 3 corners w=0, Lift 1 corner
        # Corners: BL(0,0), BR(L,0), TL(0,L), TR(L,L)
        tol = 1e-3
        coords = fem.node_coords
        
        # Find corner indices
        idx_bl = jnp.argmin(coords[:,0]**2 + coords[:,1]**2)
        idx_br = jnp.argmin((coords[:,0]-fem.Lx)**2 + coords[:,1]**2)
        idx_tl = jnp.argmin(coords[:,0]**2 + (coords[:,1]-fem.Ly)**2)
        idx_tr = jnp.argmin((coords[:,0]-fem.Lx)**2 + (coords[:,1]-fem.Ly)**2)
        
        corners = {'bl': idx_bl, 'br': idx_br, 'tl': idx_tl, 'tr': idx_tr}
        target_idx = corners[self.corner]
        
        fixed_dofs = []
        fixed_vals = []
        F = jnp.zeros(fem.total_dof)
        
        # Fix 3 others w=0
        for k, idx in corners.items():
            if k != self.corner:
                fixed_dofs.append(idx*3 + 0)
                fixed_vals.append(0.0)
        
        if self.mode == 'disp':
            fixed_dofs.append(target_idx*3 + 0)
            fixed_vals.append(self.value)
        elif self.mode == 'force':
            # Fix w=0? No, free w, apply Force
            F = F.at[target_idx*3 + 0].set(self.value)
        
        return jnp.array(fixed_dofs, dtype=jnp.int32), jnp.array(fixed_vals, dtype=jnp.float64), F

# --- Model Manager ---

class EquivalentSheetModel:
    def __init__(self, Lx, Ly, Nx, Ny):
        self.fem = PlateFEM(Lx, Ly, Nx, Ny)
        self.cases = []
        self.targets = [] # Ground Truth Responses
        self.optimized_params = None
        
    def add_case(self, case: LoadCase):
        self.cases.append(case)
        print(f"Added Case: {case.name}")

    def generate_targets(self, resolution_high=(50, 20)):
        print("Generating Ground Truth Targets...")
        self.resolution_high = resolution_high
        fem_high = PlateFEM(self.fem.Lx, self.fem.Ly, resolution_high[0], resolution_high[1])
        
        # Target Params (Fixed)
        X_grid = fem_high.node_coords[:,0].reshape(resolution_high[0]+1, -1)
        Y_grid = fem_high.node_coords[:,1].reshape(resolution_high[0]+1, -1)
        
        t_field = get_thickness_field(X_grid, Y_grid)
        t_field = t_field.flatten() # Ensure 1D
        
        rho_field = get_density_field(X_grid, Y_grid, seed=42)
        rho_field = rho_field.flatten()
        
        E_field = get_E_field(X_grid, Y_grid, seed=42)
        E_field = E_field.flatten()
        
        print(f"High Res Params Shape: t={t_field.shape}, rho={rho_field.shape}, E={E_field.shape}")
        
        params_high = {
            't': t_field,
            'rho': rho_field,
            'E': E_field
        }
        
        K_h, M_h = fem_high.assemble(params_high)
        
        def solve_one(case):
            try:
                print(f"Solving Target: {case.name}")
                fd, fv, F = case.get_bcs(fem_high)
                
                # Check shapes
                # print(f"DEBUG: {case.name} fd={fd.shape} fv={fv.shape} F={F.shape}")
                
                all_dofs = np.arange(fem_high.total_dof)
                free = np.setdiff1d(all_dofs, fd)
                u = fem_high.solve_static_partitioned(K_h, F, jnp.array(free), fd, fv)
                
                # Compute all strain/stress metrics
                curvature = fem_high.compute_curvature(u)  # (num_nodes, 3)
                moment = fem_high.compute_moment(u, params_high)  # (num_nodes, 3)
                strain_energy = fem_high.compute_strain_energy_density(u, params_high)  # (num_nodes,)
                max_stress = fem_high.compute_max_surface_stress(u, params_high)  # (num_nodes,)
                max_strain = fem_high.compute_max_surface_strain(u, params_high)  # (num_nodes,)
                
                return {
                    'case_name': case.name,
                    'weight': case.weight,
                    'u_static': u[0::3],
                    'curvature': curvature,
                    'moment': moment,
                    'strain_energy_density': strain_energy,
                    'max_surface_stress': max_stress,
                    'max_surface_strain': max_strain,
                    'params': params_high
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e
            
        # Sequential Execution (JAX has issues with threading)
        self.targets = []
        for case in self.cases:
            self.targets.append(solve_one(case))
             
        # Eigen (Global)
        print("Solving Target Eigenmodes...")
        vals, vecs = fem_high.solve_eigen(K_h, M_h, num_modes=10)
        self.target_eigen = {
            'vals': vals[3:8],
            'modes': vecs[0::3, 3:8]
        }
        
        # Calculate Target Total Mass (for mass constraint)
        # Mass = integral of (rho * t) over the area
        # Using nodal values and trapezoidal integration
        dx_h = fem_high.Lx / resolution_high[0]
        dy_h = fem_high.Ly / resolution_high[1]
        cell_area = dx_h * dy_h
        
        # Mass per unit area = rho * t (nodal values)
        mass_density = rho_field * t_field  # (num_nodes,)
        
        # Sum over nodes with weighting (corner=1/4, edge=1/2, interior=1)
        Nx_h, Ny_h = resolution_high
        mass_density_2d = mass_density.reshape(Nx_h+1, Ny_h+1)
        
        # Create weight matrix for trapezoidal integration
        weights_2d = np.ones((Nx_h+1, Ny_h+1))
        weights_2d[0, :] *= 0.5
        weights_2d[-1, :] *= 0.5
        weights_2d[:, 0] *= 0.5
        weights_2d[:, -1] *= 0.5
        
        self.target_mass = np.sum(mass_density_2d * weights_2d) * cell_area
        print(f"Target Total Mass: {self.target_mass:.6f} tonne ({self.target_mass * 1e6:.3f} g)")

    def optimize(self, opt_config, loss_weights, use_smoothing=True, use_curvature=False, use_moment=False, use_strain_energy=False, use_surface_stress=False, use_surface_strain=False, use_mass_constraint=False, mass_tolerance=0.05, max_iterations=200):
        print("Starting Optimization...")
        
        # 1. Interpolate Targets to Low Res
        self.targets_low = [] # Store for verification
        from scipy.interpolate import griddata
        
        # Uses detected or stored resolution
        if hasattr(self, 'resolution_high'):
             Nx_h, Ny_h = self.resolution_high
        else:
             print("Warning: resolution_high not found, using default (50, 20)")
             Nx_h, Ny_h = 50, 20
             
        xh = np.linspace(0, self.fem.Lx, Nx_h+1)
        yh = np.linspace(0, self.fem.Ly, Ny_h+1)
        Xh, Yh = np.meshgrid(xh, yh, indexing='ij')
        pts_h = np.column_stack([Xh.flatten(), Yh.flatten()])
        
        xl = self.fem.node_coords
        
        for tgt in self.targets:
            u_h = tgt['u_static'] # (N,)
            u_l = griddata(pts_h, u_h, xl, method='cubic')
            
            tgt_low = {
                'case_name': tgt['case_name'],
                'u_static': jnp.array(u_l),
                'weight': tgt['weight']
            }
            
            # Interpolate curvature if requested
            if use_curvature:
                curv_h = tgt['curvature']  # shape: (Nh, 3)
                curv_l = np.zeros((xl.shape[0], 3))
                for comp in range(3):
                    curv_l[:, comp] = griddata(pts_h, curv_h[:, comp], xl, method='cubic')
                tgt_low['curvature'] = jnp.array(curv_l)
            
            # Interpolate moment if requested
            if use_moment:
                mom_h = tgt['moment']  # shape: (Nh, 3)
                mom_l = np.zeros((xl.shape[0], 3))
                for comp in range(3):
                    mom_l[:, comp] = griddata(pts_h, mom_h[:, comp], xl, method='cubic')
                tgt_low['moment'] = jnp.array(mom_l)
            
            # Interpolate strain energy density if requested
            if use_strain_energy:
                sed_h = tgt['strain_energy_density']  # shape: (Nh,)
                sed_l = griddata(pts_h, sed_h, xl, method='cubic')
                tgt_low['strain_energy_density'] = jnp.array(sed_l)
            
            # Interpolate max surface stress if requested
            if use_surface_stress:
                stress_h = tgt['max_surface_stress']  # shape: (Nh,)
                stress_l = griddata(pts_h, stress_h, xl, method='cubic')
                tgt_low['max_surface_stress'] = jnp.array(stress_l)
            
            # Interpolate max surface strain if requested
            if use_surface_strain:
                strain_h = tgt['max_surface_strain']  # shape: (Nh,)
                strain_l = griddata(pts_h, strain_h, xl, method='cubic')
                tgt_low['max_surface_strain'] = jnp.array(strain_l)
            
            self.targets_low.append(tgt_low)
            
        # Interpolate Modes
        t_vals = self.target_eigen['vals']
        t_modes_h = self.target_eigen['modes']
        t_modes_l = []
        for i in range(5):
             m = griddata(pts_h, t_modes_h[:,i], xl, method='cubic')
             t_modes_l.append(m)
        t_modes_l = jnp.stack(t_modes_l, axis=1)
        
        # 2. Pre-calculate BCs
        bcs_list = []
        all_dofs = np.arange(self.fem.total_dof)
        
        for case, tgt_l in zip(self.cases, self.targets_low):
            fd, fv, F = case.get_bcs(self.fem)
            free = np.setdiff1d(all_dofs, fd)
            bc_dict = {
                'fd': fd, 'fv': fv, 'free': jnp.array(free), 'F': F,
                'target_u': tgt_l['u_static'],
                'weight': case.weight
            }
            
            # Add curvature/moment targets if requested
            if use_curvature and 'curvature' in tgt_l:
                bc_dict['target_curvature'] = tgt_l['curvature']
            if use_moment and 'moment' in tgt_l:
                bc_dict['target_moment'] = tgt_l['moment']
            if use_strain_energy and 'strain_energy_density' in tgt_l:
                bc_dict['target_strain_energy'] = tgt_l['strain_energy_density']
            if use_surface_stress and 'max_surface_stress' in tgt_l:
                bc_dict['target_surface_stress'] = tgt_l['max_surface_stress']
            if use_surface_strain and 'max_surface_strain' in tgt_l:
                bc_dict['target_surface_strain'] = tgt_l['max_surface_strain']
            
            bcs_list.append(bc_dict)
            
        # ... Loss function and Loop (Same as before) ...
        # (I'll keep the rest of the method body consistent)
        
        # Define Loss Code here again? 
        # Since I'm replacing the whole method block, I must allow the rest.
        
        # Pre-calculate mass integration weights for low-res mesh
        dx_l = self.fem.Lx / self.fem.nx
        dy_l = self.fem.Ly / self.fem.ny
        cell_area_l = dx_l * dy_l
        
        weights_mass = jnp.ones((self.fem.nx+1, self.fem.ny+1))
        weights_mass = weights_mass.at[0, :].multiply(0.5)
        weights_mass = weights_mass.at[-1, :].multiply(0.5)
        weights_mass = weights_mass.at[:, 0].multiply(0.5)
        weights_mass = weights_mass.at[:, -1].multiply(0.5)
        
        target_mass = self.target_mass if hasattr(self, 'target_mass') else 1.0
        
        # 3. Loss Function
        def loss_fn(params):
            K, M = self.fem.assemble(params)
            
            # Static Loss (Displacement, Curvature, Moment)
            l_static = 0.0
            l_curvature = 0.0
            l_moment = 0.0
            l_strain_energy = 0.0
            l_surface_stress = 0.0
            l_surface_strain = 0.0
            l_mass = 0.0
            sum_w = 0.0
            
            for bc in bcs_list:
                u = self.fem.solve_static_partitioned(K, bc['F'], bc['free'], bc['fd'], bc['fv'])
                
                # Displacement loss
                z = u[0::3]
                l_static += jnp.mean((z - bc['target_u'])**2) * bc['weight']
                
                # Curvature loss
                if use_curvature and 'target_curvature' in bc:
                    curvature = self.fem.compute_curvature(u)
                    target_curv = bc['target_curvature']
                    # Normalize by typical curvature magnitude
                    curv_scale = jnp.mean(jnp.abs(target_curv)) + 1e-8
                    l_curvature += jnp.mean((curvature - target_curv)**2) / curv_scale * bc['weight']
                
                # Moment loss
                if use_moment and 'target_moment' in bc:
                    moment = self.fem.compute_moment(u, params)
                    target_mom = bc['target_moment']
                    # Normalize by typical moment magnitude
                    mom_scale = jnp.mean(jnp.abs(target_mom)) + 1e-8
                    l_moment += jnp.mean((moment - target_mom)**2) / mom_scale * bc['weight']
                
                # Strain energy density loss
                if use_strain_energy and 'target_strain_energy' in bc:
                    sed = self.fem.compute_strain_energy_density(u, params)
                    target_sed = bc['target_strain_energy']
                    sed_scale = jnp.mean(jnp.abs(target_sed)) + 1e-8
                    l_strain_energy += jnp.mean((sed - target_sed)**2) / sed_scale * bc['weight']
                
                # Max surface stress loss
                if use_surface_stress and 'target_surface_stress' in bc:
                    surface_stress = self.fem.compute_max_surface_stress(u, params)
                    target_stress = bc['target_surface_stress']
                    stress_scale = jnp.mean(jnp.abs(target_stress)) + 1e-8
                    l_surface_stress += jnp.mean((surface_stress - target_stress)**2) / stress_scale * bc['weight']
                
                # Max surface strain loss
                if use_surface_strain and 'target_surface_strain' in bc:
                    surface_strain = self.fem.compute_max_surface_strain(u, params)
                    target_strain = bc['target_surface_strain']
                    strain_scale = jnp.mean(jnp.abs(target_strain)) + 1e-8
                    l_surface_strain += jnp.mean((surface_strain - target_strain)**2) / strain_scale * bc['weight']
                
                sum_w += bc['weight']
            
            l_static /= (sum_w + 1e-6)
            if use_curvature:
                l_curvature /= (sum_w + 1e-6)
            if use_moment:
                l_moment /= (sum_w + 1e-6)
            if use_strain_energy:
                l_strain_energy /= (sum_w + 1e-6)
            if use_surface_stress:
                l_surface_stress /= (sum_w + 1e-6)
            if use_surface_strain:
                l_surface_strain /= (sum_w + 1e-6)
            
            # Mass Constraint Loss
            if use_mass_constraint:
                # Compute current total mass
                t_field = params['t']  # shape: (nx+1, ny+1)
                rho_field = params['rho']  # shape: (nx+1, ny+1)
                mass_density = rho_field * t_field
                current_mass = jnp.sum(mass_density * weights_mass) * cell_area_l
                
                # Relative mass error (soft constraint)
                # Penalize deviations beyond tolerance
                mass_error_rel = jnp.abs(current_mass - target_mass) / target_mass
                
                # Use smooth penalty: quadratic within tolerance, linear beyond
                # This allows small deviations within tolerance
                l_mass = jnp.where(
                    mass_error_rel <= mass_tolerance,
                    (mass_error_rel / mass_tolerance) ** 2,  # Soft penalty within tolerance
                    2.0 * mass_error_rel / mass_tolerance - 1.0  # Linear penalty beyond
                )
            
            # Dynamic Loss
            l_freq = 0.0
            l_mode = 0.0
            
            if loss_weights['freq'] > 0 or loss_weights['mode'] > 0:
                vals, vecs = self.fem.solve_eigen(K, M, num_modes=5)
                my_vals = vals[3:8]
                my_modes = vecs[0::3, 3:8]
                
                if loss_weights['freq'] > 0:
                    l_freq = jnp.mean((my_vals - t_vals)**2) / (jnp.mean(t_vals)**2) * 10.0
                    
                if loss_weights['mode'] > 0:
                    l_m = 0.0
                    for i in range(5):
                        v1 = my_modes[:, i] / jnp.linalg.norm(my_modes[:, i])
                        v2 = t_modes_l[:, i] / jnp.linalg.norm(t_modes_l[:, i])
                        mac = (jnp.dot(v1, v2))**2
                        l_m += (1.0 - mac)
                    l_mode = l_m
            
            # Regularization Loss (Total Variation - smoothness penalty)
            l_reg = 0.0
            if loss_weights.get('reg', 0.0) > 0:
                # Total Variation: penalize large gradients
                # Sum of absolute differences between adjacent cells
                for key in ['t', 'rho', 'E']:
                    field = params[key]  # shape: (nx, ny)
                    # Horizontal differences
                    diff_x = jnp.sum(jnp.abs(field[1:, :] - field[:-1, :]))
                    # Vertical differences
                    diff_y = jnp.sum(jnp.abs(field[:, 1:] - field[:, :-1]))
                    # Normalize by field size
                    field_size = field.shape[0] * field.shape[1]
                    l_reg += (diff_x + diff_y) / field_size
                
                # Normalize by number of fields (3)
                l_reg /= 3.0
            
            total = (loss_weights['static'] * l_static + 
                     loss_weights['freq'] * l_freq + 
                     loss_weights['mode'] * l_mode +
                     loss_weights.get('curvature', 0.0) * l_curvature +
                     loss_weights.get('moment', 0.0) * l_moment +
                     loss_weights.get('strain_energy', 0.0) * l_strain_energy +
                     loss_weights.get('surface_stress', 0.0) * l_surface_stress +
                     loss_weights.get('surface_strain', 0.0) * l_surface_strain +
                     loss_weights.get('reg', 0.0) * l_reg +
                     loss_weights.get('mass', 0.0) * l_mass)
            return total, (l_static, l_freq, l_mode, l_curvature, l_moment, l_strain_energy, l_surface_stress, l_surface_strain, l_reg, l_mass)

        loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        
        params = {
            't': jnp.full((self.fem.nx+1, self.fem.ny+1), Base_t),
            'rho': jnp.full((self.fem.nx+1, self.fem.ny+1), Base_rho),
            'E': jnp.full((self.fem.nx+1, self.fem.ny+1), Base_E)
        }
        
        m = jax.tree_util.tree_map(jnp.zeros_like, params)
        v = jax.tree_util.tree_map(jnp.zeros_like, params)
        beta1, beta2, lr = 0.9, 0.999, 1e-2
        epsilon = 1e-8
        
        kernel = jnp.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
        kernel /= kernel.sum()

        @jax.jit
        def step(p, m, v, i):
            (L, aux), grads = loss_grad_fn(p)
            
            if use_smoothing:
                grads = jax.tree_util.tree_map(
                    lambda g: jax.scipy.signal.convolve2d(g, kernel, mode='same') if g.ndim==2 else g, 
                    grads
                )
            
            for key in p:
                if not opt_config[key]['opt']:
                    grads[key] = jnp.zeros_like(grads[key])
            
            m = jax.tree_util.tree_map(lambda m_i, g_i: beta1*m_i + (1-beta1)*g_i, m, grads)
            v = jax.tree_util.tree_map(lambda v_i, g_i: beta2*v_i + (1-beta2)*g_i**2, v, grads)
            
            m_hat = jax.tree_util.tree_map(lambda x: x/(1-beta1**(i+1)), m)
            v_hat = jax.tree_util.tree_map(lambda x: x/(1-beta2**(i+1)), v)
            
            p_new = {}
            for key in p:
                delta = lr * m_hat[key] / (jnp.sqrt(v_hat[key]) + epsilon)
                val = p[key] - delta
                val = jnp.clip(val, opt_config[key]['min'], opt_config[key]['max'])
                p_new[key] = val
                
            return p_new, m, v, L, aux
            
        print(f"{'Iter':<5} | {'Loss':<10} | {'Static':<10} | {'Freq':<10} | {'Mode':<10} | {'Curvat':<10} | {'Moment':<10} | {'StrainE':<10} | {'SurfStr':<10} | {'SurfEps':<10} | {'Reg':<10} | {'Mass':<10}")
        
        print_interval = max(1, max_iterations // 10)  # Print ~10 times during optimization
        for i in range(max_iterations + 1):
            params, m, v, L, aux = step(params, m, v, i)
            if i % print_interval == 0 or i == max_iterations:
                print(f"{i:<5} | {L:<10.4f} | {aux[0]:<10.4f} | {aux[1]:<10.4f} | {aux[2]:<10.4f} | {aux[3]:<10.4f} | {aux[4]:<10.4f} | {aux[5]:<10.4f} | {aux[6]:<10.4f} | {aux[7]:<10.4f} | {aux[8]:<10.4f} | {aux[9]:<10.4f}")
                
        self.optimized_params = params
        return params

    def verify(self):
        print("Running Verification Plots (High Resolution)...")
        
        # 1. Setup High Res Verification Model
        if hasattr(self, 'resolution_high'):
             Nx_v, Ny_v = self.resolution_high
        else:
             Nx_v, Ny_v = 50, 20
             
        fem_verify = PlateFEM(self.fem.Lx, self.fem.Ly, Nx_v, Ny_v)
        
        # 2. Interpolate Optimized Params to High Res
        from scipy.interpolate import griddata
        
        # Low Res Coords
        xl = np.linspace(0, self.fem.Lx, self.fem.nx+1)
        yl = np.linspace(0, self.fem.Ly, self.fem.ny+1)
        Xl, Yl = np.meshgrid(xl, yl, indexing='ij')
        
        # Element Centers for Parameters?
        # Solver uses nodal params averaged to elements.
        # But optimized_params are separate fields t, rho, E at NODES (based on init).
        # Let's check init: params are full((nx, ny)). Wait, PlateFEM init uses nodal or element?
        # In optimize(), params[key] has shape (nx, ny).
        # But solver.assemble expects inputs that flattened match nodes?
        # Let's check assemble call in optimize: self.fem.assemble(params).
        # And params was init as (nx, ny).
        # PlateFEM logic:
        #   assemble calls params['t'].flatten().
        #   Then averages to elements.
        #   But params shape (nx, ny) has size nx*ny = num_elements.
        #   Ah! params in optimize ARE element-based or node-based?
        #   Init in optimize: jnp.full((self.fem.nx, self.fem.ny), ...).
        #   This is nx * ny. 
        #   But num_nodes = (nx+1)*(ny+1).
        #   So currently optimization is controlling ELEMENT properties directly?
        #   Let's check solver.assemble logic again.
        #     flattened_t = params['t'].flatten()
        #     flattened_rho = params['rho'].flatten()
        #     ... 
        #     t_elem_nodes = flattened_t[self.elements]
        #     This implies flattened_t MUST be Nodal values! (Size Num_Nodes)
        #     If input is (nx, ny), flattening gives nx*ny values.
        #     self.elements contains indices up to (nx+1)*(ny+1).
        #     IndexError would occur if params are smaller than nodes!
        #
        # WAIT. In optimize init:
        #   params = { 't': jnp.full((self.fem.nx, self.fem.ny), Base_t) }
        #   This creates arrays of size elements.
        #   If assemble expects nodes, this code must be crashing or I misunderstood current solver.
        #
        #   Let's look at solver.py again or previous Step 797.
        #   Line 408: flattened_t = params['t'].flatten()
        #   Line 417: t_elem_nodes = flattened_t[self.elements]
        #   self.elements has max index = num_nodes - 1.
        #   If params['t'] has size nx*ny, and num_nodes > nx*ny, this WILL FAIL.
        #   
        #   Did the previous run rely on broadcasting or something?
        #   The previous run worked... how?
        #   Maybe I init params with correct shape?
        #   In Step 760: params = { 't': jnp.full((self.fem.nx, self.fem.ny), Base_t) ... }
        #   Wait, num_nodes = (nx+1)*(ny+1).
        #   So params are definitely too small.
        #   Why did Step 810 succeed? "Start Optimization" printed. "Iter 0" printed.
        #   Maybe self.fem.nx and ny are high res? No, Nx_low.
        #
        #   Hypothesis: implicit behavior or I missed where params shape is set.
        #   Let's CORRECT this now regardless. Params should be Nodal to allow smooth interpolation.
        #   Or Element-based.
        #   If Element-based, we don't map node->elem.
        #   The solver assemble assumes Nodal input currently (t_elem_nodes = ...).
        #   
        #   So I should init params as (nx+1, ny+1).
        #   I will fix that in optimize() first? No, verify() is the task.
        #   But I need to interpolate whatever 'params' is.
        #   I will assume params corresponds to Low Res Nodes (size (nx+1, ny+1)).
        
        #   Recalculate Grid for Interpolation:
        #   If optimized_params are (nx, ny), I should fix optimize too?
        #   Let's assume for verification I treat them as "whatever shape matches low res".
        #   I will define grid_l based on params shape.
        
        param_shape = self.optimized_params['t'].shape
        if param_shape == (self.fem.nx, self.fem.ny):
             # Element centers
             xc = np.linspace(self.fem.dx/2, self.fem.Lx - self.fem.dx/2, self.fem.nx)
             yc = np.linspace(self.fem.dy/2, self.fem.Ly - self.fem.dy/2, self.fem.ny)
             X_src, Y_src = np.meshgrid(xc, yc, indexing='ij')
        else:
             # Nodes
             xn = np.linspace(0, self.fem.Lx, self.fem.nx+1)
             yn = np.linspace(0, self.fem.Ly, self.fem.ny+1)
             X_src, Y_src = np.meshgrid(xn, yn, indexing='ij')
             
        pts_src = np.column_stack([X_src.flatten(), Y_src.flatten()])
        
        # Destination: High Res Nodes
        xv = np.linspace(0, self.fem.Lx, Nx_v+1)
        yv = np.linspace(0, self.fem.Ly, Ny_v+1)
        X_dst, Y_dst = np.meshgrid(xv, yv, indexing='ij')
        pts_dst = np.column_stack([X_dst.flatten(), Y_dst.flatten()])
        
        params_v = {}
        for k in ['t', 'rho', 'E']:
             val_l = self.optimized_params[k].flatten()
             val_v = griddata(pts_src, val_l, pts_dst, method='cubic', fill_value=np.mean(val_l))
             # If element-based source, extrapolate might be needed? 
             # Cubic handles it usually, or use nearest for boundary.
             # Actually griddata isn't great for extrapolation.
             # 'nearest' is safer for edges. Or 'linear'.
             # Let's use 'cubic' but fallback? 
             # Actually if Source is Element Centers, we are inside the domain. Nodes are boundary.
             # Element centers don't cover 0 or L. 
             # So nodes will require extrapolation.
             if param_shape == (self.fem.nx, self.fem.ny):
                 # Fix extrapolation: use 'nearest' for outside?
                 # Or just use MapCoordinates?
                 pass
             params_v[k] = jnp.array(val_v) # Should be (Num_Nodes_High,)
             
        # 3. Assemble High Res
        K_v, M_v = fem_verify.assemble(params_v)
        
        # 4. Verify Matches
        # Using self.targets (High Res) directly
        
        x_plt = np.linspace(0, self.fem.Lx, Nx_v+1)
        y_plt = np.linspace(0, self.fem.Ly, Ny_v+1)
        
        for i, case in enumerate(self.cases):
            tgt = self.targets[i] # High Res Target
            
            # Solve Optimized High Res
            fd, fv, F = case.get_bcs(fem_verify)
            all_dofs = np.arange(fem_verify.total_dof)
            free = np.setdiff1d(all_dofs, fd)
            
            u = fem_verify.solve_static_partitioned(K_v, F, jnp.array(free), fd, fv)
            
            # Displacement
            z_opt = u[0::3].reshape(Nx_v+1, Ny_v+1)
            z_ref = tgt['u_static'].reshape(Nx_v+1, Ny_v+1)
            
            # Compute max surface stress and strain (physically meaningful)
            stress_opt = fem_verify.compute_max_surface_stress(u, params_v)  # von Mises stress
            strain_opt = fem_verify.compute_max_surface_strain(u, params_v)  # von Mises strain
            
            # Get target values
            stress_ref = tgt['max_surface_stress']
            strain_ref = tgt['max_surface_strain']
            
            # Reshape for plotting
            stress_opt = stress_opt.reshape(Nx_v+1, Ny_v+1)
            stress_ref = stress_ref.reshape(Nx_v+1, Ny_v+1)
            strain_opt = strain_opt.reshape(Nx_v+1, Ny_v+1)
            strain_ref = strain_ref.reshape(Nx_v+1, Ny_v+1)
            
            # Create 3x3 plot
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            
            def get_robust_levels(data, n_sigma=2.0, num_levels=31):
                """Calculate robust min/max using Mean +/- N*Sigma and return levels."""
                mu = np.mean(data)
                sigma = np.std(data)
                vmin = max(np.min(data), mu - n_sigma * sigma)
                vmax = min(np.max(data), mu + n_sigma * sigma)
                # Ensure vmin < vmax
                if vmax <= vmin:
                    vmin, vmax = np.min(data), np.max(data)
                if vmax <= vmin: # Still same? (e.g. constant field)
                    vmax = vmin + 1e-8
                return np.linspace(vmin, vmax, num_levels)

            # Determine consistent robust levels for all plots based on target data
            disp_levels = get_robust_levels(z_ref, n_sigma=2.5) # Displacement is usually smooth, use higher sigma
            strain_levels = get_robust_levels(strain_ref * 1000, n_sigma=2.0) # Stress/Strain have peaks, use 2 sigma
            stress_levels = get_robust_levels(stress_ref, n_sigma=2.0)
            
            def add_stats_text(ax, data_ref, data_opt, unit=""):
                """Add actual Min/Max text at the bottom of the axis."""
                # We show stats for BOTH to help user see if absolute peaks matched
                txt = (f"Target: min={np.min(data_ref):.3f}, max={np.max(data_ref):.3f} {unit}\n"
                       f"Optimized: min={np.min(data_opt):.3f}, max={np.max(data_opt):.3f} {unit}")
                # Position below x-axis (transform=ax.transAxes uses 0-1 coordinate system)
                ax.text(0.5, -0.15, txt, transform=ax.transAxes, 
                        ha='center', va='top', fontsize=8, color='darkblue')

            # Row 1: Displacement
            im0 = axes[0, 0].contourf(x_plt, y_plt, z_ref.T, levels=disp_levels, cmap='jet', extend='both')
            axes[0, 0].set_title(f"{case.name} - Displacement Target (mm)")
            axes[0, 0].set_aspect('equal')
            plt.colorbar(im0, ax=axes[0, 0])
            
            im1 = axes[0, 1].contourf(x_plt, y_plt, z_opt.T, levels=disp_levels, cmap='jet', extend='both')
            axes[0, 1].set_title(f"{case.name} - Displacement Optimized (mm)")
            axes[0, 1].set_aspect('equal')
            plt.colorbar(im1, ax=axes[0, 1])
            add_stats_text(axes[0, 1], z_ref, z_opt, "mm")
            
            im2 = axes[0, 2].contourf(x_plt, y_plt, np.abs(z_opt - z_ref).T, levels=30, cmap='magma')
            axes[0, 2].set_title("Displacement Error (mm)")
            axes[0, 2].set_aspect('equal')
            plt.colorbar(im2, ax=axes[0, 2])
            
            # Row 2: Max Surface Strain (ε = z·κ at top/bottom surface)
            im3 = axes[1, 0].contourf(x_plt, y_plt, strain_ref.T * 1000, levels=strain_levels, cmap='viridis', extend='both')
            axes[1, 0].set_title("Max Surface Strain Target (×10⁻³)")
            axes[1, 0].set_aspect('equal')
            plt.colorbar(im3, ax=axes[1, 0])
            
            im4 = axes[1, 1].contourf(x_plt, y_plt, strain_opt.T * 1000, levels=strain_levels, cmap='viridis', extend='both')
            axes[1, 1].set_title("Max Surface Strain Optimized (×10⁻³)")
            axes[1, 1].set_aspect('equal')
            plt.colorbar(im4, ax=axes[1, 1])
            add_stats_text(axes[1, 1], strain_ref * 1000, strain_opt * 1000, "×10⁻³")
            
            im5 = axes[1, 2].contourf(x_plt, y_plt, np.abs(strain_opt - strain_ref).T * 1000, levels=30, cmap='magma')
            axes[1, 2].set_title("Strain Error (×10⁻³)")
            axes[1, 2].set_aspect('equal')
            plt.colorbar(im5, ax=axes[1, 2])
            
            # Row 3: Max Surface Stress (σ = 6M/t² von Mises)
            im6 = axes[2, 0].contourf(x_plt, y_plt, stress_ref.T, levels=stress_levels, cmap='plasma', extend='both')
            axes[2, 0].set_title("Max Surface Stress Target (MPa)")
            axes[2, 0].set_aspect('equal')
            plt.colorbar(im6, ax=axes[2, 0])
            
            im7 = axes[2, 1].contourf(x_plt, y_plt, stress_opt.T, levels=stress_levels, cmap='plasma', extend='both')
            axes[2, 1].set_title("Max Surface Stress Optimized (MPa)")
            axes[2, 1].set_aspect('equal')
            plt.colorbar(im7, ax=axes[2, 1])
            add_stats_text(axes[2, 1], stress_ref, stress_opt, "MPa")
            
            im8 = axes[2, 2].contourf(x_plt, y_plt, np.abs(stress_opt - stress_ref).T, levels=30, cmap='magma')
            axes[2, 2].set_title("Stress Error (MPa)")
            axes[2, 2].set_aspect('equal')
            plt.colorbar(im8, ax=axes[2, 2])
            
            plt.tight_layout()
            filename = f"verify_{case.name}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Saved {os.path.abspath(filename)} (High Res)")
            
        # 4.5 Visualize Optimized Parameters
        print("Plotting Optimized Parameter Distributions...")
        
        # Get High Res interpolated params
        t_opt_highres = params_v['t'].reshape(Nx_v+1, Ny_v+1)
        rho_opt_highres = params_v['rho'].reshape(Nx_v+1, Ny_v+1)
        E_opt_highres = params_v['E'].reshape(Nx_v+1, Ny_v+1)
        
        # Get Target params for comparison
        t_target = self.targets[0]['params']['t'].reshape(Nx_v+1, Ny_v+1)
        rho_target = self.targets[0]['params']['rho'].reshape(Nx_v+1, Ny_v+1)
        E_target = self.targets[0]['params']['E'].reshape(Nx_v+1, Ny_v+1)
        
        fig, axes = plt.figure(figsize=(18, 10)), None
        axes = fig.subplots(2, 3)
        
        # Row 1: Target (Discrete visualization)
        im0 = axes[0, 0].pcolormesh(x_plt, y_plt, t_target.T, cmap='viridis', shading='nearest')
        axes[0, 0].set_title('Target Thickness (mm)')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0, 0])
        
        im1 = axes[0, 1].pcolormesh(x_plt, y_plt, rho_target.T * 1e9, cmap='plasma', shading='nearest')
        axes[0, 1].set_title('Target Density (×10⁻⁹ tonne/mm³)')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 1])
        
        im2 = axes[0, 2].pcolormesh(x_plt, y_plt, E_target.T / 1000, cmap='inferno', shading='nearest')
        axes[0, 2].set_title('Target Young\'s Modulus (GPa)')
        axes[0, 2].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Row 2: Optimized (Discrete visualization)
        im3 = axes[1, 0].pcolormesh(x_plt, y_plt, t_opt_highres.T, cmap='viridis', shading='nearest')
        axes[1, 0].set_title('Optimized Thickness (mm)')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(im3, ax=axes[1, 0])
        
        im4 = axes[1, 1].pcolormesh(x_plt, y_plt, rho_opt_highres.T * 1e9, cmap='plasma', shading='nearest')
        axes[1, 1].set_title('Optimized Density (×10⁻⁹ tonne/mm³)')
        axes[1, 1].set_aspect('equal')
        plt.colorbar(im4, ax=axes[1, 1])
        
        im5 = axes[1, 2].pcolormesh(x_plt, y_plt, E_opt_highres.T / 1000, cmap='inferno', shading='nearest')
        axes[1, 2].set_title('Optimized Young\'s Modulus (GPa)')
        axes[1, 2].set_aspect('equal')
        plt.colorbar(im5, ax=axes[1, 2])
        
        plt.tight_layout()
        filename_params = "verify_parameters.png"
        plt.savefig(filename_params)
        plt.close()
        print(f"Saved {os.path.abspath(filename_params)}")
            
        # 5. Verify Modes
        print("Verifying Modes (High Res)...")
        vals, vecs = fem_verify.solve_eigen(K_v, M_v, num_modes=10)
        
        # Target modes
        t_vals = self.target_eigen['vals']
        t_modes = self.target_eigen['modes']
        
        # Opt Modes
        o_vals = vals[3:8] # Skip 3 rigid
        o_modes = vecs[0::3, 3:8]
        
        num_modes_plot = min(5, len(o_vals), len(t_vals))
        
        freq_t = np.sqrt(np.abs(t_vals[:num_modes_plot])) / (2*np.pi)
        freq_o = np.sqrt(np.abs(o_vals[:num_modes_plot])) / (2*np.pi)
        
        macs = []
        for j in range(num_modes_plot):
            v1 = o_modes[:, j] / jnp.linalg.norm(o_modes[:, j])
            v2 = t_modes[:, j] / jnp.linalg.norm(t_modes[:, j])
            macs.append((jnp.dot(v1, v2))**2)
            
        # Plot Frequency & MAC comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        x_idx = np.arange(1, num_modes_plot+1)
        w = 0.35
        plt.bar(x_idx - w/2, freq_t, width=w, label='Target', color='gray')
        plt.bar(x_idx + w/2, freq_o, width=w, label='Opt (High Res)', color='red')
        plt.legend()
        plt.title('Frequency Comparison (Hz)')
        plt.xlabel('Mode')
        plt.ylabel('Frequency (Hz)')
        
        plt.subplot(1, 2, 2)
        plt.bar(x_idx, macs, color='purple')
        plt.ylim(0, 1.1)
        plt.axhline(0.9, color='k', linestyle='--')
        plt.title('MAC')
        plt.xlabel('Mode')
        plt.ylabel('MAC Value')
        
        plt.tight_layout()
        plt.savefig("verify_modes.png")
        plt.close()
        print(f"Saved {os.path.abspath('verify_modes.png')}")
        
        # Plot Mode Shapes
        print("Plotting Mode Shapes...")
        fig = plt.figure(figsize=(20, 8))
        
        for j in range(num_modes_plot):
            # Target Mode
            ax1 = fig.add_subplot(2, num_modes_plot, j+1)
            mode_t = t_modes[:, j].reshape(Nx_v+1, Ny_v+1)
            im = ax1.contourf(x_plt, y_plt, mode_t.T, levels=20, cmap='RdBu_r')
            ax1.set_title(f'Target Mode {j+1}\nf={freq_t[j]:.2f} Hz', fontsize=9)
            ax1.axis('equal')
            plt.colorbar(im, ax=ax1, fraction=0.046)
            
            # Optimized Mode
            ax2 = fig.add_subplot(2, num_modes_plot, num_modes_plot + j+1)
            mode_o = o_modes[:, j].reshape(Nx_v+1, Ny_v+1)
            im = ax2.contourf(x_plt, y_plt, mode_o.T, levels=20, cmap='RdBu_r')
            ax2.set_title(f'Opt Mode {j+1}\nf={freq_o[j]:.2f} Hz\nMAC={macs[j]:.3f}', fontsize=9)
            ax2.axis('equal')
            plt.colorbar(im, ax=ax2, fraction=0.046)
        
        plt.tight_layout()
        filename_modeshapes = "verify_mode_shapes.png"
        plt.savefig(filename_modeshapes)
        plt.close()
        print(f"Saved {os.path.abspath(filename_modeshapes)}")
        
        # ============================================================
        # 6. Generate Text Report
        # ============================================================
        print("\n" + "="*80)
        print("VERIFICATION REPORT")
        print("="*80)
        
        report_lines = []
        report_lines.append("# Equivalent Sheet Model - Verification Report")
        report_lines.append("")
        
        # --- 6.1 Modal Analysis Comparison ---
        report_lines.append("## 1. Modal Analysis Comparison")
        report_lines.append("")
        report_lines.append("| Mode | Target Freq (Hz) | Opt Freq (Hz) | Freq Error (%) | MAC |")
        report_lines.append("|------|------------------|---------------|----------------|-----|")
        
        for j in range(num_modes_plot):
            freq_err = abs(freq_o[j] - freq_t[j]) / freq_t[j] * 100 if freq_t[j] > 0 else 0
            mac_val = float(macs[j])
            report_lines.append(f"| {j+1} | {freq_t[j]:.4f} | {freq_o[j]:.4f} | {freq_err:.2f} | {mac_val:.4f} |")
        
        avg_freq_err = np.mean([abs(freq_o[j] - freq_t[j]) / freq_t[j] * 100 for j in range(num_modes_plot)])
        avg_mac = np.mean([float(macs[j]) for j in range(num_modes_plot)])
        
        report_lines.append(f"| **AVG** | - | - | **{avg_freq_err:.2f}** | **{avg_mac:.4f}** |")
        report_lines.append("")
        
        # Mode-by-mode detailed analysis
        report_lines.append("### Mode-by-Mode Quality Assessment")
        report_lines.append("")
        for j in range(num_modes_plot):
            mac_val = float(macs[j])
            freq_err_j = abs(freq_o[j] - freq_t[j]) / freq_t[j] * 100 if freq_t[j] > 0 else 0
            mac_quality = "Excellent" if mac_val >= 0.95 else "Good" if mac_val >= 0.90 else "Acceptable" if mac_val >= 0.80 else "Poor"
            freq_quality = "Excellent" if freq_err_j < 2 else "Good" if freq_err_j < 5 else "Acceptable" if freq_err_j < 10 else "Poor"
            report_lines.append(f"- **Mode {j+1}**: f_tgt={freq_t[j]:.4f}Hz, f_opt={freq_o[j]:.4f}Hz, Δf={freq_err_j:.2f}% ({freq_quality}), MAC={mac_val:.4f} ({mac_quality})")
        report_lines.append("")
        report_lines.append("> **MAC Interpretation:** ≥0.95: Excellent | ≥0.90: Good | ≥0.80: Acceptable | <0.80: Poor")
        report_lines.append("")
        
        print("\n".join(report_lines[-15:]))  # Print modal section
        
        # --- 6.2 Static Analysis Comparison ---
        report_lines.append("## 2. Static Analysis Comparison")
        report_lines.append("")
        
        # Helper function: Normalized RMSE -> Similarity %
        def calc_similarity(ref, opt):
            """Calculate similarity as 100% - NRMSE (normalized by range)"""
            rmse = np.sqrt(np.mean((ref - opt)**2))
            data_range = np.max(ref) - np.min(ref)
            if data_range < 1e-12:
                return 100.0 if rmse < 1e-12 else 0.0
            nrmse = rmse / data_range
            similarity = max(0.0, (1.0 - nrmse) * 100)
            return similarity
        
        # Helper function: R² (Coefficient of Determination)
        def calc_r2(ref, opt):
            """Calculate R² score (1.0 = perfect, 0.0 = mean prediction)"""
            ss_res = np.sum((ref - opt)**2)
            ss_tot = np.sum((ref - np.mean(ref))**2)
            if ss_tot < 1e-12:
                return 1.0 if ss_res < 1e-12 else 0.0
            r2 = 1.0 - ss_res / ss_tot
            return max(0.0, r2) * 100  # Return as percentage
        
        # Helper function: Robust statistics (Mean + N*Sigma)
        def calc_robust_max(data, n_sigma=2.5):
            """Calculate robust max using Mean + N*Sigma to exclude outliers"""
            mu = np.mean(data)
            sigma = np.std(data)
            robust_max = mu + n_sigma * sigma
            actual_max = np.max(data)
            return min(robust_max, actual_max)  # Don't exceed actual max
        
        static_results = []
        
        for i, case in enumerate(self.cases):
            tgt = self.targets[i]
            
            # Solve for optimized model
            fd, fv, F = case.get_bcs(fem_verify)
            all_dofs_v = np.arange(fem_verify.total_dof)
            free_v = np.setdiff1d(all_dofs_v, fd)
            u_opt = fem_verify.solve_static_partitioned(K_v, F, jnp.array(free_v), fd, fv)
            
            # Extract fields
            z_opt = np.array(u_opt[0::3])
            z_ref = np.array(tgt['u_static'])
            
            stress_opt_arr = np.array(fem_verify.compute_max_surface_stress(u_opt, params_v))
            stress_ref_arr = np.array(tgt['max_surface_stress'])
            
            strain_opt_arr = np.array(fem_verify.compute_max_surface_strain(u_opt, params_v))
            strain_ref_arr = np.array(tgt['max_surface_strain'])
            
            # Strain Energy (total over domain)
            sed_opt = np.array(fem_verify.compute_strain_energy_density(u_opt, params_v))
            sed_ref = np.array(tgt['strain_energy_density'])
            
            # Integrate strain energy density over area
            dx_v = self.fem.Lx / Nx_v
            dy_v = self.fem.Ly / Ny_v
            cell_area_v = dx_v * dy_v
            
            # Trapezoidal integration weights
            sed_opt_2d = sed_opt.reshape(Nx_v+1, Ny_v+1)
            sed_ref_2d = sed_ref.reshape(Nx_v+1, Ny_v+1)
            weights_v = np.ones((Nx_v+1, Ny_v+1))
            weights_v[0, :] *= 0.5
            weights_v[-1, :] *= 0.5
            weights_v[:, 0] *= 0.5
            weights_v[:, -1] *= 0.5
            
            total_energy_opt = np.sum(sed_opt_2d * weights_v) * cell_area_v
            total_energy_ref = np.sum(sed_ref_2d * weights_v) * cell_area_v
            
            energy_ratio = (total_energy_opt / total_energy_ref * 100) if total_energy_ref > 0 else 0
            
            # Calculate similarities
            disp_sim = calc_similarity(z_ref, z_opt)
            disp_r2 = calc_r2(z_ref, z_opt)
            stress_sim = calc_similarity(stress_ref_arr, stress_opt_arr)
            stress_r2 = calc_r2(stress_ref_arr, stress_opt_arr)
            strain_sim = calc_similarity(strain_ref_arr, strain_opt_arr)
            strain_r2 = calc_r2(strain_ref_arr, strain_opt_arr)
            
            static_results.append({
                'case': case.name,
                'disp_sim': disp_sim,
                'disp_r2': disp_r2,
                'stress_sim': stress_sim,
                'stress_r2': stress_r2,
                'strain_sim': strain_sim,
                'strain_r2': strain_r2,
                'energy_opt': total_energy_opt,
                'energy_ref': total_energy_ref,
                'energy_ratio': energy_ratio,
                # Displacement values (mm)
                'disp_max_ref': np.max(np.abs(z_ref)),
                'disp_max_opt': np.max(np.abs(z_opt)),
                'disp_avg_ref': np.mean(np.abs(z_ref)),
                'disp_avg_opt': np.mean(np.abs(z_opt)),
                # Stress values (MPa)
                'stress_max_ref': np.max(stress_ref_arr),
                'stress_max_opt': np.max(stress_opt_arr),
                'stress_avg_ref': np.mean(stress_ref_arr),
                'stress_avg_opt': np.mean(stress_opt_arr),
                'stress_robust_ref': calc_robust_max(stress_ref_arr, 2.5),
                'stress_robust_opt': calc_robust_max(stress_opt_arr, 2.5),
                # Strain values (dimensionless, ×10⁻³)
                'strain_max_ref': np.max(strain_ref_arr) * 1000,
                'strain_max_opt': np.max(strain_opt_arr) * 1000,
                'strain_avg_ref': np.mean(strain_ref_arr) * 1000,
                'strain_avg_opt': np.mean(strain_opt_arr) * 1000,
                'strain_robust_ref': calc_robust_max(strain_ref_arr, 2.5) * 1000,
                'strain_robust_opt': calc_robust_max(strain_opt_arr, 2.5) * 1000,
            })
        
        # Print Static Results Table
        report_lines.append("### 2.1 Similarity Metrics")
        report_lines.append("")
        report_lines.append("| Case | Disp Sim% | Disp R²% | Stress Sim% | Stress R²% | Strain Sim% | Strain R²% |")
        report_lines.append("|------|-----------|----------|-------------|------------|-------------|------------|")
        
        for res in static_results:
            report_lines.append(f"| {res['case']} | {res['disp_sim']:.2f} | {res['disp_r2']:.2f} | {res['stress_sim']:.2f} | {res['stress_r2']:.2f} | {res['strain_sim']:.2f} | {res['strain_r2']:.2f} |")
        
        # Average
        avg_disp_sim = np.mean([r['disp_sim'] for r in static_results])
        avg_disp_r2 = np.mean([r['disp_r2'] for r in static_results])
        avg_stress_sim = np.mean([r['stress_sim'] for r in static_results])
        avg_stress_r2 = np.mean([r['stress_r2'] for r in static_results])
        avg_strain_sim = np.mean([r['strain_sim'] for r in static_results])
        avg_strain_r2 = np.mean([r['strain_r2'] for r in static_results])
        
        report_lines.append(f"| **AVERAGE** | **{avg_disp_sim:.2f}** | **{avg_disp_r2:.2f}** | **{avg_stress_sim:.2f}** | **{avg_stress_r2:.2f}** | **{avg_strain_sim:.2f}** | **{avg_strain_r2:.2f}** |")
        report_lines.append("")
        
        # 2.2 Displacement Values Table
        report_lines.append("### 2.2 Displacement Values (mm)")
        report_lines.append("")
        report_lines.append("| Case | Max\|w\| (Tgt) | Max\|w\| (Opt) | Avg\|w\| (Tgt) | Avg\|w\| (Opt) |")
        report_lines.append("|------|----------------|----------------|----------------|----------------|")
        for res in static_results:
            report_lines.append(f"| {res['case']} | {res['disp_max_ref']:.4f} | {res['disp_max_opt']:.4f} | {res['disp_avg_ref']:.4f} | {res['disp_avg_opt']:.4f} |")
        report_lines.append("")
        
        # 2.3 Stress Values Table
        report_lines.append("### 2.3 Stress Values (MPa)")
        report_lines.append("")
        report_lines.append("| Case | Max (Tgt) | Max (Opt) | Avg (Tgt) | Avg (Opt) | Robust (Tgt) | Robust (Opt) |")
        report_lines.append("|------|-----------|-----------|-----------|-----------|--------------|--------------|")
        for res in static_results:
            report_lines.append(f"| {res['case']} | {res['stress_max_ref']:.3f} | {res['stress_max_opt']:.3f} | {res['stress_avg_ref']:.3f} | {res['stress_avg_opt']:.3f} | {res['stress_robust_ref']:.3f} | {res['stress_robust_opt']:.3f} |")
        report_lines.append("")
        
        # 2.4 Strain Values Table
        report_lines.append("### 2.4 Strain Values (×10⁻³)")
        report_lines.append("")
        report_lines.append("| Case | Max (Tgt) | Max (Opt) | Avg (Tgt) | Avg (Opt) | Robust (Tgt) | Robust (Opt) |")
        report_lines.append("|------|-----------|-----------|-----------|-----------|--------------|--------------|")
        for res in static_results:
            report_lines.append(f"| {res['case']} | {res['strain_max_ref']:.4f} | {res['strain_max_opt']:.4f} | {res['strain_avg_ref']:.4f} | {res['strain_avg_opt']:.4f} | {res['strain_robust_ref']:.4f} | {res['strain_robust_opt']:.4f} |")
        report_lines.append("")
        
        # Metric explanations
        report_lines.append("> **Metric Definitions:**")
        report_lines.append("> - **Similarity%** = (1 - NRMSE) × 100, where NRMSE = RMSE/(max-min)")
        report_lines.append("> - **R²** = 1 - SS_res/SS_tot (100% = perfect, 0% = mean prediction)")
        report_lines.append("> - **Robust Max** = min(μ + 2.5σ, actual_max) - excludes outliers")
        report_lines.append("")
        
        # --- 6.3 Strain Energy Comparison ---
        report_lines.append("## 3. Strain Energy Comparison")
        report_lines.append("")
        report_lines.append("| Case | Target Energy (N·mm) | Opt Energy (N·mm) | Ratio (%) |")
        report_lines.append("|------|----------------------|-------------------|-----------|")
        
        for res in static_results:
            report_lines.append(f"| {res['case']} | {res['energy_ref']:.6e} | {res['energy_opt']:.6e} | {res['energy_ratio']:.2f} |")
        
        avg_energy_ratio = np.mean([r['energy_ratio'] for r in static_results])
        report_lines.append(f"| **AVERAGE** | - | - | **{avg_energy_ratio:.2f}** |")
        report_lines.append("")
        report_lines.append("> **Strain Energy:** U = ∫(0.5·κᵀ·D·κ)dA. Ratio = (Opt/Target) × 100%. Ideal = 100%")
        report_lines.append("")
        
        # --- 6.4 Mass Comparison ---
        report_lines.append("## 4. Total Mass Comparison")
        report_lines.append("")
        
        # Calculate optimized mass (at high resolution for fair comparison)
        t_opt_v = np.array(params_v['t']).reshape(Nx_v+1, Ny_v+1)
        rho_opt_v = np.array(params_v['rho']).reshape(Nx_v+1, Ny_v+1)
        mass_density_opt = t_opt_v * rho_opt_v
        
        dx_v = self.fem.Lx / Nx_v
        dy_v = self.fem.Ly / Ny_v
        cell_area_v = dx_v * dy_v
        
        weights_v = np.ones((Nx_v+1, Ny_v+1))
        weights_v[0, :] *= 0.5
        weights_v[-1, :] *= 0.5
        weights_v[:, 0] *= 0.5
        weights_v[:, -1] *= 0.5
        
        opt_mass = np.sum(mass_density_opt * weights_v) * cell_area_v
        target_mass = self.target_mass if hasattr(self, 'target_mass') else 0
        
        mass_error = abs(opt_mass - target_mass) / target_mass * 100 if target_mass > 0 else 0
        mass_ratio = opt_mass / target_mass * 100 if target_mass > 0 else 0
        
        report_lines.append("| Property | Target | Optimized |")
        report_lines.append("|----------|--------|-----------|")
        report_lines.append(f"| Mass (tonne) | {target_mass:.6e} | {opt_mass:.6e} |")
        report_lines.append(f"| Mass (g) | {target_mass * 1e6:.4f} | {opt_mass * 1e6:.4f} |")
        report_lines.append(f"| Mass (kg) | {target_mass * 1e3:.6f} | {opt_mass * 1e3:.6f} |")
        report_lines.append("")
        report_lines.append(f"- **Mass Ratio:** {mass_ratio:.2f}%")
        report_lines.append(f"- **Mass Error:** {mass_error:.2f}%")
        report_lines.append("")
        report_lines.append("> **Mass Calculation:** M = ∫(ρ·t)dA, integrated using trapezoidal rule")
        report_lines.append("")
        
        # --- Summary ---
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("## Summary")
        report_lines.append("")
        report_lines.append("| Metric | Value |")
        report_lines.append("|--------|-------|")
        report_lines.append(f"| Modal - Avg Freq Error | {avg_freq_err:.2f}% |")
        report_lines.append(f"| Modal - Avg MAC | {avg_mac:.4f} |")
        report_lines.append(f"| Displacement - Avg Similarity | {avg_disp_sim:.2f}% |")
        report_lines.append(f"| Displacement - Avg R² | {avg_disp_r2:.2f}% |")
        report_lines.append(f"| Stress - Avg Similarity | {avg_stress_sim:.2f}% |")
        report_lines.append(f"| Stress - Avg R² | {avg_stress_r2:.2f}% |")
        report_lines.append(f"| Strain - Avg Similarity | {avg_strain_sim:.2f}% |")
        report_lines.append(f"| Strain - Avg R² | {avg_strain_r2:.2f}% |")
        report_lines.append(f"| Energy - Avg Ratio | {avg_energy_ratio:.2f}% |")
        report_lines.append(f"| Mass - Ratio | {mass_ratio:.2f}% |")
        report_lines.append(f"| Mass - Error | {mass_error:.2f}% |")
        report_lines.append("")
        
        # Print and save Markdown report
        full_report = "\n".join(report_lines)
        print(full_report)
        
        report_filename = "verification_report.md"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print(f"\nSaved Markdown report: {os.path.abspath(report_filename)}")
        
        # ============================================================
        # 7. Generate HTML Report
        # ============================================================
        html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Equivalent Sheet Model - Verification Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #e8f4f8; }}
        .summary-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .summary-box h2 {{ color: white; border-left-color: white; }}
        .metric-good {{ color: #27ae60; font-weight: bold; }}
        .metric-ok {{ color: #f39c12; font-weight: bold; }}
        .metric-poor {{ color: #e74c3c; font-weight: bold; }}
        .note {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; font-size: 0.9em; }}
        .avg-row {{ background: #e8e8e8 !important; font-weight: bold; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Equivalent Sheet Model - Verification Report</h1>
    
    <h2>1. Modal Analysis Comparison</h2>
    <table>
        <tr><th>Mode</th><th>Target Freq (Hz)</th><th>Opt Freq (Hz)</th><th>Freq Error (%)</th><th>MAC</th><th>Quality</th></tr>
'''
        for j in range(num_modes_plot):
            freq_err_j = abs(freq_o[j] - freq_t[j]) / freq_t[j] * 100 if freq_t[j] > 0 else 0
            mac_val = float(macs[j])
            quality_class = "metric-good" if mac_val >= 0.90 else "metric-ok" if mac_val >= 0.80 else "metric-poor"
            quality_text = "Excellent" if mac_val >= 0.95 else "Good" if mac_val >= 0.90 else "Acceptable" if mac_val >= 0.80 else "Poor"
            html_content += f'        <tr><td>{j+1}</td><td>{freq_t[j]:.4f}</td><td>{freq_o[j]:.4f}</td><td>{freq_err_j:.2f}</td><td class="{quality_class}">{mac_val:.4f}</td><td class="{quality_class}">{quality_text}</td></tr>\n'
        
        html_content += f'''        <tr class="avg-row"><td>AVG</td><td>-</td><td>-</td><td>{avg_freq_err:.2f}</td><td>{avg_mac:.4f}</td><td>-</td></tr>
    </table>
    <div class="note"><strong>MAC Interpretation:</strong> ≥0.95 = Excellent, ≥0.90 = Good, ≥0.80 = Acceptable, &lt;0.80 = Poor</div>
    
    <h2>2. Static Analysis Comparison</h2>
    
    <h3>2.1 Similarity Metrics</h3>
    <table>
        <tr><th>Case</th><th>Disp Sim%</th><th>Disp R²%</th><th>Stress Sim%</th><th>Stress R²%</th><th>Strain Sim%</th><th>Strain R²%</th></tr>
'''
        for res in static_results:
            html_content += f'        <tr><td>{res["case"]}</td><td>{res["disp_sim"]:.2f}</td><td>{res["disp_r2"]:.2f}</td><td>{res["stress_sim"]:.2f}</td><td>{res["stress_r2"]:.2f}</td><td>{res["strain_sim"]:.2f}</td><td>{res["strain_r2"]:.2f}</td></tr>\n'
        html_content += f'        <tr class="avg-row"><td>AVERAGE</td><td>{avg_disp_sim:.2f}</td><td>{avg_disp_r2:.2f}</td><td>{avg_stress_sim:.2f}</td><td>{avg_stress_r2:.2f}</td><td>{avg_strain_sim:.2f}</td><td>{avg_strain_r2:.2f}</td></tr>\n'
        
        html_content += '''    </table>
    
    <h3>2.2 Displacement Values (mm)</h3>
    <table>
        <tr><th>Case</th><th>Max |w| (Target)</th><th>Max |w| (Opt)</th><th>Avg |w| (Target)</th><th>Avg |w| (Opt)</th></tr>
'''
        for res in static_results:
            html_content += f'        <tr><td>{res["case"]}</td><td>{res["disp_max_ref"]:.4f}</td><td>{res["disp_max_opt"]:.4f}</td><td>{res["disp_avg_ref"]:.4f}</td><td>{res["disp_avg_opt"]:.4f}</td></tr>\n'
        
        html_content += '''    </table>
    
    <h3>2.3 Stress Values (MPa)</h3>
    <table>
        <tr><th>Case</th><th>Max (Tgt)</th><th>Max (Opt)</th><th>Avg (Tgt)</th><th>Avg (Opt)</th><th>Robust (Tgt)</th><th>Robust (Opt)</th></tr>
'''
        for res in static_results:
            html_content += f'        <tr><td>{res["case"]}</td><td>{res["stress_max_ref"]:.3f}</td><td>{res["stress_max_opt"]:.3f}</td><td>{res["stress_avg_ref"]:.3f}</td><td>{res["stress_avg_opt"]:.3f}</td><td>{res["stress_robust_ref"]:.3f}</td><td>{res["stress_robust_opt"]:.3f}</td></tr>\n'
        
        html_content += '''    </table>
    
    <h3>2.4 Strain Values (×10⁻³)</h3>
    <table>
        <tr><th>Case</th><th>Max (Tgt)</th><th>Max (Opt)</th><th>Avg (Tgt)</th><th>Avg (Opt)</th><th>Robust (Tgt)</th><th>Robust (Opt)</th></tr>
'''
        for res in static_results:
            html_content += f'        <tr><td>{res["case"]}</td><td>{res["strain_max_ref"]:.4f}</td><td>{res["strain_max_opt"]:.4f}</td><td>{res["strain_avg_ref"]:.4f}</td><td>{res["strain_avg_opt"]:.4f}</td><td>{res["strain_robust_ref"]:.4f}</td><td>{res["strain_robust_opt"]:.4f}</td></tr>\n'
        
        html_content += f'''    </table>
    
    <div class="note">
        <strong>Metric Definitions:</strong><br>
        • <strong>Similarity%</strong> = (1 - NRMSE) × 100, where NRMSE = RMSE / (max - min). 100% = perfect match.<br>
        • <strong>R²</strong> (Coefficient of Determination) = 1 - SS_res/SS_tot. 100% = perfect prediction.<br>
        • <strong>Robust Max</strong> = min(μ + 2.5σ, actual_max). Excludes outliers beyond 2.5 standard deviations.
    </div>
    
    <h2>3. Strain Energy Comparison</h2>
    <table>
        <tr><th>Case</th><th>Target Energy (N·mm)</th><th>Opt Energy (N·mm)</th><th>Ratio (%)</th></tr>
'''
        total_energy_ref_sum = sum([r['energy_ref'] for r in static_results])
        total_energy_opt_sum = sum([r['energy_opt'] for r in static_results])
        for res in static_results:
            html_content += f'        <tr><td>{res["case"]}</td><td>{res["energy_ref"]:.6e}</td><td>{res["energy_opt"]:.6e}</td><td>{res["energy_ratio"]:.2f}</td></tr>\n'
        html_content += f'        <tr class="avg-row"><td>TOTAL</td><td>{total_energy_ref_sum:.6e}</td><td>{total_energy_opt_sum:.6e}</td><td>{avg_energy_ratio:.2f}</td></tr>\n'
        
        html_content += f'''    </table>
    <div class="note"><strong>Strain Energy:</strong> U = ∫(0.5·κᵀ·D·κ)dA. Ratio = (Opt/Target) × 100%. Ideal = 100%.</div>
    
    <h2>4. Total Mass Comparison</h2>
    <table>
        <tr><th>Property</th><th>Target</th><th>Optimized</th></tr>
        <tr><td>Mass (tonne)</td><td>{target_mass:.6e}</td><td>{opt_mass:.6e}</td></tr>
        <tr><td>Mass (g)</td><td>{target_mass * 1e6:.4f}</td><td>{opt_mass * 1e6:.4f}</td></tr>
        <tr><td>Mass (kg)</td><td>{target_mass * 1e3:.6f}</td><td>{opt_mass * 1e3:.6f}</td></tr>
    </table>
    <p><strong>Mass Ratio:</strong> {mass_ratio:.2f}% | <strong>Mass Error:</strong> {mass_error:.2f}%</p>
    <div class="note"><strong>Mass Calculation:</strong> M = ∫(ρ·t)dA, integrated using trapezoidal rule.</div>
    
    <div class="summary-box">
        <h2>Summary</h2>
        <p><strong>Modal:</strong> Avg Freq Error = {avg_freq_err:.2f}%, Avg MAC = {avg_mac:.4f}</p>
        <p><strong>Displacement:</strong> Avg Similarity = {avg_disp_sim:.2f}%, Avg R² = {avg_disp_r2:.2f}%</p>
        <p><strong>Stress:</strong> Avg Similarity = {avg_stress_sim:.2f}%, Avg R² = {avg_stress_r2:.2f}%</p>
        <p><strong>Strain:</strong> Avg Similarity = {avg_strain_sim:.2f}%, Avg R² = {avg_strain_r2:.2f}%</p>
        <p><strong>Energy:</strong> Avg Ratio = {avg_energy_ratio:.2f}%</p>
        <p><strong>Mass:</strong> Ratio = {mass_ratio:.2f}%, Error = {mass_error:.2f}%</p>
    </div>
</div>
</body>
</html>
'''
        
        html_filename = "verification_report.html"
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Saved HTML report: {os.path.abspath(html_filename)}")

if __name__ == '__main__':
    # XLA Flags for Threading
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
    
    # 1. Init Model
    model = EquivalentSheetModel(Lx, Ly, Nx_low, Ny_low)
    
    # 2. Add Cases
    model.add_case(TwistCase('twist_x', value=1.5, mode='angle'))
    model.add_case(TwistCase('twist_y', value=1.5, mode='angle'))
    model.add_case(PureBendingCase('bend_y', value=3.0, mode='angle'))
    model.add_case(PureBendingCase('bend_x', value=1.0, mode='angle'))
    model.add_case(CornerLiftCase('lift_br', corner='br', value=1.0))
    model.add_case(CornerLiftCase('lift_bl', corner='bl', value=3.0))
    
    # 3. Generate Truth
    model.generate_targets(resolution_high=(Nx_high, Ny_high))
    
    # 4. Optimize
    cfg = {
        't': {'opt': True, 'min': 0.5, 'max': 3.0},
        'rho': {'opt': True, 'min': 5e-9, 'max': 1e-8},
        'E': {'opt': True, 'min': 40e3, 'max': 250e3}
    }
    
    # Loss weights
    # Physical quantity matching:
    # - 'static': Displacement matching (always enabled)
    # - 'freq': Eigenfrequency matching
    # - 'mode': Mode shape (MAC) matching
    # 
    # Legacy (for debugging - NOT physical strain/stress):
    # - 'curvature': Curvature (κ) matching
    # - 'moment': Bending moment (M) matching
    #
    # Recommended metrics (physically meaningful):
    # - 'strain_energy': Strain energy density matching (0.1~1.0)
    #                    U = 0.5·κ^T·D·κ (energy-based optimization)
    # - 'surface_stress': Max surface von Mises stress (0.1~1.0)
    #                     σ_max = 6M/t² (strength-based optimization)
    # - 'surface_strain': Max surface von Mises strain (0.1~1.0)
    #                     ε_max = (t/2)·κ (failure prediction)
    #
    # Regularization:
    # - 'reg': Total variation smoothness (0.01~0.5)
    
    weights = {
        'static': 1.0,           # Displacement
        'freq': 1.0,             # Eigenfrequency
        'mode': 1.0,             # Mode shape (MAC)
        'curvature': 0.0,        # [Legacy] Curvature (κ) - debug only
        'moment': 0.0,           # [Legacy] Moment (M) - debug only
        'strain_energy': 0.0,    # Strain energy density
        'surface_stress': 1.0,   # Max surface stress (recommended)
        'surface_strain': 1.0,   # Max surface strain (recommended)
        'reg': 0.05,             # Regularization (smoothness)
        'mass': 1.0              # Mass constraint (5% tolerance)
    }
    
    model.optimize(cfg, weights, use_smoothing=False, 
                   use_curvature=False, use_moment=False,
                   use_strain_energy=False, use_surface_stress=True, use_surface_strain=True,
                   use_mass_constraint=True, mass_tolerance=0.05,
                   max_iterations=200)
    
    # 5. Verify
    model.verify()
