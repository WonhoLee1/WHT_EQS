# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from solver import PlateFEM
import time
import os
import concurrent.futures
import optax

# --- Configuration ---
Lx = 1000.0  # mm
Ly = 400.0   # mm
Base_t = 1.0 # mm
Bead_t = 2.0 # mm
Base_E = 200000.0 # MPa
Base_rho = 7.5e-9 # tonne/mm^3

# Resolutions
Nx_high = 20 
Ny_high = 10
Nx_low = 20
Ny_low = 10

# --- PyVista Visualization Helper ---
import pyvista as pv

def visualize_pattern_interactive(X, Y, t, z, pattern_name):
    """
    Visualizes the initial pattern (Thickness and Z-coordinate) by saving screenshots (Non-Blocking).
    """
    try:
        print("\n[Visualizing Initial Pattern - Saving Screenshots...]")
        
        # Infer grid size
        n_nodes = X.shape[0]
        unique_x = np.unique(X)
        unique_y = np.unique(Y)
        nx = len(unique_x)
        ny = len(unique_y)
        
        if nx * ny != n_nodes:
            # Fallback
            grid = pv.PolyData(np.column_stack((X, Y, z)))
        else:
            # Structured Grid
            grid = pv.StructuredGrid()
            grid.points = np.column_stack((X, Y, z))
            grid.dimensions = [ny, nx, 1] 
            
        grid["Thickness"] = t
        grid["Z_Coordinate"] = z
        
        # 1. Save Thickness Plot
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(grid, scalars="Thickness", show_edges=True, cmap="jet", label="Thickness")
        plotter.add_axes()
        plotter.add_title(f"Pattern: {pattern_name} (Thickness)")
        plotter.view_xy()
        plotter.screenshot("initial_pattern_thickness.png")
        plotter.close()
        print("Saved initial_pattern_thickness.png")

        # 2. Save Z Plot
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(grid, scalars="Z_Coordinate", show_edges=True, cmap="terrain", label="Z-Coord")
        plotter.add_axes()
        plotter.add_title(f"Pattern: {pattern_name} (Z-Coord)")
        plotter.view_isometric()
        plotter.screenshot("initial_pattern_z.png")
        plotter.close()
        print("Saved initial_pattern_z.png")
        
    except Exception as e:
        print(f"Visualization screenshot failed: {e}")


def get_pattern_field(X, Y, Lx, Ly, pattern_str, val_dict, base_val):
    """
    Generic function to generate a field based on pattern string and value dictionary.
    """
    if val_dict is None: val_dict = {}
    
    num_chars = len(pattern_str)
    char_width = Lx / num_chars
    
    # Initialize map with base value
    field_map = jnp.full_like(X, base_val)
    
    w_physical = 60.0 # mm
    
    def dist_segment(px, py, x1, y1, x2, y2):
        X1, Y1 = x1 * char_width, y1 * Ly
        X2, Y2 = x2 * char_width, y2 * Ly
        PX, PY = px * char_width, py * Ly
        
        dx, dy = X2-X1, Y2-Y1
        len_sq = dx**2 + dy**2 + 1e-6
        t = ((PX-X1)*dx + (PY-Y1)*dy) / len_sq
        t = jnp.clip(t, 0.0, 1.0)
        closest_x = X1 + t*dx
        closest_y = Y1 + t*dy
        return (PX-closest_x)**2 + (PY-closest_y)**2

    for i, char in enumerate(pattern_str):
        if char == ' ': continue
        
        # Determine value for this character
        if isinstance(val_dict, (int, float)):
             current_val = val_dict
        else:
             current_val = val_dict.get(char, base_val)

        # Define Character Region
        x_start = i * char_width
        x_end = (i + 1) * char_width
        
        region_mask = (X >= x_start) & (X < x_end)
        
        # Normalized Local Coords (u, v) in [0, 1]
        u = (X - x_start) / char_width
        v = Y / Ly
        
        dist_sq = jnp.full_like(X, 1e9) # Min dist squared
        
        # Define Strokes for each char
        strokes = []
        if char == 'A':
            strokes = [(0.2, 0.1, 0.5, 0.9), (0.8, 0.1, 0.5, 0.9), (0.35, 0.5, 0.65, 0.5)]
        elif char == 'B':
            strokes = [(0.2, 0.1, 0.2, 0.9), (0.2, 0.9, 0.7, 0.9), (0.2, 0.5, 0.6, 0.5), 
                       (0.2, 0.1, 0.7, 0.1), (0.7, 0.9, 0.8, 0.7), (0.8, 0.7, 0.6, 0.5),
                       (0.6, 0.5, 0.8, 0.3), (0.8, 0.3, 0.7, 0.1)]
        elif char == 'C':
            strokes = [(0.25, 0.2, 0.25, 0.8), (0.25, 0.8, 0.7, 0.8), (0.25, 0.2, 0.7, 0.2)]
        elif char == 'T':
            strokes = [(0.5, 0.1, 0.5, 0.9), (0.2, 0.9, 0.8, 0.9)]
        elif char == 'N':
            strokes = [(0.2, 0.1, 0.2, 0.9), (0.2, 0.9, 0.8, 0.1), (0.8, 0.1, 0.8, 0.9)]
        elif char == 'Y':
            strokes = [(0.2, 0.9, 0.5, 0.5), (0.8, 0.9, 0.5, 0.5), (0.5, 0.5, 0.5, 0.1)]
        
        # Compute min distance to any stroke
        if strokes:
            for s in strokes:
                d2 = dist_segment(u, v, s[0], s[1], s[2], s[3])
                dist_sq = jnp.minimum(dist_sq, d2)
                
            is_stroke = dist_sq < (w_physical/2)**2
            
            # Update map
            field_map = jnp.where(region_mask & is_stroke, current_val, field_map)
    
    return field_map

def get_z_field(X, Y, Lx=1000.0, Ly=400.0, pattern_pz="TNY", pz_dict=None):
    if pz_dict is None: pz_dict = {}
    return get_pattern_field(X, Y, Lx, Ly, pattern_pz, pz_dict, base_val=0.0)

def get_thickness_field(X, Y, Lx=1000.0, Ly=400.0, pattern_str="A", base_t=1.0, bead_t=2.0):
    return get_pattern_field(X, Y, Lx, Ly, pattern_str, bead_t, base_val=base_t)

def get_density_field(X, Y, base_rho=7.5e-9, seed=42):
    """
    Generate spatially varying density field with 3x2 grid regions.
    Each region has random density within ±30% of base value (7.5e-9).
    """
    np.random.seed(seed)
    
    # Define 3x2 grid boundaries
    x_boundaries = [0, Lx/3, 2*Lx/3, Lx]
    y_boundaries = [0, Ly/2, Ly]
    
    # Generate random density for each of 6 regions (within ±30%)
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

def get_E_field(X, Y, base_E=200000.0, seed=42):
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
        
        if self.axis == 'x':
            # Twist about X axis
            left_nodes = jnp.where(fem.node_coords[:, 0] < tol)[0]
            right_nodes = jnp.where(fem.node_coords[:, 0] > Lx - tol)[0]
            
            yc = Ly / 2.0
            
            if self.mode == 'angle':
                angle_rad = self.value * np.pi / 180.0
                
                # Left (x=0)
                y_left = fem.node_coords[left_nodes, 1]
                w_left = (y_left - yc) * np.tan(-angle_rad)
                
                for i, node in enumerate(left_nodes):
                    fixed_dofs.extend([node*6 + 2, node*6 + 3]) # w (idx 2), tx (idx 3)
                    fixed_vals.extend([w_left[i], -angle_rad])
                
                # Right (x=L)
                y_right = fem.node_coords[right_nodes, 1]
                w_right = (y_right - yc) * np.tan(angle_rad)
                
                for i, node in enumerate(right_nodes):
                    fixed_dofs.extend([node*6 + 2, node*6 + 3])
                    fixed_vals.extend([w_right[i], angle_rad])
                
                # Fix Ty at one node (0,0) to prevent y-drift
                n0 = 0 
                fixed_dofs.append(n0*6 + 4) # ty is idx 4
                fixed_vals.append(0.0)
                
                # Fix u, v at one node to prevent rigid body drift
                fixed_dofs.extend([n0*6+0, n0*6+1])
                fixed_vals.extend([0.0, 0.0])

            elif self.mode == 'moment':
                # Apply Moment Mx at ends
                m_node = self.value / len(right_nodes)
                
                F = F.at[right_nodes * 6 + 3].add(m_node)
                F = F.at[left_nodes * 6 + 3].add(-m_node)
                
                # Fix w, ty, tz at center
                center_node = jnp.argmin((fem.node_coords[:,0]-Lx/2)**2 + (fem.node_coords[:,1]-Ly/2)**2)
                fixed_dofs.extend([center_node*6+0, center_node*6+1, center_node*6+2, center_node*6+4]) 
                fixed_vals.extend([0.0, 0.0, 0.0, 0.0])
                
        elif self.axis == 'y':
            # Twist about Y axis
            bot_nodes = jnp.where(fem.node_coords[:, 1] < tol)[0]
            top_nodes = jnp.where(fem.node_coords[:, 1] > Ly - tol)[0]
            xc = Lx / 2.0
            
            if self.mode == 'angle':
                angle_rad = self.value * np.pi / 180.0
                
                # Bot: w = (x-xc)*tan(-angle)
                x_bot = fem.node_coords[bot_nodes, 0]
                w_bot = (x_bot - xc) * np.tan(-angle_rad)
                
                for i, node in enumerate(bot_nodes):
                    fixed_dofs.extend([node*6 + 2, node*6 + 4]) # w, ty
                    fixed_vals.extend([w_bot[i], -angle_rad])
                    
                # Top (y=L): w = (x-xc)*tan(angle)
                x_top = fem.node_coords[top_nodes, 0]
                w_top = (x_top - xc) * np.tan(angle_rad)
                
                for i, node in enumerate(top_nodes):
                    fixed_dofs.extend([node*6 + 2, node*6 + 4])
                    fixed_vals.extend([w_top[i], angle_rad])
                    
                # Fix Tx at center (0,0)
                n0 = 0
                fixed_dofs.append(n0*6+3) # tx is idx 3
                fixed_vals.append(0.0)
                
                # Fix u, v
                fixed_dofs.extend([n0*6+0, n0*6+1])
                fixed_vals.extend([0.0, 0.0])

        return jnp.array(fixed_dofs, dtype=jnp.int32), jnp.array(fixed_vals, dtype=jnp.float64), F

class PureBendingCase(LoadCase):
    def __init__(self, name, axis=None, value=3.0, mode='angle', weight=1.0):
        super().__init__(name, weight)
        if axis is None:
            if '_x' in name.lower():
                axis = 'x'
            elif '_y' in name.lower():
                axis = 'y'
            else:
                axis = 'y'  # default
        self.axis = axis 
        self.value = value
        self.mode = mode
        
    def get_bcs(self, fem):
        # Pure Bending
        tol = 1e-3
        Lx, Ly = fem.Lx, fem.Ly
        F = jnp.zeros(fem.total_dof)
        fixed_dofs = []
        fixed_vals = []
        
        if self.axis == 'y':
            # Bend Y: Curvature in XZ plane
            left_nodes = jnp.where(fem.node_coords[:, 0] < tol)[0]
            right_nodes = jnp.where(fem.node_coords[:, 0] > Lx - tol)[0]
            
            if self.mode == 'angle':
                angle_rad = self.value * np.pi / 180.0
                
                for node in left_nodes:
                    fixed_dofs.extend([node*3+0, node*3+2]) # w, ty
                    fixed_vals.extend([0.0, angle_rad]) 
                    
                for node in right_nodes:
                    fixed_dofs.extend([node*6+2, node*6+4]) # w, ty
                    fixed_vals.extend([0.0, -angle_rad])
                
                # Fix Tx at one node
                fixed_dofs.append(left_nodes[0]*6 + 3)
                fixed_vals.append(0.0)
                
        elif self.axis == 'x':
            # Bend X: Curvature in YZ plane
            bot_nodes = jnp.where(fem.node_coords[:, 1] < tol)[0]
            top_nodes = jnp.where(fem.node_coords[:, 1] > Ly - tol)[0]
            
            if self.mode == 'angle':
                angle_rad = self.value * np.pi / 180.0
                
                for node in bot_nodes:
                     fixed_dofs.extend([node*6+2, node*6+3]) # w, tx
                     fixed_vals.extend([0.0, -angle_rad])
                     
                for node in top_nodes:
                     fixed_dofs.extend([node*6+2, node*6+3]) # w, tx
                     fixed_vals.extend([0.0, angle_rad])
                     
                # Fix Ty at one node
                fixed_dofs.append(bot_nodes[0]*6 + 4)
                fixed_vals.append(0.0)
                
                # Anchor u, v at one node (BL)
                fixed_dofs.extend([bot_nodes[0]*6+0, bot_nodes[0]*6+1])
                fixed_vals.extend([0.0, 0.0])

        return jnp.array(fixed_dofs, dtype=jnp.int32), jnp.array(fixed_vals, dtype=jnp.float64), F

class CornerLiftCase(LoadCase):
    def __init__(self, name, corner='br', value=1.0, mode='disp', weight=1.0):
        super().__init__(name, weight)
        self.corner = corner 
        self.value = value
        self.mode = mode
        
    def get_bcs(self, fem):
        # Fix 3 corners w=0, Lift 1 corner
        coords = fem.node_coords
        
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
                fixed_dofs.append(idx*6 + 2) # w
                fixed_vals.append(0.0)
                
        # Anchor u, v at one corner
        non_target_corners = [k for k in corners if k != self.corner]
        if non_target_corners:
             anchor = corners[non_target_corners[0]]
             fixed_dofs.extend([anchor*6+0, anchor*6+1])
             fixed_vals.extend([0.0, 0.0])
        
        if self.mode == 'disp':
            fixed_dofs.append(target_idx*6 + 2) # w
            fixed_vals.append(self.value)
        elif self.mode == 'force':
            F = F.at[target_idx*6 + 2].set(self.value)
        
        return jnp.array(fixed_dofs, dtype=jnp.int32), jnp.array(fixed_vals, dtype=jnp.float64), F

# --- Model Manager ---

class EquivalentSheetModel:
    def __init__(self, Lx, Ly, Nx, Ny):
        self.fem = PlateFEM(Lx, Ly, Nx, Ny)
        self.cases = []
        self.targets = [] 
        self.optimized_params = None
        
    def add_case(self, case: LoadCase):
        self.cases.append(case)
        print(f"Added Case: {case.name}")

    def generate_targets(self, resolution_high=(50, 20), num_modes_save=5, target_config=None):
        print("Generating Ground Truth Targets...")
        self.resolution_high = resolution_high
        self.num_modes_truth = num_modes_save
        fem_high = PlateFEM(self.fem.Lx, self.fem.Ly, resolution_high[0], resolution_high[1])
        
        # Target Params (Fixed)
        X_grid = fem_high.node_coords[:,0].reshape(resolution_high[0]+1, -1)
        Y_grid = fem_high.node_coords[:,1].reshape(resolution_high[0]+1, -1)
        
        if target_config is None: target_config = {}
        
        # Extract target properties
        base_t = target_config.get('base_t', 1.0)
        bead_t = target_config.get('bead_t', 2.0)
        base_rho = target_config.get('base_rho', 7.5e-9)
        base_E = target_config.get('base_E', 200000.0)
        pattern = target_config.get('pattern', 'A') 
        
        # Parse Z-pattern properties
        pattern_pz = target_config.get('pattern_pz', '')
        bead_pz = target_config.get('bead_pz', {})

        print(f"Target Properties: t={base_t}/{bead_t}, rho={base_rho}, E={base_E}, Pattern='{pattern}'")
        if pattern_pz:
            print(f"Topography Pattern: '{pattern_pz}' with heights {bead_pz}", flush=True)
        
        print(f"DEBUG: X_grid shape logic", flush=True)
        print(f"DEBUG: node_coords shape: {fem_high.node_coords.shape}", flush=True)
        # Check slices
        nc0 = fem_high.node_coords[:,0]
        print(f"DEBUG: node_coords[:,0] shape: {nc0.shape}", flush=True)
        
        X_grid = nc0.reshape(resolution_high[0]+1, -1)
        Y_grid = fem_high.node_coords[:,1].reshape(resolution_high[0]+1, -1)
        print(f"DEBUG: X_grid shape: {X_grid.shape}", flush=True)
        
        t_field = get_thickness_field(X_grid, Y_grid, Lx=self.fem.Lx, Ly=self.fem.Ly, pattern_str=pattern, base_t=base_t, bead_t=bead_t)
        print(f"DEBUG: t_field raw shape: {t_field.shape}", flush=True)
        t_field = t_field.flatten() 
        print(f"DEBUG: t_field flattened shape: {t_field.shape}", flush=True)
        
        # Generate Z-field (Topography)
        if pattern_pz:
            z_field = get_z_field(X_grid, Y_grid, Lx=self.fem.Lx, Ly=self.fem.Ly, pattern_pz=pattern_pz, pz_dict=bead_pz)
            z_field = z_field.flatten()
        else:
            z_field = jnp.zeros_like(t_field)

        rho_field = get_density_field(X_grid, Y_grid, base_rho=base_rho, seed=42)
        rho_field = rho_field.flatten()
        
        E_field = jnp.full_like(t_field, base_E)
        
        print(f"High Res Params Shape: t={t_field.shape}, rho={rho_field.shape}, E={E_field.shape}, z={z_field.shape}")
        
        # Non-Blocking Visualization (Save Screenshots)
        try:
             visualize_pattern_interactive(X_grid.flatten(), Y_grid.flatten(), np.array(t_field), np.array(z_field), pattern)
        except Exception as e:
             print(f"Visualization skipped: {e}", flush=True)

        params_high = {
            't': t_field,
            'rho': rho_field,
            'E': E_field,
            'z': z_field
        }
        
        # Calculate Ground Truth Responses
        print("Solving High-Res FEM for Targets...", flush=True)
        K_d, M_d = fem_high.assemble(params_high)
        
        # Static Targets
        self.targets = []
        all_dofs = np.arange(fem_high.total_dof)
        
        for case in self.cases:
            fd, fv, F = case.get_bcs(fem_high)
            free = np.setdiff1d(all_dofs, fd)
            
            # Solve
            u_high = fem_high.solve_static_partitioned(K_d, F, jnp.array(free), fd, fv)
            
            # Compute derived
            tgt_data = {
                'case_name': case.name,
                'weight': case.weight,
                'u_static': u_high
            }
            
            # Stress/Strain
            tgt_data['max_surface_stress'] = fem_high.compute_max_surface_stress(u_high, params_high)
            tgt_data['max_surface_strain'] = fem_high.compute_max_surface_strain(u_high, params_high)
            tgt_data['strain_energy_density'] = fem_high.compute_strain_energy_density(u_high, params_high)
            
            self.targets.append(tgt_data)
            print(f"DEBUG: Generated target for {case.name}", flush=True)

        # Modal Analysis (Cantilever BCs for Modes)
        # Boundary Conditions (Cantilever at x=0)
        fixed_nodes = jnp.where(jnp.isclose(fem_high.node_coords[:, 0], 0.0))[0]
        print(f"DEBUG: Found {len(fixed_nodes)} fixed nodes at x=0")
        
        fixed_dofs = []
        for n in fixed_nodes:
             fixed_dofs.extend([n*6+i for i in range(6)])
        
        fixed_dofs = jnp.array(fixed_dofs)
        print(f"DEBUG: Fixed {len(fixed_dofs)} DOFs. Total DOFs: {fem_high.total_dof}")
        print(f"DEBUG: K_d norm: {jnp.linalg.norm(K_d)}, M_d norm: {jnp.linalg.norm(M_d)}")
        
        K_ff, M_ff, free_dofs = fem_high.apply_boundary_conditions(K_d, M_d, fixed_dofs)
        
        print("DEBUG: Calling solve_eigen...", flush=True)
        # Solve Eigenvalues
        vals_high, vecs_high = fem_high.solve_eigen(K_ff, M_ff, num_modes=num_modes_save + 10)
        print(f"DEBUG: solve_eigen returned. vals shape: {vals_high.shape}", flush=True)
        
        self.target_vals = vals_high[:num_modes_save]
        
        # Store FULL target eigenvectors
        full_vecs = jnp.zeros((fem_high.total_dof, num_modes_save))
        
        for m in range(num_modes_save):
            v_reduced = vecs_high[:, m]
            v_full = jnp.zeros(fem_high.total_dof)
            v_full = v_full.at[free_dofs].set(v_reduced)
            full_vecs = full_vecs.at[:, m].set(v_full)
            
        self.target_vecs = full_vecs
        self.target_params = params_high
        self.target_eigen = {'vals': vals_high, 'modes': full_vecs} 
        
        print(f"Ground Truth Generated. Target Frequencies: {jnp.sqrt(jnp.abs(self.target_vals))/(2*jnp.pi)}")

        # Calculate Target Total Mass (for mass constraint)
        # Calculate Target Total Mass (for mass constraint)
        # Convert to numpy to avoid JAX async issues during debug
        rho_np = np.array(rho_field).flatten()
        t_np = np.array(t_field).flatten()
        
        if rho_np.shape != t_np.shape:
             print(f"WARNING: shape mismatch in mass calc: rho={rho_np.shape}, t={t_np.shape}", flush=True)
             n_min = min(len(rho_np), len(t_np))
             rho_np = rho_np[:n_min]
             t_np = t_np[:n_min]
             
        mass_density = rho_np * t_np
        
        dx_h = fem_high.Lx / resolution_high[0]
        dy_h = fem_high.Ly / resolution_high[1]
        cell_area = dx_h * dy_h
        
        Nx_h, Ny_h = resolution_high
        try:
             mass_density_2d = mass_density.reshape(Nx_h+1, Ny_h+1)
        except Exception as e:
             print(f"CRASH in mass reshape (numpy): {e}", flush=True)
             print(f"mass_density size: {mass_density.size}", flush=True)
             print(f"Target shape: ({Nx_h+1}, {Ny_h+1})", flush=True)
             import traceback
             traceback.print_exc()
             raise e
             
        weights_2d = np.ones((Nx_h+1, Ny_h+1))
        weights_2d[0, :] *= 0.5
        weights_2d[-1, :] *= 0.5
        weights_2d[:, 0] *= 0.5
        weights_2d[:, -1] *= 0.5
        
        self.target_mass = np.sum(mass_density_2d * weights_2d) * cell_area
        print(f"Target Total Mass: {self.target_mass:.6f} tonne ({self.target_mass * 1e6:.3f} g)", flush=True)
