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
             print(f"Visualization skipped: {e}")

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
        
        print(f"DEBUG MASS: rho_np shape={rho_np.shape}, dtype={rho_np.dtype}", flush=True)
        print(f"DEBUG MASS: t_np shape={t_np.shape}, dtype={t_np.dtype}", flush=True)
        
        if rho_np.shape != t_np.shape:
             print(f"WARNING: shape mismatch in mass calc: rho={rho_np.shape}, t={t_np.shape}", flush=True)
             n_min = min(len(rho_np), len(t_np))
             rho_np = rho_np[:n_min]
             t_np = t_np[:n_min]
             
        mass_density = rho_np * t_np
        print(f"DEBUG MASS: mass_density shape={mass_density.shape}", flush=True)
        
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

    def optimize(self, opt_config, loss_weights, use_smoothing=True, use_curvature=False, use_moment=False, use_strain_energy=False, use_surface_stress=False, use_surface_strain=False, use_mass_constraint=False, mass_tolerance=0.05, max_iterations=200, use_early_stopping=True, early_stop_patience=None, early_stop_tol=1e-6, num_modes_loss=None):
        """
        Runs the optimization process to find equivalent sheet properties.
        """
        
        # Use stored number of modes if not provided
        if num_modes_loss is None:
            if hasattr(self, 'num_modes_truth'):
                num_modes_loss = self.num_modes_truth
            else:
                num_modes_loss = 5 # Fallback default
        
        print("OPTIMIZE METHOD ENTERED", flush=True)
        print(f"Starting Optimization (Modes for Loss: {num_modes_loss})...", flush=True)
        
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
            print(f"DEBUG Interpolation START: pts_h={pts_h.shape}, u_static len={len(tgt['u_static'])}", flush=True)
            try:
                 u_h = tgt['u_static'].reshape(-1, 6) # (N, 6)
                 print(f"DEBUG Interpolation: u_h reshaped={u_h.shape}", flush=True)
                 u_l = griddata(pts_h, u_h, xl, method='cubic')
            except Exception as e:
                 print(f"CRASH in griddata (u_static): {e}", flush=True)
                 print(f"pts_h shape: {pts_h.shape}", flush=True)
                 print(f"u_h shape: {u_h.shape} (inferred)", flush=True)
                 print(f"xl shape: {xl.shape}", flush=True)
                 raise e

            tgt_low = {
                'case_name': tgt['case_name'],
                'u_static': jnp.array(u_l.flatten()),
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
        t_vals = self.target_eigen['vals'][:num_modes_loss]
        t_modes_h = self.target_eigen['modes'][:, :num_modes_loss]
        t_modes_l = []
        for i in range(num_modes_loss):
             # Reshape to (N, 6) because griddata expects values per point
             mode_h_reshaped = t_modes_h[:,i].reshape(-1, 6)
             # Interpolate vectors
             m_reshaped = griddata(pts_h, mode_h_reshaped, xl, method='cubic')
             # Flatten back to (total_dof,)
             m = m_reshaped.flatten()
             t_modes_l.append(m)
             
        if len(t_modes_l) > 0:
            t_modes_l = jnp.stack(t_modes_l, axis=1) # (dof, num_modes)
        else:
            t_modes_l = jnp.zeros((xl.shape[0]*6, 0))
        
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
            
        # Initialize Base Parameters
        def get_init_val(key, default):
            cfg = opt_config.get(key, {})
            if 'init' in cfg:
                return cfg['init']
            if 'min' in cfg and 'max' in cfg:
                return (cfg['min'] + cfg['max']) / 2.0
            return default

        Base_t = get_init_val('t', 1.0)
        Base_rho = get_init_val('rho', 1.0)
        Base_E = get_init_val('E', 1.0)
        
        print(f"Initialized Params: t={Base_t:.4f}, rho={Base_rho:.4e}, E={Base_E:.4e}")
            
        # Initialize optimizable parameters at NODES (matches solver expectations for field input)
        # Assuming params initialized as flat arrays if not specified, 
        # but JAX optimization works better with structured dict of arrays.
        # Ensure shape is (num_nodes,)
        num_nodes = (self.fem.nx + 1) * (self.fem.ny + 1)
        
        params = {
            't': jnp.full(num_nodes, Base_t),
            'rho': jnp.full(num_nodes, Base_rho),
            'E': jnp.full(num_nodes, Base_E)
        }
        
        # Initialize z (Topography) if enabled in config
        # Check if 'z' key exists roughly in opt_config or implicitly handled
        # We will add 'z' to params if we want to optimize it.
        # Assuming 'z' optimization is desired if we call optimize? Or config dependent?
        # Let's check config.
        # If 'z' is in opt_config, we use it.
        if 'z' in opt_config:
            # Init z (usually 0)
            z_init = opt_config['z'].get('init', 0.0)
            params['z'] = jnp.full(num_nodes, z_init)
            print(f"Topography Optimization Enabled. Initial Z={z_init:.4f}")
        
        start_params = params
        
        # Pre-calculate mass integration weights for low-res mesh
        dx_l = self.fem.Lx / self.fem.nx
        dy_l = self.fem.Ly / self.fem.ny
        cell_area_l = dx_l * dy_l
        
        weights_mass = jnp.ones((self.fem.nx+1, self.fem.ny+1))
        weights_mass = weights_mass.at[0, :].multiply(0.5)
        weights_mass = weights_mass.at[-1, :].multiply(0.5)
        weights_mass = weights_mass.at[:, 0].multiply(0.5)
        weights_mass = weights_mass.at[:, -1].multiply(0.5)
        weights_mass = weights_mass.flatten() # Flatten to match nodal params
        
        target_mass = self.target_mass if hasattr(self, 'target_mass') else 1.0
        
        # 3. Loss Function
        @jax.jit
        def loss_fn(params):
            K, M = self.fem.assemble(params)
            
            total_loss = 0.0
            aux = {} # Store individual loss components
            
            # A. Static Displacement Loss
            loss_static = 0.0
            loss_curvature = 0.0
            loss_moment = 0.0
            loss_strain_energy = 0.0
            loss_surface_stress = 0.0
            loss_surface_strain = 0.0
            
            for bc in bcs_list:
                # Solve Static
                u = self.fem.solve_static_partitioned(K, bc['F'], bc['free'], bc['fd'], bc['fv'])
                
                # 1. Displacement MSE
                # Focus on Z-displacement (w) mostly? Or all components?
                # Target u_static is full 6-DOF vector.
                # However, 6-DOF vs 5-DOF (if solver changed) might be issue?
                # Assuming dimension match.
                diff = u - bc['target_u']
                
                # Weighting: mostly care about w?
                # Let's weight w higher or just MSE on all.
                # w indices: 2::6
                w_diff = diff[2::6]
                mse = jnp.mean(w_diff**2)
                loss_static += mse * bc['weight']
                
                # 2. Curvature Loss (Legacy)
                if use_curvature and 'target_curvature' in bc:
                    # Compute curvature
                    curv = self.fem.compute_curvature(u)
                    mse_curv = jnp.mean((curv - bc['target_curvature'])**2)
                    loss_curvature += mse_curv * bc['weight']
                    
                # 3. Moment Loss (Legacy)
                if use_moment and 'target_moment' in bc:
                    # Compute moment
                    params_vals = params # passed to moment calc?
                    mom = self.fem.compute_moment(u, params)
                    mse_mom = jnp.mean((mom - bc['target_moment'])**2)
                    loss_moment += mse_mom * bc['weight']
                
                # 4. Strain Energy Density Loss (New)
                if use_strain_energy and 'target_strain_energy' in bc:
                     sed = self.fem.compute_strain_energy_density(u, params)
                     # Normalize?
                     mse_sed = jnp.mean((sed - bc['target_strain_energy'])**2)
                     # Maybe normalize by target energy magnitude?
                     # scale = 1.0 / (jnp.mean(bc['target_strain_energy']**2) + 1e-9)
                     loss_strain_energy += mse_sed * bc['weight']

                # 5. Surface Stress Loss (New)
                if use_surface_stress and 'target_surface_stress' in bc:
                     stress = self.fem.compute_max_surface_stress(u, params)
                     mse_stress = jnp.mean((stress - bc['target_surface_stress'])**2)
                     loss_surface_stress += mse_stress * bc['weight']

                # 6. Surface Strain Loss (New)
                if use_surface_strain and 'target_surface_strain' in bc:
                     strain = self.fem.compute_max_surface_strain(u, params)
                     mse_strain = jnp.mean((strain - bc['target_surface_strain'])**2)
                     loss_surface_strain += mse_strain * bc['weight']

            total_loss += loss_static * loss_weights.get('static', 1.0)
            aux['static'] = loss_static
            
            if use_curvature:
                total_loss += loss_curvature * loss_weights.get('curvature', 0.0)
                aux['curvature'] = loss_curvature
            else:
                aux['curvature'] = 0.0
                
            if use_moment:
                total_loss += loss_moment * loss_weights.get('moment', 0.0)
                aux['moment'] = loss_moment
            else:
                aux['moment'] = 0.0

            if use_strain_energy:
                total_loss += loss_strain_energy * loss_weights.get('strain_energy', 0.0)
                aux['strain_energy'] = loss_strain_energy
            else:
                aux['strain_energy'] = 0.0

            if use_surface_stress:
                total_loss += loss_surface_stress * loss_weights.get('surface_stress', 0.0)
                aux['surface_stress'] = loss_surface_stress
            else:
                aux['surface_stress'] = 0.0

            if use_surface_strain:
                total_loss += loss_surface_strain * loss_weights.get('surface_strain', 0.0)
                aux['surface_strain'] = loss_surface_strain
            else:
                aux['surface_strain'] = 0.0
            
            # B. Eigenfrequency & Mode Shape Loss
            loss_freq = 0.0
            loss_mode = 0.0
            
            if loss_weights.get('freq', 0) > 0 or loss_weights.get('mode', 0) > 0:
                vals, vecs = self.fem.solve_eigen(K, M, num_modes=num_modes_loss + 3) # +3 for rigid body
                
                # Assume sorted. Skip first 3 rigid body modes (near 0)
                vals_opt = vals[3:]
                vecs_opt = vecs[:, 3:]
                
                # Dimensions match?
                n_modes = min(len(vals_opt), len(t_vals))
                
                # Freq Error
                if loss_weights.get('freq', 0) > 0:
                    # Use relative error squared
                    err = (vals_opt[:n_modes] - t_vals[:n_modes]) / (t_vals[:n_modes] + 1e-6)
                    loss_freq = jnp.mean(err**2)
                    total_loss += loss_freq * loss_weights.get('freq', 0)
                
                # MAC Loss (1 - MAC)
                if loss_weights.get('mode', 0) > 0:
                    # Target modes t_modes_l are (N_l, num_modes)
                    # Opt modes vecs_opt are (dof, num_modes)
                    # We need to extract w-component or full vector?
                    # t_modes_l provided from generate_targets are FULL vectors interpolated?
                    # verify: griddata on t_modes_h (N_h, dof?). 
                    # Step 728: t_modes_h = self.target_eigen['modes'].
                    # Solver returns (total_dof, num_modes).
                    # So griddata was likely wrong if applied to flattened DOF vector directly roughly?
                    # The griddata in step 731: griddata(..., t_modes_h[:,i], ...) implies t_modes_h[:,i] is values at nodes?
                    # But t_modes_h has size total_dof = 6*N.
                    # Griddata expects values at points.
                    # This interpolation of modes was likely buggy if passed raw DOF vector to N points.
                    # Assuming we fix this or ignore mode loss for now if broken.
                    # Let's assume t_modes_l is valid (dof, n_modes).
                    
                    # Compute MAC
                    for i in range(n_modes):
                        v1 = vecs_opt[:, i]
                        v2 = t_modes_l[:, i]
                        
                        # Normalize
                        v1 = v1 / jnp.linalg.norm(v1)
                        v2 = v2 / jnp.linalg.norm(v2)
                        
                        mac = (jnp.dot(v1, v2))**2
                        loss_mode += (1.0 - mac)
                        
                    loss_mode /= n_modes
                    total_loss += loss_mode * loss_weights.get('mode', 0)
            
            aux['freq'] = loss_freq
            aux['mode'] = loss_mode
            
            # C. Regularization (TV)
            loss_reg = 0.0
            if loss_weights.get('reg', 0) > 0:
                # TV of thickness
                t_field = params['t'].reshape(self.fem.nx + 1, self.fem.ny + 1)
                tv_x = jnp.mean(jnp.abs(t_field[1:, :] - t_field[:-1, :]))
                tv_y = jnp.mean(jnp.abs(t_field[:, 1:] - t_field[:, :-1]))
                loss_reg += (tv_x + tv_y)
                
                # TV of Z (Topography) - Important for smooth shapes
                if 'z' in params:
                     z_field = params['z'].reshape(self.fem.nx + 1, self.fem.ny + 1)
                     tv_z_x = jnp.mean(jnp.abs(z_field[1:, :] - z_field[:-1, :]))
                     tv_z_y = jnp.mean(jnp.abs(z_field[:, 1:] - z_field[:, :-1]))
                     loss_reg += (tv_z_x + tv_z_y) * 0.1 # Scale z-smoothing?

                total_loss += loss_reg * loss_weights.get('reg', 0)
            aux['reg'] = loss_reg
            
            # D. Mass Constraint (Penalty)
            loss_mass = 0.0
            if use_mass_constraint:
                # Calculate current mass
                # M = sum(rho * t * weights * cell_area)
                current_mass = jnp.sum(params['rho'] * params['t'] * weights_mass) * cell_area_l
                
                # Constraint: |M - M_tgt| / M_tgt <= tol
                # Penalty: max(0, error - tol)^2
                rel_error = jnp.abs(current_mass - target_mass) / target_mass
                violation = jnp.maximum(0.0, rel_error - mass_tolerance)
                
                loss_mass = violation * 10.0 # Strict penalty
                total_loss += loss_mass * loss_weights.get('mass', 1.0)
                
            aux['mass'] = loss_mass
            
            return total_loss, aux
        
        # 4. Optimization Loop
        # Define optimizer
        optimizer = optax.adam(learning_rate=0.01) # Slower LR for stability
        opt_state = optimizer.init(params)
        
        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            
            # Mask gradients for non-optimized parameters
            grads_masked = {}
            for k, g in grads.items():
                if opt_config.get(k, {}).get('opt', False) or (k == 'z' and 'z' in opt_config):
                    grads_masked[k] = g
                else:
                    grads_masked[k] = jnp.zeros_like(g)
            
            updates, new_opt_state = optimizer.update(grads_masked, opt_state, params)
            p_new = optax.apply_updates(params, updates)
            
            # Enforce Bounds
            for k in p_new:
                if k in opt_config:
                    info = opt_config[k]
                    if 'min' in info and 'max' in info:
                        p_new[k] = jnp.clip(p_new[k], info['min'], info['max'])
                elif k == 'z':
                     # Default bounds for Z if not specified? 
                     # E.g. +/- 30mm
                     p_new[k] = jnp.clip(p_new[k], -50.0, 50.0)

            return p_new, new_opt_state, loss, aux
        
        # Build dynamic header
        loss_names = ['Static', 'Freq', 'Mode', 'StrainE', 'SurfStr', 'SurfEps', 'Reg', 'Mass']
        active_flags = [
            True,
            loss_weights.get('freq', 0) > 0,
            loss_weights.get('mode', 0) > 0,
            use_strain_energy,
            use_surface_stress,
            use_surface_strain,
            loss_weights.get('reg', 0) > 0,
            use_mass_constraint
        ]
        active_indices = [i for i, active in enumerate(active_flags) if active]
        
        header = f"{'Iter':<5} | {'Loss':<10}"
        for idx in active_indices:
            header += f" | {loss_names[idx]:<10}"
        print(header)
        
        print_interval = max(1, max_iterations // 10)
        best_loss = 1e9
        patience_counter = 0
        
        start_time = time.time()
        for i in range(max_iterations + 1):
            params, opt_state, L, aux = step(params, opt_state)
            
            # Map aux dictionary to list for printing
            aux_vals = [
                aux['static'], aux['freq'], aux['mode'], 
                aux['strain_energy'], aux['surface_stress'], aux['surface_strain'],
                aux['reg'], aux['mass']
            ]
            
            if i % print_interval == 0 or i == max_iterations:
                row = f"{i:<5} | {L:<10.4f}"
                for idx in active_indices:
                    row += f" | {aux_vals[idx]:<10.4f}"
                print(row)
            
            # Early stopping
            if use_early_stopping:
                loss_val = float(L)
                if loss_val < best_loss - early_stop_tol:
                    best_loss = loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= (early_stop_patience or 20) and i > 20:
                    print(f"Early stopping at iteration {i}")
                    break
        
        print(f"Optimization finished in {time.time()-start_time:.2f}s")
        self.optimized_params = params
        return params

    def verify(self, num_modes_compare=None):
        if num_modes_compare is None:
            if hasattr(self, 'num_modes_truth'):
                num_modes_compare = self.num_modes_truth
            else:
                num_modes_compare = 5

        print(f"Running Verification Plots (High Resolution, Modes: {num_modes_compare})...")
        
        # 1. Setup High Res Verification Model
        if hasattr(self, 'resolution_high'):
             Nx_v, Ny_v = self.resolution_high
        else:
             Nx_v, Ny_v = 50, 20
             
        fem_verify = PlateFEM(self.fem.Lx, self.fem.Ly, Nx_v, Ny_v)
        
        # 2. Interpolate Optimized Params to High Res
        from scipy.interpolate import griddata
        
        # Low Res Coords
        # Ensure we use Nodes or Element Centers correctly.
        # Check optimize() init: params size is (nx+1)*(ny+1) -> NODAL
        xl = np.linspace(0, self.fem.Lx, self.fem.nx+1)
        yl = np.linspace(0, self.fem.Ly, self.fem.ny+1)
        Xl, Yl = np.meshgrid(xl, yl, indexing='ij')
        pts_src = np.column_stack([Xl.flatten(), Yl.flatten()])
        
        # Destination: High Res Nodes
        xv = np.linspace(0, self.fem.Lx, Nx_v+1)
        yv = np.linspace(0, self.fem.Ly, Ny_v+1)
        X_dst, Y_dst = np.meshgrid(xv, yv, indexing='ij')
        pts_dst = np.column_stack([X_dst.flatten(), Y_dst.flatten()])
        
        params_v = {}
        # Ensure params keys exist.
        for k in ['t', 'rho', 'E']:
             if k in self.optimized_params:
                 val_l = self.optimized_params[k].flatten()
                 # Interpolate
                 val_v = griddata(pts_src, val_l, pts_dst, method='cubic', fill_value=np.mean(val_l))
                 
                 # Fix NaNs from griddata (extrapolation)
                 mask = np.isnan(val_v)
                 if np.any(mask):
                     val_v[mask] = griddata(pts_src, val_l, pts_dst[mask], method='nearest')
                     
                 params_v[k] = jnp.array(val_v)
             else:
                 # Should not happen if initialized properly
                 print(f"Warning: Param {k} missing in optimized_params!")

        # Ensure z is present (if not optimized, it's 0)
        if 'z' in self.optimized_params:
            val_l = self.optimized_params['z'].flatten()
            val_v = griddata(pts_src, val_l, pts_dst, method='cubic', fill_value=0.0)
            mask = np.isnan(val_v)
            if np.any(mask):
                 val_v[mask] = griddata(pts_src, val_l, pts_dst[mask], method='nearest')
            params_v['z'] = jnp.array(val_v)
        else:
            params_v['z'] = jnp.zeros(pts_dst.shape[0])

        # 3. Assemble High Res
        K_v, M_v = fem_verify.assemble(params_v)
        
        # 4. Verify Matches
        x_plt = np.linspace(0, self.fem.Lx, Nx_v+1)
        y_plt = np.linspace(0, self.fem.Ly, Ny_v+1)
        
        static_results = []
        
        for i, case in enumerate(self.cases):
            tgt = self.targets[i] # High Res Target
            
            # Solve Optimized High Res
            fd, fv, F = case.get_bcs(fem_verify)
            
            all_dofs = np.arange(fem_verify.total_dof)
            free = np.setdiff1d(all_dofs, fd)
            
            u = fem_verify.solve_static_partitioned(K_v, F, jnp.array(free), fd, fv)
            
            # Displacement (w is index 2)
            z_opt = u[0::6].flatten() # Wait, u structure [u,v,w,tx,ty,tz] -> w is 2::6
            # My previous code said 0::3 for 3-DOF solver, but 6-DOF is 2::6?
            # Let's check PlateFEM solver definition.
            # Assuming 6-DOF per node as updated.
            # u structure: [u, v, w, tx, ty, tz]
            # w -> 2
            
            w_opt = u[2::6].reshape(Nx_v+1, Ny_v+1)
            # Target u_static is also 6-DOF vector from Generate Targets
            w_ref = tgt['u_static'][2::6].reshape(Nx_v+1, Ny_v+1)
            
            # Compute max surface stress and strain
            stress_opt = fem_verify.compute_max_surface_stress(u, params_v).reshape(Nx_v+1, Ny_v+1)
            strain_opt = fem_verify.compute_max_surface_strain(u, params_v).reshape(Nx_v+1, Ny_v+1)
            
            stress_ref = tgt['max_surface_stress'].reshape(Nx_v+1, Ny_v+1)
            strain_ref = tgt['max_surface_strain'].reshape(Nx_v+1, Ny_v+1)
            
            # Metrics Calculation
            def calc_metrics(ref, opt):
                rmse = np.sqrt(np.mean((ref - opt)**2))
                data_range = np.max(ref) - np.min(ref)
                sim = 100.0 if data_range < 1e-9 else max(0.0, (1.0 - rmse/data_range)*100)
                return sim, rmse
            
            disp_sim, disp_rmse = calc_metrics(w_ref, w_opt)
            stress_sim, stress_rmse = calc_metrics(stress_ref, stress_opt)
            strain_sim, strain_rmse = calc_metrics(strain_ref, strain_opt)
            
            static_results.append({
                'case': case.name,
                'disp_sim': disp_sim, 'stress_sim': stress_sim, 'strain_sim': strain_sim,
                'disp_max_ref': np.max(np.abs(w_ref)), 'disp_max_opt': np.max(np.abs(w_opt))
            })

            # Create 3x3 plot
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            
            def get_robust_levels(data, n_sigma=2.0, num_levels=31):
                """Calculate robust min/max using Mean +/- N*Sigma and return levels."""
                mu = np.mean(data)
                sigma = np.std(data)
                vmin = max(np.min(data), mu - n_sigma * sigma)
                vmax = min(np.max(data), mu + n_sigma * sigma)
                if vmax <= vmin:
                    vmin, vmax = np.min(data), np.max(data)
                if vmax <= vmin: 
                    vmax = vmin + 1e-8
                return np.linspace(vmin, vmax, num_levels)

            # Determine consistent robust levels
            disp_levels = get_robust_levels(w_ref, n_sigma=2.5) 
            strain_levels = get_robust_levels(strain_ref * 1000, n_sigma=2.0)
            stress_levels = get_robust_levels(stress_ref, n_sigma=2.0)
            
            def add_stats_text(ax, data_ref, data_opt, unit=""):
                txt = (f"Target: min={np.min(data_ref):.3f}, max={np.max(data_ref):.3f} {unit}\n"
                       f"Optimized: min={np.min(data_opt):.3f}, max={np.max(data_opt):.3f} {unit}")
                ax.text(0.5, -0.15, txt, transform=ax.transAxes, 
                        ha='center', va='top', fontsize=8, color='darkblue')

            # Row 1: Displacement (w)
            im0 = axes[0, 0].contourf(x_plt, y_plt, w_ref.T, levels=disp_levels, cmap='jet', extend='both')
            axes[0, 0].set_title(f"{case.name} - Height(w) Target (mm)")
            axes[0, 0].set_aspect('equal')
            plt.colorbar(im0, ax=axes[0, 0])
            
            im1 = axes[0, 1].contourf(x_plt, y_plt, w_opt.T, levels=disp_levels, cmap='jet', extend='both')
            axes[0, 1].set_title(f"{case.name} - Height(w) Optimized (mm)")
            axes[0, 1].set_aspect('equal')
            plt.colorbar(im1, ax=axes[0, 1])
            add_stats_text(axes[0, 1], w_ref, w_opt, "mm")
            
            im2 = axes[0, 2].contourf(x_plt, y_plt, np.abs(w_opt - w_ref).T, levels=30, cmap='magma')
            axes[0, 2].set_title("Height(w) Error (mm)")
            axes[0, 2].set_aspect('equal')
            plt.colorbar(im2, ax=axes[0, 2])
            
            # Row 2: Max Surface Strain
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
            
            # Row 3: Max Surface Stress
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
            
        # 4.5 Plot Optimized Parameters
        t_opt = params_v['t'].reshape(Nx_v+1, Ny_v+1)
        z_opt = params_v['z'].reshape(Nx_v+1, Ny_v+1)
        # Plot Thickness & Z
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        im0 = axes[0].pcolormesh(x_plt, y_plt, t_opt.T, cmap='viridis', shading='nearest')
        axes[0].set_title('Optimized Thickness (mm)')
        plt.colorbar(im0, ax=axes[0])
        axes[0].set_aspect('equal')
        
        im1 = axes[1].pcolormesh(x_plt, y_plt, z_opt.T, cmap='terrain', shading='nearest')
        axes[1].set_title('Optimized Topography Z (mm)')
        plt.colorbar(im1, ax=axes[1])
        axes[1].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig("verify_parameters.png")
        plt.close()
             
        # 5. Verify Modes
        print("Verifying Modes (High Res)...")
        vals, vecs = fem_verify.solve_eigen(K_v, M_v, num_modes=num_modes_compare + 10)
        
        # Target modes
        t_vals = self.target_eigen['vals']
        t_modes = self.target_eigen['modes'] # Full (dof, num_modes)
        
        # Opt Modes
        o_vals = vals[3:] # Skip 3 rigid body modes
        o_vecs = vecs[:, 3:]
        
        # Filter Rigid Body Modes based on Frequency
        freq_t_all = np.sqrt(np.abs(t_vals)) / (2*np.pi)
        freq_o_all = np.sqrt(np.abs(o_vals)) / (2*np.pi)
        
        # Compare first N modes
        num_modes_plot = min(len(freq_t_all), num_modes_compare)
        freq_t = freq_t_all[:num_modes_plot]
        freq_o = freq_o_all[:num_modes_plot]
        
        macs = []
        for j in range(num_modes_plot):
            v1 = o_vecs[:, j]
            # Use interpolated target eigenmode at high res?
            # self.target_vecs is (dof, num_modes) at High Res.
            # So we can compare directly if fem_verify shape is same.
            # fem_verify is High Res. And generate_targets uses same High Res (resolution_high).
            # So shapes should match exactly.
            v2 = self.target_vecs[:, j]
            
            # Normalize
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            
            mac = (np.dot(v1, v2))**2
            macs.append(mac)
            
        print(f"Modal Analysis: Evaluated {num_modes_plot} modes.")
        for j in range(num_modes_plot):
             print(f"Mode {j+1}: tgt={freq_t[j]:.2f}Hz, opt={freq_o[j]:.2f}Hz, MAC={macs[j]:.4f}")

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
    # Define Target Config with Z-pattern
    target_config = {
        'base_t': 1.0, 
        'bead_t': {'A': 2.0, 'B': 2.5, 'C': 1.5},  
        'bead_pz': {'T': 20.0, 'N': 10.0, 'Y': -20.0},
        'base_rho': 7.85e-9,  
        'base_E': 210000.0,   
        'pattern': 'ABC',       
        'pattern_pz': 'TNY' 
    }
    try:
        print("DEBUG: Calling generate_targets...", flush=True)
        model.generate_targets(resolution_high=(Nx_high, Ny_high), num_modes_save=6, target_config=target_config)
        print("DEBUG: generate_targets completed.", flush=True)
    except Exception as e:
        print(f"CRASH in generate_targets: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 4. Optimize
    cfg = {
        't': {'opt': True, 'min': 0.5, 'max': 3.0, 'init': 1.0},
        'rho': {'opt': False, 'min': 1e-9, 'max': 1e-8, 'init': 7.85e-9},
        'E': {'opt': False, 'min': 40e3, 'max': 250e3, 'init': 210e3},
        'z': {'opt': True, 'init': 0.0} # Enable Z optimization
    }
    
    weights = { 
        'static': 1.0,           
        'freq': 1.0,             
        'mode': 1.0,             
        'curvature': 0.0,        
        'moment': 0.0,           
        'strain_energy': 2.0,    
        'surface_stress': 1.0,   
        'surface_strain': 1.0,   
        'reg': 0.05,             
        'mass': 1.0              
    }
    
    try:
        print("DEBUG: Calling model.optimize...", flush=True)
        model.optimize(cfg, weights, use_smoothing=False, 
                       use_curvature=False, use_moment=False,
                       use_strain_energy=True, use_surface_stress=True, use_surface_strain=True,
                       use_mass_constraint=True, mass_tolerance=0.05,
                       max_iterations=300, 
                       use_early_stopping=False, 
                       early_stop_patience=None, 
                       early_stop_tol=1e-8,
                       num_modes_loss=None)
        print("DEBUG: model.optimize completed.", flush=True)
    except Exception as e:
        print(f"CRASH in optimize: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 5. Verify
    model.verify(num_modes_compare=None)
