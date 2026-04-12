# ==============================================================================
# WHT_EQS_load_cases.py
# ==============================================================================
import jax.numpy as jnp
import numpy as np
from opt_targets import OptTarget, TargetType, CompareMode

class LoadCase:
    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight
        self.opt_targets = []

    def get_bcs(self, fem):
        raise NotImplementedError("Subclasses must implement get_bcs()")

# ==============================================================================
# TWIST LOAD CASE (6-DOF Aligned)
# ==============================================================================
class TwistCase(LoadCase):
    def __init__(self, name, axis='x', value=1.5, mode='angle', weight=1.0, tol=1e-3):
        super().__init__(name, weight)
        self.axis = axis
        self.value = value
        self.mode = mode
        self.tol = tol
        # Default opt_target: compare RBE residual magnitude by default for twist
        try:
            self.opt_targets = [OptTarget(target_type=TargetType.RBE_REACTION,
                                          rbe_id='residual',
                                          component='magnitude',
                                          compare_mode=CompareMode.RELATIVE,
                                          weight=self.weight)]
        except Exception:
            self.opt_targets = []
        
    def _get_bcs_lowlevel_v1(self, fem):
        """[BACKUP] Legacy indexing-based BC implementation."""
        Lx, Ly = fem.Lx, fem.Ly
        F = jnp.zeros(fem.total_dof)
        fixed_dofs = []
        fixed_vals = []
        
        # DOF Mapping: 0:u, 1:v, 2:w, 3:tx, 4:ty, 5:tz
        if self.axis == 'x':
            left_nodes = jnp.where(fem.node_coords[:, 0] < self.tol)[0]
            right_nodes = jnp.where(fem.node_coords[:, 0] > Lx - self.tol)[0]
            yc = Ly / 2.0
            
            if self.mode == 'angle':
                angle_rad = self.value * np.pi / 180.0
                y_left = fem.node_coords[left_nodes, 1]
                w_left = (y_left - yc) * np.tan(-angle_rad)
                for i, node in enumerate(left_nodes):
                    # LEFT GRIP: Fully Fixed (Anchor)
                    # Fix w, tx (twist) + u, v, tz (rigid body anchor)
                    fixed_dofs.extend([node*6 + 0, node*6 + 1, node*6 + 2, node*6 + 3, node*6 + 5]) 
                    fixed_vals.extend([0.0, 0.0, w_left[i], -angle_rad, 0.0])
                
                y_right = fem.node_coords[right_nodes, 1]
                w_right = (y_right - yc) * np.tan(angle_rad)
                for i, node in enumerate(right_nodes):
                    # RIGHT GRIP: Release Axial (u)
                    # Fix w, tx (twist) + v (prevent transverse)
                    fixed_dofs.extend([node*6 + 1, node*6 + 2, node*6 + 3]) 
                    fixed_vals.extend([0.0, w_right[i], angle_rad])
                
        elif self.axis == 'y':
            bot_nodes = jnp.where(fem.node_coords[:, 1] < self.tol)[0]
            top_nodes = jnp.where(fem.node_coords[:, 1] > Ly - self.tol)[0]
            xc = Lx / 2.0
            
            if self.mode == 'angle':
                angle_rad = self.value * np.pi / 180.0
                x_bot = fem.node_coords[bot_nodes, 0]
                w_bot = (x_bot - xc) * np.tan(-angle_rad)
                for i, node in enumerate(bot_nodes):
                    # BOTTOM GRIP: Fully Fixed (Anchor)
                    # Fix w, ty (twist) + u, v, tz (rigid body anchor)
                    fixed_dofs.extend([node*6 + 0, node*6 + 1, node*6 + 2, node*6 + 4, node*6 + 5])
                    fixed_vals.extend([0.0, 0.0, w_bot[i], -angle_rad, 0.0])
                
                x_top = fem.node_coords[top_nodes, 0]
                w_top = (x_top - xc) * np.tan(angle_rad)
                for i, node in enumerate(top_nodes):
                    # TOP GRIP: Release Axial (v)
                    # Fix w, ty (twist) + u (prevent transverse)
                    fixed_dofs.extend([node*6 + 0, node*6 + 2, node*6 + 4]) 
                    fixed_vals.extend([0.0, w_top[i], angle_rad])

        return jnp.array(fixed_dofs), jnp.array(fixed_vals), F

    def get_bcs(self, fem):
        """High-level implementation using PlateFEM functional APIs."""
        fem.clear_bcs()
        Lx, Ly = fem.Lx, fem.Ly
        angle_rad = self.value * np.pi / 180.0

        if self.axis == 'x':
            yc = Ly / 2.0
            # LEFT GRIP: Rigid rotation by -angle around X
            fem.add_constraint_field(x_range=(None, self.tol), dofs=[0,1,2,3,4,5], 
                                     func=lambda x,y,z: [0.0, (y-yc)*(np.cos(angle_rad)-1.0) + z*np.sin(angle_rad), -(y-yc)*np.sin(angle_rad) + z*(np.cos(angle_rad)-1.0), -angle_rad, 0.0, 0.0])
            # RIGHT GRIP: Rigid rotation by +angle around X (x free to slide -> u is free)
            fem.add_constraint_field(x_range=(Lx-self.tol, None), dofs=[1,2,3,4,5],
                                     func=lambda x,y,z: [(y-yc)*(np.cos(angle_rad)-1.0) - z*np.sin(angle_rad), (y-yc)*np.sin(angle_rad) + z*(np.cos(angle_rad)-1.0), angle_rad, 0.0, 0.0])
                
        elif self.axis == 'y':
            xc = Lx / 2.0
            # BOTTOM GRIP: Rigid rotation by -angle around Y
            fem.add_constraint_field(y_range=(None, self.tol), dofs=[0,1,2,3,4,5],
                                     func=lambda x,y,z: [(x-xc)*(np.cos(angle_rad)-1.0) - z*np.sin(angle_rad), 0.0, (x-xc)*np.sin(angle_rad) + z*(np.cos(angle_rad)-1.0), 0.0, -angle_rad, 0.0])
            # TOP GRIP: Rigid rotation by +angle around Y (y free to slide -> v is free)
            fem.add_constraint_field(y_range=(Ly-self.tol, None), dofs=[0,2,3,4,5],
                                     func=lambda x,y,z: [(x-xc)*(np.cos(angle_rad)-1.0) + z*np.sin(angle_rad), -(x-xc)*np.sin(angle_rad) + z*(np.cos(angle_rad)-1.0), 0.0, angle_rad, 0.0])

        return fem.export_bcs()

# ==============================================================================
# PURE BENDING LOAD CASE (6-DOF Aligned)
# ==============================================================================
class PureBendingCase(LoadCase):
    def __init__(self, name, axis='y', value=3.0, mode='angle', weight=1.0, tol=1e-3):
        super().__init__(name, weight)
        self.axis = axis
        self.value = value
        self.mode = mode
        self.tol = tol
        
    def _get_bcs_lowlevel_v1(self, fem):
        """[BACKUP] Legacy indexing-based BC implementation."""
        Lx, Ly = fem.Lx, fem.Ly
        F = jnp.zeros(fem.total_dof)
        fixed_dofs = []
        fixed_vals = []
        
        if self.axis == 'y':
            left_nodes = jnp.where(fem.node_coords[:, 0] < self.tol)[0]
            right_nodes = jnp.where(fem.node_coords[:, 0] > Lx - self.tol)[0]
            if self.mode == 'angle':
                angle_rad = self.value * np.pi / 180.0
                for node in left_nodes:
                    # LEFT GRIP: Fully Fixed (Anchor)
                    # Fix w, ty (bend) + u, v, tz (rigid body)
                    fixed_dofs.extend([node*6+0, node*6+1, node*6+2, node*6+4, node*6+5])
                    fixed_vals.extend([0.0, 0.0, 0.0, angle_rad, 0.0]) 
                for node in right_nodes:
                    # RIGHT GRIP: Release Axial (u)
                    # Fix w, ty (bend) + v (transverse)
                    fixed_dofs.extend([node*6+1, node*6+2, node*6+4])
                    fixed_vals.extend([0.0, 0.0, -angle_rad])
                
        elif self.axis == 'x':
            bot_nodes = jnp.where(fem.node_coords[:, 1] < self.tol)[0]
            top_nodes = jnp.where(fem.node_coords[:, 1] > Ly - self.tol)[0]
            if self.mode == 'angle':
                angle_rad = self.value * np.pi / 180.0
                for node in bot_nodes:
                     # BOTTOM GRIP: Fully Fixed (Anchor)
                     # Fix w, tx (bend) + u, v, tz (rigid body)
                     fixed_dofs.extend([node*6+0, node*6+1, node*6+2, node*6+3, node*6+5])
                     fixed_vals.extend([0.0, 0.0, 0.0, -angle_rad, 0.0])
                for node in top_nodes:
                     # TOP GRIP: Release Axial (v)
                     # Fix w, tx (bend) + u (transverse)
                     fixed_dofs.extend([node*6+0, node*6+2, node*6+3])
                     fixed_vals.extend([0.0, 0.0, angle_rad])

        return jnp.array(fixed_dofs), jnp.array(fixed_vals), F

    def get_bcs(self, fem):
        """High-level implementation using PlateFEM APIs."""
        fem.clear_bcs()
        Lx, Ly = fem.Lx, fem.Ly
        angle_rad = self.value * np.pi / 180.0

        if self.axis == 'y':
            # LEFT GRIP: ty = angle, [u,v,w,tx,tz]=0 (Fully Clamped Edge)
            fem.add_constraint(x_range=(None, self.tol), dofs=[0,1,2,3,4,5], value=[0.0, 0.0, 0.0, 0.0, angle_rad, 0.0])
            # RIGHT GRIP: ty = -angle, [v,w,tx,tz]=0 (u is free to slide)
            fem.add_constraint(x_range=(Lx-self.tol, None), dofs=[1,2,3,4,5], value=[0.0, 0.0, 0.0, -angle_rad, 0.0])
                
        elif self.axis == 'x':
            # BOTTOM GRIP: tx = -angle, [u,v,w,ty,tz]=0 (Fully Clamped Edge)
            fem.add_constraint(y_range=(None, self.tol), dofs=[0,1,2,3,4,5], value=[0.0, 0.0, 0.0, -angle_rad, 0.0, 0.0])
            # TOP GRIP: tx = angle, [u,w,ty,tz]=0 (v is free to slide)
            fem.add_constraint(y_range=(Ly-self.tol, None), dofs=[0,2,3,4,5], value=[0.0, 0.0, angle_rad, 0.0, 0.0])

        return fem.export_bcs()



# ==============================================================================
# CORNER LIFT LOAD CASE (6-DOF Aligned)
# ==============================================================================
class CornerLiftCase(LoadCase):
    def __init__(self, name, corner='br', value=5.0, mode='disp', weight=1.0):
        super().__init__(name, weight)
        self.corner = corner
        self.value = value
        self.mode = mode
        
    def _get_bcs_lowlevel_v1(self, fem):
        """[BACKUP] Legacy indexing-based BC implementation."""
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
        
        # Rigid body constraint at BL
        fixed_dofs.extend([idx_bl*6+0, idx_bl*6+1, idx_bl*6+5]) # u, v, tz
        fixed_vals.extend([0.0, 0.0, 0.0])

        for k, idx in corners.items():
            if k != self.corner:
                fixed_dofs.append(idx*6 + 2) # Fix w
                fixed_vals.append(0.0)
        
        if self.mode == 'disp':
            fixed_dofs.append(target_idx*6 + 2) # Lift w
            fixed_vals.append(self.value)
        elif self.mode == 'force':
            F = F.at[target_idx*6 + 2].set(self.value)
        
        return jnp.array(fixed_dofs), jnp.array(fixed_vals), F

    def get_bcs(self, fem):
        """High-level implementation using PlateFEM APIs."""
        fem.clear_bcs()
        
        # Helper to find corner indices
        coords = fem.node_coords
        def find_corner(x, y):
            dist = (coords[:,0]-x)**2 + (coords[:,1]-y)**2
            return int(jnp.argmin(dist))
            
        idx_bl = find_corner(0, 0)
        idx_br = find_corner(fem.Lx, 0)
        idx_tl = find_corner(0, fem.Ly)
        idx_tr = find_corner(fem.Lx, fem.Ly)
        
        corners = {'bl': idx_bl, 'br': idx_br, 'tl': idx_tl, 'tr': idx_tr}
        
        # 1. Anchor BL: u, v, w, tx, ty, tz = 0 (Fully Clamped)
        fem.constraints.append((idx_bl*6+0, 0.0))
        fem.constraints.append((idx_bl*6+1, 0.0))
        fem.constraints.append((idx_bl*6+2, 0.0))
        fem.constraints.append((idx_bl*6+3, 0.0))
        fem.constraints.append((idx_bl*6+4, 0.0))
        fem.constraints.append((idx_bl*6+5, 0.0))

        # 2. Fix W, tx, ty on other inactive corners (Clamped Support)
        for k, idx in corners.items():
            if k != self.corner and k != 'bl':
                fem.constraints.append((idx*6+2, 0.0))
                fem.constraints.append((idx*6+3, 0.0))
                fem.constraints.append((idx*6+4, 0.0))
        
        # 3. Apply Load/Disp on target
        target_idx = corners[self.corner]
        if self.mode == 'disp':
            # Lift while keeping it flat (tx=ty=0)
            fem.constraints.append((target_idx*6+2, float(self.value)))
            fem.constraints.append((target_idx*6+3, 0.0))
            fem.constraints.append((target_idx*6+4, 0.0))
        else:
            fem.loads.append((target_idx*6+2, float(self.value)))
            
        return fem.export_bcs()

# ==============================================================================
# TWO CORNER LIFT LOAD CASE (Two active, two fully fixed)
# ==============================================================================
class TwoCornerLiftCase(LoadCase):
    def __init__(self, name, corners=['br', 'tl'], value=5.0, mode='disp', weight=1.0):
        """
        Args:
            corners (list): List of 2 corners to lift (e.g., ['br', 'tl'])
            value (float): Displacement (mm) or Force (N)
            mode (str): 'disp' or 'force'
        """
        super().__init__(name, weight)
        if len(corners) != 2:
            raise ValueError("TwoCornerLiftCase requires exactly 2 corners.")
        self.corners = corners
        self.value = value
        self.mode = mode
        
    def _get_bcs_lowlevel_v1(self, fem):
        """[BACKUP] Legacy indexing-based BC implementation."""
        coords = fem.node_coords
        idx_bl = jnp.argmin(coords[:,0]**2 + coords[:,1]**2)
        idx_br = jnp.argmin((coords[:,0]-fem.Lx)**2 + coords[:,1]**2)
        idx_tl = jnp.argmin(coords[:,0]**2 + (coords[:,1]-fem.Ly)**2)
        idx_tr = jnp.argmin((coords[:,0]-fem.Lx)**2 + (coords[:,1]-fem.Ly)**2)
        
        all_corners = {'bl': idx_bl, 'br': idx_br, 'tl': idx_tl, 'tr': idx_tr}
        
        fixed_dofs = []
        fixed_vals = []
        F = jnp.zeros(fem.total_dof)
        
        for name, idx in all_corners.items():
            if name in self.corners:
                # Active Corner: Apply Z-load or Z-disp
                if self.mode == 'disp':
                    fixed_dofs.append(idx*6 + 2) # Fix w
                    fixed_vals.append(self.value)
                elif self.mode == 'force':
                    F = F.at[idx*6 + 2].set(self.value)
                # Other DOFs (u, v, tx, ty, tz) remain FREE
            else:
                # Inactive Corner: FULLY FIXED (6-DOF)
                for dof_offset in range(6):
                    fixed_dofs.append(idx*6 + dof_offset)
                    fixed_vals.append(0.0)
        
        return jnp.array(fixed_dofs), jnp.array(fixed_vals), F

    def get_bcs(self, fem):
        """High-level implementation using PlateFEM APIs."""
        fem.clear_bcs()
        
        coords = fem.node_coords
        def find_corner(x, y):
            dist = (coords[:,0]-x)**2 + (coords[:,1]-y)**2
            return int(jnp.argmin(dist))
            
        all_corners = {
            'bl': find_corner(0, 0), 'br': find_corner(fem.Lx, 0),
            'tl': find_corner(0, fem.Ly), 'tr': find_corner(fem.Lx, fem.Ly)
        }
        
        active_ids = []
        for name, idx in all_corners.items():
            if name in self.corners:
                active_ids.append(idx)
            else:
                for dof in range(6):
                    fem.constraints.append((idx*6+dof, 0.0))
                    
        if self.mode == 'disp':
            for idx in active_ids:
                fem.constraints.append((idx*6+2, float(self.value)))
                fem.constraints.append((idx*6+3, 0.0))
                fem.constraints.append((idx*6+4, 0.0))
        else:
            for idx in active_ids:
                fem.loads.append((idx*6+2, float(self.value)))
                    
        return fem.export_bcs()
# ==============================================================================
# POSITIONAL LOAD CASE (General Mesh / External / ROI)
# ==============================================================================
class PositionalCase(LoadCase):
    def __init__(self, name, weight=1.0):
        super().__init__(name, weight)
        self.fixed_regions = [] # List of {'box': (min, max) or 'radius': (center, r), 'dofs': [], 'vals': []}
        self.loads = []         # List of {'box': ..., 'dofs_vals': {dof_idx: val}}

    def add_fixed_box(self, x_range=None, y_range=None, z_range=None, dofs=[0,1,2,3,4,5], vals=0.0):
        self.fixed_regions.append({
            'type': 'box', 'range': (x_range, y_range, z_range), 
            'dofs': dofs, 'vals': vals
        })

    def add_fixed_radius(self, center, radius, dofs=[0,1,2,3,4,5], vals=0.0):
        self.fixed_regions.append({
            'type': 'radius', 'center': center, 'radius': radius, 
            'dofs': dofs, 'vals': vals
        })

    def add_load_box(self, x_range=None, y_range=None, z_range=None, dof_val_dict={2: -1.0}):
        self.loads.append({
            'type': 'box', 'range': (x_range, y_range, z_range), 
            'dofs_vals': dof_val_dict
        })

    def _get_bcs_lowlevel_v1(self, fem):
        """[BACKUP] Legacy PositionalCase implementation."""
        import WHT_EQS_mesh as mesh
        coords = fem.node_coords
        F = jnp.zeros(fem.total_dof)
        fixed_dofs = []
        fixed_vals = []

        # 1. Process Fixed Regions
        for reg in self.fixed_regions:
            if reg['type'] == 'box':
                xr, yr, zr = reg['range']
                nodes = mesh.get_nodes_in_box(coords, xr, yr, zr)
            else:
                nodes = mesh.get_nodes_in_radius(coords, reg['center'], reg['radius'])
            
            for node in nodes:
                idx_start = node * 6
                for d_offset in reg['dofs']:
                    fixed_dofs.append(idx_start + d_offset)
                    if isinstance(reg['vals'], (list, np.ndarray, jnp.ndarray)):
                        fixed_vals.append(reg['vals'][d_offset])
                    else:
                        fixed_vals.append(reg['vals'])

        # 2. Process Loads
        for load in self.loads:
            if load['type'] == 'box':
                xr, yr, zr = load['range']
                nodes = mesh.get_nodes_in_box(coords, xr, yr, zr)
            
            if len(nodes) > 0:
                for node in nodes:
                    for d_offset, val in load['dofs_vals'].items():
                        F = F.at[node * 6 + d_offset].add(val)

        return jnp.array(fixed_dofs), jnp.array(fixed_vals), F

    def get_bcs(self, fem):
        """Unified implementation: Redirects to PlateFEM high-level APIs."""
        fem.clear_bcs()
        
        # Apply Fixed Regions
        for reg in self.fixed_regions:
            if reg['type'] == 'box':
                xr, yr, zr = reg['range']
                fem.add_constraint(xr, yr, zr, dofs=reg['dofs'], value=reg['vals'])
            else:
                fem.add_constraint_radius(reg['center'], reg['radius'], dofs=reg['dofs'], value=reg['vals'])
                
        # Apply Loads
        for load in self.loads:
            if load['type'] == 'box':
                xr, yr, zr = load['range']
                for dof, val in load['dofs_vals'].items():
                    fem.add_force(xr, yr, zr, dof=dof, value=val, is_total=False)
                    
        return fem.export_bcs()

# ==============================================================================
# CANTILEVER LOAD CASE (One Edge Clamped, Opposite Edge Loaded)
# ==============================================================================
class CantileverCase(LoadCase):
    def __init__(self, name, axis='x', value=-5.0, mode='disp', weight=1.0, tol=1e-3):
        super().__init__(name, weight)
        self.axis = axis       # 'x' means clamped at x=0, loaded at x=Lx
        self.value = value     # force or displacement amount
        self.mode = mode       # 'disp' or 'force'
        self.tol = tol
        
    def _get_bcs_lowlevel_v1(self, fem):
        # Placeholder for compatibility
        return self.get_bcs(fem)

    def get_bcs(self, fem):
        fem.clear_bcs()
        Lx, Ly = fem.Lx, fem.Ly
        coords = fem.node_coords

        if self.axis == 'x':
            # 1. Fully Clamp LEFT Edge (x=0)
            fem.add_constraint(x_range=(None, self.tol), dofs=[0,1,2,3,4,5], value=0.0)

            # 2. Apply Load/Disp to RIGHT Edge (x=Lx)
            if self.mode == 'disp':
                fem.add_constraint(x_range=(Lx-self.tol, None), dofs=[2], value=float(self.value))
            else:
                fem.add_force(x_range=(Lx-self.tol, None), dof=2, value=self.value, is_total=True)

        elif self.axis == 'y':
            # 1. Fully Clamp BOTTOM Edge (y=0)
            fem.add_constraint(y_range=(None, self.tol), dofs=[0,1,2,3,4,5], value=0.0)

            # 2. Apply Load/Disp to TOP Edge (y=Ly)
            if self.mode == 'disp':
                fem.add_constraint(y_range=(Ly-self.tol, None), dofs=[2], value=float(self.value))
            else:
                fem.add_force(y_range=(Ly-self.tol, None), dof=2, value=self.value, is_total=True)

        return fem.export_bcs()
# ==============================================================================
# PRESSURE LOAD CASE (All Edges Clamped, Uniform Pressure on Surface)
# ==============================================================================
class PressureCase(LoadCase):
    def __init__(self, name, value=-10.0, weight=1.0, tol=1e-3):
        """
        value: total applied Z-force distributed across the entire area
        """
        super().__init__(name, weight)
        self.value = value     
        self.tol = tol
        
    def _get_bcs_lowlevel_v1(self, fem):
        return self.get_bcs(fem)

    def get_bcs(self, fem):
        fem.clear_bcs()
        Lx, Ly = fem.Lx, fem.Ly
        
        # 1. Fully Clamp All 4 Edges (6-DOFs locked)
        fem.add_constraint(x_range=(None, self.tol), dofs=[0,1,2,3,4,5], value=0.0)      # Left
        fem.add_constraint(x_range=(Lx-self.tol, None), dofs=[0,1,2,3,4,5], value=0.0)   # Right
        fem.add_constraint(y_range=(None, self.tol), dofs=[0,1,2,3,4,5], value=0.0)      # Bottom
        fem.add_constraint(y_range=(Ly-self.tol, None), dofs=[0,1,2,3,4,5], value=0.0)   # Top
        
        # 2. Apply Uniform Z-force across the whole plate
        fem.add_force(dof=2, value=self.value, is_total=True)
                
        return fem.export_bcs()
