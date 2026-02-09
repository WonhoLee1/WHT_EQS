# ==============================================================================
# WHT_EQS_load_cases.py
# ==============================================================================
import jax.numpy as jnp
import numpy as np

class LoadCase:
    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight

    def get_bcs(self, fem):
        raise NotImplementedError("Subclasses must implement get_bcs()")

# ==============================================================================
# TWIST LOAD CASE (6-DOF Aligned)
# ==============================================================================
class TwistCase(LoadCase):
    def __init__(self, name, axis='x', value=1.5, mode='angle', weight=1.0):
        super().__init__(name, weight)
        self.axis = axis
        self.value = value
        self.mode = mode
        
    def get_bcs(self, fem):
        tol = 1e-3
        Lx, Ly = fem.Lx, fem.Ly
        F = jnp.zeros(fem.total_dof)
        fixed_dofs = []
        fixed_vals = []
        
        # DOF Mapping: 0:u, 1:v, 2:w, 3:tx, 4:ty, 5:tz
        if self.axis == 'x':
            left_nodes = jnp.where(fem.node_coords[:, 0] < tol)[0]
            right_nodes = jnp.where(fem.node_coords[:, 0] > Lx - tol)[0]
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
            bot_nodes = jnp.where(fem.node_coords[:, 1] < tol)[0]
            top_nodes = jnp.where(fem.node_coords[:, 1] > Ly - tol)[0]
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

# ==============================================================================
# PURE BENDING LOAD CASE (6-DOF Aligned)
# ==============================================================================
class PureBendingCase(LoadCase):
    def __init__(self, name, axis='y', value=3.0, mode='angle', weight=1.0):
        super().__init__(name, weight)
        self.axis = axis
        self.value = value
        self.mode = mode
        
    def get_bcs(self, fem):
        tol = 1e-3
        Lx, Ly = fem.Lx, fem.Ly
        F = jnp.zeros(fem.total_dof)
        fixed_dofs = []
        fixed_vals = []
        
        if self.axis == 'y':
            left_nodes = jnp.where(fem.node_coords[:, 0] < tol)[0]
            right_nodes = jnp.where(fem.node_coords[:, 0] > Lx - tol)[0]
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
            bot_nodes = jnp.where(fem.node_coords[:, 1] < tol)[0]
            top_nodes = jnp.where(fem.node_coords[:, 1] > Ly - tol)[0]
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



# ==============================================================================
# CORNER LIFT LOAD CASE (6-DOF Aligned)
# ==============================================================================
class CornerLiftCase(LoadCase):
    def __init__(self, name, corner='br', value=5.0, mode='disp', weight=1.0):
        super().__init__(name, weight)
        self.corner = corner
        self.value = value
        self.mode = mode
        
    def get_bcs(self, fem):
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
