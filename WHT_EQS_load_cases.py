# ==============================================================================
# WHT_EQS_load_cases.py
# ==============================================================================
# PURPOSE:
#   Load case definitions for plate verification and optimization.
#   Provides boundary condition generators for various loading scenarios.
#
# LOAD CASES:
#   - TwistCase: Torsional loading about X or Y axis
#   - PureBendingCase: Cylindrical bending deformation
#   - CornerLiftCase: Three-point support with one corner displacement
#
# USAGE:
#   from WHT_EQS_load_cases import TwistCase, PureBendingCase, CornerLiftCase
#   
#   # Create load case instances
#   twist = TwistCase('twist_x', axis='x', value=1.5, mode='angle')
#   bend = PureBendingCase('bend_y', axis='y', value=3.0, mode='angle')
#   lift = CornerLiftCase('lift_br', corner='br', value=1.0, mode='disp')
#   
#   # Get boundary conditions for FEM solver
#   fixed_dofs, fixed_vals, F = twist.get_bcs(fem)
#
# AUTHOR: Advanced FEM Team
# DATE: 2026-02-09
# ==============================================================================

import jax.numpy as jnp
import numpy as np


# ==============================================================================
# BASE CLASS
# ==============================================================================

class LoadCase:
    """
    Abstract base class for load cases.
    
    All load case classes must inherit from this and implement get_bcs().
    
    Attributes:
    -----------
    name : str
        Unique identifier for this load case
    weight : float
        Weight factor for optimization (default: 1.0)
        Higher weight = more emphasis on matching this case
    """
    
    def __init__(self, name, weight=1.0):
        """
        Initialize load case.
        
        Parameters:
        -----------
        name : str
            Load case identifier (e.g., 'twist_x', 'bend_y')
        weight : float
            Optimization weight factor (default: 1.0)
        """
        self.name = name
        self.weight = weight

    def get_bcs(self, fem):
        """
        Generate boundary conditions for this load case.
        
        Parameters:
        -----------
        fem : PlateFEM
            Finite element model instance
            
        Returns:
        --------
        tuple of (fixed_dofs, fixed_vals, F)
            fixed_dofs : jax.numpy.ndarray (int32)
                Indices of constrained DOFs
            fixed_vals : jax.numpy.ndarray (float64)
                Prescribed values for constrained DOFs
            F : jax.numpy.ndarray (float64)
                Global force vector
                
        Raises:
        -------
        NotImplementedError
            Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_bcs()")


# ==============================================================================
# TWIST LOAD CASE
# ==============================================================================

class TwistCase(LoadCase):
    """
    Torsional loading about X or Y axis.
    
    Simulates twisting deformation by prescribing rotations and displacements
    at opposite edges of the plate.
    
    Twist Modes:
    ------------
    - 'angle': Prescribe rotation angle at edges (kinematic control)
    - 'moment': Apply twisting moments at edges (force control)
    
    Coordinate Systems:
    -------------------
    - Twist about X: Ends at x=0 and x=Lx rotate oppositely
    - Twist about Y: Ends at y=0 and y=Ly rotate oppositely
    
    DOF Convention (3-DOF thin plate):
    -----------------------------------
    - DOF 0, 3, 6, ... : w (vertical displacement)
    - DOF 1, 4, 7, ... : θx (rotation about X-axis)
    - DOF 2, 5, 8, ... : θy (rotation about Y-axis)
    
    Example:
    --------
    >>> # Twist 1.5° about X-axis
    >>> case = TwistCase('twist_x', axis='x', value=1.5, mode='angle', weight=1.0)
    >>> fixed_dofs, fixed_vals, F = case.get_bcs(fem)
    """
    
    def __init__(self, name, axis=None, value=1.5, mode='angle', weight=1.0):
        """
        Initialize twist load case.
        
        Parameters:
        -----------
        name : str
            Load case name (e.g., 'twist_x_15deg')
        axis : str, optional
            Twist axis: 'x' or 'y'
            If None, auto-detects from name ('_x' or '_y' suffix)
        value : float
            - If mode='angle': Rotation angle in degrees
            - If mode='moment': Applied moment in N·mm
        mode : str
            'angle' (kinematic) or 'moment' (force)
        weight : float
            Optimization weight factor
        """
        super().__init__(name, weight)
        
        # Auto-detect axis from name if not provided
        if axis is None:
            if '_x' in name.lower():
                axis = 'x'
            elif '_y' in name.lower():
                axis = 'y'
            else:
                axis = 'x'  # default
        
        self.axis = axis      # 'x' or 'y'
        self.value = value    # degrees or N·mm
        self.mode = mode      # 'angle' or 'moment'
        
    def get_bcs(self, fem):
        """
        Generate boundary conditions for twist loading.
        
        Returns fixed DOFs, prescribed values, and force vector.
        See base class docstring for details.
        """
        tol = 1e-3
        Lx, Ly = fem.Lx, fem.Ly
        F = jnp.zeros(fem.total_dof)
        fixed_dofs = []
        fixed_vals = []
        
        if self.axis == 'x':
            # =====================================================
            # TWIST ABOUT X-AXIS
            # =====================================================
            # Plate edges at x=0 (left) and x=Lx (right)
            left_nodes = jnp.where(fem.node_coords[:, 0] < tol)[0]
            right_nodes = jnp.where(fem.node_coords[:, 0] > Lx - tol)[0]
            
            yc = Ly / 2.0  # Y-axis center (twist axis passes through here)
            
            if self.mode == 'angle':
                # Prescribe rotation angles at edges
                angle_rad = self.value * np.pi / 180.0
                
                # Left edge (x=0): θx = -angle, w = (y-yc)·tan(-angle)
                y_left = fem.node_coords[left_nodes, 1]
                w_left = (y_left - yc) * np.tan(-angle_rad)
                
                for i, node in enumerate(left_nodes):
                    fixed_dofs.extend([node*3 + 0, node*3 + 1])  # Fix w, θx
                    fixed_vals.extend([w_left[i], -angle_rad])

                # Right edge (x=Lx): θx = +angle, w = (y-yc)·tan(angle)
                y_right = fem.node_coords[right_nodes, 1]
                w_right = (y_right - yc) * np.tan(angle_rad)
                
                for i, node in enumerate(right_nodes):
                    fixed_dofs.extend([node*3 + 0, node*3 + 1])  # Fix w, θx
                    fixed_vals.extend([w_right[i], angle_rad])
                
                # Prevent rigid body motion in Y-direction
                n0 = 0  # First node
                fixed_dofs.append(n0*3 + 2)  # Fix θy
                fixed_vals.append(0.0)

            elif self.mode == 'moment':
                # Apply distributed twisting moments at edges
                m_node = self.value / len(right_nodes)  # Moment per node
                
                # Opposite moments at opposite edges
                F = F.at[right_nodes * 3 + 1].add(m_node)   # Right: +Mx
                F = F.at[left_nodes * 3 + 1].add(-m_node)  # Left: -Mx
                
                # Fix center node to prevent rigid body motion
                center_node = jnp.argmin((fem.node_coords[:,0]-Lx/2)**2 + 
                                          (fem.node_coords[:,1]-Ly/2)**2)
                fixed_dofs.extend([center_node*3+0, center_node*3+2])  # Fix w, θy
                fixed_vals.extend([0.0, 0.0])
                
        elif self.axis == 'y':
            # =====================================================
            # TWIST ABOUT Y-AXIS
            # =====================================================
            # Plate edges at y=0 (bottom) and y=Ly (top)
            bot_nodes = jnp.where(fem.node_coords[:, 1] < tol)[0]
            top_nodes = jnp.where(fem.node_coords[:, 1] > Ly - tol)[0]
            xc = Lx / 2.0  # X-axis center (twist axis passes through here)
            
            if self.mode == 'angle':
                # Prescribe rotation angles at edges
                angle_rad = self.value * np.pi / 180.0
                
                # Bottom edge (y=0): θy = -angle, w = (x-xc)·tan(-angle)
                x_bot = fem.node_coords[bot_nodes, 0]
                w_bot = (x_bot - xc) * np.tan(-angle_rad)
                
                for i, node in enumerate(bot_nodes):
                    fixed_dofs.extend([node*3 + 0, node*3 + 2])  # Fix w, θy
                    fixed_vals.extend([w_bot[i], -angle_rad])
                    
                # Top edge (y=Ly): θy = +angle, w = (x-xc)·tan(angle)
                x_top = fem.node_coords[top_nodes, 0]
                w_top = (x_top - xc) * np.tan(angle_rad)
                
                for i, node in enumerate(top_nodes):
                    fixed_dofs.extend([node*3 + 0, node*3 + 2])  # Fix w, θy
                    fixed_vals.extend([w_top[i], angle_rad])
                    
                # Prevent rigid body motion in X-direction
                n0 = 0  # First node
                fixed_dofs.append(n0*3+1)  # Fix θx
                fixed_vals.append(0.0)

        return (jnp.array(fixed_dofs, dtype=jnp.int32), 
                jnp.array(fixed_vals, dtype=jnp.float64), 
                F)


# ==============================================================================
# PURE BENDING LOAD CASE
# ==============================================================================

class PureBendingCase(LoadCase):
    """
    Pure bending (cylindrical curvature) load case.
    
    Creates constant curvature along one direction by prescribing
    opposite rotations at opposite edges.
    
    Bending Modes:
    --------------
    - 'angle': Prescribe rotation angles at edges (kinematic)
    - 'moment': Apply bending moments at edges (force) [not implemented]
    
    Geometric Interpretation:
    -------------------------
    - Bend about Y: Plate curves in XZ plane (saddle along X)
    - Bend about X: Plate curves in YZ plane (saddle along Y)
    
    Example:
    --------
    >>> # Create 3° cylindrical bend about Y-axis
    >>> case = PureBendingCase('bend_y_3deg', axis='y', value=3.0, mode='angle')
    >>> fixed_dofs, fixed_vals, F = case.get_bcs(fem)
    """
    
    def __init__(self, name, axis=None, value=3.0, mode='angle', weight=1.0):
        """
        Initialize pure bending load case.
        
        Parameters:
        -----------
        name : str
            Load case name (e.g., 'bend_y_3deg')
        axis : str, optional
            Bending axis: 'x' or 'y'
            If None, auto-detects from name
        value : float
            - If mode='angle': Rotation angle in degrees
            - If mode='moment': Applied moment (not implemented)
        mode : str
            Currently only 'angle' is supported
        weight : float
            Optimization weight factor
        """
        super().__init__(name, weight)
        
        # Auto-detect axis from name if not provided
        if axis is None:
            if '_x' in name.lower():
                axis = 'x'
            elif '_y' in name.lower():
                axis = 'y'
            else:
                axis = 'y'  # default
        
        self.axis = axis   # bending axis (curvature about this axis)
        self.value = value
        self.mode = mode
        
    def get_bcs(self, fem):
        """
        Generate boundary conditions for pure bending.
        
        Returns fixed DOFs, prescribed values, and force vector.
        See base class docstring for details.
        """
        tol = 1e-3
        Lx, Ly = fem.Lx, fem.Ly
        F = jnp.zeros(fem.total_dof)
        fixed_dofs = []
        fixed_vals = []
        
        if self.axis == 'y':
            # =====================================================
            # BEND ABOUT Y: Curvature in XZ plane
            # =====================================================
            # Edges at x=0 (left) and x=Lx (right)
            left_nodes = jnp.where(fem.node_coords[:, 0] < tol)[0]
            right_nodes = jnp.where(fem.node_coords[:, 0] > Lx - tol)[0]
            
            if self.mode == 'angle':
                angle_rad = self.value * np.pi / 180.0
                
                # Left edge: θy = +angle, w = 0
                # (Positive rotation = left edge slopes down)
                for node in left_nodes:
                    fixed_dofs.extend([node*3+0, node*3+2])  # Fix w, θy
                    fixed_vals.extend([0.0, angle_rad]) 
                    
                # Right edge: θy = -angle, w = 0
                # (Negative rotation = right edge slopes up)
                for node in right_nodes:
                    fixed_dofs.extend([node*3+0, node*3+2])  # Fix w, θy
                    fixed_vals.extend([0.0, -angle_rad])
                
                # Prevent rigid body rotation about X
                fixed_dofs.append(left_nodes[0]*3 + 1)  # Fix θx at one node
                fixed_vals.append(0.0)
                
        elif self.axis == 'x':
            # =====================================================
            # BEND ABOUT X: Curvature in YZ plane
            # =====================================================
            # Edges at y=0 (bottom) and y=Ly (top)
            bot_nodes = jnp.where(fem.node_coords[:, 1] < tol)[0]
            top_nodes = jnp.where(fem.node_coords[:, 1] > Ly - tol)[0]
            
            if self.mode == 'angle':
                angle_rad = self.value * np.pi / 180.0
                
                # Bottom edge: θx = -angle, w = 0
                for node in bot_nodes:
                     fixed_dofs.extend([node*3+0, node*3+1])  # Fix w, θx
                     fixed_vals.extend([0.0, -angle_rad])
                     
                # Top edge: θx = +angle, w = 0
                for node in top_nodes:
                     fixed_dofs.extend([node*3+0, node*3+1])  # Fix w, θx
                     fixed_vals.extend([0.0, angle_rad])
                     
                # Prevent rigid body rotation about Y
                fixed_dofs.append(bot_nodes[0]*3 + 2)  # Fix θy at one node
                fixed_vals.append(0.0)

        return (jnp.array(fixed_dofs, dtype=jnp.int32), 
                jnp.array(fixed_vals, dtype=jnp.float64), 
                F)


# ==============================================================================
# CORNER LIFT LOAD CASE
# ==============================================================================

class CornerLiftCase(LoadCase):
    """
    Three-point support with one corner lifted.
    
    Simulates a common test scenario where three corners are fixed
    and one corner is lifted (displacement or force controlled).
    
    Corner Naming Convention:
    -------------------------
    - 'bl': Bottom-left  (x=0, y=0)
    - 'br': Bottom-right (x=Lx, y=0)
    - 'tl': Top-left     (x=0, y=Ly)
    - 'tr': Top-right    (x=Lx, y=Ly)
    
    Load Modes:
    -----------
    - 'disp': Prescribe corner displacement (kinematic)
    - 'force': Apply corner force (force control)
    
    Example:
    --------
    >>> # Lift bottom-right corner by 1.0 mm
    >>> case = CornerLiftCase('lift_br', corner='br', value=1.0, mode='disp')
    >>> 
    >>> # Apply 100 N upward force at top-left corner
    >>> case = CornerLiftCase('lift_tl_force', corner='tl', value=100.0, mode='force')
    """
    
    def __init__(self, name, corner='br', value=1.0, mode='disp', weight=1.0):
        """
        Initialize corner lift load case.
        
        Parameters:
        -----------
        name : str
            Load case name (e.g., 'lift_br_1mm')
        corner : str
            Corner to lift: 'bl', 'br', 'tl', or 'tr'
        value : float
            - If mode='disp': Vertical displacement in mm (positive = upward)
            - If mode='force': Applied force in N (positive = upward)
        mode : str
            'disp' (displacement control) or 'force' (force control)
        weight : float
            Optimization weight factor
        """
        super().__init__(name, weight)
        self.corner = corner  # 'tl', 'tr', 'bl', 'br'
        self.value = value
        self.mode = mode
        
    def get_bcs(self, fem):
        """
        Generate boundary conditions for corner lift.
        
        Fixes three corners at w=0 and either prescribes displacement
        or applies force at the fourth corner.
        
        Returns fixed DOFs, prescribed values, and force vector.
        See base class docstring for details.
        """
        tol = 1e-3
        coords = fem.node_coords
        
        # Find corner node indices by geometric proximity
        # (Uses L2 distance to identify nearest node to each corner)
        idx_bl = jnp.argmin(coords[:,0]**2 + coords[:,1]**2)
        idx_br = jnp.argmin((coords[:,0]-fem.Lx)**2 + coords[:,1]**2)
        idx_tl = jnp.argmin(coords[:,0]**2 + (coords[:,1]-fem.Ly)**2)
        idx_tr = jnp.argmin((coords[:,0]-fem.Lx)**2 + (coords[:,1]-fem.Ly)**2)
        
        corners = {
            'bl': idx_bl, 
            'br': idx_br, 
            'tl': idx_tl, 
            'tr': idx_tr
        }
        target_idx = corners[self.corner]  # The corner to lift
        
        fixed_dofs = []
        fixed_vals = []
        F = jnp.zeros(fem.total_dof)
        
        # Fix three corners at w=0
        for k, idx in corners.items():
            if k != self.corner:  # Not the lifted corner
                fixed_dofs.append(idx*3 + 0)  # Fix w only
                fixed_vals.append(0.0)
        
        # Apply loading/constraint at the lifted corner
        if self.mode == 'disp':
            # Displacement control: prescribe w value
            fixed_dofs.append(target_idx*3 + 0)  # Fix w
            fixed_vals.append(self.value)
        elif self.mode == 'force':
            # Force control: apply vertical force, w is free
            F = F.at[target_idx*3 + 0].set(self.value)
        
        return (jnp.array(fixed_dofs, dtype=jnp.int32), 
                jnp.array(fixed_vals, dtype=jnp.float64), 
                F)
