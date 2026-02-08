# ==============================================================================
# WHT_EQS_pattern_generator.py
# ==============================================================================
# PURPOSE:
#   Bead pattern generation for equivalent sheet models.
#   Supports arbitrary alphanumeric strings with stroke-based rendering.
#
# FEATURES:
#   - Full character support: A-Z, 0-9, -, _, +
#   - Stroke-based font rendering (vectorized line segments)
#   - Thickness pattern generation (for bead cross-sections)
#   - Topography pattern generation (for Z-coordinate variations)
#   - Material property field generation (density, Young's modulus)
#
# USAGE:
#   from WHT_EQS_pattern_generator import get_thickness_field, get_z_field
#   
#   # Generate thickness pattern for "ABC" string
#   t_field = get_thickness_field(X, Y, Lx=1000, Ly=400, 
#                                  pattern_str="ABC", base_t=1.0, bead_t=2.0)
#
# AUTHOR: Advanced FEM Team
# DATE: 2026-02-09
# ==============================================================================

import jax.numpy as jnp
import numpy as np

# ==============================================================================
# FONT STROKE DEFINITIONS
# ==============================================================================
# Each character is defined by line segments in normalized coordinates [0,1]
# Format: (x1, y1, x2, y2) where (x1,y1) is start, (x2,y2) is end
# Coordinates are relative to character bounding box

FONT_STROKES = {
    # Uppercase Letters A-Z
    'A': [(0.2, 0.1, 0.5, 0.9), (0.8, 0.1, 0.5, 0.9), (0.35, 0.5, 0.65, 0.5)],
    'B': [(0.2, 0.1, 0.2, 0.9), (0.2, 0.9, 0.7, 0.9), (0.2, 0.5, 0.6, 0.5), 
          (0.2, 0.1, 0.7, 0.1), (0.7, 0.9, 0.8, 0.7), (0.8, 0.7, 0.6, 0.5),
          (0.6, 0.5, 0.8, 0.3), (0.8, 0.3, 0.7, 0.1)],
    'C': [(0.25, 0.15, 0.25, 0.85), (0.25, 0.85, 0.75, 0.85), (0.25, 0.15, 0.75, 0.15)],
    'D': [(0.2, 0.1, 0.2, 0.9), (0.2, 0.9, 0.6, 0.9), (0.2, 0.1, 0.6, 0.1), 
          (0.6, 0.9, 0.8, 0.5), (0.8, 0.5, 0.6, 0.1)],
    'E': [(0.2, 0.1, 0.2, 0.9), (0.2, 0.1, 0.8, 0.1), (0.2, 0.5, 0.7, 0.5), (0.2, 0.9, 0.8, 0.9)],
    'F': [(0.2, 0.1, 0.2, 0.9), (0.2, 0.5, 0.7, 0.5), (0.2, 0.9, 0.8, 0.9)],
    'G': [(0.8, 0.7, 0.8, 0.8), (0.8, 0.8, 0.2, 0.8), (0.2, 0.8, 0.2, 0.2), 
          (0.2, 0.2, 0.8, 0.2), (0.8, 0.2, 0.8, 0.5), (0.8, 0.5, 0.5, 0.5)],
    'H': [(0.2, 0.1, 0.2, 0.9), (0.8, 0.1, 0.8, 0.9), (0.2, 0.5, 0.8, 0.5)],
    'I': [(0.5, 0.1, 0.5, 0.9), (0.2, 0.1, 0.8, 0.1), (0.2, 0.9, 0.8, 0.9)],
    'J': [(0.2, 0.3, 0.5, 0.1), (0.5, 0.1, 0.8, 0.1), (0.8, 0.1, 0.8, 0.9), (0.2, 0.9, 0.8, 0.9)],
    'K': [(0.2, 0.1, 0.2, 0.9), (0.2, 0.5, 0.8, 0.9), (0.2, 0.5, 0.8, 0.1)],
    'L': [(0.2, 0.9, 0.2, 0.1), (0.2, 0.1, 0.8, 0.1)],
    'M': [(0.2, 0.1, 0.2, 0.9), (0.2, 0.9, 0.5, 0.5), (0.5, 0.5, 0.8, 0.9), (0.8, 0.9, 0.8, 0.1)],
    'N': [(0.2, 0.1, 0.2, 0.9), (0.2, 0.9, 0.8, 0.1), (0.8, 0.1, 0.8, 0.9)],
    'O': [(0.2, 0.2, 0.2, 0.8), (0.2, 0.8, 0.8, 0.8), (0.8, 0.8, 0.8, 0.2), (0.8, 0.2, 0.2, 0.2)],
    'P': [(0.2, 0.1, 0.2, 0.9), (0.2, 0.9, 0.8, 0.9), (0.8, 0.9, 0.8, 0.5), (0.8, 0.5, 0.2, 0.5)],
    'Q': [(0.2, 0.2, 0.2, 0.8), (0.2, 0.8, 0.8, 0.8), (0.8, 0.8, 0.8, 0.2), 
          (0.8, 0.2, 0.2, 0.2), (0.6, 0.4, 0.9, 0.1)],
    'R': [(0.2, 0.1, 0.2, 0.9), (0.2, 0.9, 0.8, 0.9), (0.8, 0.9, 0.8, 0.5), 
          (0.8, 0.5, 0.2, 0.5), (0.5, 0.5, 0.8, 0.1)],
    'S': [(0.2, 0.2, 0.8, 0.2), (0.8, 0.2, 0.8, 0.5), (0.8, 0.5, 0.2, 0.5), 
          (0.2, 0.5, 0.2, 0.8), (0.2, 0.8, 0.8, 0.8)],
    'T': [(0.5, 0.1, 0.5, 0.9), (0.2, 0.9, 0.8, 0.9)],
    'U': [(0.2, 0.9, 0.2, 0.2), (0.2, 0.2, 0.8, 0.2), (0.8, 0.2, 0.8, 0.9)],
    'V': [(0.2, 0.9, 0.5, 0.1), (0.5, 0.1, 0.8, 0.9)],
    'W': [(0.2, 0.9, 0.3, 0.1), (0.3, 0.1, 0.5, 0.5), (0.5, 0.5, 0.7, 0.1), (0.7, 0.1, 0.8, 0.9)],
    'X': [(0.2, 0.1, 0.8, 0.9), (0.2, 0.9, 0.8, 0.1)],
    'Y': [(0.2, 0.9, 0.5, 0.5), (0.8, 0.9, 0.5, 0.5), (0.5, 0.5, 0.5, 0.1)],
    'Z': [(0.2, 0.9, 0.8, 0.9), (0.8, 0.9, 0.2, 0.1), (0.2, 0.1, 0.8, 0.1)],
    
    # Digits 0-9
    '0': [(0.2, 0.1, 0.2, 0.9), (0.2, 0.9, 0.8, 0.9), (0.8, 0.9, 0.8, 0.1), (0.8, 0.1, 0.2, 0.1)],
    '1': [(0.3, 0.7, 0.5, 0.9), (0.5, 0.9, 0.5, 0.1), (0.2, 0.1, 0.8, 0.1)],
    '2': [(0.2, 0.7, 0.5, 0.9), (0.5, 0.9, 0.8, 0.7), (0.8, 0.7, 0.2, 0.1), (0.2, 0.1, 0.8, 0.1)],
    '3': [(0.2, 0.8, 0.8, 0.8), (0.8, 0.8, 0.5, 0.5), (0.5, 0.5, 0.8, 0.2), (0.8, 0.2, 0.2, 0.2)],
    '4': [(0.2, 0.9, 0.2, 0.5), (0.2, 0.5, 0.8, 0.5), (0.6, 0.9, 0.6, 0.1)],
    '5': [(0.8, 0.9, 0.2, 0.9), (0.2, 0.9, 0.2, 0.5), (0.2, 0.5, 0.8, 0.5), 
          (0.8, 0.5, 0.8, 0.1), (0.8, 0.1, 0.2, 0.1)],
    '6': [(0.8, 0.8, 0.2, 0.5), (0.2, 0.5, 0.2, 0.2), (0.2, 0.2, 0.8, 0.2), 
          (0.8, 0.2, 0.8, 0.5), (0.8, 0.5, 0.2, 0.5)],
    '7': [(0.2, 0.9, 0.8, 0.9), (0.8, 0.9, 0.4, 0.1)],
    '8': [(0.2, 0.1, 0.2, 0.9), (0.2, 0.9, 0.8, 0.9), (0.8, 0.9, 0.8, 0.1), 
          (0.8, 0.1, 0.2, 0.1), (0.2, 0.5, 0.8, 0.5)],
    '9': [(0.8, 0.2, 0.2, 0.5), (0.2, 0.5, 0.8, 0.5), (0.8, 0.5, 0.8, 0.8), 
          (0.8, 0.8, 0.2, 0.8), (0.2, 0.8, 0.2, 0.5)],
    
    # Special Characters
    '-': [(0.3, 0.5, 0.7, 0.5)],                      # Hyphen
    '_': [(0.2, 0.1, 0.8, 0.1)],                      # Underscore
    '+': [(0.5, 0.3, 0.5, 0.7), (0.3, 0.5, 0.7, 0.5)],  # Plus sign
}


# ==============================================================================
# PATTERN GENERATION FUNCTIONS
# ==============================================================================

def get_pattern_field(X, Y, Lx, Ly, pattern_str, val_dict, base_val=1.0):
    """
    Generate a 2D field with character patterns using stroke-based rendering.
    
    This is the core pattern generation function. It divides the plate width 
    into equal segments (one per character) and renders each character as a 
    series of line strokes with specified field values.
    
    Parameters:
    -----------
    X, Y : jax.numpy.ndarray
        2D mesh grid coordinates (mm). Shape: (Nx+1, Ny+1)
    Lx, Ly : float
        Plate dimensions in X and Y directions (mm)
    pattern_str : str
        String of characters to render (e.g., "ABC", "D7-X", "HELLO")
        Each character occupies equal width: Lx / len(pattern_str)
    val_dict : dict or float
        - If dict: maps characters to field values {'A': 2.0, 'B': 2.5}
        - If float: single value applied to all characters
    base_val : float, optional
        Background field value where no pattern exists (default: 1.0)
        
    Returns:
    --------
    jax.numpy.ndarray
        2D field with pattern values. Shape matches input X, Y
        
    Technical Details:
    ------------------
    - Stroke width: Fixed at 60mm physical width regardless of character size
    - Distance calculation: Uses finite line segment projection (clamped t ∈ [0,1])
    - Coordinate system: Normalized local coords [0,1] converted to physical space
    - Missing characters: Ignored (space ' ' always skipped)
    
    Example:
    --------
    >>> import jax.numpy as jnp
    >>> x = jnp.linspace(0, 1000, 51)
    >>> y = jnp.linspace(0, 400, 21)
    >>> X, Y = jnp.meshgrid(x, y)
    >>> 
    >>> # Thickness pattern with per-character values
    >>> thickness = get_pattern_field(X, Y, 1000, 400, "ABC", 
    ...                                {'A': 2.0, 'B': 2.5, 'C': 1.5}, 
    ...                                base_val=1.0)
    >>> 
    >>> # Uniform pattern
    >>> pattern = get_pattern_field(X, Y, 1000, 400, "D7-X", 3.0, base_val=1.0)
    """
    if val_dict is None: 
        val_dict = {}
    
    num_chars = len(pattern_str)
    char_width = Lx / (num_chars if num_chars > 0 else 1)
    field_map = jnp.full_like(X, base_val)
    w_physical = 60.0  # Physical stroke width in mm (controls bead/line thickness)
    
    def dist_segment(px, py, x1, y1, x2, y2):
        """
        Calculate squared distance from point (px, py) to finite line segment.
        
        All coordinates are in normalized space [0,1] relative to character box,
        then converted to physical space to maintain consistent stroke width.
        
        Parameters:
        -----------
        px, py : float
            Point coordinates in normalized space [0,1]
        x1, y1, x2, y2 : float
            Line segment endpoints in normalized space [0,1]
            
        Returns:
        --------
        float
            Squared distance in physical units (mm²)
            
        Algorithm:
        ----------
        1. Convert normalized coords to physical: X = u * char_width, Y = v * Ly
        2. Project point onto infinite line: t = dot(P-A, B-A) / |B-A|²
        3. Clamp projection to segment: t_clamped = clamp(t, 0, 1)
        4. Return squared distance: |P - (A + t_clamped * (B-A))|²
        """
        # Convert normalized coordinates to physical coordinates
        X1, Y1 = x1 * char_width, y1 * Ly
        X2, Y2 = x2 * char_width, y2 * Ly
        PX, PY = px * char_width, py * Ly
        
        # Calculate closest point on segment using vector projection
        dx, dy = X2 - X1, Y2 - Y1
        len_sq = dx**2 + dy**2 + 1e-6  # Add small epsilon to avoid division by zero
        t = jnp.clip(((PX - X1) * dx + (PY - Y1) * dy) / len_sq, 0.0, 1.0)
        
        # Return squared distance to closest point
        closest_x = X1 + t * dx
        closest_y = Y1 + t * dy
        return (PX - closest_x)**2 + (PY - closest_y)**2
    
    # Process each character in the pattern string
    for i, char in enumerate(pattern_str):
        if char == ' ':  # Skip spaces (no pattern)
            continue
        
        # Determine the field value for this character
        if isinstance(val_dict, dict):
            current_val = val_dict.get(char, base_val)
        else:
            current_val = val_dict  # Uniform value for all characters
        
        # Define spatial region for this character
        x_start = i * char_width
        x_end = (i + 1) * char_width
        region_mask = (X >= x_start) & (X < x_end)
        
        # Convert to normalized local coordinates [0,1] within character box
        u = (X - x_start) / char_width
        v = Y / Ly
        
        # Get stroke definitions for this character (uppercase lookup)
        strokes = FONT_STROKES.get(char.upper(), [])
        
        if strokes:
            # Calculate minimum distance to any stroke
            dist_sq = jnp.full_like(X, 1e9)  # Initialize with large value
            for stroke in strokes:
                d2 = dist_segment(u, v, stroke[0], stroke[1], stroke[2], stroke[3])
                dist_sq = jnp.minimum(dist_sq, d2)
            
            # Apply thickness threshold: points within w_physical/2 are "on stroke"
            is_stroke = dist_sq < (w_physical / 2)**2
            
            # Update field map: set current_val where strokes exist
            field_map = jnp.where(region_mask & is_stroke, current_val, field_map)
    
    return field_map


def get_thickness_field(X, Y, Lx=1000.0, Ly=400.0, pattern_str="A", base_t=1.0, bead_t=2.0):
    """
    Generate thickness field with bead pattern.
    
    Convenience wrapper around get_pattern_field() specifically for thickness.
    
    Parameters:
    -----------
    X, Y : ndarray
        2D mesh grid coordinates (mm)
    Lx, Ly : float
        Plate dimensions (mm)
    pattern_str : str
        Pattern string (e.g., "ABC", "D7-X")
    base_t : float
        Base thickness (mm) - sheet metal without beads
    bead_t : dict or float
        Bead thickness (mm) - per character or uniform
        
    Returns:
    --------
    ndarray
        Thickness field (mm)
        
    Example:
    --------
    >>> t = get_thickness_field(X, Y, Lx=1000, Ly=400, pattern_str="D7-X",
    ...                          base_t=1.0, bead_t={'D': 2.0, '7': 2.5, 'X': 3.0})
    """
    return get_pattern_field(X, Y, Lx, Ly, pattern_str, bead_t, base_val=base_t)


def get_z_field(X, Y, Lx=1000.0, Ly=400.0, pattern_pz="", pz_dict=None):
    """
    Generate topography (Z-coordinate) field with bead pattern.
    
    Convenience wrapper around get_pattern_field() for Z-height (topography).
    Useful for simulating embossed or debossed surface features.
    
    Parameters:
    -----------
    X, Y : ndarray
        2D mesh grid coordinates (mm)
    Lx, Ly : float
        Plate dimensions (mm)
    pattern_pz : str
        Topography pattern string (e.g., "TNY", "123")
    pz_dict : dict or float
        Z-height (mm) per character or uniform
        Positive values = raised features
        Negative values = recessed features
        
    Returns:
    --------
    ndarray
        Z-coordinate field (mm), base level = 0
        
    Example:
    --------
    >>> z = get_z_field(X, Y, Lx=1000, Ly=400, pattern_pz="TNY",
    ...                 pz_dict={'T': 20.0, 'N': 10.0, 'Y': -20.0})
    """
    if pz_dict is None:
        pz_dict = {}
    return get_pattern_field(X, Y, Lx, Ly, pattern_pz, pz_dict, base_val=0.0)


def get_density_field(X, Y, Lx, Ly, base_rho=7.5e-9, seed=42):
    """
    Generate spatially varying density field with 3x2 grid regions.
    
    Creates a random density distribution to simulate material variations
    or manufacturing tolerances. Useful for testing robustness of 
    equivalent sheet optimization.
    
    Parameters:
    -----------
    X, Y : numpy.ndarray
        Mesh grid coordinates (mm)
    Lx, Ly : float
        Plate dimensions (mm) - used to define region boundaries
    base_rho : float
        Base density value (tonne/mm³)
        Default: 7.5e-9 (steel density ~7850 kg/m³)
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    jax.numpy.ndarray
        Density field with ±30% random variations across 6 regions
        
    Grid Layout:
    ------------
    Plate is divided into 3x2 = 6 rectangular regions:
    
        ┌─────┬─────┬─────┐
        │  R1 │  R2 │  R3 │  ← Top half (y > Ly/2)
        ├─────┼─────┼─────┤
        │  R4 │  R5 │  R6 │  ← Bottom half (y < Ly/2)
        └─────┴─────┴─────┘
          x<L/3  x<2L/3  x>2L/3
          
    Each region gets random density: rho = base_rho * (1 + variation)
    where variation ∈ [-0.3, +0.3] (i.e., ±30%)
    
    Example:
    --------
    >>> rho = get_density_field(X, Y, Lx=1000, Ly=400, 
    ...                          base_rho=7.85e-9, seed=42)
    >>> print(f"Density range: {rho.min():.2e} to {rho.max():.2e}")
    """
    np.random.seed(seed)
    
    # Define 3x2 grid boundaries
    x_boundaries = [0, Lx/3, 2*Lx/3, Lx]
    y_boundaries = [0, Ly/2, Ly]
    
    # Initialize with base density
    rho_field = jnp.ones_like(X) * base_rho
    
    # Apply random variations to each of 6 regions
    for i in range(3):  # X divisions (3 columns)
        for j in range(2):  # Y divisions (2 rows)
            # Random variation: ±30%
            variation = np.random.uniform(-0.3, 0.3)
            region_rho = base_rho * (1 + variation)
            
            # Define region mask
            mask = ((X >= x_boundaries[i]) & (X < x_boundaries[i+1]) &
                    (Y >= y_boundaries[j]) & (Y < y_boundaries[j+1]))
            
            rho_field = jnp.where(mask, region_rho, rho_field)
    
    return rho_field


def get_E_field(X, Y, Lx, Ly, base_E=200000.0, seed=42):
    """
    Generate spatially varying Young's modulus field with 3x2 grid regions.
    
    Creates random stiffness distribution to simulate material variations.
    Values are always ≤ base_E (simulating degradation, not strengthening).
    
    Parameters:
    -----------
    X, Y : numpy.ndarray
        Mesh grid coordinates (mm)
    Lx, Ly : float
        Plate dimensions (mm)
    base_E : float
        Base Young's modulus (MPa)
        Default: 200000.0 (steel ~200 GPa)
    seed : int
        Random seed for reproducibility
        Different seed from density (seed + 100) for independence
        
    Returns:
    --------
    jax.numpy.ndarray
        Young's modulus field with -50% to 0% variations across 6 regions
        
    Variation Range:
    ----------------
    E_region = base_E * (1 + variation)
    where variation ∈ [-0.5, 0.0]
    
    This means:
    - Minimum possible: base_E * 0.5 (50% reduction)
    - Maximum possible: base_E * 1.0 (no change)
    
    Rationale:
    ----------
    Material degradation (corrosion, fatigue, defects) is more common than
    strengthening in practice, hence the asymmetric variation range.
    
    Example:
    --------
    >>> E = get_E_field(X, Y, Lx=1000, Ly=400, 
    ...                 base_E=210000.0, seed=42)
    >>> print(f"Stiffness range: {E.min():.0f} to {E.max():.0f} MPa")
    """
    np.random.seed(seed + 100)  # Different seed from density for independence
    
    # Define 3x2 grid boundaries
    x_boundaries = [0, Lx/3, 2*Lx/3, Lx]
    y_boundaries = [0, Ly/2, Ly]
    
    # Initialize with base value
    E_field = jnp.ones_like(X) * base_E
    
    # Apply random variations to each of 6 regions
    for i in range(3):  # X divisions (3 columns)
        for j in range(2):  # Y divisions (2 rows)
            # Random variation: -50% to 0% (always ≤ base_E)
            variation = np.random.uniform(-0.5, 0.0)
            region_E = base_E * (1 + variation)
            
            # Define region mask
            mask = ((X >= x_boundaries[i]) & (X < x_boundaries[i+1]) &
                    (Y >= y_boundaries[j]) & (Y < y_boundaries[j+1]))
            
            E_field = jnp.where(mask, region_E, E_field)
    
    return E_field
