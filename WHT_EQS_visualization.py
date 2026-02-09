# ==============================================================================
# WHT_EQS_visualization.py
# ==============================================================================
# PURPOSE:
#   Interactive 3D visualization for equivalent sheet optimization.
#   Provides user-controlled PyVista windows at key workflow stages.
#
# STAGES:
#   1. PRE-GENERATION: Pattern verification before FEM analysis
#   2. POST-GENERATION: Ground truth results inspection
#   3. POST-OPTIMIZATION: Target vs optimized comparison
#
# FEATURES:
#   - Interactive terminal menus with repeat-ask logic
#   - PyVista 3D visualization with professional document theme
#   - Synchronized split-screen views for comparison
#   - Deformed shape visualization with magnification
#
# USAGE:
#   from WHT_EQS_visualization import (
#       stage1_visualize_patterns,
#       stage2_visualize_ground_truth,
#       stage3_visualize_comparison
#   )
#   
#   # Stage 1: Inspect patterns before analysis
#   stage1_visualize_patterns(nx, ny, x_grid, y_grid, t_field, z_field)
#   
#   # Stage 2: View analysis results
#   stage2_visualize_ground_truth(fem, targets, params)
#   
#   # Stage 3: Compare optimization results
#   stage3_visualize_comparison(fem_high, targets, opt_params, tgt_params)
#
# DEPENDENCIES:
#   - pyvista: 3D visualization library
#   - numpy: Array operations
#
# AUTHOR: Advanced FEM Team
# DATE: 2026-02-09
# ==============================================================================

import pyvista as pv
import numpy as np


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def setup_plotter(title="Visualization", shape=(1, 1)):
    """
    Create and configure a PyVista plotter with professional document theme.
    
    Parameters:
    -----------
    title : str
        Window title text
    shape : tuple of int
        Subplot grid shape (rows, cols). Default (1,1) = single view
        
    Returns:
    --------
    pyvista.Plotter
        Configured plotter instance with white background and document theme
        
    Example:
    --------
    >>> p = setup_plotter("My Visualization", shape=(1,2))  # Side-by-side plots
    """
    p = pv.Plotter(title=title, shape=shape)
    p.set_background("white")
    pv.set_plot_theme('document')  # Professional appearance with sans-serif fonts
    p.add_axes()
    return p


# ==============================================================================
# STAGE 1: PRE-GENERATION PATTERN VERIFICATION
# ==============================================================================

def stage1_visualize_patterns(nx, ny, x_grid, y_grid, t_field, z_field):
    """
    [STAGE 1] Interactive visualization of bead patterns BEFORE ground truth generation.
    
    Purpose:
    --------
    Allows user to visually verify thickness and topography patterns before running
    expensive FEM analysis. User can inspect bead geometry and check for errors.
    
    Parameters:
    -----------
    nx, ny : int
        Number of mesh elements in X and Y directions
    x_grid, y_grid : ndarray
        2D mesh grid coordinates (mm)
    t_field : ndarray
        Thickness field values (mm) - flattened or 2D
    z_field : ndarray
        Z-coordinate topography field values (mm) - flattened or 2D
        
    User Interface:
    ---------------
    Menu Loop:
        1: View thickness pattern (bead cross-section)
        2: View Z-shape (topography height)
        0 or Enter: Continue to next stage
        
    Technical Details:
    ------------------
    - Creates PyVista StructuredGrid for surface rendering
    - Uses 9pt font size for labels (professional appearance)
    - Repeats menu until user selects 0 or presses Enter
    - Colormap: viridis for thickness, plasma for topography
    
    Example:
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1000, 51)
    >>> y = np.linspace(0, 400, 21)
    >>> X, Y = np.meshgrid(x, y)
    >>> t = np.ones_like(X) * 1.5  # Uniform thickness
    >>> z = np.zeros_like(X)       # Flat plate
    >>> stage1_visualize_patterns(50, 20, X, Y, t, z)
    """
    # Create structured grid for PyVista visualization
    grid = pv.StructuredGrid()
    grid.points = np.column_stack([
        x_grid.flatten(), 
        y_grid.flatten(), 
        z_field.flatten()
    ])
    grid.dimensions = [ny + 1, nx + 1, 1]
    grid["Thickness"] = np.array(t_field).flatten()
    grid["Z-Shape"] = np.array(z_field).flatten()

    # Interactive menu loop
    while True:
        print("\n" + "="*70)
        print(" [STAGE 1] PRE-GENERATION PATTERN VERIFICATION")
        print("="*70)
        print(" Inspect bead patterns before running FEM analysis")
        print(" 1: Visualize Thickness Pattern (Bead Cross-Section)")
        print(" 2: Visualize Z-Shape Pattern (Topography Height)")
        print(" 0: Cancel / Continue to Ground Truth Generation (or press Enter)")
        print("="*70)
        choice = input(">> Your choice [0-2]: ").strip()
        
        # Exit on 0 or empty input
        if not choice or choice == '0':
            print("✓ Proceeding to Ground Truth generation...")
            break
            
        # Setup plotter for this view
        p = setup_plotter("Initial Pattern Verification")
        
        if choice == '1':
            # Visualize thickness with viridis colormap
            p.add_mesh(
                grid, 
                scalars="Thickness", 
                show_edges=True, 
                cmap="viridis", 
                scalar_bar_args={
                    'title': "Thickness (mm)", 
                    'label_font_size': 9, 
                    'title_font_size': 10
                }
            )
            p.add_text(
                "Thickness Pattern\n(Bead Cross-Section)", 
                position='upper_left', 
                font_size=10
            )
            
        elif choice == '2':
            # Visualize topography with plasma colormap
            p.add_mesh(
                grid, 
                scalars="Z-Shape", 
                show_edges=True, 
                cmap="plasma", 
                scalar_bar_args={
                    'title': "Z-Height (mm)", 
                    'label_font_size': 9, 
                    'title_font_size': 10
                }
            )
            p.add_text(
                "Topography Pattern\n(Z-Height Distribution)", 
                position='upper_left', 
                font_size=10
            )
            
        else:
            print("⚠ Invalid choice. Please enter 1, 2, or 0.")
            continue
        
        # Show visualization window (blocks until user closes it)
        p.show()


# ==============================================================================
# STAGE 2: POST-GENERATION RESULTS VIEWER
# ==============================================================================

def stage2_visualize_ground_truth(fem, targets, params, eigen_data=None):
    """
    [STAGE 2] Interactive visualization of ground truth analysis results.
    
    Purpose:
    --------
    After FEM analysis is complete, allows user to view deformed shapes
    and contour plots for each load case. Useful for understanding structural
    behavior and identifying problematic load cases.
    
    Parameters:
    -----------
    fem : PlateFEM
        High-resolution FEM solver instance (already solved)
    targets : list of dict
        Target results, each containing:
        - 'case_name': Name of load case
        - 'u_static': Static displacement solution (6*N_nodes)
        - Other fields like stress, strain, etc.
    params : dict
        Material parameter fields with keys:
        - 't': Thickness field
        - 'z': Topography field
        - 'rho': Density field
        - 'E': Young's modulus field
        
    User Interface:
    ---------------
    Menu Loop:
        1-N: View specific load case with deformed shape
        0 or Enter: Continue to optimization
        
    Technical Details:
    ------------------
    - Extracts W-displacement (DOF 2, 8, 14, ... from 6-DOF formulation)
    - Applies 30x magnification for visual clarity
    - Color contour shows vertical displacement magnitude
    - Mesh edges are shown for better geometry perception
    
    Example:
    --------
    >>> # After running FEM analysis
    >>> stage2_visualize_ground_truth(fem_high, model.targets, params_high)
    """
    # Extract mesh information
    nx, ny = fem.nx, fem.ny
    x = np.array(fem.node_coords[:, 0])
    y = np.array(fem.node_coords[:, 1])
    z = np.array(params['z']).flatten()
    
    # Default deformation scale
    scale = 5.0
    
    # Interactive menu loop
    while True:
        print("\n" + "="*70)
        print(" [STAGE 2] GROUND TRUTH ANALYSIS RESULTS VIEWER")
        print("="*70)
        print(" View deformed shapes and contours for each load case")
        
        # List all available load cases
        for idx, tgt in enumerate(targets):
            print(f" {idx+1}: {tgt['case_name']}")
        
        print(f" S: Set Deformation Scale (Current: {scale:.1f}x)")
        if eigen_data:
            print(" M: View Eigenmodes (Vibration Analysis)")
        print(" 0: Cancel / Continue to Optimization (or press Enter)")
        print("="*70)
        choice = input(">> Select choice: ").strip()
        
        # Handle Eigenmode Visualization
        if choice.lower() == 'm' and eigen_data:
            vals = eigen_data['vals']
            modes = eigen_data['modes']
            
            print("\n Available Modes:")
            for i, val in enumerate(vals):
                freq_hz = np.sqrt(max(0, val)) / (2 * np.pi)
                print(f" Mode {i+1}: {freq_hz:.2f} Hz")
                
            try:
                m_idx = int(input(">> Select Mode Number: ")) - 1
                if 0 <= m_idx < len(vals):
                    freq_hz = np.sqrt(max(0, vals[m_idx])) / (2 * np.pi)
                    
                    # Prepare Mesh for Mode Shape
                    w_mode = np.array(modes[:, m_idx])
                    grid = pv.StructuredGrid()
                    z_deformed = z + w_mode * scale # Use current scale
                    grid.points = np.column_stack([x, y, z_deformed])
                    grid.dimensions = [ny + 1, nx + 1, 1]
                    grid["Mode Shape"] = w_mode
                    
                    # Plot
                    title = f"Mode {m_idx+1}: {freq_hz:.2f} Hz (Scale: {scale}x)"
                    p = setup_plotter(title)
                    p.add_mesh(grid, scalars="Mode Shape", show_edges=True, cmap="coolwarm")
                    p.add_text(f"Frequency: {freq_hz:.2f} Hz", position='upper_right', font_size=10, color='black')
                    p.show()
                else:
                    print("⚠ Invalid mode number.")
            except ValueError:
                print("⚠ Invalid input.")
            continue
        
        # Handle Scale Change
        if choice.lower() == 's':
            try:
                new_scale = float(input(f"Enter new scale factor (current: {scale}): "))
                scale = new_scale
                print(f"✓ Scale updated to {scale}x")
            except ValueError:
                print("⚠ Invalid input. Scale unchanged.")
            continue

        # Exit on 0 or empty input
        if not choice or choice == '0':
            print("✓ Proceeding to optimization...")
            break
            
        # Parse and validate user input
        try:
            sel_idx = int(choice) - 1
            if sel_idx < 0 or sel_idx >= len(targets):
                print("⚠ Invalid case number. Please try again.")
                continue
            tgt = targets[sel_idx]
        except ValueError:
            print("⚠ Invalid input. Please enter a number or 'S'.")
            continue
        
        # Sub-menu for Result Type
        print("-" * 50)
        print(" Select Result Type to Visualize:")
        print("  A: Displacement (W-Component)")
        print("  B: Equivalent Strain (Max Surface)")
        print("  C: Equivalent Stress (Max Surface)")
        print("-" * 50)
        type_choice = input(">> Select Type [A/B/C] (Default: A): ").strip().upper()
        
        # Default to A
        if not type_choice:
            type_choice = 'A'
            
        if type_choice not in ['A', 'B', 'C']:
            print("⚠ Invalid type. Defaulting to Displacement (A).")
            type_choice = 'A'

        # Select Data Field
        if type_choice == 'A':
            field_name = "W-Displacement (mm)"
            scalars = np.array(tgt['u_static'])
            cmap = 'jet'
            title_prefix = "Displacement"
        elif type_choice == 'B':
            field_name = "Max Surface Strain (-)"
            scalars = np.array(tgt['max_surface_strain'])
            cmap = 'plasma'
            title_prefix = "Strain"
        elif type_choice == 'C':
            field_name = "Max Surface Stress (MPa)"
            scalars = np.array(tgt['max_surface_stress'])
            cmap = 'inferno'
            title_prefix = "Stress"

        # Compute Summary Statistics (Displayed on Screen)
        # 1. Total Reaction Force (Sum of R components)
        if 'reaction_full' in tgt:
            R = tgt['reaction_full'] # (6*N)
            Rx_sum = np.sum(R[0::6])
            Ry_sum = np.sum(R[1::6])
            Rz_sum = np.sum(R[2::6])
            
            # 2. Average Displacement (Mean of U components)
            U = tgt['u_full'] # (6*N)
            Ux_avg = np.mean(U[0::6])
            Uy_avg = np.mean(U[1::6])
            Uz_avg = np.mean(U[2::6])
            
            stats_text = (
                "SUMMARY STATISTICS:\n"
                f"Total Reaction Force: Rx={Rx_sum:.2e}, Ry={Ry_sum:.2e}, Rz={Rz_sum:.2e} (N)\n"
                f"Avg Displacement:     Ux={Ux_avg:.2e}, Uy={Uy_avg:.2e}, Uz={Uz_avg:.2e} (mm)"
            )
        else:
            stats_text = "Statistics not available (Run with updated main_verification.py)"

        # Prepare Mesh
        w_disp = np.array(tgt['u_static'])
        grid = pv.StructuredGrid()
        z_deformed = z + w_disp * scale
        grid.points = np.column_stack([x, y, z_deformed])
        grid.dimensions = [ny + 1, nx + 1, 1]
        grid[field_name] = scalars
        
        # Setup Plotter
        title = f"{title_prefix}: {tgt['case_name']} (Scale: {scale}x)"
        p = setup_plotter(title)
        
        p.add_mesh(
            grid, 
            scalars=field_name, 
            show_edges=True, 
            cmap=cmap, 
            scalar_bar_args={
                'title': field_name, 
                'label_font_size': 9,
                'title_font_size': 10
            }
        )
        
        # Add Statistics Text (8pt)
        p.add_text(
            stats_text,
            position='upper_right',
            font_size=8,
            color='black',
            font='courier',
            shadow=False
        )

        
        # VISUALIZE BOUNDARY CONDITIONS (Bold Blue Points)
        if 'fixed_dofs' in tgt:
            fixed_dofs = tgt['fixed_dofs']
            # Map DOFs to nodes (integer division by 6)
            fixed_nodes = np.unique(fixed_dofs // 6)
            
            # Extract coordinates for fixed nodes
            bc_coords = np.column_stack([
                x[fixed_nodes], 
                y[fixed_nodes], 
                z_deformed[fixed_nodes]
            ])
            
            # Add BC points
            p.add_points(
                bc_coords, 
                color="blue", 
                point_size=15, 
                render_points_as_spheres=True,
                name="Boundary_Conditions"
            )
            p.add_text("Blue Points: Fixed BCs", position='upper_left', font_size=8, color='blue')

        # VISUALIZE LOADS (Red Points)
        if 'force_vector' in tgt:
            F = tgt['force_vector']
            # Find nodes with non-zero force magnitude
            F_reshaped = F.reshape(-1, 6)
            # Check magnitude of force (translational + rotational)
            force_mag = np.linalg.norm(F_reshaped[:, :3], axis=1) # check translational forces
            moment_mag = np.linalg.norm(F_reshaped[:, 3:], axis=1) # check moments
            loaded_nodes = np.where((force_mag > 1e-6) | (moment_mag > 1e-6))[0]
            
            if len(loaded_nodes) > 0:
                load_coords = np.column_stack([
                    x[loaded_nodes], 
                    y[loaded_nodes], 
                    z_deformed[loaded_nodes]
                ])
                
                # Add Load points
                p.add_points(
                    load_coords, 
                    color="red", 
                    point_size=15, 
                    render_points_as_spheres=True,
                    name="Loads"
                )
                p.add_text("Red Points: Applied Loads", position='upper_right', font_size=8, color='red')
        
        # Add informative text overlay
        p.add_text(
            f"Load Case: {tgt['case_name']}\nWarp Scale: {scale}x\n(Close window to return to menu)", 
            position='lower_left', 
            font_size=9, 
            color='black'
        )
        
        # Show visualization window
        p.show()


# ==============================================================================
# STAGE 3: POST-OPTIMIZATION COMPARISON
# ==============================================================================

def stage3_visualize_comparison(fem_high, targets, optimized_params, target_params, opt_eigen=None, tgt_eigen=None):
    """
    [STAGE 3] Interactive side-by-side comparison of target vs optimized results.
    
    Purpose:
    --------
    After optimization is complete, allows user to visually compare target and
    optimized parameters. Split-screen layout with synchronized camera views
    enables easy assessment of optimization quality.
    
    Parameters:
    -----------
    fem_high : PlateFEM
        High-resolution FEM instance (for mesh coordinates)
    targets : list of dict
        Target results (not used in current version but available for extension)
    optimized_params : dict
        Optimized parameter fields:
        - 't': Thickness field (interpolated to high-res mesh)
        - 'z': Topography field
        - 'rho': Density field
        - 'E': Young's modulus field
    target_params : dict
        Ground truth parameter fields (same structure as optimized_params)
        
    User Interface:
    ---------------
    Menu Loop:
        1: Compare thickness fields
        2: Compare Z-shape (topography) fields  
        0 or Enter: Exit program
        
    Technical Details:
    ------------------
    - Uses (1, 2) subplot layout for side-by-side comparison
    - Views are linked: rotating one viewport rotates the other
    - Same colormap scale for both sides (facilitates comparison)
    - Mesh edges shown for spatial reference
    
    Extensions:
    -----------
    Future versions could add:
    - Difference plot (center panel showing target - optimized)
    - Histogram comparison of parameter distributions
    - Line profiles along plate centerlines
    
    Example:
    --------
    >>> # After optimization completes
    >>> stage3_visualize_comparison(
    ...     fem_high=model.fem_high,
    ...     targets=model.targets,
    ...     optimized_params=model.optimized_params,
    ...     target_params=model.target_params_high
    ... )
    """
    # Extract mesh coordinates
    nx, ny = fem_high.nx, fem_high.ny
    x = np.array(fem_high.node_coords[:, 0])
    y = np.array(fem_high.node_coords[:, 1])
    
    # Interactive menu loop
    while True:
        print("\n" + "="*70)
        print(" [STAGE 3] POST-OPTIMIZATION COMPARISON VIEWER")
        print("="*70)
        print(" Compare target and optimized parameters side-by-side")
        print(" 1: Compare Thickness Fields (Target vs Optimized)")
        print(" 2: Compare Z-Shape Fields (Target vs Optimized)")
        if opt_eigen and tgt_eigen:
            print(" 3: Compare Eigenmodes (Frequencies & Shapes)")
        print(" 0: Cancel / Exit Program (or press Enter)")
        print("="*70)
        choice = input(">> Your choice [0-3]: ").strip()
        
        # Exit on 0 or empty input
        if not choice or choice == '0':
            print("✓ Exiting visualization system...")
            break
            
        # Setup split-screen plotter (1 row, 2 columns)
        p = setup_plotter(
            "Final Comparison: TARGET (Left) vs OPTIMIZED (Right)", 
            shape=(1, 2)
        )
        
        # Select data field based on user choice
        if choice == '1':
            data_target = np.array(target_params['t']).flatten()
            data_optimized = np.array(optimized_params['t']).flatten()
            field_name = "Thickness"
            units = "mm"
        elif choice == '2':
            data_target = np.array(target_params['z']).flatten()
            data_optimized = np.array(optimized_params['z']).flatten()
            field_name = "Z-Height"
            units = "mm"
            cmap = "viridis"
        elif choice == '3' and opt_eigen and tgt_eigen:
            # Mode Shape Selection
            print("\n Select Mode for Comparison:")
            for i in range(len(tgt_eigen['vals'])):
                f_t = np.sqrt(max(0, tgt_eigen['vals'][i])) / (2*np.pi)
                f_o = np.sqrt(max(0, opt_eigen['vals'][i])) / (2*np.pi)
                print(f" Mode {i+1}: Target={f_t:.1f}Hz, Opt={f_o:.1f}Hz")
            try:
                m_idx = int(input(">> Mode number: ")) - 1
                data_target = tgt_eigen['modes'][:, m_idx]
                data_optimized = opt_eigen['modes'][:, m_idx]
                f_target = np.sqrt(max(0, tgt_eigen['vals'][m_idx])) / (2*np.pi)
                f_opt = np.sqrt(max(0, opt_eigen['vals'][m_idx])) / (2*np.pi)
                field_name = f"Mode_{m_idx+1}_Shape"
                units = "Rel"
                cmap = "coolwarm"
            except:
                print("⚠ Invalid mode selection.")
                continue
        else:
            print("⚠ Invalid choice.")
            continue

        # Helper function to add mesh to specific subplot
        def add_comparison_mesh(plotter, col_idx, data, title):
            """
            Add mesh to subplot at specified column index.
            
            Parameters:
            -----------
            plotter : pyvista.Plotter
                Main plotter instance
            col_idx : int
                Column index (0=left, 1=right)
            data : ndarray
                Scalar field data to visualize
            title : str
                Subplot title ("TARGET" or "OPTIMIZED")
            """
            plotter.subplot(0, col_idx)  # Select subplot (row=0, col=col_idx)
            
            # Create structured grid
            grid = pv.StructuredGrid()
            grid.points = np.column_stack([x, y, np.zeros_like(x)])  # Flat geometry
            grid.dimensions = [ny + 1, nx + 1, 1]
            grid.point_data[field_name] = data
            
            # Add mesh with contour coloring
            plotter.add_mesh(
                grid, 
                scalars=field_name, 
                show_edges=True, 
                cmap=cmap, # Use dynamic cmap (viridis or coolwarm)
                scalar_bar_args={
                    'label_font_size': 9, 
                    'title_font_size': 10,
                    'title': f"{field_name} ({units})"
                }
            )
            
            # Add subplot title
            plotter.add_text(
                title, 
                position='upper_edge', 
                font_size=10, 
                color='black'
            )

        # Add both meshes (left=target, right=optimized)
        title_l = "TARGET"
        title_r = "OPTIMIZED"
        if choice == '3':
            title_l += f" ({f_target:.2f} Hz)"
            title_r += f" ({f_opt:.2f} Hz)"
            
        add_comparison_mesh(p, 0, data_target, title_l)
        add_comparison_mesh(p, 1, data_optimized, title_r)
        
        # Link camera views for synchronized rotation/zoom
        p.link_views()
        
        # Show split-screen visualization
        p.show()
