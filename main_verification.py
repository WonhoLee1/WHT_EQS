# ==============================================================================
# main_verification.py - Equivalent Sheet Model Optimization & Verification
# ==============================================================================
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import optax
import pyvista as pv
from scipy.interpolate import griddata

# Custom Modules
from solver import PlateFEM
from WHT_EQS_pattern_generator import (
    get_thickness_field, 
    get_z_field, 
    get_density_field, 
    get_E_field
)
from WHT_EQS_load_cases import (
    TwistCase, 
    PureBendingCase, 
    CornerLiftCase
)
from WHT_EQS_visualization import (
    stage1_visualize_patterns, 
    stage2_visualize_ground_truth, 
    stage3_visualize_comparison
)

# Enable JAX 64-bit precision
jax.config.update("jax_enable_x64", True)

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
Lx, Ly = 1000.0, 400.0
Nx_high, Ny_high = 50, 20
Nx_low, Ny_low = 20, 10

# ==============================================================================
# MODEL MANAGER
# ==============================================================================
class EquivalentSheetModel:
    def __init__(self, Lx, Ly, Nx, Ny):
        self.fem = PlateFEM(Lx, Ly, Nx, Ny)
        self.cases = []
        self.targets = []
        self.optimized_params = None
        
    def add_case(self, case):
        self.cases.append(case)
        print(f"Added Case: {case.name}")

    def generate_targets(self, resolution_high=(50, 20), num_modes_save=5, target_config=None):
        print("\n" + "="*70)
        print(" GENERATING GROUND TRUTH TARGETS")
        print("="*70)
        
        self.resolution_high = resolution_high
        self.num_modes_truth = num_modes_save
        fem_high = PlateFEM(self.fem.Lx, self.fem.Ly, resolution_high[0], resolution_high[1])
        
        X_grid = fem_high.node_coords[:,0].reshape(resolution_high[0]+1, -1)
        Y_grid = fem_high.node_coords[:,1].reshape(resolution_high[0]+1, -1)
        
        if target_config is None: target_config = {}
        
        # 1. Generate Fields
        t_field = get_thickness_field(X_grid, Y_grid, Lx=Lx, Ly=Ly, 
                                       pattern_str=target_config.get('pattern', 'A'), 
                                       base_t=target_config.get('base_t', 1.0), 
                                       bead_t=target_config.get('bead_t', 2.0))
        
        z_field = get_z_field(X_grid, Y_grid, Lx=Lx, Ly=Ly,
                               pattern_pz=target_config.get('pattern_pz', ''), 
                               pz_dict=target_config.get('bead_pz', {}))
        
        rho_field = get_density_field(X_grid, Y_grid, Lx=Lx, Ly=Ly, 
                                       base_rho=target_config.get('base_rho', 7.5e-9))
        
        E_field = get_E_field(X_grid, Y_grid, Lx=Lx, Ly=Ly, 
                               base_E=target_config.get('base_E', 200000.0))
        
        params_high = {
            't': t_field.flatten(),
            'rho': rho_field.flatten(),
            'E': E_field.flatten(),
            'z': z_field.flatten()
        }
        
        # [STAGE 1] Pattern Verification
        stage1_visualize_patterns(resolution_high[0], resolution_high[1], 
                                  X_grid, Y_grid, t_field, z_field)
        
        # 2. FEM Analysis
        K_h, M_h = fem_high.assemble(params_high)
        self.targets = []
        for case in self.cases:
            print(f"Solving Target: {case.name}")
            fd, fv, F = case.get_bcs(fem_high)
            free = np.setdiff1d(np.arange(fem_high.total_dof), fd)
            u = fem_high.solve_static_partitioned(K_h, F, jnp.array(free), fd, fv)
            
            self.targets.append({
                'case_name': case.name,
                'weight': case.weight,
                'u_static': u[2::6],  # W displacement only (3rd DOF in 6-DOF system)
                'max_surface_stress': fem_high.compute_max_surface_stress(u, params_high),
                'max_surface_strain': fem_high.compute_max_surface_strain(u, params_high),
                'params': params_high
            })
            
        print("Solving Target Eigenmodes...")
        vals, vecs = fem_high.solve_eigen(K_h, M_h, num_modes=num_modes_save + 10)
        # Skip 6 rigid body modes for 6-DOF plate
        self.target_eigen = {
            'vals': vals[6:6+num_modes_save],
            'modes': vecs[2::6, 6:6+num_modes_save] # Extract W component (idx 2)
        }
        
        # Mass calculation for constraint
        dx_h, dy_h = Lx/resolution_high[0], Ly/resolution_high[1]
        weights = np.ones((resolution_high[0]+1, resolution_high[1]+1))
        weights[0,:] *= 0.5; weights[-1,:] *= 0.5; weights[:,0] *= 0.5; weights[:,-1] *= 0.5
        self.target_mass = np.sum(t_field * rho_field * weights) * dx_h * dy_h
        print(f"Target Total Mass: {self.target_mass:.6f} tonne")
        
        self.fem_high = fem_high
        self.target_params_high = params_high
        
        # [STAGE 2] Results Visualization
        stage2_visualize_ground_truth(fem_high, self.targets, params_high)

    def optimize(self, opt_config, loss_weights, max_iterations=200, **kwargs):
        print(f"\nStarting Advanced Optimization (Max Iters: {max_iterations})...")
        Nx_l, Ny_l = self.fem.nx, self.fem.ny
        pts_l = self.fem.node_coords
        
        # 1. Target Data Interpolation to Low-Res Mesh
        self.targets_low = []
        Nx_h, Ny_h = self.resolution_high
        xh, yh = jnp.linspace(0, self.fem.Lx, Nx_h+1), jnp.linspace(0, self.fem.Ly, Ny_h+1)
        Xh, Yh = jnp.meshgrid(xh, yh, indexing='ij')
        pts_h = jnp.stack([Xh.flatten(), Yh.flatten()], axis=1)
        
        for tgt in self.targets:
            # Interpolate all metrics needed for loss
            u_l = griddata(pts_h, tgt['u_static'], pts_l, method='cubic')
            stress_l = griddata(pts_h, tgt['max_surface_stress'], pts_l, method='cubic')
            strain_l = griddata(pts_h, tgt['max_surface_strain'], pts_l, method='cubic')
            
            self.targets_low.append({
                'u_static': jnp.array(u_l),
                'max_stress': jnp.array(stress_l),
                'max_strain': jnp.array(strain_l),
                'weight': tgt['weight']
            })
            
        # Modal Targets
        t_vals = self.target_eigen['vals']
        t_modes_l = [griddata(pts_h, self.target_eigen['modes'][:, i], pts_l, method='cubic') 
                     for i in range(len(t_vals))]
        t_modes_l = jnp.stack(t_modes_l, axis=1)

        # 2. Optimization Parameters Setup
        params = {
            't': jnp.full((Nx_l+1, Ny_l+1), opt_config['t'].get('init', 1.0)),
            'rho': jnp.full((Nx_l+1, Ny_l+1), opt_config['rho'].get('init', 7.5e-9)),
            'E': jnp.full((Nx_l+1, Ny_l+1), opt_config['E'].get('init', 200000.0))
        }
        
        optimizer = optax.adam(learning_rate=kwargs.get('learning_rate', 0.01))
        opt_state = optimizer.init(params)

        @jax.jit
        def loss_fn(p):
            K, M = self.fem.assemble(p)
            loss = 0.0
            
            # --- Static Component Loss ---
            for i, case in enumerate(self.cases):
                fd, fv, F = case.get_bcs(self.fem)
                free = jnp.setdiff1d(jnp.arange(self.fem.total_dof), fd)
                u = self.fem.solve_static_partitioned(K, F, free, fd, fv)
                w = u[2::6] # W-displacement
                
                # Displacement Loss
                loss += loss_weights.get('static', 0.0) * jnp.mean((w - self.targets_low[i]['u_static'])**2) * case.weight
                
                # Stress/Strain Loss (High-order metrics)
                if loss_weights.get('surface_stress', 0.0) > 0:
                    curr_stress = self.fem.compute_max_surface_stress(u, p)
                    loss += loss_weights['surface_stress'] * jnp.mean((curr_stress - self.targets_low[i]['max_stress'])**2)
                if loss_weights.get('surface_strain', 0.0) > 0:
                    curr_strain = self.fem.compute_max_surface_strain(u, p)
                    loss += loss_weights['surface_strain'] * jnp.mean((curr_strain - self.targets_low[i]['max_strain'])**2)

            # --- Modal Component Loss ---
            if loss_weights.get('freq', 0.0) > 0 or loss_weights.get('mode', 0.0) > 0:
                vals, vecs = self.fem.solve_eigen(K, M, num_modes=len(t_vals)+10)
                # Freq matching (Skip 6 RB modes)
                loss += loss_weights.get('freq', 0.0) * jnp.mean((vals[6:6+len(t_vals)] - t_vals)**2)
                
                # Mode Shape matching (MAC - Modal Assurance Criterion)
                if loss_weights.get('mode', 0.0) > 0:
                    modes_w = vecs[2::6, 6:6+len(t_vals)] # W-component
                    for j in range(len(t_vals)):
                        # Simple MAC-based loss: 1 - Correlation^2
                        num = jnp.sum(modes_w[:, j] * t_modes_l[:, j])**2
                        den = jnp.sum(modes_w[:, j]**2) * jnp.sum(t_modes_l[:, j]**2)
                        loss += loss_weights['mode'] * (1.0 - num/den)

            # --- Constraints & Regularization ---
            # Total Mass Constraint
            if loss_weights.get('mass', 0.0) > 0:
                dx, dy = self.fem.Lx/Nx_l, self.fem.Ly/Ny_l
                curr_mass = jnp.sum(p['t'] * p['rho']) * dx * dy # Simple approx
                loss += loss_weights['mass'] * ((curr_mass - self.target_mass)/self.target_mass)**2

            # Smoothness (Total Variation Penalty)
            if loss_weights.get('smoothness', 0.0) > 0:
                diff_x = jnp.diff(p['t'], axis=0)**2
                diff_y = jnp.diff(p['t'], axis=1)**2
                loss += loss_weights['smoothness'] * (jnp.mean(diff_x) + jnp.mean(diff_y))

            return loss

        # 3. Optimization Loop with Early Stopping
        patience = kwargs.get('patience', 20)
        best_loss = float('inf')
        wait = 0
        
        for i in range(max_iterations):
            val, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            # Parameter Clipping (Safety)
            for k in params:
                params[k] = jnp.clip(params[k], opt_config[k]['min'], opt_config[k]['max'])
            
            if i % 10 == 0:
                print(f"Iter {i:3d} | Total Loss: {val:.6e}")

            # Early Stopping Check
            if val < best_loss - 1e-8:
                best_loss = val
                wait = 0
                self.optimized_params = params
            else:
                wait += 1
                if wait >= patience:
                    print(f"✔ Convergence reached. Early stopping at iteration {i}.")
                    break

        return self.optimized_params

    def verify(self):
        print("\nFinal Verification...")
        # (Simplified verification for this clean version)
        stage3_visualize_comparison(self.fem_high, self.targets, 
                                     {'t': griddata(np.linspace(0,Lx,Nx_low+1), self.optimized_params['t'], np.linspace(0,Lx,Nx_high+1), method='cubic'),
                                      'z': jnp.zeros(self.fem_high.num_nodes)}, 
                                     self.target_params_high)

# ==============================================================================
# MAIN EXECUTION: COMPREHENSIVE CONFIGURATION TEMPLATE
# ==============================================================================
if __name__ == '__main__':
    # 1. Initialize Model Manager (Low-res mesh for optimization)
    model = EquivalentSheetModel(Lx, Ly, Nx_low, Ny_low)
    
    # 2. Define Comprehensive Load Cases for Optimization
    # Supports multiple cases: Twist, Pure Bending, and Corner Lift
    model.add_case(TwistCase("twist_x", axis='x', value=1.5, mode='angle', weight=1.0))
    model.add_case(TwistCase("twist_y", axis='y', value=1.5, mode='angle', weight=0.5))
    model.add_case(PureBendingCase("bend_y", axis='y', value=3.0, mode='angle', weight=1.0))
    model.add_case(CornerLiftCase("lift_br", corner='br', value=5.0, mode='disp', weight=0.8))
    
    # 3. Ground Truth Generation Configuration (target_config)
    # This defines the "Physical Reality" we want our optimization to match.
    target_config = {
        # --- Thickness (Bead) Patterns ---
        'pattern': 'ABC',           # Alphanumeric bead pattern string
        'base_t': 1.0,              # Base sheet thickness (mm)
        'bead_t': {                 # Per-character bead thickness (mm)
            'A': 2.0, 
            'B': 2.5, 
            'C': 1.5,
            'default': 2.0          # Fallback if character not in dict
        },
        
        # --- Topography (Z-Shape) Patterns ---
        'pattern_pz': 'TNY',        # Topography pattern string (embossed features)
        'bead_pz': {                # Per-character Z-height (mm)
            'T': 2.0,               # Raised +2mm
            'N': 1.0,               # Raised +1mm
            'Y': -2.0               # Recessed -2mm
        },
        
        # --- Base Material Properties ---
        'base_rho': 7.85e-9,        # Base Density (tonne/mm³) - e.g., steel
        'base_E': 210000.0,         # Base Young's Modulus (MPa) - e.g., 210 GPa
    }
    
    # Generate ground truth responses (Includes interactive visualization stages)
    model.generate_targets(
        resolution_high=(50, 20),   # Resolution for high-fidelity "truth" mesh
        num_modes_save=5,           # <<< NUMBER OF EIGENMODES TO MATCH >>>
                                    # (e.g., 5 means matching the first 5 natural frequencies)
        target_config=target_config # The configuration defined above
    )
    
    # 4. Optimization Strategy Configuration (opt_config)
    # Define search space and initial guesses for equivalent parameters.
    opt_config = {
        't': {                      # Thickness Search Space
            'init': 1.2,            # Initial starting guess (mm)
            'min': 0.5,             # Lower bound for optimization
            'max': 5.0              # Upper bound for optimization
        },
        'rho': {                    # Density Search Space
            'init': 7.5e-9,         # Initial density guess (tonne/mm³)
            'min': 5.0e-9, 
            'max': 1.0e-8
        },
        'E': {                      # Stiffness (E) Search Space
            'init': 200000.0,       # Initial Young's Modulus guess (MPa)
            'min': 50000.0, 
            'max': 300000.0
        }
    }
    
    # 5. Run Optimization Loop (Advanced Configuration)
    # loss_weights now supports: static, freq, mode, surface_stress, surface_strain, mass, smoothness
    loss_weights = {
        'static': 1.0,           # Match static displacement fields
        'freq': 0.5,             # Match natural frequencies
        'mode': 0.5,             # Match mode shapes (MAC)
        'surface_stress': 0.5,   # Match max surface stress distribution
        'surface_strain': 0.5,   # Match max surface strain distribution
        'mass': 0.2,             # Match total model mass
        'smoothness': 0.05       # Ensure manufacturable (smooth) thickness fields
    }
    
    model.optimize(
        opt_config=opt_config, 
        loss_weights=loss_weights, 
        max_iterations=300,      # Increased for better convergence
        learning_rate=0.01,      # Optimizer step size
        patience=30              # Early stopping patience (iters without progress)
    )
    
    # 6. Final Comparative Verification
    # Launches 3D Side-by-Side comparison of Target vs Optimized fields.
    model.verify()
