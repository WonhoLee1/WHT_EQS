# ==============================================================================
# main_verification.py (Explicit & Comprehensive Version)
# ==============================================================================
import jax
import jax.numpy as jnp
import optax
import numpy as np
from scipy.interpolate import griddata
import os

# Import our modular components
from solver import PlateFEM
from WHT_EQS_pattern_generator import get_thickness_field, get_z_field
from WHT_EQS_load_cases import TwistCase, PureBendingCase, CornerLiftCase
from WHT_EQS_visualization import (
    stage1_visualize_patterns,
    stage2_visualize_ground_truth,
    stage3_visualize_comparison
)

# Mesh Settings
Lx, Ly = 1000.0, 400.0
Nx_high, Ny_high = 50, 20      # High-res ground truth
Nx_low, Ny_low = 30, 12        # Low-res optimization mesh

class EquivalentSheetModel:
    def __init__(self, Lx, Ly, nx, ny):
        self.fem = PlateFEM(Lx, Ly, nx, ny)
        self.cases = []
        self.targets = []
        self.resolution_high = (50, 20)
        self.target_mass = 0.0
        self.optimized_params = None
        self.target_params_high = None

    def add_case(self, case):
        self.cases.append(case)

    def generate_targets(self, resolution_high=(50, 20), num_modes_save=5, target_config=None):
        print("\n" + "="*70)
        print(" [STAGE 1] TARGET GENERATION & PATTERN VERIFICATION")
        print("="*70)
        self.resolution_high = resolution_high
        Nx_h, Ny_h = resolution_high
        
        # 1. Create High-Resolution Mesh for "Ground Truth"
        self.fem_high = PlateFEM(self.fem.Lx, self.fem.Ly, Nx_h, Ny_h)
        fem_high = self.fem_high
        xh = np.linspace(0, self.fem.Lx, Nx_h+1)
        yh = np.linspace(0, self.fem.Ly, Ny_h+1)
        Xh, Yh = np.meshgrid(xh, yh, indexing='ij')
        
        # 2. Build High-fidelity Property Fields
        t_h = get_thickness_field(
            Xh, Yh, Lx=self.fem.Lx, Ly=self.fem.Ly, 
            pattern_str=target_config.get('pattern', 'ABC'),
            base_t=target_config.get('base_t', 1.0),
            bead_t=target_config.get('bead_t', 2.0)
        )
        z_h = get_z_field(
            Xh, Yh, Lx=self.fem.Lx, Ly=self.fem.Ly, 
            pattern_pz=target_config.get('pattern_pz', 'TNY'),
            pz_dict=target_config.get('bead_pz', {})
        )
        rho_h = jnp.full_like(Xh, target_config.get('base_rho', 7.85e-9))
        E_h = jnp.full_like(Xh, target_config.get('base_E', 210000.0))
        
        params_high = {'t': t_h, 'z': z_h, 'rho': rho_h, 'E': E_h}
        self.target_params_high = params_high
        
        # Calculate target mass
        dx, dy = self.fem.Lx/Nx_h, self.fem.Ly/Ny_h
        self.target_mass = float(jnp.sum(t_h * rho_h) * dx * dy)
        
        # [STAGE 1] Interactive Pattern Check
        stage1_visualize_patterns(Nx_h, Ny_h, Xh, Yh, t_h, z_h)
        
        # 3. Solve FEM for each load case (High Fidelity)
        print("\nSolving High-Resolution Ground Truth...")
        K_h, M_h = fem_high.assemble(params_high)
        self.targets = []
        
        for case in self.cases:
            print(f" -> Solving Case: {case.name}")
            fixed_dofs, fixed_vals, F = case.get_bcs(fem_high)
            free_dofs = jnp.setdiff1d(jnp.arange(fem_high.total_dof), fixed_dofs)
            u = fem_high.solve_static_partitioned(K_h, F, free_dofs, fixed_dofs, fixed_vals)
            
            # Compute Full Reaction Forces (R = K*u - F_ext)
            # F_int represents internal nodal forces. At free nodes F_int = F_ext.
            # At fixed nodes, F_int = F_ext + R. So R = F_int - F_ext.
            F_int = K_h @ u
            R_residual = F_int - F
            
            self.targets.append({
                'case_name': case.name,
                'weight': case.weight,
                'u_static': np.array(u[2::6]), # W-displacement only
                'u_full': np.array(u),         # Full 6-DOF displacement
                'reaction_full': np.array(R_residual), # Full reaction force vector
                'max_surface_stress': np.array(fem_high.compute_max_surface_stress(u, params_high)),
                'max_surface_strain': np.array(fem_high.compute_max_surface_strain(u, params_high)),
                'strain_energy_density': np.array(fem_high.compute_strain_energy_density(u, params_high)),
                'params': params_high,
                'fixed_dofs': np.array(fixed_dofs), # Store BCs for visualization
                'force_vector': np.array(F)         # Store Loads for visualization
            })
            
        print("Solving Target Eigenmodes...")
        vals, vecs = fem_high.solve_eigen(K_h, M_h, num_modes=num_modes_save + 10)
        self.target_eigen = {
            'vals': vals[6:6+num_modes_save],
            'modes': vecs[2::6, 6:6+num_modes_save] # W-component
        }
        
        # [STAGE 2] Results Visualization
        stage2_visualize_ground_truth(fem_high, self.targets, params_high, eigen_data=self.target_eigen)

    def optimize(self, opt_config, loss_weights, 
                 use_smoothing=False, 
                 use_strain_energy=True, 
                 use_surface_stress=True, 
                 use_surface_strain=True,
                 use_mass_constraint=True, 
                 mass_tolerance=0.05,
                 max_iterations=300, 
                 use_early_stopping=True, 
                 early_stop_patience=30, 
                 early_stop_tol=1e-8,
                 learning_rate=0.01,
                 num_modes_loss=None):
                 
        print("\n" + "="*70)
        print(" [STAGE 3] ADVANCED OPTIMIZATION (EXPLICIT LOGIC)")
        print("="*70)
        
        Nx_l, Ny_l = self.fem.nx, self.fem.ny
        pts_l = self.fem.node_coords
        
        # 1. Interpolate High-Res Targets to Low-Res Mesh
        Nx_h, Ny_h = self.resolution_high
        xh, yh = jnp.linspace(0, self.fem.Lx, Nx_h+1), jnp.linspace(0, self.fem.Ly, Ny_h+1)
        Xh, Yh = jnp.meshgrid(xh, yh, indexing='ij')
        pts_h = jnp.stack([Xh.flatten(), Yh.flatten()], axis=1)
        
        self.targets_low = []
        for tgt in self.targets:
            u_l = griddata(pts_h, tgt['u_static'], pts_l, method='cubic')
            stress_l = griddata(pts_h, tgt['max_surface_stress'], pts_l, method='cubic')
            strain_l = griddata(pts_h, tgt['max_surface_strain'], pts_l, method='cubic')
            
            # Interpolate Strain Energy Density
            sed_l = griddata(pts_h, tgt['strain_energy_density'], pts_l, method='cubic')
            
            self.targets_low.append({
                'u_static': jnp.array(u_l),
                'max_stress': jnp.array(stress_l),
                'max_strain': jnp.array(strain_l),
                'strain_energy_density': jnp.array(sed_l),
                'weight': tgt.get('weight', 1.0)
            })
            
        # Modal Targets Interpolation
        n_loss = num_modes_loss if num_modes_loss is not None else 5
        t_vals = self.target_eigen['vals'][:n_loss]
             
        t_modes_l = [griddata(pts_h, self.target_eigen['modes'][:, i], pts_l, method='cubic') 
                     for i in range(len(t_vals))]
        t_modes_l = jnp.stack(t_modes_l, axis=1)

        # 2. Optimization Parameters Setup
        # 3. Optimization Loop with Detailed Logging
        print(f"Starting Optimization (Modes for Loss: {n_loss})...")
        
        # Initial Parameters
        params = {
            't': jnp.full((Nx_l+1, Ny_l+1), opt_config['t'].get('init', 1.0)),
            'rho': jnp.full((Nx_l+1, Ny_l+1), opt_config['rho'].get('init', 7.5e-9)),
            'E': jnp.full((Nx_l+1, Ny_l+1), opt_config['E'].get('init', 210000.0))
        }
        
        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(params)

        # Pre-compute BCs to avoid jnp.where inside JIT
        case_bcs = []
        for case in self.cases:
            fd, fv, F = case.get_bcs(self.fem)
            free = jnp.setdiff1d(jnp.arange(self.fem.total_dof), fd)
            case_bcs.append({'fd': fd, 'fv': fv, 'F': F, 'free': free})

        @jax.jit
        def loss_fn(p):
            K, M = self.fem.assemble(p)
            
            # Initialize Loss Components
            l_static = 0.0
            l_stress = 0.0
            l_strain = 0.0
            l_energy = 0.0
            l_freq = 0.0
            l_mode = 0.0
            l_mass = 0.0
            l_reg = 0.0
            
            # --- 1. Static Responses Loss ---
            for i, case in enumerate(self.cases):
                fd = case_bcs[i]['fd']
                fv = case_bcs[i]['fv']
                F  = case_bcs[i]['F']
                free = case_bcs[i]['free']
                
                u = self.fem.solve_static_partitioned(K, F, free, fd, fv)
                w = u[2::6] # Vertical displacement
                
                # Displacement matching (RMSE)
                if loss_weights.get('static', 0.0) > 0:
                    delta = w - self.targets_low[i]['u_static']
                    scale = jnp.mean(jnp.abs(self.targets_low[i]['u_static'])) + 1e-6
                    l_static += jnp.mean((delta / scale)**2) * self.targets_low[i]['weight']
                
                # Surface Stress matching
                if use_surface_stress and loss_weights.get('surface_stress', 0.0) > 0:
                    curr_stress = self.fem.compute_max_surface_stress(u, p)
                    delta = curr_stress - self.targets_low[i]['max_stress']
                    scale = jnp.mean(jnp.abs(self.targets_low[i]['max_stress'])) + 1e-6
                    l_stress += jnp.mean((delta / scale)**2)
                    
                # Surface Strain matching
                if use_surface_strain and loss_weights.get('surface_strain', 0.0) > 0:
                    curr_strain = self.fem.compute_max_surface_strain(u, p)
                    delta = curr_strain - self.targets_low[i]['max_strain']
                    scale = jnp.mean(jnp.abs(self.targets_low[i]['max_strain'])) + 1e-6
                    l_strain += jnp.mean((delta / scale)**2)
                
                # Strain Energy matching
                if use_strain_energy and loss_weights.get('strain_energy', 0.0) > 0:
                    curr_energy = self.fem.compute_strain_energy_density(u, p)
                    # For now, if we don't have target energy interpolated, skip or use approximation
                    # Assuming we target 0 for minimization or match specific profile? 
                    # The USER requirement implies we should match. 
                    # If target not available, maybe skip to avoid NaN? 
                    # Let's assume passed in targets_low if available, else skip.
                    if 'strain_energy_density' in self.targets_low[i]:
                         tgt_e = self.targets_low[i]['strain_energy_density']
                         scale = jnp.mean(jnp.abs(tgt_e)) + 1e-6
                         l_energy += jnp.mean(((curr_energy - tgt_e)/scale)**2)
                    else:
                         pass # Skip if no target

            # Normalize Static Losses by number of cases
            n_cases = len(self.cases)
            l_static /= n_cases
            l_stress /= n_cases
            l_strain /= n_cases
            l_energy /= n_cases

            # --- 2. Modal Responses Loss ---
            if loss_weights.get('freq', 0.0) > 0 or loss_weights.get('mode', 0.0) > 0:
                vals, vecs = self.fem.solve_eigen(K, M, num_modes=len(t_vals)+10)
                
                # Eigenfrequency matching (Skip 6 RB modes)
                if loss_weights.get('freq', 0.0) > 0:
                    # Log diff
                    curr_vals = vals[6:6+len(t_vals)]
                    diff = jnp.log(curr_vals) - jnp.log(t_vals)
                    l_freq = jnp.mean(diff**2)
                
                # Mode Shape matching (MAC)
                if loss_weights.get('mode', 0.0) > 0:
                    modes_w = vecs[2::6, 6:6+len(t_vals)]
                    for j in range(len(t_vals)):
                        num = jnp.sum(modes_w[:, j] * t_modes_l[:, j])**2
                        den = jnp.sum(modes_w[:, j]**2) * jnp.sum(t_modes_l[:, j]**2)
                        mac = num / (den + 1e-12)
                        l_mode += (1.0 - mac)
                    l_mode /= len(t_vals)

            # --- 3. Geometric Constraints & Regularization ---
            # Mass Constraint
            if use_mass_constraint and loss_weights.get('mass', 0.0) > 0:
                dx_l, dy_l = self.fem.Lx/Nx_l, self.fem.Ly/Ny_l
                curr_mass = jnp.sum(p['t'] * p['rho']) * dx_l * dy_l
                mass_err = (curr_mass - self.target_mass) / self.target_mass
                l_mass = jnp.abs(mass_err) # Linear penalty for robustness

            # Smoothness (TV Penalty)
            if use_smoothing or loss_weights.get('reg', 0.0) > 0:
                reg_weight = loss_weights.get('reg', loss_weights.get('smoothness', 0.0))
                diff_x = jnp.diff(p['t'], axis=0)**2
                diff_y = jnp.diff(p['t'], axis=1)**2
                l_reg = (jnp.mean(diff_x) + jnp.mean(diff_y))

            # Weighted Sum
            total_loss = (
                l_static * loss_weights.get('static', 0.0) +
                l_stress * loss_weights.get('surface_stress', 0.0) +
                l_strain * loss_weights.get('surface_strain', 0.0) + 
                l_energy * loss_weights.get('strain_energy', 0.0) +
                l_freq   * loss_weights.get('freq', 0.0) +
                l_mode   * loss_weights.get('mode', 0.0) + 
                l_mass   * loss_weights.get('mass', 0.0) +
                l_reg    * loss_weights.get('reg', 0.0)
            )
            
            metrics = {
                'Total': total_loss,
                'Static': l_static,
                'Stress': l_stress,
                'Strain': l_strain,
                'Energy': l_energy,
                'Freq': l_freq,
                'Mode': l_mode,
                'Mass': l_mass,
                'Reg': l_reg
            }
            return total_loss, metrics

        # 3. Optimization Loop with Early Stopping
        best_loss = float('inf')
        wait = 0
        
        print(f"{'Iter':<5} | {'Total':<10} | {'Static':<9} | {'Stress':<9} | {'Strain':<9} | {'Energy':<9} | {'Mass':<9} | {'Reg':<9}")
        print("-" * 95)

        for i in range(max_iterations):
            (val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            
            # Check for NaN
            if jnp.isnan(val):
                print(f"⚠ NaN Loss detected at iter {i}. Stopping.")
                break
                
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            # Explicit Param Clipping
            params['t'] = jnp.clip(params['t'], opt_config['t']['min'], opt_config['t']['max'])
            params['rho'] = jnp.clip(params['rho'], opt_config['rho']['min'], opt_config['rho']['max'])
            params['E'] = jnp.clip(params['E'], opt_config['E']['min'], opt_config['E']['max'])
            
            if i % 10 == 0:
                 print(f"{i:<5d} | {val:<10.4e} | {metrics['Static']:<9.2e} | {metrics['Stress']:<9.2e} | {metrics['Strain']:<9.2e} | {metrics['Energy']:<9.2e} | {metrics['Mass']:<9.2e} | {metrics['Reg']:<9.2e}")

            # Early Stopping Logic (Explicit)
            if use_early_stopping:
                if val < best_loss - early_stop_tol:
                    best_loss = val
                    wait = 0
                    self.optimized_params = params
                else:
                    wait += 1
                    if wait >= (early_stop_patience or 30):
                        print(f"✔ Early stopping triggered at iteration {i}. Convergence reached.")
                        break
            else:
                self.optimized_params = params

        return self.optimized_params

    def verify(self):
        print("\n" + "="*70)
        print(" [STAGE 4] FINAL COMPARATIVE VERIFICATION")
        print("="*70)
        # Interpolate optimized field back to high-res for visualization
        Nx_h, Ny_h = self.resolution_high
        xh_new = np.linspace(0, self.fem.Lx, Nx_h+1)
        yh_new = np.linspace(0, self.fem.Ly, Ny_h+1)
        Xh_new, Yh_new = np.meshgrid(xh_new, yh_new, indexing='ij')
        
        # We need a 1D coordinate array for griddata
        xl_1d = np.linspace(0, self.fem.Lx, Nx_low+1)
        yl_1d = np.linspace(0, self.fem.Ly, Ny_low+1)
        Xl_grid, Yl_grid = np.meshgrid(xl_1d, yl_1d, indexing='ij')
        pts_low = np.column_stack([Xl_grid.flatten(), Yl_grid.flatten()])
        
        t_opt_h = griddata(pts_low, np.array(self.optimized_params['t']).flatten(), 
                           (Xh_new, Yh_new), method='cubic')
        z_opt_h = griddata(pts_low, np.array(self.optimized_params['z' if 'z' in self.optimized_params else 't']).flatten(), 
                           (Xh_new, Yh_new), method='cubic') # Placeholder for topography if optimized
        
        opt_params_h = {
            't': t_opt_h,
            'z': self.target_params_high['z'], # Hold topography constant for now
            'rho': self.target_params_high['rho'],
            'E': self.target_params_high['E']
        }
        
        stage3_visualize_comparison(self.fem_high, self.targets, opt_params_h, self.target_params_high)

# ==============================================================================
# MAIN EXECUTION: COMPREHENSIVE CONFIGURATION TEMPLATE
# ==============================================================================
if __name__ == '__main__':
    # XLA Flags (Optimized for CPU)
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"

    model = EquivalentSheetModel(Lx, Ly, Nx_low, Ny_low)
    
    # 2. Add Load Cases
    model.add_case(TwistCase("twist_x", axis='x', value=1.5, mode='angle', weight=1.0))
    model.add_case(TwistCase("twist_y", axis='y', value=1.5, mode='angle', weight=1.0))
    model.add_case(PureBendingCase("bend_y", axis='y', value=3.0, mode='angle', weight=1.0))
    model.add_case(CornerLiftCase("lift_br", corner='br', value=5.0, mode='disp', weight=1.0))
    
    # 3. Ground Truth Generation (target_config)
    target_config = {
        'pattern': 'ABC',           'base_t': 1.0, 
        'bead_t': {'A': 2.0, 'B': 2.5, 'C': 1.5},
        'pattern_pz': 'TNY',        'bead_pz': {'T': 2.0, 'N': 1.0, 'Y': -2.0},
        'base_rho': 7.85e-9,        'base_E': 210000.0,
    }
    
    model.generate_targets(resolution_high=(50, 20), num_modes_save=5, target_config=target_config)
    
    # 4. Optimization Search Space (opt_config)
    opt_config = {
        't':   {'init': 1.2,      'min': 0.5,    'max': 5.0},
        'rho': {'init': 7.85e-9,   'min': 5e-9,   'max': 1e-8},
        'E':   {'init': 200000.0,  'min': 50000,  'max': 300000}
    }
    
    # 5. Full Loss Weights (as previously defined)
    weights = {
        'static': 1.0,           # Displacement matching
        'freq': 0.0,             # Eigenfrequency matching (Disabled to prevent NaN)
        'mode': 0.0,             # Mode shape (MAC) matching (Disabled to prevent NaN)
        'curvature': 0.0,        # [Legacy]
        'moment': 0.0,           # [Legacy]
        'strain_energy': 2.0,    # Strain energy density matching
        'surface_stress': 1.0,   # Max surface stress matching
        'surface_strain': 1.0,   # Max surface strain matching
        'reg': 0.05,             # Total Variation (Smoothness)
        'mass': 1.0              # Mass constraints
    }
    
    # Run Optimization with EXPLICIT flags
    model.optimize(
        opt_config, weights, 
        use_smoothing=True, 
        use_strain_energy=True, 
        use_surface_stress=True, 
        use_surface_strain=True,
        use_mass_constraint=True, 
        mass_tolerance=0.05,
        max_iterations=300, 
        use_early_stopping=True, 
        early_stop_patience=30, 
        early_stop_tol=1e-8,
        learning_rate=0.01
    )
    
    model.verify()
