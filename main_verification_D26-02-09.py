# ==============================================================================
# main_verification.py (Explicit & Comprehensive Version)
# ==============================================================================
import jax
import jax.numpy as jnp
import optax
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
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
        
        # Dynamically find the first elastic mode (> 0.01 Hz)
        all_freqs_h = np.sqrt(np.maximum(np.array(vals), 0.0)) / (2 * np.pi)
        valid_indices = np.where(all_freqs_h > 0.01)[0]
        start_idx = valid_indices[0] if len(valid_indices) > 0 else 6
        print(f" -> Found first elastic mode at index {start_idx} ({all_freqs_h[start_idx]:.2f} Hz)")

        self.target_eigen = {
            'vals': vals[start_idx : start_idx + num_modes_save],
            'modes': vecs[2::6, start_idx : start_idx + num_modes_save] # W-component
        }
        
        # Pre-output target frequencies for user overview
        target_freqs = np.sqrt(np.maximum(np.array(self.target_eigen['vals']), 0.0)) / (2 * np.pi)
        print(" -> Target frequencies (Hz): " + ", ".join([f"{f:.2f}" for f in target_freqs]))
        
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
        print(f"Starting Optimization (Modes for Loss: {n_loss})...")
        
        # --- IMPROVED: Selective Optimization with Parameter Scaling ---
        # Scaling maps physical values to ~1.0 for balanced gradients
        self.scaling = {
            't': opt_config['t'].get('init', 1.0),
            'rho': opt_config['rho'].get('init', 7.85e-9),
            'E': opt_config['E'].get('init', 210000.0),
            'pz': 1.0 # pz usually centered around 0, keep as is or shift
        }
        
        full_params_scaled = {}
        for key_name in ['t', 'rho', 'E', 'pz']:
            if key_name in opt_config:
                init_phys = opt_config[key_name].get('init', 1.0 if key_name == 't' else 0.0)
                # Scale: phys = scaled * scale_factor => scaled = phys / scale_factor
                scale_factor = self.scaling.get(key_name, 1.0)
                full_params_scaled[key_name] = jnp.full((Nx_l+1, Ny_l+1), init_phys / scale_factor)
        
        # Initialize with small jitter to break symmetry
        key = jax.random.PRNGKey(42)
        full_params_scaled['t'] = full_params_scaled['t'] + 1e-4 * jax.random.uniform(key, full_params_scaled['t'].shape)

        # Split into optimized (trainable) and fixed parameters
        params = {}
        fixed_params_scaled = {}
        for k, v in full_params_scaled.items():
            if opt_config.get(k, {}).get('opt', True):
                params[k] = v
            else:
                fixed_params_scaled[k] = v
        
        print(f" -> Optimized variables (Scaled): {list(params.keys())}")
        print(f" -> Fixed variables (Scaled): {list(fixed_params_scaled.keys())}")

        # Use a more robust optimizer chain
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=0.005)
        )
        opt_state = optimizer.init(params)

        # Pre-compute BCs to avoid jnp.where inside JIT
        case_bcs = []
        for case in self.cases:
            fd, fv, F = case.get_bcs(self.fem)
            free = jnp.setdiff1d(jnp.arange(self.fem.total_dof), fd)
            case_bcs.append({'fd': fd, 'fv': fv, 'F': F, 'free': free})

        @jax.jit
        def loss_fn(p_scaled, fixed_p_scaled, scales):
            # 1. Unscale to physical units
            combined_phys = {}
            for k, v in p_scaled.items():
                combined_phys[k] = v * scales[k]
            for k, v in fixed_p_scaled.items():
                combined_phys[k] = v * scales[k]
            
            # 2. Map 'pz' to 'z' for the FEM solver
            if 'pz' in combined_phys:
                combined_phys['z'] = combined_phys['pz']
            
            K, M = self.fem.assemble(combined_phys)
            p_actual = combined_phys
            
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
                    curr_stress = self.fem.compute_max_surface_stress(u, p_actual)
                    delta = curr_stress - self.targets_low[i]['max_stress']
                    scale = jnp.mean(jnp.abs(self.targets_low[i]['max_stress'])) + 1e-6
                    l_stress += jnp.mean((delta / scale)**2)
                    
                # Surface Strain matching
                if use_surface_strain and loss_weights.get('surface_strain', 0.0) > 0:
                    curr_strain = self.fem.compute_max_surface_strain(u, p_actual)
                    delta = curr_strain - self.targets_low[i]['max_strain']
                    scale = jnp.mean(jnp.abs(self.targets_low[i]['max_strain'])) + 1e-6
                    l_strain += jnp.mean((delta / scale)**2)
                
                # Strain Energy matching
                if use_strain_energy and loss_weights.get('strain_energy', 0.0) > 0:
                    curr_energy = self.fem.compute_strain_energy_density(u, p_actual)
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
            if (loss_weights.get('freq', 0.0) > 0 or loss_weights.get('mode', 0.0) > 0 or True):
                # Solve global eigenvalue problem
                vals, vecs = self.fem.solve_eigen(K, M, num_modes=len(t_vals)+10)
                
                # Eigenfrequency matching (Filter for > 0.01 Hz)
                all_freqs_curr = jnp.sqrt(jnp.maximum(vals, 0.0)) / (2 * jnp.pi)
                elastic_mask = (all_freqs_curr > 0.01)
                start_idx_l = jnp.argmax(elastic_mask) # First True index (e.g. index 7)
                
                if loss_weights.get('freq', 0.0) > 0:
                    curr_vals = jax.lax.dynamic_slice_in_dim(vals, start_idx_l, len(t_vals))
                    curr_vals = jnp.abs(curr_vals)
                    l_freq = jnp.mean((jnp.log(curr_vals + 1e-4) - jnp.log(t_vals + 1e-4))**2)
                    l_freq = jnp.nan_to_num(l_freq, nan=0.0)
                
                # Mode Shape matching (MAC)
                l_mode = 0.0
                modes_w = jax.lax.dynamic_slice_in_dim(vecs[2::6, :], start_idx_l, len(t_vals), axis=1)
                for j in range(len(t_vals)):
                    num_i = jnp.sum(modes_w[:, j] * t_modes_l[:, j])**2
                    den_i = (jnp.sum(modes_w[:, j]**2) + 1e-10) * (jnp.sum(t_modes_l[:, j]**2) + 1e-10)
                    mac = num_i / den_i
                    l_mode += (1.0 - mac)
                l_mode = jnp.nan_to_num(l_mode / len(t_vals), nan=0.0)

                # Track first frequency > 0.01 Hz for monitoring
                f1_hz = jnp.where(jnp.any(elastic_mask), all_freqs_curr[start_idx_l], 0.0)

            # --- 3. Geometric Constraints & Regularization ---
            # Mass Constraint
            if use_mass_constraint and loss_weights.get('mass', 0.0) > 0:
                dx_l, dy_l = self.fem.Lx/Nx_l, self.fem.Ly/Ny_l
                curr_mass = jnp.sum(p_actual['t'] * p_actual['rho']) * dx_l * dy_l
                mass_err = (curr_mass - self.target_mass) / self.target_mass
                l_mass = jnp.abs(mass_err) # Linear penalty for robustness

            # Smoothness (TV Penalty)
            if use_smoothing or loss_weights.get('reg', 0.0) > 0:
                l_reg = 0.0
                if 't' in p_scaled:
                    l_reg += (jnp.mean(jnp.diff(p_scaled['t'], axis=0)**2) + jnp.mean(jnp.diff(p_scaled['t'], axis=1)**2))
                if 'pz' in p_scaled:
                    l_reg += 5.0 * (jnp.mean(jnp.diff(p_scaled['pz'], axis=0)**2) + jnp.mean(jnp.diff(p_scaled['pz'], axis=1)**2))

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
            # Final NaN Guard
            total_loss = jnp.nan_to_num(total_loss, nan=1e10)
            
            metrics = {
                'Total': total_loss,
                'Disp': l_static,     # Renamed from Static for clarity
                'Start': l_stress,    # Renamed/shortened from Stress
                'Strn': l_strain,     # Renamed/shortened from Strain
                'Engy': l_energy,     # Renamed/shortened from Energy
                'Freq': l_freq,
                'Mode': l_mode,
                'Mass': l_mass,
                'Reg': l_reg,
                'f1_hz': f1_hz
            }
            return total_loss, metrics

        # 3. Optimization Loop with Early Stopping
        best_loss = float('inf')
        wait = 0
        
        print(f"{'Iter':<5} | {'Total':<10} | {'Disp':<9} | {'Strs':<9} | {'Strn':<9} | {'Engy':<9} | {'Freq':<9} | {'Mode':<9} | {'Freq1':<6} | {'G-Norm':<9}")
        print("-" * 115)

        for i in range(max_iterations):
            # Standard Optimization Step
            (val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, fixed_params_scaled, self.scaling)
            
            # Global Safe-Guard: Filter NaNs/Infs from gradients
            grads = jax.tree_util.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
                
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            # Explicit Param Clipping for Scaled Variables
            for k in params.keys():
                s_factor = self.scaling[k]
                params[k] = jnp.clip(params[k], opt_config[k]['min']/s_factor, opt_config[k]['max']/s_factor)
            
            # Combine and Unscale for report/save
            self.optimized_params = {}
            for k, v in params.items(): self.optimized_params[k] = v * self.scaling[k]
            for k, v in fixed_params_scaled.items(): self.optimized_params[k] = v * self.scaling[k]
            
            if i % 10 == 0:
                 gnorm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
                 print(f"{i:<5d} | {val:<10.4e} | {metrics['Disp']:<9.2e} | {metrics['Start']:<9.2e} | {metrics['Strn']:<9.2e} | {metrics['Engy']:<9.2e} | {metrics['Freq']:<9.2e} | {metrics['Mode']:<9.2e} | {metrics['f1_hz']:<6.2f} | {gnorm:<9.2e}")

            # Early Stopping Logic (Explicit)
            if use_early_stopping:
                if val < best_loss - early_stop_tol:
                    best_loss = val
                    wait = 0
                    # self.optimized_params already updated at lines 383-385
                else:
                    wait += 1
                    if wait >= (early_stop_patience or 30):
                        print(f"✔ Early stopping triggered at iteration {i}. Convergence reached.")
                        break
            # Final physical params are already in self.optimized_params from lines 383-385

        return self.optimized_params

    def verify(self):
        print("\n" + "="*70)
        print(" [STAGE 4] COMPREHENSIVE FINAL VERIFICATION")
        print("="*70)
        
        # 1. Setup High Res Verification
        Nx_h, Ny_h = self.resolution_high
        xh = np.linspace(0, self.fem.Lx, Nx_h+1)
        yh = np.linspace(0, self.fem.Ly, Ny_h+1)
        Xh, Yh = np.meshgrid(xh, yh, indexing='ij')
        pts_high = np.column_stack([Xh.flatten(), Yh.flatten()])
        
        # Low Res Coords for Interpolation
        xl = np.linspace(0, self.fem.Lx, self.fem.nx+1)
        yl = np.linspace(0, self.fem.Ly, self.fem.ny+1)
        Xl, Yl = np.meshgrid(xl, yl, indexing='ij')
        pts_low = np.column_stack([Xl.flatten(), Yl.flatten()])
        
        # Interpolate Optimized Params
        opt_params_h = {
            't': griddata(pts_low, np.array(self.optimized_params['t']).flatten(), (Xh, Yh), method='cubic'),
            'rho': griddata(pts_low, np.array(self.optimized_params['rho']).flatten(), (Xh, Yh), method='cubic'),
            'E': griddata(pts_low, np.array(self.optimized_params['E']).flatten(), (Xh, Yh), method='cubic'),
            
            'z': griddata(pts_low, np.array(self.optimized_params.get('pz', jnp.zeros_like(self.optimized_params['t']))).flatten(), (Xh, Yh), method='cubic')
        }
        
        # 2. Assemble and Solve Optimized Model at High Res
        print("Solving Optimized High-Res Model for All Cases...")
        K_opt, M_opt = self.fem_high.assemble(opt_params_h)
        
        # 3. Static Performance Analysis & Matplotlib Plotting
        static_summary = []
        for i, case in enumerate(self.cases):
            tgt = self.targets[i]
            print(f" -> Verifying Case: {case.name}")
            
            # Solve Optimized
            fd, fv, F = case.get_bcs(self.fem_high)
            all_dofs = np.arange(self.fem_high.total_dof)
            free = np.setdiff1d(all_dofs, fd)
            u_opt = self.fem_high.solve_static_partitioned(K_opt, F, jnp.array(free), fd, fv)
            
            # Extract Fields
            w_ref = tgt['u_static'].reshape(Nx_h+1, Ny_h+1)
            w_opt = u_opt[2::6].reshape(Nx_h+1, Ny_h+1)
            
            stress_ref = tgt['max_surface_stress'].reshape(Nx_h+1, Ny_h+1)
            stress_opt = self.fem_high.compute_max_surface_stress(u_opt, opt_params_h).reshape(Nx_h+1, Ny_h+1)
            
            strain_ref = tgt['max_surface_strain'].reshape(Nx_h+1, Ny_h+1)
            strain_opt = self.fem_high.compute_max_surface_strain(u_opt, opt_params_h).reshape(Nx_h+1, Ny_h+1)

            # Calculation of Metric: Similarity % = 100 * (1 - RMSE / range)
            def get_metrics(ref, opt):
                rmse = np.sqrt(np.mean((ref - opt)**2))
                drange = np.max(ref) - np.min(ref) + 1e-12
                sim = max(0.0, 100.0 * (1.0 - rmse/drange))
                return rmse, sim

            metrics_w = get_metrics(w_ref, w_opt)
            metrics_s = get_metrics(stress_ref, stress_opt)
            metrics_e = get_metrics(strain_ref, strain_opt)
            
            static_summary.append({
                'name': case.name,
                'disp_sim': metrics_w[1],
                'stress_sim': metrics_s[1],
                'strain_sim': metrics_e[1]
            })

            # Create 3x3 Plot
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            fig.suptitle(f"Verification: {case.name}\n(High Resolution {Nx_h}x{Ny_h})", fontsize=16)
            
            # Plot Helper
            def plot_field(ax, data, title, cmap='jet', levels=None):
                if levels is None:
                    im = ax.contourf(xh, yh, data.T, 30, cmap=cmap)
                else:
                    im = ax.contourf(xh, yh, data.T, levels=levels, cmap=cmap, extend='both')
                ax.set_title(title)
                ax.set_aspect('equal')
                plt.colorbar(im, ax=ax)

            # Disp
            levels_w = np.linspace(np.min(w_ref), np.max(w_ref), 30)
            plot_field(axes[0,0], w_ref, "Target Disp (mm)", levels=levels_w, cmap='jet')
            plot_field(axes[0,1], w_opt, "Optimized Disp (mm)", levels=levels_w, cmap='jet')
            plot_field(axes[0,2], np.abs(w_opt - w_ref), "Error (mm)", cmap='YlOrRd')
            
            # Stress
            levels_s = np.linspace(0, np.max(stress_ref), 30)
            plot_field(axes[1,0], stress_ref, "Target Stress (MPa)", levels=levels_s, cmap='jet')
            plot_field(axes[1,1], stress_opt, "Optimized Stress (MPa)", levels=levels_s, cmap='jet')
            plot_field(axes[1,2], np.abs(stress_opt - stress_ref), "Error (MPa)", cmap='YlOrRd')
            
            # Strain
            levels_e = np.linspace(0, np.max(strain_ref)*1000, 30)
            plot_field(axes[2,0], strain_ref*1000, "Target Strain (e-3)", levels=levels_e, cmap='jet')
            plot_field(axes[2,1], strain_opt*1000, "Optimized Strain (e-3)", levels=levels_e, cmap='jet')
            plot_field(axes[2,2], np.abs(strain_opt - strain_ref)*1000, "Error (e-3)", cmap='YlOrRd')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_file = f"verify_3d_{case.name}.png"
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"   -> Saved: {out_file}")

        # 4. Parameter Comparison Plot
        print("Generating Parameter comparison plots...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        t_ref = self.target_params_high['t'].reshape(Nx_h+1, Ny_h+1)
        t_opt = opt_params_h['t'].reshape(Nx_h+1, Ny_h+1)
        
        im0 = axes[0,0].contourf(xh, yh, t_ref.T, 30, cmap='viridis')
        axes[0,0].set_title("Target Hard-coded Thickness (mm)")
        plt.colorbar(im0, ax=axes[0,0])
        
        im1 = axes[0,1].contourf(xh, yh, t_opt.T, 30, cmap='viridis')
        axes[0,1].set_title("Optimized Equivalent Thickness (mm)")
        plt.colorbar(im1, ax=axes[0,1])
        
        # Difference
        im2 = axes[1,0].contourf(xh, yh, (t_opt - t_ref).T, 30, cmap='RdBu_r')
        axes[1,0].set_title("Thickness Difference (mm)")
        plt.colorbar(im2, ax=axes[1,0])
        
        # Z-Shape (constant in this script)
        im3 = axes[1,1].contourf(xh, yh, opt_params_h['z'].reshape(Nx_h+1, Ny_h+1).T, 30, cmap='terrain')
        axes[1,1].set_title("Topography (Z-Height)")
        plt.colorbar(im3, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig("verify_3d_parameters.png", dpi=150)
        plt.close()
        
        # 5. Modal Verification
        print("Verifying Modal Response...")
        n_modes = len(self.target_eigen['vals'])
        vals_opt, vecs_opt = self.fem_high.solve_eigen(K_opt, M_opt, num_modes=n_modes + 10)
        
        # Filter Rigid Body (> 0.01 Hz)
        freq_opt_all = np.sqrt(np.maximum(np.array(vals_opt), 0.0)) / (2*np.pi)
        valid_opt = np.where(freq_opt_all > 0.01)[0]
        s_idx_opt = valid_opt[0] if len(valid_opt) > 0 else 6
        
        freq_ref = np.sqrt(np.abs(np.array(self.target_eigen['vals']))) / (2*np.pi)
        freq_opt = freq_opt_all[s_idx_opt : s_idx_opt + n_modes]
        
        modes_ref = np.array(self.target_eigen['modes'])
        modes_opt = np.array(vecs_opt[2::6, s_idx_opt : s_idx_opt + n_modes])
        
        macs = []
        for j in range(n_modes):
            v1 = modes_opt[:, j] / np.linalg.norm(modes_opt[:, j])
            v2 = modes_ref[:, j] / np.linalg.norm(modes_ref[:, j])
            macs.append(float((np.dot(v1, v2))**2))
            
        # Modal Plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(np.arange(n_modes)-0.2, freq_ref, width=0.4, label='Target')
        plt.bar(np.arange(n_modes)+0.2, freq_opt, width=0.4, label='Optimized')
        plt.title("Frequency Comparison (Hz)")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(n_modes), macs, color='purple')
        plt.axhline(0.9, color='red', linestyle='--')
        plt.title("Modal Assurance Criterion (MAC)")
        plt.ylim(0, 1.1)
        plt.savefig("verify_3d_modes.png", dpi=150)
        plt.close()
        
        # 5.5 Mode Shape Visual Comparison Plot
        print("Generating Mode Shape contour comparison plots...")
        n_plot_modes = min(3, n_modes)
        fig, axes = plt.subplots(n_plot_modes, 2, figsize=(10, 4 * n_plot_modes))
        if n_plot_modes == 1: axes = np.expand_dims(axes, axis=0) # Handle single mode case

        for j in range(n_plot_modes):
            # Reshape to 2D
            phi_ref = modes_ref[:, j].reshape(Nx_h+1, Ny_h+1)
            phi_opt = modes_opt[:, j].reshape(Nx_h+1, Ny_h+1)
            
            # Normalize and handle sign flip (eigenvectors are sign-invariant)
            phi_ref = phi_ref / (np.max(np.abs(phi_ref)) + 1e-12)
            phi_opt = phi_opt / (np.max(np.abs(phi_opt)) + 1e-12)
            
            # Flip sign if negatively correlated
            if np.sum(phi_ref * phi_opt) < 0:
                phi_opt = -phi_opt
                
            # Plot
            im0 = axes[j,0].contourf(xh, yh, phi_ref.T, 30, cmap='RdBu_r', levels=np.linspace(-1, 1, 31))
            axes[j,0].set_title(f"Target Mode {j+1} ({freq_ref[j]:.2f} Hz)")
            axes[j,0].set_aspect('equal')
            plt.colorbar(im0, ax=axes[j,0])
            
            im1 = axes[j,1].contourf(xh, yh, phi_opt.T, 30, cmap='RdBu_r', levels=np.linspace(-1, 1, 31))
            axes[j,1].set_title(f"Optimized Mode {j+1} ({freq_opt[j]:.2f} Hz, MAC: {macs[j]:.3f})")
            axes[j,1].set_aspect('equal')
            plt.colorbar(im1, ax=axes[j,1])
            
        plt.tight_layout()
        plt.savefig("verify_mode_shapes.png", dpi=150)
        plt.close()
        print(f"   -> Saved: verify_mode_shapes.png")
        
        # 6. Report Generation
        report = []
        report.append("# Equivalent Sheet Optimization Verification Report")
        report.append(f"Model Size: {self.fem.Lx} x {self.fem.Ly} mm")
        report.append(f"Target Resolution: {Nx_h} x {Ny_h}")
        report.append(f"Optimizer Resolution: {self.fem.nx} x {self.fem.ny}\n")
        
        report.append("## 1. Static Results Similarity")
        report.append("| Case | Disp Similarity (%) | Stress Similarity (%) | Strain Similarity (%) |")
        report.append("| :--- | :---: | :---: | :---: |")
        for res in static_summary:
            report.append(f"| {res['name']} | {res['disp_sim']:.2f}% | {res['stress_sim']:.2f}% | {res['strain_sim']:.2f}% |")
        
        report.append("\n## 2. Modal Results")
        report.append("| Mode | Target Freq (Hz) | Opt Freq (Hz) | Error (%) | MAC |")
        report.append("| :--- | :---: | :---: | :---: | :---: |")
        for j in range(n_modes):
            err = abs(freq_opt[j] - freq_ref[j])/freq_ref[j]*100
            report.append(f"| {j+1} | {freq_ref[j]:.2f} | {freq_opt[j]:.2f} | {err:.2f}% | {macs[j]:.4f} |")
        
        with open("verification_report.md", "w", encoding='utf-8') as f:
            f.write("\n".join(report))
        print("\n✔ Verification report saved to: verification_report.md")
        print("✔ Verification plots saved: verify_3d_*.png")

        # 7. Final Interactive Stage (PyVista) - Optional but good for inspection
        opt_eigen = {'vals': vals_opt[6:6+n_modes], 'modes': vecs_opt[2::6, 6:6+n_modes]}
        stage3_visualize_comparison(
            self.fem_high, self.targets, opt_params_h, self.target_params_high,
            opt_eigen=opt_eigen, tgt_eigen=self.target_eigen
        )

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
        'pattern_pz': 'TNY',       # 'bead_pz': {'T': 2.0, 'N': 1.0, 'Y': -2.0},
        'base_rho': 7.85e-9,        'base_E': 210000.0,
    }
    
    model.generate_targets(resolution_high=(50, 20), num_modes_save=5, target_config=target_config)
    
    # 4. Optimization Search Space (opt_config)
    opt_config = {
        't':   {'opt': True, 'init': 1.2,      'min': 0.5,    'max': 5.0},
        'rho': {'opt': True, 'init': 7.85e-9,   'min': 5e-9,   'max': 1e-8},
        'E':   {'opt': True, 'init': 200000.0,  'min': 50000,  'max': 300000},
        'pz':   {'opt': True, 'init': 0.0,  'min': -5.0,  'max': 5.0},
    }
    
    # 5. Full Loss Weights (as previously defined)
    weights = {
        'static': 1.0,           # Displacement matching
        'freq': 1.0,            # [Very Low] frequency matching
        'mode': 0.01,           # [Gentle] Mode matching (MAC) - Start small to avoid shock
        'curvature': 0.0,        # [Legacy]
        'moment': 0.0,           # [Legacy]
        'strain_energy': 2.0,    # Strain energy density matching
        'surface_stress': 2.0,   # Surface stress matching
        'surface_strain': 2.0,   # Surface strain matching
        'mass': 0.0,            # Mass constraint
        'reg': 0.001             # Regularization
    }
    
    # Run Optimization
    # Enable all metrics for comprehensive tracking
    try:
        model.optimize(opt_config, weights, use_smoothing=False, 
                       use_strain_energy=True, use_surface_stress=True, use_surface_strain=True)
        
        # 6. Verify and Report
        model.verify()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[CRITICAL ERROR] Optimization failed: {e}")
