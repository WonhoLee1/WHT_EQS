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
import msvcrt
import datetime

# Import our modular components
from solver import PlateFEM
from WHT_EQS_pattern_generator import get_thickness_field, get_z_field
from WHT_EQS_load_cases import TwistCase, PureBendingCase, CornerLiftCase, TwoCornerLiftCase
from WHT_EQS_visualization import (
    stage1_visualize_patterns,
    stage2_visualize_ground_truth,
    stage3_visualize_comparison
)

# Mesh Settings
Lx, Ly = 1000.0, 400.0
Nx_high, Ny_high = 25, 10      # High-res ground truth
Nx_low, Ny_low = 25, 10        # Low-res optimization mesh

class EquivalentSheetModel:
    def __init__(self, Lx, Ly, nx, ny):
        self.fem = PlateFEM(Lx, Ly, nx, ny)
        self.cases = []
        self.targets = []
        self.resolution_high = (50, 20)
        self.target_mass = 0.0
        self.target_start_idx = 6 # Default for Free-Free
        self.optimized_params = None
        self.target_params_high = None

    def add_case(self, case):
        self.cases.append(case)

    def generate_targets(self, resolution_high=(50, 20), 
                        num_modes_save=5, 
                        target_config=None
                        ):
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
            pattern_str=target_config.get('pattern', '   '),
            base_t=target_config.get('base_t', 1.0),
            bead_t=target_config.get('bead_t', 1.0)
        )
        z_h = get_z_field(
            Xh, Yh, Lx=self.fem.Lx, Ly=self.fem.Ly, 
            pattern_pz=target_config.get('pattern_pz', '   '),
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
            
            # Compute Full Reaction Forces
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
            
        # --- NEW: Generate Summary Ground Truth Plot ---
        print("Generating Ground Truth Summary Plot (3xN)...")
        n_cases = len(self.cases)
        plt.rcParams.update({'font.size': 8})
        fig, axes = plt.subplots(3, n_cases, figsize=(4*n_cases, 10), squeeze=False)
        fig.suptitle(f"Ground Truth Analysis Summary (Resolution: {Nx_h}x{Ny_h})\nRows: Disp, Stress, Strain | Colormap: jet", fontsize=10)
        
        xh, yh = np.linspace(0, self.fem.Lx, Nx_h+1), np.linspace(0, self.fem.Ly, Ny_h+1)
        
        for i, target in enumerate(self.targets):
            # Row 0: Displacement
            data_w = target['u_static'].reshape(Nx_h+1, Ny_h+1)
            im0 = axes[0, i].contourf(xh, yh, data_w.T, 30, cmap='jet')
            axes[0, i].set_title(f"Case: {target['case_name']}\nMax Disp: {np.max(np.abs(data_w)):.3f}mm", fontsize=8)
            axes[0, i].set_aspect('equal')
            plt.colorbar(im0, ax=axes[0, i], shrink=0.7)
            
            # Row 1: Max Surface Stress
            data_s = target['max_surface_stress'].reshape(Nx_h+1, Ny_h+1)
            im1 = axes[1, i].contourf(xh, yh, data_s.T, 30, cmap='jet')
            axes[1, i].set_title(f"Max Stress: {np.max(data_s):.2f} MPa", fontsize=8)
            axes[1, i].set_aspect('equal')
            plt.colorbar(im1, ax=axes[1, i], shrink=0.7)
            
            # Row 2: Max Surface Strain
            data_e = target['max_surface_strain'].reshape(Nx_h+1, Ny_h+1) * 1000 # to microstrain/e-3
            im2 = axes[2, i].contourf(xh, yh, data_e.T, 30, cmap='jet')
            axes[2, i].set_title(f"Max Strain: {np.max(data_e):.3f} e-3", fontsize=8)
            axes[2, i].set_aspect('equal')
            plt.colorbar(im2, ax=axes[2, i], shrink=0.7)
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("ground_truth_3d_loadcases.png", dpi=150)
        plt.close()
        print(" -> Saved: ground_truth_3d_loadcases.png")
        
        # 4. Solve Eigenmodes
        print("Solving Target Eigenmodes...")
        vals, vecs = fem_high.solve_eigen(K_h, M_h, num_modes=num_modes_save + 10)
        
        # Smart Elastic Mode Detection (Log-Gap Method)
        # Using logarithmic jump detection to robustly find the first true elastic mode.
        all_freqs_h = np.sqrt(np.maximum(np.array(vals), 0.0)) / (2 * np.pi)
        log_f = np.log(np.maximum(all_freqs_h[:15], 1e-6))
        gaps = np.concatenate([[0], log_f[1:] - log_f[:-1]])
        
        # A mode is elastic if it's > 1Hz and follows a major jump, or if it's clearly high freq (>20Hz)
        is_elastic = (all_freqs_h > 1.0) & (gaps > 1.0) | (all_freqs_h > 20.0)
        start_idx = np.argmax(is_elastic) if np.any(is_elastic) else (6 if len(all_freqs_h) > 6 else 0)
        
        print(f" -> Found first elastic mode at index {start_idx} ({all_freqs_h[start_idx]:.2f} Hz)")

        self.target_start_idx = start_idx
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
                 early_stop_patience=100, 
                 early_stop_tol=1e-8,
                 learning_rate=0.01,
                 num_modes_loss=None,
                 min_bead_width=10.0):
                 
        print("\n" + "="*70)
        print(" [STAGE 3] ADVANCED OPTIMIZATION (EXPLICIT LOGIC)")
        print(f" -> Setting: Min Bead Width = {min_bead_width} mm")
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
            sed_l = griddata(pts_h, tgt['strain_energy_density'], pts_l, method='cubic')
            
            self.targets_low.append({
                'u_static': jnp.array(u_l),
                'max_stress': jnp.array(stress_l),
                'max_strain': jnp.array(strain_l),
                'strain_energy_density': jnp.array(sed_l),
                'weight': tgt.get('weight', 1.0)
            })
            
        n_loss = num_modes_loss if num_modes_loss is not None else 5
        t_vals = self.target_eigen['vals'][:n_loss]
        t_modes_l = [griddata(pts_h, self.target_eigen['modes'][:, i], pts_l, method='cubic') for i in range(len(t_vals))]
        t_modes_l = jnp.stack(t_modes_l, axis=1)

        # 2. Optimization Parameters Setup
        self.scaling = {}
        for k in ['t', 'rho', 'E', 'pz']:
            cfg = opt_config.get(k, {})
            val = cfg.get('init', 0.0); self.scaling[k] = val if abs(val) > 1e-15 else 1.0
        
        full_params_scaled = {}
        for k in ['t', 'rho', 'E', 'pz']:
            if k in opt_config:
                init_phys = opt_config[k].get('init', 1.0 if k == 't' else 0.0)
                full_params_scaled[k] = jnp.full((Nx_l+1, Ny_l+1), init_phys / self.scaling[k])
        
        key = jax.random.PRNGKey(42); key, k1, k2 = jax.random.split(key, 3)
        # ONLY add jitter to parameters that are actually being optimized
        if 't' in full_params_scaled and opt_config.get('t', {}).get('opt', False): 
            full_params_scaled['t'] += 1e-4 * jax.random.uniform(k1, full_params_scaled['t'].shape)
        if 'pz' in full_params_scaled and opt_config.get('pz', {}).get('opt', False):
            full_params_scaled['pz'] += 1e-4 * jax.random.uniform(k2, full_params_scaled['pz'].shape)

        params = {k: v for k, v in full_params_scaled.items() if opt_config.get(k, {}).get('opt', True)}
        fixed_params_scaled = {k: v for k, v in full_params_scaled.items() if not opt_config.get(k, {}).get('opt', True)}
        
        # Re-applying robust auto-rate scheduling (Warmup + Cosine Decay)
        warmup_steps = max(5, max_iterations // 20)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate / 5.0,  # Start at 20% of peak for immediate progress
            peak_value=learning_rate,        # User-defined max LR
            warmup_steps=warmup_steps,
            decay_steps=max_iterations,
            end_value=learning_rate * 0.01   # Final fine-tuning phase
        )

        # --- Pre-compute Topography Filter Kernel ---
        filter_kernel = None
        if min_bead_width > 0:
            dx_l, dy_l = self.fem.Lx/Nx_l, self.fem.Ly/Ny_l
            rx, ry = int(np.ceil(min_bead_width / (2*dx_l))), int(np.ceil(min_bead_width / (2*dy_l)))
            kx, ky = np.linspace(-rx, rx, 2*rx+1), np.linspace(-ry, ry, 2*ry+1)
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            dist = np.sqrt((KX*dx_l)**2 + (KY*dy_l)**2)
            kernel = np.maximum(0, (min_bead_width/2.0) - dist)
            filter_kernel = jnp.array(kernel / np.sum(kernel))

        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr_schedule))
        opt_state = optimizer.init(params)

        case_bcs = []
        for case in self.cases:
            fd, fv, F = case.get_bcs(self.fem)
            case_bcs.append({'fd': fd, 'fv': fv, 'F': F, 'free': jnp.setdiff1d(jnp.arange(self.fem.total_dof), fd)})

        @jax.jit
        def loss_fn(p_scaled, fixed_p_scaled, scales):
            combined_phys = {k: v * scales[k] for k, v in p_scaled.items()}
            for k, v in fixed_p_scaled.items(): combined_phys[k] = v * scales[k]
            
            if 'pz' in combined_phys:
                if filter_kernel is not None:
                    combined_phys['z'] = jax.scipy.signal.convolve2d(combined_phys['pz'], filter_kernel, mode='same')
                else:
                    combined_phys['z'] = combined_phys['pz']
            
            K, M = self.fem.assemble(combined_phys); p_actual = combined_phys
            l_static, l_stress, l_strain, l_energy, l_freq, l_mode, l_mass, l_reg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            for i, case in enumerate(self.cases):
                b = case_bcs[i]
                u = self.fem.solve_static_partitioned(K, b['F'], b['free'], b['fd'], b['fv'])
                w = u[2::6]
                
                # Displacement: Normalized by target mean
                delta, scale = w - self.targets_low[i]['u_static'], jnp.mean(jnp.abs(self.targets_low[i]['u_static'])) + 1e-6
                l_static += jnp.mean((delta / scale)**2) * self.targets_low[i]['weight']
                
                if use_surface_stress:
                    delta, scale = self.fem.compute_max_surface_stress(u, p_actual) - self.targets_low[i]['max_stress'], jnp.mean(jnp.abs(self.targets_low[i]['max_stress'])) + 1e-6
                    l_stress += jnp.mean((delta / scale)**2)
                if use_surface_strain:
                    delta, scale = self.fem.compute_max_surface_strain(u, p_actual) - self.targets_low[i]['max_strain'], jnp.mean(jnp.abs(self.targets_low[i]['max_strain'])) + 1e-6
                    l_strain += jnp.mean((delta / scale)**2)
                if use_strain_energy:
                    delta, scale = self.fem.compute_strain_energy_density(u, p_actual) - self.targets_low[i]['strain_energy_density'], jnp.mean(jnp.abs(self.targets_low[i]['strain_energy_density'])) + 1e-6
                    l_energy += jnp.mean((delta / scale)**2)

            n_cases = len(self.cases)
            l_static /= n_cases; l_stress /= n_cases; l_strain /= n_cases; l_energy /= n_cases

            vals, vecs = self.fem.solve_eigen(K, M, num_modes=len(t_vals)+10)
            freqs = jnp.sqrt(jnp.maximum(vals, 0.0)) / (2 * jnp.pi)
            
            # 1. Dynamic Elastic Mode Selection (Physically Consistent)
            # Find the actual first elastic mode index in the current iteration
            # to handle cases where the model starts Flat (6 RBMs) and evolves to Topography (possibly fewer/different RBMs)
            log_f = jnp.log(jnp.maximum(freqs[:15], 1e-6))
            gaps = jnp.concatenate([jnp.zeros(1), log_f[1:] - log_f[:-1]])
            current_idx = jnp.argmax((freqs > 1.0) & (gaps > 1.0) | (freqs > 20.0))
            
            n_tgt = len(t_vals)
            search_window = 2  # Allow searching in n_tgt + 2 candidates to handle swaps
            
            # 2. Frequency Matching (Relative Error on Eigenvalues)
            # Match current elastic modes to target elastic modes
            curr_vals = jax.lax.dynamic_slice_in_dim(vals, current_idx, n_tgt)
            l_freq = jnp.mean((curr_vals / (t_vals + 1e-6) - 1.0)**2)
            
            # 3. Best-Fit MAC Matrix Matching (Mode Tracking)
            # Slice candidates starting from the detected elastic index
            cand_modes = jax.lax.dynamic_slice_in_dim(vecs[2::6, :], current_idx, n_tgt + search_window, axis=1)
            
            # Compute MAC Matrix [n_target, n_candidates]
            dots = jnp.dot(t_modes_l.T, cand_modes) # (n_tgt, n_tgt + search_window)
            norm_t = jnp.sum(t_modes_l**2, axis=0, keepdims=True) # (1, n_tgt)
            norm_c = jnp.sum(cand_modes**2, axis=0, keepdims=True) # (1, n_tgt + search_window)
            mac_matrix = (dots**2) / (norm_t.T @ norm_c + 1e-10)
            
            # For each target mode, find the highest MAC value in the candidate window
            best_mac_per_target = jnp.max(mac_matrix, axis=1)
            l_mode = jnp.mean(1.0 - best_mac_per_target)
            
            f1_hz = jnp.sqrt(jnp.maximum(vals[current_idx], 0.0)) / (2 * jnp.pi)

            l_mass = jnp.abs((jnp.sum(p_actual['t'] * p_actual['rho']) * (self.fem.Lx/Nx_l)*(self.fem.Ly/Ny_l) - self.target_mass) / self.target_mass) if use_mass_constraint else 0.0
            for k in ['t', 'pz']:
                if k in p_scaled: l_reg += (jnp.mean(jnp.diff(p_scaled[k], axis=0)**2) + jnp.mean(jnp.diff(p_scaled[k], axis=1)**2))

            total_loss = (l_static * loss_weights.get('static', 0.0) +
                          l_stress * loss_weights.get('surface_stress', 0.0) +
                          l_strain * loss_weights.get('surface_strain', 0.0) + 
                          l_energy * loss_weights.get('strain_energy', 0.0) +
                          l_freq   * loss_weights.get('freq', 0.0) +
                          l_mode   * loss_weights.get('mode', 0.0) + 
                          l_mass   * loss_weights.get('mass', 0.0) +
                          l_reg    * loss_weights.get('reg', 0.0))
                          
            return total_loss, {'Total': total_loss, 'Disp': l_static, 'Strs': l_stress, 'Strn': l_strain, 'Engy': l_energy, 'Freq': l_freq, 'Mode': l_mode, 'Mass': l_mass, 'Reg': l_reg, 'f1_hz': f1_hz}

        # Clear initial metrics logic
        print("Optimization Engine Ready. Initializing search...")

        best_loss, wait = float('inf'), 0
        best_params = params
        best_iter = 0
        self.history = [] # To store optimization trajectory
        
        print("\n [Tip] Press 'q' to stop optimization and use the best result found so far.\n")
        print(f"{'Iter':<5} | {'Total_Norm':<10} | {'Disp':<9} | {'Strs':<9} | {'Strn':<9} | {'Engy':<9} | {'Freq':<9} | {'Mode':<9} | {'Mass':<9} | {'Freq1':<6}")
        print("-" * 117)

        for i in range(max_iterations):
            (val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, fixed_params_scaled, self.scaling)
            
            # Store history (on CPU as regular floats)
            self.history.append({k: float(v) for k, v in metrics.items()})
            
            updates, opt_state = optimizer.update(jax.tree_util.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads), opt_state)
            params = optax.apply_updates(params, updates)
            
            # --- Record Parameter Statistics ---
            current_phys = {k: v * self.scaling[k] for k, v in params.items()}
            current_phys.update({k: v * self.scaling[k] for k, v in fixed_params_scaled.items()})
            
            p_stats = {}
            for pk in ['t', 'rho', 'E', 'pz']:
                if pk in current_phys:
                    arr = current_phys[pk]
                    p_stats.update({
                        f'{pk}_mean': float(jnp.mean(arr)),
                        f'{pk}_std':  float(jnp.std(arr)),
                        f'{pk}_min':  float(jnp.min(arr)),
                        f'{pk}_max':  float(jnp.max(arr))
                    })
            self.history[-1].update(p_stats)
            
            # Parameter Clipping (Safety)
            for k in params:
                cfg = opt_config[k]
                params[k] = jnp.clip(params[k], cfg['min']/self.scaling[k], cfg['max']/self.scaling[k])

            # Check for 'q' key press to stop early
            if msvcrt.kbhit():
                char = msvcrt.getch()
                if char in [b'q', b'Q']:
                    print(f"\n [USER INTERRUPT] 'q' pressed. Terminating and reverting to best params (Iter {best_iter})...")
                    params = best_params
                    break

            if val < best_loss - early_stop_tol:
                best_loss, wait = val, 0
                best_params = params
                best_iter = i
            else:
                wait += 1
                if use_early_stopping and wait >= (early_stop_patience or 30):
                    print(f"\n [EARLY STOP] No improvement for {wait} iters. Reverting to best params (Iter {best_iter})...")
                    params = best_params
                    break

            if i % 10 == 0: print(f"{i:<5d} | {val:<10.3f} | {metrics['Disp']:<9.2e} | {metrics['Strs']:<9.2e} | {metrics['Strn']:<9.2e} | {metrics['Engy']:<9.2e} | {metrics['Freq']:<9.2e} | {metrics['Mode']:<9.2e} | {metrics['Mass']:<9.2e} | {metrics['f1_hz']:<6.2f}")

        # Final Assignment
        params = best_params
        self.optimized_params = {**{k: v * self.scaling[k] for k, v in params.items()}, **{k: v * self.scaling[k] for k, v in fixed_params_scaled.items()}}
        print(f"\n -> Optimization Finished. Best Result found at Iteration {best_iter} with Loss: {best_loss:.6f}")
        
        # Generate convergence plot automatically
        self.plot_optimization_history()
        
        return self.optimized_params

    def plot_optimization_history(self):
        """Generates a grid of line plots for each optimization metric to handle scale differences."""
        if not hasattr(self, 'history') or len(self.history) == 0:
            print(" -> No history data to plot.")
            return

        print("Generating Detailed Convergence History Grid (N x M)...")
        iters = np.arange(len(self.history))
        
        # Identify all keys to plot (excluding metadata/statistics)
        all_keys = ['Total', 'Disp', 'Strs', 'Strn', 'Engy', 'Freq', 'Mode', 'Mass', 'Reg']
        keys_to_plot = [k for k in all_keys if k in self.history[0]]
        
        n_plots = len(keys_to_plot)
        cols = 3
        rows = int(np.ceil(n_plots / cols))
        
        plt.figure(figsize=(15, 4 * rows))
        plt.rcParams.update({'font.size': 8})
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, n_plots))
        
        for idx, k in enumerate(keys_to_plot):
            plt.subplot(rows, cols, idx + 1)
            
            vals = np.array([h[k] for h in self.history])
            init_val = vals[0] if abs(vals[0]) > 1e-15 else 1.0
            relative_vals = vals / init_val
            
            plt.plot(iters, relative_vals, color=colors[idx], linewidth=1.5)
            plt.axhline(1.0, color='red', linestyle='--', alpha=0.3, linewidth=1) # Baseline
            
            # Use log scale only if the reduction is significant (more than 2 decades)
            if np.max(relative_vals) / (np.min(relative_vals) + 1e-15) > 100:
                plt.yscale('log')
            
            plt.title(f"{k} Convergence", fontsize=10, fontweight='bold')
            plt.xlabel("Iteration")
            plt.ylabel("Rel. Value (v/v0)")
            plt.grid(True, which="both", ls="-", alpha=0.2)
            
            # Print final improvement in the plot
            final_rel = relative_vals[-1]
            plt.text(0.95, 0.95, f"Final: {final_rel:.2e}", transform=plt.gca().transAxes, 
                     ha='right', va='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

        plt.suptitle("Optimization Convergence Metrics (Independent Scaling)", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig("verify_3d_opt_history.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(" -> Saved: verify_3d_opt_history.png")
        
        self.plot_parameter_evolution()

    def plot_parameter_evolution(self):
        """Generates a 2x2 grid plot showing the evolution of design variable statistics."""
        if not hasattr(self, 'history') or len(self.history) == 0: return

        print("Generating Parameter Evolution Plot (2x2)...")
        iters = np.arange(len(self.history))
        params_to_plot = [p for p in ['t', 'rho', 'E', 'pz'] if f'{p}_mean' in self.history[0]]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.rcParams.update({'font.size': 8})
        
        for idx, pk in enumerate(['t', 'rho', 'E', 'pz']):
            ax = axes[idx // 2, idx % 2]
            if pk not in params_to_plot:
                ax.text(0.5, 0.5, f"Variable '{pk}' not optimized/fixed", ha='center', va='center')
                continue
                
            means = np.array([h[f'{pk}_mean'] for h in self.history])
            stds  = np.array([h[f'{pk}_std'] for h in self.history])
            mins  = np.array([h[f'{pk}_min'] for h in self.history])
            maxs  = np.array([h[f'{pk}_max'] for h in self.history])
            
            # Left Axis: Mean, Min, Max
            ax.plot(iters, means, 'b-', label='Mean', linewidth=2)
            ax.fill_between(iters, mins, maxs, color='blue', alpha=0.1, label='Min-Max Range')
            ax.set_xlabel("Iteration")
            ax.set_ylabel(f"{pk} Value", color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            ax.grid(True, alpha=0.3)
            
            # Right Axis: Standard Deviation (Fluctuation/Heterogeneity)
            ax2 = ax.twinx()
            ax2.plot(iters, stds, 'r--', label='Std Dev', alpha=0.7)
            ax2.set_ylabel("Std Dev (Heterogeneity)", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax.set_title(f"Evolution of {pk.upper()}", fontsize=10, fontweight='bold')
            
            # Legend handling
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=7)

        plt.tight_layout()
        plt.savefig("verify_3d_param_evolution.png", dpi=150)
        plt.close()
        print(" -> Saved: verify_3d_param_evolution.png")

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
        plt.rcParams.update({'font.size': 8})
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        t_ref = self.target_params_high['t'].reshape(Nx_h+1, Ny_h+1)
        z_ref = self.target_params_high['z'].reshape(Nx_h+1, Ny_h+1)
        t_opt = opt_params_h['t'].reshape(Nx_h+1, Ny_h+1)
        z_opt = opt_params_h['z'].reshape(Nx_h+1, Ny_h+1)
        
        # Calculate comparison metrics (Safe correlation handles constant arrays)
        def safe_corr(a, b):
            a_flat, b_flat = a.flatten(), b.flatten()
            if np.std(a_flat) < 1e-12 or np.std(b_flat) < 1e-12:
                return 1.0 if np.allclose(a_flat, b_flat, atol=1e-5) else 0.0
            return np.corrcoef(a_flat, b_flat)[0,1]

        t_rmse = np.sqrt(np.mean((t_ref - t_opt)**2))
        t_corr = safe_corr(t_ref, t_opt)
        z_rmse = np.sqrt(np.mean((z_ref - z_opt)**2))
        z_corr = safe_corr(z_ref, z_opt)
        
        fig.suptitle(f"Geometric Parameter Verification (Resolution: {Nx_h}x{Ny_h})\n"
                     f"Thickness: RMSE={t_rmse:.4f}, Corr={t_corr:.4f} | Topography: RMSE={z_rmse:.4f}, Corr={z_corr:.4f}", 
                     fontsize=10, y=0.98)

        def plot_param(ax, data, title, cmap='jet'):
            im = ax.contourf(xh, yh, data.T, 30, cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax, shrink=0.8)

        # Row 1: Thickness
        plot_param(axes[0,0], t_ref, "Target Thickness (mm)")
        plot_param(axes[0,1], t_opt, "Optimized Thickness (mm)")
        
        # Row 2: Topography (Z)
        plot_param(axes[1,0], z_ref, "Target Topography (Z)")
        plot_param(axes[1,1], z_opt, "Optimized Topography (Z)")
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig("verify_3d_parameters.png", dpi=150)
        plt.close()
        print(f"   -> Saved: verify_3d_parameters.png")
        
        # 5. Modal Verification
        print("Verifying Modal Response...")
        n_modes = len(self.target_eigen['vals'])
        vals_opt, vecs_opt = self.fem_high.solve_eigen(K_opt, M_opt, num_modes=n_modes + 10)
        
        # Robust Elastic Mode detection using Log-Gap
        freq_opt_all = np.sqrt(np.maximum(np.array(vals_opt), 0.0)) / (2*np.pi)
        log_f = np.log(np.maximum(freq_opt_all[:15], 1e-6))
        gaps = np.concatenate([[0], log_f[1:] - log_f[:-1]])
        s_idx_opt = np.argmax((freq_opt_all > 1.0) & (gaps > 1.0) | (freq_opt_all > 20.0))
        
        print(f" -> Verification: Found first elastic mode at index {s_idx_opt} ({freq_opt_all[s_idx_opt]:.2f} Hz)")
        
        freq_ref = np.sqrt(np.abs(np.array(self.target_eigen['vals']))) / (2*np.pi)
        freq_opt = freq_opt_all[s_idx_opt : s_idx_opt + n_modes]
        
        modes_ref = np.array(self.target_eigen['modes'])
        modes_opt = np.array(vecs_opt[2::6, s_idx_opt : s_idx_opt + n_modes])
        
        macs = []
        for j in range(n_modes):
            v1 = modes_opt[:, j] / (np.linalg.norm(modes_opt[:, j]) + 1e-12)
            v2 = modes_ref[:, j] / (np.linalg.norm(modes_ref[:, j]) + 1e-12)
            macs.append(float((np.dot(v1, v2))**2))
            
        # Modal Bar Plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(np.arange(n_modes)-0.2, freq_ref, width=0.4, label='Target', color='royalblue')
        plt.bar(np.arange(n_modes)+0.2, freq_opt, width=0.4, label='Optimized', color='orange')
        plt.title("Frequency Comparison (Hz)", fontsize=10)
        plt.legend()
        plt.xticks(np.arange(n_modes), np.arange(1, n_modes+1))
        
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(n_modes), macs, color='purple')
        plt.axhline(0.9, color='red', linestyle='--', alpha=0.5)
        plt.title("Modal Assurance Criterion (MAC)", fontsize=10)
        plt.ylim(0, 1.1)
        plt.xticks(np.arange(n_modes), np.arange(1, n_modes+1))
        
        plt.tight_layout()
        plt.savefig("verify_3d_modes.png", dpi=150)
        plt.close()
        
        # 5.5 Mode Shape Visual Comparison Plot
        print("Generating Mode Shape contour comparison plots...")
        n_plot_modes = n_modes # Plot all non-rigid modes as requested
        fig = plt.figure(figsize=(3.5 * n_plot_modes, 7))
        fig.suptitle(f"Mode Shape Comparison (Target vs Optimized)\nColormap: jet", fontsize=10)
        
        for j in range(n_plot_modes):
            # Normalization and Sign Alignment
            phi_ref = modes_ref[:, j].reshape(Nx_h+1, Ny_h+1)
            phi_ref = phi_ref / (np.max(np.abs(phi_ref)) + 1e-12)
            
            phi_opt = modes_opt[:, j].reshape(Nx_h+1, Ny_h+1)
            phi_opt = phi_opt / (np.max(np.abs(phi_opt)) + 1e-12)
            if np.sum(phi_ref * phi_opt) < 0:
                phi_opt = -phi_opt

            # Target (Top Row)
            ax_tgt = fig.add_subplot(2, n_plot_modes, j+1)
            im0 = ax_tgt.contourf(xh, yh, phi_ref.T, 30, cmap='jet', levels=np.linspace(-1, 1, 31))
            ax_tgt.set_title(f"Target Mode {j+1}\n({freq_ref[j]:.2f} Hz)", fontsize=8)
            ax_tgt.set_aspect('equal')
            if j == 0: # Only show colorbar in the first column
                cbar = plt.colorbar(im0, ax=ax_tgt, shrink=0.7)
                cbar.ax.set_ylabel('Amplitude', fontsize=8)
            
            # Optimized (Bottom Row)
            ax_opt = fig.add_subplot(2, n_plot_modes, n_plot_modes + j+1)
            im1 = ax_opt.contourf(xh, yh, phi_opt.T, 30, cmap='jet', levels=np.linspace(-1, 1, 31))
            ax_opt.set_title(f"Opt Mode {j+1}\n({freq_opt[j]:.2f} Hz, MAC: {macs[j]:.3f})", fontsize=8)
            ax_opt.set_aspect('equal')
            if j == 0: # Only show colorbar in the first column
                cbar = plt.colorbar(im1, ax=ax_opt, shrink=0.7)
                cbar.ax.set_ylabel('Amplitude', fontsize=8)
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("verify_3d_mode_shapes.png", dpi=150)
        plt.close()
        print(f"   -> Saved: verify_3d_mode_shapes.png")
        print(f"   -> Saved: verify_3d_mode_shapes.png")
        
        # 6. Comprehensive Report Generation
        print("Generating Enhanced Verification Report...")
        
        def calc_r2(ref, opt):
            ss_res = np.sum((ref - opt)**2)
            ss_tot = np.sum((ref - np.mean(ref))**2)
            return 1 - (ss_res / (ss_tot + 1e-12))

        def calc_mse(ref, opt):
            return np.mean((ref - opt)**2)

        # 6.1 Collect Case-by-Case Detailed Metrics
        case_results = []
        for i, case in enumerate(self.cases):
            tgt = self.targets[i]
            fd, fv, F_ext = case.get_bcs(self.fem_high)
            free = np.setdiff1d(np.arange(self.fem_high.total_dof), fd)
            u_opt = self.fem_high.solve_static_partitioned(K_opt, F_ext, jnp.array(free), fd, fv)
            u_ref = tgt['u_full']
            
            # 1. Displacement
            w_ref, w_opt = tgt['u_static'], u_opt[2::6]
            max_w_ref, max_w_opt = np.max(np.abs(w_ref)), np.max(np.abs(w_opt))
            
            # 2. Reaction Forces (at fixed DOFs)
            R_vec_ref = (np.array(K_opt @ u_ref) - F_ext)[2::6] # Simplified Z-component reactions
            R_vec_opt = (np.array(K_opt @ u_opt) - F_ext)[2::6]
            max_R_ref, max_R_opt = np.max(np.abs(tgt['reaction_full'][2::6])), np.max(np.abs(R_vec_opt))
            
            # 3. Bending Moments (Using compute_moment)
            M_ref_vec = self.fem_high.compute_moment(u_ref, self.target_params_high)
            M_opt_vec = self.fem_high.compute_moment(u_opt, opt_params_h)
            max_M_ref = np.max(np.sqrt(np.sum(M_ref_vec**2, axis=1))) # Max resultant moment magnitude
            max_M_opt = np.max(np.sqrt(np.sum(M_opt_vec**2, axis=1)))
            
            # Overall Accuracy for this case
            _, sim = get_metrics(w_ref, w_opt)
            r2 = calc_r2(w_ref, w_opt)
            mse = calc_mse(w_ref, w_opt)

            case_results.append({
                'name': case.name,
                'max_w': (max_w_ref, max_w_opt),
                'max_R': (max_R_ref, max_R_opt),
                'max_M': (max_M_ref, max_M_opt),
                'sim': sim, 'r2': r2, 'mse': mse
            })

        report = []
        report.append("# 📊 Professional Structural Optimization Verification Report")
        report.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Domain:** {self.fem.Lx}mm x {self.fem.Ly}mm | **Material:** E={target_config.get('base_E', 210000)}MPa, v=0.3")
        report.append(f"**Resolution:** Target({Nx_h}x{Ny_h}) vs. Optimized({self.fem.nx}x{self.fem.ny})\n")

        report.append("## 1. 🎯 Optimization Metric Guide")
        report.append("| Metric | Full Name | Physical Meaning | Target |")
        report.append("| :--- | :--- | :--- | :---: |")
        report.append("| **R²** | Coeff. of Determination | Statistical correlation (1.0 is perfect) | > 0.90 |")
        report.append("| **MAC** | Modal Assurance Criterion | Mode shape similarity (1.0 is identical) | > 0.85 |")
        report.append("| **Similarity** | Accuracy Index | Range-scaled error metric | > 90% |")
        report.append("\n")

        report.append("## 2. 🏗️ Static Response Comparison")
        report.append("Detailed comparison of peak structural responses across all load cases.")
        report.append("| Load Case | Metric | Target Result | Optimized Result | Error (%) | Status |")
        report.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
        
        for res in case_results:
            def row(label, ref, opt, unit, tol=5.0):
                err = abs(ref - opt) / (abs(ref) + 1e-12) * 100
                stat = "✔" if err < tol else "⚠"
                return f"| {res['name']:<10} | {label:<10} | {ref:10.3f} {unit} | {opt:10.3f} {unit} | {err:8.2f}% | {stat:^6} |"
            
            report.append(row("Max Disp", res['max_w'][0], res['max_w'][1], "mm"))
            report.append(row("Max Reac", res['max_R'][0], res['max_R'][1], "N"))
            report.append(row("Max Moment", res['max_M'][0], res['max_M'][1], "Nmm"))
            report.append("| " + "-"*10 + " | " + "-"*10 + " | " + "-"*12 + " | " + "-"*12 + " | " + "-"*10 + " | " + "-"*6 + " |")
        report.append("\n")

        report.append("## 3. 📈 Correlation Statistics")
        report.append("| Load Case | Similarity Index | R² (Disp) | MSE (Disp) | Result Status |")
        report.append("| :--- | :---: | :---: | :---: | :---: |")
        for res in case_results:
            status = "✔ EXCELLENT" if res['r2'] > 0.95 else ("OK" if res['r2'] > 0.85 else "❌ FAIL")
            report.append(f"| {res['name']:<10} | {res['sim']:15.2f}% | {res['r2']:10.4f} | {res['mse']:10.2e} | {status:^12} |")
        report.append("\n")

        report.append("## 4. 🎵 Dynamic Modal Performance")
        report.append("| Mode No. | Target Freq (Hz) | Opt Freq (Hz) | Error (%) | MAC Value | Status |")
        report.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
        for j in range(n_modes):
            err = abs(freq_opt[j] - freq_ref[j])/freq_ref[j]*100
            status = "✔ PASS" if macs[j] > 0.9 else "⚠ CHECK"
            report.append(f"| {j+1:^8} | {freq_ref[j]:15.2f} | {freq_opt[j]:12.2f} | {err:8.2f}% | {macs[j]:9.4f} | {status:^6} |")
        report.append("\n")

        report.append("## 5. 📐 Geometry Accuracy")
        report.append("| Parameter | RMSE | Correlation | Mean (Target) | Mean (Opt) |")
        report.append("| :--- | :---: | :---: | :---: | :---: |")
        report.append(f"| Thickness (t) | {t_rmse:8.4f} | {t_corr:11.4f} | {np.mean(t_ref):10.3f} | {np.mean(t_opt):8.3f} |")
        report.append(f"| Topography (z) | {z_rmse:8.4f} | {z_corr:11.4f} | {np.mean(z_ref):10.3f} | {np.mean(z_opt):8.3f} |")
        report.append("\n")

        report.append("---")
        report.append("*End of Automated Verification Report.*")

        with open("verification_report.md", "w", encoding='utf-8') as f:
            f.write("\n".join(report))
        print("\n✔ Comprehensive verification report saved: verification_report.md")
        print("✔ Verification plots saved: verify_3d_*.png")

        # 7. Final Interactive Stage (PyVista) - Optional but good for inspection
        opt_eigen = {'vals': vals_opt[s_idx_opt:s_idx_opt+n_modes], 'modes': vecs_opt[2::6, s_idx_opt:s_idx_opt+n_modes]}
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
    model.add_case(PureBendingCase("bend_x", axis='x', value=3.0, mode='angle', weight=1.0))
    model.add_case(CornerLiftCase("lift_br", corner='br', value=5.0, mode='disp', weight=1.0))
    model.add_case(CornerLiftCase("lift_tl", corner='tl', value=5.0, mode='disp', weight=1.0))
    model.add_case(TwoCornerLiftCase("lift_tl_br", corners=['br', 'tl'], value=5.0, mode='disp', weight=1.0))
    
    # 3. Ground Truth Generation (target_config)
    target_config = {
        'pattern': 'ABC',           'base_t': 1.0, 
        #'bead_t': {'A': 2.0, 'B': 2.5, 'C': 1.5},
        'pattern_pz': 'TNY',        'bead_pz': {'T': 1.0, 'N': 1.0, 'Y': 1.0},
        'base_rho': 7.85e-9,        'base_E': 210000.0,
    }
    
    model.generate_targets(resolution_high=(Nx_high, Ny_high), num_modes_save=5, target_config=target_config)
    
    # 4. Optimization Search Space (opt_config)
    opt_config = {
        't':   {'opt': False, 'init': 1.0,      'min': 0.1,    'max': 10.0},
        'rho': {'opt': False, 'init': 7.85e-9,   'min': 7.85e-10,   'max': 7.85e-8},
        'E':   {'opt': False, 'init': 210000.0,  'min': 210000,  'max': 300000.0},
        'pz':   {'opt': True, 'init': 0.0,  'min': -10.0,  'max': 10.0},
    }
    
    # 5. Full Loss Weights (as previously defined)
    weights = {
        'static': 1.0,           # Displacement matching
        'freq': 1.0,            # [Very Low] frequency matching
        'mode': 1.0,           # [Gentle] Mode matching (MAC) - Start small to avoid shock
        'curvature': 0.0,        # [Legacy]
        'moment': 0.0,           # [Legacy]
        'strain_energy': 1.0,    # Strain energy density matching
        'surface_stress': 1.0,   # Surface stress matching
        'surface_strain': 1.0,   # Surface strain matching
        'mass': 1.0,            # Mass constraint
        'reg': 0.01             # Regularization
    }
    
    # Run Optimization
    # Enable all metrics for comprehensive tracking
    # 참고: optax.warmup_cosine_decay_schedule 
    try:
        model.optimize(opt_config, weights, 
                       use_smoothing=False, 
                       use_strain_energy=True, 
                       use_surface_stress=True, 
                       use_surface_strain=True,                 
                       use_mass_constraint=True, 
                       mass_tolerance=0.05,
                       max_iterations=100, 
                       use_early_stopping=True, 
                       early_stop_patience=100, 
                       early_stop_tol=1e-8,
                       learning_rate=0.5,
                       num_modes_loss=5,
                       min_bead_width=50.0)
        
        # 6. Verify and Report
        model.verify()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[CRITICAL ERROR] Optimization failed: {e}")

# ==============================================================================
# [DEVELOPMENT NOTES & FUTURE CONSIDERATIONS]
# ==============================================================================
# The following features were implemented but removed in the 2026-02-12 update 
# due to convergence stability issues. They remains as candidates for future tuning:
#
# 1. Automatic Loss Normalization (safe_norm):
#    - Concept: Dividing each loss component by its initial value to auto-scale weights.
#    - Issue: Highly accurate initial metrics (e.g., matching frequencies) were 
#      over-weighted, preventing progress in high-error domains like displacement.
#
# 2. Log-scale Frequency Matching (jnp.log):
#    - Concept: Matching frequency order-of-magnitude.
#    - Issue: Less sensitive to precision matching (< 1Hz) compared to linear scale 
#      residual: (curr/tgt - 1)^2.
#
# 3. LR Warm-up Cosine Decay Schedule:
#    - Concept: Starting with very low LR to stabilize initial gradients.
#    - Issue: Delayed initial convergence for well-posed topography tasks. 
#      Linear decay from peak LR proved more efficient for current load cases.
# ==============================================================================
