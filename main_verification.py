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
import msvcrt # Windows Key Press Detection

# Import our modular components
from solver import PlateFEM
from WHT_EQS_pattern_generator import get_thickness_field, get_z_field
from WHT_EQS_load_cases import TwistCase, PureBendingCase, CornerLiftCase
from WHT_EQS_visualization import (
    stage1_visualize_patterns,
    stage2_visualize_ground_truth,
    stage3_visualize_comparison
)
import WHT_EQS_topo as topo
import WHT_EQS_mesh as mesh

# Mesh Settings
Lx, Ly = 1000.0, 400.0
Nx_high, Ny_high = 50, 20      # High-res ground truth
Nx_low, Ny_low = 30, 12        # Low-res optimization mesh

class EquivalentSheetModel:
    def __init__(self, Lx, Ly, nx, ny, solver_type='grid'):
        self.solver_type = solver_type
        if solver_type == 'grid':
            self.fem = PlateFEM(Lx, Ly, nx, ny)
        else:
            # For 'general' solver, we need an initial mesh. 
            # We'll create a default grid mesh and pass it to GeneralShellSolver.
            temp_fem = PlateFEM(Lx, Ly, nx, ny)
            self.fem = topo.GeneralShellSolver(temp_fem.node_coords, temp_fem.elements)
            # Inject grid attributes for curvature/stress calculation
            self.fem.nx, self.fem.ny = nx, ny
            self.fem.dx, self.fem.dy = Lx/nx, Ly/ny
            
        self.cases = []
        self.targets = []
        self.resolution_high = (50, 20)
        self.target_mass = 0.0
        self.optimized_params = None
        self.target_params_high = None

    def add_case(self, case):
        self.cases.append(case)

    def load_external_mesh(self, filepath, file_format='msh'):
        """
        Load an external mesh file and switch to 'general' solver.
        """
        print(f"\n [MESH] Loading External Mesh: {filepath} ({file_format})")
        if file_format == 'msh':
            nodes, elements = mesh.load_mesh_msh(filepath)
        elif file_format == 'f06':
            nodes, elements = mesh.load_mesh_f06(filepath)
        else:
            raise ValueError(f"Unsupported mesh format: {file_format}")
            
        if nodes is not None and elements is not None:
            self.solver_type = 'general'
            self.fem = topo.GeneralShellSolver(nodes, elements)
            print(f" -> Mesh Loaded: {len(nodes)} nodes, {len(elements)} elements. Switched to 'general' solver.")
        else:
            print(" ??Mesh loading failed. Keeping current solver.")

    def generate_targets(self, resolution_high=(50, 20), num_modes_save=5, target_config=None):
        print("\n" + "="*70)
        print(" [STAGE 1] TARGET GENERATION & PATTERN VERIFICATION")
        print("="*70)
        self.resolution_high = resolution_high
        Nx_h, Ny_h = resolution_high
        
        # 1. Create High-Resolution Mesh for "Ground Truth"
        if self.solver_type == 'general':
            # Create temporary high-res grid and convert to General Solver
            temp_fem = PlateFEM(self.fem.Lx, self.fem.Ly, Nx_h, Ny_h)
            node_coords_3d = jnp.column_stack([temp_fem.node_coords, jnp.zeros(temp_fem.num_nodes)])
            self.fem_high = topo.GeneralShellSolver(node_coords_3d, temp_fem.elements)
            # Inject grid attributes for high-res solver
            self.fem_high.nx, self.fem_high.ny = Nx_h, Ny_h
            self.fem_high.dx, self.fem_high.dy = self.fem.Lx/Nx_h, self.fem.Ly/Ny_h
        else:
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
        
        params_high = {
            't': t_h.flatten(), 
            'z': z_h.flatten(), 
            'rho': rho_h.flatten(), 
            'E': E_h.flatten()
        }
        self.target_params_high = params_high
        
        # Calculate target mass
        dx, dy = self.fem.Lx/Nx_h, self.fem.Ly/Ny_h
        self.target_mass = float(jnp.sum(t_h * rho_h) * dx * dy)
        
        # [STAGE 1] Interactive Pattern Check
        stage1_visualize_patterns(Nx_h, Ny_h, Xh, Yh, t_h, z_h)
        
        # 3. Solve FEM for each load case (High Fidelity)
        print("\nSolving High-Resolution Ground Truth...")
        
        # Ensure params match solver requirements (General solver needs flat keys)
        K_h, M_h = self.fem_high.assemble(params_high)
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
                 num_modes_loss=None,
                 auto_lr=False): # Default to False (Manual LR)
        """
        [理쒖쟻???ㅽ뻾 ?⑥닔]
        ?뚮씪誘명꽣 ?ㅻ챸:
        - opt_config: 理쒖쟻?????蹂???ㅼ젙 (t, rho, E, pz ?? 諛?踰붿쐞(min/max)
        - loss_weights: 媛?紐⑹쟻?⑥닔 ??ぉ蹂?媛以묒튂 (1.0 = ?쒖?, 5.0 = 以묒슂)
        
        - use_smoothing: ?됲솢??Total Variation) ?ъ슜 ?щ? (True 沅뚯옣)
        - use_strain_energy: 蹂???먮꼫吏 諛??留ㅼ묶 ?ъ슜 ?щ?
        - use_surface_stress: 理쒕? ?쒕㈃ ?묐젰 留ㅼ묶 ?ъ슜 ?щ? (以묒슂)
        - use_surface_strain: 理쒕? ?쒕㈃ 蹂?뺣쪧 留ㅼ묶 ?ъ슜 ?щ?
        - use_mass_constraint: 吏덈웾 ?쒖빟 議곌굔 ?ъ슜 ?щ? (True 沅뚯옣)
        - mass_tolerance: 吏덈웾 ?ㅼ감 ?덉슜 踰붿쐞 (鍮꾩쑉, ?? 0.05 = 5%)
        
        - max_iterations: 理쒕? 諛섎났 ?잛닔 (1000??沅뚯옣)
        - use_early_stopping: 議곌린 醫낅즺 湲곕뒫 ?ъ슜 ?щ? (True 沅뚯옣)
        - early_stop_patience: 議곌린 醫낅즺 ?湲??잛닔 (?? 100???숈븞 媛쒖꽑 ?놁쑝硫?以묐떒)
        - early_stop_tol: 議곌린 醫낅즺 ?덉슜 ?ㅼ감 (留ㅼ슦 ?묒? 媛?
        
        - learning_rate: ?섎룞 ?숈뒿瑜?(auto_lr=False?????ъ슜)
        - num_modes_loss: 紐⑤뱶 ?댁꽍???ъ슜??紐⑤뱶 媛쒖닔 (湲곕낯媛? None=5媛?
        - auto_lr: [?ㅻ쭏??湲곕뒫] ?쒖옉 ?숈뒿瑜??먮룞 ?쒕떇 ?ъ슜 ?щ?
        """
                 
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
        
        # Initial Parameters (Breaking Symmetry with small jitter)
        key = jax.random.PRNGKey(42)
        params = {
            't': jnp.full((Nx_l+1, Ny_l+1), opt_config['t'].get('init', 1.0)),
            'rho': jnp.full((Nx_l+1, Ny_l+1), opt_config['rho'].get('init', 7.5e-9)),
            'E': jnp.full((Nx_l+1, Ny_l+1), opt_config['E'].get('init', 210000.0))
        }
        # Add 'pz' (Topography) if configured, else assume 0
        if 'pz' in opt_config:
             params['pz'] = jnp.full((Nx_l+1, Ny_l+1), opt_config['pz'].get('init', 0.0))
        else:
             params['pz'] = jnp.zeros((Nx_l+1, Ny_l+1))
             
        params['t'] = params['t'] + 1e-4 * jax.random.uniform(key, params['t'].shape)
        
        # Remove OLD optimizer init (Moved inside loop)
        # optimizer = optax.chain(...)
        # opt_state = optimizer.init(params)

        # Pre-compute BCs to avoid jnp.where inside JIT
        case_bcs = []
        for case in self.cases:
            fd, fv, F = case.get_bcs(self.fem)
            free = jnp.setdiff1d(jnp.arange(self.fem.total_dof), fd)
            case_bcs.append({'fd': fd, 'fv': fv, 'F': F, 'free': free})

        # Removed comment
        # Removed comment
        def compute_raw_losses(p):
            # Map 'pz' to 'z' for solver
            solver_p = p.copy()
            if 'pz' in p:
                solver_p['z'] = p['pz']
            
            # If General Solver, ensure inputs are flattened appropriately
            if self.solver_type == 'general':
                for k in ['t', 'rho', 'E', 'z']:
                    if k in solver_p:
                        solver_p[k] = solver_p[k].flatten()

            K, M = self.fem.assemble(solver_p)
            
            # Initialize Raw Loss Dictionary
            raw_losses = {
                'static': 0.0, 'surface_stress': 0.0, 'surface_strain': 0.0, 'strain_energy': 0.0,
                'freq': 0.0, 'mode': 0.0, 'mass': 0.0, 'reg': 0.0
            }
            
            # --- 1. Static Responses Loss ---
            for i, case in enumerate(self.cases):
                fd = case_bcs[i]['fd']
                fv = case_bcs[i]['fv']
                F  = case_bcs[i]['F']
                free = case_bcs[i]['free']
                
                u = self.fem.solve_static_partitioned(K, F, free, fd, fv)
                
                # Extract displacement for matching (Default to W for pattern matching)
                # For general 6-DOF, index 2 is W.
                w = u[2::6] 
                
                # Displacement matching (RMSE)
                delta = w - self.targets_low[i]['u_static']
                scale = jnp.mean(jnp.abs(self.targets_low[i]['u_static'])) + 1e-6
                raw_losses['static'] += jnp.mean((delta / scale)**2) * self.targets_low[i]['weight']
                
                # Surface Stress matching
                if use_surface_stress:
                    curr_stress = self.fem.compute_max_surface_stress(u, solver_p)
                    delta = curr_stress - self.targets_low[i]['max_stress']
                    scale = jnp.mean(jnp.abs(self.targets_low[i]['max_stress'])) + 1e-6
                    raw_losses['surface_stress'] += jnp.mean((delta / scale)**2)
                    
                # Surface Strain matching
                if use_surface_strain:
                    curr_strain = self.fem.compute_max_surface_strain(u, solver_p)
                    delta = curr_strain - self.targets_low[i]['max_strain']
                    scale = jnp.mean(jnp.abs(self.targets_low[i]['max_strain'])) + 1e-6
                    raw_losses['surface_strain'] += jnp.mean((delta / scale)**2)
                
                # Strain Energy matching
                if use_strain_energy:
                    curr_energy = self.fem.compute_strain_energy_density(u, solver_p)
                    if 'strain_energy_density' in self.targets_low[i]:
                         tgt_e = self.targets_low[i]['strain_energy_density']
                         scale = jnp.mean(jnp.abs(tgt_e)) + 1e-6
                         raw_losses['strain_energy'] += jnp.mean(((curr_energy - tgt_e)/scale)**2)

            # Normalize Static Losses by number of cases
            n_cases = len(self.cases)
            raw_losses['static'] /= n_cases
            raw_losses['surface_stress'] /= n_cases
            raw_losses['surface_strain'] /= n_cases
            raw_losses['strain_energy'] /= n_cases

            # --- 2. Modal Responses Loss ---
            # Add diagonal shift for numerical stability during eigen-solve
            K_stable = K + 1e-6 * jnp.mean(jnp.abs(jnp.diag(K))) * jnp.eye(K.shape[0])
            vals, vecs = self.fem.solve_eigen(K_stable, M, num_modes=len(t_vals)+10)
            
            # Eigenfrequency matching (Skip 6 RB modes)
            curr_vals = jnp.abs(vals[6:6+len(t_vals)])
            res_diff = (curr_vals - t_vals) / (t_vals + 1e-4)
            raw_losses['freq'] = jnp.nan_to_num(jnp.mean(res_diff**2), nan=0.0)
            
            # Mode Shape matching (MAC)
            modes_w = vecs[2::6, 6:6+len(t_vals)]
            mac_loss = 0.0
            for j in range(len(t_vals)):
                num = jnp.sum(modes_w[:, j] * t_modes_l[:, j])**2
                den = (jnp.sum(modes_w[:, j]**2) + 1e-8) * (jnp.sum(t_modes_l[:, j]**2) + 1e-8)
                mac = num / den
                mac_loss += (1.0 - mac)
            raw_losses['mode'] = jnp.nan_to_num(mac_loss / len(t_vals), nan=0.0)

            # --- 3. Geometric Constraints & Regularization ---
            # Mass Constraint
            if use_mass_constraint:
                dx_l, dy_l = self.fem.Lx/Nx_l, self.fem.Ly/Ny_l
                curr_mass = jnp.sum(p['t'] * p['rho']) * dx_l * dy_l
                mass_err = (curr_mass - self.target_mass) / self.target_mass
                raw_losses['mass'] = jnp.abs(mass_err) # Linear penalty

            # Smoothness (TV Penalty)
            if use_smoothing:
                diff_x = jnp.diff(p['t'], axis=0)**2
                diff_y = jnp.diff(p['t'], axis=1)**2
                raw_losses['reg'] = (jnp.mean(diff_x) + jnp.mean(diff_y))

            return raw_losses

        # [Auto-Normalization REMOVED]
        # Using raw physical values directly without scaling.
        print("Loss scaling: Raw Physical Values (scales dict removed).")


        # Removed comment
        # Removed comment
        # Removed comment
        print(f"\n占?Starting Optimization with Learning Rate: {learning_rate}")
        current_lr = learning_rate
        
        initial_params = params.copy()
        
        # Define JIT Loss Function (Uses captured 'scales')
        @jax.jit
        def loss_fn(p):
            raw = compute_raw_losses(p)
            total_loss = 0.0
            metrics = {'Total': 0.0}
            keys = ['static', 'surface_stress', 'surface_strain', 'strain_energy', 'freq', 'mode', 'mass', 'reg']
            for k in keys:
                w_loss = raw[k] * scales[k] * loss_weights.get(k, 0.0)
                total_loss += w_loss
                if k == 'surface_stress': metric_key = 'Stress'
                elif k == 'surface_strain': metric_key = 'Strain'
                elif k == 'strain_energy': metric_key = 'Energy'
                else: metric_key = k.capitalize()
                metrics[metric_key] = raw[k]
            total_loss = jnp.nan_to_num(total_loss, nan=1e10)
            metrics['Total'] = total_loss
            return total_loss, metrics

        # 3. Optimization Loop (Wrapped in Function for Smart Retries)
        def run_optimizer_loop(current_params, lr, patience):
            
            # Use a more robust optimizer chain with selective clipping
            optimizer = optax.chain(
                optax.clip_by_global_norm(0.5), # More restrictive clipping
                optax.adam(learning_rate=lr)
            )
            opt_state = optimizer.init(current_params)
            
            p = current_params
            best_loss = float('inf')
            wait = 0
            
            # Removed comment
            # Items: Iter, Total, Disp, Stress, Strain, Energy, Freq, Mode, Mass, Reg
            print("\n" + "="*80)
            print(" [理쒖쟻??吏???ㅻ챸 媛?대뱶 (Optimization Metrics Guide)]")
            print("="*80)
            print(" * Total : ?꾩껜 媛以묒튂媛 諛섏쁺??珥??먯떎 ?⑥닔 媛?(?묒쓣?섎줉 醫뗭쓬)")
            print(" * Disp  : 蹂???ㅼ감 (Displacement). ?꾩껜 ?몃뱶 蹂??李⑥씠??[?됯퇏 ?쒓낢洹?RMSE)]")
            print(" * Strss : ?묐젰 ?ㅼ감 (Surface Stress). ?꾩껜 ?쒕㈃ ?묐젰 李⑥씠??[?됯퇏 ?쒓낢洹?")
            print(" * Strn  : 蹂?뺣쪧 ?ㅼ감 (Surface Strain). ?꾩껜 ?쒕㈃ 蹂?뺣쪧 李⑥씠??[?됯퇏 ?쒓낢洹?")
            print(" * Engy  : 蹂???먮꼫吏 諛???ㅼ감 (Strain Energy). ?꾩껜 ?붿냼 ?먮꼫吏 李⑥씠??[?됯퇏 ?쒓낢洹?")
            print(" * Freq  : 怨좎쑀吏꾨룞???ㅼ감. ?寃?紐⑤뱶蹂?二쇳뙆???ㅼ감?⑥쓽 [?됯퇏]")
            print(" * Mode  : 紐⑤뱶 ?뺤긽 ?ㅼ감 (1 - MAC). ?寃?紐⑤뱶 ?뺤긽 遺덉씪移섎룄??[?됯퇏]")
            print(" * Mass  : 吏덈웾 ?ㅼ감?? 紐⑺몴 吏덈웾 ?鍮?李⑥씠???덈뙎媛?[?⑥씪媛?")
            print(" * Reg   : ?됲솢??Smoothness) ?섎꼸?? ?몄젒 ?붿냼 ?먭퍡 李⑥씠??[?됯퇏]")
            print("-" * 80)
            
            print(f"??Tip: Press 'q' at any time to stop optimization safely.\n")
            print(f"{'It':<3} | {'Total':<8} | {'Disp':<8} | {'Strss':<8} | {'Strn':<8} | {'Engy':<8} | {'Freq':<8} | {'Mode':<8} | {'Mass':<8} | {'Reg':<7}")
            print("-" * 115)

            for i in range(max_iterations):
                (val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
            
                # ??DEFENSIVE: Guard against NaN gradients (especially from Modal components)
                grads = jax.tree_util.tree_map(lambda g: jnp.where(jnp.isnan(g), 0.0, g), grads)
                
                # Zero out gradients for non-optimized parameters
                for key in p:
                    if key in opt_config and not opt_config[key].get('opt', False):
                        grads[key] = jnp.zeros_like(grads[key])
                
                # Check for NaN in value
                if jnp.isnan(val):
                    print(f"??NaN Loss detected at iter {i}. Stopping.")
                    break
                
                # Check for User Interrupt (Press 'q')
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key.lower() == b'q':
                         print(f"\n[USER INTERRUPT] 'q' pressed at iter {i}. Stopping optimization safely...")
                         self.optimized_params = p
                         return self.optimized_params, best_loss, 'STOP' # Return STOP signal

                    
                updates, opt_state = optimizer.update(grads, opt_state)
                # Apply updates to 'p' (current parameters), NOT global 'params'
                p = optax.apply_updates(p, updates)
                
                # Explicit Param Clipping
                if 't' in p: p['t'] = jnp.clip(p['t'], opt_config['t']['min'], opt_config['t']['max'])
                if 'rho' in p: p['rho'] = jnp.clip(p['rho'], opt_config['rho']['min'], opt_config['rho']['max'])
                if 'E' in p: p['E'] = jnp.clip(p['E'], opt_config['E']['min'], opt_config['E']['max'])
                if 'pz' in p and 'pz' in opt_config: # Clip pz locally
                     if 'min' in opt_config['pz'] and 'max' in opt_config['pz']:
                         p['pz'] = jnp.clip(p['pz'], opt_config['pz']['min'], opt_config['pz']['max'])
                
                if i % 10 == 0:
                     # Use detailed metrics
                     d_loss = metrics.get('Static_Disp', metrics.get('Static', 0.0)) 
                     s_loss = metrics.get('Stress', 0.0)
                     n_loss = metrics.get('Strain', 0.0)
                     e_loss = metrics.get('Energy', 0.0)
                     
                     print(f"{i:<3d} | {val:<8.2e} | "
                           f"{d_loss:<8.2e} | {s_loss:<8.2e} | {n_loss:<8.2e} | {e_loss:<8.2e} | "
                           f"{metrics['Freq']:<8.2e} | {metrics['Mode']:<8.2e} | "
                           f"{metrics['Mass']:<8.2e} | {metrics['Reg']:<7.2e}")

                # Early Stopping Logic (Explicit)
                if use_early_stopping:
                    # Tolerance check
                    if val < best_loss - early_stop_tol:
                        best_loss = val
                        wait = 0
                        self.optimized_params = p
                    else:
                        wait += 1
                        if wait >= (early_stop_patience or 30):
                            print(f"??Optimization stalled. No improvement for {wait} iterations. Stopping iter.")
                            return self.optimized_params, best_loss, False # False = Not Converged/Stalled

                else:
                    self.optimized_params = p

            if i == max_iterations - 1:
                print("??Maximum iterations reached.")
            
            return self.optimized_params, best_loss, True # True = Completed/Converged

        # --- Execute Smart Optimization ---
        # [INIT]
        print("Starting Optimization Process...")

        # Run Loop
        final_params = initial_params
        
        # Call the optimizer loop
        # Note: We need to define loss_fn here or use the one inside. 
        # To avoid scope issues, we define loss_fn OUTSIDE the loop in previous edits, 
        # but here we need to ensure the optimizer calls it correctly.
        # Let's adjust the replacement structure carefully.
        
        # RE-STRUCTURING logic due to tool limitation: 
        # I will inject the loss_fn definition right after the attempt loop start,
        # then the run_optimizer_loop function, then call it.
        
        # Define JIT Loss Function (Uses captured 'scales')
        @jax.jit
        def loss_fn(p):
            raw = compute_raw_losses(p)
            total_loss = 0.0
            metrics = {'Total': 0.0}
            keys = ['static', 'surface_stress', 'surface_strain', 'strain_energy', 'freq', 'mode', 'mass', 'reg']
            for k in keys:
                # Weighted Loss = Raw * UserWeight (No Scaling)
                w_val = raw[k] * loss_weights.get(k, 0.0)
                total_loss += w_val
                
                if k == 'surface_stress': metric_key = 'Stress'
                elif k == 'surface_strain': metric_key = 'Strain'
                elif k == 'strain_energy': metric_key = 'Energy'
                else: metric_key = k.capitalize()
                
                metrics[metric_key] = raw[k]         # Removed comment
                metrics[metric_key + '_W'] = w_val   # Removed comment

            # [LOGGING FIX] Aggregate all static metrics into 'Static' for terminal output
            # Current 'Static' only holds displacement loss. rewrite it to sum.
            metrics['Static_Disp'] = metrics['Static'] # Save original
            metrics['Static'] = (
                metrics['Static'] + 
                metrics.get('Stress', 0.0) + 
                metrics.get('Strain', 0.0) + 
                metrics.get('Energy', 0.0)
            )
            # Weighted version too
            metrics['Static_W'] = (
                metrics['Static_W'] + 
                metrics.get('Stress_W', 0.0) + 
                metrics.get('Strain_W', 0.0) + 
                metrics.get('Energy_W', 0.0)
            )

            total_loss = jnp.nan_to_num(total_loss, nan=1e10)
            metrics['Total'] = total_loss
            return total_loss, metrics

        # Removed comment
        self.optimized_params = initial_params
        global_best_loss = float('inf')
        working_params = initial_params
        
        try:
            for attempt in range(1, 4): # Max 3 attempts
                print(f"\n?? Optimization Attempt {attempt}/3 (LR={current_lr:.1e})")
                
                # Removed comment
                opt_params, final_loss, success = run_optimizer_loop(working_params, current_lr, early_stop_patience)
                
                # [Interruption Check] Check for 'STOP' signal
                if success == 'STOP':
                    print("?썞 Optimization stopped by user request. Proceeding to verification...")
                    self.optimized_params = opt_params
                    break
                
                # Removed comment
                if final_loss < global_best_loss:
                    global_best_loss = final_loss
                    self.optimized_params = opt_params # Keep safe copy of best result
                
                # Removed comment
                # Removed comment
                if success and final_loss < init_total_loss * 0.5: 
                    print("??Optimization Success (Converged below 50% initial loss)!")
                    break
                else:
                    if attempt < 3:
                        print(f"??Optimization stalled (Loss: {final_loss:.2e}). Fine-tuning with lower LR...")
                        current_lr *= 0.5 # Decay LR
                        
                        # Removed comment
                        if final_loss > init_total_loss * 5.0:
                            # Removed comment
                            print("   -> Divergence detected (Loss exploaded). Resetting to original parameters with lower LR.")
                            working_params = initial_params
                        else:
                            # Removed comment
                            # Removed comment
                            print("   -> Starting from global best parameters with lower LR for finer search.")
                            working_params = self.optimized_params
                    else:
                        print("??Max attempts reached. Returning best available result.")
                        
        except KeyboardInterrupt:
            print("\n" + "!"*70)
            print(" ??USER INTERRUPT DETECTED (Ctrl+C)")
            print(" ?ъ슜?먭? 理쒖쟻?붾? 以묐떒?덉뒿?덈떎. ?꾩옱源뚯???理쒖쟻 寃곌낵濡?寃利앹쓣 吏꾪뻾?⑸땲??")
            # Removed comment
            print("!"*70)

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
        opt_params_h = {}
        for k in ['t', 'rho', 'E']:
             # Use self.optimized_params which holds the GLOBAL BEST result
             val_l = np.array(self.optimized_params[k]).flatten()
             val_v = griddata(pts_low, val_l, pts_high, method='cubic', fill_value=np.mean(val_l))
             opt_params_h[k] = jnp.array(val_v)
             
        # [CRITICAL FIX] Handle 'z' (Topography)
        # User Instruction: "Do NOT use target z directly."
        # If 'pz' is in optimized_params, use it. Otherwise, assume FLAT plate (z=0).
        if 'pz' in self.optimized_params:
             val_l = np.array(self.optimized_params['pz']).flatten()
             val_v = griddata(pts_low, val_l, pts_high, method='cubic', fill_value=0.0) 
             opt_params_h['z'] = jnp.array(val_v)
        else:
             # Default to Flat Plate if pz not optimized
             print("   -> 'pz' not optimized. Using FLAT topography (z=0).")
             opt_params_h['z'] = jnp.zeros_like(pts_high[:, 0])



        
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
            plot_field(axes[0,0], w_ref, "Target Disp (mm)", levels=levels_w)
            plot_field(axes[0,1], w_opt, "Optimized Disp (mm)", levels=levels_w)
            plot_field(axes[0,2], np.abs(w_opt - w_ref), "Error (mm)", cmap='magma')
            
            # Stress
            levels_s = np.linspace(0, np.max(stress_ref), 30)
            plot_field(axes[1,0], stress_ref, "Target Stress (MPa)", levels=levels_s, cmap='viridis')
            plot_field(axes[1,1], stress_opt, "Optimized Stress (MPa)", levels=levels_s, cmap='viridis')
            plot_field(axes[1,2], np.abs(stress_opt - stress_ref), "Error (MPa)", cmap='magma')
            
            # Strain
            levels_e = np.linspace(0, np.max(strain_ref)*1000, 30)
            plot_field(axes[2,0], strain_ref*1000, "Target Strain (e-3)", levels=levels_e, cmap='plasma')
            plot_field(axes[2,1], strain_opt*1000, "Optimized Strain (e-3)", levels=levels_e, cmap='plasma')
            plot_field(axes[2,2], np.abs(strain_opt - strain_ref)*1000, "Error (e-3)", cmap='magma')
            
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
        
        # Z-Shape
        z_plt = opt_params_h['z'].reshape(Nx_h+1, Ny_h+1)
        im3 = axes[1,1].contourf(xh, yh, z_plt.T, 30, cmap='terrain')
        axes[1,1].set_title("Topography (Z-Height)")
        plt.colorbar(im3, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig("verify_3d_parameters.png", dpi=150)
        plt.close()
        
        # 5. Modal Verification
        print("Verifying Modal Response...")
        n_modes = len(self.target_eigen['vals'])
        vals_opt, vecs_opt = self.fem_high.solve_eigen(K_opt, M_opt, num_modes=n_modes + 10)
        
        # Filter Rigid Body
        freq_ref = np.sqrt(np.abs(self.target_eigen['vals'])) / (2*np.pi)
        freq_opt = np.sqrt(np.abs(vals_opt[6:6+n_modes])) / (2*np.pi) # Skip 6 RB
        
        modes_ref = self.target_eigen['modes']
        modes_opt = vecs_opt[2::6, 6:6+n_modes]
        
        macs = []
        for j in range(n_modes):
            v1 = modes_opt[:, j] / np.linalg.norm(modes_opt[:, j])
            v2 = modes_ref[:, j] / np.linalg.norm(modes_ref[:, j])
            macs.append(float((np.dot(v1, v2))**2))
            
        # Modal Freq & MAC Bar Plot
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

        # Plot Mode Shapes (Contour)
        print("Plotting Mode Shapes...")
        fig = plt.figure(figsize=(20, 8))
        
        for j in range(n_modes):
            # Target Mode
            ax1 = fig.add_subplot(2, n_modes, j+1)
            mode_t = modes_ref[:, j].reshape(Nx_h+1, Ny_h+1)
            # Normalize for visuals
            mode_t = mode_t / (np.max(np.abs(mode_t)) + 1e-8)
            im = ax1.contourf(xh, yh, mode_t.T, levels=20, cmap='RdBu_r')
            ax1.set_title(f'Target Mode {j+1}\nf={freq_ref[j]:.2f} Hz', fontsize=9)
            ax1.axis('equal')
            plt.colorbar(im, ax=ax1, fraction=0.046)
            
            # Optimized Mode
            ax2 = fig.add_subplot(2, n_modes, n_modes + j+1)
            mode_o = modes_opt[:, j].reshape(Nx_h+1, Ny_h+1)
            mode_o = mode_o / (np.max(np.abs(mode_o)) + 1e-8)
            # Flip sign if anti-aligned (check dot product sign)
            if np.dot(modes_ref[:, j], modes_opt[:, j]) < 0:
                 mode_o = -mode_o
                 
            im = ax2.contourf(xh, yh, mode_o.T, levels=20, cmap='RdBu_r')
            ax2.set_title(f'Opt Mode {j+1}\nf={freq_opt[j]:.2f} Hz\nMAC={macs[j]:.3f}', fontsize=9)
            ax2.axis('equal')
            plt.colorbar(im, ax=ax2, fraction=0.046)
        
        plt.tight_layout()
        plt.savefig("verify_3d_mode_shapes.png", dpi=150)
        plt.close()
        print(f"Saved {os.path.abspath('verify_3d_mode_shapes.png')}")
        
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
        print("\n??Verification report saved to: verification_report.md")
        print("??Verification plots saved: verify_3d_*.png")

        # 7. Final Interactive Stage (PyVista) - Optional but good for inspection
        opt_eigen = {'vals': vals_opt[6:6+n_modes], 'modes': vecs_opt[2::6, 6:6+n_modes]}
        stage3_visualize_comparison(
            self.fem_high, self.targets, opt_params_h, self.target_params_high,
            opt_eigen=opt_eigen, tgt_eigen=self.target_eigen
        )

# ==============================================================================
# EXAMPLES & CLI MENU
# ==============================================================================

def run_matching_example():
    """ [Example 1] Pattern Matching (Bead/Thickness) """
    print("\n" + "#"*70)
    print(" [EXAMPLE 1] PATTERN MATCHING OPTIMIZATION")
    print("#"*70)
    
    model = EquivalentSheetModel(Lx, Ly, Nx_low, Ny_low, solver_type='general')
    
    # Add Standard Load Cases
    model.add_case(TwistCase("twist_x", axis='x', value=1.5, mode='angle'))
    model.add_case(TwistCase("twist_y", axis='y', value=1.5, mode='angle'))
    model.add_case(PureBendingCase("bend_x", axis='x', value=3.0, mode='angle'))
    model.add_case(PureBendingCase("bend_y", axis='y', value=3.0, mode='angle'))
    #model.add_case(CornerLiftCase("lift_tr", corner='tr', value=1.0, mode='force'))
    #model.add_case(CornerLiftCase("lift_tl", corner='tl', value=1.0, mode='force'))
    #model.add_case(CornerLiftCase("lift_br", corner='br', value=1.0, mode='force'))
    #model.add_case(CornerLiftCase("lift_bl", corner='bl', value=1.0, mode='force'))
    model.add_case(CornerLiftCase("lift_tr", corner='tr', value=1.0, mode='disp'))
    model.add_case(CornerLiftCase("lift_tl", corner='tl', value=2.0, mode='disp'))
    model.add_case(CornerLiftCase("lift_br", corner='br', value=1.0, mode='disp'))
    model.add_case(CornerLiftCase("lift_bl", corner='bl', value=2.0, mode='disp'))
    
    # Target Configuration
    target_config = {
        'pattern': 'ABC',           'base_t': 1.0, 
        'bead_t': {'A': 2.0, 'B': 2.5, 'C': 1.5},
        'pattern_pz': 'TNY',        'bead_pz': {'T': 2.0, 'N': 1.0, 'Y': -2.0},
        'base_rho': 7.85e-9,        'base_E': 210000.0,
    }
    
    model.generate_targets(resolution_high=(50, 20), num_modes_save=5, target_config=target_config)
    
    opt_config = {
        't':   {'opt': True,  'init': 1.2,       'min': 0.5,     'max': 5.0},
        'rho': {'opt': True, 'init': 7.85e-9,   'min': 5e-9,    'max': 1e-8},
        'E':   {'opt': True, 'init': 200000.0,  'min': 50000,   'max': 300000},
        'pz':  {'opt': True, 'init': 0.0, 'min': -5.,   'max': 5.},
    }
    
    # [Weights for Non-Normalized Optimization]
    # EXACT COPY of Main_Verification_D26-02-09 Settings
    weights = {
        'static': 1.0, 
        'freq': 1.0,             # Reverted to 1.0 (Match Reference)
        'mode': 1.0,             # Reverted to 1.0 (Match Reference)
        'strain_energy': 2.0,    # Increased to 2.0 (Match Reference)
        'surface_stress': 1.0,  
        'surface_strain': 1.0,  
        'mass': 1.0, 
        'reg': 0.05              
    }
    
    # Set explicit learning_rate=0.005 to match reference
    model.optimize(opt_config, weights, max_iterations=300, learning_rate=0.005, auto_lr=False)
    model.verify()

def example_topology_optimization():
    """ [Example 2] Topology Optimization (Stiffness Max, 6-DOF Shell) """
    print("\n" + "#"*70)
    print(" [EXAMPLE 2] TOPOLOGY OPTIMIZATION (6-DOF SHELL)")
    print("#"*70)
    
    # Use 'general' solver for 6-DOF capabilities
    model = EquivalentSheetModel(Lx, Ly, Nx_low, Ny_low, solver_type='general')
    
    # Use PositionalCase for flexible BC
    case = PositionalCase("Cantilever_EndLoad")
    # Fix left edge (x=0) using Bounding Box
    case.add_fixed_box(x_range=(0, 5), dofs=[0,1,2,3,4,5], vals=0.0)
    # Apply vertical load at right edge (x=Lx) center
    case.add_load_box(x_range=(Lx-5, Lx), y_range=(Ly/2-10, Ly/2+10), dof_val_dict={2: -100.0})
    model.add_case(case)
    
    # In Topology Optimization, we usually don't have a "target" field, 
    # but for verification logic we set a uniform flat field as reference.
    model.generate_targets(target_config={'pattern': 'NONE', 'base_t': 2.0})
    
    opt_config = {
        't':   {'opt': True,  'init': 1.0,       'min': 0.01,    'max': 4.0},
        'rho': {'opt': False, 'init': 7.85e-9},
        'E':   {'opt': False, 'init': 210000.0}
    }
    
    # Important: Enable SIMP in topo mode (to be handled in loss_fn/WHT_EQS_topo)
    # Weights focus on Compliance minimization
    weights = {'static': 10.0, 'mass': 1.0, 'reg': 0.5}
    
    model.optimize(opt_config, weights, max_iterations=300, use_mass_constraint=True, mass_tolerance=0.0)
    model.verify()

def example_topography_optimization():
    """ [Example 3] Topography Optimization (Bead Optimization) """
    print("\n" + "#"*70)
    print(" [EXAMPLE 3] TOPOGRAPHY (BEAD) OPTIMIZATION")
    print("#"*70)
    
    model = EquivalentSheetModel(Lx, Ly, Nx_low, Ny_low, solver_type='grid')
    
    case = PositionalCase("FourCorner_Fixed")
    # Fix four corners using Radius selection
    case.add_fixed_radius(center=(0, 0, 0), radius=20, dofs=[2]) # Fix vertical at corners
    case.add_fixed_radius(center=(Lx, 0, 0), radius=20, dofs=[2])
    case.add_fixed_radius(center=(0, Ly, 0), radius=20, dofs=[2])
    case.add_fixed_radius(center=(Lx, Ly, 0), radius=20, dofs=[2])
    # Anchor BL completely
    case.add_fixed_radius(center=(0, 0, 0), radius=1, dofs=[0, 1, 5])
    
    # Central Load
    case.add_load_box(x_range=(Lx/2-20, Lx/2+20), y_range=(Ly/2-20, Ly/2+20), dof_val_dict={2: -500.0})
    model.add_case(case)
    
    model.generate_targets(target_config={'pattern': 'NONE', 'base_t': 1.0})
    
    opt_config = {
        't':   {'opt': False, 'init': 1.0},
        'rho': {'opt': False, 'init': 7.85e-9},
        'E':   {'opt': False, 'init': 210000.0},
        'pz':  {'opt': True,  'init': 0.0,       'min': -20.0,   'max': 20.0}
    }
    
    weights = {'static': 1.0, 'reg': 1.0}
    model.optimize(opt_config, weights, max_iterations=300, use_smoothing=True)
    model.verify()


if __name__ == '__main__':
    # XLA Flags (Optimized for CPU)
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"

    print("\n" + "="*80)
    print(" EQUIVALENT SHEET OPTIMIZATION FRAMEWORK v2.0")
    print("="*80)
    print(" [1] Pattern Matching Example (Default)")
    print(" [2] Topology Optimization Example (6-DOF Shell, Cantilever)")
    print(" [3] Topography Optimization Example (Bead Optimization, 4-Corners)")
    print("="*80)
    
    try:
        choice = input(">> Select Example Number: ").strip()
        if choice == '1':
            run_matching_example()
        elif choice == '2':
            example_topology_optimization()
        elif choice == '3':
            example_topography_optimization()
        else:
            print("Running default matching example...")
            run_matching_example()
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
