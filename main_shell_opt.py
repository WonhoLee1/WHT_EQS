import os
import sys

# [WHTOOLS] 대화형 시각화를 위해 NON_INTERACTIVE 기본값을 0으로 설정합니다 (환경변수 미설정 시)
if os.environ.get("NON_INTERACTIVE") is None:
    os.environ["NON_INTERACTIVE"] = "0"

# [CRITICAL FIX] Windows 콘솔 인코딩 문제 해결 (Python 3.7+ 표준 방식)
if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import pickle
import msvcrt
import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import koreanize_matplotlib
import datetime
import math

# Import dependencies from modular components
from ShellFemSolver.shell_solver import ShellFEM
from ShellFemSolver.mesh_utils import generate_rect_mesh_quads, generate_tray_mesh_quads
from WHT_EQS_analysis import PlateFEM, StructuralResult
from opt_targets import (
    ResultBundle, Mode, apply_case_targets_from_spec, 
    map_legacy_flags_to_targets, OptTarget, TargetType, parse_opt_targets
)
from WHT_EQS_pattern_generator import get_thickness_field, get_z_field
from WHT_EQS_load_cases import (
    TwistCase, PureBendingCase, CornerLiftCase, 
    TwoCornerLiftCase, CantileverCase, PressureCase
)
from WHT_EQS_visualization import (
    stage1_visualize_patterns,
    stage2_visualize_ground_truth,
    stage3_visualize_comparison
)
from solver import safe_eigh  # [CRITICAL FIX 27] 안전한 고유치 미분을 위한 임포트
from wh_utils import WHTable, wh_print_banner # [UI] 분리된 전용 유틸리티 사용

# Mesh Settings (Synced with verification script)
Lx, Ly = 1450.0, 850.0
Nx_high, Ny_high = int(Lx/30.), int(Ly/30.)      # High-res ground truth
Nx_low, Ny_low = int(Lx/60.), int(Ly/60.)        # Optimization mesh resolution sync

class EquivalentSheetModel:
    def __init__(self, Lx, Ly, nx, ny, wall_width=50.0, wall_height=50.0):
        self.wall_width = wall_width
        self.wall_height = wall_height
        self.fem = PlateFEM(Lx, Ly, nx, ny, wall_width=wall_width, wall_height=wall_height) 
        self.cases = []
        self.targets = []
        self.targets_bundles = []
        self.resolution_high = (Nx_high, Ny_high)
        self.target_mass = 0.0
        self.target_start_idx = 6 # Default for Free-Free
        self.optimized_params = None
        self.target_params_high = None
        self.config = {}
        
        # [PERFORMANCE] Pre-calculate assembly indices for high-speed JAX execution
        self.fem.fem._prepare_assembly_cache()

    def add_case(self, case):
        self.cases.append(case)

    def generate_targets(self, resolution_high=(Nx_high, Ny_high), num_modes_save=5, cache_file="target_cache.pkl", target_config={}, use_cache_override=None, wall_width=None, wall_height=None):
        """
        Generates ground truth data using a high-fidelity model.
        """
        wh_print_banner("STAGE 1: TARGET GENERATION & PATTERN VERIFICATION")
        
        # Override geometry if provided for GT model
        gt_wall_width = wall_width if wall_width is not None else self.wall_width
        gt_wall_height = wall_height if wall_height is not None else self.wall_height
        
        # [BUGFIX] Store target_config in model for later use in verify()
        self.config['target_config'] = target_config
        self.resolution_high = resolution_high
        Nx_h, Ny_h = resolution_high
        
        # 1. Create High-Resolution Mesh for "Ground Truth"
        self.fem_high = PlateFEM(self.fem.Lx, self.fem.Ly, Nx_h, Ny_h, wall_width=gt_wall_width, wall_height=gt_wall_height)
        fem_high = self.fem_high
        xh = np.linspace(0, self.fem.Lx, Nx_h+1)
        yh = np.linspace(0, self.fem.Ly, Ny_h+1)
        Xh, Yh = np.meshgrid(xh, yh, indexing='xy')

        # --- Cache Check ---
        use_cache = False
        if use_cache_override is not None:
            use_cache = use_cache_override
        elif cache_file and os.environ.get("NON_INTERACTIVE") == "1":
            use_cache = os.path.exists(cache_file)
            print(f"[NON-INTERACTIVE] File exists: {use_cache}. Loading if possible.")
        elif cache_file and os.path.exists(cache_file):
            print(f"\n[CACHE DETECTED] Found existing Ground Truth data: {cache_file}")
            if os.environ.get("NON_INTERACTIVE") != "1":
                choice = input(" -> Do you want to load from cache? [y/N] (Default: N - Re-interpret): ").strip().upper()
                if choice == 'Y':
                    use_cache = True
        
        if use_cache:
            try:
                print("Loading Ground Truth from cache...")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self.target_params_high = cache_data['target_params_high']
                self.target_mass = cache_data['target_mass']
                self.targets = cache_data['targets']
                self.target_eigen = cache_data['target_eigen']
                print("[OK] Cache loaded successfully.")
            except Exception as e:
                print(f"[WARNING] Failed to load cache ({e}). Recalculating...")
                use_cache = False
                
        if not use_cache:
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
            
            # Combine Pattern Z (pz) with Base Mesh Z (Tray Height)
            base_z_h = fem_high.nodes[:, 2].reshape(Xh.shape)
            z_h_full = base_z_h + z_h
            
            self.target_params_high = {
                't': t_h.flatten(), 
                'z': z_h.flatten(), 
                'rho': rho_h.flatten(), 
                'E': E_h.flatten()
            }
            
            # Calculate target mass
            dx, dy = self.fem.Lx/Nx_h, self.fem.Ly/Ny_h
            self.target_mass = float(jnp.sum(t_h * rho_h) * dx * dy)
            
        # 2.5 Pattern Visualization
        t_plot = self.target_params_high['t'].reshape(Xh.shape)
        z_plot = self.target_params_high['z'].reshape(Xh.shape)
        base_z_h = self.fem_high.nodes[:, 2].reshape(Xh.shape)
        z_full_plot = base_z_h + z_plot
        stage1_visualize_patterns(Nx_h, Ny_h, Xh, Yh, t_plot, z_full_plot)
            
        if not use_cache:
            # 3. Solve FEM for each load case (High Fidelity - Sparse Solve)
            print("\nSolving High-Resolution Ground Truth (Sparse)...")
            K_h, M_h = fem_high.assemble(self.target_params_high, sparse=True)
            self.targets = []
            self.targets_bundles = []
            
            for i, case in enumerate(self.cases):
                print(f" -> Solving Case: {case.name}")
                fixed_dofs, fixed_vals, F = case.get_bcs(fem_high)
                free_dofs = np.setdiff1d(np.arange(fem_high.total_dof), fixed_dofs)
                u = fem_high.solve_static_sparse(K_h, F, free_dofs, fixed_dofs, fixed_vals)
                
                # Compute Detailed Field Results
                f_res = fem_high.compute_field_results(u, self.target_params_high)
                
                # Reaction Forces
                F_int = K_h @ jnp.array(u)
                R_residual = F_int - np.array(F)
                
                # Export to ParaView
                res_fields = {
                    'displacement_vec': np.array(u),
                    'stress_vm': np.array(f_res['stress_vm']),
                    'strain_equiv': np.array(f_res['strain_equiv_nodal']),
                    'sed': np.array(f_res['sed'])
                }
                res = StructuralResult(res_fields, np.array(self.fem_high.nodes), fem_high.elements)
                res.save_vtkhdf(f"gt_static_{case.name}.vtkhdf")
                
                target_data = {
                    'case_name': case.name,
                    'u_static': np.array(u[2::6]), 
                    'u_full': np.array(u),
                    'reaction_full': np.array(R_residual),
                    'max_stress': np.array(f_res['stress_vm']),
                    'max_strain': np.array(f_res['strain_equiv_nodal']),
                    'strain_energy_density': np.array(f_res['sed']),
                    'params': self.target_params_high
                }
                self.targets.append(target_data)

                # ResultBundle for OptTarget
                bundle = ResultBundle(
                    fields={
                        'displacement_vec': np.array(u),
                        'u_static': np.array(u[2::6]),
                        'stress_vm': np.array(f_res['stress_vm']),
                        'max_stress': np.array(f_res['stress_vm']),
                        'strain_equiv': np.array(f_res['strain_equiv_nodal']),
                        'max_strain': np.array(f_res['strain_equiv_nodal']),
                        'sed': np.array(f_res['sed']),
                        'strain_energy_density': np.array(f_res['sed'])
                    },
                    rbe_reactions={'residual': np.array(R_residual)},
                    node_disps={}, # Simplified for now
                    mass=float(self.target_mass),
                    modes=[],
                    meta={'params': self.target_params_high}
                )
                self.targets_bundles.append(bundle)
                    
            # Summary Ground Truth Plot
            print("\nGenerating Ground Truth Summary Plot (3xN)...")
            n_cases = len(self.cases)
            plt.rcParams.update({'font.size': 8})
            fig, axes = plt.subplots(3, n_cases, figsize=(4*n_cases, 10), squeeze=False)
            fig.suptitle(f"Ground Truth Analysis Summary (Resolution: {Nx_h}x{Ny_h})\nRows: Disp, Stress, Strain", fontsize=10)
            
            for i, target in enumerate(self.targets):
                xh_l, yh_l = np.linspace(0, self.fem.Lx, Nx_h+1), np.linspace(0, self.fem.Ly, Ny_h+1)
                data_w = target['u_static'].reshape(Ny_h+1, Nx_h+1)
                im0 = axes[0, i].contourf(xh_l, yh_l, data_w, 30, cmap='jet')
                axes[0, i].set_title(f"Case: {target['case_name']}\nMax Disp: {np.max(np.abs(data_w)):.3f}mm", fontsize=8)
                axes[0, i].set_aspect('equal'); plt.colorbar(im0, ax=axes[0, i], shrink=0.7)
                
                data_s = target['max_stress'].reshape(Ny_h+1, Nx_h+1)
                im1 = axes[1, i].contourf(xh_l, yh_l, data_s, 30, cmap='jet')
                axes[1, i].set_title(f"Max Stress: {np.max(data_s):.2f} MPa", fontsize=8)
                axes[1, i].set_aspect('equal'); plt.colorbar(im1, ax=axes[1, i], shrink=0.7)
                
                data_e = target['max_strain'].reshape(Ny_h+1, Nx_h+1) * 1000
                im2 = axes[2, i].contourf(xh_l, yh_l, data_e, 30, cmap='jet')
                axes[2, i].set_title(f"Max Strain: {np.max(data_e):.3f} e-3", fontsize=8)
                axes[2, i].set_aspect('equal'); plt.colorbar(im2, ax=axes[2, i], shrink=0.7)
                
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig("ground_truth_3d_loadcases.png", dpi=150); plt.close()
            
            # Modal
            vals, vecs = fem_high.solve_eigen_sparse(K_h, M_h, num_modes=num_modes_save + 10)
            start_idx = 6
            self.target_eigen = {'vals': np.array(vals[start_idx : start_idx + num_modes_save]), 
                                 'modes': np.array(vecs[2::6, start_idx : start_idx + num_modes_save])}
        
            modal_res = StructuralResult({}, np.array(self.fem_high.nodes), fem_high.elements)
            steps_dict = {'values': self.target_eigen['vals'], 
                          'point_data': {'mode_shape_vec': [vecs[:, i] for i in range(start_idx, start_idx + num_modes_save)]}}
            modal_res.save_vtkhdf("gt_modal_results.vtkhdf", steps_dict=steps_dict)
            
            if cache_file:
                with open(cache_file, 'wb') as f:
                    pickle.dump({'target_params_high': self.target_params_high, 'target_mass': self.target_mass, 
                                 'targets': self.targets, 'target_eigen': self.target_eigen}, f)
        
        stage2_visualize_ground_truth(fem_high, self.targets, self.target_params_high, eigen_data=self.target_eigen)


    def optimize_v2(self, opt_config, opt_target_config,
                    use_bead_smoothing=False,
                    max_iterations=200,
                    use_early_stopping=True,
                    early_stop_patience=30,
                    early_stop_tol=1e-8,
                    learning_rate=0.3,
                    min_bead_width=150.0,
                    init_pz_from_gt=False,
                    gt_init_scale=1.0,
                    eigen_freq=20,
                    reg_weight=0.01,
                    eigen_solver='lobpcg'):
        """
        [OptTarget 기반 위상/두께 최적화 수행기 (V2)]
        JSON 명세 또는 Dictionary 형태의 opt_target_config를 직접 입력받아
        각 케이스/글로벌 목적 함수를 동적으로 평가하고 JAX 최적화 루프에 반영합니다.
        """
        wh_print_banner("STAGE 3: OPT-TARGET DRIVEN OPTIMIZATION (V2)")
    
        Nx_l, Ny_l = self.fem.nx, self.fem.ny
        pts_l = np.array(self.fem.nodes)[:, :3]
        pts_h = np.array(self.fem_high.nodes)[:, :3]
        
        if hasattr(self, 'global_opt_targets'):
            self.global_opt_targets.clear()
        for case in self.cases:
            if hasattr(case, 'opt_targets'):
                case.opt_targets.clear()
                
        apply_case_targets_from_spec(self, opt_target_config)
        
        from opt_targets import parse_opt_targets
        if not hasattr(self, 'global_opt_targets'):
            self.global_opt_targets = []
        raw_globals = opt_target_config.get('global_targets', [])
        if raw_globals:
            self.global_opt_targets.extend(parse_opt_targets(raw_globals))
            print(f" -> Global targets registered: {[ot.target_type.value for ot in self.global_opt_targets]}")
        
        print(" -> OptTarget configuration applied to model cases.")
    
        # 2. GT(Ground Truth) 보간 (고해상도 -> 저해상도)
        self.targets_low = []
        Nx_h, Ny_h = self.resolution_high
        is_same_res = bool(self.resolution_high[0] == Nx_l and self.resolution_high[1] == Ny_l)
        
        def safe_interp(src_pts, src_vals, dst_pts):
            val = griddata(src_pts, np.array(src_vals), dst_pts, method='linear')
            nan_mask = np.isnan(val)
            if np.any(nan_mask):
                # [FIX] Handle multi-column data (e.g. R_h_nodes)
                if val.ndim > 1:
                    row_mask = np.any(nan_mask, axis=1)
                    val[row_mask] = griddata(src_pts, np.array(src_vals), dst_pts[row_mask], method='nearest')
                else:
                    val[nan_mask] = griddata(src_pts, np.array(src_vals), dst_pts[nan_mask], method='nearest')
            return jnp.array(val)
    
        for tgt in self.targets:
            if is_same_res:
                u_l = np.array(tgt['u_static'])
                stress_l = np.array(tgt['max_stress'])
                strain_l = np.array(tgt['max_strain'])
                sed_l = np.array(tgt['strain_energy_density'])
            else:
                u_l = safe_interp(pts_h, tgt['u_static'], pts_l)
                print(f" DEBUG: Case {tgt.get('case_name','unknown')} | GT Max Disp (W): {np.max(np.abs(tgt['u_static'])):.4e} -> Interp Max: {np.max(np.abs(u_l)):.4e}")
                stress_l = safe_interp(pts_h, tgt['max_stress'], pts_l)
                strain_l = safe_interp(pts_h, tgt['max_strain'], pts_l)
                sed_l = safe_interp(pts_h, tgt['strain_energy_density'], pts_l)
                
            R_tgt = tgt['reaction_full']
            target_R_sums = jnp.stack([
                jnp.sum(jnp.abs(R_tgt[0::6])), jnp.sum(jnp.abs(R_tgt[1::6])),
                jnp.sum(jnp.abs(R_tgt[2::6])), jnp.sum(jnp.abs(R_tgt[3::6])),
                jnp.sum(jnp.abs(R_tgt[4::6])), jnp.sum(jnp.abs(R_tgt[5::6]))
            ])
            
            if is_same_res:
                R_l_full = np.array(R_tgt)
            else:
                R_h_nodes = np.array(R_tgt).reshape(-1, 6)
                R_l_nodes = safe_interp(pts_h, R_h_nodes, pts_l)
                area_scale = (Nx_h * Ny_h) / max(1.0, float(Nx_l * Ny_l))
                R_l_full = np.array(R_l_nodes).flatten() * area_scale
            
            self.targets_low.append({
                'u_static': jnp.array(u_l),
                'max_stress': jnp.array(stress_l),
                'max_strain': jnp.array(strain_l),
                'strain_energy_density': jnp.array(sed_l),
                'reaction_sums': target_R_sums,
                'reaction_full': jnp.array(R_l_full)
            })
    
        num_modes_loss = 5
        for global_target in getattr(self, 'global_opt_targets', []):
            if hasattr(global_target, 'target_type') and global_target.target_type == TargetType.MODES:
                if hasattr(global_target, 'num_modes') and global_target.num_modes:
                    num_modes_loss = global_target.num_modes
    
        t_modes_h_np = np.array(self.target_eigen['modes'])
        num_modes_available = t_modes_h_np.shape[1]
        num_modes_loss = min(num_modes_loss, num_modes_available)
        t_vals = np.array(self.target_eigen['vals'])[:num_modes_loss]
    
        xh_base = np.linspace(0, self.fem.Lx, Nx_high + 1)
        yh_base = np.linspace(0, self.fem.Ly, Ny_high + 1)
        XH, YH = np.meshgrid(xh_base, yh_base, indexing='xy')
        
        xl_base = np.linspace(0, self.fem.Lx, Nx_l + 1)
        yl_base = np.linspace(0, self.fem.Ly, Ny_l + 1)
        XL, YL = np.meshgrid(xl_base, yl_base, indexing='xy')
    
        n_h = len(pts_h)
        if is_same_res:
            t_modes_l = jnp.array(t_modes_h_np[:n_h, :num_modes_loss])
        else:
            t_modes_h_z = t_modes_h_np[:n_h, :]
            pts_h_actual = np.array(pts_h[:, :2])
            pts_l_actual = np.array(self.fem.nodes[:, :2])
            
            t_modes_l_list = []
            for i in range(num_modes_loss):
                mapped = griddata(pts_h_actual, t_modes_h_z[:, i], pts_l_actual, method='linear')
                nan_mask = np.isnan(mapped)
                if np.any(nan_mask):
                    mapped[nan_mask] = griddata(pts_h_actual, t_modes_h_z[:, i], pts_l_actual[nan_mask], method='nearest')
                t_modes_l_list.append(mapped)
            t_modes_l = jnp.array(np.stack(t_modes_l_list, axis=1))
        
        print(f" -> Fundamental Multi-Res Mapping: {t_modes_l.shape} nodes/modes matched.")
            
        target_z_low = None
        if 'z' in self.target_params_high:
            z_h_flat = self.target_params_high['z']
            target_z_low = jnp.array(z_h_flat if is_same_res else safe_interp(pts_h, z_h_flat, pts_l)).reshape(Nx_l+1, Ny_l+1)
    
        opt_config = {k: v.copy() for k, v in opt_config.items()}
        if init_pz_from_gt and target_z_low is not None and 'pz' in opt_config:
            opt_config['pz']['init'] = target_z_low * gt_init_scale
    
        self.scaling = {}
        full_params_scaled = {}
        for k in ['t', 'rho', 'E', 'pz']:
            if k in opt_config:
                val = opt_config[k].get('init', 0.0)
                v_scalar = float(jnp.mean(jnp.abs(val))) if isinstance(val, (np.ndarray, jnp.ndarray)) else float(abs(val))
                self.scaling[k] = v_scalar if v_scalar > 1e-15 else 1.0
                init_phys = opt_config[k].get('init', 1.0 if k == 't' else 0.0)
                if opt_config[k].get('type') == 'global' or not opt_config[k].get('opt', True):
                    val = jnp.mean(jnp.array(init_phys)) if isinstance(init_phys, (np.ndarray, jnp.ndarray)) else float(init_phys)
                    full_params_scaled[k] = jnp.array([val / self.scaling[k]])
                else:
                    if isinstance(init_phys, (np.ndarray, jnp.ndarray)):
                        full_params_scaled[k] = jnp.array(init_phys) / self.scaling[k]
                    else:
                        full_params_scaled[k] = jnp.full((Nx_l+1, Ny_l+1), init_phys / self.scaling[k])
    
        params = {k: v for k, v in full_params_scaled.items() if opt_config.get(k, {}).get('opt', True)}
        fixed_params_scaled = {k: v for k, v in full_params_scaled.items() if not opt_config.get(k, {}).get('opt', True)}
        scaling_consts = {k: float(v) for k, v in self.scaling.items()}
    
        filter_kernel = None
        if use_bead_smoothing and min_bead_width > 0:
            dx_l, dy_l = self.fem.Lx/Nx_l, self.fem.Ly/Ny_l
            rx, ry = int(np.ceil(min_bead_width / (2*dx_l))), int(np.ceil(min_bead_width / (2*dy_l)))
            kx, ky = np.linspace(-rx, rx, 2*rx+1), np.linspace(-ry, ry, 2*ry+1)
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            dist = np.sqrt((KX*dx_l)**2 + (KY*dy_l)**2)
            kernel = np.maximum(0, (min_bead_width/2.0) - dist)
            filter_kernel = jnp.array(kernel / np.sum(kernel))
    
        warmup_steps = min(max_iterations // 2, max(1, max_iterations // 20))
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate / 5.0, peak_value=learning_rate,
            warmup_steps=warmup_steps, decay_steps=max_iterations, end_value=learning_rate * 0.01
        )
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr_schedule))
        opt_state = optimizer.init(params)
        
        case_bcs = []
        z_lock_nodes = set()
        for case in self.cases:
            fd, fv, F = case.get_bcs(self.fem)
            case_bcs.append({'fd': fd, 'fv': fv, 'F': F, 'free': jnp.setdiff1d(jnp.arange(self.fem.total_dof), fd)})
            z_dofs = [int(d) for d in fd if d % 6 == 2]
            z_lock_nodes.update([d // 6 for d in z_dofs])
            
        z_lock_mask_1d = jnp.ones((Nx_l+1) * (Ny_l+1))
        if z_lock_nodes:
            z_lock_mask_1d = z_lock_mask_1d.at[jnp.array(list(z_lock_nodes))].set(0.0)
    
        targets_jax = []
        target_info = [] 
        for ci, case in enumerate(self.cases):
            for ot in getattr(case, 'opt_targets', []):
                desc = {'case_idx': ci, 'type': ot.target_type.value, 'weight': float(ot.weight)}
                desc['compare_mode'] = ot.compare_mode.value
                if ot.target_type == TargetType.FIELD_STAT:
                    desc['ref_key'] = {'stress_vm': 'max_stress', 'max_strain': 'max_strain', 'strain_energy_density': 'strain_energy_density'}.get(ot.field, 'u_static')
                    desc['reduction'] = ot.reduction.value
                elif ot.target_type == TargetType.RBE_REACTION:
                    desc['ref_key'] = 'reaction_full'
                targets_jax.append(desc)
                target_info.append({'case': case.name, 'type': ot.target_type.value, 'field': getattr(ot, 'field', None) or getattr(ot, 'component', None) or '', 'mode': ot.compare_mode.value})
                
        for ot in getattr(self, 'global_opt_targets', []):
            desc = {'case_idx': None, 'type': ot.target_type.value, 'weight': float(ot.weight)}
            desc['compare_mode'] = ot.compare_mode.value
            if ot.target_type == TargetType.MASS:
                desc['ref_value'] = float(ot.ref_value) if ot.ref_value else float(self.target_mass)
                desc['tolerance'] = float(getattr(ot, 'tolerance', 0.05))
            elif ot.target_type == TargetType.MODES:
                desc['num_modes'] = getattr(ot, 'num_modes', num_modes_loss)
                fw = float(getattr(ot, 'freq_weight', 0.0))
                desc['freq_weight'] = fw
                if fw > 0.0:
                    mac_desc = desc.copy(); mac_desc['sub_type'] = 'mac'; mac_desc['weight'] = float(ot.weight) * (1.0 - fw)
                    targets_jax.append(mac_desc)
                    target_info.append({'case': 'Global', 'type': 'modes', 'field': 'MAC (Shape)', 'mode': ot.compare_mode.value, 'weight': mac_desc['weight']})
                    freq_desc = desc.copy(); freq_desc['sub_type'] = 'freq'; freq_desc['weight'] = float(ot.weight) * fw
                    targets_jax.append(freq_desc)
                    target_info.append({'case': 'Global', 'type': 'modes', 'field': 'Freq (Hz)', 'mode': ot.compare_mode.value, 'weight': freq_desc['weight']})
                    continue 
                else: desc['sub_type'] = 'mac'
            targets_jax.append(desc)
            target_info.append({'case': 'Global', 'type': ot.target_type.value, 'field': getattr(ot, 'field', None) or getattr(ot, 'component', None) or '', 'mode': ot.compare_mode.value, 'weight': float(ot.weight)})
    
        case_needs = {i: {'u': False, 'stress': False, 'strain': False, 'sed': False, 'reaction': False} for i in range(len(self.cases))}
        for desc in targets_jax:
            ci = desc.get('case_idx')
            cases_to_update = [int(ci)] if ci is not None else range(len(self.cases))
            for idx in cases_to_update:
                ref_key = desc.get('ref_key')
                if ref_key == 'max_stress': case_needs[idx]['stress'] = True
                elif ref_key == 'max_strain': case_needs[idx]['strain'] = True
                elif ref_key == 'strain_energy_density': case_needs[idx]['sed'] = True
                elif ref_key == 'reaction_full': case_needs[idx]['reaction'] = True
                elif ref_key == 'u_static': case_needs[idx]['u'] = True
    
        def loss_fn(p_scaled, fixed_p_scaled, do_eigen=False, cached_freqs=None, cached_vecs=None):
            combined_phys = {k: v * scaling_consts[k] for k, v in p_scaled.items()}
            for k, v in fixed_p_scaled.items(): combined_phys[k] = v * scaling_consts[k]
            if filter_kernel is not None:
                pad_h, pad_w = filter_kernel.shape[0]//2, filter_kernel.shape[1]//2
                for k in combined_phys:
                    if combined_phys[k].ndim > 1:
                        padded = jnp.pad(combined_phys[k], ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
                        combined_phys[k] = jax.scipy.signal.convolve2d(padded, filter_kernel, mode='valid')
            if 'pz' in combined_phys:
                pz_val = combined_phys['pz']
                z_map = jnp.full((Nx_l+1, Ny_l+1), pz_val[0]) if (pz_val.ndim == 1 and pz_val.shape[0] == 1) else combined_phys['pz']
                z_masked = (z_map.flatten() * z_lock_mask_1d)
                combined_phys['z'] = z_masked
            def broadcast_nodal(val):
                return jnp.full(((Nx_l+1)*(Ny_l+1),), val[0]) if val.ndim == 1 and val.shape[0] == 1 else val.flatten()
            p_fem = {k: broadcast_nodal(v) for k, v in combined_phys.items() if k != 'pz'}
            K, M = self.fem.assemble(p_fem)
            
            freqs, vecs_filtered = None, None
            if do_eigen:
                m_diag = jnp.maximum(jnp.diag(M), 1e-15)
                m_inv_sqrt = 1.0 / jnp.sqrt(m_diag)
                K_stable = (K + K.T) / 2.0
                A = K_stable * m_inv_sqrt[:, None] * m_inv_sqrt[None, :]
                eps_unique = jnp.logspace(-9, -4, A.shape[0])
                A = A + jnp.diag(eps_unique)
                vals, vecs_v = safe_eigh(A)
                vecs = vecs_v * m_inv_sqrt[:, None]
                vals = jnp.maximum(vals, 0.0)
                search_window = 3
                freqs = jnp.sqrt(jnp.maximum(vals[6:num_modes_loss + 6 + search_window], 1e-12)) / (2 * jnp.pi)
                vecs_filtered = vecs[:, 6:num_modes_loss + 6 + search_window]
            elif cached_vecs is not None:
                rayleigh_num = jnp.sum(cached_vecs * (K @ cached_vecs), axis=0)
                rayleigh_den = jnp.sum(cached_vecs * (M @ cached_vecs), axis=0)
                approx_vals = rayleigh_num / jnp.maximum(rayleigh_den, 1e-15)
                freqs = jnp.sqrt(jnp.maximum(approx_vals, 1e-12)) / (2 * jnp.pi)
                vecs_filtered = cached_vecs
            else:
                freqs = cached_freqs; vecs_filtered = cached_vecs
            
            per_w, per_stress, per_strain, per_sed, per_reaction = [], [], [], [], []
            for i, case in enumerate(self.cases):
                req = case_needs[i]
                if not any(req.values()):
                    per_w.append(jnp.zeros(0)); per_stress.append(jnp.zeros(0)); per_strain.append(jnp.zeros(0)); per_sed.append(jnp.zeros(0)); per_reaction.append(jnp.zeros(0))
                    continue
                b = case_bcs[i]
                u = self.fem.solve_static_partitioned(K, b['F'], b['free'], b['fd'], b['fv'])
                per_w.append(u[2::6] if req['u'] else jnp.zeros(0))
                f_res = self.fem.compute_field_results(u, p_fem) if (req['stress'] or req['strain'] or req['sed']) else None
                per_stress.append(self.fem.compute_max_surface_stress(u, p_fem, field_results=f_res) if req['stress'] else jnp.zeros(0))
                per_strain.append(self.fem.compute_max_surface_strain(u, p_fem, field_results=f_res) if req['strain'] else jnp.zeros(0))
                per_sed.append(self.fem.compute_strain_energy_density(u, p_fem, field_results=f_res) if req['sed'] else jnp.zeros(0))
                per_reaction.append(K @ u - b['F'] if req['reaction'] else jnp.zeros(0))
    
            def get_case_arr(key, idx):
                if key == 'u_static': return per_w[idx]
                if key == 'max_stress': return per_stress[idx]
                if key == 'max_strain': return per_strain[idx]
                if key == 'strain_energy_density': return per_sed[idx]
                if key == 'reaction_full': return per_reaction[idx]
                return per_w[idx]
    
            opt_loss = 0.0
            target_evals = [] 
            matched_freqs, best_mac = None, None
            for desc in targets_jax:
                ci = desc.get('case_idx')
                typ, wgt, comp_mode = desc['type'], desc['weight'], desc.get('compare_mode', 'absolute')
                val, ref_val, terr = 0.0, 0.0, 0.0
                if ci is not None:
                    idx = int(ci); data = get_case_arr(desc.get('ref_key','u_static'), idx)
                    ref_arr = self.targets_low[idx].get(desc.get('ref_key','u_static'), 0.0)
                    if typ == 'field_stat' and desc.get('reduction') == 'mse':
                        ref_rms = jnp.sqrt(jnp.mean(ref_arr ** 2)); scale = jnp.maximum(ref_rms, 1e-4)
                        terr_arr = (data - ref_arr) / scale if comp_mode == 'relative' else (data - ref_arr)
                        opt_loss += wgt * jnp.mean(terr_arr ** 2)
                        val, ref_val, terr = jnp.sqrt(jnp.mean(data**2)), ref_rms, jnp.sqrt(jnp.mean(terr_arr ** 2))
                    elif typ == 'field_stat':
                        red = desc.get('reduction', 'max')
                        val = jnp.nanmax(data) if red == 'max' else (jnp.nanmean(data) if red == 'mean' else jnp.sqrt(jnp.nanmean(data**2)))
                        ref_val = jnp.max(ref_arr) if red == 'max' else (jnp.mean(ref_arr) if red == 'mean' else jnp.sqrt(jnp.mean(ref_arr**2)))
                        scale = jnp.maximum(jnp.abs(ref_val), 1e-4); terr = (val - ref_val) / scale if comp_mode == 'relative' else (val - ref_val)
                        opt_loss += wgt * (terr ** 2)
                    elif typ == 'rbe_reaction':
                        fd = case_bcs[idx]['fd']; val_f, ref_f = data[fd], ref_arr[fd]
                        ref_rms = jnp.sqrt(jnp.mean(ref_f**2)); scale = jnp.maximum(ref_rms, 1e-3)
                        terr_arr = (val_f - ref_f) / scale if comp_mode == 'relative' else (val_f - ref_f)
                        opt_loss += wgt * jnp.mean(terr_arr ** 2)
                        val, ref_val, terr = jnp.sqrt(jnp.mean(val_f**2)), ref_rms, jnp.sqrt(jnp.mean(terr_arr ** 2))
                else:
                    if typ == 'mass':
                        t_2d, rho_2d = p_fem['t'].reshape(Nx_l+1, Ny_l+1), p_fem['rho'].reshape(Nx_l+1, Ny_l+1)
                        dx_v, dy_v = self.fem.Lx / Nx_l, self.fem.Ly / Ny_l
                        W = jnp.ones_like(t_2d).at[0, :].multiply(0.5).at[-1, :].multiply(0.5).at[:, 0].multiply(0.5).at[:, -1].multiply(0.5)
                        val = jnp.sum(t_2d * rho_2d * W) * dx_v * dy_v
                        ref_val, tol = float(desc.get('ref_value', self.target_mass)), float(desc.get('tolerance', 0.05))
                        rel_err = jnp.abs(val - ref_val) / (jnp.abs(ref_val) + 1e-12)
                        terr = rel_err if comp_mode == 'relative' else jnp.abs(val - ref_val)
                        opt_loss += wgt * (rel_err ** 2 * 0.1 + jnp.maximum(0.0, rel_err - tol) ** 2 * 100.0)
                    elif typ == 'modes' and vecs_filtered is not None:
                        cand_modes = vecs_filtered[2::6, :]; nmode = int(desc.get('num_modes') or len(t_vals))
                        eps_mac = 1e-10; t_n = t_modes_l / jnp.sqrt(jnp.sum(t_modes_l**2, axis=0, keepdims=True) + eps_mac)
                        c_n = cand_modes / jnp.sqrt(jnp.sum(cand_modes**2, axis=0, keepdims=True) + eps_mac)
                        mac_matrix = jnp.dot(t_n.T, c_n)**2; best_mac = jnp.max(mac_matrix, axis=1); best_match_idx = jnp.argmax(mac_matrix, axis=1)
                        if desc.get('sub_type') == 'mac':
                            val, ref_val = jnp.mean(best_mac[:nmode]), 1.0; terr = jnp.mean((1.0 - best_mac[:nmode])**2)
                        else:
                            matched_freqs = freqs[best_match_idx[:nmode]]
                            val, ref_val = jnp.mean(matched_freqs), jnp.mean(t_vals[:nmode])
                            terr = jnp.mean(((matched_freqs - t_vals[:nmode]) / (t_vals[:nmode] + 1e-12))**2)
                        opt_loss += wgt * terr
                target_evals.append(jnp.stack([val, ref_val, terr, wgt]))
    
            l_reg = 0.0
            for k in ['t', 'pz']:
                if k in p_scaled and p_scaled[k].ndim > 1:
                    p_min, p_max = opt_config[k].get('min', 0.0) / scaling_consts[k], opt_config[k].get('max', 1.0) / scaling_consts[k]
                    p_range = jnp.maximum(p_max - p_min, 1e-3)
                    l_reg += (jnp.mean(jnp.diff(p_scaled[k], axis=0)**2) + jnp.mean(jnp.diff(p_scaled[k], axis=1)**2)) / (p_range**2)
            opt_loss += l_reg * reg_weight
            return opt_loss, {'Total': opt_loss, 'Reg': l_reg * reg_weight, 'target_evals': jnp.stack(target_evals) if target_evals else jnp.zeros((0,4)), 'freqs': freqs, 'vecs_filtered': vecs_filtered, 'matched_freqs': matched_freqs, 'best_mac': best_mac, 'f1_hz': freqs[0] if freqs is not None and len(freqs)>0 else 0.0}
    
        loss_vg_with_eigen = jax.jit(jax.value_and_grad(lambda p, fp: loss_fn(p, fp, True, None, None), has_aux=True))
        loss_vg_no_eigen = jax.jit(jax.value_and_grad(lambda p, fp, cf, cv: loss_fn(p, fp, False, cf, cv), has_aux=True))
        
        best_loss, best_params, wait = float('inf'), jax.tree_util.tree_map(jnp.array, params), 0
        self.history = []; cached_freqs, cached_vecs = None, None
        last_matched_freqs, last_best_mac = None, None
        has_mode_target = any('modes' in str(ot.get('type','')).lower() for ot in opt_target_config.get('global_targets', []))
        
        print("\n" + "═"*115); print(f" {'Optimization Progress Log':^115}"); print("═"*115)
        print(f" {'Status':<16} | {'Time':<7} | {'Current Loss':<14} | {'Convergence Metrics / Physical State':<40}"); print("─"*115)
    
        for i in range(max_iterations):
            iter_start = time.time()
            if i == 0: print(f"\n [Iter {i:04d}/{max_iterations}] Initializing and Compiling JAX graph...")
            else: print(f" [Iter {i:04d}/{max_iterations:04d}] Optimizing... ", end="", flush=True)
            
            do_eigen = (i == 0) or ((i % eigen_freq == 0) and has_mode_target)
            if do_eigen:
                (val, aux), grads = loss_vg_with_eigen(params, fixed_params_scaled)
                cached_freqs, cached_vecs = aux['freqs'], aux['vecs_filtered']
                last_matched_freqs, last_best_mac = aux['matched_freqs'], aux['best_mac']
            else:
                (val, aux), grads = loss_vg_no_eigen(params, fixed_params_scaled, cached_freqs, cached_vecs)
                if aux.get('matched_freqs') is None: aux['matched_freqs'], aux['best_mac'] = last_matched_freqs, last_best_mac
            
            if val < best_loss - early_stop_tol: best_loss, wait, best_params = val, 0, jax.tree_util.tree_map(jnp.array, params)
            else: wait += 1
            
            hist_dict = {'Total': float(val), 'Reg': float(aux.get('Reg', 0.0))}
            evals_np = np.array(aux['target_evals'])
            for d_idx, info in enumerate(target_info):
                k = info['field'] if info['field'] else info['type']
                hist_dict[k] = hist_dict.get(k, 0.0) + float(evals_np[d_idx, 2] * evals_np[d_idx, 3])
            self.history.append(hist_dict)
            
            updates, opt_state = optimizer.update(jax.tree_util.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads), opt_state)
            params = optax.apply_updates(params, updates)
            for k in params:
                if k in opt_config: params[k] = jnp.clip(params[k], opt_config[k].get('min', -1e12)/scaling_consts[k], opt_config[k].get('max', 1e12)/scaling_consts[k])
            
            if i > 0:
                dt_max = float(jnp.abs(params['t'] - prev_params['t']).max() * scaling_consts['t'])
                dpz_max = float(jnp.abs(params['pz'] - prev_params['pz']).max() * scaling_consts['pz']) if 'pz' in params else 0.0
                dt_avg = float(jnp.abs(params['t'] - prev_params['t']).mean() * scaling_consts['t'])
                dpz_avg = float(jnp.abs(params['pz'] - prev_params['pz']).mean() * scaling_consts['pz']) if 'pz' in params else 0.0
                print(f" 🔄 [Iter {i:04d}/{max_iterations:04d}] {time.time()-iter_start:6.2f}s | Loss: {float(val):12.6e}")
                print(f"    ╰─> Sens: dt:{dt_max:8.6f}(avg:{dt_avg:10.7f}) | dz:{dpz_max:7.4f}(avg:{dpz_avg:8.5f}) | t_avg:{float(jnp.mean(params.get('t', 0.0))) * scaling_consts['t']:7.4f}mm")
            prev_params = jax.tree_util.tree_map(jnp.array, params)
            
            if i % 10 == 0 or i == 0:
                evals = np.array(aux['target_evals'])
                if not hasattr(self, '_prev_report_errors'): self._prev_report_errors = np.zeros(len(target_info))
                table = WHTable(["Case", "Type", "Field", "Target Val", "Current Val", "Error", "Delta", "Status"], title=f"OPTIMIZATION TARGET REPORT [ITER {i:04d}]")
                table.set_aligns(['left', 'left', 'left', 'right', 'right', 'right', 'center', 'center'])
                for d_idx, info in enumerate(target_info):
                    v, r_v, err, w = evals[d_idx]; e_diff = abs(float(err)) - abs(float(self._prev_report_errors[d_idx]))
                    delta = "🟢 ▼" if e_diff < -1e-5 else ("🔴 ▲" if e_diff > 1e-5 else "⚪ =")
                    if i == 0: delta = "-"
                    status = "✅" if abs(err) < 0.05 else ("🟡" if abs(err) < 0.20 else "❌")
                    table.add_row([info['case'], info['type'], info['field'], f"{r_v:.4e}", f"{v:.4e}", f"{err:+.4f}", delta, status])
                self._prev_report_errors = np.array([abs(float(e[2])) for e in evals]); table.print()
                if cached_freqs is not None:
                    m_freqs_raw = aux.get('matched_freqs')
                    c_hz = np.atleast_1d(np.array(m_freqs_raw if m_freqs_raw is not None else cached_freqs))
                    t_hz = np.atleast_1d(np.array(t_vals))
                    best_mac_raw = aux.get('best_mac')
                    best_mac = np.atleast_1d(np.array(best_mac_raw if best_mac_raw is not None else np.zeros_like(t_hz)))
                    
                    if c_hz[0] is not None:
                        n_f = min(len(c_hz), len(t_hz))
                        m_table = WHTable(["Mode", "Target (Hz)", "Current (Hz)", "Error (%)", "MAC", "Quality"], title="MODAL PERFORMANCE ANALYTICS")
                        for m_idx in range(n_f):
                            if c_hz[m_idx] is not None:
                                err_hz, mac_val = (c_hz[m_idx]-t_hz[m_idx])/t_hz[m_idx]*100, float(best_mac[m_idx])
                                q = "Excellent" if mac_val > 0.95 else ("High" if mac_val > 0.85 else ("Good" if mac_val > 0.5 else "Poor"))
                                m_table.add_row([f"#{m_idx+1}", f"{t_hz[m_idx]:.2f}", f"{c_hz[m_idx]:.2f}", f"{err_hz:+.2f}%", f"{mac_val:.4f}", f"{q} Quality"])
                        m_table.print()
                print(f"💰 [SYSTEM] Total Loss (with Reg): {float(val):.6e}")
            if msvcrt.kbhit() and msvcrt.getch() in [b'q', b'Q']: params = best_params; break
            if use_early_stopping and wait >= early_stop_patience: params = best_params; break
        self.optimized_params = {**{k: v * scaling_consts[k] for k, v in params.items()}, **{k: v * scaling_consts[k] for k, v in fixed_params_scaled.items()}}
        print(f"\n -> Optimization Finished. Best Loss: {best_loss:.6f}"); return self.optimized_params

    def verify(self):
        print("\n" + "="*70); wh_print_banner("STAGE 4: COMPREHENSIVE FINAL VERIFICATION"); print("="*70)
        def get_metrics(ref, opt):
            rmse = np.sqrt(np.mean((ref - opt)**2)); drange = np.max(ref) - np.min(ref) + 1e-12
            return rmse, max(0.0, 100.0 * (1.0 - rmse/drange))
        Nx_h, Ny_h = self.resolution_high; xh, yh = np.linspace(0, self.fem.Lx, Nx_h+1), np.linspace(0, self.fem.Ly, Ny_h+1)
        Xh, Yh = np.meshgrid(xh, yh, indexing='xy'); xl, yl = np.linspace(0, self.fem.Lx, self.fem.nx+1), np.linspace(0, self.fem.Ly, self.fem.ny+1)
        Xl, Yl = np.meshgrid(xl, yl, indexing='xy'); pts_low, pts_high = np.column_stack([Xl.flatten(), Yl.flatten()]), np.column_stack([Xh.flatten(), Yh.flatten()])
        def get_interp(k):
            val = np.array(self.optimized_params.get(k, jnp.zeros_like(self.optimized_params['t'] if k != 'pz' else self.optimized_params['t'])))
            if val.ndim == 1 and val.shape[0] == 1: return np.full((Ny_h+1, Nx_h+1), val[0]).flatten()
            return griddata(pts_low, val.flatten(), (Xh, Yh), method='linear').flatten()
        opt_params_h = {k: get_interp(v) for k, v in {'t':'t','rho':'rho','E':'E','z':'pz'}.items()}
        print("Solving Optimized High-Res Model..."); K_opt, M_opt = self.fem_high.assemble(opt_params_h, sparse=True)
        static_summary = []
        for i, case in enumerate(self.cases):
            tgt = self.targets[i]; print(f" -> Verifying: {case.name}")
            fd, fv, F = case.get_bcs(self.fem_high); free = np.setdiff1d(np.arange(self.fem_high.total_dof), fd)
            u_opt = self.fem_high.solve_static_sparse(K_opt, F, free, fd, fv)
            w_ref, w_opt = tgt['u_static'].reshape(Ny_h+1, Nx_h+1), u_opt[2::6].reshape(Ny_h+1, Nx_h+1)
            f_opt = self.fem_high.compute_field_results(u_opt, opt_params_h)
            stress_ref, stress_opt = tgt['max_stress'].reshape(Ny_h+1, Nx_h+1), np.array(f_opt['stress_vm']).reshape(Ny_h+1, Nx_h+1)
            strain_ref, strain_opt = tgt['max_strain'].reshape(Ny_h+1, Nx_h+1), np.array(f_opt['strain_equiv_nodal']).reshape(Ny_h+1, Nx_h+1)
            sim_w = get_metrics(w_ref, w_opt)[1]; sim_s = get_metrics(stress_ref, stress_opt)[1]; sim_e = get_metrics(strain_ref, strain_opt)[1]
            static_summary.append({'name':case.name, 'disp_sim':sim_w, 'stress_sim':sim_s, 'strain_sim':sim_e})
            fig, axes = plt.subplots(3, 3, figsize=(15, 12)); fig.suptitle(f"Verification: {case.name} | Sim: {sim_w:.1f}%", fontsize=12, fontweight='bold')
            def plot_f(ax, data, title, cmap='jet', levels=None):
                im = ax.contourf(xh, yh, data, levels=levels if levels is not None else 30, cmap=cmap)
                ax.set_title(title); ax.set_aspect('equal'); plt.colorbar(im, ax=ax)
            lv_w = np.linspace(np.min(w_ref), np.max(w_ref), 30); plot_f(axes[0,0], w_ref, "Target Disp", levels=lv_w)
            plot_f(axes[0,1], w_opt, "Opt Disp", levels=lv_w); plot_f(axes[0,2], np.abs(w_opt-w_ref), "Error", cmap='YlOrRd')
            lv_s = np.linspace(0, np.max(stress_ref), 30); plot_f(axes[1,0], stress_ref, "Target Stress", levels=lv_s)
            plot_f(axes[1,1], stress_opt, "Opt Stress", levels=lv_s); plot_f(axes[1,2], np.abs(stress_opt-stress_ref), "Error", cmap='YlOrRd')
            lv_e = np.linspace(0, np.max(strain_ref)*1000, 30); plot_f(axes[2,0], strain_ref*1000, "Target Strain", levels=lv_e)
            plot_f(axes[2,1], strain_opt*1000, "Opt Strain", levels=lv_e); plot_f(axes[2,2], np.abs(strain_opt-strain_ref)*1000, "Error", cmap='YlOrRd')
            plt.savefig(f"verify_3d_{case.name}.png", dpi=150); plt.close()
            res_opt = StructuralResult({'displacement_vec':u_opt, 'stress_vm':f_opt['stress_vm'], 'strain_equiv':f_opt['strain_equiv_nodal'], 'sed':f_opt['sed']}, np.array(self.fem_high.nodes), self.fem_high.elements)
            res_opt.save_vtkhdf(f"opt_static_{case.name}.vtkhdf")
        n_modes = len(self.target_eigen['vals']); vals_opt, vecs_opt = self.fem_high.solve_eigen_sparse(K_opt, M_opt, num_modes=n_modes + 10)
        s_idx, freq_ref, freq_opt = 6, self.target_eigen['vals'], np.array(vals_opt[s_idx : s_idx + n_modes])
        modes_ref, modes_opt = self.target_eigen['modes'], np.array(vecs_opt[2::6, s_idx : s_idx + n_modes])
        macs = [float((np.dot(modes_opt[:,j], modes_ref[:,j])**2)/(np.dot(modes_opt[:,j],modes_opt[:,j])*np.dot(modes_ref[:,j],modes_ref[:,j])+1e-12)) for j in range(n_modes)]
        report = ["# Optimization Verification Report", f"**Date:** {datetime.datetime.now()}", "## Static Results", "| Case | Disp Sim | Stress Sim |"]
        for s in static_summary: report.append(f"| {s['name']} | {s['disp_sim']:.1f}% | {s['stress_sim']:.1f}% |")
        report.extend(["\n## Modal Results", "| Mode | Ref Freq | Opt Freq | MAC |"])
        for j in range(n_modes): report.append(f"| {j+1} | {freq_ref[j]:.1f} | {freq_opt[j]:.1f} | {macs[j]:.3f} |")
        with open("verification_report.md", "w") as f: f.write("\n".join(report))
        K_l, M_l = self.fem.assemble(self.optimized_params, sparse=True); freq_l, vecs_l = self.fem.solve_eigen_sparse(K_l, M_l, num_modes=n_modes)
        stage3_visualize_comparison(self.fem_high, self.fem, self.targets, self.optimized_params, self.target_params_high, opt_eigen={'vals':(2*np.pi*freq_l)**2, 'modes':vecs_l[2::6, :n_modes]}, tgt_eigen=self.target_eigen)

if __name__ == '__main__':

    '''
    # 1. 기본 실행 (50회 반복)
    python main_shell_opt.py --run

    # 2. 반복 횟수 지정 및 자동 모드 (150회)
    python main_shell_opt.py --run --max_iter 150 --non-interactive

    # 3. 도움말 확인
    python main_shell_opt.py --help
    '''

    import argparse
    parser = argparse.ArgumentParser(
        description="[WHTOOLS] Shell Topography Optimization Pipeline (V2)",
        epilog="Examples:\n  python main_shell_opt.py --run --max_iter 150\n  python main_shell_opt.py --run --non-interactive",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--run', action='store_true', help='최적화 파이프라인 전체 실행')
    parser.add_argument('--max_iter', type=int, default=50, help='최대 최적화 반복 횟수 (default: 50)')
    parser.add_argument('--non-interactive', action='store_true', help='사용자 입력 대기 없이 자동 실행')
    parser.add_argument('--use_cache', action='store_true', help='기존 Ground Truth 캐시(target_cache.pkl) 강제 사용')
    parser.add_argument('--tray_h', type=float, default=50.0, help='Tray 외곽 벽 높이 (default: 50.0)')
    parser.add_argument('--cores', type=int, default=4, help='JAX/XLA가 사용할 CPU 코어(스레드) 수 (default: 6)')
    
    args = parser.parse_args()

    if not args.run:
        parser.print_help()
        print("\n[INFO] 최적화를 실행하려면 --run 인자를 추가하십시오.")
        sys.exit(0)

    if args.non_interactive:
        os.environ["NON_INTERACTIVE"] = "1"

    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={args.cores} --xla_backend_opt_level=1"
    wh_print_banner("WHTOOLS SHELL OPTIMIZATION SYSTEM START")
    
    model = EquivalentSheetModel(Lx, Ly, Nx_low, Ny_low, wall_height=args.tray_h)
    model.add_case(TwistCase("twist_x", axis='x', value=1.5, mode='angle', weight=1.0))
    model.add_case(TwistCase("twist_y", axis='y', value=1.5, mode='angle', weight=1.0))
    model.add_case(PureBendingCase("bend_y", axis='y', value=3.0, mode='angle', weight=1.0))
    model.add_case(PureBendingCase("bend_x", axis='x', value=3.0, mode='angle', weight=1.0))
    model.add_case(CornerLiftCase("lift_br", corner='br', value=5.0, mode='disp', weight=1.0))
    model.add_case(CornerLiftCase("lift_tl", corner='tl', value=5.0, mode='disp', weight=1.0))
    model.add_case(TwoCornerLiftCase("lift_tl_br", corners=['br', 'tl'], value=5.0, mode='disp', weight=1.0))
    model.add_case(PressureCase("pressure_z", value=-10.0, weight=1.0))
    
    opt_target_config = {
        "twist_x": { "opt_targets": [
            {"target_type": "field_stat", "field": "u_static", "reduction": "mse", "compare_mode": "relative", "weight": 2.0},
            {"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 1.0},
            {"target_type": "field_stat", "field": "stress_vm", "reduction": "mse", "compare_mode": "relative", "weight": 0.5}
        ]},
        "twist_y": { "opt_targets": [
            {"target_type": "field_stat", "field": "u_static", "reduction": "mse", "compare_mode": "relative", "weight": 2.0},
            {"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 1.0},
            {"target_type": "field_stat", "field": "stress_vm", "reduction": "mse", "compare_mode": "relative", "weight": 0.5}
        ]},
        "bend_y": { "opt_targets": [
            {"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 2.0},
            {"target_type": "field_stat", "field": "max_strain", "reduction": "mse", "compare_mode": "relative", "weight": 0.5}
        ]},
        "bend_x": { "opt_targets": [
            {"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 2.0},
            {"target_type": "field_stat", "field": "max_strain", "reduction": "mse", "compare_mode": "relative", "weight": 0.5}
        ]},
        "lift_br": {"opt_targets": [{"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 1.0}]},
        "lift_tl": {"opt_targets": [{"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 1.0}]},
        "lift_tl_br": {"opt_targets": [{"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 1.0}]},        
        "pressure_z": {"opt_targets": [{"target_type": "field_stat", "field": "u_static", "reduction": "rms", "compare_mode": "relative", "weight": 1.0}]},
        "global_targets": [{"target_type": "modes", "compare_mode": "mac", "num_modes": 3, "freq_weight": 0.5, "weight": 5.0}]
    }
    target_config = { 'pattern': 'ABC', 'base_t': 1.0, 'bead_t': {'A': 2.0, 'B': 2.5, 'C': 1.5}, 'pattern_pz': 'TNYBV', 'bead_pz': {'T': 12.0, 'N': 10.0, 'Y': 15.0, 'B': 12.0, 'V': 4.0}, 'base_rho': 7.85e-9, 'base_E': 210000.0 }
    
    use_cache_req = args.use_cache
    cache_file = "target_cache.pkl"
    if not use_cache_req and os.path.exists(cache_file):
        if os.environ.get("NON_INTERACTIVE") == "1": use_cache_req = False
        else:
            print(f"\n[CACHE CHECK] 기존 Ground Truth 데이터({cache_file})가 존재합니다.")
            choice = input(" -> 캐시를 로드하시겠습니까? [y/N]: ").strip().upper()
            use_cache_req = (choice == 'Y')
    
    model.generate_targets(resolution_high=(Nx_high, Ny_high), num_modes_save=3, target_config=target_config, use_cache_override=use_cache_req, wall_width=5.0, wall_height=5.0)
    opt_config = { 't': {'opt': True, 'init': 1.0, 'min': 0.8, 'max': 1.2, 'type': 'local'}, 'rho': {'opt': True, 'init': 7.85e-9, 'min': 1e-10, 'max': 1e-7, 'type': 'local'}, 'E': {'opt': False, 'init': 210000.0, 'type': 'local'}, 'pz': {'opt': True, 'init': 0.0, 'min': -20.0, 'max': 20.0, 'type': 'local'} }
    
    try:
        wh_print_banner(f"RUNNING OPTIMIZATION (MAX_ITER: {args.max_iter})")
        model.optimize_v2(opt_config, opt_target_config, max_iterations=args.max_iter, use_bead_smoothing=True, use_early_stopping=True, early_stop_patience=40, learning_rate=0.5, min_bead_width=150.0, init_pz_from_gt=True, gt_init_scale=0.0, eigen_freq=3, eigen_solver='lobpcg')               
        model.verify()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[CRITICAL ERROR] Optimization failed: {e}")