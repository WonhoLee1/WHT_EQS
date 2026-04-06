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
import pickle

# Import our modular components
from ShellFemSolver.shell_solver import ShellFEM
from ShellFemSolver.mesh_utils import generate_rect_mesh_quads, generate_tray_mesh_quads

def PlateFEM(Lx, Ly, nx, ny):
    # 'vertical' 또는 'sloped' 모드를 선택할 수 있습니다.
    nodes, elements = generate_tray_mesh_quads(Lx, Ly, wall_width=50.0, wall_height=50.0, nx=nx, ny=ny, mode='vertical')
    fem = ShellFEM(nodes, elements)
    fem.Lx, fem.Ly, fem.nx, fem.ny = Lx, Ly, nx, ny
    return fem
from WHT_EQS_pattern_generator import get_thickness_field, get_z_field
from WHT_EQS_load_cases import TwistCase, PureBendingCase, CornerLiftCase, TwoCornerLiftCase
from WHT_EQS_visualization import (
    stage1_visualize_patterns,
    stage2_visualize_ground_truth,
    stage3_visualize_comparison
)

# Mesh Settings
Lx, Ly = 1450.0, 850.0
Nx_high, Ny_high = int(Lx/30.), int(Ly/30.)      # High-res ground truth
#Nx_high, Ny_high = int(Lx/60.), int(Ly/60.)      # High-res ground truth
Nx_low, Ny_low = int(Lx/60.), int(Ly/60.)        # Optimization mesh resolution sync

class EquivalentSheetModel:
    def __init__(self, Lx, Ly, nx, ny):
        self.fem = PlateFEM(Lx, Ly, nx, ny) # 호환성 함수 호출 (ShellFEM)
        self.cases = []
        self.targets = []
        self.resolution_high = (50, 20)
        self.target_mass = 0.0
        self.target_start_idx = 6 # Default for Free-Free
        self.optimized_params = None
        self.target_params_high = None
        self.config = {}

    def add_case(self, case):
        self.cases.append(case)

    def generate_targets(self, resolution_high=(50, 20), 
                        num_modes_save=5, 
                        target_config=None,
                        cache_file="ground_truth_cache.pkl"
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
        Xh, Yh = np.meshgrid(xh, yh, indexing='xy')

        # --- Cache Check ---
        use_cache = False
        if os.environ.get("NON_INTERACTIVE"):
            use_cache = os.path.exists(cache_file)
            print(f"[NON-INTERACTIVE] File exists: {use_cache}. Loading if possible.")
        elif os.path.exists(cache_file):
            print(f"\n[CACHE DETECTED] Found existing Ground Truth data: {cache_file}")
            while True:
                choice = input("Do you want to load from cache? [y/N]: ").strip().upper()
                if choice == 'Y':
                    use_cache = True
                    break
                elif choice in ('N', ''):
                    use_cache = False
                    break
        
        if use_cache:
            try:
                print("Loading Ground Truth from cache...")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self.target_params_high = cache_data['target_params_high']
                self.target_mass = cache_data['target_mass']
                self.targets = cache_data['targets']
                self.target_eigen = cache_data['target_eigen']
                
                # Assign back to variables needed for visualization
                params_high = self.target_params_high
                t_h = params_high['t']
                z_h = params_high['z']
                
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
            # This ensures visualization shows the 3D tray shape, not just the flat pattern.
            base_z_h = fem_high.nodes[:, 2].reshape(Xh.shape)
            z_h_full = base_z_h + z_h
            
            params_high = {
                't': t_h.flatten(), 
                'z': z_h.flatten(), # Solver uses 'add' logic now, so we keep the relative pattern here
                'rho': rho_h.flatten(), 
                'E': E_h.flatten()
            }
            self.target_params_high = params_high
            
            # Calculate target mass
            dx, dy = self.fem.Lx/Nx_h, self.fem.Ly/Ny_h
            self.target_mass = float(jnp.sum(t_h * rho_h) * dx * dy)
            
        # --- Visualization must happen AFTER cache/generation is settled ---
        # 2.5 Combine Pattern Z (pz) with Base Mesh Z (Tray Height) for 3D View
        # This ensures visualization shows the 3D tray shape, even if loading from an old flat cache.
        t_h = self.target_params_high['t'].reshape(Xh.shape)
        z_h = self.target_params_high['z'].reshape(Xh.shape)
        base_z_h = self.fem_high.nodes[:, 2].reshape(Xh.shape)
        z_h_full = base_z_h + z_h
        
        # [STAGE 1] Interactive Pattern Check - Always use latest mesh coords
        stage1_visualize_patterns(Nx_h, Ny_h, Xh, Yh, t_h, z_h_full)
            
        # 3. Solve FEM for each load case (High Fidelity - Sparse Solve)
        print("\nSolving High-Resolution Ground Truth (Sparse)...")
        K_h, M_h = fem_high.assemble(self.target_params_high, sparse=True)
        self.targets = []
        
        for case in self.cases:
            print(f" -> Solving Case: {case.name}")
            fixed_dofs, fixed_vals, F = case.get_bcs(fem_high)
            free_dofs = np.setdiff1d(np.arange(fem_high.total_dof), fixed_dofs)
            u = fem_high.solve_static_sparse(K_h, F, free_dofs, fixed_dofs, fixed_vals)
            
            # Compute Full Reaction Forces (Sparse Dot)
            F_int = K_h.dot(np.array(u))
            R_residual = F_int - np.array(F)
            
            self.targets.append({
                'case_name': case.name,
                'weight': case.weight,
                'u_static': np.array(u[2::6]), # W-displacement only
                'u_full': np.array(u),         # Full 6-DOF displacement
                'reaction_full': np.array(R_residual), # Full reaction force vector
                'max_surface_stress': np.array(fem_high.compute_max_surface_stress(u, params_high, K=K_h)),
                'max_surface_strain': np.array(fem_high.compute_max_surface_strain(u, params_high, K=K_h)),
                'strain_energy_density': np.array(fem_high.compute_strain_energy_density(u, params_high, K=K_h)),
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
                # Node locations for reshaping
                # Node (i,j) has index j*(Nx_h+1) + i
                # So to reshape back, we can use (Nx_h+1, Ny_h+1, ...) with order='F'
                
                # Row 0: Displacement
                data_w = target['u_static'].reshape(Ny_h+1, Nx_h+1)
                im0 = axes[0, i].contourf(xh, yh, data_w, 30, cmap='jet')
                axes[0, i].set_title(f"Case: {target['case_name']}\nMax Disp: {np.max(np.abs(data_w)):.3f}mm", fontsize=8)
                axes[0, i].set_aspect('equal')
                plt.colorbar(im0, ax=axes[0, i], shrink=0.7)
                
                # Row 1: Max Surface Stress
                data_s = target['max_surface_stress'].reshape(Ny_h+1, Nx_h+1)
                im1 = axes[1, i].contourf(xh, yh, data_s, 30, cmap='jet')
                axes[1, i].set_title(f"Max Stress: {np.max(data_s):.2f} MPa", fontsize=8)
                axes[1, i].set_aspect('equal')
                plt.colorbar(im1, ax=axes[1, i], shrink=0.7)
                
                # Row 2: Max Surface Strain
                data_e = target['max_surface_strain'].reshape(Ny_h+1, Nx_h+1) * 1000 # to microstrain/e-3
                im2 = axes[2, i].contourf(xh, yh, data_e, 30, cmap='jet')
                axes[2, i].set_title(f"Max Strain: {np.max(data_e):.3f} e-3", fontsize=8)
                axes[2, i].set_aspect('equal')
                plt.colorbar(im2, ax=axes[2, i], shrink=0.7)
                
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig("ground_truth_3d_loadcases.png", dpi=150)
            plt.close()
            print(" -> Saved: ground_truth_3d_loadcases.png")
            
            # 4. Solve Eigenmodes (Sparse for speed - Already assembled)
            print("Solving Target Eigenmodes (Sparse computation)...")
            # K_h, M_h already assembled as sparse above
            vals, vecs = fem_high.solve_eigen_sparse(K_h, M_h, num_modes=num_modes_save + 10)
            
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
            
            # Save the recalculation to cache
            print(f"Saving Ground Truth to cache ({cache_file})...")
            cache_data = {
                'target_params_high': self.target_params_high,
                'target_mass': self.target_mass,
                'targets': self.targets,
                'target_eigen': self.target_eigen
            }
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
            except Exception as e:
                print(f"[WARNING] Failed to save cache: {e}")

        # Pre-output target frequencies for user overview
        target_freqs = np.sqrt(np.maximum(np.array(self.target_eigen['vals']), 0.0)) / (2 * np.pi)
        print(" -> Target frequencies (Hz): " + ", ".join([f"{f:.2f}" for f in target_freqs]))
        
        # [STAGE 2] Results Visualization
        stage2_visualize_ground_truth(fem_high, self.targets, self.target_params_high, eigen_data=self.target_eigen)

    def optimize(self, opt_config, loss_weights, 
                 use_bead_smoothing=False, 
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
                 min_bead_width=150.0,
                 mac_search_window=2,
                 mode_match_type='hybrid',
                 init_pz_from_gt=False,
                 gt_init_scale=1.0):
        """
        [위상 최적화 (Topography Optimization) 수행기]

        매개변수 설명:
        - opt_config          : 최적화할 변수(t, rho, E, pz)의 초기값, 상하한선 및 최적화 켜기/끄기 설정 딕셔너리.
        - loss_weights        : 각 평가 지표(Displacement, Energy, Freq, MAC, Mass 등)의 손실 함수 반영 가중치.
        - use_bead_smoothing  : (매우 중요) 독립 노드가 톱니나 모자이크처럼 깨지는 물리적 편법(Checkerboarding)을 막고 
                                부드러운 산등성이 모양의 비드(Bead)가 형성되도록 강제하는 필터 사용 여부.
        - use_strain_energy   : 손실 함수에 변형 에너지 밀도(SED) 오차를 포함할지 여부 (미분 경로가 완벽하여 켜두는 것을 강력 권장).
        - use_surface_stress  : 응력 오차 반영 여부 (현재 JAX 미분 구조상 pz 변수 사용 시 비활성화 권장).
        - use_surface_strain  : 변형률 오차 반영 여부 (동일 사유로 비활성화 권장).
        - use_mass_constraint : 질량 증가를 방지하기 위한 페널티 부여 여부.
        - mass_tolerance      : 타겟 모델 대비 허용되는 최대 질량 오차율 (예: 0.05 = 5%). 넘어갈 시 거대한 페널티 부과.
        - max_iterations      : JAX 옵티마이저(Adam)가 수행할 최대 반복(런) 횟수.
        - use_early_stopping  : 손실값이 개선되지 않을 때 조기에 최적화를 종료할지 여부.
        - early_stop_patience : 손실값이 개선되지 않는 상태를 몇 번의 반복 횟수까지 봐줄 것인지 (예: 100번).
        - early_stop_tol      : 손실값 비교 시 유의미한 개선이라고 판단할 최소 변화량 오차 한계.
        - learning_rate       : Adam 옵티마이저의 최대 학습률 (Cosine Decay로 점진적으로 스케줄링됨).
        - num_modes_loss      : 주파수 및 MAC 손실 함수에 반영할 모드의 개수 (None이면 타겟 모드 개수와 동일).
        - min_bead_width      : use_bead_smoothing이 True일 때, 비드의 최소 폭(mm). (기본 추천값: 150.0mm 이상)
        - mac_search_window   : 모드 추적(Mode Tracking) 시 몇 번째까지 이웃한 모드들과 MAC를 비교하여 형태를 찾을지 결정.
        - mode_match_type     : 모드 형상 비교 방식 설정 ('mac', 'direct', 'hybrid'). 
                                'mac': 각도 기반 비교 (기울기 소실 우려), 
                                'direct': 노드별 직접 MSE 매칭 (매우 민감함), 
                                'hybrid': MAC와 Direct를 적절한 비율로 혼합 (기본값).
        - init_pz_from_gt     : [NEW 옵션] 최적화 초기 형상(pz_init)을 타겟 Ground Truth 형상으로 덮어쓸지 여부.
                                (초기부터 비슷한 형태를 갖추고 시작하여 로컬 미니마 탈출 및 빠른 수렴 가능)
        - gt_init_scale       : init_pz_from_gt=True 시, 타겟 변위를 얼마나 가져올지에 대한 곱셈 비율 (기본 1.0 = 100%)
        """
                 
        print("\n" + "="*70)
        print(" [STAGE 3] ADVANCED OPTIMIZATION (EXPLICIT LOGIC)")
        print(f" -> Setting: Min Bead Width = {min_bead_width} mm")
        if not use_bead_smoothing:
            print(" -> WARNING: Bead smoothing (use_bead_smoothing=False) is disabled. Results may show numerical checkerboarding.")
        print("="*70)
        
        Nx_l, Ny_l = self.fem.nx, self.fem.ny
        pts_l = self.fem.node_coords[:, :2]  # X,Y only for griddata
        
        # [NEW] Save optimization configuration for later use (e.g., in verify())
        self.config = {
            'mode_match_type': mode_match_type,
            'num_modes_loss': num_modes_loss,
            'use_bead_smoothing': use_bead_smoothing
        }
        
        # 1. Interpolate High-Res Targets to Low-Res Mesh
        Nx_h, Ny_h = self.resolution_high
        xh, yh = jnp.linspace(0, self.fem.Lx, Nx_h+1), jnp.linspace(0, self.fem.Ly, Ny_h+1)
        Xh, Yh = jnp.meshgrid(xh, yh, indexing='xy')
        pts_h = jnp.stack([Xh.flatten(), Yh.flatten()], axis=1)
        
        self.targets_low = []
        is_same_res = (Nx_h == Nx_l and Ny_h == Ny_l)
        if is_same_res:
            print(" -> Resolution Match (Target Low-Res): Skipping griddata for bit-exact mapping.")

        for tgt in self.targets:
            if is_same_res:
                u_l = np.array(tgt['u_static'])
                stress_l = np.array(tgt['max_surface_stress'])
                strain_l = np.array(tgt['max_surface_strain'])
                sed_l = np.array(tgt['strain_energy_density'])
            else:
                u_l = griddata(pts_h, tgt['u_static'], pts_l, method='linear')
                stress_l = griddata(pts_h, tgt['max_surface_stress'], pts_l, method='linear')
                strain_l = griddata(pts_h, tgt['max_surface_strain'], pts_l, method='linear')
                sed_l = griddata(pts_h, tgt['strain_energy_density'], pts_l, method='linear')
            
            R_tgt = tgt['reaction_full']
            target_R_sums = jnp.array([
                jnp.sum(jnp.abs(R_tgt[0::6])),
                jnp.sum(jnp.abs(R_tgt[1::6])),
                jnp.sum(jnp.abs(R_tgt[2::6])),
                jnp.sum(jnp.abs(R_tgt[3::6])),
                jnp.sum(jnp.abs(R_tgt[4::6])),
                jnp.sum(jnp.abs(R_tgt[5::6]))
            ])
            
            self.targets_low.append({
                'u_static': jnp.array(u_l),
                'max_stress': jnp.array(stress_l),
                'max_strain': jnp.array(strain_l),
                'strain_energy_density': jnp.array(sed_l),
                'reaction_sums': target_R_sums,
                'weight': tgt.get('weight', 1.0)
            })
            
        n_loss = num_modes_loss if num_modes_loss is not None else 5
        t_vals = self.target_eigen['vals'][:n_loss]
        if is_same_res:
            t_modes_l = self.target_eigen['modes'][:, :n_loss]
        else:
            t_modes_l = [griddata(pts_h, self.target_eigen['modes'][:, i], pts_l, method='linear') for i in range(len(t_vals))]
            t_modes_l = jnp.stack(t_modes_l, axis=1)
        
        # --- [NEW] Map Target Z-Coordinates (gt_z) for direct optimization matching ---
        target_z_low = None
        if 'z' in self.target_params_high:
            z_h_flat = self.target_params_high['z']
            if is_same_res:
                z_l_flat = np.array(z_h_flat)
            else:
                z_l_flat = griddata(pts_h, z_h_flat, pts_l, method='linear')
            target_z_low = jnp.array(z_l_flat).reshape(Ny_l+1, Nx_l+1)

        # 2. Optimization Parameters Setup
        opt_config = {k: v.copy() for k, v in opt_config.items()} # 원본 딕셔너리 보호용 얕은 복사
        
        if init_pz_from_gt and target_z_low is not None and 'pz' in opt_config:
            opt_config['pz']['init'] = target_z_low * gt_init_scale
            print(f" -> [INIT] Using Ground Truth Z-coordinates for 'pz' (Scale: {gt_init_scale})")
            
        self.scaling = {}
        for k in ['t', 'rho', 'E', 'pz']:
            cfg = opt_config.get(k, {})
            val = cfg.get('init', 0.0)
            if isinstance(val, (np.ndarray, jnp.ndarray)):
                val_scalar = float(jnp.mean(jnp.abs(val)))
            else:
                val_scalar = float(abs(val))
            self.scaling[k] = val_scalar if val_scalar > 1e-15 else 1.0
        
        full_params_scaled = {}
        for k in ['t', 'rho', 'E', 'pz']:
            if k in opt_config:
                cfg = opt_config[k]
                init_phys = cfg.get('init', 1.0 if k == 't' else 0.0)
                # Check for global type
                is_global = (cfg.get('type') == 'global')
                
                if is_global:
                    # If global, treat as a single scalar (stored in a (1,) array for JAX)
                    if isinstance(init_phys, (np.ndarray, jnp.ndarray)):
                        val = jnp.mean(jnp.array(init_phys))
                    else:
                        val = float(init_phys)
                    full_params_scaled[k] = jnp.array([val / self.scaling[k]])
                    print(f" -> Parameter '{k}' set to GLOBAL (Uniform) mode.")
                else:
                    if isinstance(init_phys, (np.ndarray, jnp.ndarray)):
                        full_params_scaled[k] = jnp.array(init_phys) / self.scaling[k]
                    else:
                        full_params_scaled[k] = jnp.full((Ny_l+1, Nx_l+1), init_phys / self.scaling[k])
        
        key = jax.random.PRNGKey(42); key, k1, k2 = jax.random.split(key, 3)
        # [REMOVED JITTER] Removed 1e-4 jitter for exact bit-match debugging during GT-initialization.
        # if 't' in full_params_scaled and opt_config.get('t', {}).get('opt', False): 
        #     full_params_scaled['t'] += 1e-4 * jax.random.uniform(k1, full_params_scaled['t'].shape)
        # if 'pz' in full_params_scaled and opt_config.get('pz', {}).get('opt', False):
        #     full_params_scaled['pz'] += 1e-4 * jax.random.uniform(k2, full_params_scaled['pz'].shape)

        params = {k: v for k, v in full_params_scaled.items() if opt_config.get(k, {}).get('opt', True)}
        fixed_params_scaled = {k: v for k, v in full_params_scaled.items() if not opt_config.get(k, {}).get('opt', True)}
        
        # Re-applying robust auto-rate scheduling (Warmup + Cosine Decay)
        warmup_steps = min(max_iterations // 2, max(1, max_iterations // 20))
        if max_iterations < 2: warmup_steps = 0 # No warmup for single step
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate / 5.0,  # Start at 20% of peak for immediate progress
            peak_value=learning_rate,        # User-defined max LR
            warmup_steps=warmup_steps,
            decay_steps=max_iterations,
            end_value=learning_rate * 0.01   # Final fine-tuning phase
        )

        # --- Pre-compute Topography Filter Kernel ---
        filter_kernel = None
        if use_bead_smoothing and min_bead_width > 0:
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
            
            # Special case for 'z' calculation from 'pz' (Topography)
            if 'pz' in combined_phys:
                pz_val = combined_phys['pz']
                if pz_val.ndim == 1 and pz_val.shape[0] == 1: # Global PZ
                    combined_phys['z'] = jnp.full((Ny_l+1, Nx_l+1), pz_val[0])
                elif filter_kernel is not None:
                    combined_phys['z'] = jax.scipy.signal.convolve2d(pz_val, filter_kernel, mode='same')
                else:
                    combined_phys['z'] = pz_val
            
            # Prepare flattened parameters for FEM assembly/analysis
            def broadcast_nodal(val):
                # If scalar (global), broadcast to (num_nodes,)
                if val.ndim == 1 and val.shape[0] == 1:
                    return jnp.full(((Ny_l+1)*(Nx_l+1),), val[0])
                return val.flatten()

            p_fem = {k: broadcast_nodal(v) for k, v in combined_phys.items() if k != 'pz'}
            
            K, M = self.fem.assemble(p_fem)
            p_actual = p_fem # Use flattened version for analysis calls
            l_static, l_stress, l_strain, l_energy, l_reaction, l_freq, l_mode, l_mass, l_reg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            for i, case in enumerate(self.cases):
                b = case_bcs[i]
                u = self.fem.solve_static_partitioned(K, b['F'], b['free'], b['fd'], b['fv'])
                w = u[2::6]
                # [FIXED] Static Errors: Normalization bug fixes and robust scaling (eps=1e-3)
                eps = 1e-3
                
                # Displacement: Normalized by target mean
                scale_disp = jnp.mean(jnp.abs(self.targets_low[i]['u_static'])) + eps
                delta_disp = w - self.targets_low[i]['u_static']
                l_static += jnp.mean((delta_disp / scale_disp)**2) * self.targets_low[i]['weight']
                
                if use_surface_stress:
                    scale_stress = jnp.mean(jnp.abs(self.targets_low[i]['max_stress'])) + eps
                    delta_stress = self.fem.compute_max_surface_stress(u, p_actual, K=K) - self.targets_low[i]['max_stress']
                    l_stress += jnp.mean((delta_stress / scale_stress)**2)
                if use_surface_strain:
                    scale_strain = jnp.mean(jnp.abs(self.targets_low[i]['max_strain'])) + eps
                    delta_strain = self.fem.compute_max_surface_strain(u, p_actual, K=K) - self.targets_low[i]['max_strain']
                    l_strain += jnp.mean((delta_strain / scale_strain)**2)
                if use_strain_energy:
                    scale_energy = jnp.mean(jnp.abs(self.targets_low[i]['strain_energy_density'])) + eps
                    delta_energy = self.fem.compute_strain_energy_density(u, p_actual, K=K) - self.targets_low[i]['strain_energy_density']
                    l_energy += jnp.mean((delta_energy / scale_energy)**2)
                
                # Reaction loss
                R_opt = K @ u - b['F']
                opt_R_sums = jnp.array([
                    jnp.sum(jnp.abs(R_opt[0::6])),
                    jnp.sum(jnp.abs(R_opt[1::6])),
                    jnp.sum(jnp.abs(R_opt[2::6])),
                    jnp.sum(jnp.abs(R_opt[3::6])),
                    jnp.sum(jnp.abs(R_opt[4::6])),
                    jnp.sum(jnp.abs(R_opt[5::6]))
                ])
                tgt_R_sums = self.targets_low[i]['reaction_sums']
                scale_R = jnp.mean(tgt_R_sums) + eps
                l_reaction += jnp.mean(((opt_R_sums - tgt_R_sums) / scale_R)**2)

            n_cases = len(self.cases)
            l_static /= n_cases; l_stress /= n_cases; l_strain /= n_cases; l_energy /= n_cases; l_reaction /= n_cases

            l_freq, l_mode, f1_hz = 0.0, 0.0, 0.0
            
            # --- [SPEED OPTIMIZATION] ---
            # If we don't care about frequency matching or mode shape matching, 
            # bypass the expensive O(N^3) dense eigen solver entirely!
            if loss_weights.get('freq', 0.0) > 1e-6 or loss_weights.get('mode', 0.0) > 1e-6:
                vals, vecs = self.fem.solve_eigen(K, M, num_modes=len(t_vals)+10)
                freqs = jnp.sqrt(jnp.maximum(vals, 0.0)) / (2 * jnp.pi)
                
                # 1. Dynamic Elastic Mode Selection (Physically Consistent)
                log_f = jnp.log(jnp.maximum(freqs[:15], 1e-6))
                gaps = jnp.concatenate([jnp.zeros(1), log_f[1:] - log_f[:-1]])
                current_idx = jnp.argmax((freqs > 1.0) & (gaps > 1.0) | (freqs > 20.0))
                
                n_tgt = len(t_vals)
                search_window = mac_search_window  # Allow searching in n_tgt + candidates to handle swaps
                
                # 2. Frequency Matching (Relative Error on Eigenvalues)
                curr_vals = jax.lax.dynamic_slice_in_dim(vals, current_idx, n_tgt)
                l_freq = jnp.mean((curr_vals / (t_vals + 1e-6) - 1.0)**2)
                
                # 3. Best-Fit MAC Matrix Matching (Mode Tracking)
                cand_modes = jax.lax.dynamic_slice_in_dim(vecs[2::6, :], current_idx, n_tgt + search_window, axis=1)
                
                # Compute MAC Matrix [n_target, n_candidates]
                dots = jnp.dot(t_modes_l.T, cand_modes) # (n_tgt, n_tgt + search_window)
                norm_t = jnp.sum(t_modes_l**2, axis=0, keepdims=True) # (1, n_tgt)
                norm_c = jnp.sum(cand_modes**2, axis=0, keepdims=True) # (1, n_tgt + search_window)
                mac_matrix = (dots**2) / (norm_t.T @ norm_c + 1e-10)
                
                best_mac_per_target = jnp.max(mac_matrix, axis=1)
                
                if mode_match_type == 'mac':
                    l_mode = jnp.mean((1.0 - best_mac_per_target)**2)
                else:
                    best_match_idx = jnp.argmax(mac_matrix, axis=1) # (n_tgt,)
                    best_cand_modes = cand_modes[:, best_match_idx] # (n_dofs, n_tgt)
                    
                    best_dots = jnp.diagonal(jnp.dot(t_modes_l.T, best_cand_modes)) # (n_tgt,)
                    signs = jnp.where(best_dots < 0, -1.0, 1.0)
                    aligned_modes = best_cand_modes * signs
                    
                    target_normed = t_modes_l / jnp.sqrt(norm_t.reshape(1, -1) + 1e-10)
                    cand_normed = aligned_modes / jnp.sqrt(jnp.sum(aligned_modes**2, axis=0, keepdims=True) + 1e-10)
                    
                    l_shape_mse = jnp.mean((target_normed - cand_normed)**2)
                    
                    if mode_match_type == 'direct':
                        l_mode = l_shape_mse * 100.0
                    else: # 'hybrid'
                        l_mode = jnp.mean((1.0 - best_mac_per_target)**2) + (l_shape_mse * 50.0)
                
                f1_hz = jnp.sqrt(jnp.maximum(vals[current_idx], 0.0)) / (2 * jnp.pi)
            
            # --- [FIXED] 질량 제약(Mass Constraint)에 mass_tolerance 적용 ---
            # 질량이 허용 한도(tolerance)를 넘어서면 폭발적인 페널티(Quadratic Penalty)를 부여합니다.
            if use_mass_constraint:
                current_mass = jnp.sum(p_actual['t'] * p_actual['rho']) * (self.fem.Lx/Nx_l)*(self.fem.Ly/Ny_l)
                mass_ratio_diff = jnp.abs(current_mass - self.target_mass) / self.target_mass
                # tolerance를 넘어가는 부분에 대해서만 벌점을 강력하게 부여 (RELu 형태 모방)
                l_mass = jnp.maximum(0.0, mass_ratio_diff - mass_tolerance)**2 * 100.0
            else:
                l_mass = 0.0
            for k in ['t', 'pz']:
                if k in p_scaled:
                    val = p_scaled[k]
                    if val.ndim > 1: # Skip global scalar [1]
                        l_reg += (jnp.mean(jnp.diff(val, axis=0)**2) + jnp.mean(jnp.diff(val, axis=1)**2))

            # 4) Gt_Z loss (Ground Truth Z-Coordinate matching)
            l_gt_z = 0.0
            if target_z_low is not None and loss_weights.get('gt_z', 0.0) > 0.0:
                l_gt_z = jnp.mean((combined_phys['z'] - target_z_low)**2)

            total_loss = (l_static * loss_weights.get('static', 0.0) +
                          l_stress * loss_weights.get('surface_stress', 0.0) +
                          l_strain * loss_weights.get('surface_strain', 0.0) + 
                          l_energy * loss_weights.get('strain_energy', 0.0) +
                          l_reaction * loss_weights.get('reaction', 0.0) +
                          l_freq   * loss_weights.get('freq', 0.0) +
                          l_mode   * loss_weights.get('mode', 0.0) + 
                          l_gt_z   * loss_weights.get('gt_z', 0.0) +
                          l_mass   * loss_weights.get('mass', 0.0) +
                          l_reg    * loss_weights.get('reg', 0.0))
                          
            return total_loss, {'Total': total_loss, 'Disp': l_static, 'Strs': l_stress, 'Strn': l_strain, 'Engy': l_energy, 'Reac': l_reaction, 'Freq': l_freq, 'Mode': l_mode, 'Gt_Z': l_gt_z, 'Mass': l_mass, 'Reg': l_reg, 'f1_hz': f1_hz}

        # Clear initial metrics logic
        print("Optimization Engine Ready. Initializing search...")

        best_loss, wait = float('inf'), 0
        best_params = params
        best_iter = 0
        self.history = [] # To store optimization trajectory
        
        print("\n [Tip] Press 'q' to stop optimization and use the best result found so far.\n")
        print(f"{'Iter':<5} | {'Total_Norm':<10} | {'Disp':<9} | {'Strs':<9} | {'Strn':<9} | {'Engy':<9} | {'Reac':<9} | {'Freq':<9} | {'Mode':<9} | {'Gt_Z':<9} | {'Mass':<9} | {'Freq1':<6}")
        print("-" * 128)

        for i in range(max_iterations):
            (val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, fixed_params_scaled, self.scaling)
            
            # [FIXED LOGIC] Store BEST params BEFORE they are updated by the optimizer
            if val < best_loss - early_stop_tol:
                best_loss, wait = val, 0
                best_params = {k: v.copy() for k, v in params.items()} # Deep copy of JAX arrays
                best_iter = i
            else:
                wait += 1

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
                p_min = cfg.get('min', -1e12) / self.scaling[k]
                p_max = cfg.get('max',  1e12) / self.scaling[k]
                params[k] = jnp.clip(params[k], p_min, p_max)

            # Check for 'q' key press to stop early
            if msvcrt.kbhit():
                char = msvcrt.getch()
                if char in [b'q', b'Q']:
                    print(f"\n [USER INTERRUPT] 'q' pressed. Terminating and reverting to best params (Iter {best_iter})...")
                    params = best_params
                    break

            if use_early_stopping and wait >= (early_stop_patience or 30):
                print(f"\n [EARLY STOP] No improvement for {wait} iters. Reverting to best params (Iter {best_iter})...")
                params = best_params
                break

            if i % 10 == 0: print(f"{i:<5d} | {val:<10.2e} | {metrics['Disp']:<9.2e} | {metrics['Strs']:<9.2e} | {metrics['Strn']:<9.2e} | {metrics['Engy']:<9.2e} | {metrics['Reac']:<9.2e} | {metrics['Freq']:<9.2e} | {metrics['Mode']:<9.2e} | {metrics['Gt_Z']:<9.2e} | {metrics['Mass']:<9.2e} | {metrics['f1_hz']:<6.2f}")

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
        all_keys = ['Total', 'Disp', 'Strs', 'Strn', 'Engy', 'Reac', 'Freq', 'Mode', 'Gt_Z', 'Mass', 'Reg']
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
        Xh, Yh = np.meshgrid(xh, yh, indexing='xy')
        pts_high = np.column_stack([Xh.flatten(), Yh.flatten()])
        
        # Low Res Coords for Interpolation
        xl = np.linspace(0, self.fem.Lx, self.fem.nx+1)
        yl = np.linspace(0, self.fem.Ly, self.fem.ny+1)
        Xl, Yl = np.meshgrid(xl, yl, indexing='xy')
        pts_low = np.column_stack([Xl.flatten(), Yl.flatten()])
        
        # Interpolate Optimized Params and Flatten to Nodal 1D arrays
        # [OPTIMIZATION] If resolution is the same, skip griddata to avoid smoothing loss
        def broadcast_to_high(val, target_res):
            target_shape = (target_res[1]+1, target_res[0]+1)
            if val.ndim == 1 and val.shape[0] == 1:
                return np.full(target_shape, val[0])
            elif val.ndim == 0:
                return np.full(target_shape, val)
            return val

        if Nx_h == self.fem.nx and Ny_h == self.fem.ny:
            print(" -> Resolution Match: Skipping interpolation to preserve shape integrity.")
            opt_params_h = {
                't': broadcast_to_high(np.array(self.optimized_params['t']), (Nx_h, Ny_h)).flatten(),
                'rho': broadcast_to_high(np.array(self.optimized_params['rho']), (Nx_h, Ny_h)).flatten(),
                'E': broadcast_to_high(np.array(self.optimized_params['E']), (Nx_h, Ny_h)).flatten(),
                'z': broadcast_to_high(np.array(self.optimized_params.get('pz', jnp.zeros_like(self.optimized_params['t']))), (Nx_h, Ny_h)).flatten()
            }
        else:
            def get_interp(k):
                val = np.array(self.optimized_params.get(k, jnp.zeros_like(self.optimized_params['t'] if k != 'pz' else self.optimized_params['t'])))
                if val.ndim == 1 and val.shape[0] == 1:
                    return np.full((Ny_h+1, Nx_h+1), val[0]).flatten()
                elif val.ndim == 0:
                    return np.full((Ny_h+1, Nx_h+1), val).flatten()
                return griddata(pts_low, val.flatten(), (Xh, Yh), method='linear').flatten()

            opt_params_h = {
                't': get_interp('t'),
                'rho': get_interp('rho'),
                'E': get_interp('E'),
                'z': get_interp('pz')
            }
        
        # 2. Assemble and Solve Optimized Model at High Res (Sparse)
        print("Solving Optimized High-Res Model for All Cases (Sparse Implementation)...")
        K_opt, M_opt = self.fem_high.assemble(opt_params_h, sparse=True)
        
        # 3. Static Performance Analysis & Matplotlib Plotting
        static_summary = []
        for i, case in enumerate(self.cases):
            tgt = self.targets[i]
            print(f" -> Verifying Case: {case.name}")
            
            # Solve Optimized (Sparse)
            fd, fv, F = case.get_bcs(self.fem_high)
            free = np.setdiff1d(np.arange(self.fem_high.total_dof), fd)
            u_opt = self.fem_high.solve_static_sparse(K_opt, F, free, fd, fv)
            
            # Extract Fields and reshape to (Ny+1, Nx+1) for Matplotlib
            w_ref = tgt['u_static'].reshape(Ny_h+1, Nx_h+1)
            w_opt = u_opt[2::6].reshape(Ny_h+1, Nx_h+1)
            
            stress_ref = tgt['max_surface_stress'].reshape(Ny_h+1, Nx_h+1)
            stress_opt = self.fem_high.compute_max_surface_stress(u_opt, opt_params_h, K=K_opt).reshape(Ny_h+1, Nx_h+1)
            
            strain_ref = tgt['max_surface_strain'].reshape(Ny_h+1, Nx_h+1)
            strain_opt = self.fem_high.compute_max_surface_strain(u_opt, opt_params_h, K=K_opt).reshape(Ny_h+1, Nx_h+1)

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
            
            # Calculate and display max reaction/moment
            R_full_tgt = tgt['reaction_full']
            max_force_z_tgt = np.max(np.abs(R_full_tgt[2::6])) # Primary Z reaction
            M_x_tgt = np.sum(np.abs(R_full_tgt[3::6])) # Approximating global moments
            M_y_tgt = np.sum(np.abs(R_full_tgt[4::6]))
            max_mom_tgt = np.sqrt(M_x_tgt**2 + M_y_tgt**2)

            # Calculate for Optimized (Sparse Dot)
            R_full_opt = K_opt.dot(np.array(u_opt)) - np.array(F)
            max_force_z_opt = np.max(np.abs(R_full_opt[2::6]))
            M_x_opt = np.sum(np.abs(R_full_opt[3::6]))
            M_y_opt = np.sum(np.abs(R_full_opt[4::6]))
            max_mom_opt = np.sqrt(M_x_opt**2 + M_y_opt**2)

            title_str = f"Verification: {case.name} | Resolution: {Nx_h}x{Ny_h}\n"
            
            # Check if this case is driven by displacement or angle (stiffness testing)
            if hasattr(case, 'mode') and case.mode in ['disp', 'angle']:
                title_str += f"[Stiffness Test] TGT RecZ: {max_force_z_tgt:.1f}N, Mom: {max_mom_tgt:.1f}Nmm | OPT RecZ: {max_force_z_opt:.1f}N, Mom: {max_mom_opt:.1f}Nmm"
            else:
                title_str += f"Target -> Max Reac Z: {max_force_z_tgt:.1f} N, Global Resultant Moment: {max_mom_tgt:.1f} Nmm"

            fig.suptitle(title_str, fontsize=12, fontweight='bold')
            
            # Plot Helper
            def plot_field(ax, data, title, cmap='jet', levels=None):
                if levels is None:
                    im = ax.contourf(xh, yh, data, 30, cmap=cmap)
                else:
                    im = ax.contourf(xh, yh, data, levels=levels, cmap=cmap, extend='both')
                ax.set_title(title)
                ax.set_aspect('equal')
                plt.colorbar(im, ax=ax)

            # Disp
            levels_w = np.linspace(np.min(w_ref), np.max(w_ref), 30)
            plot_field(axes[0,0], w_ref, "Target Disp (mm)", levels=levels_w, cmap='jet')
            plot_field(axes[0,1], w_opt, "Optimized Disp (mm)", levels=levels_w, cmap='jet')
            plot_field(axes[0,2], np.abs(w_opt - w_ref), "Error (mm)", cmap='YlOrRd')
            
            # Stress
            max_s = np.max(stress_ref)
            levels_s = np.linspace(0, max_s if max_s > 1e-12 else 1.0, 30)
            plot_field(axes[1,0], stress_ref, "Target Stress (MPa)", levels=levels_s, cmap='jet')
            plot_field(axes[1,1], stress_opt, "Optimized Stress (MPa)", levels=levels_s, cmap='jet')
            plot_field(axes[1,2], np.abs(stress_opt - stress_ref), "Error (MPa)", cmap='YlOrRd')
            
            # Strain
            max_e = np.max(strain_ref)*1000
            levels_e = np.linspace(0, max_e if max_e > 1e-12 else 1.0, 30)
            plot_field(axes[2,0], strain_ref*1000, "Target Strain (e-3)", levels=levels_e, cmap='jet')
            plot_field(axes[2,1], strain_opt*1000, "Optimized Strain (e-3)", levels=levels_e, cmap='jet')
            plot_field(axes[2,2], np.abs(strain_opt - strain_ref)*1000, "Error (e-3)", cmap='YlOrRd')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_file = f"verify_3d_{case.name}.png"
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"   -> Saved: {out_file}")

        # 4. Parameter Evolution Visualization (Ref vs Opt)
        print("Generating Parameter comparison plots...")
        plt.rcParams.update({'font.size': 8})
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        t_ref = np.array(self.target_params_high['t']).reshape(Ny_h+1, Nx_h+1)
        z_ref = np.array(self.target_params_high['z']).reshape(Ny_h+1, Nx_h+1)
        
        t_opt = np.array(opt_params_h['t']).reshape(Ny_h+1, Nx_h+1)
        z_opt = np.array(opt_params_h['z']).reshape(Ny_h+1, Nx_h+1)
        
        rho_ref = np.array(self.target_params_high['rho']).reshape(Ny_h+1, Nx_h+1)
        E_ref = np.array(self.target_params_high['E']).reshape(Ny_h+1, Nx_h+1)
        rho_opt = np.array(opt_params_h['rho']).reshape(Ny_h+1, Nx_h+1)
        E_opt = np.array(opt_params_h['E']).reshape(Ny_h+1, Nx_h+1)
        
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

        def plot_param(ax, data, title, cmap='viridis'):
            im = ax.contourf(xh, yh, data, 30, cmap=cmap)
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
        vals_opt, vecs_opt = self.fem_high.solve_eigen_sparse(K_opt, M_opt, num_modes=n_modes + 10)
        
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
        match_type = self.config.get('mode_match_type', 'mac')
        if match_type == 'direct':
            # Calculate Direct Nodal RMSE Correlation for Mode Shapes
            direct_diffs = []
            for j in range(n_modes):
                v1 = modes_opt[:, j] / (np.linalg.norm(modes_opt[:, j]) + 1e-12)
                v2 = modes_ref[:, j] / (np.linalg.norm(modes_ref[:, j]) + 1e-12)
                if np.sum(v1*v2) < 0: v1 = -v1
                rmse = np.sqrt(np.mean((v1 - v2)**2))
                direct_diffs.append(1.0 - min(1.0, rmse)) # Similarity Index
            
            plt.bar(np.arange(n_modes), direct_diffs, color='teal')
            plt.axhline(0.9, color='red', linestyle='--', alpha=0.5)
            plt.title("Mode Shape Similarity (Direct)", fontsize=10)
            plt.ylabel("1.0 - RMSE")
        else:
            plt.bar(np.arange(n_modes), macs, color='purple')
            plt.axhline(0.9, color='red', linestyle='--', alpha=0.5)
            plt.title("Modal Assurance Criterion (MAC)", fontsize=10)
            plt.ylabel("MAC Value")
            
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
            phi_ref = modes_ref[:, j].reshape(Ny_h+1, Nx_h+1)
            phi_ref = phi_ref / (np.max(np.abs(phi_ref)) + 1e-12)
            
            phi_opt = modes_opt[:, j].reshape(Ny_h+1, Nx_h+1)
            phi_opt = phi_opt / (np.max(np.abs(phi_opt)) + 1e-12)
            if np.sum(phi_ref * phi_opt) < 0:
                phi_opt = -phi_opt

            # Target (Top Row)
            ax_tgt = fig.add_subplot(2, n_plot_modes, j+1)
            im0 = ax_tgt.contourf(xh, yh, phi_ref, 30, cmap='jet', levels=np.linspace(-1, 1, 31))
            ax_tgt.set_title(f"Target Mode {j+1}\n({freq_ref[j]:.2f} Hz)", fontsize=8)
            ax_tgt.set_aspect('equal')
            if j == 0: # Only show colorbar in the first column
                cbar = plt.colorbar(im0, ax=ax_tgt, shrink=0.7)
                cbar.ax.set_ylabel('Amplitude', fontsize=8)
            
            # Optimized (Bottom Row)
            ax_opt = fig.add_subplot(2, n_plot_modes, n_plot_modes + j+1)
            im1 = ax_opt.contourf(xh, yh, phi_opt, 30, cmap='jet', levels=np.linspace(-1, 1, 31))
            ax_opt.set_title(f"Opt Mode {j+1}\n({freq_opt[j]:.2f} Hz, MAC: {macs[j]:.3f})", fontsize=8)
            ax_opt.set_aspect('equal')
            if j == 0: # Only show colorbar in the first column
                cbar = plt.colorbar(im1, ax=ax_opt, shrink=0.7)
                cbar.ax.set_ylabel('Amplitude', fontsize=8)
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("verify_3d_mode_shapes.png", dpi=150)
        plt.close()
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
            u_opt = self.fem_high.solve_static_sparse(K_opt, F_ext, free, fd, fv)
            u_ref = tgt['u_full']
            
            # 1. Displacement
            w_ref, w_opt = tgt['u_static'], u_opt[2::6]
            max_w_ref, max_w_opt = np.max(np.abs(w_ref)), np.max(np.abs(w_opt))
            
            # 2. Reaction Forces (at fixed DOFs) - Sparse Dot
            R_vec_ref = (K_opt.dot(np.array(u_ref)) - np.array(F_ext))[2::6] # Simplified Z-component reactions
            R_vec_opt = (K_opt.dot(np.array(u_opt)) - np.array(F_ext))[2::6]
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
        print("\n[OK] Comprehensive verification report saved: verification_report.md")
        print("[OK] Verification plots saved: verify_3d_*.png")

        # 7. Final Interactive Stage (PyVista) - Resolution Independent Comparison
        print("\nSolving Optimized Low-Res Model for 3D Comparison...")
        K_l, M_l = self.fem.assemble(self.optimized_params, sparse=True)
        vals_l, vecs_l = self.fem.solve_eigen_sparse(K_l, M_l, num_modes=n_modes + 10)
        # Match elastic modes for low-res
        freq_l_all = np.sqrt(np.maximum(np.array(vals_l), 0.0)) / (2*np.pi)
        log_fl = np.log(np.maximum(freq_l_all[:15], 1e-6))
        gapsl = np.concatenate([[0], log_fl[1:] - log_fl[:-1]])
        s_idx_l = np.argmax((freq_l_all > 1.0) & (gapsl > 1.0) | (freq_l_all > 20.0))
        opt_eigen_l = {
            'vals': vals_l[s_idx_l : s_idx_l + n_modes], 
            'modes': vecs_l[2::6, s_idx_l : s_idx_l + n_modes]
        }

        stage3_visualize_comparison(
            self.fem_high, self.fem, self.targets, self.optimized_params, self.target_params_high,
            opt_eigen=opt_eigen_l, tgt_eigen=self.target_eigen
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
        'bead_t': {'A': 2.0, 'B': 2.5, 'C': 1.5},
        'pattern_pz': 'TNYBV',  'bead_pz': {'T': 12.0, 'N': 10.0, 'Y': 15.0, 'B': 12.0, 'V': 4.0},
        'base_rho': 7.85e-9,        'base_E': 210000.0,
    }
    
    model.generate_targets(resolution_high=(Nx_high, Ny_high), num_modes_save=5, target_config=target_config)
        
    # 4. Optimization Search Space (opt_config)
    opt_config = {
        # 'type': 'global'을 추가하면 전체 평판의 두께가 동일한 값으로 최적화됩니다.
        't':   {'opt': True, 'init': 1.0, 'min': 0.1, 'max': 10.0, 'type': 'global'},        
        # 'nodal' (또는 생략)은 기존처럼 위치마다 다른 값을 갖는 위상 최적화 모드입니다.
        'rho': {'opt': True, 'init': 7.85e-9, 'min': 1e-10, 'max': 1e-7, 'type': 'global'},
        'E':   {'opt': False, 'init': 210000.0, 'type': 'global'},        
        # pz(형상)도 global로 설정하면 평판이 구부러지는 게 아니라 전체가 위아래로만 움직입니다.
        'pz':  {'opt': True, 'init': 0.0, 'min': -20.0, 'max': 20.0, 'type': 'local'}, 
    }
    
    # 5. Full Loss Weights (as previously defined)
    weights = {
        'static': 1.0,           # Displacement matching
        'reaction': 1.0,         # Reaction force matching (NEW)
        'freq': 1.0,            # [CRITICAL] Increase frequency priority
        'mode': 1.000,           # [Gentle] Mode matching (MAC) - Start small to avoid shock
        'curvature': 0.0,        # [Legacy]
        'moment': 0.0,           # [Legacy]
        'strain_energy': 1.0,    # Strain energy density matching
        'surface_stress':.0,   # Surface stress matching
        'surface_strain': .0,   # Surface strain matching
        'gt_z': 0.0,             # [NEW 옵션] Ground Truth 좌표 1:1 직접 매칭 여부 (강제 복원용). 현재는 0.0 (끔)
        'mass': 1.0,            # Mass constraint
        'reg': 0.001              # Regularization
    }
        
    # 6. Run Optimization - Two-Stage approach
    try:
        # --- STAGE 1: BASE SHAPE OPTIMIZATION ---
        print("\n\n" + "#"*70)
        print(" [RUN STAGE 1] 초기 뼈대 및 모드 기초 런 (Low MAC Weight)")
        print("#"*70)
        
        weights_stage1 = weights.copy()
        #weights_stage1['mode'] = 100.0  # 사용자 설정 반영 (Shape 매칭 강화)
        
        # [NEW 옵션 적용 예시] 완전히 똑같은 형상을 강제로 100% 매칭하고 싶다면 주석 해제 (단, 검증/Verification 용으로만 유효)
        # weights_stage1['gt_z'] = 100.0  
        # weights_stage1['static'] = 0.0
        # weights_stage1['strain_energy'] = 0.0
        
        best_params_s1 = model.optimize(opt_config, weights_stage1, 
                                        use_bead_smoothing=True,      # [필수] 저-고해상도 간 물리적 합치(Consistency)를 위해 활성화
                                        use_strain_energy=True,      # 사용 가능!
                                        use_surface_stress=False,     
                                        use_surface_strain=False,    
                                        use_mass_constraint=True,    
                                        mass_tolerance=0.05,         
                                        max_iterations=80,          
                                        use_early_stopping=True, 
                                        early_stop_patience=500, 
                                        early_stop_tol=1e-8,
                                        learning_rate=1.0,           
                                        num_modes_loss=5,
                                        min_bead_width=120.0,        # [조정] 메시 해상도에 적합한 비드 크기 유도
                                        mac_search_window=5,         # 모드 순서가 뒤집혀도 찾을 수 있도록 여유로운 비교 범위 부여
                                        mode_match_type='mac',    # 'mac', 'direct', 'hybrid' (제안된 형상 직접 매칭 방식)
                                        init_pz_from_gt=True,        # [NEW 옵션] 타겟 Ground Truth 좌표를 기반으로 초기 형태 전사 (사용)
                                        gt_init_scale=0.3)           # 타겟 Z 변위량의 10% 스케일을 최초 초기값으로 부과
                                        
        # --- STAGE 2: MAC FINE-TUNING ---
        #print("\n\n" + "#"*70)
        #print(" [RUN STAGE 2] MAC 미세 조정 런 (High MAC Weight & Wider Search)")
        #print("#"*70)
        
        # Stage 1 결과를 초기값으로 세팅
        # opt_config_stage2 = {
        #     't':   {'opt': opt_config['t']['opt'], 'init': best_params_s1['t'], 'min': opt_config['t']['min'], 'max': opt_config['t']['max']},
        #     'rho': {'opt': opt_config['rho']['opt'], 'init': best_params_s1['rho'], 'min': opt_config['rho']['min'], 'max': opt_config['rho']['max']},
        #     'E':   {'opt': opt_config['E']['opt'], 'init': best_params_s1['E'], 'min': opt_config['E']['min'], 'max': opt_config['E']['max']},
        #     'pz':  {'opt': opt_config['pz']['opt'], 'init': best_params_s1.get('pz', opt_config['pz']['init']), 'min': opt_config['pz']['min'], 'max': opt_config['pz']['max']},
        # }
        
        # weights_stage2 = weights.copy()
        # weights_stage2['mode'] = 5.   # MAC 가중치 크게 상향
        
        # model.optimize(opt_config_stage2, weights_stage2, 
        #                use_bead_smoothing=True,     # Stage 1에서 찾은 둥근 형태가 깨지지 않도록 유지
        #                use_strain_energy=True,      
        #                use_surface_stress=False,     # 이제 미분 가능
        #                use_surface_strain=False,     # 이제 미분 가능
        #                use_mass_constraint=True, 
        #                mass_tolerance=0.05,
        #                max_iterations=350,           # Stage 2: MAC 위주의 미세 조정이므로 반복 횟수를 짧게 설정
        #                use_early_stopping=True, 
        #                early_stop_patience=150, 
        #                early_stop_tol=1e-8,
        #                learning_rate=0.1,          # [안정성 확보] Fine Tuning에서 폭주(발산)하지 않도록 매우 좁은 보폭 적용
        #                num_modes_loss=5,
        #                min_bead_width=150.0,        # [핵심] Stage 1과 반드시 동일해야 초기 변형장이 파괴되지 않음
        #                mac_search_window=5,         # 형상이 복잡해지며 튀어나오는 모드들도 확실하게 캐치하기 위해 넓게 유지
        #                mode_match_type='hybrid',    # 'mac', 'direct', 'hybrid' (제안된 형상 직접 매칭 방식)
        #                init_pz_from_gt=False,        # [NEW 옵션] 타겟 Ground Truth 좌표를 기반으로 초기 형태 전사 (사용)
        #                gt_init_scale=0.5)           # 타겟 Z 변위량의 10% 스케일을 최초 초기값으로 부과

        # 7. Verify and Report
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
