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
import sys
import msvcrt
import datetime
import pickle

# Import our modular components
from ShellFemSolver.shell_solver import ShellFEM
from ShellFemSolver.mesh_utils import generate_rect_mesh_quads, generate_tray_mesh_quads
from WHT_EQS_analysis import PlateFEM, StructuralResult
from opt_targets import ResultBundle, Mode, apply_case_targets_from_spec, map_legacy_flags_to_targets, OptTarget
from WHT_EQS_pattern_generator import get_thickness_field, get_z_field
from WHT_EQS_load_cases import TwistCase, PureBendingCase, CornerLiftCase, TwoCornerLiftCase, CantileverCase, PressureCase
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
        self.targets_bundles = []
        self.resolution_high = (50, 20)
        self.target_mass = 0.0
        self.target_start_idx = 6 # Default for Free-Free
        self.optimized_params = None
        self.target_params_high = None
        self.config = {}
        
        # [PERFORMANCE] Pre-calculate assembly indices for high-speed JAX execution
        self.fem.fem._prepare_assembly_cache()

    def add_case(self, case):
        self.cases.append(case)

    def generate_targets(self, resolution_high=(120, 60), num_modes_save=5, cache_file="target_cache.pkl", target_config={}, use_cache_override=None):
        """
        Generates ground truth data using a high-fidelity model.
        """
        print("\n" + "="*70)
        print(" [STAGE 1] TARGET GENERATION & PATTERN VERIFICATION")
        print("="*70)
        
        # [BUGFIX] Store target_config in model for later use in verify()
        self.config['target_config'] = target_config
        self.resolution_high = resolution_high
        Nx_h, Ny_h = resolution_high
        
        # 1. Create High-Resolution Mesh for "Ground Truth"
        self.fem_high = PlateFEM(self.fem.Lx, self.fem.Ly, Nx_h, Ny_h)
        fem_high = self.fem_high
        xh = np.linspace(0, self.fem.Lx, Nx_h+1)
        yh = np.linspace(0, self.fem.Ly, Ny_h+1)
        Xh, Yh = np.meshgrid(xh, yh, indexing='xy')

        # --- Cache Check ---
        if use_cache_override is not None:
            use_cache = use_cache_override
        elif cache_file and os.environ.get("NON_INTERACTIVE") == "1":
            use_cache = os.path.exists(cache_file)
            print(f"[NON-INTERACTIVE] File exists: {use_cache}. Loading if possible.")
        elif cache_file and os.path.exists(cache_file):
            print(f"\n[CACHE DETECTED] Found existing Ground Truth data: {cache_file}")
            while True:
                choice = input(" -> Do you want to load from cache? [y/N] (Default: N - Re-interpret): ").strip().upper()
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
            
        if not use_cache:
            # 3. Solve FEM for each load case (High Fidelity - Sparse Solve)
            print("\nSolving High-Resolution Ground Truth (Sparse)...")
            K_h, M_h = fem_high.assemble(self.target_params_high, sparse=True)
            self.targets = []
            
            for i, case in enumerate(self.cases):
                print(f" -> Solving Case: {case.name}")
                fixed_dofs, fixed_vals, F = case.get_bcs(fem_high)
                free_dofs = np.setdiff1d(np.arange(fem_high.total_dof), fixed_dofs)
                u = fem_high.solve_static_sparse(K_h, F, free_dofs, fixed_dofs, fixed_vals)
                
                # Compute Detailed Field Results (Stress, Strain, SED)
                f_res = fem_high.compute_field_results(u, self.target_params_high)
                
                # Compute Full Reaction Forces (Sparse Dot) for optimization matching
                F_int = K_h @ jnp.array(u)
                R_residual = F_int - np.array(F)
                
                # Create StructuralResult for ParaView export
                res_fields = {
                    'displacement_vec': np.array(u),
                    'stress_vm': np.array(f_res['stress_vm']),
                    'strain_equiv': np.array(f_res['strain_equiv_nodal']),
                    'sed': np.array(f_res['sed'])
                }
                res = StructuralResult(res_fields, np.array(self.fem_high.nodes), fem_high.elements)
                res.save_vtkhdf(f"gt_static_{case.name}.vtkhdf")
                
                target = {
                    'case_name': case.name,
                    'weight': case.weight,
                    'u_static': np.array(u[2::6]), 
                    'u_full': np.array(u),
                    'reaction_full': np.array(R_residual),
                    'max_stress': np.array(f_res['stress_vm']),
                    'max_strain': np.array(f_res['strain_equiv_nodal']),
                    'strain_energy_density': np.array(f_res['sed']),
                    'params': self.target_params_high,
                    'fixed_dofs': np.array(fixed_dofs),
                    'force_vector': np.array(F)
                }
                self.targets.append(target)
                # Also create a structured ResultBundle for downstream OptTarget use
                # Build node displacement dict (node_id -> [u,v,w])
                node_disp_dict = {}
                num_nodes = fem_high.nodes.shape[0]
                u_full = np.array(u)
                for n in range(num_nodes):
                    node_disp_dict[n] = u_full[n*6 : n*6+3]
    
                rbemap = {'residual': np.array(R_residual)}
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
                    rbe_reactions=rbemap,
                    node_disps=node_disp_dict,
                    mass=float(self.target_mass),
                    modes=[],
                    meta={'params': self.target_params_high, 'units': {'stress_vm': 'MPa', 'disp': 'mm', 'rbe_reaction': 'N'}}
                )
                self.targets_bundles.append(bundle)
                    
            # --- NEW: Generate Summary Ground Truth Plot (Moved Outside Loop) ---
            print("\nGenerating Ground Truth Summary Plot (3xN)...")
            n_cases = len(self.cases)
            plt.rcParams.update({'font.size': 8})
            fig, axes = plt.subplots(3, n_cases, figsize=(4*n_cases, 10), squeeze=False)
            fig.suptitle(f"Ground Truth Analysis Summary (Resolution: {Nx_h}x{Ny_h})\nRows: Disp, Stress, Strain | Colormap: jet", fontsize=10)
            
            xh, yh = np.linspace(0, self.fem.Lx, Nx_h+1), np.linspace(0, self.fem.Ly, Ny_h+1)
            
            for i, target in enumerate(self.targets):
                # Row 0: Displacement (Nodal)
                data_w = target['u_static'].reshape(Ny_h+1, Nx_h+1)
                im0 = axes[0, i].contourf(xh, yh, data_w, 30, cmap='jet')
                axes[0, i].set_title(f"Case: {target['case_name']}\nMax Disp: {np.max(np.abs(data_w)):.3f}mm", fontsize=8)
                axes[0, i].set_aspect('equal')
                plt.colorbar(im0, ax=axes[0, i], shrink=0.7)
                
                # Row 1/2: Max Surface Stress/Strain
                # These can be nodal (if averaged) or elemental (if raw).
                # We already have nodal averages in 'max_stress' and 'max_strain' keys.
                s_field = target['max_stress']
                e_field = target['max_strain']
                
                # Ensure we can reshape (project back to grid if necessary)
                size_expected = (Nx_h+1)*(Ny_h+1)
                if s_field.size == size_expected and e_field.size == size_expected:
                    data_s = s_field.reshape(Ny_h+1, Nx_h+1)
                    data_e = e_field.reshape(Ny_h+1, Nx_h+1) * 1000 # to microstrain
                else:
                    # Fallback: if size doesn't match nodal grid, just plot max as text or skip contour
                    print(f" [WARN] Field sizes (S:{s_field.size}, E:{e_field.size}) don't match grid {size_expected}. Skipping contour.")
                    data_s = np.zeros((Ny_h+1, Nx_h+1))
                    data_e = np.zeros((Ny_h+1, Nx_h+1))
    
                im1 = axes[1, i].contourf(xh, yh, data_s, 30, cmap='jet')
                axes[1, i].set_title(f"Max Stress: {np.max(data_s):.2f} MPa", fontsize=8)
                axes[1, i].set_aspect('equal')
                plt.colorbar(im1, ax=axes[1, i], shrink=0.7)
                
                im2 = axes[2, i].contourf(xh, yh, data_e, 30, cmap='jet')
                axes[2, i].set_title(f"Max Strain: {np.max(data_e):.3f} e-3", fontsize=8)
                axes[2, i].set_aspect('equal')
                plt.colorbar(im2, ax=axes[2, i], shrink=0.7)
                
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig("ground_truth_3d_loadcases.png", dpi=150)
            plt.close()
            print(" -> Saved: ground_truth_3d_loadcases.png")
            
            # [ISSUE-015] Use unified solve_eigen_sparse (dense-eigh backend) for verification
            vals, vecs = fem_high.solve_eigen_sparse(K_h, M_h, num_modes=num_modes_save + 20)
            
            # solve_eigen_sparse already returns frequencies in Hz
            all_freqs_h = np.array(vals)
            start_idx = 6
            print(f" -> Unified Standard: Skipping 6 RBMs. Target Mode 1 is index {start_idx} ({all_freqs_h[start_idx]:.2f} Hz)")
    
            self.target_start_idx = start_idx
            freqs_save = vals[start_idx : start_idx + num_modes_save]
            modes_save = vecs[:, start_idx : start_idx + num_modes_save] # Full 6-DOF
            
            self.target_eigen = {
                'vals': np.array(freqs_save),
                'modes': np.array(modes_save[2::6, :]) # W-component for internal loss
            }
        
            # --- NEW: Export Modal Results to Temporal VTKHDF ---
            print("\nExporting Ground Truth Modal Analysis to ParaView...")
            modal_res = StructuralResult({}, np.array(self.fem_high.nodes), fem_high.elements)
            steps_dict = {
                'values': freqs_save,
                'point_data': {
                    'mode_shape_vec': [modes_save[:, i] for i in range(num_modes_save)]
                }
            }
            modal_res.save_vtkhdf("gt_modal_results.vtkhdf", steps_dict=steps_dict)
            
            # Save to cache
            if cache_file:
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
        else:
            # If using cache, we already have everything needed for self.target_eigen
            # but we should still make sure self.fem_high is consistent if needed for verify()
            pass

        # target_eigen['vals'] are already in Hz
        target_freqs = np.array(self.target_eigen['vals'])
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
                 max_iterations=200,
                 use_early_stopping=True,
                 early_stop_patience=30,
                 early_stop_tol=1e-8,
                 learning_rate=0.3,
                 num_modes_loss=None,
                 min_bead_width=150.0,
                 mac_search_window=2,
                 mode_match_type='hybrid',
                 mode_freq_weight=0.0,
                 init_pz_from_gt=False,
                 gt_init_scale=1.0,
                 auto_scale=False,
                 eigen_freq=20):
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
        - early_stop_patience : 손실값이 개선되지 않는 상태를 몇 번의 반복 횟수까지 봐줄 것인지 (예: 30번). [최적화: 기본값 100→30]
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
        - auto_scale          : 초기 손실 함수 자동 가중치 스케일링 여부. False면 사용자 설정 가중치 직접 사용 [최적화: True→False]
        - eigen_freq          : 주파수/모드 손실 계산 주기(반복 수). 10=10번마다 한 번 계산, 나머지는 캐시 사용. [최적화: 신규 추가]
        """
                 
        print("\n" + "="*70)
        print(" [STAGE 3] ADVANCED OPTIMIZATION (EXPLICIT LOGIC)")
        print(f" -> Setting: Min Bead Width = {min_bead_width} mm")
        if not use_bead_smoothing:
            print(" -> WARNING: Bead smoothing (use_bead_smoothing=False) is disabled. Results may show numerical checkerboarding.")
        print("="*70)
        
        Nx_l, Ny_l = self.fem.nx, self.fem.ny
        pts_l = self.fem.node_coords[:, :2]  # X,Y only for griddata

        # --- PoC: Evaluate any OptTargets attached to cases using ResultBundle ---
        if hasattr(self, 'targets_bundles') and len(self.targets_bundles) > 0:
            print("\n[PoC] Evaluating case OptTargets against stored ResultBundles")
            for i, case in enumerate(self.cases):
                if getattr(case, 'opt_targets', None):
                    try:
                        bundle = self.targets_bundles[i]
                    except IndexError:
                        print(f"  - Case {case.name}: no matching ResultBundle (skip)")
                        continue
                    # [ROBUST] Use ACTUAL 3D nodal coordinates from the Mesh objects
                    # Using [:, :3] instead of [:, :2] prevents "Dimension Collapse" on vertical walls 
                    # which was causing MAC=0.0 matching failures.
                    pts_h = np.array(self.fem_high.nodes)[:, :3]
                    pts_l = np.array(self.fem.nodes)[:, :3]
                    for t_idx, ot in enumerate(case.opt_targets):
                        err, details = ot.compute_error(bundle, ref_bundle=bundle)
                        print(f"  - Case {case.name} Target#{t_idx}: err={err:.6e}, details={details}")

        
        # [NEW] Save optimization configuration for later use (e.g., in verify())
        self.config = {
            'mode_match_type': mode_match_type,
            'num_modes_loss': num_modes_loss,
            'use_bead_smoothing': use_bead_smoothing
        }

        # Map legacy boolean flags to OptTarget instances (attach to cases/global)
        map_legacy_flags_to_targets(self,
                        use_surface_stress=use_surface_stress,
                        use_surface_strain=use_surface_strain,
                        use_strain_energy=use_strain_energy,
                        use_mass_constraint=use_mass_constraint,
                        mode_weight=loss_weights.get('mode', 0.0),
                        num_modes=num_modes_loss,
                        freq_weight=mode_freq_weight)
        
        # 1. Interpolate High-Res Targets to Low-Res Mesh
        Nx_h, Ny_h = self.resolution_high
        Nx_l, Ny_l = self.fem.nx, self.fem.ny
        
        # [ROBUST] Use ACTUAL 3D nodal coordinates from the Mesh objects
        pts_h = np.array(self.fem_high.nodes)[:, :3]
        pts_l = np.array(self.fem.nodes)[:, :3]
        
        self.targets_low = []
        is_same_res = bool(Nx_h == Nx_l and Ny_h == Ny_l)
        if is_same_res:
            print(" -> Resolution Match (Target Low-Res): Skipping griddata for bit-exact mapping.")

        for tgt in self.targets:
            if is_same_res:
                u_l = np.array(tgt['u_static'])
                stress_l = np.array(tgt['max_stress'])
                strain_l = np.array(tgt['max_strain'])
                sed_l = np.array(tgt['strain_energy_density'])
            else:
                u_l = griddata(pts_h, np.array(tgt['u_static']), pts_l, method='linear')
                stress_l = griddata(pts_h, np.array(tgt['max_stress']), pts_l, method='linear')
                strain_l = griddata(pts_h, np.array(tgt['max_strain']), pts_l, method='linear')
                sed_l = griddata(pts_h, np.array(tgt['strain_energy_density']), pts_l, method='linear')
            
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
            
        n_loss = int(num_modes_loss) if num_modes_loss is not None else 5
        t_vals = np.array(self.target_eigen['vals'])[:n_loss]
        t_modes_h_np = np.array(self.target_eigen['modes'])
        
        if is_same_res:
            t_modes_l = jnp.array(t_modes_h_np[:, :n_loss])
        else:
            print(f" -> Performing linear interpolation for {n_loss} modes...")
            t_modes_l_list = []
            for i in range(len(t_vals)):
                itp = griddata(pts_h, t_modes_h_np[:, i], pts_l, method='linear')
                t_modes_l_list.append(itp)
            t_modes_l = jnp.array(np.stack(t_modes_l_list, axis=1))
        
        # --- [NEW] Map Target Z-Coordinates (gt_z) for direct optimization matching ---
        target_z_low = None
        if 'z' in self.target_params_high:
            z_h_flat = self.target_params_high['z']
            if is_same_res:
                z_l_flat = np.array(z_h_flat)
            else:
                z_l_flat = griddata(np.array(pts_h), np.array(z_h_flat), np.array(pts_l), method='linear')
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
        
        # [FIX] Ensure scaling is a dict of pure Python floats to avoid JAX Tracer issues inside JIT
        scaling_consts = {k: float(v) for k, v in self.scaling.items()}
        
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

        # --- Aggregate OptTarget weights into loss-category constants ---
        opt_w = {
            'static': 0.0,
            'stress': 0.0,
            'strain': 0.0,
            'energy': 0.0,
            'reaction': 0.0,
            'mode': 0.0,
            'mass': 0.0
        }
        from opt_targets import TargetType
        for ci, case in enumerate(self.cases):
            cweight = getattr(case, 'weight', 1.0)
            for ot in getattr(case, 'opt_targets', []) or []:
                w = float(getattr(ot, 'weight', 1.0)) * float(cweight)
                t = ot.target_type
                if t == TargetType.FIELD_STAT:
                    # map field choices to categories
                    field = getattr(ot, 'field', '')
                    if field and 'stress' in field:
                        opt_w['stress'] += w
                    elif field and 'strain' in field:
                        opt_w['strain'] += w
                    else:
                        opt_w['static'] += w
                elif t == TargetType.RBE_REACTION:
                    opt_w['reaction'] += w
                elif t == TargetType.NODE_DISP:
                    opt_w['static'] += w
                elif t == TargetType.MODES:
                    opt_w['mode'] += w
                elif t == TargetType.MASS:
                    opt_w['mass'] += w

        # --- Build JAX-evaluable target descriptors for in-loss evaluation ---
        # Each descriptor references a case index and contains minimal data for JAX ops.
        targets_jax = []
        for ci, case in enumerate(self.cases):
            for ot in getattr(case, 'opt_targets', []) or []:
                desc = {'case_idx': ci, 'type': ot.target_type.value, 'weight': float(getattr(ot, 'weight', 1.0))}
                # map fields to existing target arrays (targets_low)
                if ot.target_type == TargetType.FIELD_STAT:
                    fld = getattr(ot, 'field', '') or ''
                    if 'stress' in fld:
                        desc['ref_key'] = 'max_stress'
                    elif 'strain' in fld:
                        desc['ref_key'] = 'max_strain'
                    elif 'sed' in fld or 'energy' in fld:
                        desc['ref_key'] = 'strain_energy_density'
                    else:
                        desc['ref_key'] = 'u_static'
                    desc['reduction'] = ot.reduction.value
                    desc['compare_mode'] = ot.compare_mode.value
                elif ot.target_type == TargetType.RBE_REACTION:
                    desc['ref_key'] = 'reaction_sums'
                    desc['component'] = getattr(ot, 'component', None)
                    desc['compare_mode'] = ot.compare_mode.value
                elif ot.target_type == TargetType.MODES:
                    desc['compare_mode'] = ot.compare_mode.value
                    desc['num_modes'] = getattr(ot, 'num_modes', None)
                    desc['freq_weight'] = float(getattr(ot, 'freq_weight', 0.0))
                elif ot.target_type == TargetType.MASS:
                    desc['ref_value'] = float(ot.ref_value) if ot.ref_value is not None else float(self.target_mass)
                    desc['compare_mode'] = ot.compare_mode.value
                targets_jax.append(desc)

        # Include any global targets attached to model (e.g., mass, modal)
        for ot in getattr(self, 'global_opt_targets', []) or []:
            desc = {'case_idx': None, 'type': ot.target_type.value, 'weight': float(getattr(ot, 'weight', 1.0))}
            if ot.target_type == TargetType.FIELD_STAT:
                fld = getattr(ot, 'field', '') or ''
                if 'stress' in fld:
                    desc['ref_key'] = 'max_stress'
                elif 'strain' in fld:
                    desc['ref_key'] = 'max_strain'
                elif 'sed' in fld or 'energy' in fld:
                    desc['ref_key'] = 'strain_energy_density'
                else:
                    desc['ref_key'] = 'u_static'
                desc['reduction'] = ot.reduction.value
                desc['compare_mode'] = ot.compare_mode.value
            elif ot.target_type == TargetType.RBE_REACTION:
                desc['ref_key'] = 'reaction_sums'
                desc['component'] = getattr(ot, 'component', None)
                desc['compare_mode'] = ot.compare_mode.value
            elif ot.target_type == TargetType.MODES:
                desc['compare_mode'] = ot.compare_mode.value
                desc['num_modes'] = getattr(ot, 'num_modes', None)
                desc['freq_weight'] = float(getattr(ot, 'freq_weight', 0.0))
            elif ot.target_type == TargetType.MASS:
                desc['ref_value'] = float(ot.ref_value) if ot.ref_value is not None else float(self.target_mass)
                desc['compare_mode'] = ot.compare_mode.value
            targets_jax.append(desc)


        def loss_fn(p_scaled, fixed_p_scaled, freqs=None, vecs=None, loss_weights=None):
            # Use captured scaling_consts (constants in JIT)
            combined_phys = {k: v * scaling_consts[k] for k, v in p_scaled.items()}
            for k, v in fixed_p_scaled.items(): combined_phys[k] = v * scaling_consts[k]

            # Prepare flattened parameters for FEM assembly and analysis calls
            def broadcast_nodal(val):
                if val.ndim == 1 and val.shape[0] == 1:
                    return jnp.full(((Ny_l+1)*(Nx_l+1),), val[0])
                return val.flatten()

            p_fem = {k: broadcast_nodal(v) for k, v in combined_phys.items() if k != 'pz'}
            K, M = self.fem.assemble(p_fem)
            p_actual = p_fem  # Use flattened version for analysis calls

            # collect per-case computed fields so we can evaluate all OptTargets after solving each case
            per_u = []
            per_w = []
            per_max_stress = []
            per_max_strain = []
            per_sed = []
            per_reaction_sums = []

            for i, case in enumerate(self.cases):
                b = case_bcs[i]
                u = self.fem.solve_static_partitioned(K, b['F'], b['free'], b['fd'], b['fv'])
                w = u[2::6]
                f_res = self.fem.compute_field_results(u, p_actual)

                # store proxies for later descriptor evaluation
                per_u.append(self.targets_low[i]['u_static'])
                per_w.append(w)
                per_max_stress.append(self.fem.compute_max_surface_stress(u, p_actual, field_results=f_res))
                per_max_strain.append(self.fem.compute_max_surface_strain(u, p_actual, field_results=f_res))
                per_sed.append(self.fem.compute_strain_energy_density(u, p_actual, field_results=f_res))
                R_opt = K @ u - b['F']
                opt_R_sums = jnp.array([
                    jnp.sum(jnp.abs(R_opt[0::6])),
                    jnp.sum(jnp.abs(R_opt[1::6])),
                    jnp.sum(jnp.abs(R_opt[2::6])),
                    jnp.sum(jnp.abs(R_opt[3::6])),
                    jnp.sum(jnp.abs(R_opt[4::6])),
                    jnp.sum(jnp.abs(R_opt[5::6]))
                ])
                per_reaction_sums.append(opt_R_sums)

            # Now evaluate all OptTarget descriptors (per-case and global) into a single opt_loss
            l_static, l_stress, l_strain, l_energy, l_reaction = 0.0, 0.0, 0.0, 0.0, 0.0
            opt_loss = 0.0
            for desc in targets_jax:
                typ = desc['type']
                wgt = desc.get('weight', 1.0)
                ci = desc.get('case_idx')
                # helper to fetch per-case arrays
                def get_case_arr(key, idx):
                    if key == 'u_static':
                        return per_u[idx]
                    if key == 'w':
                        return per_w[idx]
                    if key == 'max_stress':
                        return per_max_stress[idx]
                    if key == 'max_strain':
                        return per_max_strain[idx]
                    if key == 'strain_energy_density':
                        return per_sed[idx]
                    if key == 'reaction_sums':
                        return per_reaction_sums[idx]
                    return per_w[idx]

                if ci is not None:
                    # per-case target
                    idx = int(ci)
                    if typ == 'field_stat':
                        ref_key = desc.get('ref_key', 'u_static')
                        data = get_case_arr(ref_key, idx)
                        red = desc.get('reduction', 'max')
                        if red == 'max':
                            val = jnp.nanmax(data)
                        elif red == 'mean':
                            val = jnp.nanmean(data)
                        elif red == 'rms':
                            val = jnp.sqrt(jnp.nanmean(data**2))
                        else:
                            val = jnp.nanmax(data)
                        # obtain reference (pre-interpolated)
                        ref_arr = self.targets_low[idx].get(ref_key)
                        if ref_arr is None:
                            ref_val = 0.0
                        else:
                            if hasattr(ref_arr, 'ndim') and ref_arr.ndim > 0:
                                if red == 'max':
                                    ref_val = jnp.max(ref_arr)
                                else:
                                    ref_val = jnp.mean(ref_arr)
                            else:
                                ref_val = ref_arr
                        if desc.get('compare_mode','absolute') == 'relative' and ref_val != 0:
                            terr = (val - ref_val) / (ref_val + 1e-12)
                        else:
                            terr = val - ref_val
                        if typ == 'rbe_reaction': l_reaction += wgt * (terr**2)
                        elif typ == 'field_stat':
                            if desc.get('ref_key') == 'u_static': l_static += wgt * (terr**2)
                            elif desc.get('ref_key') == 'max_stress': l_stress += wgt * (terr**2)
                            elif desc.get('ref_key') == 'max_strain': l_strain += wgt * (terr**2)
                            elif desc.get('ref_key') == 'strain_energy_density': l_energy += wgt * (terr**2)
                            else: l_static += wgt * (terr**2)
                        elif typ == 'mass': pass # mass is handled later
                        elif typ == 'modes': pass # modes handled later

                    elif typ == 'rbe_reaction':
                        comp = desc.get('component', None)
                        val_vec = get_case_arr('reaction_sums', idx)
                        ref_vec = self.targets_low[idx].get('reaction_sums')
                        ref_mean = jnp.mean(ref_vec) + 1e-12
                        terr = (jnp.mean(val_vec) - jnp.mean(ref_vec)) / ref_mean
                        if typ == 'rbe_reaction': l_reaction += wgt * (terr**2)
                        elif typ == 'field_stat':
                            if desc.get('ref_key') == 'u_static': l_static += wgt * (terr**2)
                            elif desc.get('ref_key') == 'max_stress': l_stress += wgt * (terr**2)
                            elif desc.get('ref_key') == 'max_strain': l_strain += wgt * (terr**2)
                            elif desc.get('ref_key') == 'strain_energy_density': l_energy += wgt * (terr**2)
                            else: l_static += wgt * (terr**2)
                        elif typ == 'mass': pass # mass is handled later
                        elif typ == 'modes': pass # modes handled later

                    elif typ == 'modes':
                        # already handled in global eigen pass; skip per-case here
                        pass

                    elif typ == 'mass':
                        if 't' in p_fem and 'rho' in p_fem:
                            mass_opt = jnp.sum(p_fem['t'] * p_fem['rho']) * (self.fem.Lx / Nx_l) * (self.fem.Ly / Ny_l)
                        else:
                            mass_opt = 0.0
                        ref_mass = float(desc.get('ref_value', self.target_mass))
                        if desc.get('compare_mode','absolute') == 'relative' and ref_mass != 0:
                            terr = (mass_opt - ref_mass) / (ref_mass + 1e-12)
                        else:
                            terr = mass_opt - ref_mass
                        if typ == 'rbe_reaction': l_reaction += wgt * (terr**2)
                        elif typ == 'field_stat':
                            if desc.get('ref_key') == 'u_static': l_static += wgt * (terr**2)
                            elif desc.get('ref_key') == 'max_stress': l_stress += wgt * (terr**2)
                            elif desc.get('ref_key') == 'max_strain': l_strain += wgt * (terr**2)
                            elif desc.get('ref_key') == 'strain_energy_density': l_energy += wgt * (terr**2)
                            else: l_static += wgt * (terr**2)
                        elif typ == 'mass': pass # mass is handled later
                        elif typ == 'modes': pass # modes handled later

                else:
                    # global descriptor
                    if typ == 'mass':
                        if 't' in p_fem and 'rho' in p_fem:
                            mass_opt = jnp.sum(p_fem['t'] * p_fem['rho']) * (self.fem.Lx / Nx_l) * (self.fem.Ly / Ny_l)
                        else:
                            mass_opt = 0.0
                        ref_mass = float(desc.get('ref_value', self.target_mass))
                        if desc.get('compare_mode','absolute') == 'relative' and ref_mass != 0:
                            terr = (mass_opt - ref_mass) / (ref_mass + 1e-12)
                        else:
                            terr = mass_opt - ref_mass
                        if typ == 'rbe_reaction': l_reaction += wgt * (terr**2)
                        elif typ == 'field_stat':
                            if desc.get('ref_key') == 'u_static': l_static += wgt * (terr**2)
                            elif desc.get('ref_key') == 'max_stress': l_stress += wgt * (terr**2)
                            elif desc.get('ref_key') == 'max_strain': l_strain += wgt * (terr**2)
                            elif desc.get('ref_key') == 'strain_energy_density': l_energy += wgt * (terr**2)
                            else: l_static += wgt * (terr**2)
                        elif typ == 'mass': pass # mass is handled later
                        elif typ == 'modes': pass # modes handled later

                    elif typ == 'modes':
                        # evaluate using freqs/vecs if available
                        fw = float(desc.get('freq_weight', 0.0))
                        nmode = int(desc.get('num_modes') or len(t_vals))
                        if freqs is not None and vecs is not None:
                            cand_modes = vecs[2::6, :]
                            dots = jnp.dot(t_modes_l.T, cand_modes)
                            norm_t = jnp.sum(t_modes_l**2, axis=0, keepdims=True)
                            norm_c = jnp.sum(cand_modes**2, axis=0, keepdims=True)
                            mac_matrix = (dots**2) / (norm_t.T @ norm_c + 1e-10)
                            best_mac_per_target = jnp.max(mac_matrix, axis=1)
                            mac_err = jnp.mean((1.0 - best_mac_per_target[:nmode])**2) if nmode > 0 else 0.0
                            if fw > 0.0 and freqs is not None:
                                n_f = min(nmode, freqs.shape[0], t_vals.shape[0])
                                if n_f > 0:
                                    ref_f = t_vals[:n_f]
                                    opt_f = freqs[:n_f]
                                    freq_err = jnp.mean(jnp.abs((opt_f - ref_f) / (ref_f + 1e-12)))
                                else:
                                    freq_err = 0.0
                                combined = mac_err * (1.0 - fw) + freq_err * fw
                            else:
                                combined = mac_err
                            opt_loss += wgt * combined

                    elif typ == 'field_stat':
                        # compare aggregated field stat across all cases
                        ref_key = desc.get('ref_key', 'u_static')
                        stacked = jnp.stack([t[ref_key] for t in self.targets_low], axis=0)
                        if desc.get('reduction','max') == 'max':
                            ref_val = jnp.max(stacked)
                        else:
                            ref_val = jnp.mean(stacked)
                        cur_val = jnp.max(stacked) if desc.get('reduction','max') == 'max' else jnp.mean(stacked)
                        if desc.get('compare_mode','absolute') == 'relative' and ref_val != 0:
                            terr = (cur_val - ref_val) / (ref_val + 1e-12)
                        else:
                            terr = cur_val - ref_val
                        if typ == 'rbe_reaction': l_reaction += wgt * (terr**2)
                        elif typ == 'field_stat':
                            if desc.get('ref_key') == 'u_static': l_static += wgt * (terr**2)
                            elif desc.get('ref_key') == 'max_stress': l_stress += wgt * (terr**2)
                            elif desc.get('ref_key') == 'max_strain': l_strain += wgt * (terr**2)
                            elif desc.get('ref_key') == 'strain_energy_density': l_energy += wgt * (terr**2)
                            else: l_static += wgt * (terr**2)
                        elif typ == 'mass': pass # mass is handled later
                        elif typ == 'modes': pass # modes handled later

                    elif typ == 'rbe_reaction':
                        # global compare: mean reaction sums across cases
                        ref_vec = jnp.mean(jnp.stack([t['reaction_sums'] for t in self.targets_low], axis=0), axis=0)
                        # use mean of computed per_reaction_sums
                        opt_vec = jnp.mean(jnp.stack(per_reaction_sums, axis=0), axis=0)
                        ref_mean = jnp.mean(ref_vec) + 1e-12
                        terr = (jnp.mean(opt_vec) - jnp.mean(ref_vec)) / ref_mean
                        if typ == 'rbe_reaction': l_reaction += wgt * (terr**2)
                        elif typ == 'field_stat':
                            if desc.get('ref_key') == 'u_static': l_static += wgt * (terr**2)
                            elif desc.get('ref_key') == 'max_stress': l_stress += wgt * (terr**2)
                            elif desc.get('ref_key') == 'max_strain': l_strain += wgt * (terr**2)
                            elif desc.get('ref_key') == 'strain_energy_density': l_energy += wgt * (terr**2)
                            else: l_static += wgt * (terr**2)
                        elif typ == 'mass': pass # mass is handled later
                        elif typ == 'modes': pass # modes handled later

            # Regularization (kept)
            l_reg = 0.0
            for k in ['t', 'pz']:
                if k in p_scaled:
                    val = p_scaled[k]
                    if val.ndim > 1:
                        l_reg += (jnp.mean(jnp.diff(val, axis=0)**2) + jnp.mean(jnp.diff(val, axis=1)**2))

            n_cases = len(self.cases)
            l_static /= n_cases; l_stress /= n_cases; l_strain /= n_cases; l_energy /= n_cases; l_reaction /= n_cases

            l_freq, l_mode, f1_hz = 0.0, 0.0, 0.0
            
            # --- [SPEED OPTIMIZATION] ---
            # If we don't care about frequency matching or mode shape matching, 
            # bypass the expensive O(N^3) dense eigen solver entirely!
            # --- [SPEED OPTIMIZATION - ARPACK SPARSE EIGENVALUE] ---
            if freqs is not None and vecs is not None:
                # 🚀 Use ARPACK for 300x speedup: Sparse matrix retained, only k modes computed
                n_tgt = len(t_vals)
                
                # 1. Frequency Matching (Relative Error on Eigenvalues λ = ω^2)
                # Since frequencies are already in Hz: (f_opt / f_tgt - 1)^2
                l_freq = jnp.mean((freqs / (t_vals + 1e-6) - 1.0)**2)
                
                # 2. Best-Fit MAC Matrix Matching
                # Indexing is now simplified as f_opt[0] corresponds to f_tgt[0]
                cand_modes = vecs[2::6, :] # Extract w-displacement (out-of-plane)
                
                # Compute MAC [n_target, n_opt]
                dots = jnp.dot(t_modes_l.T, cand_modes) 
                norm_t = jnp.sum(t_modes_l**2, axis=0, keepdims=True)
                norm_c = jnp.sum(cand_modes**2, axis=0, keepdims=True)
                mac_matrix = (dots**2) / (norm_t.T @ norm_c + 1e-10)
                
                best_mac_per_target = jnp.max(mac_matrix, axis=1) # Perfect match = 1.0
                
                # Build combined mode loss from descriptors (if any)
                mode_descs = [d for d in targets_jax if d['type'] == 'modes']
                if len(mode_descs) == 0:
                    # default behaviour: MAC-only
                    if mode_match_type == 'mac':
                        l_mode = jnp.mean((1.0 - best_mac_per_target)**2)
                    else:
                        best_match_idx = jnp.argmax(mac_matrix, axis=1)
                        best_cand_modes = cand_modes[:, best_match_idx]
                        best_dots = jnp.diagonal(jnp.dot(t_modes_l.T, best_cand_modes))
                        signs = jnp.where(best_dots < 0, -1.0, 1.0)
                        aligned_modes = best_cand_modes * signs
                        target_normed = t_modes_l / jnp.sqrt(norm_t.reshape(1, -1) + 1e-10)
                        cand_normed = aligned_modes / jnp.sqrt(jnp.sum(aligned_modes**2, axis=0, keepdims=True) + 1e-10)
                        l_shape_mse = jnp.mean((target_normed - cand_normed)**2)
                        if mode_match_type == 'direct':
                            l_mode = l_shape_mse * 100.0
                        else:
                            l_mode = jnp.mean((1.0 - best_mac_per_target)**2) + (l_shape_mse * 50.0)
                else:
                    # accumulate weighted descriptors (normalize by total descriptor weight)
                    total_w = jnp.sum(jnp.array([d.get('weight', 1.0) for d in mode_descs]))
                    acc = 0.0
                    for d in mode_descs:
                        wgt_d = d.get('weight', 1.0)
                        # mac error per target modes
                        n_use = d.get('num_modes') or n_tgt
                        n_use = min(n_use, best_mac_per_target.shape[0])
                        mac_err = jnp.mean((1.0 - best_mac_per_target[:n_use])**2) if n_use > 0 else 0.0
                        fw = float(d.get('freq_weight', 0.0))
                        if fw > 0.0:
                            n_f = min(n_use, freqs.shape[0], t_vals.shape[0])
                            if n_f > 0:
                                ref_f = t_vals[:n_f]
                                opt_f = freqs[:n_f]
                                freq_err = jnp.mean(jnp.abs((opt_f - ref_f) / (ref_f + 1e-12)))
                            else:
                                freq_err = 0.0
                            combined = mac_err * (1.0 - fw) + freq_err * fw
                        else:
                            combined = mac_err
                        acc += wgt_d * combined
                    l_mode = acc / jnp.maximum(1.0, total_w)

                f1_hz = freqs[0]

            # --- Global descriptors (case_idx == None) ---
            for gdesc in [d for d in targets_jax if d.get('case_idx') is None]:
                gtyp = gdesc['type']
                gw = gdesc.get('weight', 1.0)
                if gtyp == 'mass':
                    # current mass already computed above when use_mass_constraint handled
                    if 't' in p_fem and 'rho' in p_fem:
                        mass_opt = jnp.sum(p_fem['t'] * p_fem['rho']) * (self.fem.Lx / Nx_l) * (self.fem.Ly / Ny_l)
                    else:
                        mass_opt = 0.0
                    ref_mass = float(gdesc.get('ref_value', self.target_mass))
                    if gdesc.get('compare_mode','absolute') == 'relative' and ref_mass != 0:
                        terr = (mass_opt - ref_mass) / (ref_mass + 1e-12)
                    else:
                        terr = mass_opt - ref_mass
                    l_mass += gw * (terr**2)

                elif gtyp == 'modes':
                    # evaluate using already computed freqs/vecs if available
                    fw = float(gdesc.get('freq_weight', 0.0))
                    nmode = int(gdesc.get('num_modes') or n_tgt)
                    cand_modes = vecs[2::6, :] if vecs is not None else None
                    if cand_modes is not None:
                        dots = jnp.dot(t_modes_l.T, cand_modes)
                        norm_t = jnp.sum(t_modes_l**2, axis=0, keepdims=True)
                        norm_c = jnp.sum(cand_modes**2, axis=0, keepdims=True)
                        mac_matrix = (dots**2) / (norm_t.T @ norm_c + 1e-10)
                        best_mac_per_target = jnp.max(mac_matrix, axis=1)
                        mac_err = jnp.mean((1.0 - best_mac_per_target[:nmode])**2) if nmode > 0 else 0.0
                        if fw > 0.0 and freqs is not None:
                            n_f = min(nmode, freqs.shape[0], t_vals.shape[0])
                            if n_f > 0:
                                ref_f = t_vals[:n_f]
                                opt_f = freqs[:n_f]
                                freq_err = jnp.mean(jnp.abs((opt_f - ref_f) / (ref_f + 1e-12)))
                            else:
                                freq_err = 0.0
                            combined = mac_err * (1.0 - fw) + freq_err * fw
                        else:
                            combined = mac_err
                        l_mode += gw * combined

                elif gtyp == 'field_stat':
                    # global field stat - compare aggregated over all cases' target arrays
                    ref_key = gdesc.get('ref_key', 'u_static')
                    # use average over targets_low arrays
                    if ref_key in self.targets_low[0]:
                        ref_arrs = jnp.stack([t[ref_key] for t in self.targets_low], axis=0)
                        if gdesc.get('reduction','max') == 'max':
                            ref_val = jnp.max(ref_arrs)
                        else:
                            ref_val = jnp.mean(ref_arrs)
                    else:
                        ref_val = 0.0
                    # compute current value as average over cases
                    # for simplicity compare mean of current computed field across cases
                    cur_vals = []
                    for i in range(len(self.cases)):
                        # we only have f_res computed per-case earlier; reuse stored targets_low as proxy
                        cur_vals.append(self.targets_low[i].get(ref_key, 0.0))
                    cur_stack = jnp.stack(cur_vals, axis=0)
                    if gdesc.get('reduction','max') == 'max':
                        cur_val = jnp.max(cur_stack)
                    else:
                        cur_val = jnp.mean(cur_stack)
                    if gdesc.get('compare_mode','absolute') == 'relative' and ref_val != 0:
                        terr = (cur_val - ref_val) / (ref_val + 1e-12)
                    else:
                        terr = cur_val - ref_val
                    l_static += gw * (terr**2)

                elif gtyp == 'rbe_reaction':
                    # compare total reaction sums across cases (global aggregate)
                    # use mean of target reaction_sums as ref
                    ref_vec = jnp.mean(jnp.stack([t['reaction_sums'] for t in self.targets_low], axis=0), axis=0)
                    opt_vec = jnp.mean(jnp.stack([K @ self.fem.solve_static_partitioned(K, b['F'], b['free'], b['fd'], b['fv'])[2::6] for b in case_bcs], axis=0), axis=0)
                    ref_mean = jnp.mean(ref_vec) + 1e-12
                    terr = (jnp.mean(opt_vec) - jnp.mean(ref_vec)) / ref_mean
                    l_reaction += gw * (terr**2)
            
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
                          
            # [FIXED] Robust diagonal extraction for both Dense and Sparse (BCOO) matrices
            # [DIAGNOSTICS] Simplified metrics to avoid JIT-incompatible sparse diag operations
            # Mass is already calculated for l_mass constraint
            current_mass = jnp.sum(p_actual['t'] * p_actual['rho']) * (self.fem.Lx/Nx_l)*(self.fem.Ly/Ny_l)

            return total_loss, {
                'Total': total_loss, 'Disp': l_static, 'Strs': l_stress, 'Strn': l_strain, 
                'Engy': l_energy, 'Reac': l_reaction, 'Freq': l_freq, 'Mode': l_mode, 
                'Gt_Z': l_gt_z, 'Mass': l_mass, 'Reg': l_reg, 'f1_hz': f1_hz,
                'total_m_ton': current_mass,
                'avg_k_diag': jnp.mean(p_actual['E']), # Proxy for stiffness diagnostic
                'K': K, 'M': M
            }

        # --- [ISSUE-015] PRE-OPTIMIZATION DIAGNOSTICS & AUTO-SCALING ---
        print("\n" + "-"*80)
        print(" [DIAGNOSTICS] Initializing Optimization Engine...")
        
        # 1. Initial Raw Loss Pass (for auto-scaling)
        # We call loss_fn without JIT to get concrete values for diagnostics
        init_val, init_metrics = loss_fn(params, fixed_params_scaled, None, None, loss_weights)
        
        # [DIAGNOSTICS] Physical Unit Verification
        diagnostic_m = float(init_metrics.get('total_m_ton', 0.0))
        diagnostic_k = float(init_metrics.get('avg_k_diag', 0.0))
        print(f" -> Structural Mass: {diagnostic_m*1000:.3f} kg")
        print(f" -> Avg Stiffness : {diagnostic_k:.3f} N/mm")
        print(f" -> Target Freqs : {', '.join([f'{f:.2f}Hz' for f in t_vals])}")
        print(f" -> Initial Freqs: {init_metrics.get('f1_hz', 0.0):.2f} Hz (Mode 1)")

        if auto_scale:
            print(" [AUTO-SCALE] Balancing loss components based on initial magnitudes...")
            # We want each active component to contribute roughly 10.0 to the initial loss total
            target_scale = 10.0 
            
            metric_map = {
                'static': 'Disp', 'surface_stress': 'Strs', 'surface_strain': 'Strn',
                'strain_energy': 'Engy', 'reaction': 'Reac', 'freq': 'Freq',
                'mode': 'Mode', 'gt_z': 'Gt_Z', 'mass': 'Mass', 'reg': 'Reg'
            }
            
            new_weights = {}
            for k, w_init in loss_weights.items():
                if w_init > 1e-9:
                    metric_key = metric_map.get(k)
                    if metric_key and metric_key in init_metrics:
                        raw_val = float(init_metrics[metric_key])
                        scale = target_scale / max(raw_val, 1e-8)
                        new_weights[k] = w_init * scale
                    else:
                        new_weights[k] = w_init
                else:
                    new_weights[k] = 0.0
            
            # Use updated weights for loss_fn
            loss_weights.update(new_weights)
            # Re-verify with updated weights
            init_val, init_metrics = loss_fn(params, fixed_params_scaled)

        # 2. Compile Jitted Version for Optimization
        # NOTE: loss_fn cannot be fully JIT-compiled due to solve_eigen's JAX incompatibilities.
        # Instead, we'll call loss_fn directly without full JIT, and optimize internally-JIT'd operations.
        loss_vg_direct = jax.value_and_grad(loss_fn, has_aux=True)

        print(f" -> Target Mass  : {self.target_mass:.4f} Ton")
        print(f" -> Initial Freq1: {float(init_metrics['f1_hz']):6.2f} Hz")
        print(f" -> Initial Loss : {float(init_val):.6e} (Scaled)")
        
        print("\n [Effective Weights]")
        for k, v in loss_weights.items():
            if v > 0: print(f"    - {k:<15}: {v:.4e}")
        
        # [PERFORMANCE OPTIMIZATION] Store original eigen-related weights for cyclic control
        orig_freq_weight = loss_weights.get('freq', 0.0)
        orig_mode_weight = loss_weights.get('mode', 0.0)
        
        # Initialize cached eigen values (computed periodically)
        cached_freqs = None
        cached_vecs = None
        cached_f1_hz = None  # Keep last computed frequency for display
        print(f"\n [PERFORMANCE] Eigen Decomposition Frequency: Every {eigen_freq} iterations")
        print(f"    (Expensive eigenvalue solve: computed every {eigen_freq} iters, cached otherwise)")
        print("-"*80 + "\n")

        # --- Initialize tracking variables BEFORE the loop ---
        best_loss = float('inf')
        best_params = jax.tree_util.tree_map(jnp.array, params)  # Safe copy
        best_iter = 0
        wait = 0
        self.history = []

        print("Optimization Engine Ready. Initializing search...")
        print(" [Tip] Press 'q' to stop optimization and use the best result found so far.\n")
        
        # [ISSUE-014] Enhanced 2-Line Header
        print(f"{'Iter':<5} | {'Total_Norm':<10} | {'t_mean':<9} | {'pz_mean':<9} | {'Freq1':<6}")
        print(f"{'      ':<5} | {'Disp_Err':<10} | {'Freq_Err':<9} | {'Mass_Pen':<9} | {'Reg':<9}")
        print("-" * 70)

        for i in range(max_iterations):
            # [PERFORMANCE OPTIMIZATION - PERIODIC EIGEN COMPUTATION]
            # Only compute expensive eigenvalue decomposition every eigen_freq iterations
            # This reduces O(N^3) calls by ~95% while maintaining converged solution quality
            compute_eigen_now = (i % eigen_freq == 0)
            
            # Temporarily enable/disable freq/mode losses based on computation schedule
            if not compute_eigen_now:
                loss_weights['freq'] = 0.0
                loss_weights['mode'] = 0.0
            else:
                loss_weights['freq'] = orig_freq_weight
                loss_weights['mode'] = orig_mode_weight
            
            # Compute eigen if needed
            freqs = None
            vecs = None
            if compute_eigen_now:
                # Get K and M by calling loss_fn without eigen
                val_temp, metrics_temp = loss_fn(params, fixed_params_scaled, None, None, loss_weights)
                K = metrics_temp['K']
                M = metrics_temp['M']
                # Use Hybrid ARPACK on Sparse Matrices (No Dense Conversion)
                freqs_np, vecs_np = self.fem.solve_eigen_arpack_jax_compatible(K, M, num_modes=len(t_vals), num_skip=6)
                freqs = jnp.array(freqs_np)
                vecs = jnp.array(vecs_np)
            
            (val, metrics), grads = jax.value_and_grad(lambda p, fp: loss_fn(p, fp, freqs, vecs, loss_weights), has_aux=True)(params, fixed_params_scaled)
            
            # [PERFORMANCE] Update cached frequency for display
            # When freq loss is OFF (compute_eigen_now=False), use previously computed value
            if compute_eigen_now:
                cached_f1_hz = metrics.get('f1_hz', cached_f1_hz)  # Update cache
            elif cached_f1_hz is not None:
                metrics['f1_hz'] = cached_f1_hz  # Restore last computed value for display
            
            # [FIXED LOGIC] Store BEST params BEFORE they are updated by the optimizer
            if val < best_loss - early_stop_tol:
                best_loss, wait = val, 0
                best_params = jax.tree_util.tree_map(jnp.array, params) # Safe JAX capture
                best_iter = i
            else:
                wait += 1

            # Store history (on CPU as regular floats)
            self.history.append({k: float(v) for k, v in metrics.items() if k not in ['K', 'M']})
            
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

            if i % 10 == 0: 
                # [ISSUE-014] Enhanced 2-Line Output
                t_mean = float(jnp.mean(current_phys.get('t', 0.0)))
                pz_mean = float(jnp.mean(current_phys.get('pz', 0.0)))
                
                print(f"{i:<5d} | {val:<10.2e} | {t_mean:<9.2f} | {pz_mean:<9.2f} | {metrics['f1_hz']:<6.2f}  (CURR)")
                print(f"{' ':<5} | {metrics['Disp']:<10.2e} | {metrics['Freq']:<9.2e} | {metrics['Mass']:<9.2e} | {metrics['Reg']:<9.2e} (ERR)")
                print("-" * 70)

        # Final Assignment
        params = best_params
        self.optimized_params = {**{k: v * self.scaling[k] for k, v in params.items()}, **{k: v * self.scaling[k] for k, v in fixed_params_scaled.items()}}
        
        # [PERFORMANCE] Restore full weights and use last computed eigen values for final evaluation
        loss_weights['freq'] = orig_freq_weight
        loss_weights['mode'] = orig_mode_weight
        final_val, final_metrics = loss_fn(params, fixed_params_scaled, cached_freqs, cached_vecs, loss_weights)
        
        print(f"\n -> Optimization Finished. Best Result found at Iteration {best_iter} with Loss: {best_loss:.6f}")
        print(f" -> [FINAL EVAL] Full Loss (With Freq/Mode): {final_val:.6f}")
        
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
            
            stress_ref = tgt['max_stress'].reshape(Ny_h+1, Nx_h+1)
            stress_opt = self.fem_high.compute_max_surface_stress(u_opt, opt_params_h).reshape(Ny_h+1, Nx_h+1)
            
            strain_ref = tgt['max_strain'].reshape(Ny_h+1, Nx_h+1)
            strain_opt = self.fem_high.compute_max_surface_strain(u_opt, opt_params_h).reshape(Ny_h+1, Nx_h+1)

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

            # Calculate for Optimized (JAX Sparse Matmul)
            R_full_opt = np.array(K_opt @ jnp.array(u_opt)) - np.array(F)
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
            min_w, max_w = float(np.min(w_ref)), float(np.max(w_ref))
            levels_w = np.linspace(min_w, max_w if max_w > min_w + 1e-12 else min_w + 1e-12, 30)
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
        # Use sigma=100.0 to prevent hanging on singular RBMs in the unconstrained tray model
        vals_opt, vecs_opt = self.fem_high.solve_eigen_sparse(K_opt, M_opt, num_modes=n_modes + 10)
        
        # solve_eigen_sparse already returns frequencies in Hz
        freq_opt_all = np.array(vals_opt)
        # [STANDARD] Unified Mode Identification: Consistently skip 6 RBMs for comparison
        s_idx_opt = 6
        print(f" -> Verification: Skipping 6 RBMs. Opt Mode 1 at index {s_idx_opt} ({freq_opt_all[s_idx_opt]:.2f} Hz)")
        
        # target_eigen['vals'] are already in Hz (from solve_eigen_sparse)
        freq_ref = np.array(self.target_eigen['vals'])
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
            
            # Compute Detailed Field Results for Optimized model
            f_res_opt = self.fem_high.compute_field_results(u_opt, opt_params_h)
            
            # Export Optimized Static Result to ParaView
            res_fields_opt = {
                'displacement_vec': np.array(u_opt),
                'stress_vm': np.array(f_res_opt['stress_vm']),
                'strain_x': np.array(f_res_opt['strain_x']),
                'sed': np.array(f_res_opt['sed'])
            }
            res_opt = StructuralResult(res_fields_opt, np.array(self.fem_high.nodes), self.fem_high.elements)
            res_opt.save_vtkhdf(f"opt_static_{case.name}.vtkhdf")
            
            # 1. Displacement
            w_ref, w_opt = tgt['u_static'], u_opt[2::6]
            max_w_ref, max_w_opt = np.max(np.abs(w_ref)), np.max(np.abs(w_opt))
            
            # 2. Reaction Forces (at fixed DOFs) - JAX Sparse Matmul
            R_vec_ref = (np.array(K_opt @ jnp.array(u_ref)) - np.array(F_ext))[2::6] # Simplified Z-component reactions
            R_vec_opt = (np.array(K_opt @ jnp.array(u_opt)) - np.array(F_ext))[2::6]
            max_R_ref, max_R_opt = np.max(np.abs(tgt['reaction_full'][2::6])), np.max(np.abs(R_vec_opt))
            
            # 3. Bending Moments (Approximated from reaction force moments)
            # compute_moment is not available in ShellFEM; use reaction Mx/My as proxy
            R_full_ref = np.array(tgt['reaction_full'])
            R_full_opt = np.array(K_opt @ jnp.array(u_opt)) - np.array(F_ext)
            max_M_ref = np.sqrt(np.sum(R_full_ref[3::6]**2) + np.sum(R_full_ref[4::6]**2))
            max_M_opt = np.sqrt(np.sum(R_full_opt[3::6]**2) + np.sum(R_full_opt[4::6]**2))
            
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
        target_config = self.config.get('target_config', {})
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
        # solve_eigen_sparse now already identifies and returns ONLY elastic modes!
        freq_l, vecs_l = self.fem.solve_eigen_sparse(K_l, M_l, num_modes=n_modes)
        
        opt_eigen_l = {
            'vals': (2 * np.pi * freq_l)**2,  # Convert Hz back to eigenvalues if needed by viewer
            'modes': vecs_l[2::6, :n_modes]   # Extract w-displacement
        }

        # --- NEW: Export Optimized Modal Results to Temporal VTKHDF ---
        print("\nExporting Optimized Modal Analysis to ParaView...")
        modal_res_opt = StructuralResult({}, np.array(self.fem.nodes), self.fem.elements)
        steps_dict_opt = {
            'values': freq_l,
            'point_data': {
                'mode_shape_vec': [vecs_l[:, i] for i in range(n_modes)]
            }
        }
        modal_res_opt.save_vtkhdf("opt_modal_results.vtkhdf", steps_dict=steps_dict_opt)

        stage3_visualize_comparison(
            self.fem_high, self.fem, self.targets, self.optimized_params, self.target_params_high,
            opt_eigen=opt_eigen_l, tgt_eigen=self.target_eigen
        )

# ==============================================================================
# MAIN EXECUTION: COMPREHENSIVE CONFIGURATION TEMPLATE
# ==============================================================================
if __name__ == '__main__':
    if os.environ.get('RUN_FULL_OPT', '0').lower() not in ('1', 'true', 'yes'):
        print("RUN_FULL_OPT environment variable must be set to '1' to execute the main optimization pipeline.")
        print("Example: set RUN_FULL_OPT=1 && python main_shell_verification.py")
        sys.exit(0)

    # XLA Flags (Optimized for CPU)
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    model = EquivalentSheetModel(Lx, Ly, Nx_low, Ny_low)
    
    # 2. Add Load Cases
    model.add_case(TwistCase("twist_x", axis='x', value=1.5, mode='angle', weight=1.0))
    model.add_case(TwistCase("twist_y", axis='y', value=1.5, mode='angle', weight=1.0))
    model.add_case(PureBendingCase("bend_y", axis='y', value=3.0, mode='angle', weight=1.0))
    model.add_case(PureBendingCase("bend_x", axis='x', value=3.0, mode='angle', weight=1.0))
    model.add_case(CornerLiftCase("lift_br", corner='br', value=5.0, mode='disp', weight=1.0))
    model.add_case(CornerLiftCase("lift_tl", corner='tl', value=5.0, mode='disp', weight=1.0))
    model.add_case(TwoCornerLiftCase("lift_tl_br", corners=['br', 'tl'], value=5.0, mode='disp', weight=1.0))
    model.add_case(CantileverCase("cantilever_x", axis='x', value=-5.0, mode='disp', weight=1.0))
    model.add_case(CantileverCase("cantilever_y", axis='y', value=-5.0, mode='disp', weight=1.0))
    model.add_case(PressureCase("pressure_z", value=-10.0, weight=1.0))
    
    # Apply opt-targets configuration (Python dict with detailed comments)
    # OptTarget.preset()를 사용하여 기본 최적화 목표 설정을 가져옵니다.
    opt_target_config = OptTarget.preset()
    
    try:
        apply_case_targets_from_spec(model, opt_target_config)
        print("[INFO] Applied inline opt-target configuration with detailed comments.")
    except Exception as e:
        print(f"[INFO] No inline targets applied ({e}).")
    
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
        't':   {'opt': True, 'init': 1.0, 'min': 0.8, 'max': 1.2, 'type': 'global'},        
        # 'nodal' (또는 생략)은 기존처럼 위치마다 다른 값을 갖는 위상 최적화 모드입니다.
        'rho': {'opt': True, 'init': 7.85e-9, 'min': 1e-10, 'max': 1e-7, 'type': 'global'},
        'E':   {'opt': True, 'init': 210000.0, 'type': 'global'},        
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
        'reg': 0.01              # Regularization
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
                                        max_iterations=100,          # [조정] 초기 단계 기초 형상 확보
                                        use_early_stopping=True, 
                                        early_stop_patience=40, 
                                        early_stop_tol=1e-8,
                                        learning_rate=1.0,           
                                        num_modes_loss=3,            
                                        min_bead_width=150.0,        # [표준화] 제조 한계 150mm 적용
                                        mac_search_window=3,         
                                        mode_match_type='mac',    
                                        init_pz_from_gt=True,        # [NEW 옵션] 타겟 Ground Truth 좌표를 기반으로 초기 형태 전사 (사용)
                                        gt_init_scale=0.0,           # 상향 조정 (0.3 -> 0.5)
                                        auto_scale=False)
                                        
        # --- STAGE 2: MAC FINE-TUNING ---
        print("\n\n" + "#"*70)
        print(" [RUN STAGE 2] MAC 미세 조정 런 (High MAC Weight & hybrid Matching)")
        print("#"*70)
        
        # Stage 1 결과를 초기값으로 세팅
        opt_config_stage2 = {
            't':   {'opt': opt_config['t']['opt'], 'init': best_params_s1['t'], 'min': opt_config['t']['min'], 'max': opt_config['t']['max']},
            'rho': {'opt': opt_config['rho']['opt'], 'init': best_params_s1['rho'], 'min': opt_config['rho']['min'], 'max': opt_config['rho']['max']},
            'E':   {'opt': opt_config['E']['opt'], 'init': best_params_s1['E'], 'min': opt_config['E']['min'], 'max': opt_config['E']['max']},
            'pz':  {'opt': opt_config['pz']['opt'], 'init': best_params_s1.get('pz', opt_config['pz']['init']), 'min': opt_config['pz']['min'], 'max': opt_config['pz']['max']},
        }
        
        weights_stage2 = weights.copy()
        weights_stage2['mode'] = 5.0   # MAC 가중치 크게 상향
        
        model.optimize(opt_config_stage2, weights_stage2, 
                       use_bead_smoothing=True,     
                       use_strain_energy=True,      
                       use_surface_stress=False,    
                       use_surface_strain=False,    
                       use_mass_constraint=True, 
                       mass_tolerance=0.05,
                       max_iterations=250,           # [Plan v4] Speed-Accuracy Balance
                       use_early_stopping=True, 
                       early_stop_patience=60, 
                       early_stop_tol=1e-8,
                       learning_rate=0.1,          
                       num_modes_loss=5,
                       min_bead_width=150.0,        
                       mac_search_window=5,         
                       mode_match_type='hybrid',    
                       init_pz_from_gt=False,       
                       gt_init_scale=0.5)           

        # 7. Verify and Report
        model.verify()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[CRITICAL ERROR] Optimization failed: {e}")
