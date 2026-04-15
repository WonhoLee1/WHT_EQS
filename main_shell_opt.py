# ==============================================================================
# main_shell_opt.py (OptTarget Driven Optimization Version)
# ==============================================================================
"""
[이론적 배경 및 AI Agent 이해를 위한 아키텍처 가이드]
이 스크립트는 6-DOF 쉘(Shell) 요소를 기반으로 한 '등가 평판(Equivalent Sheet) 최적화' 엔진입니다.
복잡한 3D 형상(비드, 리브 등)을 가진 원본 모델의 기계적 거동(변위, 응력, 진동수 등)을 
단순한 평판의 두께(t), 밀도(rho), 강성(E), 그리고 국부적 높이(pz, Topography) 변수로 모사합니다.

핵심 기술:
1. JAX 기반 자동 미분(Auto-Diff): 모든 FEM 행렬 조립과 해석 과정이 미분 가능하도록 설계되어, Adam 옵티마이저가 손실 함수의 기울기(Gradient)를 직접 계산합니다.
2. AOT (Ahead-Of-Time) Pruning: 파이썬 조건문을 활용해 JAX JIT 컴파일 전 불필요한 하중 케이스 연산을 완벽히 제거(가지치기)합니다.
3. Two-Track JIT: 고유치 해석(Eigen)이 포함된 무거운 연산과 정적 해석만 포함된 가벼운 연산을 분리 컴파일하여 연산 속도와 미분 무결성을 모두 확보합니다.
"""
import os
os.environ["NON_INTERACTIVE"] = "1"
import sys
# [CRITICAL FIX] Windows 콘솔 인코딩 문제 해결 (Python 3.7+ 표준 방식)
if sys.platform == "win32":
    # sys.stdout이 이미 io.TextIOWrapper라면 reconfigure() 사용 가능
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import msvcrt
import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
from scipy.interpolate import griddata

# Import dependencies from the existing verification script
from main_shell_verification import (
    Lx, Ly, Nx_low, Ny_low, Nx_high, Ny_high,
    EquivalentSheetModel, TwistCase, PureBendingCase, 
    CornerLiftCase, TwoCornerLiftCase, CantileverCase, PressureCase
)
from opt_targets import OptTarget, TargetType, apply_case_targets_from_spec
from solver import safe_eigh  # [CRITICAL FIX 27] 안전한 고유치 미분을 위한 임포
from wh_utils import WHTable, wh_print_banner # [UI] 분리된 전용 유틸리티 사용

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
    # [ROBUST] Use ACTUAL 3D nodal coordinates from the Mesh objects
    # Using [:, :3] instead of [:, :2] prevents "Dimension Collapse" on vertical walls 
    # which was causing MAC=0.0 matching failures.
    pts_l = np.array(self.fem.nodes)[:, :3]
    pts_h = np.array(self.fem_high.nodes)[:, :3]
    
    # 1. 대상 모델에 OptTarget 설정 적용
    if hasattr(self, 'global_opt_targets'):
        self.global_opt_targets.clear()
    for case in self.cases:
        if hasattr(case, 'opt_targets'):
            case.opt_targets.clear()
            
    apply_case_targets_from_spec(self, opt_target_config)
    
    # [FINAL DEFINITIVE FIX] apply_case_targets_from_spec\uc740 'global_targets' \ud0a4\uc758 list\ub97c
    # case\uc774\ub984\uc73c\ub85c \ucc2d\uace0 \ud574\ub2f9 case\uac00 \uc5c6\uc73c\uba74 \uc870\uc6a9\ud788 \ubb35\uc0b4\ud569\ub2c8\ub2e4.
    # -> \ud574\ub2f9 \ud0c0\uac9f\uc774 self.global_opt_targets\uc5d0 \uc808\ub300 \ubc14\uc778\ub529\ub418\uc9c0 \uc54a\uc544 modes/mass\ub97c \uc801\uc6a9 \ubd88\ub9c9 -> MAC=0, dt=0
    # \uc774\ub97c \uc9c1\uc811 \ud30c\uc2f1\ud558\uc5ec \ub4f1\ub85d\ud569\ub2c8\ub2e4.
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
    
    # 안전한 보간을 위한 헬퍼 함수 (Boundary NaN 방지)
    def safe_interp(src_pts, src_vals, dst_pts):
        val = griddata(src_pts, np.array(src_vals), dst_pts, method='linear')
        nan_mask = np.isnan(val)
        if np.any(nan_mask):
            # 선형 보간 실패(경계 밖) 지점은 가장 가까운 노드의 값으로 복구
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
            # [CRITICAL FIX 29] Reaction Force Dimensional Collapse (유령 지지대 매핑) 방지
            # 해상도가 다를 때 고해상도(N_high)의 반력 배열을 저해상도(N_low) 인덱스로 그대로 참조하면,
            # 물리적으로 완전히 엉뚱한 내부 노드의 힘을 경계 반력으로 오인하여 학습이 붕괴합니다.
            R_h_nodes = np.array(R_tgt).reshape(-1, 6)
            R_l_nodes = safe_interp(pts_h, R_h_nodes, pts_l)
            
            # 총 하중(Total Force) 보존을 위해 노드 밀도(면적) 비율로 스케일링 보정
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

    # 모드 및 Z-좌표 매핑
    num_modes_loss = 5
    for global_target in getattr(self, 'global_opt_targets', []):
        if hasattr(global_target, 'target_type') and global_target.target_type == TargetType.MODES:
            if hasattr(global_target, 'num_modes') and global_target.num_modes:
                num_modes_loss = global_target.num_modes

    # [CRITICAL FIX] num_modes_loss가 실제 로드된 데이터 개수보다 크면 IndexError 발생
    # 데이터의 실제 열(Column) 개수를 기준으로 자동 클램핑합니다.
    t_modes_h_np = np.array(self.target_eigen['modes'])
    num_modes_available = t_modes_h_np.shape[1]
    num_modes_loss = min(num_modes_loss, num_modes_available)
    
    t_vals = np.array(self.target_eigen['vals'])[:num_modes_loss]


    
    # [STAGE 1: FUNDAMENTAL FIX] Parametric Modal Projection (Multi-Resolution Tech)
    # 3D jittered 보간은 지형 변화 시 매핑 위치가 어긋나는 본질적 불안정함이 있습니다.
    # 대신 셸 본연의 파라미터 공간(Lx, Ly)에서의 2D 정규화 보간을 통해 정합성을 100% 확보합니다.
    xh_base = np.linspace(0, self.fem.Lx, Nx_high + 1)
    yh_base = np.linspace(0, self.fem.Ly, Ny_high + 1)
    XH, YH = np.meshgrid(xh_base, yh_base, indexing='xy')
    
    xl_base = np.linspace(0, self.fem.Lx, Nx_l + 1)
    yl_base = np.linspace(0, self.fem.Ly, Ny_l + 1)
    XL, YL = np.meshgrid(xl_base, yl_base, indexing='xy')

    n_h = len(pts_h)
    
    if is_same_res:
        # [BUG FIX 1] target_eigen['modes']는 generate_targets에서 이미 Z-변위만 저장됨
        # (N_nodes, n_modes) 형태 — reshape(-1,6,...) 이중 슬라이싱하면 데이터 오염!
        t_modes_l = jnp.array(t_modes_h_np[:n_h, :num_modes_loss])
    else:
        # 6-DOF 전체 데이터에서 Z-변위(index 2) 추출 (이미 Z-only면 패스)
        n_total_rows = t_modes_h_np.shape[0]
        if n_total_rows == n_h * 6:
            # [rare] full 6-DOF stored
            t_modes_res = t_modes_h_np.reshape(-1, 6, t_modes_h_np.shape[1])
            t_modes_h_z = t_modes_res[:n_h, 2, :]
        elif n_total_rows == n_h:
            # [normal] already Z-only (N_h_nodes, n_modes)
            t_modes_h_z = t_modes_h_np
        else:
            # fallback: take first n_h rows
            t_modes_h_z = t_modes_h_np[:n_h, :]

        # [WHTOOLS Technology] 2D Parametric Projection
        # 수직벽이 있는 트레이 형상이라도 모드 형상(w)은 바닥면 그리드 위에서 정의되므로
        # 2D 좌표(XH, YH) -> (XL, YL) 매핑이 가장 정확하고 본질적인 매핑 방법입니다.
        from scipy.interpolate import griddata as scipy_griddata
        pts_h_2d = np.stack([XH.flatten(), YH.flatten()], axis=1)
        pts_l_2d = np.stack([XL.flatten(), YL.flatten()], axis=1)
        
        t_modes_l_list = []
        for i in range(num_modes_loss):
            mapped = scipy_griddata(pts_h_2d, t_modes_h_z[:, i], pts_l_2d, method='linear')
            # NaN 발생 시 Nearest로 보정하여 불연속성 최소화
            nan_mask = np.isnan(mapped)
            if np.any(nan_mask):
                mapped[nan_mask] = scipy_griddata(pts_h_2d, t_modes_h_z[:, i], pts_l_2d[nan_mask], method='nearest')
            t_modes_l_list.append(mapped)
            
        t_modes_l = jnp.array(np.stack(t_modes_l_list, axis=1))
    
    print(f" -> Fundamental Multi-Res Mapping: {t_modes_l.shape} nodes/modes matched via 2D Parametric Space.")
        
    target_z_low = None
    if 'z' in self.target_params_high:
        z_h_flat = self.target_params_high['z']
        target_z_low = jnp.array(z_h_flat if is_same_res else safe_interp(pts_h, z_h_flat, pts_l)).reshape(Nx_l+1, Ny_l+1)

    # 3. 최적화 파라미터 초기화 및 스케일링
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
            # [PERFORMANCE] If not optimized, keep as scalar even if type is 'local' to reduce JAX graph complexity
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

    # 필터 커널 및 옵티마이저 설정
    # [이론적 배경: Topography 필터링 및 Checkerboarding 방지]
    # 요소별 두께(t)나 높이(pz)를 독립적으로 최적화하면, 수치적 불안정성으로 인해 
    # 체스판 무늬(Checkerboarding) 형태의 비현실적인 요철(Singularity)이 발생합니다.
    # 이를 막고 실제 프레스 성형(제조)이 가능한 곡면(최소 비드 폭, min_bead_width)을 얻기 위해, 
    # 저주파 통과 필터(2D Convolution)를 JAX 그래프 내부에 포함시켜 파라미터 변화를 부드럽게(Smooth) 만듭니다.
    filter_kernel = None
    if use_bead_smoothing and min_bead_width > 0:
        dx_l, dy_l = self.fem.Lx/Nx_l, self.fem.Ly/Ny_l
        rx, ry = int(np.ceil(min_bead_width / (2*dx_l))), int(np.ceil(min_bead_width / (2*dy_l)))
        kx, ky = np.linspace(-rx, rx, 2*rx+1), np.linspace(-ry, ry, 2*ry+1)
        KX, KY = np.meshgrid(kx, ky, indexing='ij') # [최종 수정] 파라미터 차원이 (Nx, Ny)로 정상화됨에 따라 필터 커널도 'ij'로 완벽 동기화
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
        # [CRITICAL FIX 22] 수직(Z) 방향 변위가 구속된 모든 지지대 노드 수집 (DOF % 6 == 2)
        z_dofs = [int(d) for d in fd if d % 6 == 2]
        z_lock_nodes.update([d // 6 for d in z_dofs])
        
    # 동적 지지대 마스크 생성 (1.0=이동가능, 0.0=이동불가)
    z_lock_mask_1d = jnp.ones((Nx_l+1) * (Ny_l+1))
    if z_lock_nodes:
        z_lock_mask_1d = z_lock_mask_1d.at[jnp.array(list(z_lock_nodes))].set(0.0)

    # 4. OptTarget 목록 직렬화 (JIT 내부 순회용)
    targets_jax = []
    target_info = [] # 테이블 출력용 메타데이터
    
    # 케이스별 타겟
    for ci, case in enumerate(self.cases):
        for ot in getattr(case, 'opt_targets', []):
            desc = {'case_idx': ci, 'type': ot.target_type.value, 'weight': float(ot.weight)}
            desc['compare_mode'] = ot.compare_mode.value
            if ot.target_type == TargetType.FIELD_STAT:
                desc['ref_key'] = {'stress_vm': 'max_stress', 'max_strain': 'max_strain', 'strain_energy_density': 'strain_energy_density'}.get(ot.field, 'u_static')
                desc['reduction'] = ot.reduction.value
            elif ot.target_type == TargetType.RBE_REACTION:
                desc['ref_key'] = 'reaction_full'  # 스칼라 합이 아닌 전체 벡터 분포 추종
            targets_jax.append(desc)
            target_info.append({'case': case.name, 'type': ot.target_type.value, 'field': getattr(ot, 'field', None) or getattr(ot, 'component', None) or '', 'mode': ot.compare_mode.value})
            
    # 글로벌 타겟
    for ot in getattr(self, 'global_opt_targets', []):
        desc = {'case_idx': None, 'type': ot.target_type.value, 'weight': float(ot.weight)}
        desc['compare_mode'] = ot.compare_mode.value
        if ot.target_type == TargetType.MASS:
            desc['ref_value'] = float(ot.ref_value) if ot.ref_value else float(self.target_mass)
            desc['tolerance'] = float(getattr(ot, 'tolerance', 0.05))
        elif ot.target_type == TargetType.MODES:
            desc['num_modes'] = getattr(ot, 'num_modes', num_modes_loss)
            desc['freq_weight'] = float(getattr(ot, 'freq_weight', 0.0))
        targets_jax.append(desc)
        target_info.append({'case': 'Global', 'type': ot.target_type.value, 'field': getattr(ot, 'field', None) or getattr(ot, 'component', None) or '', 'mode': ot.compare_mode.value})

    # 4.5. 사전 의존성(Dependency) 분석
    # [이론적 배경: AOT(Ahead-Of-Time) 가지치기 및 JAX Meta-programming]
    # JAX의 @jax.jit은 파이썬 코드를 처음 실행(Trace)할 때, 조건문(if)의 상태가 정적(Static)이면
    # False로 평가된 블록을 C++ 연산 그래프(XLA)에서 완전히 제거(Dead Code Elimination)합니다.
    # 이를 이용해 현재 사용자가 최적화 목표(OptTarget)에 등록하지 않은 하중 케이스나 
    # 불필요한 파생 물리량(응력, 변형률 등)의 무거운 행렬 연산을 컴파일 단계에서 원천 차단하여 
    # 미분 연산량과 메모리 사용량을 극단적으로 줄입니다.
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

    # 5. JIT 컴파일 가능한 목적 함수 (Loss Function) 정의
    def loss_fn(p_scaled, fixed_p_scaled, do_eigen=False, cached_freqs=None, cached_vecs=None):
        combined_phys = {k: v * scaling_consts[k] for k, v in p_scaled.items()}
        for k, v in fixed_p_scaled.items(): combined_phys[k] = v * scaling_consts[k]
        
        # [CRITICAL FIX 18] 범용 스무딩(Universal Filtering) 적용
        # pz뿐만 아니라 t, rho, E 등 최적화 대상인 모든 2D 로컬 파라미터에 스무딩 필터를 적용하여
        # 체스판 무늬(Checkerboarding) 수치 불안정을 완벽히 차단합니다.
        if filter_kernel is not None:
            pad_h, pad_w = filter_kernel.shape[0]//2, filter_kernel.shape[1]//2
            for k in combined_phys:
                if combined_phys[k].ndim > 1:
                    padded = jnp.pad(combined_phys[k], ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
                    combined_phys[k] = jax.scipy.signal.convolve2d(padded, filter_kernel, mode='valid')

        if 'pz' in combined_phys:
            pz_val = combined_phys['pz']
            z_map = jnp.full((Nx_l+1, Ny_l+1), pz_val[0]) if (pz_val.ndim == 1 and pz_val.shape[0] == 1) else combined_phys['pz']
            z_masked = (z_map.flatten() * z_lock_mask_1d)  # 1D flat (N_nodes,)
            combined_phys['z'] = z_masked  # [BUG FIX 2] 솔버는 1D flat 배열만 인식: shape==(n_n,)

        def broadcast_nodal(val):
            return jnp.full(((Nx_l+1)*(Ny_l+1),), val[0]) if val.ndim == 1 and val.shape[0] == 1 else val.flatten()

        p_fem = {k: broadcast_nodal(v) for k, v in combined_phys.items() if k != 'pz'}
        K, M = self.fem.assemble(p_fem)
        
        freqs, vecs_filtered = None, None
        if do_eigen:
            # [이론적 배경: JAX 호환 미분 가능 고유치 해석 (Differentiable Eigensolver)]
            # 구조 동역학의 일반화된 고유치 문제(Generalized Eigenvalue Problem): Kx = λMx
            # JAX는 eigh(A, B)의 다중 행렬 인자 미분을 지원하지 않으므로, 이를 표준 고유치 문제(Ax=λx)로 변환해야 합니다.
            # 질량 행렬(M)이 집중 질량(Lumped Mass)으로 구성되어 대각(Diagonal) 성분이 지배적이라는 점을 이용해,
            # A = M^(-1/2) * K * M^(-1/2) 형태의 대칭 행렬로 변환한 후 jnp.linalg.eigh(A)를 수행합니다.
            m_diag = jnp.maximum(jnp.diag(M), 1e-15)
            m_inv_sqrt = 1.0 / jnp.sqrt(m_diag)
            K_stable = (K + K.T) / 2.0
            A = K_stable * m_inv_sqrt[:, None] * m_inv_sqrt[None, :]
            
            # [이론적 배경: 미세 섭동 트릭 (Micro-Perturbation Trick)]
            # 강체 모드(Rigid Body Modes)처럼 동일한 고유치(λ=0)가 여러 개 존재할 경우, (Repeated Roots)
            # JAX 내부의 고유벡터 미분 공식에서 분모(λ_i - λ_j)가 0이 되어 NaN 그라디언트가 폭주합니다.
            # 대각선에 서로 다른 미세한 로그 스케일의 값(eps_unique)을 더해 중복근을 강제로 찢어주어 미분 안정성을 확보합니다.
            eps_unique = jnp.logspace(-9, -4, A.shape[0])
            A = A + jnp.diag(eps_unique)
            
            # [CRITICAL FIX 27] 대칭 구조 고유치 미분 폭주(Symmetric Gradient Explosion) 원천 차단
            # eps_unique로 중복근을 찢어주더라도, JAX의 기본 eigh는 1/1e-9 = 10^9 의 거대한 미분값을 발생시켜 
            # 대칭 형태의 모드(예: 정방형 평판)에서 그래디언트를 폭주(NaN)시킵니다.
            vals, vecs_v = safe_eigh(A)
            vecs = vecs_v * m_inv_sqrt[:, None]
            vals = jnp.maximum(vals, 0.0)
            
            # [CRITICAL FIX 3] Mode Swapping 추적을 위한 Search Window(여유분 3개) 부여
            search_window = 3
            # [CRITICAL FIX 6] NaN Gradient Bomb 방지: 0.0 대신 1e-12를 사용하여 sqrt 미분 시 분모가 0이 되는 현상 차단
            freqs = jnp.sqrt(jnp.maximum(vals[6:num_modes_loss + 6 + search_window], 1e-12)) / (2 * jnp.pi)
            vecs_filtered = vecs[:, 6:num_modes_loss + 6 + search_window]
        else:
            # [CRITICAL FIX 16] Ghost Eigen Gradient (Rayleigh Quotient) 복원
            # eigh를 건너뛰는 스텝에서 freqs를 단순 상수로 취급하면 JAX 미분값이 0이 되어 학습이 붕괴(Sawtooth)됩니다.
            # 캐시된 고유벡터와 업데이트된 K, M 행렬을 이용해 레일리 몫(Rayleigh Quotient)을 계산하면,
            # O(N^3) 비용 없이도 '미분 가능한(Differentiable)' 주파수를 완벽하게 근사해 미분 사슬을 살려냅니다!
            if cached_vecs is not None:
                rayleigh_num = jnp.sum(cached_vecs * (K @ cached_vecs), axis=0)
                rayleigh_den = jnp.sum(cached_vecs * (M @ cached_vecs), axis=0)
                approx_vals = rayleigh_num / jnp.maximum(rayleigh_den, 1e-15)
                freqs = jnp.sqrt(jnp.maximum(approx_vals, 1e-12)) / (2 * jnp.pi)
            else:
                freqs = cached_freqs
            vecs_filtered = cached_vecs
        
        per_w, per_stress, per_strain, per_sed, per_reaction = [], [], [], [], []
        for i, case in enumerate(self.cases):
            req = case_needs[i]
            
            # [초고속 최적화] 해당 케이스에 요구되는 정적 목표(Target)가 아예 없다면, FEM 해석 자체를 건너뜁니다!
            if not any(req.values()):
                per_w.append(jnp.zeros(0)); per_stress.append(jnp.zeros(0))
                per_strain.append(jnp.zeros(0)); per_sed.append(jnp.zeros(0)); per_reaction.append(jnp.zeros(0))
                continue

            b = case_bcs[i]
            u = self.fem.solve_static_partitioned(K, b['F'], b['free'], b['fd'], b['fv'])
            
            # 1. 변위 성분 추출
            per_w.append(u[2::6] if req['u'] else jnp.zeros(0))
            
            # 2. 파생 물리량 계산 (응력, 변형률, 변형 에너지 중 하나라도 필요할 때만 계산)
            f_res = self.fem.compute_field_results(u, p_fem) if (req['stress'] or req['strain'] or req['sed']) else None
            
            per_stress.append(self.fem.compute_max_surface_stress(u, p_fem, field_results=f_res) if req['stress'] else jnp.zeros(0))
            per_strain.append(self.fem.compute_max_surface_strain(u, p_fem, field_results=f_res) if req['strain'] else jnp.zeros(0))
            per_sed.append(self.fem.compute_strain_energy_density(u, p_fem, field_results=f_res) if req['sed'] else jnp.zeros(0))
            
            # 3. 반력(Reaction) 계산
            if req['reaction']:
                R_opt = K @ u - b['F']
                per_reaction.append(R_opt) # [CRITICAL FIX 23] 반력 분포 전체를 저장
            else:
                per_reaction.append(jnp.zeros(0))

        def get_case_arr(key, idx):
            if key == 'u_static': return per_w[idx]
            if key == 'max_stress': return per_stress[idx]
            if key == 'max_strain': return per_strain[idx]
            if key == 'strain_energy_density': return per_sed[idx]
            if key == 'reaction_full': return per_reaction[idx]
            return per_w[idx]

        opt_loss = 0.0
        target_evals = [] # [Current Value, Target Value, Error, Weight]
        
        for desc in targets_jax:
            ci = desc.get('case_idx')
            typ = desc['type']
            wgt = desc['weight']
            comp_mode = desc.get('compare_mode', 'absolute')
            
            val, ref_val, terr = 0.0, 0.0, 0.0
            
            if ci is not None:
                idx = int(ci)
                if typ == 'field_stat':
                    data = get_case_arr(desc['ref_key'], idx)
                    red = desc.get('reduction', 'max')
                    
                    ref_arr = self.targets_low[idx].get(desc['ref_key'], 0.0)
                    
                    # [CRITICAL FIX 1] 필드 전체 형상 매칭 (Field MSE) 복구
                    # 옵티마이저가 꼼수를 쓰지 못하도록, 노드별 변형/응력 맵 전체의 평균제곱오차를 강제합니다.
                    if red == 'mse':
                        # [CRITICAL FIX 13] Zero-Scale Explosion 방지: 물리적 하한선(1e-3)을 두어 노이즈 증폭 차단
                        scale = jnp.maximum(jnp.max(jnp.abs(ref_arr)), 1e-3)
                        terr_arr = (data - ref_arr) / scale if comp_mode == 'relative' else (data - ref_arr)
                        opt_loss += wgt * jnp.mean(terr_arr ** 2)
                        # 출력용 대푯값
                        # [FIX] Display Peak Magnitude for field matches instead of Mean
                        # (Mean can be 0 for symmetric/localized fields, causing misleading Target Val 0)
                        val, ref_val = jnp.max(jnp.abs(data)), jnp.max(jnp.abs(ref_arr))
                        terr = jnp.sqrt(jnp.mean(terr_arr ** 2))
                    else:
                        # 기존 스칼라 통계량 매칭
                        val = jnp.nanmax(data) if red == 'max' else (jnp.nanmean(data) if red == 'mean' else jnp.sqrt(jnp.nanmean(data**2)))
                        if red == 'max': ref_val = jnp.max(ref_arr)
                        elif red == 'mean': ref_val = jnp.mean(ref_arr)
                        else: ref_val = jnp.sqrt(jnp.mean(jnp.square(ref_arr)))
                        
                        # [STABILIZE] Add robust epsilon based on scale for relative error
                        scale = jnp.maximum(jnp.abs(ref_val), 1e-4)
                        terr = (val - ref_val) / scale if comp_mode == 'relative' else (val - ref_val)
                        opt_loss += wgt * (terr ** 2)
                    
                elif typ == 'rbe_reaction':
                    # [CRITICAL FIX 23] Single Node Anchor Cheat 원천 차단
                    # 반력의 합(Sum)만 맞추는 꼼수를 쓰지 못하도록 고정된 경계면(Fixed DOFs) 전체의
                    # 반력 분포(Spatial Distribution) 벡터를 1:1로 강제 매칭합니다.
                    val_arr = get_case_arr('reaction_full', idx)
                    ref_arr = self.targets_low[idx].get('reaction_full', jnp.zeros_like(val_arr))
                    
                    fd = case_bcs[idx]['fd']
                    val_fixed = val_arr[fd]
                    ref_fixed = ref_arr[fd]
                    
                    if comp_mode == 'relative':
                        is_force = (fd % 6) < 3
                        force_scale = jnp.maximum(jnp.max(jnp.where(is_force, jnp.abs(ref_fixed), 0.0)), 1e-3)
                        moment_scale = jnp.maximum(jnp.max(jnp.where(~is_force, jnp.abs(ref_fixed), 0.0)), 1e-3)
                        scale_arr = jnp.where(is_force, force_scale, moment_scale)
                        terr_arr = (val_fixed - ref_fixed) / scale_arr
                    else:
                        terr_arr = (val_fixed - ref_fixed)
                        
                    opt_loss += wgt * jnp.mean(terr_arr ** 2)
                    
                    # 출력용 대푯값
                    # [FIX] Display RMS or Max instead of Mean sign-canceled values
                    val, ref_val = jnp.sqrt(jnp.mean(val_fixed**2)), jnp.sqrt(jnp.mean(ref_fixed**2))
                    terr = jnp.sqrt(jnp.mean(terr_arr ** 2))
                    
            else:
                if typ == 'mass':
                    # [토포그라피 최적화 핵심 보완]
                    # 단순 2D 투영 면적이 아닌 Z 좌표의 기울기를 반영한 '실제 3D 표면적' 기반으로 질량을 계산합니다.
                    # 이를 통해 옵티마이저가 강성 확보를 위해 불필요하게 비드(Z)를 무한정 높이는 편법을 원천 차단합니다.
                    if 't' in p_fem and 'rho' in p_fem:
                        t_2d = p_fem['t'].reshape(Nx_l+1, Ny_l+1)
                        rho_2d = p_fem['rho'].reshape(Nx_l+1, Ny_l+1)
                        dx_val, dy_val = self.fem.Lx / Nx_l, self.fem.Ly / Ny_l
                        
                        # 사다리꼴 적분 가중치 (경계 모서리 절반)
                        W = jnp.ones_like(t_2d)
                        W = W.at[0, :].multiply(0.5).at[-1, :].multiply(0.5)
                        W = W.at[:, 0].multiply(0.5).at[:, -1].multiply(0.5)
                        
                        # [CRITICAL FIX 20] 질량 제약(Mass Constraint) 차원 불일치 버그 수정
                        # Ground Truth의 target_mass는 2D 평면 기준으로 계산됩니다.
                        # 최적화 중에 Z-비드가 생성되며 늘어나는 3D 표면적을 질량에 강제로 반영하면,
                        # 옵티마이저는 불가능한 2D 질량 타겟을 맞추기 위해 모든 비드를 강제로 평평하게 짓눌러버립니다.
                        # 실제 표면적 팽창 억제는 이미 TV 정규화(l_reg)가 담당하고 있으므로, 
                        # 여기서는 2D 투영 질량을 사용하여 타겟과의 수학적 차원을 완벽히 일치시켜야 합니다!
                        val = jnp.sum(t_2d * rho_2d * W) * dx_val * dy_val
                    else:
                        val = 0.0
                    ref_val = float(desc.get('ref_value', self.target_mass))
                    tol = float(desc.get('tolerance', 0.05))
                    
                    if comp_mode == 'relative':
                        rel_err = jnp.abs(val - ref_val) / (jnp.abs(ref_val) + 1e-12)
                        # [CRITICAL FIX 25] C1-Continuous Mass Penalty
                        # 경계면에서 발생하는 기울기 단절(Kink)을 없애고 완벽한 C1 연속성을 확보하여 옵티마이저가 부드럽게 안착하게 유도
                        opt_loss += wgt * (rel_err ** 2 * 0.1 + jnp.maximum(0.0, rel_err - tol) ** 2 * 100.0)
                    else:
                        abs_err = jnp.abs(val - ref_val)
                        opt_loss += wgt * (abs_err ** 2 * 0.1 + jnp.maximum(0.0, abs_err - tol * ref_val) ** 2 * 100.0)
                    
                elif typ == 'modes':
                    fw = float(desc.get('freq_weight', 0.0))
                    nmode = int(desc.get('num_modes') or len(t_vals))
                    if vecs_filtered is not None and freqs is not None:
                        # [MAC ROOT-CAUSE FIX]
                        # vecs_filtered는 전체 DOF 공간(total_dof, n_modes)에서 이미 증덧행렬으로
                        # 쪼바로 계산된 고유벡터입니다.
                        # self.fem.free_dof가 없으므로, 제로패딩 후 scatter는 증복 인덱싱 오류를 유발.
                        # -> 단순히 [2::6] 슬라이싱만으로 Z-변위 성분을 정확히 추출합니다.
                        cand_modes = vecs_filtered[2::6, :]
                        
                        # [SHAPE DIAGNOSTIC - MAC=0 추적]
                        if jax.numpy.ndim(cand_modes) == 2:
                            import jax.debug as jdebug
                            jdebug.print("[SHAPE] t_modes_l: {s1}, cand_modes: {s2}", 
                                        s1=t_modes_l.shape[0], s2=cand_modes.shape[0])
                            jdebug.print("[NORM]  norm_t_max: {nt:.4e}, norm_c_max: {nc:.4e}",
                                        nt=jnp.max(jnp.sum(t_modes_l**2, axis=0)),
                                        nc=jnp.max(jnp.sum(cand_modes**2, axis=0)))
                        
                        # [DEBUG] MAC 연산 차원 정합성 강제 확인 (Nodes, Modes)
                        # dots shape: (TargetModes, CandModes)
                        dots = jnp.dot(t_modes_l.T, cand_modes)
                        norm_t = jnp.sum(t_modes_l**2, axis=0, keepdims=True)
                        norm_c = jnp.sum(cand_modes**2, axis=0, keepdims=True)
                        mac_matrix = (dots**2) / (norm_t.T @ norm_c + 1e-10)
                        best_mac = jnp.max(mac_matrix, axis=1)
                        
                        # [CRITICAL FIX 5] Mode Swapping에 따른 주파수(Frequency) 엇갈림 추적 및 형상 스케일 복구
                        best_match_idx = jnp.argmax(mac_matrix, axis=1)
                        
                        if comp_mode == 'mac':
                            mac_err = jnp.mean((1.0 - best_mac[:nmode])**2) if nmode > 0 else 0.0
                        else:
                            best_cand_modes = cand_modes[:, best_match_idx]
                            
                            best_dots = jnp.diagonal(jnp.dot(t_modes_l.T, best_cand_modes))
                            signs = jnp.where(best_dots < 0, -1.0, 1.0)
                            aligned_modes = best_cand_modes * signs
                            
                            target_normed = t_modes_l / jnp.sqrt(norm_t.reshape(1, -1) + 1e-10)
                            cand_normed = aligned_modes / jnp.sqrt(jnp.sum(aligned_modes**2, axis=0, keepdims=True) + 1e-10)
                            
                            # 노드 개수(해상도)에 의해 에러가 소실되는 것을 막기 위해 노드 축(axis=0)에 대해 sum 적용
                            l_shape_mse = jnp.mean(jnp.sum((target_normed[:, :nmode] - cand_normed[:, :nmode])**2, axis=0)) if nmode > 0 else 0.0
                            mac_err = l_shape_mse * 2.0 if comp_mode == 'direct' else jnp.mean((1.0 - best_mac[:nmode])**2) + (l_shape_mse * 1.0)
                        
                        val = jnp.mean(best_mac[:nmode]) if nmode > 0 else 0.0
                        ref_val = 1.0

                        if fw > 0.0:
                            # 억지로 순서를 맞추지 않고, MAC이 가장 높은(진짜 짝꿍인) 모드의 주파수를 가져와서 목표와 비교
                            matched_freqs = freqs[best_match_idx[:nmode]]
                            # [CRITICAL FIX 7] V-Shape Oscillation 방지: abs() 대신 **2를 사용하여 Adam 옵티마이저의 부드러운 수렴(U-Shape) 유도
                            freq_err = jnp.mean(((matched_freqs - t_vals[:nmode]) / (t_vals[:nmode] + 1e-12))**2) if nmode > 0 else 0.0
                            terr = mac_err * (1.0 - fw) + freq_err * fw
                        else:
                            terr = mac_err
                        opt_loss += wgt * terr
                    else:
                        val, ref_val, terr = 0.0, 1.0, 0.0

            target_evals.append(jnp.stack([val, ref_val, terr, wgt]))

        # 정규화 항 (Regularization)
        l_reg = 0.0
        for k in ['t', 'pz']:
            if k in p_scaled and p_scaled[k].ndim > 1:
                # [CRITICAL FIX 28] 정규화 스케일 압사(Scale Imbalance) 방지
                # pz(수십 mm)와 t(수 mm)의 물리적 스케일 차이로 인해 pz가 과도한 평활화 페널티를 받는 현상을 차단합니다.
                # 각 파라미터의 (최대-최소) 허용 범위를 기준으로 스케일을 평준화합니다.
                p_min = opt_config[k].get('min', 0.0) / scaling_consts[k]
                p_max = opt_config[k].get('max', 1.0) / scaling_consts[k]
                p_range = jnp.maximum(p_max - p_min, 1e-3)
                l_reg += (jnp.mean(jnp.diff(p_scaled[k], axis=0)**2) + jnp.mean(jnp.diff(p_scaled[k], axis=1)**2)) / (p_range**2)
        opt_loss += l_reg * reg_weight

        return opt_loss, {
            'Total': opt_loss,
            'Reg': l_reg * reg_weight,
            'target_evals': jnp.stack(target_evals) if target_evals else jnp.zeros((0,4)),
            'K': K, 'M': M,
            'f1_hz': freqs[0] if freqs is not None and len(freqs) > 0 else 0.0,
            'freqs': freqs,
            'vecs_filtered': vecs_filtered,
            'matched_freqs': matched_freqs if (has_mode_target and 'matched_freqs' in locals()) else None,
            'best_mac': best_mac if 'best_mac' in locals() else None
        }

    # [이론적 배경: Two-Track JIT 아키텍처와 미분 무결성 (Gradient Integrity)]
    # 고유치 해석(eigh)은 O(N^3)의 무거운 연산이므로 매 이터레이션마다 수행하면 속도가 극도로 느려집니다.
    # 따라서 eigen_freq(예: 10회) 주기로 가끔씩만 계산하고, 나머지 스텝에서는 이전 결과(cached)를 재사용합니다.
    # 
    # 문제점: 이전 결과를 JAX 배열 인자로 전달하면 JAX 미분기(value_and_grad)는 이를 상수로 취급하여 고유진동수에 대한 파라미터의 미분값(Gradient)을 0으로 만듭니다.
    # 해결책: 고유치 해석을 내부에서 수행하여 파라미터와 미분 사슬(Chain Rule)이 완벽히 연결된 함수(with_eigen)와, 
    #         정적 해석만 수행하여 가벼운 함수(no_eigen) 2개를 분리 컴파일(Two-Track)하고 상황에 맞게 번갈아 호출합니다.
    #         이렇게 하면 연산 속도를 챙기면서도 학습의 방향성(Gradient)을 잃지 않습니다.
    loss_vg_with_eigen = jax.jit(jax.value_and_grad(lambda p, fp: loss_fn(p, fp, True, None, None), has_aux=True))
    loss_vg_no_eigen = jax.jit(jax.value_and_grad(lambda p, fp, cf, cv: loss_fn(p, fp, False, cf, cv), has_aux=True))

    # 최적화 루프
    best_loss = float('inf')
    best_params = jax.tree_util.tree_map(jnp.array, params)
    self.history = [] # [CRITICAL FIX 15] 시각화 파이프라인 복원을 위한 히스토리 초기화
    wait = 0
    
    cached_freqs, cached_vecs = None, None
    cached_f1_hz = 0.0
    # [WHTOOLS] [NEW] Pre-initialize modal result variables for Iter 0 reporting
    last_matched_freqs, last_best_mac = None, None
    print(f"\n [PERFORMANCE] Eigen Decomposition occurs every {eigen_freq} iterations.")
    
    # [FINAL ROOT-CAUSE FIX] 'global' 키가 아니라 'global_targets'가 실제 키입니다.
    # -> 항상 빈 리스트 반환 -> has_mode_target = False -> Iter 1부터 고유치 해석 완전 차단
    # -> vecs_filtered = None 고정 -> MAC 연산 자체가 실행되지 않음 -> MAC=0 고정
    has_mode_target = (
        any(str(d.get('type','')).lower() in ['modes', 'modal', 'frequency', 'freq'] 
            for d in opt_target_config.get('global_targets', []))  # correct key
        or any(str(d.get('type','')).lower() in ['modes', 'modal', 'frequency', 'freq']
               for d in targets_jax if d.get('case_idx') is None)  # fallback from serialised
    )
    print(f" [MODAL] Mode target detected: {has_mode_target}")

    for i in range(max_iterations):
        iter_start = time.time()
        if i == 0:
            print(f"\n [Iter {i:04d}/{max_iterations}] Initializing and Compiling JAX graph... (This may take several minutes)")
        else:
            print(f" [Iter {i:04d}/{max_iterations}] Optimizing...", end="", flush=True)

        # 모드 매칭 타겟이 존재할 때만 주기적으로 고유치 해석 수행
        # [WHTOOLS] [PERFORMANCE] 최적화 속도를 위해 고유치 해석 주기를 eigen_freq에 맞춥니다.
        # i=0 (첫 실행)에는 결과를 보여주기 위해 타겟 여부와 상관없이 강제 수행합니다.
        compute_eigen_now = (i == 0) or ((i % eigen_freq == 0) and has_mode_target)

        
        if compute_eigen_now:
            (val, aux), grads = loss_vg_with_eigen(params, fixed_params_scaled)
            
            # [DIAGNOSTIC] Gradient Flow Monitoring
            # 왜 dt=0 인지, 본질적인 기울기 소멸 여부를 확인합니다.
            if i == 0:
                print("\n [DIAGNOSTIC] Gradient Norms at Iter 0:")
                for k, g in grads.items():
                    print(f"  ╰── {k}: {jnp.linalg.norm(g):.4e}")
            
            cached_freqs = aux['freqs']
            cached_vecs = aux['vecs_filtered']
            cached_f1_hz = aux['f1_hz']
            # [WHTOOLS] [NEW] Export MAC results for immediate reporting
            last_matched_freqs = aux.get('matched_freqs')
            last_best_mac = aux.get('best_mac')
        else:
            (val, aux), grads = loss_vg_no_eigen(params, fixed_params_scaled, cached_freqs, cached_vecs)
            # [CRITICAL FIX] do not overwrite with stale cached values from outer loop.
            # Rayleigh quotient-based results in aux should be preserved for accurate reporting.
            if 'matched_freqs' not in aux or aux['matched_freqs'] is None:
                aux['matched_freqs'] = last_matched_freqs if 'last_matched_freqs' in locals() else None
            if 'best_mac' not in aux or aux['best_mac'] is None:
                aux['best_mac'] = last_best_mac if 'last_best_mac' in locals() else None

        
        if val < best_loss - early_stop_tol:
            best_loss, wait, best_params = val, 0, jax.tree_util.tree_map(jnp.array, params)
        else:
            wait += 1

        # [CRITICAL FIX 15] V1 호환 리포팅 파이프라인 복원 (플롯 자동 생성용)
        hist_dict = {'Total': float(val), 'Reg': float(aux.get('Reg', 0.0))}
        evals_np = np.array(aux['target_evals'])
        for d_idx, info in enumerate(target_info):
            k_name = info['field'] if info['field'] else info['type']
            # 에러(terr) * 가중치(weight)를 히스토리에 누적
            hist_dict[k_name] = hist_dict.get(k_name, 0.0) + float(evals_np[d_idx, 2] * evals_np[d_idx, 3])
        self.history.append(hist_dict)

        updates, opt_state = optimizer.update(jax.tree_util.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads), opt_state)
        params = optax.apply_updates(params, updates)
        
        for k in params:
            if k in opt_config:
                params[k] = jnp.clip(params[k], opt_config[k].get('min', -1e12)/scaling_consts[k], opt_config[k].get('max', 1e12)/scaling_consts[k])

        # [WHTOOLS] [NEW] Track Design Parameter Sensitivity (Change since last step)
        if i > 0:
            dt_max = float(jnp.abs(params['t'] - prev_params['t']).max() * scaling_consts['t']) if 't' in params else 0.0
            dpz_max = float(jnp.abs(params['pz'] - prev_params['pz']).max() * scaling_consts['pz']) if 'pz' in params else 0.0
            dt_avg = float(jnp.abs(params['t'] - prev_params['t']).mean() * scaling_consts['t']) if 't' in params else 0.0
            dpz_avg = float(jnp.abs(params['pz'] - prev_params['pz']).mean() * scaling_consts['pz']) if 'pz' in params else 0.0
            sens_info = f" | dt:{dt_max:.6f}(avg:{dt_avg:.7f}) | dz:{dpz_max:.4f}(avg:{dpz_avg:.5f})"
        else:
            sens_info = ""
        prev_params = jax.tree_util.tree_map(jnp.array, params)
        
        # [WHTOOLS] [NEW] Track and Print Global Physical Values
        if i % 5 == 0:
            # [CRITICAL FIX] type: local 모드에서도 정상 작동하도록 float() 변환 전 jnp.mean() 취해 스칼라화
            t_phys = float(jnp.mean(params.get('t', fixed_params_scaled.get('t')))) * scaling_consts['t']
            rho_phys = float(jnp.mean(params.get('rho', fixed_params_scaled.get('rho')))) * scaling_consts['rho']
            E_phys = float(jnp.mean(params.get('E', fixed_params_scaled.get('E')))) * scaling_consts['E']
            print(f" ╰─── 🧪 Physical Stats (Mean): t={t_phys:.5f}mm | rho={rho_phys:.3e} | E={E_phys:.1f}MPa")

        iter_time = time.time() - iter_start
        if i != 0:
            print(f"\r 🔄 [Iter {i:04d}/{max_iterations}] {iter_time:4.2f}s | Loss: {float(val):.5e}{sens_info}")
        else:
            print(f" 🚀 [Iter {i:04d}/{max_iterations}] {iter_time:4.2f}s (C+E) | Loss: {float(val):.5e}")

        # 상세 출력 및 보고 - [WHTOOLS] 첫 이터레이션(Iter 0) 및 10회 주기마다 출력
        if i % 10 == 0 or i == 0:
            evals = np.array(aux['target_evals'])
            
            # --- OptTarget Evaluation Report Improvement ---
            # [WHTOOLS] ICE 엔진을 사용하여 헤더와 내용에 이모지가 자동 할당됩니다.
            table = WHTable(["Case", "Type", "Field", "Target Val", "Current Val", "Error", "Status"], 
                            title=f"OPTIMIZATION TARGET REPORT [ITER {i:04d}]")
            table.set_aligns(['left', 'left', 'left', 'right', 'right', 'right', 'center'])
            
            for d_idx, info in enumerate(target_info):
                v, r_v, err, w = evals[d_idx]
                
                # Status Emoji (자동 할당 지원 안 하므로 수동 처리 유지)
                abs_err = abs(float(err))
                if abs_err < 0.05: status = "✅"
                elif abs_err < 0.20: status = "🟡"
                else: status = "❌"
                
                # Case 및 Type 컬럼은 WHTable 내부의 ICE 엔진이 "Twist", "Stress", "Mass" 등의 키워드를 포착하여
                # 자동으로 이모지를 붙여줍니다.
                table.add_row([
                    info['case'],
                    info['type'],
                    info['field'],
                    f"{r_v:.4e}",
                    f"{v:.4e}",
                    f"{err:+.4f}",
                    status
                ], smart_cols=[0, 1, 2]) # 특정 컬럼에만 스마트 이모지 적용 가능
            table.print()
            
            # --- Mode Frequencies Output Improvement ---
            if cached_freqs is not None:
                m_table = WHTable(["Mode", "Target (Hz)", "Current (Hz)", "Error (%)", "MAC", "Quality"], 
                                  title="MODAL PERFORMANCE ANALYTICS")
                m_table.set_aligns(['center', 'right', 'right', 'right', 'right', 'center'])
                
                c_hz = np.array(aux['matched_freqs']) if aux.get('matched_freqs') is not None else np.array(cached_freqs)
                t_hz = np.array(t_vals)
                best_mac = np.array(aux['best_mac']) if aux.get('best_mac') is not None else None
                n_f = min(len(c_hz), len(t_hz))
                
                for m_idx in range(n_f):
                    err_hz = (c_hz[m_idx] - t_hz[m_idx]) / (t_hz[m_idx] + 1e-12) * 100.0
                    mac_val = float(best_mac[m_idx]) if best_mac is not None else 0.0
                    
                    # MAC Quality 등급 (ICE 엔진의 'quality' 키워드로 보조 가능)
                    if mac_val > 0.95: q_desc = "Excellent"
                    elif mac_val > 0.85: q_desc = "High"
                    elif mac_val > 0.50: q_desc = "Good"
                    else: q_desc = "Poor"
                    
                    m_table.add_row([
                        f"#{m_idx+1}",
                        f"{t_hz[m_idx]:.2f}",
                        f"{c_hz[m_idx]:.2f}",
                        f"{err_hz:+.2f}%",
                        f"{mac_val:.4f}",
                        f"{q_desc} Quality" # Quality 키워드 포함 시 ICE 엔진이 메달 이모지 등을 붙여줌
                    ])
                m_table.print()
            print(f"💰 [SYSTEM] Total Loss (with Reg): {float(val):.6e}")

        if msvcrt.kbhit() and msvcrt.getch() in [b'q', b'Q']:
            print("\n [USER INTERRUPT] Terminating early...")
            params = best_params; break
            
        if use_early_stopping and wait >= early_stop_patience:
            print(f"\n [EARLY STOP] No improvement for {wait} iters. Reverting to best.")
            params = best_params; break

    self.optimized_params = {**{k: v * scaling_consts[k] for k, v in params.items()}, **{k: v * scaling_consts[k] for k, v in fixed_params_scaled.items()}}
    print(f"\n -> Optimization Finished. Best Loss: {best_loss:.6f}")
    return self.optimized_params

# ---------------------------------------------------------
# Monkey-patching the new method to the existing model class
# ---------------------------------------------------------
EquivalentSheetModel.optimize_v2 = optimize_v2


if __name__ == '__main__':
    if os.environ.get('RUN_FULL_OPT', '0').lower() not in ('1', 'true', 'yes'):
        print("RUN_FULL_OPT environment variable must be set to '1' to execute the optimization pipeline.")
        print("CMD Example: set RUN_FULL_OPT=1 && python main_shell_opt.py")
        print("PowerShell Example: $env:RUN_FULL_OPT=\"1\"; python main_shell_opt.py")
        sys.exit(0)

    # XLA 성능 설정
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=6 --xla_backend_opt_level=1"

    model = EquivalentSheetModel(Lx, Ly, Nx_low, Ny_low)
    
    # 하중 케이스 추가
    model.add_case(TwistCase("twist_x", axis='x', value=1.5, mode='angle', weight=1.0))
    model.add_case(TwistCase("twist_y", axis='y', value=1.5, mode='angle', weight=1.0))
    model.add_case(PureBendingCase("bend_y", axis='y', value=3.0, mode='angle', weight=1.0))
    model.add_case(PureBendingCase("bend_x", axis='x', value=3.0, mode='angle', weight=1.0))
    model.add_case(CornerLiftCase("lift_br", corner='br', value=5.0, mode='disp', weight=1.0))
    model.add_case(CornerLiftCase("lift_tl", corner='tl', value=5.0, mode='disp', weight=1.0))
    model.add_case(TwoCornerLiftCase("lift_tl_br", corners=['br', 'tl'], value=5.0, mode='disp', weight=1.0))
    #model.add_case(CantileverCase("cantilever_x", axis='x', value=-5.0, mode='disp', weight=1.0))
    #model.add_case(CantileverCase("cantilever_y", axis='y', value=-5.0, mode='disp', weight=1.0))
    model.add_case(PressureCase("pressure_z", value=-10.0, weight=1.0))
    
    # 시나리오 테스트용 커스텀 OptTarget 설정
    opt_target_config = {
        # [비틀림 하중] 상대 비교 (Relative): 반력과 폰미세스 응력 타겟
        "twist_x": {
            "opt_targets": [
                {"target_type": "field_stat", "field": "u_static", "reduction": "mse", "compare_mode": "relative", "weight": 2.0},
                {"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 1.0},
                # [CRITICAL FIX 8] Stress Spike Cheating 방지: 응력도 max 대신 mse를 사용하여 꼼수 방지
                {"target_type": "field_stat", "field": "stress_vm", "reduction": "mse", "compare_mode": "relative", "weight": 0.5}
            ]
        },
        "twist_y": {
            "opt_targets": [
                {"target_type": "field_stat", "field": "u_static", "reduction": "mse", "compare_mode": "relative", "weight": 2.0},
                {"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 1.0},
                {"target_type": "field_stat", "field": "stress_vm", "reduction": "mse", "compare_mode": "relative", "weight": 0.5}
            ]
        },
        # [굽힘 하중] 각도(Angle) 제어 시 변위보다는 단면의 저항(Reaction Moment) 일치가 물리적으로 타당함
        "bend_y": {
            "opt_targets": [
                {"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 2.0},
                {"target_type": "field_stat", "field": "max_strain", "reduction": "mse", "compare_mode": "relative", "weight": 0.5}
            ]
        },
        "bend_x": {
            "opt_targets": [
                {"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 2.0},
                {"target_type": "field_stat", "field": "max_strain", "reduction": "mse", "compare_mode": "relative", "weight": 0.5}
            ]
        },
        # 변위 제어 케이스들 (리프트 및 캔틸레버): 목표 변위에 따른 반력(Reaction Force) 강성 일치 타겟
        "lift_br": {
            "opt_targets": [
                {"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 1.0}
            ]
        },
        "lift_tl": {
            "opt_targets": [
                {"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 1.0}
            ]
        },
        "lift_tl_br": {
            "opt_targets": [
                {"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 1.0}
            ]
        },        
        #"cantilever_x": {
        #    "opt_targets": [
        #        {"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 1.0}
        #    ]
        #},
        #"cantilever_y": {
        #    "opt_targets": [
        #        {"target_type": "rbe_reaction", "compare_mode": "relative", "weight": 1.0}
        #    ]
        #},
        # [압력 하중] 전체 변형 분포 제어를 위해 RMS(제곱평균제곱근) 변위 사용
        "pressure_z": {
            "opt_targets": [
                {"target_type": "field_stat", "field": "u_static", "reduction": "rms", "compare_mode": "relative", "weight": 1.0}
            ]
        },
        # [글로벌 제약 조건] 질량 및 고유 진동 모드
        # [BUG FIX 3] mass weight(5) >> modes weight(1) 이었으산 두께를 즌이면 주파수를 올리는 방향보다
        # 질량을 줄이는 방향이 5배 더 크게 보상 주어졌으므로 dt=0.8로 수렴 → 교정
        "global_targets": [
        #    {"target_type": "mass", "compare_mode": "relative", "tolerance": 0.05, "weight": 1.0},
            {"target_type": "modes", "compare_mode": "mac", "num_modes": 3, "freq_weight": 0.5, "weight": 5.0}
        ]
    }
    
    target_config = {
        'pattern': 'ABC', 'base_t': 1.0, 
        'bead_t': {'A': 2.0, 'B': 2.5, 'C': 1.5},
        'pattern_pz': 'TNYBV', 'bead_pz': {'T': 12.0, 'N': 10.0, 'Y': 15.0, 'B': 12.0, 'V': 4.0},
        'base_rho': 7.85e-9, 'base_E': 210000.0,
    }
    
    model.generate_targets(resolution_high=(Nx_high, Ny_high), num_modes_save=3, target_config=target_config)
        
    # 최적화 탐색 공간 설정
    opt_config = {
        't':   {'opt': True, 'init': 1.0, 'min': 0.8, 'max': 1.2, 'type': 'local'},        
        'rho': {'opt': True, 'init': 7.85e-9, 'min': 1e-10, 'max': 1e-7, 'type': 'local'},
        'E':   {'opt': False, 'init': 210000.0, 'type': 'local'},        
        'pz':  {'opt': True, 'init': 0.0, 'min': -20.0, 'max': 20.0, 'type': 'local'}, 
    }
    
    try:
        print("\n\n" + "#"*70)
        print(" [RUN] OptTarget V2 실행 (상세 리포트 포함)")
        print("#"*70)
        
        model.optimize_v2(opt_config, opt_target_config, 
                          use_bead_smoothing=True,      
                          max_iterations=150,
                          use_early_stopping=True, 
                          early_stop_patience=40, 
                          learning_rate=0.5,           
                          min_bead_width=150.0,        
                          init_pz_from_gt=True,        
                          gt_init_scale=0.0,
                          eigen_freq=10,
                          eigen_solver='lobpcg')               
                          
        model.verify()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[CRITICAL ERROR] Optimization failed: {e}")