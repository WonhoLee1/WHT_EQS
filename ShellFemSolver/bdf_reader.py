# -*- coding: utf-8 -*-
"""
================================================================================
ShellFemSolver / bdf_reader.py
================================================================================

■ 목적
    Nastran BDF / Altair OptiStruct (.fem) 파일을 파싱하여
    ShellFEM 클래스가 소비할 수 있는 데이터 구조로 변환한다.

■ 지원 카드 (Cards)
    GRID        — 노드 좌표 (x, y, z)
    CQUAD4      — 4절점 사각형 쉘 요소
    CTRIA3      — 3절점 삼각형 쉘 요소
    PSHELL      — 쉘 물성 (두께 t)
    MAT1        — 등방성 재료 (E, nu, rho)
    SPC / SPC1  — 단일점 구속 경계조건
    EIGRL       — 실수 고유값 해석 카드 (num_modes 추출용)

■ 지원 포맷
    - Nastran Small Field (8자 고정폭)
    - Nastran Large Field (16자 고정폭, GRID* 등)
    - Free Field (쉼표 구분)
    - OptiStruct .fem (위 모두 혼합 가능)

■ 비지원 카드 처리
    인식하지 못하는 카드는 조용히 무시 (경고만 출력).

■ 사용 예시
    from ShellFemSolver.bdf_reader import read_fem_file

    mesh = read_fem_file("model.fem")
    # mesh['nodes']    : (N, 3) ndarray  — XYZ 좌표
    # mesh['quads']    : (Q, 4) ndarray  — CQUAD4 노드 인덱스 (0-based)
    # mesh['trias']    : (T, 3) ndarray  — CTRIA3 노드 인덱스 (0-based)
    # mesh['t_node']   : (N,)   ndarray  — 노드별 두께 (PSHELL 참조)
    # mesh['E_node']   : (N,)   ndarray  — 노드별 탄성계수
    # mesh['nu_node']  : (N,)   ndarray  — 노드별 포아송비
    # mesh['rho_node'] : (N,)   ndarray  — 노드별 밀도
    # mesh['spc_dofs'] : list of (node_0based, dof_0based)  — 구속 자유도
    # mesh['nid_map']  : dict {nastran_id -> 0based_index}  — ID 매핑

================================================================================
"""

import numpy as np
import re
import os


def _parse_real(s):
    """Nastran 고정폭 부동소수점 파싱. 'D' 지수 표기도 처리."""
    s = s.strip()
    if not s:
        return 0.0
    # 1.5-3 (부호 없는 지수), 1.5+3, 1.5D-3 형식 처리
    s = s.replace('D', 'E').replace('d', 'e')
    # 내장 ± 기호가 지수 위치에 있을 경우 처리: "1.5-3" -> "1.5e-3"
    s = re.sub(r'([0-9])([+-])([0-9])', r'\1e\2\3', s)
    try:
        return float(s)
    except ValueError:
        return 0.0


def _split_small_field(line):
    """Nastran Small Field: 8자 단위로 분할 (최대 10 필드)."""
    # Remove comment and trailing whitespace
    line = line.rstrip()
    fields = []
    for i in range(0, min(len(line), 80), 8):
        fields.append(line[i:i+8].strip())
    while len(fields) < 10:
        fields.append('')
    return fields


def _split_large_field(line):
    """Nastran Large Field: 첫 필드 8자, 나머지 16자 단위."""
    line = line.rstrip()
    fields = [line[0:8].strip()]
    for i in range(8, min(len(line), 72), 16):
        fields.append(line[i:i+16].strip())
    while len(fields) < 6:
        fields.append('')
    return fields


def _split_free_field(line):
    """Free Field: 쉼표 구분. 빈 필드(,,) 보존."""
    line = line.rstrip()
    # 쉼표는 보존하면서 분리
    parts = line.split(',')
    return [p.strip() for p in parts]


def _detect_format(first_field):
    """카드 포맷 감지."""
    if first_field.endswith('*'):
        return 'large'
    elif ',' in first_field:
        return 'free'
    else:
        return 'small'


def read_fem_file(filepath):
    """
    Nastran BDF / OptiStruct .fem 파일을 파싱합니다.

    Parameters
    ----------
    filepath : str
        .bdf, .fem, .dat 파일 경로

    Returns
    -------
    dict with keys:
        nodes    : np.ndarray (N, 3)
        quads    : np.ndarray (Q, 4)  or empty (0, 4)
        trias    : np.ndarray (T, 3)  or empty (0, 3)
        t_node   : np.ndarray (N,)
        E_node   : np.ndarray (N,)
        nu_node  : np.ndarray (N,)
        rho_node : np.ndarray (N,)
        spc_dofs : list of (node_0based_idx, dof_0based_idx)
        nid_map  : dict {nastran_id -> 0based_index}
    """
    print(f"[BDF Reader] Reading: {os.path.basename(filepath)}")

    # ── 1단계: 파일 로드 및 연속행 처리 ────────────────────────────
    raw_lines = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            raw_lines = f.readlines()
    except Exception as e:
        raise IOError(f"파일을 열 수 없습니다: {filepath}\n{e}")

    # 연속행 처리: Large Field의 두 번째 행은 *로 시작
    lines = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        # 주석($ 또는 공백/탭만 있는 행) 제거
        stripped = line.lstrip()
        if stripped.startswith('$') or not stripped.strip():
            i += 1
            continue
        # BULK DATA 구분자는 무시
        if stripped.startswith('BEGIN BULK') or stripped.startswith('ENDDATA'):
            i += 1
            continue
        lines.append(line.rstrip('\n').rstrip('\r'))
        i += 1

    # ── 2단계: 카드별 파싱 ─────────────────────────────────────────
    # 임시 저장소
    nid_to_xyz  = {}   # {nastran_node_id: [x, y, z]}
    pid_to_prop = {}   # {prop_id: {'t': float, 'mid': int}}
    mid_to_mat  = {}   # {mat_id: {'E': float, 'nu': float, 'rho': float}}
    quad_raw    = []   # [(n1, n2, n3, n4, pid)]
    trias_raw   = []   # [(n1, n2, n3, pid)]
    spc_raw     = []   # [(nid, dof_str)]
    rbe2_raw    = []   # [(master, [slaves], dof_str)]
    conm2_raw   = []   # [(nid, mass, I11, I22, I33)]
    beam_raw    = []   # [(n1, n2, pid)]
    pbar_props  = {}   # {pid: {'mid': int, 'A': float, 'I1': float, 'I2': float, 'J': float}}
    num_modes   = 10   # EIGRL에서 추출

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue

        # 포맷 감지
        first8 = line[:8].strip().upper()
        fmt = _detect_format(line[:8])

        if fmt == 'free':
            flds = _split_free_field(line)
            card = flds[0].upper().rstrip('*')
        elif fmt == 'large':
            flds = _split_large_field(line)
            card = flds[0].upper().rstrip('*')
        else:
            flds = _split_small_field(line)
            card = flds[0].upper()

        # ── GRID ──────────────────────────────────────────────────
        if card == 'GRID':
            if fmt == 'large':
                # Large Field GRID*: 다음 행과 결합
                flds2 = _split_large_field(lines[idx+1]) if idx+1 < len(lines) else [''] * 6
                nid = int(flds[1]) if flds[1] else 0
                x   = _parse_real(flds[3])
                y   = _parse_real(flds[4])
                z   = _parse_real(flds2[1]) if len(flds2) > 1 else 0.0
                idx += 2
            else:
                nid = int(flds[1]) if flds[1] else 0
                x   = _parse_real(flds[3]) if len(flds) > 3 else 0.0
                y   = _parse_real(flds[4]) if len(flds) > 4 else 0.0
                z   = _parse_real(flds[5]) if len(flds) > 5 else 0.0
                idx += 1
            nid_to_xyz[nid] = [x, y, z]

        # ── CQUAD4 ────────────────────────────────────────────────
        elif card == 'CQUAD4':
            # flds: [CQUAD4, EID, PID, G1, G2, G3, G4, ...]
            try:
                pid = int(flds[2])
                n1, n2, n3, n4 = int(flds[3]), int(flds[4]), int(flds[5]), int(flds[6])
                quad_raw.append((n1, n2, n3, n4, pid))
            except (ValueError, IndexError):
                pass
            idx += 1

        # ── CTRIA3 ────────────────────────────────────────────────
        elif card == 'CTRIA3':
            try:
                pid = int(flds[2])
                n1, n2, n3 = int(flds[3]), int(flds[4]), int(flds[5])
                trias_raw.append((n1, n2, n3, pid))
            except (ValueError, IndexError):
                pass
            idx += 1

        # ── RBE2 ──────────────────────────────────────────────────
        elif card == 'RBE2':
            # RBE2, EID, GN, CM, GM1, GM2, ...
            try:
                master = int(flds[2])
                dof_str = str(flds[3])
                slaves = []
                j = 4
                while j < len(flds) and flds[j]:
                    if flds[j].upper() == 'THRU':
                        st = slaves.pop()
                        en = int(flds[j+1])
                        slaves.extend(range(st, en+1))
                        j += 2
                    else:
                        slaves.append(int(flds[j]))
                        j += 1
                rbe2_raw.append((master, slaves, dof_str))
            except (ValueError, IndexError):
                pass
            idx += 1

        # ── CONM2 ─────────────────────────────────────────────────
        elif card == 'CONM2':
            # CONM2, EID, GRID, CID, M, X1, X2, X3 | I11, I21, I22, I31, I32, I33
            try:
                nid  = int(flds[2])
                mass = _parse_real(flds[4])
                # Principal Inertias (I11, I22, I33 from next row/fields)
                i11 = _parse_real(flds[8]) if len(flds) > 8 else 0.0
                # If there's a next line with + continuation
                i22, i33 = 0.0, 0.0
                if idx + 1 < len(lines) and lines[idx+1].strip().startswith('+'):
                    idx += 1
                    flds2 = _split_small_field(lines[idx])
                    i22 = _parse_real(flds2[2])
                    i33 = _parse_real(flds2[5])
                conm2_raw.append((nid, mass, i11, i22, i33))
            except (ValueError, IndexError):
                pass
            idx += 1

        # ── CBAR ──────────────────────────────────────────────────
        elif card == 'CBAR':
            try:
                pid = int(flds[2])
                n1, n2 = int(flds[3]), int(flds[4])
                beam_raw.append((n1, n2, pid))
            except (ValueError, IndexError):
                pass
            idx += 1

        # ── PSHELL ────────────────────────────────────────────────
        elif card == 'PSHELL':
            try:
                pid = int(flds[1])
                mid = int(flds[2]) if flds[2] else 1
                t   = _parse_real(flds[3])
                pid_to_prop[pid] = {'t': t, 'mid': mid}
            except (ValueError, IndexError):
                pass
            idx += 1

        # ── PBAR ──────────────────────────────────────────────────
        elif card == 'PBAR':
            try:
                pid = int(flds[1])
                mid = int(flds[2])
                A   = _parse_real(flds[3])
                i1  = _parse_real(flds[4])
                i2  = _parse_real(flds[5])
                j_v = _parse_real(flds[6])
                pbar_props[pid] = {'mid': mid, 'A': A, 'I1': i1, 'I2': i2, 'J': j_v}
            except (ValueError, IndexError):
                pass
            idx += 1

        # ── MAT1 ──────────────────────────────────────────────────
        elif card == 'MAT1':
            try:
                mid = int(flds[1])
                E   = _parse_real(flds[2])
                nu  = _parse_real(flds[4]) if len(flds) > 4 else 0.3
                rho = _parse_real(flds[5]) if len(flds) > 5 else 7.85e-9
                mid_to_mat[mid] = {'E': E, 'nu': nu, 'rho': rho}
            except (ValueError, IndexError):
                pass
            idx += 1

        # ── SPC (단일점 구속, 직접 지정) ─────────────────────────
        elif card == 'SPC':
            # SPC, SID, G1, C1, D1, G2, C2, D2
            try:
                nid1 = int(flds[2])
                dof1 = str(flds[3])
                spc_raw.append((nid1, dof1))
                if flds[5]:
                    nid2 = int(flds[5])
                    dof2 = str(flds[6])
                    spc_raw.append((nid2, dof2))
            except (ValueError, IndexError):
                pass
            idx += 1

        # ── SPC1 (범위 지정 구속) ─────────────────────────────────
        elif card == 'SPC1':
            # SPC1, SID, C, G1, G2, ..., or THRU
            try:
                dof_str = str(flds[2])
                node_ids = []
                j = 3
                while j < len(flds) and flds[j]:
                    if flds[j].upper() == 'THRU':
                        start_n = int(flds[j-1])
                        end_n   = int(flds[j+1])
                        node_ids.extend(range(start_n, end_n+1))
                        j += 2
                    else:
                        try:
                            node_ids.append(int(flds[j]))
                        except ValueError:
                            pass
                        j += 1
                for nid in node_ids:
                    spc_raw.append((nid, dof_str))
            except (ValueError, IndexError):
                pass
            idx += 1

        # ── EIGRL ─────────────────────────────────────────────────
        elif card == 'EIGRL':
            try:
                nd = int(flds[7]) if len(flds) > 7 and flds[7] else 10
                num_modes = nd
            except (ValueError, IndexError):
                pass
            idx += 1

        else:
            idx += 1

    # ── 3단계: 0-based 인덱스로 변환 ──────────────────────────────
    sorted_nids = sorted(nid_to_xyz.keys())
    nid_map     = {nid: i for i, nid in enumerate(sorted_nids)}
    N           = len(sorted_nids)

    nodes = np.array([nid_to_xyz[nid] for nid in sorted_nids], dtype=np.float64)

    # 요소 변환
    def _nid(n):
        return nid_map.get(n, 0)

    quads = np.array([[_nid(r[0]), _nid(r[1]), _nid(r[2]), _nid(r[3])]
                       for r in quad_raw], dtype=np.int32) if quad_raw else np.zeros((0,4), dtype=np.int32)

    trias = np.array([[_nid(r[0]), _nid(r[1]), _nid(r[2])]
                       for r in trias_raw], dtype=np.int32) if trias_raw else np.zeros((0,3), dtype=np.int32)

    rbe2 = []
    for master, slaves, dof_str in rbe2_raw:
        m_idx = _nid(master)
        s_idxs = [_nid(s) for s in slaves]
        rbe2.append((m_idx, s_idxs, dof_str))

    # CONM2 인덱싱
    conm2 = []
    for nid, m, i11, i22, i33 in conm2_raw:
        n_idx = _nid(nid)
        conm2.append((n_idx, m, i11, i22, i33))

    # BEAM (CBAR) 인덱싱 및 속성 추출
    beams = np.array([[_nid(r[0]), _nid(r[1])] for r in beam_raw], dtype=np.int32) if beam_raw else np.zeros((0,2), dtype=np.int32)
    b_A, b_I1, b_I2, b_J = [], [], [], []
    for n1, n2, pid in beam_raw:
        prop = pbar_props.get(pid, {'A': 1.0, 'I1': 1.0, 'I2': 1.0, 'J': 1.0})
        b_A.append(prop['A']); b_I1.append(prop['I1']); b_I2.append(prop['I2']); b_J.append(prop['J'])
    b_A  = np.array(b_A)
    b_I1 = np.array(b_I1)
    b_I2 = np.array(b_I2)
    b_J  = np.array(b_J)

    # ── 4단계: 노드별 재료/두께 분배 ──────────────────────────────
    # 기본값 (MAT1에서 첫 번째 재료를 전역 기본값으로 사용)
    default_E, default_nu, default_rho, default_t = 210000.0, 0.3, 7.85e-9, 1.0
    if mid_to_mat:
        first_mid = min(mid_to_mat.keys())
        m = mid_to_mat[first_mid]
        default_E   = m['E']   if m['E']   > 0 else default_E
        default_nu  = m['nu']  if m['nu']  > 0 else default_nu
        default_rho = m['rho'] if m['rho'] > 0 else default_rho

    if pid_to_prop:
        first_pid = min(pid_to_prop.keys())
        default_t = pid_to_prop[first_pid]['t'] if pid_to_prop[first_pid]['t'] > 0 else default_t

    # 요소별 두께를 노드로 전파 (단순 평균)
    t_node   = np.full(N, default_t)
    E_node   = np.full(N, default_E)
    nu_node  = np.full(N, default_nu)
    rho_node = np.full(N, default_rho)

    def _set_elem_props(node_indices, pid):
        prop = pid_to_prop.get(pid, None)
        if prop is None:
            return
        t_val = prop['t'] if prop['t'] > 0 else default_t
        mid   = prop['mid']
        mat   = mid_to_mat.get(mid, None)
        E_val   = mat['E']   if mat and mat['E']   > 0 else default_E
        nu_val  = mat['nu']  if mat and mat['nu']  > 0 else default_nu
        rho_val = mat['rho'] if mat and mat['rho'] > 0 else default_rho
        for ni in node_indices:
            t_node[ni]   = t_val
            E_node[ni]   = E_val
            nu_node[ni]  = nu_val
            rho_node[ni] = rho_val

    for r in quad_raw:
        idxs = [_nid(r[0]), _nid(r[1]), _nid(r[2]), _nid(r[3])]
        _set_elem_props(idxs, r[4])

    for r in trias_raw:
        idxs = [_nid(r[0]), _nid(r[1]), _nid(r[2])]
        _set_elem_props(idxs, r[3])

    # ── 5단계: SPC → DOF 인덱스 변환 ─────────────────────────────
    # DOF 코드: 1=Tx, 2=Ty, 3=Tz, 4=Rx, 5=Ry, 6=Rz
    # 0-based: DOF코드-1
    spc_dofs = []
    for nid, dof_str in spc_raw:
        ni = nid_map.get(nid, None)
        if ni is None:
            continue
        for ch in str(dof_str):
            if ch.isdigit() and 1 <= int(ch) <= 6:
                spc_dofs.append((ni, int(ch) - 1))

    # ── 6단계: 결과 출력 ───────────────────────────────────────────
    nq = len(quads)
    nt = len(trias)
    print(f"[BDF Reader] Nodes: {N}, CQUAD4: {nq}, CTRIA3: {nt}")
    print(f"[BDF Reader] Materials: {len(mid_to_mat)}, Properties: {len(pid_to_prop)}")
    print(f"[BDF Reader] SPCs: {len(spc_dofs)} constrained DOFs")
    if mid_to_mat:
        print(f"[BDF Reader] E={default_E:.0f} MPa, nu={default_nu:.3f}, rho={default_rho:.3e}")
    print(f"[BDF Reader] Default thickness: {default_t:.3f} mm")

    return {
        'nodes'    : nodes,
        'quads'    : quads,
        'trias'    : trias,
        't_node'   : t_node,
        'E_node'   : E_node,
        'nu_node'  : nu_node,
        'rho_node' : rho_node,
        'spc_dofs' : spc_dofs,
        'rbe2'     : rbe2,
        'conm2'    : conm2,
        'beams'    : beams,
        'b_A'      : b_A,
        'b_I1'     : b_I1,
        'b_I2'     : b_I2,
        'b_J'      : b_J,
        'nid_map'  : nid_map,
        'num_modes': num_modes,
    }


def build_fem_from_bdf(filepath, E=None, nu=None, rho=None, t=None):
    """
    BDF/FEM 파일을 읽어 ShellFEM 객체와 params를 반환합니다.
    E, nu, rho, t 인자가 제공되면 BDF 파일의 정보를 무시하고 해당 값으로 오버라이드합니다.
    """
    import jax.numpy as jnp
    from ShellFemSolver.shell_solver import ShellFEM

    mesh = read_fem_file(filepath)

    quads = mesh['quads'] if len(mesh['quads']) > 0 else None
    trias = mesh['trias'] if len(mesh['trias']) > 0 else None
    beams = mesh['beams'] if len(mesh['beams']) > 0 else None

    fem = ShellFEM(mesh['nodes'], quads=quads, trias=trias, beams=beams)
    fem.rbe2  = mesh['rbe2']   # Store for assembly
    fem.conm2 = mesh['conm2']  # Store for assembly

    # 파라미터 구성 (오버라이드 우선순위 적용)
    n_nodes = fem.num_nodes
    p_E   = jnp.full(n_nodes, E)   if E   is not None else jnp.array(mesh['E_node'])
    p_nu  = jnp.full(n_nodes, nu)  if nu  is not None else jnp.array(mesh['nu_node'])
    p_rho = jnp.full(n_nodes, rho) if rho is not None else jnp.array(mesh['rho_node'])
    p_t   = jnp.full(n_nodes, t)   if t   is not None else jnp.array(mesh['t_node'])

    params = {
        't'  : p_t,
        'E'  : p_E,
        'rho': p_rho,
        'nu' : p_nu,
        'b_A' : jnp.array(mesh['b_A']) if len(mesh['b_A']) > 0 else None,
        'b_I1': jnp.array(mesh['b_I1']) if len(mesh['b_I1']) > 0 else None,
        'b_I2': jnp.array(mesh['b_I2']) if len(mesh['b_I2']) > 0 else None,
        'b_J' : jnp.array(mesh['b_J']) if len(mesh['b_J']) > 0 else None,
    }

    return fem, params, mesh['spc_dofs'], mesh['num_modes']


def run_eigen_from_bdf(filepath, num_modes=None, E=None, nu=None, rho=None, t=None):
    """
    BDF/OptiStruct FEM 파일에서 고유진동수를 계산합니다.
    E, nu, rho, t 인자가 제공되면 파일 정보를 무시하고 해당 값을 사용합니다.
    """
    import jax.numpy as jnp
    import numpy as np

    fem, params, spc_dofs, n_modes_file = build_fem_from_bdf(
        filepath, E=E, nu=nu, rho=rho, t=t
    )

    if num_modes is None:
        num_modes = n_modes_file

    # Use Sparse Assembly
    K_s, M_s = fem.assemble(params, sparse=True)

    # CONM2 (Concentrated Mass) Assembly
    if hasattr(fem, 'conm2') and fem.conm2:
        from scipy.sparse import coo_matrix
        m_rows, m_cols, m_vals = [], [], []
        for n_idx, m, i11, i22, i33 in fem.conm2:
            base = n_idx * 6
            # Translation mass (X, Y, Z)
            for d in range(3):
                m_rows.append(base + d); m_cols.append(base + d); m_vals.append(m)
            # Rotational inertia (Rx, Ry, Rz)
            if i11 > 0: m_rows.append(base + 3); m_cols.append(base + 3); m_vals.append(i11)
            if i22 > 0: m_rows.append(base + 4); m_cols.append(base + 4); m_vals.append(i22)
            if i33 > 0: m_rows.append(base + 5); m_cols.append(base + 5); m_vals.append(i33)
        M_conm = coo_matrix((m_vals, (m_rows, m_cols)), shape=M_s.shape).tocsr()
        M_s = M_s + M_conm

    # RBE2 Penalty Stiffness
    if hasattr(fem, 'rbe2') and fem.rbe2:
        from scipy.sparse import coo_matrix
        rows, cols, vals = [], [], []
        penalty = 1.0e10  # Very large stiffness
        for master, slaves, dof_str in fem.rbe2:
            dofs = [int(c)-1 for c in dof_str if c.isdigit()]
            for val in slaves:
                for d in dofs:
                    m_dof = master * 6 + d
                    s_dof = val * 6 + d
                    # Penalty: (u_m - u_s)^2 * k/2 -> k [1 -1; -1 1]
                    rows.extend([m_dof, m_dof, s_dof, s_dof])
                    cols.extend([m_dof, s_dof, m_dof, s_dof])
                    vals.extend([penalty, -penalty, -penalty, penalty])
        K_penalty = coo_matrix((vals, (rows, cols)), shape=K_s.shape).tocsr()
        K_s = K_s + K_penalty

    # 경계조건 적용 (Sparse 조작)
    fixed_global = sorted(set([ni * 6 + di for ni, di in spc_dofs]))
    all_dofs = np.arange(fem.total_dof)
    mask = np.ones(fem.total_dof, dtype=bool)
    if fixed_global:
        mask[fixed_global] = False
    free = np.where(mask)[0]

    # Sparse matrix partitioning
    # Filter out DOFs that have zero diagonal in both K and M
    diag_k = np.array(K_s.diagonal())
    diag_m = np.array(M_s.diagonal())
    valid_dofs = np.where((diag_k > 1e-12) | (diag_m > 1e-12))[0]
    
    # Active DOFs are both free (not SPC) and valid (connected)
    free_set = set(free)
    valid_set = set(valid_dofs)
    active_dofs = sorted(list(free_set.intersection(valid_set)))
    active_dofs = np.array(active_dofs)

    K_ff = K_s[active_dofs, :][:, active_dofs]
    M_ff = M_s[active_dofs, :][:, active_dofs]

    print(f"[Eigen] Solving for {num_modes} modes using Sparse Solver...")
    vals, vecs = fem.solve_eigen_sparse(K_ff, M_ff, num_modes=num_modes + 6)
    freqs = np.sqrt(np.maximum(np.array(vals), 0.0)) / (2 * 3.14159265358979)

    # 강체 모드 제거 (1Hz 이하)
    valid = freqs > 1.0
    freqs_valid = freqs[valid][:num_modes]

    print(f"\n[Eigen] 고유진동수 상위 {len(freqs_valid)} 모드:")
    for i, f in enumerate(freqs_valid[:num_modes]):
        print(f"  Mode {i+1:2d}: {float(f):10.4f} Hz")

    return freqs_valid, fem
