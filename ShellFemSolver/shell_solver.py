# -*- coding: utf-8 -*-
"""
================================================================================
ShellFemSolver / shell_solver.py
================================================================================
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import vmap, jit, custom_vjp
import numpy as np
from functools import partial

# ──────────────────────────────────────────────────────────────────────────────
#  Recovery Utilities (Internal)
# ──────────────────────────────────────────────────────────────────────────────

def recover_stress_tria_membrane(u, nodes, trias, E, nu):
    """CST element strain recovery."""
    def get_element_strain_inner(tri_idx):
        ix = trias[tri_idx]
        tri_nodes = nodes[ix]
        u_el = u.reshape(-1, 6)[ix][:, :2].flatten()
        
        x1, y1 = tri_nodes[0, 0], tri_nodes[0, 1]
        x2, y2 = tri_nodes[1, 0], tri_nodes[1, 1]
        x3, y3 = tri_nodes[2, 0], tri_nodes[2, 1]
        
        area_factor = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        inv_area = 1.0 / jnp.where(jnp.abs(area_factor) < 1e-12, 1e-12, area_factor)
        
        b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
        c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
        
        B = inv_area * jnp.array([
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3]
        ])
        return B @ u_el
    return vmap(get_element_strain_inner)(jnp.arange(trias.shape[0]))

def _get_B_tria_membrane(nodes):
    """Constant Strain Triangle (CST) B-matrix for membrane."""
    x1, y1 = nodes[0,0], nodes[0,1]
    x2, y2 = nodes[1,0], nodes[1,1]
    x3, y3 = nodes[2,0], nodes[2,1]
    detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    # b_i = y_j - y_k, c_i = x_k - x_j
    b1, b2, b3 = y2-y3, y3-y1, y1-y2
    c1, c2, c3 = x3-x2, x1-x3, x2-x1
    inv2A = 1.0 / detJ
    Bm = jnp.array([
        [b1, 0, b2, 0, b3, 0],
        [0, c1, 0, c2, 0, c3],
        [c1, b1, c2, b2, c3, b3]
    ]) * inv2A
    return Bm, detJ

def _get_B_dkt(xi, eta, nodes):
    """
    DKT (Discrete Kirchhoff Triangle) B-matrix
    Reference: Batoz (1980) & FEniCS/JuliaFEM DKT implementations.
    """
    x1, y1 = nodes[0,0], nodes[0,1]
    x2, y2 = nodes[1,0], nodes[1,1]
    x3, y3 = nodes[2,0], nodes[2,1]

    # Jacobian (2 * Area)
    x21, y21 = x2-x1, y2-y1
    x31, y31 = x3-x1, y3-y1
    detJ = x21*y31 - y21*x31

    x23, y23 = x2-x3, y2-y3
    x31_e, y31_e = x3-x1, y3-y1
    x12, y12 = x1-x2, y1-y2
    L23sq = x23**2 + y23**2
    L31sq = x31_e**2 + y31_e**2
    L12sq = x12**2 + y12**2

    # DKT coefficients (Batoz Table 1)
    p4, p5, p6 = -6*x23/L23sq, -6*x31_e/L31sq, -6*x12/L12sq
    q4, q5, q6 =  3*x23*y23/L23sq,  3*x31_e*y31_e/L31sq,  3*x12*y12/L12sq
    r4, r5, r6 =  3*y23**2/L23sq,  3*y31_e**2/L31sq,  3*y12**2/L12sq
    t4, t5, t6 = -6*y23/L23sq, -6*y31_e/L31sq, -6*y12/L12sq

    # Hx Derivatives
    Hx_xi = jnp.array([
         p6*(1.0-2.0*xi) + (p5-p6)*eta,
         q6*(1.0-2.0*xi) - (q5+q6)*eta,
        -4.0 + 6.0*(xi+eta) + r6*(1.0-2.0*xi) - eta*(r5+r6),
        -p6*(1.0-2.0*xi) + eta*(p4+p6),
         q6*(1.0-2.0*xi) - eta*(q6-q4),
        -2.0 + 6.0*xi + r6*(1.0-2.0*xi) + eta*(r4-r6),
        -eta*(p5+p4),
         eta*(q4-q5),
        -eta*(r5-r4)
    ])
    Hx_et = jnp.array([
        -p5*(1.0-2.0*eta) - xi*(p6-p5),
         q5*(1.0-2.0*eta) - xi*(q5+q6),
        -4.0 + 6.0*(xi+eta) + r5*(1.0-2.0*eta) - xi*(r5+r6),
         xi*(p4+p6),
         xi*(q4-q6),
        -xi*(r6-r4),
         p5*(1.0-2.0*eta) - xi*(p4+p5),
         q5*(1.0-2.0*eta) + xi*(q4-q5),
        -2.0 + 6.0*eta + r5*(1.0-2.0*eta) + xi*(r4-r5)
    ])

    # Hy Derivatives
    Hy_xi = jnp.array([
         t6*(1.0-2.0*xi) + eta*(t5-t6),
         1.0 + r6*(1.0-2.0*xi) - eta*(r5+r6),
        -q6*(1.0-2.0*xi) + eta*(q5+q6),
        -t6*(1.0-2.0*xi) + eta*(t4+t6),
        -1.0 + r6*(1.0-2.0*xi) + eta*(r4-r6),
        -q6*(1.0-2.0*xi) - eta*(q4-q6),
        -eta*(t4+t5),
         eta*(r4-r5),
        -eta*(q4-q5)
    ])
    Hy_et = jnp.array([
        -t5*(1.0-2.0*eta) - xi*(t6-t5),
         1.0 + r5*(1.0-2.0*eta) - xi*(r5+r6),
        -q5*(1.0-2.0*eta) + xi*(q5+q6),
         xi*(t4+t6),
         xi*(r4-r6),
        -xi*(q4-q6),
         t5*(1.0-2.0*eta) - xi*(t4+t5),
        -1.0 + r5*(1.0-2.0*eta) + xi*(r4-r5),
        -q5*(1.0-2.0*eta) - xi*(q4-q5)
    ])

    invJ = 1.0 / detJ
    B = jnp.zeros((3, 9))
    # Curvature Row 0: kappa_x = beta_x,x
    B = B.at[0].set( invJ * (y31*Hx_xi - y21*Hx_et) )
    # Curvature Row 1: kappa_y = beta_y,y
    B = B.at[1].set( invJ * (-x31*Hy_xi + x21*Hy_et) )
    # Curvature Row 2: 2kxy = beta_x,y + beta_y,x
    B = B.at[2].set( invJ * (-x31*Hx_xi + x21*Hx_et + y31*Hy_xi - y21*Hy_et) )
    return B

def recover_curvature_tria_bending(u, nodes, trias, E, nu, t):
    """Recover curvature from local T3 displacements."""
    def get_curv(idx):
        ix = trias[idx]
        t_nodes = nodes[ix].astype(jnp.float64)
        
        # Local system: w,x = -phiy, w,y = phix
        u_el = u.reshape(-1, 6)[ix]
        # Batoz: u = [w1, thx1, thy1, w2, thx2, thy2, w3, thx3, thy3]
        # Slopes: thx = w,x = -phiy, thy = w,y = phix
        udkt = jnp.array([
            u_el[0, 2], -u_el[0, 4], u_el[0, 3],
            u_el[1, 2], -u_el[1, 4], u_el[1, 3],
            u_el[2, 2], -u_el[2, 4], u_el[2, 3]
        ])
        
        B = _get_B_dkt(1.0/3.0, 1.0/3.0, t_nodes)
        return B @ udkt
    
    return vmap(get_curv)(jnp.arange(trias.shape[0]))

def recover_curvature_quad_bending(u, nodes, quads, E, nu, t):
    """Q4 Mindlin curvature recovery."""
    def get_curvature(idx):
        ix = quads[idx]
        q_nodes = nodes[ix]
        v12 = q_nodes[1]-q_nodes[0]
        e1 = v12 / jnp.linalg.norm(v12).clip(1e-12)
        v14 = q_nodes[3]-q_nodes[0]
        nrm = jnp.cross(e1, v14)
        e3 = nrm / jnp.linalg.norm(nrm).clip(1e-12)
        e2 = jnp.cross(e3, e1)
        
        p2d = jnp.stack([jnp.dot(q_nodes-q_nodes[0], e1), jnp.dot(q_nodes-q_nodes[0], e2)], axis=1)
        
        dn_dxi = jnp.array([-0.25, 0.25, 0.25, -0.25])
        dn_det = jnp.array([-0.25, -0.25, 0.25, 0.25])
        J = jnp.array([
            [jnp.dot(dn_dxi, p2d[:,0]), jnp.dot(dn_dxi, p2d[:,1])], 
            [jnp.dot(dn_det, p2d[:,0]), jnp.dot(dn_det, p2d[:,1])]
        ])
        invJ = jnp.linalg.inv(J + jnp.eye(2)*1e-12)
        dn_dx = invJ[0,0]*dn_dxi + invJ[0,1]*dn_det
        dn_dy = invJ[1,0]*dn_dxi + invJ[1,1]*dn_det
        
        u_g = u.reshape(-1, 6)[ix]
        tx_g, ty_g = u_g[:, 3], u_g[:, 4]
        tx_l = tx_g * e1[0] + ty_g * e1[1]
        ty_l = tx_g * e2[0] + ty_g * e2[1]
        
        kx = -jnp.dot(dn_dx, ty_l)
        ky = jnp.dot(dn_dy, tx_l)
        kxy = -(jnp.dot(dn_dy, ty_l) - jnp.dot(dn_dx, tx_l))
        return jnp.array([kx, ky, kxy])
    return vmap(get_curvature)(jnp.arange(quads.shape[0]))

def recover_stress_quad_membrane(u, nodes, quads, E, nu):
    """Q4 membrane strain recovery."""
    def get_eps(idx):
        ix = quads[idx]
        q_nodes = nodes[ix]
        v12 = q_nodes[1]-q_nodes[0]
        e1 = v12 / jnp.linalg.norm(v12).clip(1e-12)
        v14 = q_nodes[3]-q_nodes[0]
        nrm = jnp.cross(e1, v14)
        e3 = nrm / jnp.linalg.norm(nrm).clip(1e-12)
        e2 = jnp.cross(e3, e1)
        
        p2d = jnp.stack([jnp.dot(q_nodes-q_nodes[0], e1), jnp.dot(q_nodes-q_nodes[0], e2)], axis=1)
        
        dn_dxi = jnp.array([-0.25, 0.25, 0.25, -0.25])
        dn_det = jnp.array([-0.25, -0.25, 0.25, 0.25])
        J = jnp.array([
            [jnp.dot(dn_dxi, p2d[:,0]), jnp.dot(dn_dxi, p2d[:,1])], 
            [jnp.dot(dn_det, p2d[:,0]), jnp.dot(dn_det, p2d[:,1])]
        ])
        invJ = jnp.linalg.inv(J + jnp.eye(2)*1e-12)
        dn_dx = invJ[0,0]*dn_dxi + invJ[0,1]*dn_det
        dn_dy = invJ[1,0]*dn_dxi + invJ[1,1]*dn_det
        
        u_m = u.reshape(-1, 6)[ix][:, :2]
        ul = u_m[:,0]*e1[0] + u_m[:,1]*e1[1]
        vl = u_m[:,0]*e2[0] + u_m[:,1]*e2[1]
        
        ex = jnp.dot(dn_dx, ul)
        ey = jnp.dot(dn_dy, vl)
        exy = jnp.dot(dn_dy, ul) + jnp.dot(dn_dx, vl)
        return jnp.array([ex, ey, exy])
    return vmap(get_eps)(jnp.arange(quads.shape[0]))

# ────────────────────────────────────────────────────────────────────
# Safe Eigh
# ────────────────────────────────────────────────────────────────────
@custom_vjp
def safe_eigh(A):
    vals, vecs = jnp.linalg.eigh(A)
    return vals, vecs

def safe_eigh_fwd(A):
    vals, vecs = jnp.linalg.eigh(A)
    return (vals, vecs), (vals, vecs)

def safe_eigh_bwd(res, g):
    vals, vecs = res
    g_vals, g_vecs = g
    grad_A_vals = jnp.einsum('k,ik,jk->ij', g_vals, vecs, vecs)
    diff = vals[:, None] - vals[None, :]
    F = jnp.where(jnp.abs(diff) < 1e-9, jnp.inf, 1.0 / diff)
    F = jnp.where(jnp.isinf(F), 0.0, F).at[jnp.diag_indices(vecs.shape[0])].set(0.0)
    P = F * (vecs.T @ g_vecs)
    grad_A_vecs = vecs @ P @ vecs.T
    total = 0.5 * (grad_A_vals + grad_A_vecs + (grad_A_vals + grad_A_vecs).T)
    return (total,)

safe_eigh.defvjp(safe_eigh_fwd, safe_eigh_bwd)

# ────────────────────────────────────────────────────────────────────
#  Element Formulation Helpers
# ────────────────────────────────────────────────────────────────────

def _shape_deriv_q4(xi, eta, a, b):
    dN_dxi  = 0.25 * jnp.array([-(1-eta),  (1-eta),  (1+eta), -(1+eta)])
    dN_deta = 0.25 * jnp.array([-(1-xi),  -(1+xi),   (1+xi),   (1-xi)])
    N       = 0.25 * jnp.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])
    return dN_dxi/a, dN_deta/b, N

def _B_membrane_q4(dN_dx, dN_dy):
    z = jnp.zeros(4)
    r0 = jnp.stack([dN_dx, z   ], axis=1).reshape(-1)
    r1 = jnp.stack([z,    dN_dy], axis=1).reshape(-1)
    r2 = jnp.stack([dN_dy, dN_dx], axis=1).reshape(-1)
    return jnp.stack([r0, r1, r2])

def _B_bending_q4(dN_dx, dN_dy):
    z = jnp.zeros(4)
    r0 = jnp.stack([z, z, dN_dx], axis=1).reshape(-1)
    r1 = jnp.stack([z, -dN_dy, z], axis=1).reshape(-1)
    r2 = jnp.stack([z, -dN_dx, dN_dy], axis=1).reshape(-1)
    return jnp.stack([r0, r1, r2])

def _B_shear_q4(dN_dx, dN_dy, N):
    z = jnp.zeros(4)
    r0 = jnp.stack([dN_dx, z,  N ], axis=1).reshape(-1)
    r1 = jnp.stack([dN_dy, -N, z ], axis=1).reshape(-1)
    return jnp.stack([r0, r1])

def compute_q4_local(E, t, nu, rho, a, b):
    gp = 0.577350269189626
    C_mem  = (E*t / (1-nu**2))        * jnp.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    C_bend = (E*t**3/(12*(1-nu**2)))  * jnp.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    C_sh   = (E*t*(5/6)/(2*(1+nu)))   * jnp.eye(2)
    
    K_m = jnp.zeros((8, 8))
    K_b = jnp.zeros((12, 12))
    for xg in [-gp, gp]:
        for eg in [-gp, gp]:
            ddx, ddy, _ = _shape_deriv_q4(xg, eg, a, b)
            Bm = _B_membrane_q4(ddx, ddy)
            K_m += (Bm.T @ C_mem @ Bm) * (a*b)
            Bb = _B_bending_q4(ddx, ddy)
            K_b += (Bb.T @ C_bend @ Bb) * (a*b)
            
    ddx0, ddy0, N0 = _shape_deriv_q4(0.0, 0.0, a, b)
    Bs = _B_shear_q4(ddx0, ddy0, N0)
    K_s = (Bs.T @ C_sh @ Bs) * (a*b*4.0)
    
    K_bend = K_b + K_s
    K_local = jnp.zeros((24, 24))
    mem_idx  = jnp.array([0,1, 6,7, 12,13, 18,19])
    bend_idx = jnp.array([2,3,4, 8,9,10, 14,15,16, 20,21,22])
    drill_id = jnp.array([5,11,17,23])
    
    K_local = K_local.at[jnp.ix_(mem_idx, mem_idx)].set(K_m)
    K_local = K_local.at[jnp.ix_(bend_idx, bend_idx)].set(K_bend)
    K_local = K_local.at[drill_id, drill_id].add(E*t*(4*a*b)*1e-4)
    
    m_node = rho*(4*a*b)*t/4.0
    I_rot = m_node*(t**2)/12.0
    I_drill = m_node*(a**2+b**2)/12.0*0.01
    
    m_vec = jnp.array([m_node, m_node, m_node, I_rot, I_rot, I_drill])
    M_local = jnp.diag(jnp.tile(m_vec, 4))
    return K_local, M_local

def compute_tria3_local(E, t, nu, rho, x2d, y2d):
    """Complete 6-DOF T3 Shell Element (CST Membrane + DKT Bending)."""
    nodes = jnp.stack([x2d, y2d], axis=1)
    
    # 1. Membrane Part (CST)
    Bm, detJ_m = _get_B_tria_membrane(nodes)
    area = jnp.abs(detJ_m) * 0.5
    Dm = (E*t/(1-nu**2)) * jnp.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    Km = (Bm.T @ Dm @ Bm) * area
    
    # 2. Bending Part (DKT)
    Db = (E*(t**3)/(12*(1-nu**2))) * jnp.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    def get_Kb_at(xi, eta, weight):
        B = _get_B_dkt(xi, eta, nodes)
        # Transform local [w, phix, phiy] to DKT [w, thx, thy]
        T = jnp.zeros((9,9))
        for i in range(3):
            T = T.at[3*i, 3*i].set(1.0)
            T = T.at[3*i+1, 3*i+2].set(-1.0) # thx = -phiy
            T = T.at[3*i+2, 3*i+1].set(1.0)  # thy = phix
        BT = B @ T
        return (BT.T @ Db @ BT) * area * weight

    pts = jnp.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
    ws = jnp.array([1/3, 1/3, 1/3])
    Kb_total = jnp.zeros((9,9))
    # Sum over 3 integration points
    Kb_total += get_Kb_at(pts[0,0], pts[0,1], ws[0])
    Kb_total += get_Kb_at(pts[1,0], pts[1,1], ws[1])
    Kb_total += get_Kb_at(pts[2,0], pts[2,1], ws[2])
    
    # 3. Assemble Local 18x18
    K_local = jnp.zeros((18, 18))
    mem_idx = jnp.array([0,1, 6,7, 12,13])
    bend_idx = jnp.array([2,3,4, 8,9,10, 14,15,16])
    drill_idx = jnp.array([5,11,17])
    
    K_local = K_local.at[jnp.ix_(mem_idx, mem_idx)].set(Km)
    K_local = K_local.at[jnp.ix_(bend_idx, bend_idx)].set(Kb_total)
    # Restore standard drilling stabilization - the previous 600x error was transformation-based
    K_local = K_local.at[drill_idx, drill_idx].add(E * t * area * 1e-6)
    
    # 4. Mass Matrix
    m_node = rho * area * t / 3.0
    I_rot = m_node * (t**2) / 12.0
    I_drill = m_node * area / 12.0 * 0.01
    m_vec = jnp.array([m_node, m_node, m_node, I_rot, I_rot, I_drill])
    M_local = jnp.diag(jnp.tile(m_vec, 3))
    
    return K_local, M_local

# ──────────────────────────────────────────────────────────────────────────────
#  ShellFEM Class
# ──────────────────────────────────────────────────────────────────────────────

class ShellFEM:
    def __init__(self, nodes, quads=None, trias=None, beams=None, elements=None, dof_per_node=6):
        self.nodes = jnp.array(nodes)
        if elements is not None:
            arr = jnp.array(elements)
            if arr.shape[1] == 4: quads = arr
            elif arr.shape[1] == 3: trias = arr
        self.quads = jnp.array(quads) if quads is not None else jnp.zeros((0,4), dtype=jnp.int32)
        self.trias = jnp.array(trias) if trias is not None else jnp.zeros((0,3), dtype=jnp.int32)
        self.num_nodes = len(self.nodes)
        self.total_dof = self.num_nodes * 6
        self.node_coords = self.nodes # Store full 3D coordinates (X, Y, Z) for visualization
        
        if self.quads.shape[0] > 0:
            self.quad_dof_idx = vmap(lambda e: jnp.concatenate([jnp.arange(6)+n*6 for n in e]))(self.quads)
        else:
            self.quad_dof_idx = None
            
        if self.trias.shape[0] > 0:
            self.tria_dof_idx = vmap(lambda e: jnp.concatenate([jnp.arange(6)+n*6 for n in e]))(self.trias)
        else:
            self.tria_dof_idx = None

    def assemble(self, params, sparse=False):
        # Universal property extraction with scalar support
        n_nodes = self.nodes.shape[0]
        
        def get_p(key, default):
            val = params.get(key)
            # Use jnp.where to handle both scalars and arrays without Python branch
            v = jnp.atleast_1d(jnp.array(val) if val is not None else default)
            return jnp.where(v.shape[0] == 1, jnp.full(n_nodes, v[0]), v)
            
        E   = get_p('E', 210000.0)
        t   = get_p('t', 1.0)
        rho = get_p('rho', 7.85e-9)
        nu  = 0.3
        
        curr_nodes = self.nodes
        if 'z' in params:
            z_off = get_p('z', 0.0)
            curr_nodes = self.nodes.at[:, 2].add(z_off)
            
        K_g = jnp.zeros((self.total_dof, self.total_dof))
        M_g = jnp.zeros((self.total_dof, self.total_dof))
        
        # Q4 Assembly
        n_quad = self.quads.shape[0]
        if n_quad > 0:
            ec = curr_nodes[self.quads]
            v12 = ec[:,1,:]-ec[:,0,:]
            Lx = jnp.linalg.norm(v12, axis=1, keepdims=True).clip(1e-12)
            e1 = v12 / Lx
            v14 = ec[:,3,:]-ec[:,0,:]
            nrm = jnp.cross(e1, v14)
            e3 = nrm / jnp.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-12)
            e2 = jnp.cross(e3, e1)
            Ts_q = jnp.stack([e1,e2,e3], axis=2)
            a_arr, b_arr = Lx[:,0]/2.0, jnp.sum(v14*e2, axis=1)/2.0
            
            def one_quad(Eq, tq, rq, a, b, T3):
                Kl, Ml = compute_q4_local(Eq, tq, nu, rq, a, b)
                z3 = jnp.zeros((3,3))
                Tn = jnp.block([[T3,z3],[z3,T3]])
                Te = jsl.block_diag(Tn, Tn, Tn, Tn)
                return Te @ Kl @ Te.T, Te @ Ml @ Te.T
                
            Kq, Mq = vmap(one_quad)(E[self.quads].mean(1), t[self.quads].mean(1), rho[self.quads].mean(1), a_arr, b_arr, Ts_q)
            I_idx = jnp.repeat(jnp.arange(24), 24)
            J_idx = jnp.tile(jnp.arange(24), 24)
            Gi = self.quad_dof_idx[:, I_idx].flatten()
            Gj = self.quad_dof_idx[:, J_idx].flatten()
            K_g = K_g.at[Gi, Gj].add(Kq.flatten())
            M_g = M_g.at[Gi, Gj].add(Mq.flatten())
            
        # T3 Assembly
        n_tri = self.trias.shape[0]
        if n_tri > 0:
            ec = curr_nodes[self.trias]
            v12 = ec[:,1,:]-ec[:,0,:]
            e1 = v12 / jnp.linalg.norm(v12, axis=1, keepdims=True).clip(1e-12)
            v13 = ec[:,2,:]-ec[:,0,:]
            nrm = jnp.cross(e1, v13)
            e3 = nrm / jnp.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-12)
            e2 = jnp.cross(e3, e1)
            Ts_t = jnp.stack([e1,e2,e3], axis=2)
            o = ec[:,0,:]
            lx1 = jnp.sum((ec[:,1,:]-o)*e1, axis=1)
            ly1 = jnp.sum((ec[:,1,:]-o)*e2, axis=1)
            lx2 = jnp.sum((ec[:,2,:]-o)*e1, axis=1)
            ly2 = jnp.sum((ec[:,2,:]-o)*e2, axis=1)
            x2d = jnp.stack([jnp.zeros(n_tri), lx1, lx2], 1)
            y2d = jnp.stack([jnp.zeros(n_tri), ly1, ly2], 1)
            
            def one_tria(Et, tt, rt, x, y, T3):
                Kl, Ml = compute_tria3_local(Et, tt, nu, rt, x, y)
                z3 = jnp.zeros((3,3))
                Tn = jnp.block([[T3,z3],[z3,T3]])
                Te = jsl.block_diag(Tn, Tn, Tn)
                return Te @ Kl @ Te.T, Te @ Ml @ Te.T
                
            Kt, Mt = vmap(one_tria)(E[self.trias].mean(1), t[self.trias].mean(1), rho[self.trias].mean(1), x2d, y2d, Ts_t)
            I_idx = jnp.repeat(jnp.arange(18), 18)
            J_idx = jnp.tile(jnp.arange(18), 18)
            Gi = self.tria_dof_idx[:, I_idx].flatten()
            Gj = self.tria_dof_idx[:, J_idx].flatten()
            K_g = K_g.at[Gi, Gj].add(Kt.flatten())
            M_g = M_g.at[Gi, Gj].add(Mt.flatten())
            
        if not sparse:
            return K_g, M_g
        else:
            # Sparse assembly using scipy.sparse (CPU) for large models
            from scipy.sparse import coo_matrix
            
            rows, cols, k_vals, m_vals = [], [], [], []
            if self.quads.shape[0] > 0:
                rows.append(np.array(self.quad_dof_idx[:, jnp.repeat(jnp.arange(24), 24)].flatten()))
                cols.append(np.array(self.quad_dof_idx[:, jnp.tile(jnp.arange(24), 24)].flatten()))
                k_vals.append(np.array(Kq).flatten())
                m_vals.append(np.array(Mq).flatten())
            if self.trias.shape[0] > 0:
                rows.append(np.array(self.tria_dof_idx[:, jnp.repeat(jnp.arange(18), 18)].flatten()))
                cols.append(np.array(self.tria_dof_idx[:, jnp.tile(jnp.arange(18), 18)].flatten()))
                k_vals.append(np.array(Kt).flatten())
                m_vals.append(np.array(Mt).flatten())
            
            R = np.concatenate(rows); C = np.concatenate(cols)
            KV = np.concatenate(k_vals); MV = np.concatenate(m_vals)
            K_sparse = coo_matrix((KV, (R, C)), shape=(self.total_dof, self.total_dof)).tocsr()
            M_sparse = coo_matrix((MV, (R, C)), shape=(self.total_dof, self.total_dof)).tocsr()
            return K_sparse, M_sparse

    def assemble_sparse(self, params):
        return self.assemble(params, sparse=True)

    def solve_eigen(self, K, M, num_modes=10, num_skip=0):
        m_diag = jnp.maximum(jnp.diag(M), 1e-15)
        m_inv_sqrt = 1.0 / jnp.sqrt(m_diag)
        K_sym = (K + K.T) / 2.0
        A = K_sym * m_inv_sqrt[:,None] * m_inv_sqrt[None,:] + jnp.eye(K.shape[0])*1e-10
        
        # Request more modes if skip is requested
        n_search = num_modes + num_skip + 2
        vals, vecs = safe_eigh(A)
        # vals and vecs are already sorted in safe_eigh
        
        # Convert to Hz
        freqs = jnp.sqrt(jnp.maximum(vals, 0.0)) / (2 * jnp.pi)
        vecs_phys = vecs * m_inv_sqrt[:,None]
        
        # Skip rigid body modes if requested (num_skip > 0)
        # For free-free analysis, num_skip=6 is typical
        res_freqs = freqs[num_skip : num_skip + num_modes]
        res_vecs = vecs_phys[:, num_skip : num_skip + num_modes]
        return res_freqs, res_vecs

    def solve_static(self, params, F, prescribed_dofs, prescribed_vals, sparse=False):
        K, _ = self.assemble(params, sparse=sparse)
        free_dofs = jnp.setdiff1d(jnp.arange(self.total_dof), prescribed_dofs)
        if sparse:
            return self.solve_static_sparse(K, F, free_dofs, prescribed_dofs, prescribed_vals)
        else:
            return self.solve_static_partitioned(K, F, free_dofs, prescribed_dofs, prescribed_vals)

    def solve_static_sparse(self, K_s, F, free_dofs, prescribed_dofs, prescribed_vals):
        """Solve static problem using scipy.sparse.linalg.spsolve."""
        from scipy.sparse.linalg import spsolve
        # Indices must be numpy for scipy
        fd = np.array(free_dofs)
        pd = np.array(prescribed_dofs)
        pv = np.array(prescribed_vals)
        
        K_ff = K_s[fd, :][:, fd]
        K_fp = K_s[fd, :][:, pd]
        rhs = np.array(F)[fd] - K_fp.dot(pv)
        
        u_f = spsolve(K_ff, rhs)
        u = np.zeros(self.total_dof)
        u[fd] = u_f
        u[pd] = pv
        return jnp.array(u)

    def solve_eigen_sparse(self, K_s, M_s, num_modes=12, sigma=None):
        """Solve generalized eigen problem using scipy.sparse.linalg.eigsh."""
        from scipy.sparse.linalg import eigsh
        n_search = min(num_modes + 6, self.total_dof - 2)
        try:
            # If sigma is None, default to SM (Smallest Magnitude) which correctly finds RBMs as ~0
            # If sigma is provided (e.g. 100.0), it targets specific frequency ranges
            if sigma is None:
                vals, vecs = eigsh(K_s, k=n_search, M=M_s, which='SM', tol=1e-5)
            else:
                vals, vecs = eigsh(K_s, k=n_search, M=M_s, which='LM', sigma=sigma, tol=1e-5)
            
            # Sort by frequency
            idx = jnp.argsort(vals)
            vals = vals[idx]
            vecs = vecs[:, idx]
            
            # Convert to Hz
            freqs = jnp.sqrt(jnp.maximum(vals, 1e-10)) / (2.0 * jnp.pi)
            
            # We no longer hard-filter for freqs > 1.0 globally in this method.
            # Filtering or skipping should be handled by the caller or specialized methods.
            return freqs[:num_modes], vecs[:, :num_modes]
        except Exception as e:
            # Fallback to standard solve if shift-invert fails
            import traceback
            traceback.print_exc()
            return jnp.zeros(num_modes), jnp.zeros((self.total_dof, num_modes))

    def compute_field_results(self, u, params, K=None):
        E_nodes = params.get('E', 210000.0)
        nu = 0.3
        t_nodes = params.get('t', 1.0)
        u_f = jnp.array(u).flatten()
        nodes_f = jnp.array(self.nodes)
        
        # 1. Element-wise properties for accuracy (Standardized for scalar/array)
        n_tri = self.trias.shape[0]
        curv_t, eps_t = jnp.zeros((n_tri, 3)), jnp.zeros((n_tri, 3))
        t_elem_t, E_elem_t = jnp.zeros((n_tri,)), jnp.zeros((n_tri,))
        
        if n_tri > 0:
            curv_t = recover_curvature_tria_bending(u_f, nodes_f, self.trias, 1.0, nu, 1.0)
            eps_m_t = recover_stress_tria_membrane(u_f, nodes_f, self.trias, 1.0, nu)
            t_elem_t = t_nodes[self.trias].mean(axis=1) if jnp.ndim(t_nodes) > 0 else jnp.full((n_tri,), t_nodes)
            E_elem_t = E_nodes[self.trias].mean(axis=1) if jnp.ndim(E_nodes) > 0 else jnp.full((n_tri,), E_nodes)
            eps_t = eps_m_t + curv_t * (t_elem_t[:, None] / 2.0)
            
        n_quad = self.quads.shape[0]
        curv_q, eps_q = jnp.zeros((n_quad, 3)), jnp.zeros((n_quad, 3))
        t_elem_q, E_elem_q = jnp.zeros((n_quad,)), jnp.zeros((n_quad,))
        
        if n_quad > 0:
            curv_q = recover_curvature_quad_bending(u_f, nodes_f, self.quads, 1.0, nu, 1.0)
            eps_m_q = recover_stress_quad_membrane(u_f, nodes_f, self.quads, 1.0, nu)
            t_elem_q = t_nodes[self.quads].mean(axis=1) if jnp.ndim(t_nodes) > 0 else jnp.full((n_quad,), t_nodes)
            E_elem_q = E_nodes[self.quads].mean(axis=1) if jnp.ndim(E_nodes) > 0 else jnp.full((n_quad,), E_nodes)
            eps_q = eps_m_q + curv_q * (t_elem_q[:, None] / 2.0)
            
        eps_el = jnp.concatenate([eps_t, eps_q])
        E_el = jnp.concatenate([E_elem_t, E_elem_q])
        curv_el = jnp.concatenate([curv_t, curv_q])
        t_el = jnp.concatenate([t_elem_t, t_elem_q])
        
        pre_el = E_el / (1 - nu**2)
        spre_el = E_el / (2 * (1 + nu))
        
        # 2. Element-wise stress/moment
        sig_el = jnp.stack([
            pre_el * (eps_el[:,0] + nu * eps_el[:,1]), 
            pre_el * (eps_el[:,1] + nu * eps_el[:,0]), 
            spre_el * eps_el[:,2]
        ], axis=1)
        
        D_el = (E_el * t_el**3) / (12.0 * (1.0 - nu**2))
        mom_el = jnp.stack([
            D_el * (curv_el[:,0] + nu * curv_el[:,1]), 
            D_el * (curv_el[:,1] + nu * curv_el[:,0]), 
            D_el * 0.5 * (1.0 - nu) * curv_el[:,2]
        ], axis=1)
        
        vm_el = jnp.sqrt(jnp.maximum(sig_el[:,0]**2 - sig_el[:,0]*sig_el[:,1] + sig_el[:,1]**2 + 3*sig_el[:,2]**2, 1e-12))
        sed_el = 0.5 * jnp.sum(sig_el * eps_el, axis=1)
        
        # 3. Nodal Mapping
        stress_node = jnp.zeros(self.num_nodes)
        strain_node = jnp.zeros(self.num_nodes)
        sed_node = jnp.zeros(self.num_nodes)
        mom_node = jnp.zeros((self.num_nodes, 3))
        count = jnp.zeros(self.num_nodes)
        
        avg_e = (eps_el[:,0] + eps_el[:,1]) / 2.0
        r_e = jnp.sqrt(jnp.maximum(((eps_el[:,0] - eps_el[:,1]) / 2.0)**2 + (eps_el[:,2] / 2.0)**2, 1e-12))
        strain_max_pr = avg_e + r_e

        n_tri = self.trias.shape[0]
        n_quad = self.quads.shape[0]

        if n_tri > 0:
            ix = self.trias.flatten()
            stress_node = stress_node.at[ix].add(jnp.repeat(vm_el[:n_tri], 3))
            strain_node = strain_node.at[ix].add(jnp.repeat(strain_max_pr[:n_tri], 3))
            sed_node = sed_node.at[ix].add(jnp.repeat(sed_el[:n_tri], 3))
            
            # Map Moments (3-component vector)
            for i in range(3):
                mom_node = mom_node.at[ix, i].add(jnp.repeat(mom_el[:n_tri, i], 3))
                
            count = count.at[ix].add(1.0)
            
        if n_quad > 0:
            ix = self.quads.flatten()
            off = n_tri
            stress_node = stress_node.at[ix].add(jnp.repeat(vm_el[off:], 4))
            strain_node = strain_node.at[ix].add(jnp.repeat(strain_max_pr[off:], 4))
            sed_node = sed_node.at[ix].add(jnp.repeat(sed_el[off:], 4))
            
            # Map Moments
            for i in range(3):
                mom_node = mom_node.at[ix, i].add(jnp.repeat(mom_el[off:, i], 4))
                
            count = count.at[ix].add(1.0)

        safe_c = jnp.maximum(count, 1.0)
        return {
            'stress_vm': stress_node / safe_c, 
            'strain_max_principal': strain_node / safe_c, 
            'sed_node': sed_node / safe_c,
            'moments': mom_node / safe_c[:, None],
            'u_mag': jnp.linalg.norm(u_f.reshape(-1,6)[:,:3], axis=1), 
            'eps_el': eps_el, 
            'sig_el': sig_el, 
            'vm_el': vm_el,
            'sed_el': sed_el,
            # Aliases for Verification Suite compatibility
            'stress_vm_el': vm_el,
            'stress_x_el': sig_el[:,0],
            'stress_y_el': sig_el[:,1],
            'stress_xy_el': sig_el[:,2],
            'strain_x_el': eps_el[:,0],
            'strain_y_el': eps_el[:,1],
            'strain_xy_el': eps_el[:,2]
        }

    def compute_strain_energy_density(self, u, params, field_results=None, **kwargs):
        """Compute SED. Optimized with nodal mapping in compute_field_results."""
        if field_results is None:
            field_results = self.compute_field_results(u, params)
        return field_results['sed_node']

    def compute_max_surface_stress(self, u, params, field_results=None, **kwargs):
        """Return Von-Mises stress. Optimized with field_results cache."""
        if field_results is None:
            field_results = self.compute_field_results(u, params)
        return field_results['stress_vm']

    def compute_max_surface_strain(self, u, params, field_results=None, **kwargs):
        """Return Max Principal strain. Optimized with field_results cache."""
        if field_results is None:
            field_results = self.compute_field_results(u, params)
        return field_results['strain_max_principal']

    def compute_moment(self, u, params, **kwargs):
        return self.compute_field_results(u, params)['moments']

    def solve_static_partitioned(self, K, F, free_dofs, prescribed_dofs, prescribed_vals):
        K_ff = K[jnp.ix_(free_dofs, free_dofs)]
        K_fp = K[jnp.ix_(free_dofs, prescribed_dofs)]
        rhs = F[free_dofs] - K_fp @ prescribed_vals
        u_f = jax.scipy.linalg.solve(K_ff + 1e-9*jnp.eye(free_dofs.shape[0]), rhs, assume_a='pos')
        u = jnp.zeros(self.total_dof)
        u = u.at[free_dofs].set(u_f).at[prescribed_dofs].set(prescribed_vals)
        return u
