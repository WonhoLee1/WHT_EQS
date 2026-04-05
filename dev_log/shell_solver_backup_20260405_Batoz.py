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
    return vmap(get_element_strain_inner)(jnp.arange(len(trias)))

def _get_B_dkt(nodes, xi, eta):
    """Explicit Batoz (1980) DKT B-matrix (3x9)."""
    x21 = nodes[1,0]-nodes[0,0]; y21 = nodes[1,1]-nodes[0,1]
    x32 = nodes[2,0]-nodes[1,0]; y32 = nodes[2,1]-nodes[1,1]
    x13 = nodes[0,0]-nodes[2,0]; y13 = nodes[0,1]-nodes[2,1]
    
    area = 0.5 * jnp.abs(x21*(-y13) - (-x13)*y21)
    inv2A = 0.5 / area
    
    L_sq = jnp.array([x32**2+y32**2, x13**2+y13**2, x21**2+y21**2])
    invL2 = 1.0 / jnp.where(L_sq < 1e-12, 1e-12, L_sq)
    
    p4, p5, p6 = -6*x32*invL2[0], -6*x13*invL2[1], -6*x21*invL2[2]
    q4, q5, q6 = 3*x32*y32*invL2[0], 3*x13*y13*invL2[1], 3*x21*y21*invL2[2]
    r4, r5, r6 = 3*y32**2*invL2[0], 3*y13**2*invL2[1], 3*y21**2*invL2[2]
    t4, t5, t6 = -6*y32*invL2[0], -6*y13*invL2[1], -6*y21*invL2[2]
    
    Hx_xi = jnp.array([
        p6*(1-2*xi)+(p5-p6)*eta, 
        q6*(1-2*xi)-(q5+q6)*eta, 
        -4+6*(xi+eta)+r6*(1-2*xi)-eta*(r5+r6), 
        -p6*(1-2*xi)+eta*(p4+p6), 
        q6*(1-2*xi)-eta*(q6-q4), 
        -2+6*xi+r6*(1-2*xi)+eta*(r4-r6), 
        -eta*(p5+p4), 
        eta*(q4-q5), 
        -eta*(r5-r4)
    ])
    
    Hx_eta = jnp.array([
        -p5*(1-2*eta)-xi*(p6-p5), 
        q5*(1-2*eta)-xi*(q5+q6), 
        -4+6*(xi+eta)+r5*(1-2*eta)-xi*(r5+r6), 
        xi*(p4+p6), 
        xi*(q4-q6), 
        -xi*(r6-r4), 
        p5*(1-2*eta)-xi*(p4+p5), 
        q5*(1-2*eta)+xi*(q4-q5), 
        -2+6*eta+r5*(1-2*eta)+xi*(r4-r5)
    ])
    
    Hy_xi = jnp.array([
        t6*(1-2*xi)+eta*(t5-t6), 
        1+r6*(1-2*xi)-eta*(r5+r6), 
        -q6*(1-2*xi)+eta*(q5+q6), 
        -t6*(1-2*xi)+eta*(t4+t6), 
        -1+r6*(1-2*xi)+eta*(r4-r6), 
        -q6*(1-2*xi)-eta*(q4-q6), 
        -eta*(t4+t5), 
        eta*(r4-r5), 
        -eta*(q4-q5)
    ])
    
    Hy_eta = jnp.array([
        -t5*(1-2*eta)-xi*(t6-t5), 
        1+r5*(1-2*eta)-xi*(r5+r6), 
        -q5*(1-2*eta)+xi*(q5+q6), 
        xi*(t4+t6), 
        xi*(r4-r6), 
        -xi*(q4-q6), 
        t5*(1-2*eta)-xi*(t4+t5), 
        -1+r5*(1-2*eta)+xi*(r4-r5), 
        -q5*(1-2*eta)-xi*(q4-q5)
    ])
    
    B1 = inv2A * ((-y13)*Hx_xi + y21*Hx_eta)
    B2 = inv2A * (x13*Hy_xi + (-x21)*Hy_eta)
    B3 = inv2A * (x13*Hx_xi + (-x21)*Hx_eta + (-y13)*Hy_xi + y21*Hy_eta)
    
    return jnp.stack([B1, B2, B3])

def recover_curvature_tria_bending(u, nodes, trias, E, nu, t):
    """Batoz DKT curvature recovery."""
    def get_curvature(idx):
        ix = trias[idx]
        tri_nodes = nodes[ix]
        u_all = u.reshape(-1, 6)[ix]
        u_b = u_all[:, 2:5] # w, tx, ty
        
        # Mapping: Batoz's [w,Beta_x,Beta_y] -> our [w,theta_x,theta_y]
        Tnode = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        T9 = jsl.block_diag(Tnode, Tnode, Tnode)
        
        # Center of triangle (1/3, 1/3)
        B = _get_B_dkt(tri_nodes, 1./3., 1./3.) @ T9
        kappa = B @ u_b.flatten()
        return -kappa # Kirchhoff convention
    return vmap(get_curvature)(jnp.arange(len(trias)))

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
        
        kx = jnp.dot(dn_dx, ty_l)
        ky = -jnp.dot(dn_dy, tx_l)
        kxy = jnp.dot(dn_dy, ty_l) - jnp.dot(dn_dx, tx_l)
        return jnp.array([kx, ky, kxy])
    return vmap(get_curvature)(jnp.arange(len(quads)))

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
    return vmap(get_eps)(jnp.arange(len(quads)))

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
    F = jnp.where(jnp.isinf(F), 0.0, F).at[jnp.diag_indices_from(A)].set(0.0)
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
    area = 0.5*jnp.abs((x2d[1]-x2d[0])*(y2d[2]-y2d[0])-(x2d[2]-x2d[0])*(y2d[1]-y2d[0]))
    nodes = jnp.stack([x2d, y2d], axis=1)
    C_mem = (E*t/(1-nu**2)) * jnp.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    
    y23, y31, y12 = y2d[1]-y2d[2], y2d[2]-y2d[1], y2d[0]-y2d[1]
    x32, x13, x21 = x2d[2]-x2d[1], x2d[0]-x2d[2], x2d[1]-x2d[0]
    B_m = (0.5/area) * jnp.array([
        [y23,0, y31,0, y12,0], 
        [0,x32, 0,x13, 0,x21], 
        [x32,y23, x13,y31, x21,y12]
    ])
    K_m = area * (B_m.T @ C_mem @ B_m)
    
    D_bend = (E*t**3/(12*(1-nu**2))) * jnp.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    Tnode = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    T9 = jsl.block_diag(Tnode, Tnode, Tnode)
    
    pts = jnp.array([[2/3, 1/6], [1/6, 2/3], [1/6, 1/6]])
    w_gp = 1.0/6.0
    K_b = jnp.zeros((9,9))
    for i in range(3):
        B = _get_B_dkt(nodes, pts[i,0], pts[i,1]) @ T9
        K_b += (w_gp * 2*area) * (B.T @ D_bend @ B)
        
    K_local = jnp.zeros((18,18))
    mem_id = jnp.array([0,1, 6,7, 12,13])
    bend_id = jnp.array([2,3,4, 8,9,10, 14,15,16])
    drill_id = jnp.array([5,11,17])
    
    K_local = K_local.at[jnp.ix_(mem_id, mem_id)].set(K_m)
    K_local = K_local.at[jnp.ix_(bend_id, bend_id)].set(K_b)
    K_local = K_local.at[drill_id, drill_id].add(E*t*area*1e-4)
    
    m_node = rho*area*t/3.0
    I_rot = m_node*(t**2)/12.0
    I_drill = m_node*area/12.0*0.01
    
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
        self.node_coords = self.nodes[:, :2]
        
        if len(self.quads) > 0:
            self.quad_dof_idx = vmap(lambda e: jnp.concatenate([jnp.arange(6)+n*6 for n in e]))(self.quads)
        else:
            self.quad_dof_idx = None
            
        if len(self.trias) > 0:
            self.tria_dof_idx = vmap(lambda e: jnp.concatenate([jnp.arange(6)+n*6 for n in e]))(self.trias)
        else:
            self.tria_dof_idx = None

    def assemble(self, params, sparse=False):
        E, t, rho, nu = params.get('E'), params.get('t'), params.get('rho'), 0.3
        K_g = jnp.zeros((self.total_dof, self.total_dof))
        M_g = jnp.zeros((self.total_dof, self.total_dof))
        
        # Q4 Assembly
        if len(self.quads) > 0:
            ec = self.nodes[self.quads]
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
        if len(self.trias) > 0:
            ec = self.nodes[self.trias]
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
            x2d = jnp.stack([jnp.zeros(len(self.trias)), lx1, lx2], 1)
            y2d = jnp.stack([jnp.zeros(len(self.trias)), ly1, ly2], 1)
            
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
            
        return K_g, M_g

    def solve_eigen(self, K, M, num_modes=10):
        m_diag = jnp.maximum(jnp.diag(M), 1e-15)
        m_inv_sqrt = 1.0 / jnp.sqrt(m_diag)
        K_sym = (K + K.T) / 2.0
        A = K_sym * m_inv_sqrt[:,None] * m_inv_sqrt[None,:] + jnp.eye(K.shape[0])*1e-10
        vals, vecs = safe_eigh(A)
        return jnp.maximum(vals, 0.0)[:num_modes], (vecs * m_inv_sqrt[:,None])[:, :num_modes]

    def solve_static(self, params, F, prescribed_dofs, prescribed_vals):
        K, _ = self.assemble(params)
        free_dofs = jnp.setdiff1d(jnp.arange(self.total_dof), prescribed_dofs)
        return self.solve_static_partitioned(K, F, free_dofs, prescribed_dofs, prescribed_vals)

    def compute_field_results(self, u, params, K=None):
        E_nodes = params.get('E', 210000.0)
        nu = 0.3
        t_nodes = params.get('t', 1.0)
        u_f = jnp.array(u).flatten()
        nodes_f = jnp.array(self.nodes)
        
        curv_t = jnp.zeros((0,3))
        eps_t = jnp.zeros((0,3))
        if len(self.trias) > 0:
            curv_t = recover_curvature_tria_bending(u_f, nodes_f, self.trias, 1.0, nu, 1.0)
            eps_m_t = recover_stress_tria_membrane(u_f, nodes_f, self.trias, 1.0, nu)
            eps_t = eps_m_t + curv_t * (t_nodes.mean() / 2.0)
            
        curv_q = jnp.zeros((0,3))
        eps_q = jnp.zeros((0,3))
        if len(self.quads) > 0:
            curv_q = recover_curvature_quad_bending(u_f, nodes_f, self.quads, 1.0, nu, 1.0)
            eps_m_q = recover_stress_quad_membrane(u_f, nodes_f, self.quads, 1.0, nu)
            eps_q = eps_m_q + curv_q * (t_nodes.mean() / 2.0)
            
        eps_el = jnp.concatenate([eps_t, eps_q])
        curv_el = jnp.concatenate([curv_t, curv_q])
        
        E_val = E_nodes.mean()
        D_val = (E_val * t_nodes.mean()**3) / (12 * (1 - nu**2))
        pre = E_val / (1 - nu**2)
        s_pre = E_val / (2 * (1 + nu))
        
        sig_el = jnp.stack([
            pre * (eps_el[:,0] + nu * eps_el[:,1]), 
            pre * (eps_el[:,1] + nu * eps_el[:,0]), 
            s_pre * eps_el[:,2]
        ], axis=1)
        
        mom_el = jnp.stack([
            D_val * (curv_el[:,0] + nu * curv_el[:,1]), 
            D_val * (curv_el[:,1] + nu * curv_el[:,0]), 
            D_val * 0.5 * (1 - nu) * curv_el[:,2]
        ], axis=1)
        
        vm_el = jnp.sqrt(jnp.maximum(sig_el[:,0]**2 - sig_el[:,0]*sig_el[:,1] + sig_el[:,1]**2 + 3*sig_el[:,2]**2, 1e-12))
        avg_e = (eps_el[:,0] + eps_el[:,1]) / 2.0
        r_e = jnp.sqrt(jnp.maximum(((eps_el[:,0] - eps_el[:,1]) / 2.0)**2 + (eps_el[:,2] / 2.0)**2, 1e-12))
        
        stress_node = jnp.zeros(self.num_nodes)
        strain_node = jnp.zeros(self.num_nodes)
        mom_node = jnp.zeros((self.num_nodes, 3))
        count = jnp.zeros(self.num_nodes)
        
        if len(self.trias) > 0:
            ix = self.trias.flatten()
            stress_node = stress_node.at[ix].add(jnp.repeat(vm_el[:len(self.trias)], 3))
            strain_node = strain_node.at[ix].add(jnp.repeat(avg_e[:len(self.trias)] + r_e[:len(self.trias)], 3))
            count = count.at[ix].add(1.0)
            for i in range(3):
                mom_node = mom_node.at[ix, i].add(jnp.repeat(mom_el[:len(self.trias), i], 3))
                
        if len(self.quads) > 0:
            ix = self.quads.flatten()
            off = len(self.trias)
            stress_node = stress_node.at[ix].add(jnp.repeat(vm_el[off:], 4))
            strain_node = strain_node.at[ix].add(jnp.repeat(avg_e[off:] + r_e[off:], 4))
            count = count.at[ix].add(1.0)
            for i in range(3):
                mom_node = mom_node.at[ix, i].add(jnp.repeat(mom_el[off:, i], 4))
                
        safe_c = jnp.maximum(count, 1.0)
        return {
            'stress_vm': stress_node / safe_c, 
            'strain_max_principal': strain_node / safe_c, 
            'u_mag': jnp.linalg.norm(u_f.reshape(-1,6)[:,:3], axis=1), 
            'eps_el': eps_el, 
            'sig_el': sig_el, 
            'moments': mom_node / safe_c[:,None],
            'stress_vm_el': vm_el,
            'stress_x_el': sig_el[:, 0],
            'strain_x_el': eps_el[:, 0]
        }

    def compute_strain_energy_density(self, u, params, K=None):
        res = self.compute_field_results(u, params)
        sed_el = 0.5 * jnp.sum(res['sig_el'] * res['eps_el'], axis=1)
        sed_node = jnp.zeros(self.num_nodes)
        count = jnp.zeros(self.num_nodes)
        if len(self.trias) > 0:
            sed_node = sed_node.at[self.trias.flatten()].add(jnp.repeat(sed_el[:len(self.trias)], 3))
            count = count.at[self.trias.flatten()].add(1.0)
        if len(self.quads) > 0:
            off = len(self.trias)
            sed_node = sed_node.at[self.quads.flatten()].add(jnp.repeat(sed_el[off:], 4))
            count = count.at[self.quads.flatten()].add(1.0)
        return sed_node / jnp.maximum(count, 1.0)

    def compute_max_surface_stress(self, u, params, K=None):
        return self.compute_field_results(u, params)['stress_vm']

    def compute_max_surface_strain(self, u, params, K=None):
        return self.compute_field_results(u, params)['strain_max_principal']

    def compute_moment(self, u, params, K=None):
        return self.compute_field_results(u, params)['moments']

    def solve_static_partitioned(self, K, F, free_dofs, prescribed_dofs, prescribed_vals):
        K_ff = K[jnp.ix_(free_dofs, free_dofs)]
        K_fp = K[jnp.ix_(free_dofs, prescribed_dofs)]
        rhs = F[free_dofs] - K_fp @ prescribed_vals
        u_f = jax.scipy.linalg.solve(K_ff + 1e-9*jnp.eye(len(free_dofs)), rhs, assume_a='pos')
        u = jnp.zeros(self.total_dof)
        u = u.at[free_dofs].set(u_f).at[prescribed_dofs].set(prescribed_vals)
        return u
