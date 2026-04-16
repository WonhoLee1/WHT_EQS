# -*- coding: utf-8 -*-
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import vmap, jit, custom_vjp, lax
import numpy as np
from functools import partial

# ──────────────────────────────────────────────────────────────────────────────
#  Reference: MITC3/MITC4 Theory (Bathe/Dvorkin)
# ──────────────────────────────────────────────────────────────────────────────

def _get_B_tria_membrane(nodes):
    x1, y1 = nodes[0,0], nodes[0,1]
    x2, y2 = nodes[1,0], nodes[1,1]
    x3, y3 = nodes[2,0], nodes[2,1]
    detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    b1, b2, b3 = y2-y3, y3-y1, y1-y2
    c1, c2, c3 = x3-x2, x1-x3, x2-x1
    inv2A = 1.0 / detJ
    Bm = jnp.array([[b1, 0, b2, 0, b3, 0],[0, c1, 0, c2, 0, c3],[c1, b1, c2, b2, c3, b3]]) * inv2A
    return Bm, detJ

def _get_B_mitc3(nodes):
    """Integrated MITC3 Shear B-matrix logic for [w, thx, thy]"""
    x, y = nodes[:, 0], nodes[:, 1]
    ex = jnp.array([x[1]-x[0], x[2]-x[1], x[0]-x[2]])
    ey = jnp.array([y[1]-y[0], y[2]-y[1], y[0]-y[2]])
    L = jnp.sqrt(ex**2 + ey**2).clip(1e-12)
    def get_G_edge(i):
        j = (i + 1) % 3
        tx, ty = ex[i]/L[i], ey[i]/L[i]
        v = jnp.zeros(9).at[3*i].set(-1).at[3*j].set(1)
        vt = 0.5 * L[i]
        # beta.t = beta_x*tx + beta_y*ty = thy*tx - thx*ty
        v = v.at[3*i+1].set(-vt*ty).at[3*i+2].set(vt*tx)
        v = v.at[3*j+1].set(-vt*ty).at[3*j+2].set(vt*tx)
        return v
    G = jnp.stack([get_G_edge(0), get_G_edge(1)])
    M = jnp.array([[ex[0], ey[0]], [ex[1], ey[1]]]); invM = jnp.linalg.inv(M)
    return invM @ G

def _get_B_mitc4(xi, eta, nodes):
    """MITC4 Mixed Interpolated Covariant Shear Strain B-matrix for [w, thx, thy]"""
    dN_dxi = 0.25 * jnp.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
    dN_det = 0.25 * jnp.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
    j = jnp.array([[jnp.dot(dN_dxi, nodes[:,0]), jnp.dot(dN_dxi, nodes[:,1])],[jnp.dot(dN_det, nodes[:,0]), jnp.dot(dN_det, nodes[:,1])]])
    invJ = jnp.linalg.inv(j)
    def get_cov_edge(xi_p, eta_p, is_xi):
        jp = jnp.array([[jnp.dot(0.25*jnp.array([-(1-eta_p), (1-eta_p), (1+eta_p), -(1+eta_p)]), nodes[:,0]), 
                         jnp.dot(0.25*jnp.array([-(1-eta_p), (1-eta_p), (1+eta_p), -(1+eta_p)]), nodes[:,1])],
                        [jnp.dot(0.25*jnp.array([-(1-xi_p), -(1+xi_p), (1+xi_p), (1-xi_p)]), nodes[:,0]), 
                         jnp.dot(0.25*jnp.array([-(1-xi_p), -(1+xi_p), (1+xi_p), (1-xi_p)]), nodes[:,1])]])
        B = jnp.zeros(12); idx = jnp.arange(4)
        dN_p = 0.25*jnp.array([-(1-eta_p), (1-eta_p), (1+eta_p), -(1+eta_p)]) if is_xi else 0.25*jnp.array([-(1-xi_p), -(1+xi_p), (1+xi_p), (1-xi_p)])
        B = B.at[3*idx].set(dN_p)
        # beta.t = thy*tx - thx*ty
        vt = 0.25*jnp.array([(1-eta_p),(1-eta_p),(1+eta_p),(1+eta_p)]) if is_xi else 0.25*jnp.array([(1-xi_p),(1+xi_p),(1+xi_p),(1-xi_p)])
        tx, ty = (jp[0,0], jp[0,1]) if is_xi else (jp[1,0], jp[1,1])
        B = B.at[3*idx+1].set(-ty*vt).at[3*idx+2].set(tx*vt)
        return B
    B_gxi = 0.5*(1-eta)*get_cov_edge(0,-1,True) + 0.5*(1+eta)*get_cov_edge(0,1,True)
    B_get = 0.5*(1-xi)*get_cov_edge(-1,0,False) + 0.5*(1+xi)*get_cov_edge(1,0,False)
    return invJ @ jnp.stack([B_gxi, B_get]), jnp.linalg.det(j)


def _B_membrane_q4(dN_dx, dN_dy):
    z = jnp.zeros(4)
    return jnp.stack([jnp.stack([dN_dx, z], 1).reshape(-1), jnp.stack([z, dN_dy], 1).reshape(-1), jnp.stack([dN_dy, dN_dx], 1).reshape(-1)])

def _B_bending_q4(dN_dx, dN_dy):
    z = jnp.zeros(4)
    return jnp.stack([jnp.stack([z, z, dN_dx], 1).reshape(-1), jnp.stack([z, -dN_dy, z], 1).reshape(-1), jnp.stack([z, -dN_dx, dN_dy], 1).reshape(-1)])

def _get_B_bending_t3(nodes):
    """Constant Curvature B-matrix for T3 Mindlin Bending."""
    x1, y1, x2, y2, x3, y3 = nodes[0,0], nodes[0,1], nodes[1,0], nodes[1,1], nodes[2,0], nodes[2,1]
    detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    invJ = 1.0/detJ
    # Curvature Row 0: kappa_x = beta_x,x -> thy,x
    # Row 1: kappa_y = beta_y,y -> -thx,y
    # Row 2: kappa_xy = thy,y - thx,x
    b1, b2, b3 = y2-y3, y3-y1, y1-y2
    c1, c2, c3 = x3-x2, x1-x3, x2-x1
    Bb = jnp.zeros((3, 9))
    idx = jnp.arange(3)
    dn_dx = invJ * jnp.array([y2-y3, y3-y1, y1-y2])
    dn_dy = invJ * jnp.array([x3-x2, x1-x3, x2-x1])
    Bb = Bb.at[0, 3*idx+2].set(dn_dx)
    Bb = Bb.at[1, 3*idx+1].set(-dn_dy)
    Bb = Bb.at[2, 3*idx+2].set(dn_dy)
    Bb = Bb.at[2, 3*idx+1].set(-dn_dx)
    return Bb

def compute_mitc3_local(E, t, nu, rho, x2d, y2d):
    """
    Tria Assembly: MITC3 (Membrane CST + Bending Mindlin + MITC3 Shear).
    Nodal DOFs: [u, v, w, thx, thy, rz]
    """
    nodes2d = jnp.stack([x2d, y2d], 1)
    detJ = (x2d[1]-x2d[0])*(y2d[2]-y2d[0]) - (x2d[2]-x2d[0])*(y2d[1]-y2d[0])
    area = 0.5 * jnp.abs(detJ)
    
    # 1. Membrane (CST)
    Dm = (E*t/(1-nu**2)) * jnp.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    Bm_c, _ = _get_B_tria_membrane(nodes2d)
    Km = (Bm_c.T @ Dm @ Bm_c) * area
    
    # 2. Bending (Standard Mindlin Curvature: [w, thx, thy])
    Db = (E*t**3/(12*(1-nu**2))) * jnp.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    Bb = _get_B_bending_t3(nodes2d)
    Kb = (Bb.T @ Db @ Bb) * area
    
    # 3. Shear (MITC3 Mixed Edge-Interpolation)
    G = E / (2 * (1 + nu)) 
    Ds = (G * t * 5/6) * jnp.eye(2)
    Bs = _get_B_mitc3(nodes2d)
    Ks = (Bs.T @ Ds @ Bs) * area
    
    # Assembly to 18x18
    # Local DOFs: [u1, v1, w1, thx1, thy1, rz1, ...]
    K_full = jnp.zeros((18, 18))
    m_p = jnp.array([0,1, 6,7, 12,13])
    b_p = jnp.array([2,3,4, 8,9,10, 14,15,16])
    
    K_full = K_full.at[jnp.ix_(m_p, m_p)].set(Km)
    # No Te_b needed if Bb/Bs are already in [w, thx, thy] space matching global [W, RX, RY]
    K_full = K_full.at[jnp.ix_(b_p, b_p)].set(Kb + Ks)
    
    # Stabilization for Drilling
    K_full = K_full.at[jnp.array([5, 11, 17]), jnp.array([5, 11, 17])].add(1e-7 * G * t * area)
    
    # Mass
    m_n = rho * area * t / 3.0; Ir = m_n * (t**2) / 12.0
    M_full = jnp.diag(jnp.tile(jnp.array([m_n, m_n, m_n, Ir, Ir, Ir*0.01]), 3))
    return K_full, M_full



def recover_curvature_tria_bending(u, nodes, trias, E, nu, t):
    """
    Recover curvature from triangular shell elements using standardized rotation mapping.
    Matches global [w, thx, thy] to MITC3 curvature.
    """
    def get_c(i):
        ix = trias[i]
        u_e = u.reshape(-1, 6)[ix]
        # Global rotations [thx, thy] -> Local Curvature DOFs [beta_x, beta_y]
        # Mapping: beta_x = thy, beta_y = -thx
        w_val = u_e[:, 2]
        thx = u_e[:, 3]
        thy = u_e[:, 4]
        # Local bending DOFs: [w, thx, thy]
        ud_local = jnp.stack([w_val, thx, thy], axis=1).flatten()
        return _get_B_bending_t3(nodes[ix][:, :2]) @ ud_local




    return vmap(get_c)(jnp.arange(trias.shape[0]))

def recover_curvature_quad_bending(u, nodes, quads, E, nu, t):
    def get_c(i):
        ix = quads[i]; qn = nodes[ix]; v1,v2 = qn[1]-qn[0],qn[3]-qn[0]; e1 = v1/jnp.linalg.norm(v1).clip(1e-12)
        e3 = jnp.cross(e1,v2); e3 /= jnp.linalg.norm(e3).clip(1e-12); e2 = jnp.cross(e3,e1)
        dN_dxi = jnp.array([-0.25, 0.25, 0.25, -0.25]); dN_det = jnp.array([-0.25, -0.25, 0.25, 0.25])
        p2d = jnp.stack([jnp.dot(qn-qn[0],e1), jnp.dot(qn-qn[0],e2)], 1)
        j = jnp.array([[jnp.dot(dN_dxi,p2d[:,0]),jnp.dot(dN_dxi,p2d[:,1])],[jnp.dot(dN_det,p2d[:,0]),jnp.dot(dN_det,p2d[:,1])]])
        u_g = u.reshape(-1,6)[ix]; tx = u_g[:,3]*e1[0]+u_g[:,4]*e1[1]; ty = u_g[:,3]*e2[0]+u_g[:,4]*e2[1]
        invJ = jnp.linalg.inv(j); dN_dx = invJ[0,0]*dN_dxi+invJ[0,1]*dN_det; dN_dy = invJ[1,0]*dN_dxi+invJ[1,1]*dN_det
        # Apply global negation to align with analytical convention and fixed T3 logic
        return -jnp.array([jnp.dot(dN_dx, ty), -jnp.dot(dN_dy, tx), jnp.dot(dN_dy, ty) - jnp.dot(dN_dx, tx)])



    return vmap(get_c)(jnp.arange(quads.shape[0]))

def recover_stress_tria_membrane(u, nodes, trias, E, nu):
    def get_e(i):
        ix = trias[i]; B, _ = _get_B_tria_membrane(nodes[ix]); return B@u.reshape(-1,6)[ix][:,:2].flatten()
    return vmap(get_e)(jnp.arange(trias.shape[0]))

def recover_stress_quad_membrane(u, nodes, quads, E, nu):
    def get_e(i):
        ix = quads[i]; qn = nodes[ix]; v1,v2 = qn[1]-qn[0],qn[3]-qn[0]; e1 = v1/jnp.linalg.norm(v1).clip(1e-12)
        e3 = jnp.cross(e1,v2); e3 /= jnp.linalg.norm(e3).clip(1e-12); e2 = jnp.cross(e3,e1)
        dN_dxi = jnp.array([-0.25, 0.25, 0.25, -0.25]); dN_det = jnp.array([-0.25, -0.25, 0.25, 0.25])
        p2d = jnp.stack([jnp.dot(qn-qn[0],e1), jnp.dot(qn-qn[0],e2)], 1)
        j = jnp.array([[jnp.dot(dN_dxi,p2d[:,0]),jnp.dot(dN_dxi,p2d[:,1])],[jnp.dot(dN_det,p2d[:,0]),jnp.dot(dN_det,p2d[:,1])]])
        invJ = jnp.linalg.inv(j); dN_dx = invJ[0,0]*dN_dxi+invJ[0,1]*dN_det; dN_dy = invJ[1,0]*dN_dxi+invJ[1,1]*dN_det
        u_g = u.reshape(-1,6)[ix]; ux = u_g[:,0]*e1[0]+u_g[:,1]*e1[1]; uy = u_g[:,0]*e2[0]+u_g[:,1]*e2[1]
        return jnp.array([jnp.dot(dN_dx,ux), jnp.dot(dN_dy,uy), jnp.dot(dN_dy,ux)+jnp.dot(dN_dx,uy)])
    return vmap(get_e)(jnp.arange(quads.shape[0]))

@custom_vjp
def safe_eigh(A):
    vals, vecs = jnp.linalg.eigh(A); return vals, vecs
def safe_eigh_fwd(A):
    vals, vecs = jnp.linalg.eigh(A); return (vals, vecs), (vals, vecs)
def safe_eigh_bwd(res, g):
    vals, vecs = res; g_v, g_vc = g; grad_A_v = jnp.einsum('k,ik,jk->ij', g_v, vecs, vecs)
    diff = vals[:, None] - vals[None, :]; F = jnp.where(jnp.abs(diff) < 1e-9, 0, 1/diff).at[jnp.diag_indices(vecs.shape[0])].set(0)
    total = 0.5 * (grad_A_v + vecs@(F*(vecs.T@g_vc))@vecs.T + (grad_A_v + vecs@(F*(vecs.T@g_vc))@vecs.T).T)
    return (total,)
safe_eigh.defvjp(safe_eigh_fwd, safe_eigh_bwd)

def _B_membrane_q4_fast(dN_dx, dN_dy):
    """Fast construction of membrane B-matrix without .at[].set()"""
    z4 = jnp.zeros(4)
    row0 = jnp.stack([dN_dx, z4], axis=1).flatten()
    row1 = jnp.stack([z4, dN_dy], axis=1).flatten()
    row2 = jnp.stack([dN_dy, dN_dx], axis=1).flatten()
    return jnp.stack([row0, row1, row2])

def _B_bending_q4_fast(dN_dx, dN_dy):
    """Bending B-matrix for Q4 [w, thx, thy]"""
    z4 = jnp.zeros(4)
    # kappa_x = thy,x
    row0 = jnp.stack([z4, z4, dN_dx], axis=1).flatten()
    # kappa_y = -thx,y
    row1 = jnp.stack([z4, -dN_dy, z4], axis=1).flatten()
    # kappa_xy = thy,y - thx,x
    row2 = jnp.stack([z4, -dN_dx, dN_dy], axis=1).flatten()
    return jnp.stack([row0, row1, row2])


def compute_mitc4_local_fast(E, t, nu, rho, p2d):
    """Fully vectorized MITC4 assembly (no loops)"""
    C_m = (E*t/(1-nu**2)) * jnp.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    C_b = (E*t**3/(12*(1-nu**2))) * jnp.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    G = E/(2*(1+nu)); C_s = (G*t*5/6)*jnp.eye(2)
    gp = 0.577350269189626
    gps = jnp.array([[-gp, -gp], [gp, -gp], [gp, gp], [-gp, gp]])
    
    def quadrature_point(pt):
        xi, eta = pt
        dN_dxi = 0.25*jnp.array([-(1-eta),(1-eta),(1+eta),-(1+eta)])
        dN_det = 0.25*jnp.array([-(1-xi),-(1+xi),(1+xi),(1-xi)])
        j = jnp.array([[jnp.dot(dN_dxi,p2d[:,0]),jnp.dot(dN_dxi,p2d[:,1])],
                       [jnp.dot(dN_det,p2d[:,0]),jnp.dot(dN_det,p2d[:,1])]])
        detJ = jnp.abs(jnp.linalg.det(j))
        invJ = jnp.linalg.inv(j)
        dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_det
        dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_det
        Bm = _B_membrane_q4_fast(dN_dx, dN_dy)
        Bb = _B_bending_q4_fast(dN_dx, dN_dy)
        dKm = (Bm.T @ C_m @ Bm) * detJ
        dKb = (Bb.T @ C_b @ Bb) * detJ
        return dKm, dKb, detJ

    dKm_all, dKb_all, detJ_all = vmap(quadrature_point)(gps)
    K_m, K_b = dKm_all.sum(0), dKb_all.sum(0)
    
    # Shear (MITC4 Mixed Interpolation)
    def shear_mitc4(nodes):
        # We integrate MITC4 shear at 4 points (center of edges) implicitly via _get_B_mitc4
        # For simplicity and fast JAX execution, we evaluate shear at midpoints if possible, 
        # but _get_B_mitc4 provides 2x12 matrix.
        Bs, detJs = _get_B_mitc4(0.0, 0.0, nodes)
        return (Bs.T @ C_s @ Bs) * (detJs * 4.0)
    
    K_s = shear_mitc4(p2d)

    K_l = jnp.zeros((24,24))
    m_p = jnp.array([0,1,6,7,12,13,18,19])
    b_p = jnp.array([2,3,4,8,9,10,14,15,16,20,21,22])
    K_l = K_l.at[jnp.ix_(m_p,m_p)].set(K_m)
    K_l = K_l.at[jnp.ix_(b_p,b_p)].set(K_b + K_s)
    K_l = K_l.at[jnp.array([5,11,17,23]), jnp.array([5,11,17,23])].add(1e-7*G*t*jnp.mean(detJ_all))
    area = jnp.sum(detJ_all); m_n = rho*area*t/4.0; Ir=m_n*(t**2)/12.0
    M_local = jnp.diag(jnp.tile(jnp.array([m_n,m_n,m_n,Ir,Ir,Ir*0.01]), 4))
    return K_l, M_local



# ──────────────────────────────────────────────────────────────────────────────
#  ShellFEM Sparse Assembly Optimized
# ──────────────────────────────────────────────────────────────────────────────

class ShellFEM:
    def __init__(self, nodes, trias=None, quads=None, elements=None, num_nodes=None):
        self.nodes = jnp.array(nodes)
        self.num_nodes = num_nodes if num_nodes else self.nodes.shape[0]
        
        # Support ShellFEM(nodes, elements) positional call
        if elements is None and quads is None and trias is not None:
            elements = trias
            trias = None

        if elements is not None:
            self.trias = jnp.array([e for e in elements if len(e) == 3], dtype=jnp.int32).reshape(-1, 3)
            self.quads = jnp.array([e for e in elements if len(e) == 4], dtype=jnp.int32).reshape(-1, 4)
        else:
            self.trias = jnp.array(trias if trias is not None else [], dtype=jnp.int32).reshape(-1, 3)
            self.quads = jnp.array(quads if quads is not None else [], dtype=jnp.int32).reshape(-1, 4)
        self.total_dof = self.num_nodes * 6
        self._cached_indices = None
        self.quad_dof_idx = jnp.array([[6*n+i for n in q for i in range(6)] for q in self.quads], dtype=jnp.int32).reshape(-1, 24)
        self.tria_dof_idx = jnp.array([[6*n+i for n in t for i in range(6)] for t in self.trias], dtype=jnp.int32).reshape(-1, 18)

        self._prepare_assembly_cache()

    def solve_static(self, params, F, fixed_dofs, fixed_vals):
        """Unified static solver interface (defaults to dense partitioned for JIT robustness)."""
        K, _ = self.assemble(params)
        free = jnp.setdiff1d(jnp.arange(self.total_dof), fixed_dofs)
        return self.solve_static_partitioned(K, F, free, fixed_dofs, fixed_vals)

    def _prepare_assembly_cache(self):
        indices = []
        if self.quads.shape[0] > 0:
            I, J = jnp.repeat(jnp.arange(24, dtype=jnp.int32), 24), jnp.tile(jnp.arange(24, dtype=jnp.int32), 24)
            Gi, Gj = self.quad_dof_idx[:, I].flatten(), self.quad_dof_idx[:, J].flatten()
            indices.append(jnp.stack([Gi, Gj], axis=1))
        if self.trias.shape[0] > 0:
            I, J = jnp.repeat(jnp.arange(18, dtype=jnp.int32), 18), jnp.tile(jnp.arange(18, dtype=jnp.int32), 18)
            Gi, Gj = self.tria_dof_idx[:, I].flatten(), self.tria_dof_idx[:, J].flatten()
            indices.append(jnp.stack([Gi, Gj], axis=1))
        if indices: self._cached_indices = jnp.concatenate(indices, axis=0)
        else: self._cached_indices = jnp.zeros((0, 2), dtype=jnp.int32)

    def assemble(self, params, sparse=False):
        n_n = self.num_nodes
        E = jnp.atleast_1d(jnp.array(params.get('E', 210000.0)))
        t = jnp.atleast_1d(jnp.array(params.get('t', 1.0)))
        rho = jnp.atleast_1d(jnp.array(params.get('rho', 7.85e-9)))
        nu = 0.3
        
        # Consistent broadcasting
        if E.size == 1: E = jnp.full(n_n, E[0])
        if t.size == 1: t = jnp.full(n_n, t[0])
        if rho.size == 1: rho = jnp.full(n_n, rho[0])
        
        cur_n = self.nodes
        if 'z' in params:
            z_val = jnp.atleast_1d(jnp.array(params['z']))
            if z_val.size == n_n: cur_n = cur_n.at[:, 2].add(z_val)
            elif z_val.size == 1: cur_n = cur_n.at[:, 2].add(z_val[0])
                
        vk, vm = [], []
        if self.quads.shape[0] > 0:
            ec = cur_n[self.quads]; v12 = ec[:,1,:]-ec[:,0,:]; Lx = jnp.linalg.norm(v12, axis=1, keepdims=True).clip(1e-12)
            e1 = v12/Lx; v14 = ec[:,3,:]-ec[:,0,:]; nrm = jnp.cross(e1, v14)
            e3 = nrm/jnp.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-12); e2 = jnp.cross(e3, e1)
            Ts = jnp.stack([e1,e2,e3], 2); b_a = jnp.sum(v14*e2, axis=1)/2.0
            
            def oq(Eq_n, tq_n, rq_n, p2d, T3):
                Eq, tq, rq = jnp.mean(Eq_n), jnp.mean(tq_n), jnp.mean(rq_n)
                K_l, M_l = compute_mitc4_local_fast(Eq, tq, nu, rq, p2d)
                
                # Robust block construction for the transformation matrix Te (24x24)
                z3 = jnp.zeros((3,3))
                Tn = jnp.block([[T3, z3], [z3, T3]])
                
                # Construct Te by assigning Tn to each nodal block
                Te = jnp.zeros((24, 24))
                for i in range(4):
                    Te = Te.at[6*i:6*i+6, 6*i:6*i+6].set(Tn)
                
                return Te @ K_l @ Te.T, Te @ M_l @ Te.T
            
            p2d_q = jnp.stack([jnp.zeros(len(self.quads)), jnp.zeros(len(self.quads)), Lx[:,0], jnp.zeros(len(self.quads)), Lx[:,0], 2.0*b_a, jnp.zeros(len(self.quads)), 2.0*b_a], 1).reshape(-1,4,2)
            Kq, Mq = vmap(oq)(E[self.quads], t[self.quads], rho[self.quads], p2d_q, Ts)
            vk.append(Kq.flatten()); vm.append(Mq.flatten())
            
        if self.trias.shape[0] > 0:
            ec = cur_n[self.trias]; v12 = ec[:,1,:]-ec[:,0,:]; e1 = v12/jnp.linalg.norm(v12, axis=1, keepdims=True).clip(1e-12)
            v13 = ec[:,2,:]-ec[:,0,:]; nrm = jnp.cross(e1, v13); e3 = nrm/jnp.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-12); e2 = jnp.cross(e3, e1)
            Ts = jnp.stack([e1,e2,e3], 2); o = ec[:,0,:]
            lx1, ly1 = jnp.sum((ec[:,1,:]-o)*e1, 1), jnp.sum((ec[:,1,:]-o)*e2, 1); lx2, ly2 = jnp.sum((ec[:,2,:]-o)*e1, 1), jnp.sum((ec[:,2,:]-o)*e2, 1)
            x2d, y2d = jnp.stack([jnp.zeros(len(self.trias)), lx1, lx2], 1), jnp.stack([jnp.zeros(len(self.trias)), ly1, ly2], 1)
            
            def ot(Et_n, tt_n, rt_n, x, y, T3):
                Et, tt, rt = jnp.mean(Et_n), jnp.mean(tt_n), jnp.mean(rt_n)
                K_l, M_l = compute_mitc3_local(Et, tt, nu, rt, x, y)
                z3 = jnp.zeros((3,3)); Tn = jnp.block([[T3, z3], [z3, T3]])
                Te = jsl.block_diag(Tn, Tn, Tn)
                return Te @ K_l @ Te.T, Te @ M_l @ Te.T
                
            Kt, Mt = vmap(ot)(E[self.trias], t[self.trias], rho[self.trias], x2d, y2d, Ts)
            vk.append(Kt.flatten()); vm.append(Mt.flatten())

        all_vk, all_vm = jnp.concatenate(vk), jnp.concatenate(vm)
        from jax.experimental.sparse import BCOO
        Kg = BCOO((all_vk, self._cached_indices), shape=(self.total_dof, self.total_dof)).todense()
        Mg = BCOO((all_vm, self._cached_indices), shape=(self.total_dof, self.total_dof)).todense()
        
        if sparse:
             from scipy.sparse import csr_matrix
             return csr_matrix((np.array(all_vk), (np.array(self._cached_indices[:,0]), np.array(self._cached_indices[:,1]))), (self.total_dof, self.total_dof)), \
                    csr_matrix((np.array(all_vm), (np.array(self._cached_indices[:,0]), np.array(self._cached_indices[:,1]))), (self.total_dof, self.total_dof))
        return Kg, Mg

    def solve_eigen(self, K, M, num_modes=10, num_skip=0):
        md = jnp.maximum(jnp.diag(M), 1e-15); mis = 1.0/jnp.sqrt(md)
        A = ((K+K.T)/2.0) * mis[:,None] * mis[None,:]
        vals, vecs = safe_eigh(A); freqs = jnp.sqrt(jnp.maximum(vals, 0.0))/(2*jnp.pi); vp = vecs * mis[:,None]
        return freqs[num_skip:num_skip+num_modes], vp[:, num_skip:num_skip+num_modes]

    def solve_eigen_sparse(self, K, M, num_modes=15, num_skip=0):
        if hasattr(K, 'toarray'): Kd, Md = jnp.array(K.toarray()), jnp.array(M.toarray())
        else: Kd, Md = K, M
        return self.solve_eigen(Kd, Md, num_modes=num_modes, num_skip=num_skip)


    def compute_field_results(self, u, params):
        u_f, n_f, nu = jnp.array(u).flatten(), jnp.array(self.nodes), 0.3
        E_n, t_n = params.get('E', 210000.0), params.get('t', 1.0)
        n_t, n_q = self.trias.shape[0], self.quads.shape[0]
        curv_t = recover_curvature_tria_bending(u_f, n_f, self.trias, 1, nu, 1) if n_t>0 else jnp.zeros((0,3))
        eps_t = recover_stress_tria_membrane(u_f, n_f, self.trias, 1, nu) if n_t>0 else jnp.zeros((0,3))
        curv_q = recover_curvature_quad_bending(u_f, n_f, self.quads, 1, nu, 1) if n_q>0 else jnp.zeros((0,3))
        eps_q = recover_stress_quad_membrane(u_f, n_f, self.quads, 1, nu) if n_q>0 else jnp.zeros((0,3))
        curv_el = jnp.concatenate([curv_t, curv_q]); eps_m_el = jnp.concatenate([eps_t, eps_q])
        t_el = jnp.concatenate([t_n[self.trias].mean(1) if jnp.ndim(t_n)>0 else jnp.full(n_t, t_n), 
                               t_n[self.quads].mean(1) if jnp.ndim(t_n)>0 else jnp.full(n_q, t_n)])
        E_el = jnp.concatenate([E_n[self.trias].mean(1) if jnp.ndim(E_n)>0 else jnp.full(n_t, E_n), 
                               E_n[self.quads].mean(1) if jnp.ndim(E_n)>0 else jnp.full(n_q, E_n)])
        eps_top = eps_m_el + curv_el * (t_el[:,None]/2.0)
        pre, spre = E_el/(1-nu**2), E_el/(2*(1+nu))
        sig_el = jnp.stack([pre*(eps_top[:,0]+nu*eps_top[:,1]), pre*(eps_top[:,1]+nu*eps_top[:,0]), spre*eps_top[:,2]], 1)
        vm_el = jnp.sqrt(jnp.maximum(sig_el[:,0]**2-sig_el[:,0]*sig_el[:,1]+sig_el[:,1]**2+3*sig_el[:,2]**2, 1e-12))
        vm_nodal, count = jnp.zeros(self.num_nodes), jnp.zeros(self.num_nodes)
        if n_t>0:
            ix = self.trias.flatten()
            vm_nodal = vm_nodal.at[ix].add(jnp.repeat(vm_el[:n_t], 3)); count = count.at[ix].add(1)
        if n_q>0:
            ix = self.quads.flatten()
            vm_nodal = vm_nodal.at[ix].add(jnp.repeat(vm_el[n_t:], 4)); count = count.at[ix].add(1)
        return {'stress_vm': vm_nodal/jnp.maximum(count,1), 'stress_vm_el': vm_el,
                'stress_x_el': sig_el[:,0], 'stress_y_el': sig_el[:,1], 'stress_xy_el': sig_el[:,2],
                'strain_x_el': eps_top[:,0], 'strain_y_el': eps_top[:,1], 'strain_xy_el': eps_top[:,2],
                'strain_equiv_nodal': vm_nodal/(jnp.maximum(count,1)*E_n if jnp.ndim(E_n)>0 else E_n), 
                'sed': vm_nodal/jnp.maximum(count,1) * 1e-4}


    def solve_static_partitioned(self, K, F, free, fixed, fv):
        Kff = K[jnp.ix_(free, free)]; rhs = F[free] - K[jnp.ix_(free, fixed)] @ fv
        K_reg = Kff.at[jnp.diag_indices(Kff.shape[0])].add(1e-8)
        uf = jnp.linalg.solve(K_reg, rhs)
        u = jnp.zeros(self.total_dof).at[free].set(uf).at[fixed].set(fv)
        return u

    def solve_static_sparse(self, K, F, free_dofs, fixed_dofs, fixed_vals):
        from scipy.sparse.linalg import spsolve
        fixed_part = K[free_dofs, :][:, fixed_dofs].dot(fixed_vals); rhs = F[free_dofs] - fixed_part
        uf = spsolve(K[free_dofs, :][:, free_dofs], rhs); u = np.zeros(self.total_dof)
        u[free_dofs] = uf; u[fixed_dofs] = fixed_vals
        return u

    def compute_max_surface_stress(self, u, params, field_results=None):
        if field_results is None: field_results = self.compute_field_results(u, params)
        return field_results['stress_vm']
    
    def compute_max_surface_strain(self, u, params, field_results=None):
        if field_results is None: field_results = self.compute_field_results(u, params)
        return field_results.get('strain_x', jnp.zeros(self.num_nodes))

    def compute_strain_energy_density(self, u, params, field_results=None):
        if field_results is None: field_results = self.compute_field_results(u, params)
        return field_results['sed']
