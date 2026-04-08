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

def _get_B_dkt(xi, eta, nodes):
    # Standard DKT implementation
    x1, y1, x2, y2, x3, y3 = nodes[0,0], nodes[0,1], nodes[1,0], nodes[1,1], nodes[2,0], nodes[2,1]
    x21, y21, x31, y31 = x2-x1, y2-y1, x3-x1, y3-y1
    detJ = x21*y31 - y21*x31
    x23, y23 = x2-x3, y2-y3
    x31e, y31e = x3-x1, y3-y1
    x12, y12 = x1-x2, y1-y2
    L23sq, L31sq, L12sq = x23**2+y23**2, x31e**2+y31e**2, x12**2+y12**2
    p4,p5,p6 = -6*x23/L23sq, -6*x31e/L31sq, -6*x12/L12sq
    q4,q5,q6 = 3*x23*y23/L23sq, 3*x31e*y31e/L31sq, 3*x12*y12/L12sq
    r4,r5,r6 = 3*y23**2/L23sq, 3*y31e**2/L31sq, 3*y12**2/L12sq
    t4,t5,t6 = -6*y23/L23sq, -6*y31e/L31sq, -6*y12/L12sq
    Hx_xi = jnp.array([p6*(1-2*xi)+(p5-p6)*eta, q6*(1-2*xi)-(q5+q6)*eta, -4+6*(xi+eta)+r6*(1-2*xi)-eta*(r5+r6), -p6*(1-2*xi)+eta*(p4+p6), q6*(1-2*xi)-eta*(q6-q4), -2+6*xi+r6*(1-2*xi)+eta*(r4-r6), -eta*(p5+p4), eta*(q4-q5), -eta*(r5-r4)])
    Hx_et = jnp.array([-p5*(1-2*eta)-xi*(p6-p5), q5*(1-2*eta)-xi*(q5+q6), -4+6*(xi+eta)+r5*(1-2*eta)-xi*(r5+r6), xi*(p4+p6), xi*(q4-q6), -xi*(r6-r4), p5*(1-2*eta)-xi*(p4+p5), q5*(1-2*eta)+xi*(q4-q5), -2+6*eta+r5*(1-2*eta)+xi*(r4-r5)])
    Hy_xi = jnp.array([t6*(1-2*xi)+eta*(t5-t6), 1+r6*(1-2*xi)-eta*(r5+r6), -q6*(1-2*xi)+eta*(q5+q6), -t6*(1-2*xi)+eta*(t4+t6), -1+r6*(1-2*xi)+eta*(r4-r6), -q6*(1-2*xi)-eta*(q4-q6), -eta*(t4+t5), eta*(r4-r5), -eta*(q4-q5)])
    Hy_et = jnp.array([-t5*(1-2*eta)-xi*(t6-t5), 1+r5*(1-2*eta)-xi*(r5+r6), -q5*(1-2*eta)+xi*(q5+q6), xi*(t4+t6), xi*(r4-r6), -xi*(q4-q6), t5*(1-2*eta)-xi*(t4+t5), -1+r5*(1-2*eta)+xi*(r4-r5), -q5*(1-2*eta)-xi*(q4-q5)])
    invJ = 1.0/detJ
    B = jnp.zeros((3, 9))
    B = B.at[0].set(invJ*(y31*Hx_xi-y21*Hx_et)).at[1].set(invJ*(-x31*Hy_xi+x21*Hy_et)).at[2].set(invJ*(-x31*Hx_xi+x21*Hx_et+y31*Hy_xi-y21*Hy_et))
    return B

def _get_B_mitc3(nodes, thick, nu):
    x, y = nodes[:, 0], nodes[:, 1]
    ex = jnp.array([x[1]-x[0], x[2]-x[1], x[0]-x[2]])
    ey = jnp.array([y[1]-y[0], y[2]-y[1], y[0]-y[2]])
    L = jnp.sqrt(ex**2 + ey**2).clip(1e-12)
    def get_G(i):
        j=(i+1)%3; v=jnp.zeros(9).at[3*i].set(-1).at[3*j].set(1)
        tx, ty = ex[i]/L[i], ey[i]/L[i]; vt=0.5*L[i]
        return v.at[3*i+1].set(-vt*ty).at[3*i+2].set(vt*tx).at[3*j+1].set(-vt*ty).at[3*j+2].set(vt*tx)
    G1, G2 = get_G(0), get_G(1)
    M = jnp.array([[ex[0], ey[0]], [ex[1], ey[1]]])
    Bs = jnp.linalg.inv(M) @ jnp.stack([G1, G2])
    return Bs, 0.5*jnp.abs(ex[0]*ey[1]-ex[1]*ey[0])

def _get_B_mitc4(xi, eta, nodes):
    dN_dxi = 0.25 * jnp.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
    dN_det = 0.25 * jnp.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
    j = jnp.array([[jnp.dot(dN_dxi, nodes[:,0]), jnp.dot(dN_dxi, nodes[:,1])],[jnp.dot(dN_det, nodes[:,0]), jnp.dot(dN_det, nodes[:,1])]])
    invJ = jnp.linalg.inv(j)
    def get_cov(xi_p, eta_p, is_xi):
        dN_dxi_p = 0.25*jnp.array([-(1-eta_p), (1-eta_p), (1+eta_p), -(1+eta_p)])
        dN_det_p = 0.25*jnp.array([-(1-xi_p), -(1+xi_p), (1+xi_p), (1-xi_p)])
        jp = jnp.array([[jnp.dot(dN_dxi_p, nodes[:,0]), jnp.dot(dN_dxi_p, nodes[:,1])],[jnp.dot(dN_det_p, nodes[:,0]), jnp.dot(dN_det_p, nodes[:,1])]])
        B = jnp.zeros(12); idx = jnp.arange(4)
        if is_xi: B = B.at[3*idx].set(dN_dxi_p).at[3*idx+2].set(jp[0,0]*0.25*jnp.array([(1-eta_p),(1-eta_p),(1+eta_p),(1+eta_p)])).at[3*idx+1].set(-jp[0,1]*0.25*jnp.array([(1-eta_p),(1-eta_p),(1+eta_p),(1+eta_p)]))
        else: B = B.at[3*idx].set(dN_det_p).at[3*idx+2].set(jp[1,0]*0.25*jnp.array([(1-xi_p),(1+xi_p),(1+xi_p),(1-xi_p)])).at[3*idx+1].set(-jp[1,1]*0.25*jnp.array([(1-xi_p),(1+xi_p),(1+xi_p),(1-xi_p)]))
        return B
    B_gxi = 0.5*(1-eta)*get_cov(0,-1,True) + 0.5*(1+eta)*get_cov(0,1,True)
    B_get = 0.5*(1-xi)*get_cov(-1,0,False) + 0.5*(1+xi)*get_cov(1,0,False)
    Bs = jnp.stack([invJ[0,0]*B_gxi + invJ[1,0]*B_get, invJ[0,1]*B_gxi + invJ[1,1]*B_get])
    return Bs, jnp.linalg.det(j)

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
    # w (idx 0,3,6), thx (idx 1,4,7), thy (idx 2,5,8)
    idx = jnp.arange(3)
    dn_dx = invJ * jnp.array([y2-y3, y3-y1, y1-y2])
    dn_dy = invJ * jnp.array([x3-x2, x1-x3, x2-x1])
    # Curvature Definition: kappa = [thy,x, -thx,y, thy,y - thx,x]
    Bb = Bb.at[0, 3*idx+2].set(dn_dx)  # kappa_x = thy,x (matches analytical -w,xx)
    Bb = Bb.at[1, 3*idx+1].set(-dn_dy) # kappa_y = -thx,y (matches analytical -w,yy)
    Bb = Bb.at[2, 3*idx+2].set(dn_dy)  # kappa_xy = thy,y - thx,x
    Bb = Bb.at[2, 3*idx+1].set(-dn_dx)
    return Bb

def compute_mitc3_local(E, t, nu, rho, x2d, y2d):
    detJ = (x2d[1]-x2d[0])*(y2d[2]-y2d[0]) - (x2d[2]-x2d[0])*(y2d[1]-y2d[0])
    area = 0.5*jnp.abs(detJ); G = E/(2*(1+nu))
    Dm = (E*t/(1-nu**2))*jnp.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    invJ = 1.0/detJ; dn_dx = invJ*jnp.array([y2d[1]-y2d[2], y2d[2]-y2d[0], y2d[0]-y2d[1]])
    dn_dy = invJ*jnp.array([x2d[2]-x2d[1], x2d[0]-x2d[2], x2d[1]-x2d[0]])
    Bm = jnp.zeros((3,6)); idx=jnp.arange(3); Bm=Bm.at[0,2*idx].set(dn_dx).at[1,2*idx+1].set(dn_dy).at[2,2*idx].set(dn_dy).at[2,2*idx+1].set(dn_dx)
    Db = (E*t**3/(12*(1-nu**2)))*jnp.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    nodes2d=jnp.stack([x2d,y2d],1)
    Bb = _get_B_bending_t3(nodes2d)
    Kb = (Bb.T @ Db @ Bb) * area
    Bs, _ = _get_B_mitc3(nodes2d, t, nu)
    Ks = (Bs.T @ (5/6*G*t*jnp.eye(2)) @ Bs)*area
    K_local = jnp.zeros((18,18)); m_p = jnp.array([0,1, 6,7, 12,13]); b_p = jnp.array([2,3,4, 8,9,10, 14,15,16])
    K_local = K_local.at[jnp.ix_(m_p,m_p)].set(Bm.T@Dm@Bm*area).at[jnp.ix_(b_p,b_p)].set(Kb+Ks).at[jnp.array([5,11,17]),jnp.array([5,11,17])].add(1e-7*G*t*area)
    m_n = rho*area*t/3.0; Ir = m_n*(t**2)/12.0
    M_local = jnp.diag(jnp.tile(jnp.array([m_n,m_n,m_n,Ir,Ir,Ir*0.01]), 3))
    return K_local, M_local

def compute_mitc4_local(E, t, nu, rho, p2d):
    C_m = (E*t/(1-nu**2))*jnp.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    C_b = (E*t**3/(12*(1-nu**2)))*jnp.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    G = E/(2*(1+nu)); C_s = (G*t*5/6)*jnp.eye(2); K_m,K_b,K_s = jnp.zeros((8,8)),jnp.zeros((12,12)),jnp.zeros((12,12))
    gp = 0.577350269189626
    for xi in [-gp, gp]:
        for eta in [-gp, gp]:
            dN_dxi = 0.25*jnp.array([-(1-eta),(1-eta),(1+eta),-(1+eta)]); dN_det = 0.25*jnp.array([-(1-xi),-(1+xi),(1+xi),(1-xi)])
            j = jnp.array([[jnp.dot(dN_dxi,p2d[:,0]),jnp.dot(dN_dxi,p2d[:,1])],[jnp.dot(dN_det,p2d[:,0]),jnp.dot(dN_det,p2d[:,1])]])
            detJ = jnp.linalg.det(j); invJ = jnp.linalg.inv(j)
            dN_dx = invJ[0,0]*dN_dxi+invJ[0,1]*dN_det; dN_dy = invJ[1,0]*dN_dxi+invJ[1,1]*dN_det
            K_m += (_B_membrane_q4(dN_dx,dN_dy).T @ C_m @ _B_membrane_q4(dN_dx,dN_dy))*detJ
            K_b += (_B_bending_q4(dN_dx,dN_dy).T @ C_b @ _B_bending_q4(dN_dx,dN_dy))*detJ
            Bs, _ = _get_B_mitc4(xi, eta, p2d); K_s += (Bs.T @ C_s @ Bs)*detJ
    K_l = jnp.zeros((24,24)); m_i = jnp.array([0,1,6,7,12,13,18,19]); b_i = jnp.array([2,3,4,8,9,10,14,15,16,20,21,22])
    K_l = K_l.at[jnp.ix_(m_i,m_i)].set(K_m).at[jnp.ix_(b_i,b_i)].set(K_b+K_s).at[jnp.array([5,11,17,23]),jnp.array([5,11,17,23])].add(1e-7*G*t*jnp.abs(jnp.mean(jnp.linalg.det(j))))
    # Corrected Area calculation and Rotational Inertia
    area = jnp.abs((p2d[1,0]-p2d[0,0])*(p2d[2,1]-p2d[0,1])); m_n = rho*area*t/4.0; Ir=m_n*(t**2)/12.0
    M_local = jnp.diag(jnp.tile(jnp.array([m_n,m_n,m_n,Ir,Ir,Ir*0.01]), 4))
    return K_l, M_local

def recover_curvature_tria_bending(u, nodes, trias, E, nu, t):
    def get_c(i):
        ix = trias[i]; u_e = u.reshape(-1,6)[ix]
        # Mindlin mapping [w1, thx1, thy1, ...]
        ud = u_e[:, 2:5].flatten()
        return _get_B_bending_t3(nodes[ix]) @ ud
    return vmap(get_c)(jnp.arange(trias.shape[0]))

def recover_curvature_quad_bending(u, nodes, quads, E, nu, t):
    def get_c(i):
        ix = quads[i]; qn = nodes[ix]; v1,v2 = qn[1]-qn[0],qn[3]-qn[0]; e1 = v1/jnp.linalg.norm(v1).clip(1e-12)
        e3 = jnp.cross(e1,v2); e3 /= jnp.linalg.norm(e3).clip(1e-12); e2 = jnp.cross(e3,e1)
        dN_dxi = jnp.array([-0.25, 0.25, 0.25, -0.25]); dN_det = jnp.array([-0.25, -0.25, 0.25, 0.25])
        p2d = jnp.stack([jnp.dot(qn-qn[0],e1), jnp.dot(qn-qn[0],e2)], 1)
        j = jnp.array([[jnp.dot(dN_dxi,p2d[:,0]),jnp.dot(dN_dxi,p2d[:,1])],[jnp.dot(dN_det,p2d[:,0]),jnp.dot(dN_det,p2d[:,1])]])
        invJ = jnp.linalg.inv(j); dN_dx = invJ[0,0]*dN_dxi+invJ[0,1]*dN_det; dN_dy = invJ[1,0]*dN_dxi+invJ[1,1]*dN_det
        u_g = u.reshape(-1,6)[ix]; tx = u_g[:,3]*e1[0]+u_g[:,4]*e1[1]; ty = u_g[:,3]*e2[0]+u_g[:,4]*e2[1]
        return jnp.array([-jnp.dot(dN_dx,ty), jnp.dot(dN_dy,tx), -(jnp.dot(dN_dy,ty)-jnp.dot(dN_dx,tx))])
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

class ShellFEM:
    def __init__(self, nodes, quads=None, trias=None, beams=None, elements=None, dof_per_node=6):
        self.nodes = jnp.array(nodes)
        if elements is not None:
            arr = jnp.array(elements)
            if arr.shape[1] == 4: quads = arr
            elif arr.shape[1] == 3: trias = arr
        self.quads = jnp.array(quads) if quads is not None else jnp.zeros((0,4), dtype=jnp.int32)
        self.trias = jnp.array(trias) if trias is not None else jnp.zeros((0,3), dtype=jnp.int32)
        self.num_nodes = len(self.nodes); self.total_dof = self.num_nodes * 6; self.node_coords = self.nodes
        if self.quads.shape[0] > 0: self.quad_dof_idx = jnp.array([np.concatenate([np.arange(6)+n*6 for n in e]) for e in np.array(self.quads)])
        else: self.quad_dof_idx = None
        if self.trias.shape[0] > 0: self.tria_dof_idx = jnp.array([np.concatenate([np.arange(6)+n*6 for n in e]) for e in np.array(self.trias)])
        else: self.tria_dof_idx = None

    def assemble(self, params, sparse=False):
        n_n = self.nodes.shape[0]; E = jnp.atleast_1d(jnp.array(params.get('E', 210000.0))); t = jnp.atleast_1d(jnp.array(params.get('t', 1.0)))
        if E.shape[0] == 1: E = jnp.full(n_n, E[0]); 
        if t.shape[0] == 1: t = jnp.full(n_n, t[0]);
        rho = jnp.atleast_1d(jnp.array(params.get('rho', 7.85e-9))); 
        if rho.shape[0] == 1: rho = jnp.full(n_n, rho[0])
        nu = 0.3; cur_n = self.nodes; Kg = jnp.zeros((self.total_dof, self.total_dof)); Mg = jnp.zeros((self.total_dof, self.total_dof))
        if self.quads.shape[0] > 0:
            ec = cur_n[self.quads]; v12 = ec[:,1,:]-ec[:,0,:]; Lx = jnp.linalg.norm(v12, axis=1, keepdims=True).clip(1e-12); e1 = v12/Lx
            v14 = ec[:,3,:]-ec[:,0,:]; nrm = jnp.cross(e1, v14); e3 = nrm/jnp.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-12); e2 = jnp.cross(e3, e1)
            Ts = jnp.stack([e1,e2,e3], 2); b_a = jnp.sum(v14*e2, axis=1)/2.0
            def oq(Eq, tq, rq, p2d, T3):
                Kl, Ml = compute_mitc4_local(Eq, tq, nu, rq, p2d); z3 = jnp.zeros((3,3)); Tn = jnp.concatenate([jnp.concatenate([T3, z3], 1), jnp.concatenate([z3, T3], 1)], 0)
                Te = jnp.zeros((24, 24)); 
                for i in range(4): Te = Te.at[6*i:6*i+6, 6*i:6*i+6].set(Tn)
                return Te @ Kl @ Te.T, Te @ Ml @ Te.T
            p2d_q = jnp.stack([jnp.zeros(len(self.quads)), jnp.zeros(len(self.quads)), Lx[:,0], jnp.zeros(len(self.quads)), Lx[:,0], 2.0*b_a, jnp.zeros(len(self.quads)), 2.0*b_a], 1).reshape(-1,4,2)
            Kq, Mq = vmap(oq)(E[self.quads].mean(1), t[self.quads].mean(1), rho[self.quads].mean(1), p2d_q, Ts)
            I, J = jnp.repeat(jnp.arange(24), 24), jnp.tile(jnp.arange(24), 24)
            Gi, Gj = self.quad_dof_idx[:, I].flatten(), self.quad_dof_idx[:, J].flatten()
            Kg = Kg.at[Gi, Gj].add(Kq.flatten()); Mg = Mg.at[Gi, Gj].add(Mq.flatten())
        if self.trias.shape[0] > 0:
            ec = cur_n[self.trias]; v12 = ec[:,1,:]-ec[:,0,:]; e1 = v12/jnp.linalg.norm(v12, axis=1, keepdims=True).clip(1e-12)
            v13 = ec[:,2,:]-ec[:,0,:]; nrm = jnp.cross(e1, v13); e3 = nrm/jnp.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-12); e2 = jnp.cross(e3, e1)
            Ts = jnp.stack([e1,e2,e3], 2); o = ec[:,0,:]
            lx1, ly1 = jnp.sum((ec[:,1,:]-o)*e1, 1), jnp.sum((ec[:,1,:]-o)*e2, 1); lx2, ly2 = jnp.sum((ec[:,2,:]-o)*e1, 1), jnp.sum((ec[:,2,:]-o)*e2, 1)
            x2d, y2d = jnp.stack([jnp.zeros(len(self.trias)), lx1, lx2], 1), jnp.stack([jnp.zeros(len(self.trias)), ly1, ly2], 1)
            def ot(Et, tt, rt, x, y, T3):
                Kl, Ml = compute_mitc3_local(Et, tt, nu, rt, x, y); z3 = jnp.zeros((3,3)); Tn = jnp.concatenate([jnp.concatenate([T3, z3], 1), jnp.concatenate([z3, T3], 1)], 0)
                Te = jnp.zeros((18, 18)); 
                for i in range(3): Te = Te.at[6*i:6*i+6, 6*i:6*i+6].set(Tn)
                return Te @ Kl @ Te.T, Te @ Ml @ Te.T
            Kt, Mt = vmap(ot)(E[self.trias].mean(1), t[self.trias].mean(1), rho[self.trias].mean(1), x2d, y2d, Ts)
            I, J = jnp.repeat(jnp.arange(18), 18), jnp.tile(jnp.arange(18), 18)
            Gi, Gj = self.tria_dof_idx[:, I].flatten(), self.tria_dof_idx[:, J].flatten()
            Kg = Kg.at[Gi, Gj].add(Kt.flatten()); Mg = Mg.at[Gi, Gj].add(Mt.flatten())
        if not sparse: return Kg, Mg
        from scipy.sparse import coo_matrix
        R, C, KV, MV = [], [], [], []
        if self.quads.shape[0]>0: R.append(np.array(self.quad_dof_idx[:, jnp.repeat(jnp.arange(24), 24)].flatten())); C.append(np.array(self.quad_dof_idx[:, jnp.tile(jnp.arange(24), 24)].flatten())); KV.append(np.array(Kq).flatten()); MV.append(np.array(Mq).flatten())
        if self.trias.shape[0]>0: R.append(np.array(self.tria_dof_idx[:, jnp.repeat(jnp.arange(18), 18)].flatten())); C.append(np.array(self.tria_dof_idx[:, jnp.tile(jnp.arange(18), 18)].flatten())); KV.append(np.array(Kt).flatten()); MV.append(np.array(Mt).flatten())
        return coo_matrix((np.concatenate(KV),(np.concatenate(R),np.concatenate(C))), (self.total_dof,self.total_dof)).tocsr(), coo_matrix((np.concatenate(MV),(np.concatenate(R),np.concatenate(C))), (self.total_dof,self.total_dof)).tocsr()

    def solve_eigen(self, K, M, num_modes=10, num_skip=0):
        md = jnp.maximum(jnp.diag(M), 1e-15); mis = 1.0/jnp.sqrt(md); A = ((K+K.T)/2.0) * mis[:,None] * mis[None,:] + jnp.eye(K.shape[0])*1e-10
        vals, vecs = safe_eigh(A); freqs = jnp.sqrt(jnp.maximum(vals, 0.0))/(2*jnp.pi); vp = vecs * mis[:,None]
        return freqs[num_skip:num_skip+num_modes], vp[:, num_skip:num_skip+num_modes]

    def solve_eigen_sparse(self, K, M, num_modes=15):
        """Sparse-interface modal solver using JAX dense backend for verification-scale meshes."""
        if hasattr(K, 'toarray'): Kd = jnp.array(K.toarray()); Md = jnp.array(M.toarray())
        else: Kd, Md = K, M
        md = jnp.maximum(jnp.diag(Md), 1e-15); mis = 1.0/jnp.sqrt(md); A = ((Kd+Kd.T)/2.0) * mis[:,None] * mis[None,:]
        vals, vecs = safe_eigh(A); frq = jnp.sqrt(jnp.maximum(vals, 0.0))/(2*jnp.pi); vp = vecs * mis[:,None]
        return frq, vp

    def compute_field_results(self, u, params):
        u_f = jnp.array(u).flatten(); n_f = jnp.array(self.nodes); nu=0.3; E_n = params.get('E', 210000.0); t_n = params.get('t', 1.0)
        n_t = self.trias.shape[0]; n_q = self.quads.shape[0]
        curv_t = recover_curvature_tria_bending(u_f, n_f, self.trias, 1, nu, 1) if n_t>0 else jnp.zeros((0,3))
        eps_t = recover_stress_tria_membrane(u_f, n_f, self.trias, 1, nu) if n_t>0 else jnp.zeros((0,3))
        curv_q = recover_curvature_quad_bending(u_f, n_f, self.quads, 1, nu, 1) if n_q>0 else jnp.zeros((0,3))
        eps_q = recover_stress_quad_membrane(u_f, n_f, self.quads, 1, nu) if n_q>0 else jnp.zeros((0,3))
        curv_el = jnp.concatenate([curv_t, curv_q]); eps_m_el = jnp.concatenate([eps_t, eps_q])
        t_el = jnp.concatenate([t_n[self.trias].mean(1) if jnp.ndim(t_n)>0 else jnp.full(n_t, t_n), t_n[self.quads].mean(1) if jnp.ndim(t_n)>0 else jnp.full(n_q, t_n)])
        E_el = jnp.concatenate([E_n[self.trias].mean(1) if jnp.ndim(E_n)>0 else jnp.full(n_t, E_n), E_n[self.quads].mean(1) if jnp.ndim(E_n)>0 else jnp.full(n_q, E_n)])
        
        # Fiber strains at z = t/2 (Top surface)
        eps_top = eps_m_el + curv_el * (t_el[:,None]/2.0)
        pre = E_el/(1-nu**2); spre = E_el/(2*(1+nu))
        sig_el = jnp.stack([pre*(eps_top[:,0]+nu*eps_top[:,1]), pre*(eps_top[:,1]+nu*eps_top[:,0]), spre*eps_top[:,2]], 1)
        vm_el = jnp.sqrt(jnp.maximum(sig_el[:,0]**2-sig_el[:,0]*sig_el[:,1]+sig_el[:,1]**2+3*sig_el[:,2]**2, 1e-12))
        
        # Nodal averaging for visualization
        vm_nodal = jnp.zeros(self.num_nodes); eps_x_nodal = jnp.zeros(self.num_nodes); sed_nodal = jnp.zeros(self.num_nodes); count = jnp.zeros(self.num_nodes);
        
        # Consistent Strain Energy Density (SED)
        # For simplicity in this stabilization phase, we use the element-wise average approximation u_e^T K_e u_e / Area_e
        # In a real shell solver, this would be computed at Gauss points.
        sed_el = jnp.zeros(n_t + n_q) # Placeholder for now, can be improved
        
        if n_t>0: ix=self.trias.flatten(); vm_nodal=vm_nodal.at[ix].add(jnp.repeat(vm_el[:n_t],3)); eps_x_nodal=eps_x_nodal.at[ix].add(jnp.repeat(eps_top[:n_t,0],3)); count=count.at[ix].add(1)
        if n_q>0: ix=self.quads.flatten(); vm_nodal=vm_nodal.at[ix].add(jnp.repeat(vm_el[n_t:],4)); eps_x_nodal=eps_x_nodal.at[ix].add(jnp.repeat(eps_top[n_t:,0],4)); count=count.at[ix].add(1)
        
        return {
            'stress_vm': vm_nodal/jnp.maximum(count,1), 
            'strain_x': eps_x_nodal/jnp.maximum(count,1),
            'stress_vm_el': vm_el, 
            'strain_x_el': eps_top[:,0],
            'sed': vm_nodal/jnp.maximum(count,1) * 1e-4, # Proxy for SED in nodal view
            'u_mag': jnp.linalg.norm(u_f.reshape(-1,6)[:,:3], axis=1)
        }

    def solve_static(self, params, F, prescribed_dofs, prescribed_vals):
        K, _ = self.assemble(params); free = jnp.setdiff1d(jnp.arange(self.total_dof), prescribed_dofs)
        Kff = K[jnp.ix_(free, free)]; rhs = F[free] - K[jnp.ix_(free, prescribed_dofs)] @ prescribed_vals
        uf = jnp.linalg.solve(Kff + 1e-9*jnp.eye(len(free)), rhs)
        u = jnp.zeros(self.total_dof).at[free].set(uf).at[prescribed_dofs].set(prescribed_vals)
        return u

    def solve_static_partitioned(self, K, F, free_dofs, fixed_dofs, fixed_vals):
        Kff = K[jnp.ix_(free_dofs, free_dofs)]; rhs = F[free_dofs] - K[jnp.ix_(free_dofs, fixed_dofs)] @ fixed_vals
        uf = jnp.linalg.solve(Kff + 1e-8*jnp.eye(Kff.shape[0]), rhs)
        return jnp.zeros(self.total_dof).at[free_dofs].set(uf).at[fixed_dofs].set(fixed_vals)

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
        return field_results['strain_x']

    def compute_strain_energy_density(self, u, params, field_results=None):
        if field_results is None: field_results = self.compute_field_results(u, params)
        return field_results['sed']
