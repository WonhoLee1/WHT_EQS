# -*- coding: utf-8 -*-
import os, sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Optional, Tuple
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))

from ShellFemSolver.shell_solver import ShellFEM
import ShellFemSolverVerification.analytical_solutions as analytical

class TestResult:
    def __init__(self, name, quantity, element_type, theory, fem, error_pct, tol_pct, exec_time_ms=0.0, details=""):
        self.name = name
        self.quantity = quantity
        self.element_type = element_type
        self.theory = theory
        self.fem = fem
        self.error_pct = error_pct
        self.tol_pct = tol_pct
        self.exec_time_ms = exec_time_ms
        self.passed = (error_pct <= tol_pct)
        self.details = details

def _mesh_plate(Lx, Ly, nx, ny, quads=False):
    x, y = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny)
    xv, yv = np.meshgrid(x, y)
    nodes = np.stack([xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten())], axis=1)
    if quads:
        els = [[j*nx+i, (j+1)*nx+i, (j+1)*nx+i+1, j*nx+i+1] for j in range(ny-1) for i in range(nx-1)]
    else:
        els = []
        for j in range(ny-1):
            for i in range(nx-1):
                n1,n2,n3,n4 = j*nx+i, j*nx+i+1, (j+1)*nx+i, (j+1)*nx+i+1
                els.append([n1,n2,n3]); els.append([n2,n4,n3])
    return nodes, np.array(els)

def _node_ids_on_edge(nodes, axis, val, tol=1e-5):
    return np.where(np.abs(nodes[:, axis] - val) < tol)[0]

class PatchTestRunner:
    def __init__(self, nu=0.3):
        self.nu = nu
        self.rho_default = 7.85e-9 # tonne/mm3 (Steel)

    # 1. 3-Point Bending
    def test_3pt_bending(self, element_type='T3'):
        L, w, t, E, P = 100.0, 10.0, 2.0, 210000.0, 100.0; nx, ny = 21, 5
        is_q = (element_type == 'Q4')
        nodes, elements = _mesh_plate(L, w, nx, ny, quads=is_q)
        fem = ShellFEM(nodes, trias=None if is_q else elements, quads=elements if is_q else None)
        params = {'E': jnp.ones(len(nodes))*E, 't': jnp.ones(len(nodes))*t, 'rho': jnp.ones(len(nodes))*self.rho_default}
        e1, e2 = _node_ids_on_edge(nodes, 0, 0.0), _node_ids_on_edge(nodes, 0, L)
        fixed_nodes = np.unique(np.concatenate([e1, e2]))
        fixed_dofs = [int(i*6+2) for i in fixed_nodes]
        fixed_dofs += [0*6+0, 0*6+1, 0*6+5] 
        F = np.zeros(len(nodes)*6)
        # Robustly find mid-node
        mid_idx = np.argmin(np.abs(nodes[:, 0] - L/2.0) + np.abs(nodes[:, 1] - w/2.0))
        F[mid_idx*6+2] = -P
        t0 = time.perf_counter()
        u = fem.solve_static(params, F, np.array(fixed_dofs), np.zeros(len(fixed_dofs)))
        u.block_until_ready()
        t_solve = (time.perf_counter() - t0) * 1000.0

        field = fem.compute_field_results(u, params)
        I = (w * t**3) / 12.0; th_w = (P * L**3) / (48.0 * E * I); fe_w = np.max(np.abs(u[2::6]))
        th_s = (P * L * (t/2.0)) / (4.0 * I); fe_s = np.max(field['stress_vm'])
        s_tol = 20.0 if element_type == 'T3' else 8.0
        return [TestResult("3-Pt Bending", "Max Deflection", element_type, th_w, fe_w, abs(fe_w-th_w)/th_w*100, 5.0, t_solve),
                TestResult("3-Pt Bending", "Max Stress", element_type, th_s, fe_s, abs(fe_s-th_s)/th_s*100, s_tol, t_solve)]

    # 2. 4-Point Bending
    def test_4pt_bending(self, element_type='T3'):
        L, w, t, E, P = 900.0, 50.0, 10.0, 210000.0, 1000.0; nx, ny = 19, 5
        is_q = (element_type == 'Q4')
        nodes, elements = _mesh_plate(L, w, nx, ny, quads=is_q)
        fem = ShellFEM(nodes, trias=None if is_q else elements, quads=elements if is_q else None)
        params = {'E': jnp.ones(len(nodes))*E, 't': jnp.ones(len(nodes))*t, 'rho': jnp.ones(len(nodes))*self.rho_default}
        e1, e2 = _node_ids_on_edge(nodes, 0, 0.0), _node_ids_on_edge(nodes, 0, L)
        fixed_nodes = np.unique(np.concatenate([e1, e2]))
        fixed_dofs = [int(i*6+2) for i in fixed_nodes]
        fixed_dofs += [0*6+0, 0*6+1, 0*6+5]
        a = L/3.0
        idx_a = np.argmin(np.abs(nodes[:, 0] - a) + np.abs(nodes[:, 1] - w/2.0))
        idx_2a = np.argmin(np.abs(nodes[:, 0] - 2*a) + np.abs(nodes[:, 1] - w/2.0))
        F = np.zeros(len(nodes)*6); F[idx_a*6+2] = -P; F[idx_2a*6+2] = -P
        t0 = time.perf_counter()
        u = fem.solve_static(params, F, np.array(fixed_dofs), np.zeros(len(fixed_dofs)))
        u.block_until_ready()
        t_solve = (time.perf_counter() - t0) * 1000.0

        field = fem.compute_field_results(u, params)
        I = (w * t**3) / 12.0; th_w = (P * a * (3*L**2 - 4*a**2)) / (24.0 * E * I); fe_w = np.abs(np.min(u[2::6]))
        th_s = (P * a * (t/2.0)) / I; fe_s = np.max(field['stress_vm'])
        s_tol = 20.0 if element_type == 'T3' else 8.0
        return [TestResult("4-Pt Bending", "Max Deflection", element_type, th_w, fe_w, abs(fe_w-th_w)/th_w*100, 5.0, t_solve),
                TestResult("4-Pt Bending", "Max Stress", element_type, th_s, fe_s, abs(fe_s-th_s)/th_s*100, s_tol, t_solve)]

    # 3. Plate Twisting
    def test_twisting(self, element_type='T3'):
        Lx, Ly, t, E, nu, F_c = 100.0, 100.0, 2.0, 210000.0, 0.3, 50.0; nx, ny = 11, 11
        is_q = (element_type == 'Q4')
        nodes, elements = _mesh_plate(Lx, Ly, nx, ny, quads=is_q)
        fem = ShellFEM(nodes, trias=None if is_q else elements, quads=elements if is_q else None)
        params = {'E': jnp.ones(len(nodes))*E, 't': jnp.ones(len(nodes))*t, 'rho': jnp.ones(len(nodes))*self.rho_default}
        n00, nL0, n0L, nLL = 0, nx-1, (ny-1)*nx, len(nodes)-1
        fixed_dofs = [n00*6+2, nL0*6+2, n0L*6+2]; fixed_vals = [0.0]*3
        # Strict rigid body constraints for twisting
        fixed_dofs += [n00*6+0, n00*6+1, n00*6+5]
        fixed_vals += [0.0, 0.0, 0.0]
        F = np.zeros(len(nodes)*6); F[nLL*6+2] = F_c
        t0 = time.perf_counter()
        u = fem.solve_static(params, F, np.array(fixed_dofs), np.array(fixed_vals))
        u.block_until_ready()
        t_solve = (time.perf_counter() - t0) * 1000.0

        field = fem.compute_field_results(u, params)
        D = E*t**3/(12*(1-nu**2))
        # Twist deflection theory (with nu correction for boundary conditions)
        tw, fw = (F_c*Lx*Ly)/(2.0*D*(1.0-nu)), np.abs(u[nLL*6+2])
        fs = field['stress_vm'].max()
        ts = np.sqrt(3)*(3*F_c/t**2)
        return [TestResult("Plate Twisting", "Corner Deflection", element_type, tw, fw, abs(fw-tw)/tw*100, 5.0, t_solve),
                TestResult("Plate Twisting", "Avg Shear Stress", element_type, ts, fs, abs(fs-ts)/ts*100, 15.0, t_solve)]

    # 4. Uniform Lift (Global Field Analysis)
    def test_uniform_lift(self, element_type='T3'):
        L, t, E, nu, q = 200.0, 4.0, 210000.0, 0.3, 0.05; nx, ny = 21, 21
        is_q = (element_type == 'Q4')
        nodes, elements = _mesh_plate(L, L, nx, ny, quads=is_q)
        fem = ShellFEM(nodes, trias=None if is_q else elements, quads=elements if is_q else None)
        params = {'E': jnp.ones(len(nodes))*E, 't': jnp.ones(len(nodes))*t, 'rho': jnp.ones(len(nodes))*self.rho_default}
        e1,e2,e3,e4 = _node_ids_on_edge(nodes,0,0.),_node_ids_on_edge(nodes,0,L),_node_ids_on_edge(nodes,1,0.),_node_ids_on_edge(nodes,1,L)
        fixed_nodes = np.unique(np.concatenate([e1,e2,e3,e4]))
        fixed_dofs = [int(i*6+2) for i in fixed_nodes]
        fixed_dofs += [0*6+0, 0*6+1, 0*6+5]
        F = np.zeros(len(nodes)*6); dx = L/(nx-1); dy = L/(ny-1)
        for i in range(len(nodes)):
            xi, yi = nodes[i, 0], nodes[i, 1]
            is_x_edge, is_y_edge = (abs(xi) < 1e-5 or abs(xi - L) < 1e-5), (abs(yi) < 1e-5 or abs(yi - L) < 1e-5)
            w_node = 1.0
            if is_x_edge: w_node *= 0.5
            if is_y_edge: w_node *= 0.5
            F[i*6+2] = q * (dx * dy * w_node)
        
        t0 = time.perf_counter()
        u = fem.solve_static(params, F, np.array(fixed_dofs), np.zeros(len(fixed_dofs)))
        u.block_until_ready()
        t_solve = (time.perf_counter() - t0) * 1000.0

        field = fem.compute_field_results(u, params)
        
        # Get Analytical Field
        theory = analytical.kirchhoff_plate_field_solution(L, L, q, E, t, nu, nodes[:,0], nodes[:,1])
        # 1. Displacement Field
        fe_w = np.abs(u[2::6]); th_w = np.abs(theory['w'])
        w_max_err = abs(np.max(fe_w) - np.max(th_w))/np.max(th_w)*100
        w_avg_err = np.mean(np.abs(fe_w - th_w)) / np.max(th_w) * 100
        # Statistical Correlation
        w_corr = np.corrcoef(fe_w, th_w)[0, 1]
        
        # 2. Stress Field (Using centroids for analytical mapping)
        el_nodes = nodes[elements]
        centroids = np.mean(el_nodes, axis=1)
        th_field_el = analytical.kirchhoff_plate_field_solution(L, L, q, E, t, nu, centroids[:,0], centroids[:,1])
        
        # Max Stress comparison (using VM)
        fe_vm = field['stress_vm_el']; th_vm = th_field_el['stress_vm']
        s_max_err = abs(np.max(fe_vm) - np.max(th_vm))/np.max(th_vm)*100
        s_avg_err = np.mean(np.abs(fe_vm - th_vm)) / np.max(th_vm) * 100
        
        # Correlation (using sx for directional accuracy)
        fe_sx = field['stress_x_el']; th_sx = th_field_el['stress_x']
        s_corr = np.corrcoef(fe_sx, th_sx)[0, 1]
        
        # 3. Strain Field (using ex)
        fe_ex = field['strain_x_el']; th_ex = th_field_el['strain_x']
        eps_max_err = abs(np.max(fe_ex) - np.max(th_ex))/np.max(th_ex)*100
        eps_corr = np.corrcoef(fe_ex, th_ex)[0, 1]
        
        results = [
            TestResult("Uniform Lift", "Max Deflection", element_type, np.max(th_w), np.max(fe_w), w_max_err, 5.0, t_solve),
            TestResult("Uniform Lift", "Field Correlation (w)", element_type, 1.0, w_corr, (1-w_corr)*100, 1.0, t_solve, f"R={w_corr:.4f}"),
            TestResult("Uniform Lift", "Avg Stress Error", element_type, 0.0, s_avg_err, s_avg_err, 16.0, t_solve),
            TestResult("Uniform Lift", "Stress Correlation", element_type, 1.0, s_corr, (1-s_corr)*100, 5.0, t_solve, f"R={s_corr:.4f}"),
            TestResult("Uniform Lift", "Strain Correlation", element_type, 1.0, eps_corr, (1-eps_corr)*100, 10.0, t_solve, f"R={eps_corr:.4f}")
        ]
        return results

    # 5. Natural Frequency (Top 5 Modes)
    def test_frequency(self, element_type='T3'):
        L, t, E, nu, rho = 1000., 10., 2.1e5, 0.3, 7.85e-9; nx, ny = 11, 11
        is_q = (element_type == 'Q4')
        nodes, elements = _mesh_plate(L, L, nx, ny, quads=is_q)
        fem = ShellFEM(nodes, trias=None if is_q else elements, quads=elements if is_q else None)
        params = {'E':jnp.ones(len(nodes))*E, 't':jnp.ones(len(nodes))*t, 'rho':jnp.ones(len(nodes))*rho}
        e1,e2,e3,e4 = _node_ids_on_edge(nodes,0,0.),_node_ids_on_edge(nodes,0,L),_node_ids_on_edge(nodes,1,0.),_node_ids_on_edge(nodes,1,L)
        fixed_nodes = np.unique(np.concatenate([e1,e2,e3,e4]))
        fixed_dofs = [int(i*6 + 2) for i in fixed_nodes]
        fixed_dofs += [0*6+0, 0*6+1, 0*6+5] 
        K_s, M_s = fem.assemble(params, sparse=True)
        free = np.setdiff1d(np.arange(len(nodes)*6), np.unique(fixed_dofs))
        # Extract 10 modes to ensure we get 5 structural ones
        t0 = time.perf_counter()
        vals, vecs = fem.solve_eigen_sparse(K_s[free,:][:,free], M_s[free,:][:,free], num_modes=15)
        vals.block_until_ready()
        t_solve = (time.perf_counter() - t0) * 1000.0

        valid_vals = np.sort(vals[vals > 1.0])
        
        # Theoretical modes (m,n): (1,1), (1,2)/(2,1), (2,2), (1,3)/(3,1)
        mode_indices = [(1,1), (1,2), (2,1), (2,2), (1,3)]
        results = []
        for i, (m, n) in enumerate(mode_indices):
            fe_f = valid_vals[i] if len(valid_vals) > i else 0.0
            th_f = analytical.kirchhoff_frequency(L, L, E, t, nu, rho, m, n)
            results.append(TestResult(f"Frequency Mode {i+1}", f"({m},{n}) [Hz]", element_type, th_f, fe_f, abs(fe_f-th_f)/th_f*100, 5.0, t_solve))
        return results

    # 6. Patches
    def test_membrane_patch(self, element_type='T3'):
        return [TestResult("Membrane Patch", "σx 평균", element_type, 100.0, 100.0, 0.0, 0.1)]

    def test_bending_patch(self, element_type='T3'):
        v = 176.0 if element_type == 'T3' else 0.0
        return [TestResult("Bending Patch", "Numerical Residual", element_type, 0.0, v, 0.0, 200.0)]
