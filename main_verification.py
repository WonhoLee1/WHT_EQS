# ==============================================================================
# main_verification.py - Equivalent Sheet Model Optimization & Verification
# ==============================================================================
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import optax
import pyvista as pv
from scipy.interpolate import griddata

# Custom Modules
from solver import PlateFEM
from WHT_EQS_pattern_generator import (
    get_thickness_field, 
    get_z_field, 
    get_density_field, 
    get_E_field
)
from WHT_EQS_load_cases import (
    TwistCase, 
    PureBendingCase, 
    CornerLiftCase
)
from WHT_EQS_visualization import (
    stage1_visualize_patterns, 
    stage2_visualize_ground_truth, 
    stage3_visualize_comparison
)

# Enable JAX 64-bit precision
jax.config.update("jax_enable_x64", True)

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
Lx, Ly = 1000.0, 400.0
Nx_high, Ny_high = 50, 20
Nx_low, Ny_low = 20, 10

# ==============================================================================
# MODEL MANAGER
# ==============================================================================
class EquivalentSheetModel:
    def __init__(self, Lx, Ly, Nx, Ny):
        self.fem = PlateFEM(Lx, Ly, Nx, Ny)
        self.cases = []
        self.targets = []
        self.optimized_params = None
        
    def add_case(self, case):
        self.cases.append(case)
        print(f"Added Case: {case.name}")

    def generate_targets(self, resolution_high=(50, 20), num_modes_save=5, target_config=None):
        print("\n" + "="*70)
        print(" GENERATING GROUND TRUTH TARGETS")
        print("="*70)
        
        self.resolution_high = resolution_high
        self.num_modes_truth = num_modes_save
        fem_high = PlateFEM(self.fem.Lx, self.fem.Ly, resolution_high[0], resolution_high[1])
        
        X_grid = fem_high.node_coords[:,0].reshape(resolution_high[0]+1, -1)
        Y_grid = fem_high.node_coords[:,1].reshape(resolution_high[0]+1, -1)
        
        if target_config is None: target_config = {}
        
        # 1. Generate Fields
        t_field = get_thickness_field(X_grid, Y_grid, Lx=Lx, Ly=Ly, 
                                       pattern_str=target_config.get('pattern', 'A'), 
                                       base_t=target_config.get('base_t', 1.0), 
                                       bead_t=target_config.get('bead_t', 2.0))
        
        z_field = get_z_field(X_grid, Y_grid, Lx=Lx, Ly=Ly,
                               pattern_pz=target_config.get('pattern_pz', ''), 
                               pz_dict=target_config.get('bead_pz', {}))
        
        rho_field = get_density_field(X_grid, Y_grid, Lx=Lx, Ly=Ly, 
                                       base_rho=target_config.get('base_rho', 7.5e-9))
        
        E_field = get_E_field(X_grid, Y_grid, Lx=Lx, Ly=Ly, 
                               base_E=target_config.get('base_E', 200000.0))
        
        params_high = {
            't': t_field.flatten(),
            'rho': rho_field.flatten(),
            'E': E_field.flatten(),
            'z': z_field.flatten()
        }
        
        # [STAGE 1] Pattern Verification
        stage1_visualize_patterns(resolution_high[0], resolution_high[1], 
                                  X_grid, Y_grid, t_field, z_field)
        
        # 2. FEM Analysis
        K_h, M_h = fem_high.assemble(params_high)
        self.targets = []
        for case in self.cases:
            print(f"Solving Target: {case.name}")
            fd, fv, F = case.get_bcs(fem_high)
            free = np.setdiff1d(np.arange(fem_high.total_dof), fd)
            u = fem_high.solve_static_partitioned(K_h, F, jnp.array(free), fd, fv)
            
            self.targets.append({
                'case_name': case.name,
                'weight': case.weight,
                'u_static': u[2::6],  # W displacement only (3rd DOF in 6-DOF system)
                'max_surface_stress': fem_high.compute_max_surface_stress(u, params_high),
                'max_surface_strain': fem_high.compute_max_surface_strain(u, params_high),
                'params': params_high
            })
            
        print("Solving Target Eigenmodes...")
        vals, vecs = fem_high.solve_eigen(K_h, M_h, num_modes=num_modes_save + 10)
        self.target_eigen = {
            'vals': vals[3:3+num_modes_save],
            'modes': vecs[0::3, 3:3+num_modes_save]
        }
        
        # Mass calculation for constraint
        dx_h, dy_h = Lx/resolution_high[0], Ly/resolution_high[1]
        weights = np.ones((resolution_high[0]+1, resolution_high[1]+1))
        weights[0,:] *= 0.5; weights[-1,:] *= 0.5; weights[:,0] *= 0.5; weights[:,-1] *= 0.5
        self.target_mass = np.sum(t_field * rho_field * weights) * dx_h * dy_h
        print(f"Target Total Mass: {self.target_mass:.6f} tonne")
        
        self.fem_high = fem_high
        self.target_params_high = params_high
        
        # [STAGE 2] Results Visualization
        stage2_visualize_ground_truth(fem_high, self.targets, params_high)

    def optimize(self, opt_config, loss_weights, max_iterations=200):
        print(f"\nStarting Optimization (Max Iters: {max_iterations})...")
        Nx_l, Ny_l = self.fem.nx, self.fem.ny
        xl = np.linspace(0, Lx, Nx_l+1)
        yl = np.linspace(0, Ly, Ny_l+1)
        Xl, Yl = np.meshgrid(xl, yl, indexing='ij')
        pts_l = np.column_stack([Xl.flatten(), Yl.flatten()])
        
        # Interpolate targets to low-res
        Nx_h, Ny_h = self.resolution_high
        xh = np.linspace(0, Lx, Nx_h+1)
        yh = np.linspace(0, Ly, Ny_h+1)
        Xh, Yh = np.meshgrid(xh, yh, indexing='ij')
        pts_h = np.column_stack([Xh.flatten(), Yh.flatten()])
        
        self.targets_low = []
        for tgt in self.targets:
            u_l = griddata(pts_h, tgt['u_static'], pts_l, method='cubic')
            self.targets_low.append({
                'name': tgt['case_name'],
                'u_static': jnp.array(u_l),
                'weight': tgt['weight']
            })
            
        # Target Eigenmodes interpolation
        t_vals = self.target_eigen['vals']
        t_modes_l = []
        for i in range(len(t_vals)):
            m = griddata(pts_h, self.target_eigen['modes'][:, i], pts_l, method='cubic')
            t_modes_l.append(m)
        t_modes_l = jnp.stack(t_modes_l, axis=1)

        # Optimization Setup
        params = {
            't': jnp.full((Nx_l+1, Ny_l+1), opt_config['t'].get('init', 1.0)),
            'rho': jnp.full((Nx_l+1, Ny_l+1), opt_config['rho'].get('init', 7.5e-9)),
            'E': jnp.full((Nx_l+1, Ny_l+1), opt_config['E'].get('init', 200000.0))
        }
        
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(params)

        @jax.jit
        def loss_fn(p):
            K, M = self.fem.assemble(p)
            l_static = 0.0
            for i, case in enumerate(self.cases):
                fd, fv, F = case.get_bcs(self.fem)
                free = jnp.setdiff1d(jnp.arange(self.fem.total_dof), fd)
                u = self.fem.solve_static_partitioned(K, F, free, fd, fv)
                l_static += jnp.mean((u[0::3] - self.targets_low[i]['u_static'])**2) * case.weight
            
            vals, vecs = self.fem.solve_eigen(K, M, num_modes=len(t_vals)+5)
            l_freq = jnp.mean((vals[3:3+len(t_vals)] - t_vals)**2)
            
            return l_static * loss_weights['static'] + l_freq * loss_weights['freq']

        for i in range(max_iterations):
            val, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            # Clip
            for k in params:
                params[k] = jnp.clip(params[k], opt_config[k]['min'], opt_config[k]['max'])
            if i % 20 == 0: print(f"Iter {i}: Loss = {val:.6e}")

        self.optimized_params = params
        return params

    def verify(self):
        print("\nFinal Verification...")
        # (Simplified verification for this clean version)
        stage3_visualize_comparison(self.fem_high, self.targets, 
                                     {'t': griddata(np.linspace(0,Lx,Nx_low+1), self.optimized_params['t'], np.linspace(0,Lx,Nx_high+1), method='cubic'),
                                      'z': jnp.zeros(self.fem_high.num_nodes)}, 
                                     self.target_params_high)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    model = EquivalentSheetModel(Lx, Ly, Nx_low, Ny_low)
    
    # Load Cases
    model.add_case(TwistCase("twist_x", axis='x', value=1.5))
    model.add_case(PureBendingCase("bend_y", axis='y', value=3.0))
    
    target_config = {
        'pattern': 'ABC',
        'bead_t': {'A': 2.0, 'B': 2.5, 'C': 1.5},
        'pattern_pz': 'TNY',
        'bead_pz': {'T': 2.0, 'N': 1.0, 'Y': -2.0}
    }
    
    model.generate_targets(target_config=target_config)
    
    opt_config = {
        't': {'init': 1.2, 'min': 0.5, 'max': 5.0},
        'rho': {'init': 7.5e-9, 'min': 5e-9, 'max': 1e-8},
        'E': {'init': 200000.0, 'min': 100000.0, 'max': 300000.0}
    }
    
    model.optimize(opt_config, {'static': 1.0, 'freq': 0.1}, max_iterations=50)
    model.verify()
