
    def optimize(self, opt_config, loss_weights, use_smoothing=True, use_curvature=False, use_moment=False, use_strain_energy=False, use_surface_stress=False, use_surface_strain=False, use_mass_constraint=False, mass_tolerance=0.05, max_iterations=200, use_early_stopping=True, early_stop_patience=None, early_stop_tol=1e-6, num_modes_loss=None):
        """
        Runs the optimization process to find equivalent sheet properties.
        """
        
        # Use stored number of modes if not provided
        if num_modes_loss is None:
            if hasattr(self, 'num_modes_truth'):
                num_modes_loss = self.num_modes_truth
            else:
                num_modes_loss = 5 # Fallback default
        
        print("OPTIMIZE METHOD ENTERED", flush=True)
        print(f"Starting Optimization (Modes for Loss: {num_modes_loss})...", flush=True)
        
        # 1. Interpolate Targets to Low Res
        self.targets_low = [] # Store for verification
        from scipy.interpolate import griddata
        
        # Uses detected or stored resolution
        if hasattr(self, 'resolution_high'):
             Nx_h, Ny_h = self.resolution_high
        else:
             print("Warning: resolution_high not found, using default (50, 20)")
             Nx_h, Ny_h = 50, 20
             
        xh = np.linspace(0, self.fem.Lx, Nx_h+1)
        yh = np.linspace(0, self.fem.Ly, Ny_h+1)
        Xh, Yh = np.meshgrid(xh, yh, indexing='ij')
        pts_h = np.column_stack([Xh.flatten(), Yh.flatten()])
        
        xl = self.fem.node_coords
        
        for tgt in self.targets:
            print(f"DEBUG Interpolation START: pts_h={pts_h.shape}, u_static len={len(tgt['u_static'])}", flush=True)
            try:
                 u_h = tgt['u_static'].reshape(-1, 6) # (N, 6)
                 print(f"DEBUG Interpolation: u_h reshaped={u_h.shape}", flush=True)
                 u_l = griddata(pts_h, u_h, xl, method='cubic')
            except Exception as e:
                 print(f"CRASH in griddata (u_static): {e}", flush=True)
                 print(f"pts_h shape: {pts_h.shape}", flush=True)
                 print(f"u_h shape: {u_h.shape} (inferred)", flush=True)
                 print(f"xl shape: {xl.shape}", flush=True)
                 raise e

            tgt_low = {
                'case_name': tgt['case_name'],
                'u_static': jnp.array(u_l.flatten()),
                'weight': tgt['weight']
            }
            
            # Interpolate curvature if requested
            if use_curvature:
                curv_h = tgt['curvature']  # shape: (Nh, 3)
                curv_l = np.zeros((xl.shape[0], 3))
                for comp in range(3):
                    curv_l[:, comp] = griddata(pts_h, curv_h[:, comp], xl, method='cubic')
                tgt_low['curvature'] = jnp.array(curv_l)
            
            # Interpolate moment if requested
            if use_moment:
                mom_h = tgt['moment']  # shape: (Nh, 3)
                mom_l = np.zeros((xl.shape[0], 3))
                for comp in range(3):
                    mom_l[:, comp] = griddata(pts_h, mom_h[:, comp], xl, method='cubic')
                tgt_low['moment'] = jnp.array(mom_l)
            
            # Interpolate strain energy density if requested
            if use_strain_energy:
                sed_h = tgt['strain_energy_density']  # shape: (Nh,)
                sed_l = griddata(pts_h, sed_h, xl, method='cubic')
                tgt_low['strain_energy_density'] = jnp.array(sed_l)
            
            # Interpolate max surface stress if requested
            if use_surface_stress:
                stress_h = tgt['max_surface_stress']  # shape: (Nh,)
                stress_l = griddata(pts_h, stress_h, xl, method='cubic')
                tgt_low['max_surface_stress'] = jnp.array(stress_l)
            
            # Interpolate max surface strain if requested
            if use_surface_strain:
                strain_h = tgt['max_surface_strain']  # shape: (Nh,)
                strain_l = griddata(pts_h, strain_h, xl, method='cubic')
                tgt_low['max_surface_strain'] = jnp.array(strain_l)
            
            self.targets_low.append(tgt_low)
            
        # Interpolate Modes
        t_vals = self.target_eigen['vals'][:num_modes_loss]
        t_modes_h = self.target_eigen['modes'][:, :num_modes_loss]
        t_modes_l = []
        for i in range(num_modes_loss):
             # Reshape to (N, 6) because griddata expects values per point
             mode_h_reshaped = t_modes_h[:,i].reshape(-1, 6)
             # Interpolate vectors
             m_reshaped = griddata(pts_h, mode_h_reshaped, xl, method='cubic')
             # Flatten back to (total_dof,)
             m = m_reshaped.flatten()
             t_modes_l.append(m)
             
        if len(t_modes_l) > 0:
            t_modes_l = jnp.stack(t_modes_l, axis=1) # (dof, num_modes)
        else:
            t_modes_l = jnp.zeros((xl.shape[0]*6, 0))
        
        # 2. Pre-calculate BCs
        bcs_list = []
        all_dofs = np.arange(self.fem.total_dof)
        
        for case, tgt_l in zip(self.cases, self.targets_low):
            fd, fv, F = case.get_bcs(self.fem)
            free = np.setdiff1d(all_dofs, fd)
            bc_dict = {
                'fd': fd, 'fv': fv, 'free': jnp.array(free), 'F': F,
                'target_u': tgt_l['u_static'],
                'weight': case.weight
            }
            
            # Add curvature/moment targets if requested
            if use_curvature and 'curvature' in tgt_l:
                bc_dict['target_curvature'] = tgt_l['curvature']
            if use_moment and 'moment' in tgt_l:
                bc_dict['target_moment'] = tgt_l['moment']
            if use_strain_energy and 'strain_energy_density' in tgt_l:
                bc_dict['target_strain_energy'] = tgt_l['strain_energy_density']
            if use_surface_stress and 'max_surface_stress' in tgt_l:
                bc_dict['target_surface_stress'] = tgt_l['max_surface_stress']
            if use_surface_strain and 'max_surface_strain' in tgt_l:
                bc_dict['target_surface_strain'] = tgt_l['max_surface_strain']
            
            bcs_list.append(bc_dict)
            
        # Initialize Base Parameters
        def get_init_val(key, default):
            cfg = opt_config.get(key, {})
            if 'init' in cfg:
                return cfg['init']
            if 'min' in cfg and 'max' in cfg:
                return (cfg['min'] + cfg['max']) / 2.0
            return default

        Base_t = get_init_val('t', 1.0)
        Base_rho = get_init_val('rho', 1.0)
        Base_E = get_init_val('E', 1.0)
        
        print(f"Initialized Params: t={Base_t:.4f}, rho={Base_rho:.4e}, E={Base_E:.4e}")
            
        # Initialize optimizable parameters at NODES (matches solver expectations for field input)
        # Assuming params initialized as flat arrays if not specified, 
        # but JAX optimization works better with structured dict of arrays.
        # Ensure shape is (num_nodes,)
        num_nodes = (self.fem.nx + 1) * (self.fem.ny + 1)
        
        params = {
            't': jnp.full(num_nodes, Base_t),
            'rho': jnp.full(num_nodes, Base_rho),
            'E': jnp.full(num_nodes, Base_E)
        }
        
        # Initialize z (Topography) if enabled in config
        # Check if 'z' key exists roughly in opt_config or implicitly handled
        # We will add 'z' to params if we want to optimize it.
        # Assuming 'z' optimization is desired if we call optimize? Or config dependent?
        # Let's check config.
        # If 'z' is in opt_config, we use it.
        if 'z' in opt_config:
            # Init z (usually 0)
            z_init = opt_config['z'].get('init', 0.0)
            params['z'] = jnp.full(num_nodes, z_init)
            print(f"Topography Optimization Enabled. Initial Z={z_init:.4f}")
        
        start_params = params
        
        # Pre-calculate mass integration weights for low-res mesh
        dx_l = self.fem.Lx / self.fem.nx
        dy_l = self.fem.Ly / self.fem.ny
        cell_area_l = dx_l * dy_l
        
        weights_mass = jnp.ones((self.fem.nx+1, self.fem.ny+1))
        weights_mass = weights_mass.at[0, :].multiply(0.5)
        weights_mass = weights_mass.at[-1, :].multiply(0.5)
        weights_mass = weights_mass.at[:, 0].multiply(0.5)
        weights_mass = weights_mass.at[:, -1].multiply(0.5)
        weights_mass = weights_mass.flatten() # Flatten to match nodal params
        
        target_mass = self.target_mass if hasattr(self, 'target_mass') else 1.0
        
        # 3. Loss Function
        @jax.jit
        def loss_fn(params):
            K, M = self.fem.assemble(params)
            
            total_loss = 0.0
            aux = {} # Store individual loss components
            
            # A. Static Displacement Loss
            loss_static = 0.0
            loss_curvature = 0.0
            loss_moment = 0.0
            loss_strain_energy = 0.0
            loss_surface_stress = 0.0
            loss_surface_strain = 0.0
            
            for bc in bcs_list:
                # Solve Static
                u = self.fem.solve_static_partitioned(K, bc['F'], bc['free'], bc['fd'], bc['fv'])
                
                # 1. Displacement MSE
                # Focus on Z-displacement (w) mostly? Or all components?
                # Target u_static is full 6-DOF vector.
                # However, 6-DOF vs 5-DOF (if solver changed) might be issue?
                # Assuming dimension match.
                diff = u - bc['target_u']
                
                # Weighting: mostly care about w?
                # Let's weight w higher or just MSE on all.
                # w indices: 2::6
                w_diff = diff[2::6]
                mse = jnp.mean(w_diff**2)
                loss_static += mse * bc['weight']
                
                # 2. Curvature Loss (Legacy)
                if use_curvature and 'target_curvature' in bc:
                    # Compute curvature
                    curv = self.fem.compute_curvature(u)
                    mse_curv = jnp.mean((curv - bc['target_curvature'])**2)
                    loss_curvature += mse_curv * bc['weight']
                    
                # 3. Moment Loss (Legacy)
                if use_moment and 'target_moment' in bc:
                    # Compute moment
                    params_vals = params # passed to moment calc?
                    mom = self.fem.compute_moment(u, params)
                    mse_mom = jnp.mean((mom - bc['target_moment'])**2)
                    loss_moment += mse_mom * bc['weight']
                
                # 4. Strain Energy Density Loss (New)
                if use_strain_energy and 'target_strain_energy' in bc:
                     sed = self.fem.compute_strain_energy_density(u, params)
                     # Normalize?
                     mse_sed = jnp.mean((sed - bc['target_strain_energy'])**2)
                     # Maybe normalize by target energy magnitude?
                     # scale = 1.0 / (jnp.mean(bc['target_strain_energy']**2) + 1e-9)
                     loss_strain_energy += mse_sed * bc['weight']

                # 5. Surface Stress Loss (New)
                if use_surface_stress and 'target_surface_stress' in bc:
                     stress = self.fem.compute_max_surface_stress(u, params)
                     mse_stress = jnp.mean((stress - bc['target_surface_stress'])**2)
                     loss_surface_stress += mse_stress * bc['weight']

                # 6. Surface Strain Loss (New)
                if use_surface_strain and 'target_surface_strain' in bc:
                     strain = self.fem.compute_max_surface_strain(u, params)
                     mse_strain = jnp.mean((strain - bc['target_surface_strain'])**2)
                     loss_surface_strain += mse_strain * bc['weight']

            total_loss += loss_static * loss_weights.get('static', 1.0)
            aux['static'] = loss_static
            
            if use_curvature:
                total_loss += loss_curvature * loss_weights.get('curvature', 0.0)
                aux['curvature'] = loss_curvature
            else:
                aux['curvature'] = 0.0
                
            if use_moment:
                total_loss += loss_moment * loss_weights.get('moment', 0.0)
                aux['moment'] = loss_moment
            else:
                aux['moment'] = 0.0

            if use_strain_energy:
                total_loss += loss_strain_energy * loss_weights.get('strain_energy', 0.0)
                aux['strain_energy'] = loss_strain_energy
            else:
                aux['strain_energy'] = 0.0

            if use_surface_stress:
                total_loss += loss_surface_stress * loss_weights.get('surface_stress', 0.0)
                aux['surface_stress'] = loss_surface_stress
            else:
                aux['surface_stress'] = 0.0

            if use_surface_strain:
                total_loss += loss_surface_strain * loss_weights.get('surface_strain', 0.0)
                aux['surface_strain'] = loss_surface_strain
            else:
                aux['surface_strain'] = 0.0
            
            # B. Eigenfrequency & Mode Shape Loss
            loss_freq = 0.0
            loss_mode = 0.0
            
            if loss_weights.get('freq', 0) > 0 or loss_weights.get('mode', 0) > 0:
                vals, vecs = self.fem.solve_eigen(K, M, num_modes=num_modes_loss + 3) # +3 for rigid body
                
                # Assume sorted. Skip first 3 rigid body modes (near 0)
                vals_opt = vals[3:]
                vecs_opt = vecs[:, 3:]
                
                # Dimensions match?
                n_modes = min(len(vals_opt), len(t_vals))
                
                # Freq Error
                if loss_weights.get('freq', 0) > 0:
                    # Use relative error squared
                    err = (vals_opt[:n_modes] - t_vals[:n_modes]) / (t_vals[:n_modes] + 1e-6)
                    loss_freq = jnp.mean(err**2)
                    total_loss += loss_freq * loss_weights.get('freq', 0)
                
                # MAC Loss (1 - MAC)
                if loss_weights.get('mode', 0) > 0:
                    # Target modes t_modes_l are (N_l, num_modes)
                    # Opt modes vecs_opt are (dof, num_modes)
                    # We need to extract w-component or full vector?
                    # t_modes_l provided from generate_targets are FULL vectors interpolated?
                    # verify: griddata on t_modes_h (N_h, dof?). 
                    # Step 728: t_modes_h = self.target_eigen['modes'].
                    # Solver returns (total_dof, num_modes).
                    # So griddata was likely wrong if applied to flattened DOF vector directly roughly?
                    # The griddata in step 731: griddata(..., t_modes_h[:,i], ...) implies t_modes_h[:,i] is values at nodes?
                    # But t_modes_h has size total_dof = 6*N.
                    # Griddata expects values at points.
                    # This interpolation of modes was likely buggy if passed raw DOF vector to N points.
                    # Assuming we fix this or ignore mode loss for now if broken.
                    # Let's assume t_modes_l is valid (dof, n_modes).
                    
                    # Compute MAC
                    for i in range(n_modes):
                        v1 = vecs_opt[:, i]
                        v2 = t_modes_l[:, i]
                        
                        # Normalize
                        v1 = v1 / jnp.linalg.norm(v1)
                        v2 = v2 / jnp.linalg.norm(v2)
                        
                        mac = (jnp.dot(v1, v2))**2
                        loss_mode += (1.0 - mac)
                        
                    loss_mode /= n_modes
                    total_loss += loss_mode * loss_weights.get('mode', 0)
            
            aux['freq'] = loss_freq
            aux['mode'] = loss_mode
            
            # C. Regularization (TV)
            loss_reg = 0.0
            if loss_weights.get('reg', 0) > 0:
                # TV of thickness
                t_field = params['t'].reshape(self.fem.nx + 1, self.fem.ny + 1)
                tv_x = jnp.mean(jnp.abs(t_field[1:, :] - t_field[:-1, :]))
                tv_y = jnp.mean(jnp.abs(t_field[:, 1:] - t_field[:, :-1]))
                loss_reg += (tv_x + tv_y)
                
                # TV of Z (Topography) - Important for smooth shapes
                if 'z' in params:
                     z_field = params['z'].reshape(self.fem.nx + 1, self.fem.ny + 1)
                     tv_z_x = jnp.mean(jnp.abs(z_field[1:, :] - z_field[:-1, :]))
                     tv_z_y = jnp.mean(jnp.abs(z_field[:, 1:] - z_field[:, :-1]))
                     loss_reg += (tv_z_x + tv_z_y) * 0.1 # Scale z-smoothing?

                total_loss += loss_reg * loss_weights.get('reg', 0)
            aux['reg'] = loss_reg
            
            # D. Mass Constraint (Penalty)
            loss_mass = 0.0
            if use_mass_constraint:
                # Calculate current mass
                # M = sum(rho * t * weights * cell_area)
                current_mass = jnp.sum(params['rho'] * params['t'] * weights_mass) * cell_area_l
                
                # Constraint: |M - M_tgt| / M_tgt <= tol
                # Penalty: max(0, error - tol)^2
                rel_error = jnp.abs(current_mass - target_mass) / target_mass
                violation = jnp.maximum(0.0, rel_error - mass_tolerance)
                
                loss_mass = violation * 10.0 # Strict penalty
                total_loss += loss_mass * loss_weights.get('mass', 1.0)
                
            aux['mass'] = loss_mass
            
            return total_loss, aux
        
        # 4. Optimization Loop
        # Define optimizer
        optimizer = optax.adam(learning_rate=0.01) # Slower LR for stability
        opt_state = optimizer.init(params)
        
        @jax.jit
        def step(params, opt_state):
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            
            # Mask gradients for non-optimized parameters
            grads_masked = {}
            for k, g in grads.items():
                if opt_config.get(k, {}).get('opt', False) or (k == 'z' and 'z' in opt_config):
                    grads_masked[k] = g
                else:
                    grads_masked[k] = jnp.zeros_like(g)
            
            updates, new_opt_state = optimizer.update(grads_masked, opt_state, params)
            p_new = optax.apply_updates(params, updates)
            
            # Enforce Bounds
            for k in p_new:
                if k in opt_config:
                    info = opt_config[k]
                    if 'min' in info and 'max' in info:
                        p_new[k] = jnp.clip(p_new[k], info['min'], info['max'])
                elif k == 'z':
                     # Default bounds for Z if not specified? 
                     # E.g. +/- 30mm
                     p_new[k] = jnp.clip(p_new[k], -50.0, 50.0)

            return p_new, new_opt_state, loss, aux
        
        # Build dynamic header
        loss_names = ['Static', 'Freq', 'Mode', 'StrainE', 'SurfStr', 'SurfEps', 'Reg', 'Mass']
        active_flags = [
            True,
            loss_weights.get('freq', 0) > 0,
            loss_weights.get('mode', 0) > 0,
            use_strain_energy,
            use_surface_stress,
            use_surface_strain,
            loss_weights.get('reg', 0) > 0,
            use_mass_constraint
        ]
        active_indices = [i for i, active in enumerate(active_flags) if active]
        
        header = f"{'Iter':<5} | {'Loss':<10}"
        for idx in active_indices:
            header += f" | {loss_names[idx]:<10}"
        print(header)
        
        print_interval = max(1, max_iterations // 10)
        best_loss = 1e9
        patience_counter = 0
        
        start_time = time.time()
        for i in range(max_iterations + 1):
            params, opt_state, L, aux = step(params, opt_state)
            
            # Map aux dictionary to list for printing
            aux_vals = [
                aux['static'], aux['freq'], aux['mode'], 
                aux['strain_energy'], aux['surface_stress'], aux['surface_strain'],
                aux['reg'], aux['mass']
            ]
            
            if i % print_interval == 0 or i == max_iterations:
                row = f"{i:<5} | {L:<10.4f}"
                for idx in active_indices:
                    row += f" | {aux_vals[idx]:<10.4f}"
                print(row)
            
            # Early stopping
            if use_early_stopping:
                loss_val = float(L)
                if loss_val < best_loss - early_stop_tol:
                    best_loss = loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= (early_stop_patience or 20) and i > 20:
                    print(f"Early stopping at iteration {i}")
                    break
        
        print(f"Optimization finished in {time.time()-start_time:.2f}s")
        self.optimized_params = params
        return params
