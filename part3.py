
    def verify(self, num_modes_compare=None):
        if num_modes_compare is None:
            if hasattr(self, 'num_modes_truth'):
                num_modes_compare = self.num_modes_truth
            else:
                num_modes_compare = 5

        print(f"Running Verification Plots (High Resolution, Modes: {num_modes_compare})...")
        
        # 1. Setup High Res Verification Model
        if hasattr(self, 'resolution_high'):
             Nx_v, Ny_v = self.resolution_high
        else:
             Nx_v, Ny_v = 50, 20
             
        fem_verify = PlateFEM(self.fem.Lx, self.fem.Ly, Nx_v, Ny_v)
        
        # 2. Interpolate Optimized Params to High Res
        from scipy.interpolate import griddata
        
        # Low Res Coords
        # Ensure we use Nodes or Element Centers correctly.
        # Check optimize() init: params size is (nx+1)*(ny+1) -> NODAL
        xl = np.linspace(0, self.fem.Lx, self.fem.nx+1)
        yl = np.linspace(0, self.fem.Ly, self.fem.ny+1)
        Xl, Yl = np.meshgrid(xl, yl, indexing='ij')
        pts_src = np.column_stack([Xl.flatten(), Yl.flatten()])
        
        # Destination: High Res Nodes
        xv = np.linspace(0, self.fem.Lx, Nx_v+1)
        yv = np.linspace(0, self.fem.Ly, Ny_v+1)
        X_dst, Y_dst = np.meshgrid(xv, yv, indexing='ij')
        pts_dst = np.column_stack([X_dst.flatten(), Y_dst.flatten()])
        
        params_v = {}
        # Ensure params keys exist.
        for k in ['t', 'rho', 'E']:
             if k in self.optimized_params:
                 val_l = self.optimized_params[k].flatten()
                 # Interpolate
                 val_v = griddata(pts_src, val_l, pts_dst, method='cubic', fill_value=np.mean(val_l))
                 
                 # Fix NaNs from griddata (extrapolation)
                 mask = np.isnan(val_v)
                 if np.any(mask):
                     val_v[mask] = griddata(pts_src, val_l, pts_dst[mask], method='nearest')
                     
                 params_v[k] = jnp.array(val_v)
             else:
                 # Should not happen if initialized properly
                 print(f"Warning: Param {k} missing in optimized_params!")

        # Ensure z is present (if not optimized, it's 0)
        if 'z' in self.optimized_params:
            val_l = self.optimized_params['z'].flatten()
            val_v = griddata(pts_src, val_l, pts_dst, method='cubic', fill_value=0.0)
            mask = np.isnan(val_v)
            if np.any(mask):
                 val_v[mask] = griddata(pts_src, val_l, pts_dst[mask], method='nearest')
            params_v['z'] = jnp.array(val_v)
        else:
            params_v['z'] = jnp.zeros(pts_dst.shape[0])

        # 3. Assemble High Res
        K_v, M_v = fem_verify.assemble(params_v)
        
        # 4. Verify Matches
        x_plt = np.linspace(0, self.fem.Lx, Nx_v+1)
        y_plt = np.linspace(0, self.fem.Ly, Ny_v+1)
        
        static_results = []
        
        for i, case in enumerate(self.cases):
            tgt = self.targets[i] # High Res Target
            
            # Solve Optimized High Res
            fd, fv, F = case.get_bcs(fem_verify)
            
            all_dofs = np.arange(fem_verify.total_dof)
            free = np.setdiff1d(all_dofs, fd)
            
            u = fem_verify.solve_static_partitioned(K_v, F, jnp.array(free), fd, fv)
            
            # Displacement (w is index 2)
            z_opt = u[0::6].flatten() # Wait, u structure [u,v,w,tx,ty,tz] -> w is 2::6
            # My previous code said 0::3 for 3-DOF solver, but 6-DOF is 2::6?
            # Let's check PlateFEM solver definition.
            # Assuming 6-DOF per node as updated.
            # u structure: [u, v, w, tx, ty, tz]
            # w -> 2
            
            w_opt = u[2::6].reshape(Nx_v+1, Ny_v+1)
            # Target u_static is also 6-DOF vector from Generate Targets
            w_ref = tgt['u_static'][2::6].reshape(Nx_v+1, Ny_v+1)
            
            # Compute max surface stress and strain
            stress_opt = fem_verify.compute_max_surface_stress(u, params_v).reshape(Nx_v+1, Ny_v+1)
            strain_opt = fem_verify.compute_max_surface_strain(u, params_v).reshape(Nx_v+1, Ny_v+1)
            
            stress_ref = tgt['max_surface_stress'].reshape(Nx_v+1, Ny_v+1)
            strain_ref = tgt['max_surface_strain'].reshape(Nx_v+1, Ny_v+1)
            
            # Metrics Calculation
            def calc_metrics(ref, opt):
                rmse = np.sqrt(np.mean((ref - opt)**2))
                data_range = np.max(ref) - np.min(ref)
                sim = 100.0 if data_range < 1e-9 else max(0.0, (1.0 - rmse/data_range)*100)
                return sim, rmse
            
            disp_sim, disp_rmse = calc_metrics(w_ref, w_opt)
            stress_sim, stress_rmse = calc_metrics(stress_ref, stress_opt)
            strain_sim, strain_rmse = calc_metrics(strain_ref, strain_opt)
            
            static_results.append({
                'case': case.name,
                'disp_sim': disp_sim, 'stress_sim': stress_sim, 'strain_sim': strain_sim,
                'disp_max_ref': np.max(np.abs(w_ref)), 'disp_max_opt': np.max(np.abs(w_opt))
            })

            # Create 3x3 plot
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            
            def get_robust_levels(data, n_sigma=2.0, num_levels=31):
                """Calculate robust min/max using Mean +/- N*Sigma and return levels."""
                mu = np.mean(data)
                sigma = np.std(data)
                vmin = max(np.min(data), mu - n_sigma * sigma)
                vmax = min(np.max(data), mu + n_sigma * sigma)
                if vmax <= vmin:
                    vmin, vmax = np.min(data), np.max(data)
                if vmax <= vmin: 
                    vmax = vmin + 1e-8
                return np.linspace(vmin, vmax, num_levels)

            # Determine consistent robust levels
            disp_levels = get_robust_levels(w_ref, n_sigma=2.5) 
            strain_levels = get_robust_levels(strain_ref * 1000, n_sigma=2.0)
            stress_levels = get_robust_levels(stress_ref, n_sigma=2.0)
            
            def add_stats_text(ax, data_ref, data_opt, unit=""):
                txt = (f"Target: min={np.min(data_ref):.3f}, max={np.max(data_ref):.3f} {unit}\n"
                       f"Optimized: min={np.min(data_opt):.3f}, max={np.max(data_opt):.3f} {unit}")
                ax.text(0.5, -0.15, txt, transform=ax.transAxes, 
                        ha='center', va='top', fontsize=8, color='darkblue')

            # Row 1: Displacement (w)
            im0 = axes[0, 0].contourf(x_plt, y_plt, w_ref.T, levels=disp_levels, cmap='jet', extend='both')
            axes[0, 0].set_title(f"{case.name} - Height(w) Target (mm)")
            axes[0, 0].set_aspect('equal')
            plt.colorbar(im0, ax=axes[0, 0])
            
            im1 = axes[0, 1].contourf(x_plt, y_plt, w_opt.T, levels=disp_levels, cmap='jet', extend='both')
            axes[0, 1].set_title(f"{case.name} - Height(w) Optimized (mm)")
            axes[0, 1].set_aspect('equal')
            plt.colorbar(im1, ax=axes[0, 1])
            add_stats_text(axes[0, 1], w_ref, w_opt, "mm")
            
            im2 = axes[0, 2].contourf(x_plt, y_plt, np.abs(w_opt - w_ref).T, levels=30, cmap='magma')
            axes[0, 2].set_title("Height(w) Error (mm)")
            axes[0, 2].set_aspect('equal')
            plt.colorbar(im2, ax=axes[0, 2])
            
            # Row 2: Max Surface Strain
            im3 = axes[1, 0].contourf(x_plt, y_plt, strain_ref.T * 1000, levels=strain_levels, cmap='viridis', extend='both')
            axes[1, 0].set_title("Max Surface Strain Target (×10⁻³)")
            axes[1, 0].set_aspect('equal')
            plt.colorbar(im3, ax=axes[1, 0])
            
            im4 = axes[1, 1].contourf(x_plt, y_plt, strain_opt.T * 1000, levels=strain_levels, cmap='viridis', extend='both')
            axes[1, 1].set_title("Max Surface Strain Optimized (×10⁻³)")
            axes[1, 1].set_aspect('equal')
            plt.colorbar(im4, ax=axes[1, 1])
            add_stats_text(axes[1, 1], strain_ref * 1000, strain_opt * 1000, "×10⁻³")
            
            im5 = axes[1, 2].contourf(x_plt, y_plt, np.abs(strain_opt - strain_ref).T * 1000, levels=30, cmap='magma')
            axes[1, 2].set_title("Strain Error (×10⁻³)")
            axes[1, 2].set_aspect('equal')
            plt.colorbar(im5, ax=axes[1, 2])
            
            # Row 3: Max Surface Stress
            im6 = axes[2, 0].contourf(x_plt, y_plt, stress_ref.T, levels=stress_levels, cmap='plasma', extend='both')
            axes[2, 0].set_title("Max Surface Stress Target (MPa)")
            axes[2, 0].set_aspect('equal')
            plt.colorbar(im6, ax=axes[2, 0])
            
            im7 = axes[2, 1].contourf(x_plt, y_plt, stress_opt.T, levels=stress_levels, cmap='plasma', extend='both')
            axes[2, 1].set_title("Max Surface Stress Optimized (MPa)")
            axes[2, 1].set_aspect('equal')
            plt.colorbar(im7, ax=axes[2, 1])
            add_stats_text(axes[2, 1], stress_ref, stress_opt, "MPa")
            
            im8 = axes[2, 2].contourf(x_plt, y_plt, np.abs(stress_opt - stress_ref).T, levels=30, cmap='magma')
            axes[2, 2].set_title("Stress Error (MPa)")
            axes[2, 2].set_aspect('equal')
            plt.colorbar(im8, ax=axes[2, 2])
            
            plt.tight_layout()
            filename = f"verify_{case.name}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Saved {os.path.abspath(filename)} (High Res)")
            
        # 4.5 Plot Optimized Parameters
        t_opt = params_v['t'].reshape(Nx_v+1, Ny_v+1)
        z_opt = params_v['z'].reshape(Nx_v+1, Ny_v+1)
        # Plot Thickness & Z
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        im0 = axes[0].pcolormesh(x_plt, y_plt, t_opt.T, cmap='viridis', shading='nearest')
        axes[0].set_title('Optimized Thickness (mm)')
        plt.colorbar(im0, ax=axes[0])
        axes[0].set_aspect('equal')
        
        im1 = axes[1].pcolormesh(x_plt, y_plt, z_opt.T, cmap='terrain', shading='nearest')
        axes[1].set_title('Optimized Topography Z (mm)')
        plt.colorbar(im1, ax=axes[1])
        axes[1].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig("verify_parameters.png")
        plt.close()
             
        # 5. Verify Modes
        print("Verifying Modes (High Res)...")
        vals, vecs = fem_verify.solve_eigen(K_v, M_v, num_modes=num_modes_compare + 10)
        
        # Target modes
        t_vals = self.target_eigen['vals']
        t_modes = self.target_eigen['modes'] # Full (dof, num_modes)
        
        # Opt Modes
        o_vals = vals[3:] # Skip 3 rigid body modes
        o_vecs = vecs[:, 3:]
        
        # Filter Rigid Body Modes based on Frequency
        freq_t_all = np.sqrt(np.abs(t_vals)) / (2*np.pi)
        freq_o_all = np.sqrt(np.abs(o_vals)) / (2*np.pi)
        
        # Compare first N modes
        num_modes_plot = min(len(freq_t_all), num_modes_compare)
        freq_t = freq_t_all[:num_modes_plot]
        freq_o = freq_o_all[:num_modes_plot]
        
        macs = []
        for j in range(num_modes_plot):
            v1 = o_vecs[:, j]
            # Use interpolated target eigenmode at high res?
            # self.target_vecs is (dof, num_modes) at High Res.
            # So we can compare directly if fem_verify shape is same.
            # fem_verify is High Res. And generate_targets uses same High Res (resolution_high).
            # So shapes should match exactly.
            v2 = self.target_vecs[:, j]
            
            # Normalize
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            
            mac = (np.dot(v1, v2))**2
            macs.append(mac)
            
        print(f"Modal Analysis: Evaluated {num_modes_plot} modes.")
        for j in range(num_modes_plot):
             print(f"Mode {j+1}: tgt={freq_t[j]:.2f}Hz, opt={freq_o[j]:.2f}Hz, MAC={macs[j]:.4f}")
