
if __name__ == '__main__':
    # XLA Flags for Threading
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
    
    # 1. Init Model
    model = EquivalentSheetModel(Lx, Ly, Nx_low, Ny_low)
    
    # 2. Add Cases
    model.add_case(TwistCase('twist_x', value=1.5, mode='angle'))
    model.add_case(TwistCase('twist_y', value=1.5, mode='angle'))
    model.add_case(PureBendingCase('bend_y', value=3.0, mode='angle'))
    model.add_case(PureBendingCase('bend_x', value=1.0, mode='angle'))
    model.add_case(CornerLiftCase('lift_br', corner='br', value=1.0))
    model.add_case(CornerLiftCase('lift_bl', corner='bl', value=3.0))
    
    # 3. Generate Truth 
    # Define Target Config with Z-pattern
    target_config = {
        'base_t': 1.0, 
        'bead_t': {'A': 2.0, 'B': 2.5, 'C': 1.5},  
        'bead_pz': {'T': 20.0, 'N': 10.0, 'Y': -20.0},
        'base_rho': 7.85e-9,  
        'base_E': 210000.0,   
        'pattern': 'ABC',       
        'pattern_pz': 'TNY' 
    }
    try:
        print("DEBUG: Calling generate_targets...", flush=True)
        model.generate_targets(resolution_high=(Nx_high, Ny_high), num_modes_save=6, target_config=target_config)
        print("DEBUG: generate_targets completed.", flush=True)
    except Exception as e:
        print(f"CRASH in generate_targets: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 4. Optimize
    cfg = {
        't': {'opt': True, 'min': 0.5, 'max': 3.0, 'init': 1.0},
        'rho': {'opt': False, 'min': 1e-9, 'max': 1e-8, 'init': 7.85e-9},
        'E': {'opt': False, 'min': 40e3, 'max': 250e3, 'init': 210e3},
        'z': {'opt': True, 'init': 0.0} # Enable Z optimization
    }
    
    weights = { 
        'static': 1.0,           
        'freq': 1.0,             
        'mode': 1.0,             
        'curvature': 0.0,        
        'moment': 0.0,           
        'strain_energy': 2.0,    
        'surface_stress': 1.0,   
        'surface_strain': 1.0,   
        'reg': 0.05,             
        'mass': 1.0              
    }
    
    try:
        print("DEBUG: Calling model.optimize...", flush=True)
        model.optimize(cfg, weights, use_smoothing=False, 
                       use_curvature=False, use_moment=False,
                       use_strain_energy=True, use_surface_stress=True, use_surface_strain=True,
                       use_mass_constraint=True, mass_tolerance=0.05,
                       max_iterations=300, 
                       use_early_stopping=False, 
                       early_stop_patience=None, 
                       early_stop_tol=1e-8,
                       num_modes_loss=None)
        print("DEBUG: model.optimize completed.", flush=True)
    except Exception as e:
        print(f"CRASH in optimize: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 5. Verify
    model.verify(num_modes_compare=None)
