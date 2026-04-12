import numpy as np
import h5py
import os
from WHT_EQS_analysis import PlateFEM, StructuralResult

def test_modal_export_logic():
    print("Testing Modal VTKHDF Export Logic...")
    
    # 1. Setup a simple grid
    Lx, Ly = 200, 100
    nx, ny = 10, 5
    fem = PlateFEM(Lx, Ly, nx, ny)
    
    # 2. Mock modal results
    num_modes = 3
    num_points = len(fem.nodes)
    freqs = [10.5, 25.3, 45.7]
    
    # Mock mode shapes (z-displacement pattern)
    modes = []
    x = fem.nodes[:, 0]
    for i in range(num_modes):
        # Create different wave patterns
        shape = np.zeros((num_points, 6))
        shape[:, 2] = np.sin((i+1) * np.pi * x / Lx) # W-displacement
        modes.append(shape.flatten())
        
    # 3. Use StructuralResult to save
    res = StructuralResult({}, fem.nodes, fem.elements)
    steps_dict = {
        'values': freqs,
        'point_data': {
            'mode_shape_vec': modes
        }
    }
    
    output_file = "test_modal_v3.vtkhdf"
    res.save_vtkhdf(output_file, steps_dict=steps_dict)
    
    # 4. Verify HDF5 structure
    print(f"\nVerifying {output_file} internal structure...")
    with h5py.File(output_file, 'r') as f:
        v = f['VTKHDF']
        print(f" - Version: {v.attrs['Version']}")
        print(f" - Type: {v.attrs['Type']}")
        
        steps = v['Steps']
        print(f" - NSteps: {steps.attrs['NSteps']}")
        print(f" - Step Values: {list(steps['Values'])}")
        
        pd = v['PointData']
        print(f" - PointData Fields: {list(pd.keys())}")
        shape_data = pd['mode_shape_vec']
        print(f" - mode_shape_vec Shape: {shape_data.shape}")
        
        # Expected shape: (n_steps * n_points, 3) 
        # since StructuralResult collapses 6-DOF to 3-DOF for vectors
        expected_shape = (num_modes * num_points, 3)
        if shape_data.shape == expected_shape:
            print(" [OK] Shape is correct!")
        else:
            print(f" [FAIL] Expected shape {expected_shape}, got {shape_data.shape}")

if __name__ == "__main__":
    test_modal_export_logic()
