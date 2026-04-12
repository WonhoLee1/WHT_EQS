# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os

def verify_vtkhdf_structure(filepath):
    print("\n" + "="*50)
    print(f" [VERIFY] VTKHDF STRUCTURE: {filepath}")
    print("="*50)
    
    with h5py.File(filepath, 'r') as f:
        # 1. Header Checks
        vtkhdf = f['VTKHDF']
        
        # Check Attributes (Identity for ParaView)
        v_attr = vtkhdf.attrs['Version']
        t_attr = vtkhdf.attrs['Type']
        if isinstance(t_attr, bytes): t_attr = t_attr.decode('utf-8')
        print(f" -> Attributes: Version={v_attr}, Type={t_attr}")
        assert np.array_equal(v_attr, [1, 0])
        assert t_attr == "UnstructuredGrid"
        
        # Check Datasets (Data Spec)
        v_ds = vtkhdf['Version'][:]
        t_ds = vtkhdf['Type'][()].decode('utf-8')
        print(f" -> Datasets: Version={v_ds}, Type={t_ds}")
        assert np.array_equal(v_ds, [1, 0])
        assert t_ds == "UnstructuredGrid"
        
        # 2. Topology Checks
        # MUST BE 1D ARRAYS of size 1
        num_points_ds = vtkhdf['NumberOfPoints'][:]
        num_cells_ds = vtkhdf['NumberOfCells'][:]
        print(f" -> NumberOfPoints Dataset: {num_points_ds} (Shape: {num_points_ds.shape})")
        
        num_points = num_points_ds[0]
        num_cells = num_cells_ds[0]
        print(f" -> Evaluated Points: {num_points}, Cells: {num_cells}")
        
        assert num_points_ds.ndim == 1
        assert num_cells_ds.ndim == 1
        
        offsets = vtkhdf['Offsets'][:]
        types = vtkhdf['Types'][:]
        print(f" -> Offsets Size: {len(offsets)}, Types Size: {len(types)}")
        assert len(offsets) == num_cells + 1
        assert len(types) == num_cells
        
        # 3. Data Integrity
        pts = vtkhdf['Points'][:]
        assert pts.shape == (num_points, 3)
        
        pd = vtkhdf['PointData']
        print(" -> Fields in PointData:")
        for field in pd.keys():
            data = pd[field][:]
            print(f"   - {field}: {data.shape}")
            assert data.shape[0] == num_points
            
    print("\n" + "="*50)
    print(" [SUCCESS] VTKHDF structure is valid.")
    print("="*50)

if __name__ == "__main__":
    # Use a unique filename for verification to avoid locking issues with ParaView
    verify_file = "verify_result.vtkhdf"
    
    from WHT_EQS_analysis import PlateFEM
    fem = PlateFEM.from_cad("resources/a.step", mesh_size=12.0)
    fem.add_constraint_on_entity(1, 5, dofs=[0,1,2,3,4,5], value=0.0) # Fixed bottom
    res = fem.solve_static({'t': 2.0, 'E': 210000.0, 'rho': 7.85e-9})
    res.save_vtkhdf(verify_file)
    
    if os.path.exists(verify_file):
        verify_vtkhdf_structure(verify_file)
    else:
        print("[ERROR] Verification VTKHDF file not found.")
