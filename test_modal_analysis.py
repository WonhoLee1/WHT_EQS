# -*- coding: utf-8 -*-
import numpy as np
from WHT_EQS_analysis import PlateFEM

def test_modal_logic():
    print("\n" + "="*50)
    print(" [TEST] MODAL ANALYSIS LOGIC VERIFICATION")
    print("="*50)

    # 1. Simple Beam-like Plate (100x20)
    Lx, Ly = 100.0, 20.0
    nx, ny = 20, 4
    fem = PlateFEM(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    
    # 2. Clamped-Free (Cantilever)
    # Fix nodes at x=0
    fem.add_constraint(x_range=(0, 0.1), y_range=None, z_range=None, dofs=[0,1,2,3,4,5], value=0.0)
    
    # 3. Solve Eigen
    params = {'t': 2.0, 'E': 210000.0, 'rho': 7.85e-9}
    n_modes = 5
    result = fem.solve_eigen(params, n_modes=n_modes)
    
    # 4. Verification
    print("\n [해석 결과 확인]")
    freqs = []
    for field in sorted(result.results.keys()):
        if field.startswith("Mode_") and not field.endswith("_vec"):
            # Extract frequency from name "Mode_01_125.4Hz"
            f_hz = float(field.split('_')[2].replace('Hz', ''))
            freqs.append(f_hz)
            print(f" -> {field}: {f_hz:.2f} Hz")
            
    assert len(freqs) == n_modes
    assert np.all(np.diff(freqs) >= 0) # Frequencies must be non-decreasing
    print(" [OK] Frequencies are non-decreasing.")
    
    # Mode shape normalization check
    for field in result.results.keys():
        if field.endswith("_vec"):
            mode_max = np.max(np.abs(result.results[field]))
            print(f" -> {field} Max Amp: {mode_max:.4f}")
            assert np.isclose(mode_max, 1.0, atol=1e-5)
    print(" [OK] Mode shapes are correctly normalized.")

    print("\n" + "="*50)
    print(" [SUCCESS] Modal analysis verification passed.")
    print("="*50)

if __name__ == "__main__":
    test_modal_logic()
