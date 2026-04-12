
from paraview.simple import *
# 1. Load Data
reader = OpenDataFile('C:/Users/GOODMAN/code_sheet/output_result.vtkhdf')
if not reader:
    print('Failed to open file')
else:
    UpdatePipeline()
    # 2. Setup View
    view = GetActiveViewOrCreate('RenderView')
    view.Background = [1, 1, 1] # Clean White
    
    # --- BEFORE: Original Mesh (Wireframe) ---
    original_display = Show(reader, view)
    original_display.Representation = 'Wireframe'
    original_display.DiffuseColor = [0.7, 0.7, 0.7] # Subtle Gray
    original_display.Opacity = 0.2

    # --- AFTER: Deformed Mesh (Warp + Surface) ---
    if 'displacement_vec':
        warp = WarpByVector(Input=reader)
        warp.Vectors = ['POINTS', 'displacement_vec']
        # Auto-scale: roughly 10% of bbox size
        warp.ScaleFactor = 20.0 # Default factor for visual wow
        
        UpdatePipeline()
        display = Show(warp, view)
        display.Representation = 'Surface With Edges'
        
        # 3. Coloring
        if 'stress_vm':
            ColorBy(display, ('POINTS', 'stress_vm'))
            lut = GetColorTransferFunction('stress_vm')
            lut.ApplyPreset('Cool to Warm', (True))
            display.SetScalarBarVisibility(view, True)
    else:
        # Fallback to simple show if no vectors
        display = Show(reader, view)
        display.Representation = 'Surface With Edges'

    # 4. Camera & Render
    ResetCamera()
    Render()
