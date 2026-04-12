# -*- coding: utf-8 -*-
import numpy as np
from WHT_EQS_analysis import PlateFEM

def test_geometric_api():
    print("\n" + "="*50)
    print(" [TEST] CAD GEOMETRIC API VERIFICATION")
    print("="*50)

    # 1. Load CAD
    cad_path = "resources/a.step"
    print(f" -> Loading {cad_path}...")
    fem = PlateFEM.from_cad(cad_path, mesh_size=20.0)
    
    # 2. Check BBox
    bbox = fem.get_cad_bbox()
    print(f" -> CAD BBox: {bbox}")
    assert bbox is not None
    assert bbox[3] > bbox[0]
    print(" [OK] BBox Metadata Valid.")

    # 3. Find Nearest Entity
    # Corner of the box is roughly (0,0,0) as per BBOX inspection
    target_pos = [0, 0, 0]
    nearest, dist = fem.find_nearest_entity(target_pos, dim=0) # Search for Vertex
    print(f" -> Nearest Vertex to {target_pos}: {nearest}, dist={dist:.4e}")
    assert nearest is not None
    assert nearest[0] == 0
    print(" [OK] Nearest Search Functional.")

    # 4. Entity Nodes Mapping
    nodes = fem.get_entity_nodes(nearest[0], nearest[1])
    print(f" -> Nodes on Vertex {nearest[1]}: {nodes}")
    assert len(nodes) > 0
    print(" [OK] Node Mapping Functional.")

    # 5. Entity BC Application
    print(" -> Applying BC on Edge nearest to (10, 0, 0)")
    nearest_edge, d_edge = fem.find_nearest_entity([10, 0, 0], dim=1)
    print(f"    Found Edge: {nearest_edge}, dist={d_edge:.4e}")
    fem.add_constraint_on_entity(nearest_edge[0], nearest_edge[1], dofs=[0,1,2], value=0.0)
    
    # Check if constraints were added
    edge_nodes = fem.get_entity_nodes(nearest_edge[0], nearest_edge[1])
    added_dofs = [c[0] for c in fem.constraints]
    for node in edge_nodes:
        assert node*6 + 0 in added_dofs
        assert node*6 + 1 in added_dofs
        assert node*6 + 2 in added_dofs
    print(" [OK] Entity-based BC Application Verified.")

    print("\n" + "="*50)
    print(" [SUCCESS] CAD Geometric API verification passed.")
    print("="*50)

if __name__ == "__main__":
    test_geometric_api()
