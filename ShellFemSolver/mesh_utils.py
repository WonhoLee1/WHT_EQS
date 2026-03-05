import numpy as np

def generate_rect_mesh_triangles(Lx, Ly, nx, ny):
    """
    평판(Plate)을 사각형이 아닌 3절점 삼각형(Triangle) Shell 요소로 분할(Meshing)합니다.
    기존 PlateFEM과 동일한 공간(Lx, Ly)을 커버하되, 내부 격자를 구성합니다.
    
    Returns:
        nodes: (N, 3) 3D coordinate array
        elements: (E, 3) connectivity array
    """
    dx = Lx / nx
    dy = Ly / ny
    
    # 1. 노드(Nodes) 생성
    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([i * dx, j * dy, 0.0])
    nodes = np.array(nodes)
    
    # 2. 요소(Elements) 생성 (사각형 1개당 대각선 분할 2개의 삼각형)
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n3 = n1 + (nx + 1)
            n4 = n3 + 1
            
            # 2 Triangles per Rectangle cell
            # 삼각형 1: (Bottom-Left, Bottom-Right, Top-Left)
            elements.append([n1, n2, n3])
            # 삼각형 2: (Bottom-Right, Top-Right, Top-Left)
            elements.append([n2, n4, n3])
            
    elements = np.array(elements)
    
    print(f"[MeshGen] Rectangle {Lx}x{Ly} split into {len(nodes)} Nodes and {len(elements)} Triangles.")
    return nodes, elements


def generate_rect_mesh_quads(Lx, Ly, nx, ny):
    import numpy as np
    dx = Lx / nx
    dy = Ly / ny
    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([i * dx, j * dy, 0.0])
    nodes = np.array(nodes)
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n3 = n2 + (nx + 1)
            n4 = n1 + (nx + 1)
            elements.append([n1, n2, n3, n4])
    elements = np.array(elements)
    print(f'[MeshGen] Rectangle {Lx}x{Ly} split into {len(nodes)} Nodes and {len(elements)} Quads.')
    return nodes, elements
