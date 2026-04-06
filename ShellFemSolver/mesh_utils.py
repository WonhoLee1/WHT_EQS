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

def generate_tray_mesh_quads(Lx_outer, Ly_outer, wall_width, wall_height, nx, ny, mode='sloped'):
    """
    도시락(Box/Tray) 형태의 5면체 구조 메쉬를 생성합니다.
    
    Args:
        Lx_outer, Ly_outer: 전체 외곽 치수
        wall_width: 테두리 영역의 폭 (X, Y 방향)
        wall_height: 올라오는 높이
        nx, ny: 메쉬 분할 수
        mode: 'sloped' (완만한 경사) 또는 'vertical' (날카로운 수직벽)
    """
    import numpy as np
    dx = Lx_outer / nx
    dy = Ly_outer / ny
    
    nodes = []
    
    for j in range(ny + 1):
        y = j * dy
        for i in range(nx + 1):
            x = i * dx
            
            # 1. 외곽으로부터의 거리 산출
            dist_x = 0.0
            if x < wall_width: dist_x = wall_width - x
            elif x > Lx_outer - wall_width: dist_x = x - (Lx_outer - wall_width)
            
            dist_y = 0.0
            if y < wall_width: dist_y = wall_width - y
            elif y > Ly_outer - wall_width: dist_y = y - (Ly_outer - wall_width)
            
            dist = max(dist_x, dist_y)
            
            # 2. 모드에 따른 높이(Z) 할당
            if mode == 'vertical':
                # transition_width(2mm) 이내로 급격히 상승시켜 수직벽 모사
                transition = min(2.0, wall_width * 0.1) if wall_width > 0 else 0.5
                ratio = min(1.0, dist / transition) if transition > 0 else 1.0
            else:
                # wall_width 전체를 경사면으로 활용 (기본값)
                ratio = min(1.0, dist / wall_width) if wall_width > 0 else 0.0
                
            z = wall_height * ratio
            nodes.append([x, y, z])
            
    nodes = np.array(nodes)
    z_min, z_max = np.min(nodes[:, 2]), np.max(nodes[:, 2])
    print(f" -> Mesh Nodes Z-Range: {z_min:.2f}mm to {z_max:.2f}mm")
    
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n3 = n2 + (nx + 1)
            n4 = n1 + (nx + 1)
            elements.append([n1, n2, n3, n4])
    elements = np.array(elements)
    
    print(f'[MeshGen] Tray/Box {Lx_outer}x{Ly_outer} (H={wall_height}) split into {len(nodes)} Nodes and {len(elements)} Quads.')
    return nodes, elements
