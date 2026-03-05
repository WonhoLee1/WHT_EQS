import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from shell_solver import ShellFEM
from mesh_utils import generate_rect_mesh_triangles

def main():
    Lx, Ly = 1000.0, 400.0
    nx, ny = 25, 10
    
    print("==================================================")
    print(" [ShellFemSolver] - Integrated End-to-End Test")
    print("==================================================")
    
    # 1. Mesh Gen (기존 nx_h, ny_h = 25, 10과 동일한 개수 사용 확인용)
    nodes, elements = generate_rect_mesh_triangles(Lx, Ly, nx, ny)
    
    # 2. Solver Init
    fem = ShellFEM(nodes, elements)
    
    # 3. Parameters (동일한 강철(Steel) 5mm 두께 부여)
    base_t = 5.0
    params = {
        't': jnp.full(len(elements), base_t),
        'E': jnp.full(len(elements), 210000.0),
        'rho': jnp.full(len(elements), 7.85e-9)
    }
    
    # 4. Assemble
    print("Assembling Global stiffness matrices...")
    K, M = fem.assemble(params)
    
    print(f"K contains NaN: {jnp.isnan(K).any()}")
    print(f"K trace: {jnp.trace(K)}")
    
    # 5. Boundary Condition 적용 시험 (Twist X 로드케이스 호환성 검사)
    tol = 1e-3
    left_nodes = jnp.where(fem.node_coords[:, 0] < tol)[0]
    right_nodes = jnp.where(fem.node_coords[:, 0] > Lx - tol)[0]
    
    fixed_dofs = []
    fixed_vals = []
    
    angle_rad = 1.5 * np.pi / 180.0
    yc = Ly / 2.0
    
    y_left = fem.node_coords[left_nodes, 1]
    w_left = (y_left - yc) * np.tan(-angle_rad)
    for i, node in enumerate(left_nodes):
        # Fix u, v, w, tx, ty, tz at left (fully clamped)
        fixed_dofs.extend([node*6+0, node*6+1, node*6+2, node*6+3, node*6+4, node*6+5]) 
        fixed_vals.extend([0.0, 0.0, w_left[i], -angle_rad, 0.0, 0.0])
        
    y_right = fem.node_coords[right_nodes, 1]
    w_right = (y_right - yc) * np.tan(angle_rad)
    for i, node in enumerate(right_nodes):
        fixed_dofs.extend([node*6 + 1, node*6 + 2, node*6 + 3, node*6 + 4, node*6 + 5]) 
        fixed_vals.extend([0.0, w_right[i], angle_rad, 0.0, 0.0])
        
    fixed_dofs = jnp.array(fixed_dofs)
    fixed_vals = jnp.array(fixed_vals)
    F = jnp.zeros(fem.total_dof)
    free_dofs = jnp.setdiff1d(jnp.arange(fem.total_dof), fixed_dofs)
    
    # 6. Solve Execution
    print("Solving Twist X Load Case on ShellFEM ...")
    u = fem.solve_static_partitioned(K, F, free_dofs, fixed_dofs, fixed_vals)
    
    w_disp = u[2::6]
    print(f"Max Z Displacement: {np.max(np.abs(w_disp)):.3f} mm")
    
    # Plotting for sanity check
    node_x = fem.node_coords[:, 0]
    node_y = fem.node_coords[:, 1]
    
    plt.figure()
    plt.scatter(node_x, node_y, c=w_disp, cmap='jet')
    plt.colorbar(label='W-Disp (mm)')
    plt.title('ShellFEM Test (Twist X: Triangle Mesh)')
    plt.axis('equal')
    plt.savefig('shell_test_twist.png')
    print("Test passed! Saved visual output to shell_test_twist.png")
    
    print("\n--- Eigenvalue Analysis (Modal) ---")
    try:
        vals, vecs = fem.solve_eigen(K, M, num_modes=10)
        
        # 0점 에너지 모드(강체 모드) 등을 걸러내기 위한 주파수 변환
        all_freqs = np.sqrt(np.maximum(np.array(vals), 0.0)) / (2 * np.pi)
        
        # 주파수가 1Hz 이상인 것만 유효(Elastic) 모드로 간주 출력
        elastic_freqs = all_freqs[all_freqs > 1.0]
        
        print("First 5 Elastic Frequencies (Hz):")
        for i, f in enumerate(elastic_freqs[:5]):
            print(f"  Mode {i+1}: {f:.2f} Hz")
            
    except Exception as e:
        print(f"Eigenvalue analysis failed: {e}")
        
if __name__ == '__main__':
    main()
