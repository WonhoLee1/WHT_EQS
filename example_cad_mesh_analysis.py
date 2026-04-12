# -*- coding: utf-8 -*-
"""
[WHTOOLS] 고급 예제 - CAD 기하 기반 엔티티 해석 (CAD Feature-based Analysis)
`resources/a.step` 파일을 정밀하게 제어하고 해석하는 방법 시연.
"""
import numpy as np
import matplotlib.pyplot as plt
from WHT_EQS_analysis import PlateFEM, StructuralResult

def run_cad_geometric_example():
    print("\n" + "="*70)
    print(" [EXAMPLE] ADVANCED CAD-INTEGRATED ANALYSIS (a.step)")
    print("="*70)

    # 1. STEP 파일로부터 모델 생성 (Gmsh 자동 연동)
    cad_path = "resources/a.step"
    mesh_size = 10.0 # 10mm 격자 크기 목표
    print(f" -> CAD 파일 로드 중... (경로: {cad_path})")
    fem_prob = PlateFEM.from_cad(cad_path, mesh_size=mesh_size, curvature_adaptation=True)
    
    bbox = fem_prob.get_cad_bbox()
    print(f" -> CAD Bounding Box: {bbox[0]:.2f}~{bbox[3]:.2f}, {bbox[1]:.2f}~{bbox[4]:.2f}, "
          f"{bbox[2]:.2f}~{bbox[5]:.2f}")
    print(f" -> 메쉬 생성 완료: {fem_prob.total_dof} 자유도")

    # 2. 기하 엔티티(Entity) 기반 조건 부여
    # (0, 0, 0) 근처의 하단 모서리(Edge)를 좌표로 자동 검색하여 고속으로 구속함
    print(" -> 하단 모서리(Edge) 검색 및 고정 조건 부여")
    nearest, dist = fem_prob.find_nearest_entity(pos=[0, 0, 0], dim=1) # dim=1: Edge
    if nearest:
        dim, tag = nearest
        print(f"    [검색결과] Edge Tag: {tag}, 검색 좌표와의 거리: {dist:.4e}mm")
        # 해당 모서리에 포함된 모든 노드에 고정 조건(Clamped) 부여
        fem_prob.add_constraint_on_entity(dim, tag, dofs=[0,1,2,3,4,5], value=0.0)
    
    # 3. 추가적인 기하 기반 하중 부여 (선택)
    # 다른 모서리에 X방향 하중 부여 시연
    f_nearest, f_dist = fem_prob.find_nearest_entity(pos=[18, 100, 100], dim=1)
    if f_nearest:
        print(f" -> 상단 모서리(Edge {f_nearest[1]})에 X방향 200N 하중 부여")
        fem_prob.add_force_on_entity(f_nearest[0], f_nearest[1], dof=0, value=200.0, is_total=True)

    # 4. 정적 해석 실행
    params = {
        't': 2.0,           # 두께 2mm
        'E': 210000.0,      # 탄성계수 (Steel)
        'rho': 7.85e-9      # 밀도
    }
    print(" -> 해석 실행 중... (JAX Sparse Solver)")
    result = fem_prob.solve_static(params)
    print(" [OK] 해석 완료.")

    # 5. 결과 분석 (Nodal & Probe)
    vm_stress = result.get_nodal_result('stress_vm')
    max_stress = np.max(vm_stress)
    print(f"\n [해석 결과 요약]")
    print(f" -> 최대 폰-미세스 응력: {max_stress:.2f} MPa")
    
    u_mag = result.get_nodal_result('u_mag')
    print(f" -> 최대 변위: {np.max(u_mag):.4f} mm")

    # [최고급 기능] 좌표 기반 결과 보간 (Probe)
    # 특정 관심 지점 (Center surface)에서의 응력 확인
    probe_x, probe_y, probe_z = 10.0, 50.0, 50.0
    val_at_center = result.get_value_at_point('stress_vm', probe_x, probe_y, probe_z)
    print(f" -> 지점({probe_x}, {probe_y}, {probe_z}) 보간 응력: {val_at_center:.2f} MPa")

    print("\n" + "="*70)
    print(" [SUCCESS] CAD 기반 해석 예제가 성공적으로 완료되었습니다.")
    print("="*70)

if __name__ == "__main__":
    run_cad_geometric_example()
