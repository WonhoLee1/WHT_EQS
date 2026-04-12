# -*- coding: utf-8 -*-
"""
[WHTOOLS] 기초 예제 - 캔틸레버 보 해석 (Cantilever Beam Analysis)
영역(Bounding Box)을 이용한 경계 조건 및 하중 부여 방법 시연.
"""
import numpy as np
from WHT_EQS_analysis import PlateFEM, StructuralResult

def run_cantilever_example():
    print("\n" + "="*60)
    print(" [EXAMPLE] BASIC CANTILEVER BEAM ANALYSIS")
    print("="*60)

    # 1. 모델 생성 (1000x500mm 플레이트, 30x15 격자)
    Lx, Ly = 1000.0, 500.0
    nx, ny = 30, 15
    # Tray 메쉬 생성 (모든 노드 Z=0 기준)
    fem_prob = PlateFEM(Lx, Ly, nx, ny)
    print(f" -> 모델 생성 완료: {fem_prob.total_dof} 자유도")

    # 2. 경계 조건 부여 (영역 지정)
    # X 좌표 0~5mm 사이의 모든 노드를 전구속(Fixed)
    print(" -> 왼쪽 끝단 고정 조건 (X: 0~5mm) 부여")
    fem_prob.add_constraint(x_range=(0, 5), dofs=[0,1,2,3,4,5], value=0.0)

    # 3. 하중 조건 부여 (영역 지정)
    # X 좌표 950~1000mm 사이의 모든 노드에 총 하중 -1000N을 분배하여 부가
    print(" -> 오른쪽 끝단 집중 하중 (-1000N, Z방향) 부여")
    fem_prob.add_force(x_range=(950, 1000), dof=2, value=-1000.0, is_total=True)

    # 4. 정적 해석 실행
    params = {
        't': 2.0,           # 두께 2mm
        'E': 210000.0,      # 탄성계수 (Steel)
        'rho': 7.85e-9      # 밀도
    }
    print(" -> 해석 실행 중...")
    result = fem_prob.solve_static(params)
    print(" [OK] 해석 완료.")

    # 5. 결과 분석
    # 최대 변위 확인
    u_mag = result.get_nodal_result('u_mag')
    print(f"\n [결과 데이터 요약]")
    print(f" -> 최대 변위: {np.max(u_mag):.4f} mm")
    
    # 최대 등가 응력 확인
    vm_stress = result.get_nodal_result('stress_vm')
    print(f" -> 최대 Von-Mises 응력: {np.max(vm_stress):.2f} MPa")
    
    # 특정 좌표 (중앙부) 결과 보간 확인
    mid_stress = result.get_value_at_point('stress_vm', 500.0, 250.0, 0.0)
    print(f" -> 중앙 지점(500, 250) 보간 응력: {mid_stress:.2f} MPa")

    print("\n" + "="*60)
    print(" [SUCCESS] 예제 실행이 성공적으로 완료되었습니다.")
    print("="*60)

if __name__ == "__main__":
    run_cantilever_example()
