# -*- coding: utf-8 -*-
"""
[WHTOOLS] 통합 CAE 포스트프로세싱 데모 (Analysis-to-ParaView Workflow)
CAD 모델 해석부터 VTKHDF 저장 및 ParaView 자동 실행까지의 전 과정을 시연합니다.
"""
import os
from WHT_EQS_analysis import PlateFEM

def run_cae_workflow_demo():
    print("\n" + "="*70)
    print(" [DEMO] END-TO-END CAE WORKFLOW & POST-PROCESSING")
    print("="*70)

    # 1. 모델 생성 및 메싱 (resources/a.step 활용)
    cad_path = "resources/a.step"
    if not os.path.exists(cad_path):
        print(f"[ERROR] CAD 파일을 찾을 수 없습니다: {cad_path}")
        return

    print(f" -> CAD 모델 로드 및 자동 메싱 중: {cad_path}")
    fem = PlateFEM.from_cad(cad_path, mesh_size=10.0)
    
    # 2. 경계 조건 및 하중 설정 (기하 기반)
    # 하단(Y=0) 고정
    nearest, _ = fem.find_nearest_entity(pos=[0, 0, 0], dim=1)
    if nearest:
        print(f" -> 경계 조건 부여: 하단 Edge {nearest[1]} 고정")
        fem.add_constraint_on_entity(nearest[0], nearest[1], dofs=[0,1,2,3,4,5], value=0.0)
    
    # 상단(Y=100) 하중
    f_nearest, _ = fem.find_nearest_entity(pos=[18, 100, 100], dim=1)
    if f_nearest:
        print(f" -> 하중 조건 부여: 상단 Edge {f_nearest[1]}에 X-하중 500N")
        fem.add_force_on_entity(f_nearest[0], f_nearest[1], dof=0, value=500.0, is_total=True)

    # 3. 정적 해석 실행
    params = {'t': 2.0, 'E': 210000.0, 'rho': 7.85e-9}
    print(" -> 해석 실행 중 (JAX Solver)...")
    result = fem.solve_static(params)
    print(" [OK] 해석 완료.")

    # 4. 현대적 포스트프로세싱 (VTKHDF & ParaView)
    output_hdf = "output_result.vtkhdf"
    output_glb = "output_presentation.glb"
    
    print(f"\n -> 필드 데이터 저장 중: {output_hdf} (VTKHDF 1.0)")
    result.save_vtkhdf(output_hdf)
    
    print(f" -> 프리미엄 시각화 파일 생성 중: {output_glb} (GLB)")
    result.save_glb(output_glb)

    print("\n" + "*"*70)
    print(" [MAGIC] ParaView 자동 실행 및 결과 로드")
    print("*"*70)
    
    # ParaView 자동 실행 연계 (설치된 경우)
    result.open_paraview()

    print("\n" + "="*70)
    print(" [SUCCESS] CAE 워크플로우 데모가 완료되었습니다.")
    print(" [INFO] 생성된 파일:")
    print(f"   - {output_hdf} (ParaView 정밀 분석용)")
    print(f"   - {output_glb} (3D 뷰어/브리핑 공유용)")
    print("="*70)

if __name__ == "__main__":
    run_cae_workflow_demo()
