# -*- coding: utf-8 -*-
"""
[WHTOOLS] 고유 진동수(Modal Analysis) 해석 데모
CAD 모델의 고유 진동수 및 모드 형상을 산출하고 ParaView로 내보내는 과정을 시연합니다.
"""
from WHT_EQS_analysis import PlateFEM

def run_modal_analysis_demo():
    print("\n" + "="*70)
    print(" [DEMO] CAD MODAL ANALYSIS (NATURAL FREQUENCIES)")
    print("="*70)

    # 1. 모델 생성 및 메싱 (resources/a.step)
    cad_path = "resources/a.step"
    print(f" -> CAD 모델 로드 및 메싱 중: {cad_path}")
    fem = PlateFEM.from_cad(cad_path, mesh_size=10.0)
    
    # 2. 구속 조건 설정 (고유진동수 해석을 위한 경계 조건)
    # 하단(Y=0)을 완전히 고정한 Cantilever 상태로 해석
    nearest, _ = fem.find_nearest_entity(pos=[0, 0, 0], dim=1)
    if nearest:
        print(f" -> 경계 조건: 하단 Edge {nearest[1]} 고정 (Cantilever)")
        fem.add_constraint_on_entity(nearest[0], nearest[1], dofs=[0,1,2,3,4,5], value=0.0)

    # 3. 고유 진동수 해석 실행 (10차 모드까지)
    params = {
        't': 2.0,           # 두께 2mm
        'E': 210000.0,      # 탄성계수 (Steel)
        'rho': 7.85e-9      # 밀도
    }
    
    num_modes = 10
    result = fem.solve_eigen(params, n_modes=num_modes)
    
    # 4. 결과 저장 및 ParaView 실행
    output_file = "modal_results.vtkhdf"
    print(f"\n -> 10차 모드 형상 저장 중: {output_file}")
    result.save_vtkhdf(output_file)
    
    print("\n" + "*"*70)
    print(" [MAGIC] ParaView에서 진동 모드를 확인하세요.")
    print(" Tip: ParaView의 필드 선택기에서 'Mode_01...', 'Mode_02...' 등을 선택하여 볼 수 있습니다.")
    print("*"*70)
    
    result.open_paraview()

    print("\n" + "="*70)
    print(" [SUCCESS] 고유 진동수 해석 데모가 완료되었습니다.")
    print("="*70)

if __name__ == "__main__":
    run_modal_analysis_demo()
