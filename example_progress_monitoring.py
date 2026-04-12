# -*- coding: utf-8 -*-
"""
[WHTOOLS] 진행 상황 모니터링 데모 (Progress Monitoring Demo)
고밀도 메쉬 해석 시 진행 막대(tqdm)와 솔버 잔차 출력을 시연합니다.
"""
from WHT_EQS_analysis import PlateFEM

def run_progress_demo():
    print("\n" + "="*70)
    print(" [DEMO] PROGRESS MONITORING & ITERATIVE SOLVER")
    print("="*70)

    # 1. 고밀도 모델 생성 (진행 막대 확인을 위해 mesh_size를 작게 설정)
    cad_path = "resources/a.step"
    print(f" -> CAD 모델 로드 및 고밀도 메싱 중: {cad_path}")
    fem = PlateFEM.from_cad(cad_path, mesh_size=3.0) 
    
    # 2. 경계 조건
    nearest, _ = fem.find_nearest_entity(pos=[0, 0, 0], dim=1)
    if nearest:
        fem.add_constraint_on_entity(nearest[0], nearest[1], dofs=[0,1,2,3,4,5], value=0.0)
    
    f_nearest, _ = fem.find_nearest_entity(pos=[18, 100, 100], dim=1)
    if f_nearest:
        fem.add_force_on_entity(f_nearest[0], f_nearest[1], dof=0, value=1000.0, is_total=True)

    # 3. 진행 상황이 표시되는 해석 실행
    # method='cg'를 사용하여 반복법 솔버의 잔차 변화를 확인합니다.
    params = {'t': 2.0, 'E': 210000.0, 'rho': 7.85e-9}
    
    print("\n" + ">"*30 + " [STAGE] ANALYSIS IN PROGRESS " + "<"*30)
    result = fem.solve_static(params, method='cg')
    print(">"*70 + "\n")

    # 4. 결과 출력
    print(f" [OK] 해석이 성공적으로 완료되었습니다. (Nodes: {len(fem.nodes)})")
    
    # 시각화 (선택 사항)
    # result.save_vtkhdf("progress_demo.vtkhdf")

if __name__ == "__main__":
    run_progress_demo()
