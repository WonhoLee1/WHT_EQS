# FEM 구조해석 일반화 및 CAD 통합 계획서 (RBE2/Gmsh/보간 포함)

현재 `generate_targets` 함수 내부에 파편화되어 있는 해석 과정을 일반화하여, 사용자 친화적인 FEM 해석 워크플로우를 구축합니다. 위치 기반 조건 부여, RBE2 강체 연결, CAD 기반 자동 메싱, 그리고 체계적인 결과 접근 및 보간(Interpolation) 기능을 포함합니다.

## User Review Required

> [!IMPORTANT]
> - 기존 `PlateFEM` 함수를 클래스로 전환하거나, 이를 래핑하는 `AnalyticalManager` 구조를 도입할 예정입니다. 기존 `main_shell_verification.py`에서의 호출 방식에 일부 변경이 생길 수 있습니다 (예: `fem = PlateFEM(...)` -> `problem = PlateFEM(...)`).
> - 주응력(Principal Stress) 및 주변형률(Principal Strain) 계산 로직이 `ShellFEM` 클래스 내부에 추가됩니다.
> - **RBE2 (Rigid Body Element)** 기능이 도입되어, 다수의 노드를 하나의 마스터 노드에 강체로 결합하고 중앙 집중 하중을 부여하는 기능이 추가됩니다.
> - **CAD-to-Mesh 통합**: STEP, IGES 등 CAD 파일을 읽어 `gmsh`를 통해 자동으로 메쉬(Triangle/Quad)를 생성하고 `ShellFEM`에서 즉시 해석에 사용하는 워크플로우를 추가합니다.

## Proposed Changes

### 1. `ShellFemSolver` 및 결과 처리 보강

#### [MODIFY] [shell_solver.py](file:///C:/Users/GOODMAN/code_sheet/ShellFemSolver/shell_solver.py)
- `compute_field_results` 함수를 확장하여 다음 물리량들을 계산하도록 수정합니다.
    - **Displacement**: Node별 (u, v, w) 성분 분리 제공
    - **Stress**: Principal Stresses ($\sigma_1, \sigma_2$), Von Mises Stress ($\sigma_{vm}$), Shear Stress ($\tau_{xy}$), Principal Directions
    - **Strain**: Principal Strains ($\epsilon_1, \epsilon_2$), Equivalent Strain ($\epsilon_{eq}$), Membrane/Bending 성분 분리
    - **RBE2 Support**: `ShellFEM` 어셈블리 과정에서 MPC(Multi-Point Constraint) 변환 행렬을 적용하여 강체 연결 구현
- 결과 데이터를 체계적으로 관리할 `StructuralResult` 클래스를 정의합니다.
    - `get_nodal_result(field_name)`: 노드 기반 데이터 반환
    - `get_element_result(field_name)`: 요소 기반 데이터 반환
    - `get_value_at_point(field_name, x, y, z)` [NEW]: 임의의 좌표에서 물리량을 보간(Interpolation)하여 반환하는 기능 제공

### 2. 고수준 API (`PlateFEM` 클래스화)

#### [MODIFY] [main_shell_verification.py](file:///C:/Users/GOODMAN/code_sheet/main_shell_verification.py)
- 기존 `def PlateFEM(...)`을 `class PlateFEM`으로 전환하여 고수준 API를 제공합니다.
- 다음 메서드들을 추가합니다.
    - `add_constraint(x_range=None, y_range=None, z_range=None, dofs=[0,1,2,3,4,5], value=0.0)`
    - `add_force(x_range=None, y_range=None, z_range=None, dof=2, value=-1.0, is_total=False)`
        - `is_total=True`: 입력된 값을 해당 영역 노드 수로 나누어 분배 (총 하중 개념)
        - `is_total=False`: 각 노드에 입력된 값을 그대로 부여 (압력/분포 하중 개념)
    - `add_rbe2(master_node_pos, slave_region_box=None, slave_node_ids=None)`
        - 마스터 노드 생성 및 하위 노드들과의 강체 연결 정의
        - 마스터 노드에 직접 하중을 부여하고 그에 대한 응답(변위-하중)을 추출하는 기능 제공
    - `from_cad(cad_path, mesh_size=20.0, element_type='quad', curvature_adaptation=True)` [Static Method]
        - Gmsh API를 활용하여 STEP/IGES 파일을 메싱하고 `PlateFEM` 인스턴스를 생성하여 반환
    - `solve_static(params)`: 내부적으로 `ShellFEM`의 어셈블리 및 솔버를 호출하고 `StructuralResult`를 반환
- `WHT_EQS_mesh.py`의 `get_nodes_in_box` 등을 활용하여 위치 기반 노드 검색을 수행합니다.

### 3. 유틸리티 및 데이터 구조 개선

#### [NEW] `dev_log/issue_tracker_2026-04-09.md` [NEW]
- 이번 작업을 포함하여 반복되는 이슈(인코딩, 최신 Mujoco/JAX 호환성 등)를 관리하기 위한 이슈 트래커 파일을 최신화합니다.

## Open Questions

- **보간 방식**: 3D 공간 상의 보간을 위해 `scipy.interpolate.LinearNDInterpolator`를 기본으로 사용할까요, 아니면 특정 투영(Projection) 기반의 보간을 선호하시나요?

## Verification Plan

### Automated Tests
- `test_generalized_fem.py` 스크립트를 생성하여 다음을 검증합니다.
    - **보간 테스트**: 특정 노드 사이의 무작위 좌표에서 물리량 값이 노드 값 사이로 정상 보간되는지 확인
    - **CAD-to-Mesh 테스트**: STEP 파일을 메싱하고 해석이 가능함을 확인
    - **RBE2 테스트**: 마스터 노드 하중에 따른 거동 정합성 확인

### Manual Verification
- `main_shell_verification.py`에서 보간 기능을 활용하여 특정 관심 지점(ROI)의 응답을 추출하고 확인합니다.
