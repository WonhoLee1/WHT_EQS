# [WHTOOLS] Generalized FEM structural analysis API 사용 가이드

안녕하세요, **WHTOOLS**입니다. 
본 문서는 기계공학 및 소프트웨어 엔지니어링 지식을 바탕으로 설계된 **Generalized FEM API**의 사용법을 안내합니다. 본 API는 JAX 기반의 고성능 연산 엔진을 탑재하고 있으며, CAD 모델링과 연동된 직관적인 기하 기반 조건 부여를 핵심으로 합니다.

---

## 1. 개요 (Overview)

본 API는 복잡한 유한요소 해석 과정을 단순화하고, 엔지니어가 물리적 형상과 조건에만 집중할 수 있도록 설계되었습니다. 
주요 구성 요소는 다음과 같습니다:
- **`PlateFEM`**: 전처리(Preprocessing) 및 해석 실행을 담당하는 메인 클래스.
- **`StructuralResult`**: 해석 결과 데이터 관리 및 결과 보간(Interpolation)을 담당하는 결과 처리 클래스.

---

## 2. 주요 클래스 및 메서드

### 2.1. PlateFEM (해석 매니저)

#### 2.1.1. 인스턴스 생성
```python
from WHT_EQS_analysis import PlateFEM

# 방법 1: 사각형 Tray 메쉬(nx, ny)로 직접 생성
fem = PlateFEM(Lx=1000.0, Ly=500.0, nx=20, ny=10)

# 방법 2: CAD 파일(STEP/IGES)로부터 생성 (Gmsh 연동)
fem = PlateFEM.from_cad("resources/a.step", mesh_size=15.0)
```

#### 2.1.2. 경계 조건 및 하중 부여 (영역 기반)
> [!TIP]
> 엔지니어가 직접 노드 번호를 찾을 필요 없이, 공간 좌표 범위를 지정하여 다량의 노드에 동시 조건을 부여할 수 있습니다.

```python
# 영역(Bounding Box) 내 노드 고정
fem.add_constraint(x_range=(0, 10), dofs=[0,1,2,3,4,5], value=0.0)

# 특정 영역에 하중 분배 (is_total=True 시 전체 하중을 노드 수로 나누어 분배)
fem.add_force(x_range=(450, 550), y_range=(200, 300), dof=2, value=-500.0, is_total=True)
```

#### 2.1.3. CAD 기하 기반 조건 부여 (Advanced)
CAD 기반 모델링 시, Vertex, Edge, Face 정보를 직접 활용할 수 있습니다.

```python
# 특정 좌표에서 가장 가까운 Edge 찾기
nearest, dist = fem.find_nearest_entity(pos=[500, 0, 0], dim=1)
dim, tag = nearest

# 해당 Edge 전체에 구속 조건 부여
fem.add_constraint_on_entity(dim, tag, dofs=[0,1,2,3,4,5], value=0.0)
```

---

### 2.2. StructuralResult (결과 처리)

#### 2.2.1. 필드 데이터 조회
```python
# 노드 기반 결과 (Von-Mises Stress, Displacement 등)
vm_stress = result.get_nodal_result('stress_vm')
u_mag = result.get_nodal_result('u_mag')

# 요소 기반 결과
vm_stress_el = result.get_element_result('stress_vm')
```

#### 2.2.2. 좌표 기반 결과 보간 (Probe)
> [!IMPORTANT]
> 쉘(Shell) 구조의 특성을 반영하여, 입력 좌표에서 가장 가까운 표면으로 투영(Projection) 후 형상 함수를 이용해 정밀한 값을 산출합니다.

```python
# 특정 좌표 (100.5, 250.0, 50.0)에서의 Von-Mises 응력 확인
probe_val = result.get_value_at_point('stress_vm', 100.5, 250.0, 50.0)
print(f"Probe Result: {probe_val:.2f} MPa")
```

---

## 3. 예제 코드 (Example)

상세한 활용 예제는 다음 파일들을 참고해 주세요:
- [`example_fem_cantilever.py`](file:///C:/Users/GOODMAN/code_sheet/example_fem_cantilever.py): 영역 기반 BC/Load 기본.
- [`example_cad_mesh_analysis.py`](file:///C:/Users/GOODMAN/code_sheet/example_cad_mesh_analysis.py): CAD 통합 및 기하 엔티티 활용 고급 예제.

---

**WHTOOLS**는 엔지니어의 생산성을 최우선으로 생각합니다. 
본 도구를 통해 데이터 기반의 신속하고 정확한 의사결정을 내리시길 바랍니다. 

[^1]: **RBE2**: 강체 연결(Rigid Body Element)을 의미하며, 집중 하중 시 현장의 물리적 상태를 모사하는 데 사용됩니다.
