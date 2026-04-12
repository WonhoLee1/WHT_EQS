# VTKHDF 1.0 Implementation Guide for ParaView Compatibility (Backup: 2026-04-10)

본 문서는 **WHTOOLS** 프로젝트 진행 중 터득한 JAX/Python 기반의 **VTKHDF 1.0 (UnstructuredGrid)** 포맷 구현 핵심 노하우를 정리한 것입니다. 이 가이드는 향후 유사한 CAE 포스트프로세싱 파이프라인 구축 시 AI 에이전트와 개발자가 오류 없이 즉시 구현할 수 있도록 돕는 "Source of Truth" 역할을 합니다.

---

## 1. 식별 헤더 (Identity & Type)

ParaView가 파일을 리서(Reader) 없이 자동으로 인식하게 하려면 `/VTKHDF` 그룹에 식별 정보를 **중복 기록**해야 합니다.

### 1.1. Attributes (ParaView Discovery용)
HDF5 그룹 속성으로 다음 정보를 기록합니다.
- `Version`: `[1, 0]` (int32 배열)
- `Type`: `"UnstructuredGrid"` (ASCII 문자열)
  > [!IMPORTANT]
  > `Type`은 반드시 UTF-16이 아닌 **ASCII(Fixed-length)**로 기록해야 합니다. `h5py`에서 `np.bytes_("UnstructuredGrid")`를 활용하십시오.

### 1.2. Datasets (VTK Spec 준수용)
그룹 내부 데이터셋으로도 동일 정보를 기록합니다.
- `Version`: 데이터셋 (Shape: (2,), dtype: int32)
- `Type`: 데이터셋 (Shape: (), dtype: fixed-length ASCII string)

---

## 2. 메타데이터 차원의 함정 (The 1D Array Rule)

ParaView의 `vtkHDFReader`는 고정 크기 메타데이터에 대해 매우 엄격합니다.

- **잘못된 예**: `NumberOfPoints = 147` (Scalar)
- **올바른 예**: `NumberOfPoints = [147]` (Shape: (1,), dtype: int64)

### 필수 메타데이터 목록 (UnstructuredGrid)
1. `NumberOfPoints`: 총 노드 수
2. `NumberOfCells`: 총 요소 수
3. `NumberOfConnectivityIds`: Connectivity 배열의 총 길이 (누락 시 Reader 오류 발생)

---

## 3. 메쉬 데이터 구조 (Mesh Data)

### 3.1. Points
- **Shape**: `(N, 3)`
- **Dtype**: `float64` (float32도 가능하나 정밀도를 위해 64 권장)

### 3.2. Connectivity & Offsets
- **Connectivity**: 모든 요소의 노드 인덱스를 일렬로 나열한 배열. (0-based 인덱스 사용 필수)
- **Offsets**: 각 요소의 끝 지점을 가리키는 인덱스 배열. (첫 값은 항상 0이 아니며, 첫 요소의 끝 지점부터 기록)
  - 예: 삼각형(3)과 사각형(4)이 있다면, Offsets는 `[3, 7, ...]`
- **Types**: VTK 셀 타입 번호 (uint8)
  - `5`: VTK_TRIANGLE
  - `9`: VTK_QUAD

---

## 4. 윈도우 환경 및 시각화 자동화

### 4.1. Win32 파일 잠금 (File Locking)
ParaView가 파일을 열고 있는 동안 `h5py`로 덮어쓰기를 시도하면 `OSError (GetLastError=33)`가 발생합니다.
- **해결책**: 파일 쓰기 시 `try-except` 루프를 돌며 파일명에 접미사(예: `_1`, `_2`)를 붙여 활성 파일을 회피하십시오.

### 4.2. "Zero-Click" 자동화 스크립트
단순히 파일을 파라미터로 넘기는 대신, `--script=auto.py` 방식을 사용하면 사용자의 클릭 없이 즉시 시각화가 가능합니다.

**추천 스크립트 패턴 (`paraview.simple`):**
```python
from paraview.simple import *
reader = OpenDataFile('path/to/result.vtkhdf')
UpdatePipeline()
view = GetActiveViewOrCreate('RenderView')
display = Show(reader, view)
display.Representation = 'Surface With Edges'
ColorBy(display, ('POINTS', 'stress_vm'))
ResetCamera()
Render()
```

---

## 5. 결론 및 체크리스트
- [ ] `Type`이 ASCII 문자열인가?
- [ ] `NumberOfConnectivityIds`가 포함되었는가?
- [ ] 모든 크기 정보(Number of...)가 1D 배열인가?
- [ ] ParaView가 켜져 있어도 저장이 가능한가?

이 노하우를 준수하면 어떤 JAX 프로젝트에서도 **프리미엄급 CAE 포스트프로세서**를 즉시 구축할 수 있습니다.
