# MITC Shell Theory Upgrade Guide for JAX

This document contains the core mathematical and structural concepts required to upgrade the current DKT-based shell solver to the **MITC (Mixed Interpolation of Tensorial Components)** formulation. 이 문서는 향후 Gemini 3.0 Flash 또는 다른 에이전트가 코드를 작성할 때 즉시 참고할 수 있는 핵심 알고리즘 가이드입니다.

## 1. 개요 (Overview)
*   **목적**: Tray 형상이나 복잡한 곡면에서 발생하는 수치적 불안정성(가짜 강체 모드 등) 및 Shear Locking을 해결하기 위함.
*   **이론적 기반**: Mindlin-Reissner 쉘 이론 + MITC 전단 변형률 보간법 (Bathe & Dvorkin).
*   **자유도(DOFs)**: 노드당 6자유도 ($u, v, w, \theta_x, \theta_y, \theta_z$). 

## 2. 핵심 수학적 정식화 (MITC Formulation)

### 2.1. 독립적인 변위장 및 회전장
Mindlin 쉘에서는 수직 처짐($w$)과 회전장($\theta_x, \theta_y$)이 독립적입니다.
*   전단 변형률 (Transverse Shear Strains): $\gamma_{xz} = \frac{\partial w}{\partial x} + \theta_y$, $\gamma_{yz} = \frac{\partial w}{\partial y} - \theta_x$
*   **Locking 문제**: 요소가 얇아지면 $\gamma \rightarrow 0$을 만족해야 하는데, 저차 요소에서는 이 구속 조건을 수치적으로 완벽히 표현할 수 없어 요소가 비정상적으로 뻣뻣해집니다.

### 2.2. MITC 보간법 (The MITC Cure)
Shear locking을 피하기 위해, 전단 변형률을 면 내의 특정 지점(Tying Points)에서만 직접 계산한 뒤, 이 값들을 요소 전체로 **혼합 보간(Mixed Interpolation)**합니다.

#### MITC3 (3절점 삼각형 요소)
*   **Tying Points**: 세 변의 중점 (Mid-points of the 3 edges).
*   각 중점에서 가장자리(Edge) 방향으로의 접선 전단 변형률(Tangential shear strain)을 평가합니다.
*   수식적 접근 (JAX 구현 시):
    1.  각 변 $k=1,2,3$ 의 방향 벡터 $\mathbf{s}_k$ 계산.
    2.  각 중점에서 가장자리 전단 변형률 $\gamma_k = (\nabla w + \mathbf{\theta}) \cdot \mathbf{s}_k$ 계산.
    3.  요소 내의 전단 변형률 $\gamma_{xz}, \gamma_{yz}$를 $\gamma_1, \gamma_2, \gamma_3$로부터 보간하는 전용 행렬 $B_s^{MITC}$ 구성.

#### MITC4 (4절점 사각형 요소)
*   **Tying Points**: 4개 변의 중점 (A, B, C, D).
*   $\xi$ 방향 가장자리 중점에서는 $\gamma_{\xi z}$, $\eta$ 방향 가장자리 중점에서는 $\gamma_{\eta z}$만 평가.
*   $\gamma_{\xi z}$는 $\eta$ 방향에 대해 상수로, $\gamma_{\eta z}$는 $\xi$ 방향에 대해 상수로 보간.

## 3. JAX 구현 시 핵심 고려사항 (JAX Implementation Notes)

1.  **Fully Vectorized $B_s$ Matrix**:
    기존의 DKT 벤딩 $B_b$와 멤브레인 $B_m$ 외에, 완전히 독립적인 $B_s$ (전단) 매트릭스 생성 함수를 추가해야 합니다.
    ```python
    @jit
    def _compute_Bs_mitc3(nodes):
        # 1. 3개 변의 벡터 및 길이 계산
        # 2. 형상 함수를 이용한 Edge Shear Strain Nodal 행렬 생성
        # 3. 보간 행렬(Interpolation Matrix)을 곱하여 최종 Bs 도출
        pass
    ```

2.  **적분점 (Integration Points)**:
    *   **굽힘 및 막(Membrane/Bending)**: T3의 경우 3점 적분, Q4의 경우 2x2 Gauss 직교 적용.
    *   **전단 (Shear)**: MITC를 사용하므로 Full Integration을 사용해도 Shear Locking이 발생하지 않음. 따라서 T3는 1점 또는 3점, Q4는 2x2 유지 가능. (Reduced Integration 불필요)

3.  **드릴링 자유도 ($\theta_z$) 강화**:
    단순 페널티($10^{-4}$) 대신, 요소의 형상 자코비안을 반영한 Allman Formulation이나 MacNeal-Harder 페널티 스케일을 사용하여 물리적으로 더 타당한 평면 회전 강성을 제공해야 합니다.
    ```python
    # 기존 단순 상수가 아닌, 요소 특성(두께, 면적, 전단 계수 G)을 반영한 자동 스케일링
    drilling_stiffness = alpha * G * V # V = area * thickness
    ```

4.  **Local to Global Transformation**:
    Tray 형태(3D 일반 쉘)이므로 기존에 구현된 $T$ (방향 여현 행렬, Local-to-Global 변환)은 유지되며, 6x6 노드 변환이 새로운 $K_{local}$ (18x18 for T3, 24x24 for Q4)에 그대로 적용됩니다.

## 4. 진행 순서 (Workflow for the Agent)

1.  `shell_solver.py` 내부에 `_compute_Bs_mitc3`, `_compute_Bs_mitc4` 헬퍼 함수 작성 (수식 최적화 및 `jax.jit` 호환성 확보).
2.  `compute_mitc3_local` 및 `compute_mitc4_local` 함수 뼈대 작성. (막 강성, 굽힘 강성, 전단 강성 통합).
3.  기존 `assemble` 로직 내에서 옵션 (ex: `element_type='mitc'`) 형태 또는 전면 교체로 통합.
4.  평판 벤치마크 수행 (DKT와 비교하여 변위 해가 일치하는지 검증).
5.  Tray 형상에서 Eigenvalue ($>0$ Hz) 및 0Hz 6개 강체 모드 명확성 확인.
