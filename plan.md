너는 기계공학자, 시뮬레이션 전문가, 인공지능 프로그래머 이다.

sheet metal된 3D surface CAD 파트 모델이 있다.
이 모델에 대해서 어떤 정해진 하중 및 경계 조건에 대해서 변형의 형태를 알고 있다.
더불어 이 모델에 대한 고유진동수 및 모드 형상도 알고 있다.
이 CAD는 다양한 비드의 형태 또는 홀 그리고 벤딩의 형태들을 가지고 있다.
단, 두께는 일정하다.

이 파트를 평평한 동일한 크기의 sheet body로 동등한 특성을 가지는 조건을 찾고 싶다.
sheet body를 n by m 구역으로 나누거나 element로 분할한다고 하면
각 위치에 대한 두께, 밀도, 강성을 변경하여 
대상 sheet metal 파트와 동일한 기계적 변형 특성을 가지는 조건을 찾는 것이다.
앞서 언급한 정해진 하중 및 경계 조건에 대해서 동일한 변형의 형태를 가지고
더불어 고유진동수와 모드 형상도 최대한 동일하게 되는 조건을 찾고 싶다.

이를 구하기 위한 파이썬 코드를 개발한다.

목표 모델에 대한 결과는 제공받을 수 있다.

하중 및 경계 조건에 대한 변형의 결과 (위치별)
고유진동수와 모드 형상 (위치별 변위 정보)

모사할 sheet body는 크기와 초기치 두께, 초기치 밀도, 초기치 강성을 제공할 수 있다.
두께를 고정하거나 밀도를 고정하거나 강성을 고정할 수 있는 옵션을 제시할 수 있다.

유한요소법을 python에서 직접 고속으로 수행해 결과를 얻고 목표 모델의 결과와 비교해서
오차를 최소화하는 조건을 찾을 수 있다.

추가로 검토해볼 아이디어는 deeponet 등을 이용하는 것이다.
많은 경우의 수에 대해서 해석의 결과를 deeponet으로 학습해놓고
주어진 대상 파트의 조건을 만족하는 위치별 두께, 밀도, 강성을 찾는 것이다. 
이 경우도 의견을 달라.

개발한 코드의 검증을 위한 기본 모델은 
임의의 특성을 가지는 평판 1000 x 400 mm와 두께 1mm을 가지는 평판에 알파벳 A의 형태로 비드가 올라온 것을 감안한 두께가 다른 형태 형성된 모양으로 가정하자.
이 모델에 대해서 양끝단을 고정하고 중앙부을 1N으로 눌렀을 때의 변형 상태와 
자유 상태의 리지드바디모션을 제외한 고유진동수 5개 그리고 각각의 모드 형상을 목적으로 하자.

이를 만족하는 sheet body의 위치별 조건을 찾자. N by M 구역으로 나눌 때 N=10, M=5로 하자.




SciPy vs JAX+Adam 비교
특성	SciPy (minimize)	JAX + Adam
Gradient 계산	수치 미분 (finite difference) 또는 사용자 제공	자동 미분 (autodiff)
메모리	Hessian 저장 필요 (L-BFGS-B)	1차/2차 모멘트만 저장
스케일링	변수 ~수천개까지	변수 수백만개도 OK
GPU 가속	❌ CPU only	✅ 네이티브 지원
배치 처리	❌	✅ 미니배치 가능
SciPy 대표 알고리즘
python
from scipy.optimize import minimize
# L-BFGS-B: 제한 조건 있는 quasi-Newton
result = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds)
# SLSQP: 제한 조건 있는 sequential quadratic programming
result = minimize(loss_fn, x0, method='SLSQP', constraints=cons)
장점: 수렴 보장, 제한 조건 처리 우수, 적은 iteration
단점: 고차원에서 느림, GPU 미지원
JAX + Adam (현재 코드)
python
# 자동 미분으로 gradient 계산
grads = jax.grad(loss_fn)(params)
# Adam으로 업데이트
params = params - lr * adam_update(grads)
장점: 대규모 파라미터, GPU 가속, 딥러닝 친화적
단점: 하이퍼파라미터 튜닝 필요, 수렴 보장 없음
이 프로젝트에서 JAX를 선택한 이유
고차원 파라미터: 격자 노드마다 t, rho, E → 수천 개 변수
FEM 계산의 미분: jax.grad()로 전체 FEM 연산 자동 미분
JIT 컴파일: 반복 최적화 속도 향상
SciPy의 minimize도 사용 가능하지만, 이 문제 규모에서는 JAX가 더 효율적이에요.


현재 구현의 한계와 개선 방향
1. 학습률 스케줄링 (권장 ⭐)
현재: 고정 lr = 0.01

python
# 개선: Cosine Annealing 또는 Step Decay
lr = lr_init * 0.5 ** (epoch // decay_step)
# 또는
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
2. Early Stopping
수렴 감지 후 조기 종료:

python
if abs(loss - prev_loss) < tol:
    break
3. 다른 Optimizer
Optimizer	특징
AdamW	Weight decay 분리 → 더 나은 일반화
LBFGS	2차 정보 활용 → 빠른 수렴 (Optax 지원)
RAdam	Warmup 불필요 → 안정적 시작
4. Multi-Start / Global Optimization
Local minima 회피:

python
# 여러 초기값에서 시작
for seed in range(N):
    params_init = random_init(seed)
    result = optimize(params_init)
    best = min(best, result)
5. Optax 라이브러리 활용
JAX 공식 optimizer 라이브러리:

python
import optax
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Gradient clipping
    optax.adam(learning_rate=1e-2),
    optax.scale_by_schedule(cosine_schedule)
)
6. 물리 기반 제약 강화
PDE 제약 (Physics-Informed Neural Networks 스타일)
Augmented Lagrangian으로 질량 제약 처리



┌─────────────────────────────────────────────────────────────┐
│  Target (Ground Truth)                                      │
│  ┌─────────────┐                                            │
│  │ 복잡한 구조  │ ──FEM (고해상도)──▶ 변위, 응력, 모드      │
│  │ (허니컴 등) │                                            │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ Loss 비교
┌─────────────────────────────────────────────────────────────┐
│  Optimized Sheet (등가 모델)                                │
│  ┌─────────────┐                                            │
│  │ 균질 판재   │ ──FEM (저해상도)──▶ 변위, 응력, 모드      │
│  │ (t, ρ, E)  │      ▲                                     │
│  └─────────────┘      │                                     │
│                       │                                     │
│               Optimizer가 파라미터 업데이트                 │
└─────────────────────────────────────────────────────────────┘
핵심: 양쪽 모두 FEM을 통해 물리법칙을 만족하는 해를 구하고 있어서, PDE 제약을 추가로 "강제"할 필요는 크지 않아요.

대신 유용할 수 있는 개선:

Early stopping - 수렴 감지
Learning rate decay - 후반부 미세 조정
Multi-objective balancing - 여러 loss 항목 자동 가중치 조절



# 자, 이제 매우 특별한 시도를 하겠다. 
topography를 추가한다.
: z 방향으로의 노드 좌표를 이동하는 변수를 추가하는 것이다.
: ground_truth를 생성할 때, bead의 두께를 제시하는 변수가 있었는데
  이와 별개로 bead_z을 제시하는 설정 변수를 추가한다.
: bead_pz으로 패턴 이름에 대한 z좌표를 정의한다.
  pattern_pz에 -x부터 +x로 이어지는 패턴을 알파벳이나 문자로 정의한다.

    target_config = {
        'base_t': 1.0, 
        'bead_t': {'A': 2.0, 'B': 2.5, 'C': 1.5},
        'bead_pz': {'T': 1.0, 'N': 0.5, 'Y': -1.0},
        'base_rho': 7.85e-9,  # Steel density
        'base_E': 210000.0,   # Steel Young's Modulus
        'pattern': 'ABC'       # Example Pattern
        'pattern_pz': 'TNY'       # Example Pattern
    }

: ground truth를 생성 전 형성된 패턴의 비드 두께와 형상을 확인하기 위해서
  pyvista를 이용해서 visualize를 생성해서 먼저 확인할 수 있도록 해달라.
  표시 전에 터미널로 contour로 표시할 내용이 z 좌표, 두께 중 선택합니다.    
  0는 cancel이고 아무런 값을 입력하지 않아도 그렇다.
  cancel 상태가 될 때까지 반복 ask한다.


: ground truth를 생성하는 해석이 끝난 후에는
  각 해석 결과를 pyvista로 볼 수 있는 visualizer로 준비한다.
  터미널에서 어떤 결과를 볼 지 선택할 수 있도록 한다. 0는 cancel이고 아무런 값을 입력하지 않아도 그렇다.
  변형된 형상으로 나타나고 결과 값은 contour로 나타나게 한다.
  cancel 상태가 될 때까지 반복 ask한다.


: 최적화가 끝난 후에 왼쪽은 타켓, 오른쪽에는 최종의 결과가 표현되는 pyvista로 시각화한다.
  마찬가지로 터미널에서 어떤 결과를 볼 지 선택할 수 있도록 한다. 0는 cancel이고 아무런 값을 입력하지 않아도 그렇다.
  cancel 상태가 될 때까지 반복 ask한다. 

: pyvista의 모든 폰트의 크기는 9pt 수준에 맞는 크기로 한다.

# 최적화 기능 중에 bead_pz을 조정하는 변수를 추가하겠다.
  사용 여부는 cfg에서 position_z_enable를 True로 설정하면 된다.
  범위의 기본값은 -1.0 ~ 1.0으로 해놓자. 기본값은 0이다.
