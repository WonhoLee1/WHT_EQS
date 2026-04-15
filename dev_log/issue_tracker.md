# WHTOOLS Structural Solver Issue Tracker

본 문서는 **ShellFEM 솔버**의 개발 및 검증 과정에서 발생한 주요 기술적 이슈와 해결 방안을 기록하여, 동일한 문제가 반복되지 않도록 관리하기 위한 문서입니다.

## 1. 진행 중인 이슈 및 해결 기록

| ID | 날짜 | 이슈 항목 | 상태 | 해결 방안 및 방지책 |
| :---: | :---: | :--- | :---: | :--- |
| #001 | 2026-04-05 | 고유 진동수 해석 시 강체 모드(0Hz) 간섭 | **RESOLVED** | 수치적 강체 모드(1.0 Hz 이하)를 필터링하고 실제 구조적 모드만 추출하도록 `test_frequency` 로직 보정. |
| #002 | 2026-04-05 | Q4 요소 응력 정밀도 부족 (SED 기반) | **RESOLVED** | Strain Energy Density 기반 근사를 폐기하고, 자코비안 기반 **B-matrix Stress Recovery**를 구현하여 굽힘 응력 오차 1% 미만 달성. |
| #003 | 2026-04-05 | 등분포 하중 인가 시 경계부 응력 오차 | **RESOLVED** | 모든 노드에 동일 하중을 주던 방식에서 **면적 가중치(A, A/2, A/4)**를 적용한 노달 하중 할당 방식으로 개선. |
| #004 | 2026-04-05 | 응력 추출 지점(Sampling Point) 불일치 | **RESOLVED** | `np.max()`에 의존하지 않고, 이론해 좌표에 대응하는 **Geometric Center** 기반 요소 샘플링 로직 도입. |
| #005 | 2026-04-05 | 고차 모드(Higher Modes) 검증 부재 | **RESOLVED** | 1차 모드 외에 상위 5개 모드까지 검증 범위를 확대하여 동역학적 신뢰성 강화 완료 (Mode 1~5 PASS). |
| #006 | 2026-04-05 | 통합 과정에서의 기능 회귀(Regression) | **RESOLVED** | 새로운 기능(예: 5차 진동수) 추가 시 기존 해결책(예: 하중/응력 보정)이 유실되는 현상 방지를 위해 형상 관리 및 전수 검증 프로세스 강화. |

| #007 | 2026-04-16 | 대규모 조립(Assembly) 속도 저하 | **RESOLVED** | `.at[].set()` 루프를 제거하고 `jnp.concatenate` 및 `einsum` 기반 블록 조립 방식으로 전면 교체하여 연산 속도 10배 이상 향상. |
| #008 | 2026-04-16 | 최적화 중 주파수 리포팅 정체 현상 | **RESOLVED** | 레일리 몫(Rayleigh Quotient) 근사치를 출력부에서 캐시값으로 덮어씌우던 버그 수정으로 실시간 모니터링 확보. |
| #009 | 2026-04-16 | 고유치 해석 주기 설정 무시 버그 | **RESOLVED** | `eigen_freq` 인자가 무시되고 15회로 고정되어 있던 하드코딩 오류 수정 및 사용자 설정 연동 완료. |

## 2. 재발 방지 체크리스트

- [ ] **Boundary Conditions**: 주 모드 외의 강체 모드 발생 가능성을 항상 체크하고 Essential Constraints가 충분한지 확인.
- [ ] **Load Allocation**: 압력 또는 등분포 하중 적용 시 경계 노드의 기여 면적 보정 여부 확인.
- [ ] **Stress Recovery**: 새로운 요소 도입 시 반드시 B-matrix 기반 고정밀 응력 사출 로직 적용.
- [ ] **Differentiability**: JAX 최적화 시 캐시된 결과가 미분 사슬(Chain Rule)을 끊지 않는지 `aux` 데이터 흐름 상시 점검.
- [ ] **Hardcoding Audit**: 매개변수화된 인자가 실제 루프 제어 변수에 1:1로 매핑되어 있는지 확인.

---
> **Last Updated**: 2026-04-16  
> **Maintainer**: WHTOOLS (Expert Structural Analyst)

