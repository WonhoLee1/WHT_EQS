"""
opt_targets.py - 최적화 목표(OptTarget) 정의 및 관리 모듈

이 모듈은 구조 해석 최적화에서 사용되는 목표 함수들을 정의하고 관리합니다.
OptTarget 클래스를 통해 다양한 유형의 목표(예: 응력, 변형률, 질량, 모드 등)를 설정하고,
결과 번들(ResultBundle)로부터 오차를 계산할 수 있습니다.

주요 기능:
- OptTarget: 최적화 목표를 정의하는 데이터 클래스
- ResultBundle: 해석 결과를 저장하는 데이터 클래스
- ResultAccessor: 결과 번들에서 값을 추출하는 유틸리티 클래스
- 파싱 함수들: JSON이나 딕셔너리로부터 OptTarget을 생성
- 레거시 플래그 변환: 기존 부울 플래그를 OptTarget으로 변환

사용법:
1. OptTarget 인스턴스 생성:
   target = OptTarget(target_type=TargetType.FIELD_STAT, field='stress_vm', reduction=Reduction.MAX)

2. 결과 번들 생성:
   bundle = ResultBundle(fields={'stress_vm': stress_array}, mass=total_mass)

3. 오차 계산:
   error, details = target.compute_error(bundle, ref_bundle)

4. JSON에서 파싱:
   targets = parse_opt_targets('opt_target_examples.json')

5. 모델에 적용:
   assigned = apply_case_targets_from_spec(model, spec_source)

참고: 이 모듈은 PlateFEM 기반 구조 최적화에 특화되어 있습니다.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import json
import os


# 최적화 목표의 유형을 정의하는 열거형
class TargetType(Enum):
    FIELD_STAT = 'field_stat'      # 필드 통계 (응력, 변형률 등)
    RBE_REACTION = 'rbe_reaction'  # RBE 반력
    NODE_DISP = 'node_disp'        # 노드 변위
    MASS = 'mass'                  # 질량
    MODES = 'modes'                # 모드 (진동 모드)


# 필드 값의 축소 방법을 정의하는 열거형
class Reduction(Enum):
    MAX = 'max'           # 최대값
    MIN = 'min'           # 최소값
    MEAN = 'mean'         # 평균값
    RMS = 'rms'           # RMS (제곱 평균 제곱근)
    PERCENTILE = 'percentile'  # 백분위수


# 비교 모드를 정의하는 열거형
class CompareMode(Enum):
    ABSOLUTE = 'absolute'  # 절대 비교
    RELATIVE = 'relative'  # 상대 비교 (참조값으로 나눔)
    MAC = 'mac'            # MAC (Modal Assurance Criterion) - 모드 비교
    ANGLE = 'angle'        # 각도 비교


# 목표의 방향을 정의하는 열거형
class Sense(Enum):
    MATCH = 'match'        # 일치시키기 (오차 최소화)
    MINIMIZE = 'minimize'  # 최소화
    MAXIMIZE = 'maximize'  # 최대화


@dataclass
class ResultValue:
    """결과 값과 메타데이터를 저장하는 데이터 클래스"""
    value: float              # 실제 값
    units: str = ''           # 단위 (예: 'MPa', 'mm')
    meta: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터


@dataclass
class Mode:
    """진동 모드를 표현하는 데이터 클래스"""
    index: int                          # 모드 인덱스
    frequency: float                    # 고유 진동수 (Hz)
    vector: Optional[np.ndarray] = None # 모드 벡터 (변위 모양)

    def mac_with(self, other: 'Mode') -> Optional[float]:
        """
        다른 모드와의 Modal Assurance Criterion (MAC)을 계산합니다.
        MAC은 두 모드 벡터의 유사성을 측정합니다 (0~1, 1이 완전 일치).

        Args:
            other: 비교할 다른 Mode 객체

        Returns:
            MAC 값 (0~1) 또는 벡터가 없는 경우 None
        """
        if self.vector is None or other.vector is None:
            return None
        v1 = np.asarray(self.vector).ravel()
        v2 = np.asarray(other.vector).ravel()
        num = float(np.dot(v1, v2)) ** 2
        den = float(np.dot(v1, v1)) * float(np.dot(v2, v2))
        return float(num / den) if den > 0 else 0.0


@dataclass
class ResultBundle:
    """해석 결과를 저장하는 데이터 클래스"""
    # 필드 데이터: 필드명 -> 배열 (요소별 또는 노드별)
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    # RBE 반력: ID -> 3차원 벡터 (또는 더 많은 성분)
    rbe_reactions: Dict[str, np.ndarray] = field(default_factory=dict)
    # 노드 변위: 노드ID -> 벡터
    node_disps: Dict[int, np.ndarray] = field(default_factory=dict)
    mass: float = 0.0                    # 총 질량
    modes: List[Mode] = field(default_factory=list)  # 진동 모드 목록
    meta: Dict[str, Any] = field(default_factory=dict)  # 메타데이터 (단위 등)


class ResultAccessor:
    """
    ResultBundle에서 값을 편리하게 추출하는 유틸리티 클래스.
    PlateFEM의 결과 저장 방식에 따라 구현을 확장할 수 있도록 최소한으로 유지합니다.
    """

    @staticmethod
    def get_field_stat(bundle: ResultBundle, field: str, region: Optional[Any] = None,
                       reduction: Reduction = Reduction.MAX, percentile: float = 95.0) -> ResultValue:
        """
        필드 통계를 계산합니다.

        Args:
            bundle: 결과 번들
            field: 필드명 (예: 'stress_vm', 'max_strain')
            region: 영역 제한 (현재 미사용)
            reduction: 축소 방법
            percentile: 백분위수 (reduction이 PERCENTILE일 때 사용)

        Returns:
            ResultValue 객체
        """
        arr = bundle.fields.get(field)
        if arr is None:
            raise KeyError(f'Field "{field}" not found in result bundle')
        data = np.asarray(arr)
        if reduction == Reduction.MAX:
            val = float(np.nanmax(data))
        elif reduction == Reduction.MIN:
            val = float(np.nanmin(data))
        elif reduction == Reduction.MEAN:
            val = float(np.nanmean(data))
        elif reduction == Reduction.RMS:
            val = float(np.sqrt(np.nanmean(np.square(data))))
        elif reduction == Reduction.PERCENTILE:
            val = float(np.nanpercentile(data, percentile))
        else:
            raise ValueError('Unknown reduction')
        return ResultValue(value=val, units=bundle.meta.get('units', {}).get(field, ''))

    @staticmethod
    def get_rbe_reaction(bundle: ResultBundle, rbe_id: str, component: Optional[str] = None) -> ResultValue:
        """
        RBE 반력을 추출합니다.

        Args:
            bundle: 결과 번들
            rbe_id: RBE ID
            component: 성분 ('magnitude', 'fx', 'fy', 'fz' 또는 인덱스)

        Returns:
            ResultValue 객체
        """
        vec = bundle.rbe_reactions.get(rbe_id)
        if vec is None:
            raise KeyError(f'RBE reaction "{rbe_id}" not found')
        v = np.asarray(vec)
        if component is None or component == 'magnitude':
            val = float(np.linalg.norm(v))
            units = bundle.meta.get('units', {}).get('rbe_reaction', '')
        else:
            # component could be 'fx','fy','fz' or index
            idx = {'fx': 0, 'fy': 1, 'fz': 2}.get(component, None)
            if isinstance(component, int):
                idx = component
            if idx is None:
                raise ValueError('Unknown component')
            val = float(v[idx])
            units = bundle.meta.get('units', {}).get('rbe_reaction', '')
        return ResultValue(value=val, units=units)

    @staticmethod
    def get_node_disp(bundle: ResultBundle, node_id: int, comp: str = 'magnitude') -> ResultValue:
        """
        노드 변위를 추출합니다.

        Args:
            bundle: 결과 번들
            node_id: 노드 ID
            comp: 성분 ('magnitude', 'x', 'y', 'z')

        Returns:
            ResultValue 객체
        """
        vec = bundle.node_disps.get(node_id)
        if vec is None:
            raise KeyError(f'Node displacement for node {node_id} not found')
        v = np.asarray(vec)
        if comp == 'magnitude':
            val = float(np.linalg.norm(v))
        elif comp in ('x', 'y', 'z'):
            idx = {'x': 0, 'y': 1, 'z': 2}[comp]
            val = float(v[idx])
        else:
            raise ValueError('Unknown component')
        return ResultValue(value=val, units=bundle.meta.get('units', {}).get('disp', ''))

    @staticmethod
    def get_modes(bundle: ResultBundle) -> List[Mode]:
        """모드 목록을 반환합니다."""
        return list(bundle.modes)

    @staticmethod
    def compute_mac_matrix(bundle: ResultBundle, ref_bundle: ResultBundle) -> np.ndarray:
        """
        두 결과 번들 간의 MAC 행렬을 계산합니다.

        Args:
            bundle: 현재 결과 번들
            ref_bundle: 참조 결과 번들

        Returns:
            MAC 행렬 (shape: (len(bundle.modes), len(ref_bundle.modes)))
        """
        modes = bundle.modes
        ref_modes = ref_bundle.modes
        M = np.zeros((len(modes), len(ref_modes)), dtype=float)
        for i, m in enumerate(modes):
            for j, rm in enumerate(ref_modes):
                mac = m.mac_with(rm)
                M[i, j] = 0.0 if mac is None else float(mac)
        return M


@dataclass
class OptTarget:
    """
    최적화 목표를 정의하는 데이터 클래스.
    다양한 유형의 목표(필드 통계, 반력, 질량, 모드 등)를 설정하고,
    결과 번들로부터 오차를 계산합니다.
    """
    target_type: TargetType  # 목표 유형

    # 필드 통계용 파라미터
    field: Optional[str] = None          # 필드명 (예: 'stress_vm')
    region: Optional[Any] = None         # 영역 제한
    reduction: Reduction = Reduction.MAX # 축소 방법
    percentile: float = 95.0             # 백분위수

    # RBE/노드용 파라미터
    rbe_id: Optional[str] = None         # RBE ID
    node_id: Optional[int] = None        # 노드 ID
    component: Optional[str] = None      # 성분

    # 모드용 파라미터
    num_modes: Optional[int] = None      # 고려할 모드 수
    freq_weight: float = 0.0             # 진동수 오차 가중치

    # 비교/목표 파라미터
    compare_mode: CompareMode = CompareMode.ABSOLUTE  # 비교 모드
    sense: Sense = Sense.MATCH                        # 목표 방향
    weight: float = 1.0                               # 가중치
    tolerance: Optional[float] = None                 # 허용 오차
    ref_value: Optional[float] = None                 # 참조 값

    def compute_error(self, bundle: ResultBundle, ref_bundle: Optional[ResultBundle] = None,
                      accessor: ResultAccessor = ResultAccessor) -> Tuple[float, Dict[str, Any]]:
        """
        결과 번들로부터 오차를 계산합니다.

        Args:
            bundle: 현재 결과 번들
            ref_bundle: 참조 결과 번들 (선택적)
            accessor: 결과 접근자 클래스

        Returns:
            (가중치 적용된 오차, 세부 정보 딕셔너리)의 튜플
        """
        details: Dict[str, Any] = {'target_type': self.target_type.value}

        # 질량 목표 처리
        if self.target_type == TargetType.MASS:
            val = float(bundle.mass)
            ref = self.ref_value if self.ref_value is not None else (ref_bundle.mass if ref_bundle is not None else 0.0)
            if self.compare_mode == CompareMode.RELATIVE and ref != 0:
                err = (val - ref) / ref
            else:
                err = val - ref
            details.update({'value': val, 'ref': ref})
            return float(self.weight * float(err)), details

        # RBE 반력 목표 처리
        if self.target_type == TargetType.RBE_REACTION:
            rv = accessor.get_rbe_reaction(bundle, self.rbe_id, self.component)
            val = rv.value
            ref = self.ref_value if self.ref_value is not None else (accessor.get_rbe_reaction(ref_bundle, self.rbe_id, self.component).value if ref_bundle is not None else 0.0)
            if self.compare_mode == CompareMode.RELATIVE and ref != 0:
                err = (val - ref) / ref
            else:
                err = val - ref
            details.update({'value': val, 'ref': ref})
            return float(self.weight * float(err)), details

        # 필드 통계 목표 처리
        if self.target_type == TargetType.FIELD_STAT:
            rv = accessor.get_field_stat(bundle, self.field, self.region, self.reduction, self.percentile)
            val = rv.value
            if ref_bundle is not None:
                ref_rv = accessor.get_field_stat(ref_bundle, self.field, self.region, self.reduction, self.percentile)
                ref = ref_rv.value
            else:
                ref = self.ref_value if self.ref_value is not None else 0.0
            if self.compare_mode == CompareMode.RELATIVE and ref != 0:
                err = (val - ref) / ref
            else:
                err = val - ref
            details.update({'value': val, 'ref': ref})
            return float(self.weight * float(err)), details

        # 모드 목표 처리 (MAC 기반)
        if self.target_type == TargetType.MODES:
            # MAC (Modal Assurance Criterion) 비교 지원
            if self.compare_mode == CompareMode.MAC and ref_bundle is not None:
                M = accessor.compute_mac_matrix(bundle, ref_bundle)
                # 각 참조 모드에 대해 최적 매칭의 평균을 취함
                best = M.max(axis=0) if M.size > 0 else np.array([])
                score = float(np.mean(best)) if best.size > 0 else 1.0
                # 오차 = 1 - 평균 MAC
                mac_err = 1.0 - score
                details.update({'mean_mac': score})

                # 선택적 진동수 매칭 성분
                if self.freq_weight and hasattr(bundle, 'modes') and hasattr(ref_bundle, 'modes'):
                    try:
                        ref_freqs = np.array([m.frequency for m in ref_bundle.modes])
                        cur_freqs = np.array([m.frequency for m in bundle.modes])
                        if self.num_modes is not None:
                            ref_freqs = ref_freqs[:self.num_modes]
                            cur_freqs = cur_freqs[:self.num_modes]
                        # 길이를 동일하게 맞춤
                        n = min(len(ref_freqs), len(cur_freqs))
                        if n > 0:
                            ref_f = ref_freqs[:n]
                            cur_f = cur_freqs[:n]
                            freq_err = float(np.mean(np.abs((cur_f - ref_f) / (ref_f + 1e-12))))
                        else:
                            freq_err = 0.0
                    except Exception:
                        freq_err = 0.0
                    details.update({'freq_err': freq_err})
                    combined = (mac_err * (1.0 - float(self.freq_weight))) + (float(freq_err) * float(self.freq_weight))
                    return float(self.weight * combined), details
                else:
                    return float(self.weight * float(mac_err)), details
            else:
                return 0.0, details

        raise NotImplementedError(f'OptTarget compute_error not implemented for {self.target_type}')

    @classmethod
    def preset(cls) -> Dict[str, Any]:
        """
        기본 최적화 목표 설정을 반환합니다.
        get_default_opt_target_config 함수를 호출합니다.
        """
        return get_default_opt_target_config()


def _resolve_enum(enum_cls, value, default=None):
    """
    문자열이나 enum 값을 enum 클래스로 변환합니다.

    Args:
        enum_cls: 대상 enum 클래스
        value: 변환할 값 (문자열, enum, 또는 None)
        default: 기본값

    Returns:
        변환된 enum 값 또는 default
    """
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        v_low = value.strip().lower()
        for m in enum_cls:
            if m.value == v_low or m.name.lower() == v_low:
                return m
    return default


def parse_opt_targets(source) -> List[OptTarget]:
    """
    opt-targets 명세를 파싱하여 OptTarget 인스턴스 리스트를 반환합니다.

    지원되는 source 형식:
    - JSON 파일 경로 (문자열)
    - Python 딕셔너리 (단일 목표 또는 매핑)
    - 목표 딕셔너리 리스트

    Args:
        source: 파싱할 소스 (파일 경로, 딕셔너리, 또는 리스트)

    Returns:
        OptTarget 인스턴스 리스트

    Raises:
        FileNotFoundError: 파일 경로가 존재하지 않을 때
        TypeError: 지원되지 않는 소스 타입일 때
    """
    # 원시 데이터 로드
    data = None
    if isinstance(source, str):
        # JSON 파일로 취급
        if not os.path.exists(source):
            raise FileNotFoundError(f'File not found: {source}')
        with open(source, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif isinstance(source, (list, tuple)):
        data = list(source)
    elif isinstance(source, dict):
        # 여러 딕셔너리 형태 지원: {'targets': [...]}, {case_name: [...]}, 또는 단일 목표 딕셔너리
        if 'targets' in source and isinstance(source['targets'], (list, tuple)):
            data = list(source['targets'])
        else:
            # 매핑 내 리스트 수집
            collected = []
            for v in source.values():
                if isinstance(v, (list, tuple)):
                    collected.extend(v)
            if collected:
                data = collected
            else:
                # 단일 목표 명세로 가정
                data = [source]
    else:
        raise TypeError('Unsupported source type for parse_opt_targets')

    out: List[OptTarget] = []
    for spec in data:
        if not isinstance(spec, dict):
            continue
        ttype = _resolve_enum(TargetType, spec.get('target_type') or spec.get('type'), None)
        if ttype is None:
            # 유효하지 않은 항목 건너뜀
            continue
        ot = OptTarget(
            target_type=ttype,
            field=spec.get('field'),
            region=spec.get('region'),
            reduction=_resolve_enum(Reduction, spec.get('reduction'), Reduction.MAX),
            percentile=float(spec.get('percentile', 95.0)),
            rbe_id=spec.get('rbe_id') or spec.get('rbe'),
            node_id=spec.get('node_id'),
            component=spec.get('component'),
            compare_mode=_resolve_enum(CompareMode, spec.get('compare_mode') or spec.get('mode'), CompareMode.ABSOLUTE),
            sense=_resolve_enum(Sense, spec.get('sense'), Sense.MATCH),
            weight=float(spec.get('weight', 1.0)),
            tolerance=spec.get('tolerance'),
            ref_value=spec.get('ref_value')
        )
        out.append(ot)

    return out


def apply_case_targets_from_spec(model, spec_source) -> Dict[str, List[OptTarget]]:
    """
    모델의 케이스에 opt-targets 명세를 적용합니다.

    spec_source는 JSON 파일 경로, 딕셔너리 (opt_target_examples.json과 유사),
    또는 리스트일 수 있습니다. 함수는 케이스별 항목을 찾아
    case.opt_targets 리스트에 OptTarget 인스턴스를 첨부합니다.
    할당된 목표의 case_name -> OptTarget 리스트 매핑을 반환합니다.

    Args:
        model: 목표를 적용할 모델 객체 (cases 속성 필요)
        spec_source: 명세 소스 (파일 경로, 딕셔너리, 또는 리스트)

    Returns:
        할당된 목표의 딕셔너리: {case_name: [OptTarget, ...]}
    """
    # 원시 명세 로드 (파싱 로직 재사용)
    if isinstance(spec_source, str):
        if not os.path.exists(spec_source):
            raise FileNotFoundError(spec_source)
        with open(spec_source, 'r', encoding='utf-8') as f:
            spec = json.load(f)
    else:
        spec = spec_source

    assigned: Dict[str, List[OptTarget]] = {}

    if isinstance(spec, dict):
        for key, val in spec.items():
            # 케이스 항목: {'opt_targets': [...], 'name': optional}
            if isinstance(val, dict) and 'opt_targets' in val:
                case_name = val.get('name') or key
                # 이름으로 케이스 찾기
                target_case = next((c for c in getattr(model, 'cases', []) if getattr(c, 'name', None) == case_name), None)
                if target_case is not None:
                    ots = parse_opt_targets(val['opt_targets'])
                    setattr(target_case, 'opt_targets', ots)
                    assigned[case_name] = ots
            # 최상위 키가 케이스 이름에 해당하는 목표 딕셔너리 리스트
            elif isinstance(val, (list, tuple)):
                case_name = key
                target_case = next((c for c in getattr(model, 'cases', []) if getattr(c, 'name', None) == case_name), None)
                if target_case is not None:
                    ots = parse_opt_targets(val)
                    setattr(target_case, 'opt_targets', ots)
                    assigned[case_name] = ots
            # 독립형 단일 목표 명세 -> 글로벌 풀에 첨부
            elif isinstance(val, dict) and 'target_type' in val:
                # model.global_opt_targets에 첨부
                if not hasattr(model, 'global_opt_targets'):
                    model.global_opt_targets = []
                ots = parse_opt_targets([val])
                model.global_opt_targets.extend(ots)

    elif isinstance(spec, (list, tuple)):
        # 항목 리스트; 'case_name' 또는 'case'를 가진 딕셔너리 허용
        for item in spec:
            if not isinstance(item, dict):
                continue
            case_name = item.get('case_name') or item.get('case')
            if case_name:
                target_case = next((c for c in getattr(model, 'cases', []) if getattr(c, 'name', None) == case_name), None)
                if target_case is not None:
                    ots = parse_opt_targets([item])
                    setattr(target_case, 'opt_targets', getattr(target_case, 'opt_targets', []) + ots)
                    assigned.setdefault(case_name, []).extend(ots)
            else:
                # 독립형 목표 -> 글로벌
                if not hasattr(model, 'global_opt_targets'):
                    model.global_opt_targets = []
                model.global_opt_targets.extend(parse_opt_targets([item]))

    return assigned


def map_legacy_flags_to_targets(model,
                                use_surface_stress: bool = True,
                                use_surface_strain: bool = False,
                                use_strain_energy: bool = True,
                                use_mass_constraint: bool = True,
                                mode_weight: float = 0.0,
                                num_modes: Optional[int] = None,
                                freq_weight: float = 0.0):
    """
    레거시 부울 플래그를 OptTarget 인스턴스로 변환하여 모델에 첨부합니다.
    중복 목표 추가를 방지하기 위해 멱등적입니다.

    Args:
        model: 목표를 적용할 모델 객체
        use_surface_stress: 표면 응력 목표 사용 여부
        use_surface_strain: 표면 변형률 목표 사용 여부
        use_strain_energy: 변형률 에너지 목표 사용 여부
        use_mass_constraint: 질량 제약 사용 여부
        mode_weight: 모드 목표 가중치
        num_modes: 고려할 모드 수
        freq_weight: 진동수 오차 가중치

    Returns:
        수정된 모델 객체
    """
    # 기존 유사 목표 확인 헬퍼
    def has_target(case, ttype, field=None):
        for ot in getattr(case, 'opt_targets', []) or []:
            if ot.target_type == ttype:
                if field is None or getattr(ot, 'field', None) == field:
                    return True
        return False

    # 케이스별 추가
    for case in getattr(model, 'cases', []):
        if use_surface_stress and not has_target(case, TargetType.FIELD_STAT, 'stress_vm'):
            case.opt_targets.append(OptTarget(target_type=TargetType.FIELD_STAT,
                                              field='stress_vm', reduction=Reduction.MAX,
                                              compare_mode=CompareMode.RELATIVE, weight=case.weight))
        if use_surface_strain and not has_target(case, TargetType.FIELD_STAT, 'max_strain'):
            case.opt_targets.append(OptTarget(target_type=TargetType.FIELD_STAT,
                                              field='max_strain', reduction=Reduction.MAX,
                                              compare_mode=CompareMode.RELATIVE, weight=case.weight))
        if use_strain_energy and not has_target(case, TargetType.FIELD_STAT, 'strain_energy_density'):
            case.opt_targets.append(OptTarget(target_type=TargetType.FIELD_STAT,
                                              field='strain_energy_density', reduction=Reduction.MEAN,
                                              compare_mode=CompareMode.ABSOLUTE, weight=case.weight))

    # 글로벌 질량 제약
    if use_mass_constraint:
        if not hasattr(model, 'global_opt_targets'):
            model.global_opt_targets = []
        # 중복 질량 목표 방지
        existing_mass = any((ot.target_type == TargetType.MASS) for ot in getattr(model, 'global_opt_targets', []))
        if not existing_mass:
            model.global_opt_targets.append(OptTarget(target_type=TargetType.MASS,
                                                      compare_mode=CompareMode.RELATIVE,
                                                      ref_value=getattr(model, 'target_mass', None),
                                                      weight=10.0))

    # 모드 목표 (글로벌)
    if mode_weight and mode_weight > 0.0:
        if not hasattr(model, 'global_opt_targets'):
            model.global_opt_targets = []
        # 중복 모드 목표 방지
        existing_modes = any((ot.target_type == TargetType.MODES) for ot in getattr(model, 'global_opt_targets', []))
        if not existing_modes:
            model.global_opt_targets.append(OptTarget(target_type=TargetType.MODES,
                                                      compare_mode=CompareMode.MAC,
                                                      num_modes=num_modes,
                                                      freq_weight=freq_weight,
                                                      weight=mode_weight))

    return model


# === 추가 사용법 예제 ===

"""
고급 사용법:

1. JSON 파일에서 목표 로드:
   targets = parse_opt_targets('opt_target_examples.json')

2. 모델에 케이스별 목표 적용:
   assigned = apply_case_targets_from_spec(model, 'config.json')

3. 레거시 플래그에서 변환:
   model = map_legacy_flags_to_targets(model, use_surface_stress=True, use_mass_constraint=True)

4. 수동 목표 생성:
   stress_target = OptTarget(
       target_type=TargetType.FIELD_STAT,
       field='stress_vm',
       reduction=Reduction.MAX,
       compare_mode=CompareMode.RELATIVE,
       weight=1.0
   )

5. 오차 계산:
   bundle = ResultBundle(fields={'stress_vm': stress_data}, mass=total_mass)
   error, details = stress_target.compute_error(bundle, ref_bundle)

참고: 모든 목표는 가중치(weight)를 통해 상대적 중요도를 조정할 수 있습니다.
모드 목표의 경우 MAC과 진동수 오차를 결합하여 사용할 수 있습니다.
"""


def get_default_opt_target_config() -> Dict[str, Any]:
    """
    기본 최적화 목표 설정을 반환합니다.
    각 케이스별 최적화 목표와 글로벌 목표를 포함합니다.
    """
    opt_target_config = {
        # 비틀림 하중 케이스들: 반력과 응력을 최적화 목표로 설정
        "twist_x": {
            "name": "twist_x",
            "opt_targets": [
                {
                    "target_type": "rbe_reaction",  # RBE 반력 목표
                    "rbe_id": "residual",           # 잔여 반력 ID
                    "component": "magnitude",      # 크기 성분
                    "compare_mode": "relative",    # 상대 비교 모드
                    "weight": 1.0                  # 가중치
                },
                {
                    "target_type": "field_stat",   # 필드 통계 목표
                    "field": "stress_vm",          # 폰 미제스 응력 필드
                    "reduction": "max",            # 최대값 축소
                    "compare_mode": "relative",    # 상대 비교
                    "weight": 0.3                  # 낮은 가중치 (보조 목표)
                }
            ]
        },
        "twist_y": {
            "name": "twist_y", 
            "opt_targets": [
                {
                    "target_type": "rbe_reaction",
                    "rbe_id": "residual", 
                    "component": "magnitude",
                    "compare_mode": "relative",
                    "weight": 1.0
                },
                {
                    "target_type": "field_stat",
                    "field": "stress_vm",
                    "reduction": "max", 
                    "compare_mode": "relative",
                    "weight": 0.3
                }
            ]
        },
        
        # 굽힘 하중 케이스들: 최대 변형률을 목표로 설정
        "bend_x": {
            "name": "bend_x",
            "opt_targets": [
                {
                    "target_type": "field_stat",
                    "field": "max_strain",         # 최대 변형률 필드
                    "reduction": "max",
                    "compare_mode": "relative",
                    "weight": 1.0
                }
            ]
        },
        "bend_y": {
            "name": "bend_y", 
            "opt_targets": [
                {
                    "target_type": "field_stat",
                    "field": "max_strain",
                    "reduction": "max",
                    "compare_mode": "relative", 
                    "weight": 1.0
                }
            ]
        },
        
        # 변위 제어 케이스들 (리프트 및 캔틸레버): 목표 변위 도달 시 발생하는 반력(Reaction Force) 일치 타겟
        "lift_br": {
            "name": "lift_br",
            "opt_targets": [
                {
                    "target_type": "rbe_reaction",
                    "component": "magnitude",
                    "compare_mode": "relative",
                    "weight": 1.0
                }
            ]
        },
        "lift_tl": {
            "name": "lift_tl",
            "opt_targets": [
                {
                    "target_type": "rbe_reaction",
                    "component": "magnitude",
                    "compare_mode": "relative",
                    "weight": 1.0
                }
            ]
        },
        "lift_tl_br": {
            "name": "lift_tl_br",
            "opt_targets": [
                {
                    "target_type": "rbe_reaction",
                    "component": "magnitude",
                    "compare_mode": "relative",
                    "weight": 1.0
                }
            ]
        },
        
        "cantilever_x": {
            "name": "cantilever_x",
            "opt_targets": [
                {
                    "target_type": "rbe_reaction",
                    "component": "magnitude",
                    "compare_mode": "relative",
                    "weight": 1.0
                }
            ]
        },
        "cantilever_y": {
            "name": "cantilever_y",
            "opt_targets": [
                {
                    "target_type": "rbe_reaction",
                    "component": "magnitude",
                    "compare_mode": "relative",
                    "weight": 1.0
                }
            ]
        },
        
        # 압력 하중 케이스: RMS 변위 제어 (균일한 변형 분포 목표)
        "pressure_z": {
            "name": "pressure_z",
            "opt_targets": [
                {
                    "target_type": "field_stat",
                    "field": "u_static",
                    "reduction": "rms",            # RMS 값 (제곱 평균 제곱근)
                    "compare_mode": "absolute",
                    "ref_value": 10.0,             # 목표 RMS 변위
                    "weight": 1.0
                }
            ]
        }
    }
    
    # 글로벌 최적화 목표 추가 (모든 케이스에 적용되는 제약조건)
    global_targets = [
        {
            "target_type": "mass",              # 질량 제약
            "compare_mode": "relative",         # 상대 비교 (목표 질량 대비)
            "ref_value": 5.0,                   # 목표 질량 (톤)
            "tolerance": 0.05,                  # 허용 오차 (5%)
            "weight": 10.0                      # 높은 가중치 (중요 제약)
        },
        {
            "target_type": "modes",             # 모드 매칭 목표
            "compare_mode": "mac",              # MAC (Modal Assurance Criterion) 사용
            "weight": 1.0,                      # 기본 가중치
            "num_modes": 5                      # 비교할 모드 수
        }
    ]
    
    # 글로벌 목표를 opt_target_config에 추가
    opt_target_config["global_targets"] = global_targets
    
    return opt_target_config
