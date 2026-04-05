# -*- coding: utf-8 -*-
"""
================================================================================
ShellFemSolverVerification / analytical_solutions.py
================================================================================
■ 목적
    패치 테스트 검증에 사용할 쉘(판) 이론의 해석적(Analytical) 해를 제공합니다.
    단위계: mm / MPa(N/mm²) / tonne  (기존 shell_solver.py와 동일)
================================================================================
"""
import numpy as np


def kirchhoff_plate_field_solution(
    Lx: float, Ly: float, q: float,
    E: float, t: float, nu: float,
    x_coords: np.ndarray, y_coords: np.ndarray,
    n_terms: int = 15
) -> dict:
    """
    단순지지 사각형 판의 모든 노드 위치에서 이론적 해(필드)를 계산합니다.
    
    Args:
        Lx, Ly   : 판의 가로, 세로 길이 [mm]
        q        : 균일 하중 [MPa]
        E, t, nu : 재료 물성 및 두께
        x_coords : 계산 지점 X 좌표 배열 [mm]
        y_coords : 계산 지점 Y 좌표 배열 [mm]
        n_terms  : 급수 연산 항 수
    Returns:
        dict: {
            'w'        : 처짐 [mm],
            'kappa_x'  : X 곡률 [1/mm],
            'kappa_y'  : Y 곡률 [1/mm],
            'kappa_xy' : 비틀림 곡률 [1/mm],
            'strain_x' : 상면(z=-t/2) X 변형률 [-],
            'stress_vm': 상면 Von-Mises 응력 [MPa]
        }
    """
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    w = np.zeros_like(x_coords)
    kx = np.zeros_like(x_coords)
    ky = np.zeros_like(x_coords)
    kxy = np.zeros_like(x_coords)

    # Navier series summation
    for m in range(1, 2 * n_terms, 2):
        for n in range(1, 2 * n_terms, 2):
            # Load coefficient
            q_mn = 16.0 * q / (np.pi**2 * m * n)
            # Stiffness factor
            denom = np.pi**4 * D * ((m / Lx)**2 + (n / Ly)**2)**2
            coeff = q_mn / denom
            
            # Sine terms
            smx = np.sin(m * np.pi * x_coords / Lx)
            sny = np.sin(n * np.pi * y_coords / Ly)
            cmx = np.cos(m * np.pi * x_coords / Lx)
            cny = np.cos(n * np.pi * y_coords / Ly)
            
            # Displacement w
            w += coeff * smx * sny
            
            # Curvatures kappa = -w,ij
            kx += coeff * (m * np.pi / Lx)**2 * smx * sny
            ky += coeff * (n * np.pi / Ly)**2 * smx * sny
            kxy -= coeff * (m * np.pi / Lx) * (n * np.pi / Ly) * cmx * cny

    # Stress/Strain at z = t/2
    factor_eps = t/2.0
    ex = kx * factor_eps
    ey = ky * factor_eps
    exy = kxy * factor_eps
    
    factor_sig = E / (1.0 - nu**2)
    sx = factor_sig * (ex + nu * ey)
    sy = factor_sig * (ey + nu * ex)
    sxy = (E / (2.0 * (1.0 + nu))) * exy
    
    vm = np.sqrt(sx**2 - sx*sy + sy**2 + 3.0*sxy**2)

    return {
        'w': w,
        'kappa_x': kx, 'kappa_y': ky, 'kappa_xy': kxy,
        'strain_x': ex,
        'stress_x': sx,
        'stress_y': sy,
        'stress_xy': sxy,
        'stress_vm': vm
    }


def kirchhoff_frequency(
    Lx: float, Ly: float, E: float, t: float, nu: float, rho: float,
    m: int = 1, n: int = 1
) -> float:
    """단순지지 사각형 판의 고유 진동수 (Hz)."""
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    f = (np.pi / 2.0) * np.sqrt(D / (rho * t)) * ((m / Lx)**2 + (n / Ly)**2)
    return f


def beam_cantilever_tip_deflection(L: float, E: float, I: float, P: float) -> float:
    """
    외팔보 끝단 집중하중 P → 끝단 처짐.

    Args:
        L : 보 길이 [mm]
        E : 탄성계수 [MPa]
        I : 단면 2차 모멘트 [mm⁴]
        P : 끝단 집중하중 [N]
    Returns:
        w_tip [mm]
    """
    return P * L**3 / (3.0 * E * I)


def beam_cantilever_root_stress(L: float, P: float, I: float, c: float) -> float:
    """
    외팔보 고정단 최대 굽힘 응력.

    Args:
        L : 보 길이 [mm]
        P : 끝단 집중하중 [N]
        I : 단면 2차 모멘트 [mm⁴]
        c : 중립축에서 외면까지의 거리 = t/2 [mm]
    Returns:
        sigma_max [MPa]
    """
    M_root = P * L
    return M_root * c / I


def plane_stress_uniaxial(E: float, nu: float, sigma_x: float):
    """
    단축 평면 응력 상태의 이론 변형률.

    Args:
        E       : 탄성계수 [MPa]
        nu      : 포아송비 [-]
        sigma_x : 적용 응력 [MPa]
    Returns:
        (eps_x, eps_y, eps_z) : 주요 방향 변형률
    """
    eps_x = sigma_x / E
    eps_y = -nu * sigma_x / E
    eps_z = -nu * sigma_x / E
    return eps_x, eps_y, eps_z


def beam_3point_bending_deflection(L: float, E: float, I: float, P: float) -> float:
    """3점 굽힘 중앙 처짐: w = PL^3 / (48EI)"""
    return (P * L**3) / (48.0 * E * I)

def beam_4point_bending_deflection(L: float, a: float, E: float, I: float, P_total: float) -> float:
    """4점 굽힘 중앙 처짐: w = (Pa / 24EI) * (3L^2 - 4a^2)
    P_total은 두 지점 하중의 합. 각 지점 하중 P = P_total/2
    a는 지점부터 하중점까지의 거리.
    """
    P = P_total / 2.0
    return (P * a / (24.0 * E * I)) * (3 * L**2 - 4 * a**2)

def beam_3point_bending_stress(L: float, b: float, t: float, P: float) -> float:
    """3점 굽힘 중앙하면 최대 굽힘 응력: sigma = 3PL / (2bt^2)"""
    return (3.0 * P * L) / (2.0 * b * t**2)

def beam_4point_bending_stress(a: float, b: float, t: float, P_total: float) -> float:
    """4점 굽힘 일정 구간 최대 응력: sigma = 6Pa / (bt^2) (P=P_total/2)"""
    P = P_total / 2.0
    return (6.0 * P * a) / (b * t**2)

def plate_twisting_stress(Mxy: float, t: float) -> float:
    """판 비틀림 전단 응력 대등 Von-Mises: Mxy에 의한 Tau_xy = 6Mxy/t^2
    Von-Mises sigma = sqrt(3) * Tau_xy
    """
    tau = (6.0 * Mxy) / (t**2)
    return np.sqrt(3.0) * tau

def plate_simply_supported_max_stress(Lx: float, Ly: float, t: float, q: float, nu: float) -> float:
    """단순지지 판 등분포 하중(q) 중앙 최대 응력 (Navier 해 기반 근사).
    Lx=Ly 정방형 판의 경우 sigma = 0.287 * q * Lx^2 / t^2 (nu=0.3)
    """
    # 여기서는 좀 더 일반적인 Naviers 급수 기반 계수를 약식 사용 (정방형 기준)
    aspect = Lx / Ly
    if abs(aspect - 1.0) < 0.1: beta = 0.287
    elif aspect > 1.5: beta = 0.45 
    else: beta = 0.35
    return beta * q * Lx**2 / t**2


def pure_bending_curvature(M: float, E: float, t: float, nu: float):
    """
    순수 굽힘 하중의 이론 곡률 및 표면 응력.

    Args:
        M  : 굽힘 모멘트 결합력 (단위 폭당) [N·mm/mm]
        E  : 탄성계수 [MPa]
        t  : 두께 [mm]
        nu : 포아송비 [-]
    Returns:
        kappa_x : X 방향 곡률 [1/mm]
        sigma_top : 표면 응력 [MPa] (z = +t/2)
    """
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    kappa_x = M / D
    sigma_top = -6.0 * M / (t**2)   # 상면 (z = +t/2)  M>0 → 압축
    return kappa_x, sigma_top
