# Equivalent Sheet Model - Verification Report

## 1. Modal Analysis Comparison

| Mode   | Target Freq (Hz) | Opt Freq (Hz)   | Freq Error (%) | MAC    |
|--------|------------------|-----------------|----------------|--------|
| 1      |           6.9376 |          8.1081 |          16.87 | 0.9824 |
| 2      |          12.4962 |         12.5320 |           0.29 | 0.9865 |
| 3      |          15.7843 |         17.9216 |          13.54 | 0.9414 |
| 4      |          20.5174 |         21.3100 |           3.86 | 0.9769 |
| 5      |          36.6464 |         37.2852 |           1.74 | 0.3389 |
| **AVG** |                - |               - |       **7.26** | **0.8452** |

### Mode-by-Mode Quality Assessment

- **Mode 1**: f_tgt=6.9376Hz, f_opt=8.1081Hz, Δf=16.87% (Poor), MAC=0.9824 (Excellent)
- **Mode 2**: f_tgt=12.4962Hz, f_opt=12.5320Hz, Δf=0.29% (Excellent), MAC=0.9865 (Excellent)
- **Mode 3**: f_tgt=15.7843Hz, f_opt=17.9216Hz, Δf=13.54% (Poor), MAC=0.9414 (Good)
- **Mode 4**: f_tgt=20.5174Hz, f_opt=21.3100Hz, Δf=3.86% (Good), MAC=0.9769 (Excellent)
- **Mode 5**: f_tgt=36.6464Hz, f_opt=37.2852Hz, Δf=1.74% (Excellent), MAC=0.3389 (Poor)

> **MAC Interpretation:** ≥0.95: Excellent | ≥0.90: Good | ≥0.80: Acceptable | <0.80: Poor

## 2. Static Analysis Comparison

### 2.1 Similarity & Correlation Metrics

| Case        | Disp Sim% | Disp R²% | Corr% | Stress Sim% | Stress R²% | Corr% |
|-------------|-----------|----------|-------|-------------|------------|-------|
| twist_x     |     98.63 |    99.38 |  99.7 |       90.52 |       0.00 |  74.8 |
| twist_y     |     98.09 |    98.81 |  99.5 |       89.48 |      43.49 |  95.0 |
| bend_y      |     96.37 |    98.58 |  99.7 |       79.56 |       0.00 |  51.1 |
| bend_x      |     94.58 |    96.22 |  99.3 |       81.56 |      11.17 |  66.0 |
| lift_br     |     99.32 |    99.87 | 100.0 |       77.45 |       0.00 |  65.1 |
| lift_bl     |     99.38 |    99.91 | 100.0 |       77.45 |       0.00 |  65.1 |
| **AVERAGE** |     97.73 |    98.80 |  99.7 |       82.67 |       9.11 |  69.5 |

### 2.2 Prediction Error (MSE)

| Case        | Disp MSE     | Stress MSE   | Strain MSE (x1e-6)   |
|-------------|--------------|--------------|----------------------|
| twist_x     |     0.020706 |      12.2348 |               0.0010 |
| twist_y     |     0.250591 |    3779.6655 |               0.4266 |
| bend_y      |     0.156780 |      39.2862 |               0.0008 |
| bend_x      |     0.012041 |      19.3303 |               0.0006 |
| lift_br     |     0.000073 |       0.0294 |               0.0000 |
| lift_bl     |     0.000658 |       0.2649 |               0.0000 |
| **AVERAGE** |     0.073475 |     641.8019 |               0.0715 |

### 2.3 Displacement Values (mm)

| Case        | Max|w| (Tgt)  | Max|w| (Opt)  | Avg|w| (Tgt)  | Avg|w| (Opt)  |
|-------------|---------------|---------------|---------------|---------------|
| twist_x     |        5.2372 |        5.2372 |        1.4251 |        1.3989 |
| twist_y     |       13.0930 |       13.0930 |        3.4445 |        3.4065 |
| bend_y      |       10.9124 |       11.2114 |        7.4444 |        7.7548 |
| bend_x      |        2.0235 |        2.3531 |        1.0794 |        1.1518 |
| lift_br     |        1.0000 |        1.0000 |        0.2051 |        0.2102 |
| lift_bl     |        3.0000 |        3.0000 |        0.7131 |        0.6990 |

### 2.4 Stress Values (MPa)

| Case        | Max (Tgt)  | Max (Opt)  | Avg (Tgt)  | Avg (Opt)  | Robust (Tgt) | Robust (Opt) |
|-------------|------------|------------|------------|------------|--------------|--------------|
| twist_x     |     38.111 |     46.709 |      7.074 |      9.569 |       14.838 |       18.696 |
| twist_y     |    584.683 |    800.975 |     54.375 |     77.074 |      258.827 |      399.110 |
| bend_y      |     32.044 |     31.695 |      9.892 |     13.519 |       22.371 |       26.841 |
| bend_x      |     26.243 |     23.073 |      9.849 |     12.468 |       21.511 |       21.240 |
| lift_br     |      0.817 |      1.062 |      0.334 |      0.448 |        0.675 |        0.861 |
| lift_bl     |      2.451 |      3.187 |      1.003 |      1.344 |        2.024 |        2.583 |

### 2.5 Strain Values (×10⁻³)

| Case        | Max (Tgt)  | Max (Opt)  | Avg (Tgt)  | Avg (Opt)  | Robust (Tgt) | Robust (Opt) |
|-------------|------------|------------|------------|------------|--------------|--------------|
| twist_x     |     0.3101 |     0.3880 |     0.1125 |     0.1137 |       0.2617 |       0.2180 |
| twist_y     |     7.0857 |     9.5636 |     0.7090 |     0.8654 |       3.4270 |       4.8394 |
| bend_y      |     0.1692 |     0.1736 |     0.0706 |     0.0756 |       0.1469 |       0.1484 |
| bend_x      |     0.1899 |     0.1245 |     0.0660 |     0.0647 |       0.1401 |       0.1114 |
| lift_br     |     0.0128 |     0.0131 |     0.0053 |     0.0053 |       0.0122 |       0.0102 |
| lift_bl     |     0.0383 |     0.0393 |     0.0160 |     0.0160 |       0.0365 |       0.0306 |

> **Metric Definitions:**
> - **Similarity%** = (1 - NRMSE) × 100, where NRMSE = RMSE/(max-min)
> - **R²** = 1 - SS_res/SS_tot (100% = perfect, 0% = mean prediction)
> - **MSE** = Mean Squared Error (Lower is better, ideal = 0)
> - **Robust Max** = min(μ + 2.5σ, actual_max) - excludes outliers

> **지표 설명 (Korean):**
> - **Similarity% (유사도)**: 전체 범위 대비 오차 비율을 100%에서 뺀 값. 높을수록 좋음.
> - **R² (결정계수)**: 데이터의 변동을 모델이 얼마나 설명하는지. 100%는 완벽, 0%는 평균값 예측.
> - **MSE (평균제곱오차)**: 예측값과 실제값 차이의 제곱 평균. 0에 가까울수록 좋음.
> - **Robust Max**: 이상치(튀는 값)를 제외한 최대값. (평균 + 2.5 × 표준편차)로 제한하여 평가.

## 3. Strain Energy Comparison

| Case        | Target Energy (N·mm) | Opt Energy (N·mm)    | Ratio (%) |
|-------------|----------------------|----------------------|-----------|
| twist_x     |         2.917122e+01 |         3.616301e+01 |    123.97 |
| twist_y     |         2.185689e+03 |         5.565782e+03 |    254.65 |
| bend_y      |         6.129834e+01 |         8.867794e+01 |    144.67 |
| bend_x      |         8.677112e+01 |         9.363422e+01 |    107.91 |
| lift_br     |         6.435941e-02 |         7.828599e-02 |    121.64 |
| lift_bl     |         5.792347e-01 |         7.045739e-01 |    121.64 |
| **AVERAGE** |                    - |                    - |    145.74 |

> **Strain Energy:** U = ∫(0.5·κᵀ·D·κ)dA. Ratio = (Opt/Target) × 100%. Ideal = 100%

## 4. Total Mass Comparison

| Property       | Target              | Optimized           |
|----------------|---------------------|---------------------|
| Mass (tonne)   |        4.497954e-03 |        4.418441e-03 |
| Mass (g)       |           4497.9537 |           4418.4409 |
| Mass (kg)      |            4.497954 |            4.418441 |

- **Mass Ratio:** 98.23%
- **Mass Error:** 1.77%

> **Mass Calculation:** M = ∫(ρ·t)dA, integrated using trapezoidal rule

---

## Summary

| Metric                         | Value         |
|--------------------------------|---------------|
| Modal - Avg Freq Error         |        7.26% |
| Modal - Avg MAC                |       0.8452 |
| Displacement - Avg Similarity  |       97.73% |
| Displacement - Avg R²          |       98.80% |
| Displacement - Avg MSE         |    0.073475 |
| Stress - Avg Similarity        |       82.67% |
| Stress - Avg R²                |        9.11% |
| Stress - Avg MSE               |    641.8019 |
| Strain - Avg Similarity        |       86.96% |
| Strain - Avg R²                |       52.62% |
| Strain - Avg MSE (x1e-6)       |      0.0715 |
| Energy - Avg Ratio             |      145.74% |
| Mass - Ratio                   |       98.23% |
| Mass - Error                   |        1.77% |
