# CLAUDE.md — MSc Thesis Project

## Project Identity

**Institution**: Stockholm School of Economics (SSE)  
**Degree**: MSc in Finance  
**Topic**: Realized Variance Forecasting for the EURO STOXX 50 Index using 5-Minute Intraday Data  
**Reference Paper**: Christensen et al. (2023) — located at `Papers/Christensen et al. (2023).pdf`

This is an academic thesis. Professionalism, structure, and clarity are non-negotiable standards — in code, notebooks, and the final LaTeX document. All work should be reproducible, well-commented, and suitable for academic scrutiny.

---

## Research Overview

The thesis replicates and extends the methodology of Christensen et al. (2023) in a European equity market context. The focus is on one-step-ahead realized variance (RV) forecasting for the **EURO STOXX 50 single index** (not its individual constituents). A systematic comparison of traditional econometric benchmark models, regularized linear models, tree-based ensemble methods, and deep learning architectures is conducted across two feature sets.

**Key extensions beyond the reference paper:**
- Application to the European equity market (EURO STOXX 50)
- Inclusion of macro-financial predictors (MALL feature set) to assess whether external information improves volatility forecasts
- Systematic benchmarking across a broader model space than the reference paper

**Primary evaluation metric**: Mean Squared Error (MSE)  
**Secondary metrics**: RMSE, MAE, Mean Error

---

## Data

### Source & Coverage
- **Raw data**: Intraday minute-level price data for EURO STOXX 50 (`.STOXX50E`)
- **Full sample**: 1999-01-05 to 2020-08-06 (~21 years of trading days)
- **Test period**: 2016-04-21 to 2020-08-06 (1,099 observations)

### Train/Validation/Test Split (70/10/20)
| Split | Period | Observations |
|---|---|---|
| Train | 1999-01-05 to 2014-02-26 | 3,842 |
| Validation | 2014-02-27 to 2016-04-20 | 548 |
| Test | 2016-04-21 to 2020-08-06 | 1,099 |

**Rolling window size**: 4,390 (train + validation) — used for benchmark model estimation.

### Cleaned Data Files
| File | Rows | Columns | Description |
|---|---|---|---|
| `Cleaned Data Sets/MHAR_5min_intraday.csv` | 5,674 | 13 | HAR-family features |
| `Cleaned Data Sets/MALL_5min.csv` | 5,489 | 32 | Extended macro-financial features |

---

## Feature Sets

### MHAR Feature Set (13 columns)
Core realized variance measures constructed from 5-minute intraday returns:

| Feature | Description |
|---|---|
| `RVD` | Daily realized variance |
| `RVW` | 5-day rolling average RV (weekly) |
| `RVM` | 22-day rolling average RV (monthly) |
| `logRVD`, `logRVW`, `logRVM` | Log-transformed variants |
| `RVD_pos`, `RVD_neg` | Positive/negative semivariance (for SHAR) |
| `RD_neg`, `RW_neg`, `RM_neg` | Negative return aggregates (for LevHAR) |
| `RQ` | Realized quarticity (fourth moment) |

### MALL Feature Set (32 columns)
MHAR features + 19 macro-financial predictors:

| Feature | Description |
|---|---|
| `VSTOXX`, `logVSTOXX` | European volatility index |
| `EPU`, `logEPU` | Economic Policy Uncertainty index |
| `EPU_1m_lag`, `logEPU_1m_lag` | 1-month lagged EPU |
| `EURUSD`, `EURUSD_ret` | EUR/USD exchange rate and returns |
| `BRENT`, `BRENT_ret` | Brent crude oil price and returns |
| `NIKKEI_ret`, `NIKKEI_sq` | Nikkei returns and squared returns |
| `SP500_ret_lag`, `SP500_sq_lag` | Lagged S&P 500 returns and squared returns |
| `M1W`, `M1M` | Weekly and monthly monetary aggregates |
| `RV_target`, `logRV_target` | Next-day RV (forecasting target) |

---

## Methodology

- **Forecasting horizon**: One-step-ahead (t+1)
- **Estimation approach**: Rolling window OLS for benchmark models; fixed split for ML/DL models
- **Insanity filter**: Negative forecasts replaced with the minimum in-sample realized variance (following Bollerslev et al., 2016)
- **Standardization**: Predictors standardized using training-sample mean and standard deviation before feeding into neural networks

---

## Model Inventory

### MHAR Models (15 total)

**Benchmark Models** (6):
- `HAR` — Heterogeneous Autoregressive (Corsi, 2009)
- `LogHAR` — HAR on log-transformed RV
- `HARQ` — HAR augmented with Realized Quarticity
- `SHAR` — Semivariance HAR (positive/negative decomposition)
- `LevHAR` — Leverage-augmented HAR
- `HARX` — HAR with exogenous variables

**Regularized Linear Models** (3): Ridge Regression (RR), LASSO (LA), Elastic Net (EN)

**Tree-Based Ensemble Models** (3): Random Forest (RF), Gradient Boosting (GB), Bagging (BG)

**Neural Network Models** (4): NN1 (1 hidden layer), NN2 (2 hidden layers), NN3 (3 hidden layers), HARNet (CNN architecture)

### MALL Models (10 total)
HARX benchmark + all regularized, tree-based, and NN models (excluding HAR variants specific to MHAR).

---

## Key Results (Current)

| Model | Feature Set | MSE |
|---|---|---|
| HARQ | MHAR | 4.792e-08 |
| Random Forest | MALL | 4.840e-08 |
| NN1 | MALL | 4.964e-08 |
| LASSO | MALL | 5.078e-08 |
| HAR | MHAR | 5.664e-08 |

MALL features improve 7 out of 9 comparable models, with regularized linear models showing the largest gains (10–16% MSE reduction).

---

## Folder Structure

```
5-min RV/
├── CLAUDE.md                          ← This file
├── Python Code/
│   ├── Cleaning Raw Data/
│   │   └── MHAR_5min.ipynb            ← Data cleaning & MHAR construction
│   ├── Constructing MALL Feature Set/
│   │   └── MALL_5min.ipynb            ← Macro feature engineering
│   ├── MHAR Results/
│   │   ├── Benchmark Models/          ← 6 econometric benchmarks
│   │   ├── Regularized Models/        ← Ridge, LASSO, Elastic Net
│   │   ├── Tree-based Models/         ← RF, GB, Bagging
│   │   ├── Neural Networks/           ← NN1, NN2, NN3, HARNet
│   │   └── MHAR Empirical Results/    ← Aggregated performance analysis
│   └── MALL Results/
│       ├── Benchmark Models/
│       ├── Regularized Models/
│       ├── Tree-based Models/
│       └── Neural Networks/
├── Cleaned Data Sets/
│   ├── MHAR_5min_intraday.csv
│   ├── MALL_5min.csv
│   ├── Forecasts 5-min/               ← Model forecast CSVs
│   └── MSE 5-min/                     ← Performance metric CSVs
├── Figures/
│   ├── Data/                          ← EDA visualizations
│   └── Results/                       ← Model comparison plots
└── Papers/
    └── Christensen et al. (2023).pdf  ← Primary reference
```

---

## Output File Conventions

| Output Type | Naming Convention | Location |
|---|---|---|
| Forecasts | `forecasts_[MODEL].csv` | `Cleaned Data Sets/Forecasts 5-min/` |
| Performance metrics | `mse_summary_[MODEL].csv` | `Cleaned Data Sets/MSE 5-min/` |

Forecast CSVs contain: `Date`, `RV_actual`, `RV_forecast` (1,099 rows — test period).  
MSE summary CSVs contain: `MSE`, `RMSE`, `MAE`, `Mean_Error`. Neural network files include a second row for the top-10 ensemble forecast.

---

## Code Quality & Conventions

- **Code origin**: Much of the code was generated by Codex (an AI agent). **Always verify logic, formulas, and indexing before trusting results.** Known risk areas: rolling window alignment, date handling, and feature lag construction.
- **Notebooks**: Each notebook should be self-contained, clearly structured with markdown headers, and runnable top-to-bottom without errors.
- **Academic standard**: As an MSc Thesis, all code, comments, and outputs must meet a professional academic standard. Avoid throwaway variable names, magic numbers without explanation, or undocumented design choices.
- **Do not modify cleaned data files** without updating the cleaning notebook to match.
- **LaTeX document**: Figures and tables exported for the thesis should be publication-quality (appropriate font sizes, axis labels, consistent styling).

---

## Important Warnings

- Never assume Codex-generated code is correct — audit formulas against the reference paper before using results.
- The rolling window in benchmark models uses a window of exactly 4,390 observations (train + validation). Do not change this without justification.
- The insanity filter must be applied consistently across all models for fair comparison.
- MALL has fewer rows than MHAR (5,489 vs 5,674) due to missing macro data at the start of the sample — account for this when aligning datasets.
