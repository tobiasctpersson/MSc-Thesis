"""
Random Forest Volatility Forecasting for EURO STOXX 50
=======================================================
Replicates and extends Christensen, Siggaard & Veliyev (2023) for European data.

Data assumed: Daily realized variance for EURO STOXX 50, 1998-2020,
plus additional predictors described below.

Structure:
    1. Data loading and preprocessing
    2. Feature engineering (HAR lags + additional predictors)
    3. Train / validation / test split
    4. Random Forest with rolling-window out-of-sample forecasting
    5. HAR benchmark
    6. Forecast evaluation (MSE, R², Diebold-Mariano test)
    7. Variable importance via ALE (Accumulated Local Effects)
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
matplotlib.use("MacOSX")
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import chi2
import statsmodels.api as sm

# For ALE plots (pip install PyALE)
try:
    from PyALE import ale
    ALE_AVAILABLE = True
except ImportError:
    ALE_AVAILABLE = False
    print("PyALE not installed. ALE plots will be skipped.")
    print("Install with: pip install PyALE")

# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load your dataset. Expected columns (adjust to your actual column names):
    
        date        : datetime index
        rv          : daily realized variance (annualized, in percent²)
        vstoxx      : VSTOXX implied volatility index
        vix         : CBOE VIX
        volume      : log-differenced trading volume
        epu_europe  : European Economic Policy Uncertainty index
        rate        : ECB policy rate (first-differenced)
        term_spread : 10yr Bund - short rate
        credit_spread: iTraxx Europe or similar
        eur_usd_rv  : EUR/USD realized variance
        hsi_sq      : Hang Seng squared log-return (overnight spillover)
        m1w         : 1-week cumulative return
        ecb_meeting : dummy = 1 on ECB meeting days
    
    If you don't have all variables, the code will still run on what's available.
    """
    df = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
    df = df.sort_index()
    df = df.loc["1998-01-01":"2020-12-31"]
    return df


def simulate_data(n: int = 5800, seed: int = 42) -> pd.DataFrame:
    """
    Simulate a realistic daily RV dataset for demonstration when no real data
    is available. Uses a simple HAR-type DGP with noise.
    Replace this with load_data() for your actual research.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("1998-01-02", periods=n)
    
    # Simulate RV via a HAR-type process
    rv = np.zeros(n)
    rv[0] = 20.0  # start at ~20% annualised vol²
    for t in range(1, n):
        rv_d  = rv[t-1]
        rv_w  = rv[max(0,t-5):t].mean()
        rv_m  = rv[max(0,t-22):t].mean()
        shock = rng.normal(0, 1) * np.sqrt(rv[t-1]) * 0.3
        rv[t] = max(0.5, 0.05 + 0.25*rv_d + 0.35*rv_w + 0.20*rv_m + shock)
    
    # Crisis bumps: GFC and sovereign debt crisis
    gfc   = (dates >= "2008-09-01") & (dates <= "2009-06-30")
    sdc   = (dates >= "2010-04-01") & (dates <= "2012-12-31")
    covid = (dates >= "2020-02-20") & (dates <= "2020-05-31")
    rv[gfc]   *= rng.uniform(1.5, 3.5, gfc.sum())
    rv[sdc]   *= rng.uniform(1.2, 2.0, sdc.sum())
    rv[covid] *= rng.uniform(2.0, 5.0, covid.sum())
    rv = np.clip(rv, 0.5, 500)
    
    df = pd.DataFrame({"rv": rv}, index=dates)
    
    # Simulate correlated macro/financial predictors
    df["vstoxx"]       = np.sqrt(rv) * rng.uniform(0.9, 1.3, n) + rng.normal(0, 1, n)
    df["vix"]          = df["vstoxx"] * rng.uniform(0.7, 1.2, n) + rng.normal(0, 2, n)
    df["volume"]       = rng.normal(0, 1, n)  # already log-differenced
    df["epu_europe"]   = np.abs(rng.normal(100, 60, n))
    df["rate"]         = rng.normal(0, 0.1, n)  # first-differenced ECB rate
    df["term_spread"]  = rng.normal(1.5, 0.8, n)
    df["credit_spread"]= np.sqrt(rv) * 0.05 + rng.normal(0, 0.5, n)
    df["eur_usd_rv"]   = rv * 0.1 + rng.normal(0, 1, n)
    df["hsi_sq"]       = rng.exponential(0.0002, n)
    df["m1w"]          = rng.normal(0, 2, n)
    df["ecb_meeting"]  = 0
    # Roughly 8 ECB meetings/year
    ecb_idx = rng.choice(n, size=int(n * 8 / 252), replace=False)
    df.loc[df.index[ecb_idx], "ecb_meeting"] = 1
    
    return df


# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

def build_features(df: pd.DataFrame, target_col: str = "rv") -> pd.DataFrame:
    """
    Constructs all predictors from the raw dataframe.

    HAR lags
    --------
    RVD  : yesterday's RV  (lag 1)
    RVW  : average RV over past week  (lags 1-5)
    RVM  : average RV over past month (lags 1-22)

    Semivariances (if rv_pos / rv_neg available)
    --------------------------------------------
    RVP, RVN: positive and negative semivariance lags

    All other columns are lagged by 1 day to avoid look-ahead bias.
    """
    out = pd.DataFrame(index=df.index)
    rv  = df[target_col]

    # --- HAR lags ---
    out["RVD"] = rv.shift(1)
    out["RVW"] = rv.shift(1).rolling(5).mean()
    out["RVM"] = rv.shift(1).rolling(22).mean()

    # --- Optional: semivariance lags ---
    if "rv_pos" in df.columns and "rv_neg" in df.columns:
        out["RVP"] = df["rv_pos"].shift(1)
        out["RVN"] = df["rv_neg"].shift(1)
        out["RVP_W"] = df["rv_pos"].shift(1).rolling(5).mean()
        out["RVN_W"] = df["rv_neg"].shift(1).rolling(5).mean()

    # --- All other predictors: lag 1 to avoid look-ahead ---
    extra_cols = [c for c in df.columns if c not in [target_col, "rv_pos", "rv_neg"]]
    for col in extra_cols:
        out[col] = df[col].shift(1)

    # --- Target (next day RV) ---
    out["target"] = rv  # already the current day RV = next day relative to features

    # Drop rows with NaNs introduced by rolling windows (first ~22 observations)
    out = out.dropna()
    return out


# =============================================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# =============================================================================

def time_series_split(df: pd.DataFrame,
                      train_frac: float = 0.70,
                      val_frac: float   = 0.10):
    """
    Chronological split: 70% train, 10% validation, 20% test.
    Returns indices (integer positions) for each split.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))
    
    idx_train = list(range(0, train_end))
    idx_val   = list(range(train_end, val_end))
    idx_test  = list(range(val_end, n))
    
    print(f"Total observations : {n}")
    print(f"Training set       : {len(idx_train)} obs  "
          f"({df.index[0].date()} – {df.index[train_end-1].date()})")
    print(f"Validation set     : {len(idx_val)} obs  "
          f"({df.index[train_end].date()} – {df.index[val_end-1].date()})")
    print(f"Test set           : {len(idx_test)} obs  "
          f"({df.index[val_end].date()} – {df.index[-1].date()})")
    return idx_train, idx_val, idx_test


# =============================================================================
# 4.  RANDOM FOREST — ROLLING WINDOW FORECASTING
# =============================================================================

def tune_rf_on_validation(X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val:   np.ndarray,
                          y_val:   np.ndarray) -> dict:
    """
    Light hyperparameter search over a small grid on the validation set.
    Consistent with the paper's conservative tuning philosophy.

    Tuned parameters:
        n_estimators : number of trees
        max_features : fraction of features considered at each split (= J/3 default)
        min_samples_leaf : controls tree depth / complexity

    Fixed at defaults:
        bootstrap = True (i.i.d. bootstrap as in paper)
        criterion = 'squared_error'
    """
    param_grid = {
        "n_estimators"    : [200, 500],
        "max_features"    : ["sqrt", 1/3],   # sqrt ≈ Breiman default; 1/3 as in paper
        "min_samples_leaf": [3, 5, 10],
    }
    
    best_mse   = np.inf
    best_params = {}
    
    for n_est in param_grid["n_estimators"]:
        for max_feat in param_grid["max_features"]:
            for min_leaf in param_grid["min_samples_leaf"]:
                rf = RandomForestRegressor(
                    n_estimators     = n_est,
                    max_features     = max_feat,
                    min_samples_leaf = min_leaf,
                    bootstrap        = True,
                    n_jobs           = -1,
                    random_state     = 42,
                )
                rf.fit(X_train, y_train)
                val_mse = mean_squared_error(y_val, rf.predict(X_val))
                if val_mse < best_mse:
                    best_mse    = val_mse
                    best_params = {
                        "n_estimators"    : n_est,
                        "max_features"    : max_feat,
                        "min_samples_leaf": min_leaf,
                    }
    
    print(f"\nBest RF hyperparameters (val MSE = {best_mse:.4f}):")
    for k, v in best_params.items():
        print(f"   {k}: {v}")
    return best_params


def rolling_rf_forecast(data:        pd.DataFrame,
                        feature_cols: list,
                        target_col:   str,
                        idx_train:    list,
                        idx_val:      list,
                        idx_test:     list,
                        best_params:  dict,
                        roll_every:   int = 21) -> np.ndarray:
    """
    Produces out-of-sample one-step-ahead forecasts using a rolling window.

    At each step t in the test set:
        - Training window = all data up to t-1 (expanding window)
          OR fixed-length rolling window of (train+val) size
        - Fit RF with best_params
        - Predict RV at t

    roll_every : re-estimate the model every `roll_every` days (default: monthly)
                 to balance accuracy and computation time.
    
    Returns array of forecasts aligned with test set dates.
    """
    X = data[feature_cols].values
    y = data[target_col].values
    
    # Combine train+val as the initial estimation window
    init_window = idx_train + idx_val
    
    forecasts = np.zeros(len(idx_test))
    rf_model  = None
    
    for i, t in enumerate(idx_test):
        # Re-fit every roll_every days (or on first step)
        if i % roll_every == 0:
            # Expanding window: use all data up to t-1
            est_idx   = init_window + idx_test[:i]  # all available history
            X_est     = X[est_idx]
            y_est     = y[est_idx]
            
            rf_model  = RandomForestRegressor(
                n_estimators     = best_params["n_estimators"],
                max_features     = best_params["max_features"],
                min_samples_leaf = best_params["min_samples_leaf"],
                bootstrap        = True,
                n_jobs           = -1,
                random_state     = 42,
            )
            rf_model.fit(X_est, y_est)
        
        # One-step-ahead forecast
        forecasts[i] = rf_model.predict(X[[t]])[0]
        
        # Insanity filter: replace negative forecasts with minimum in-sample RV
        if forecasts[i] < 0:
            forecasts[i] = y[init_window].min()
    
    return forecasts


# =============================================================================
# 5. HAR BENCHMARK
# =============================================================================

def har_forecast_rolling(data:        pd.DataFrame,
                         feature_cols: list,
                         target_col:   str,
                         idx_train:    list,
                         idx_val:      list,
                         idx_test:     list) -> np.ndarray:
    """
    OLS HAR benchmark with rolling window (re-estimated daily).
    Uses only RVD, RVW, RVM as regressors.
    """
    har_cols  = [c for c in ["RVD", "RVW", "RVM"] if c in feature_cols]
    X_all     = data[har_cols].values
    y_all     = data[target_col].values
    init_idx  = idx_train + idx_val
    forecasts = np.zeros(len(idx_test))
    
    for i, t in enumerate(idx_test):
        est_idx  = init_idx + idx_test[:i]
        X_est    = sm.add_constant(X_all[est_idx])
        y_est    = y_all[est_idx]
        model    = sm.OLS(y_est, X_est).fit()
        x_t      = np.array([1.0] + list(X_all[t]))
        pred     = model.predict(x_t)[0]
        forecasts[i] = max(pred, y_est.min())  # insanity filter
    
    return forecasts


# =============================================================================
# 6. FORECAST EVALUATION
# =============================================================================

def mse_ratio(forecasts_model: np.ndarray,
              forecasts_bench: np.ndarray,
              y_true:          np.ndarray) -> float:
    """Ratio of model MSE to benchmark (HAR) MSE. < 1 means model wins."""
    mse_m = mean_squared_error(y_true, forecasts_model)
    mse_b = mean_squared_error(y_true, forecasts_bench)
    return mse_m / mse_b


def r2_oos(forecasts: np.ndarray, y_true: np.ndarray) -> float:
    """
    Out-of-sample R² as in Campbell & Thompson (2008):
        R²_OOS = 1 - MSE_model / MSE_prev_mean
    where MSE_prev_mean uses the historical mean as the forecast.
    """
    mse_model = mean_squared_error(y_true, forecasts)
    mse_mean  = mean_squared_error(y_true, np.full_like(y_true, y_true.mean()))
    return 1 - mse_model / mse_mean


def diebold_mariano_test(e1: np.ndarray,
                         e2: np.ndarray,
                         h:  int = 1) -> tuple[float, float]:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.
    H0: E[d_t] = 0  where d_t = e1_t² - e2_t²
    H1: E[d_t] > 0  (model 2 is better than model 1)

    Uses Newey-West HAC standard errors for h > 1.
    Returns: (DM statistic, p-value one-sided)
    """
    d   = e1**2 - e2**2        # loss differential
    d_bar = d.mean()
    
    # Newey-West variance with bandwidth h-1
    T   = len(d)
    nw  = sm.stats.sandwich_covariance.cov_hac_simple(
              sm.OLS(d, np.ones(T)).fit(), nlags=max(1, h-1)
          )
    se  = np.sqrt(nw[0, 0] / T)
    
    dm_stat = d_bar / se
    p_val   = 1 - stats.norm.cdf(dm_stat)   # one-sided: H1 e2 better
    return float(dm_stat), float(p_val)


def evaluate_all(y_true:     np.ndarray,
                 fc_rf:      np.ndarray,
                 fc_har:     np.ndarray,
                 test_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Compile a summary table of evaluation metrics."""
    e_rf  = y_true - fc_rf
    e_har = y_true - fc_har
    
    dm_stat, dm_pval = diebold_mariano_test(e_har, e_rf)  # H1: RF better than HAR
    
    results = pd.DataFrame({
        "Model": ["HAR", "Random Forest"],
        "MSE"  : [mean_squared_error(y_true, fc_har),
                  mean_squared_error(y_true, fc_rf)],
        "RMSE" : [np.sqrt(mean_squared_error(y_true, fc_har)),
                  np.sqrt(mean_squared_error(y_true, fc_rf))],
        "R²_OOS": [r2_oos(fc_har, y_true),
                   r2_oos(fc_rf,  y_true)],
        "MSE Ratio (vs HAR)": [1.0, mse_ratio(fc_rf, fc_har, y_true)],
    })
    
    print("\n" + "="*60)
    print("OUT-OF-SAMPLE FORECAST EVALUATION")
    print("="*60)
    print(results.to_string(index=False))
    print(f"\nDiebold-Mariano test (H1: RF beats HAR)")
    print(f"   DM statistic : {dm_stat:.3f}")
    print(f"   p-value (1-sided): {dm_pval:.4f}")
    if dm_pval < 0.01:
        print("   *** Significant at 1% level")
    elif dm_pval < 0.05:
        print("   **  Significant at 5% level")
    elif dm_pval < 0.10:
        print("   *   Significant at 10% level")
    else:
        print("   Not significant at conventional levels")
    
    return results


# =============================================================================
# 7. VARIABLE IMPORTANCE VIA ALE
# =============================================================================

def compute_ale_importance(rf_model:     RandomForestRegressor,
                           X_train_df:   pd.DataFrame,
                           feature_cols: list,
                           top_n:        int = 10) -> pd.DataFrame:
    """
    Compute variable importance using:
      (a) RF built-in MDI (Mean Decrease in Impurity) — fast but biased toward
          high-cardinality features
      (b) ALE-based importance following Greenwell et al. (2018), as in the paper
          (requires PyALE package)

    Returns a DataFrame ranked by importance.
    """
    # --- (a) MDI importance (always available) ---
    mdi = pd.Series(
        rf_model.feature_importances_,
        index=feature_cols,
        name="MDI_importance"
    ).sort_values(ascending=False)
    
    print("\nTop features by MDI importance:")
    print(mdi.head(top_n).to_string())
    
    # --- (b) ALE importance (if PyALE available) ---
    ale_importance = pd.Series(dtype=float)
    
    if ALE_AVAILABLE:
        print("\nComputing ALE importance (this may take a moment)...")
        ale_scores = {}
        for feat in feature_cols:
            try:
                ale_df = ale(
                    X        = X_train_df,
                    model    = rf_model,
                    feature  = [feat],
                    grid_size= 50,
                    plot     = False,
                )
                # ALE importance = standard deviation of ALE values (Greenwell 2018)
                ale_scores[feat] = ale_df["eff"].std()
            except Exception:
                ale_scores[feat] = np.nan
        
        ale_importance = pd.Series(ale_scores, name="ALE_importance")
        ale_importance = ale_importance / ale_importance.sum()  # normalise
        ale_importance = ale_importance.sort_values(ascending=False)
        
        print("\nTop features by ALE importance (normalised):")
        print(ale_importance.head(top_n).to_string())
    
    # Combine into summary
    importance_df = pd.DataFrame({"MDI": mdi})
    if not ale_importance.empty:
        importance_df["ALE"] = ale_importance
    importance_df = importance_df.sort_values("MDI", ascending=False)
    
    return importance_df


# =============================================================================
# 8. PLOTTING
# =============================================================================

def plot_forecasts(y_true:     np.ndarray,
                  fc_rf:      np.ndarray,
                  fc_har:     np.ndarray,
                  test_dates: pd.DatetimeIndex,
                  save_path:  str = None):
    """Plot actual vs. forecasted realized variance in the test window."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # -- Panel A: Level forecasts --
    ax = axes[0]
    ax.plot(test_dates, y_true, color="black",  lw=0.8, label="Realized RV", alpha=0.9)
    ax.plot(test_dates, fc_har, color="steelblue", lw=1.0, ls="--", label="HAR", alpha=0.85)
    ax.plot(test_dates, fc_rf,  color="tomato",    lw=1.0, label="Random Forest", alpha=0.85)
    ax.set_ylabel("Realized Variance")
    ax.set_title("Out-of-Sample Volatility Forecasts — EURO STOXX 50")
    ax.legend(frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    
    # -- Panel B: Squared errors (rolling 22-day average) --
    ax2 = axes[1]
    se_har = pd.Series((y_true - fc_har)**2, index=test_dates).rolling(22).mean()
    se_rf  = pd.Series((y_true - fc_rf)**2,  index=test_dates).rolling(22).mean()
    ax2.plot(test_dates, se_har, color="steelblue", lw=1.0, ls="--", label="HAR squared error")
    ax2.plot(test_dates, se_rf,  color="tomato",    lw=1.0, label="RF squared error")
    ax2.set_ylabel("Squared Error (22-day MA)")
    ax2.set_title("Forecast Errors Over Time")
    ax2.legend(frameon=True)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def plot_variable_importance(importance_df: pd.DataFrame,
                             top_n: int = 12,
                             save_path: str = None):
    """Horizontal bar chart of variable importance."""
    df = importance_df.head(top_n).copy()
    
    n_metrics = df.shape[1]
    fig, axes  = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for ax, col in zip(axes, df.columns):
        vals = df[col].dropna().sort_values(ascending=True)
        ax.barh(vals.index, vals.values, color="steelblue", edgecolor="white")
        ax.set_xlabel("Importance")
        ax.set_title(f"Variable Importance ({col})")
        ax.spines[["top", "right"]].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


