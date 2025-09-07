import numpy as np
import pandas as pd

# --------------------------------------------
# 1) Détection sur RETURNS SIMPLES (pct_change)
#    r_t = P_t / P_{t-1} - 1
# --------------------------------------------
def detect_splits_from_returns(returns_df: pd.DataFrame, tol=0.05) -> pd.DataFrame:
    """
    returns_df: index = dates, colonnes = tickers, valeurs = returns simples (ex: df.pct_change()).
    tol: tolérance relative (±5% par défaut).

    Renvoie un DataFrame (ticker, date, observed_return, matched_split).
    """
    ratios = {
        "2-for-1": -0.5,
        "3-for-1": -2/3,
        "4-for-1": -0.75,
        "5-for-1": -0.8,
        "10-for-1": -0.9,
        "1-for-2 (reverse)": 1.0,
        "1-for-3 (reverse)": 2.0,
        "1-for-4 (reverse)": 3.0,
        "1-for-5 (reverse)": 4.0,
        "1-for-10 (reverse)": 9.0,
    }
    events = []
    for col in returns_df.columns:
        s = returns_df[col].dropna().astype(float)
        for date, val in s.items():
            # on zappe les variations "normales" (accélère et évite les faux positifs)
            if -0.3 <= val <= 0.3:
                continue
            best_name, best_err = None, np.inf
            for name, target in ratios.items():
                # erreur relative; pour target=0 (n'existe pas ici), on ferait abs(val)
                rel_err = abs(val - target) / abs(target)
                if rel_err < best_err:
                    best_name, best_err = name, rel_err
            if best_err <= tol:
                events.append({
                    "ticker": col,
                    "date": pd.to_datetime(date),
                    "observed_value": float(val),
                    "matched_split": best_name,
                    "space": "simple_return",
                    "rel_error": float(best_err),
                })
    return pd.DataFrame(events).sort_values(["ticker", "date"])


# --------------------------------------------
# 2) Détection sur LOG‑RETURNS
#    l_t = ln(P_t / P_{t-1})
# --------------------------------------------
def detect_splits_from_log_returns(log_returns_df: pd.DataFrame, tol=0.05) -> pd.DataFrame:
    """
    log_returns_df: index = dates, colonnes = tickers, valeurs = log-returns (ex: np.log(P/P.shift(1))).
    tol: tolérance relative (±5% par défaut).

    Renvoie un DataFrame (ticker, date, observed_log_return, matched_split).
    """
    ratios_log = {
        "2-for-1": np.log(1/2),     # ≈ -0.6931
        "3-for-1": np.log(1/3),     # ≈ -1.0986
        "4-for-1": np.log(1/4),     # ≈ -1.3863
        "5-for-1": np.log(1/5),     # ≈ -1.6094
        "10-for-1": np.log(1/10),   # ≈ -2.3026
        "1-for-2 (reverse)": np.log(2),   # ≈ 0.6931
        "1-for-3 (reverse)": np.log(3),   # ≈ 1.0986
        "1-for-4 (reverse)": np.log(4),   # ≈ 1.3863
        "1-for-5 (reverse)": np.log(5),   # ≈ 1.6094
        "1-for-10 (reverse)": np.log(10), # ≈ 2.3026
    }
    events = []
    for col in log_returns_df.columns:
        s = log_returns_df[col].dropna().astype(float)
        for date, val in s.items():
            if -0.3 <= val <= 0.3:
                continue
            best_name, best_err = None, np.inf
            for name, target in ratios_log.items():
                rel_err = abs(val - target) / abs(target)
                if rel_err < best_err:
                    best_name, best_err = name, rel_err
            if best_err <= tol:
                events.append({
                    "ticker": col,
                    "date": pd.to_datetime(date),
                    "observed_value": float(val),
                    "matched_split": best_name,
                    "space": "log_return",
                    "rel_error": float(best_err),
                })
    if not events:
        return pd.DataFrame(columns=["ticker", "date", "observed_value", "matched_split", "space", "rel_error"])
    return pd.DataFrame(events).sort_values(["date"])


# --------------------------------------------
# 3) Helper pour résumer rapidement par ticker
# --------------------------------------------
def summarize_events(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=["ticker", "n_events"])
    out = (events_df
           .groupby("ticker")["matched_split"]
           .count()
           .rename("n_events")
           .reset_index()
           .sort_values("n_events", ascending=False))
    return out


def import_sector_data(sector, data_dir):
    sector_dir = f"{data_dir}/data/processed/sectors/{sector.lower().replace(' ', '_')}"
    sector_returns = pd.read_csv(f"{sector_dir}/returns.csv", index_col=0)
    sector_log_returns = pd.read_csv(f"{sector_dir}/log_returns.csv", index_col=0)
    
    # Remove first row of NaN values
    sector_returns = sector_returns.iloc[1:]
    sector_log_returns = sector_log_returns.iloc[1:]
    
    return sector_returns, sector_log_returns


def perform_pca(data, n_components=None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(scaled_data)
    
    return pca, transformed_data, pca.explained_variance_ratio_

def analyze_pca_results(pca, transformed_data, explained_variance_ratio, data):
    print("Shape of transformed data:", transformed_data.shape)
    print("\nExplained variance ratio:")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
    
    print(f"\nCumulative variance explained: {sum(explained_variance_ratio)*100:.2f}%")
    
    # Create DataFrame with principal components
    pc_df = pd.DataFrame(
        transformed_data,
        columns=[f'PC{i+1}' for i in range(transformed_data.shape[1])],
        index=data.index
    )
    return pc_df

def make_pca_analysis(sector, n_components=3):
    _, sector_log_returns = import_sector_data(sector)

    pca, transformed_data, explained_variance_ratio = perform_pca(sector_log_returns, n_components=n_components)
    pc_df = analyze_pca_results(pca, transformed_data, explained_variance_ratio, sector_log_returns)
    return pc_df


def get_variance_by_sector(sector_list):
    """ Just a quick function to get the variance by sector and have a df directly """
    variance_by_sector = []

    for sector in sector_list:
        if sector != 'unknown':
            # Get data and perform PCA
            _, sector_log_returns = import_sector_data(sector)
            pca, _, explained_variance_ratio = perform_pca(sector_log_returns, n_components=3)
            
            # Store results
            variance_by_sector.append({
                'Sector': sector,
                'PC1 (%)': explained_variance_ratio[0] * 100,
                'PC2 (%)': explained_variance_ratio[1] * 100, 
                'PC3 (%)': explained_variance_ratio[2] * 100,
                'Total (%)': sum(explained_variance_ratio) * 100
            })

    # Create and sort DataFrame
    variance_df = pd.DataFrame(variance_by_sector).sort_values('Total (%)', ascending=False)
    variance_df = variance_df.round(2)
    return variance_df