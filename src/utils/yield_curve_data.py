"""
Utility functions for fetching and preprocessing FRED yield curve data.
"""

import pandas as pd
from pandas_datareader import data as web
from datetime import datetime
import numpy as np
from pathlib import Path


def fetch_fred_yield_curve(start="1990-01-01", end=None, out_path="data/raw/fred_yield_curve.csv"):
    """
    Fetch FRED yield curve data.
    
    Parameters
    ----------
    start : str
        Start date in 'YYYY-MM-DD' format
    end : str, optional
        End date in 'YYYY-MM-DD' format. Defaults to today.
    out_path : str
        Path to save the CSV file
        
    Returns
    -------
    pd.DataFrame
        Yield curve data with columns for different maturities
    """
    # Try to load existing file first
    out_path = Path(out_path)
    if out_path.exists():
        try:
            df = pd.read_csv(out_path, index_col=0, parse_dates=True)
            print(f"Loading existing file: {out_path}")
            print(f"FRED Yield Curve: {df.shape[0]} rows × {df.shape[1]} cols")
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
            print("Downloading data...")

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # FRED tickers for different maturities
    fred_tickers = {
        "DGS1MO": "1M",
        "DGS3MO": "3M", 
        "DGS6MO": "6M",
        "DGS1": "1Y",
        "DGS2": "2Y",
        "DGS5": "5Y", 
        "DGS10": "10Y",
        "DGS30": "30Y"
    }

    # Download data for each maturity
    frames = {}
    for code, label in fred_tickers.items():
        try:
            s = web.DataReader(code, "fred", start, end).rename(columns={code: label})
            frames[label] = s
        except Exception as e:
            print(f"Error downloading {code}: {e}")

    # Combine all series
    df = pd.concat(frames.values(), axis=1)
    
    # Forward fill for holidays and resample to business days
    df = df.ffill().dropna(how="all")
    df = df.asfreq("B")
    df = df.ffill()

    # Save to CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=True)
    
    print(f"FRED Yield Curve: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"     Saved to: {out_path}")
    
    return df


def preprocess_yield_curve(df, drop_short_maturities=True):
    """
    Preprocess yield curve data with winsorization and two types of normalization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw yield curve dataset
    drop_short_maturities : bool
        If True, drop 1M maturity due to many NaNs
        
    Returns
    -------
    tuple
        (cross_section_zscore_df, column_zscore_df)
    """
    # Drop columns with too many NaNs (typically 1M maturity)
    if drop_short_maturities:
        cols_to_drop = []
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"Column {col} has {nan_count} NaNs")
                if nan_count > len(df) * 0.1:  # Drop if more than 10% NaNs
                    cols_to_drop.append(col)
        
        if cols_to_drop:
            print(f"Dropping columns: {cols_to_drop}")
            df = df.drop(cols_to_drop, axis=1)
    
    # Winsorize each column at 0.1% and 99.9%
    df_winsorized = df.copy()
    for col in df.columns:
        lower = df[col].quantile(0.001)
        upper = df[col].quantile(0.999)
        df_winsorized[col] = df[col].clip(lower=lower, upper=upper)
    
    # A) Cross-section z-score (standardize each date/row)
    # This emphasizes the shape of the yield curve at each point in time
    cross_section_zscore = (df_winsorized.sub(df_winsorized.mean(axis=1), axis=0)
                           .div(df_winsorized.std(axis=1), axis=0))
    
    # B) Column-wise z-score (standardize each maturity/column)
    # This emphasizes the time series behavior of each maturity
    column_zscore = (df_winsorized - df_winsorized.mean()) / df_winsorized.std()
    
    return cross_section_zscore, column_zscore


def load_preprocessed_yield_curve(start="1990-01-01", end=None, 
                                 out_path="data/raw/fred_yield_curve.csv",
                                 normalization="cross_section"):
    """
    Load and preprocess yield curve data in one step.
    
    Parameters
    ----------
    start : str
        Start date in 'YYYY-MM-DD' format
    end : str, optional
        End date in 'YYYY-MM-DD' format
    out_path : str
        Path to save/load the raw CSV file
    normalization : str
        Type of normalization: 'cross_section', 'column_wise', or 'both'
        
    Returns
    -------
    pd.DataFrame or tuple
        Preprocessed data. If normalization='both', returns tuple.
    """
    # Fetch raw data
    df = fetch_fred_yield_curve(start=start, end=end, out_path=out_path)
    
    # Preprocess
    cross_section_zscore, column_zscore = preprocess_yield_curve(df)
    
    print(f"Cross-section z-score shape: {cross_section_zscore.shape}")
    print(f"Column-wise z-score shape: {column_zscore.shape}")
    
    if normalization == "cross_section":
        return cross_section_zscore
    elif normalization == "column_wise":
        return column_zscore
    elif normalization == "both":
        return cross_section_zscore, column_zscore
    else:
        raise ValueError(f"Unknown normalization type: {normalization}")


def analyze_pca_components(df, n_components=None):
    """
    Perform PCA analysis on yield curve data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed yield curve data
    n_components : int, optional
        Number of components to compute. None for all.
        
    Returns
    -------
    dict
        Dictionary with PCA results
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(df)
    
    # Components needed for 95% variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.argmax(cumsum >= 0.95) + 1
    
    results = {
        'pca': pca,
        'transformed': pca_transformed,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': cumsum,
        'n_components_95': n_95,
        'components': pca.components_,
        'mean': pca.mean_
    }
    
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {cumsum}")
    print(f"Components needed for 95% variance: {n_95}")
    
    return results


def prepare_yield_curve_for_kan(start="1990-01-01", end=None, 
                               normalization="cross_section",
                               test_size=0.2):
    """
    Prepare yield curve data for KAN autoencoder training.
    
    Parameters
    ----------
    start : str
        Start date
    end : str, optional
        End date
    normalization : str
        Type of normalization to use
    test_size : float
        Proportion of data for testing
        
    Returns
    -------
    dict
        Dictionary with train/test splits and metadata
    """
    # Load preprocessed data
    data = load_preprocessed_yield_curve(start=start, end=end, 
                                        normalization=normalization)
    
    # Convert to numpy if it's a single DataFrame
    if isinstance(data, pd.DataFrame):
        X = data.values
        dates = data.index
    else:
        # If both normalizations, use the first one
        X = data[0].values
        dates = data[0].index
    
    # Create train/test split (temporal)
    n_samples = len(X)
    n_train = int(n_samples * (1 - test_size))
    
    X_train = X[:n_train]
    X_test = X[n_train:]
    dates_train = dates[:n_train]
    dates_test = dates[n_train:]
    
    # PCA analysis on training data
    pca_results = analyze_pca_components(pd.DataFrame(X_train))
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'dates_train': dates_train,
        'dates_test': dates_test,
        'n_features': X.shape[1],
        'n_components_95': pca_results['n_components_95'],
        'pca_results': pca_results,
        'normalization': normalization
    }


# Example usage function
def example_usage():
    """Example of how to use these functions."""
    # 1. Simple loading
    cross_section_data = load_preprocessed_yield_curve(
        start="2000-01-01",
        normalization="cross_section"
    )
    
    # 2. Get both normalizations
    cross_section, column_wise = load_preprocessed_yield_curve(
        start="2000-01-01",
        normalization="both"
    )
    
    # 3. Prepare for KAN training
    kan_data = prepare_yield_curve_for_kan(
        start="2000-01-01",
        normalization="cross_section",
        test_size=0.2
    )
    
    print(f"Training data shape: {kan_data['X_train'].shape}")
    print(f"Test data shape: {kan_data['X_test'].shape}")
    print(f"Recommended latent dim (95% variance): {kan_data['n_components_95']}")
    
    return kan_data


if __name__ == "__main__":
    # Run example
    example_usage()
