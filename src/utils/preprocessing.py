"""
Fonctions de prétraitement des données pour KAN-memoire
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, Any


def load_and_preprocess_data(data_path: str) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Charge les données, applique le prétraitement et split en train/val/test
    Args:
        data_path: chemin vers le dossier contenant les données (CSV ou npy)
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    # Exemple : on suppose un fichier data.npy avec X, y
    # Remplacez par votre logique de chargement réelle
    X = np.load(f"{data_path}/X.npy")
    y = np.load(f"{data_path}/y.npy")
    
    # Normalisation
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    
    # Split train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Conversion en tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def causal_preprocessing_sectorial(
    data: pd.DataFrame,
    window_days: int = 60,
    min_periods: int = 40,
    clip_range: Tuple[float, float] = (-3.0, 3.0),
    min_tickers_per_date: int = 5,
    date_col: str = 'date',
    ticker_col: str = 'ticker',
    return_col: str = 'log_return'
) -> Dict[str, Any]:
    """
    Preprocessing causal pour log-returns sectoriels.
    
    Args:
        data: DataFrame avec colonnes [date, ticker, log_return, ...]
        window_days: Taille de la fenêtre rolling (60j)
        min_periods: Minimum d'observations pour calculer la rolling stat (40)
        clip_range: Range de clipping (-3, +3)
        min_tickers_per_date: Nombre minimum de tickers valides par date
        date_col: Nom de la colonne date
        ticker_col: Nom de la colonne ticker
        return_col: Nom de la colonne des log returns
        
    Returns:
        Dict contenant:
        - 'processed_df': DataFrame traité
        - 'tensor_data': torch.Tensor des données (dates, tickers)
        - 'masks': torch.Tensor masks binaires (0/1)
        - 'weights': torch.Tensor poids de fiabilité (0..1)
        - 'date_index': Index des dates
        - 'ticker_index': Index des tickers
    """
    print("🔄 Début du preprocessing causal...")
    
    # Copie pour éviter de modifier l'original
    df = data.copy()
    
    # S'assurer que la date est en datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Trier par ticker et date
    df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)
    
    print(f"📊 Données originales: {len(df)} observations, {df[ticker_col].nunique()} tickers")
    
    # 1. Rolling Z-Score par ticker (CAUSAL)
    print("⚙️  Étape 1/5: Rolling Z-Score causal...")
    
    def rolling_zscore_causal(series):
        """Z-Score rolling avec ddof=0 (causale)"""
        rolling_mean = series.rolling(window=window_days, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window_days, min_periods=min_periods).std(ddof=0)
        # Éviter division par zéro
        rolling_std = rolling_std.fillna(1.0)
        rolling_std = rolling_std.replace(0.0, 1.0)
        zscore = (series - rolling_mean) / rolling_std
        return zscore, rolling_mean, rolling_std
    
    # Appliquer le rolling z-score par ticker
    df['zscore'], df['rolling_mean'], df['rolling_std'] = zip(*df.groupby(ticker_col)[return_col].apply(rolling_zscore_causal))
    df['zscore'] = df['zscore'].astype(float)
    
    # Calculer le nombre d'observations utilisées pour chaque z-score
    df['n_obs'] = df.groupby(ticker_col)[return_col].transform(
        lambda x: x.rolling(window=window_days, min_periods=min_periods).count()
    )
    
    print(f"✅ Z-Scores calculés. NaN: {df['zscore'].isna().sum()}")
    
    # 2. De-mean cross-sectionnel par date (médiane)
    print("⚙️  Étape 2/5: De-mean cross-sectionnel...")
    
    # Calculer la médiane cross-sectionnelle par date
    daily_medians = df.groupby(date_col)['zscore'].median()
    df['daily_median'] = df[date_col].map(daily_medians)
    df['zscore_demeaned'] = df['zscore'] - df['daily_median']
    
    print(f"✅ De-mean effectué. Médiane quotidienne calculée pour {len(daily_medians)} dates")
    
    # 3. Clipping
    print("⚙️  Étape 3/5: Clipping...")
    df['zscore_clipped'] = df['zscore_demeaned'].clip(clip_range[0], clip_range[1])
    
    clipped_count = (df['zscore_demeaned'] != df['zscore_clipped']).sum()
    print(f"✅ Clipping effectué. {clipped_count} valeurs clippées")
    
    # 4. Génération des masks et poids
    print("⚙️  Étape 4/5: Génération masks et poids...")
    
    # Mask dur : 1 si pas NaN et n_obs >= min_periods, 0 sinon
    df['mask_hard'] = (~df['zscore_clipped'].isna() & (df['n_obs'] >= min_periods)).astype(int)
    
    # Poids soft : interpolation linéaire entre min_periods et window_days
    # n_obs = min_periods -> poids = 0
    # n_obs = window_days -> poids = 1
    df['weight_soft'] = np.clip(
        (df['n_obs'] - min_periods) / (window_days - min_periods), 0.0, 1.0
    )
    # Si mask_hard = 0, alors weight_soft = 0
    df['weight_soft'] = df['weight_soft'] * df['mask_hard']
    
    print(f"✅ Masks et poids générés. Observations valides: {df['mask_hard'].sum()}")
    
    # 5. Filtrage des dates avec peu de tickers valides
    print("⚙️  Étape 5/5: Filtrage des dates...")
    
    # Compter le nombre de tickers valides par date
    valid_tickers_per_date = df.groupby(date_col)['mask_hard'].sum()
    valid_dates = valid_tickers_per_date[valid_tickers_per_date >= min_tickers_per_date].index
    
    df_filtered = df[df[date_col].isin(valid_dates)].copy()
    
    removed_dates = len(df[date_col].unique()) - len(valid_dates)
    print(f"✅ Filtrage effectué. {removed_dates} dates supprimées, {len(valid_dates)} dates conservées")
    
    # Création des indices
    unique_dates = sorted(df_filtered[date_col].unique())
    unique_tickers = sorted(df_filtered[ticker_col].unique())
    
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    ticker_to_idx = {ticker: idx for idx, ticker in enumerate(unique_tickers)}
    
    df_filtered['date_idx'] = df_filtered[date_col].map(date_to_idx)
    df_filtered['ticker_idx'] = df_filtered[ticker_col].map(ticker_to_idx)
    
    # Création des tensors
    print("🔧 Création des tensors...")
    
    n_dates = len(unique_dates)
    n_tickers = len(unique_tickers)
    
    # Initialiser avec NaN/0
    tensor_data = torch.full((n_dates, n_tickers), float('nan'))
    tensor_masks = torch.zeros((n_dates, n_tickers), dtype=torch.float32)
    tensor_weights = torch.zeros((n_dates, n_tickers), dtype=torch.float32)
    
    # Remplir les tensors
    for _, row in df_filtered.iterrows():
        d_idx = int(row['date_idx'])
        t_idx = int(row['ticker_idx'])
        
        if not np.isnan(row['zscore_clipped']):
            tensor_data[d_idx, t_idx] = row['zscore_clipped']
        tensor_masks[d_idx, t_idx] = row['mask_hard']
        tensor_weights[d_idx, t_idx] = row['weight_soft']
    
    # Remplacer les NaN par 0 dans tensor_data
    tensor_data = torch.nan_to_num(tensor_data, nan=0.0)
    
    print(f"📊 Résultats finaux:")
    print(f"   • Shape tensors: {tensor_data.shape}")
    print(f"   • Observations valides: {tensor_masks.sum().item():.0f}")
    print(f"   • Poids moyen: {tensor_weights[tensor_masks == 1].mean().item():.3f}")
    print(f"   • Dates: {n_dates}, Tickers: {n_tickers}")
    
    return {
        'processed_df': df_filtered,
        'tensor_data': tensor_data,
        'masks': tensor_masks,
        'weights': tensor_weights,
        'date_index': unique_dates,
        'ticker_index': unique_tickers,
        'date_to_idx': date_to_idx,
        'ticker_to_idx': ticker_to_idx,
        'preprocessing_stats': {
            'original_obs': len(df),
            'final_obs': len(df_filtered),
            'valid_obs': int(tensor_masks.sum().item()),
            'n_dates': n_dates,
            'n_tickers': n_tickers,
            'removed_dates': removed_dates,
            'mean_weight': float(tensor_weights[tensor_masks == 1].mean().item()) if tensor_masks.sum() > 0 else 0.0
        }
    } 