"""
Fonctions de prétraitement des données pour KAN-memoire
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


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