import pandas as pd
import numpy as np
import torch
from models.ae_kan import KANAutoencoder
import matplotlib.pyplot as plt



def preprocessing_dataset(
    log_returns_df: pd.DataFrame,
    win: int = 60,
    min_periods: int = 40,
    clip_val: float = 3.0,
    min_valid_per_day: int = 30,
    use_median: bool = True,
    soft_weights: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prétraitement CAUSAL (dates x tickers) pour KAN/PCA.
    Retourne:
      X_df: z-score 60j -> de-mean par date -> clip [-clip_val, +clip_val]
      W_df: poids soft 0..1 (ou mask binaire en float si soft_weights=False)
      M_df: mask dur 0/1 (uint8)
    """
    df = log_returns_df.sort_index().copy()

    # 1) Rolling z-score causal (pas de center, pas d'info future)
    mu  = df.rolling(win, min_periods=min_periods).mean()
    sig = df.rolling(win, min_periods=min_periods).std(ddof=0)
    z = (df - mu) / sig

    # Mask dur: z défini ET sigma>0
    M_df = (z.notna()) & (sig.gt(0))

    # 2) De-mean cross-sectionnel (médiane robuste par défaut)
    center = z.median(axis=1) if use_median else z.mean(axis=1)
    z_dm = z.sub(center, axis=0)

    # 3) Clip dans [-clip_val, +clip_val]
    X_df = z_dm.clip(-clip_val, clip_val)

    # 4) Poids soft de fiabilité (0..1) selon taille de fenêtre disponible
    if soft_weights:
        cnt = df.rolling(win, min_periods=1).count()  # 1..win
        denom = max(1, win - min_periods)             # ex: 20 si 60/40
        rel = ((cnt - min_periods) / denom).clip(lower=0.0, upper=1.0)
        W_df = (M_df.astype(float) * rel).astype(float)
    else:
        W_df = M_df.astype(float)

    # 5) Filtrer les dates trop creuses
    keep = M_df.sum(axis=1) >= int(min_valid_per_day)
    X_df = X_df.loc[keep]
    M_df = M_df.loc[keep]
    W_df = W_df.loc[keep]

    # 6) Fill NA après avoir figé le mask/poids
    X_df = X_df.fillna(0.0)
    W_df = W_df.fillna(0.0)
    M_df = M_df.astype(np.uint8)

    return X_df, W_df, M_df


import matplotlib.pyplot as plt

def plot_training_history(history, hyperparameters):
    # Create a figure with 2 rows and 2 columns of subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot training and validation loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss') 
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot regularization loss
    ax2.plot(history['regularization'])
    ax2.set_title('Regularization Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)

    # Plot skip gain evolution
    ax3.plot(history['skip_gain'])
    ax3.set_title('Skip Gain Evolution')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Skip Gain')
    ax3.grid(True)

    # Plot skip weight norm
    ax4.plot(history['skip_weight_norm'])
    ax4.set_title('Skip Weight Norm')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Weight Norm')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()
    
    # Add hyperparameters as text on the figure
    hyperparams_text = "\n".join([
        f"{k}: {v}" for k, v in hyperparameters.items()
    ])
    
    # Add text box with hyperparameters
    fig.text(1.02, 0.5, hyperparams_text, 
             bbox=dict(facecolor='white', alpha=0.8),
             fontsize=8, 
             transform=plt.gcf().transFigure,
             verticalalignment='center')
    
    plt.subplots_adjust(right=0.85)
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(f'figures/training_history_{timestamp}.png',
                bbox_inches='tight', 
                dpi=300)
    
    
    

def sample_and_train(hyperparameters_grid, X_train, X_test, device='cuda', epochs=100):
    import random
    
    sampled_params = {}
    for param_name, param_values in hyperparameters_grid.items():
        if isinstance(param_values, list):
            sampled_params[param_name] = random.choice(param_values)
        else:
            sampled_params[param_name] = param_values
            
    return sampled_params

def evaluate_model(model, X_test, device):
    model.eval()
    with torch.no_grad():
        X_test_recon = model(X_test)
        test_loss = model.loss_fn(X_test_recon, X_test).item()
        print(f"Test loss: {test_loss:.4f}")
        
        
def train_kan_model(hyperparameters, X_train, X_test, device='cuda', epochs=100):
    # Convert device string to torch.device
    device = torch.device(device)
    
    # Move data to device
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    
    model = KANAutoencoder(
        input_dim=hyperparameters['input_dim'],
        hidden_dims=hyperparameters['hidden_dims_choices'],
        k=hyperparameters['latent_dims'],
        basis_type=hyperparameters['basis_types'],
        M=hyperparameters['M_values'],
        poly_degree=hyperparameters['poly_degrees'],
        use_silu=hyperparameters['use_silu_choices'],
        dropout_p=hyperparameters['dropout_rates'],
        use_global_skip=hyperparameters['use_global_skip'],
        use_skip=hyperparameters['use_skip_choices'],
        skip_init=hyperparameters['skip_init_choices'],
        skip_gain=hyperparameters['skip_gain_values'],
        lambda_alpha=hyperparameters['lambda_alpha_values'],
        lambda_group=hyperparameters['lambda_group_values'],
        lambda_tv=hyperparameters['lambda_tv_values'],
        lambda_poly_decay=hyperparameters['lambda_poly_decay_values'],
        lambda_skip_l2=hyperparameters['lambda_skip_l2_values'],
        loss_type=hyperparameters['loss_types'],
        huber_delta=hyperparameters['huber_deltas']
        
    ).to(device)
    
    history = model.fit(X_train,
            epochs=epochs,
            batch_size=hyperparameters['batch_sizes'],
            learning_rate=hyperparameters['learning_rates'],
            weight_decay=hyperparameters['weight_decays'],
            validation_split=0.2,
            patience=15,
            verbose=True,
            lambda_reg=hyperparameters['lambda_reg_values'],
            device=device
            )
    plot_training_history(history, hyperparameters)
    print(f"Training time: {history['training_time']:.2f} seconds")
    evaluate_model(model, X_test, device)