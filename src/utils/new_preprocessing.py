import pandas as pd
import numpy as np
import torch
from models.ae_kan import KANAutoencoder
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from pathlib import Path


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
    
    
    
#______________________________________________________________________________


def simple_train_kan(sector_name, sector_data, hyperparams, epochs=100, save_results=True, plot_results=True, save_here=None, device='cuda'):
    print(f"🚀 Entraînement simple pour le secteur: {sector_name}")
    
    # Convert device string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)
    
    X_train = sector_data['train']['X']
    X_test = sector_data['test']['X']
    input_dim = X_train.shape[1]
    
    print(f"📊 Dimensions: Train {X_train.shape}, Test {X_test.shape}")
    
    # Générer la grille d'hyperparamètres
    # sampled_params = create_hyperparams_grid(input_dim)
    
    print()
    print(f"   - Architecture: {input_dim} → {hyperparams['hidden_dims_choices']} → {hyperparams['latent_dims']}")
    print(f"   - Basis: {hyperparams['basis_types']}, M: {hyperparams['M_values']}")
    print(f"   - Learning rate: {hyperparams['learning_rates']:.1e}")
    print(f"   - Batch size: {hyperparams['batch_sizes']}")
    print(f"   - Dropout: {hyperparams['dropout_rates']}")
    print(f"   - Use SiLU: {hyperparams['use_silu_choices']}")
    print(f"   - Skip connections: {hyperparams['use_skip_choices']}")
    print(f"   - Skip init: {hyperparams['skip_init_choices']}")
    print(f"   - Skip gain: {hyperparams['skip_gain_values']}")
    print(f"   - Max skip gain: {hyperparams['max_skip_gain']}")
    print(f"   - Global skip: {hyperparams['use_global_skip']}")
    print(f"   - Loss type: {hyperparams['loss_types']}")
    print(f"   - Huber delta: {hyperparams['huber_deltas']}")
    print(f"   - Weight decay: {hyperparams['weight_decays']}")
    print(f"   - Lambda reg: {hyperparams['lambda_reg_values']}")
    print(f"   - Lambda alpha: {hyperparams['lambda_alpha_values']}")
    print(f"   - Lambda group: {hyperparams['lambda_group_values']}")
    print(f"   - Lambda TV: {hyperparams['lambda_tv_values']}")
    print(f"   - Lambda poly decay: {hyperparams['lambda_poly_decay_values']}")
    print(f"   - Lambda skip L2: {hyperparams['lambda_skip_l2_values']}")
    print(f"   - Poly degree: {hyperparams['poly_degrees']}")
    print()
    
    # Entraîner le modèle
    start_time = time.time()
    
    try:
        model = KANAutoencoder(
            input_dim=input_dim,
            hidden_dims=hyperparams['hidden_dims_choices'],
            k=hyperparams['latent_dims'],
            basis_type=str(hyperparams['basis_types']),  # Convert to string
            M=hyperparams['M_values'],
            poly_degree=hyperparams['poly_degrees'],
            use_silu=hyperparams['use_silu_choices'],
            dropout_p=hyperparams['dropout_rates'],
            use_global_skip=hyperparams['use_global_skip'],
            use_skip=hyperparams['use_skip_choices'],
            skip_init=hyperparams['skip_init_choices'],
            skip_gain=hyperparams['skip_gain_values'],
            max_skip_gain=hyperparams['max_skip_gain'],
            lambda_alpha=hyperparams['lambda_alpha_values'],
            lambda_group=hyperparams['lambda_group_values'],
            lambda_tv=hyperparams['lambda_tv_values'],
            lambda_poly_decay=hyperparams['lambda_poly_decay_values'],
            lambda_skip_l2=hyperparams['lambda_skip_l2_values'],
            loss_type=str(hyperparams['loss_types']),  # Convert to string
            huber_delta=hyperparams['huber_deltas']
        ).to(device)
        
        # Entraînement
        history = model.fit(
            X=X_train,
            epochs=epochs,
            batch_size=int(hyperparams['batch_sizes']),
            learning_rate=float(hyperparams['learning_rates']),
            weight_decay=float(hyperparams['weight_decays']),
            validation_split=0.2,
            patience=15,
            verbose=True,
            lambda_reg=float(hyperparams['lambda_reg_values']),
            device=device
        )
        
        training_time = time.time() - start_time
        
        # Évaluation sur le test
        model.eval()
        with torch.no_grad():
            X_test_recon, latent_test = model(X_test.to(device))
            test_criterion = model.get_loss_criterion()
            test_loss = test_criterion(X_test_recon, X_test.to(device)).item()
            
            # Métriques supplémentaires
            mse_loss = torch.nn.MSELoss()(X_test_recon, X_test.to(device)).item()
            mae_loss = torch.nn.L1Loss()(X_test_recon, X_test.to(device)).item()
        
        print(f"✅ Entraînement terminé en {training_time:.1f}s")
        print(f"📊 Test Loss: {test_loss:.6f}")
        print(f"📏 MSE: {mse_loss:.6f}, MAE: {mae_loss:.6f}")
        
        # Visualiser les résultats si demandé
        if plot_results:
            plot_training_history(history, hyperparams)
        
        # Préparer les résultats
        results = {
            'sector': sector_name,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'hyperparameters': hyperparams,
            'history': history,
            'test_loss': test_loss,
            'mse_loss': mse_loss,
            'mae_loss': mae_loss,
            'training_time': training_time,
            'epochs_trained': len(history['train_loss']),
            'best_val_loss': min(history['val_loss']) if history['val_loss'] else None,
            'final_skip_gain': history['skip_gain'][-1] if history['skip_gain'] else None,
            'model_state': model.state_dict()
        }
        
        # Sauvegarder les résultats si demandé
        if save_results:
            results_dir = Path(save_here) if save_here is not None else Path('results')
            results_dir.mkdir(exist_ok=True)
            
            # Sauvegarder sans le model_state (trop lourd pour JSON)
            results_to_save = {k: v for k, v in results.items() if k != 'model_state'}
            
            filename = f"simple_kan_training_{sector_name}_{results['timestamp']}.json"
            filepath = results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)
            
            print(f"💾 Résultats sauvegardés: {filepath}")
        
        return results
        
    except Exception as e:
        print(f"❌ Erreur durant l'entraînement: {str(e)}")
        return None



def hyperparameter_comparison(sector_name, sector_data, base_hyperparams, hp_to_test, values_to_test, 
                             epochs=150, save_results=True, save_here="results/hyperparams_comparison"):
    """
    Compare different values of a hyperparameter and generate comparison plots.
    
    Args:
        sector_name (str): Name of the sector
        sector_data (dict): Sector data with train/test splits
        base_hyperparams (dict): Base hyperparameters dictionary
        hp_to_test (str): Name of the hyperparameter to test
        values_to_test (list): List of values to test for the hyperparameter
        epochs (int): Number of training epochs
        save_results (bool): Whether to save results
        save_here (str): Directory to save results
        
    Returns:
        dict: Dictionary containing all results and comparison plots
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import json
    
    print(f"🔬 Comparaison hyperparamètre '{hp_to_test}' pour le secteur: {sector_name}")
    print(f"📋 Valeurs à tester: {values_to_test}")
    print(f"⚙️ Epochs: {epochs}")
    print("-" * 60)
    
    results = {}
    all_histories = {}
    
    # Test each value
    for i, value in enumerate(values_to_test):
        print(f"\n🧪 Test {i+1}/{len(values_to_test)}: {hp_to_test} = {value}")
        
        # Create modified hyperparameters
        test_hyperparams = change_hyperparam(base_hyperparams.copy(), hp_to_test, value)
        
        # Train model
        result = simple_train_kan(
            sector_name=f"{sector_name}_{hp_to_test}_{value}",
            sector_data=sector_data,
            hyperparams=test_hyperparams,
            epochs=epochs,
            save_results=save_results,
            plot_results=False,  # We'll make our own plots
            save_here=save_here
        )
        
        if result is not None:
            results[value] = result
            all_histories[value] = result['history']
            print(f"✅ Terminé - Loss final: {result['test_loss']:.6f}")
        else:
            print(f"❌ Échec pour {hp_to_test} = {value}")
    
    # Generate comparison plots
    if results:
        print("\n📊 Génération des graphiques de comparaison...")
        fig = plot_hyperparameter_comparison(all_histories, results, hp_to_test, values_to_test)
        
        # Save comparison plot
        if save_results:
            results_dir = Path(save_here)
            results_dir.mkdir(exist_ok=True, parents=True)
            
            plot_filename = f"comparison_{sector_name}_{hp_to_test}.png"
            plot_path = results_dir / plot_filename
            fig.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"💾 Graphique sauvegardé: {plot_path}")
            
            # Save summary results
            summary = create_comparison_summary(results, hp_to_test, values_to_test)
            summary_filename = f"summary_{sector_name}_{hp_to_test}.json"
            summary_path = results_dir / summary_filename
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"💾 Résumé sauvegardé: {summary_path}")
        
        plt.show()
        
        # Print comparison table
        print_comparison_table(results, hp_to_test, values_to_test)
        
    return results


def plot_hyperparameter_comparison(all_histories, results, hp_name, values):
    """
    Create simple comparison plots like plot_training_history but with multiple hyperparameters.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a figure with 2 rows and 2 columns of subplots (same as plot_training_history)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Comparaison Hyperparamètre: {hp_name}', fontsize=16, fontweight='bold')
    
    # Couleurs distinctes et facilement reconnaissables
    base_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    colors = base_colors[:len(values)]  # Prendre autant de couleurs que de valeurs
    
    # Si on a plus de valeurs que de couleurs prédéfinies, compléter avec tab10
    if len(values) > len(base_colors):
        additional_colors = plt.cm.tab10(np.linspace(0, 1, len(values) - len(base_colors)))
        colors.extend(additional_colors)
    
    # Plot 1: Training and Validation Loss (same as original)
    for value, color in zip(values, colors):
        if value in all_histories:
            history = all_histories[value]
            ax1.plot(history['train_loss'], label=f'Train {hp_name}={value}', color=color, linewidth=2)
            if 'val_loss' in history and len(history['val_loss']) > 0:
                ax1.plot(history['val_loss'], label=f'Val {hp_name}={value}', color=color, linewidth=2, linestyle='--')
    
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Regularization Loss (same as original)
    for value, color in zip(values, colors):
        if value in all_histories:
            history = all_histories[value]
            if 'regularization' in history:
                ax2.plot(history['regularization'], label=f'{hp_name}={value}', color=color, linewidth=2)
    
    ax2.set_title('Regularization Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Skip Gain Evolution (same as original)
    for value, color in zip(values, colors):
        if value in all_histories:
            history = all_histories[value]
            if 'skip_gain' in history and len(history['skip_gain']) > 0:
                ax3.plot(history['skip_gain'], label=f'{hp_name}={value}', color=color, linewidth=2)
    
    ax3.set_title('Skip Gain Evolution')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Skip Gain')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Skip Weight Norm (same as original)
    for value, color in zip(values, colors):
        if value in all_histories:
            history = all_histories[value]
            if 'skip_weight_norm' in history and len(history['skip_weight_norm']) > 0:
                ax4.plot(history['skip_weight_norm'], label=f'{hp_name}={value}', color=color, linewidth=2)
    
    ax4.set_title('Skip Weight Norm')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Weight Norm')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save the figure like the original plot_training_history
    from datetime import datetime
    from pathlib import Path
    
    # Create figures directory if it doesn't exist
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = figures_dir / f'hyperparams_comparison_{hp_name}_{timestamp}.png'
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"💾 Graphique sauvegardé: {save_path}")
    
    return fig


def create_comparison_summary(results, hp_name, values):
    """
    Create a summary of hyperparameter comparison results.
    """
    summary = {
        'hyperparameter': hp_name,
        'values_tested': values,
        'results': {}
    }
    
    best_loss = float('inf')
    best_value = None
    
    for value in values:
        if value in results:
            result = results[value]
            summary['results'][str(value)] = {
                'test_loss': result['test_loss'],
                'training_time': result['training_time'],
                'epochs_trained': len(result['history']['train_loss'])
            }
            
            if result['test_loss'] < best_loss:
                best_loss = result['test_loss']
                best_value = value
    
    summary['best_value'] = best_value
    summary['best_loss'] = best_loss
    
    return summary


def print_comparison_table(results, hp_name, values):
    """
    Print a formatted comparison table.
    """
    print(f"\n📋 RÉSUMÉ DE COMPARAISON - {hp_name}")
    print("=" * 80)
    print(f"{'Valeur':<15} {'Test Loss':<12} {'Temps (s)':<12} {'Epochs':<8} {'Rang':<6}")
    print("-" * 80)
    
    # Sort by test loss
    sorted_results = []
    for value in values:
        if value in results:
            sorted_results.append((value, results[value]))
    
    sorted_results.sort(key=lambda x: x[1]['test_loss'])
    
    for rank, (value, result) in enumerate(sorted_results, 1):
        print(f"{str(value):<15} {result['test_loss']:<12.6f} {result['training_time']:<12.1f} "
              f"{len(result['history']['train_loss']):<8} {rank:<6}")
    
    print("-" * 80)
    if sorted_results:
        best_value, best_result = sorted_results[0]
        print(f"🏆 MEILLEUR: {hp_name} = {best_value} (Loss: {best_result['test_loss']:.6f})")


def reconstruct_model_from_results(result, hyperparams, input_dim):
    """
    Reconstruit un modèle KANAutoencoder à partir des résultats et hyperparamètres.
    
    Args:
        result: Résultat d'entraînement contenant 'model_state'
        hyperparams: Dictionnaire d'hyperparamètres utilisés
        input_dim: Dimension d'entrée du modèle
        
    Returns:
        model: Modèle KANAutoencoder reconstruit avec les poids chargés
    """
    from models.ae_kan import KANAutoencoder
    
    # Recréer le modèle avec les mêmes paramètres
    model = KANAutoencoder(
        input_dim=input_dim,
        hidden_dims=hyperparams['hidden_dims_choices'],
        k=hyperparams['latent_dims'],
        basis_type=hyperparams['basis_types'],
        M=hyperparams['M_values'],
        poly_degree=hyperparams['poly_degrees'],
        use_silu=hyperparams['use_silu_choices'],
        dropout_p=hyperparams['dropout_rates'],
        use_global_skip=hyperparams['use_global_skip'],
        use_skip=hyperparams['use_skip_choices'],
        skip_init=hyperparams['skip_init_choices'],
        skip_gain=hyperparams['skip_gain_values'],
        max_skip_gain=hyperparams['max_skip_gain'],
        lambda_alpha=hyperparams['lambda_alpha_values'],
        lambda_group=hyperparams['lambda_group_values'],
        lambda_tv=hyperparams['lambda_tv_values'],
        lambda_poly_decay=hyperparams['lambda_poly_decay_values'],
        lambda_skip_l2=hyperparams['lambda_skip_l2_values'],
        loss_type=hyperparams['loss_types'],
        huber_delta=hyperparams['huber_deltas']
    )
    
    # Charger les poids
    model.load_state_dict(result['model_state'])
    model.eval()
    
    return model


def visualize_kan_functions(model_or_result, hyperparams=None, input_dim=None, save_path=None, max_connections=10, resolution=1000):
    """
    Visualise les fonctions apprises dans chaque couche KAN.
    
    Args:
        model_or_result: Soit un modèle KANAutoencoder, soit un dictionnaire de résultats contenant 'model_state'
        hyperparams: Dictionnaire d'hyperparamètres (requis si model_or_result est un résultat)
        input_dim: Dimension d'entrée (requis si model_or_result est un résultat)
        save_path: Chemin pour sauvegarder les graphiques
        max_connections: Nombre maximum de connexions à afficher par couche
        resolution: Résolution pour l'évaluation des fonctions
    """
    # Déterminer si on a un modèle ou un résultat
    if hasattr(model_or_result, 'encoder_layers'):
        # C'est un modèle
        model = model_or_result
    elif isinstance(model_or_result, dict) and 'model_state' in model_or_result:
        # C'est un résultat d'entraînement
        if hyperparams is None or input_dim is None:
            raise ValueError("hyperparams et input_dim sont requis quand model_or_result est un dictionnaire de résultats")
        print("🔄 Reconstruction du modèle à partir des résultats...")
        model = reconstruct_model_from_results(model_or_result, hyperparams, input_dim)
    else:
        raise ValueError("model_or_result doit être un modèle KANAutoencoder ou un dictionnaire de résultats avec 'model_state'")
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from datetime import datetime
    
    model.eval()
    
    # Récupérer toutes les couches KAN
    kan_layers = []
    layer_names = []
    
    # Encoder layers
    for i, layer in enumerate(model.encoder_layers):
        kan_layers.append(layer)
        layer_names.append(f"Encoder Layer {i+1}")
    
    # Decoder layers  
    for i, layer in enumerate(model.decoder_layers):
        kan_layers.append(layer)
        layer_names.append(f"Decoder Layer {i+1}")
    
    if not kan_layers:
        print("❌ Aucune couche KAN trouvée dans le modèle")
        return
    
    print(f"🔍 Visualisation de {len(kan_layers)} couches KAN...")
    
    # Créer la grille d'évaluation
    xmin, xmax = -3.5, 3.5  # Domaine par défaut des bases
    x_eval = torch.linspace(xmin, xmax, resolution)
    
    # Nombre de sous-graphiques
    n_layers = len(kan_layers)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_layers == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Couleurs pour différencier les connexions
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for layer_idx, (layer, layer_name) in enumerate(zip(kan_layers, layer_names)):
        row = layer_idx // n_cols
        col = layer_idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        print(f"  📊 {layer_name}: {layer.in_features}→{layer.out_features}")
        
        with torch.no_grad():
            # Évaluer les bases
            if x_eval.dim() == 1:
                x_eval_batch = x_eval.unsqueeze(0)  # (1, resolution)
            else:
                x_eval_batch = x_eval
                
            try:
                # Évaluer la base pour obtenir Phi (1, resolution, M)
                Phi = layer.basis_1d(x_eval_batch)  # (1, resolution, M)
                if Phi.dim() == 2:  # Si on a (resolution, M)
                    Phi = Phi.unsqueeze(0)  # (1, resolution, M)
                
                # Sélectionner les connexions les plus importantes
                alpha_abs = layer.alpha.abs()  # (out_features, in_features)
                
                # Prendre les connexions avec les plus grands alpha
                flat_indices = torch.argsort(alpha_abs.flatten(), descending=True)
                top_connections = flat_indices[:min(max_connections, alpha_abs.numel())]
                
                functions_plotted = 0
                for flat_idx in top_connections:
                    out_idx = flat_idx // layer.in_features
                    in_idx = flat_idx % layer.in_features
                    
                    alpha_val = layer.alpha[out_idx, in_idx].item()
                    c_coeffs = layer.c[out_idx, in_idx, :]  # (M,)
                    
                    # Calculer la fonction: alpha * (Phi @ c)
                    # Phi[0, :, :] est (resolution, M), c_coeffs est (M,)
                    func_values = alpha_val * torch.matmul(Phi[0], c_coeffs)  # (resolution,)
                    
                    # Convertir en numpy pour matplotlib
                    x_np = x_eval.cpu().numpy()
                    y_np = func_values.cpu().numpy()
                    
                    # Couleur pour cette connexion
                    color = colors[functions_plotted % len(colors)]
                    
                    # Plot avec transparence pour éviter la surcharge visuelle
                    alpha_plot = 0.8 if functions_plotted < 5 else 0.4
                    ax.plot(x_np, y_np, 
                           color=color, 
                           alpha=alpha_plot,
                           linewidth=2 if functions_plotted < 3 else 1,
                           label=f'({in_idx}→{out_idx}) α={alpha_val:.3f}')
                    
                    functions_plotted += 1
                    
                    # Limiter le nombre de légendes pour la lisibilité
                    if functions_plotted >= max_connections:
                        break
                
                # Configuration du graphique
                ax.set_title(f'{layer_name}\n({layer.basis_type}, {layer.in_features}→{layer.out_features})')
                ax.set_xlabel('Input')
                ax.set_ylabel('Function Output')
                ax.grid(True, alpha=0.3)
                
                # Légende seulement pour les premières connexions
                if functions_plotted <= 5:
                    ax.legend(fontsize=8)
                
                # Statistiques dans le titre
                max_alpha = alpha_abs.max().item()
                sparsity = (alpha_abs < 1e-6).float().mean().item()
                ax.text(0.02, 0.98, f'max(α)={max_alpha:.3f}\nsparsity={sparsity:.1%}', 
                       transform=ax.transAxes, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=8)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Erreur: {str(e)}', transform=ax.transAxes, 
                       ha='center', va='center')
                print(f"    ❌ Erreur pour {layer_name}: {str(e)}")
    
    # Masquer les axes vides
    for layer_idx in range(n_layers, n_rows * n_cols):
        row = layer_idx // n_cols
        col = layer_idx % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        elif n_cols > 1:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    
    # Sauvegarder si demandé
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualisation sauvegardée: {save_path}")
    else:
        # Sauvegarde automatique
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        figures_dir = Path('figures')
        figures_dir.mkdir(exist_ok=True)
        auto_save_path = figures_dir / f'kan_functions_{timestamp}.png'
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualisation sauvegardée: {auto_save_path}")
    
    plt.show()
    return fig


def visualize_kan_layers_detailed(model_or_result, hyperparams=None, input_dim=None, layer_indices=None, save_path=None):
    """
    Visualisation détaillée des couches KAN spécifiques avec matrices de coefficients.
    
    Args:
        model_or_result: Soit un modèle KANAutoencoder, soit un dictionnaire de résultats contenant 'model_state'
        hyperparams: Dictionnaire d'hyperparamètres (requis si model_or_result est un résultat)
        input_dim: Dimension d'entrée (requis si model_or_result est un résultat)
        layer_indices: Liste des indices de couches à visualiser (None = toutes)
        save_path: Chemin de sauvegarde
    """
    # Déterminer si on a un modèle ou un résultat
    if hasattr(model_or_result, 'encoder_layers'):
        # C'est un modèle
        model = model_or_result
    elif isinstance(model_or_result, dict) and 'model_state' in model_or_result:
        # C'est un résultat d'entraînement
        if hyperparams is None or input_dim is None:
            raise ValueError("hyperparams et input_dim sont requis quand model_or_result est un dictionnaire de résultats")
        print("🔄 Reconstruction du modèle à partir des résultats...")
        model = reconstruct_model_from_results(model_or_result, hyperparams, input_dim)
    else:
        raise ValueError("model_or_result doit être un modèle KANAutoencoder ou un dictionnaire de résultats avec 'model_state'")
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from datetime import datetime
    
    model.eval()
    
    # Récupérer les couches
    all_layers = list(model.encoder_layers) + list(model.decoder_layers)
    all_names = [f"Enc{i+1}" for i in range(len(model.encoder_layers))] + \
                [f"Dec{i+1}" for i in range(len(model.decoder_layers))]
    
    if layer_indices is None:
        layer_indices = list(range(len(all_layers)))
    
    selected_layers = [all_layers[i] for i in layer_indices]
    selected_names = [all_names[i] for i in layer_indices]
    
    print(f"🔬 Visualisation détaillée de {len(selected_layers)} couches...")
    
    for layer_idx, (layer, name) in enumerate(zip(selected_layers, selected_names)):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Analyse Détaillée - {name} ({layer.in_features}→{layer.out_features})', 
                     fontsize=16, fontweight='bold')
        
        with torch.no_grad():
            # 1. Matrice Alpha (sparsité des connexions)
            alpha_matrix = layer.alpha.cpu().numpy()
            im1 = ax1.imshow(alpha_matrix, cmap='RdBu', aspect='auto')
            ax1.set_title('Matrice Alpha (Coefficients de connexion)')
            ax1.set_xlabel('Input Features')
            ax1.set_ylabel('Output Features')
            plt.colorbar(im1, ax=ax1, label='Alpha Value')
            
            # 2. Norme des coefficients C par connexion
            c_norms = torch.sqrt((layer.c ** 2).sum(dim=-1)).cpu().numpy()
            im2 = ax2.imshow(c_norms, cmap='viridis', aspect='auto')
            ax2.set_title('Norme des Coefficients C par Connexion')
            ax2.set_xlabel('Input Features')
            ax2.set_ylabel('Output Features')
            plt.colorbar(im2, ax=ax2, label='||C|| Norm')
            
            # 3. Distribution des valeurs Alpha
            alpha_flat = alpha_matrix.flatten()
            ax3.hist(alpha_flat, bins=50, alpha=0.7, edgecolor='black')
            ax3.set_title('Distribution des Valeurs Alpha')
            ax3.set_xlabel('Alpha Value')
            ax3.set_ylabel('Frequency')
            ax3.axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero')
            ax3.legend()
            
            # 4. Statistiques de sparsité et activité
            sparsity = (np.abs(alpha_flat) < 1e-6).mean()
            active_connections = np.sum(np.abs(alpha_flat) > 1e-6)
            total_connections = alpha_flat.size
            
            stats_text = f"""Statistiques de la couche:
            
• Type: {layer.basis_type}
• Connexions totales: {total_connections}
• Connexions actives: {active_connections}
• Sparsité: {sparsity:.1%}
• Alpha max: {np.abs(alpha_flat).max():.4f}
• Alpha moyen: {np.abs(alpha_flat).mean():.4f}
• Biais max: {layer.bias.abs().max().item():.4f}
"""
            
            if layer.use_skip:
                skip_norm = layer.skip.weight.norm().item()
                skip_gain = layer.skip_gain.item()
                stats_text += f"• Skip norm: {skip_norm:.4f}\n• Skip gain: {skip_gain:.4f}"
            
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Statistiques de la Couche')
        
        plt.tight_layout()
        
        # Sauvegarder
        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(exist_ok=True, parents=True)
            layer_save_path = save_dir / f"{Path(save_path).stem}_layer_{name}.png"
            plt.savefig(layer_save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Couche {name} sauvegardée: {layer_save_path}")
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            figures_dir = Path('figures')
            figures_dir.mkdir(exist_ok=True)
            layer_save_path = figures_dir / f'kan_layer_detail_{name}_{timestamp}.png'
            plt.savefig(layer_save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Couche {name} sauvegardée: {layer_save_path}")
        
        plt.show()


def change_hyperparam(hyperparameters_grid, hp_to_change, value):
    if hp_to_change in hyperparameters_grid.keys():
        hyperparameters_grid[hp_to_change] = value
        return hyperparameters_grid
    else:
        raise ValueError(f"Hyperparameter {hp_to_change} not found in grid")
    
    
    
    
def create_hyperparams_grid(input_dim):
    hyperparameters = {
        'input_dim': input_dim,
        'hidden_dims_choices': [[64, 32], [32], [32, 16]],
        'latent_dims': [5, 8, 12, 20],
        'basis_types': 'spline',
        'M_values': [8, 16, 32],
        'poly_degrees': [3, 5, 7],
        'use_silu_choices': True,
        'dropout_rates': 0.1,
        
        'use_global_skip': True,
        'use_skip_choices': False,
        'skip_init_choices': ['zeros', 'identity', 'xavier'],
        'skip_gain_values': [0.1, 0.5, 1.0],
        'max_skip_gain': [0.3, 0.5, 0.7, 0.9],
        
        'lambda_alpha_values': [1e-3, 1e-4, 1e-5],
        'lambda_group_values': [1e-4, 1e-5, 1e-6],
        'lambda_tv_values': [5e-4, 1e-4, 1e-5],
        'lambda_poly_decay_values': [1e-4, 1e-5, 1e-6],
        'lambda_skip_l2_values': [1e-5, 1e-6, 1e-7],
        
        'loss_types': 'huber',
        'huber_deltas': [1.0, 0.5],
        
        'batch_sizes': [32, 64, 128],
        'learning_rates': [1e-3, 1e-4, 1e-5],
        'weight_decays': [1e-5, 1e-6, 1e-7],
        'lambda_reg_values': [1e-2, 1e-3, 1e-4]
    }
    # Sample one value randomly from each hyperparameter list
    sampled_params = {}
    for param_name, param_values in hyperparameters.items():
        if isinstance(param_values, (list, tuple)):
            if isinstance(param_values[0], list):
                # Handle nested lists like hidden_dims_choices
                sampled_params[param_name] = param_values[np.random.randint(len(param_values))]
            else:
                # Handle flat lists
                sampled_params[param_name] = np.random.choice(param_values)
        else:
            sampled_params[param_name] = param_values
            
    return sampled_params