
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import json
from pathlib import Path
import itertools
from tqdm.auto import tqdm
import gc

from src.models.gated_kan_ae import GatedKANAutoencoder
from src.utils.yield_curve_data import load_preprocessed_yield_curve

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')

# Configuration device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

## 1. Configuration de la grille d'hyperparamètres

# Grille d'hyperparamètres pour test nocturne - NOUVELLE CONFIGURATION
HYPERPARAMS_GRID = {
    # Hyperparamètres à tester (pour le constructeur)
    'hidden_dims': [[5, 4, 3], [6, 5, 4], [6, 4, 3]],  # 3 configurations de dimensions cachées
    
    # Hyperparamètres fixes du constructeur (basés sur les meilleures pratiques)
    'input_dim': None, # will be set later
    'k': 3, 
    'basis_type': 'spline',
    'M': 16,
    'poly_degree': 5,
    'xmin': -3.5,
    'xmax': 3.5,
    'dropout_p': 0,
    'use_silu': True,
    'gate_init': 0.5,
    'skip_rank': None,  # Sera défini selon input_dim
    'loss_type': 'huber',
    'huber_delta': 1.0,
    'lambda_alpha': 1e-4,
    'lambda_group': 1e-5,
    'lambda_tv': 1e-4,
    'lambda_poly_decay': 0.0,
    'lambda_gate_reg': 1e-4, 
    'lambda_orthogonal': 0.1
}

# Paramètres d'entraînement à tester
TRAINING_PARAMS_GRID = {
    'learning_rate': [1e-6, 1e-4, 1e-2],  # 3 valeurs
    'weight_decay': [1e-8, 1e-6, 1e-3, 1e-2]  # 4 valeurs
}

# Paramètres d'entraînement fixes
FIXED_TRAINING_PARAMS = {
    'epochs': 150,
    'batch_size': 128,
    'patience': 20,
    'lambda_reg': 1.0,
    'verbose': False  # Pas de verbose pour la grille
}

DATASETS = {
    'yield_cross_section': {
        'normalization': 'cross_section',
        'description': 'Yield Curve (Cross-section Z-score)'
    },
    'yield_column_wise': {
        'normalization': 'column_wise', 
        'description': 'Yield Curve (Column-wise Z-score)'
    }
}

print(f"🔧 Hyperparamètres à tester:")
print(f"   hidden_dims: {HYPERPARAMS_GRID['hidden_dims']}")
print(f"   learning_rate: {TRAINING_PARAMS_GRID['learning_rate']}")
print(f"   weight_decay: {TRAINING_PARAMS_GRID['weight_decay']}")
print(f"   Total combinations: {len(HYPERPARAMS_GRID['hidden_dims']) * len(TRAINING_PARAMS_GRID['learning_rate']) * len(TRAINING_PARAMS_GRID['weight_decay'])}")
print(f"   Datasets: {list(DATASETS.keys())}")
print(f"   Total experiments: {len(HYPERPARAMS_GRID['hidden_dims']) * len(TRAINING_PARAMS_GRID['learning_rate']) * len(TRAINING_PARAMS_GRID['weight_decay']) * len(DATASETS)}")

## 2. Fonctions utilitaires

def create_results_directory():
    """Crée le répertoire de résultats avec timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/gated_kan_hyperparams_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer un sous-dossier pour les graphiques de courbes
    plots_dir = results_dir / "training_curves"
    plots_dir.mkdir(exist_ok=True)
    
    # Sauvegarder la configuration
    config = {
        'hyperparams_grid': HYPERPARAMS_GRID,
        'training_params_grid': TRAINING_PARAMS_GRID,
        'fixed_training_params': FIXED_TRAINING_PARAMS,
        'datasets': DATASETS,
        'timestamp': timestamp,
        'device': str(device)
    }
    
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return results_dir

def load_dataset(normalization_type):
    """Charge un dataset de yield curve"""
    df = load_preprocessed_yield_curve(
        start="1990-01-01",
        normalization=normalization_type
    )
    
    # Conversion en tenseurs
    X = torch.tensor(df.values, dtype=torch.float32)
    dates = df.index
    
    return X, dates, df.columns

def save_training_curves(history, model_config, training_config, dataset_name, results_dir):
    """Sauvegarde les courbes d'entraînement avec hyperparamètres"""
    if not history or 'train_loss' not in history:
        return None
    
    # Créer une figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Curves - {dataset_name}\n'
                f'hidden_dims: {model_config["hidden_dims"]}, '
                f'lr: {training_config["learning_rate"]:.0e}, '
                f'wd: {training_config["weight_decay"]:.0e}', 
                fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Gate evolution
    if 'gate_values' in history:
        ax2 = axes[0, 1]
        ax2.plot(epochs, history['gate_values'], label='Gate Value', color='green', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Gate Value')
        ax2.set_title('Gate Evolution (KAN Contribution)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Regularization terms
    if 'orthogonality_violation' in history:
        ax3 = axes[1, 0]
        ax3.plot(epochs, history['orthogonality_violation'], label='Orthogonality Violation', color='orange', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Violation')
        ax3.set_title('Orthogonality Violation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Learning rate schedule (si disponible)
    if 'learning_rate' in history:
        ax4 = axes[1, 1]
        ax4.plot(epochs, history['learning_rate'], label='Learning Rate', color='purple', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # Si pas de LR schedule, afficher les paramètres
        ax4 = axes[1, 1]
        ax4.text(0.1, 0.8, f'Learning Rate: {training_config["learning_rate"]:.0e}', 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.7, f'Weight Decay: {training_config["weight_decay"]:.0e}', 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f'Batch Size: {training_config["batch_size"]}', 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.5, f'Epochs: {training_config["epochs"]}', 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.4, f'Hidden Dims: {model_config["hidden_dims"]}', 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.3, f'Gate Init: {model_config["gate_init"]}', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Hyperparameters')
        ax4.set_xticks([])
        ax4.set_yticks([])
    
    plt.tight_layout()
    
    # Nom de fichier avec hyperparamètres
    hidden_str = "_".join(map(str, model_config['hidden_dims']))
    filename = (f"curves_{dataset_name}_"
               f"h{hidden_str}_"
               f"lr{training_config['learning_rate']:.0e}_"
               f"wd{training_config['weight_decay']:.0e}.png")
    
    plots_dir = results_dir / "training_curves"
    filepath = plots_dir / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Fermer pour économiser la mémoire
    
    return filepath

def train_gated_kan_config(X, model_config, training_config, dataset_name, results_dir, cv_splits=3):
    """Entraîne un Gated KAN AE avec une configuration donnée"""
    results = {
        'dataset': dataset_name,
        'model_config': model_config.copy(),
        'training_config': training_config.copy(),
        'cv_scores': [],
        'mean_score': 0,
        'std_score': 0,
        'training_times': [],
        'final_gate_values': [],
        'orthogonality_violations': [],
        'success': False,
        'training_curve_files': []
    }
    
    try:
        # Configuration du modèle (pour le constructeur)
        model_params = model_config.copy()
        model_params['input_dim'] = X.shape[1]
        model_params['skip_rank'] = min(X.shape[1] // 4, 32)  # Heuristique
        
        # Cross-validation temporelle
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            
            # Création du modèle
            model = GatedKANAutoencoder(**model_params)
            model.to(device)
            
            # Entraînement
            start_time = datetime.now()
            history = model.fit(
                X_train=X_train,
                X_val=X_val,
                **training_config  # Utilise les paramètres d'entraînement
            )
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Sauvegarder les courbes d'entraînement pour ce fold
            curve_file = save_training_curves(
                history, model_config, training_config, 
                f"{dataset_name}_fold{fold+1}", results_dir
            )
            if curve_file:
                results['training_curve_files'].append(str(curve_file))
            
            # Évaluation
            model.eval()
            with torch.no_grad():
                X_val_hat, _, gate, _, _ = model(X_val.to(device))
                val_loss = nn.functional.mse_loss(X_val_hat, X_val.to(device)).item()
                
                # Métriques spécifiques au Gated KAN
                gate_info = model.get_gate_info()
                final_gate = gate_info['kan_contribution']
                orth_violation = history['orthogonality_violation'][-1] if 'orthogonality_violation' in history else 0.0
            
            results['cv_scores'].append(val_loss)
            results['training_times'].append(training_time)
            results['final_gate_values'].append(final_gate)
            results['orthogonality_violations'].append(orth_violation)
            
            # Nettoyage mémoire
            del model, history
            torch.cuda.empty_cache()
            gc.collect()
        
        # Statistiques finales
        results['mean_score'] = np.mean(results['cv_scores'])
        results['std_score'] = np.std(results['cv_scores'])
        results['success'] = True
        
    except Exception as e:
        print(f"❌ Erreur pour {dataset_name}: {str(e)}")
        results['error'] = str(e)
    
    return results

def save_results(results, results_dir):
    """Sauvegarde les résultats"""
    results_file = results_dir / f"results_{datetime.now().strftime('%H%M%S')}.json"
    
    # Conversion pour JSON
    json_results = []
    for result in results:
        json_result = result.copy()
        # Conversion des listes numpy
        for key in ['cv_scores', 'training_times', 'final_gate_values', 'orthogonality_violations']:
            if key in json_result:
                json_result[key] = [float(x) for x in json_result[key]]
        json_results.append(json_result)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    return results_file

## 3. Chargement des datasets

# Chargement des datasets
datasets = {}
for name, config in DATASETS.items():
    print(f"📂 Loading {name}...")
    X, dates, features = load_dataset(config['normalization'])
    datasets[name] = {
        'X': X,
        'dates': dates,
        'features': features,
        'description': config['description']
    }
    print(f"   ✅ {X.shape[0]} samples, {X.shape[1]} features")

print(f"\n🎯 Total datasets loaded: {len(datasets)}")

## 4. Grille de recherche principale

# Création du répertoire de résultats
results_dir = create_results_directory()
print(f"📁 Results directory: {results_dir}")
print(f"📊 Training curves will be saved in: {results_dir}/training_curves/")

# Génération de toutes les combinaisons
all_configs = list(itertools.product(
    HYPERPARAMS_GRID['hidden_dims'],
    TRAINING_PARAMS_GRID['learning_rate'],
    TRAINING_PARAMS_GRID['weight_decay']
))

print(f"🔍 Total configurations to test: {len(all_configs)}")
print(f"⏱️  Estimated time: {len(all_configs) * len(datasets) * 2} minutes (2min per config)")
print(f"🌙 Perfect for overnight training!")

all_results = []
start_time = datetime.now()

print(f"🚀 Starting hyperparameter grid search at {start_time.strftime('%H:%M:%S')}")
print(f"📊 Testing {len(all_configs)} configurations on {len(datasets)} datasets")
print("="*80)

for config_idx, (hidden_dims, learning_rate, weight_decay) in enumerate(tqdm(all_configs, desc="Configurations")):
    print(f"\n🔧 Config {config_idx+1}/{len(all_configs)}: hidden_dims={hidden_dims}, lr={learning_rate:.0e}, wd={weight_decay:.0e}")
    
    # Configuration du modèle pour cette itération
    model_config = HYPERPARAMS_GRID.copy()
    model_config['hidden_dims'] = hidden_dims
    
    # Configuration d'entraînement pour cette itération
    training_config = FIXED_TRAINING_PARAMS.copy()
    training_config['learning_rate'] = learning_rate
    training_config['weight_decay'] = weight_decay
    
    # Test sur chaque dataset
    for dataset_name, dataset_info in datasets.items():
        print(f"   📊 Testing on {dataset_name}...")
        
        result = train_gated_kan_config(
            X=dataset_info['X'],
            model_config=model_config,
            training_config=training_config,
            dataset_name=dataset_name,
            results_dir=results_dir
        )
        
        all_results.append(result)
        
        if result['success']:
            print(f"      ✅ Score: {result['mean_score']:.6f} ± {result['std_score']:.6f}")
            print(f"      🎛️  Gate: {np.mean(result['final_gate_values']):.3f}")
            print(f"      🔀 Orth_viol: {np.mean(result['orthogonality_violations']):.4f}")
            print(f"      📈 Curves saved: {len(result['training_curve_files'])} files")
        else:
            print(f"      ❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Sauvegarde intermédiaire tous les 5 configs
    if (config_idx + 1) % 5 == 0:
        save_results(all_results, results_dir)
        print(f"   💾 Intermediate save at config {config_idx+1}")
    
    # Nettoyage mémoire
    torch.cuda.empty_cache()
    gc.collect()

end_time = datetime.now()
total_time = (end_time - start_time).total_seconds() / 3600  # en heures

print(f"\n🎉 Grid search completed!")
print(f"⏱️  Total time: {total_time:.2f} hours")
print(f"📊 Total experiments: {len(all_results)}")
print(f"✅ Successful: {sum(1 for r in all_results if r['success'])}")
print(f"❌ Failed: {sum(1 for r in all_results if not r['success'])}")

## 5. Sauvegarde finale et analyse

# Sauvegarde finale
final_results_file = save_results(all_results, results_dir)
print(f"💾 Final results saved to: {final_results_file}")

# Compter les courbes sauvegardées
total_curves = sum(len(r.get('training_curve_files', [])) for r in all_results)
print(f"📈 Total training curves saved: {total_curves}")

# Analyse rapide des meilleurs résultats
successful_results = [r for r in all_results if r['success']]

if successful_results:
    # Meilleurs résultats par dataset
    print(f"\n🏆 BEST RESULTS BY DATASET:")
    print("="*60)
    
    for dataset_name in datasets.keys():
        dataset_results = [r for r in successful_results if r['dataset'] == dataset_name]
        if dataset_results:
            best_result = min(dataset_results, key=lambda x: x['mean_score'])
            print(f"\n📊 {dataset_name}:")
            print(f"   🎯 Best score: {best_result['mean_score']:.6f} ± {best_result['std_score']:.6f}")
            print(f"   🔧 hidden_dims: {best_result['model_config']['hidden_dims']}")
            print(f"   🔧 learning_rate: {best_result['training_config']['learning_rate']:.0e}")
            print(f"   🔧 weight_decay: {best_result['training_config']['weight_decay']:.0e}")
            print(f"   🎛️  Avg gate: {np.mean(best_result['final_gate_values']):.3f}")
            print(f"   🔀 Avg orth_viol: {np.mean(best_result['orthogonality_violations']):.4f}")
        else:
            print(f"\n❌ {dataset_name}: No successful runs")
    
    # Top 5 configurations globales
    print(f"\n🌟 TOP 5 GLOBAL CONFIGURATIONS:")
    print("="*60)
    
    top_5 = sorted(successful_results, key=lambda x: x['mean_score'])[:5]
    for i, result in enumerate(top_5, 1):
        print(f"\n{i}. {result['dataset']}:")
        print(f"   Score: {result['mean_score']:.6f}")
        print(f"   hidden_dims: {result['model_config']['hidden_dims']}")
        print(f"   learning_rate: {result['training_config']['learning_rate']:.0e}")
        print(f"   weight_decay: {result['training_config']['weight_decay']:.0e}")
        print(f"   Gate: {np.mean(result['final_gate_values']):.3f}")
        print(f"   Orth_viol: {np.mean(result['orthogonality_violations']):.4f}")

else:
    print("❌ No successful experiments!")

print(f"\n📁 All results saved in: {results_dir}")
print(f"📈 Training curves saved in: {results_dir}/training_curves/")
print(f"Ready for morning analysis!")

## 6. Visualisation rapide des résultats

if successful_results:
    # Préparation des données pour visualisation
    viz_data = []
    for result in successful_results:
        viz_data.append({
            'dataset': result['dataset'],
            'hidden_dims': str(result['model_config']['hidden_dims']),
            'learning_rate': result['training_config']['learning_rate'],
            'weight_decay': result['training_config']['weight_decay'],
            'score': result['mean_score'],
            'gate_value': np.mean(result['final_gate_values']),
            'orth_violation': np.mean(result['orthogonality_violations'])
        })
    
    df_viz = pd.DataFrame(viz_data)
    
    # Créer les graphiques
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Score vs Learning Rate
    ax = axes[0, 0]
    for dataset_name in datasets.keys():
        dataset_df = df_viz[df_viz['dataset'] == dataset_name]
        if not dataset_df.empty:
            ax.scatter(dataset_df['learning_rate'], dataset_df['score'], 
                      label=dataset_name, alpha=0.7, s=60)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Validation Loss')
    ax.set_xscale('log')
    ax.set_title('Score vs Learning Rate')
    ax.legend()
    ax.grid(True)
    
    # 2. Score vs Weight Decay
    ax = axes[0, 1]
    for dataset_name in datasets.keys():
        dataset_df = df_viz[df_viz['dataset'] == dataset_name]
        if not dataset_df.empty:
            ax.scatter(dataset_df['weight_decay'], dataset_df['score'], 
                      label=dataset_name, alpha=0.7, s=60)
    ax.set_xlabel('Weight Decay')
    ax.set_ylabel('Validation Loss')
    ax.set_xscale('log')
    ax.set_title('Score vs Weight Decay')
    ax.legend()
    ax.grid(True)
    
    # 3. Score par Hidden Dimensions
    ax = axes[0, 2]
    for dataset_name in datasets.keys():
        dataset_df = df_viz[df_viz['dataset'] == dataset_name]
        if not dataset_df.empty:
            hidden_dims_scores = dataset_df.groupby('hidden_dims')['score'].mean()
            ax.bar(range(len(hidden_dims_scores)), hidden_dims_scores.values, 
                  label=dataset_name, alpha=0.7)
    ax.set_xlabel('Hidden Dimensions')
    ax.set_ylabel('Average Validation Loss')
    ax.set_title('Score by Hidden Dimensions')
    ax.set_xticks(range(len(hidden_dims_scores)))
    ax.set_xticklabels(hidden_dims_scores.index, rotation=45)
    ax.legend()
    ax.grid(True)
    
    # 4. Gate Value vs Learning Rate
    ax = axes[1, 0]
    for dataset_name in datasets.keys():
        dataset_df = df_viz[df_viz['dataset'] == dataset_name]
        if not dataset_df.empty:
            ax.scatter(dataset_df['learning_rate'], dataset_df['gate_value'], 
                      label=dataset_name, alpha=0.7, s=60)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Gate Value (KAN Contribution)')
    ax.set_xscale('log')
    ax.set_title('Gate Value vs Learning Rate')
    ax.legend()
    ax.grid(True)
    
    # 5. Gate Value vs Weight Decay
    ax = axes[1, 1]
    for dataset_name in datasets.keys():
        dataset_df = df_viz[df_viz['dataset'] == dataset_name]
        if not dataset_df.empty:
            ax.scatter(dataset_df['weight_decay'], dataset_df['gate_value'], 
                      label=dataset_name, alpha=0.7, s=60)
    ax.set_xlabel('Weight Decay')
    ax.set_ylabel('Gate Value (KAN Contribution)')
    ax.set_xscale('log')
    ax.set_title('Gate Value vs Weight Decay')
    ax.legend()
    ax.grid(True)
    
    # 6. Distribution des scores
    ax = axes[1, 2]
    for dataset_name in datasets.keys():
        dataset_df = df_viz[df_viz['dataset'] == dataset_name]
        if not dataset_df.empty:
            ax.hist(dataset_df['score'], alpha=0.7, label=dataset_name, bins=15)
    ax.set_xlabel('Validation Loss')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Validation Losses')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    # Sauvegarde
    viz_file = results_dir / 'hyperparams_analysis.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"📊 Analysis plots saved to: {viz_file}")
    
    plt.show()
    
    # Résumé statistique
    print("\n📈 STATISTICAL SUMMARY:")
    print("="*50)
    print(f"📊 Total experiments: {len(all_results)}")
    print(f"✅ Successful: {len(successful_results)} ({len(successful_results)/len(all_results)*100:.1f}%)")
    print(f"🎯 Best score: {min(r['mean_score'] for r in successful_results):.6f}")
    print(f"📈 Worst score: {max(r['mean_score'] for r in successful_results):.6f}")
    print(f"🎛️  Gate range: [{min(np.mean(r['final_gate_values']) for r in successful_results):.3f}, {max(np.mean(r['final_gate_values']) for r in successful_results):.3f}]")
    print(f"🔀 Orth violation range: [{min(np.mean(r['orthogonality_violations']) for r in successful_results):.4f}, {max(np.mean(r['orthogonality_violations']) for r in successful_results):.4f}]")
    print(f"📈 Training curves saved: {total_curves} files in {results_dir}/training_curves/")