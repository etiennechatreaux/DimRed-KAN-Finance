# KAN-memoire : Étude comparative des Kolmogorov-Arnold Networks

## 📋 Description du projet

Ce projet de mémoire explore les **Kolmogorov-Arnold Networks (KAN)** et les compare aux réseaux de neurones traditionnels (MLP) dans le contexte de la prédiction de séries temporelles financières.

### 🎯 Objectifs

- Implémenter et analyser l'architecture des KAN
- Comparer les performances des KAN vs MLP sur des données financières
- Évaluer l'interprétabilité et l'efficacité des KAN
- Fournir des insights pour l'utilisation des KAN en finance

## 🏗️ Architecture du projet

```
KAN-memoire/
├── data/                  # Données du projet
│   ├── raw/              # Données brutes (crypto, actions, etc.)
│   └── processed/        # Données nettoyées et transformées
├── notebooks/            # Notebooks d'analyse
│   ├── 01_exploration.ipynb         # Analyse exploratoire
│   ├── 02_KAN_experiments.ipynb     # Expérimentations KAN
│   └── 03_comparaison_MLP.ipynb     # Comparaison KAN vs MLP
├── src/                  # Code source
│   ├── models/           # Implémentations des modèles
│   │   ├── kan.py        # Architecture KAN
│   │   └── mlp.py        # Modèle MLP de référence
│   ├── training/         # Scripts d'entraînement
│   │   └── train.py      # Pipeline d'entraînement
│   ├── evaluation/       # Métriques et évaluation
│   │   └── evaluate.py   # Fonctions d'évaluation
│   └── utils/            # Utilitaires
│       └── preprocessing.py # Prétraitement des données
├── outputs/              # Résultats et visualisations
│   ├── figures/          # Graphiques pour le mémoire
│   └── logs/             # Logs d'entraînement
├── models_saved/         # Modèles entraînés
├── requirements.txt      # Dépendances Python
├── README.md             # Ce fichier
└── main.py               # Point d'entrée principal
```

## 🚀 Installation et utilisation

### Prérequis

- Python 3.8+
- pip ou conda

### Installation

1. **Cloner le repository**
   ```bash
   git clone <url-du-repo>
   cd KAN-memoire
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate     # Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

### Utilisation

#### Entraînement des modèles

```bash
# Entraîner les deux modèles (KAN et MLP)
python main.py --model both --epochs 100

# Entraîner seulement le KAN
python main.py --model kan --epochs 150 --lr 0.0001

# Entraîner seulement le MLP
python main.py --model mlp --batch_size 64
```

#### Notebooks d'analyse

1. **Exploration des données** : `notebooks/01_exploration.ipynb`
2. **Expérimentations KAN** : `notebooks/02_KAN_experiments.ipynb`
3. **Comparaison des modèles** : `notebooks/03_comparaison_MLP.ipynb`

## 📊 Fonctionnalités principales

### Modèles implémentés

- **KAN (Kolmogorov-Arnold Networks)** : Architecture basée sur le théorème de Kolmogorov-Arnold
- **MLP (Multi-Layer Perceptron)** : Réseau de neurones traditionnel pour comparaison

### Métriques d'évaluation

- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Métriques financières (Sharpe ratio, drawdown, etc.)

### Visualisations

- Courbes d'apprentissage
- Comparaison des prédictions
- Analyse des erreurs
- Métriques de performance

## 🔬 Expérimentations

### Datasets utilisés

- Données crypto (Bitcoin, Ethereum)
- Données d'actions (S&P 500, NASDAQ)
- Indicateurs techniques (RSI, MACD, Bollinger Bands)

### Configurations testées

- Différentes architectures de KAN
- Variantes de MLP
- Hyperparamètres optimisés
- Stratégies de régularisation

## 📈 Résultats attendus

- Comparaison quantitative des performances
- Analyse de l'interprétabilité des modèles
- Recommandations d'utilisation
- Insights sur les avantages/inconvénients des KAN

## 🤝 Contribution

Ce projet est développé dans le cadre d'un mémoire universitaire. Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue.

## 📚 Références

- [Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
- [Théorème de Kolmogorov-Arnold](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem)
- [Deep Learning for Time Series](https://otexts.com/fpp3/)

## 📄 Licence

Ce projet est destiné à un usage académique dans le cadre d'un mémoire universitaire.

---

**Auteur** : [Votre nom]  
**Institution** : [Votre université]  
**Année** : 2024 