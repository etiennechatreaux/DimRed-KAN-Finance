# KAN Autoencoder – Dimensionality Reduction in Finance
Author : Etienne CHATREAUX  
Master thesis supervisor : Fouad BEN ABDELAZIZ  

This project explores **Kolmogorov–Arnold Networks (KANs)** as nonlinear autoencoders for dimensionality reduction. The project is part of my Master Thesis, motivated by the need for a recent, interpretable, and finance-oriented approach to factor modeling.  


## Overview
- **Step 1 – MNIST Benchmark:** Train KAN autoencoders on MNIST to validate reconstruction ability on images.
- **Step 2 – Finance Application:** Apply the KAN autoencoder to **S&P 500 daily returns (2010–2025)** for factor extraction and comparison with classical methods (PCA, MLP Autoencoders).

## Goals
- Evaluate reconstruction error and factor interpretability.
- Explore interpretability of KAN spline functions in finance.  
- Compare KANs with traditional factor models (linear and nonlinear)

## Tech Stack
- Python (PyTorch, Numpy)
- CUDA (GPU acceleration)
- Custom implementation of KAN (B-splines, polynomial)

## Structure
```
data/
results/
figures/
papers/
src/ 
├── models/ # Model implementations
│ ├── ae_kan.py # KAN Autoencoder
│ ├── ae_mlp.py # MLP Autoencoder (baseline)
│ ├── bases_1d.py # 1D basis functions (B-splines and polynomial functions)
│ └── kan_layers.py
└── utils/
.gitignore
01_data_preprocessing.ipynb
02_data_exploration.ipynb
03_mnist_test.ipynb 
README.md
requirements.txt
```

