# KAN Autoencoder – Dimensionality Reduction in Finance

This project explores **Kolmogorov–Arnold Networks (KANs)** as nonlinear autoencoders for dimensionality reduction. I have chosen this subject for my Master Thesis because I wanted something recent, interpretable and applicable to finance. 

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
