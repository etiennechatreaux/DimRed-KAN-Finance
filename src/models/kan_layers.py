# -*- coding: utf-8 -*-
"""
Couches KAN basées sur des fonctions 1D par arête :
  y_j = b_j + sum_i alpha_ij * ( Phi(x_i) @ c_ij ),
où Phi(x_i) ∈ R^{M_b} est la sortie d'une base 1D (splines ou polynômes).

Skip linéaire (optionnel) :
  y = y_KAN + gain * (W_skip x)     # W_skip: Linear(in→out)

Régularisations incluses :
  - L1 sur alpha (sparsité des connexions)
  - Group-lasso sur c par arête (||c_{ij}||_2)
  - TV (variation totale) sur c (axe des noeuds) pour les splines
  - (optionnel) décroissance polynomiale (pénalise hauts degrés)
  - (optionnel) L2 sur skip (faible)

API :
  - KANLayer(..., basis_type="spline"|"poly", M=16, poly_degree=5, ...,
             use_skip=True|False, skip_init="zeros"|"xavier"|"identity", skip_gain=1.0)
  - .regularization() -> Tensor
  - .set_regularization(...)
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from .bases_1d import Basis1D, HatSpline1D, PolyBasis1D


class KANLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        basis_type: str = "spline",   # "spline" | "poly"
        M: int = 16,                  # nb fonctions pour spline
        poly_degree: int = 5,         # nb fonctions = degree+1 pour poly
        xmin: float = -3.5,
        xmax: float = 3.5,
        alpha_init: float = 0.01,
        # régularisations KAN
        lambda_alpha: float = 1e-4,
        lambda_group: float = 1e-5,
        lambda_tv: float = 1e-4,      # utile pour spline
        lambda_poly_decay: float = 0.0, # utile pour poly
        # --- options skip linéaire ---
        use_skip: bool = True,
        skip_init: str = "zeros",     # "zeros" | "xavier" | "identity"(si in==out)
        skip_gain: float = 1.0,
        lambda_skip_l2: float = 0.0   # petite pénalité L2 optionnelle sur W_skip
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.basis_type = basis_type

        # Base 1D et dimension de base
        if basis_type == "spline":
            self.basis_1d: Basis1D = HatSpline1D(M=M, xmin=xmin, xmax=xmax)
            Bdim = M
        elif basis_type == "poly":
            self.basis_1d = PolyBasis1D(degree=poly_degree, clip=max(abs(xmin), abs(xmax)))
            Bdim = poly_degree + 1
        else:
            raise ValueError("basis_type must be 'spline' or 'poly'")

        # Paramètres par arête (o,i,m) et mélange alpha (o,i), biais (o,)
        self.c = nn.Parameter(torch.randn(out_features, in_features, Bdim) * 0.01)     # (o, i, Bdim)
        self.alpha = nn.Parameter(torch.full((out_features, in_features), alpha_init)) # (o, i)
        self.bias = nn.Parameter(torch.zeros(out_features))                             # (o,)

        # Hyperparams de régularisation (KAN)
        self.lambda_alpha = float(lambda_alpha)
        self.lambda_group = float(lambda_group)
        self.lambda_tv = float(lambda_tv)
        self.lambda_poly_decay = float(lambda_poly_decay)

        # --- Skip linéaire optionnel ---
        self.use_skip = bool(use_skip)
        self.lambda_skip_l2 = float(lambda_skip_l2)
        if self.use_skip:
            self.skip = nn.Linear(in_features, out_features, bias=False)
            # init
            if skip_init == "zeros":
                nn.init.zeros_(self.skip.weight)
            elif skip_init == "identity" and in_features == out_features:
                with torch.no_grad():
                    self.skip.weight.copy_(torch.eye(in_features))
            else:  # "xavier" (par défaut si non reconnu)
                nn.init.xavier_uniform_(self.skip.weight, gain=1.0)
            # gain apprenable (pratique pour doser la contribution du skip)
            self.skip_gain = nn.Parameter(torch.tensor(float(skip_gain)))
        else:
            # place-holder pour éviter les 'has no attribute' si éteint
            self.register_parameter("skip_gain", None)
            self.skip = None  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Nin = x.shape
        assert Nin == self.in_features

        # Évalue la base pour tout (B,N) en une fois -> (B, N, Bdim)
        Phi = self.basis_1d(x)

        # (B,N,M) × (o,N,M) -> (B,o,N)
        gij = torch.einsum("bim,oim->boi", Phi, self.c)

        # Somme sur i avec alpha + biais -> (B,o)
        y = (gij * self.alpha.unsqueeze(0)).sum(dim=2) + self.bias

        # Ajout du skip linéaire si activé
        if self.use_skip:
            y = y + self.skip_gain * self.skip(x)

        return y

    def regularization(self) -> torch.Tensor:
        """
        Renvoie la somme des pénalités :
          L1(alpha) + group-lasso(c) + TV(c) [splines] + poly-decay(c) [poly]
          + (optionnel) L2 sur skip
        """
        reg = torch.tensor(0.0, device=self.c.device, dtype=self.c.dtype)
        
        # L1 sur alpha (sparsité des connexions)
        if self.lambda_alpha > 0:
            reg = reg + self.lambda_alpha * self.alpha.abs().sum()
            
        # Group-lasso sur c par (o,i) : ||c_{oi}||_2
        if self.lambda_group > 0:
            reg = reg + self.lambda_group * torch.sqrt((self.c ** 2).sum(dim=-1) + 1e-8).sum()
            
        # TV-L1 sur l'axe des noeuds (splines)
        if self.basis_type == "spline" and self.lambda_tv > 0:
            reg = reg + self.lambda_tv * (self.c[:, :, 1:] - self.c[:, :, :-1]).abs().sum()
            
        # Décroissance polynomiale (pénalise hauts degrés)
        if self.basis_type == "poly" and self.lambda_poly_decay > 0:
            deg = self.c.shape[-1]  # degree+1
            degrees = torch.arange(deg, device=self.c.device, dtype=self.c.dtype)  # 0..p
            weights = (degrees ** 2) / max(1, deg - 1) ** 2
            reg = reg + self.lambda_poly_decay * (self.c * weights).pow(2).sum()
            
        # L2 sur le skip (optionnel)
        if self.use_skip and self.lambda_skip_l2 > 0:
            reg = reg + self.lambda_skip_l2 * (self.skip.weight.pow(2).sum())
            
        return reg

    def set_regularization(
        self,
        lambda_alpha: Optional[float] = None,
        lambda_group: Optional[float] = None,
        lambda_tv: Optional[float] = None,
        lambda_poly_decay: Optional[float] = None,
        lambda_skip_l2: Optional[float] = None,
    ) -> None:
        if lambda_alpha is not None:
            self.lambda_alpha = float(lambda_alpha)
        if lambda_group is not None:
            self.lambda_group = float(lambda_group)
        if lambda_tv is not None:
            self.lambda_tv = float(lambda_tv)
        if lambda_poly_decay is not None:
            self.lambda_poly_decay = float(lambda_poly_decay)
        if lambda_skip_l2 is not None:
            self.lambda_skip_l2 = float(lambda_skip_l2)

    def get_layer_stats(self) -> dict:
        """
        Retourne des statistiques sur la couche pour debug/monitoring.
        """
        with torch.no_grad():
            stats = {
                'layer_info': {
                    'in_features': self.in_features,
                    'out_features': self.out_features,
                    'basis_type': self.basis_type,
                    'n_params': sum(p.numel() for p in self.parameters())
                },
                'weights_stats': {
                    'alpha_mean': self.alpha.mean().item(),
                    'alpha_std': self.alpha.std().item(),
                    'alpha_min': self.alpha.min().item(),
                    'alpha_max': self.alpha.max().item(),
                    'alpha_sparsity': (self.alpha.abs() < 1e-6).float().mean().item(),
                    'c_mean': self.c.mean().item(),
                    'c_std': self.c.std().item(),
                    'c_norm': self.c.norm().item(),
                    'bias_norm': self.bias.norm().item()
                },
                'regularization': {
                    'total_reg': self.regularization().item(),
                    'alpha_l1': (self.lambda_alpha * self.alpha.abs().sum()).item(),
                    'group_lasso': (self.lambda_group * torch.sqrt((self.c ** 2).sum(dim=-1) + 1e-8).sum()).item()
                }
            }
            # Ajouter les régularisations spécifiques à chaque type de base
            if self.basis_type == "spline" and self.lambda_tv > 0:
                stats['regularization']['tv_penalty'] = (
                    self.lambda_tv * (self.c[:, :, 1:] - self.c[:, :, :-1]).abs().sum()
                ).item()
            if self.basis_type == "poly" and self.lambda_poly_decay > 0:
                deg = self.c.shape[-1]
                degrees = torch.arange(deg, device=self.c.device, dtype=self.c.dtype)
                weights = (degrees ** 2) / max(1, deg - 1) ** 2
                stats['regularization']['poly_decay'] = (
                    self.lambda_poly_decay * (self.c * weights).pow(2).sum()
                ).item()
            if self.use_skip and self.lambda_skip_l2 > 0:
                stats['regularization']['skip_l2'] = (
                    self.lambda_skip_l2 * (self.skip.weight.pow(2).sum())
                ).item()
            return stats

    def print_layer_info(self, verbose: bool = True):
        """
        Affiche les informations sur la couche.
        """
        stats = self.get_layer_stats()
        if verbose:
            print(f"🔧 KAN Layer ({self.basis_type}): {self.in_features} → {self.out_features}"
                  f"{' + skip' if self.use_skip else ''}")
            print(f"   📊 Paramètres: {stats['layer_info']['n_params']:,}")
            print(f"   ⚖️  Alpha: μ={stats['weights_stats']['alpha_mean']:.4f}, "
                  f"σ={stats['weights_stats']['alpha_std']:.4f}, "
                  f"sparsity={stats['weights_stats']['alpha_sparsity']:.2%}")
            print(f"   🎯 C coeffs: μ={stats['weights_stats']['c_mean']:.4f}, "
                  f"σ={stats['weights_stats']['c_std']:.4f}, "
                  f"||C||={stats['weights_stats']['c_norm']:.4f}")
            print(f"   📈 Régularisation totale: {stats['regularization']['total_reg']:.6f}")
        else:
            print(f"KAN{self.basis_type}({self.in_features}→{self.out_features})"
                  f"{'+skip' if self.use_skip else ''}: "
                  f"α_sparsity={stats['weights_stats']['alpha_sparsity']:.1%}, "
                  f"reg={stats['regularization']['total_reg']:.2e}")

    def check_gradients(self) -> dict:
        """
        Vérifie l'état des gradients pour debug.
        """
        grad_stats = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.abs().max().item()
                grad_mean = param.grad.mean().item()
                has_nan = torch.isnan(param.grad).any().item()
                has_inf = torch.isinf(param.grad).any().item()
                grad_stats[name] = {
                    'norm': grad_norm,
                    'max_abs': grad_max,
                    'mean': grad_mean,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'is_zero': grad_norm < 1e-10
                }
            else:
                grad_stats[name] = {'grad': None}
        return grad_stats

    def print_gradient_info(self):
        """
        Affiche des informations sur les gradients.
        """
        grad_stats = self.check_gradients()
        print(f"   🔄 Gradients pour KAN Layer {self.in_features}→{self.out_features}:")
        for param_name, stats in grad_stats.items():
            if stats.get('grad') is None:
                print(f"      {param_name}: No gradient")
            else:
                status = ""
                if stats['has_nan']:
                    status += "❌NaN "
                if stats['has_inf']:
                    status += "❌Inf "
                if stats['is_zero']:
                    status += "⚠️Zero "
                print(f"      {param_name}: norm={stats['norm']:.2e}, "
                      f"max={stats['max_abs']:.2e} {status}")

    def get_activation_stats(self, x: torch.Tensor) -> dict:
        """
        Analyse les activations pour une entrée donnée.
        """
        with torch.no_grad():
            # Forward pass (mêmes vecteurs que forward)
            B, Nin = x.shape
            Phi = self.basis_1d(x)  # (B, N, Bdim)
            gij = torch.einsum("bim,oim->boi", Phi, self.c)
            y_nl = (gij * self.alpha.unsqueeze(0)).sum(dim=2) + self.bias
            y = y_nl + (self.skip_gain * self.skip(x) if self.use_skip else 0.0)

            stats = {
                'input_stats': {
                    'mean': x.mean().item(),
                    'std': x.std().item(),
                    'min': x.min().item(),
                    'max': x.max().item()
                },
                'basis_stats': {
                    'phi_mean': Phi.mean().item(),
                    'phi_std': Phi.std().item(),
                    'phi_max': Phi.max().item()
                },
                'output_stats': {
                    'mean': y.mean().item(),
                    'std': y.std().item(),
                    'min': y.min().item(),
                    'max': y.max().item()
                }
            }
            return stats


