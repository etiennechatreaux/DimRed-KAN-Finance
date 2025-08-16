# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn

class Basis1D(nn.Module):
    """Interface minimale pour une base 1D."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class HatSpline1D(Basis1D):
    """
    Base 'hat' (linéaire par morceaux) sur grille régulière [xmin, xmax].
    Retourne:
      - si x shape (B,)    -> Phi (B, M)
      - si x shape (B, N)  -> Phi (B, N, M)
    """
    def __init__(self, M: int = 16, xmin: float = -3.5, xmax: float = 3.5):
        super().__init__()
        assert M >= 2
        self.M = M
        # La grille suit le device/dtype du module automatiquement avec .to()
        self.register_buffer("grid", torch.linspace(xmin, xmax, M), persistent=False)

    def _phi_1d(self, x: torch.Tensor) -> torch.Tensor:
        # aligne la grille sur le device/dtype de x
        grid = self.grid.to(device=x.device, dtype=x.dtype)

        # clamp dans [xmin, xmax]
        x = torch.clamp(x, grid[0], grid[-1])
        B = x.shape[0]

        # indices voisins
        idx = torch.bucketize(x, grid) - 1
        idx = torch.clamp(idx, 0, self.M - 2)

        x0, x1 = grid[idx], grid[idx + 1]
        w = (x - x0) / (x1 - x0 + 1e-8)

        # matrice Phi initialisée sur le bon device/dtype
        Phi = torch.zeros((B, self.M), device=x.device, dtype=x.dtype)
        Phi.scatter_add_(1, idx.unsqueeze(1), (1 - w).unsqueeze(1))
        Phi.scatter_add_(1, (idx + 1).unsqueeze(1), w.unsqueeze(1))
        return Phi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return self._phi_1d(x)
        elif x.dim() == 2:
            B, N = x.shape
            xf = x.reshape(-1)                    # (B*N,)
            Phi_flat = self._phi_1d(xf)           # (B*N, M)
            return Phi_flat.view(B, N, self.M)    # (B, N, M)
        else:
            raise ValueError("HatSpline1D expects x of shape (B,) or (B,N)")

class PolyBasis1D(Basis1D):
    """
    Base polynomiale: [1, x, x^2, ..., x^p]
    Retourne:
      - si x shape (B,)    -> Phi (B, p+1)
      - si x shape (B, N)  -> Phi (B, N, p+1)
    """
    def __init__(self, degree: int = 5, clip: float = 3.5):
        super().__init__()
        assert degree >= 0
        self.degree = int(degree)
        self.clip = float(clip)

    def _phi_1d(self, x: torch.Tensor) -> torch.Tensor:
        # clamp puis normalisation
        x = torch.clamp(x, -self.clip, self.clip) / self.clip
        B = x.shape[0]

        # 1, x, x^2, ..., x^degree
        Phi_list = [torch.ones(B, device=x.device, dtype=x.dtype)]
        cur = x
        for _ in range(1, self.degree + 1):
            Phi_list.append(cur)
            cur = cur * x
        return torch.stack(Phi_list, dim=1)  # (B, degree+1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return self._phi_1d(x)
        elif x.dim() == 2:
            B, N = x.shape
            xf = x.reshape(-1)                    # (B*N,)
            Phi_flat = self._phi_1d(xf)           # (B*N, degree+1)
            return Phi_flat.view(B, N, -1)        # (B, N, degree+1)
        else:
            raise ValueError("PolyBasis1D expects x of shape (B,) or (B,N)")
