"""
Autoencodeur KAN avec options de loss basées sur la corrélation et le cosinus.

Fonctions de loss supportées :
- "mse" : Mean Squared Error (par défaut)
- "huber" : Huber Loss (plus robuste aux outliers)
- "correlation" : 1 - corr(x, x̂) par colonne (invariante à l'échelle)
- "cosine" : 1 - cos(x, x̂) par colonne (invariante à l'échelle)
- "corr_mse_mix" : α*MSE + β*(1-corr) (MSE stabilise, corr force la structure)

La loss de corrélation/cosinus empêche que x̂ soit constant car la corrélation
d'un vecteur constant est indéfinie/nulle.
"""

from __future__ import annotations
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kan_layers import KANLayer


class WeightedHuberLoss(nn.Module):
    """
    Huber Loss pondérée pour gérer les masks et poids de fiabilité.
    """
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        
    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input: Prédictions (B, N) ou (B, ...)
            target: Vraies valeurs (B, N) ou (B, ...)
            weights: Poids de fiabilité (B, N) ou (B, ...), valeurs entre 0 et 1
        """
        # Calculer la Huber loss classique
        diff = input - target
        abs_diff = torch.abs(diff)
        
        # Huber loss: quadratique si |diff| <= delta, linéaire sinon
        is_small = abs_diff <= self.delta
        loss = torch.where(
            is_small,
            0.5 * diff.pow(2),
            self.delta * abs_diff - 0.5 * self.delta.pow(2)
        )
        
        # Appliquer les poids si fournis
        if weights is not None:
            # S'assurer que weights a la même shape que loss
            if weights.shape != loss.shape:
                raise ValueError(f"Shape mismatch: weights {weights.shape} vs loss {loss.shape}")
            
            loss = loss * weights
            
            # Pour le reduction, normaliser par la somme des poids non-nuls
            if self.reduction == 'mean':
                weight_sum = weights.sum()
                if weight_sum > 0:
                    return loss.sum() / weight_sum
                else:
                    return torch.tensor(0.0, device=loss.device, requires_grad=True)
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            # Pas de pondération, Huber loss classique
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss


class CorrelationLoss(nn.Module):
    """
    Loss basée sur la corrélation : 1 - corr(x, x̂) par colonne.
    Invariante à l'échelle et empêche les vecteurs constants.
    """
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Prédictions (B, N)
            target: Vraies valeurs (B, N)
        """
        # Calculer la corrélation par colonne
        # Centrer les données
        input_centered = input - input.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        
        # Calculer le numérateur et dénominateur de la corrélation
        numerator = (input_centered * target_centered).sum(dim=0)
        input_std = torch.sqrt((input_centered ** 2).sum(dim=0) + self.eps)
        target_std = torch.sqrt((target_centered ** 2).sum(dim=0) + self.eps)
        
        # Éviter la division par zéro et les corrélations indéfinies
        denominator = input_std * target_std
        valid_mask = denominator > self.eps
        
        # Calculer la corrélation seulement pour les colonnes valides
        correlation = torch.zeros_like(numerator)
        correlation[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        # Clamper la corrélation entre -1 et 1 pour éviter les erreurs numériques
        correlation = torch.clamp(correlation, -1.0, 1.0)
        
        # Loss = 1 - corrélation (on veut maximiser la corrélation)
        loss = 1.0 - correlation
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CosineLoss(nn.Module):
    """
    Loss basée sur le cosinus : 1 - cos(x, x̂) par colonne.
    Invariante à l'échelle et empêche les vecteurs constants.
    """
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Prédictions (B, N)
            target: Vraies valeurs (B, N)
        """
        # Calculer le cosinus par colonne
        # Normaliser les vecteurs colonnes
        input_norm = F.normalize(input, p=2, dim=0, eps=self.eps)
        target_norm = F.normalize(target, p=2, dim=0, eps=self.eps)
        
        # Calculer le cosinus (produit scalaire des vecteurs normalisés)
        cosine_sim = (input_norm * target_norm).sum(dim=0)
        
        # Clamper pour éviter les erreurs numériques
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
        
        # Loss = 1 - cosinus (on veut maximiser la similarité cosinus)
        loss = 1.0 - cosine_sim
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CorrelationMSELoss(nn.Module):
    """
    Loss mixte : α*MSE + β*(1-corr).
    La MSE stabilise, la corrélation force la structure (non-constance).
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.eps = eps
        self.mse_loss = nn.MSELoss(reduction='none')
        self.corr_loss = CorrelationLoss(reduction='none', eps=eps)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Prédictions (B, N)
            target: Vraies valeurs (B, N)
        """
        # Calculer MSE par colonne
        mse = self.mse_loss(input, target)
        if self.reduction == 'mean':
            mse = mse.mean(dim=0)  # Moyenne par colonne
        elif self.reduction == 'sum':
            mse = mse.sum(dim=0)   # Somme par colonne
        
        # Calculer la loss de corrélation
        corr_loss = self.corr_loss(input, target)
        
        # Combiner les deux losses
        total_loss = self.alpha * mse + self.beta * corr_loss
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


class KANAutoencoderCorrelation(nn.Module):
    def __init__(
        self,
        input_dim: int,
        k: int = 5,
        *,
        hidden_dims: Optional[List[int]] = None,   # ex: [128, 32]
        basis_type: str = "spline",                # "spline" | "poly"
        M: int = 16,                               # nb fonctions pour spline
        poly_degree: int = 5,                      # degré polynômial (nb bases = p+1)
        xmin: float = -3.5,
        xmax: float = 3.5,
        dropout_p: float = 0.05,
        use_silu: bool = True,
        # fonction de loss
        loss_type: str = "mse",                    # "mse" | "huber" | "correlation" | "cosine" | "corr_mse_mix"
        huber_delta: float = 1.0,                  # paramètre delta pour Huber Loss
        # paramètres pour corr_mse_mix
        alpha: float = 1.0,                        # poids pour MSE dans corr_mse_mix
        beta: float = 1.0,                         # poids pour corrélation dans corr_mse_mix
        
        
        # régularisations KAN
        lambda_alpha: float = 1e-4,
        lambda_group: float = 1e-5,
        lambda_tv: float = 1e-4,
        lambda_poly_decay: float = 0.0,
        # -------- options skip linéaire --------
        use_global_skip: bool = True,              # skip entrée→sortie
        use_skip: bool = False,                    # skips dans chaque KANLayer
        skip_init: str = "zeros",                  # "zeros" | "xavier" | "identity"
        skip_gain: float = 1.0,
        lambda_skip_l2: float = 0.0,
        max_skip_gain: float = 1.0                 # gain max autorisé (0 = pas de limite)
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.k = int(k)

        if hidden_dims is None:
            hidden_dims = [128, 32]
        self.hidden_dims = list(hidden_dims)
        self.use_silu = use_silu

        # Configuration de la fonction de loss
        self.loss_type = loss_type.lower()
        self.huber_delta = huber_delta
        self.alpha = alpha
        self.beta = beta
        
        if self.loss_type not in ["mse", "huber", "correlation", "cosine", "corr_mse_mix"]:
            raise ValueError(f"loss_type doit être 'mse', 'huber', 'correlation', 'cosine', ou 'corr_mse_mix', reçu: {loss_type}")

        # --------- ENCODEUR ---------
        self.encoder_layers = nn.ModuleList()
        self.encoder_activations = nn.ModuleList()
        self.encoder_dropouts = nn.ModuleList()

        prev_dim = input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            self.encoder_layers.append(
                KANLayer(
                    prev_dim, hidden_dim,
                    basis_type=basis_type, M=M, poly_degree=poly_degree,
                    xmin=xmin, xmax=xmax,
                    lambda_alpha=lambda_alpha, lambda_group=lambda_group,
                    lambda_tv=lambda_tv, lambda_poly_decay=lambda_poly_decay,
                    use_skip=use_skip, skip_init=skip_init,
                    skip_gain=skip_gain, lambda_skip_l2=lambda_skip_l2
                )
            )
            self.encoder_activations.append(nn.SiLU() if use_silu else nn.Identity())
            self.encoder_dropouts.append(nn.Dropout(dropout_p) if i == 0 else nn.Identity())
            prev_dim = hidden_dim

        # Vers l'espace latent
        self.to_latent = nn.Linear(self.hidden_dims[-1], k, bias=True)

        # --------- DECODEUR ---------
        self.from_latent = nn.Linear(k, self.hidden_dims[-1], bias=True)

        self.decoder_layers = nn.ModuleList()
        self.decoder_activations = nn.ModuleList()
        self.decoder_dropouts = nn.ModuleList()

        reversed_dims = list(reversed(self.hidden_dims))
        for i, hidden_dim in enumerate(reversed_dims[:-1]):
            next_dim = reversed_dims[i + 1]
            self.decoder_layers.append(
                KANLayer(
                    hidden_dim, next_dim,
                    basis_type=basis_type, M=M, poly_degree=poly_degree,
                    xmin=xmin, xmax=xmax,
                    lambda_alpha=lambda_alpha, lambda_group=lambda_group,
                    lambda_tv=lambda_tv, lambda_poly_decay=lambda_poly_decay,
                    use_skip=use_skip, skip_init=skip_init,
                    skip_gain=skip_gain, lambda_skip_l2=lambda_skip_l2
                )
            )
            self.decoder_activations.append(nn.SiLU() if use_silu else nn.Identity())
            self.decoder_dropouts.append(nn.Dropout(dropout_p) if i == 0 else nn.Identity())

        # Couche finale vers la sortie
        self.decoder_layers.append(
            KANLayer(
                self.hidden_dims[0], input_dim,
                basis_type=basis_type, M=M, poly_degree=poly_degree,
                xmin=xmin, xmax=xmax,
                lambda_alpha=lambda_alpha, lambda_group=lambda_group,
                lambda_tv=lambda_tv, lambda_poly_decay=lambda_poly_decay,
                use_skip=use_skip, skip_init=skip_init,
                skip_gain=skip_gain, lambda_skip_l2=lambda_skip_l2
            )
        )

        # --------- GLOBAL SKIP (entrée -> sortie) ---------
        self.use_global_skip = bool(use_global_skip)
        self.max_skip_gain = float(max_skip_gain)
        if self.use_global_skip:
            self.global_skip = nn.Linear(input_dim, input_dim, bias=False)
            if skip_init == "zeros":
                nn.init.zeros_(self.global_skip.weight)
            elif skip_init == "identity":
                with torch.no_grad():
                    nn.init.eye_(self.global_skip.weight)
            else:
                nn.init.xavier_uniform_(self.global_skip.weight, gain=1.0)
            self.global_skip_gain = nn.Parameter(torch.tensor(float(skip_gain)))
            self.lambda_global_skip_l2 = float(lambda_skip_l2)
        else:
            self.global_skip = None
            self.global_skip_gain = None
            self.lambda_global_skip_l2 = 0.0

    # ---------- API ----------
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x
        for kan_layer, activation, dropout in zip(
            self.encoder_layers, self.encoder_activations, self.encoder_dropouts
        ):
            h = dropout(activation(kan_layer(h)))
        return self.to_latent(h)

    def decode(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.from_latent(z)
        for kan_layer, activation, dropout in zip(
            self.decoder_layers[:-1], self.decoder_activations, self.decoder_dropouts
        ):
            h = dropout(activation(kan_layer(h)))
        return self.decoder_layers[-1](h)

    def forward(self, x, mask=None):
        z = self.encode(x, mask)
        x_kan = self.decode(z, mask)
        if self.use_global_skip:
            # Appliquer le clipping du gain si max_skip_gain > 0
            if self.max_skip_gain > 0:
                effective_gain = torch.clamp(self.global_skip_gain, max=self.max_skip_gain)
            else:
                effective_gain = self.global_skip_gain
            x_hat = x_kan + effective_gain * self.global_skip(x)
        else:
            x_hat = x_kan
        return x_hat, z

    def get_loss_criterion(self, weighted: bool = False):
        if self.loss_type == "mse":
            return nn.MSELoss()
        elif self.loss_type == "huber":
            return nn.HuberLoss(delta=self.huber_delta)
        elif self.loss_type == "correlation":
            return CorrelationLoss()
        elif self.loss_type == "cosine":
            return CosineLoss()
        elif self.loss_type == "corr_mse_mix":
            return CorrelationMSELoss(alpha=self.alpha, beta=self.beta)
        else:
            raise ValueError(f"Type de loss non supporté: {self.loss_type}")

    def regularization(self) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.encoder_layers:
            reg_loss += layer.regularization()
        for layer in self.decoder_layers:
            reg_loss += layer.regularization()
        if self.use_global_skip and self.lambda_global_skip_l2 > 0.0:
            reg_loss += self.lambda_global_skip_l2 * (self.global_skip.weight.pow(2).sum())
        return reg_loss

    def fit(
        self, X: torch.Tensor,
        epochs: int = 100, batch_size: int = 64,
        learning_rate: float = 0.001, weight_decay: float = 1e-5,
        validation_split: float = 0.2, patience: int = 10,
        verbose: bool = True, lambda_reg: float = 1.0,
        device: Optional[torch.device] = None
    ) -> dict:
        import time
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # split train/val
        n_samples = X.size(0)
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)
        X_train, X_val = X[indices[n_val:]], (X[indices[:n_val]] if n_val > 0 else None)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, X_train),
            batch_size=batch_size, shuffle=True,
            pin_memory=(device.type == "cuda")
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = self.get_loss_criterion()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(1, patience // 2), factor=0.5)

        history = {'train_loss': [], 'val_loss': [], 'learning_rate': [],
                   'regularization': [], 'skip_gain': [], 'skip_weight_norm': []}

        best_val_loss = float('10000')
        patience_counter = 0
        t0 = time.time()

        for epoch in range(epochs):
            self.train()
            train_loss, reg_loss = 0.0, 0.0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                x_reconstructed, _ = self(batch_x)
                recon_loss = criterion(x_reconstructed, batch_x)
                reg_term = self.regularization()
                loss = recon_loss + lambda_reg * reg_term
                loss.backward()
                optimizer.step()
                train_loss += recon_loss.item()
                reg_loss += reg_term.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_reg_loss = reg_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            history['regularization'].append(avg_reg_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # log skip evolution
            if self.use_global_skip:
                with torch.no_grad():
                    # Logger la valeur effective du gain (après clipping)
                    if self.max_skip_gain > 0:
                        effective_gain = torch.clamp(self.global_skip_gain, max=self.max_skip_gain)
                    else:
                        effective_gain = self.global_skip_gain
                    history['skip_gain'].append(float(effective_gain.item()))
                    history['skip_weight_norm'].append(float(self.global_skip.weight.norm().item()))

            # validation
            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    vh, _ = self(X_val.to(device))
                    val_loss = criterion(vh, X_val.to(device)).item()
                history['val_loss'].append(val_loss)
                scheduler.step(val_loss)

                # Comparer avec la validation loss précédente
                prev_val_loss = history['val_loss'][-2] if len(history['val_loss']) > 1 else float('inf')
                if val_loss < best_val_loss - 1e-9:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"🛑 Early stopping à l'époque {epoch+1}")
                        break

            if verbose:
                validation_symbol = "✅" if val_loss < prev_val_loss else "❌"
                print(f"📈 Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.6f} | "
                      f"Val: {val_loss if X_val is not None else 0:.6f} {validation_symbol} | "
                      f"Reg: {avg_reg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
                if self.use_global_skip:
                    print(f"   ↳ skip_gain={history['skip_gain'][-1]:.4f} | "
                          f"||W_skip||={history['skip_weight_norm'][-1]:.4f}")

        if 'best_state' in locals():
            self.load_state_dict(best_state)

        history['training_time'] = time.time() - t0
        return history
