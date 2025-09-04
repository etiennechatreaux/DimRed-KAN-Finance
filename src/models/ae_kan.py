from __future__ import annotations
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from .kan_layers import KANLayer


# ----------------------- Losses -----------------------

class WeightedHuberLoss(nn.Module):
    """
    Huber Loss pondérée (supporte poids W et masque M).
    loss = 0.5*diff^2 si |diff|<=delta, sinon delta*|diff| - 0.5*delta^2
    """
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = float(delta)
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        diff = input - target
        abs_diff = diff.abs()
        small = abs_diff <= self.delta
        loss = torch.where(small, 0.5 * diff.pow(2), self.delta * abs_diff - 0.5 * (self.delta ** 2))

        if weights is not None:
            if weights.shape != loss.shape:
                raise ValueError(f"Shape mismatch: weights {weights.shape} vs loss {loss.shape}")
            loss = loss * weights
            if self.reduction == 'mean':
                wsum = weights.sum()
                return loss.sum() / (wsum + 1e-12)
            elif self.reduction == 'sum':
                return loss.sum()
            return loss
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss
        
        
class WeightedMSE(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        loss = (input - target).pow(2)
        loss = loss * weights
        if self.reduction == 'mean':
            return loss.sum() / (weights.sum() + 1e-12)
        elif self.reduction == 'sum':
            return loss.sum()
        return loss



# ------------------- KAN Autoencoder -------------------

class KANAutoencoder(nn.Module):
    """
    Autoencodeur KAN :
      - Encoder: empilement KANLayer -> SiLU (optionnel) -> Dropout (optionnel)
      - Latent: Linear(h_last -> k)
      - Decoder: Linear(k -> h_last) -> empilement KANLayer -> SiLU/Dropout
      - Sortie: KANLayer(h1 -> input_dim)
    Skip global optionnel: y = y_kan + gain * W_skip x  (gain borné par max_skip_gain si > 0)
    """

    def __init__(
        self,
        input_dim: int,
        k: int = 5,
        *,
        hidden_dims: Optional[List[int]] = None,   # ex: [128, 32]
        basis_type: str = "spline",                # "spline" | "poly"
        M: int = 16,                               # nb bases spline
        poly_degree: int = 5,                      # pour base polynomiale (nb bases = p+1)
        xmin: float = -3.5,
        xmax: float = 3.5,
        dropout_p: float = 0.05,
        use_silu: bool = True,
        
        # Loss
        loss_type: str = "huber",                    # "mse" | "huber"
        huber_delta: float = 1.0,
        
        # Régularisations KAN (propagées dans KANLayer)
        lambda_alpha: float = 1e-4,
        lambda_group: float = 1e-5,
        lambda_tv: float = 1e-4,
        lambda_poly_decay: float = 0.0,
        
        # Skip linéaire global
        use_global_skip: bool = True,
        use_skip: bool = False,                    # skips internes aux KANLayer
        skip_init: str = "zeros",                  # "zeros" | "xavier" | "identity"
        skip_gain: float = 1.0,
        lambda_skip_l2: float = 0.0,
        max_skip_gain: float = 1.0
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.k = int(k)
        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else [128, 32]
        self.use_silu = bool(use_silu)

        # Config loss
        self.loss_type = loss_type.lower()
        if self.loss_type not in ("mse", "huber"):
            raise ValueError("loss_type must be 'mse' or 'huber'")
        self.huber_delta = float(huber_delta)

        # Stockage des paramètres d'entraînement pour fit()
        self.dropout_p = float(dropout_p)
        self.basis_type = basis_type
        self.M = int(M)
        self.poly_degree = int(poly_degree)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.lambda_alpha = float(lambda_alpha)
        self.lambda_group = float(lambda_group)
        self.lambda_tv = float(lambda_tv)
        self.lambda_poly_decay = float(lambda_poly_decay)
        self.use_skip = bool(use_skip)
        self.skip_init = skip_init
        self.skip_gain_init = float(skip_gain)
        self.lambda_skip_l2 = float(lambda_skip_l2)

        # ---------- ENCODEUR ----------
        self.encoder_layers = nn.ModuleList()
        self.encoder_activ = nn.ModuleList()
        self.encoder_drop = nn.ModuleList()

        prev = self.input_dim
        for i, h in enumerate(self.hidden_dims):
            self.encoder_layers.append(
                KANLayer(
                    prev, h,
                    basis_type=basis_type, M=M, poly_degree=poly_degree,
                    xmin=xmin, xmax=xmax,
                    lambda_alpha=lambda_alpha, lambda_group=lambda_group,
                    lambda_tv=lambda_tv, lambda_poly_decay=lambda_poly_decay,
                    use_skip=use_skip, skip_init=skip_init,
                    skip_gain=skip_gain, lambda_skip_l2=lambda_skip_l2
                )
            )
            self.encoder_activ.append(nn.SiLU() if self.use_silu else nn.Identity())
            self.encoder_drop.append(nn.Dropout(dropout_p) if dropout_p > 0 and i == 0 else nn.Identity())
            prev = h

        self.to_latent = nn.Linear(prev, self.k, bias=True)

        # ---------- DECODEUR ----------
        self.from_latent = nn.Linear(self.k, self.hidden_dims[-1], bias=True)

        self.decoder_layers = nn.ModuleList()
        self.decoder_activ = nn.ModuleList()
        self.decoder_drop = nn.ModuleList()

        rev_dims = list(reversed(self.hidden_dims))
        for i, h in enumerate(rev_dims[:-1]):
            nxt = rev_dims[i + 1]
            self.decoder_layers.append(
                KANLayer(
                    h, nxt,
                    basis_type=basis_type, M=M, poly_degree=poly_degree,
                    xmin=xmin, xmax=xmax,
                    lambda_alpha=lambda_alpha, lambda_group=lambda_group,
                    lambda_tv=lambda_tv, lambda_poly_decay=lambda_poly_decay,
                    use_skip=use_skip, skip_init=skip_init,
                    skip_gain=skip_gain, lambda_skip_l2=lambda_skip_l2
                )
            )
            self.decoder_activ.append(nn.SiLU() if self.use_silu else nn.Identity())
            self.decoder_drop.append(nn.Dropout(dropout_p) if dropout_p > 0 and i == 0 else nn.Identity())

        # Dernière KAN vers la sortie
        self.decoder_layers.append(
            KANLayer(
                self.hidden_dims[0], self.input_dim,
                basis_type=basis_type, M=M, poly_degree=poly_degree,
                xmin=xmin, xmax=xmax,
                lambda_alpha=lambda_alpha, lambda_group=lambda_group,
                lambda_tv=lambda_tv, lambda_poly_decay=lambda_poly_decay,
                use_skip=use_skip, skip_init=skip_init,
                skip_gain=skip_gain, lambda_skip_l2=lambda_skip_l2
            )
        )

        # ---------- SKIP GLOBAL ----------
        self.use_global_skip = bool(use_global_skip)
        self.max_skip_gain = float(max_skip_gain)
        if self.use_global_skip:
            self.global_skip = nn.Linear(self.input_dim, self.input_dim, bias=False)
            if skip_init == "zeros":
                nn.init.zeros_(self.global_skip.weight)
            elif skip_init == "identity":
                with torch.no_grad():
                    nn.init.eye_(self.global_skip.weight)
            else:
                nn.init.xavier_uniform_(self.global_skip.weight, gain=1.0)
            # Rendre global_skip_gain un Parameter apprenable
            self.global_skip_gain = nn.Parameter(torch.tensor(float(skip_gain), requires_grad=True))
            self.lambda_global_skip_l2 = float(lambda_skip_l2)
        else:
            self.global_skip = None
            self.global_skip_gain = None
            self.lambda_global_skip_l2 = 0.0

    # -------------------- API de passage --------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for L, A, D in zip(self.encoder_layers, self.encoder_activ, self.encoder_drop):
            h = D(A(L(h)))
        return self.to_latent(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        for L, A, D in zip(self.decoder_layers[:-1], self.decoder_activ, self.decoder_drop):
            h = D(A(L(h)))
        return self.decoder_layers[-1](h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_kan = self.decode(z)
        if self.use_global_skip:
            # Borne du gain appliquée à l'inférence
            gain = self.global_skip_gain
            if self.max_skip_gain > 0:
                gain = torch.clamp(gain, max=self.max_skip_gain)
            x_hat = x_kan + gain * self.global_skip(x)
        else:
            x_hat = x_kan
        return x_hat, z

    # -------------------- Outils internes --------------------

    def _make_criterion(self, weighted: bool) -> nn.Module:
        if self.loss_type == "mse":
            return nn.MSELoss(reduction='mean') if not weighted else WeightedMSE()
        else:  # huber
            return nn.HuberLoss(delta=self.huber_delta, reduction='mean') if not weighted else WeightedHuberLoss(delta=self.huber_delta)

    def regularization(self) -> torch.Tensor:
        reg = torch.zeros((), device=next(self.parameters()).device)
        for L in self.encoder_layers:
            reg = reg + L.regularization()
        for L in self.decoder_layers:
            reg = reg + L.regularization()
        if self.use_global_skip and self.lambda_global_skip_l2 > 0.0:
            reg = reg + self.lambda_global_skip_l2 * self.global_skip.weight.pow(2).sum()
            # Ajouter régularisation sur le gain également
            reg = reg + self.lambda_global_skip_l2 * self.global_skip_gain.pow(2)
        return reg

    # ------------------------- Fit -------------------------

    @torch.no_grad()
    def _chrono_split(self, X: torch.Tensor, frac_val: float) -> Tuple[torch.Tensor, torch.Tensor]:
        n = X.size(0)
        n_val = int(max(1, round(n * frac_val))) if n > 1 else 0
        n_train = n - n_val
        return X[:n_train], (X[n_train:] if n_val > 0 else None)

    @torch.no_grad()
    def _chrono_split_opt(self, T: Optional[torch.Tensor], frac_val: float) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if T is None:
            return None, None
        n = T.size(0)
        n_val = int(max(1, round(n * frac_val))) if n > 1 else 0
        n_train = n - n_val
        return T[:n_train], (T[n_train:] if n_val > 0 else None)

    def fit(
        self,
        X: torch.Tensor,
        W: Optional[torch.Tensor] = None,
        M: Optional[torch.Tensor] = None,
        *,
        validation_split: float = 0.2,
        chronological_split: bool = True,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        lambda_reg: float = 1.0,
        use_weighted_loss: bool = True,
        verbose: bool = True,
        device: Optional[torch.device] = None,
        num_workers: int = 0,
        seed: Optional[int] = None
    ) -> dict:
        """
        Entraîne le modèle. Le split train/val est fait **à l'intérieur** (par défaut chronologique).
        Args:
            X, W, M: tenseurs (n_samples, input_dim). W/M optionnels.
            validation_split: part de validation (0..0.5 recommandé).
            chronological_split: si True, prend la fin pour val (anti-fuite temporelle).
            patience: early stopping sur la val loss.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        X = X.to(device)
        W = W.to(device) if W is not None else None
        M = M.to(device) if M is not None else None

        # ---- Split interne (chronologique par défaut) ----
        if chronological_split:
            X_tr, X_va = self._chrono_split(X, validation_split)
            W_tr, W_va = self._chrono_split_opt(W, validation_split)
            M_tr, M_va = self._chrono_split_opt(M, validation_split)
        else:
            n = X.size(0)
            n_val = int(max(1, round(n * validation_split))) if n > 1 else 0
            idx = torch.randperm(n, device=device)
            val_idx = idx[-n_val:] if n_val > 0 else None
            tr_idx = idx[:-n_val] if n_val > 0 else idx
            X_tr = X[tr_idx]
            X_va = X[val_idx] if n_val > 0 else None
            W_tr = W[tr_idx] if (W is not None and n_val > 0) else (W if W is not None else None)
            W_va = W[val_idx] if (W is not None and n_val > 0) else None
            M_tr = M[tr_idx] if (M is not None and n_val > 0) else (M if M is not None else None)
            M_va = M[val_idx] if (M is not None and n_val > 0) else None

        # ---- Dataloaders ----
        pin = device.type == "cuda" and X_tr.device.type == "cpu"

        if use_weighted_loss and (W_tr is not None):
            train_ds = torch.utils.data.TensorDataset(X_tr, X_tr, W_tr, (M_tr if M_tr is not None else torch.ones_like(X_tr)))
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
        else:
            train_ds = torch.utils.data.TensorDataset(X_tr, X_tr)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)

        has_val = X_va is not None and X_va.size(0) > 0
        if has_val:
            if use_weighted_loss and (W_va is not None):
                val_ds = torch.utils.data.TensorDataset(X_va, X_va, W_va, (M_va if M_va is not None else torch.ones_like(X_va)))
                val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
            else:
                val_ds = torch.utils.data.TensorDataset(X_va, X_va)
                val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
        else:
            val_loader = None

        # ---- Optim / Loss / Sched ----
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        weighted = bool(use_weighted_loss and W is not None)
        criterion = self._make_criterion(weighted=weighted)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=max(1, patience // 2), factor=0.5, verbose=False
        )

        # ---- Boucle d'entraînement ----
        best_val = float('inf')
        best_state = None
        wait = 0

        history = {
            "train_loss": [],
            "val_loss": [],
            "reg": [],
            "lr": [],
            "skip_gain": [],
            "skip_weight_norm": []
        }

        for epoch in range(1, epochs + 1):
            self.train()
            running = 0.0
            reg_running = 0.0

            if weighted and (W_tr is not None):
                for bx, by, bw, bm in train_loader:
                    bx = bx.to(device, non_blocking=True)
                    by = by.to(device, non_blocking=True)
                    bw = bw.to(device, non_blocking=True)
                    bm = bm.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    yhat, _ = self(bx)
                    eff_w = bw * bm
                    recon = criterion(yhat, by, weights=eff_w)
                    reg = self.regularization()
                    loss = recon + lambda_reg * reg
                    loss.backward()
                    optimizer.step()

                    running += recon.item()
                    reg_running += reg.item()
            else:
                for bx, by in train_loader:
                    bx = bx.to(device, non_blocking=True)
                    by = by.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    yhat, _ = self(bx)
                    recon = criterion(yhat, by)
                    reg = self.regularization()
                    loss = recon + lambda_reg * reg
                    loss.backward()
                    optimizer.step()

                    running += recon.item()
                    reg_running += reg.item()

            n_batches = max(1, len(train_loader))
            train_loss = running / n_batches
            reg_loss = reg_running / n_batches
            history["train_loss"].append(train_loss)
            history["reg"].append(reg_loss)
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))

            # Log skip - Accès direct au paramètre sans torch.no_grad() pour capturer les gradients
            if self.use_global_skip:
                # Récupérer la valeur actuelle du gain (peut avoir évolué pendant l'entraînement)
                current_gain = float(self.global_skip_gain.item())
                history["skip_gain"].append(current_gain)
                history["skip_weight_norm"].append(float(self.global_skip.weight.norm().item()))
            else:
                history["skip_gain"].append(0.0)
                history["skip_weight_norm"].append(0.0)

            # ---- Validation ----
            val_loss = None
            if has_val and val_loader is not None:
                self.eval()
                v_running = 0.0
                with torch.no_grad():
                    if weighted and (W_va is not None):
                        for vx, vy, vw, vm in val_loader:
                            vx = vx.to(device, non_blocking=True)
                            vy = vy.to(device, non_blocking=True)
                            vw = vw.to(device, non_blocking=True)
                            vm = vm.to(device, non_blocking=True)
                            vhat, _ = self(vx)
                            vloss = criterion(vhat, vy, weights=vw * vm)
                            v_running += vloss.item()
                    else:
                        for vx, vy in val_loader:
                            vx = vx.to(device, non_blocking=True)
                            vy = vy.to(device, non_blocking=True)
                            vhat, _ = self(vx)
                            vloss = criterion(vhat, vy)
                            v_running += vloss.item()

                val_loss = v_running / max(1, len(val_loader))
                history["val_loss"].append(val_loss)
                scheduler.step(val_loss)

                improved = val_loss + 1e-12 < best_val
                if improved:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(f"Early stopping @ epoch {epoch} (best val={best_val:.6f})")
                        break
            else:
                history["val_loss"].append(None)

            if verbose:
                if val_loss is not None:
                    print(
                        f"Epoch {epoch:03d} | Train {train_loss:.6f} | Val {val_loss:.6f} | "
                        f"Reg {reg_loss:.6f} | LR {optimizer.param_groups[0]['lr']:.2e}"
                    )
                else:
                    print(
                        f"Epoch {epoch:03d} | Train {train_loss:.6f} | "
                        f"Reg {reg_loss:.6f} | LR {optimizer.param_groups[0]['lr']:.2e}"
                    )

        if best_state is not None:
            self.load_state_dict(best_state)

        return history
