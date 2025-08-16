# -*- coding: utf-8 -*-
"""
Autoencodeur KAN (encodeur/décodeur basés sur KANLayer) avec choix de base :
- "spline" (par défaut, M=16)
- "poly" (degré p ; stabilisé par clipping/normalisation)

Architecture flexible :
  Enc: KAN(N,h1) → (SiLU?) → Dropout → KAN(h1,h2) → (SiLU?) → ... → Linear(hn→k)
  Dec: Linear(k→hn) → KAN(hn,h(n-1)) → (SiLU?) → Dropout → ... → KAN(h1→N)
  
où h1, h2, ..., hn sont définies par le paramètre hidden_dims
"""

from __future__ import annotations
from typing import Optional, List
import torch
import torch.nn as nn
from .kan_layers import KANLayer


class KANAutoencoder(nn.Module):
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
        use_silu: bool = False,
        # régularisations KAN
        lambda_alpha: float = 1e-4,
        lambda_group: float = 1e-5,
        lambda_tv: float = 1e-4,
        lambda_poly_decay: float = 0.0,
        # -------- options skip linéaire (propagées à chaque KANLayer) --------
        use_skip: bool = True,
        skip_init: str = "zeros",                  # "zeros" | "xavier" | "identity"(si in==out)
        skip_gain: float = 1.0,
        lambda_skip_l2: float = 0.0
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.k = int(k)

        if hidden_dims is None:
            hidden_dims = [128, 32]
        self.hidden_dims = list(hidden_dims)
        self.use_silu = use_silu

        # Mémorise les options skip pour log/debug
        self._skip_opts = dict(
            use_skip=use_skip,
            skip_init=skip_init,
            skip_gain=skip_gain,
            lambda_skip_l2=lambda_skip_l2
        )

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
                    # skip options propagées
                    use_skip=use_skip,
                    skip_init=skip_init,
                    skip_gain=skip_gain,
                    lambda_skip_l2=lambda_skip_l2
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
                    # skip options propagées
                    use_skip=use_skip,
                    skip_init=skip_init,
                    skip_gain=skip_gain,
                    lambda_skip_l2=lambda_skip_l2
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
                # skip options propagées
                use_skip=use_skip,
                skip_init=skip_init,
                skip_gain=skip_gain,
                lambda_skip_l2=lambda_skip_l2
            )
        )

    # ---------- API ----------
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N) -> z: (B, k)
        """
        h = x
        # Passage à travers toutes les couches de l'encodeur
        for i, (kan_layer, activation, dropout) in enumerate(zip(
            self.encoder_layers, self.encoder_activations, self.encoder_dropouts
        )):
            h = kan_layer(h)
            h = activation(h)
            h = dropout(h)
        
        # Projection vers l'espace latent
        z = self.to_latent(h)
        return z

    def decode(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        z: (B, k) -> x_hat: (B, N)
        """
        # Projection depuis l'espace latent
        h = self.from_latent(z)
        
        # Passage à travers toutes les couches du décodeur
        for i, (kan_layer, activation, dropout) in enumerate(zip(
            self.decoder_layers[:-1], self.decoder_activations, self.decoder_dropouts
        )):
            h = kan_layer(h)
            h = activation(h)
            h = dropout(h)
            
        # Dernière couche sans activation
        h = self.decoder_layers[-1](h)
        return h
    
    def encode_batched(self, x: torch.Tensor, batch_size: int = 512, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode large datasets in batches to avoid memory issues.
        x: (N, input_dim) -> z: (N, latent_dim)
        """
        self.eval()
        encoded_batches = []
        
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i+batch_size]
                batch_mask = mask[i:i+batch_size] if mask is not None else None
                batch_encoded = self.encode(batch_x, batch_mask)
                encoded_batches.append(batch_encoded)
        
        return torch.cat(encoded_batches, dim=0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        z = self.encode(x, mask)
        x_hat = self.decode(z, mask)
        return x_hat, z

    def regularization(self) -> torch.Tensor:
        """
        Somme des régularisations de toutes les couches KAN (encodeur + décodeur).
        """
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Régularisation des couches de l'encodeur
        for layer in self.encoder_layers:
            reg_loss += layer.regularization()
        
        # Régularisation des couches du décodeur
        for layer in self.decoder_layers:
            reg_loss += layer.regularization()
        
        return reg_loss

    def set_regularization(
        self,
        lambda_alpha: Optional[float] = None,
        lambda_group: Optional[float] = None,
        lambda_tv: Optional[float] = None,
        lambda_poly_decay: Optional[float] = None,
    ) -> None:
        # Mise à jour de la régularisation pour toutes les couches KAN
        all_kan_layers = list(self.encoder_layers) + list(self.decoder_layers)
        for layer in all_kan_layers:
            layer.set_regularization(lambda_alpha, lambda_group, lambda_tv, lambda_poly_decay)
        
    def fit(
        self,
        X: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        validation_split: float = 0.2,
        patience: int = 10,
        verbose: bool = True,
        lambda_reg: float = 1.0,
        device: Optional[torch.device] = None
    ) -> dict:
        import time
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # --- split sur CPU (ne pas faire X.to(device))
        n_samples = X.size(0)
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)
        X_train = X[indices[n_val:]]
        X_val = X[indices[:n_val]] if n_val > 0 else None

        # --- DataLoader train
        train_dataset = torch.utils.data.TensorDataset(X_train, X_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            pin_memory=(device.type == "cuda"),
            num_workers=4 if device.type == "cuda" else 0,
            persistent_workers=(device.type == "cuda")
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=max(1, patience // 2), factor=0.5
        )

        history = {'train_loss': [], 'val_loss': [], 'learning_rate': [], 'regularization': []}
        best_val_loss = float('inf')
        patience_counter = 0

        if verbose:
            print("=" * 90)
            print("🚀 ENTRAÎNEMENT KAN AUTOENCODER (ae_kan)")
            print("=" * 90)
            print(f"📊 Données: {X_train.size(0)} train, {0 if X_val is None else X_val.size(0)} val")
            arch_str = f"{self.input_dim}"
            for d in self.hidden_dims: arch_str += f" -> {d}"
            arch_str += f" -> {self.k}"
            for d in reversed(self.hidden_dims): arch_str += f" -> {d}"
            arch_str += f" -> {self.input_dim}"
            print(f"🏗️  Architecture: {arch_str}")
            print(f"🔧 Device: {device}")
            print(f"⚙️  Paramètres: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
            print(f"🎯 Régularisation: λ={lambda_reg}")
            tot = sum(p.numel() for p in self.parameters())
            trn = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"🔢 Paramètres: {tot:,} total, {trn:,} entraînables")
            print("-" * 90)

        t0 = time.time()
        for epoch in range(epochs):
            # ---- TRAIN
            self.train()
            train_loss = 0.0
            reg_loss = 0.0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                x_reconstructed, _ = self(batch_x)
                mse_loss = criterion(x_reconstructed, batch_x)
                reg_term = self.regularization()
                loss = mse_loss + lambda_reg * reg_term
                loss.backward()
                optimizer.step()
                train_loss += mse_loss.item()
                reg_loss += reg_term.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_reg_loss = reg_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            history['regularization'].append(avg_reg_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # ---- VALIDATION (batchée + to(device))
            if X_val is not None:
                self.eval()
                val_dataset = torch.utils.data.TensorDataset(X_val, X_val)
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=min(batch_size, 512),
                    shuffle=False,
                    pin_memory=(device.type == "cuda"),
                    num_workers=2 if device.type == "cuda" else 0
                )
                with torch.no_grad():
                    tot_v, cnt_v = 0.0, 0
                    for vb, _ in val_loader:
                        vb = vb.to(device, non_blocking=True)
                        vh, _ = self(vb)
                        tot_v += criterion(vh, vb).item() * vb.size(0)
                        cnt_v += vb.size(0)
                    val_loss = tot_v / max(1, cnt_v)

                history['val_loss'].append(val_loss)
                scheduler.step(val_loss)

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

            # ---- LOG
            if verbose:
                elapsed = time.time() - t0
                eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
                if X_val is not None:
                    improvement = "✅" if val_loss < best_val_loss + 1e-9 else "⚠️"
                    print(f"📈 Époque {epoch+1:3d}/{epochs} | "
                        f"Train: {avg_train_loss:.6f} | "
                        f"Val: {val_loss:.6f} {improvement} | "
                        f"Reg: {avg_reg_loss:.6f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                        f"⏱️  {elapsed:.1f}s (ETA: {eta:.1f}s)")
                else:
                    print(f"📈 Époque {epoch+1:3d}/{epochs} | "
                        f"Train: {avg_train_loss:.6f} | "
                        f"Reg: {avg_reg_loss:.6f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                        f"⏱️  {elapsed:.1f}s (ETA: {eta:.1f}s)")

        # ---- restore best
        if 'best_state' in locals():
            self.load_state_dict(best_state)

        # ---- Évaluation finale par batch (même device)
        self.eval()
        with torch.no_grad():
            total_loss = 0.0
            num_samples = 0
            batch_size_eval = min(batch_size, 512)
            for i in range(0, len(X), batch_size_eval):
                batch_X = X[i:i+batch_size_eval].to(device, non_blocking=True)
                batch_reconstructed, _ = self(batch_X)
                batch_loss = criterion(batch_reconstructed, batch_X)
                total_loss += batch_loss.item() * batch_X.size(0)
                num_samples += batch_X.size(0)
            final_loss = total_loss / max(1, num_samples)
            final_reg = self.regularization().item()

        if verbose:
            elapsed = time.time() - t0
            print("-" * 90)
            print(f"🎉 ENTRAÎNEMENT TERMINÉ EN {elapsed:.2f}s")
            print("-" * 90)
            print("📊 RÉSULTATS FINAUX:")
            print(f"   • MSE finale: {final_loss:.6f}")
            print(f"   • Régularisation: {final_reg:.6f}")
            print(f"   • Loss totale: {final_loss + lambda_reg * final_reg:.6f}")
            if history['val_loss']:
                best_epoch = int(torch.tensor(history['val_loss']).argmin().item() + 1)
                print(f"   • Meilleure époque: {best_epoch}")
                print(f"   • Meilleure val loss: {min(history['val_loss']):.6f}")

        history['training_time'] = time.time() - t0
        history['final_loss'] = final_loss
        return history