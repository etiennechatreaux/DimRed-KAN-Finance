"""
GatedKANAutoencoder : Autoencodeur KAN avec skip linéaire "gated" et contrainte d'orthogonalité.

Fonctionnalités principales :
1. Skip linéaire "gated" qui dose entre voie linéaire et non-linéaire
2. Contrainte d'orthogonalité : ce qui passe dans le KAN AE est orthogonal au skip
3. Paramètre gate appris qui contrôle le mélange linéaire/non-linéaire
4. Régularisation d'orthogonalité pour maintenir l'indépendance des voies

Architecture :
  x -> Projection orthogonale -> [KAN Encoder -> Latent -> KAN Decoder] -> x_kan
  x -> Skip linéaire -> x_skip
  x_hat = gate * x_kan + (1 - gate) * x_skip

Avec contrainte : x_kan ⊥ x_skip (produit scalaire = 0)
"""

from __future__ import annotations
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kan_layers import KANLayer


class OrthogonalProjection(nn.Module):
    """
    Module qui projette l'entrée sur l'espace orthogonal au skip linéaire.
    Utilise la décomposition QR pour maintenir l'orthogonalité.
    """
    def __init__(self, input_dim: int, skip_rank: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.skip_rank = skip_rank or min(input_dim // 4, 64)  # Rang du skip par défaut
        
        # Matrice du skip linéaire (rang réduit pour l'orthogonalité)
        self.skip_basis = nn.Parameter(torch.randn(input_dim, self.skip_rank) * 0.1)
        
        # Normalisation pour maintenir l'orthogonalité
        self.register_buffer('eye', torch.eye(input_dim))
        
    def get_skip_projection(self):
        """Calcule la matrice de projection sur l'espace du skip."""
        # Orthonormalisation de la base du skip
        Q, _ = torch.linalg.qr(self.skip_basis)
        # Projection : P = Q @ Q.T
        P_skip = Q @ Q.transpose(-2, -1)
        return P_skip
    
    def get_orthogonal_projection(self):
        """Calcule la matrice de projection orthogonale au skip."""
        P_skip = self.get_skip_projection()
        # Projection orthogonale : I - P_skip
        P_orth = self.eye - P_skip
        return P_orth
    
    def forward(self, x: torch.Tensor):
        """
        Retourne les projections orthogonales et sur le skip.
        
        Returns:
            x_orth: projection orthogonale (pour le KAN)
            x_skip: projection sur le skip (pour la voie linéaire)
        """
        P_orth = self.get_orthogonal_projection()
        P_skip = self.get_skip_projection()
        
        # Projections
        x_orth = torch.matmul(x, P_orth)  # Pour le KAN
        x_skip = torch.matmul(x, P_skip)  # Pour le skip
        
        return x_orth, x_skip


class GatedMixing(nn.Module):
    """
    Module de mélange gated entre la voie KAN et la voie skip.
    """
    def __init__(self, input_dim: int, gate_init: float = 0.5):
        super().__init__()
        # Gate global appris - Solution plus propre
        import math
        gate_logit_value = math.log(gate_init / (1 - gate_init))  # logit manual
        self.gate_logit = nn.Parameter(torch.tensor(gate_logit_value))
        
        # Gates adaptatifs par dimension (optionnel)
        self.use_adaptive_gate = True
        if self.use_adaptive_gate:
            self.adaptive_gate = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.SiLU(),
                nn.Linear(input_dim // 4, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x_kan: torch.Tensor, x_skip: torch.Tensor, x_original: torch.Tensor):
        """
        Mélange gated entre les deux voies.
        
        Args:
            x_kan: Sortie de la voie KAN
            x_skip: Sortie de la voie skip 
            x_original: Entrée originale (pour le gate adaptatif)
        """
        # Gate global
        global_gate = torch.sigmoid(self.gate_logit)
        
        # Gate adaptatif basé sur l'entrée
        if self.use_adaptive_gate:
            adaptive_gate = self.adaptive_gate(x_original)
            gate = global_gate * adaptive_gate
        else:
            gate = global_gate
        
        # Mélange
        x_mixed = gate * x_kan + (1 - gate) * x_skip
        
        return x_mixed, gate


class GatedKANAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        k: int = 5,
        *,
        hidden_dims: Optional[List[int]] = None,
        basis_type: str = "spline",
        M: int = 16,
        poly_degree: int = 5,
        xmin: float = -3.5,
        xmax: float = 3.5,
        dropout_p: float = 0.05,
        use_silu: bool = True,
        # Gate parameters
        gate_init: float = 0.5,
        skip_rank: int = None,
        # Loss and regularization
        loss_type: str = "mse",
        huber_delta: float = 1.0,
        lambda_alpha: float = 1e-4,
        lambda_group: float = 1e-5,
        lambda_tv: float = 1e-4,
        lambda_poly_decay: float = 0.0,
        lambda_orthogonal: float = 1e-3,  # Pénalité pour maintenir l'orthogonalité
        lambda_gate_reg: float = 1e-4      # Régularisation sur le gate
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
        
        # Paramètres de régularisation
        self.lambda_orthogonal = float(lambda_orthogonal)
        self.lambda_gate_reg = float(lambda_gate_reg)
        
        # ===== PROJECTION ORTHOGONALE =====
        self.projection = OrthogonalProjection(input_dim, skip_rank)
        
        # ===== ENCODEUR KAN (sur la projection orthogonale) =====
        self.encoder_layers = nn.ModuleList()
        self.encoder_activations = nn.ModuleList()
        self.encoder_dropouts = nn.ModuleList()
        
        prev_dim = input_dim  # Même dimension après projection
        for i, hidden_dim in enumerate(self.hidden_dims):
            self.encoder_layers.append(
                KANLayer(
                    prev_dim, hidden_dim,
                    basis_type=basis_type, M=M, poly_degree=poly_degree,
                    xmin=xmin, xmax=xmax,
                    lambda_alpha=lambda_alpha, lambda_group=lambda_group,
                    lambda_tv=lambda_tv, lambda_poly_decay=lambda_poly_decay,
                    use_skip=False  # Pas de skip interne, on gère tout au niveau global
                )
            )
            self.encoder_activations.append(nn.SiLU() if use_silu else nn.Identity())
            self.encoder_dropouts.append(nn.Dropout(dropout_p) if i == 0 else nn.Identity())
            prev_dim = hidden_dim
        
        # Vers l'espace latent
        self.to_latent = nn.Linear(self.hidden_dims[-1], k, bias=True)
        
        # ===== DECODEUR KAN =====
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
                    use_skip=False
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
                use_skip=False
            )
        )
        
        # ===== VOIE SKIP LINÉAIRE =====
        # Simple transformation linéaire sur la projection skip
        self.skip_transform = nn.Linear(input_dim, input_dim, bias=True)
        
        # ===== GATED MIXING =====
        self.gated_mixing = GatedMixing(input_dim, gate_init)
    
    def encode(self, x_orth: torch.Tensor) -> torch.Tensor:
        """Encode la projection orthogonale via les couches KAN."""
        h = x_orth
        for kan_layer, activation, dropout in zip(
            self.encoder_layers, self.encoder_activations, self.encoder_dropouts
        ):
            h = dropout(activation(kan_layer(h)))
        return self.to_latent(h)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Décode l'espace latent via les couches KAN."""
        h = self.from_latent(z)
        for kan_layer, activation, dropout in zip(
            self.decoder_layers[:-1], self.decoder_activations, self.decoder_dropouts
        ):
            h = dropout(activation(kan_layer(h)))
        return self.decoder_layers[-1](h)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass avec projection orthogonale et gated mixing.
        
        Returns:
            x_hat: Reconstruction finale
            z: Représentation latente
            gate: Valeur du gate utilisé
            x_kan: Reconstruction KAN (pour analyse)
            x_skip: Reconstruction skip (pour analyse)
        """
        # 1. Projections orthogonales
        x_orth, x_skip_proj = self.projection(x)
        
        # 2. Voie KAN (sur la projection orthogonale)
        z = self.encode(x_orth)
        x_kan = self.decode(z)
        
        # 3. Voie skip (sur la projection skip)
        x_skip = self.skip_transform(x_skip_proj)
        
        # 4. Mélange gated
        x_hat, gate = self.gated_mixing(x_kan, x_skip, x)
        
        return x_hat, z, gate, x_kan, x_skip
    
    def orthogonality_loss(self, x_kan: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        Calcule la pénalité d'orthogonalité entre les reconstructions KAN et skip.
        """
        # Produit scalaire normalisé
        dot_product = torch.sum(x_kan * x_skip, dim=-1)  # (batch_size,)
        
        # Normes pour normalisation
        norm_kan = torch.norm(x_kan, dim=-1) + 1e-8
        norm_skip = torch.norm(x_skip, dim=-1) + 1e-8
        
        # Cosine similarity (doit être proche de 0 pour l'orthogonalité)
        cos_sim = dot_product / (norm_kan * norm_skip)
        
        # Pénalité : on veut minimiser |cos_sim|
        return torch.mean(cos_sim.pow(2))
    
    def gate_regularization(self) -> torch.Tensor:
        """
        Régularisation sur le gate pour éviter des valeurs extrêmes.
        """
        # Pénalité pour garder le gate proche de 0.5 (équilibre)
        gate_value = torch.sigmoid(self.gated_mixing.gate_logit)
        penalty = (gate_value - 0.5).pow(2)
        return penalty
    
    def regularization(self) -> torch.Tensor:
        """
        Calcule toutes les régularisations.
        """
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Régularisations KAN
        for layer in self.encoder_layers:
            reg_loss += layer.regularization()
        for layer in self.decoder_layers:
            reg_loss += layer.regularization()
        
        # Régularisation du gate
        if self.lambda_gate_reg > 0:
            reg_loss += self.lambda_gate_reg * self.gate_regularization()
        
        return reg_loss
    
    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, x_kan: torch.Tensor, 
                     x_skip: torch.Tensor, lambda_reg: float = 1.0) -> dict:
        """
        Calcule la loss totale avec toutes les composantes.
        """
        # Loss de reconstruction principale
        if self.loss_type == "mse":
            recon_loss = F.mse_loss(x_hat, x)
        elif self.loss_type == "huber":
            recon_loss = F.huber_loss(x_hat, x, delta=self.huber_delta)
        else:
            raise ValueError(f"Type de loss non supporté: {self.loss_type}")
        
        # Régularisations
        reg_loss = self.regularization()
        
        # Pénalité d'orthogonalité
        orth_loss = self.orthogonality_loss(x_kan, x_skip)
        
        # Loss totale
        total_loss = (recon_loss + 
                     lambda_reg * reg_loss + 
                     self.lambda_orthogonal * orth_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'regularization_loss': reg_loss,
            'orthogonality_loss': orth_loss
        }
    
    def get_gate_info(self, x_sample: torch.Tensor = None) -> dict:
        """Retourne des informations sur l'état du gate."""
        with torch.no_grad():
            global_gate = torch.sigmoid(self.gated_mixing.gate_logit).item()
            
            effective_gate = global_gate
            if x_sample is not None and self.gated_mixing.use_adaptive_gate:
                adaptive = self.gated_mixing.adaptive_gate(x_sample).mean().item()
                effective_gate = global_gate * adaptive
            
            return {
                'global_gate_value': global_gate,
                'effective_gate_value': effective_gate,
                'kan_contribution': effective_gate,
                'skip_contribution': 1 - effective_gate,
                'gate_logit': self.gated_mixing.gate_logit.item()
            }
    
    def print_architecture_info(self):
        """Affiche des informations sur l'architecture."""
        gate_info = self.get_gate_info()
        print(f"🚪 Gated KAN Autoencoder - Input: {self.input_dim}, Latent: {self.k}")
        print(f"   📐 Hidden dims: {self.hidden_dims}")
        print(f"   🎛️  Gate: KAN={gate_info['kan_contribution']:.3f} | Skip={gate_info['skip_contribution']:.3f}")
        print(f"   📊 Skip rank: {self.projection.skip_rank}")
        print(f"   🔧 Orthogonality λ: {self.lambda_orthogonal:.2e}")

    def fit(
        self, 
        X_train: torch.Tensor,
        W_train: Optional[torch.Tensor] = None,  # ✅ Poids d'entraînement
        M_train: Optional[torch.Tensor] = None,  # ✅ Masques d'entraînement
        X_val: Optional[torch.Tensor] = None,
        W_val: Optional[torch.Tensor] = None,    # ✅ Poids validation
        M_val: Optional[torch.Tensor] = None,    # ✅ Masques validation
        # Paramètres legacy pour compatibilité (ignorés si X_val fourni)
        validation_split: float = 0.2,
        epochs: int = 100, 
        batch_size: int = 64,
        learning_rate: float = 0.001, 
        weight_decay: float = 1e-5,
        patience: int = 10,
        verbose: bool = True, 
        lambda_reg: float = 1.0,
        use_weighted_loss: bool = True,          # ✅ Activer la loss pondérée
        device: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Entraîne le modèle Gated KAN Autoencoder.
        
        Args:
            X_train: Données d'entraînement (n_samples, input_dim)
            W_train: Poids de fiabilité d'entraînement (n_samples, input_dim)
            M_train: Masques durs d'entraînement (n_samples, input_dim)
            X_val: Données de validation (n_val_samples, input_dim). Si None, utilise validation_split
            W_val: Poids de fiabilité de validation (n_val_samples, input_dim)
            M_val: Masques durs de validation (n_val_samples, input_dim)
            validation_split: Fraction pour la validation (ignoré si X_val fourni)
            epochs: Nombre d'époques maximum
            batch_size: Taille des batches
            learning_rate: Taux d'apprentissage
            weight_decay: Décroissance des poids
            patience: Patience pour early stopping
            verbose: Affichage détaillé
            lambda_reg: Coefficient de régularisation
            use_weighted_loss: Si True, utilise les poids W dans la loss
            device: Device pour l'entraînement
            
        Returns:
            dict: Historique d'entraînement avec métriques détaillées
        """
        import time
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Gestion du split train/val
        if X_val is not None:
            # Utilisation des ensembles fournis (recommandé pour données temporelles)
            if verbose:
                print("🕒 Utilisation d'ensembles train/val séparés")
                print(f"   📊 Train: {X_train.shape[0]} échantillons")
                print(f"   📊 Val: {X_val.shape[0]} échantillons")
                if use_weighted_loss and W_train is not None:
                    print("   🎯 Loss pondérée activée avec poids W et masques M")
        else:
            # Fallback : split aléatoire (pour compatibilité legacy)
            if verbose:
                print(f"⚠️  Utilisation du split aléatoire (validation_split={validation_split})")
            
            n_samples = X_train.size(0)
            n_val = int(n_samples * validation_split)
            indices = torch.randperm(n_samples)
            X_train, X_val = X_train[indices[n_val:]], (X_train[indices[:n_val]] if n_val > 0 else None)

            # Si split aléatoire, splitter aussi W et M
            if W_train is not None:
                W_train, W_val = W_train[indices[n_val:]], (W_train[indices[:n_val]] if n_val > 0 else None)
            if M_train is not None:
                M_train, M_val = M_train[indices[n_val:]], (M_train[indices[:n_val]] if n_val > 0 else None)

        # Création des DataLoaders
        if use_weighted_loss and W_train is not None and M_train is not None:
            # DataLoader avec poids et masques
            train_dataset = torch.utils.data.TensorDataset(X_train, X_train, W_train, M_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                pin_memory=(device.type == "cuda")
            )
            has_weights = True
        else:
            # DataLoader standard (fallback)
            train_dataset = torch.utils.data.TensorDataset(X_train, X_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                pin_memory=(device.type == "cuda")
            )
            has_weights = False
            if verbose and use_weighted_loss:
                print("⚠️  Loss pondérée demandée mais W_train/M_train manquants -> fallback loss standard")

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(1, patience // 2), factor=0.5)

        # Historique étendu pour le modèle gated
        history = {
            'train_loss': [], 'val_loss': [], 'learning_rate': [],
            'reconstruction_loss': [], 'regularization_loss': [], 'orthogonality_loss': [],
            'gate_value': [], 'kan_contribution': [], 'skip_contribution': [],
            'orthogonality_violation': []  # Mesure de violation de l'orthogonalité
        }

        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            self.train()
            epoch_losses = {
                'total': 0.0, 'recon': 0.0, 'reg': 0.0, 'orth': 0.0
            }
            epoch_gate_values = []
            epoch_orth_violations = []

            # Boucle d'entraînement adaptée aux poids/masques
            if has_weights:
                for batch_x, _, batch_w, batch_m in train_loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_w = batch_w.to(device, non_blocking=True)
                    batch_m = batch_m.to(device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Forward pass
                    x_hat, z, gate, x_kan, x_skip = self(batch_x)
                    
                    # Calcul des losses avec pondération
                    losses = self.compute_weighted_loss(batch_x, x_hat, x_kan, x_skip, 
                                                      batch_w, batch_m, lambda_reg)
                    
                    # Backward pass
                    losses['total_loss'].backward()
                    optimizer.step()

                    # Accumulation des métriques
                    epoch_losses['total'] += losses['total_loss'].item()
                    epoch_losses['recon'] += losses['reconstruction_loss'].item()
                    epoch_losses['reg'] += losses['regularization_loss'].item()
                    epoch_losses['orth'] += losses['orthogonality_loss'].item()
                    
                    # Métriques du gate
                    if gate.dim() > 0:
                        epoch_gate_values.append(gate.mean().item())
                    else:
                        epoch_gate_values.append(gate.item())
                    
                    # Violation d'orthogonalité (cosine similarity)
                    with torch.no_grad():
                        cos_sim = torch.nn.functional.cosine_similarity(
                            x_kan.flatten(1), x_skip.flatten(1), dim=1
                        ).abs().mean().item()
                        epoch_orth_violations.append(cos_sim)
            else:
                # Boucle standard (sans poids)
                for batch_x, _ in train_loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Forward pass
                    x_hat, z, gate, x_kan, x_skip = self(batch_x)
                    
                    # Calcul des losses
                    losses = self.compute_loss(batch_x, x_hat, x_kan, x_skip, lambda_reg)
                    
                    # Backward pass
                    losses['total_loss'].backward()
                    optimizer.step()

                    # Accumulation des métriques
                    epoch_losses['total'] += losses['total_loss'].item()
                    epoch_losses['recon'] += losses['reconstruction_loss'].item()
                    epoch_losses['reg'] += losses['regularization_loss'].item()
                    epoch_losses['orth'] += losses['orthogonality_loss'].item()
                    
                    # Métriques du gate
                    if gate.dim() > 0:
                        epoch_gate_values.append(gate.mean().item())
                    else:
                        epoch_gate_values.append(gate.item())
                    
                    # Violation d'orthogonalité (cosine similarity)
                    with torch.no_grad():
                        cos_sim = torch.nn.functional.cosine_similarity(
                            x_kan.flatten(1), x_skip.flatten(1), dim=1
                        ).abs().mean().item()
                        epoch_orth_violations.append(cos_sim)

            # Moyennes par époque
            n_batches = len(train_loader)
            avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
            avg_gate = sum(epoch_gate_values) / len(epoch_gate_values)
            avg_orth_violation = sum(epoch_orth_violations) / len(epoch_orth_violations)
            
            # Mise à jour de l'historique
            history['train_loss'].append(avg_losses['total'])
            history['reconstruction_loss'].append(avg_losses['recon'])
            history['regularization_loss'].append(avg_losses['reg'])
            history['orthogonality_loss'].append(avg_losses['orth'])
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            history['gate_value'].append(avg_gate)
            history['kan_contribution'].append(avg_gate)
            history['skip_contribution'].append(1 - avg_gate)
            history['orthogonality_violation'].append(avg_orth_violation)

            # Validation avec ou sans poids
            val_loss = 0.0
            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    if use_weighted_loss and W_val is not None and M_val is not None:
                        # Validation pondérée
                        x_hat_val, _, _, _, _ = self(X_val.to(device))
                        effective_weights_val = W_val.to(device) * M_val.to(device)
                        
                        if self.loss_type == "mse":
                            # MSE pondérée manuelle
                            diff = (x_hat_val - X_val.to(device)) ** 2
                            weighted_diff = diff * effective_weights_val
                            val_loss = weighted_diff.sum() / effective_weights_val.sum()
                        elif self.loss_type == "huber":
                            # Utiliser WeightedHuberLoss pour validation
                            from .ae_kan import WeightedHuberLoss
                            val_criterion = WeightedHuberLoss(delta=self.huber_delta)
                            val_loss = val_criterion(x_hat_val, X_val.to(device), effective_weights_val).item()
                    else:
                        # Validation standard
                        x_hat_val, _, _, _, _ = self(X_val.to(device))
                        if self.loss_type == "mse":
                            val_loss = torch.nn.functional.mse_loss(x_hat_val, X_val.to(device)).item()
                        elif self.loss_type == "huber":
                            val_loss = torch.nn.functional.huber_loss(x_hat_val, X_val.to(device), delta=self.huber_delta).item()
                
                history['val_loss'].append(val_loss)
                scheduler.step(val_loss)

                # Early stopping
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

            # Affichage verbose
            if verbose:
                val_symbol = "✅" if len(history['val_loss']) > 1 and val_loss < history['val_loss'][-2] else "❌"
                orth_symbol = "🔀" if avg_orth_violation < 0.1 else "⚠️"
                
                print(f"📈 Epoch {epoch+1}/{epochs} | "
                      f"Train: {avg_losses['total']:.6f} | "
                      f"Val: {val_loss:.6f} {val_symbol}")
                print(f"   ↳ Recon: {avg_losses['recon']:.6f} | "
                      f"Reg: {avg_losses['reg']:.6f} | "
                      f"Orth: {avg_losses['orth']:.6f} {orth_symbol}")
                print(f"   🎛️  Gate: {avg_gate:.3f} (KAN: {avg_gate*100:.1f}%, Skip: {(1-avg_gate)*100:.1f}%) | "
                      f"Orth_viol: {avg_orth_violation:.4f}")

        # Restaurer le meilleur modèle
        if 'best_state' in locals():
            self.load_state_dict(best_state)

        history['training_time'] = time.time() - start_time
        return history

    def get_loss_criterion(self, weighted: bool = False):
        """Retourne le critère de loss pour compatibilité avec l'API existante."""
        if self.loss_type == "mse":
            return torch.nn.MSELoss()
        elif self.loss_type == "huber":
            return torch.nn.HuberLoss(delta=self.huber_delta)
        else:
            raise ValueError(f"Type de loss non supporté: {self.loss_type}")

    def plot_training_analysis(self, history: dict, X_sample: torch.Tensor = None, 
                             save_path: str = None, figsize: tuple = (20, 16)) -> None:
        """
        Comprehensive visualization of Gated KAN Autoencoder training and behavior.
        
        Args:
            history: Training history from fit() method
            X_sample: Sample data for detailed analysis (optional)
            save_path: Path to save the figure (optional)
            figsize: Figure size (width, height)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.gridspec import GridSpec
        
        # Create figure with custom layout
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Color scheme [[memory:7550163]]
        colors = {
            'kan': '#FF6B6B',      # Red
            'skip': '#4ECDC4',     # Teal/Green
            'total': '#45B7D1',    # Blue
            'validation': '#96CEB4', # Light green
            'orthogonality': '#FECA57' # Yellow/Orange
        }
        
        epochs = np.arange(1, len(history['train_loss']) + 1)
        
        # ===== 1. Training Loss Evolution =====
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(epochs, history['train_loss'], color=colors['total'], linewidth=2, label='Training Loss')
        if 'val_loss' in history and len(history['val_loss']) > 0:
            ax1.plot(epochs, history['val_loss'], color=colors['validation'], linewidth=2, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # ===== 2. Loss Components Breakdown =====
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.plot(epochs, history['reconstruction_loss'], color=colors['total'], linewidth=2, label='Reconstruction')
        ax2.plot(epochs, history['regularization_loss'], color=colors['kan'], linewidth=2, label='Regularization')
        ax2.plot(epochs, history['orthogonality_loss'], color=colors['orthogonality'], linewidth=2, label='Orthogonality')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Components')
        ax2.set_title('Loss Components Evolution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # ===== 3. Gate Evolution =====
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(epochs, np.array(history['kan_contribution']) * 100, 
                color=colors['kan'], linewidth=3, label='KAN Path (%)')
        ax3.plot(epochs, np.array(history['skip_contribution']) * 100, 
                color=colors['skip'], linewidth=3, label='Skip Path (%)')
        ax3.fill_between(epochs, 0, np.array(history['kan_contribution']) * 100, 
                        color=colors['kan'], alpha=0.3)
        ax3.fill_between(epochs, np.array(history['kan_contribution']) * 100, 100, 
                        color=colors['skip'], alpha=0.3)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Contribution (%)')
        ax3.set_title('Gate Evolution: KAN vs Skip Contributions', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # ===== 4. Orthogonality Violation =====
        ax4 = fig.add_subplot(gs[1, 2:])
        ax4.plot(epochs, history['orthogonality_violation'], 
                color=colors['orthogonality'], linewidth=2, marker='o', markersize=2)
        ax4.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High Violation (0.1)')
        ax4.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Medium Violation (0.05)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Orthogonality Violation')
        ax4.set_title('Orthogonality Constraint Violation', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # ===== 5. Learning Rate Schedule =====
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(epochs, history['learning_rate'], color='purple', linewidth=2)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Learning Rate')
        ax5.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')
        
        # ===== 6. Gate Distribution (Final) =====
        ax6 = fig.add_subplot(gs[2, 1])
        final_kan = history['kan_contribution'][-1] * 100
        final_skip = history['skip_contribution'][-1] * 100
        
        wedges, texts, autotexts = ax6.pie([final_kan, final_skip], 
                                          labels=['KAN Path', 'Skip Path'],
                                          colors=[colors['kan'], colors['skip']],
                                          autopct='%1.1f%%',
                                          startangle=90)
        ax6.set_title('Final Path Distribution', fontsize=12, fontweight='bold')
        
        # ===== 7. Training Efficiency Metrics =====
        ax7 = fig.add_subplot(gs[2, 2:])
        
        # Calculate training efficiency metrics
        loss_improvement = (history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100
        gate_stability = 1 - np.std(history['gate_value'][-10:])  # Stability in last 10 epochs
        orth_final = history['orthogonality_violation'][-1]
        
        metrics = ['Loss Improvement (%)', 'Gate Stability', 'Orthogonality\n(lower=better)']
        values = [loss_improvement, gate_stability * 100, orth_final * 100]
        
        bars = ax7.bar(metrics, values, color=[colors['total'], colors['kan'], colors['orthogonality']])
        ax7.set_ylabel('Score')
        ax7.set_title('Training Efficiency Metrics', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # ===== 8. Sample Analysis (if data provided) =====
        if X_sample is not None:
            ax8 = fig.add_subplot(gs[3, :2])
            
            self.eval()
            with torch.no_grad():
                # Get detailed breakdown
                x_orth, x_skip_proj = self.projection(X_sample)
                z = self.encode(x_orth)
                x_kan = self.decode(z)
                x_skip = self.skip_transform(x_skip_proj)
                x_hat, gate, _, _, _ = self(X_sample)
                
                # Sample a few examples for visualization
                n_samples = min(5, X_sample.shape[0])
                sample_idx = np.random.choice(X_sample.shape[0], n_samples, replace=False)
                
                # Plot reconstruction comparison
                x_orig = X_sample[sample_idx].cpu().numpy()
                x_recon = x_hat[sample_idx].cpu().numpy()
                x_kan_recon = x_kan[sample_idx].cpu().numpy()
                x_skip_recon = x_skip[sample_idx].cpu().numpy()
                
                for i in range(n_samples):
                    offset = i * 0.1
                    ax8.plot(x_orig[i] + offset, color='black', linewidth=2, alpha=0.7, label='Original' if i == 0 else "")
                    ax8.plot(x_recon[i] + offset, color=colors['total'], linewidth=1.5, label='Reconstruction' if i == 0 else "")
                    ax8.plot(x_kan_recon[i] + offset, color=colors['kan'], linewidth=1, alpha=0.7, label='KAN Only' if i == 0 else "")
                    ax8.plot(x_skip_recon[i] + offset, color=colors['skip'], linewidth=1, alpha=0.7, label='Skip Only' if i == 0 else "")
                
                ax8.set_xlabel('Feature Index')
                ax8.set_ylabel('Value')
                ax8.set_title('Sample Reconstructions Comparison', fontsize=12, fontweight='bold')
                ax8.legend()
                ax8.grid(True, alpha=0.3)
        
        # ===== 9. Model Architecture Summary =====
        ax9 = fig.add_subplot(gs[3, 2:])
        ax9.axis('off')
        
        # Get model info
        gate_info = self.get_gate_info()
        n_params = sum(p.numel() for p in self.parameters())
        
        # Create text summary
        summary_text = f"""
        MODEL ARCHITECTURE SUMMARY
        
        Input Dimension: {self.input_dim}
        Latent Dimension: {self.k}
        Hidden Layers: {self.hidden_dims}
        
        Total Parameters: {n_params:,}
        Training Time: {history.get('training_time', 0):.1f}s
        
        FINAL STATE:
        Gate Value: {gate_info['kan_contribution']:.3f}
        KAN Contribution: {gate_info['kan_contribution']*100:.1f}%
        Skip Contribution: {gate_info['skip_contribution']*100:.1f}%
        
        Orthogonality Violation: {history['orthogonality_violation'][-1]:.4f}
        Final Training Loss: {history['train_loss'][-1]:.6f}
        Final Validation Loss: {history.get('val_loss', [0])[-1]:.6f}
        
        Skip Rank: {self.projection.skip_rank}
        Basis Type: {self.encoder_layers[0].basis_type}
        """
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Main title
        fig.suptitle('Gated KAN Autoencoder: Training Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"📊 Training analysis saved to: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("🚪 GATED KAN AUTOENCODER - TRAINING SUMMARY")
        print("="*60)
        print(f"📈 Loss Improvement: {loss_improvement:.2f}%")
        print(f"🎛️  Final Gate: KAN={gate_info['kan_contribution']*100:.1f}% | Skip={gate_info['skip_contribution']*100:.1f}%")
        print(f"🔀 Orthogonality Violation: {history['orthogonality_violation'][-1]:.4f}")
        print(f"⏱️  Training Time: {history.get('training_time', 0):.1f}s")
        print(f"🎯 Best Validation Loss: {min(history.get('val_loss', [float('inf')])):.6f}")
        print("="*60)

    def analyze_latent_factors(self, X: torch.Tensor, y_labels: torch.Tensor = None, 
                             sector_labels: list = None, save_path: str = None, 
                             figsize: tuple = (20, 12)) -> dict:
        """
        Comprehensive analysis and visualization of latent factors.
        
        Args:
            X: Input data (n_samples, input_dim)
            y_labels: Optional labels for coloring (e.g., time periods, sectors)
            sector_labels: Optional sector names for interpretation
            save_path: Path to save the figure (optional)
            figsize: Figure size (width, height)
            
        Returns:
            dict: Latent analysis results including embeddings, correlations, etc.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.gridspec import GridSpec
        import seaborn as sns
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import pandas as pd
        
        self.eval()
        device = next(self.parameters()).device
        X = X.to(device)
        
        with torch.no_grad():
            # Extract latent representations
            x_orth, x_skip_proj = self.projection(X)
            z = self.encode(x_orth)  # Latent factors
            x_hat, _, gate, x_kan, x_skip = self(X)
            
            # Convert to numpy for analysis
            z_np = z.cpu().numpy()
            X_np = X.cpu().numpy()
            x_hat_np = x_hat.cpu().numpy()
            x_kan_np = x_kan.cpu().numpy()
            x_skip_np = x_skip.cpu().numpy()
            gate_np = gate.cpu().numpy() if gate.dim() > 0 else np.full(X_np.shape[0], gate.item())
        
        # Color scheme [[memory:7550163]]
        colors = {
            'primary': '#FF6B6B',      # Red
            'secondary': '#4ECDC4',    # Teal/Green  
            'accent': '#45B7D1',       # Blue
            'highlight': '#FECA57',    # Yellow
            'dark': '#2C3E50'          # Dark blue
        }
        
        # Create comprehensive figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # ===== 1. Latent Space Distribution =====
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Plot distribution of each latent dimension
        for i in range(min(z_np.shape[1], 8)):  # Show max 8 dimensions
            ax1.hist(z_np[:, i], bins=30, alpha=0.7, 
                    label=f'Z{i+1}', density=True)
        
        ax1.set_xlabel('Latent Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Latent Factors Distribution', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # ===== 2. Latent Correlation Matrix =====
        ax2 = fig.add_subplot(gs[0, 2:])
        
        corr_matrix = np.corrcoef(z_np.T)
        im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Add correlation values
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                text = ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white",
                               fontsize=8)
        
        ax2.set_title('Latent Factors Correlation Matrix', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Latent Factor')
        ax2.set_ylabel('Latent Factor')
        ax2.set_xticks(range(z_np.shape[1]))
        ax2.set_yticks(range(z_np.shape[1]))
        ax2.set_xticklabels([f'Z{i+1}' for i in range(z_np.shape[1])])
        ax2.set_yticklabels([f'Z{i+1}' for i in range(z_np.shape[1])])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Correlation')
        
        # ===== 3. 2D Latent Space Visualization =====
        ax3 = fig.add_subplot(gs[1, 0])
        
        if z_np.shape[1] >= 2:
            scatter = ax3.scatter(z_np[:, 0], z_np[:, 1], 
                                c=gate_np, cmap='viridis', alpha=0.6, s=20)
            ax3.set_xlabel('Latent Factor 1')
            ax3.set_ylabel('Latent Factor 2') 
            ax3.set_title('2D Latent Space\n(colored by gate value)', fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax3, label='Gate Value')
        else:
            ax3.text(0.5, 0.5, 'Need at least 2\nlatent dimensions', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('2D Latent Space', fontsize=12, fontweight='bold')
        
        ax3.grid(True, alpha=0.3)
        
        # ===== 4. PCA of Latent Space =====
        ax4 = fig.add_subplot(gs[1, 1])
        
        if z_np.shape[1] > 2:
            pca = PCA(n_components=2)
            z_pca = pca.fit_transform(z_np)
            
            scatter = ax4.scatter(z_pca[:, 0], z_pca[:, 1], 
                                c=gate_np, cmap='plasma', alpha=0.6, s=20)
            ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax4.set_title('PCA of Latent Space', fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax4, label='Gate Value')
        else:
            ax4.text(0.5, 0.5, 'PCA requires more\nthan 2 dimensions', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('PCA of Latent Space', fontsize=12, fontweight='bold')
        
        ax4.grid(True, alpha=0.3)
        
        # ===== 5. Latent Factor Statistics =====
        ax5 = fig.add_subplot(gs[1, 2])
        
        latent_stats = {
            'mean': np.mean(z_np, axis=0),
            'std': np.std(z_np, axis=0),
            'min': np.min(z_np, axis=0),
            'max': np.max(z_np, axis=0)
        }
        
        x_pos = np.arange(z_np.shape[1])
        width = 0.2
        
        ax5.bar(x_pos - width*1.5, latent_stats['mean'], width, 
               label='Mean', color=colors['primary'], alpha=0.8)
        ax5.bar(x_pos - width*0.5, latent_stats['std'], width, 
               label='Std', color=colors['secondary'], alpha=0.8)
        ax5.bar(x_pos + width*0.5, latent_stats['min'], width, 
               label='Min', color=colors['accent'], alpha=0.8)
        ax5.bar(x_pos + width*1.5, latent_stats['max'], width, 
               label='Max', color=colors['highlight'], alpha=0.8)
        
        ax5.set_xlabel('Latent Factor')
        ax5.set_ylabel('Value')
        ax5.set_title('Latent Factors Statistics', fontsize=12, fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([f'Z{i+1}' for i in range(z_np.shape[1])])
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # ===== 6. Reconstruction Quality by Latent Magnitude =====
        ax6 = fig.add_subplot(gs[1, 3])
        
        # Calculate reconstruction error per sample
        recon_error = np.mean((X_np - x_hat_np)**2, axis=1)
        latent_magnitude = np.linalg.norm(z_np, axis=1)
        
        scatter = ax6.scatter(latent_magnitude, recon_error, 
                            c=gate_np, cmap='coolwarm', alpha=0.6, s=20)
        ax6.set_xlabel('Latent Magnitude ||z||')
        ax6.set_ylabel('Reconstruction Error')
        ax6.set_title('Reconstruction Quality\nvs Latent Magnitude', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax6, label='Gate Value')
        ax6.grid(True, alpha=0.3)
        
        # ===== 7. Latent Factors Evolution (if time series) =====
        ax7 = fig.add_subplot(gs[2, :2])
        
        # Show evolution of latent factors over samples (assuming temporal order)
        sample_indices = np.arange(min(500, z_np.shape[0]))  # Show first 500 samples
        
        for i in range(min(z_np.shape[1], 5)):  # Show max 5 factors
            ax7.plot(sample_indices, z_np[sample_indices, i], 
                    label=f'Factor {i+1}', alpha=0.8, linewidth=1.5)
        
        ax7.set_xlabel('Sample Index (Time Order)')
        ax7.set_ylabel('Latent Factor Value')
        ax7.set_title('Latent Factors Evolution Over Time', fontsize=12, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # ===== 8. Path Contribution Analysis =====
        ax8 = fig.add_subplot(gs[2, 2:])
        
        # Analyze which samples use more KAN vs Skip
        kan_users = gate_np > 0.5
        skip_users = gate_np <= 0.5
        
        if np.any(kan_users) and np.any(skip_users):
            # Compare latent spaces for KAN vs Skip users
            z_kan_users = z_np[kan_users]
            z_skip_users = z_np[skip_users]
            
            # Calculate mean latent values for each group
            if len(z_kan_users) > 0:
                mean_kan = np.mean(z_kan_users, axis=0)
            else:
                mean_kan = np.zeros(z_np.shape[1])
                
            if len(z_skip_users) > 0:
                mean_skip = np.mean(z_skip_users, axis=0)
            else:
                mean_skip = np.zeros(z_np.shape[1])
            
            x_pos = np.arange(z_np.shape[1])
            width = 0.35
            
            ax8.bar(x_pos - width/2, mean_kan, width, 
                   label=f'KAN Users (n={len(z_kan_users)})', 
                   color=colors['primary'], alpha=0.8)
            ax8.bar(x_pos + width/2, mean_skip, width, 
                   label=f'Skip Users (n={len(z_skip_users)})', 
                   color=colors['secondary'], alpha=0.8)
            
            ax8.set_xlabel('Latent Factor')
            ax8.set_ylabel('Mean Value')
            ax8.set_title('Latent Factors: KAN vs Skip Path Users', fontsize=12, fontweight='bold')
            ax8.set_xticks(x_pos)
            ax8.set_xticklabels([f'Z{i+1}' for i in range(z_np.shape[1])])
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        else:
            # Show overall gate distribution
            ax8.hist(gate_np, bins=30, color=colors['accent'], alpha=0.7)
            ax8.set_xlabel('Gate Value')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Gate Value Distribution', fontsize=12, fontweight='bold')
            ax8.grid(True, alpha=0.3)
        
        # Main title
        fig.suptitle('Gated KAN Autoencoder: Latent Factors Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"🧠 Latent factors analysis saved to: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        # Prepare analysis results
        analysis_results = {
            'latent_embeddings': z_np,
            'reconstruction_errors': np.mean((X_np - x_hat_np)**2, axis=1),
            'gate_values': gate_np,
            'latent_statistics': latent_stats,
            'correlation_matrix': corr_matrix,
            'kan_path_samples': np.where(kan_users)[0] if np.any(kan_users) else np.array([]),
            'skip_path_samples': np.where(skip_users)[0] if np.any(skip_users) else np.array([]),
        }
        
        # Add PCA results if applicable
        if z_np.shape[1] > 2:
            analysis_results['pca_embeddings'] = z_pca
            analysis_results['pca_explained_variance'] = pca.explained_variance_ratio_
        
        # Print summary
        print("\n" + "="*60)
        print("🧠 LATENT FACTORS ANALYSIS SUMMARY")
        print("="*60)
        print(f"📐 Latent Dimensions: {z_np.shape[1]}")
        print(f"📊 Number of Samples: {z_np.shape[0]}")
        print(f"🎛️  Average Gate Value: {np.mean(gate_np):.3f}")
        print(f"🔀 KAN Path Users: {np.sum(kan_users)} ({np.mean(kan_users)*100:.1f}%)")
        print(f"🔀 Skip Path Users: {np.sum(skip_users)} ({np.mean(skip_users)*100:.1f}%)")
        print(f"📈 Latent Correlation Range: [{np.min(corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]):.3f}, {np.max(corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]):.3f}]")
        print(f"🎯 Average Reconstruction Error: {np.mean(analysis_results['reconstruction_errors']):.6f}")
        print("="*60)
        
        return analysis_results

    def get_latent_representation(self, X: torch.Tensor) -> torch.Tensor:
        """
        Simple function to extract latent representation.
        
        Args:
            X: Input data
            
        Returns:
            torch.Tensor: Latent representation
        """
        self.eval()
        device = next(self.parameters()).device
        X = X.to(device)
        
        with torch.no_grad():
            x_orth, _ = self.projection(X)
            z = self.encode(x_orth)
        
        return z

    def compute_weighted_loss(self, x: torch.Tensor, x_hat: torch.Tensor, x_kan: torch.Tensor, 
                         x_skip: torch.Tensor, weights: torch.Tensor, masks: torch.Tensor,
                         lambda_reg: float = 1.0) -> dict:
        """
        Calcule la loss totale avec pondération par les poids de fiabilité.
        
        Args:
            x: Données originales
            x_hat: Reconstruction
            x_kan: Composante KAN
            x_skip: Composante skip
            weights: Poids de fiabilité (0-1)
            masks: Masques durs (0/1)
            lambda_reg: Coefficient de régularisation
        """
        # Poids effectifs : combine poids soft et masques durs
        effective_weights = weights * masks
        
        # Loss de reconstruction pondérée
        if self.loss_type == "mse":
            # MSE pondérée manuelle
            diff_squared = (x_hat - x) ** 2
            weighted_diff = diff_squared * effective_weights
            # Normalisation par la somme des poids pour éviter biais
            recon_loss = weighted_diff.sum() / (effective_weights.sum() + 1e-8)
        elif self.loss_type == "huber":
            # Utiliser WeightedHuberLoss
            from .ae_kan import WeightedHuberLoss
            weighted_criterion = WeightedHuberLoss(delta=self.huber_delta)
            recon_loss = weighted_criterion(x_hat, x, weights=effective_weights)
        else:
            raise ValueError(f"Type de loss non supporté: {self.loss_type}")
        
        # Régularisations (identiques)
        reg_loss = self.regularization()
        
        # Pénalité d'orthogonalité (identique)
        orth_loss = self.orthogonality_loss(x_kan, x_skip)
        
        # Loss totale
        total_loss = (recon_loss + 
                 lambda_reg * reg_loss + 
                 self.lambda_orthogonal * orth_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'regularization_loss': reg_loss,
            'orthogonality_loss': orth_loss
        }
