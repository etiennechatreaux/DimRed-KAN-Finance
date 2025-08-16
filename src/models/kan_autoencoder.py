import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any
import time
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error


class KANLayer(nn.Module):
    """
    Couche KAN utilisant des B-splines comme fonctions d'activation apprises.
    
    Args:
        input_dim (int): Dimension d'entrée
        output_dim (int): Dimension de sortie
        grid_size (int): Nombre de points de grille pour les splines
        spline_order (int): Ordre des splines B (degré + 1)
        noise_scale (float): Échelle du bruit d'initialisation
        base_activation (str): Fonction d'activation de base ('silu', 'relu', 'tanh')
        grid_eps (float): Epsilon pour l'extension de la grille
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        grid_size: int = 5,
        spline_order: int = 3,
        noise_scale: float = 0.1,
        base_activation: str = 'silu',
        grid_eps: float = 0.02
    ):
        super(KANLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.noise_scale = noise_scale
        self.grid_eps = grid_eps
        
        # Grille de base pour les splines
        h = (2 * grid_eps) / grid_size
        grid = torch.arange(-grid_eps, grid_eps + h, h)[:-1]
        self.register_buffer('grid', grid)
        
        # Coefficients des splines (paramètres apprenables)
        # shape: [input_dim, output_dim, grid_size + spline_order]
        self.spline_weight = nn.Parameter(
            torch.randn(input_dim, output_dim, grid_size + spline_order) * noise_scale
        )
        
        # Fonction d'activation de base
        if base_activation == 'silu':
            self.base_activation = nn.SiLU()
        elif base_activation == 'relu':
            self.base_activation = nn.ReLU()
        elif base_activation == 'tanh':
            self.base_activation = nn.Tanh()
        else:
            raise ValueError(f"Activation non supportée: {base_activation}")
        
        # Poids de base (comme dans un MLP classique) -> [in_dim, out_dim]
        self.base_weight = nn.Parameter(torch.randn(input_dim, output_dim) * noise_scale)
        
        # >>> FIX BROADCASTING <<<
        # Facteurs d'échelle pour combiner base et spline:
        # On veut des shapes diffusables sur [batch, out_dim], PAS [in_dim, out_dim].
        # On choisit (1, out_dim) pour base/spline et un scalaire pour le bruit.
        self.scale_base   = nn.Parameter(torch.ones(1, output_dim))
        self.scale_spline = nn.Parameter(torch.ones(1, output_dim))
        self.scale_noise  = nn.Parameter(torch.tensor(0.01))  # scalaire

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcul des fonctions de base B-spline (approximation gaussienne).
        
        Args:
            x (torch.Tensor): Tensor d'entrée de forme (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Valeurs des B-splines de forme (batch_size, input_dim, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.input_dim
        
        grid = self.grid
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        
        # Extension de la grille pour les splines
        grid_extended = torch.cat([
            grid[:1] - (grid[1] - grid[0]) * torch.arange(self.spline_order, 0, -1, device=grid.device),
            grid,
            grid[-1:] + (grid[-1] - grid[-2]) * torch.arange(1, self.spline_order + 1, device=grid.device)
        ])
        
        # Approximation simplifiée par gaussiennes centrées sur grid_extended
        bases_tensor = torch.zeros(x.size(0), x.size(1), self.grid_size + self.spline_order, device=x.device)
        width = 2.0 / self.grid_size
        xs = x.squeeze(-1)  # [batch, in]
        
        for i in range(self.grid_size + self.spline_order):
            center = grid_extended[i] if i < len(grid_extended) else grid_extended[-1]
            bases_tensor[:, :, i] = torch.exp(-0.5 * ((xs - center) / width) ** 2)
        
        return bases_tensor  # [batch, in, g]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la couche KAN.
        
        Args:
            x (torch.Tensor): Tensor d'entrée de forme (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Tensor de sortie de forme (batch_size, output_dim)
        """
        assert x.size(1) == self.input_dim
        
        # Partie base (MLP)
        base_output = self.base_activation(x @ self.base_weight)  # [batch, out]
        
        # Partie spline
        spline_bases = self.b_splines(x)  # [batch, in, g]
        # spline_weight: [in, out, g]
        
        # Vectorisation avec einsum (à vérifier!!)
        # Somme sur in et g -> [batch, out]
        
        spline_output = torch.einsum('big,iog->bo', spline_bases, self.spline_weight)
        
        # --------------------------------------------------------------------
        # Double boucle for échangée avec einsum ci-dessus
        # batch_size = x.size(0)
        # spline_output = torch.zeros(batch_size, self.output_dim, device=x.device)
        # for i in range(self.input_dim):
        #     for j in range(self.output_dim):
        #         spline_output[:, j] += torch.sum(
        #             spline_bases[:, i, :] * self.spline_weight[i, j, :], dim=1
        #         )
        # -------------------------------------------------------------------
        
        # Sanity checks (une fois pour debug)
        # assert base_output.shape == (x.size(0), self.output_dim)
        # assert spline_output.shape == (x.size(0), self.output_dim)
        
        # Combinaison base + spline + bruit
        output = (
            base_output   * self.scale_base +
            spline_output * self.scale_spline +
            torch.randn_like(base_output) * self.scale_noise
        )
        
        return output


class KANAutoencoder(nn.Module):
    """
    Autoencodeur basé sur les Kolmogorov-Arnold Networks avec architecture bottleneck.
    
    Args:
        input_dim (int): Dimension d'entrée
        latent_dim (int): Dimension du bottleneck (espace latent)
        hidden_dims (List[int], optional): Dimensions des couches cachées
        grid_size (int): Taille de la grille pour les splines
        spline_order (int): Ordre des splines B
        noise_scale (float): Échelle du bruit d'initialisation
        base_activation (str): Fonction d'activation de base
        dropout_rate (float): Taux de dropout
        use_batch_norm (bool): Utiliser la normalisation par batch
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Optional[List[int]] = None,
        grid_size: int = 5,
        spline_order: int = 3,
        noise_scale: float = 0.1,
        base_activation: str = 'silu',
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        **kwargs
    ):
        super(KANAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [max(latent_dim * 2, 64)]
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.noise_scale = noise_scale
        self.base_activation = base_activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Construction de l'encodeur
        encoder_dims = [input_dim] + self.hidden_dims + [latent_dim]
        print("Encoder dims: ", encoder_dims)
        self.encoder_layers = nn.ModuleList()
        
        for i in range(len(encoder_dims) - 1):
            self.encoder_layers.append(
                KANLayer(
                    encoder_dims[i], 
                    encoder_dims[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    noise_scale=noise_scale,
                    base_activation=base_activation
                )
            )
            
            if use_batch_norm and i < len(encoder_dims) - 2:
                self.encoder_layers.append(nn.BatchNorm1d(encoder_dims[i + 1]))
            
            if dropout_rate > 0 and i < len(encoder_dims) - 2:
                self.encoder_layers.append(nn.Dropout(dropout_rate))
        
        # Construction du décodeur (symétrique)
        decoder_dims = [latent_dim] + self.hidden_dims[::-1] + [input_dim]
        print("Decoder dims: ", decoder_dims)
        self.decoder_layers = nn.ModuleList()
        
        for i in range(len(decoder_dims) - 1):
            self.decoder_layers.append(
                KANLayer(
                    decoder_dims[i], 
                    decoder_dims[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    noise_scale=noise_scale,
                    base_activation=base_activation
                )
            )
            
            if use_batch_norm and i < len(decoder_dims) - 2:
                self.decoder_layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
            
            if dropout_rate > 0 and i < len(decoder_dims) - 2:
                self.decoder_layers.append(nn.Dropout(dropout_rate))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode l'entrée vers l'espace latent.
        """
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
        return h
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Décode depuis l'espace latent vers l'espace d'origine.
        """
        h = z
        for layer in self.decoder_layers:
            h = layer(h)
        return h
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass complet de l'autoencodeur.
        """
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtient la représentation latente sans reconstruction.
        """
        self.eval()
        with torch.no_grad():
            return self.encode(x)
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruit l'entrée.
        """
        self.eval()
        with torch.no_grad():
            z = self.encode(x)
            return self.decode(z)
    
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
        plot_losses: bool = True
    ) -> Dict[str, Any]:
        """
        Entraîne l'autoencodeur KAN.
        """
        
        # Division train/validation
        n_samples = X.size(0)
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)
        
        X_train = X[indices[n_val:]]
        X_val = X[indices[:n_val]] if n_val > 0 else None
        
        # DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train, X_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Optimiseur et critère
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(1, patience//2), factor=0.5)
        
        # Historique
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        if verbose:
            print("=" * 80)
            print("🚀 DÉBUT DE L'ENTRAÎNEMENT KAN AUTOENCODER")
            print("=" * 80)
            print(f"📊 Données d'entraînement: {X_train.size(0)} échantillons")
            if X_val is not None:
                print(f"📊 Données de validation: {X_val.size(0)} échantillons")
            print(f"🏗️  Architecture: {self.input_dim} -> {' -> '.join(map(str, self.hidden_dims))} -> {self.latent_dim}")
            print(f"⚙️  Paramètres KAN: grid_size={self.grid_size}, spline_order={self.spline_order}")
            print(f"🎯 Objectif: {epochs} époques, batch_size={batch_size}")
            print(f"📚 Optimiseur: Adam (lr={learning_rate}, weight_decay={weight_decay})")
            print(f"⏰ Early stopping: patience={patience}")
            
            # Affichage du nombre de paramètres
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"🔢 Paramètres totaux: {total_params:,}")
            print(f"🔢 Paramètres entraînables: {trainable_params:,}")
            print("-" * 80)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Phase d'entraînement
            self.train()
            train_loss = 0.0
            
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                
                x_reconstructed, z = self(batch_x)
                loss = criterion(x_reconstructed, batch_x)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / max(1, len(train_loader))
            history['train_loss'].append(avg_train_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Phase de validation
            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    x_val_reconstructed, _ = self(X_val)
                    val_loss = criterion(x_val_reconstructed, X_val).item()
                    history['val_loss'].append(val_loss)
                    
                    scheduler.step(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss - 1e-9:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping à l'époque {epoch+1}")
                        break
            
            # Affichage détaillé pendant l'entraînement
            if verbose:
                # Affichage à chaque époque pour un feedback immédiat
                current_lr = optimizer.param_groups[0]['lr']
                elapsed_time = time.time() - start_time
                eta = (elapsed_time / (epoch + 1)) * (epochs - epoch - 1)
                
                if X_val is not None:
                    # Affichage avec validation
                    improvement = "✅" if val_loss < best_val_loss - 1e-9 else "⚠️"
                    print(f"📈 Époque {epoch+1:3d}/{epochs} | "
                          f"Train: {avg_train_loss:.6f} | "
                          f"Val: {val_loss:.6f} {improvement} | "
                          f"LR: {current_lr:.2e} | "
                          f"Patience: {patience_counter}/{patience} | "
                          f"⏱️  {elapsed_time:.1f}s (ETA: {eta:.1f}s)")
                    
                    # Affichage détaillé tous les 10% des époques
                    if (epoch + 1) % max(1, epochs // 10) == 0:
                        print(f"    💾 Meilleure val loss: {best_val_loss:.6f}")
                        print(f"    📊 Réduction val loss: {((history['val_loss'][0] - val_loss) / history['val_loss'][0] * 100):.1f}%")
                else:
                    # Affichage sans validation
                    print(f"📈 Époque {epoch+1:3d}/{epochs} | "
                          f"Train: {avg_train_loss:.6f} | "
                          f"LR: {current_lr:.2e} | "
                          f"⏱️  {elapsed_time:.1f}s (ETA: {eta:.1f}s)")
                    
                    # Affichage détaillé tous les 10% des époques  
                    if (epoch + 1) % max(1, epochs // 10) == 0:
                        print(f"    📊 Réduction train loss: {((history['train_loss'][0] - avg_train_loss) / history['train_loss'][0] * 100):.1f}%")
        
        training_time = time.time() - start_time
        
        if verbose:
            print("-" * 80)
            print(f"🎉 ENTRAÎNEMENT TERMINÉ EN {training_time:.2f}s")
            print("-" * 80)
            
            # Évaluation finale détaillée
            self.eval()
            with torch.no_grad():
                x_reconstructed, _ = self(X)
                final_loss = criterion(x_reconstructed, X).item()
                
                # Calcul de métriques supplémentaires
                mse_error = torch.mean((X - x_reconstructed) ** 2).item()
                mae_error = torch.mean(torch.abs(X - x_reconstructed)).item()
                
                print("📊 MÉTRIQUES FINALES:")
                print(f"   • Erreur de reconstruction (MSE): {final_loss:.6f}")
                print(f"   • Erreur absolue moyenne (MAE): {mae_error:.6f}")
                print(f"   • RMSE: {np.sqrt(mse_error):.6f}")
                
                if history['val_loss']:
                    best_epoch = np.argmin(history['val_loss']) + 1
                    improvement_pct = ((history['val_loss'][0] - min(history['val_loss'])) / history['val_loss'][0] * 100)
                    print(f"   • Meilleure époque: {best_epoch}")
                    print(f"   • Amélioration validation: {improvement_pct:.1f}%")
                
                print(f"   • Époques effectuées: {len(history['train_loss'])}/{epochs}")
                print(f"   • Temps par époque: {training_time/len(history['train_loss']):.2f}s")
        
        # Graphique des pertes
        if plot_losses and len(history['train_loss']) > 1:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
            if history['val_loss']:
                plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
            plt.title('Courbes de Perte')
            plt.xlabel('Époque')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            plt.subplot(1, 2, 2)
            plt.plot(history['learning_rate'], linewidth=2)
            plt.title('Taux d\'Apprentissage')
            plt.xlabel('Époque')
            plt.ylabel('Learning Rate')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            plt.tight_layout()
            plt.show()
        
        history['training_time'] = training_time
        history['final_loss'] = final_loss if 'final_loss' in locals() else avg_train_loss
        
        return history


# ================================
# EXEMPLE D'UTILISATION
# ================================

if __name__ == "__main__":
    # Configuration des paramètres
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=== EXEMPLE D'UTILISATION - KAN AUTOENCODER ===")
    print()
    
    # 1. Génération de données d'exemple (remplacez par vos vraies données)
    print("1. Génération de données d'exemple...")
    n_samples = 1000
    input_dim = 50
    
    # Données synthétiques avec structure latente
    true_latent_dim = 5
    latent_data = torch.randn(n_samples, true_latent_dim)
    
    # Matrice de mélange pour créer des données haute dimension
    mixing_matrix = torch.randn(input_dim, true_latent_dim)
    X = latent_data @ mixing_matrix.T + 0.1 * torch.randn(n_samples, input_dim)
    
    print(f"Données générées: {X.shape}")
    print(f"Moyenne: {X.mean():.4f}, Std: {X.std():.4f}")
    print()
    
    # 2. Initialisation de l'autoencodeur KAN
    print("2. Initialisation de l'autoencodeur KAN...")
    
    # Configuration de base
    autoencoder = KANAutoencoder(
        input_dim=input_dim,
        latent_dim=10,  # Dimension du bottleneck
        hidden_dims=[30, 20],  # Couches cachées
        grid_size=5,  # Taille de la grille pour les splines
        spline_order=3,  # Ordre des splines B
        noise_scale=0.1,  # Échelle du bruit d'initialisation
        base_activation='silu',  # Fonction d'activation de base
        dropout_rate=0.1,  # Dropout pour la régularisation
        use_batch_norm=True  # Normalisation par batch
    )
    
    print(f"Architecture créée: {input_dim} -> {autoencoder.hidden_dims} -> {autoencoder.latent_dim}")
    print(f"Nombre de paramètres: {sum(p.numel() for p in autoencoder.parameters()):,}")
    print()
    
    # 3. Entraînement
    print("3. Entraînement de l'autoencodeur...")
    
    history = autoencoder.fit(
        X=X,
        epochs=50,  # Nombre d'époques
        batch_size=32,  # Taille du batch
        learning_rate=0.001,  # Taux d'apprentissage
        weight_decay=1e-5,  # Régularisation L2
        validation_split=0.2,  # 20% pour la validation
        patience=10,  # Early stopping
        verbose=True,  # Affichage détaillé
        plot_losses=True  # Graphiques des pertes
    )
    
    print()
    
    # 4. Évaluation et utilisation
    print("4. Évaluation et utilisation...")
    
    # Obtenir la représentation latente
    latent_repr = autoencoder.get_latent_representation(X)
    print(f"Représentation latente: {latent_repr.shape}")
    
    # Reconstruction
    X_reconstructed = autoencoder.reconstruct(X)
    reconstruction_error = torch.mean((X - X_reconstructed) ** 2).item()
    print(f"Erreur de reconstruction (MSE): {reconstruction_error:.6f}")
    
    # Visualisation de l'espace latent (si 2D)
    if autoencoder.latent_dim == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(latent_repr[:, 0], latent_repr[:, 1], alpha=0.6, s=20)
        plt.title('Espace Latent 2D - KAN Autoencoder')
        plt.xlabel('Dimension Latente 1')
        plt.ylabel('Dimension Latente 2')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    print()
    print("=== EXEMPLE TERMINÉ ===")
    
    # 5. Configuration avancée (exemple)
    print()
    print("5. Exemple de configuration avancée...")
    
    advanced_autoencoder = KANAutoencoder(
        input_dim=100,
        latent_dim=10,
        hidden_dims=[80, 60, 40, 20],  # Architecture plus profonde
        grid_size=8,  # Grille plus fine
        spline_order=4,  # Splines d'ordre supérieur
        noise_scale=0.05,  # Moins de bruit
        base_activation='tanh',  # Activation différente
        dropout_rate=0.2,  # Plus de dropout
        use_batch_norm=True
    )
    
    print(f"Architecture avancée: {advanced_autoencoder.input_dim} -> {advanced_autoencoder.hidden_dims} -> {advanced_autoencoder.latent_dim}")
    print(f"Paramètres: grid_size={advanced_autoencoder.grid_size}, spline_order={advanced_autoencoder.spline_order}")
    print("Prêt pour l'entraînement sur vos données réelles !")
