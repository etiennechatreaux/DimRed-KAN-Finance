# -*- coding: utf-8 -*-
"""
Autoencodeur MLP (baseline) — architecture symétrique, tailles cachées paramétrables.

Features:
- hidden=(256,64) par ex. pour l'encodeur; le décodeur est le miroir
- activation: 'relu' | 'silu' | 'tanh' | 'gelu'
- dropout & batchnorm optionnels
- fit() avec DataLoader CPU->GPU (pin_memory), early stopping, logs
- évaluation finale par batch (évite l'OOM)
- encode_batched / reconstruct utilitaires

Exemples:
    model = MLPAutoencoder(input_dim=784, k=32, hidden=(256,64),
                           activation='relu', dropout_p=0.0, use_bn=False).to(device)
    hist = model.fit(X_train_tensor, epochs=20, batch_size=256, learning_rate=3e-3)
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn


# -------------------------
# MLP blocks
# -------------------------
def make_mlp(in_f: int,
             out_f: int,
             activation: str = "relu",
             use_bn: bool = False,
             dropout_p: float = 0.0) -> nn.Sequential:
    acts = {
        "relu": nn.ReLU(inplace=True),
        "silu": nn.SiLU(inplace=True),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "identity": nn.Identity(),
    }
    layers = [nn.Linear(in_f, out_f)]
    if use_bn:
        layers.append(nn.BatchNorm1d(out_f))
    layers.append(acts.get(activation.lower(), nn.ReLU(inplace=True)))
    if dropout_p and dropout_p > 0.0:
        layers.append(nn.Dropout(dropout_p))
    return nn.Sequential(*layers)


class MLPAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        k: int = 16,
        *,
        hidden: Tuple[int, ...] = (256, 64),   # encodeur; décodeur = miroir
        activation: str = "relu",
        use_bn: bool = False,
        dropout_p: float = 0.0,
        # loss: "mse" (par défaut) ou "bce_logits" (images binaires type MNIST)
        loss_type: str = "mse",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.k = int(k)
        self.hidden = tuple(int(h) for h in hidden)
        self.activation = activation
        self.use_bn = bool(use_bn)
        self.dropout_p = float(dropout_p)
        self.loss_type = loss_type.lower()

        # ---------- ENCODEUR ----------
        enc_dims = (self.input_dim,) + self.hidden
        enc_blocks = []
        for i in range(len(enc_dims) - 1):
            enc_blocks.append(
                make_mlp(enc_dims[i], enc_dims[i+1],
                         activation=self.activation, use_bn=self.use_bn, dropout_p=self.dropout_p)
            )
        self.encoder = nn.Sequential(*enc_blocks)
        last_enc = self.hidden[-1] if self.hidden else self.input_dim
        self.to_latent = nn.Linear(last_enc, self.k, bias=True)

        # ---------- DECODEUR (symétrique) ----------
        dec_hidden = tuple(reversed(self.hidden))
        if len(dec_hidden) > 0:
            self.from_latent = nn.Linear(self.k, dec_hidden[0], bias=True)
        else:
            self.from_latent = nn.Identity()

        dec_blocks = []
        if len(dec_hidden) >= 2:
            for i in range(len(dec_hidden)-1):
                dec_blocks.append(
                    make_mlp(dec_hidden[i], dec_hidden[i+1],
                             activation=self.activation, use_bn=self.use_bn, dropout_p=self.dropout_p)
                )
        self.decoder = nn.Sequential(*dec_blocks)
        last_dec = dec_hidden[-1] if len(dec_hidden) > 0 else self.k
        self.out = nn.Linear(last_dec, self.input_dim, bias=True)

        # sanity
        assert self.input_dim > 0 and self.k > 0

    # ---------- API ----------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x) if len(self.encoder) > 0 else x
        z = self.to_latent(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = self.decoder(h) if len(self.decoder) > 0 else h
        x_hat = self.out(h)
        return x_hat

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            x_hat, _ = self(x)
        return x_hat

    def encode_batched(self, x: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        self.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                outs.append(self.encode(x[i:i+batch_size]))
        return torch.cat(outs, dim=0)

    def _criterion(self):
        if self.loss_type == "bce_logits":
            return nn.BCEWithLogitsLoss()
        return nn.MSELoss()

    # ---------- Entraînement ----------
    def fit(
        self,
        X: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        validation_split: float = 0.2,
        patience: int = 10,
        verbose: bool = True,
        device: Optional[torch.device] = None,
        amp: bool = True,  # mixed precision
        num_workers: int = 4,
    ) -> Dict[str, Any]:
        """
        Entraîne l'AE MLP avec early stopping et logs.
        """
        import time

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Move X to CPU if needed for DataLoader with pin_memory
        if X.device.type != "cpu":
            X = X.cpu()

        # split train/val (sur CPU)
        n = X.size(0)
        n_val = int(n * validation_split)
        idx = torch.randperm(n)
        X_val = X[idx[:n_val]] if n_val > 0 else None
        X_trn = X[idx[n_val:]]

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_trn, X_trn),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
            num_workers=(num_workers if device.type == "cuda" else 0),
            persistent_workers=(device.type == "cuda" and num_workers > 0),
        )

        # loss / opt / lr scheduler
        criterion = self._criterion()
        opt = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, patience // 2))
        scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

        # logs
        hist = {"train_loss": [], "val_loss": [], "lr": []}
        best_val = float("inf")
        wait = 0

        if verbose:
            enc_arch = " -> ".join(map(str, (self.input_dim,) + self.hidden))
            dec_arch = " -> ".join(map(str, self.hidden[::-1] + (self.input_dim,)))
            print("=" * 88)
            print("🚀 ENTRAÎNEMENT MLP AUTOENCODER")
            print("=" * 88)
            print(f"📊 Data: {X_trn.size(0)} train | {0 if X_val is None else X_val.size(0)} val")
            print(f"🏗️  Arch: {enc_arch} -> {self.k} -> {dec_arch}")
            print(f"⚙️  Params: epochs={epochs}, bs={batch_size}, lr={learning_rate}, loss={self.loss_type}")
            print(f"🔧 Device: {device} | AMP: {amp}")
            tot = sum(p.numel() for p in self.parameters())
            trn = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"🔢 Paramètres: {tot:,} total | {trn:,} entraînables")
            print("-" * 88)

        t0 = time.time()
        for ep in range(1, epochs + 1):
            self.train()
            loss_sum = 0.0
            for (xb, _) in train_loader:
                xb = xb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if self.loss_type == "bce_logits":
                    # Pour BCE logits, x doit être dans [0,1] (pas besoin de sigmoid ici)
                    pass
                with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                    xh, _ = self(xb)
                    loss = criterion(xh, xb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                loss_sum += loss.item()
            train_loss = loss_sum / max(1, len(train_loader))

            # validation batchée (évite OOM)
            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    bs_val = min(batch_size, 512)
                    tot_v, cnt_v = 0.0, 0
                    for i in range(0, X_val.size(0), bs_val):
                        xvb = X_val[i:i+bs_val].to(device, non_blocking=True)
                        xvh, _ = self(xvb)
                        tot_v += criterion(xvh, xvb).item() * xvb.size(0)
                        cnt_v += xvb.size(0)
                    val_loss = tot_v / max(1, cnt_v)
                sched.step(val_loss)
                improved = val_loss + 1e-9 < best_val
                if improved:
                    best_val = val_loss
                    wait = 0
                    best_state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(f"🛑 Early stopping à l'époque {ep}")
                        break
            else:
                val_loss = float("nan")

            hist["train_loss"].append(train_loss)
            hist["val_loss"].append(val_loss)
            hist["lr"].append(opt.param_groups[0]["lr"])

            if verbose:
                elapsed = time.time() - t0
                eta = (elapsed / ep) * (epochs - ep)
                flag = "✅" if X_val is not None and val_loss + 1e-9 <= best_val else "⚠️"
                print(
                    f"📈 Époque {ep:3d}/{epochs} | "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} {flag} | "
                    f"LR: {opt.param_groups[0]['lr']:.2e} | "
                    f"⏱️ {elapsed:.1f}s (ETA: {eta:.1f}s)"
                )

        # restore best
        if X_val is not None and "best_state" in locals():
            self.load_state_dict(best_state)

        # final eval by batches
        self.eval()
        with torch.no_grad():
            bs_eval = min(batch_size, 512)
            tot_f, cnt_f = 0.0, 0
            for i in range(0, X.size(0), bs_eval):
                xb = X[i:i+bs_eval].to(device, non_blocking=True)
                xh, _ = self(xb)
                tot_f += self._criterion()(xh, xb).item() * xb.size(0)
                cnt_f += xb.size(0)
            final_loss = tot_f / max(1, cnt_f)

        if verbose:
            elapsed = time.time() - t0
            print("-" * 88)
            print(f"🎉 ENTRAÎNEMENT TERMINÉ EN {elapsed:.2f}s")
            print("-" * 88)
            print(f"📊 MSE/BCE finale (selon loss): {final_loss:.6f}")

        hist: Dict[str, Any] = hist
        hist["final_loss"] = final_loss
        hist["training_time"] = time.time() - t0
        return hist


# -------------------------
# Petit test rapide
# -------------------------
# if __name__ == "__main__":
#     torch.manual_seed(42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)

#     # Dummy data (remplace par MNIST)
#     X = torch.rand(10_000, 784)  # [0,1]
#     model = MLPAutoencoder(input_dim=784, k=32, hidden=(256, 64), activation="relu",
#                            use_bn=False, dropout_p=0.0, loss_type="mse").to(device)
#     hist = model.fit(X, epochs=3, batch_size=512, learning_rate=3e-3, patience=2, verbose=True, device=device)
