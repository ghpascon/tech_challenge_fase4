# ---------------------------------------------------------------
# TUNING OTIMIZADO do LSTM usando Optuna + Lightning + MLflow
# Gera o modelo final: tuned_model.pth
# ---------------------------------------------------------------

import logging
import joblib
import optuna
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as L
import mlflow

from app.core.config import config

from app.services.ml.petr4_train import (    
    train_petr4_model,
    set_seed,
    setup_logging
)

# ---------------------------------------------
# CONFIG Local
# ---------------------------------------------

TUNED_MODEL_PATH = Path(config.model_dir) / "tuned_model.pth"
GLOBAL_DATA = None  # <---- PRE-CACHE DO DATASET (extremamente rápido)

# ===============================================================
# 1. Callback para salvar tuned_model.pth durante re-treino final
# ===============================================================

class SaveTunedModelCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.best_val = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")

        if val_loss is None:
            return

        v = val_loss.item()
        if v < self.best_val:
            self.best_val = v
            torch.save(pl_module.state_dict(), TUNED_MODEL_PATH)
            print(f"[TUNING] Novo melhor modelo salvo → {TUNED_MODEL_PATH}")


# ===============================================================
# 2. Objective do Optuna (rápido e otimizado)
# ===============================================================

def objective(trial):
    global GLOBAL_DATA

    # -------------------------
    # Pré-carregamento do Data
    # -------------------------
    if GLOBAL_DATA is None:
        print("[INFO] Carregando dataset uma vez só para todos os trials...")
        GLOBAL_DATA = train_petr4_model()
        GLOBAL_DATA.prepare_data()

    data = GLOBAL_DATA

    # -------------------------
    # Hiperparâmetros sugeridos
    # -------------------------
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # -------------------------
    # Modelo LSTM para o trial
    # -------------------------
    class TrialLSTM(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, 1)
            self.loss_fn = nn.MSELoss()

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=lr)

        def training_step(self, batch, batch_idx):
            X, y = batch
            y_hat = self(X).squeeze()
            loss = self.loss_fn(y_hat, y.squeeze())
            return loss

        def validation_step(self, batch, batch_idx):
            X, y = batch
            y_hat = self(X).squeeze()
            loss = self.loss_fn(y_hat, y.squeeze())
            self.log("val_loss", loss)
            return {"pred": y_hat.detach(), "target": y.detach()}

    model = TrialLSTM()

    # -------------------------
    # Trainer rápido e leve
    # -------------------------
    trainer = L.Trainer(
        max_epochs=3,                 # <---- otimizado
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=False
    )

    trainer.fit(model, datamodule=data)

    # -------------------------
    # Avaliação rápida no VAL
    # -------------------------
    val_loader = data.val_dataloader()

    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            y_hat = model(X).squeeze()
            preds.append(y_hat.numpy().ravel())
            targets.append(y.numpy().ravel())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    scaler = joblib.load(config.model_dir / config.scaler_name)

    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    targets_inv = scaler.inverse_transform(targets.reshape(-1, 1)).ravel()

    rmse = np.sqrt(mean_squared_error(targets_inv, preds_inv))

    print(f"[Trial {trial.number}] RMSE = {rmse:.4f}")

    return rmse


# ===============================================================
# 3. Tuning + Treino final + Log MLflow
# ===============================================================

def realiza_tuning():
    set_seed(config.seed)

    logger, log_path = setup_logging(config.model_dir)
    logger.info("Iniciando tuning otimizado...")

    mlflow.set_experiment("lstm_tuning")
    mlflow.start_run(run_name=f"TUNING_{config.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # -----------------------------------------------------------
    # Executa estudo Optuna
    # -----------------------------------------------------------
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=30,
        timeout=600  # 10 minutos - evita loops eternos
    )

    logger.info(f"Melhores hiperparâmetros: {study.best_params}")
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_rmse", study.best_value)

    # -----------------------------------------------------------
    # Re-treino final com parâmetros vencedores
    # -----------------------------------------------------------
    best_hp = study.best_params
    logger.info("Treinando modelo final com hiperparâmetros otimizados...")

    class FinalLSTM(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=best_hp["hidden_dim"],
                num_layers=best_hp["num_layers"],
                dropout=best_hp["dropout"],
                batch_first=True
            )
            self.fc = nn.Linear(best_hp["hidden_dim"], 1)
            self.loss_fn = nn.MSELoss()

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=best_hp["lr"])

        def training_step(self, batch, batch_idx):
            X, y = batch
            y_hat = self(X).squeeze()
            loss = self.loss_fn(y_hat, y.squeeze())
            return loss

        def validation_step(self, batch, batch_idx):
            X, y = batch
            y_hat = self(X).squeeze()
            loss = self.loss_fn(y_hat, y.squeeze())
            self.log("val_loss", loss)

    final_model = FinalLSTM()
    save_cb = SaveTunedModelCallback()

    trainer = L.Trainer(
        max_epochs=20,
        callbacks=[save_cb],
        logger=False,
        enable_checkpointing=False
    )

    # reutiliza GLOBAL_DATA
    trainer.fit(final_model, datamodule=GLOBAL_DATA)

    # -----------------------------------------------------------
    # Log MLflow
    # -----------------------------------------------------------
    if TUNED_MODEL_PATH.exists():
        mlflow.log_artifact(str(TUNED_MODEL_PATH))

    scaler_path = config.model_dir / config.scaler_name
    if scaler_path.exists():
        mlflow.log_artifact(str(scaler_path))

    if log_path.exists():
        mlflow.log_artifact(str(log_path))

    mlflow.end_run()

    logger.info("Tuning concluído com sucesso!")
    logger.info(f"Modelo final salvo em: {TUNED_MODEL_PATH}")


def main():
    realiza_tuning()

if __name__ == "__main__":
    main()
