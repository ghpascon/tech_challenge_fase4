# -------------------------------------------------------------------------
# Treinamento do modelo LSTM com PyTorch Lightning + MLflow + Logging local
# -------------------------------------------------------------------------

from importlib.resources import path
import math
import logging
import joblib
import yfinance as yf
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Tuple

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as L
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from app.core.config import config

# =============================================================================
# 1. LOGGING LOCAL (idêntico ao notebook)
# =============================================================================

def setup_logging(base_dir: Path) -> Tuple[logging.Logger, Path]:
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"training_{timestamp}.log"

    logger = logging.getLogger("lstm_pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger, log_path


# =============================================================================
# 2. Funções auxiliares
# =============================================================================

def set_seed(seed):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_data():
    logger = logging.getLogger("lstm_pipeline")
    logger.setLevel(logging.INFO)
    logger.info(f"Baixando dados de {config.symbol} de {config.start_date} até {config.end_date}")
    df = yf.download(config.symbol, start=config.start_date, end=config.end_date)    
    if df.empty:
        raise ValueError("Dataset vazio!")
    logger.info(f"Dataset carregado com {df.shape[0]} linhas")
    df["y"] = df["Close"].shift(-1)
    df = df.dropna()
    return df[["Close", "y"]]


def create_sequences(values, seq_len):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X), np.array(y)


# =============================================================================
# 3. Dataset + DataModule (Lightning)
# =============================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class train_petr4_model(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self):
        logger = logging.getLogger("lstm_pipeline")
        logger.setLevel(logging.INFO)
        df = download_data()
        values = df["Close"].values.reshape(-1, 1)

        Path(config.model_dir).mkdir(parents=True, exist_ok=True)
        self.scaler.fit(values)
        logger.info("Escalonamento concluído")

        # Salva o scaler
        joblib.dump(self.scaler, config.model_dir / config.scaler_name)
        logger.info(f"Scaler salvo em: {config.model_dir / config.scaler_name}")

        values_scaled = self.scaler.transform(values)
        X, y = create_sequences(values_scaled, config.seq_len)
        logger.info(f"Sequências de {config.seq_len} geradas: X={X.shape}, y={y.shape}")

        n = len(X)
        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)

        self.X_train, self.y_train = X[:n_train], y[:n_train]
        self.X_val, self.y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
        self.X_test, self.y_test = X[n_train + n_val:], y[n_train + n_val:]
        
        logger.info(f"Divisões: train={n_train}, val={n_val}, test={len(X) - n_train - n_val}")

    def train_dataloader(self):
        return DataLoader(TimeSeriesDataset(self.X_train, self.y_train),
                          batch_size=config.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(TimeSeriesDataset(self.X_val, self.y_val),
                          batch_size=config.batch_size)

    def test_dataloader(self):
        return DataLoader(TimeSeriesDataset(self.X_test, self.y_test),
                          batch_size=config.batch_size)


# =============================================================================
# 4. Modelo LSTM (LightningModule)
# =============================================================================

class LitLSTM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout
        )
        self.fc = nn.Linear(config.hidden_dim, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=config.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=5, factor=0.5, mode="min"
        )
        logger = logging.getLogger("lstm_pipeline")
        logger.setLevel(logging.INFO)
        logger.info(f"criterion: {self.loss_fn}")
        logger.info(f"optimizer: {opt}")
        logger.info(f"scheduler: {sch}")
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze()
        loss = self.loss_fn(y_hat, y.squeeze())
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze()
        loss = self.loss_fn(y_hat, y.squeeze())
        self.log("val_loss", loss, on_epoch=True)
        return {"pred": y_hat.detach(), "target": y.detach()}

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze()
        return {"pred": y_hat.cpu(), "target": y.cpu()}


# =============================================================================
# 5. Callback para calcular RMSE REAL pós-scaler + log no MLflow
# =============================================================================

class SaveBestModelPTHCallback(Callback):
    def __init__(self, logger, log_path):
        super().__init__()
        self.logger = logger
        self.log_path = log_path
        self.best_val = float("inf")
        self.output_path = Path(config.model_dir) / "best_model.pth"

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")

        if val_loss is None:
            return

        val_loss = val_loss.item()

        # salva apenas se melhorar
        if val_loss < self.best_val:
            self.best_val = val_loss

            # salva state_dict puro (compatível com outros módulos)
            torch.save(pl_module.state_dict(), self.output_path)


            print(f"[Checkpoint] Melhor val_loss={val_loss:.6f} → modelo salvo em {self.output_path}")
            self.logger.info(f"Salvo state_dict() em: {path}")


class MetricsAndArtifactsCallback(Callback):
    def __init__(self, logger, log_path):
        super().__init__()
        self.logger = logger
        self.log_path = log_path
        # listas para acumular predições e targets durante o teste
        self._preds = []
        self._targets = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Este hook é chamado a cada batch do teste. 'outputs' é o retorno de test_step.
        Acumulamos preds e targets (em CPU e numpy) para uso posterior.
        """
        # outputs pode ser dict com keys "pred" e "target" (conforme test_step)
        if outputs is None:
            return

        # lidar com tensores ou arrays
        pred = outputs.get("pred") if isinstance(outputs, dict) else None
        target = outputs.get("target") if isinstance(outputs, dict) else None

        # se vierem tensores, mover para CPU e converter
        if pred is not None:
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy().ravel()
            else:
                pred = np.array(pred).ravel()
            self._preds.append(pred)

        if target is not None:
            if isinstance(target, torch.Tensor):
                target = target.detach().cpu().numpy().ravel()
            else:
                target = np.array(target).ravel()
            self._targets.append(target)

    def on_test_end(self, trainer, pl_module):
        """
        Chamado quando todos os batches de teste terminaram.
        Concatena os arrays acumulados, aplica inverse_transform e loga métricas + artifacts.
        """
        if len(self._preds) == 0 or len(self._targets) == 0:
            self.logger.warning("Nenhuma predição/target acumulada durante o teste.")
            return

        preds = np.concatenate(self._preds)
        targets = np.concatenate(self._targets)

        # carregar scaler
        scaler = joblib.load(config.model_dir / config.scaler_name)

        preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
        targets_inv = scaler.inverse_transform(targets.reshape(-1, 1)).ravel()

        rmse = math.sqrt(mean_squared_error(targets_inv, preds_inv))
        rmse_pct = (rmse / np.mean(targets_inv)) * 100

        self.logger.info(f"Test RMSE = {rmse:.6f}")

        # Log no MLflow (métricas finais)
        mlflow.log_metrics({
            "test_rmse": rmse,
            "test_rmse_percentage": rmse_pct
        })
               
        # registrar artefatos que existirem
        path = Path(config.model_dir) / config.model_name
        #if path and Path(path).exists():
        if path.exists():
            mlflow.log_artifact(str(path))
        else:
            self.logger.warning(f"Checkpoint não encontrado em: {path}")

        # scaler e arquivo de log
        scaler_path = config.model_dir / config.scaler_name
        if Path(scaler_path).exists():
            mlflow.log_artifact(str(scaler_path))
        else:
            self.logger.warning(f"Scaler não encontrado em: {scaler_path}")

        if Path(self.log_path).exists():
            mlflow.log_artifact(str(self.log_path))
        else:
            self.logger.warning(f"Arquivo de log não encontrado em: {self.log_path}")

        # limpar buffers caso o callback seja reutilizado
        self._preds = []
        self._targets = []

def apply_custom_config(custom_config: dict):
    """
    Atualiza os campos permitidos em config com base no JSON fornecido.
    Ignora parâmetros não reconhecidos.
    """

    if not isinstance(custom_config, dict):
        raise ValueError("custom_config deve ser um dicionário JSON.")

    # Campos permitidos e seus tipos
    allowed_fields = {
        "seq_len": int,
        "batch_size": int,
        "epochs": int,
        "hidden_dim": int,
        "dropout": float,
        "lr": float,
        "train_ratio": float,
        "val_ratio": float,
        "experiment_name": str,
    }

    logger = logging.getLogger("lstm_pipeline")
    logger.info("Aplicando custom_config...")

    for key, value in custom_config.items():
        if key not in allowed_fields:
            logger.warning(f"Campo ignorado no custom_config: {key}")
            continue

        expected_type = allowed_fields[key]

        # Tenta converter
        try:
            converted = expected_type(value)
        except Exception:
            raise ValueError(f"Valor inválido para '{key}'. Esperado tipo {expected_type.__name__}")

        setattr(config, key, converted)
        logger.info(f"config.{key} = {converted} (customizado)")


# =============================================================================
# 6. Loop principal de treinamento
# =============================================================================

def realiza_treinamento(custom_config: dict | None = None):

    if custom_config is not None:
        apply_custom_config(custom_config)

    # -------------------------------
    # Logging local
    # -------------------------------
    logger, log_path = setup_logging(config.model_dir)

    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando device: {device}")
    logger.info(f"Logging registrado em: {log_path}")

    # -------------------------------
    # Início do MLflow
    # -------------------------------
    mlflow.set_experiment(config.experiment_name)
    mlflow.start_run(run_name=f"LSTM_{config.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # hiperparâmetros
    mlflow.log_params({
        "symbol": config.symbol,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "seq_len": config.seq_len,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "learning_rate": config.lr,
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "device": str(device),
        "seed": config.seed
    })

    # -------------------------------
    # DataModule + tamanhos do dataset
    # -------------------------------
    data = train_petr4_model()
    data.prepare_data()

    mlflow.log_metrics({
        "dataset_size": len(data.X_train) + len(data.X_val) + len(data.X_test),
        "train_size": len(data.X_train),
        "val_size": len(data.X_val),
        "test_size": len(data.X_test),
        "sequence_length": config.seq_len
    })

    # -------------------------------
    # Modelo Lightning
    # -------------------------------
    model = LitLSTM()
    logger.info(f"Model LSTM:\n{model}")

    # Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.model_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True
    )

    # Logger Lightning → MLflow
    logger_mlflow = MLFlowLogger(
        experiment_name=config.experiment_name,
        run_id=mlflow.active_run().info.run_id
    )

    trainer = L.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",
        logger=logger_mlflow,
        deterministic=True,
        callbacks=[
            SaveBestModelPTHCallback(logger, log_path),
            MetricsAndArtifactsCallback(logger, log_path)
        ]
        #callbacks=[
        #    checkpoint_callback,
        #    MetricsAndArtifactsCallback(logger, log_path)
        #]
    )

    # -------------------------------
    # Treinamento + Teste
    # -------------------------------
    trainer.fit(model, datamodule=data)
    predictions = trainer.test(model, datamodule=data)

    # -------------------------------
    # Log final do modelo diretamente no MLflow
    # -------------------------------
    mlflow.pytorch.log_model(
        model,
        "lstm_model",
        registered_model_name=f"LSTM_{config.symbol}",
        extra_files=[str(config.model_dir / "scaler.joblib")]
    )

    mlflow.end_run()

    logger.info("Treinamento concluído com Lightning + MLflow!")


def main():
    realiza_treinamento()


if __name__ == "__main__":
    main()
