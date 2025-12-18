# -------------------------------------------------------------------------
# Fine-tuning do modelo LSTM previamente treinado (best_model.pth)
# Com Early Stopping + MLflow + rollback automático
# -------------------------------------------------------------------------

import logging
from pathlib import Path
from datetime import datetime

import torch
import mlflow
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback, EarlyStopping

from app.core.config import config
from app.services.ml.petr4_train import (
    setup_logging,
    set_seed,
    train_petr4_model,
    LitLSTM,
)

# =============================================================================
# Callback: salva tuned_model.pth apenas se melhorar
# =============================================================================

class SaveTunedModelCallback(Callback):
    def __init__(self, logger: logging.Logger, base_dir: Path):
        super().__init__()
        self.logger = logger
        self.best_val = float("inf")
        self.output_path = base_dir / "tuned_model.pth"

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")

        if val_loss is None:
            return

        val_loss = val_loss.item()

        if val_loss < self.best_val:
            self.best_val = val_loss
            torch.save(pl_module.state_dict(), self.output_path)

            self.logger.info(
                f"[TUNING] Melhor val_loss={val_loss:.6f} "
                f"→ tuned_model salvo em {self.output_path}"
            )

            mlflow.log_metric("best_val_loss_tuning", val_loss)


# =============================================================================
# Loop principal de tuning
# =============================================================================

def realiza_tuning():
    base_dir = Path(config.model_dir)
    best_model_path = base_dir / "best_model.pth"

    if not best_model_path.exists():
        raise FileNotFoundError(f"Modelo base não encontrado: {best_model_path}")

    # -------------------------------
    # Logging local
    # -------------------------------
    logger, log_path = setup_logging(base_dir)
    logger.info("Iniciando TUNING do modelo LSTM")

    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Modelo base: {best_model_path}")

    # -------------------------------
    # Ajustes específicos de tuning
    # -------------------------------
    original_lr = config.lr
    config.lr = config.lr * 0.1          # LR menor para fine-tuning
    tuning_epochs = 10

    logger.info(f"Tuning epochs: {tuning_epochs}")
    logger.info(f"LR original: {original_lr}")
    logger.info(f"LR tuning: {config.lr}")

    # -------------------------------
    # MLflow (run filho)
    # -------------------------------
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(
        run_name=f"TUNING_LSTM_{config.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        nested=True
    ):

        mlflow.log_params({
            "tuning": True,
            "base_model": "best_model.pth",
            "tuning_epochs": tuning_epochs,
            "learning_rate_tuning": config.lr,
            "early_stopping_patience": 5,
            "early_stopping_monitor": "val_loss",
            "seed": config.seed
        })

        # -------------------------------
        # DataModule
        # -------------------------------
        data = train_petr4_model()
        data.prepare_data()

        # -------------------------------
        # Modelo
        # -------------------------------
        model = LitLSTM()
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        logger.info("Pesos do best_model carregados com sucesso")

        # -------------------------------
        # Callbacks
        # -------------------------------
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            verbose=True
        )

        save_tuned_callback = SaveTunedModelCallback(logger, base_dir)

        # -------------------------------
        # Trainer
        # -------------------------------
        trainer = L.Trainer(
            max_epochs=tuning_epochs,
            accelerator="auto",
            deterministic=True,
            callbacks=[
                early_stopping,
                save_tuned_callback
            ],
            logger=False  # métricas já vão manualmente para o MLflow
        )

        # -------------------------------
        # Tuning
        # -------------------------------
        trainer.fit(model, datamodule=data)

        # -------------------------------
        # Artefatos
        # -------------------------------
        tuned_model_path = base_dir / "tuned_model.pth"
        if tuned_model_path.exists():
            mlflow.log_artifact(str(tuned_model_path))
        else:
            logger.warning("tuned_model.pth não foi gerado (sem melhora)")

        mlflow.log_artifact(str(log_path))

    logger.info("Tuning finalizado com sucesso")


def main():
    realiza_tuning()


if __name__ == "__main__":
    main()
