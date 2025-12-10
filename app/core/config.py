from dotenv import load_dotenv
import os
from pathlib import Path

class Settings:
    def __init__(self):
        """Application settings loader and manager."""
        load_dotenv()
        self.data = {key: value for key, value in os.environ.items()}
        
        # Configurações específicas da aplicação PETR4.SA
        if "TITLE" not in self.data:
            self.data["TITLE"] = "PETR4 Predictor - Sistema LSTM"


settings = Settings()

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DO PIPELINE
# -----------------------------------------------------------------------------

class PipelineConfig:
    seed = 42
    symbol = "PETR4.SA"
    start_date = "2015-01-01"
    end_date = "2025-10-20"
    seq_len = 10
    batch_size = 32
    epochs = 100
    hidden_dim = 64
    num_layers = 2
    dropout = 0.2
    lr = 0.001
    train_ratio = 0.7
    val_ratio = 0.15  
    model_dir = Path("model_pipeline/artifacts")
    model_name: str = "best_model.pth"
    scaler_name: str = "scaler.joblib"
    experiment_name: str = "lstm_stock_forecast"
    #experiment_name = "lstm_stock_lightning"


config = PipelineConfig()
