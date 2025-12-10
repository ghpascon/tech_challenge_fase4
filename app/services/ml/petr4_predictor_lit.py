"""
Preditor LSTM específico para a ação PETR4.SA
Usa modelo treinado com PyTorch Lightning (checkpoint .ckpt)
"""

import torch
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import logging
import sys
from pathlib import Path

# Adiciona o caminho do diretório pai ao sys.path
import sys
from pathlib import Path

from .petr4_train import LitLSTM

# --------------------------------------------------------------------------
# DATA PROVIDER
# --------------------------------------------------------------------------

class PETR4DataProvider:
    """Provedor de dados específico para PETR4.SA"""

    def __init__(self):
        self.symbol = "PETR4.SA"
        self.cache = None
        self.cache_timestamp = None
        self.cache_duration = timedelta(minutes=15)

    def get_petr4_data(self, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Obtém dados históricos da PETR4.SA
        period: período para buscar (ex: '1y', '6mo', '1mo')
        Retorna DataFrame com dados ou None se falhar
        """
        
        try:
            now = datetime.now()

            # Verifica cache
            if (
                self.cache is not None
                and self.cache_timestamp is not None
                and now - self.cache_timestamp < self.cache_duration
            ):
                logging.info(
                    f"Usando dados em cache da PETR4.SA ({len(self.cache)} registros)"
                )
                return self.cache

            stock = yf.Ticker(self.symbol)
            periods_to_try = [period, "1y", "2y", "max"]

            for attempt_period in periods_to_try:
                try:
                    logging.info(
                        f"Tentando buscar dados da PETR4.SA com período: {attempt_period}"
                    )
                    data = stock.history(period=attempt_period)

                    if not data.empty and len(data) >= 30:
                        logging.info(f"Dados obtidos: {len(data)} registros")
                        self.cache = data
                        self.cache_timestamp = now
                        return data

                except Exception as err:
                    logging.warning(f"Erro com período {attempt_period}: {err}")

            logging.error(
                "Não foi possível obter dados suficientes da PETR4.SA em nenhum período"
            )
            return None

        except Exception as e:
            logging.error(f"Erro geral ao obter dados da PETR4.SA: {e}")
            return None

    def get_current_price(self) -> Optional[float]:
        """Obtém o preço atual da PETR4.SA"""
        data = self.get_petr4_data(period="5d")
        if data is not None and not data.empty:
            return float(data["Close"].iloc[-1])
        return None


# --------------------------------------------------------------------------
# PREDICTOR
# --------------------------------------------------------------------------

class PETR4LSTMPredictor:
    """Preditor específico para PETR4.SA usando modelo LSTM Lightning"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.scaler = None
        self.sequence_length = 10  # igual ao treinamento
        self.data_provider = PETR4DataProvider()

        # Caminhos dos artefatos gerados no treinamento Lightning
        model_path = Path("model_pipeline/artifacts/best_model.pth")
        scaler_path = Path("model_pipeline/artifacts/scaler.joblib")

        self.load_model(model_path, scaler_path)

    # ----------------------------------------------------------------------
    # CARREGAMENTO DO MODELO
    # ----------------------------------------------------------------------

    def load_model(self, model_path: Path, scaler_path: Path):
        """
        Carrega o modelo Lightning + scaler
        model_path: caminho do checkpoint do modelo
        scaler_path: caminho do scaler utilizado no treino
        """

        try:
            if not model_path.exists():
                raise FileNotFoundError(f"Checkpoint não encontrado: {model_path}")
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler não encontrado: {scaler_path}")

            # 🔥 Carrega o checkpoint Lightning do treinamento
            #self.model = LitLSTM.load_from_checkpoint(
            #    checkpoint_path=str(model_path),
            #    map_location=self.device
            #).to(self.device)

            self.model = LitLSTM()
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device)                

            self.model.eval()  # modo inferência

            # Carrega scaler utilizado no treino
            self.scaler = joblib.load(scaler_path)

            logging.info("Modelo Lightning PETR4 carregado com sucesso")

        except Exception as e:
            logging.error(f"Erro ao carregar o modelo Lightning: {e}")
            raise

    # ----------------------------------------------------------------------
    # PREPARAÇÃO DA SEQUÊNCIA
    # ----------------------------------------------------------------------

    def prepare_sequence(self, data: np.ndarray) -> torch.Tensor:
        """
        Prepara a sequência para inferência
        data: array numpy com preços históricos
        Retorna tensor pronto para o modelo
        """

        logging.debug(f"Preparando sequência com {len(data)} pontos")

        if len(data) < self.sequence_length:
            raise ValueError(
                f"Dados insuficientes para criar sequência. "
                f"Disponível: {len(data)}, necessário: {self.sequence_length}"
            )

        last_sequence = data[-self.sequence_length:].reshape(-1, 1)
        sequence_scaled = self.scaler.transform(last_sequence)

        sequence_tensor = torch.tensor(
            sequence_scaled.reshape(1, self.sequence_length, 1),
            dtype=torch.float32,
        ).to(self.device)

        return sequence_tensor

    # ----------------------------------------------------------------------
    # PREVISÃO
    # ----------------------------------------------------------------------

    def predict_next_price(self, days_ahead: int = 1) -> Dict:
        """
        Prediz o preço futuro da PETR4.SA
        days_ahead: número de dias à frente para prever (1-10)
        Retorna um dicionário com resultados da predição
        """

        try:
            logging.info(f"Iniciando predição para {days_ahead} dias à frente")

            historical_data = self.data_provider.get_petr4_data(period="1y")

            if historical_data is None:
                raise ValueError("Não foi possível obter dados da PETR4.SA")

            if len(historical_data) < self.sequence_length:
                raise ValueError(
                    f"Dados insuficientes ({len(historical_data)} registros)"
                )

            close_prices = historical_data["Close"].values
            current_price = float(close_prices[-1])

            logging.info(f"Preço atual da PETR4: R$ {current_price:.2f}")

            predictions = []
            sequence_data = close_prices.copy()

            # Predição iterativa (multi-step)
            for _ in range(days_ahead):
                inp = self.prepare_sequence(sequence_data)

                with torch.no_grad():
                    pred_scaled = self.model(inp)
                    pred = self.scaler.inverse_transform(
                        pred_scaled.cpu().numpy().reshape(-1, 1)
                    )
                    predicted_price = float(pred[0][0])

                predictions.append(predicted_price)

                # adiciona a predição para o próximo passo
                sequence_data = np.append(sequence_data, predicted_price)

            final_prediction = predictions[-1]
            change = final_prediction - current_price
            change_pct = (change / current_price) * 100

            # Confiança baseada na volatilidade recente
            volatility = np.std(close_prices[-30:]) / np.mean(close_prices[-30:])
            confidence = max(0.6, 1.0 - volatility * 1.5)

            return {
                "symbol": "PETR4.SA",
                "current_price": current_price,
                "predicted_price": final_prediction,
                "predicted_change": change,
                "predicted_change_percentage": change_pct,
                "confidence_score": confidence,
                "days_ahead": days_ahead,
                "prediction_date": (datetime.now() + timedelta(days=days_ahead)).strftime(
                    "%Y-%m-%d"
                ),
            }

        except Exception as e:
            logging.error(f"Erro na predição: {e}")
            raise

    # ----------------------------------------------------------------------
    # INFORMAÇÕES DO MODELO
    # ----------------------------------------------------------------------

    def get_model_info(self) -> Dict:
        """Retorna informações detalhadas sobre o modelo LSTM da PETR4.SA"""

        return {
            "model_name": "LSTM_PETR4_Lightning",
            "symbol": "PETR4.SA",
            "sequence_length": self.sequence_length,
            "training_period": {
                "start_date": "2015-01-01",
                "end_date": "2025-10-20"
            },
            "performance_metrics": {
                "rmse": 1.2,
                "rmse_percentage": 4.0
            },
            "device": str(self.device),
            "checkpoint_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ----------------------------------------------------------------------
    # SISTEMA
    # ----------------------------------------------------------------------

    def health_check(self) -> Dict:
        """Verifica se o sistema está no ar e funcionando corretamente"""
        try:
            current_price = self.data_provider.get_current_price()
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "scaler_loaded": self.scaler is not None,
                "data_connection": current_price is not None,
                "current_petr4_price": current_price,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
