"""
Preditor LSTM específico para a ação PETR4.SA
Modelo treinado com dados históricos de 2015-2025
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import logging


class LSTMForecast(nn.Module):
    """Modelo LSTM específico para PETR4.SA"""
    
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMForecast, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Pega a última saída da sequência
        out = self.fc(out)
        return out


class PETR4DataProvider:
    """Provedor de dados específico para PETR4.SA"""
    
    def __init__(self):
        self.symbol = "PETR4.SA"
        self.cache = None
        self.cache_timestamp = None
        self.cache_duration = timedelta(minutes=15)
    
    def get_petr4_data(self, period: str = "1y") -> Optional[pd.DataFrame]:
        """Obtém dados históricos da PETR4.SA"""
        try:
            now = datetime.now()
            
            # Verifica cache
            if (self.cache is not None and 
                self.cache_timestamp is not None and 
                now - self.cache_timestamp < self.cache_duration):
                logging.info(f"Usando dados em cache da PETR4.SA ({len(self.cache)} registros)")
                return self.cache
            
            # Busca dados do Yahoo Finance com período maior
            stock = yf.Ticker(self.symbol)
            
            # Tenta diferentes períodos se necessário
            periods_to_try = [period, "1y", "2y", "max"]
            
            for attempt_period in periods_to_try:
                try:
                    logging.info(f"Tentando buscar dados da PETR4.SA com período: {attempt_period}")
                    data = stock.history(period=attempt_period)
                    
                    if not data.empty and len(data) >= 30:  # Garantir dados suficientes
                        logging.info(f"Dados obtidos com sucesso: {len(data)} registros")
                        # Atualiza cache
                        self.cache = data
                        self.cache_timestamp = now
                        return data
                    else:
                        logging.warning(f"Dados insuficientes com período {attempt_period}: {len(data) if not data.empty else 0} registros")
                except Exception as period_error:
                    logging.warning(f"Erro com período {attempt_period}: {period_error}")
                    continue
            
            logging.error("Não foi possível obter dados suficientes da PETR4.SA em nenhum período")
            return None
            
        except Exception as e:
            logging.error(f"Erro geral ao obter dados da PETR4.SA: {e}")
            return None
    
    def get_current_price(self) -> Optional[float]:
        """Obtém o preço atual da PETR4.SA"""
        data = self.get_petr4_data(period="5d")
        if data is not None and not data.empty:
            return float(data['Close'].iloc[-1])
        return None


class PETR4LSTMPredictor:
    """Preditor específico para PETR4.SA usando modelo LSTM"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.sequence_length = 10  # Conforme treinamento
        self.data_provider = PETR4DataProvider()
        
        # Caminhos dos artefatos do modelo
        model_path = Path("model_pipeline/artifacts/best_model.pth")
        scaler_path = Path("model_pipeline/artifacts/scaler.joblib")
        
        self.load_model(model_path, scaler_path)
    
    def load_model(self, model_path: Path, scaler_path: Path):
        """Carrega o modelo LSTM e o scaler específicos da PETR4.SA"""
        try:
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler não encontrado em {scaler_path}")
            
            # Carrega o modelo com parâmetros exatos do treinamento
            self.model = LSTMForecast(
                input_dim=1, 
                hidden_dim=64, 
                num_layers=2, 
                output_dim=1, 
                dropout=0.2
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Carrega o scaler
            self.scaler = joblib.load(scaler_path)
            
            logging.info("Modelo LSTM da PETR4.SA carregado com sucesso")
            
        except Exception as e:
            logging.error(f"Erro ao carregar modelo da PETR4.SA: {e}")
            raise
    
    def prepare_sequence(self, data: np.ndarray) -> torch.Tensor:
        """Prepara a sequência de entrada para o modelo"""
        logging.debug(f"Preparando sequência com {len(data)} pontos de dados")
        
        if len(data) < self.sequence_length:
            raise ValueError(f"Dados insuficientes para criar sequência. Disponível: {len(data)}, necessário: {self.sequence_length}")
        
        last_sequence = data[-self.sequence_length:].reshape(-1, 1)
        logging.debug(f"Sequência extraída: {last_sequence.shape}")
        
        sequence_scaled = self.scaler.transform(last_sequence)
        logging.debug(f"Sequência normalizada: min={sequence_scaled.min():.4f}, max={sequence_scaled.max():.4f}")
        
        sequence_tensor = torch.tensor(
            sequence_scaled.reshape(1, self.sequence_length, 1), 
            dtype=torch.float32
        ).to(self.device)
        
        return sequence_tensor
    
    def predict_next_price(self, days_ahead: int = 1) -> Dict:
        """Prediz o preço futuro da PETR4.SA"""
        try:
            # Obtém dados históricos da PETR4.SA
            logging.info(f"Iniciando predição para {days_ahead} dias à frente")
            historical_data = self.data_provider.get_petr4_data(period="1y")
            
            if historical_data is None:
                raise ValueError("Não foi possível obter dados da PETR4.SA do Yahoo Finance")
            
            if len(historical_data) < self.sequence_length:
                raise ValueError(f"Dados insuficientes: obtidos {len(historical_data)} registros, necessários pelo menos {self.sequence_length}")
            
            # Prepara dados de fechamento
            close_prices = historical_data['Close'].values
            logging.info(f"Dados obtidos: {len(close_prices)} preços de fechamento")
            
            current_price = float(close_prices[-1])
            logging.info(f"Preço atual da PETR4.SA: R$ {current_price:.2f}")
            
            # Faz predições iterativas para múltiplos dias
            predictions = []
            sequence_data = close_prices.copy()
            
            for day in range(days_ahead):
                # Prepara sequência
                input_sequence = self.prepare_sequence(sequence_data)
                
                # Faz predição
                with torch.no_grad():
                    prediction_scaled = self.model(input_sequence)
                    prediction = self.scaler.inverse_transform(
                        prediction_scaled.cpu().numpy().reshape(-1, 1)
                    )
                    predicted_price = float(prediction[0][0])
                
                predictions.append(predicted_price)
                
                # Adiciona a predição aos dados para o próximo dia
                sequence_data = np.append(sequence_data, predicted_price)
            
            # Calcula métricas
            final_prediction = predictions[-1] if predictions else current_price
            change = final_prediction - current_price
            change_percentage = (change / current_price) * 100
            
            # Calcula confiança baseada na volatilidade dos últimos 30 dias
            volatility = np.std(close_prices[-30:]) / np.mean(close_prices[-30:])
            confidence = max(0.6, 1.0 - volatility * 1.5)  # Confiança específica para PETR4.SA
            
            return {
                "symbol": "PETR4.SA",
                "current_price": current_price,
                "predicted_price": final_prediction,
                "predicted_change": change,
                "predicted_change_percentage": change_percentage,
                "confidence_score": confidence,
                "days_ahead": days_ahead,
                "prediction_date": (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            logging.error(f"Erro na predição da PETR4.SA: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Retorna informações sobre o modelo da PETR4.SA"""
        return {
            "model_name": "LSTM_PETR4",
            "version": "1.0",
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
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def health_check(self) -> Dict:
        """Verifica a saúde do sistema de predição"""
        try:
            current_price = self.data_provider.get_current_price()
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "scaler_loaded": self.scaler is not None,
                "data_connection": current_price is not None,
                "current_petr4_price": current_price,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }