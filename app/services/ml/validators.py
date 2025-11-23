"""
Schemas Pydantic simplificados para predição específica da PETR4.SA
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime


class PETR4PredictionRequest(BaseModel):
    """Schema para requisição de predição da ação PETR4.SA"""
    days_ahead: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Número de dias à frente para predição (1-10)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "days_ahead": 1
            }
        }


class PETR4PredictionResponse(BaseModel):
    """Schema para resposta de predição da PETR4.SA"""
    symbol: str = "PETR4.SA"
    current_price: float
    predicted_price: float
    predicted_change: float
    predicted_change_percentage: float
    prediction_date: str
    confidence_score: float
    model_version: str = "LSTM_v1.0"

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "PETR4.SA",
                "current_price": 30.50,
                "predicted_price": 31.20,
                "predicted_change": 0.70,
                "predicted_change_percentage": 2.30,
                "prediction_date": "2024-01-16",
                "confidence_score": 0.85,
                "model_version": "LSTM_v1.0"
            }
        }


class ModelInfoResponse(BaseModel):
    """Schema para informações do modelo LSTM da PETR4.SA"""
    model_name: str = "LSTM_PETR4"
    version: str = "1.0"
    symbol: str = "PETR4.SA"
    training_period: Dict[str, str]
    performance_metrics: Dict[str, float]
    sequence_length: int
    last_update: str

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "LSTM_PETR4",
                "version": "1.0",
                "symbol": "PETR4.SA",
                "training_period": {
                    "start_date": "2015-01-01",
                    "end_date": "2025-10-20"
                },
                "performance_metrics": {
                    "rmse": 1.2,
                    "rmse_percentage": 4.0
                },
                "sequence_length": 10,
                "last_update": "2024-01-15 08:30:00"
            }
        }


class HealthCheckResponse(BaseModel):
    """Schema para verificação de saúde do sistema"""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    data_connection: bool
    current_petr4_price: Optional[float]
    timestamp: str
    error: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "scaler_loaded": True,
                "data_connection": True,
                "current_petr4_price": 30.50,
                "timestamp": "2024-01-15T10:30:00",
                "error": None
            }
        }