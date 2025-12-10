"""
API Endpoints para predição de preços da PETR4.SA

Este módulo contém os endpoints específicos para:
- Predição do preço da PETR4.SA
- Informações do modelo LSTM
- Status de saúde do sistema
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import logging

from app.services.ml.validators import (
    PETR4PredictionRequest,
    PETR4PredictionResponse, 
    ModelInfoResponse,
    HealthCheckResponse,
    PETR4TrainConfig
)
from app.services.ml.petr4_predictor_lit import PETR4LSTMPredictor

# Configurar logging
logger = logging.getLogger(__name__)

# Criar router
router = APIRouter(prefix="/api/petr4", tags=["PETR4 Prediction"])

# Instância global do preditor
try:
    petr4_predictor = PETR4LSTMPredictor()
    logger.info("Preditor PETR4 inicializado com sucesso")
except Exception as e:
    logger.error(f"Erro ao inicializar preditor PETR4: {e}")
    petr4_predictor = None


@router.post("/predict", response_model=PETR4PredictionResponse)
async def predict_petr4_price(request: PETR4PredictionRequest):
    """
    Prediz o preço futuro da PETR4.SA
    
    Utiliza o modelo LSTM treinado para fazer predições de 1 a 10 dias à frente.
    O modelo foi treinado com dados históricos da PETR4.SA de 2015 a 2025.
    """
    if petr4_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo LSTM não está disponível. Verifique os logs do sistema."
        )
    
    try:
        logger.info(f"Iniciando predição PETR4 para {request.days_ahead} dias")
        
        # Faz a predição
        prediction_result = petr4_predictor.predict_next_price(request.days_ahead)
        
        # Cria resposta estruturada
        response = PETR4PredictionResponse(
            current_price=prediction_result["current_price"],
            predicted_price=prediction_result["predicted_price"],
            predicted_change=prediction_result["predicted_change"],
            predicted_change_percentage=prediction_result["predicted_change_percentage"],
            prediction_date=prediction_result["prediction_date"],
            confidence_score=prediction_result["confidence_score"]
        )
        
        logger.info(f"Predição PETR4 concluída: {response.predicted_price}")
        return response
        
    except Exception as e:
        logger.error(f"Erro na predição PETR4: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno na predição: {str(e)}"
        )


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Retorna informações detalhadas sobre o modelo LSTM da PETR4.SA
    
    Inclui métricas de performance, período de treinamento e configurações do modelo.
    """
    if petr4_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo LSTM não está disponível"
        )
    
    try:
        model_info = petr4_predictor.get_model_info()
        
        response = ModelInfoResponse(
            training_period=model_info["training_period"],
            performance_metrics=model_info["performance_metrics"],
            sequence_length=model_info["sequence_length"],
            last_update=model_info["last_update"]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao obter informações do modelo"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Verifica o status de saúde do sistema de predição PETR4
    
    Testa a conectividade com dados, carregamento do modelo e disponibilidade do serviço.
    """
    try:
        if petr4_predictor is None:
            return HealthCheckResponse(
                status="unhealthy",
                model_loaded=False,
                scaler_loaded=False,
                data_connection=False,
                current_petr4_price=None,
                timestamp=datetime.now().isoformat(),
                error="Preditor não inicializado"
            )
        
        health_data = petr4_predictor.health_check()
        
        response = HealthCheckResponse(
            status=health_data["status"],
            model_loaded=health_data["model_loaded"],
            scaler_loaded=health_data["scaler_loaded"],
            data_connection=health_data["data_connection"],
            current_petr4_price=health_data.get("current_petr4_price"),
            timestamp=health_data["timestamp"],
            error=health_data.get("error")
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            model_loaded=False,
            scaler_loaded=False,
            data_connection=False,
            current_petr4_price=None,
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )


@router.get("/current-price")
async def get_current_petr4_price():
    """
    Obtém o preço atual da PETR4.SA
    
    Endpoint simples para verificar o preço atual sem fazer predições.
    """
    if petr4_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviço de dados não está disponível"
        )
    
    try:
        current_price = petr4_predictor.data_provider.get_current_price()
        
        if current_price is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Não foi possível obter o preço atual da PETR4.SA"
            )
        
        return {
            "symbol": "PETR4.SA",
            "current_price": current_price,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter preço atual: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao obter preço atual"
        )

@router.post("/train")
async def train_model(request: PETR4TrainConfig):
    """
    Aciona o treinamento do modelo LSTM para PETR4.SA
    
    Executa o script de treinamento (petr4_train.py).
    """
    try:
        logger.info("Iniciando treinamento do modelo PETR4")
        
        # Importar o módulo de treinamento
        from app.services.ml.petr4_train import realiza_treinamento
                
        # Executar treinamento
        #training_result = train_petr4_model()
        realiza_treinamento(request.model_dump())

        logger.info("Treinamento realizado com sucesso.")
        
        return {
            "status": "success",
            "message": "Treinamento do modelo realizado com sucesso",
            "timestamp": datetime.now().isoformat()            
        }
        
    except Exception as e:
        logger.error(f"Erro no treinamento do modelo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao treinar o modelo: {str(e)}"
        )
    

@router.post("/tuning")
async def model_tuning():
    """
    Aciona o tuning do modelo LSTM para PETR4.SA
    
    Executa o script de tuning (petr4_tuning.py).
    """
    try:
        logger.info("Iniciando tuning do modelo PETR4")
        
        # Importar o módulo de tuning
        from app.services.ml.petr4_tuning import realiza_tuning
                
        # Executar treinamento
        #training_result = train_petr4_model()
        realiza_tuning()

        logger.info("Tuning realizado com sucesso.")
        
        return {
            "status": "success",
            "message": "Tuning do modelo realizado com sucesso",
            "timestamp": datetime.now().isoformat()            
        }
        
    except Exception as e:
        logger.error(f"Erro no tuning do modelo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao realizar tuning do modelo: {str(e)}"
        )
    