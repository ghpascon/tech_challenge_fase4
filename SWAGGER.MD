# 📈 PETR4.SA Prediction API

## Descrição
API especializada para predição de preços da ação PETR4.SA (Petróleo Brasileiro S.A.) utilizando modelo LSTM (Long Short-Term Memory) de Deep Learning.

## 🚀 Características

- **Modelo Específico**: LSTM treinado exclusivamente com dados da PETR4.SA
- **Período de Treinamento**: 2015-2025 (10 anos de dados históricos)
- **Precisão**: RMSE de 1.2 (~4% de erro percentual)
- **Predição**: 1 a 10 dias à frente
- **Sequência**: Utiliza 10 dias de histórico para cada predição

## 📊 Endpoints Disponíveis

### 🎯 Predição Principal
**POST** `/api/petr4/predict`

- Faz predição do preço futuro da PETR4.SA
- Parâmetros: `days_ahead` (1-10 dias)
- Retorna: preço atual, preço previsto, variação e confiança

### 🏥 Status do Sistema
**GET** `/api/petr4/health`

- Verifica saúde do sistema de predição
- Testa conectividade com dados e carregamento do modelo
- Retorna: status, preço atual e timestamp

### ℹ️ Informações do Modelo
**GET** `/api/petr4/model/info`

- Detalhes sobre o modelo LSTM
- Métricas de performance e configuração
- Período de treinamento e última atualização

### 💰 Preço Atual
**GET** `/api/petr4/current-price`

- Obtém preço atual da PETR4.SA do Yahoo Finance
- Endpoint simples sem predições

## 🔧 Tecnologias Utilizadas

- **FastAPI**: Framework web moderno e rápido
- **PyTorch**: Framework de Deep Learning para LSTM
- **yfinance**: Coleta de dados financeiros em tempo real
- **scikit-learn**: Pré-processamento de dados
- **Pydantic v2**: Validação de dados e serialização

## 📈 Modelo LSTM

- **Arquitetura**: 2 camadas LSTM com 64 neurônios cada
- **Dropout**: 0.2 para evitar overfitting
- **Sequência**: 10 dias de preços de fechamento
- **Normalização**: MinMaxScaler para dados entre 0-1
- **Otimizador**: Adam com learning rate 0.001

## 🎯 Confiança da Predição
O score de confiança é calculado baseado na:

- Volatilidade histórica dos últimos 30 dias
- Consistência dos padrões nos dados
- Performance do modelo em dados de validação

## 🔄 Cache e Performance

- Cache de dados por 15 minutos para otimizar performance
- Fallbacks automáticos para diferentes períodos de dados
- Logs detalhados para monitoramento e debugging
