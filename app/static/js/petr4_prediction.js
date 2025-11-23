/**
 * JavaScript para predição específica da PETR4.SA
 * Sistema simplificado focado apenas na ação da Petrobras
 */

// URLs dos endpoints da API
const API_ENDPOINTS = {
    predict: '/api/petr4/predict',
    modelInfo: '/api/petr4/model/info',
    health: '/api/petr4/health',
    currentPrice: '/api/petr4/current-price'
};

// Estado da aplicação
let currentPrediction = null;

/**
 * Inicialização quando o DOM estiver pronto
 */
document.addEventListener('DOMContentLoaded', function() {
    initializePETR4Prediction();
});

/**
 * Inicializa todas as funcionalidades
 */
function initializePETR4Prediction() {
    setupEventListeners();
    checkSystemHealth();
    loadCurrentPrice();
}

/**
 * Configura todos os event listeners
 */
function setupEventListeners() {
    // Formulário principal
    const petr4Form = document.getElementById('petr4Form');
    if (petr4Form) {
        petr4Form.addEventListener('submit', handlePETR4Prediction);
    }

    // Botão de atualizar dados
    const refreshBtn = document.getElementById('refreshDataBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadCurrentPrice);
    }

    // Botão de informações do modelo
    const modelInfoBtn = document.getElementById('modelInfoBtn');
    if (modelInfoBtn) {
        modelInfoBtn.addEventListener('click', showModelInfoModal);
    }
}

/**
 * Verifica a saúde do sistema
 */
async function checkSystemHealth() {
    try {
        const response = await fetch(API_ENDPOINTS.health);
        const health = await response.json();
        
        if (health.status !== 'healthy') {
            showAlert('warning', `Sistema com problemas: ${health.error || 'Verifique os logs'}`);
        }
    } catch (error) {
        console.warn('Erro ao verificar saúde do sistema:', error);
        showAlert('warning', 'Não foi possível verificar o status do sistema');
    }
}

/**
 * Carrega o preço atual da PETR4.SA
 */
async function loadCurrentPrice() {
    try {
        const response = await fetch(API_ENDPOINTS.currentPrice);
        if (response.ok) {
            const data = await response.json();
            
            // Atualizar na interface se já houver resultado
            const currentPriceElement = document.getElementById('current_price');
            if (currentPriceElement && !currentPriceElement.textContent.includes('--')) {
                currentPriceElement.textContent = formatCurrency(data.current_price);
            }
        }
    } catch (error) {
        console.warn('Erro ao carregar preço atual:', error);
    }
}

/**
 * Manipula o envio do formulário de predição
 */
async function handlePETR4Prediction(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const daysAhead = parseInt(formData.get('days_ahead'));
    
    const requestData = {
        days_ahead: daysAhead
    };
    
    try {
        showLoading(true);
        hideResult();
        
        const response = await fetch(API_ENDPOINTS.predict, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Erro na predição');
        }
        
        const result = await response.json();
        currentPrediction = result;
        displayPredictionResult(result);
        
    } catch (error) {
        console.error('Erro na predição:', error);
        showAlert('danger', `Erro: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

/**
 * Exibe o resultado da predição
 */
function displayPredictionResult(result) {
    // Preços
    document.getElementById('current_price').textContent = formatCurrency(result.current_price);
    document.getElementById('predicted_price').textContent = formatCurrency(result.predicted_price);
    document.getElementById('prediction_date').textContent = formatDate(result.prediction_date);
    
    // Variações
    const changeElement = document.getElementById('price_change');
    const percentageElement = document.getElementById('percentage_change');
    
    const isPositive = result.predicted_change >= 0;
    const changeText = (isPositive ? '+' : '') + formatCurrency(result.predicted_change);
    const percentageText = (isPositive ? '+' : '') + result.predicted_change_percentage.toFixed(2) + '%';
    
    changeElement.textContent = changeText;
    changeElement.className = isPositive ? 'text-success' : 'text-danger';
    
    percentageElement.textContent = percentageText;
    percentageElement.className = isPositive ? 'text-success' : 'text-danger';
    
    // Barra de confiança
    const confidence = result.confidence_score * 100;
    const confidenceBar = document.getElementById('confidence_bar');
    const confidenceText = document.getElementById('confidence_text');
    
    confidenceBar.style.width = `${confidence}%`;
    confidenceText.textContent = `${confidence.toFixed(1)}%`;
    
    // Cor da barra baseada na confiança
    confidenceBar.className = 'progress-bar';
    if (confidence >= 75) {
        confidenceBar.classList.add('bg-success');
    } else if (confidence >= 60) {
        confidenceBar.classList.add('bg-warning');
    } else {
        confidenceBar.classList.add('bg-danger');
    }
    
    // Interpretação
    const interpretation = generateInterpretation(result);
    document.getElementById('interpretation_text').textContent = interpretation;
    
    showResult();
}

/**
 * Gera texto de interpretação da predição
 */
function generateInterpretation(result) {
    const isPositive = result.predicted_change >= 0;
    const changePercent = Math.abs(result.predicted_change_percentage);
    const confidence = result.confidence_score * 100;
    const daysText = result.days_ahead === 1 ? 'amanhã' : `em ${result.days_ahead} dias`;
    
    let interpretation = `O modelo LSTM prevê que a PETR4.SA `;
    
    if (isPositive) {
        if (changePercent > 5) {
            interpretation += `terá uma alta significativa de ${changePercent.toFixed(2)}%`;
        } else if (changePercent > 2) {
            interpretation += `terá uma alta moderada de ${changePercent.toFixed(2)}%`;
        } else {
            interpretation += `terá uma leve alta de ${changePercent.toFixed(2)}%`;
        }
    } else {
        if (changePercent > 5) {
            interpretation += `terá uma queda significativa de ${changePercent.toFixed(2)}%`;
        } else if (changePercent > 2) {
            interpretation += `terá uma queda moderada de ${changePercent.toFixed(2)}%`;
        } else {
            interpretation += `terá uma leve queda de ${changePercent.toFixed(2)}%`;
        }
    }
    
    interpretation += ` ${daysText}. `;
    
    if (confidence >= 75) {
        interpretation += `A confiança é alta (${confidence.toFixed(1)}%), baseada em padrões consistentes nos dados históricos da Petrobras.`;
    } else if (confidence >= 60) {
        interpretation += `A confiança é moderada (${confidence.toFixed(1)}%), indicando alguma incerteza na predição devido à volatilidade do mercado.`;
    } else {
        interpretation += `A confiança é baixa (${confidence.toFixed(1)}%), sugerindo alta volatilidade ou condições atípicas no mercado.`;
    }
    
    return interpretation;
}

/**
 * Mostra/oculta o indicador de loading
 */
function showLoading(show) {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.classList.toggle('d-none', !show);
    }
}

/**
 * Mostra/oculta o resultado da predição
 */
function showResult() {
    document.getElementById('predictionResult').classList.remove('d-none');
}

function hideResult() {
    document.getElementById('predictionResult').classList.add('d-none');
}

/**
 * Mostra modal com informações do modelo
 */
async function showModelInfoModal() {
    try {
        const modal = new bootstrap.Modal(document.getElementById('modelInfoModal'));
        const content = document.getElementById('modelInfoContent');
        
        // Mostrar loading
        content.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-info" role="status">
                    <span class="visually-hidden">Carregando informações...</span>
                </div>
            </div>
        `;
        
        modal.show();
        
        // Buscar informações do modelo
        const response = await fetch(API_ENDPOINTS.modelInfo);
        
        if (!response.ok) {
            throw new Error('Erro ao carregar informações do modelo');
        }
        
        const modelInfo = await response.json();
        
        // Renderizar informações
        content.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Informações Gerais</h6>
                    <ul class="list-unstyled">
                        <li><strong>Nome:</strong> ${modelInfo.model_name}</li>
                        <li><strong>Versão:</strong> ${modelInfo.version}</li>
                        <li><strong>Ação:</strong> ${modelInfo.symbol}</li>
                        <li><strong>Sequência:</strong> ${modelInfo.sequence_length} dias</li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <h6>Período de Treinamento</h6>
                    <ul class="list-unstyled">
                        <li><strong>Início:</strong> ${modelInfo.training_period.start_date}</li>
                        <li><strong>Fim:</strong> ${modelInfo.training_period.end_date}</li>
                        <li><strong>Última Atualização:</strong> ${modelInfo.last_update}</li>
                    </ul>
                </div>
            </div>
            <hr>
            <div class="row">
                <div class="col-12">
                    <h6>Métricas de Performance</h6>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card bg-light">
                                <div class="card-body text-center">
                                    <h5 class="text-primary">${modelInfo.performance_metrics.rmse.toFixed(2)}</h5>
                                    <small>RMSE (Erro Médio)</small>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card bg-light">
                                <div class="card-body text-center">
                                    <h5 class="text-success">${modelInfo.performance_metrics.rmse_percentage.toFixed(1)}%</h5>
                                    <small>RMSE Percentual</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
    } catch (error) {
        console.error('Erro ao carregar informações do modelo:', error);
        const content = document.getElementById('modelInfoContent');
        content.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Erro ao carregar informações do modelo: ${error.message}
            </div>
        `;
    }
}

/**
 * Formatação de valores monetários
 */
function formatCurrency(value) {
    return new Intl.NumberFormat('pt-BR', {
        style: 'currency',
        currency: 'BRL'
    }).format(value);
}

/**
 * Formatação de data
 */
function formatDate(dateString) {
    return new Intl.DateTimeFormat('pt-BR', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit'
    }).format(new Date(dateString));
}

/**
 * Exibe alertas para o usuário
 */
function showAlert(type, message) {
    // Remover alertas existentes
    const existingAlerts = document.querySelectorAll('.alert-custom');
    existingAlerts.forEach(alert => alert.remove());
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show alert-custom`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container-fluid');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-remover após 5 segundos (exceto para alertas de erro)
    if (type !== 'danger') {
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}