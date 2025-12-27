import streamlit as st
import requests
import pandas as pd
import os
import yfinance as yf
from datetime import datetime
import altair as alt

API_URL = os.environ.get("API_BASE_URL", "http://localhost:5000").rstrip("/")

ENDPOINTS = {
    "predict": f"{API_URL}/api/petr4/predict",
    "train": f"{API_URL}/api/petr4/train",
    "tuning": f"{API_URL}/api/petr4/tuning",
    "info": f"{API_URL}/api/petr4/model/info",
    "health": f"{API_URL}/api/petr4/health",
    "current_price": f"{API_URL}/api/petr4/current-price",
}


@st.cache_data(ttl=3600)
def get_historical_data(symbol, period='30d'):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return pd.DataFrame()
        df = data[['Close']].reset_index()

        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Erro ao buscar dados do Yahoo Finance: {e}")
        return pd.DataFrame()

def call_api(method, endpoint, payload=None):
    """Função robusta para chamadas de API."""
    try:
        if method == 'GET':
            response = requests.get(endpoint, timeout=10)
        elif method == 'POST':
            response = requests.post(endpoint, json=payload, timeout=200)
        else:
            return None

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na requisição ({endpoint}): {e}")
        return None

@st.cache_data(ttl=10)
def get_health_check():
    return call_api('GET', ENDPOINTS['health'])

@st.cache_data(ttl=600)
def get_model_info():
    return call_api('GET', ENDPOINTS['info'])

@st.cache_data(ttl=30)
def get_current_price():
    return call_api('GET', ENDPOINTS['current_price'])

def transform_prediction_result(api_result):
    """Normaliza o retorno da API para um formato padrão de DataFrame."""
    if not api_result:
        return None
    
    if 'predictions' in api_result:
        data = api_result

    elif 'predicted_price' in api_result:
        data = {
            "symbol": api_result.get("symbol", "PETR4.SA"),
            "current_price": api_result.get("current_price", 0.0),
            "confidence_score": api_result.get("confidence_score", 0.0),
            "predictions": [{
                "prediction_date": api_result.get("prediction_date"),
                "predicted_price": api_result.get("predicted_price"),
                "predicted_change": api_result.get("predicted_change", 0.0),
                "predicted_change_percentage": api_result.get("predicted_change_percentage", 0.0),
            }]
        }
    else:
        return None
    
    return data


def render_prediction_tab():
    st.header("📈 Previsão de Preço para PETR4.SA")

    col1, col2 = st.columns([1, 3])
    with col1:
        days_ahead = st.slider("Horizonte de Predição (Dias Úteis)", 1, 10, 3)
        if st.button("Executar Predição", type="primary"):
            with st.spinner("Consultando modelo..."):
                raw_result = call_api('POST', ENDPOINTS['predict'], {"days_ahead": days_ahead})
                processed = transform_prediction_result(raw_result)
                if processed:
                    st.session_state['prediction_result'] = processed
                    st.success("Dados atualizados!")

    with col2:
        if 'prediction_result' in st.session_state and st.session_state['prediction_result']:
            res = st.session_state['prediction_result']
            df_pred = pd.DataFrame(res['predictions'])
            
            df_pred['Date'] = pd.to_datetime(df_pred['prediction_date']).dt.tz_localize(None)
            
            # KPIs
            kpi1, kpi2, kpi3 = st.columns(3)
            curr_p = res.get('current_price', 0.0)
            last_p = df_pred.iloc[-1]
            
            kpi1.metric("Preço Atual", f"R$ {curr_p:.2f}")
            kpi2.metric(f"Alvo ({last_p['Date'].strftime('%d/%m')})", 
                       f"R$ {last_p['predicted_price']:.2f}",
                       delta=f"{last_p['predicted_change_percentage']:.2f}%")
            kpi3.metric("Confiança", f"{res.get('confidence_score', 0.0) * 100:.1f}%")

            hist_df = get_historical_data(res.get('symbol', 'PETR4.SA'))
            
            if not hist_df.empty:
                hist_plot = hist_df.rename(columns={'Close': 'Valor'})
                hist_plot['Tipo'] = 'Histórico'

                pred_plot = df_pred[['Date', 'predicted_price']].rename(columns={'predicted_price': 'Valor'})
                pred_plot['Tipo'] = 'Predição'
                
                connection_point = hist_plot.tail(1).copy()
                connection_point['Tipo'] = 'Predição'
                
                full_df = pd.concat([hist_plot, connection_point, pred_plot], ignore_index=True)

                chart = alt.Chart(full_df).mark_line(interpolate='monotone').encode(
                    x=alt.X('Date:T', title='Data'),
                    y=alt.Y('Valor:Q', scale=alt.Scale(zero=False), title='Preço (R$)'),
                    color=alt.Color('Tipo:N', scale=alt.Scale(domain=['Histórico', 'Predição'], range=['#1f77b4', '#ff7f0e'])),
                    strokeDash=alt.condition(
                        alt.datum.Tipo == 'Predição',
                        alt.value([5, 5]),
                        alt.value([0])
                    )
                ).properties(height=400).interactive()

                st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Clique em 'Executar Predição' para visualizar os dados.")


def render_management_tab():
    st.header("🛠️ Gerenciamento e Re-Treinamento do Modelo")
    st.subheader("1. Acionar Novo Treinamento do Modelo")
    with st.form("train_form"):
        train_params = {"seq_len": 10, "epochs": 50, "batch_size": 32, "hidden_dim": 64, "lr": 0.001, "dropout": 0.2, "experiment_name": "lstm_petr4_manual", "train_ratio": 0.7, "val_ratio": 0.15 }
        if st.form_submit_button("🚨 Iniciar Treinamento"):
            with st.spinner("Enviando solicitação de treinamento..."):
                train_result = call_api('POST', ENDPOINTS['train'], train_params)
            if train_result: st.success(f"Treinamento iniciado: {train_result.get('message')}")
    st.markdown("---")
    st.subheader("2. Otimização de Hiperparâmetros (Tuning)")
    if st.button("🔥 Acionar Model Tuning"):
        with st.spinner("Enviando solicitação de tuning..."):
            tuning_result = call_api('POST', ENDPOINTS['tuning'])
        if tuning_result: st.success(f"Tuning iniciado: {tuning_result.get('message')}")


def render_model_info_tab():
    st.header("ℹ️ Detalhes e Métricas do Modelo LSTM")
    info = get_model_info()
    if info:
        st.subheader(f"Modelo: {info.get('model_name', 'N/A')} ({info.get('symbol', 'N/A')})")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Versão e Período de Treinamento**")
            st.metric("Versão", info.get("version", "N/A"))
        with col2:
            st.markdown("**Métricas de Performance**")
            metrics = info.get("performance_metrics", {})
            st.metric("RMSE Absoluto", f"{metrics.get('rmse', 'N/A')}")
        st.markdown("---")
        st.subheader("Configurações do Modelo (Raw)")
        st.json(info)


def render_utilities_tab():
    st.header("🩺 Utilitários e Status do Sistema")
    st.subheader("1. Status de Saúde (Health Check)")
    health = get_health_check()
    if health:
        status = health.get("status")
        col1, col2, col3, col4 = st.columns(4)
        if status == "healthy": col1.success("Status: OK")
        else: col1.error("Status: Falha")
        col2.metric("Modelo Carregado", "Sim" if health.get("model_loaded") else "Não")
        col4.metric("Conexão de Dados", "OK" if health.get("data_connection") else "Falha")
    st.markdown("---")
    st.subheader("2. Preço Atual da PETR4.SA")
    price_info = get_current_price()
    if price_info:
        st.metric(f"Preço de {price_info.get('symbol', 'PETR4.SA')}", f"R$ {price_info.get('current_price', 'N/A'):.2f}")



def main():
    st.set_page_config(page_title="PETR4 Predictor", layout="wide")
    st.title("Sistema de Predição PETR4.SA")
    
    tabs = st.tabs(["📊 Predição", "🛠️ Gerenciamento", "ℹ️ Info", "🩺 Saúde"])
    with tabs[0]: render_prediction_tab()
    with tabs[1]: render_management_tab()
    with tabs[2]: render_model_info_tab()
    with tabs[3]: render_utilities_tab()

if __name__ == "__main__":
    main()