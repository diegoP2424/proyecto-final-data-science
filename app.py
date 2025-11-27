import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predicción Fútbol IA", layout="wide")

# --- CARGAR ARTEFACTOS ---
@st.cache_resource
def load_artifacts():
    # Cargar transformadores
    imputer = joblib.load('imputer.sav')
    scaler = joblib.load('scaler.sav')
    # Cargar modelo
    model = tf.keras.models.load_model('modelo_futbol.keras')
    return imputer, scaler, model

try:
    imputer, scaler, model = load_artifacts()
    st.success("Modelo y Pipeline cargados correctamente")
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

# --- TÍTULO ---
st.title("⚽ Predicción de Partidos de Fútbol (Red Neuronal)")
st.markdown("Ingresa las estadísticas del partido para predecir si gana Local, Visitante o Empate.")

# --- INTERFAZ DE ENTRADA (Dividida en columnas) ---
st.header("Estadísticas del Partido")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Ratings Elo")
    home_elo = st.number_input("Elo Local", value=1500.0)
    away_elo = st.number_input("Elo Visitante", value=1500.0)
    
    st.subheader("Forma Reciente")
    form3_home = st.number_input("Forma Local (3 partidos)", value=1.5)
    form5_home = st.number_input("Forma Local (5 partidos)", value=2.5)
    form3_away = st.number_input("Forma Visitante (3 partidos)", value=1.5)
    form5_away = st.number_input("Forma Visitante (5 partidos)", value=2.5)

with col2:
    st.subheader("Cuotas de Apuesta (Odds)")
    odd_home = st.number_input("Cuota Local", value=2.10)
    odd_draw = st.number_input("Cuota Empate", value=3.20)
    odd_away = st.number_input("Cuota Visitante", value=3.50)
    
    st.subheader("Cuotas Máximas")
    max_home = st.number_input("Max Local", value=2.20)
    max_draw = st.number_input("Max Empate", value=3.30)
    max_away = st.number_input("Max Visitante", value=3.60)

with col3:
    st.subheader("Mercado Goles (2.5)")
    over25 = st.number_input("Over 2.5", value=1.90)
    under25 = st.number_input("Under 2.5", value=1.90)
    max_over25 = st.number_input("Max Over 2.5", value=2.00)
    max_under25 = st.number_input("Max Under 2.5", value=2.00)
    
    st.subheader("Hándicap Asiático")
    handi_size = st.number_input("Tamaño Hándicap", value=0.5)
    handi_home = st.number_input("Hándicap Local", value=1.95)
    handi_away = st.number_input("Hándicap Visitante", value=1.95)

# --- LOGICA DE PREDICCIÓN ---
if st.button("Predecir Resultado", type="primary"):
    # 1. Crear DataFrame con el mismo orden que en el entrenamiento
    features = [
        'HomeElo', 'AwayElo',
        'Form3Home', 'Form5Home',
        'Form3Away', 'Form5Away',
        'OddHome', 'OddDraw', 'OddAway',
        'MaxHome', 'MaxDraw', 'MaxAway',
        'Over25', 'Under25',
        'MaxOver25', 'MaxUnder25',
        'HandiSize', 'HandiHome', 'HandiAway'
    ]
    
    data = np.array([[
        home_elo, away_elo,
        form3_home, form5_home,
        form3_away, form5_away,
        odd_home, odd_draw, odd_away,
        max_home, max_draw, max_away,
        over25, under25,
        max_over25, max_under25,
        handi_size, handi_home, handi_away
    ]])
    
    # 2. Preprocesamiento (Pipeline: Imputer -> Scaler)
    # Importante: Usar los objetos cargados, NO crear nuevos
    data_imputed = imputer.transform(data)
    data_scaled = scaler.transform(data_imputed)
    
    # 3. Predicción
    prediction_prob = model.predict(data_scaled)
    prediction_class = np.argmax(prediction_prob, axis=1)[0]
    
    # Mapeo de clases (LabelEncoder ordena alfabéticamente: A=0, D=1, H=2)
    classes = {0: 'Visitante (Away)', 1: 'Empate (Draw)', 2: 'Local (Home)'}
    result_text = classes.get(prediction_class)
    
    # 4. Mostrar Resultados
    st.divider()
    st.subheader(f"Predicción del Modelo: {result_text}")
    
    # Mostrar probabilidades
    prob_df = pd.DataFrame(prediction_prob, columns=['Visitante', 'Empate', 'Local'])
    st.bar_chart(prob_df.T)
    
    st.info(f"Confianza: {np.max(prediction_prob)*100:.2f}%")