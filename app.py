import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Carga del Dataset para Gráficas
# -----------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    rutas = [
        os.path.join(BASE_DIR, "canciones.csv"),
        os.path.join(BASE_DIR, "models", "canciones.csv"),
        "canciones.csv",
        "Semana 9/canciones.csv"
    ]
    for r in rutas:
        if os.path.exists(r):
            return pd.read_csv(r)
    return None

df_canciones = cargar_datos()
# -----------------------------------------------------------------------------
# Configuración de la Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Panel de Predicción & Análisis Musical",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Estilos CSS Personalizados (Diseño Neón / Morado - Fucsia Coral)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Fondo principal */
    .stApp {
        background-color: #12011b;
        color: #ffffff;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Forzar visibilidad y legibilidad de textos */
    p, span, label, div, small, li, .stMarkdown {
        color: #f3e8ff !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Banner Superior */
    .header-banner {
        background: linear-gradient(90deg, #1f012e 0%, #320346 100%);
        padding: 16px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        border-bottom: 3px solid #ff007a;
    }
    .header-title {
        color: #ffffff !important;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0;
    }

    /* Tarjeta de Predicción */
    .prediction-card {
        background: linear-gradient(135deg, #ff007a 0%, #ff5e36 50%, #ff9e00 100%);
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(255, 0, 122, 0.35);
        margin-top: 10px;
    }
    .prediction-title {
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 2px;
        color: #ffffff !important;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .prediction-score {
        font-size: 52px;
        font-weight: 900;
        color: #ffffff !important;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .prediction-sub {
        font-size: 15px;
        color: #ffe8fa !important;
        margin-top: 6px;
        font-weight: 600;
    }

    /* Tarjetas de Contenido */
    .studio-card {
        background-color: #1e032b;
        border: 1px solid #ff007a;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .studio-card-title {
        color: #ff5e36 !important;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 14px;
        border-bottom: 1px solid #3d0754;
        padding-bottom: 8px;
    }

    /* FIX COMPLETO PARA SELECTBOX / MENÚS DESPLEGABLES */
    div[data-baseweb="select"] > div {
        background-color: #250338 !important;
        border-color: #ff007a !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="popover"], 
    div[data-baseweb="menu"], 
    ul[role="listbox"] {
        background-color: #250338 !important;
        border: 1px solid #ff007a !important;
    }
    li[role="option"], div[role="option"] {
        background-color: #250338 !important;
        color: #ffffff !important;
    }
    li[role="option"]:hover, 
    div[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #ff007a !important;
        color: #ffffff !important;
    }

    /* Inputs y Sliders */
    .stTextInput input, .stNumberInput input {
        background-color: #250338 !important;
        color: #ffffff !important;
        border: 1px solid #ff007a !important;
        border-radius: 8px !important;
    }
    .stSelectbox label, .stSlider label, .stTextInput label, .stNumberInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Carga de Recursos (Modelo y Datos)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE_DIR, "modelo.pkl")

@st.cache_resource
def cargar_modelo():
    if os.path.exists(RUTA_MODELO):
        return joblib.load(RUTA_MODELO)
    else:
        # Si no lo encuentra directo en la carpeta actual, intenta en 'models/modelo.pkl'
        ruta_models = os.path.join(BASE_DIR, "models", "modelo.pkl")
        if os.path.exists(ruta_models):
            return joblib.load(ruta_models)
        return None

# Cargar el artefacto
artefacto = cargar_modelo()

if artefacto is None:
    st.error("⚠️ No se encontró 'modelo.pkl'. Verifica la presencia del archivo.")
else:
    # Si el pkl es un diccionario empaquetado, extraemos el pipeline
    if isinstance(artefacto, dict):
        modelo = artefacto.get("pipeline")
        ficha = artefacto.get("ficha", {})
        columnas_entrada = artefacto.get("columnas_entrada", [])
    else:
        modelo = artefacto

# -----------------------------------------------------------------------------
# Banner Superior
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <div class="header-title">🎵 Panel de Predicción & Análisis Musical</div>
</div>
""", unsafe_allow_html=True)

if modelo is None:
    st.error("⚠️ No se encontró 'modelo_popularidad.pkl'. Verifica la presencia del archivo.")
    st.stop()

# -----------------------------------------------------------------------------
# Controles de Entrada (Generación Reactiva en Tiempo Real)
# -----------------------------------------------------------------------------
col_meta, col_audio1, col_audio2, col_audio3 = st.columns([1.2, 1, 1, 1])

with col_meta:
    artista_input = st.text_input("Artista / Grupo", value="Artista Demo")
    titulo_input = st.text_input("Nombre de la Canción", value="Midnight Pulse")
    anio_input = st.number_input("Año de Lanzamiento", min_value=1950, max_value=2030, value=2023)
    
    generos = ["hip-hop", "pop", "rock", "reggaeton", "latin", "indie", "electronic", "jazz", "metal", "classical"]
    genero_input = st.selectbox("Género Musical", options=generos, index=0)
    colab_input = 0

with col_audio1:
    energia = st.slider("Energía (0.0 a 1.0)", 0.0, 1.0, 0.80, 0.01)
    valencia = st.slider("Valencia / Positividad", 0.0, 1.0, 0.60, 0.01)
    acustica = st.slider("Acústica (0.0 a 1.0)", 0.0, 1.0, 0.20, 0.01)

with col_audio2:
    bailabilidad = st.slider("Bailabilidad", 0.0, 1.0, 0.75, 0.01)
    instrumentalidad = 0.05  # atributo reservado para la interfaz
    duracion_min = st.number_input("Duración (minutos)", min_value=1.0, max_value=10.0, value=3.50, step=0.1)

with col_audio3:
    liveness = 0.15
    speechiness = 0.10
    tempo = 120.0
    volumen_db = st.slider("Volumen (dB)", -30.0, 0.0, -6.0, 0.5)

# -----------------------------------------------------------------------------
# Inferencia del Modelo ML en Tiempo Real
# -----------------------------------------------------------------------------
duracion_c2 = (duracion_min - 3.5) ** 2
datos_entrada = pd.DataFrame([{
    'bailabilidad': bailabilidad,
    'energia': energia,
    'valencia': valencia,
    'acustica': acustica,
    'tempo': tempo,
    'duracion_min': duracion_min,
    'duracion_c2': duracion_c2,
    'volumen_db': volumen_db,
    'anio': anio_input,
    'colaboracion': colab_input,
    'genero': genero_input
}])

prediccion_raw = modelo.predict(datos_entrada)[0]
pred_clamped = round(float(np.clip(prediccion_raw, 0, 100)), 1)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Resultados e Radar Sonoro
# -----------------------------------------------------------------------------
col_res1, col_res2 = st.columns([1, 2])

with col_res1:
    st.markdown(f"""
    <div class="prediction-card">
        <div class="prediction-title">Popularidad Predicha</div>
        <div class="prediction-score">{pred_clamped} <span style="font-size:24px;">/ 100</span></div>
        <div class="prediction-sub">"{titulo_input}"</div>
        <div class="prediction-sub" style="opacity:0.85;">Artista: {artista_input} | {genero_input}</div>
    </div>
    """, unsafe_allow_html=True)

with col_res2:
    st.markdown("### 💥 Perfil Musical de la Canción Evaluada")
    
    categorias = ['Energía', 'Bailabilidad', 'Valencia', 'Acústica']
    valores = [energia, bailabilidad, valencia, acustica]
    
    categorias_plot = categorias + [categorias[0]]
    valores_plot = valores + [valores[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=valores_plot,
        theta=categorias_plot,
        fill='toself',
        fillcolor='rgba(255, 0, 122, 0.35)',
        line=dict(color='#ff5e36', width=3),
        name=titulo_input
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, linecolor="#ff007a"),
            angularaxis=dict(tickfont=dict(size=12, color="#ffffff"), linecolor="#ff007a"),
            bgcolor="#12011b"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=20, b=20),
        font=dict(color="#ffffff")
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# -----------------------------------------------------------------------------
# MÓDULO DE GRÁFICAS COMPARATIVAS (Contexto del Dataset)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📈 Posicionamiento Analítico de la Canción")

if df_canciones is not None:
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown('<div class="studio-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="studio-card-title">Posición Predicha ({pred_clamped} pts) en la Distribución</div>', unsafe_allow_html=True)
        
        fig_hist = px.histogram(
            df_canciones, 
            x="popularidad", 
            nbins=30,
            color_discrete_sequence=['#320346']
        )
        
        # Marcador de la canción evaluada
        fig_hist.add_vline(
            x=pred_clamped, 
            line_dash="solid", 
            line_color="#ff007a", 
            line_width=4,
            annotation_text=f" Tu Canción: {pred_clamped}",
            annotation_position="top right",
            annotation_font=dict(size=13, color="#ff007a")
        )
        
        mean_pop = df_canciones['popularidad'].mean()
        fig_hist.add_vline(
            x=mean_pop, 
            line_dash="dash", 
            line_color="#ff9e00", 
            line_width=2,
            annotation_text=f" Media: {mean_pop:.1f}",
            annotation_position="top left",
            annotation_font=dict(size=12, color="#ff9e00")
        )

        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e032b",
            font=dict(color="#ffffff"),
            xaxis=dict(title="Popularidad (0 - 100)", gridcolor="#3d0754", color="#ffffff"),
            yaxis=dict(title="Frecuencia", gridcolor="#3d0754", color="#ffffff")
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g_col2:
        st.markdown('<div class="studio-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="studio-card-title">Línea de Tiempo por Año ({anio_input})</div>', unsafe_allow_html=True)
        
        conteo_anios = df_canciones.groupby('anio').size().reset_index(name='canciones')
        colors = ['#320346' if a != anio_input else '#ff007a' for a in conteo_anios['anio']]

        fig_bar = go.Figure(data=[
            go.Bar(
                x=conteo_anios['anio'], 
                y=conteo_anios['canciones'],
                marker_color=colors
            )
        ])
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e032b",
            font=dict(color="#ffffff"),
            xaxis=dict(title="Año de Lanzamiento", gridcolor="#3d0754", color="#ffffff"),
            yaxis=dict(title="N° de Canciones", gridcolor="#3d0754", color="#ffffff")
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Gráfica 3: Dispersión y Regresión
    st.markdown('<div class="studio-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="studio-card-title">Regresión Lineal: Bailabilidad ({bailabilidad}) vs Popularidad ({pred_clamped} pts)</div>', unsafe_allow_html=True)
    
    x_vals = df_canciones['bailabilidad'].values
    y_vals = df_canciones['popularidad'].values
    mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
    m, b = np.polyfit(x_vals[mask], y_vals[mask], 1)
    
    x_line = np.linspace(x_vals[mask].min(), x_vals[mask].max(), 100)
    y_line = m * x_line + b

    fig_scatter = go.Figure()
    
    # Dataset
    fig_scatter.add_trace(go.Scatter(
        x=x_vals, 
        y=y_vals, 
        mode='markers',
        marker=dict(color='#320346', opacity=0.5, size=6),
        name='Canciones Dataset'
    ))
    
    # Recta
    fig_scatter.add_trace(go.Scatter(
        x=x_line, 
        y=y_line, 
        mode='lines',
        line=dict(color='#ff5e36', width=3),
        name=f'Tendencia Lineal (y = {b:.1f} + {m:.1f}x)'
    ))
    
    # Estrella de la canción evaluada
    fig_scatter.add_trace(go.Scatter(
        x=[bailabilidad], 
        y=[pred_clamped], 
        mode='markers+text',
        marker=dict(color='#ff007a', size=18, symbol='star', line=dict(color='#ffffff', width=2)),
        text=[f"  <b>{titulo_input}</b> ({pred_clamped} pts)"],
        textposition="top right",
        textfont=dict(color='#ff007a', size=14),
        name=f"Tu Canción: {titulo_input}"
    ))

    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e032b",
        font=dict(color="#ffffff"),
        xaxis=dict(title="Bailabilidad", gridcolor="#3d0754", color="#ffffff"),
        yaxis=dict(title="Popularidad", gridcolor="#3d0754", color="#ffffff"),
        legend=dict(font=dict(color="#ffffff"))
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)