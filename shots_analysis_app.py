import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gk_charts import *
from mplsoccer import Pitch

# Título de la aplicación
st.title("Shot Analysis")

# Cargar datos
df = pd.read_csv('historical_shot_model_pred.csv')

# Función para generar el mapa de disparos
def plot_shot_map(df_model):
    # Crear el campo de fútbol
    pitch = Pitch(pitch_type='opta', line_color='black')
    fig, ax = pitch.draw(figsize=(10, 6))
    
    # Graficar los puntos
    sc = ax.scatter(df_model["x"], df_model["y"], c=df_model["pred_proba"], cmap="RdYlGn", alpha=0.4)
    
    # Agregar barra de colores
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Probabilidad de gol")
    
    # Título del gráfico
    ax.set_title("Mapa de disparos", fontsize=14)
    
    return fig

# Crear pestañas
tab1, tab2 = st.tabs(["GoalKeeper Analysis", "Historical Shot Analysis"])

with tab1:
    st.subheader("GoalKeeper Analysis")
    
    # Filtro de selección de portero
    selected_gk = st.selectbox("Selección de Portero para el análisis", df['NaPlayer_gk'].unique())
    
    # Filtrar datos por portero seleccionado
    df_filtered = df[df['NaPlayer_gk'] == selected_gk]
    
    # Crear una disposición en columnas para mostrar los gráficos en la misma fila
    col1, col2 = st.columns(2)
    
    with col1:
        # Generar y mostrar el gráfico del análisis del portero
        fig = plot_goalkeeper_analysis(df_filtered)
        st.pyplot(fig)
        
        # Generar y mostrar el gráfico del mapa de disparos
        fig_prob_shot_map = plot_shot_map(df_filtered)
        st.pyplot(fig_prob_shot_map)

    with col2:
        fig = plot_performance_heatmap(df_filtered)
        st.pyplot(fig)
        
        fig_shot_map = plot_goal_vs_miss(df_filtered)
        st.pyplot(fig_shot_map)    

with tab2:
    st.subheader("Historical Shot Analysis")