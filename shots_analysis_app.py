import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gk_charts import *
from mplsoccer import Pitch

# Título de la aplicación
st.title("⚽ Shot Analysis ⚽")

# Cargar datos
df = pd.read_csv('historical_shot_model_pred.csv')

# Crear pestañas
tab1, tab2 = st.tabs(["GoalKeeper Analysis", "Historical Shot Analysis"])

with tab1:
    st.subheader("🥅 GoalKeeper Analysis 🥅")
    
    # Filtro de selección de portero
    selected_gk = st.selectbox("Selección de Portero para el análisis", df.sort_values('NaPlayer_gk')['NaPlayer_gk'].unique())
    
    # Filtrar datos por portero seleccionado
    df_filtered = df[df['NaPlayer_gk'] == selected_gk]

    # Calcular métricas
    total_shots = df_filtered[df_filtered['NaEventType'].isin(["Goal", "Attempt Saved", "Post"])].shape[0]
    total_goals = df_filtered[df_filtered['NaEventType'] == "Goal"].shape[0]
    total_saves = df_filtered[df_filtered['NaEventType'] == "Attempt Saved"].shape[0]
    effectiveness = total_saves / total_shots if total_shots > 0 else 0
    total_performance = df_filtered["pred_proba"].sum() - total_goals
    
    # Mostrar métricas
    st.markdown("📈 Estadísticas Generales")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Total Disparos", value=total_shots)
        st.metric(label="Total Goles", value=total_goals)
    
    with col2:
        st.metric(label="Total Intentos Salvados", value=total_saves)
        st.metric(label="Eficacia (%)", value=f"{effectiveness:.2%}")
    
    with col3:
        st.metric(label="Rendimiento Real vs Esperado", value=f"{total_performance:.2f}")
    
    # Crear una disposición en columnas para mostrar los gráficos en la misma fila
    col1, col2 = st.columns(2)
    
    with col1:
        # Generar y mostrar el gráfico del análisis del portero
        fig = plot_goalkeeper_analysis(df_filtered)
        st.pyplot(fig)

        fig = plot_performance_heatmap(df_filtered, 'Attempt Saved', "Atajadas", 4, 3, "Greens")
        st.pyplot(fig)
        
        # Generar y mostrar el gráfico del mapa de disparos
        fig_prob_shot_map = plot_shot_map(df_filtered)
        st.pyplot(fig_prob_shot_map)

    with col2:
        fig = plot_performance_heatmap(df, 4, 3)
        st.pyplot(fig)

        fig = plot_performance_heatmap(df_filtered, 'Goal', "Goles", 4, 3, "Reds")
        st.pyplot(fig)
        
        fig_shot_map = plot_goal_vs_miss(df_filtered)
        st.pyplot(fig_shot_map)    

with tab2:
    st.subheader("Historical Shot Analysis")