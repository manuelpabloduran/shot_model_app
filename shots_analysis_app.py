import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gk_charts import *
from mplsoccer import Pitch

# Heatmaps size
bin_y = 6
bin_z = 3

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

        fig = plot_event_heatmap(df_filtered, 'Attempt Saved', "Atajadas", bin_y, bin_z, "Greens")
        st.pyplot(fig)
        
        # Generar y mostrar el gráfico del mapa de disparos
        fig_prob_shot_map = plot_shot_map(df_filtered)
        st.pyplot(fig_prob_shot_map)

    with col2:
        fig = plot_performance_heatmap(df_filtered, bin_y, bin_z)
        st.pyplot(fig)

        fig = plot_event_heatmap(df_filtered, 'Goal', "Goles", bin_y, bin_z, "Reds")
        st.pyplot(fig)
        
        fig_shot_map = plot_goal_vs_miss(df_filtered)
        st.pyplot(fig_shot_map)    

with tab2:
    st.subheader("Historical Shot Analysis")
    
    # Crear un campo de fútbol interactivo
    st.markdown("### Selecciona una posición en el campo")
    pitch = Pitch(pitch_type='opta', line_color='black')
    fig, ax = pitch.draw(figsize=(10, 6))
    
    # Primera fila: Posición del jugador
    st.markdown("#### Posicion Jugador")
    col1, col2 = st.columns(2)
    with col1:
        player_x = st.slider("Posición X del jugador", 0, 100, 50)
    with col2:
        player_y = st.slider("Posición Y del jugador", 0, 100, 50)

    # Segunda fila: Posición del portero
    st.markdown("#### Posicion Portero")
    col3, col4 = st.columns(2)
    with col3:
        gk_x = st.slider("Posición X del portero", 0, 100, 50)
    with col4:
        gk_y = st.slider("Posición Y del portero", 0, 100, 50)
    
    # Palos del arco
    x_goal = 100
    y_post1 = 45.2
    y_post2 = 54.8

    # Cálculo de variables
    distancia_tiro = np.sqrt((x_goal - player_x) ** 2 + (50 - player_y) ** 2)
    angulo_palo1 = np.arctan2(y_post1 - player_y, x_goal - player_x)
    angulo_palo2 = np.arctan2(y_post2 - player_y, x_goal - player_x)
    angulo_vision_arco = np.abs(angulo_palo2 - angulo_palo1)
    gk_distance_to_player = np.sqrt((player_x - gk_x) ** 2 + (player_y - gk_y) ** 2)
    gk_distance_to_goal = np.sqrt((gk_x - x_goal) ** 2 + (gk_y - 50) ** 2)
    y_gk_distance_to_y_player = gk_y - player_y

    # Mostrar métricas calculadas
    st.markdown("### 📊 Cálculos de Variables")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Distancia del Tiro", value=f"{distancia_tiro:.2f} m")
        st.metric(label="Ángulo Visión al Arco", value=f"{np.degrees(angulo_vision_arco):.2f}°")

    with col2:
        st.metric(label="Distancia GK a Jugador", value=f"{gk_distance_to_player:.2f} m")
        st.metric(label="Distancia GK a Arco", value=f"{gk_distance_to_goal:.2f} m")

    with col3:
        st.metric(label="Distancia Y GK vs Jugador", value=f"{y_gk_distance_to_y_player:.2f} m")

    # Dibujar las posiciones
    ax.scatter(player_x, player_y, color='blue', s=200, label='Jugador')
    ax.scatter(gk_x, gk_y, color='red', s=200, label='Portero')

    # Dibujar el tiro (flecha)
    pitch.arrows(player_x, player_y, x_goal, 50, color="blue", ax=ax, width=2, headwidth=10, headlength=10, label="Tiro")

    # Dibujar el área del ángulo de visión
    vision_area = [(player_x, player_y), (x_goal, y_post1), (x_goal, y_post2)]
    polygon = plt.Polygon(vision_area, color="gray", alpha=0.3, label="Ángulo de visión")
    ax.add_patch(polygon)

    ax.legend()
    st.pyplot(fig)
