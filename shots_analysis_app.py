import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mplsoccer import Pitch
from itertools import product
import pickle
from scipy.stats import percentileofscore

from gk_charts import *
from model_functions import *

# Heatmaps size
bin_y = 6
bin_z = 4

buckets = {
    "Palo Corto 1": (88.5, 100, 21, 29),
    "Palo Corto 2": (88.5, 100, 29, 37),
    "PP 1": (83, 88.5, 21, 37),
    
    "Palo Corto 3": (88.5, 100, 63, 72),
    "Palo Corto 4": (88.5, 100, 72, 79),
    "PP 5": (83, 88.5, 63, 79),

    "PP 4": (83, 88.5, 54.33, 63),
    "2do Palo Área": (88.5, 94, 54.33, 63),
    "PP 3": (83, 88.5, 45.66, 54.33),
    "Zona Central": (88.5, 94, 45.66, 54.33),
    "PP 2": (83, 88.5, 37, 45.66),
    "1er Palo Área": (88.5, 94, 37, 45.66),

    "2do Palo": (94, 100, 54.33, 63),
    "Zona GK": (94, 100, 45.66, 54.33),
    "1er Palo": (94, 100, 37, 45.66),
    "Zona Lateral 1": (70, 83, 0, 21),
    "Zona Corner 1": (83, 100, 0, 21),
    "Zona Corner 2": (83, 100, 79, 100),
    "Zona Lateral 2": (70, 83, 79, 100),
    "Frontal 1": (70, 83, 21, 37),
    "Frontal 2": (70, 83, 37, 63),
    "Frontal 3": (70, 83, 63, 79)
}

# Título de la aplicación
st.title("⚽ ANÁLISIS PORTEROS ⚽")

# Cargar datos
df = pd.read_csv('3___model_predict_xg_xgot.csv')
df_new = df[(df['xgot']>0)&(df['NaPlayer_gk']!="0")]
df_new['date'] = pd.to_datetime(df_new['TsEvent']).dt.date

# Deflection Correction
# TODO: Eliminar y hacer correccion en el pipeline de databricks
df_new['xgot'] = np.where((df_new['Deflection']==-1)&(df_new['outcome']==1), 1, df_new['xgot'])
df_new['xgot'] = np.where(
    df_new['NaPlayer_gk'].isin(['Jan Oblak', 'Thibaut Courtois']),
    np.minimum(1.05 * df_new['xgot'], 1),
    df_new['xgot']
)

# Rango de fechas disponible
min_date = df_new['date'].min()
max_date = df_new['date'].max()

# Crear pestañas
tab1, tab2 = st.tabs(["Análisis Rendimiento Individual", "Optimización Posicionamiento"])

with tab1:
    st.subheader("🥅 GoalKeeper Analysis 🥅")

    st.markdown("🔍 Selección de Filtros")
    # Filtro de selección de portero
    selected_gk = st.selectbox("Selección de Portero para el análisis", df_new.sort_values('NaPlayer_gk')['NaPlayer_gk'].unique())

    # Selector de rango de fechas
    st.markdown("📅 **Selecciona el periodo de análisis**")
    date_range = st.slider(
        "Rango de Fechas:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    df_new = df_new[(df_new['date'] >= date_range[0]) & (df_new['date'] <= date_range[1])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        ### ONE ON ONE FILTER ###
        one_vs_one = st.checkbox("1 vs 1")

        # Aplicar el filtro si el checkbox está activado
        if one_vs_one:
            df_new = df_new[df_new["1_on_1"] == 1]
        
        ### REGULTAR PLAY FILTER ###
        
        filter_regular_play = st.checkbox("Regular Play")
        filter_penalty = st.checkbox("Penalty")
        filter_corner = st.checkbox("Corner")
        filter_free_kick = st.checkbox("Free Kick")
        filter_assisted = st.checkbox("Assisted")
        filter_individual_play = st.checkbox("Individual Play")
        filter_big_chance = st.checkbox("Big Chance")

        # Lista para almacenar condiciones de filtro
        conditions_play = []

        # Agregar condiciones según los checkboxes seleccionados
        if filter_regular_play:
            conditions_play.append(df_new["Regular_play"] == 1)
        if filter_penalty:
            conditions_play.append(df_new["Penalty"] == 1)
        if filter_corner:
            conditions_play.append(df_new["From_corner"] == 1)
        if filter_free_kick:
            conditions_play.append(df_new["Free_kick"] == 1)
        if filter_assisted:
            conditions_play.append(df_new["Assisted"] == 1)
        if filter_individual_play:
            conditions_play.append(df_new["Individual_Play"] == 1)
        if filter_big_chance:
            conditions_play.append(df_new["Big_Chance"] == 1)

        # Aplicar filtro si hay alguna condición seleccionada
        if conditions_play:
            df_new = df_new[pd.concat(conditions_play, axis=1).any(axis=1)]
        
        
    with col2:

        # 1. Obtener valores únicos de IdSeason
        season_options = df_new['IdSeason'].unique()

        # 2. Crear un multiselect en Streamlit
        selected_seasons = st.multiselect("Selecciona Temporadas", season_options, default=season_options)

        # 3. Filtrar el DataFrame con los valores seleccionados
        df_new = df_new[df_new['IdSeason'].isin(selected_seasons)]

        # Crear checkboxes para cada opción
        filter_small_box = st.checkbox("Small Box")
        filter_box = st.checkbox("Box")
        filter_out_box = st.checkbox("Out of Box")

        # Lista para almacenar condiciones de filtro
        conditions = []

        # Agregar condiciones según los checkboxes seleccionados
        if filter_small_box:
            conditions.append(df_new["Small_box"] == 1)
        if filter_box:
            conditions.append(df_new["box"] == 1)
        if filter_out_box:
            conditions.append(df_new["out_of_box"] == 1)

        # Aplicar filtro si hay alguna condición seleccionada
        if conditions:
            df_new = df_new[pd.concat(conditions, axis=1).any(axis=1)]


    # 4. Mostrar un mensaje si no hay datos
    if df_new.empty:
        st.warning("No hay datos para las temporadas seleccionadas.")
    else:
        st.write(f"Se analizarán {len(df_new[(df_new['NaPlayer_gk'] == selected_gk) & (df_new['NaEventType'] != "Miss")])} eventos de las temporadas seleccionadas.")
    
    # Filtrar datos por portero seleccionado
    df_filtered = df_new[(df_new['NaPlayer_gk'] == selected_gk) & (df_new['NaEventType'] != "Miss")]

    # Aplicar la clasificación a las coordenadas del tiro
    df_filtered["pitch_zone_shot"] = df_filtered.apply(
        lambda row: classify_pitch_zone_dynamic(row["x"], row["y"]), axis=1
    )

    # Calcular métricas
    total_shots = df_filtered[df_filtered['NaEventType'].isin(["Goal", "Attempt Saved", "Post"])].shape[0]
    total_goals = df_filtered[df_filtered['NaEventType'] == "Goal"].shape[0]
    total_saves = df_filtered[df_filtered['NaEventType'] == "Attempt Saved"].shape[0]
    effectiveness = total_saves / total_shots if total_shots > 0 else 0
    total_performance = df_filtered["xgot"].sum() - total_goals

    # 2. Contar partidos por portero
    partidos_por_gk = df_new.groupby('NaPlayer_gk')['date'].nunique()

    # 3. Filtrar porteros con al menos 20 partidos
    gks_validos = partidos_por_gk[partidos_por_gk >= 20].index

    # 4. Calcular total_performance por portero
    performance_por_gk = (
        df_new[df_new['NaPlayer_gk'].isin(gks_validos)]
        .groupby('NaPlayer_gk')
        .apply(lambda g: g['xgot'].sum() - (g['NaEventType'] == 'Goal').sum())
    )

    selected_performance = total_performance  # ya lo calculaste antes para el portero actual
    percentil = percentileofscore(performance_por_gk.values, selected_performance)
    
    # Rendimiento global del portero
    st.markdown(
    "<h2 style='text-align: center; color: black;'>🥅 RENDIMIENTO GLOBAL DEL PORTERO </h2>",
    unsafe_allow_html=True)

    # Gráfico swarmplot en Streamlit
    fig_swarm = plot_total_goles_prevenidos(df_new, selected_gk)
    st.pyplot(fig_swarm)

    st.markdown("📈 Evolución del Rendimiento")
    x_axis_option = st.radio(
        "Elegir eje X:",
        ('fecha', 'equipo', 'fecha_equipo'),
        format_func=lambda x: {'fecha':'Por Fecha', 'equipo':'Por Equipo', 'fecha_equipo':'Por Fecha y Equipo'}[x]
    )

    fig = plot_goals_vs_xgot(df_filtered, x_axis=x_axis_option)
    st.pyplot(fig)

    
    
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
        st.metric(label="Percentil del Portero", value=f"{percentil:.1f} %")
    
    # Rendimiento en el arco
    st.markdown(
    "<h2 style='text-align: center; color: black;'>🥅 DISTRIBUCIÓN EN EL ARCO </h2>",
    unsafe_allow_html=True)
    
    # Crear una disposición en columnas para mostrar los gráficos en la misma fila
    col1, col2 = st.columns(2)
    
    with col1:
        # Generar y mostrar el gráfico del análisis del portero
        fig = plot_goalkeeper_analysis(df_filtered)
        st.pyplot(fig)

        fig = plot_event_heatmap(df_filtered[df_filtered['NaEventType']=="Attempt Saved"], "Atajadas", bin_y, bin_z, "Greens")
        st.pyplot(fig)

    with col2:
        fig = plot_performance_heatmap(df_filtered, bin_y, bin_z)
        st.pyplot(fig)

        fig = plot_event_heatmap(df_filtered[df_filtered['NaEventType']=="Goal"], "Goles", bin_y, bin_z, "Reds")
        st.pyplot(fig)
    
    # Rendimiento en el arco
    st.markdown(
    "<h2 style='text-align: center; color: black;'>🥅 POSICIONAMIENTO EN EL CAMPO </h2>",
    unsafe_allow_html=True)
    
    fig_gk_perf_map = plot_gk_performance_map(df_filtered)
    st.pyplot(fig_gk_perf_map)

    # Crear una disposición en columnas para mostrar los gráficos en la misma fila
    col1, col2 = st.columns(2)
    
    with col1:
        fig_gk_perf_map = plot_gk_saves_map(df_filtered[df_filtered['NaEventType']=="Goal"], "Goles", cmap_name="Reds")
        st.pyplot(fig_gk_perf_map)

        fig_gk_kdeplot = plot_gk_kde(df_filtered[df_filtered['NaEventType']=="Goal"], "Goles", cmap_name="Reds")
        st.pyplot(fig_gk_kdeplot)

    
    with col2:
        fig_gk_perf_map = plot_gk_saves_map(df_filtered[df_filtered['NaEventType']=="Attempt Saved"], "Atajadas", cmap_name="Greens")
        st.pyplot(fig_gk_perf_map)

        fig_gk_kdeplot = plot_gk_kde(df_filtered[df_filtered['NaEventType']=="Attempt Saved"], "Atajadas", cmap_name="Greens")
        st.pyplot(fig_gk_kdeplot)
    
    # Rendimiento en el arco
    st.markdown(
    "<h2 style='text-align: center; color: black;'>🥅 LOCACIÓN DEL DISPARO </h2>",
    unsafe_allow_html=True)
    
    # Crear una disposición en columnas para mostrar los gráficos en la misma fila
    col1, col2 = st.columns(2)
    
    with col1:
        # Generar y mostrar el gráfico del mapa de disparos
        fig_prob_shot_map = plot_shot_map(df_filtered)
        st.pyplot(fig_prob_shot_map)
    
    with col2:
        fig_shot_map = plot_goal_vs_miss(df_filtered)
        st.pyplot(fig_shot_map)    


# Funciones que ya definimos antes:
# get_nearest_shots() y plot_with_gk_heatmap_scaled()

# -------------------------------
# TAB 2 IMPLEMENTACIÓN
# -------------------------------
with tab2:
    st.header("Análisis de Posicionamiento del Portero")

    # Sliders para posición del jugador
    x_player = st.slider("Posición X del jugador", 70, 100, 90)
    y_player = st.slider("Posición Y del jugador", 15, 85, 50)

    # Filtros para el DataFrame
    st.subheader("Filtros de Jugada")
    col1, col2 = st.columns(2)

    with col1:
        filter_big_chance = st.checkbox("Big Chance Situations")
        filter_one_on_one = st.checkbox("1 vs 1 Situations")
        filter_saves = st.checkbox("Solo Atajadas")

    with col2:
        st.write("Parte del cuerpo")
        filter_right = st.checkbox("Pie Derecho")
        filter_left = st.checkbox("Pie Izquierdo")
        filter_head = st.checkbox("Cabeza")
    
    # Aplicar filtros dinámicos
    df_filtered = df.copy()

    if filter_big_chance:
        df_filtered = df_filtered[df_filtered['Big_Chance'] == 1]
    if filter_one_on_one:
        df_filtered = df_filtered[df_filtered['1_on_1'] == 1]
    if filter_saves:
        df_filtered = df_filtered[df_filtered['NaEventType'] == "Attempt Saved"]

    body_filters = []
    if filter_right:
        body_filters.append('Right_footed')
    if filter_left:
        body_filters.append('Left_footed')
    if filter_head:
        body_filters.append('Head')

    if body_filters:
        df_filtered = df_filtered[df_filtered[body_filters].sum(axis=1) > 0]

    st.write(f"Total jugadas después de filtros: **{len(df_filtered)}**")

    if len(df_filtered) == 0:
        st.warning("No hay datos para los filtros seleccionados.")
    else:
        # Calcular N como 5% del dataset filtrado
        N = max(10, round(0.05 * len(df_filtered)))

        if st.button("Generar análisis", key="btn_generate"):
            
            nearest = get_nearest_shots(df_filtered, x_player, y_player, N=N)
            
            fig, metrics = plot_with_gk_heatmap_scaled(x_player, y_player, nearest, side='right')

            # Mostrar métricas arriba del gráfico
            st.subheader("Métricas calculadas")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Dist GK → Bisectriz", f"{metrics['Dist GK → Bisectriz']:.2f} m")
                st.metric("Dist Jugador → GK", f"{metrics['Dist Jug → GK']:.2f} m")
            with col2:
                st.metric("Dist GK → Recta Centro Arco", f"{metrics['Dist GK → Jug-Centro']:.2f} m")
                st.metric("Dist GK → Centro Arco", f"{metrics['Dist GK → Centro Arco']:.2f} m")

            # Crear 3 columnas: vacío - gráfico - vacío
            col1, col2 = st.columns([1, 1])  # col central más grande
            with col1:
                st.pyplot(fig)
            
            with col2:
                fig_gk_scatter = plot_with_gk_scatter_scaled(x_player, y_player, nearest, side='right')
                st.pyplot(fig_gk_scatter)