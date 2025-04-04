import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mplsoccer import Pitch
from itertools import product
import pickle

from gk_charts import *
from model_functions import *

# Heatmaps size
dkslañdkasñld
bin_y = 2
bin_z = 3

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
st.title("⚽ Shot Analysis ⚽")

# Cargar datos
#df = pd.read_csv('historical_shot_model_pred.csv')
df_new = pd.read_csv('gk_shots_model_predictions.csv')

# Aplicar la clasificación a las coordenadas del tiro
df_new["pitch_zone_shot"] = df_new.apply(
    lambda row: classify_pitch_zone_dynamic(row["x"], row["y"]), axis=1
)

# Crear pestañas
tab1, tab2 = st.tabs(["GoalKeeper Analysis", "Historical Shot Analysis"])

with tab1:
    st.subheader("🥅 GoalKeeper Analysis 🥅")

    st.markdown("🔍 Selección de Filtros")
    # Filtro de selección de portero
    selected_gk = st.selectbox("Selección de Portero para el análisis", df_new.sort_values('NaPlayer_gk')['NaPlayer_gk'].unique())
    
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

        fig = plot_event_heatmap(df_filtered[df_filtered['NaEventType']=="Attempt Saved"], "Atajadas", bin_y, bin_z, "Greens")
        st.pyplot(fig)
        
        # Generar y mostrar el gráfico del mapa de disparos
        fig_prob_shot_map = plot_shot_map(df_filtered)
        st.pyplot(fig_prob_shot_map)

    with col2:
        fig = plot_performance_heatmap(df_filtered, bin_y, bin_z)
        st.pyplot(fig)

        fig = plot_event_heatmap(df_filtered[df_filtered['NaEventType']=="Goal"], "Goles", bin_y, bin_z, "Reds")
        st.pyplot(fig)
        
        fig_shot_map = plot_goal_vs_miss(df_filtered)
        st.pyplot(fig_shot_map)    
    
    fig_gk_perf_map = plot_gk_performance_map(df_filtered)
    st.pyplot(fig_gk_perf_map)

    # Crear una disposición en columnas para mostrar los gráficos en la misma fila
    col1, col2 = st.columns(2)
    
    with col1:
        fig_gk_perf_map = plot_gk_saves_map(df_filtered[df_filtered['NaEventType']=="Goal"], "Goles", cmap_name="Reds")
        st.pyplot(fig_gk_perf_map)
    
    with col2:
        fig_gk_perf_map = plot_gk_saves_map(df_filtered[df_filtered['NaEventType']=="Attempt Saved"], "Atajadas", cmap_name="Greens")
        st.pyplot(fig_gk_perf_map)

    

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
        player_x = st.slider("Posición X del jugador", 70, 100, 85)
    with col2:
        player_y = st.slider("Posición Y del jugador", 35, 65, 50)

    # Segunda fila: Posición del portero
    st.markdown("#### Posicion Portero")
    col3, col4 = st.columns(2)
    with col3:
        gk_x = st.slider("Posición X del portero", 80, 100, 95)
    with col4:
        gk_y = st.slider("Posición Y del portero", 35, 65, 50)
    
    # Validación: Si el portero no está en la visión, mostrar un warning
    if not check_gk_in_vision(player_x, player_y, gk_x, gk_y):
        st.warning("⚠️ El portero no está en la visión del jugador. Ajusta su posición para continuar.")
    else:
        st.success("✅")

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

        # Rango de valores para y_end y z_end
        y_end_values = np.arange(45.2, 54.8, 1)  # de 45.2 a 54.8 con paso de 1
        z_end_values = np.arange(0, 34.8, 1)  # de 0 a 34 con paso de 1

        # Generar todas las combinaciones posibles
        data = []
        for y_end, z_end in product(y_end_values, z_end_values):
            data.append([
                player_x, player_y, distancia_tiro, angulo_vision_arco, y_end, z_end,
                gk_x, gk_y, gk_distance_to_player, gk_distance_to_goal
            ])

        # Crear el DataFrame
        df_model = pd.DataFrame(data, columns=[
            'x', 'y', 'distancia_tiro', 'angulo_vision_arco', 'y_end_fixed', 'z_end_fixed',
            'GK_X_Coordinate', 'GK_Y_Coordinate', 'gk_distance_to_player', 'gk_distance_to_goal'
        ])

        df_model['x_end'] = 100
        df_model['Small_box'] = 0
        df_model['box'] = 1
        df_model['gk_in_vision'] = 1
        df_model['Penalty'] = 0
    
    st.markdown("#### Selección de Parámetros")
    col5, col6 = st.columns(2)
    with col5:
        # Lista de opciones
        play_types = ["Free_kick", "From_corner", "Regular_play"]

        # Selectbox para seleccionar una sola opción
        selected_play_type = st.selectbox("Selecciona tipo de jugada", play_types)

        # Lista de opciones
        assist_type = ["Individual_Play", "Intentional_assist"]

        # Selectbox para seleccionar una sola opción
        selected_situation_type = st.selectbox("Selecciona situación", assist_type)

        
        # Asignar 1 a la seleccionada y 0 al resto
        for play in play_types:
            df_model[play] = 1 if play == selected_play_type else 0
        
        for assist in assist_type:
            df_model[assist] = 1 if assist == selected_situation_type else 0
        
    with col6:
        ### ONE ON ONE FILTER ###
        one_vs_one_cb = st.checkbox("One vs One")

        # Aplicar el filtro si el checkbox está activado
        if one_vs_one_cb:
            df_model["1_on_1"] = 1
        else:
            df_model["1_on_1"] = 0

        # Lista de opciones
        body_part = ["Head", "Left_footed", "Right_footed"]

        # Selectbox para seleccionar una sola opción
        selected_body_type = st.selectbox("Selecciona Parte del cuerpo para definición", body_part)

        for body in body_part:
            df_model[body] = 1 if body == selected_body_type else 0

    

    # Aplicar la función a todo el DataFrame
    df_model['gk_dist_to_shot_line_proy'] = df_model.apply(
        lambda row: gk_distance_to_shot(row['x'], row['y'], row['x_end'], row['y_end_fixed'], row['GK_X_Coordinate'], row['GK_Y_Coordinate']), axis=1
    )

    df_model['gk_dist_to_shot_line_proy'] = df_model['gk_dist_to_shot_line_proy'].fillna(df_model['gk_dist_to_shot_line_proy'].max())

    features = ["x", "y", "distancia_tiro", "angulo_vision_arco", "y_end_fixed", "z_end_fixed", "gk_dist_to_shot_line_proy", 'GK_X_Coordinate', 'GK_Y_Coordinate', "gk_in_vision", "gk_distance_to_player", "gk_distance_to_goal", "Small_box", "box", "1_on_1", "Free_kick", "From_corner", "Head", "Individual_Play", "Intentional_assist", "Left_footed", "Right_footed", "Penalty"]
    
    print(df_model.columns)

    df_model = df_model[features]
    
    # Cargar el modelo
    with open("model_goal_proba_prediction_20250402.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    df_prediction = df_model.copy()

    df_prediction['pred'] = loaded_model.predict(df_model)
    df_prediction['model_proba'] = loaded_model.predict_proba(df_model)[:, 1]

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

    player_zone = classify_pitch_zone_dynamic(player_x, player_y)

    df_shot_zone = df_new[df_new['pitch_zone_shot']==player_zone]
    
    # Mostrar métricas calculadas
    st.markdown("### 📊 Probabilidades del tiro")
    # Crear columnas para los gráficos
    col1, col2 = st.columns(2)

    with col1:
        fig = plot_success_probability_heatmap(df_prediction, num_bins_y=18, num_bins_z=6)
        st.pyplot(fig)

        # Generar y mostrar el gráfico del análisis del portero
        fig = plot_goalkeeper_analysis(df_shot_zone)
        st.pyplot(fig)

        fig = plot_event_heatmap(df_shot_zone[df_shot_zone['NaEventType'].isin(['Attempt Saved', 'Post'])], "Errados", bin_y, bin_z, "Reds")
        st.pyplot(fig)
        
        # Generar y mostrar el gráfico del mapa de disparos
        fig_prob_shot_map = plot_goal_percentage_heatmap(df_shot_zone[df_shot_zone['NaEventType'].isin(['Goal'])], bin_y, bin_z, "Greens")
        st.pyplot(fig_prob_shot_map)

    with col2:
        fig = plot_interpolated_probability_contour(df_prediction, num_bins_y=18, num_bins_z=6)
        st.pyplot(fig)

        fig = plot_performance_heatmap(df_shot_zone, bin_y, bin_z)
        st.pyplot(fig)

        fig = plot_event_heatmap(df_shot_zone[df_shot_zone['NaEventType'].isin(['Attempt Saved', 'Miss', 'Post'])], "Errados (Corrección)", bin_y, bin_z, "Reds", y_col='y_end_fixed', z_col='z_end_fixed')
        st.pyplot(fig)

        fig = plot_event_heatmap(df_shot_zone[df_shot_zone['NaEventType'].isin(['Goal'])], "Goles", bin_y, bin_z, "Greens")
        st.pyplot(fig)