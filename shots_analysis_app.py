import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Título de la aplicación
st.title("Shot Analysis")

# Cargar datos
df = pd.read_csv('xgot_model_version_17_02.csv')

# Crear pestañas
tab1, tab2 = st.tabs(["GoalKeeper Analysis", "Historical Shot Analysis"])

with tab1:
    st.subheader("GoalKeeper Analysis")
    
    # Filtro de selección de portero
    selected_gk = st.selectbox("Selección de Portero para el análisis", df['NaPlayer_gk'].unique())
    
    # Filtrar datos por portero seleccionado
    df_filtered = df[df['NaPlayer_gk'] == selected_gk]
    
    # Definir los límites del arco
    y_post1 = 45.2
    y_post2 = 54.8
    z_min = 0
    z_max = 34.8
    
    # Cargar la imagen de fondo
    goal_img = Image.open("images/goal_fondo_2.jpg")
    
    # Filtrar eventos
    df_goal = df_filtered[df_filtered['NaEventType'] == "Goal"]
    df_saved = df_filtered[df_filtered['NaEventType'] == "Attempt Saved"]
    
    # Normalizar el tamaño de los puntos (entre 50 y 300 por visibilidad)
    def scale_size(proba, min_size=50, max_size=300):
        return min_size + (max_size - min_size) * proba
    
    df_goal['size'] = df_goal['pred_proba'].apply(scale_size)
    df_saved['size'] = df_saved['pred_proba'].apply(scale_size)
    
    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Mostrar la imagen de fondo ajustada a los límites del arco
    ax.imshow(goal_img, extent=[y_post1, y_post2, z_min, z_max], aspect='auto')
    
    # Graficar los goles en rojo con opacidad y tamaño según pred_proba
    ax.scatter(df_goal['y_end'], df_goal['z_end'], color='red', label='Gol', 
               s=df_goal['size'], edgecolors='black', alpha=0.6)
    
    # Graficar los intentos salvados en verde con opacidad y tamaño según pred_proba
    ax.scatter(df_saved['y_end'], df_saved['z_end'], color='green', label='Intento Desviado', 
               s=df_saved['size'] * 1.3, edgecolors='black', alpha=0.6)
    
    # Ajustar límites de los ejes
    ax.set_xlim(y_post1, y_post2)
    ax.set_ylim(z_min, z_max)
    
    # Eliminar los valores y etiquetas de los ejes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Ajustar etiquetas y título
    ax.set_title('Distribución de Goles y Paradas en el Arco', fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Agregar leyenda
    ax.legend()
    
    # Invertir el eje X para que coincida con la perspectiva del arco
    ax.invert_xaxis()
    
    # Mostrar gráfico en Streamlit
    st.pyplot(fig)

with tab2:
    st.subheader("Historical Shot Analysis")
    st.write("Aquí se pueden agregar más análisis históricos.")