import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gk_charts import *

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
    
    # Generar y mostrar el gráfico
    fig = plot_goalkeeper_analysis(df_filtered)
    st.pyplot(fig)

with tab2:
    st.subheader("Historical Shot Analysis")
    st.write("Aquí se pueden agregar más análisis históricos.")