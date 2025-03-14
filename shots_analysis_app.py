import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
    
    # Crear gráfico de dispersión
    fig, ax = plt.subplots()
    ax.scatter(df_filtered['x'], df_filtered['y'], alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Gráfico de dispersión de X vs Y para {selected_gk}")
    
    # Mostrar gráfico en Streamlit
    st.pyplot(fig)

with tab2:
    st.subheader("Historical Shot Analysis")
    st.write("Aquí se pueden agregar más análisis históricos.")
