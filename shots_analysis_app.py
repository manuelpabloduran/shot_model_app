import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Shot Analysis")

# Subtítulo
st.subheader("Estudio Histórico de Tiros y Estadísticas")

# Mensaje de bienvenida
st.write("Bienvenido a mi aplicación. Aquí puedes agregar más funcionalidades.")

df = pd.read_csv('xgot_model_version_17_02.csv')

# Crear gráfico de dispersión
fig, ax = plt.subplots()
ax.scatter(df['x'], df['y'], alpha=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Gráfico de dispersión de X vs Y")

# Mostrar gráfico en Streamlit
st.pyplot(fig)