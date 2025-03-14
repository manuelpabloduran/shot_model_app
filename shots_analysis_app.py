import streamlit as st
import pandas as pd

# Título de la aplicación
st.title("Shot Analysis")

# Subtítulo
st.subheader("Estudio Histórico de Tiros y Estadísticas")

# Mensaje de bienvenida
st.write("Bienvenido a mi aplicación. Aquí puedes agregar más funcionalidades.")

df = pd.read_csv('xgot_model_version_17_02.csv')