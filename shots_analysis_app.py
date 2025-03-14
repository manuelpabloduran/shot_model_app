import streamlit as st
import pandas as pd
import os

# Título de la aplicación
st.title("Shot Analysis")

# Subtítulo
st.subheader("Estudio Histórico de Tiros y Estadísticas")

# Mensaje de bienvenida
st.write("Bienvenido a mi aplicación. Aquí puedes agregar más funcionalidades.")

# Importar archivos
def excel_total(pathCarpeta):
    files = [f for f in os.listdir(pathCarpeta) if f.endswith('.xlsx') or f.endswith('.xls')]
    dfs = [pd.read_excel(os.path.join(pathCarpeta, f), sheet_name='Eventos') for f in files]
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

league = 'Spain La Liga'
folder_path = f'/Partidos/{league}/24-25'
df = excel_total(f'{folder_path}')