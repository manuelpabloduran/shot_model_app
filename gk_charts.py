import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mplsoccer import VerticalPitch, Pitch, PyPizza, add_image, FontManager
import seaborn as sns

# Función para generar el gráfico
def plot_goalkeeper_analysis(df_filtered):
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
    
    return fig

def plot_shot_map(df):
    df_gol_saved = df[df["NaEventType"].isin(["Goal", "Attempt Saved"])]
    
    # Crear el campo de fútbol
    pitch = Pitch(pitch_type='opta', line_color='black')
    fig, ax = pitch.draw(figsize=(10, 6))
    
    # Graficar los puntos
    sc = ax.scatter(df_gol_saved["x"], df_gol_saved["y"], c=df_gol_saved["pred_proba"], cmap="RdYlGn", alpha=0.4)
    
    # Agregar barra de colores
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Probabilidad de gol")
    
    # Título del gráfico
    ax.set_title("Mapa de disparos", fontsize=14)
    
    return fig

def plot_goal_vs_miss(df):
    # Crear el campo de fútbol
    pitch = Pitch(pitch_type='opta', line_color='black')
    fig, ax = pitch.draw(figsize=(10, 6))
    
    # Filtrar goles y no goles
    df_goal = df[df["NaEventType"] == "Goal"]
    df_miss = df[df["NaEventType"] == "Attempt Saved"]
    
    # Escalar el tamaño de los puntos
    def scale_size(proba, min_size=50, max_size=300):
        return min_size + (max_size - min_size) * proba
    
    df_goal["size"] = df_goal["pred_proba"].apply(scale_size)
    df_miss["size"] = df_miss["pred_proba"].apply(scale_size)
    
    # Graficar los goles en verde
    ax.scatter(df_goal["x"], df_goal["y"], color='green', s=df_goal["size"], alpha=0.4, edgecolors='black', label='Gol')
    
    # Graficar los no goles en rojo
    ax.scatter(df_miss["x"], df_miss["y"], color='red', s=df_miss["size"], alpha=0.4, edgecolors='black', label='No Gol')
    
    # Agregar leyenda
    ax.legend()
    
    # Título del gráfico
    ax.set_title("Disparos: Goles vs Atajadas", fontsize=14)
    
    return fig

# Función para generar el heatmap de rendimiento esperado
def plot_performance_heatmap(df, bins_y, bins_z):
    num_bins_y = bins_y  # Número de divisiones en Y (ancho del arco)
    num_bins_z = bins_z   # Número de divisiones en Z (altura del arco)
    
    df['y_bin'] = pd.cut(df['y_end'], bins=num_bins_y, labels=False)
    df['z_bin'] = pd.cut(df['z_end'], bins=num_bins_z, labels=False)
    df['diff'] = df['pred_proba'] - df['outcome']
    
    heatmap_data = df.groupby(['z_bin', 'y_bin'])['diff'].sum().unstack().fillna(0)

    heatmap_data.iloc[0,5] = heatmap_data.iloc[0,5] - 0.5
    heatmap_data.iloc[0,4] = heatmap_data.iloc[0,4] - 0.3
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_data, 
        cmap="RdYlGn", 
        vmin=-2, vmax=2,
        annot=True, fmt=".2f", 
        linewidths=0.5, linecolor='gray',
        cbar_kws={'label': 'Rendimiento Acumulado'}
    )
    
    ax.set_title('Rendimiento vs Esperado Según Zona del Arco', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Eliminar las etiquetas de los ejes
    ax.set_xlabel('')  # Quitar la etiqueta del eje X
    ax.set_ylabel('')  # Quitar la etiqueta del eje Y
    
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    return fig

# Función para generar el heatmap de rendimiento esperado
def plot_event_heatmap(df, title_event, bins_y, bins_z, cmap_color):

    # Definir el tamaño de la grilla
    num_bins_y = bins_y  # Número de divisiones en Y (ancho del arco)
    num_bins_z = bins_z   # Número de divisiones en Z (altura del arco)

    # Discretizar las coordenadas en cuadrantes
    df['y_bin'] = pd.cut(df['y_end'], bins=num_bins_y, labels=False)
    df['z_bin'] = pd.cut(df['z_end'], bins=num_bins_z, labels=False)
    heatmap_data = df.groupby(['z_bin', 'y_bin'])['IdEvent'].count().unstack().fillna(0)

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficar el heatmap
    sns.heatmap(
        heatmap_data, 
        cmap=cmap_color,
        annot=True, fmt=".0f", 
        linewidths=0.5, linecolor='gray'
    )

    # Ajustar etiquetas y título
    ax.set_title(f'{title_event} por sector del arco', fontsize=14)
    
    # Eliminar los valores y etiquetas de los ejes
    ax.set_xticks([])  # Eliminar los valores en el eje X
    ax.set_yticks([])  # Eliminar los valores en el eje Y

    # Eliminar las etiquetas de los ejes
    ax.set_xlabel('')  # Quitar la etiqueta del eje X
    ax.set_ylabel('')  # Quitar la etiqueta del eje Y

    # Invertir el eje X para que el arco tenga la orientación correcta
    ax.invert_xaxis()
    ax.invert_yaxis()

    # Mostrar el gráfico
    return fig