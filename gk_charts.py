import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mplsoccer import VerticalPitch, Pitch, PyPizza, add_image, FontManager
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm

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

    heatmap_data.iloc[0,5] = heatmap_data.iloc[0,5] - 0.4
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

def plot_gk_performance_map(df_gk):
    """
    Genera un gráfico de rendimiento real vs esperado para el arquero.

    Parámetros:
    df_gk : DataFrame
        DataFrame con las columnas ['x_gk', 'y_gk', 'diff'].
    player : str
        Nombre del arquero a analizar.
    """

    # Definir las zonas del campo
    buckets_gk = {
        "Área Penalti 1": (88, 91, 54.33, 63),
        "Zona Penalti 2": (88, 91, 45.66, 54.33),
        "Área Penalti 3": (88, 91, 37, 45.66),
        "Área Penalti 4": (91, 94, 54.33, 63),
        "Zona Penalti 5": (91, 94, 45.66, 54.33),
        "Área Penalti 6": (91, 94, 37, 45.66),
        "Área Chica 1": (97, 100, 37, 45.66),
        "Área Chica 2": (97, 100, 45.66, 48.55),
        "Área Chica 3": (97, 100, 48.55, 51.44),
        "Área Chica 4": (97, 100, 51.44, 54.33),
        "Área Chica 5": (97, 100, 54.33, 63),
        "Área Chica 6": (94, 97, 37, 45.66),
        "Área Chica 7": (94, 97, 45.66, 54.33),
        "Área Chica 8": (94, 97, 54.33, 63)
    }

    # Clasificar las zonas del arquero
    def classify_pitch_zone(x, y):
        for zone, (x_min, x_max, y_min, y_max) in buckets_gk.items():
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return zone
        return "Fuera de zona"

    # Aplicar la clasificación al DataFrame
    df_gk["pitch_zone_gk"] = df_gk.apply(lambda row: classify_pitch_zone(row["x_gk"], row["y_gk"]), axis=1)

    # Calcular la suma acumulada de "diff" por zona
    zone_values = df_gk.groupby("pitch_zone_gk")["diff"].sum().to_dict()

    # Definir normalización de colores (centrado en 0)
    vmin, vmax = min(zone_values.values()), max(zone_values.values())
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Dibujar el campo
    pitch = Pitch(pitch_type='opta', pitch_color='white', line_color='black',
                  stripe=False, corner_arcs=True, goal_type='box', half=True)
    fig, ax = pitch.draw(figsize=(14, 10))

    # Dibujar cada zona en el campo
    for zone, (x_min, x_max, y_min, y_max) in buckets_gk.items():
        value = zone_values.get(zone, 0)  # Si no hay datos, asignar 0

        color = plt.cm.RdYlGn(norm(value))  # Color basado en el rendimiento
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                 linewidth=1.5, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)

        # Mostrar el valor dentro de la zona
        ax.text((x_min + x_max) / 2, (y_min + y_max) / 2, f"{value:.1f}",
                ha="center", va="center", fontsize=10, color="black")

    # Agregar barra de color
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Diferencia Rendimiento Real vs Esperado", fontsize=12)

    # Título del gráfico
    ax.set_title(f"Rendimiento Real vs Esperado", fontsize=15)

    return fig

import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm

def plot_gk_saves_map(df, name_event, cmap_name="Greens"):
    """
    Genera un gráfico de calor de atajadas del arquero por zona.

    Parámetros:
    df_gk : DataFrame
        DataFrame con las columnas ['x_gk', 'y_gk', 'NaEventType', 'IdEvent'].
    player : str
        Nombre del arquero a analizar.
    cmap_name : str, opcional
        Nombre del colormap de Matplotlib a utilizar (por defecto "Greens").
    """

    # Definir zonas del campo
    buckets_gk = {
        "Área Penalti 1": (88, 91, 54.33, 63),
        "Zona Penalti 2": (88, 91, 45.66, 54.33),
        "Área Penalti 3": (88, 91, 37, 45.66),
        "Área Penalti 4": (91, 94, 54.33, 63),
        "Zona Penalti 5": (91, 94, 45.66, 54.33),
        "Área Penalti 6": (91, 94, 37, 45.66),
        "Área Chica 1": (97, 100, 37, 45.66),
        "Área Chica 2": (97, 100, 45.66, 48.55),
        "Área Chica 3": (97, 100, 48.55, 51.44),
        "Área Chica 4": (97, 100, 51.44, 54.33),
        "Área Chica 5": (97, 100, 54.33, 63),
        "Área Chica 6": (94, 97, 37, 45.66),
        "Área Chica 7": (94, 97, 45.66, 54.33),
        "Área Chica 8": (94, 97, 54.33, 63)
    }

    # Calcular la cantidad de atajadas por zona
    zone_values = df.groupby("pitch_zone_gk")["IdEvent"].count().to_dict()

    # Definir normalización de colores (centrado en vmax/2)
    vmin, vmax = min(zone_values.values(), default=0), max(zone_values.values(), default=1)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vmax/2, vmax=vmax)

    # Dibujar el campo
    pitch = Pitch(pitch_type='opta', pitch_color='white', line_color='black',
                  stripe=False, corner_arcs=True, goal_type='box', half=True)
    fig, ax = pitch.draw(figsize=(14, 10))

    # Dibujar cada zona en el campo
    cmap = plt.get_cmap(cmap_name)  # Obtener el colormap
    for zone, (x_min, x_max, y_min, y_max) in buckets_gk.items():
        value = zone_values.get(zone, 0)  # Si no hay datos, asignar 0

        # Color en escala de colores elegidos
        color = cmap(norm(value))

        # Dibujar la zona
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                 linewidth=1.5, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)

        # Mostrar el valor en la zona
        ax.text((x_min + x_max) / 2, (y_min + y_max) / 2, f"{value:.0f}",
                ha="center", va="center", fontsize=10, color="black")

    # Agregar barra de color
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(f"Cantidad de {name_event}", fontsize=12)

    # Título del gráfico
    ax.set_title(f"Mapa de Atajadas", fontsize=15)

    return fig
