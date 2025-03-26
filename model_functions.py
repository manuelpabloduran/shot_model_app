import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

def gk_distance_to_shot(x, y, x_end, y_end, x_gk, y_gk):
    """
    Calcula la distancia en Y entre el arquero y el tiro en el momento en que el tiro cruza x_gk.
    
    Parámetros:
    x, y       -> Coordenadas de inicio del tiro
    x_end, y_end -> Coordenadas donde termina el tiro
    x_gk, y_gk -> Coordenadas del arquero
    
    Retorna:
    - Distancia en Y entre el arquero y el tiro cuando x = x_gk, o NaN si el tiro no cruza x_gk.
    """
    # Verificar que el tiro no es vertical (para evitar dividir por 0)
    if x_end == x:
        return abs(y_gk - y) if x == x_gk else np.nan

    # Calcular la pendiente de la trayectoria del tiro
    m = (y_end - y) / (x_end - x)  # Pendiente
    b = y - m * x  # Intersección con el eje Y

    # Calcular el Y del tiro en la posición X del arquero
    y_tiro = m * x_gk + b

    # Si el arquero está entre el rango de Y del disparo, devolver distancia
    if x <= x_gk <= x_end:
        return abs(y_gk - y_tiro)
    else:
        return np.nan  # Si no, devuelve NaN
    
def plot_success_probability_heatmap(df, num_bins_y=18, num_bins_z=6):
    """
    Genera un heatmap de probabilidad de éxito en el arco.
    
    Parámetros:
    df : DataFrame
        DataFrame con las columnas ['y_end', 'z_end', 'model_proba'].
    num_bins_y : int
        Número de divisiones en Y (ancho del arco).
    num_bins_z : int
        Número de divisiones en Z (altura del arco).
    """
    # Discretizar las coordenadas en cuadrantes
    df['y_bin'] = pd.cut(df['y_end'], bins=num_bins_y, labels=False)
    df['z_bin'] = pd.cut(df['z_end'], bins=num_bins_z, labels=False)

    # Calcular la probabilidad promedio en cada celda del grid
    heatmap_data = df.groupby(['z_bin', 'y_bin'])['model_proba'].mean().unstack()

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficar el heatmap
    sns.heatmap(
        heatmap_data, 
        cmap="RdYlGn", 
        vmin=0, vmax=1,  # Mantener la escala fija 0 = rojo, 1 = verde
        annot=True, fmt=".2f", 
        linewidths=0.5, linecolor='gray',
        cbar_kws={'label': 'Probabilidad de Éxito'}
    )

    # Ajustar etiquetas y título
    ax.set_title('Probabilidad de Éxito por Zona del Arco', fontsize=14)
    ax.invert_xaxis()
    ax.invert_yaxis()

    return fig

def plot_interpolated_probability_contour(df, num_bins_y=18, num_bins_z=6):
    """
    Genera un mapa de contornos interpolado de probabilidad de éxito en el arco.

    Parámetros:
    df : DataFrame
        DataFrame con las columnas ['y_end', 'z_end', 'model_proba'].
    num_bins_y : int
        Número de divisiones en Y (ancho del arco).
    num_bins_z : int
        Número de divisiones en Z (altura del arco).
    """

    # Discretizar coordenadas
    df['y_bin'] = pd.cut(df['y_end'], bins=num_bins_y, labels=False)
    df['z_bin'] = pd.cut(df['z_end'], bins=num_bins_z, labels=False)

    # Obtener centros de los bins
    y_centers = df.groupby('y_bin')['y_end'].mean().values
    z_centers = df.groupby('z_bin')['z_end'].mean().values

    # Obtener la malla de los bins
    y_mesh, z_mesh = np.meshgrid(y_centers, z_centers)
    proba_values = df.groupby(['z_bin', 'y_bin'])['model_proba'].mean().unstack().values.flatten()

    # Crear una malla más fina
    y_fine = np.linspace(df['y_end'].min(), df['y_end'].max(), 100)
    z_fine = np.linspace(df['z_end'].min(), df['z_end'].max(), 100)
    y_fine_mesh, z_fine_mesh = np.meshgrid(y_fine, z_fine)

    # Interpolación
    proba_fine = griddata((y_mesh.flatten(), z_mesh.flatten()), proba_values, 
                          (y_fine_mesh, z_fine_mesh), method='cubic')

    # Crear la figura y graficar
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(y_fine_mesh, z_fine_mesh, proba_fine, cmap="RdYlGn", levels=30, vmin=0, vmax=1)
    cbar = plt.colorbar(contour)
    cbar.set_label("Probabilidad de Éxito")

    # Ajustar visualización
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.invert_xaxis()

    return fig

# Función para clasificar las coordenadas dentro de los buckets correctos según el lado del córner
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

def classify_pitch_zone_dynamic(x, y):    
    # Buscar en qué zona cae el punto
    for zone, (x_min, x_max, y_min, y_max) in buckets.items():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return zone
    return "Fuera de zona"

def plot_goal_percentage_heatmap(df, bins_y, bins_z, cmap_color="Greens"):
    """
    Genera un heatmap de porcentaje de goles en cada zona del arco con respecto al total de goles.

    Parámetros:
    df : DataFrame
        DataFrame con las columnas ['y_end', 'z_end', 'NaEventType'].
    bins_y : int
        Número de divisiones en Y (ancho del arco).
    bins_z : int
        Número de divisiones en Z (altura del arco).
    cmap_color : str
        Color del mapa de calor.
    """
    # Discretizar las coordenadas en cuadrantes
    df['y_bin'] = pd.cut(df['y_end'], bins=bins_y, labels=False)
    df['z_bin'] = pd.cut(df['z_end'], bins=bins_z, labels=False)

    # Filtrar solo los goles
    df_goals = df[df['NaEventType'] == 'Goal']

    # Contar el total de goles
    total_goals = df_goals.shape[0]

    # Contar goles por cada bin
    goals_per_bin = df_goals.groupby(['z_bin', 'y_bin']).size()

    # Normalizar para obtener el porcentaje respecto al total de goles
    goal_percentage_per_bin = (goals_per_bin / total_goals) * 100

    # Crear un DataFrame con los porcentajes
    heatmap_data = goal_percentage_per_bin.unstack().fillna(0)

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficar el heatmap con el "%" en los valores
    sns.heatmap(
        heatmap_data, 
        cmap=cmap_color, 
        annot=True, fmt=".2%",  # Aquí se le agrega el "%" a los valores
        linewidths=0.5, linecolor='gray',
        cbar_kws={'label': 'Porcentaje de Goles'}
    )

    # Ajustar etiquetas y título
    ax.set_title(f'Porcentaje de Goles por Zona del Arco', fontsize=14)
    ax.invert_xaxis()
    ax.invert_yaxis()

    return fig
