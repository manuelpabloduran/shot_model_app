import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
