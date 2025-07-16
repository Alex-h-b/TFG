from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Cambiar el backend a 'Agg'
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from matplotlib import font_manager
from pathlib import Path
import os
import io
import base64
from PIL import Image
import seaborn as sns

from shiny import App, ui, render
from pathlib import Path
import os
import urllib.parse

# StatsBomb and football analytics specific
from statsbombpy import sb
from mplsoccer import (Pitch, Sbopen, VerticalPitch, 
                       Radar, PyPizza, add_image)

# Machine learning and similarity
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from funciones import *
import kagglehub


# Orden fijo para las estadísticas
orden_estadisticas = [
    "Goles", "Tiros", "Tiros a puerta", "Posesión (%)", 
    "Pases", "Faltas", "Tarjetas amarillas", "Tarjetas rojas", 
    "Fueras de juego"
]

pos_map = {
    "Porteros": "GK",
    "Defensas": "DF",
    "Centrocampistas": "MF",
    "Delanteros": "FW"
}

estadisticas_por_posicion = {
    "GK": {
        "labels": [
            "xG Recibido (PSxG)", 
            "Rendimiento xG (+/-)",
            "% Paradas (Save%)",
            "Paradas por 90'",
            "Goles Recibidos/90'",
            "Despejes por 90'",
            "Tiros Recibidos Totales"
        ],
        "columns": ["PSxG", "PSxG+/-", "Save%", "Saves_90", "GA90", "Clr_90", "SoTA"],
        "low": [0, 0, 0, 0, 0, 0, 0],
        "high": [100, 100, 100, 100, 100, 100, 100],
        "lower_is_better": ["GA90"]
    },
    "DF": {
        "labels": [
            "Pases Completos/90'", 
            "% Éxito Pases",
            "Pases Progresivos/90'",
            "Acciones Defensivas/90'",
            "% Duelos Ganados",
            "% Duelos Aéreos Ganados",
            "Conducciones Progresivas/90'",
            "xAG (Asistencias)/90'",
            "Centros al Área"
        ],
        "columns": ["Cmp_90", "Cmp%", "PrgP_90", "Tkl+Int_90", "Tkl%", "Won%", "PrgC_90", "xAG_90", "CrsPA_90"],
        "low": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "high": [100, 100, 100, 100, 100, 100, 100, 100, 100]
    },
    "MF": {
        "labels": [
            "% Duelos Ganados",
            "Acciones Defensivas/90'", 
            "Conducciones Progresivas/90'",
            "Pases Progresivos/90'",
            "Pases Clave/90'",
            "xAG (Asistencias)/90'",
            "Regates Exitosos/90'",
            "% Pases Completos",
            "Pases Completos/90'"
        ],
        "columns": ["Tkl%", "Tkl+Int_90", "PrgC_90", "PrgP_90", "KP_90", "xAG_90", "SuccDrib_90", "Cmp%", "Cmp_90"],
        "low": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "high": [100, 100, 100, 100, 100, 100, 100, 100, 100]
    },
    "FW": {
        "labels": [
            "% Duelos Aéreos Ganados", 
            "Goles sin Penaltis/90'",
            "xG (Goles Esperados)/90'",
            "Pases al Área/90'",
            "Pases Clave/90'",
            "xAG (Asistencias Esperadas)/90",
            "Regates Exitosos/90'",
            "Acciones ofensivas/90'",  
            "Conducciones Progresivas/90'"
        ],
        "columns": ["Won%", "G-PK_90", "xG_90", "PPA_90", "KP_90", "xAG_90", "SuccDrib_90", "Att_Act_90", "PrgC_90"],
        "low": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "high": [100, 100, 100, 100, 100, 100, 100, 100, 100]
    }
}

estadisticas_por_posicion_buscar = {
    "GK": {
        "stats": ["PSxG", "PSxG+/-", "Save%", "Saves_90", "GA90", "Clr_90", "SoTA"],
        "labels": {
            "PSxG": "xG Recibido (PSxG)",
            "PSxG+/-": "Rendimiento xG (+/-)",
            "Save%": "% Paradas (Save%)",
            "Saves_90": "Paradas por 90'",
            "GA90": "Goles Recibidos/90'",
            "Clr_90": "Despejes por 90'",
            "SoTA": "Tiros Recibidos Totales"
        }
    },
    "DF": {
        "stats": ["Cmp_90", "Cmp%", "PrgP_90", "Tkl+Int_90", "Tkl%", "Won%", "PrgC_90", "xAG_90", "CrsPA_90"],
        "labels": {
            "Cmp_90": "Pases Completos/90'",
            "Cmp%": "% Éxito Pases",
            "PrgP_90": "Pases Progresivos/90'",
            "Tkl+Int_90": "Acciones Defensivas/90'",
            "Tkl%": "% Duelos Ganados",
            "Won%": "% Duelos Aéreos Ganados",
            "PrgC_90": "Conducciones Progresivas/90'",
            "xAG_90": "xAG (Asistencias)/90'",
            "CrsPA_90": "Centros al Área"
        }
    },
    "MF": {
        "stats": ["Tkl%", "Tkl+Int_90", "PrgC_90", "PrgP_90", "KP_90", "xAG_90", "SuccDrib_90", "Cmp%", "Cmp_90"],
        "labels": {
            "Tkl%": "% Duelos Ganados",
            "Tkl+Int_90": "Acciones Defensivas/90'",
            "PrgC_90": "Conducciones Progresivas/90'",
            "PrgP_90": "Pases Progresivos/90'",
            "KP_90": "Pases Clave/90'",
            "xAG_90": "xAG (Asistencias)/90'",
            "SuccDrib_90": "Regates Exitosos/90'",
            "Cmp%": "% Pases Completos",
            "Cmp_90": "Pases Completos/90'"
        }
    },
    "FW": {
        "stats": ["Won%", "G-PK_90", "xG_90", "PPA_90", "KP_90", "xAG_90", "SuccDrib_90", "Att_Act_90", "PrgC_90"],
        "labels": {
            "Won%": "% Duelos Aéreos Ganados",
            "G-PK_90": "Goles sin Penaltis/90'",
            "xG_90": "xG (Goles Esperados)/90'",
            "PPA_90": "Pases al Área/90'",
            "KP_90": "Pases Clave/90'",
            "xAG_90": "xAG (Asistencias Esperadas)/90",
            "SuccDrib_90": "Regates Exitosos/90'",
            "Att_Act_90": "Acciones ofensivas/90'",
            "PrgC_90": "Conducciones Progresivas/90'"
        }
    }
}


estadisticas_por_posicion_pizza = {
    "GK": {
        "stats": [
            # Defensa (5)
            "GA90", "Save%", "Saves_90", "PSxG", "PSxG+/-",
            # Posesión (5)
            "SoTA", "Clr_90", "Won%", "Cmp_90"
            # Los porteros solo tienen 9 stats 
        ],
        "labels": {
            "GA90": "Goles Recibidos/90'",
            "Save%": "% Paradas",
            "Saves_90": "Paradas/90'",
            "PSxG": "xG Enfrentados (PSxG)",
            "PSxG+/-": "Diferencia PSxG +/-",
            "SoTA": "Tiros Recibidos",
            "Clr_90": "Despejes/90'",
            "Won%": "% Duelos Aéreos Ganados",
            "Cmp_90": "Pases Completos/90'",
        }
    },
    "DF": {
        "stats": [
            # Defensa (5)
            "Tkl+Int_90", "Int", "Clr_90", "Won%", "Blocks_90",
            # Posesión (5)
            "PrgC_90", "Cmp_90", "PrgP_90", "PPA_90", "CrsPA_90",
            # Ataque (5)
            "Ast_90", "SCA90", "Gls_90", "xG_90", "G-PK_90"
        ],
        "labels": {
            "Tkl+Int_90": "Acciones Defensivas/90'",
            "Int": "Intercepciones",
            "Clr_90": "Despejes/90'",
            "Won%": "% Duelos Aéreos Ganados",
            "Blocks_90": "Bloqueos/90'",
            "PrgC_90": "Conducciones Progresivas/90'",
            "Cmp_90": "Pases Completos/90'",
            "PrgP_90": "Pases Progresivos/90'",
            "PPA_90": "Pases al Área/90'",
            "CrsPA_90": "Centros al Área/90'",
            "Ast_90": "Asistencias/90'",
            "SCA90": "Creación de Tiros/90'",
            "Gls_90": "Goles/90'",
            "xG_90": "xG/90'",
            "G-PK_90": "Goles sin Penalti/90'"
        }
    },
    "MF": {
        "stats": [
            # Defensa (5)
            "Tkl+Int_90", "Int", "Att_stats_defense", "Won%", "Recov_90",
            # Posesión (5)
            "PrgP_90", "KP_90", "PrgC_90", "Cmp_90", "PPA_90",
            # Ataque (5)
            "Ast_90", "xAG_90", "SCA90", "Gls_90", "xG_90" 
        ],
        "labels": {
            "Tkl+Int_90": "Acciones Defensivas/90'",
            "Int": "Intercepciones",
            "Att_stats_defense": "Acciones Defensivas",
            "Won%": "% Duelos Aéreos Ganados",
            "Recov_90": "Recuperaciones/90'",
            "PrgP_90": "Pases Progresivos/90'",
            "KP_90": "Pases Clave/90'",
            "PrgC_90": "Conducciones Progresivas/90'",
            "Cmp_90": "Pases Completos/90'",
            "PPA_90": "Pases al Área/90'",
            "Ast_90": "Asistencias/90'",
            "xAG_90": "xAG/90'",
            "SCA90": "Creación de Tiros/90'",
            "Gls_90": "Goles/90'",
            "xG_90": "xG/90'"
        }
    },
    "FW": {
        "stats": [
            # Defensa (5)
            "Tkl+Int_90", "Int", "Att_stats_defense", "Won%", "Recov_90",
            # Posesión (5)
            "PrgC_90", "SuccDrib_90", "KP_90", "PPA_90", "CrsPA_90",
            # Ataque (5)
             "Ast_90", "G-PK_90","SCA90", "Gls_90", "xG_90"
        ],
        "labels": {
            "Tkl+Int_90": "Acciones Defensivas/90'",
            "Int": "Intercepciones",
            "Att_stats_defense": "Acciones Defensivas",
            "Won%": "% Duelos Aéreos Ganados",
            "Recov_90": "Recuperaciones/90'",
            "PrgC_90": "Conducciones Progresivas/90'",
            "SuccDrib_90": "Regates Exitosos/90'",
            "KP_90": "Pases Clave/90'",
            "PPA_90": "Pases al Área/90'",
            "CrsPA_90": "Centros al Área/90'",
            "Ast_90": "Asistencias/90'",
            "G-PK_90": "Goles sin Penalti/90'",
            "SCA90": "Creación de Tiros/90'",
            "Gls_90": "Goles/90'",
            "xG_90": "xG/90'"
        }
    }
}





# Definir columnas a excluir y nombres descriptivos
excluded_columns = {'Player', 'Nation', 'Squad', 'Comp', 'Pos', 'Age', 'MP', 'Starts', 'Min', '90s','indice_rendimiento'}

column_display_names = {
    "indice_rendimiento": "Índice de Rendimiento",

    # --- ESTADÍSTICAS OFENSIVAS (GOLES) ---
    'Gls': 'Goles totales',
    'Gls_90': 'Goles por 90 minutos',
    'G-PK': 'Goles sin penaltis',
    'G-PK_90': 'Goles sin penaltis por 90 minutos',
    'xG': 'Goles esperados (xG) totales',
    'xG_90': 'Goles esperados (xG) por 90 minutos',
    'npxG': 'Goles esperados sin penaltis (npxG) totales',
    'npxG_90': 'Goles esperados sin penaltis (npxG) por 90 minutos',
    'G-xG': 'Diferencia Goles - xG',
    'G-xG_90': 'Diferencia Goles - xG por 90 minutos',
    'np:G-xG': 'Diferencia Goles sin penaltis - npxG',
    'np:G-xG_90': 'Diferencia Goles sin penaltis - npxG por 90 minutos',

    # --- CREACIÓN DE JUEGO (ASISTENCIAS Y CREACIÓN) ---
    'Ast': 'Asistencias totales',
    'Ast_90': 'Asistencias por 90 minutos',
    'xAG': 'Asistencias esperadas (xAG) totales',
    'xAG_90': 'Asistencias esperadas (xAG) por 90 minutos',
    'xG+xAG': 'xG + xAG totales',
    'xG+xAG_90': 'xG + xAG por 90 minutos',
    'A-xAG': 'Diferencia Asistencias - xAG',
    'G+A': 'Goles + Asistencias totales',
    'G+A_90': 'Goles + Asistencias por 90 minutos',
    'SCA': 'Acciones de creación de tiros totales',
    'SCA90': 'Acciones de creación de tiros por 90 minutos',
    'GCA': 'Acciones de creación de gol',
    'Att_Act_90': 'Acciones ofensivas por 90 minutos',
    'xA': 'Asistencias esperadas (xA)',

    # --- PASE ---
    'Cmp': 'Pases completados',
    'Cmp_90': 'Pases completados por 90 minutos',
    'Att': 'Pases intentados',
    'Cmp%': 'Porcentaje de pases completados (%)',
    'PrgP': 'Pases progresivos',
    'PrgP_90': 'Pases progresivos por 90 minutos',
    'KP': 'Pases clave',
    'KP_90': 'Pases clave por 90 minutos',
    'PPA': 'Pases al área',
    'PPA_90': 'Pases al área por 90 minutos',
    'CrsPA': 'Centros al área',
    'CrsPA_90': 'Centros al área por 90 minutos',
    'PassLive': 'Pases en jugada',
    'PassDead': 'Pases a balón parado',
    '1/3': 'Pases en el último tercio',
    '1/3_90': 'Pases en el último tercio por 90 minutos',

    # --- CONDUCCIÓN Y REGATES ---
    'PrgR': 'Carreras progresivas',
    'PrgR_90': 'Carreras progresivas por 90 minutos',
    'PrgC': 'Conducciones progresivas',
    'PrgC_90': 'Conducciones progresivas por 90 minutos',
    'Succ': 'Regates exitosos',
    'SuccDrib_90': 'Regates exitosos por 90 minutos',
    'Carries': 'Conducciones totales',
    'Touches': 'Toques de balón',
    'Touches_90': 'Toques de balón por 90 minutos',
    'Att Pen': 'Acciones en el área rival',
    'Att Pen_90': 'Acciones en el área rival por 90 minutos',
    'Rec': 'Recepciones de pase',

    # --- DISTANCIAS ---
    'TotDist': 'Distancia total recorrida con el balón',
    'PrgDist': 'Distancia progresiva recorrida con el balón',

    # --- DEFENSA ---
    'Tkl': 'Entradas exitosas',
    'TklW': 'Entradas ganadas',
    'Tkl%': 'Porcentaje de entradas ganadas (%)',
    'Int': 'Intercepciones',
    'Tkl+Int': 'Entradas + Intercepciones',
    'Tkl+Int_90': 'Entradas + Intercepciones por 90 minutos',
    'Recov': 'Recuperaciones',
    'Recov_90': 'Recuperaciones por 90 minutos',
    'Att_stats_defense': 'Acciones defensivas intentadas',
    'Blocks_stats_defense': 'Bloqueos',
    'Blocks_90': 'Bloqueos por 90 minutos',
    'Clr': 'Despejes',
    'Clr_90': 'Despejes por 90 minutos',
    'Won%': 'Porcentaje de duelos aéreos ganados (%)',

    # --- PORTERO ---
    'PSxG': 'Goles esperados enfrentados (PSxG)',
    'PSxG+/-': 'Diferencia PSxG +/-',
    'GA': 'Goles concedidos',
    'GA90': 'Goles concedidos por 90 minutos',
    'Saves': 'Paradas',
    'Saves_90': 'Paradas por 90 minutos',
    'Save%': 'Porcentaje de paradas (%)',
    'SoTA': 'Tiros a puerta enfrentados',

    # --- DISPAROS ---
    'Sh': 'Tiros totales',
    'Sh_90': 'Tiros por 90 minutos',
    'SoT': 'Tiros a puerta',
    'SoT_90': 'Tiros a puerta por 90 minutos',
    'G/Sh': 'Goles por tiro',
    'G/SoT': 'Goles por tiro a puerta',
    'Dist': 'Distancia media de tiro',
    'npxG/Sh': 'npxG por tiro',
}



categorias_metricas_adicionales = {
    "Indice Rendimiento": {
        'indice_rendimiento': 'Índice de Rendimiento',
    },

    "⚽ Ofensivas (Goles y Tiros)": {
        'Gls': 'Goles totales',
        'G-PK': 'Goles sin penaltis',
        'Gls_90': 'Goles por 90 minutos',
        'G-PK_90': 'Goles sin penaltis por 90 minutos',
        'xG': 'Goles esperados (xG) totales',
        'xG_90': 'Goles esperados (xG) por 90 minutos',
        'npxG': 'Goles esperados sin penaltis (npxG) totales',
        'npxG_90': 'Goles esperados sin penaltis (npxG) por 90 minutos',
        'G-xG': 'Diferencia Goles - xG',
        'G-xG_90': 'Diferencia Goles - xG por 90 minutos',
        'np:G-xG': 'Diferencia Goles sin penaltis - npxG',
        'np:G-xG_90': 'Diferencia Goles sin penaltis - npxG por 90 minutos',
        'Sh': 'Tiros totales',
        'Sh_90': 'Tiros por 90 minutos',
        'SoT': 'Tiros a puerta',
        'SoT_90': 'Tiros a puerta por 90 minutos',
        'G/Sh': 'Goles por tiro',
        'G/SoT': 'Goles por tiro a puerta',
        'Dist': 'Distancia media de tiro',
        'npxG/Sh': 'npxG por tiro'
    },
    "🎯 Creación de Juego": {
        'Ast': 'Asistencias totales',
        'Ast_90': 'Asistencias por 90 minutos',
        'xAG': 'Asistencias esperadas (xAG) totales',
        'xAG_90': 'Asistencias esperadas (xAG) por 90 minutos',
        'xG+xAG': 'xG + xAG totales',
        'xG+xAG_90': 'xG + xAG por 90 minutos',
        'A-xAG': 'Diferencia Asistencias - xAG',
        'G+A': 'Goles + Asistencias totales',
        'G+A_90': 'Goles + Asistencias por 90 minutos',
        'SCA': 'Acciones de creación de tiros totales',
        'SCA90': 'Acciones de creación de tiros por 90 minutos',
        'GCA': 'Acciones de creación de gol',
        'Att_Act_90': 'Acciones ofensivas por 90 minutos',
    },
    "🔄 Pases": {
        'Cmp': 'Pases completados',
        'Cmp_90': 'Pases completados por 90 minutos',
        'Att': 'Pases intentados',
        'Cmp%': 'Porcentaje de pases completados (%)',
        'PrgP': 'Pases progresivos',
        'PrgP_90': 'Pases progresivos por 90 minutos',
        'KP': 'Pases clave',
        'KP_90': 'Pases clave por 90 minutos',
        'PPA': 'Pases al área',
        'PPA_90': 'Pases al área por 90 minutos',
        'CrsPA': 'Centros al área',
        'CrsPA_90': 'Centros al área por 90 minutos',
        'PassLive': 'Pases en jugada',
        'PassDead': 'Pases a balón parado',
        '1/3': 'Pases al último tercio',
        '1/3_90': 'Pases al último tercio por 90 minutos'
    },
    "🏃 Conducción y Regates": {
        'PrgR': 'Carreras progresivas',
        'PrgR_90': 'Carreras progresivas por 90 minutos',
        'PrgC': 'Conducciones progresivas',
        'PrgC_90': 'Conducciones progresivas por 90 minutos',
        'Succ': 'Regates exitosos',
        'SuccDrib_90': 'Regates exitosos por 90 minutos',
        'Carries': 'Conducciones totales',
        'Touches': 'Toques de balón',
        'Touches_90': 'Toques de balón por 90 minutos',
        'Att Pen': 'Acciones en el área rival',
        'Att Pen_90': 'Acciones en el área rival por 90 minutos',
        'Rec': 'Recepciones',
        'TotDist': 'Distancia total recorrida con el balón',
        'PrgDist': 'Distancia progresiva recorrida con el balón'
    },
    "🛡️ Defensivas": {
        'Tkl': 'Entradas exitosas',
        'TklW': 'Entradas ganadas',
        'Tkl%': 'Porcentaje de entradas ganadas (%)',
        'Int': 'Intercepciones',
        'Tkl+Int': 'Entradas + Intercepciones',
        'Tkl+Int_90': 'Entradas + Intercepciones por 90 minutos',
        'Recov': 'Recuperaciones',
        'Recov_90': 'Recuperaciones por 90 minutos',
        'Att_stats_defense': 'Acciones defensivas intentadas',
        'Blocks_stats_defense': 'Bloqueos',
        'Blocks_90': 'Bloqueos por 90 minutos',
        'Clr': 'Despejes',
        'Clr_90': 'Despejes por 90 minutos',
        'Won%': 'Porcentaje de duelos aéreos ganados (%)'
    },
    "🧤 Porteros": {
        'PSxG': 'Goles esperados enfrentados (PSxG)',
        'PSxG+/-': 'Diferencia PSxG +/-',
        'GA': 'Goles concedidos',
        'GA90': 'Goles concedidos por 90 minutos',
        'Saves': 'Paradas',
        'Saves_90': 'Paradas por 90 minutos',
        'Save%': 'Porcentaje de paradas (%)',
        'SoTA': 'Tiros a puerta enfrentados',
        'Cs': 'Porterías a cero',
        'Cs%': 'Porcentaje de porterías a cero (%)'
    }
}





# Download latest version
path = kagglehub.dataset_download("hubertsidorowicz/football-players-stats-2024-2025")

csv_file = os.path.join(path, "players_data-2024_2025.csv")

df_jugadores = pd.read_csv(csv_file)

df_jugadores_filtered = df_jugadores[df_jugadores['Min'] >= 900]

df_jugadores_filtered = calcular_metricas_avanzadas(df_jugadores_filtered)

df_jugadores_filtered = combinar_jugadores_duplicados(df_jugadores_filtered)

df_jugadores_filtered = calcular_indice_rendimiento(df_jugadores_filtered, eliminar_percentil_cols=True)

# 1. Filtra las columnas que están tanto en df.columns como en column_display_names
metric_columns = [col for col in column_display_names.keys() if col in df_jugadores_filtered.columns]

# 2. axis_choices ya puede usarse directamente desde column_display_names (filtrado)
axis_choices = {col: column_display_names[col] for col in metric_columns}

df_jugadores_filtered = calcular_percentiles_por_posicion(df_jugadores_filtered, estadisticas_por_posicion)




base_dir = os.path.dirname(os.path.abspath(__file__))  # carpeta donde está el script

csv_goles_path = os.path.join(base_dir, "data", "top_goles_predichos_xgboost.csv")
csv_asistentes_path = os.path.join(base_dir, "data", "top_asistentes_predichas_25_26.csv")

top_goleadores_pred = pd.read_csv(csv_goles_path)
top_goleadores_pred = top_goleadores_pred.rename(columns={
    "Player": "Jugador",
    "stats_Squad": "Equipo",         # si tienes esta columna
    "target_Gls": "Goles Temporada Actual",
    "goles_pred_xgb": "Goles Temporada Próxima"
})

top_asistentes_pred = pd.read_csv(csv_asistentes_path)
top_asistentes_pred = top_asistentes_pred.rename(columns={
    "Player": "Jugador",
    "stats_Squad": "Equipo",         # si tienes esta columna
    "target_Ast": "Asistencias Temporada Actual",
    "ast_predichas_25_26": "Asistencias Temporada Próxima"
})



# Inicializar el parser de StatsBomb
parser = Sbopen()

# Datos de ejemplo (se mantienen los datos originales)
free_comps = sb.competitions()
euro_2024 = sb.matches(competition_id=55, season_id=282)

# Extraer nombres únicos de equipos
teams = pd.unique(euro_2024[['home_team', 'away_team']].values.ravel('K'))


# Ruta al archivo CSS del tema
ruta_tema = Path(__file__).parent / "bootstrap.css"

# Configuración de rutas ABSOLUTAS
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

# Debug: Verifica archivos
print(f"\n📂 Contenido de static/: {os.listdir(STATIC_DIR)}\n")


# Interfaz de usuario
app_ui = ui.page_navbar(
    ui.head_content(ui.include_css(ruta_tema)),  # Incluir el archivo CSS
    ui.nav_panel("Mi aplicación",
        ui.h2("Bienvenido a la aplicación"),
        ui.p("Esta es la pantalla de inicio. Selecciona una sección en el menú superior para comenzar."),
        ui.p("""
        Esta aplicación interactiva está diseñada para explorar en profundidad datos avanzados de fútbol. Ofrece múltiples módulos que permiten analizar competiciones, estudiar el rendimiento individual de jugadores, realizar comparativas visuales con gráficos personalizados y aplicar técnicas de visión artificial a secuencias reales de partidos. 
        """),
        ui.p("""
        A través de la sección "📊 Análisis de Competiciones", gracias a datos extraidos de Statsbomb puedes examinar estadísticas detalladas de equipos y partidos reales, incluyendo mapas de tiros, redes de pases, recuperaciones y goles. 
        """),
        ui.p("""
        En "🧍 Análisis de Jugadores", puedes explorar el rendimiento de los jugadores en base a las métricas pertenecientes a Fbref, generar gráficos radar o pizza según la posición, comparar atributos y predecir su rendimiento en la siguiente temporada mediante modelos de aprendizaje automático. Además, la sección de "🔎 Buscador" permite filtrar jugadores por posición, edad y métricas clave para encontrar perfiles específicos, ideal para scouting o análisis técnico.
        """),
        ui.p("""
         Finalmente, en el apartado de "🎥🧠 Visión por Computadora", gracias a modelos preentrenados de Roboflow se visualizan ejemplos prácticos del uso de inteligencia artificial para rastrear jugadores, analizar posturas y posiciones y extraer información táctica a partir de vídeo.
        """),
        ui.p("""
        Todo el análisis se realiza de forma dinámica con datos reales de competiciones europeas, utilizando herramientas de ciencia de datos, machine learning y visualización avanzada. Esta aplicación está pensada para analistas, entrenadores, aficionados o investigadores que deseen profundizar en el fútbol desde una perspectiva cuantitativa y visual.
        """)
    ),
    ui.nav_panel("📊 Análisis de Competiciones",
        ui.h2("📊 Análisis de Competiciones"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select("equipo", "🏟️ Selecciona un equipo:", teams.tolist(), selected="Spain"),
                ui.input_select("partido", "📅 Selecciona un partido:", []),
            ),
            ui.navset_tab(
                ui.nav_panel("🏟️ Estadísticas de Equipos",
                    ui.h4("📈 Estadísticas de Equipos"),
                    ui.layout_columns(
                        ui.card(
                            ui.output_table("stats_equipo"),
                            height="780px",
                        ),
                        ui.card(
                            ui.input_checkbox("incluir_penaltis", "⚽ Incluir penaltis en el xG", value=True),
                            ui.output_plot("grafico_xg"),
                            ui.layout_columns(
                                ui.output_plot("mapa_tiros_local"),
                                ui.output_plot("mapa_tiros_visitante")
                            ),
                        ),
                        col_widths=[5, 7],
                    ),
                    ui.input_select("equipo_pases", "📌 Selecciona el equipo para analizar:", []),
                    ui.h4("🔁 Pases"),
                    ui.output_plot("grafico_pases"),
                    ui.br(),
                    ui.output_plot("grafico_pases_completos_incompletos_apilados"),
                    ui.br(),
                    ui.h4("🗺️ Pases y Posiciones Promedio"),
                    ui.layout_columns(
                        ui.card(ui.output_plot("grafico_red_pases")),
                        ui.card(ui.output_plot("grafico_pases_equipo")),
                        col_widths=[6, 6],
                    ),
                    ui.h4("🛡️ Recuperaciones"),
                    ui.layout_columns(
                        ui.card(ui.output_plot("mapa_calor_recuperaciones")),
                        ui.card(ui.output_plot("pieplot_recuperaciones")),
                        col_widths=[6, 6],
                    ),
                    ui.h4("🥅 Goles"),
                    ui.output_ui("selector_goles"),
                    ui.output_plot("grafico_goles"),
                ),
                ui.nav_panel("🧍 Estadísticas de Jugadores",
                    ui.input_select("jugador", "🎽 Selecciona un jugador:", []),
                    ui.layout_columns(
                        ui.card(
                            ui.output_table("stats_jugador"),
                        ),
                        ui.card(
                            ui.layout_columns(
                                ui.card(
                                    ui.h4("🔁 Pases"),
                                    ui.output_plot("grafico_pases_jugador")
                                ),
                                ui.card(
                                    ui.h4("🔥 Mapa de calor"),
                                    ui.output_plot("heatmap")
                                ),
                                ui.card(
                                    ui.h4("🏃‍♂️ Conducciones"),
                                    ui.output_plot("grafico_conducciones")
                                ),
                                ui.card(
                                    ui.h4("⚽ Tiros"),
                                    ui.output_plot("mapa_tiros_jugador")
                                ),
                                col_widths=[6, 6, 6, 6]
                            ),
                        ),
                        col_widths=[3, 9]
                    )
                )
            )
        )
    ),
    ui.nav_panel("🧍 Análisis de Jugadores",
        ui.h2("🎽 Análisis de Jugadores"),
        ui.navset_tab(
            ui.nav_panel("⭐ Top Jugadores",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_select("liga_top", "🏆 Selecciona liga:", 
                            ["Todas"] + sorted(df_jugadores_filtered['Comp'].unique().tolist())
                        ),
                        ui.input_select("posicion_top", "🎯 Filtrar por posición:", 
                            ["Todos", "Delantero", "Mediocentro", "Defensa", "Portero"]
                        ),
                        ui.input_select("edad_top", "⏳ Filtrar por edad:", 
                            choices=["Todos", "U25", "U23", "U21", "U19"], 
                            selected="Todos"
                        ),
                        width=250
                    ),
                    ui.navset_tab(
                        ui.nav_panel("⚽ Goleadores",
                            ui.card(
                                ui.card_header("🥇 Top 10 Goleadores"),
                                ui.output_data_frame("top_goleadores")
                            )
                        ),
                        ui.nav_panel("🎯 Asistentes",
                            ui.card(
                                ui.card_header("🥇 Top 10 Asistentes"),
                                ui.output_data_frame("top_asistentes")
                            )
                        ),
                        ui.nav_panel("🧤 Porteros",
                            ui.card(
                                ui.card_header("🥇 Top 10 Porteros"),
                                ui.output_data_frame("top_porteros")
                            )
                        ),
                        ui.nav_panel("🔮 Predicciones",
                            ui.card(
                                ui.card_header("🔮 Predicción de Goleadores y Asistentes (Temporada Próxima)"),
                                ui.row(
                                    ui.column(6, ui.output_ui("tabla_predicciones_goles")),
                                    ui.column(6, ui.output_ui("tabla_predicciones_asistencias")),
                                )
                            )
                        ),
                        ui.nav_panel("📈 Rankings",
                            ui.row(
                                ui.input_select("indice_metrica", "📊 Seleccionar Métrica:",
                                    categorias_metricas_adicionales,
                                    selected="indice_rendimiento"
                                ),
                                ui.input_numeric("num_jugadores", "🔢 Nº de jugadores a mostrar:",
                                    min=5, max=50, value=15, step=5
                                ),
                                width=300
                            ),
                            ui.layout_columns(
                                ui.output_ui("tabla_indices")
                            )
                        )
                    )
                )
            ),
            ui.nav_panel("📊 Gráficos",
                ui.navset_tab(
                    ui.nav_panel("📈 Radar",
                        ui.layout_sidebar(
                            ui.sidebar(
                                ui.input_select("posicion_radar", "🎯 Selecciona posición:",
                                    ["Porteros", "Defensas", "Centrocampistas", "Delanteros"]),
                                ui.input_selectize(
                                    "jugadores", "👥 Selecciona jugadores (1-3):",
                                    choices=df_jugadores_filtered["Player"].tolist(),
                                    multiple=True,
                                    options={"maxItems": 3, "create": False}
                                ),
                            ),
                            ui.layout_columns(
                                ui.card(ui.output_ui("tabla_estadisticas_radar")),
                                ui.card(ui.output_plot("grafico_radar_dinamico")),
                                col_widths=[5, 7]
                            ),
                            ui.card(
                                ui.h4("🔍 Jugadores Similares"),
                                ui.output_data_frame("jugadores_similares")
                            )
                        )
                    ),

                    ui.nav_panel("🍕 Pizza Plot",
                        ui.layout_sidebar(
                            ui.sidebar(
                                ui.input_select(
                                    "posicion_pizza", 
                                    "🎯 Selecciona posición:",
                                    ["Porteros", "Defensas", "Centrocampistas", "Delanteros"],
                                    selected="Centrocampistas"
                                ),
                                ui.input_selectize(
                                    "jugadores_pizza", 
                                    "👤 Selecciona un jugador:",
                                    choices=[],
                                    multiple=True,
                                    options={"maxItems": 1, "create": False}
                                ),
                                width=300
                            ),
                            ui.layout_columns(
                                ui.card(
                                    ui.output_plot("grafico_pizza", height="650px"),
                                    ui.output_ui("fortalezas_debilidades"),
                                )
                            )
                        )
                    ),

                    ui.nav_panel("📉 Scatterplot",
                        ui.row(
                            ui.column(2, ui.input_select("posicion_scatter", "🎯 Posición", choices=list(pos_map.keys()))),
                            ui.column(2, ui.input_select("x_axis", "➖ Eje X", categorias_metricas_adicionales, selected="Gls_90")),
                            ui.column(2, ui.input_select("y_axis", "➕ Eje Y", categorias_metricas_adicionales, selected="Ast_90")),
                            ui.column(2, ui.input_select("liga_scatter", "🏆 Competición", choices=["Todas"] + sorted(df_jugadores_filtered["Comp"].dropna().unique().tolist()), selected="Todas")),
                            ui.column(2, ui.input_select("edad_scatter", "⏳ Filtro de edad", choices=["Todos", "U25", "U23", "U21", "U19"], selected="Todos")),
                        ),
                        ui.layout_sidebar(
                            ui.sidebar(
                                ui.input_selectize(
                                    "jugadores_scatter", "👥 Selecciona jugadores (1-20):",
                                    choices=df_jugadores_filtered["Player"].tolist(),
                                    multiple=True,
                                    options={"maxItems": 20, "create": False}
                                ),
                                ui.input_action_button("limpiar_jugadores", "🧹 Limpiar selección", class_="btn-danger"),
                                ui.input_checkbox("mostrar_nombres", "📝 Mostrar todos los nombres", False),
                                ui.input_checkbox("mostrar_mediana", "📊 Mostrar líneas de mediana", False),
                            ),
                            ui.card(ui.output_plot("scatter_plot", height="550px"))
                        )
                    )
                )
            ),
            ui.nav_panel(
                "🔎 Buscador",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_select("posicion", "🎯 Posición", choices=["GK", "DF", "MF", "FW"], selected="GK"),
                        ui.output_ui("filtros_base"),
                        ui.input_numeric("age_min", "📉 Edad mínima", value=16, min=15, max=50, step=1),
                        ui.input_numeric("age_max", "📈 Edad máxima", value=37, min=15, max=50, step=1),
                        ui.output_ui("select_metricas_adicionales"),
                    ),
                    ui.card(
                        ui.output_ui("tabla_filtrada")
                    )
                )
            ),
        )
    ),
    ui.nav_panel("🎥🧠 Visión por Computadora",
    ui.h2("🎥🧠 Visión por Computadora"),
    ui.p("Bienvenido a la sección de Visión por Computadora. Aquí puedes visualizar fragmentos de partidos a los que se les ha aplicado modelos de visión artificial."),

    ui.input_select(
        id="video_selector",
        label="Selecciona un vídeo:",
        choices={
            "prueba1.mp4": "Partido 1",
            "prueba2.mp4": "Partido 2",
            "prueba3.mp4": "Partido 3",
        }
    ),

   ui.layout_columns(
       
        ui.div(
            ui.output_ui("video_player"),
            width=6
        ),

        ui.div(
            ui.h4("¿Qué hace este módulo?"),
            ui.p("""Este módulo permite visualizar técnicas avanzadas de visión artificial para detectar jugadores y balón, seguir su movimiento a lo largo del campo y analizar aspectos clave del juego como la posesión, la velocidad o la postura."""),
            ui.p("""El sistema usado combina modelos de detección de objetos, estimación de poses clave y algoritmos de seguimiento multiobjeto. Todo el procesamiento se realiza en Google Colab con soporte de GPU, y modelos de inferencia optimizados de Roboflow y HuggingFace."""),
            ui.h5("Fases del procesamiento:"),
            ui.tags.ul(
                ui.tags.li("🔍 Detección de jugadores y balón en cada fotograma."),
                ui.tags.li("🧍 Estimación de poses clave para analizar posturas."),
                ui.tags.li("📍 Seguimiento continuo con asignación de ID únicos."),
                ui.tags.li("📊 Análisis de posesión, trayectorias y velocidades."),
                ui.tags.li("🎯 Visualización de resultados en vídeo procesado.")
            ),
            ui.p("Este enfoque permite representar el juego desde una nueva dimensión, facilitando el análisis táctico y físico de cada jugador en tiempo real."),
            width=6
        )
    )

),
    id="navbar",
    selected="Mi aplicación",
)

# Lógica del servidor
def server(input: Inputs, output: Outputs, session: Session):
    colores = ['#66d8ba', '#f28500', '#c94c4c']

    @reactive.Effect
    @reactive.event(input.limpiar_jugadores)
    def _():
        ui.update_selectize("jugadores_scatter", selected=[])
    
    # Actualizar la lista de partidos en función del equipo seleccionado
    @reactive.Effect
    def _():
        equipo_seleccionado = input.equipo()
        if equipo_seleccionado:
            # Filtrar los partidos del equipo seleccionado
            df_equipo = euro_2024[
                (euro_2024['home_team'] == equipo_seleccionado) | 
                (euro_2024['away_team'] == equipo_seleccionado)
            ]
            
            # Ordenar los partidos por fecha (cronológicamente)
            df_equipo = df_equipo.sort_values(by='match_date')

            # Obtener la lista de match_id ordenados cronológicamente
            match_ids = df_equipo['match_id'].tolist()

            # Asignar una descripción dinámica a cada partido
            descripcion_partidos = {}
            fases = ["Fase de Grupos - Partido", "Octavos de Final", "Cuartos de Final", "Semifinal", "Final"]

            for i, match_id in enumerate(match_ids):
                if i < 3:  # Los primeros 3 partidos son de fase de grupos
                    descripcion_partidos[match_id] = f"{fases[0]} {i+1}"
                else:  # Los siguientes son de fases eliminatorias
                    fase_index = i - 2  # Ajustamos el índice porque los primeros 3 ya son fase de grupos
                    if fase_index < len(fases):
                        descripcion_partidos[match_id] = fases[fase_index]
                    else:
                        descripcion_partidos[match_id] = f"Partido {i+1}"

            # Actualizar la lista de selección con nombres descriptivos en vez de IDs
            ui.update_select("partido", choices=descripcion_partidos)

    # Actualizar la lista de jugadores y los nombres de los equipos según el partido seleccionado
    @reactive.Effect
    def _():
        partido_seleccionado = input.partido()
        if partido_seleccionado:
            # Obtener el ID del partido seleccionado
            match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
            
            # Cargar los eventos del partido
            eventos_partido = parser.event(match_id)[0]  # Asumiendo que el parser devuelve un DataFrame con los eventos
            
            # Verificar la estructura del DataFrame
            print("Columnas disponibles en eventos_partido:", eventos_partido.columns.tolist())
            
            # Obtener la lista de jugadores únicos
            if 'player_name' in eventos_partido.columns:
                jugadores = eventos_partido['player_name'].dropna().unique().tolist()
            else:
                print("La columna 'player_name' no está presente en el DataFrame.")
                jugadores = []  # Lista vacía si no se encuentra la columna
            
            
            
            # Obtener los nombres de los equipos local y visitante
            partido_info = euro_2024[euro_2024['match_id'] == match_id].iloc[0]
            equipo_local = partido_info['home_team']
            equipo_visitante = partido_info['away_team']

            jugadores_local = eventos_partido[eventos_partido['team_name'] == equipo_local]['player_name'].dropna().unique().tolist()
            jugadores_visitante = eventos_partido[eventos_partido['team_name'] == equipo_visitante]['player_name'].dropna().unique().tolist()


            choices = {
                equipo_local: {nombre: nombre for nombre in jugadores_local},
                equipo_visitante: {nombre: nombre for nombre in jugadores_visitante}
            }

            ui.update_select("jugador", choices=choices)
            
            # Actualizar el selector de equipos para pases
            ui.update_select("equipo_pases", choices=[equipo_local, equipo_visitante])

    # Mostrar las estadísticas del equipo seleccionado
    @output
    @render.table
    def stats_equipo():
        equipo_seleccionado = input.equipo()
        partido_seleccionado = input.partido()
        if equipo_seleccionado and partido_seleccionado:
            # Obtener el match_id del partido seleccionado
            match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
            
            # Cargar los eventos del partido
            eventos_partido = parser.event(match_id)[0]  # Asumiendo que el parser devuelve un DataFrame con los eventos
            
            # Obtener el equipo contrario
            partido_info = euro_2024[euro_2024['match_id'] == match_id].iloc[0]
            equipo_contrario = partido_info['away_team'] if partido_info['home_team'] == equipo_seleccionado else partido_info['home_team']
            
            # Calcular las estadísticas del partido
            resumen = resumen_estadisticas_partido(eventos_partido, equipo_seleccionado, equipo_contrario)
            
            # Crear un DataFrame con las estadísticas de ambos equipos
            df_local = pd.DataFrame(resumen[equipo_seleccionado].items(), columns=["Estadística", equipo_seleccionado])
            df_visitante = pd.DataFrame(resumen[equipo_contrario].items(), columns=["Estadística", equipo_contrario])

            # Combinar ambos DataFrames en uno solo
            df_combinado = pd.merge(df_local, df_visitante, on="Estadística", how="outer")

            # Forzar el orden de las estadísticas
            df_combinado["Estadística"] = pd.Categorical(df_combinado["Estadística"], categories=orden_estadisticas, ordered=True)
            df_combinado = df_combinado.sort_values("Estadística").reset_index(drop=True)

            return df_combinado

    # Gráfico de xG acumulado
    @output
    @render.plot
    def grafico_xg():
        partido_seleccionado = input.partido()
        incluir_penaltis = input.incluir_penaltis()  # Obtener la selección del checkbox
        if partido_seleccionado:
            # Obtener el ID del partido seleccionado
            match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
            
            # Cargar los eventos del partido
            eventos_partido = sb.events(match_id)  # Asumiendo que el parser devuelve un DataFrame con los eventos
            
            # Obtener los nombres de los equipos local y visitante
            partido_info = euro_2024[euro_2024['match_id'] == match_id].iloc[0]
            equipo_local = partido_info['home_team']
            equipo_visitante = partido_info['away_team']
            
            # Crear el gráfico de xG acumulado
            fig = plot_xg_acumulado(eventos_partido, equipo_local, equipo_visitante, incluir_penales=incluir_penaltis)
            return fig

    @output
    @render.plot
    def grafico_pases():
        partido_seleccionado = input.partido()
        equipo_pases = input.equipo_pases()

        if partido_seleccionado and equipo_pases:
            match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
            eventos_partido = parser.event(match_id)[0]

            pases_totales = analizar_pases(eventos_partido, equipo_pases)

            # Estética
            sns.set(style="whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))

            # Colores
            color_barra = "#4682B4"  # Steel Blue

            # Gráfico
            ax = sns.barplot(
                data=pases_totales,
                y="player_name",
                x="Porcentaje %",
                orient="h",
                color=color_barra
            )

            # Etiquetas internas con contraste
            for i, (valor, jugador) in enumerate(zip(pases_totales['Porcentaje %'], pases_totales['player_name'])):
                ax.text(valor - 3, i, f"{valor:.1f}%", va="center", ha="right", color="white", fontsize=10, fontweight="bold")

            # Estética general
            ax.set_title(f"Top 10 jugadores por porcentaje de pases completos - {equipo_pases}", fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel("Porcentaje de pases completos (%)", fontsize=12)
            ax.set_ylabel("Jugador", fontsize=12)
            sns.despine(left=True, bottom=True)
            # plt.tight_layout()
            # plt.close(fig)

            return fig


    @output
    @render.plot
    def grafico_pases_completos_incompletos_apilados():
        partido_seleccionado = input.partido()
        equipo_pases = input.equipo_pases()

        if partido_seleccionado and equipo_pases:
            match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
            eventos_partido = parser.event(match_id)[0]
            pases_totales = analizar_pases(eventos_partido, equipo_pases)

            # Calcular total y ordenar
            pases_totales['total_pases'] = pases_totales['completos'] + pases_totales['incompletos']
            pases_totales = pases_totales.sort_values(by='total_pases', ascending=True)

            # Crear figura sin estilo de cuadrícula
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.style.use("default")  # Elimina el estilo grid si estaba activo
            ax.grid(False)  # Asegura que no haya grid

            # Datos y posiciones
            y_pos = np.arange(len(pases_totales))
            complet = pases_totales['completos'].values
            incomplet = pases_totales['incompletos'].values
            jugadores = pases_totales['player_name'].values

            # Colores
            color_completos = "#4C9F70"
            color_incompletos = "#E57373"

            # Barras horizontales
            ax.barh(y_pos, complet, color=color_completos, label="Completos")
            ax.barh(y_pos, incomplet, left=complet, color=color_incompletos, label="Incompletos")

            # Etiquetas
            for i, (c, ic) in enumerate(zip(complet, incomplet)):
                if c > 0:
                    ax.text(c / 2, i, f"{int(c)}", va="center", ha="center", color="white", fontsize=9, fontweight="bold")
                if ic > 0:
                    ax.text(c + ic / 2, i, f"{int(ic)}", va="center", ha="center", color="black", fontsize=9, fontweight="bold")

            # Ejes
            ax.set_yticks(y_pos)
            ax.set_yticklabels(jugadores)
            ax.set_xlabel("Número de pases", fontsize=12)
            ax.set_title(f"Pases Completos e Incompletos - {equipo_pases}", fontsize=14, fontweight="bold", pad=15)
            ax.legend(loc="lower right", frameon=True, fontsize=10)

            ax.xaxis.grid(True, linestyle="--", alpha=0.5)
            ax.yaxis.grid(False)

            # Limpiar bordes y márgenes
            sns.despine(left=True, bottom=True)
            # plt.tight_layout()
            # plt.close(fig)

            return fig


  

    @output
    @render.table
    def stats_jugador():
        partido_seleccionado = input.partido()
        jugador_seleccionado = input.jugador()
        
        # Verificar si se ha seleccionado un partido y un jugador
        if not partido_seleccionado or not jugador_seleccionado:
            return pd.DataFrame(columns=["Tipo de Evento", "Eventos"])
        
        # Obtener el ID del partido seleccionado
        match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
        
        # Obtener los eventos del partido seleccionado
        eventos_partido = parser.event(match_id)[0]  # Asumiendo que parser.event retorna los eventos del partido
        
        # Filtrar los eventos del jugador seleccionado
        eventos_jugador = eventos_partido[eventos_partido['player_name'] == jugador_seleccionado]
        
        # Lista de eventos a excluir
        eventos_excluir = ["Substitution", "Miscontrol", "Dribbled Past", "Pressure", "Dispossessed", 
                        "Injury Stoppage", "Foul Won", "Player Off", "Player On"]
        
        # Filtrar los eventos eliminando los no deseados
        eventos_jugador = eventos_jugador[~eventos_jugador['type_name'].isin(eventos_excluir)]
        
        # Contar los goles - verificar si existe la columna 'outcome_name'
        goles = 0
        if 'Shot' in eventos_jugador['type_name'].values and 'outcome_name' in eventos_jugador.columns:
            goles = eventos_jugador[(eventos_jugador['type_name'] == 'Shot') & 
                                (eventos_jugador['outcome_name'] == 'Goal')].shape[0]

        # Contar las asistencias - verificar si existe la columna 'pass_goal_assist'
        asistencias = 0
        if 'Pass' in eventos_jugador['type_name'].values and 'pass_goal_assist' in eventos_jugador.columns:
            asistencias = eventos_jugador[(eventos_jugador['type_name'] == 'Pass') & 
                                        (eventos_jugador['pass_goal_assist'] == True)].shape[0]
            
        pases_clave = eventos_jugador[(eventos_jugador['type_name'] == 'Pass') & 
                                        (eventos_jugador['pass_shot_assist'] == True)].shape[0]

        # Agrupar por tipo de evento y contar los eventos
        df_eventos = eventos_jugador.groupby('type_name').size().reset_index(name='Eventos')
        
        # Renombrar la columna 'type_name' a 'Tipo de Evento'
        df_eventos = df_eventos.rename(columns={"type_name": "Tipo de Evento"})

        # Agregar los goles y asistencias al DataFrame
        extra_eventos = pd.DataFrame({
            "Tipo de Evento": ["Goals", "Assists", "Key Passes"],
            "Eventos": [goles, asistencias, pases_clave]
        })

        df_eventos = pd.concat([df_eventos, extra_eventos], ignore_index=True)

        return df_eventos

    @output
    @render.plot
    def mapa_tiros_local():
        partido_seleccionado = input.partido()
        if partido_seleccionado:
            # Obtener el ID del partido seleccionado
            match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
            
            # Cargar los eventos del partido
            eventos_partido = sb.events(match_id)  # Asumiendo que el parser devuelve un DataFrame con los eventos
            
            # Obtener el nombre del equipo local
            partido_info = euro_2024[euro_2024['match_id'] == match_id].iloc[0]
            equipo_local = partido_info['home_team']
            equipo_visitante = partido_info['away_team']
            # Crear el gráfico del mapa de tiros para el equipo local
            fig = mapa_tiros_por_equipo(eventos_partido, equipo_local, equipo_visitante)
            return fig

    @output
    @render.plot
    def mapa_tiros_visitante():
        partido_seleccionado = input.partido()
        if partido_seleccionado:
            # Obtener el ID del partido seleccionado
            match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
            
            # Cargar los eventos del partido
            eventos_partido = sb.events(match_id)  # Asumiendo que el parser devuelve un DataFrame con los eventos
            
            # Obtener el nombre del equipo visitante
            partido_info = euro_2024[euro_2024['match_id'] == match_id].iloc[0]
            equipo_visitante = partido_info['away_team']
            equipo_local = partido_info['home_team']
            # Crear el gráfico del mapa de tiros para el equipo visitante
            fig = mapa_tiros_por_equipo(eventos_partido, equipo_visitante, equipo_local)
            return fig
        
    @output
    @render.plot
    def grafico_red_pases():
        partido_seleccionado = input.partido()
        equipo_pases = input.equipo_pases()  # Obtener la selección de equipo (nombre del equipo)
        
        if partido_seleccionado and equipo_pases:
            # Obtener el ID del partido seleccionado
            match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
            
            # Cargar los eventos del partido
            eventos_partido = parser.event(match_id=match_id)[0]  
            lineups = sb.lineups(match_id)
            alineacion = lineups[equipo_pases] 
            
            # Procesar los datos para la red de pases (sin mapear nombres aún)
            scatter_df, lines_df = procesar_datos_red_pases(eventos_partido, equipo_pases)
            
            # Verificar si hay datos para graficar
            if scatter_df.empty or lines_df.empty:
                print("No hay datos suficientes para graficar la red de pases.")
                return None
            
            # Crear el campo de fútbol con un tamaño grande
            pitch = Pitch(line_color='black', pitch_color='white')
            fig, ax = pitch.draw(figsize=(20, 10))  # Tamaño grande
            
            # Verificar si hay nicknames en la alineación
            if 'player_nickname' in alineacion.columns:
                # Mapear nombres de jugadores a sus apodos (si existen)
                player_name_to_nickname = {
                    name: nickname if pd.notna(nickname) else name
                    for name, nickname in zip(alineacion['player_name'], alineacion['player_nickname'])
                }
            else:
                # Si no hay nicknames, crear un diccionario donde el nombre es igual al apodo
                player_name_to_nickname = {name: name for name in alineacion['player_name']}
            
            # Mapear nombres en scatter_df
            scatter_df['player_name'] = scatter_df['player_name'].map(player_name_to_nickname).fillna(scatter_df['player_name'])
            
            # Dibujar jugadores
            pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, color='#2f5fed', edgecolors='grey', linewidth=1, alpha=0.8, ax=ax, zorder=3)
            
            # Anotar nombres con un tamaño de letra más pequeño
            for i, row in scatter_df.iterrows():
                pitch.annotate(
                    row.player_name, 
                    xy=(row.x, row.y), 
                    c='black', 
                    va='center', 
                    ha='center', 
                    size=8,  # Tamaño de letra más pequeño
                    ax=ax, 
                    zorder=4
                )
            
            # Dibujar líneas de pases
            for _, row in lines_df.iterrows():
                player1, player2 = row["pair_key"].split("_")
                
                # Mapear nombres completos a apodos (o nombres completos si no hay apodos)
                player1 = player_name_to_nickname.get(player1, player1)
                player2 = player_name_to_nickname.get(player2, player2)
                
                # Verificar si player1 y player2 están en scatter_df
                if player1 in scatter_df["player_name"].values and player2 in scatter_df["player_name"].values:
                    player1_x = scatter_df.loc[scatter_df["player_name"] == player1, 'x'].iloc[0]
                    player1_y = scatter_df.loc[scatter_df["player_name"] == player1, 'y'].iloc[0]
                    player2_x = scatter_df.loc[scatter_df["player_name"] == player2, 'x'].iloc[0]
                    player2_y = scatter_df.loc[scatter_df["player_name"] == player2, 'y'].iloc[0]
                    num_passes = row["pass_count"]
                    line_width = (num_passes / lines_df['pass_count'].max() * 5)

                    pitch.lines(player1_x, player1_y, player2_x, player2_y, alpha=1, lw=line_width, zorder=2, color="#2f5fed", ax=ax)
                else:
                    print(f"Advertencia: No se encontraron datos para el par {player1} - {player2}.")
            
            # Eliminar márgenes innecesarios
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Sin márgenes

            return plt.gcf()
        
    @output
    @render.plot
    def mapa_calor_recuperaciones():
        # Obtener el ID del partido seleccionado
        partido_seleccionado = input.partido()
        match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
        team = input.equipo_pases()  # Obtener el equipo seleccionado
        eventos_partido = sb.events(match_id)

        # Filtrar las recuperaciones del equipo
        df1_recuperaciones = eventos_partido[eventos_partido['type'] == 'Ball Recovery']
        df1_recuperaciones_team = df1_recuperaciones[df1_recuperaciones['team'] == team]

        # Extraer las coordenadas (x, y) de las recuperaciones
        df1_recuperaciones_team = df1_recuperaciones_team.copy()
        df1_recuperaciones_team['location_x'] = df1_recuperaciones_team['location'].apply(lambda loc: loc[0])
        df1_recuperaciones_team['location_y'] = df1_recuperaciones_team['location'].apply(lambda loc: loc[1])

        # Crear una figura para el mapa de calor
        fig, ax = plt.subplots(figsize=(8, 6))

        # Crear el campo de fútbol
        pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
        pitch.draw(ax=ax)

        # Graficar el mapa de calor (heatmap) de las recuperaciones
        sns.kdeplot(
            x=df1_recuperaciones_team['location_x'],
            y=df1_recuperaciones_team['location_y'],
            cmap="Reds",  # Color del heatmap
            fill=True,    # Rellenar el heatmap
            alpha=0.5,    # Transparencia del heatmap
            ax=ax         # Superponer el heatmap en el campo de fútbol
        )

        # Graficar las recuperaciones como puntos en el campo
        pitch.scatter(
            df1_recuperaciones_team['location_x'],
            df1_recuperaciones_team['location_y'],
            ax=ax,
            color='red',      # Color de los puntos
            edgecolor='black', # Borde de los puntos
            s=100,            # Tamaño de los puntos
            label='Recuperaciones'  # Leyenda
        )

        # Devolver la figura
        return fig

    @output
    @render.plot
    def pieplot_recuperaciones():
        # Obtener el ID del partido seleccionado
        partido_seleccionado = input.partido()
        match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
        team = input.equipo_pases()  # Obtener el equipo seleccionado
        eventos_partido = sb.events(match_id)

        # Filtrar las recuperaciones del equipo
        df1_recuperaciones = eventos_partido[eventos_partido['type'] == 'Ball Recovery']
        df1_recuperaciones_team = df1_recuperaciones[df1_recuperaciones['team'] == team]

        # Extraer las coordenadas (x, y) de las recuperaciones
        df1_recuperaciones_team = df1_recuperaciones_team.copy()
        df1_recuperaciones_team['location_x'] = df1_recuperaciones_team['location'].apply(lambda loc: loc[0])

        # Definir zonas del campo (ofensiva o defensiva)
        df1_recuperaciones_team['zona'] = df1_recuperaciones_team['location_x'].apply(
            lambda x: 'Zona Ofensiva' if x > 60 else 'Zona Defensiva'
        )

        # Contar las recuperaciones por zona
        recuperaciones_por_zona = df1_recuperaciones_team['zona'].value_counts()

        # Función personalizada para mostrar porcentaje + total
        def format_label(pct, all_vals):
            total = int(round(pct / 100. * sum(all_vals)))
            return f'{pct:.1f}%\n({total})'

        # Crear la figura
        fig, ax = plt.subplots(figsize=(8, 6))

        # Crear el pieplot con etiqueta personalizada
        wedges, texts, autotexts = ax.pie(
            recuperaciones_por_zona,
            labels=recuperaciones_por_zona.index,
            autopct=lambda pct: format_label(pct, recuperaciones_por_zona),
            colors=['lightblue', 'lightcoral'],
            startangle=90,
            wedgeprops={'edgecolor': 'black'},
            textprops={'fontsize': 11}
        )

        # Título opcional
        ax.set_title(f"Recuperaciones por zona - {team}", fontsize=14)

        plt.tight_layout()
        plt.close(fig)
        return fig
    
    @output
    @render.plot
    def grafico_pases_equipo():
        partido_seleccionado = input.partido()
        match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
        team = input.equipo_pases()  # Obtener el equipo seleccionado
        eventos_partido = parser.event(match_id)[0]
        fig = plot_pases_por_equipo(eventos_partido, team)
        return fig 
    
    @output
    @render.plot
    def grafico_pases_jugador():
        partido_seleccionado = input.partido()
        jugador_seleccionado = input.jugador()
        match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
        eventos_partido = parser.event(match_id)[0]
        fig = plot_pases_por_jugador(eventos_partido, jugador_seleccionado)
        return fig 
    

    @output
    @render.plot
    def heatmap():
        partido_seleccionado = input.partido()
        jugador_seleccionado = input.jugador()
        match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
        eventos_partido = parser.event(match_id)[0]
        fig = graficar_heatmap(eventos_partido, jugador_seleccionado)
        return fig 
    
    @output
    @render.plot
    def grafico_conducciones():
        partido_seleccionado = input.partido()
        jugador_seleccionado = input.jugador()
        match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
        eventos_partido = sb.events(match_id)
        fig = plot_conducciones_jugador(eventos_partido, jugador_seleccionado)
        return fig 
    
    @output
    @render.plot
    def mapa_tiros_jugador():
        partido_seleccionado = input.partido()
        jugador_seleccionado = input.jugador()
        match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
        eventos_partido = sb.events(match_id)
        
        # Obtener el equipo del jugador para el título
        jugador_data = eventos_partido[eventos_partido['player'] == jugador_seleccionado]
        equipo_jugador = jugador_data['team'].iloc[0] if not jugador_data.empty else ""
        
        fig = crear_mapa_tiros_jugador(eventos_partido, jugador_seleccionado, equipo_jugador)
        return fig
    
    # Funciones para el selector y visualización de goles
    @reactive.Effect
    @reactive.event(input.equipo_pases, input.partido)
    def _():
        partido_seleccionado = input.partido()
        equipo_pases = input.equipo_pases()
        
        if not partido_seleccionado or not equipo_pases:
            return
        
        match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
        
        try:
            eventos_partido = parser.event(match_id)[0]
            #("Columnas disponibles en eventos_partido:", eventos_partido.columns.tolist())  
            
            goal_shot_ids = get_shot_ids_for_team(eventos_partido, equipo_pases)
            
            if goal_shot_ids:
                options = {str(shot_id): f"Gol {i+1}" for i, shot_id in enumerate(goal_shot_ids)}
                ui.update_select("gol_seleccionado", choices=options)
            else:
                ui.update_select("gol_seleccionado", choices={})
        except Exception as e:
            print(f"Error al cargar goles: {str(e)}")
            ui.update_select("gol_seleccionado", choices={})

    @output
    @render.ui
    def selector_goles():
        partido_seleccionado = input.partido()
        equipo_pases = input.equipo_pases()
        
        if not partido_seleccionado or not equipo_pases:
            return ui.p("Selecciona un partido y equipo para ver los goles.")
        
        match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
        eventos_partido = parser.event(match_id)[0]
        
        goal_shot_ids = get_shot_ids_for_team(eventos_partido, equipo_pases)
        
        if not goal_shot_ids:
            return ui.p("")
        
        return ui.input_select("gol_seleccionado", "Selecciona un gol para ver:", 
                             {str(shot_id): f"Gol {i+1}" for i, shot_id in enumerate(goal_shot_ids)})

    @output
    @render.plot
    def grafico_goles():
        partido_seleccionado = input.partido()
        equipo_pases = input.equipo_pases()
        
        if not partido_seleccionado or not equipo_pases:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Selecciona un partido y equipo para ver los goles.", 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        
        match_id = int(partido_seleccionado.split(" - ")[-1]) if " - " in partido_seleccionado else int(partido_seleccionado)
        
        try:
            eventos_partido = parser.event(match_id)[0]
            goal_shot_ids = get_shot_ids_for_team(eventos_partido, equipo_pases)
            
            if not goal_shot_ids:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, f"{equipo_pases} no marcó goles en este partido.", 
                        ha='center', va='center', fontsize=12)
                ax.axis('off')
                return fig
            
            gol_seleccionado = input.gol_seleccionado()
            
            # Cambio clave: Eliminar la conversión a int() y comparar directamente como strings
            if not gol_seleccionado or gol_seleccionado not in goal_shot_ids:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "Selecciona un gol válido para visualizar", 
                        ha='center', va='center', fontsize=12)
                ax.axis('off')
                return fig
            
            freeze = parser.event(match_id)[2]
            lineup = parser.lineup(match_id)
            lineup = lineup[['player_id', 'jersey_number', 'team_name']].copy()
            
            fig = plot_shot_freeze_frame(gol_seleccionado, freeze, eventos_partido, lineup)
            return fig
        
        except Exception as e:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error al cargar el gol: {str(e)}", 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig

    def get_shot_ids_for_team(df, team_name):
        """
        Gets the shot IDs of goals scored by a given team.
        Args:
            df: The StatsBomb event data DataFrame.
            team_name: The name of the team.
        Returns:
            A list of shot IDs (as strings) corresponding to goals scored by the team.
        """
        try:
            team_goals = df[
                (df['team_name'] == team_name) &
                (df['type_name'] == 'Shot') &
                (df['outcome_name'] == 'Goal')
            ]
            # Devolver los IDs como strings (sin conversión a int)
            return team_goals['id'].astype(str).tolist()
        except Exception as e:
            print(f"Error al obtener goles: {str(e)}")
            return []
        
    
    @render.data_frame
    def top_goleadores():
        df = df_jugadores_filtered.copy()
        posicion = input.posicion_top()
        # Aplicar filtro de liga
        if input.liga_top() != "Todas":
            df = df[df['Comp'] == input.liga_top()]
        
        # Aplicar filtro de posición con mapeo más flexible
        if posicion != "Todos":
            pos_map = {
                "Delantero": ["FW", "ST", "CF", "LW", "RW", "RF", "LF"],
                "Mediocentro": ["MF", "CM", "DM", "AM", "LM", "RM", "CAM", "CDM"],
                "Defensa": ["DF", "CB", "LB", "RB", "WB", "LWB", "RWB"],
                "Portero": ["GK"]
            }
            posiciones = pos_map.get(posicion, [])

            # Nos quedamos con la primera posición
            df["Primera_Pos"] = df["Pos"].str.split(",").str[0].str.strip()

            # Filtramos por si la primera posición está en la lista
            df = df[df["Primera_Pos"].isin(posiciones)]
        
        edad_map = {"U25": 25, "U23": 23, "U21": 21, "U19": 19}
        if input.edad_top() != "Todos":
            df = df[df["Age"] <= edad_map[input.edad_top()]]
            
        
        # Ordenar y seleccionar top 10
        top = df.sort_values('Gls', ascending=False).head(10)
        
        # Seleccionar columnas disponibles
        columnas = {
            'Player': 'Jugador',
            'Squad': 'Equipo',
            'Gls': 'Goles',
            'xG': 'xG',
            'G-xG': 'G-xG',
            'npxG': 'xG no penal',
            'G-PK': 'Goles sin penaltis',
            'Gls_90': 'Goles/90'
        }
        columnas_seleccionadas = [k for k in columnas.keys() if k in df.columns]
        top = top[columnas_seleccionadas]
        top = top.rename(columns=columnas)
        
        return top.round(2)

    @render.data_frame
    def top_asistentes():
        df = df_jugadores_filtered.copy()

        posicion = input.posicion_top()

        if input.liga_top() != "Todas":
            df = df[df['Comp'] == input.liga_top()]
        
        if posicion != "Todos":
            pos_map = {
                "Delantero": ["FW", "ST", "CF", "LW", "RW", "RF", "LF"],
                "Mediocentro": ["MF", "CM", "DM", "AM", "LM", "RM", "CAM", "CDM"],
                "Defensa": ["DF", "CB", "LB", "RB", "WB", "LWB", "RWB"],
                "Portero": ["GK"]
            }
            posiciones = pos_map.get(posicion, [])

            # Nos quedamos con la primera posición
            df["Primera_Pos"] = df["Pos"].str.split(",").str[0].str.strip()

            # Filtramos por si la primera posición está en la lista
            df = df[df["Primera_Pos"].isin(posiciones)]

        edad_map = {"U25": 25, "U23": 23, "U21": 21, "U19": 19}
        if input.edad_top() != "Todos":
            df = df[df["Age"] <= edad_map[input.edad_top()]]
        
        top = df.sort_values('Ast', ascending=False).head(10)
        
        columnas = {
            'Player': 'Jugador',
            'Squad': 'Equipo',
            'Ast': 'Asistencias',
            'xAG': 'xAG',
            'Ast_90': 'Asistencias/90',
            'A-xAG':'A-xAG',
            'KP': 'Pases clave',
            'CrsPA': 'Centros al área'
        }
        columnas_seleccionadas = [k for k in columnas.keys() if k in df.columns]
        top = top[columnas_seleccionadas]
        top = top.rename(columns=columnas)
        
        return top.round(2)

    @render.data_frame
    def top_porteros():
        df = df_jugadores_filtered.copy()
        
        # Filtramos solo porteros independientemente del filtro de posición
        df = df[df['Pos'] == 'GK']
        
        if input.liga_top() != "Todas":
            df = df[df['Comp'] == input.liga_top()]

        edad_map = {"U25": 25, "U23": 23, "U21": 21, "U19": 19}
        if input.edad_top() != "Todos":
            df = df[df["Age"] <= edad_map[input.edad_top()]]
        
        # Calculamos % de paradas si no existe
        if 'Save%' not in df.columns and 'Saves' in df.columns and 'SoTA' in df.columns:
            df['Save%'] = (df['Saves'] / df['SoTA'] * 100).round(1)
        
        sort_column = 'PSxG+/-' if 'PSxG+/-' in df.columns else 'Saves'
        top = df.sort_values(sort_column, ascending=False).head(10)
        
        columnas = {
            'Player': 'Jugador',
            'Squad': 'Equipo',
            'PSxG+/-': 'Diferencia de goles esperados',
            'Save%': '% Paradas',
            'Saves': 'Paradas',
            'SoTA': 'Tiros recibidos',
            'GA': 'Goles encajados',
            'GA90': 'Goles encajados/90'
        }
        columnas_seleccionadas = [k for k in columnas.keys() if k in df.columns]
        top = top[columnas_seleccionadas]
        top = top.rename(columns=columnas)
        
        return top.round(2)
    
    @reactive.Calc
    def df_filtrado_por_posicion():
        posicion_ui = input.posicion_radar()
        posicion_df = pos_map.get(posicion_ui, None)
    
        if posicion_df:
            return df_jugadores_filtered[
                df_jugadores_filtered["Pos"].str.split(",").str[0].str.strip() == posicion_df
            ]
        return df_jugadores_filtered

    @reactive.Effect
    def actualizar_selectize_jugadores():
        jugadores = df_filtrado_por_posicion()["Player"].tolist()
        ui.update_selectize("jugadores", choices=jugadores)
    

    @reactive.Calc
    def jugadores_seleccionados():
        nombres = input.jugadores()
        if not nombres:
            return []
        
        df = df_filtrado_por_posicion()
        seleccionados = df[df["Player"].isin(nombres)]
        
        # Mapear las estadísticas a las necesarias para el radar
        return seleccionados.to_dict(orient="records")

    def get_estadisticas_posicion():
        pos_ui = input.posicion_radar()
        return estadisticas_por_posicion.get(pos_map.get(pos_ui, ""), {})


    @output
    @render.plot
    def grafico_radar_dinamico():
        players = input.jugadores()
        if not players:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, "Selecciona jugadores para visualizarlos", 
                    ha='center', va='center', fontsize=14, color='gray')
            ax.axis('off')
            return fig

        df = df_filtrado_por_posicion()
        config = get_estadisticas_posicion()

        params = config.get("labels", [])
        cols = config.get("columns", [])
        low = config.get("low", [])
        high = config.get("high", [])
        lower_is_better = config.get("lower_is_better", [])

        radar = Radar(params, low, high,
                    round_int=[False]*len(params),
                    num_rings=4, ring_width=1, center_circle_radius=1)

        fig, ax = radar.setup_axis()
        fig.set_size_inches(12, 12)
        radar.draw_circles(ax=ax, facecolor='none', edgecolor='black', zorder=1)

        for idx, name in enumerate(players):
            row = df[df["Player"] == name]
            if row.empty:
                continue
            row = row.iloc[0]

            values = []
            for col in cols:
                val = row.get(f"percentil_{col}", 0)
                if col in lower_is_better:
                    val = val  # invertir el percentil
                values.append(val)

            color = colores[idx % len(colores)]
            radar_output = radar.draw_radar(values, ax=ax,
                                            kwargs_radar={'facecolor': color, 'alpha': 0.2},
                                            kwargs_rings={'facecolor': color, 'alpha': 0})
            _, _, vertices = radar_output
            ax.scatter(vertices[:, 0], vertices[:, 1], c=color, edgecolors=color, marker='o', s=150, zorder=3)
            ax.plot(np.append(vertices[:, 0], vertices[0, 0]),
                    np.append(vertices[:, 1], vertices[0, 1]),
                    color=color, linewidth=2, linestyle='-', zorder=1, alpha=0.8)

        lines = radar.spoke(ax=ax, color='black', linestyle='-', zorder=2)
        radar.draw_param_labels(ax=ax, fontsize=10)

        return fig


    @output
    @render.ui
    def tabla_estadisticas_radar():
        player_names = input.jugadores()
        if not player_names:
            return "Selecciona jugadores para ver las estadísticas."
        
        players_data = jugadores_seleccionados()
        ordered_players = []
        for name in player_names:
            for player in players_data:
                if player["Player"] == name:
                    ordered_players.append(player)
                    break

        # Obtener columnas y etiquetas de radar según posiciones
        estadisticas = []
        for player in ordered_players:
            main_pos = player["Pos"].split(",")[0].strip()
            config = estadisticas_por_posicion.get(main_pos, {})
            labels = config.get("labels", [])
            columns = config.get("columns", [])
            estadisticas = list(zip(labels, columns))
            break  # Solo usamos la primera posición encontrada

        # Columnas adicionales: Equipo, Edad, Minutos
        info_extra = [
            ("Equipo", "Squad"),
            ("Edad", "Age"),
        ]
        estadisticas_minutos = [
            ("Minutos", "Min")
        ]

        # Construir tabla
        html_table = """
        <style>
            .custom-table {
                width: 100%;
                border-collapse: collapse;
                font-family: Arial, sans-serif;
            }
            .custom-table th, .custom-table td {
                padding: 8px;
                border: 1px solid #ddd;
                text-align: left;
            }
            .custom-table th {
                background-color: #f8f9fa;
                font-weight: bold;
            }
        </style>
        <table class='custom-table'>
        <thead>
            <tr><th>Estadística</th>
        """

        # Encabezados con nombres y colores
        for idx, player in enumerate(ordered_players):
            color = colores[idx % len(colores)]
            html_table += f"<th style='color: {color};'>{player['Player']}</th>"
        html_table += "</tr></thead><tbody>"

        # Equipo y Edad
        for label, col in info_extra:
            html_table += f"<tr><td><strong>{label}</strong></td>"
            for player in ordered_players:
                value = player.get(col, "N/A")
                if col == "Age" and isinstance(value, float):
                    value = int(value)
                html_table += f"<td>{value}</td>"
            html_table += "</tr>"

        # Estadísticas técnicas con percentiles
        for label, col in estadisticas:
            html_table += f"<tr><td><strong>{label}</strong></td>"
            for player in ordered_players:
                value = player.get(col, "N/A")
                perc_col = f"percentil_{col}"
                percentil = player.get(perc_col, None)

                if isinstance(value, float):
                    value = f"{value:.2f}"
                elif isinstance(value, int):
                    value = f"{value}"
                
                if value != "NaN" and percentil is not None:
                    value = f"{value} ({int(percentil)})"

                html_table += f"<td>{value}</td>"
            html_table += "</tr>"

        # Minutos jugados
        for label, col in estadisticas_minutos:
            html_table += f"<tr><td><strong>{label}</strong></td>"
            for player in ordered_players:
                value = player.get(col, "N/A")
                if isinstance(value, float):
                    value = f"{value:.2f}"
                html_table += f"<td>{value}</td>"
            html_table += "</tr>"

        html_table += "</tbody></table>"
        
        return ui.HTML(html_table)

    @output
    @render.data_frame
    def jugadores_similares():
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import RobustScaler
        from sklearn.metrics.pairwise import cosine_similarity

        # 1. Verificar si hay jugadores seleccionados
        players = input.jugadores()
        if not players:
            return pd.DataFrame(columns=["Player", "Squad", "Pos", "Age", "Similitud"])
        
        # 2. Obtener datos del jugador de referencia
        jugador_ref = players[0]
        df = df_jugadores_filtered.copy()
        
        # 3. Seleccionar columnas relevantes para la similitud
        columnas_similitud = [
            col for col in df.columns 
            if df[col].dtype in [float, int] 
            and col not in ['Age', 'Min', 'Player', 'Squad', 'Comp', 'Pos', 'Nation']
            and not col.endswith(('percentil_'))
        ]
        
        # 4. Limpieza de datos
        df_sim = df[columnas_similitud].copy()
        
        # Reemplazar infinitos y valores extremadamente grandes
        df_sim = df_sim.replace([np.inf, -np.inf], np.nan)
        
        # Eliminar columnas con todos valores nulos
        df_sim = df_sim.dropna(axis=1, how='all')
        
        # Imputar valores nulos con la mediana (más robusta que la media)
        df_sim = df_sim.fillna(df_sim.median())
        
        # 5. Verificación de datos válidos
        if df_sim.empty or len(df_sim) < 2:
            return pd.DataFrame({"Mensaje": ["Datos insuficientes para comparación"]})
        
        # 6. Escalado robusto (mejor para datos con outliers)
        try:
            scaler = RobustScaler()
            stats_scaled = scaler.fit_transform(df_sim)
        except Exception as e:
            return pd.DataFrame({"Error": [f"Error en escalado: {str(e)}"]})
        
        # 7. Encontrar índice del jugador referencia
        try:
            mask = df['Player'] == jugador_ref
            referencia_idx = df[mask].index[0]
        except IndexError:
            return pd.DataFrame({"Error": [f"Jugador {jugador_ref} no encontrado"]})
        
        # 8. Validación de índice
        if referencia_idx >= len(stats_scaled):
            return pd.DataFrame({"Error": ["Índice de referencia inválido"]})
        
        # 9. Cálculo de similitud del coseno
        try:
            similitudes = cosine_similarity(
                [stats_scaled[referencia_idx]], 
                stats_scaled
            )[0]
        except Exception as e:
            return pd.DataFrame({"Error": [f"Error en cálculo de similitud: {str(e)}"]})
        
        # 10. Crear DataFrame con resultados
        df_resultado = df.assign(Similitud=similitudes)
        
        # 11. Filtrar y ordenar resultados
        similares = df_resultado[
            (df_resultado['Player'] != jugador_ref)
        ].sort_values('Similitud', ascending=False).head(5)
        
        # 12. Formatear salida
        columnas_mostrar = ['Player', 'Squad', 'Pos', 'Age', 'Similitud']
        
        if not similares.empty:
            # Redondear y convertir similitud a porcentaje
            similares['Similitud'] = (similares['Similitud'] * 100).round(1)
            return similares[columnas_mostrar]
        else:
            return pd.DataFrame({"Mensaje": ["No se encontraron jugadores similares"]})



    def plot_pizza_chart(player_name, position, params, values):  # Añade position como parámetro
        font_normal = font_manager.FontProperties(family='DejaVu Sans')
        font_bold = font_manager.FontProperties(family='DejaVu Sans', weight='bold')
        font_italic = font_manager.FontProperties(family='DejaVu Sans', style='italic')

        # Usamos la posición pasada como parámetro en lugar de input.posicion_pizza()
        if position == "GK":
            # Mapeo entre nombres de visualización y columnas técnicas
            param_to_column = {
                "xG Recibido (PSxG)": "PSxG",
                "Rendimiento xG (+/-)": "PSxG+/-",
                "% Paradas (Save%)": "Save%",
                "Paradas por 90'": "Saves_90",
                "Goles Recibidos/90'": "GA90",
                "Despejes por 90'": "Clr_90",
                "Tiros Recibidos Totales": "SoTA"
            }
            

            # Definición de qué métricas son mejores cuando son más bajas (True) o más altas (False)                
            lower_is_better = {
                "GA90": True,         # Goles recibidos por 90': menor es mejor
                "Won%": False,        # % de duelos aéreos ganados: mayor es mejor
                "PSxG": True,        # xG recibido: menor es mejor
                "PSxG+/-": False,     # Rendimiento xG: mayor es mejor (diferencia positiva)
                "Save%": False,       # % de paradas: mayor es mejor
                "Saves_90": False,    # Paradas por 90': mayor es mejor
                "Clr_90": False,      # Despejes por 90': mayor es mejor
                "SoTA": False          # Tiros recibidos: menor es mejor (indica menos presión)
            }
            
            # Ajustar los valores según si es mejor alto o bajo
            adjusted_values = []
            inverted_metrics = []
            for param, val in zip(params, values):
                col_name = param_to_column.get(param, param)
                if lower_is_better.get(col_name, False):
                    adjusted_values.append(100 - val)
                    inverted_metrics.append(param)
                else:
                    adjusted_values.append(val)
            
            # Configuración visual
            slice_colors = ["#1A78CF"] * len(params)
            value_colors = ["white"] * len(params)

            baker = PyPizza(
                params=params,
                background_color="#FFFFFF",
                straight_line_color="#EBEBE9",
                straight_line_lw=1,
                last_circle_lw=0,
                other_circle_lw=0,
                inner_circle_size=20
            )

            fig, ax = baker.make_pizza(
                adjusted_values,
                figsize=(6, 6),
                color_blank_space="same",
                slice_colors=slice_colors,
                value_colors=value_colors,
                value_bck_colors=slice_colors,
                blank_alpha=0.4,
                param_location=115,
                kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
                kwargs_params=dict(
                    color="#000000", fontsize=8,
                    fontproperties=font_normal, va="center"
                ),
                kwargs_values=dict(
                    fontsize=11,
                    fontproperties=font_bold,
                    zorder=3,
                    color="#000000",
                    bbox=dict(
                        edgecolor="#000000",
                        facecolor=slice_colors,
                        boxstyle="round,pad=0.2",
                        lw=1
                    )
                )
            )

            # Título y subtítulos
            fig.text(0.02, 0.94, player_name, size=14, fontproperties=font_bold, color="#000000", ha="left")
            fig.text(0.02, 0.91, "Percentile Rank vs Top-Five League Goalkeepers", 
                    size=10, fontproperties=font_normal, color="#000000", ha="left")
            fig.text(0.02, 0.88, "Temporada 2024-25", 
                    size=10, fontproperties=font_normal, color="#000000", ha="left")

            # Leyenda mejorada
            leyenda_x = 0.80
            leyenda_y = 0.92
            fig.text(leyenda_x, leyenda_y, "Leyenda:", size=10, fontproperties=font_bold, color="#000000")
            fig.patches.append(
                plt.Rectangle((leyenda_x, leyenda_y - 0.03), 0.02, 0.02, 
                            color="#1A78CF", transform=fig.transFigure, figure=fig)
            )
            fig.text(leyenda_x + 0.025, leyenda_y - 0.02, "Portería", 
                    size=9, fontproperties=font_normal, va="center")
            
            # Nota sobre métricas invertidas
            if inverted_metrics:
                inverted_text = "Métricas invertidas (menor = mejor):\n" + "\n".join(
                    f"• {m}" for m in inverted_metrics[:3])  # Muestra máximo 3
                if len(inverted_metrics) > 3:
                    inverted_text += "\n• ..."
                
                fig.text(
                    0.80, 0.15, inverted_text,
                    size=7, fontproperties=font_italic, color="#555555", ha="left"
                )


        else:
            slice_colors = ["#1A78CF"] * 5 + ["#FF9300"] * 5 + ["#D70232"] * 5
            slice_colors = slice_colors[:len(params)]
            value_colors = ["white"] * len(params)

            baker = PyPizza(
                params=params,
                background_color="#FFFFFF",
                straight_line_color="#EBEBE9",
                straight_line_lw=1,
                last_circle_lw=0,
                other_circle_lw=0,
                inner_circle_size=20
            )

            fig, ax = baker.make_pizza(
                values,
                figsize=(6, 6),
                color_blank_space="same",
                slice_colors=slice_colors,
                value_colors=value_colors,
                value_bck_colors=slice_colors,
                blank_alpha=0.4,
                param_location=115,
                kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
                kwargs_params=dict(
                    color="#000000", fontsize=8,
                    fontproperties=font_normal, va="center"
                ),
                kwargs_values=dict(
                    fontsize=11, 
                    fontproperties=font_bold,
                    zorder=3,
                    color="#000000",
                    bbox=dict(
                        edgecolor="#000000", 
                        facecolor=slice_colors,
                        boxstyle="round,pad=0.2", 
                        lw=1
                    )
                )
            )

            # 🏷️ Título
            # Obtener el equipo del jugador (si está disponible)
            equipo = ""
            try:
                # Buscar el equipo en el DataFrame global si existe
                if "df_jugadores_filtered" in globals():
                    df = globals()["df_jugadores_filtered"]
                    equipo = df.loc[df["Player"] == player_name, "Squad"].values[0]
            except Exception:
                equipo = ""

            titulo = player_name
            if equipo:
                if player_name == "Vitinha":
                    if position == "FW":
                        titulo += " (Genoa)"
                    else:
                        titulo += " (Paris S-G)"
                else:
                    titulo += f" ({equipo})"

            fig.text(
                0.02, 0.94, titulo,
                size=14, fontproperties=font_bold, color="#000000", ha="left"
            )

            # 🧾 Subtítulo
            fig.text(
                0.02, 0.91, "Percentile Rank vs Top-Five League Players",
                size=10, fontproperties=font_normal, color="#000000", ha="left"
            )
            fig.text(
                0.02, 0.88, "Temporada 2024-25",
                size=10, fontproperties=font_normal, color="#000000", ha="left"
            )

            # 📘 Leyenda
            leyenda_x = 0.80
            leyenda_y = 0.92
            fig.text(leyenda_x, leyenda_y, "Categories:", size=10, fontproperties=font_bold, color="#000000")
            fig.patches.extend([
                plt.Rectangle((leyenda_x, leyenda_y - 0.03), 0.02, 0.02, color="#1A78CF", transform=fig.transFigure, figure=fig),
                plt.Rectangle((leyenda_x, leyenda_y - 0.06), 0.02, 0.02, color="#FF9300", transform=fig.transFigure, figure=fig),
                plt.Rectangle((leyenda_x, leyenda_y - 0.09), 0.02, 0.02, color="#D70232", transform=fig.transFigure, figure=fig),
            ])
            fig.text(leyenda_x + 0.025, leyenda_y - 0.02, "Defensa", size=9, fontproperties=font_normal, va="center")
            fig.text(leyenda_x + 0.025, leyenda_y - 0.05, "Posesion", size=9, fontproperties=font_normal, va="center")
            fig.text(leyenda_x + 0.025, leyenda_y - 0.08, "Ataque", size=9, fontproperties=font_normal, va="center")

        # 📄 Fuente (común para ambos casos)
        fig.text(
            0.99, 0.02,
            "Data: StatsBomb and FBref\nInspired by @Worville, @FootballSlices, @somazerofc, @Soumyaj15209314 & MPLSoccer",
            size=8, fontproperties=font_italic, color="#000000", ha="right"
        )

        return fig


    def generar_pizza_con_percentiles(nombre_jugador, posicion_ui, df_jugadores_filtered, estadisticas_por_posicion_pizza):
        posicion = pos_map.get(posicion_ui)
        if not posicion:
            raise ValueError("Invalid position")

        config = estadisticas_por_posicion_pizza.get(posicion)
        if not config:
            raise ValueError("Invalid position")

        stats = config["stats"]
        labels_dict = config["labels"]
        labels = [labels_dict[stat] for stat in stats]

        # Obtener todos los jugadores con ese nombre y posición
        df_jugador_candidatos = df_jugadores_filtered[
            (df_jugadores_filtered["Player"] == nombre_jugador) &
            (df_jugadores_filtered["Pos"].str.split(",").str[0].str.strip() == posicion)
        ]

        if df_jugador_candidatos.empty:
            raise ValueError(f"No se encontró al jugador '{nombre_jugador}' con posición '{posicion}'.")

        # Elegir el jugador del equipo más reciente (último en el DataFrame)
        jugador_data = df_jugador_candidatos.iloc[-1]

        # Filtrar jugadores de la misma posición
        df_posicion = df_jugadores_filtered[
            df_jugadores_filtered["Pos"].str.split(",").str[0].str.strip() == posicion
        ]

        percentiles = []
        for stat in stats:
            if stat not in df_posicion.columns:
                raise KeyError(f"Column {stat} not found in DataFrame")

            valores = df_posicion[stat].dropna()
            valor_jugador = jugador_data[stat]

            pct = np.round((valores < valor_jugador).mean() * 100, 0)
            percentiles.append(pct)

        fig = plot_pizza_chart(nombre_jugador, posicion, labels, percentiles)
        return fig



    @reactive.Calc
    def df_filtrado_por_posicion_pizza():
        posicion_ui = input.posicion_pizza()
        posicion_codigo = pos_map.get(posicion_ui, None)
        if posicion_codigo:
            return df_jugadores_filtered[
                df_jugadores_filtered["Pos"].str.split(",").str[0].str.strip() == posicion_codigo
            ]
        return df_jugadores_filtered
    

    @reactive.Effect
    def actualizar_selectize_jugadores_pizza():
        jugadores_pizza = df_filtrado_por_posicion_pizza()["Player"].tolist()
        ui.update_selectize("jugadores_pizza", choices=jugadores_pizza)


    @output
    @render.plot
    def grafico_pizza():
        jugadores = input.jugadores_pizza()
        posicion_ui = input.posicion_pizza()
        if not jugadores or len(jugadores) == 0 or not posicion_ui:
            return None
        jugador = jugadores[0]

        try:
            fig = generar_pizza_con_percentiles(jugador, posicion_ui, df_jugadores_filtered, estadisticas_por_posicion_pizza)
            return fig
        except Exception as e:
            print(f"Error en generación gráfica: {e}")
            return None

    def calcular_fortalezas_y_debilidades(jugador, posicion_ui, df_jugadores_filtered, column_display_names):
        columnas_excluir = excluded_columns
        portero_cols = [
            'PSxG', 'PSxG+/-', 'GA', 'GA90', 'Saves', 'Saves_90',
            'Save%', 'SoTA', 'Cs', 'Cs%'
        ]

        posicion_codigo = pos_map.get(posicion_ui)

        df_pos = df_jugadores_filtered[
            df_jugadores_filtered["Pos"].str.split(",").str[0].str.strip() == posicion_codigo
        ]

        columnas_numericas = [
            col for col in df_jugadores_filtered.select_dtypes(include=[np.number]).columns
            if "percentil" not in col.lower() and col not in columnas_excluir
        ]

        if posicion_codigo == "GK":
            columnas_numericas = [col for col in columnas_numericas if col in portero_cols]
        else:
            columnas_numericas = [col for col in columnas_numericas if col not in portero_cols]

        # Buscar jugador exacto por nombre y posición, tomar el más reciente
        df_jugador_candidatos = df_jugadores_filtered[
            (df_jugadores_filtered["Player"] == jugador) &
            (df_jugadores_filtered["Pos"].str.split(",").str[0].str.strip() == posicion_codigo)
        ]

        if df_jugador_candidatos.empty:
            return ui.HTML("⚠️ No se encontró una fila válida para este jugador y posición.")

        valores_jugador = df_jugador_candidatos.iloc[-1]

        columnas_validas = [
            col for col in columnas_numericas
            if not pd.isna(valores_jugador[col])
        ]

        lower_is_better_cols = {
            'GA', 'GA90'
        }

        rankings = {}
        for col in columnas_validas:
            serie = df_pos[col].dropna()
            ascending = col in lower_is_better_cols
            serie_ordenada = serie.sort_values(ascending=ascending)
            serie_ranking = serie_ordenada.rank(method="min", ascending=ascending)

            try:
                ranking_jugador = int(serie_ranking[df_pos["Player"] == jugador].values[0])
                rankings[col] = ranking_jugador
            except Exception:
                continue

        if not rankings:
            return ui.HTML("⚠️ No se pudo calcular el ranking del jugador.")

        mejores = sorted(rankings.items(), key=lambda x: x[1])[:5]
        peores = sorted(rankings.items(), key=lambda x: x[1], reverse=True)[:5]

        def traducir(col):
            return column_display_names.get(col, col)

        html = "<h4>🔝 Puntos fuertes</h4><ul>"
        for col, rank in mejores:
            valor = valores_jugador[col]
            if "_90" in col:
                valor_str = f"{valor:.2f}"
            else:
                valor_str = f"{int(round(valor))}"
            html += f"<li><b>{traducir(col)}</b>: {valor_str} (top {rank} de {len(df_pos)})</li>"

        html += "</ul><h4>⚠️ Puntos débiles</h4><ul>"
        for col, rank in peores:
            valor = valores_jugador[col]
            if "_90" in col:
                valor_str = f"{valor:.2f}"
            else:
                valor_str = f"{int(round(valor))}"
            html += f"<li><b>{traducir(col)}</b>: {valor_str} (top {rank} de {len(df_pos)})</li>"
        html += "</ul>"


        return ui.HTML(html)


    
    @output
    @render.ui
    def fortalezas_debilidades():
        jugadores = input.jugadores_pizza()
        posicion_ui = input.posicion_pizza()
        if not jugadores or len(jugadores) == 0 or not posicion_ui:
            return ui.HTML("<i>Selecciona un jugador para ver fortalezas y debilidades</i>")
        
        jugador = jugadores[0]
        try:
            return calcular_fortalezas_y_debilidades(jugador, posicion_ui, df_jugadores_filtered, column_display_names)
        except Exception as e:
            print(f"Error calculando fortalezas/debilidades: {e}")
            return ui.HTML("⚠️ No se pudo calcular el ranking del jugador.")


    @reactive.Calc
    def df_jugadores_filtrados_scatter():
        posicion_ui = input.posicion_scatter()
        posicion_codigo = pos_map.get(posicion_ui, None)
        liga = input.liga_scatter()
        edad = input.edad_scatter()

        df_filtrado = df_jugadores_filtered.copy()
        
        # Filtro por posición principal (primera posición listada)
        if posicion_codigo:
            df_filtrado = df_filtrado[
                df_filtrado["Pos"].str.split(",").str[0].str.strip() == posicion_codigo
            ]
        
        # Filtro por liga
        if liga != "Todas":
            df_filtrado = df_filtrado[df_filtrado["Comp"] == liga]
        
        # Filtro por edad
        edad_map = {"U25": 25, "U23": 23, "U21": 21, "U19": 19}
        if edad != "Todos":
            df_filtrado = df_filtrado[df_filtrado["Age"] <= edad_map[edad]]
        
        return df_filtrado

    
    @reactive.Effect
    def actualizar_selectize_jugadores_scatter():
        jugadores_scatter = df_jugadores_filtrados_scatter()["Player"].tolist()
        ui.update_selectize("jugadores_scatter", choices=jugadores_scatter)


    @output
    @render.plot
    def scatter_plot():
        df_actual = df_jugadores_filtrados_scatter()

        fig, ax = plt.subplots(figsize=(10, 6))
        minutos = df_actual["Min"]
        tamano_puntos = (minutos - minutos.min()) / (minutos.max() - minutos.min()) * 300 + 50

        ax.scatter(
            df_actual[input.x_axis()],
            df_actual[input.y_axis()],
            c="#add8e6",
            alpha=0.6,
            edgecolor="w",
            s=tamano_puntos
        )

        if input.mostrar_nombres():
            for _, row in df_actual.iterrows():
                ax.text(
                    row[input.x_axis()],
                    row[input.y_axis()] + 0.015 * df_actual[input.y_axis()].max(),
                    row["Player"],
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    alpha=0.6
                )

        if input.jugadores_scatter():
            colores = plt.cm.tab20.colors[:20]
            jugadores_ordenados = input.jugadores_scatter()[:20]
            color_map = {jugador: colores[i] for i, jugador in enumerate(jugadores_ordenados)}

            df_jugadores_resaltados = df_actual[
                df_actual["Player"].isin(jugadores_ordenados)
            ].sort_values(
                "Player",
                key=lambda x: pd.Categorical(x, categories=jugadores_ordenados, ordered=True)
            )

            for _, row in df_jugadores_resaltados.iterrows():
                color = color_map[row["Player"]]
                ax.scatter(
                    row[input.x_axis()],
                    row[input.y_axis()],
                    c=[color],
                    edgecolor="black",
                    s=300,
                    alpha=0.9,
                    zorder=10
                )
                ax.text(
                    row[input.x_axis()],
                    row[input.y_axis()] + 0.025 * df_actual[input.y_axis()].max(),
                    row["Player"],
                    fontsize=9,
                    ha="center",
                    va="bottom",
                    fontweight="regular",
                    color="black",
                    alpha=0.9
                )

        if input.mostrar_mediana():
            med_x = df_actual[input.x_axis()].median()
            med_y = df_actual[input.y_axis()].median()
            ax.axvline(med_x, color="black", linestyle="--", linewidth=1.5)
            ax.axhline(med_y, color="black", linestyle="--", linewidth=1.5)

        ax.set_xlabel(column_display_names.get(input.x_axis(), input.x_axis()), fontsize=12)
        ax.set_ylabel(column_display_names.get(input.y_axis(), input.y_axis()), fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

        return fig
    
    # Reactive calculation for index data
    @reactive.Calc
    def datos_indices():
        df = df_jugadores_filtered.copy()
        posicion = input.posicion_top()
        liga = input.liga_top()
        edad = input.edad_top()
        metric = input.indice_metrica()

        # FILTROS BÁSICOS

        if posicion != "Todos":
            pos_map = {
                "Delantero": ["FW", "ST", "CF", "LW", "RW", "RF", "LF"],
                "Mediocentro": ["MF", "CM", "DM", "AM", "LM", "RM", "CAM", "CDM"],
                "Defensa": ["DF", "CB", "LB", "RB", "WB", "LWB", "RWB"],
                "Portero": ["GK"]
            }
            posiciones = pos_map.get(posicion, [])

            # Nos quedamos con la primera posición
            df["Primera_Pos"] = df["Pos"].str.split(",").str[0].str.strip()

            # Filtramos por si la primera posición está en la lista
            df = df[df["Primera_Pos"].isin(posiciones)]



        if liga != "Todas":
            df = df[df["Comp"] == liga]

        edad_map = {"U25": 25, "U23": 23, "U21": 21, "U19": 19}
        if edad != "Todos":
            df = df[df["Age"] <= edad_map[edad]]

        # Calcular percentil si no está en el dataframe
        if f'percentil_{metric}' not in df.columns:
            df[f'percentil_{metric}'] = df[metric].rank(pct=True) * 100

        # FILTRO ESPECIAL PARA MÉTRICAS PORCENTUALES CON MÍNIMO DE INTENTOS
        metricas_porcentaje_min_requisitos = {
            'Cmp%': 'Att',
            'Tkl%': 'Tkl',
            'Won%': 'Att_stats_defense',
            'Save%': 'SoTA'
        }

        min_requeridos = metricas_porcentaje_min_requisitos.get(metric)
        if min_requeridos and min_requeridos in df.columns:
            df = df[df[min_requeridos] >= 10]

        

        # ORDENAR Y MOSTRAR TOP N
        return df.sort_values(metric, ascending=False).head(input.num_jugadores())
    
    # Dynamic title
    @output
    @render.text
    def indice_titulo():
        metric = input.indice_metrica()
        return f"Top {input.num_jugadores()} - {column_display_names.get(metric, metric)}"

    

    @output
    @render.ui
    def tabla_indices():
        df = datos_indices()
        metric = input.indice_metrica()
        # Extraer solo la primera posición
        df["Pos"] = df["Pos"].str.split(",").str[0].str.strip()

        df["Age"] = df["Age"].astype(int)
        # Definir nombres visibles
        cols = {
            'Player': 'Jugador',
            'Squad': 'Equipo',
            'Pos': 'Posición',
            'Age': 'Edad',
            metric: column_display_names.get(metric, metric)
        }

        df = df[list(cols.keys())].rename(columns=cols).round(2)

        # Generar HTML para la tabla centrada
        html = """
        <table style="width:100%; border-collapse: collapse; text-align: center;">
            <thead>
                <tr>
        """
        for col in df.columns:
            html += f"<th style='border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;'>{col}</th>"
        html += "</tr></thead><tbody>"

        for _, row in df.iterrows():
            html += "<tr>"
            for value in row:
                html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{value}</td>"
            html += "</tr>"
        html += "</tbody></table>"

        return ui.HTML(html)

   # En el servidor, conservamos esto:
    def sanitize_id(id_str: str) -> str:
        """Convierte nombres de estadísticas a IDs válidos para Shiny"""
        return id_str.replace("%", "_pct").replace("+", "_plus").replace("-", "_minus").replace("/","_")

    # Y modificamos las funciones clave:
    @output
    @render.ui
    def filtros_base():
        pos = input.posicion()
        if not pos:
            return ui.p("Selecciona una posición primero")
        
        # Obtener estadísticas fijas para la posición
        stats_config = estadisticas_por_posicion_buscar.get(pos, {})
        stats_fijas = stats_config.get("stats", [])
        labels = stats_config.get("labels", {})
        
        # Obtener métricas adicionales seleccionadas
        adicionales = input.metricas_adicionales() if "metricas_adicionales" in input else []
        
        # Combinar todas las estadísticas
        stats_todas = stats_fijas + list(adicionales)
        
        # Crear sliders para todas las stats
        sliders = []
        for stat in stats_todas:
            # Usar el nombre descriptivo si está disponible
            label = labels.get(stat, column_display_names.get(stat, stat))
            sliders.append(
                ui.input_slider(
                    id=sanitize_id(stat),
                    label=label,
                    min=0,
                    max=100,
                    value=0,
                    step=1
                )
            )
        
        return ui.div(*sliders)

    @output
    @render.ui
    def tabla_filtrada():
        pos = input.posicion()
        if not pos:
            return ui.HTML("<p>Selecciona una posición</p>")

        df = df_jugadores_filtered.copy()
        df = df[df["Pos"].str.contains(pos.split(",")[0].strip())]

        # Filtrar por edad
        age_min = input.age_min()
        age_max = input.age_max()
        df = df[(df["Age"] >= age_min) & (df["Age"] <= age_max)]

        # Obtener todas las estadísticas a considerar
        stats_config = estadisticas_por_posicion_buscar.get(pos, {})
        stats_fijas = stats_config.get("stats", [])
        adicionales = input.metricas_adicionales() if "metricas_adicionales" in input else []
        stats_todas = stats_fijas + list(adicionales)

        # Aplicar filtros de sliders
        for stat in stats_todas:
            slider_id = sanitize_id(stat)
            if slider_id in input:
                valor_slider = input[slider_id]()
                if valor_slider > 0:
                    col_percentil = f"percentil_{stat}"
                    if col_percentil in df.columns:
                        df = df[df[col_percentil] >= valor_slider]

        # Preparar columnas para mostrar
        columnas_base = ["Player", "Squad", "Age", "Min"]
        columnas_stats = []

        for stat in stats_todas:
            if stat in df.columns:
                if f"percentil_{stat}" in df.columns:
                    df = df.copy()
                    df[stat] = df.apply(lambda row: formatear_row(row, stat), axis=1)
                columnas_stats.append(stat)

        resultado = df[columnas_base + columnas_stats].copy()

        # 🛑 VERIFICAR SI EL RESULTADO ESTÁ VACÍO
        if resultado.empty:
            return ui.HTML("<p style='color: red; font-weight: bold'>⚠️ No existe ningún jugador que cumpla con esos registros.</p>")

        resultado["Player"] = resultado.apply(
            lambda row: f"<b>{row['Player']}</b><br><span style='color:gray; font-style:italic'>{row['Squad']}, {int(row['Age'])}, {int(row['Min'])} min</span>", 
            axis=1
        )
        resultado = resultado.sort_values("Min", ascending=False)
        resultado = resultado.drop(columns=["Squad", "Age", "Min"])

        resultado.columns = [
            "Jugador" if col == "Player" else column_display_names.get(col, col).replace(" ", "<br>")
            for col in resultado.columns
        ]

        tabla_html = resultado.to_html(
            classes="compact-table",
            escape=False,
            index=False,
            border=0
        )

        estilos_css = """
            <style>
            .compact-table {
                font-size: 13px;
                border-collapse: collapse;
                width: 100%;
            }
            .compact-table th, .compact-table td {
                padding: 10px 6px;
                text-align: center;
                border: 1px solid #ccc;
                height: 40px;
            }
            .compact-table th {
                background-color: #f2f2f2;
                font-size: 11px;
            }
            </style>
            """

        return ui.HTML(estilos_css + tabla_html)


    @output
    @render.ui
    def select_metricas_adicionales():
        pos = input.posicion()
        if not pos:
            return None
        # Calcular percentil si no está en el dataframe
        for metric in column_display_names:
            if f'percentil_{metric}' not in df_jugadores_filtered.columns and metric in df_jugadores_filtered.columns:
                df_jugadores_filtered[f'percentil_{metric}'] = df_jugadores_filtered[metric].rank(pct=True) * 100

        # Obtener todas las columnas disponibles que tienen percentiles
        todas_columnas = [col.replace("percentil_", "") for col in df_jugadores_filtered.columns 
                        if col.startswith("percentil_")]
        print(f"Columnas disponibles: {todas_columnas}")
        
        # Obtener estadísticas fijas para la posición
        stats_fijas = estadisticas_por_posicion_buscar.get(pos, {}).get("stats", [])
        
        # Filtrar para obtener solo las métricas adicionales (que no están en stats_fijas)
        adicionales = [s for s in todas_columnas 
                    if s not in stats_fijas and s in column_display_names]
        
        # Crear opciones con nombres descriptivos
        opciones = {}
        for categoria, metricas in categorias_metricas_adicionales.items():
            subgrupo = {
                m: nombre for m, nombre in metricas.items()
                if m in adicionales
            }
            if subgrupo:
                opciones[categoria] = subgrupo
        
        return ui.input_selectize(
            id="metricas_adicionales",
            label="Métricas adicionales para filtrar",
            choices=opciones,
            multiple=True
        )
    
    def formatear_row(row, stat):
        p = row.get(f'percentil_{stat}')
        if pd.isna(p):
            p_str = "N/A"  # O cualquier texto que prefieras para valores NaN
        else:
            p_str = str(int(p))
        return f"{row[stat]:.2f} <span style='color:gray; font-size:0.85em;'>({p_str})</span>"
    

    # ---------- Tabla predicción de GOLES ----------
    @output
    @render.ui
    def tabla_predicciones_goles():
        df = top_goleadores_pred.copy()

        # Convertir la predicción a int
        df["Goles Temporada Próxima"] = df["Goles Temporada Próxima"].astype(int)

        # Columna combinada Nombre + Equipo
        df["Jugador"] = df.apply(
            lambda row: (
                f"<b>{row['Jugador']}</b>"
                f"<br><span style='color:gray; font-style:italic'>{row['Equipo']}</span>"
            ),
            axis=1,
        )

        # Mantener columnas deseadas
        df = df[
            ["Jugador", "Goles Temporada Actual", "Goles Temporada Próxima"]
        ]

        # Convertir tabla a HTML
        tabla_html = df.to_html(
            escape=False,
            index=False,
            border=0,
            classes="comparative-table",
        )

        estilos = """
        <style>
        .comparative-table {
            font-size: 12px;
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }
        .comparative-table th,
        .comparative-table td {
            padding: 8px;
            text-align: center;
            border: 1px solid #ccc;
            word-wrap: break-word;
        }
        .comparative-table th {
            background-color: #f2f2f2;
        }
        </style>
        """

        return ui.HTML(estilos + tabla_html)


    # ---------- Tabla predicción de ASISTENCIAS ----------
    @output
    @render.ui
    def tabla_predicciones_asistencias():
        df = top_asistentes_pred.copy()

        # Convertir la predicción a int
        df["Asistencias Temporada Próxima"] = df["Asistencias Temporada Próxima"].astype(int)

        # Columna combinada Nombre + Equipo
        df["Jugador"] = df.apply(
            lambda row: (
                f"<b>{row['Jugador']}</b>"
                f"<br><span style='color:gray; font-style:italic'>{row['Equipo']}</span>"
            ),
            axis=1,
        )

        # Mantener columnas deseadas
        df = df[
            ["Jugador", "Asistencias Temporada Actual", "Asistencias Temporada Próxima"]
        ]

        # Convertir tabla a HTML
        tabla_html = df.to_html(
            escape=False,
            index=False,
            border=0,
            classes="comparative-table",
        )

        estilos = """
        <style>
        .comparative-table {
            font-size: 12px;
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }
        .comparative-table th,
        .comparative-table td {
            padding: 8px;
            text-align: center;
            border: 1px solid #ccc;
            word-wrap: break-word;
        }
        .comparative-table th {
            background-color: #f2f2f2;
        }
        </style>
        """

        return ui.HTML(estilos + tabla_html)

    @output
    @render.ui
    def video_player():
        selected_video = input.video_selector()
        return ui.tags.video(
            ui.tags.source(src=f"static/{selected_video}", type="video/mp4"),
            controls=True,
            width="640",
            height="360",
        )


app = App(
    app_ui,
    server,
    static_assets={"/static": str(STATIC_DIR.resolve())}
)

# Ejecutar la aplicación
app.run()