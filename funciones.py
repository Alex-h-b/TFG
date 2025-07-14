import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsbombpy import sb
from mplsoccer import Pitch,Sbopen,VerticalPitch
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import matplotlib.patches as patches


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
        "low": [0, -5, 50, 1, 0, 0, 1],
        "high": [10, 5, 90, 6, 3, 5, 10]
    },
    "DF": {
        "labels": [
            "Pases Completos/90'", 
            "% Éxito Pases",
            "Pases Progresivos/90'",
            "Acciones Defensivas/90'",
            "% Duelos Ganados",
            "% Duelos Aéreos Ganados",
            "Carreras Progresivas/90'",
            "xAG (Asistencias)/90'",
            "Centros al Área"
        ],
        "columns": ["Cmp_90", "Cmp%", "PrgP_90", "Tkl+Int_90", "TklW%", "Won%", "PrgC_90", "xAG_90", "CrsPa"],
        "low": [10, 60, 1, 1, 30, 30, 1, 0.1, 1],
        "high": [80, 95, 10, 10, 80, 80, 10, 2, 15]
    },
    "MF": {
        "labels": [
            "% Duelos Ganados",
            "Acciones Defensivas/90'", 
            "Carreras Progresivas/90'",
            "Pases Progresivos/90'",
            "Pases Clave/90'",
            "xAG (Asistencias)/90'",
            "Regates Exitosos/90'",
            "% Pases Completos",
            "Pases Completos/90'"
        ],
        "columns": ["Won%", "Tkl+Int_90", "PrgC_90", "PrgP_90", "KP_90", "xAG_90", "SuccDrib_90", "Cmp%", "Cmp_90"],
        "low": [30, 1, 1, 1, 0.5, 0.1, 0.5, 60, 10],
        "high": [80, 10, 15, 15, 5, 2, 6, 95, 80]
    },
    "FW": {
        "labels": [
            "% Duelos Aéreos Ganados", 
            "Goles sin Penaltis/90'",
            "xG (Expected Goals)/90'",
            "Diferencia Goles - xG",
            "Pases Clave/90'",
            "Asistencias/90'",
            "Regates Exitosos/90'",
            "Acciones de Creación de tiros/90'",
            "Carreras Progresivas/90'"
        ],
        "columns": ["Won%", "G-PK_90", "xG_90", "G-xG", "KP_90", "Ast_90", "SuccDrib_90", "SCA_90", "PrgC_90"],
        "low": [10, 0.1, 0.1, -0.5, 0.3, 0.1, 0.5, 2, 2],
        "high": [80, 1.0, 1.0, 0.5, 5, 2, 6, 10, 10]
    }
}

def resumen_estadisticas_partido(df, equipo_local, equipo_visitante):
    # Filtrar eventos por equipo
    eventos_local = df[df['team_name'] == equipo_local]
    eventos_visitante = df[df['team_name'] == equipo_visitante]

    # Calcular goles normales y goles en el minuto 120 o posterior
    goles_local_normales = eventos_local[(eventos_local['type_name'] == 'Shot') & 
                                         (eventos_local['outcome_name'] == 'Goal') & 
                                         (eventos_local['minute'] < 120)].shape[0]
    goles_local_120 = eventos_local[(eventos_local['type_name'] == 'Shot') & 
                                    (eventos_local['outcome_name'] == 'Goal') & 
                                    (eventos_local['minute'] >= 120)].shape[0]

    goles_visitante_normales = eventos_visitante[(eventos_visitante['type_name'] == 'Shot') & 
                                                 (eventos_visitante['outcome_name'] == 'Goal') & 
                                                 (eventos_visitante['minute'] < 120)].shape[0]
    goles_visitante_120 = eventos_visitante[(eventos_visitante['type_name'] == 'Shot') & 
                                            (eventos_visitante['outcome_name'] == 'Goal') & 
                                            (eventos_visitante['minute'] >= 120)].shape[0]

    # Calcular tiros
    tiros_local = eventos_local[eventos_local['type_name'] == 'Shot'].shape[0]
    tiros_visitante = eventos_visitante[eventos_visitante['type_name'] == 'Shot'].shape[0]

    # Calcular tiros a puerta
    tiros_puerta_local = eventos_local[(eventos_local['type_name'] == 'Shot') & 
                                       (eventos_local['outcome_name'].isin(['Goal', 'Saved']))].shape[0]
    tiros_puerta_visitante = eventos_visitante[(eventos_visitante['type_name'] == 'Shot') & 
                                               (eventos_visitante['outcome_name'].isin(['Goal', 'Saved']))].shape[0]

    # Calcular posesión (basado en la duración de las posesiones)
    posesion_local = df[df['possession_team_name'] == equipo_local]['duration'].sum()
    posesion_visitante = df[df['possession_team_name'] == equipo_visitante]['duration'].sum()
    total_posesion = posesion_local + posesion_visitante
    porcentaje_posesion_local = (posesion_local / total_posesion) * 100 if total_posesion > 0 else 0
    porcentaje_posesion_visitante = (posesion_visitante / total_posesion) * 100 if total_posesion > 0 else 0

    # Calcular pases
    pases_local = eventos_local[eventos_local['type_name'] == 'Pass'].shape[0]
    pases_visitante = eventos_visitante[eventos_visitante['type_name'] == 'Pass'].shape[0]

    # Calcular faltas
    faltas_local = eventos_local[eventos_local['type_name'] == 'Foul Committed'].shape[0]
    faltas_visitante = eventos_visitante[eventos_visitante['type_name'] == 'Foul Committed'].shape[0]

    # Calcular tarjetas amarillas y rojas
    amarillas_local = eventos_local[(eventos_local['type_name'] == 'Bad Behaviour') & 
                                    (eventos_local['foul_committed_card_name'] == 'Yellow Card')].shape[0]
    rojas_local = eventos_local[(eventos_local['type_name'] == 'Bad Behaviour') & 
                                (eventos_local['foul_committed_card_name'] == 'Red Card')].shape[0]

    amarillas_visitante = eventos_visitante[(eventos_visitante['type_name'] == 'Bad Behaviour') & 
                                            (eventos_visitante['foul_committed_card_name'] == 'Yellow Card')].shape[0]
    rojas_visitante = eventos_visitante[(eventos_visitante['type_name'] == 'Bad Behaviour') & 
                                        (eventos_visitante['foul_committed_card_name'] == 'Red Card')].shape[0]

    # Calcular fueras de juego
    fueras_juego_local = eventos_local[eventos_local['type_name'] == 'Offside'].shape[0]
    fueras_juego_visitante = eventos_visitante[eventos_visitante['type_name'] == 'Offside'].shape[0]

    # Crear resumen
    resumen = {
        equipo_local: {
            'Goles': f"{goles_local_normales}({goles_local_120})" if goles_local_120 > 0 else f"{goles_local_normales}",
            'Tiros': int(tiros_local),
            'Tiros a puerta': int(tiros_puerta_local),
            'Posesión (%)': round(porcentaje_posesion_local, 2),
            'Pases': int(pases_local),
            'Faltas': int(faltas_local),
            'Tarjetas amarillas': int(amarillas_local),
            'Tarjetas rojas': int(rojas_local),
            'Fueras de juego': int(fueras_juego_local)
        },
        equipo_visitante: {
            'Goles': f"{goles_visitante_normales}({goles_visitante_120})" if goles_visitante_120 > 0 else f"{goles_visitante_normales}",
            'Tiros': int(tiros_visitante),
            'Tiros a puerta': int(tiros_puerta_visitante),
            'Posesión (%)': round(porcentaje_posesion_visitante, 2),
            'Pases': int(pases_visitante),
            'Faltas': int(faltas_visitante),
            'Tarjetas amarillas': int(amarillas_visitante),
            'Tarjetas rojas': int(rojas_visitante),
            'Fueras de juego': int(fueras_juego_visitante)
        }
    }
    return resumen

# Función para analizar pases
def analizar_pases(df1, equipo):
    # Filtrar solo los pases del equipo seleccionado
    df1_pases = df1[df1['type_name'] == 'Pass']
    pases_equipo = df1_pases[df1_pases["team_name"] == equipo]

    # Contar los pases totales por jugador
    pases_totales = pases_equipo.groupby(['player_name'])['player_name'].count().to_frame()

    # Filtrar pases completos e incompletos
    completos = pases_equipo[pases_equipo['outcome_name'].isnull()]  # Pases completos no tienen outcome
    incompletos = pases_equipo[pases_equipo['outcome_name'].notnull()]  # Pases incompletos tienen outcome

    # Agregar la cuenta de los pases completos e incompletos
    pases_totales['completos'] = completos.groupby(['player_name'])['player_name'].count()
    pases_totales['incompletos'] = incompletos.groupby(['player_name'])['player_name'].count()

    # Rellenar NaN con 0 y renombrar columnas
    pases_totales = pases_totales.fillna(0)
    pases_totales = pases_totales.rename(columns={'player_name': 'pases totales'})

    # Restablecer el índice
    pases_totales = pases_totales.reset_index()

    # Calcular el porcentaje de pases completos
    pases_totales['Porcentaje %'] = pases_totales['completos'] / pases_totales['pases totales'] * 100

    # Filtrar jugadores con al menos 10 pases
    pases_totales = pases_totales[pases_totales['pases totales'] >= 10]

    # Ordenar y seleccionar el Top 10
    pases_totales_sorted = pases_totales.sort_values(by="Porcentaje %", ascending=False).head(10)

    return pases_totales_sorted

def plot_xg_acumulado(df, equipo_local, equipo_visitante, incluir_penales=True):
    """
    Grafica el xG acumulado para cada equipo a lo largo de un partido (90 o 120 minutos), excluyendo los eventos del período 5 (Penalty Shootout).
    La leyenda muestra el valor concreto del xG acumulado al final del partido.

    Args:
        df (pd.DataFrame): DataFrame con datos del partido, incluyendo timestamps, valores de xG y nombres de equipos.
        equipo_local (str): Nombre del equipo local.
        equipo_visitante (str): Nombre del equipo visitante.
        incluir_penales (bool): Si es True, incluye penales en el cálculo del xG. Si es False, los excluye.

    Returns:
        plt.Figure: Un gráfico de Seaborn con el xG acumulado.
    """
    # Excluir eventos del período 5 (Penalty Shootout)
    df = df[df['period'] != 5]

    # Excluir penales si no se deben incluir
    if not incluir_penales:
        df = df[df['shot_type'] != 'Penalty']

    # Ajustar el rango de minutos hasta 120 si hay prórroga
    max_minuto = min(max(df['minute'].max(), 90), 120)
    
    # Calcular xG acumulado para ambos equipos
    def calcular_xg(datos, equipo):
        xg = datos[datos['team'] == equipo].groupby('minute')['shot_statsbomb_xg'].sum().cumsum()
        # Asegurar que todos los minutos estén presentes (1 a max_minuto)
        xg = xg.reindex(range(1, max_minuto + 1), method='ffill').fillna(0)
        return xg

    xg_local = calcular_xg(df, equipo_local)
    xg_visitante = calcular_xg(df, equipo_visitante)

    # Obtener el valor final del xG acumulado
    xg_final_local = xg_local.iloc[-1]  # Último valor de xG acumulado para el equipo local
    xg_final_visitante = xg_visitante.iloc[-1]  # Último valor de xG acumulado para el equipo visitante

    # Filtrar goles, excluyendo cualquier penalti (incluidos los del minuto 120)
    goles = df[(df['shot_outcome'] == 'Goal') & (df['shot_type'] != 'Penalty')]
    goles_local = goles[goles['team'] == equipo_local]['minute'].tolist()
    goles_visitante = goles[goles['team'] == equipo_visitante]['minute'].tolist()

    # Crear el gráfico
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=xg_local.index, y=xg_local.values, label=f"{equipo_local} (xG: {xg_final_local:.2f})", color='red')
    sns.lineplot(x=xg_visitante.index, y=xg_visitante.values, label=f"{equipo_visitante} (xG: {xg_final_visitante:.2f})", color='blue')

    # Marcar los goles en el gráfico
    plt.scatter(goles_local, [xg_local.get(minuto, 0) for minuto in goles_local], color='red', s=100, label=f"Goles {equipo_local}", marker='o')
    plt.scatter(goles_visitante, [xg_visitante.get(minuto, 0) for minuto in goles_visitante], color='blue', s=100, label=f"Goles {equipo_visitante}", marker='o')

    # Personalizar el gráfico
    plt.title(f"xG Acumulado para {equipo_local} vs {equipo_visitante}")
    plt.xlabel("Minuto")
    plt.ylabel("xG Acumulado")
    plt.legend(title="Equipos", loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(1, max_minuto)  # Limitar el eje X hasta donde haya datos
    plt.ylim(0, max(max(xg_local), max(xg_visitante)))  # Ajustar el eje Y al máximo xG

    return plt.gcf()  # Devolver la figura de Matplotlib

def mapa_tiros_por_equipo(df, equipo, equipo_comparado):
    """
    Genera un mapa de tiros para un equipo específico, normalizando el colorbar según el rango de xG conjunto con otro equipo.

    Args:
        df (pd.DataFrame): DataFrame con los datos del partido.
        equipo (str): Nombre del equipo a visualizar.
        equipo_comparado (str): Nombre del equipo usado para normalizar el rango de xG.
    """
    df_shots = df[df['type'] == 'Shot']
    df_equipo = df_shots[df_shots['team'] == equipo].copy()
    df_comparado = df_shots[df_shots['team'] == equipo_comparado].copy()

    # Excluir penaltis del período 5 (tanda de penaltis)
    df_equipo = df_equipo[~((df_equipo["shot_type"] == "Penalty") & (df_equipo["period"] == 5))]
    df_comparado = df_comparado[~((df_comparado["shot_type"] == "Penalty") & (df_comparado["period"] == 5))]

    # Mapear partes del cuerpo
    body_part_mapping = {
        "Right Foot": "Pie", "Left Foot": "Pie", "Head": "Cabeza", "Other": "Otro"
    }
    df_equipo["shot_body_part_ESP.name"] = df_equipo["shot_body_part"].map(body_part_mapping)

    # Formas para cada parte del cuerpo
    body_shapes = {"Cabeza": "o", "Pie": "s", "Otro": "D"}

    # Crear la cancha
    fig, ax = plt.subplots(figsize=(7, 15))
    pitch = VerticalPitch(pitch_type='statsbomb', line_color='black', pitch_color="white", goal_type='box', half=True)
    pitch.draw(ax=ax)

    # Leyenda
    legend_handles = {}

    # Coordenadas
    df_equipo[['location_x', 'location_y']] = pd.DataFrame(df_equipo['location'].tolist(), index=df_equipo.index)

    # Normalización del xG con ambos equipos
    min_xg = min(df_equipo["shot_statsbomb_xg"].min(), df_comparado["shot_statsbomb_xg"].min())
    max_xg = max(df_equipo["shot_statsbomb_xg"].max(), df_comparado["shot_statsbomb_xg"].max())

    # Estadísticas generales
    total_shots = len(df_equipo)
    total_goals = len(df_equipo[df_equipo["shot_outcome"] == "Goal"])
    total_xg = df_equipo["shot_statsbomb_xg"].sum()

    # Dibujar tiros
    for body_part, marker in body_shapes.items():
        subset = df_equipo[df_equipo["shot_body_part_ESP.name"] == body_part]
        goals = subset[subset["shot_outcome"] == "Goal"]
        misses = subset[subset["shot_outcome"] != "Goal"]

        scatter_goals = pitch.scatter(goals["location_x"], goals["location_y"], ax=ax,
                                      c=goals["shot_statsbomb_xg"], cmap="viridis", vmin=min_xg, vmax=max_xg,
                                      edgecolors="black", linewidth=2.5, s=350, marker=marker, alpha=0.9)

        scatter_misses = pitch.scatter(misses["location_x"], misses["location_y"], ax=ax,
                                       c=misses["shot_statsbomb_xg"], cmap="viridis", vmin=min_xg, vmax=max_xg,
                                       edgecolors="black", linewidth=1, s=250, marker=marker, alpha=0.8)

        if body_part not in legend_handles:
            legend_handles[body_part] = mlines.Line2D([0], [0], marker=marker, color='w',
                                                      markerfacecolor='black', markersize=10,
                                                      label=f"{body_part} ({len(subset)} tiros, {len(goals)} goles)",
                                                      linestyle='None')

    # Colorbar
    cbar = plt.colorbar(scatter_misses, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Valor de xG", fontsize=12)

    # Título
    ax.set_title(f"Mapa de Tiros - {equipo}\n{total_shots} tiros, {total_goals} goles, xG: {total_xg:.2f}",
                 fontsize=14, color="black", pad=10)

    # Leyenda
    ax.legend(handles=list(legend_handles.values()), title="Parte del cuerpo", loc="lower right",
              frameon=True, fancybox=True, shadow=True, fontsize=8)

    return fig


import matplotlib
matplotlib.use('Agg')  # Cambiar el backend a 'Agg'
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Procesar y visualizar la red de pases

import pandas as pd
import numpy as np

def procesar_datos_red_pases(df, equipo):
    """
    Filtra y procesa los datos de los pases para la visualización de la red de pases.
    """
    # Filtrar los titulares (jugadores en los primeros 45 minutos)
    titulares = df[df['minute'] < 45]['player_name'].unique()
    df = df[df['player_name'].isin(titulares) & df['pass_recipient_name'].isin(titulares)]

    # Filtrar los pases del equipo (que no sean saques de banda y que no hayan salido)
    team_passes = (df['type_name'] == 'Pass') & (df['team_name'] == equipo) & (df['outcome_name'].isnull()) & (df['sub_type_name'] != "Throw-in")
    team_pass = df.loc[team_passes, ['x', 'y', 'end_x', 'end_y', "player_name", "pass_recipient_name"]]

    # Verificar si hay pases registrados
    if team_pass.empty:
        print("Advertencia: No hay pases registrados para el equipo en este período.")
        return pd.DataFrame(columns=["player_name", "x", "y", "no", "marker_size"]), pd.DataFrame(columns=["pair_key", "pass_count"])

    # DataFrame para posiciones promedio
    scatter_df = pd.DataFrame()

    for i, name in enumerate(team_pass["player_name"].unique()):
        passx = team_pass.loc[team_pass["player_name"] == name]["x"].to_numpy()
        recx = team_pass.loc[team_pass["pass_recipient_name"] == name]["end_x"].to_numpy()
        passy = team_pass.loc[team_pass["player_name"] == name]["y"].to_numpy()
        recy = team_pass.loc[team_pass["pass_recipient_name"] == name]["end_y"].to_numpy()

        scatter_df.at[i, "player_name"] = name
        scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
        scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))
        scatter_df.at[i, "no"] = team_pass.loc[team_pass["player_name"] == name].shape[0]

    # Verificar si la columna "no" contiene valores antes de calcular "marker_size"
    if not scatter_df.empty and "no" in scatter_df.columns and scatter_df["no"].max() > 0:
        scatter_df['marker_size'] = (scatter_df["no"] / scatter_df["no"].max() * 1000)
    else:
        scatter_df['marker_size'] = 100  # Asignar un valor por defecto si no hay datos válidos

    # Contar los pases entre jugadores
    team_pass["pair_key"] = team_pass.apply(lambda x: "_".join(sorted([x["player_name"], x["pass_recipient_name"]])), axis=1)
    lines_df = team_pass.groupby(["pair_key"])['x'].count().reset_index()
    lines_df.rename(columns={'x': 'pass_count'}, inplace=True)

    # Filtrar solo pares con más de 2 pases
    lines_df = lines_df[lines_df['pass_count'] > 2]

    return scatter_df, lines_df


def plot_pases_por_equipo(df, teamname):
    filtro_pases = (df['type_name'] == 'Pass') & (df['team_name'] == teamname) & (df['outcome_name'].isnull()) & (df['sub_type_name'] != "Throw-in")
    df_pases = df.loc[filtro_pases, ['x', 'y', 'end_x', 'end_y', "player_name", "pass_recipient_name"]]

    if df_pases.empty:
        fig, ax = plt.subplots(figsize=(20, 10), dpi=200)  # Mismo tamaño que grafico_red_pases
        ax.text(0.5, 0.5, f"No se encontraron pases para {teamname}.", 
                fontsize=10, ha='center', va='center')
        ax.axis('off')
        plt.close(fig)
        return fig

    df_pases['x'] = df_pases['x'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    df_pases['y'] = df_pases['y'].apply(lambda x: x[1] if isinstance(x, (list, np.ndarray)) else x)
    df_pases['end_x'] = df_pases['end_x'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    df_pases['end_y'] = df_pases['end_y'].apply(lambda x: x[1] if isinstance(x, (list, np.ndarray)) else x)

    pases_ultimo_tercio = df_pases[(df_pases['x'] > 80) & (df_pases['end_x'] > 80)]
    pases_hacia_ultimo_tercio = df_pases[(df_pases['x'] < 80) & (df_pases['end_x'] > 80)]

    # Crear el campo de fútbol con el mismo tamaño que grafico_red_pases
    
    pitch = Pitch(line_color='black', pitch_color='white')
    fig, ax = pitch.draw(figsize=(20, 10))  # Tamaño grande

    # Dibujar los pases
    pitch.arrows(df_pases.x, df_pases.y, df_pases.end_x, df_pases.end_y, color="blue", ax=ax, width=1, alpha=0.5, label='Pases')
    pitch.arrows(pases_ultimo_tercio.x, pases_ultimo_tercio.y, pases_ultimo_tercio.end_x, pases_ultimo_tercio.end_y, color="red", ax=ax, width=1.5, alpha=0.8, label='Pases en último tercio')
    pitch.arrows(pases_hacia_ultimo_tercio.x, pases_hacia_ultimo_tercio.y, pases_hacia_ultimo_tercio.end_x, pases_hacia_ultimo_tercio.end_y, color="orange", ax=ax, width=1.5, alpha=0.8, label='Pases hacia último tercio')

    # Mover la leyenda fuera del gráfico
    ax.legend(facecolor='white', edgecolor='black', fontsize=8, loc='upper left')

    # Eliminar márgenes innecesarios (igual que en grafico_red_pases)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Sin márgenes

    plt.close(fig)
    return fig

def plot_pases_por_jugador(df, jugador):
    filtro_pases = (df['type_name'] == 'Pass') & (df['player_name'] == jugador)  & (df['sub_type_name'] != "Throw-in")
    df_pases = df.loc[filtro_pases, ['x', 'y', 'end_x', 'end_y', "player_name", "pass_recipient_name"]]

    if df_pases.empty:
        fig, ax = plt.subplots(figsize=(20, 10), dpi=200)  # Mismo tamaño que grafico_red_pases
        ax.text(0.5, 0.5, f"No se encontraron pases para {jugador}.", 
                fontsize=10, ha='center', va='center')
        ax.axis('off')
        plt.close(fig)
        return fig

    df_pases['x'] = df_pases['x'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    df_pases['y'] = df_pases['y'].apply(lambda x: x[1] if isinstance(x, (list, np.ndarray)) else x)
    df_pases['end_x'] = df_pases['end_x'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    df_pases['end_y'] = df_pases['end_y'].apply(lambda x: x[1] if isinstance(x, (list, np.ndarray)) else x)

    pases_ultimo_tercio = df_pases[(df_pases['x'] > 80) & (df_pases['end_x'] > 80)]
    pases_hacia_ultimo_tercio = df_pases[(df_pases['x'] < 80) & (df_pases['end_x'] > 80)]

    # Crear el campo de fútbol con el mismo tamaño que grafico_red_pases
    
    pitch = Pitch(line_color='black', pitch_color='white')
    fig, ax = pitch.draw(figsize=(20, 10))  # Tamaño grande

    # Dibujar los pases
    pitch.arrows(df_pases.x, df_pases.y, df_pases.end_x, df_pases.end_y, color="blue", ax=ax, width=1, alpha=0.5, label='Pases')
    pitch.arrows(pases_ultimo_tercio.x, pases_ultimo_tercio.y, pases_ultimo_tercio.end_x, pases_ultimo_tercio.end_y, color="red", ax=ax, width=1.5, alpha=0.8, label='Pases en último tercio')
    pitch.arrows(pases_hacia_ultimo_tercio.x, pases_hacia_ultimo_tercio.y, pases_hacia_ultimo_tercio.end_x, pases_hacia_ultimo_tercio.end_y, color="orange", ax=ax, width=1.5, alpha=0.8, label='Pases hacia último tercio')

    # Mover la leyenda fuera del gráfico
    ax.legend(facecolor='white', edgecolor='black', fontsize=8, loc='upper left')

    # Eliminar márgenes innecesarios (igual que en grafico_red_pases)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Sin márgenes

    plt.close(fig)
    return fig

def graficar_heatmap(df, jugador=""):
    """
    Función para graficar el mapa de calor de los pases de un jugador de un equipo.

    :param df: DataFrame que contiene los datos de los eventos del partido.
    :param jugador: Nombre del jugador cuyo mapa de calor se quiere graficar (obligatorio).
    """
    # Filtrar los pases del equipo que no sean saques de banda
    filtro_pases = (df['type_name'] == 'Pass') & (df['player_name'] == jugador) & (df['outcome_name'].isnull()) & (df['sub_type_name'] != "Throw-in")
    df_pases = df.loc[filtro_pases, ['x', 'y', 'end_x', 'end_y', 'player_name']].copy()

    # Eliminar valores NaN en coordenadas
    df_pases = df_pases.dropna(subset=['x', 'y'])

    # Verificar si el jugador tiene al menos 2 pases
    if len(df_pases) < 2:
        print(f"El jugador {jugador} no tiene suficientes pases para generar el mapa de calor.")
        return

    # Configuración del campo de juego
    pitch = Pitch(line_color='black', pitch_color='white')

    # Crear la figura
    fig, ax = pitch.draw(figsize=(20, 10))  # Usar directamente el objeto Axes devuelto

    # Crear el mapa de calor para este jugador
    try:
        pitch.kdeplot(
            x=df_pases['x'],
            y=df_pases['y'],
            fill=True,
            alpha=0.5,
            n_levels=10,
            cmap='Reds',
            ax=ax,  # Usar directamente el objeto Axes
            bw_adjust=1.5,  # Ajusta el suavizado para evitar errores con pocos datos
            levels=np.linspace(0, 1, 11)  # Niveles explícitos de 0 a 1
        )

        # Eliminar márgenes innecesarios (igual que en grafico_red_pases)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Sin márgenes
    
    except ValueError:
        print(f"No se pudo generar el heatmap para {jugador} debido a la falta de datos o distribución de los mismos.")
        return  # Si hay un error, terminar la función

    return fig

def plot_conducciones_jugador(df, player):
    """Grafica las conducciones de un jugador específico en un campo de fútbol con mplsoccer."""

    # Filtrar conducciones del equipo y jugador específico
    df_carries = df[(df['type'] == 'Carry') & (df['player'] == player)].copy()

    # Filtrar conducciones progresivas (posición final > posición inicial)
    #df_carries = df_carries[df_carries['carry_end_location'].apply(lambda x: x[0]) > df_carries['location'].apply(lambda x: x[0])]
    #df_carries = df_carries[abs(df_carries['location'].apply(lambda x: x[0]) - df_carries['carry_end_location'].apply(lambda x: x[0])) > 2]

    # Extraer coordenadas
    df_carries['start_x'] = df_carries['location'].apply(lambda loc: loc[0])
    df_carries['start_y'] = df_carries['location'].apply(lambda loc: loc[1])
    df_carries['end_x'] = df_carries['carry_end_location'].apply(lambda loc: loc[0])
    df_carries['end_y'] = df_carries['carry_end_location'].apply(lambda loc: loc[1])

    # Si no hay conducciones, imprimir mensaje y salir
    if df_carries.empty:
        print(f"No se encontraron conducciones para {player}.")
        return df_carries[['player', 'start_x', 'start_y', 'end_x', 'end_y']]

    # Crear el campo de fútbol con mplsoccer
    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black', linewidth=1.5)
    fig, ax = pitch.draw(figsize=(12, 8))

    # Modificar el borde del campo
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='black', facecolor='none'))

    # Dibujar las conducciones con un color más suave y grosor de flechas ajustado
    for _, row in df_carries.iterrows():
        pitch.arrows(row['start_x'], row['start_y'], row['end_x'], row['end_y'],
                     ax=ax, color='#FF5733', width=1.5, headwidth=2.4, headlength=2.5, alpha=0.8)


    # Ajustar márgenes y mostrar gráfico
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    

    return fig


def crear_mapa_tiros_jugador(df, jugador, equipo=""):
    df_shots = df[df['type'] == 'Shot']
    df_jugador = df_shots[df_shots['player'] == jugador].copy()
    df_jugador = df_jugador[~((df_jugador["shot_type"] == "Penalty") & (df_jugador["period"] == 5))]

    if df_jugador.empty:
        fig, ax = plt.subplots(figsize=(7, 2))
        fig.text(0.5, 0.5, 
                 f"{jugador}\nNo realizó tiros en este partido.", 
                 ha='center', va='center', fontsize=14, color='black')
        plt.axis('off')
        plt.close(fig)
        return fig

    # Rango global de xG
    vmin = df_jugador["shot_statsbomb_xg"].min()
    vmax = df_jugador["shot_statsbomb_xg"].max()

    pitch = VerticalPitch(pitch_type='statsbomb', line_color='black', pitch_color="white", goal_type='box', half=True)
    fig, ax = plt.subplots(figsize=(7, 15))
    pitch.draw(ax=ax)

    df_jugador[['location_x', 'location_y']] = pd.DataFrame(df_jugador['location'].tolist(), index=df_jugador.index)

    total_shots = len(df_jugador)
    total_goals = len(df_jugador[df_jugador["shot_outcome"] == "Goal"])
    total_xg = df_jugador["shot_statsbomb_xg"].sum().round(2)

    body_part_mapping = {"Right Foot": "Pie", "Left Foot": "Pie", "Head": "Cabeza", "Other": "Otro"}
    df_jugador["shot_body_part_ESP.name"] = df_jugador["shot_body_part"].map(body_part_mapping)
    body_shapes = {"Cabeza": "o", "Pie": "s", "Otro": "D"}

    legend_handles = {}
    scatter_referencia = None  # Para el colorbar

    for body_part, marker in body_shapes.items():
        subset = df_jugador[df_jugador["shot_body_part_ESP.name"] == body_part]
        if subset.empty:
            continue

        goals = subset[subset["shot_outcome"] == "Goal"]
        misses = subset[subset["shot_outcome"] != "Goal"]

        if not goals.empty:
            pitch.scatter(goals["location_x"], goals["location_y"], ax=ax,
                          c=goals["shot_statsbomb_xg"], cmap="viridis",
                          vmin=vmin, vmax=vmax,
                          edgecolors="black", linewidth=2.5, s=350,
                          marker=marker, alpha=0.9)

        if not misses.empty:
            scatter_referencia = pitch.scatter(misses["location_x"], misses["location_y"], ax=ax,
                                               c=misses["shot_statsbomb_xg"], cmap="viridis",
                                               vmin=vmin, vmax=vmax,
                                               edgecolors="black", linewidth=1, s=250,
                                               marker=marker, alpha=0.8)

        legend_handles[body_part] = mlines.Line2D([0], [0], marker=marker, color='w',
                                                  markerfacecolor='black', markersize=10,
                                                  label=f"{body_part} ({len(subset)} tiros, {len(goals)} goles)",
                                                  linestyle='None')

    # Colorbar
    if scatter_referencia is not None:
        cbar = plt.colorbar(scatter_referencia, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Valor de xG", fontsize=10)

    # Título con salto de línea y centrado
    titulo = f"{jugador}"
    if equipo:
        titulo += f" ({equipo})"
    subtitulo = f"Total: {total_shots} tiros | {total_goals} goles | xG: {total_xg}"

    fig.suptitle(f"{titulo}\n{subtitulo}", fontsize=15, color="black", y=0.94, ha='center')

    # Leyenda
    if legend_handles:
        ax.legend(handles=list(legend_handles.values()),
                  title="Parte del cuerpo", loc="lower right",
                  frameon=True, fancybox=True, shadow=True, fontsize=8)

    return fig


def plot_shot_freeze_frame(shot_id, df_freeze, df_event, df_lineup):
    try:
        # Filtramos los datos
        df_freeze_frame = df_freeze[df_freeze.id == shot_id].copy()
        df_shot_event = df_event[df_event.id == shot_id].dropna(axis=1, how='all').copy()
        
        # Verificamos que tenemos datos
        if df_freeze_frame.empty or df_shot_event.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No hay datos disponibles para este gol", 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig

        # Añadir el número de camiseta
        df_freeze_frame = df_freeze_frame.merge(df_lineup, how='left', on='player_id')

        # Nombres de los equipos
        team1 = df_shot_event.team_name.iloc[0]
        team2 = list(set(df_event.team_name.unique()) - {team1})[0]

        # Subconjuntos de jugadores
        df_team1 = df_freeze_frame[df_freeze_frame.team_name == team1]
        df_team2_goal = df_freeze_frame[(df_freeze_frame.team_name == team2) & 
                                      (df_freeze_frame.position_name == 'Goalkeeper')]
        df_team2_other = df_freeze_frame[(df_freeze_frame.team_name == team2) & 
                                       (df_freeze_frame.position_name != 'Goalkeeper')]

        # Configuración del campo con más espacio para el título
        pitch = VerticalPitch(half=True, goal_type='box', pad_bottom=-20)
        
        # Crear figura con más altura y ajustar los márgenes
        fig = plt.figure(figsize=(10, 9))  # Aumentamos la altura para el título
        ax = fig.add_subplot(111)
        
        # Ajustar los márgenes para dejar espacio para el título
        plt.subplots_adjust(top=0.85)  # Deja más espacio en la parte superior
        
        pitch.draw(ax=ax)

        # Dibujar los jugadores
        pitch.scatter(df_team1.x, df_team1.y, s=600, c='#727cce', label='Atacante', ax=ax)
        pitch.scatter(df_team2_other.x, df_team2_other.y, s=600, c='#5ba965', label='Defensor', ax=ax)
        pitch.scatter(df_team2_goal.x, df_team2_goal.y, s=600, c='#c15ca5', label='Portero', ax=ax)

        # Dibujar el tiro
        pitch.scatter(df_shot_event.x, df_shot_event.y, marker='football',
                     s=600, ax=ax, label='Tirador', zorder=1.2)
        pitch.lines(df_shot_event.x, df_shot_event.y,
                   df_shot_event.end_x, df_shot_event.end_y, comet=True,
                   label='tiro', color='#cb5a4c', ax=ax)

        # Ángulo hacia el gol
        pitch.goal_angle(df_shot_event.x, df_shot_event.y, ax=ax, alpha=0.2, 
                        zorder=1.1, color='#cb5a4c', goal='right')

        # Números de camiseta
        for i, label in enumerate(df_freeze_frame.jersey_number):
            pitch.annotate(label, (df_freeze_frame.x.iloc[i], df_freeze_frame.y.iloc[i]),
                         va='center', ha='center', color='white', fontsize=15, ax=ax)

        # Leyenda
        legend = ax.legend(loc='center left', labelspacing=1.5)
        for text in legend.get_texts():
            text.set_fontsize(15)
            text.set_va('center')

        # Título con posición ajustada
        xg_value = df_shot_event['shot_statsbomb_xg'].iloc[0] if 'shot_statsbomb_xg' in df_shot_event.columns else 0.0
        fig.suptitle(f'{df_shot_event.player_name.iloc[0]}\n{team1} vs. {team2}\nxG: {xg_value:.2f}', 
                    fontsize=16, y=0.92)  # Ajustamos la posición vertical (y) del título
        
        return fig

    except Exception as e:
        print(f"Error al generar el gráfico: {str(e)}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error al mostrar el gol: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    

def calcular_metricas_avanzadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas avanzadas para jugadores de fútbol y filtra las columnas relevantes.
    
    Args:
        df (pd.DataFrame): DataFrame con estadísticas originales de jugadores.
    
    Returns:
        pd.DataFrame: DataFrame con las métricas calculadas y las columnas filtradas.
    """
    df_filtered = df.copy()
    
    # Métricas ofensivas
    df_filtered['Gls_90'] = df_filtered['Gls'] / df_filtered['90s']
    df_filtered['Ast_90'] = df_filtered['Ast'] / df_filtered['90s']
    df_filtered['G+A'] = (df_filtered['Gls'] + df_filtered['Ast']) 
    df_filtered['G+A_90'] = (df_filtered['Gls'] + df_filtered['Ast']) / df_filtered['90s']
    df_filtered['G-PK_90'] = df_filtered['G-PK'] / df_filtered['90s']
    df_filtered['xG_90'] = df_filtered['xG'] / df_filtered['90s']
    df_filtered['xAG_90'] = df_filtered['xAG'] / df_filtered['90s']
    df_filtered['xG+xAG'] = (df_filtered['xG'] + df_filtered['xAG'])
    df_filtered['xG+xAG_90'] = (df_filtered['xG'] + df_filtered['xAG']) / df_filtered['90s']
    df_filtered['npxG_90'] = df_filtered['npxG'] / df_filtered['90s']
    df_filtered["Att_Act_90"] = df_filtered["Sh/90"] + (df_filtered["KP"] / df_filtered["90s"]) + (df_filtered["Succ"] / df_filtered["90s"])
    df_filtered['Att_Pen_90'] = df_filtered['Att Pen'] / df_filtered['90s']
    # Goles vs expectativas por 90
    df_filtered['G-xG_90'] = df_filtered['G-xG'] / df_filtered['90s']
    df_filtered['np:G-xG_90'] = df_filtered['np:G-xG'] / df_filtered['90s']

    # Disparos por 90
    df_filtered['Sh_90'] = df_filtered['Sh'] / df_filtered['90s']
    df_filtered['SoT_90'] = df_filtered['SoT'] / df_filtered['90s']

    
    # Creación de juego y posesión
    df_filtered['Cmp_90'] = df_filtered['Cmp'] / df_filtered['90s']
    df_filtered['Cmp%'] = (df_filtered['Cmp'] / df_filtered['Att']) * 100
    df_filtered['PrgP_90'] = df_filtered['PrgP'] / df_filtered['90s']
    df_filtered['KP_90'] = df_filtered['KP'] / df_filtered['90s']
    df_filtered['PrgC_90'] = df_filtered['PrgC'] / df_filtered['90s']
    df_filtered['SuccDrib_90'] = df_filtered['Succ'] / df_filtered['90s']
    df_filtered['PPA_90'] = df_filtered['PPA'] / df_filtered['90s']
    df_filtered['CrsPA_90'] = df_filtered['CrsPA'] / df_filtered['90s']
    df_filtered['1/3_90'] = df_filtered['1/3'] / df_filtered['90s']
    df_filtered['Touches_90'] = df_filtered['Touches'] / df_filtered['90s']
    
    # Métricas defensivas
    df_filtered['Tkl+Int_90'] = df_filtered['Tkl+Int'] / df_filtered['90s']
    df_filtered['Blocks_90'] = df_filtered['Blocks_stats_defense'] / df_filtered['90s']
    df_filtered['Clr_90'] = df_filtered['Clr'] / df_filtered['90s']
    df_filtered['Recov_90'] = df_filtered['Recov'] / df_filtered['90s']
    
    # Métricas para porteros
    df_filtered['Saves_90'] = df_filtered['Saves'] / df_filtered['90s']
    
    # Columnas originales utilizadas
    columnas_originales_utilizadas = [
        'Player', 'Nation', 'Squad', 'Comp','Pos', 'Age', 'MP', 'Starts', 'Min',
        'Gls', 'Ast', 'G-PK', 'xG', 'xAG', 'npxG', 'G-xG', 'np:G-xG', 'SCA90', 'GCA', 'Att_Act_90',
        'Cmp', 'Att', 'PrgP', 'KP', 'CrsPA', 'A-xAG','PPA',
        'PrgC', 'Succ',
        'Tkl', 'Int', 'Tkl+Int', 'TklW' ,'Tkl%', 'Att_stats_defense', 'Blocks_stats_defense', 'Clr',
        'Won%', 'Recov',
        '90s',
        'PSxG', 'PSxG+/-', 'GA', 'GA90', 'Saves', 'Save%', 'SoTA', 'Cs', 'Cs%',
        'Sh', 'SoT', 'G/Sh', 'G/SoT', 'Dist', 'npxG/Sh', 'xA', 'SCA', 
        'PassLive', 'PassDead', '1/3', 'PrgR', 'Carries', 'Touches', 
        'Att Pen', 'Rec', 'TotDist', 'PrgDist'
    ]
    
    # Nuevas columnas creadas
    nuevas_columnas = [
        'G-xG_90', 'np:G-xG_90', 'Sh_90', 'SoT_90',
        'Gls_90', 'Ast_90','G+A_90', 'G+A', 'G-PK_90', 'xG_90', 'xAG_90', 'xG+xAG', 'xG+xAG_90', 'npxG_90',
        'Cmp_90', 'Cmp%', 'PrgP_90', 'KP_90', 'PPA_90', 'CrsPA_90',
        'PrgC_90', 'SuccDrib_90', '1/3_90', 'Touches_90',
        'Tkl+Int_90',  'Blocks_90', 'Clr_90','Recov_90',
         'Saves_90'
    ]
    
    # Filtrar columnas manteniendo solo las necesarias
    columnas_a_mantener = [col for col in columnas_originales_utilizadas if col in df_filtered.columns] + nuevas_columnas
    df_filtered = df_filtered[columnas_a_mantener]
    
    return df_filtered

def calcular_percentiles_por_posicion(df, config_dict, columna_posicion="Pos"):
    df_resultado = df.copy()

    # Limpieza inicial de datos
    df_resultado[columna_posicion] = df_resultado[columna_posicion].fillna('Desconocido')
    
    # Crear lista de posiciones normalizada
    df_resultado["pos_list"] = (
        df_resultado[columna_posicion]
        .str.split(",")
        .apply(lambda x: [pos.strip().upper() for pos in x] if isinstance(x, list) else [])
    )

    # Para cada posición en la configuración
    for pos, config in config_dict.items():
        pos_upper = pos.upper()
        columnas = config.get("columns", [])
        lower_is_better = config.get("lower_is_better", [])

        # 1. Filtrar jugadores que tengan esta posición
        mask = df_resultado["pos_list"].apply(lambda positions: pos_upper in positions)
        subset = df_resultado[mask]

        if subset.empty:
            print(f"Advertencia: No hay jugadores en la posición {pos}")
            continue

        # 2. Calcular percentiles para cada columna relevante
        for col in columnas:
            if col not in subset.columns:
                print(f"Advertencia: Columna {col} no existe para posición {pos}")
                continue

            try:
                # Calcular percentil normal
                percentiles = subset[col].rank(pct=True, na_option='keep') * 100

                # Si es métrica donde menos es mejor, invertir el percentil
                if col in lower_is_better:
                    percentiles = 100 - percentiles

                # Rellenar NaN con la mediana
                median_value = percentiles.median()
                percentiles_filled = percentiles.fillna(median_value if not pd.isna(median_value) else 50)

                # Asignar al dataframe
                df_resultado.loc[subset.index, f"percentil_{col}"] = percentiles_filled.astype(int)

            except Exception as e:
                print(f"Error calculando percentil para {col} en {pos}: {str(e)}")
                df_resultado.loc[subset.index, f"percentil_{col}"] = 50  # Valor por defecto

    # Limpieza final
    df_resultado.drop("pos_list", axis=1, inplace=True, errors="ignore")
    return df_resultado

    

def combinar_jugadores_duplicados(df):
    import pandas as pd

    # Detectar columnas de porcentaje y columnas por 90 minutos
    columnas_porcentaje = [col for col in df.columns if col.endswith('%')]
    columnas_por90 = [col for col in df.columns if '90' in col]

    # Columnas no numéricas a conservar del primer registro
    columnas_identidad = ['Player', 'Age', 'Squad', 'Pos', 'Nation', 'Comp']

    # Filtrar primero los jugadores que NO son Vitinha
    df_no_vitinha = df[df['Player'] != 'Vitinha']
    df_vitinha = df[df['Player'] == 'Vitinha']

    # Obtener duplicados por nombre y edad (solo para no Vitinha)
    duplicados = df_no_vitinha[df_no_vitinha.duplicated(subset=['Player', 'Age', 'Nation'], keep=False)]

    # Filtrar los no duplicados (de los no Vitinha)
    no_duplicados = pd.concat([
        df_no_vitinha.drop(duplicados.index),
        df_vitinha  # Añadimos todos los registros de Vitinha como no duplicados
    ])

    # Primer registro de cada duplicado (para conservar identidad)
    primeros = (
        duplicados
        .sort_index()
        .groupby(['Player', 'Age'], sort=False)
        .first()
        .reset_index()[['Player', 'Age'] + columnas_identidad]
    )

    # Función de agregación personalizada
    def agg_func(col):
        if col.name in columnas_porcentaje or col.name in columnas_por90:
            return col.mean()
        elif col.name in columnas_identidad:
            return col.iloc[0]
        else:
            return col.sum()

    # Agrupar duplicados con lógica personalizada (solo no Vitinha)
    agrupados = (
        duplicados
        .groupby(['Player', 'Age'], sort=False)
        .agg(agg_func)
        .reset_index()
    )

    # Combinar datos de identidad y datos agregados
    resultado_duplicados = primeros.drop(columns=['Player', 'Age']).combine_first(agrupados).reset_index(drop=True)

    # Concatenar todo
    df_final = pd.concat([no_duplicados, resultado_duplicados], ignore_index=True)

    return df_final


# Variables por grupo de posición
pos_variables = {
    "GK": ["PSxG", "PSxG+/-", "Save%", "Saves_90", "GA90", "Clr_90", "SoTA", "Min"],
    "DF": ["Tkl+Int_90", "Tkl%", "Blocks_90", "Clr_90", "Won%", "Int", "Tkl", "PrgP_90", "PrgC_90", "Cmp_90", "Min"],
    "MF": [
        "TklW%", "Tkl+Int_90", "PrgC_90", "PrgP_90", "KP_90", "xAG_90", 
        "SuccDrib_90", "Cmp%", "Cmp_90", "SCA90", "GCA", "G+A_90", "Min"
    ],
    "FW": [
        "Gls_90", "xG_90", "G-xG", "SCA90", "PPA_90", "KP_90", "Ast_90",
        "G+A_90", "xG+xAG_90", "SuccDrib_90", "Won%", "Att_Act_90", "Min"
    ]
}


#Ponderaciones reales ajustadas por posición
pos_weights = {
    "GK": {
        "PSxG": 0.177,
        "PSxG+/-": 0.221,
        "Save%": 0.177,
        "Saves_90": 0.133,
        "GA90": 0.044,
        "Clr_90": 0.088,
        "SoTA": 0.133,
        "Min": 0.027
    },
    "DF": {
        "Tkl+Int_90": 0.194,
        "Tkl%": 0.146,
        "Blocks_90": 0.097,
        "Clr_90": 0.097,
        "Won%": 0.097,
        "Int": 0.097,
        "Tkl": 0.097,
        "PrgP_90": 0.049,
        "PrgC_90": 0.049,
        "Cmp_90": 0.049,
        "Min": 0.029
    },
    "MF": {
        "TklW%": 0.078,
        "Tkl+Int_90": 0.078,
        "PrgC_90": 0.097,
        "PrgP_90": 0.097,
        "KP_90": 0.117,
        "xAG_90": 0.097,
        "SuccDrib_90": 0.068,
        "Cmp%": 0.078,
        "Cmp_90": 0.078,
        "SCA90": 0.068,
        "GCA": 0.068,
        "G+A_90": 0.049,
        "Min": 0.029
    },
    "FW": {
        "Gls_90": 0.136,
        "xG_90": 0.080,
        "SCA90": 0.117,
        "PPA_90": 0.110,
        "KP_90": 0.115,
        "Ast_90": 0.136,
        "xAG_90": 0.075,
        "SuccDrib_90": 0.085,
        "Won%": 0.019,
        "Att_Act_90": 0.100,
        "Min": 0.029
    }
}




# # Variables por posición (extraídas de 'estadisticas_por_posicion')
# pos_variables = {
#     "GK": ["PSxG", "PSxG+/-", "Save%", "Saves_90", "GA90", "Clr_90", "SoTA"],
#     "DF": ["Cmp_90", "Cmp%", "PrgP_90", "Tkl+Int_90", "Tkl%", "Won%", "PrgC_90", "xAG_90", "CrsPA_90"],
#     "MF": ["Won%", "Tkl+Int_90", "PrgC_90", "PrgP_90", "KP_90", "xAG_90", "SuccDrib_90", "Cmp%", "Cmp_90"],
#     "FW": ["Won%", "G-PK_90", "xG_90", "PPA_90", "KP_90", "xAG_90", "SuccDrib_90", "Att_Act_90", "PrgC_90"]
# }

# # Ponderaciones (puedes ajustarlas si lo deseas)
# pos_weights = {
#     "GK": {
#         "PSxG": 0.15,
#         "PSxG+/-": 0.25,
#         "Save%": 0.20,
#         "Saves_90": 0.15,
#         "GA90": 0.05,         # lower is better
#         "Clr_90": 0.10,
#         "SoTA": 0.10
#     },
#     "DF": {
#         "Cmp_90": 0.10,
#         "Cmp%": 0.10,
#         "PrgP_90": 0.10,
#         "Tkl+Int_90": 0.20,
#         "Tkl%": 0.10,
#         "Won%": 0.10,
#         "PrgC_90": 0.10,
#         "xAG_90": 0.10,
#         "CrsPA_90": 0.10
#     },
#     "MF": {
#         "Won%": 0.10,
#         "Tkl+Int_90": 0.10,
#         "PrgC_90": 0.05,
#         "PrgP_90": 0.15,
#         "KP_90": 0.15,
#         "xAG_90": 0.15,
#         "SuccDrib_90": 0.1,
#         "Cmp%": 0.10,
#         "Cmp_90": 0.10
#     },
#     "FW": {
#         "Won%": 0.05,
#         "G-PK_90": 0.15,
#         "xG_90": 0.12,
#         "PPA_90": 0.12,
#         "KP_90": 0.12,
#         "xAG_90": 0.10,
#         "SuccDrib_90": 0.08,
#         "Att_Act_90": 0.08,
#         "PrgC_90": 0.08
#     }
# }



# Asignación de grupo de posición
def map_position(pos):
    if pos == "GK":
        return "GK"
    elif pos in ["CB", "LB", "RB", "DF", "LCB", "RCB", "LWB", "RWB"]:
        return "DF"
    elif pos in ["CM", "CDM", "CAM", "LM", "RM", "MF", "LCM", "RCM"]:
        return "MF"
    else:
        return "FW"

# Calcular percentiles por grupo de posición
def calcular_percentiles(df, pos_variables):
    for pos, variables in pos_variables.items():
        df_pos = df[df["pos_group"] == pos]
        for metric in variables:
            if metric in df.columns:
                col_name = f"percentil_{metric}"
                if metric == "GA90":
                    df.loc[df["pos_group"] == pos, col_name] = 100 * (1 - df_pos[metric].rank(pct=True))
                else:
                    df.loc[df["pos_group"] == pos, col_name] = 100 * df_pos[metric].rank(pct=True)
    return df

# Calcular índice ponderado para una fila
def calcular_indice_fila_ponderado(row, pos_variables, pos_weights):
    pos = row["pos_group"]
    if pos not in pos_variables:
        return np.nan

    total = 0
    total_pesos = 0
    for metric in pos_variables[pos]:
        col_percentil = f"percentil_{metric}"
        peso = pos_weights[pos].get(metric, 0)
        if col_percentil in row and pd.notnull(row[col_percentil]):
            total += row[col_percentil] * peso
            total_pesos += peso

    return total / total_pesos if total_pesos else np.nan

# Función principal
def calcular_indice_rendimiento(df, eliminar_percentil_cols=False):
    df = df.copy()
    df['pos_group'] = df['Pos'].apply(map_position)

    df = calcular_percentiles(df, pos_variables)

    df["indice_rendimiento"] = df.apply(
        calcular_indice_fila_ponderado, axis=1,
        args=(pos_variables, pos_weights)
    )

    if eliminar_percentil_cols:
        percentil_cols = [f"percentil_{col}" for cols in pos_variables.values() for col in cols]
        df.drop(columns=[c for c in percentil_cols if c in df.columns], inplace=True)

    return df

