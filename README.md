# Fútbol Analytics Platform: Plataforma Interactiva de Análisis Estadístico y Visión por Computadora

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Shiny](https://img.shields.io/badge/Shiny%20for%20Python-v0.6%2B-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 📋 Descripción del Proyecto

[cite_start]Este repositorio contiene el código fuente de una **plataforma interactiva para el análisis de fútbol**, diseñada como una prueba de concepto orientada a cuerpos técnicos, analistas tácticos y departamentos de *scouting*[cite: 4]. [cite_start]El objetivo principal es integrar la ciencia de datos, el modelado estadístico avanzado y la visión por computadora en una herramienta centralizada y accesible, eliminando la dependencia de software privativo[cite: 4, 7].

[cite_start]La aplicación web ha sido construida íntegramente utilizando **Shiny for Python**, proporcionando una interfaz de usuario reactiva y dinámica para explorar métricas avanzadas y procesar vídeo de manera automatizada[cite: 6, 280].

---

## 🏗️ Arquitectura y Bloques Principales

[cite_start]El proyecto se divide en tres módulos fundamentales independientes y complementarios:

### 1. Análisis de Competiciones Oficiales (Datos de Eventos)
* [cite_start]**Fuente:** StatsBomb Open Data (vía API `statsbombpy`)[cite: 100].
* [cite_start]**Alcance actual:** Análisis de los 51 partidos de la Eurocopa 2024[cite: 100, 139].
* [cite_start]**Funcionalidades:** * Exploración granular de eventos (pases, disparos, recuperaciones) con coordenadas espaciales ($X, Y$)[cite: 107, 129].
  * [cite_start]Visualización avanzada mediante **mplsoccer**, incluyendo mapas de calor de presión, redes de pases y mapas de disparos[cite: 129, 281].
  * [cite_start]*Selector reactivo de goles:* Generación de un "freeze frame" táctico que muestra la posición exacta de todos los jugadores en el instante previo a un disparo convertido en gol[cite: 265, 272, 273].

### 2. Evaluación de Jugadores y Modelos Predictivos (*Scouting*)
* [cite_start]**Fuente:** Dataset de Kaggle basado en web scraping de FBref (temporada 2024-2025), cubriendo más de 2000 jugadores de las cinco grandes ligas europeas[cite: 110, 141].
* **Modelado Predictivo:**
  * [cite_start]**Predicción de Goles:** Algoritmo optimizado basado en **XGBoost** empleando variables de rendimiento histórico[cite: 2, 588].
  * [cite_start]**Predicción de Asistencias:** Modelo basado en **Random Forest**[cite: 2, 588].
* **Funcionalidades de Exploración:**
  * [cite_start]Rankings dinámicos basados en un Índice de Rendimiento personalizado (con filtros mínimos para mitigar sesgos por muestras pequeñas)[cite: 313, 316].
  * [cite_start]Gráficos interactivos de rendimiento: *Radar Plots* y *Pizza Charts* adaptados según la posición específica del futbolista[cite: 302, 317].
  * [cite_start]Buscador avanzado por perfil estadístico y percentiles a través de controles deslizantes (*sliders*) reactivos[cite: 334, 342, 348].

### 3. Visión por Computadora Aplicada al Vídeo
* [cite_start]**Tecnologías:** OpenCV, Ultralytics (YOLOv8) y la librería `supervision` de Roboflow[cite: 57, 171, 173].
* **Canal de Procesamiento (*Pipeline*):**
  1. [cite_start]**Detección de Objetos:** Identificación cuadro a cuadro de jugadores, porteros, árbitros y balón usando YOLOv8 especializado en deportes[cite: 173, 174, 204].
  2. [cite_start]**Clasificación por Equipos:** Extracción de *bounding boxes* de los jugadores y paso por un clasificador secundario basado en color de la equipación para asignar pertenencia a un equipo[cite: 175, 177].
  3. [cite_start]**Seguimiento Multiobjeto:** Algoritmo **ByteTrack** para mantener la identidad (ID persistente) de los elementos a lo largo de la secuencia de vídeo[cite: 180, 204].
  4. [cite_start]**Homografía y Proyección Espacial:** Detección de puntos clave del terreno de juego para proyectar las posiciones tridimensionales a un plano táctico 2D (vista aérea)[cite: 207].
  5. [cite_start]**Generación de Salidas Visuales:** Exportación de tres retransmisiones analíticas: vídeo anotado con ID/equipo, vídeo con **Diagramas de Voronoi dinámicos** (control territorial) y mapa táctico animado de vista zenital[cite: 200, 210, 212, 214].

---

## 🛠️ Tecnologías y Dependencias

El ecosistema de librerías de Python requeridas incluye:

* [cite_start]**Estructura Web:** `shiny`, `htmltools`, `faicons`[cite: 6, 280].
* [cite_start]**Procesamiento de Datos:** `pandas`, `numpy`, `scikit-learn`, `xgboost`[cite: 2, 130].
* [cite_start]**Visualización:** `matplotlib`, `seaborn`, `mplsoccer`[cite: 281, 332].
* [cite_start]**Visión Artificial:** `opencv-python`, `ultralytics`, `supervision`, `inference-gpu`[cite: 171, 202].
* [cite_start]**APIs de Datos:** `statsbombpy`[cite: 100].

---


