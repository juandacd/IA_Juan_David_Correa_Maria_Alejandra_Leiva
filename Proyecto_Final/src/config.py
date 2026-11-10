"""Configuración central del proyecto: rutas, estaciones, fechas, umbrales y modelos de lenguaje.

Todo lo que alguien podría querer cambiar está aquí, en un solo lugar.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- rutas
RAIZ = Path(__file__).resolve().parent.parent          # carpeta principal del repositorio
DATA_RAW = RAIZ / "data" / "raw"                       # archivos .xls originales de SISAIRE
DATA_PROC = RAIZ / "data" / "processed"                # datos ya limpios
CSV_DIARIO = DATA_PROC / "pm25_diario.csv"
CORPUS = RAIZ / "corpus"                               # documentos que lee el agente (RAG)
RESULTS = RAIZ / "results"                             # tablas de resultados y evaluaciones
CACHE = RAIZ / ".cache"                                # índice del RAG (se crea solo)

# Si existe un archivo .env, se leen de ahí las variables (por ejemplo la API key de NVIDIA)
try:
    from dotenv import load_dotenv
    load_dotenv(RAIZ / ".env")
except ImportError:                                    # si no está instalado python-dotenv, no pasa nada
    pass

# ---------------------------------------------------------------- datos
# archivo .xls -> nombre de la estación. Almacén EPM se descartó: su serie empieza en dic-2023.
ESTACIONES = {
    "parque_biblioteca": "Parque Biblioteca",
    "santa_elena": "Santa Elena",
    "trafico_centro": "Tráfico Centro",
    "villahermosa": "Villahermosa",
}
INICIO, FIN = "2019-01-01", "2026-09-30"               # período de los datos
FECHA_VAL = "2025-01-01"                               # entrenar: antes | validar: 2025
FECHA_CORTE = "2026-01-01"                             # probar: desde esta fecha (nunca se usa para elegir nada)

# Coordenadas APROXIMADAS de las estaciones (latitud, longitud), solo para el mapa.
COORDS = {
    "Parque Biblioteca": (6.2790, -75.6330),
    "Santa Elena": (6.2100, -75.4960),
    "Tráfico Centro": (6.2518, -75.5700),
    "Villahermosa": (6.2550, -75.5430),
}

# ---------------------------------------------------------------- categorías
CLASES = ["Buena", "Moderada", "Alerta"]
UMBRALES = (12.0, 37.0)                                # µg/m³: Buena <=12, Moderada <=37, Alerta >37

# Pistas del Naive Bayes elegidas con la validación de 2025 (configuración "F")
PISTAS_FINAL = ["cat_hoy", "cat_ayer", "estado_hmm", "dow_manana", "mes_manana"]

# ---------------------------------------------------------------- agente (LLM)
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
# Lista de modelos: se prueba el primero y, si falla, el siguiente. Se puede cambiar con la variable NVIDIA_MODELOS.
MODELOS_LLM = [m.strip() for m in os.getenv(
    "NVIDIA_MODELOS", "nvidia/nemotron-3.5-lightning-30b-a3b,nvidia/nemotron-3.5-bolt-e4b").split(",") if m.strip()]
T_MAX = 5                                              # máximo de vueltas del loop del agente
MAX_CONSULTAS = 3                                      # máximo de herramientas por pregunta

# ---------------------------------------------------------------- RAG
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS", "intfloat/multilingual-e5-small")
TAM_CHUNK = 160                                        # palabras por fragmento
SOLAPE = 30                                            # palabras repetidas entre fragmentos vecinos

# Nombre legible de cada documento (así se citan las fuentes)
NOMBRES_FUENTE = {
    "res_2254_2017_ocr.txt": "Resolución 2254 de 2017 (MinAmbiente)",
    "res_2254_2017_tablas_pm25.txt": "Resolución 2254 de 2017 - tablas de PM2.5",
    "informe_amva_2021.pdf": "Informe AMVA/SIATA (08-jul-2021)",
    "informe_amva_2026.pdf": "Informe diario AMVA/SIATA (28-jul-2026)",
    "oms_2021_resumen.pdf": "OMS 2021 - Directrices mundiales de calidad del aire (resumen ejecutivo)",
}
