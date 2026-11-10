"""Datos: limpieza de los archivos de SISAIRE, categorías de PM2.5, división temporal y métricas."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from . import config as cfg


# ------------------------------------------------------------------ limpieza
def leer_estacion(clave):
    """Lee el .xls de una estación de SISAIRE y devuelve su PM2.5 diario ya limpio."""
    ruta = cfg.DATA_RAW / f"{clave}.xls"
    # header=9: los nombres de columna están en la fila 10 (las anteriores son información del reporte)
    d = pd.read_excel(ruta, engine="xlrd", header=9)
    d = d.rename(columns={"Fecha inicial": "fecha", "PM2.5": "pm25"})
    d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce")        # lo que no se pueda leer queda vacío
    d["pm25"] = pd.to_numeric(d["pm25"], errors="coerce")

    d = (d.dropna(subset=["fecha"])                                 # quitar filas sin fecha (notas del reporte)
           .drop_duplicates("fecha")                                # quitar fechas repetidas
           .set_index("fecha").sort_index())
    d = d.loc[cfg.INICIO:cfg.FIN, ["pm25"]]                         # período del proyecto

    d.loc[d["pm25"] <= 0, "pm25"] = np.nan                          # PM2.5 = 0 exacto es falla del sensor
    d = d.reindex(pd.date_range(cfg.INICIO, cfg.FIN, freq="D"))     # un renglón por día
    d["pm25"] = d["pm25"].ffill(limit=3)                            # huecos de máx. 3 días: último valor (solo pasado)
    d.index.name = "fecha"
    return d


def preparar_datos():
    """Limpia los .xls de todas las estaciones y guarda data/processed/pm25_diario.csv."""
    frames = []
    for clave, nombre in cfg.ESTACIONES.items():
        d = leer_estacion(clave)
        d["estacion"] = nombre
        frames.append(d.reset_index())
    df = pd.concat(frames, ignore_index=True)
    cfg.DATA_PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.CSV_DIARIO, index=False)
    return df


def cargar_series(forzar=False):
    """Devuelve {estación: serie diaria de PM2.5}. Si el CSV limpio no existe (o se fuerza), lo crea."""
    if forzar or not cfg.CSV_DIARIO.exists():
        df = preparar_datos()
    else:
        df = pd.read_csv(cfg.CSV_DIARIO, parse_dates=["fecha"])
    return {est: g.set_index("fecha")["pm25"].sort_index() for est, g in df.groupby("estacion")}


# ------------------------------------------------------------------ categorías y tabla de trabajo
def a_categoria(pm):
    """PM2.5 -> 0 Buena (<=12), 1 Moderada (<=37), 2 Alerta (>37)."""
    pm = np.asarray(pm, dtype=float)
    return np.where(pm <= cfg.UMBRALES[0], 0, np.where(pm <= cfg.UMBRALES[1], 1, 2)).astype(float)


def nombre_categoria(pm):
    """Nombre de la categoría de un valor de PM2.5 ('Sin dato' si falta)."""
    return "Sin dato" if pd.isna(pm) else cfg.CLASES[int(a_categoria(pm))]


def armar_tabla(serie):
    """Una fila por día con: categoría de hoy, de ayer y de mañana (lo que se quiere predecir)."""
    t = pd.DataFrame({"pm25": serie})
    t["cat_hoy"] = np.where(serie.isna(), np.nan, a_categoria(serie))
    t["cat_ayer"] = t["cat_hoy"].shift(1)                           # corre la columna un día hacia adelante
    t["y"] = t["cat_hoy"].shift(-1)                                 # categoría de MAÑANA
    t["dow_manana"] = (t.index + pd.Timedelta(days=1)).dayofweek    # día de la semana de mañana (0 = lunes)
    t["mes_manana"] = (t.index + pd.Timedelta(days=1)).month - 1    # mes de mañana (0 = enero)
    return t.dropna(subset=["cat_hoy", "cat_ayer", "y"])            # descartar filas con huecos


def split_temporal(t):
    """Separa por fecha: el pasado para entrenar y el futuro para probar. Nunca se mezclan."""
    return t[t.index < cfg.FECHA_CORTE], t[t.index >= cfg.FECHA_CORTE]


# ------------------------------------------------------------------ métricas
def metricas(y_real, y_pred):
    """Accuracy, dos versiones de F1 macro y matriz de confusión (filas = real, columnas = predicho)."""
    y_real, y_pred = np.asarray(y_real, int), np.asarray(y_pred, int)
    presentes = np.unique(y_real)                                   # clases que sí aparecen en la prueba
    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "f1_macro_3clases": f1_score(y_real, y_pred, average="macro", labels=[0, 1, 2], zero_division=0),
        "f1_macro_presentes": f1_score(y_real, y_pred, average="macro", labels=presentes, zero_division=0),
        "matriz": confusion_matrix(y_real, y_pred, labels=[0, 1, 2]),
    }


if __name__ == "__main__":                                          # python -m src.datos  -> regenera el CSV limpio
    df = preparar_datos()
    print(f"CSV limpio guardado en {cfg.CSV_DIARIO} ({len(df)} filas)")
    print(df.groupby("estacion")["pm25"].agg(["count", "mean"]).round(1))
