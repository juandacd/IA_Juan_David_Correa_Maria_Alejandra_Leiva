"""Pipeline de la Parte 1: entrena el HMM y el Naive Bayes y ofrece las consultas de datos y predicción."""
import numpy as np
import pandas as pd

from . import config as cfg
from .datos import cargar_series, armar_tabla, split_temporal, a_categoria
from .modelos import HMMGaussiano, NaiveBayesCategorico


class Sistema:
    """Guarda las series de PM2.5 y los modelos entrenados (uno por estación)."""

    def __init__(self, series=None):
        self.series = series if series is not None else cargar_series()
        self.modelos_hmm = {}      # HMM entrenado con el pasado (antes de 2026)
        self.estados_hmm = {}      # estado oculto de cada día (usa solo datos hasta ese día)
        self.modelos_nb = {}       # Naive Bayes final
        self.datos = {}            # tablas de entrenamiento y prueba
        for est, s in self.series.items():
            self._entrenar_estacion(est, s)

    def _entrenar_estacion(self, est, s):
        obs = np.log1p(s.to_numpy())                                 # log(1+PM2.5): reduce el efecto de los picos
        n_train = int((s.index < cfg.FECHA_CORTE).sum())             # días de entrenamiento
        hmm = HMMGaussiano(n_estados=3).fit(obs[:n_train])           # aprende SOLO del pasado
        self.modelos_hmm[est] = hmm
        self.estados_hmm[est] = pd.Series(hmm.filtrar(obs).argmax(axis=1), index=s.index)

        train, test = split_temporal(armar_tabla(s))
        train, test = train.copy(), test.copy()
        train["estado_hmm"] = self.estados_hmm[est].loc[train.index].values
        test["estado_hmm"] = self.estados_hmm[est].loc[test.index].values
        self.datos[est] = {"train": train, "test": test}
        self.modelos_nb[est] = NaiveBayesCategorico().fit(train[cfg.PISTAS_FINAL], train["y"])

    # ------------------------------------------------------------ consultas
    @property
    def ultima_fecha(self):
        """Último día con dato entre todas las estaciones."""
        return max(s.dropna().index[-1] for s in self.series.values())

    def obtener_pm25(self, estacion, fecha=None):
        """PM2.5 de una estación en una fecha (por defecto, el último día con dato)."""
        if estacion not in self.series:
            return {"error": f"Estación desconocida. Opciones: {list(self.series)}"}
        s = self.series[estacion]
        fecha = pd.Timestamp(fecha) if fecha is not None else s.dropna().index[-1]
        if fecha not in s.index:
            return {"error": f"No hay datos para {fecha.date()} en {estacion}. "
                             f"Rango disponible: {s.index.min().date()} a {s.index.max().date()}."}
        if pd.isna(s.loc[fecha]):
            return {"error": f"{estacion} no tiene dato de PM2.5 el {fecha.date()}."}
        ultimos7 = s.loc[fecha - pd.Timedelta(days=6): fecha].mean()  # promedio de los últimos 7 días
        return {"estacion": estacion, "fecha": str(fecha.date()),
                "pm25_ugm3": round(float(s.loc[fecha]), 1),
                "categoria": cfg.CLASES[int(a_categoria(s.loc[fecha]))],
                "promedio_7_dias": round(float(ultimos7), 1)}

    def predecir_manana(self, estacion, fecha=None):
        """Predice la categoría del día siguiente a 'fecha', con probabilidades. Devuelve un dict (o un error)."""
        if estacion not in self.modelos_nb:
            return {"error": f"Estación desconocida. Opciones: {list(self.modelos_nb)}"}
        s = self.series[estacion]
        fecha = pd.Timestamp(fecha) if fecha is not None else s.dropna().index[-1]
        if fecha not in s.index:
            return {"error": f"No hay datos para {fecha.date()} en {estacion}."}
        ayer = fecha - pd.Timedelta(days=1)
        if pd.isna(s.loc[fecha]) or ayer not in s.index or pd.isna(s.loc[ayer]):
            return {"error": f"Faltan datos de hoy o de ayer en {estacion} para {fecha.date()}."}

        manana = fecha + pd.Timedelta(days=1)
        fila = {"cat_hoy": int(a_categoria(s.loc[fecha])), "cat_ayer": int(a_categoria(s.loc[ayer])),
                "estado_hmm": int(self.estados_hmm[estacion].loc[fecha]),
                "dow_manana": manana.dayofweek, "mes_manana": manana.month - 1}
        X = pd.DataFrame([fila])[cfg.PISTAS_FINAL]                  # mismas pistas del entrenamiento
        p = self.modelos_nb[estacion].predict_proba(X)[0]           # probabilidad de cada categoría
        return {"estacion": estacion, "fecha_base": str(fecha.date()),
                "fecha_prediccion": str(manana.date()), "pm25_hoy": round(float(s.loc[fecha]), 1),
                "categoria_hoy": cfg.CLASES[fila["cat_hoy"]],
                "categoria_prevista": cfg.CLASES[int(p.argmax())],
                "probabilidades": {c: round(float(v), 3) for c, v in zip(cfg.CLASES, p)}}


_SISTEMA = None


def get_sistema():
    """Crea el sistema una sola vez y lo reutiliza (entrenar tarda unos segundos)."""
    global _SISTEMA
    if _SISTEMA is None:
        _SISTEMA = Sistema()
    return _SISTEMA
