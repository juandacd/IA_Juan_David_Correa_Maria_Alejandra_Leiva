"""Evaluación de la Parte 1: línea base vs Naive Bayes vs Naive Bayes + HMM.

Uso (desde la carpeta principal):   python scripts/evaluar_modelos.py
Guarda los resultados en results/ (validacion_pistas.csv, metricas_modelos.csv, matrices_confusion.txt).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # para poder importar la carpeta src

import numpy as np
import pandas as pd

from src import config as cfg
from src.datos import armar_tabla, cargar_series, metricas, split_temporal
from src.modelos import HMMGaussiano, NaiveBayesCategorico

# Candidatas para la validación con 2025 (la elegida fue la F)
CANDIDATAS = {
    "A: hoy+ayer": ["cat_hoy", "cat_ayer"],
    "B: hoy+ayer+mes": ["cat_hoy", "cat_ayer", "mes_manana"],
    "C: hoy+ayer+dia+mes": ["cat_hoy", "cat_ayer", "dow_manana", "mes_manana"],
    "D: hoy+ayer+HMM": ["cat_hoy", "cat_ayer", "estado_hmm"],
    "E: hoy+ayer+HMM+mes": ["cat_hoy", "cat_ayer", "estado_hmm", "mes_manana"],
    "F: hoy+ayer+HMM+dia+mes": ["cat_hoy", "cat_ayer", "estado_hmm", "dow_manana", "mes_manana"],
    "G: hoy+HMM+mes": ["cat_hoy", "estado_hmm", "mes_manana"],
}


def estados_hmm(serie, fecha_fin_entrenamiento):
    """Entrena el HMM solo con datos ANTERIORES a la fecha dada y devuelve el estado de cada día."""
    obs = np.log1p(serie.to_numpy())
    n = int((serie.index < fecha_fin_entrenamiento).sum())
    hmm = HMMGaussiano(n_estados=3).fit(obs[:n])
    return pd.Series(hmm.filtrar(obs).argmax(axis=1), index=serie.index)


def main():
    cfg.RESULTS.mkdir(exist_ok=True)
    series = cargar_series()

    # ---------------- 1) Validación con 2025: elegir las pistas sin tocar 2026
    filas_val = []
    for est, s in series.items():
        train_full, _ = split_temporal(armar_tabla(s))
        tr = train_full[train_full.index < cfg.FECHA_VAL].copy()
        va = train_full[train_full.index >= cfg.FECHA_VAL].copy()
        est_h = estados_hmm(s, cfg.FECHA_VAL)                       # HMM entrenado antes de 2025
        tr["estado_hmm"], va["estado_hmm"] = est_h.loc[tr.index].values, est_h.loc[va.index].values
        m = metricas(va["y"], va["cat_hoy"])
        filas_val.append({"estacion": est, "config": "Línea base", "accuracy": m["accuracy"],
                          "f1_presentes": m["f1_macro_presentes"]})
        for nombre, pistas in CANDIDATAS.items():
            nb = NaiveBayesCategorico().fit(tr[pistas], tr["y"])
            m = metricas(va["y"], nb.predict(va[pistas]))
            filas_val.append({"estacion": est, "config": nombre, "accuracy": m["accuracy"],
                              "f1_presentes": m["f1_macro_presentes"]})
    val = pd.DataFrame(filas_val)
    ranking = (val.groupby("config")[["accuracy", "f1_presentes"]].mean()
                  .sort_values("f1_presentes", ascending=False).round(3))
    ranking.to_csv(cfg.RESULTS / "validacion_pistas.csv")
    print("=== Validación (2025): promedio de las estaciones ===")
    print(ranking)

    # ---------------- 2) Evaluación final en 2026 (se hace una sola vez)
    filas, matrices = [], []
    for est, s in series.items():
        train, test = split_temporal(armar_tabla(s))
        train, test = train.copy(), test.copy()
        e = estados_hmm(s, cfg.FECHA_CORTE)                         # HMM entrenado antes de 2026
        train["estado_hmm"], test["estado_hmm"] = e.loc[train.index].values, e.loc[test.index].values
        modelos = {
            "Línea base (mañana = hoy)": test["cat_hoy"].astype(int).to_numpy(),
            "Naive Bayes": NaiveBayesCategorico().fit(
                train[["cat_hoy", "cat_ayer", "dow_manana", "mes_manana"]], train["y"]).predict(
                test[["cat_hoy", "cat_ayer", "dow_manana", "mes_manana"]]),
            "Naive Bayes + HMM": NaiveBayesCategorico().fit(
                train[cfg.PISTAS_FINAL], train["y"]).predict(test[cfg.PISTAS_FINAL]),
        }
        for nombre, pred in modelos.items():
            m = metricas(test["y"], pred)
            filas.append({"estacion": est, "modelo": nombre, "accuracy": m["accuracy"],
                          "f1_macro_3clases": m["f1_macro_3clases"], "f1_macro_presentes": m["f1_macro_presentes"],
                          "dias_prueba": len(test), "dias_alerta_prueba": int((test["y"] == 2).sum())})
            matrices.append(f"{est} | {nombre}\n(filas = real, columnas = predicho: {cfg.CLASES})\n{m['matriz']}\n")
    res = pd.DataFrame(filas).round(3)
    res.to_csv(cfg.RESULTS / "metricas_modelos.csv", index=False)
    (cfg.RESULTS / "matrices_confusion.txt").write_text("\n".join(matrices), encoding="utf-8")

    print("\n=== Evaluación en 2026 ===")
    print(res.to_string(index=False))
    print("\nPromedio por modelo:")
    print(res.groupby("modelo", sort=False)[["accuracy", "f1_macro_3clases", "f1_macro_presentes"]].mean().round(3))
    print(f"\nResultados guardados en {cfg.RESULTS}")


if __name__ == "__main__":
    main()
