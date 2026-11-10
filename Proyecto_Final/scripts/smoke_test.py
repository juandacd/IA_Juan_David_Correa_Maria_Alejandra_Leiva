"""Prueba rápida: revisa cada pieza del proyecto y dice OK o FALLA con el motivo.

Uso (desde la carpeta principal):
    python scripts/smoke_test.py              # revisa todo, incluida una llamada real a NVIDIA
    python scripts/smoke_test.py --sin-llm    # no llama a NVIDIA (no gasta cuota ni necesita la key)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTADOS = []


def probar(nombre, funcion):
    """Ejecuta una comprobación y guarda si salió bien."""
    try:
        detalle = funcion()
        RESULTADOS.append((nombre, True, detalle or ""))
        print(f"OK     | {nombre}" + (f" -> {detalle}" if detalle else ""))
    except Exception as e:
        RESULTADOS.append((nombre, False, f"{type(e).__name__}: {e}"))
        print(f"FALLA  | {nombre} -> {type(e).__name__}: {str(e)[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-llm", action="store_true", help="no llamar al LLM de NVIDIA")
    args = ap.parse_args()
    from src import config as cfg

    def datos():
        from src.datos import cargar_series
        s = cargar_series()
        assert len(s) == 4, f"se esperaban 4 estaciones y hay {len(s)}"
        return f"{len(s)} estaciones, hasta {max(x.dropna().index[-1] for x in s.values()).date()}"

    def modelos():
        from src.pipeline import get_sistema
        S = get_sistema()
        for est in S.series:
            p = S.predecir_manana(est)
            assert "error" not in p, p
        return "HMM + Naive Bayes entrenados; predicción de mañana OK en las 4 estaciones"

    def herramientas():
        from src.agente import ejecutar_herramienta
        r = ejecutar_herramienta("obtener_pm25", '{"estacion": "Santa Elena"}')
        assert "pm25_ugm3" in r, r
        e = ejecutar_herramienta("obtener_pm25", '{"estacion": "Mordor"}')
        assert "error" in e, "debía devolver un error para una estación inexistente"
        return "obtener_pm25 responde y maneja errores"

    def corpus():
        from src.rag import cargar_fragmentos
        fr = cargar_fragmentos()
        assert len(fr) > 50, f"pocos fragmentos: {len(fr)}"
        return f"{len(fr)} fragmentos de {len({f['archivo'] for f in fr})} documentos"

    def rag():
        from src.rag import buscar_documentos
        r = buscar_documentos("nivel máximo permisible de PM2.5 en 24 horas", k=2)
        assert "fragmentos" in r, r
        return f"mejor resultado: {r['fragmentos'][0]['fuente']} (pág. {r['fragmentos'][0]['pagina']})"

    def llm():
        from src.agente import hay_api_key, preguntar_agente
        assert hay_api_key(), "falta NVIDIA_API_KEY (mira el README)"
        texto, _, usadas = preguntar_agente("¿Cómo está el aire hoy en Santa Elena?", verbose=False)
        assert "obtener_pm25" in usadas, f"el agente no usó la herramienta (usó {usadas}): {texto[:100]}"
        return f"el agente usó {usadas} y respondió"

    def dashboard():
        from src.dashboard import crear_interfaz
        crear_interfaz()
        return "la interfaz se construye"

    probar("1. Datos limpios", datos)
    probar("2. Modelos (HMM + Naive Bayes)", modelos)
    probar("3. Herramientas del agente", herramientas)
    probar("4. Corpus de documentos", corpus)
    probar("5. Búsqueda RAG (descarga el modelo de embeddings la primera vez)", rag)
    probar("6. Dashboard", dashboard)
    if args.sin_llm:
        print("SALTADO| 7. Agente con NVIDIA (--sin-llm)")
    else:
        probar("7. Agente con NVIDIA", llm)

    fallas = [r for r in RESULTADOS if not r[1]]
    print(f"\n{len(RESULTADOS) - len(fallas)}/{len(RESULTADOS)} comprobaciones bien.")
    sys.exit(1 if fallas else 0)


if __name__ == "__main__":
    main()
