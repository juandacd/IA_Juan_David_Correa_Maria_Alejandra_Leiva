"""Punto de entrada: python app.py  ->  abre el dashboard en http://127.0.0.1:7860"""
import argparse

from src import config as cfg
from src.agente import hay_api_key


def main():
    ap = argparse.ArgumentParser(description="Dashboard de calidad del aire (PM2.5) en Medellín")
    ap.add_argument("--puerto", type=int, default=7860, help="puerto local (por defecto 7860)")
    ap.add_argument("--share", action="store_true", help="crear además un enlace público temporal (gradio.live)")
    ap.add_argument("--sin-navegador", action="store_true", help="no abrir el navegador automáticamente")
    args = ap.parse_args()

    print("1/3 Preparando datos y entrenando los modelos (unos segundos)...")
    from src.pipeline import get_sistema
    get_sistema()                                            # limpia el CSV si hace falta y entrena HMM + Naive Bayes

    print("2/3 Preparando el buscador de documentos (RAG)...")
    try:
        from src.rag import cargar_indice
        cargar_indice()                                      # la primera vez descarga el modelo de embeddings (~470 MB)
    except Exception as e:                                   # el dashboard abre igual; el RAG se reintentará al preguntar
        print(f"  Aviso: no se pudo preparar el RAG ahora ({type(e).__name__}: {str(e)[:100]}).")

    print("3/3 Abriendo el dashboard...")
    if not hay_api_key():
        print("  Aviso: no hay NVIDIA_API_KEY; el chat estará desactivado (el resto funciona).")
    from src.dashboard import lanzar
    lanzar(share=args.share, puerto=args.puerto, abrir_navegador=not args.sin_navegador)


if __name__ == "__main__":
    main()
