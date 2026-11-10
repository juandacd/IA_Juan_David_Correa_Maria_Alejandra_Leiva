"""RAG: lee los documentos del corpus, los parte en fragmentos, los indexa con FAISS y permite buscarlos.

El modelo de embeddings y FAISS se cargan solo cuando se necesitan; así el resto del proyecto funciona
aunque estas librerías no estén instaladas o no haya internet para descargar el modelo la primera vez.
"""
import hashlib
import json
import re

import numpy as np

from . import config as cfg

_MODELO = None       # modelo de embeddings (se carga la primera vez)
_INDICE = None       # (índice FAISS, lista de fragmentos)


# ------------------------------------------------------------------ lectura y partición de documentos
def limpiar(texto):
    """Arregla saltos de línea y espacios del texto extraído."""
    texto = re.sub(r"-\n(\w)", r"\1", texto)       # une palabras partidas con guion al final de línea
    texto = re.sub(r"\s+", " ", texto)             # varios espacios o saltos de línea -> un espacio
    return texto.strip()


def leer_paginas(ruta):
    """Devuelve una lista de (número de página, texto) de un archivo del corpus."""
    if ruta.suffix == ".pdf":
        from pypdf import PdfReader
        lector = PdfReader(str(ruta))
        paginas = [(i, limpiar(p.extract_text() or "")) for i, p in enumerate(lector.pages, start=1)]
        if sum(len(t) for _, t in paginas[:5]) < 500:      # PDF escaneado: no tiene texto, se omite
            print(f"  (se omite {ruta.name}: es un PDF escaneado sin texto; se usa su versión en .txt)")
            return []
        return paginas
    # Archivos .txt: el OCR marca las páginas con "--- Página N ---"; si no hay marcas, es una sola "página"
    bloques = re.split(r"--- Página (\d+) ---", ruta.read_text(encoding="utf-8"))
    if len(bloques) == 1:
        return [(1, limpiar(bloques[0]))]
    return [(int(bloques[i]), limpiar(bloques[i + 1])) for i in range(1, len(bloques) - 1, 2)]


def partir(texto, tam=cfg.TAM_CHUNK, solape=cfg.SOLAPE):
    """Parte un texto en fragmentos de 'tam' palabras, repitiendo 'solape' palabras entre vecinos."""
    palabras = texto.split()
    paso = tam - solape
    return [" ".join(palabras[i:i + tam]) for i in range(0, max(len(palabras), 1), paso)
            if len(palabras[i:i + tam]) >= 15]               # descartar fragmentos diminutos


def cargar_fragmentos():
    """Lee todos los documentos de corpus/ y devuelve la lista de fragmentos con su fuente y página."""
    fragmentos = []
    for ruta in sorted(cfg.CORPUS.iterdir()):
        if ruta.suffix not in (".pdf", ".txt"):
            continue
        for pagina, texto in leer_paginas(ruta):
            for trozo in partir(texto):
                fragmentos.append({"id": len(fragmentos),
                                   "fuente": cfg.NOMBRES_FUENTE.get(ruta.name, ruta.name),
                                   "archivo": ruta.name, "pagina": pagina, "texto": trozo})
    return fragmentos


# ------------------------------------------------------------------ embeddings e índice
def _modelo_embeddings():
    """Carga el modelo de embeddings (la primera vez se descarga, ~470 MB)."""
    global _MODELO
    if _MODELO is None:
        from sentence_transformers import SentenceTransformer
        print(f"Cargando el modelo de embeddings '{cfg.MODELO_EMBEDDINGS}' "
              "(la primera vez se descarga, puede tardar unos minutos)...")
        _MODELO = SentenceTransformer(cfg.MODELO_EMBEDDINGS)
    return _MODELO


def _firma_corpus():
    """Huella del corpus: si cambia un documento o un parámetro, el índice guardado se reconstruye."""
    partes = [cfg.MODELO_EMBEDDINGS, str(cfg.TAM_CHUNK), str(cfg.SOLAPE)]
    for ruta in sorted(cfg.CORPUS.iterdir()):
        if ruta.suffix in (".pdf", ".txt"):
            partes.append(f"{ruta.name}:{ruta.stat().st_size}")
    return hashlib.sha256("|".join(partes).encode()).hexdigest()[:16]


def cargar_indice():
    """Devuelve (índice FAISS, fragmentos). Lo lee de .cache/ si está al día; si no, lo construye."""
    global _INDICE
    if _INDICE is not None:
        return _INDICE
    import faiss
    cfg.CACHE.mkdir(exist_ok=True)
    f_indice, f_frag, f_firma = (cfg.CACHE / "indice_faiss.bin", cfg.CACHE / "fragmentos.json",
                                 cfg.CACHE / "firma.txt")
    firma = _firma_corpus()
    if f_indice.exists() and f_frag.exists() and f_firma.exists() and f_firma.read_text() == firma:
        _INDICE = (faiss.read_index(str(f_indice)), json.loads(f_frag.read_text(encoding="utf-8")))
        return _INDICE

    print("Construyendo el índice del RAG (solo la primera vez)...")
    fragmentos = cargar_fragmentos()
    # El modelo e5 pide anteponer "passage: " a los documentos y "query: " a las preguntas
    emb = _modelo_embeddings().encode(["passage: " + f["texto"] for f in fragmentos],
                                      normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")                  # FAISS exige float32
    indice = faiss.IndexFlatIP(emb.shape[1])                # producto interno = similitud coseno (vectores normalizados)
    indice.add(emb)
    faiss.write_index(indice, str(f_indice))
    f_frag.write_text(json.dumps(fragmentos, ensure_ascii=False), encoding="utf-8")
    f_firma.write_text(firma)
    print(f"Índice listo: {indice.ntotal} fragmentos.")
    _INDICE = (indice, fragmentos)
    return _INDICE


def buscar(pregunta, k=4):
    """Devuelve los k fragmentos más parecidos a la pregunta, con su fuente, página y similitud."""
    indice, fragmentos = cargar_indice()
    q = _modelo_embeddings().encode(["query: " + pregunta], normalize_embeddings=True)
    puntajes, posiciones = indice.search(np.asarray(q, dtype="float32"), k)
    return [{**fragmentos[i], "similitud": round(float(s), 3)}
            for s, i in zip(puntajes[0], posiciones[0]) if i != -1]


# ------------------------------------------------------------------ herramienta del agente
def buscar_documentos(consulta, k=4):
    """Herramienta del agente: busca en los documentos y devuelve fragmentos con su fuente y página."""
    if not isinstance(consulta, str) or len(consulta.strip()) < 3:
        return {"error": "La consulta está vacía o es muy corta. Escribe una pregunta o palabras clave."}
    k = max(1, min(int(k), 6))                              # entre 1 y 6 fragmentos
    try:
        resultados = buscar(consulta, k=k)
    except Exception as e:                                  # p. ej. falta una librería o no se pudo descargar el modelo
        return {"error": f"No se pudo buscar en los documentos ({type(e).__name__}: {str(e)[:120]})."}
    if not resultados:
        return {"error": "No se encontraron fragmentos relevantes en los documentos."}
    return {"consulta": consulta,
            "fragmentos": [{"fuente": r["fuente"], "pagina": r["pagina"], "similitud": r["similitud"],
                            "texto": r["texto"][:900]} for r in resultados]}   # recortar para no saturar al modelo
