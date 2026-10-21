# RESPUESTAS – Workshop RAG SI3003

**Nombre:** Juan David Correa
**Código estudiantil:** 1000123360

---

## Parte A · App RAG empresarial

### A1 · App Gradio funcionando
- Notebook: `02b_rag_app_nvidia_nim_colab.ipynb`
- Evidencia: `evidencias/A1_gradio.png`
- Pregunta de prueba: *¿Cuántos días pueden trabajar remotamente los empleados?*
- Respuesta: "Hasta 3 días por semana, con aprobación del jefe directo."
- Observación: la app respondió con un documento demo y citó la fuente.

### A2 · Corpus propio
- Empresa ficticia: Nimbus Software S.A.S. (desarrollo de software, Medellín)
- Archivo: `data/company_documents.json` (4 documentos: RRHH, Finanzas, TI, Producto)
- Evidencia: `evidencias/A2_corpus.png`
- Índice: 4 documentos, 10 chunks.
- Pregunta: *¿Cuántos días pueden trabajar remotamente los empleados?*
- Respuesta: "Máximo 3 días por semana; 2 días en los primeros 90 días; aprobación del jefe de área y RRHH para más días o bloques de 2 o más días consecutivos."
- Observación: la respuesta coincide con `doc-001`. La primera versión salió cortada al final; no se repitió.
- Incidencia: una consulta devolvió `500 Failed to generate completions` del servicio de NVIDIA; se resolvió reintentando.

---

## Parte B · RAG sobre papers académicos

### B1 · Ground truth (notebook 03)
Índice: 150 chunks (55 de Attention, 95 de RAG). Dimensión de embeddings: 2048.
Incidencia: `rag_chain` devolvió `404 Not Found` en la primera ejecución; la prueba posterior funcionó.

| # | Pregunta | Respuesta del RAG (resumen) | ¿Correcta? | ¿Contexto del paper correcto? |
|---|---|---|---|---|
| Q1 | BLEU del Transformer big en WMT14 EN-DE | 28.4 | [sí/parcial/no] | sí (Attention) |
| Q2 | Combinación de memoria paramétrica y no paramétrica en RAG | BART + índice denso de Wikipedia con DPR, entrenado de extremo a extremo | [sí/parcial/no] | sí (RAG) |
| Q3 | Propósito del factor √dk | Evita que el softmax sature con productos punto grandes | [sí/parcial/no] | sí (Attention) |
| Q4 | Variantes RAG-Sequence y RAG-Token | Un documento por secuencia vs. un documento por token | [sí/parcial/no] | sí (RAG) |
| Q5 | Cabezas y dimensiones del Transformer base | h = 8, d_k = d_v = 64 | [sí/parcial/no] | sí (Attention) |
| Bonus | ¿BART es basado en Transformer? | Sí, BART es un seq2seq Transformer | [sí/parcial/no] | ambos papers aparecieron (4 fragmentos de RAG, 1 de Attention) |

**Observación:** Q2 y Q4 son correctas pero más largas de lo necesario.

### B2 · Experimentos

Configuración base del notebook: chunk_size=900, overlap=120, k=5, score_threshold=0.25. Corpus: 2 papers (150 chunks con 900). Preguntas: Q1 a Q5 del ground truth.

| Experimento | Configuración | Chunks | Aciertos (de 5) | Observación |
|---|---|---|---|---|
| chunk_size | 400 / overlap 52 | 412 | 5 | Muchos más chunks, sin mejora |
| chunk_size | 900 / overlap 120 | 150 | 5 | Configuración del notebook |
| chunk_size | 1500 / overlap 195 | 150 | 5 | Igual que 900 (ver nota) |
| k | 3 | 3 por pregunta | 5 | Ya acierta con pocos chunks |
| k | 5 | 5 por pregunta | 5 | Configuración del notebook |
| k | 8 | 8 por pregunta | 5 | Más ruido, sin mejora |
| score_threshold | 0.15 | 5 por pregunta | 5 | Sin efecto |
| score_threshold | 0.25 | 5 por pregunta | 5 | Sin efecto |
| score_threshold | 0.40 | 5 por pregunta | 5 | Sin efecto |
| score_threshold | 0.80 | 0.4 por pregunta | 1 | Filtra casi todo |

**Nota sobre chunk_size:** 900 y 1500 producen el mismo número de chunks (150). Verifiqué con el splitter directamente. Probablemente la extracción de los PDFs genera segmentos cortos, así que el splitter nunca llega al límite de 900 caracteres.

**Conclusión:**
- El parámetro que más cambió el número de chunks fue `k`, que siempre devuelve exactamente ese número.
- `chunk_size` generó muchos más fragmentos al bajar a 400, pero no cambió los aciertos en ninguna configuración.
- `score_threshold` no tuvo efecto entre 0.15 y 0.40, y a 0.80 eliminó casi toda la recuperación (1/5).
- La configuración del notebook (900, k=5, threshold=0.25) es suficiente para este corpus; no encontré una mejor en estas pruebas.
- Limitación: solo 5 preguntas, así que las diferencias de aciertos son pequeñas o nulas.

### B3 · Preguntas trampa (pendiente)

### B4 · Preguntas en español (pendiente)

### B5 · Quiz propio (pendiente)

---

## Parte R · Reflexión (pendiente)