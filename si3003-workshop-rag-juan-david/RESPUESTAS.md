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

### B3 · Preguntas trampa

Preguntas escritas sobre temas que no están en los papers. Objetivo: comprobar si el RAG admite que no sabe o inventa.

| # | Pregunta | Respuesta del RAG (resumen) | Clasificación |
|---|---|---|---|
| T1 | ¿Qué accuracy logró RAG en clasificación de ImageNet? | Los extractos no mencionan ImageNet | Admite que no sabe |
| T2 | ¿Qué BLEU logró RAG-Token en WMT 2020 alemán-inglés? | Los extractos no tienen resultados de WMT 2020 | Admite que no sabe |
| T3 | ¿Quién es el autor del reporte técnico de GPT-4? | Los extractos no contienen información sobre GPT-4 | Admite que no sabe |

**Cambios de prompt:** ninguno. Ninguna respuesta inventó datos, así que no fue necesario modificar el prompt del sistema.

**Observación:** las respuestas mencionan "excerpts", lo que muestra que el modelo solo ve los fragmentos recuperados y no el paper completo. Eso explica por qué un RAG depende de la calidad de la recuperación.

### B4 · Preguntas en español

Traduje Q1, Q3 y Q4 al español y comparé con la versión en inglés usando el índice de 900 (k=5, threshold=0.25).

| Pregunta | Chunks del paper correcto (ES) | Respuesta (ES) vs. EN | Observación |
|---|---|---|---|
| Q1 · BLEU Transformer big (WMT14 EN-DE) | 5/5 de attention_paper.pdf | Igual de buena (28.4) | Recuperación sin pérdida |
| Q3 · Factor sqrt(dk) | 5/5 de attention_paper.pdf | Igual de buena | La respuesta en español explica el mismo mecanismo |
| Q4 · Variantes RAG-Sequence y RAG-Token | 5/5 de rag_paper.pdf | Igual de buena | Respuesta más larga que en inglés |

**Hipótesis:** nemotron-3-embed-1b es multilingüe, así que la similitud entre una pregunta en español y un chunk en inglés sigue siendo suficiente para recuperar el paper correcto. La respuesta sale en español porque el prompt pide responder en el idioma de la pregunta.

**Cómo lo comprobaría:** compararía los scores de similitud de la misma pregunta en ambos idiomas (con `similarity_search_with_score`) y miraría si el orden de los chunks cambia. Si el score cae mucho en español, la hipótesis sobre el modelo multilingüe no se sostiene del todo.

### B5 · Quiz propio (pendiente)

---

## Parte R · Reflexión

**R1. ¿Qué parámetro afectó más los resultados y por qué?**
El parámetro que más cambió el número de chunks fue `k`, porque el retriever siempre devuelve exactamente ese número de fragmentos. Pasar de 3 a 8 aumentó los chunks sin mejorar los aciertos, que se mantuvieron en 5/5. `chunk_size` sí generó muchos más fragmentos al bajar a 400 (412 chunks), pero tampoco cambió los aciertos. El único parámetro que sí afectó la recuperación fue `score_threshold` en 0.80, donde los aciertos cayeron a 1/5.

**R2. ¿Cuándo un RAG debe decir "no lo sé"? Cita tu prueba de B3.**
Un RAG debe decir que no sabe cuando la respuesta no está en los fragmentos recuperados. En B3, las tres preguntas sobre ImageNet, WMT 2020 y el autor de GPT-4 recibieron respuestas que admitían no tener la información, sin inventar cifras. Eso es lo esperado, porque el modelo solo ve los extractos que el retriever le entrega, no el paper completo.

**R3. ¿Por qué la búsqueda en español funcionó (o no) con papers en inglés? Cita B4.**
En B4, las tres preguntas traducidas recuperaron los 5 fragmentos del paper correcto, y las respuestas fueron igual de buenas que en inglés. Mi hipótesis es que `nemotron-3-embed-1b` es multilingüe, así que la similitud entre una pregunta en español y un chunk en inglés basta para recuperar el paper correcto. Lo comprobaría comparando los scores de similitud de la misma pregunta en ambos idiomas.

**R4. ¿Qué cambiarías del corpus o del prompt si el sistema fuera para una empresa real?**
Cambiaría el corpus para incluir documentos con cifras y excepciones claras, como el de reembolsos, porque ahí las respuestas dependen de montos exactos. También revisaría el prompt para que cite el documento y la sección, no solo el título. Y probaría un umbral intermedio (entre 0.40 y 0.80) para evitar que el sistema pierda recuperaciones correctas.

**R5. ¿Qué fue lo más difícil del workshop y cómo lo resolviste?**
Lo más difícil fue el error 404 en la generación de B1, que se resolvió al volver a ejecutar la celda, y el 500 de NVIDIA en A2, que se solucionó reintentando. También me costó construir el experimento de B2 porque las funciones usaban nombres distintos a los del notebook (`Qdrant` en vez de `QdrantVectorStore`), así que tuve que adaptarlas.