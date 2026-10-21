# RESPUESTAS – Workshop RAG SI3003

**Nombre:** Juan David Correa
**Código estudiantil:** [pon tu código]

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

### B2 · Experimentos (pendiente)

### B3 · Preguntas trampa (pendiente)

### B4 · Preguntas en español (pendiente)

### B5 · Quiz propio (pendiente)

---

## Parte R · Reflexión (pendiente)