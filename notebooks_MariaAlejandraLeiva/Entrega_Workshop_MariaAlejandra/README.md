# Workshop RAG · SI3003 Inteligencia Artificial

- **Nombre:** Maria Alejandra Leiva
- **Código estudiantil:** 1000303844
- **Usuario de GitHub:** alejaleiva12
- **Fecha:** 21 de octubre de 2026
- **Entrega a:** jipadillab@eafit.edu.co

## Progreso

| ID | Tarea | Puntos | Hecha |
|---|---|---:|:---:|
| T1 | Cuántos nodos viven en cada capa | 2 | ⬜ |
| T2 | ef, k y el tamaño de la cola | 2 | ⬜ |
| T3 | Inserción frente a búsqueda | 2 | ⬜ |
| T4 | El invariante de FAISSVectorStore | 2 | ⬜ |
| T5 | El costo de la fuerza bruta | 2 | ⬜ |
| A1 | Ejecuta la app RAG de punta a punta | 4 | ⬜ |
| A2 | Tu propio corpus de documentos | 6 | ⬜ |
| A3 | Ingeniería del prompt (antes y después) | 5 | ⬜ |
| A4 | Muestra el score de similitud en las fuentes | 5 | ⬜ |
| B1 | Ejecuta el RAG de papers y evalúa con el ground truth | 6 | ⬜ |
| B2 | Experimentos con chunk_size, k y score_threshold | 8 | ⬜ |
| B3 | Preguntas trampa: ¿el RAG admite que no sabe? | 6 | ⬜ |
| B4 | Preguntas en español sobre papers en inglés | 5 | ⬜ |
| B5 | Escribe tu propio quiz con ground truth | 5 | ⬜ |
| R1 | Reflexión final con tus datos | 10 | ⬜ |
| C1 | HNSW contra fuerza bruta con FAISS (reto) | 15 | ⬜ |
| C2 | Un tercer paper: el RAG que explica HNSW (reto) | 12 | ⬜ |
| C3 | App Gradio para los papers con filtro por paper (reto) | 15 | ⬜ |
| C4 | Evaluación automática de la recuperación (reto) | 12 | ⬜ |
| C5 | MMR frente a similitud pura (reto) | 8 | ⬜ |
| C6 | Persistencia del índice en disco (reto) | 8 | ⬜ |

**Autoevaluación:** 0 pts de núcleo + 0 pts de retos (tope 30) = **0 / 100** → bonificación estimada **0.00 / 0.50**.

## Bloque 0 · Teoría

### T1 · Nodos por capa

P(l ≥ k) = P(U ≤ e^(−k·ln M)) = M^(−k). Con M = 16 y N = 10⁶:

- Capa ≥ 1: 10⁶ / 16 = **62.500**
- Capa ≥ 2: 10⁶ / 256 ≈ **3.906**
- Capa ≥ 3: 10⁶ / 4096 ≈ **244**

La simulación realizada en el código dio los mismo resultados. 

### T2 · ef, k y la cola

1. Con `k=10` y `ef=5`, la cola solo guarda 5 candidatos, así que es imposible devolver 10 vecinos. Siempre debe cumplirse `ef ≥ k`.
2. Se empieza con `ef = 2·k` (o 50–100 para RAG con k de 5 a 10) y se sube si el recall es insuficiente.
3. `ef`, porque es un parámetro de consulta y no cambia el grafo. `ef_construction` solo actúa al insertar, así que cambiarlo exige reconstruir el índice.

### T3 · Inserción frente a búsqueda

| | Inserción | Búsqueda |
|---|---|---|
| ¿Se sortea la capa? | **Sí.** Hay que decidir hasta qué capa vivirá el nodo nuevo; el azar evita depender del orden de inserción. | **No.** No hay nodo nuevo que ubicar; se entra siempre por la capa más alta desde el nodo de entrada guardado. |
| ef en capas altas | **ef = 1**, de la capa más alta hasta l+1. Solo se busca una buena zona de entrada. | **ef = 1**, de la capa más alta hasta la 1. Es una aproximación gruesa y barata. |
| Parámetro de la fase principal | **ef_construction**, de la capa l a la 0. Más candidatos dan mejores conexiones. | **ef** en la capa 0. Más candidatos dan más recall y más latencia. |
| ¿Modifica el índice? | **Sí.** Crea aristas bidireccionales y poda las más débiles. | **No.** Solo lee el grafo y devuelve los k más cercanos. |

En el notebook 03, `vectorstore.add_documents(...)` dispara la **inserción** y `retriever.invoke(...)` dispara la **búsqueda**.

### T4 · El invariante de FAISSVectorStore

| Pos. | FAISS (sin tocar) | Lista antes | Lista tras borrar B |
|---|---|---|---|
| 0 | vector A | A | A |
| 1 | vector B | B | **C** |
| 2 | vector C | C | **D** |
| 3 | vector D | D | *(no existe)* |

La búsqueda que antes daba `[1, 2, 3]` (B, C, D) ahora resuelve así:
- idx 1 → `documents[1]` = **C**. Debía ser B.
- idx 2 → `documents[2]` = **D**. Debía ser C.
- idx 3 → `documents[3]` → **IndexError**.

Los dos primeros son errores **silenciosos**: el score corresponde al vector de B, pero el texto mostrado es el de C. Solo el tercero lanza una excepción.

Esto conecta con Qdrant (y Weaviate) porque `IndexFlatIP` no tiene borrado y la lista paralela se desincroniza. Las bases vectoriales gestionan las mutaciones de forma nativa y guardan la metadata junto al vector.

### T5 · Costo de la fuerza bruta

1. 768 × 10⁶ multiplicaciones × 100 consultas/s = **7,68 × 10¹⁰ mult/s**, unos 77 mil millones por segundo.
2. Eso es solo aritmética. Además hay que leer 10⁶ × 768 × 4 B ≈ 3 GB por consulta, unos 300 GB/s de ancho de banda a 100 consultas/s. El costo crece linealmente con N, así que ser exacto no escala. (El dato de memoria es un cálculo mío; verifícalo antes de incluirlo.)
3. En ANN se sacrifica la garantía de exactitud, con un recall típico de ~95 % o más, y se gana latencia de órdenes de magnitud, unas 10.000 veces más rápido según la presentación.

## Bloque A · App RAG empresarial

### A1

Tabla de 5 preguntas con respuesta y fuentes:

user: ¿Cuál es el precio del plan Enterprise de CloudSync Pro?

assistant: El plan Enterprise de CloudSync Pro cuesta $49/mes por usuario con compromiso anual.

user: ¿Cuántos días pueden trabajar remotamente los empleados?

assistant: Hasta 3 días por semana, con aprobación del jefe directo.

user: ¿Qué pasa con los reembolsos mayores a $100?

assistant: Requieren aprobación del supervisor. Una vez aprobados, se procesan en 3 a 5 días hábiles al método de pago original.

user: ¿Qué ocurre durante la primera semana de onboarding?

assistant: Durante la primera semana se completan módulos obligatorios de seguridad, cumplimiento y cultura de la empresa.

user: Cuales son los colores de la bandera de colombia?

assistant: No tengo esa información en el contexto proporcionado.

Ver imagen A1_gradio.png

### A2

| # | Pregunta | Respuesta obtenida del RAG | Fuente utilizada | ¿Correcta? |
|---|---|---|---|---|
| 1 | ¿Cuántos días por semana se permite trabajar remotamente en NovaTech y quién lo aprueba? | Hasta dos días por semana, y debe ser aprobado previamente por el supervisor directo. | `doc_01` — Política de trabajo remoto | Sí |
| 2 | ¿Con cuánta anticipación se deben solicitar las vacaciones? | Al menos diez días calendario de anticipación. | `doc_02` — Política de vacaciones | Sí |
| 3 | ¿Quién debe aprobar un descuento comercial del 15 %? | El gerente comercial. | `doc_04` — Política de descuentos comerciales | Sí |
| 4 | ¿Cuál es el horario de atención al cliente de NovaTech? | El horario de atención al cliente de NovaTech es de lunes a viernes, de 8:00 a. m. a 5:00 p. m., hora de Colombia. | `doc_05` — Horario de atención al cliente | Sí |
| 5 | ¿Cuál es el salario mensual del gerente general de NovaTech? | No dispongo de esa información. El contexto proporcionado no contiene detalles sobre los salarios del personal de NovaTech. | Ninguna fuente pertinente en el corpus | Sí |
| 6 | ¿Qué compañía de seguros médicos tiene contratada NovaTech? | No tengo esa información en el contexto proporcionado. | Ninguna fuente pertinente en el corpus | Sí |
| 7 | ¿Qué día de la semana es hoy? | No tengo esa información en el contexto proporcionado. | Ninguna fuente pertinente en el corpus | Sí |

Análisis de resultados: De las preguntas que se pudieron evaluar, las cuatro preguntas respondibles obtuvieron respuestas correctas y las otras preguntas fuera de contexto fueron respondidas reconociendo que la información no estaba disponible en los documentos. Esto muestra que el sistema logró recuperar información pertinente para las preguntas cubiertas por el corpus y, en estos dos casos, evitó inventar información ausente. 

Justificación de chunk_size y chunk_overlap: Se utilizó `chunk_size=400` y `chunk_overlap=40` porque los documentos del corpus son relativamente breves y describen políticas independientes. Este tamaño busca conservar las reglas junto con su contexto, mientras que el solapamiento permite mantener parte de la información contextual cuando un documento se divide en varios fragmentos.

Ver imagen A2_corpus.png

###A 3

| Pregunta | Antes del cambio | Después del cambio | Análisis |
|---|---|---|---|
| ¿Quién debe aprobar un descuento comercial del 15 %? | El gerente comercial. | El gerente comercial debe aprobar un descuento comercial del 15 %. [Política de descuentos comerciales] | La nueva respuesta conserva la información correcta e incorpora el título de la fuente entre corchetes, cumpliendo la instrucción de citar el documento utilizado. |
| ¿Cuál es el salario mensual del gerente general de NovaTech? | No dispongo de esa información. El contexto proporcionado no contiene detalles sobre los salarios del personal de NovaTech. | No encuentro esa información en los documentos. | El modelo siguió la instrucción de responder exactamente con la frase establecida cuando la información no está disponible. Además, no inventó información. |
| ¿Cuál es el horario de atención al cliente de NovaTech? | El horario de atención al cliente de NovaTech es de lunes a viernes, de 8:00 a. m. a 5:00 p. m., hora de Colombia. | El horario de atención al cliente de NovaTech es de lunes a viernes, de 8:00 a. m. a 5:00 p. m., hora de Colombia. [Horario de atención al cliente] | La información se mantuvo correcta y se añadió la cita de la fuente. La respuesta está en español y utiliza una sola oración, por lo que cumple el límite de extensión. |

Conclusión:

La instrucción que tuvo mayor efecto observable fue la de citar el título de la fuente entre corchetes, ya que las dos respuestas a preguntas respondibles incorporaron explícitamente el documento utilizado. También se observó un efecto claro en la pregunta fuera de contexto: el modelo utilizó exactamente la frase indicada en el nuevo prompt, en lugar de la explicación más extensa que producía anteriormente.

En las tres respuestas evaluadas, el modelo no ignoró ninguna de las instrucciones principales: respondió en español, utilizó un máximo de tres oraciones y reconoció la ausencia de información cuando correspondía. Sin embargo, esta prueba no permite afirmar que nunca ignore las instrucciones en otras preguntas; solo muestra que las siguió en los tres casos evaluados.


## Decisiones y aprendizajes

Escribe 5 a 8 líneas sobre lo más importante que descubriste hoy.
