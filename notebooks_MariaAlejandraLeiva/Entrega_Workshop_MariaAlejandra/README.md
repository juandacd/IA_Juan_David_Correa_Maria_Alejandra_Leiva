# Workshop RAG · SI3003 Inteligencia Artificial

- **Nombre:** Maria Alejandra Leiva
- **Código estudiantil:** 1000303844
- **Usuario de GitHub:** alejaleiva12
- **Fecha:** 21 de octubre de 2026
- **Entrega a:** jipadillab@eafit.edu.co

## Progreso

| ID | Tarea | Puntos | Hecha |
|---|---|---:|:---:|
| T1 | Cuántos nodos viven en cada capa | 2 | SI |
| T2 | ef, k y el tamaño de la cola | 2 | SI |
| T3 | Inserción frente a búsqueda | 2 | SI |
| T4 | El invariante de FAISSVectorStore | 2 | SI |
| T5 | El costo de la fuerza bruta | 2 | SI |
| A1 | Ejecuta la app RAG de punta a punta | 4 | SI |
| A2 | Tu propio corpus de documentos | 6 | SI |
| A3 | Ingeniería del prompt (antes y después) | 5 | SI |
| A4 | Muestra el score de similitud en las fuentes | 5 | SI |
| B1 | Ejecuta el RAG de papers y evalúa con el ground truth | 6 |  SI |
| B2 | Experimentos con chunk_size, k y score_threshold | 8 |  SI |
| B3 | Preguntas trampa: ¿el RAG admite que no sabe? | 6 |  SI |
| B4 | Preguntas en español sobre papers en inglés | 5 |  SI |
| B5 | Escribe tu propio quiz con ground truth | 5 |  SI |
| R1 | Reflexión final con tus datos | 10 |  SI |

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

### A3

| Pregunta | Antes del cambio | Después del cambio | Análisis |
|---|---|---|---|
| ¿Quién debe aprobar un descuento comercial del 15 %? | El gerente comercial. | El gerente comercial debe aprobar un descuento comercial del 15 %. [Política de descuentos comerciales] | La nueva respuesta conserva la información correcta e incorpora el título de la fuente entre corchetes, cumpliendo la instrucción de citar el documento utilizado. |
| ¿Cuál es el salario mensual del gerente general de NovaTech? | No dispongo de esa información. El contexto proporcionado no contiene detalles sobre los salarios del personal de NovaTech. | No encuentro esa información en los documentos. | El modelo siguió la instrucción de responder exactamente con la frase establecida cuando la información no está disponible. Además, no inventó información. |
| ¿Cuál es el horario de atención al cliente de NovaTech? | El horario de atención al cliente de NovaTech es de lunes a viernes, de 8:00 a. m. a 5:00 p. m., hora de Colombia. | El horario de atención al cliente de NovaTech es de lunes a viernes, de 8:00 a. m. a 5:00 p. m., hora de Colombia. [Horario de atención al cliente] | La información se mantuvo correcta y se añadió la cita de la fuente. La respuesta está en español y utiliza una sola oración, por lo que cumple el límite de extensión. |

Conclusión:

La instrucción que tuvo mayor efecto observable fue la de citar el título de la fuente entre corchetes, ya que las dos respuestas a preguntas respondibles incorporaron explícitamente el documento utilizado. También se observó un efecto claro en la pregunta fuera de contexto: el modelo utilizó exactamente la frase indicada en el nuevo prompt, en lugar de la explicación más extensa que producía anteriormente.

En las tres respuestas evaluadas, el modelo no ignoró ninguna de las instrucciones principales: respondió en español, utilizó un máximo de tres oraciones y reconoció la ausencia de información cuando correspondía. Sin embargo, esta prueba no permite afirmar que nunca ignore las instrucciones en otras preguntas; solo muestra que las siguió en los tres casos evaluados.

### A4

Se realizaron cinco pruebas: tres preguntas respondibles con los documentos de NovaTech y dos preguntas fuera de contexto. Para cada pregunta se registraron las tres fuentes recuperadas y sus scores de similitud.

| Pregunta | Tipo | Fuente recuperada | Score |
|---|---|---|---:|
| ¿Quién debe aprobar un descuento comercial del 15 %? | Respondible | Política de descuentos comerciales | 0.4210 |
| ¿Quién debe aprobar un descuento comercial del 15 %? | Respondible | Política de reembolso de gastos | 0.1400 |
| ¿Quién debe aprobar un descuento comercial del 15 %? | Respondible | Política de reembolso de gastos | 0.0967 |
| ¿Cuántos días por semana se permite trabajar de forma remota y quién lo aprueba? | Respondible | Política de trabajo remoto | 0.4530 |
| ¿Cuántos días por semana se permite trabajar de forma remota y quién lo aprueba? | Respondible | Política de reembolso de gastos | 0.2395 |
| ¿Cuántos días por semana se permite trabajar de forma remota y quién lo aprueba? | Respondible | Política de vacaciones | 0.2068 |
| ¿Cuál es el horario de atención al cliente de NovaTech? | Respondible | Horario de atención al cliente | 0.6239 |
| ¿Cuál es el horario de atención al cliente de NovaTech? | Respondible | Política de trabajo remoto | 0.4443 |
| ¿Cuál es el horario de atención al cliente de NovaTech? | Respondible | Política de vacaciones | 0.3635 |
| ¿Cuál es el salario mensual del gerente general de NovaTech? | Fuera de contexto | Política de descuentos comerciales | 0.3591 |
| ¿Cuál es el salario mensual del gerente general de NovaTech? | Fuera de contexto | Política de trabajo remoto | 0.3524 |
| ¿Cuál es el salario mensual del gerente general de NovaTech? | Fuera de contexto | Política de reembolso de gastos | 0.3486 |
| ¿Qué empresa de seguros médicos utiliza NovaTech para sus empleados? | Fuera de contexto | Programa de capacitación | 0.3710 |
| ¿Qué empresa de seguros médicos utiliza NovaTech para sus empleados? | Fuera de contexto | Política de vacaciones | 0.3559 |
| ¿Qué empresa de seguros médicos utiliza NovaTech para sus empleados? | Fuera de contexto | Política de trabajo remoto | 0.3463 |

 Umbral propuesto

Se propone cambiar `score_threshold` de **0.30 a 0.40**.

La fuente correcta con menor score entre las tres preguntas respondibles obtuvo 0.4210, mientras que el score más alto de las fuentes recuperadas para las dos preguntas fuera de contexto fue 0.3710. Por tanto, 0.40 queda entre ambos valores: conserva las fuentes correctas principales observadas y excluye las fuentes de las pruebas fuera de contexto en estos cinco casos.

Este umbral es una propuesta inicial basada en una muestra pequeña. Debe validarse con más preguntas antes de considerarlo definitivo.

### Conclusión

Los scores ayudan a distinguir las fuentes más relacionadas con la pregunta, pero no garantizan que una fuente contenga la respuesta. En las pruebas realizadas, las preguntas fuera de contexto también recuperaron documentos con scores relativamente altos, por lo que establecer un umbral puede reducir recuperaciones poco relevantes. La evaluación se limita a las cinco preguntas utilizadas.

Ver imagen A4_scores.

## Bloque B · RAG sobre papers académicos

### B1

| Pregunta | ¿Respuesta correcta? | ¿Se recuperó el contexto del paper correcto? | Observación |
|---|---|---|---|
| Q1. BLEU del Transformer grande | Sí | Sí, `attention_paper.pdf` | Respondió 28.4 BLEU para WMT 2014 inglés-alemán. |
| Q2. Memoria paramétrica y no paramétrica en RAG | Sí | Sí, `rag_paper.pdf` | Explicó el papel del generador BART y del índice denso recuperado mediante DPR. |
| Q3. Propósito de `1/sqrt(d_k)` | Sí | Sí, `attention_paper.pdf` | Explicó que el escalado evita productos punto demasiado grandes y gradientes muy pequeños en la softmax. |
| Q4. Variantes RAG-Sequence y RAG-Token | Sí | Sí, `rag_paper.pdf` | Diferenció el uso de un mismo documento para toda la secuencia frente a documentos diferentes por token. |
| Q5. Cabezas de atención y dimensiones | Sí | Sí, `attention_paper.pdf` | Indicó 8 cabezas y dimensiones `d_k = d_v = 64` por cabeza. |
| Bonus. Generador de RAG y arquitectura Transformer | Sí | Sí, aparecieron ambos papers | Se recuperaron cuatro fragmentos de `rag_paper.pdf` y uno de `attention_paper.pdf`. |

Conclusión B1: el RAG obtuvo 5/5 aciertos en las preguntas principales y respondió correctamente la pregunta bonus. En la bonus sí aparecieron ambos papers, aunque la mayoría de los fragmentos recuperados pertenecía al artículo de RAG.

### B2

| Experimento | Configuración | Número de chunks del índice | Aciertos |
|---|---|---:|---:|
| `chunk_size` | 400, `overlap=52` | 412 | 5/5 |
| `chunk_size` | 900, `overlap=120` | 150 | 5/5 |
| `chunk_size` | 1500, `overlap=195` | 150 | 5/5 |
| `k` | 3 | 150 | 5/5 |
| `k` | 5 | 150 | 5/5 |
| `k` | 8 | 150 | 5/5 |
| `score_threshold` | 0.15 | 150 | 5/5 |
| `score_threshold` | 0.25 | 150 | 5/5 |
| `score_threshold` | 0.40 | 150 | 5/5 |

En los experimentos realizados, el sistema obtuvo 5/5 aciertos con los tres valores de chunk_size, los tres valores de k y los umbrales de 0.15, 0.25 y 0.40. Sin embargo, el número de chunks del índice cambió de 412 con chunk_size=400 a 150 con chunk_size=900 y chunk_size=1500. Al aumentar k, se recuperaron más fragmentos, pero no mejoró el número de aciertos en estas cinco preguntas. Los umbrales de 0.15, 0.25 y 0.40 tampoco cambiaron los aciertos, aunque un experimento adicional con 0.80 redujo el resultado a 1/5. Esto sugiere que un umbral demasiado alto puede excluir información útil. Con los datos actuales conservaría chunk_size=900, overlap=120, k=5 y score_threshold=0.25 como configuración inicial, porque funcionó correctamente sin recuperar tantos fragmentos como k=8. No obstante, los cinco aciertos no demuestran que sea la mejor configuración para todas las preguntas; haría falta probar más consultas, especialmente preguntas difíciles y preguntas sin respuesta en los documentos.

### B3

| Experimento | Configuración | Aciertos | Observación |
|---|---|---:|---|
| `score_threshold` | 0.80 | 1/5 | El umbral fue demasiado estricto y redujo la recuperación a un promedio de 0.4 fragmentos por pregunta. |

En las tres preguntas trampa, el modelo reconoció que los fragmentos recuperados no contenían información suficiente para responder. No inventó una puntuación para ImageNet, un resultado BLEU de 2020 ni el autor del informe técnico de GPT-4. Las respuestas explicaron qué información faltaba y se mantuvieron dentro de los límites del contexto recuperado. Por tanto, en estas pruebas el RAG mostró un comportamiento adecuado frente a preguntas fuera del alcance de los papers. No fue necesario modificar el prompt porque las tres respuestas observadas ya reconocían la falta de evidencia. Aun así, convendría probar preguntas más ambiguas, ya que estas tres respuestas correctas no garantizan que el sistema siempre evite las alucinaciones.

### B4

| Pregunta en español | Fuente recuperada | ¿Respuesta correcta? | Observación |
|---|---|---|---|
| ¿Cuál fue la puntuación BLEU del Transformer grande en WMT 2014 inglés-alemán? | `attention_paper.pdf` | Sí | Respondió 28.4 BLEU. |
| ¿Cuál es el propósito del factor de escala `sqrt(d_k)`? | `attention_paper.pdf` | Sí | Explicó el efecto de los productos punto grandes sobre la softmax y sus gradientes. |
| ¿Qué dos variantes de RAG propone el paper y en qué se diferencian? | `rag_paper.pdf` | Sí | Diferenció RAG-Sequence de RAG-Token. |

Hipótesis y cómo comprobarla
Los resultados sugieren que el modelo de embeddings nvidia/nemotron-3-embed-1b pudo relacionar las preguntas en español con fragmentos relevantes de documentos escritos en inglés. Es posible que esto se deba a su capacidad para representar semánticamente textos en distintos idiomas, pero las ejecuciones por sí solas no demuestran cómo fue entrenado ni garantizan el mismo comportamiento en todos los idiomas.

Para comprobarlo mejor, compararía cada pregunta en inglés con su traducción al español y registraría los fragmentos recuperados, el orden de las fuentes y la corrección de las respuestas. También probaría preguntas con vocabulario técnico y traducciones menos literales. En el notebook, las tres preguntas en español recuperaron el paper correcto y produjeron respuestas sustancialmente correctas.


### B5

El notebook ya tiene cinco preguntas de referencia, pero para este bloque necesitas añadir tres preguntas nuevas, sus respuestas esperadas y la sección del paper donde se encuentra cada respuesta. Una debe combinar información de ambos artículos.

Estas tres propuestas se basan en los temas que aparecen en el notebook y en el ground truth existente. Las puedes agregar como una nueva sección de código.
Pregunta nueva 1 — Attention Is All You Need
Pregunta:
Why does the Transformer use multi-head attention instead of a single attention function with full dimensionality?

Ground truth:

The Transformer uses multi-head attention to allow the model to attend to information from different representation subspaces at different positions. Instead of performing a single attention function with full dimensionality, it projects queries, keys, and values into multiple lower-dimensional spaces, applies attention in parallel, and combines the results.

Sección: Attention Is All You Need, sección 3.2.2, “Multi-Head Attention”.
Pregunta nueva 2 — Retrieval-Augmented Generation
Pregunta:
What is the role of the dense vector index in the RAG architecture?

Ground truth:

The dense vector index provides non-parametric memory by storing document representations that can be retrieved in response to an input query. A neural retriever identifies relevant documents, and the generator conditions on those documents to produce an answer. In the paper, the index is built from Wikipedia and accessed using the DPR retriever.

Sección: Retrieval-Augmented Generation, secciones 1 y 2.
Pregunta nueva 3 — combinación de ambos papers
Pregunta:
How are BART in the RAG paper and the Transformer architecture described in “Attention Is All You Need” related?

Ground truth:

The RAG paper uses BART-large as its sequence-to-sequence generator. It describes BART as a pretrained sequence-to-sequence Transformer, linking the generator used by RAG to the Transformer architecture discussed in “Attention Is All You Need”. RAG then conditions this generator on retrieved documents and the input query to produce its output.

## Reflexión

### 1. ¿Qué ocurrió cuando el contexto no contenía información suficiente? ¿El modelo reconoció la limitación o inventó una respuesta?

En las tres preguntas trampa, el modelo reconoció que los fragmentos recuperados no contenían información suficiente para responder. Por ejemplo, ante la pregunta sobre la precisión de RAG en ImageNet, indicó que los documentos proporcionados no mencionaban esa tarea ni sus resultados. También evitó inventar una puntuación BLEU para WMT 2020 y un autor para el informe técnico de GPT-4. Esto demuestra que, en estas pruebas, el sistema fue capaz de reconocer sus limitaciones y responder de manera prudente.

### 2. ¿Qué fragmentos se recuperaron para la pregunta bonus? ¿Aparecieron ambos papers?

Para la pregunta bonus se recuperaron cinco fragmentos: cuatro del artículo *Retrieval-Augmented Generation* y uno de *Attention Is All You Need*. Por tanto, aparecieron ambos papers, aunque predominó el artículo de RAG. La respuesta identificó correctamente a BART como el generador utilizado por RAG y explicó que está basado en la arquitectura Transformer. Este experimento muestra cómo la recuperación de información de diferentes documentos puede ayudar a responder preguntas que relacionan conceptos de ambos artículos.

### 3. ¿Qué efecto tuvo modificar `chunk_size`, `chunk_overlap`, `k` o `score_threshold`?

Al cambiar `chunk_size`, el índice pasó de 412 chunks con un tamaño de 400 a 150 chunks con tamaños de 900 y 1500, mientras que las tres configuraciones obtuvieron 5/5 aciertos. Con el índice de 900, modificar `k` entre 3, 5 y 8 tampoco cambió los aciertos, aunque sí aumentó el número de fragmentos recuperados por pregunta. Los umbrales de 0.15, 0.25 y 0.40 mantuvieron los cinco aciertos, pero el experimento adicional con un umbral de 0.80 redujo el resultado a 1/5 y el promedio a 0.4 fragmentos por pregunta. Por estos resultados, conservaría inicialmente `chunk_size=900`, `overlap=120`, `k=5` y `score_threshold=0.25`, aunque probaría más preguntas antes de concluir que esta es la mejor configuración general.

### 4. ¿Por qué conviene separar el modelo de embeddings del modelo generativo?

El modelo de embeddings transforma los documentos y las preguntas en representaciones vectoriales que permiten recuperar fragmentos semánticamente relevantes. El modelo generativo utiliza la pregunta y el contexto recuperado para construir una respuesta en lenguaje natural. Separar ambos componentes permite evaluar y modificar la recuperación sin tener que cambiar necesariamente el generador, y viceversa. En este notebook, el sistema recuperó los papers adecuados para las preguntas principales y también respondió correctamente las tres preguntas formuladas en español, aunque los documentos estaban escritos en inglés.

### 5. ¿Qué cambiarías para llevar este prototipo a producción?

Primero, utilizaría una base de datos vectorial persistente para conservar el índice y evitar reconstruirlo cada vez que se inicia la aplicación. También implementaría controles de acceso para proteger los documentos y las credenciales de la API. En segundo lugar, prepararía un conjunto de evaluación más amplio con preguntas respondibles, preguntas fuera de contexto y preguntas que requieran combinar fuentes, midiendo tanto la calidad de la recuperación como la corrección de las respuestas. Además, incorporaría observabilidad para registrar latencia, errores, fragmentos recuperados, uso de tokens y costos. Finalmente, establecería límites de consumo y evaluaciones periódicas para detectar cambios en el rendimiento y mantener la fiabilidad del sistema.
