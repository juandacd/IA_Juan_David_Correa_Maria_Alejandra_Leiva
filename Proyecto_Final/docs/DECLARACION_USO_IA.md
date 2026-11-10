# Declaración de uso de IA generativa

**Proyecto:** Agente de calidad del aire del Valle de Aburrá · SI3003 – Inteligencia Artificial · EAFIT 2026-1
**Autores**: María Alejandra Leiva y Juan David Correa Duque


## 1. Herramientas usadas y para qué

| Herramienta | Para qué se usó |
|---|---|
| **Claude (Anthropic)** | Asistente de programación y guía paso a paso: orientación sobre el enunciado y la estructura del proyecto; escritura y revisión de código (limpieza de datos, Naive Bayes, HMM, evaluación, herramientas del agente, RAG, dashboard); búsqueda de errores; ayuda para organizar el repositorio; borradores del README y de esta declaración. |
| **Modelos de lenguaje de NVIDIA (NIM)**: `nvidia/nemotron-3.5-lightning-30b-a3b` y `nvidia/nemotron-3.5-bolt-e4b` | Son el "cerebro" del agente del proyecto: deciden qué herramienta usar y redactan las respuestas. No se usaron para escribir el código. |
| **Modelo de embeddings `intfloat/multilingual-e5-small`** | Convertir los fragmentos de los documentos en vectores para el buscador (RAG). |

## 2. Qué hicimos nosotros

- Elegimos el tema y las 4 estaciones y decidimos mantener las 3 categorías (Buena, Moderada, Alerta).
- Conseguimos y descargamos los datos de SISAIRE y los documentos del corpus.
- Ejecutamos todo el código en Google Colab, revisamos los resultados y los reportamos tal como salieron (incluido que el modelo no supera a la línea base).
- **Verificamos a mano contra el PDF** las tablas de la Resolución 2254 que el OCR leyó mal.
- Ejecutamos la evaluación del agente, detectamos problemas (ciclos de búsquedas, respuestas vacías, una cita inventada y una respuesta fuera de tema) y decidimos las correcciones.

## 3. Cómo entendimos el trabajo

Naive Bayes: es un clasificador probabilístico que utiliza el teorema de Bayes para estimar la probabilidad de que una observación pertenezca a cada categoría de calidad del aire. Su principal supuesto es que las variables son condicionalmente independientes dada la clase. Finalmente, asigna la categoría con mayor probabilidad estimada.

Modelo Oculto de Markov (HMM): es un modelo probabilístico que representa un sistema mediante estados ocultos que evolucionan en el tiempo y generan observaciones. En nuestro proyecto permite modelar la evolución temporal de las categorías de calidad del aire.

RAG (Retrieval-Augmented Generation): es una arquitectura que combina la recuperación de información documental con un modelo de lenguaje generativo. En nuestro proyecto, los documentos se dividen en fragmentos, se convierten en vectores mediante un modelo de embeddings y se almacenan en FAISS. Cuando el usuario realiza una pregunta, se recuperan los fragmentos semánticamente más relevantes y se proporcionan al modelo de lenguaje como contexto para generar una respuesta fundamentada en las fuentes disponibles. 

Loop agéntico: es el ciclo mediante el cual el modelo de lenguaje interpreta una pregunta, decide si necesita utilizar una herramienta, ejecuta la acción correspondiente y recibe su resultado. A partir de esta información puede decidir realizar otra consulta o generar una respuesta final. En nuestro proyecto, este mecanismo permite que el agente consulte datos históricos, obtenga predicciones y recupere información documental según las necesidades de cada pregunta.

Comparación con la línea base: los resultados muestran que Naive Bayes y la combinación de Naive Bayes con HMM no superaron la línea base, que consiste en asignar a cada día la misma categoría observada el día anterior. Una posible explicación es la persistencia temporal de las categorías de calidad del aire, que permite que una regla sencilla obtenga buenos resultados. Estos resultados muestran la importancia de evaluar los modelos frente a una referencia sencilla y no asumir que una mayor complejidad garantiza un mejor desempeño.

## 4. Limitaciones y responsabilidad

El desarrollo del proyecto contó con el apoyo de herramientas de IA generativa, por lo que el código y las explicaciones sugeridas por estas herramientas fueron revisados y contrastados por los integrantes del equipo. Las decisiones metodológicas, la ejecución de los experimentos y la interpretación de los resultados estuvieron bajo nuestra responsabilidad.

Los resultados reportados corresponden a las ejecuciones realizadas y no fueron modificados para favorecer el desempeño de los modelos. Reconocemos que los modelos predictivos presentan limitaciones y que el agente puede generar respuestas incompletas o incorrectas.

Por esta razón, el sistema se presenta como una herramienta de consulta y apoyo para explorar información sobre la calidad del aire del Valle de Aburrá, y no como un sustituto de las fuentes oficiales ni como un sistema de alerta oficial. Sus resultados no deben utilizarse como único fundamento para tomar decisiones médicas, legales o de seguridad.
