# Declaración de uso de IA generativa

**Proyecto:** Agente de calidad del aire del Valle de Aburrá · SI3003 – Inteligencia Artificial · EAFIT 2026-1
**Autores:** Aleja [COMPLETAR: apellidos] y Juan David Correa Duque

> ⚠️ **Borrador para revisar.** Esta declaración debe ser fiel a lo que de verdad hicieron. Léanla completa, corrijan lo que no coincida y completen los espacios marcados con [COMPLETAR] antes de entregar.

## 1. Herramientas usadas y para qué

| Herramienta | Para qué se usó |
|---|---|
| **Claude (Anthropic)** | Asistente de programación y guía paso a paso: orientación sobre el enunciado y la estructura del proyecto; escritura y revisión de código (limpieza de datos, Naive Bayes, HMM, evaluación, herramientas del agente, RAG, dashboard); búsqueda de errores; ayuda para organizar el repositorio; borradores del README y de esta declaración. |
| **Modelos de lenguaje de NVIDIA (NIM)**: `nvidia/nemotron-3.5-lightning-30b-a3b` y `nvidia/nemotron-3.5-bolt-e4b` | Son el "cerebro" del agente del proyecto: deciden qué herramienta usar y redactan las respuestas. No se usaron para escribir el código. |
| **Modelo de embeddings `intfloat/multilingual-e5-small`** | Convertir los fragmentos de los documentos en vectores para el buscador (RAG). |
| **Tesseract (OCR)** | Leer el PDF escaneado de la Resolución 2254 de 2017 (no es IA generativa, se menciona por transparencia). |

## 2. Qué hicimos nosotros

- Elegimos el tema y las 4 estaciones, descartamos Almacén EPM y decidimos mantener las 3 categorías (Buena, Moderada, Alerta).
- Conseguimos y descargamos los datos de SISAIRE y los documentos del corpus.
- Ejecutamos todo el código en Google Colab, revisamos los resultados y los reportamos tal como salieron (incluido que el modelo no supera a la línea base).
- **Verificamos a mano contra el PDF** las tablas de la Resolución 2254 que el OCR leyó mal.
- Ejecutamos la evaluación del agente, detectamos problemas (ciclos de búsquedas, respuestas vacías, una cita inventada y una respuesta fuera de tema) y decidimos las correcciones.
- [COMPLETAR: qué parte hizo cada integrante.]

## 3. Cómo entendimos el trabajo

[COMPLETAR: en pocas líneas, expliquen con sus palabras qué hace el HMM, el Naive Bayes, el RAG y el loop del agente, y por qué el modelo empata con la línea base. El enunciado exige comprender lo entregado.]

## 4. Limitaciones y responsabilidad

El código fue generado y revisado con ayuda de IA y puede contener errores. Los resultados reportados son los obtenidos al ejecutar el proyecto. El agente puede equivocarse y no constituye consejo médico, legal ni de seguridad.
