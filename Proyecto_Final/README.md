# Agente de calidad del aire del Valle de Aburrá

Proyecto final de **SI3003 – Inteligencia Artificial** (Universidad EAFIT, 2026-1).

**Autores:** Aleja [COMPLETAR: apellidos] y Juan David Correa Duque · **Grupo:** [COMPLETAR: nombre del grupo]

---

## ¿De qué trata?

En Medellín se habla mucho del PM2.5 (las partículas finas del aire), pero casi nadie sabe qué significa el número del día ni qué hacer con él. Este proyecto junta tres cosas:

1. **Datos reales** de PM2.5 de 4 estaciones de Medellín (Parque Biblioteca, Santa Elena, Tráfico Centro y Villahermosa), de 2019 a septiembre de 2026.
2. **Modelos hechos por nosotros desde cero** que predicen si mañana el aire estará *Buena*, *Moderada* o en *Alerta*: un Naive Bayes y un HMM (modelo oculto de Markov).
3. **Un agente con inteligencia artificial** al que se le puede conversar. Consulta los datos, usa el modelo de predicción y busca en documentos oficiales (normas y guías de salud), citando la fuente.

Todo se ve en un **dashboard** con mapa, gráficas y un chat.

> Es una herramienta académica y analítica. **No es consejo médico, legal ni de seguridad.**

---

## Cómo ejecutarlo (paso a paso)

Necesitas **Python 3.10 a 3.13** e internet (la primera vez se descarga un modelo de unos 470 MB).

### 1. Descargar el proyecto e instalar

**Windows (PowerShell):**
```powershell
git clone https://github.com/[COMPLETAR: usuario]/[COMPLETAR: repositorio].git
cd [COMPLETAR: repositorio]
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Si PowerShell no deja activar el entorno, ejecuta antes: `Set-ExecutionPolicy -Scope Process RemoteSigned`.

**Mac / Linux:**
```bash
git clone https://github.com/[COMPLETAR: usuario]/[COMPLETAR: repositorio].git
cd [COMPLETAR: repositorio]
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Poner la clave de NVIDIA (solo para el chat con el agente)

El agente usa modelos de lenguaje de NVIDIA, que tienen capa gratuita.

1. Crea una cuenta gratis en <https://build.nvidia.com> y genera una API key.
2. Copia el archivo de ejemplo y pega tu clave:
   - Windows: `copy .env.example .env`
   - Mac / Linux: `cp .env.example .env`
3. Abre el archivo `.env` y reemplaza `pega_aqui_tu_clave` por tu clave.

> **Sin clave el dashboard abre igual**: el mapa, las gráficas y la predicción funcionan. Solo el chat queda desactivado y avisa por qué.

### 3. (Recomendado) Probar que todo está bien

```bash
python scripts/smoke_test.py
```
Revisa cada pieza (datos, modelos, herramientas, documentos, buscador, dashboard y agente) y dice `OK` o `FALLA` con el motivo. Con `--sin-llm` no llama a NVIDIA.

### 4. Abrir el dashboard

```bash
python app.py
```
Se abre solo en <http://127.0.0.1:7860>. La **primera vez** tarda unos minutos: limpia los datos, entrena los modelos y construye el índice de búsqueda (descarga el modelo de embeddings). Las siguientes veces abre en segundos. Para detenerlo: `Ctrl + C`.

Opciones: `--puerto 8000` (otro puerto), `--sin-navegador` (no abrir el navegador), `--share` (crear un enlace público temporal; cualquiera con el enlace podría gastar tu cuota de NVIDIA).

### ¿Prefieres Google Colab?
```python
!git clone https://github.com/[COMPLETAR: usuario]/[COMPLETAR: repositorio].git
%cd [COMPLETAR: repositorio]
!pip install -q -r requirements.txt
import os
from google.colab import userdata
os.environ["NVIDIA_API_KEY"] = userdata.get("NVIDIA_API_KEY")   # la clave guardada en los "Secrets" de Colab
!python app.py --share --sin-navegador
```

---

## ¿Qué hay en el dashboard?

- **Selectores:** estación, fecha base (se predice el día siguiente) y cuántos días mostrar en la serie.
- **Tarjetas:** PM2.5 del día, categoría, promedio de 7 días y predicción de mañana.
- **Mapa** de las estaciones coloreado por categoría.
- **Serie de tiempo** con los umbrales (12 y 37 µg/m³) y los estados ocultos del HMM.
- **Comparación** entre estaciones.
- **Predicción:** probabilidades de cada categoría para mañana y el desempeño del modelo frente a la línea base.
- **Agente:** chat que consulta datos, predice y busca en los documentos.

---

## Cómo funciona

### Parte 1: modelos clásicos (hechos desde cero, solo con numpy)

- **Categorías:** *Buena* (≤ 12 µg/m³), *Moderada* (12–37) y *Alerta* (> 37). Equivalen al ICA oficial (Resolución 2254 de 2017): Buena = Buena, Moderada = Aceptable y Alerta = "Dañina a la salud de grupos sensibles" o peor. Ojo: la "Alerta" de nuestro modelo **no** es la declaratoria oficial de alerta del Área Metropolitana (esa es de 56 a 150 µg/m³).
- **Línea base:** "mañana será igual que hoy".
- **Naive Bayes:** cuenta, en el pasado, qué pasó al día siguiente en situaciones parecidas (cómo estuvo el aire hoy y ayer, qué día y mes es) y calcula probabilidades.
- **HMM:** supone que el aire tiene un estado oculto (más limpio o más contaminado) que dura días; lo aprende con el algoritmo Baum-Welch (con *forward*, *backward* y *Viterbi* escritos por nosotros). Su estado se usa como una pista más para el Naive Bayes.
- **Evaluación sin trampa:** se entrena con 2019–2025 y se prueba con 2026, que el modelo nunca ve. Las pistas se eligieron con 2025 como validación. Las métricas son accuracy y F1.

### Parte 2: agente con RAG

El agente usa un modelo de lenguaje de NVIDIA y **tres herramientas**:

| Herramienta | Para qué sirve |
|---|---|
| `obtener_pm25` | Consultar el PM2.5 medido en una estación y fecha |
| `predecir_manana` | Predecir la categoría del día siguiente (el modelo de la Parte 1) |
| `buscar_documentos` | Buscar en los documentos (RAG con embeddings y FAISS) y citar la fuente y la página |

Cómo se cuida el agente: tiene un máximo de vueltas y de consultas por pregunta; si una herramienta falla, el error vuelve al modelo como observación; si el modelo da una respuesta vacía, se muestra lo que encontraron las herramientas; solo responde sobre calidad del aire; no da consejo médico; y no puede inventar citas.

**Documentos del buscador** (carpeta `corpus/`):
- Resolución 2254 de 2017 del Ministerio de Ambiente (niveles y categorías del ICA). El PDF original es un escaneo, así que se leyó con OCR y **las tablas de PM2.5 se transcribieron a mano y se verificaron contra el PDF** (`res_2254_2017_tablas_pm25.txt`).
- Informes de operación del AMVA/SIATA (8 de julio de 2021 y 28 de julio de 2026).
- Directrices mundiales de la OMS sobre la calidad del aire 2021 (resumen ejecutivo).

---

## Resultados

### Modelos (prueba en 2026, promedio de las 4 estaciones)

| Modelo | Accuracy | F1 macro (clases presentes) |
|---|---|---|
| Línea base (mañana = hoy) | 0.824 | 0.659 |
| Naive Bayes | 0.816 | 0.642 |
| Naive Bayes + HMM (modelo final) | 0.819 | 0.645 |

**El modelo empata con la línea base, no la supera.** Es razonable: el PM2.5 cambia despacio de un día al siguiente, así que "mañana = hoy" ya acierta mucho. Detalle por estación en `results/metricas_modelos.csv`; matrices de confusión en `results/matrices_confusion.txt`; la validación de las pistas en `results/validacion_pistas.csv`. Para repetirlo: `python scripts/evaluar_modelos.py`.

### Agente (15 preguntas fijas)

`python scripts/evaluar_agente.py` prueba 15 preguntas (datos, predicción, documentos, una combinada, errores, seguridad y tres fuera de alcance) y guarda `results/evaluacion_agente.csv` con: si usó la herramienta correcta, si citó la fuente, si el contenido es correcto y si hay citas falsas. En nuestras dos ejecuciones finales: herramienta correcta **15/15**, contenido correcto **15/15** (revisión automática), citas falsas **0/15**. Esos resultados llegaron después de corregir problemas que la evaluación fue mostrando (ciclos de búsquedas, respuestas vacías, una cita inventada y una respuesta fuera de tema). Detalles y limitaciones en el informe (`docs/`).

> El modelo de lenguaje varía un poco entre ejecuciones, así que los números pueden cambiar levemente.

---

## Estructura del repositorio

```text
├── app.py                  Abre el dashboard (un solo comando)
├── requirements.txt        Librerías necesarias
├── .env.example            Muestra dónde va la clave de NVIDIA
├── src/
│   ├── config.py           Rutas, estaciones, fechas, umbrales y modelos de lenguaje
│   ├── datos.py            Limpieza de datos, categorías, división temporal y métricas
│   ├── modelos.py          Naive Bayes y HMM escritos desde cero
│   ├── pipeline.py         Entrenamiento y predicción de mañana
│   ├── rag.py              Lectura de documentos, embeddings, FAISS y búsqueda
│   ├── agente.py           Herramientas, instrucciones y loop del agente
│   └── dashboard.py        Interfaz (Gradio + Plotly)
├── scripts/
│   ├── evaluar_modelos.py  Resultados de la Parte 1
│   ├── evaluar_agente.py   Evaluación del agente (15 preguntas)
│   └── smoke_test.py       Prueba rápida de todas las piezas
├── data/raw/               Archivos .xls originales de SISAIRE
├── data/processed/         Datos limpios (pm25_diario.csv)
├── corpus/                 Documentos del buscador (RAG)
├── results/                Tablas de resultados y evaluaciones
├── notebooks/              Cuaderno que recorre el proyecto
└── docs/                   Informe y declaración de uso de IA
```

---

## Datos y uso responsable

- **Fuente:** SISAIRE (Subsistema de Información sobre Calidad del Aire de Colombia, IDEAM), PM2.5 diario de las estaciones Parque Biblioteca, Santa Elena, Tráfico Centro y Villahermosa. Descargados el [COMPLETAR: fecha de descarga].
- **Plan B del enunciado:** no usamos una API en tiempo real; trabajamos con históricos descargados. Por eso los datos llegan hasta el **30 de septiembre de 2026** y el dashboard predice a partir de la fecha base elegida, no del día de hoy. Para usarlo en tiempo real habría que conectar una fuente que actualice los datos.
- Se descartó la estación *Almacén EPM*: su serie empieza en diciembre de 2023 y tiene más del 60 % de los días vacíos.
- Son datos agregados de calidad del aire (no hay datos personales). Los documentos del corpus son públicos y se citan por su nombre y página.
- **Documentos del corpus:** Resolución 2254 de 2017 (MinAmbiente); Informe de operación AMVA/SIATA 08-jul-2021 y Informe diario AMVA/SIATA 28-jul-2026 (`siata.gov.co/CalidadAire`); Directrices mundiales de la OMS sobre la calidad del aire 2021, resumen ejecutivo (OPS/OMS).

## Limitaciones

- **El modelo no supera a la línea base** "mañana = hoy".
- **Hay muy pocos días de Alerta** (en 2026, 0 en tres estaciones y 3 en Tráfico Centro), así que el modelo casi no aprende esa clase y su F1 es poco confiable.
- Las probabilidades del Naive Bayes son muy extremas (por ejemplo 99.6 %): tómalas como una estimación, no como certeza.
- Solo se predice **el día siguiente**, y solo con datos de PM2.5 (no se consiguieron lluvia ni viento, por eso no se hizo la red bayesiana).
- Los datos terminan en septiembre de 2026.
- Las coordenadas del mapa son aproximadas.
- Las tablas de la Resolución 2254 se leyeron con OCR y las de PM2.5 se verificaron a mano; el resto del OCR puede tener errores en cifras.
- El agente puede equivocarse, y los modelos de lenguaje cambian entre ejecuciones. Las evaluaciones automáticas no detectan todo, por eso se complementaron con lectura manual.

## Si algo falla

| Problema | Qué hacer |
|---|---|
| El chat dice que falta la clave | Crea el archivo `.env` con `NVIDIA_API_KEY` (paso 2) |
| El agente dice que ningún modelo respondió | Los modelos gratuitos de NVIDIA cambian. Mira cuáles tienes en <https://build.nvidia.com> y ponlos en `.env` como `NVIDIA_MODELOS=modelo1,modelo2` (deben aceptar llamadas a herramientas) |
| La primera vez "se queda pensando" | Está descargando el modelo de embeddings (~470 MB). Espera unos minutos |
| "Puerto ocupado" | Usa `python app.py --puerto 8001` |
| El mapa sale sin fondo | Cambia `ESTILO_MAPA` a `"white-bg"` en `src/dashboard.py` |
| Algo más | Ejecuta `python scripts/smoke_test.py` y mira qué pieza dice `FALLA` |

## Uso de inteligencia artificial generativa

Se usó IA generativa en este proyecto: **Claude (Anthropic)** como guía y asistente de programación, y **modelos de NVIDIA (NIM)** como parte del propio agente. La declaración completa, con herramientas, propósito y lo que hicimos nosotros, está en [`docs/DECLARACION_USO_IA.md`](docs/DECLARACION_USO_IA.md).
