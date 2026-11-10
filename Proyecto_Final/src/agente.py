"""Agente LLM: herramientas, instrucciones y loop (pensar -> usar herramienta -> observar -> responder)."""
import json
import os

from . import config as cfg
from .pipeline import get_sistema
from .rag import buscar_documentos

_CLIENTE = None


# ------------------------------------------------------------------ conexión con NVIDIA NIM
def hay_api_key():
    """¿Está configurada la API key de NVIDIA?"""
    return bool(os.getenv("NVIDIA_API_KEY", "").strip())


def cliente():
    """Cliente de la API de NVIDIA (compatible con OpenAI). Se crea la primera vez que se necesita."""
    global _CLIENTE
    if _CLIENTE is None:
        from openai import OpenAI
        _CLIENTE = OpenAI(base_url=cfg.NVIDIA_BASE_URL, api_key=os.getenv("NVIDIA_API_KEY", ""))
    return _CLIENTE


# ------------------------------------------------------------------ herramientas
def obtener_pm25(estacion, fecha=None):
    """Herramienta 1: PM2.5 medido en una estación."""
    return get_sistema().obtener_pm25(estacion, fecha)


def predecir_manana(estacion, fecha=None):
    """Herramienta 2: categoría prevista para el día siguiente."""
    return get_sistema().predecir_manana(estacion, fecha)


FUNCIONES = {"obtener_pm25": obtener_pm25, "predecir_manana": predecir_manana,
             "buscar_documentos": buscar_documentos}          # herramienta 3 (RAG)

_ESTACIONES = list(cfg.ESTACIONES.values())
HERRAMIENTAS = [          # descripción de las herramientas para el LLM (formato "function calling" de OpenAI)
    {"type": "function", "function": {
        "name": "obtener_pm25",
        "description": "Consulta el PM2.5 diario (µg/m³) medido en una estación de Medellín, su categoría "
                       "(Buena, Moderada o Alerta) y el promedio de los últimos 7 días.",
        "parameters": {"type": "object", "properties": {
            "estacion": {"type": "string", "enum": _ESTACIONES, "description": "Nombre exacto de la estación."},
            "fecha": {"type": "string",
                      "description": "Fecha en formato AAAA-MM-DD. Si se omite, se usa el último día con dato."}},
            "required": ["estacion"]}}},
    {"type": "function", "function": {
        "name": "predecir_manana",
        "description": "Predice la categoría del aire (Buena, Moderada o Alerta) del día siguiente en una "
                       "estación, con probabilidades, usando el modelo Naive Bayes + HMM.",
        "parameters": {"type": "object", "properties": {
            "estacion": {"type": "string", "enum": _ESTACIONES, "description": "Nombre exacto de la estación."},
            "fecha": {"type": "string",
                      "description": "Día base AAAA-MM-DD (se predice el día siguiente). "
                                     "Si se omite, se usa el último día con dato."}},
            "required": ["estacion"]}}},
    {"type": "function", "function": {
        "name": "buscar_documentos",
        "description": "Busca en los documentos de referencia (Resolución 2254 de 2017, informes del AMVA/SIATA y "
                       "directrices de la OMS 2021). Úsala para preguntas sobre normas, niveles de prevención/alerta/"
                       "emergencia, categorías del ICA, episodios de contaminación y recomendaciones de salud.",
        "parameters": {"type": "object", "properties": {
            "consulta": {"type": "string", "description": "Pregunta o palabras clave en español."},
            "k": {"type": "integer", "description": "Cuántos fragmentos devolver (1 a 6). Por defecto 4."}},
            "required": ["consulta"]}}},
]

# ------------------------------------------------------------------ instrucciones del agente
SYSTEM_PROMPT = """Eres un asistente sobre la calidad del aire (PM2.5) en estaciones de Medellín, Valle de Aburrá.
Herramientas:
- obtener_pm25: datos medidos de una estación. predecir_manana: categoría prevista para mañana.
- buscar_documentos: normas, niveles, categorías del ICA, episodios, efectos y recomendaciones de salud.
Reglas:
- Para cualquier dato, predicción o información de normas usa las herramientas; nunca inventes cifras.
- Haz como máximo 2 búsquedas con buscar_documentos por pregunta. Después responde con lo que encontraste.
- Cuando uses buscar_documentos, basa la respuesta SOLO en los fragmentos devueltos y cita la fuente UNA vez al
  final con este formato: (Fuente: nombre del documento, pág. N). Si los fragmentos no responden, dilo.
- Si preguntan qué significa una categoría o qué efectos o recomendaciones de salud tiene, usa buscar_documentos.
  No inventes descripciones de efectos en la salud ni consejos por tu cuenta.
- Si preguntan por medicamentos o consejo médico personal, no lo des: di que no puedes dar consejo médico y
  que consulten a un profesional de la salud. Solo puedes citar lo que digan los documentos.
- Para cifras de la Resolución 2254 prefiere el documento "tablas de PM2.5": las demás páginas de la resolución
  vienen de OCR y pueden tener errores en los números.
- Responde en español, de forma breve y sencilla.
- Categorías del modelo: Buena (<=12 µg/m³), Moderada (12-37) y Alerta (>37). Equivalen al ICA oficial así:
  Buena = Buena; Moderada = Aceptable (13-37); Alerta = Dañina a la salud de grupos sensibles (38-55) o peor.
  'Alerta' del modelo NO es la declaratoria oficial de alerta del Área Metropolitana (esa es 56-150 µg/m³).
- predecir_manana predice solo la CATEGORÍA de mañana, no un valor de PM2.5. Sus probabilidades pueden ser muy
  extremas; preséntalas como una estimación del modelo, no como una certeza.
- Si una herramienta devuelve un error, explícalo al usuario o intenta corregir los parámetros.
- Esto es una herramienta analítica: no es consejo médico, legal ni de seguridad.
- NUNCA escribas una cita (Fuente: ...) si en esta respuesta no usaste buscar_documentos. Nunca escribas "pág. N"
  sin un número real de página.
- Si piden un medicamento o consejo médico personal: solo explica que no puedes darlo y que consulten a un
  profesional de la salud. No cites documentos ni agregues consejos propios.
- predecir_manana solo predice el día siguiente. Si piden más días, dilo y llama la herramienta una sola vez.
- ALCANCE: solo respondes sobre calidad del aire (PM2.5), las estaciones de Medellín, sus datos, las normas, el
  ICA y las recomendaciones de salud relacionadas. Si la pregunta es de otro tema (historia, símbolos patrios,
  matemáticas, recetas, programación, etc.), responde únicamente: "Solo puedo ayudarte con la calidad del aire
  (PM2.5) en Medellín: datos de las estaciones, predicción de mañana, normas y recomendaciones." No uses
  herramientas ni conocimiento general para esos temas."""

MENSAJE_SIN_KEY = ("El chat del agente no está disponible: falta la variable NVIDIA_API_KEY. "
                   "Consigue una gratis en https://build.nvidia.com y configúrala como explica el README. "
                   "El resto del dashboard (mapa, series, predicción) funciona sin ella.")


# ------------------------------------------------------------------ loop del agente
def llamar_llm(mensajes, con_herramientas=True):
    """Llama al LLM; si el modelo principal falla, usa el de respaldo (así la demo no se cae)."""
    ultimo_error = None
    for modelo in cfg.MODELOS_LLM:
        try:
            extra = {"tools": HERRAMIENTAS, "tool_choice": "auto"} if con_herramientas else {}
            return cliente().with_options(timeout=60, max_retries=1).chat.completions.create(
                model=modelo, messages=mensajes, temperature=0.2, max_tokens=1500, **extra)
        except Exception as e:
            ultimo_error = e
            print(f"  (el modelo {modelo} falló: {str(e)[:80]}; probando el siguiente)")
    raise RuntimeError(f"Ningún modelo respondió. Último error: {ultimo_error}")


def ejecutar_herramienta(nombre, argumentos_json):
    """Ejecuta la función pedida por el LLM; si algo falla, devuelve el error como texto (observación)."""
    if nombre not in FUNCIONES:
        return {"error": f"Herramienta desconocida: {nombre}"}
    try:
        args = json.loads(argumentos_json or "{}")           # el LLM manda los argumentos como texto JSON
        return FUNCIONES[nombre](**args)
    except Exception as e:                                    # cualquier error vuelve al LLM, no rompe el programa
        return {"error": f"{type(e).__name__}: {e}"}


def respuesta_sin_herramientas(pregunta, observaciones):
    """Pide al LLM redactar la respuesta final con lo recopilado, en texto simple y sin herramientas."""
    datos = json.dumps(observaciones, ensure_ascii=False)[:6000]
    mensajes = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Pregunta: {pregunta}\n\nResultados de las herramientas (JSON):\n{datos}\n\n"
                 "Responde la pregunta usando SOLO estos resultados. Si usas documentos, cita "
                 "(Fuente: nombre, pág. N). Si los resultados no alcanzan, dilo."}]
    for modelo in cfg.MODELOS_LLM:                            # probar el principal y luego el respaldo
        try:
            r = cliente().with_options(timeout=60, max_retries=1).chat.completions.create(
                model=modelo, messages=mensajes, temperature=0.2, max_tokens=2000)
            texto = (r.choices[0].message.content or "").strip()
            if texto:
                return texto
        except Exception as e:
            print(f"  (el modelo {modelo} falló: {str(e)[:80]})")
    return None


def respuesta_de_emergencia(observaciones):
    """Último recurso: si el LLM no redacta nada, mostrar directamente lo que encontraron las herramientas."""
    docs = [o["resultado"] for o in observaciones
            if o["herramienta"] == "buscar_documentos" and "fragmentos" in o["resultado"]]
    if docs:
        partes = [f"- {f['texto'][:300]}... (Fuente: {f['fuente']}, pág. {f['pagina']})"
                  for f in docs[-1]["fragmentos"][:2]]
        return "No pude redactar una respuesta, pero estos son los fragmentos más relevantes:\n" + "\n".join(partes)
    if observaciones:
        return "No pude redactar una respuesta. Resultado de la consulta: " + str(observaciones[-1]["resultado"])[:300]
    return "No pude generar una respuesta. Intenta reformular la pregunta."


def preguntar_agente(pregunta, historial=None, verbose=True):
    """Loop agéntico con límite de consultas y plan de respaldo. Devuelve (respuesta, historial, herramientas_usadas)."""
    mensajes = historial or [{"role": "system", "content": SYSTEM_PROMPT}]
    if not hay_api_key():                                     # sin key no se puede llamar al LLM: avisar sin romper nada
        return MENSAJE_SIN_KEY, mensajes, []
    mensajes.append({"role": "user", "content": pregunta})
    usadas, vistas, observaciones = [], set(), []

    for vuelta in range(cfg.T_MAX):
        if len(usadas) >= cfg.MAX_CONSULTAS:                  # ya consultó suficiente: pasar a responder
            break
        msg = llamar_llm(mensajes).choices[0].message

        if not msg.tool_calls:                                # el LLM respondió directamente
            texto = (msg.content or "").strip()
            if texto:
                mensajes.append({"role": "assistant", "content": texto})
                return texto, mensajes, usadas
            break                                             # respuesta vacía: ir al plan de respaldo

        mensajes.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            clave = (tc.function.name, tc.function.arguments)
            if clave in vistas:                               # misma herramienta con los mismos argumentos
                resultado = {"aviso": "Ya hiciste esta misma consulta. Responde con lo que tienes."}
            else:
                resultado = ejecutar_herramienta(tc.function.name, tc.function.arguments)
                vistas.add(clave)
                observaciones.append({"herramienta": tc.function.name,
                                      "argumentos": tc.function.arguments, "resultado": resultado})
            usadas.append(tc.function.name)
            if verbose:
                print(f"  [vuelta {vuelta+1}] {tc.function.name}({tc.function.arguments}) -> {str(resultado)[:150]}")
            mensajes.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(resultado, ensure_ascii=False)})

    # Respuesta final: primero el LLM sin herramientas; si también falla, mostrar lo encontrado
    texto = respuesta_sin_herramientas(pregunta, observaciones) or respuesta_de_emergencia(observaciones)
    mensajes.append({"role": "assistant", "content": texto})
    return texto, mensajes, usadas
