"""Dashboard interactivo (Gradio + Plotly): mapa, serie de tiempo, comparación, predicción y chat con el agente."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import config as cfg
from .agente import hay_api_key, preguntar_agente
from .datos import a_categoria, metricas, nombre_categoria
from .pipeline import get_sistema

COLORES = {"Buena": "#4caf50", "Moderada": "#ffc107", "Alerta": "#e53935", "Sin dato": "#9e9e9e"}
ESTILO_MAPA = "carto-positron"      # fondo del mapa (open-street-map bloquea muchas conexiones). Alternativa: "white-bg"
CSS = ".gradio-container {max-width: 1200px !important; margin: auto}"
EJEMPLOS = ["¿Cómo está el aire hoy en Santa Elena?",
            "¿Cómo estará mañana en Tráfico Centro?",
            "¿Cuáles son los niveles de prevención, alerta y emergencia para PM2.5?",
            "¿Qué recomienda la OMS sobre la exposición a PM2.5?"]


# ------------------------------------------------------------------ tarjetas y gráficas
def tarjetas_html(estacion, fecha):
    """Cuatro tarjetas con los datos clave del día: PM2.5, categoría, promedio de 7 días y predicción de mañana."""
    S = get_sistema()
    f = pd.Timestamp(fecha)
    s = S.series[estacion]
    pm = s.loc[f] if f in s.index else np.nan
    prom7 = s.loc[f - pd.Timedelta(days=6):f].mean()
    cat = nombre_categoria(pm)
    pred = S.predecir_manana(estacion, fecha)
    if "error" in pred:
        txt_pred, color_pred, detalle = "Sin predicción", COLORES["Sin dato"], pred["error"]
    else:
        txt_pred, color_pred = pred["categoria_prevista"], COLORES[pred["categoria_prevista"]]
        detalle = " · ".join(f"{k} {v*100:.0f} %" for k, v in pred["probabilidades"].items())

    def tarjeta(titulo, valor, sub, color="#263238"):
        return (f"<div style='flex:1;min-width:180px;padding:14px 16px;border-radius:14px;background:#fff;"
                f"box-shadow:0 2px 8px rgba(0,0,0,.12);border-top:6px solid {color}'>"
                f"<div style='font-size:12px;color:#607d8b;text-transform:uppercase'>{titulo}</div>"
                f"<div style='font-size:28px;font-weight:700;color:{color}'>{valor}</div>"
                f"<div style='font-size:12px;color:#78909c'>{sub}</div></div>")

    valor_pm = "—" if pd.isna(pm) else f"{pm:.1f} µg/m³"
    return ("<div style='display:flex;gap:12px;flex-wrap:wrap'>"
            + tarjeta("PM2.5 del día", valor_pm, f"{estacion} · {fecha}", COLORES[cat])
            + tarjeta("Categoría", cat, "Buena ≤12 · Moderada 12-37 · Alerta >37", COLORES[cat])
            + tarjeta("Promedio 7 días", "—" if pd.isna(prom7) else f"{prom7:.1f} µg/m³", "hasta la fecha elegida")
            + tarjeta("Mañana (modelo)", txt_pred, detalle, color_pred)
            + "</div>")


def fig_mapa(fecha):
    """Mapa de las estaciones: color = categoría del día elegido, tamaño = PM2.5."""
    S = get_sistema()
    f = pd.Timestamp(fecha)
    filas = []
    for est, (lat, lon) in cfg.COORDS.items():
        pm = S.series[est].loc[f] if f in S.series[est].index else np.nan
        filas.append({"Estación": est, "lat": lat, "lon": lon, "PM2.5": pm, "Categoría": nombre_categoria(pm),
                      "tamaño": 14 if pd.isna(pm) else 14 + min(pm, 60) / 2})
    d = pd.DataFrame(filas)
    func = getattr(px, "scatter_map", None) or px.scatter_mapbox     # nombre nuevo o antiguo según la versión
    kw = {"map_style": ESTILO_MAPA} if hasattr(px, "scatter_map") else {"mapbox_style": ESTILO_MAPA}
    fig = func(d, lat="lat", lon="lon", color="Categoría", size="tamaño", color_discrete_map=COLORES,
               hover_name="Estación", hover_data={"PM2.5": ":.1f", "lat": False, "lon": False, "tamaño": False},
               zoom=10.3, center={"lat": 6.245, "lon": -75.56}, height=460, **kw)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), title=f"Estaciones el {fecha}")
    return fig


def fig_serie(estacion, fecha, ventana=180):
    """Serie de tiempo del PM2.5 con los umbrales y los estados ocultos del HMM."""
    S = get_sistema()
    f = pd.Timestamp(fecha)
    s = S.series[estacion].loc[f - pd.Timedelta(days=int(ventana)):f]
    est_hmm = S.estados_hmm[estacion].reindex(s.index)               # estado oculto de cada día
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="PM2.5", line=dict(color="#546e7a", width=1.5)))
    colores_estado = ["#4caf50", "#ffc107", "#e53935"]
    for k in range(3):
        m = est_hmm == k
        fig.add_trace(go.Scatter(x=s.index[m], y=s.values[m], mode="markers", name=f"Estado HMM {k}",
                                 marker=dict(color=colores_estado[k], size=5)))
    fig.add_hline(y=12, line_dash="dash", line_color="gray")        # umbral Buena / Moderada
    fig.add_hline(y=37, line_dash="dash", line_color="red")         # umbral Moderada / Alerta
    fig.add_shape(type="line", x0=str(f.date()), x1=str(f.date()), y0=0, y1=1, yref="paper",
                  line=dict(color="black", width=1, dash="dot"))     # línea en la fecha elegida
    fig.update_layout(title=f"{estacion}: PM2.5 diario", yaxis_title="µg/m³", height=420,
                      margin=dict(l=40, r=20, t=50, b=30), legend=dict(orientation="h", y=-0.2))
    return fig


def fig_comparacion(fecha):
    """Barras con el PM2.5 de todas las estaciones en la fecha elegida."""
    S = get_sistema()
    f = pd.Timestamp(fecha)
    d = pd.DataFrame([{"Estación": e, "PM2.5": (S.series[e].loc[f] if f in S.series[e].index else np.nan)}
                      for e in S.series])
    d["Categoría"] = d["PM2.5"].apply(nombre_categoria)
    fig = px.bar(d, x="Estación", y="PM2.5", color="Categoría", color_discrete_map=COLORES,
                 text=d["PM2.5"].round(1), height=420)
    fig.add_hline(y=12, line_dash="dash", line_color="gray")
    fig.add_hline(y=37, line_dash="dash", line_color="red")
    fig.update_layout(title=f"Comparación entre estaciones el {fecha}", yaxis_title="µg/m³",
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


def fig_prediccion(estacion, fecha):
    """Barras con la probabilidad de cada categoría para mañana (según el modelo Naive Bayes + HMM)."""
    p = get_sistema().predecir_manana(estacion, fecha)
    fig = go.Figure()
    if "error" in p:                                                 # sin predicción posible: mostrar el motivo
        fig.add_annotation(text=p["error"], showarrow=False, font=dict(size=14))
        fig.update_layout(height=380, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    probs = p["probabilidades"]
    fig.add_trace(go.Bar(x=list(probs), y=[v * 100 for v in probs.values()],
                         marker_color=[COLORES[k] for k in probs],
                         text=[f"{v * 100:.1f} %" for v in probs.values()], textposition="outside"))
    fig.update_layout(title=f"{estacion}: predicción para el {p['fecha_prediccion']} (hoy: {p['categoria_hoy']})",
                      yaxis=dict(range=[0, 115], title="Probabilidad (%)"), height=380,
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


def tabla_predicciones(fecha):
    """Tabla con la predicción de mañana para las estaciones."""
    S = get_sistema()
    filas = []
    for est in S.series:
        p = S.predecir_manana(est, fecha)
        if "error" in p:
            filas.append({"Estación": est, "PM2.5 del día": None, "Categoría hoy": "—",
                          "Predicción mañana": "Sin predicción", "Buena %": None, "Moderada %": None, "Alerta %": None})
        else:
            pr = p["probabilidades"]
            filas.append({"Estación": est, "PM2.5 del día": p["pm25_hoy"], "Categoría hoy": p["categoria_hoy"],
                          "Predicción mañana": p["categoria_prevista"], "Buena %": round(pr["Buena"] * 100, 1),
                          "Moderada %": round(pr["Moderada"] * 100, 1), "Alerta %": round(pr["Alerta"] * 100, 1)})
    return pd.DataFrame(filas)


def tabla_metricas():
    """Desempeño en 2026 de la línea base y del modelo final (promedio de las estaciones)."""
    S = get_sistema()
    filas = []
    for est, d in S.datos.items():
        test = d["test"]
        pred_modelo = S.modelos_nb[est].predict(test[cfg.PISTAS_FINAL])
        for nombre, pred in [("Línea base (mañana = hoy)", test["cat_hoy"]), ("Naive Bayes + HMM", pred_modelo)]:
            m = metricas(test["y"], pred)
            filas.append({"Modelo": nombre, "Accuracy": m["accuracy"], "F1 macro (clases presentes)": m["f1_macro_presentes"]})
    return pd.DataFrame(filas).groupby("Modelo", sort=False).mean().round(3).reset_index()


# ------------------------------------------------------------------ interfaz
def actualizar(estacion, fecha, ventana):
    """Se ejecuta al cambiar un selector: recalcula tarjetas, gráficas y tabla de predicciones."""
    return (tarjetas_html(estacion, fecha), fig_mapa(fecha), fig_serie(estacion, fecha, ventana),
            fig_comparacion(fecha), fig_prediccion(estacion, fecha), tabla_predicciones(fecha))


def chatear(mensaje, chat, historial):
    """Envía la pregunta al agente y agrega la respuesta al chat. 'historial' conserva el contexto del agente."""
    mensaje = (mensaje or "").strip()
    if not mensaje:                                                  # no hacer nada si el cuadro está vacío
        return "", chat, historial
    if historial is not None and len(historial) > 30:               # conversación muy larga: empezar de nuevo
        historial = None
    try:
        respuesta, historial, usadas = preguntar_agente(mensaje, historial=historial, verbose=False)
    except Exception as e:                                           # que un error no rompa el dashboard
        respuesta, usadas = f"Ocurrió un error al consultar al agente: {e}", []
    herramientas = ", ".join(dict.fromkeys(usadas))                  # herramientas usadas, sin repetir
    if herramientas:
        respuesta += f"\n\n*Herramientas usadas: {herramientas}*"
    chat = (chat or []) + [{"role": "user", "content": mensaje}, {"role": "assistant", "content": respuesta}]
    return "", chat, historial


def crear_interfaz():
    """Construye la interfaz de Gradio (todavía sin abrirla)."""
    import gradio as gr
    S = get_sistema()
    ultima = S.ultima_fecha
    opciones_fecha = [str(d.date()) for d in pd.date_range(end=ultima, periods=365)][::-1]
    estaciones = list(S.series)

    with gr.Blocks(title="Calidad del aire - Valle de Aburrá") as demo:
        gr.Markdown("# 🌫️ Calidad del aire en Medellín (PM2.5)\n"
                    f"Datos de SISAIRE hasta {ultima.date()} · Modelo Naive Bayes + HMM · "
                    "Agente con RAG sobre normas y guías de salud")
        if not hay_api_key():
            gr.Markdown("⚠️ **El chat está desactivado**: falta la variable `NVIDIA_API_KEY` (mira el README). "
                        "Lo demás funciona con normalidad.")

        # --- Selectores: estación, fecha base y cuántos días mostrar en la serie ---
        with gr.Row():
            dd_est = gr.Dropdown(estaciones, value=estaciones[0], label="Estación")
            dd_fecha = gr.Dropdown(opciones_fecha, value=opciones_fecha[0],
                                   label="Fecha base (se predice el día siguiente)")
            sl_vent = gr.Slider(30, 365, value=180, step=15, label="Días de la serie de tiempo")

        tarjetas = gr.HTML()
        with gr.Tabs():
            with gr.Tab("🗺️ Mapa"):
                plot_mapa = gr.Plot()
            with gr.Tab("📈 Serie de tiempo"):
                plot_serie = gr.Plot()
            with gr.Tab("📊 Comparación"):
                plot_comp = gr.Plot()
            with gr.Tab("🔮 Predicción"):
                plot_pred = gr.Plot()
                gr.Markdown("**Predicción de mañana en las estaciones** (a partir de la fecha elegida)")
                tabla_pred = gr.Dataframe(interactive=False)
                gr.Markdown("**Desempeño del modelo en 2026 frente a la línea base** (promedio de las estaciones)")
                gr.Dataframe(value=tabla_metricas(), interactive=False)
                gr.Markdown("*Las probabilidades del Naive Bayes suelen ser muy extremas; tómalas como una estimación.*")
            with gr.Tab("💬 Agente"):
                chatbot = gr.Chatbot(height=420, label="Agente de calidad del aire")
                caja = gr.Textbox(placeholder="Escribe tu pregunta y presiona Enter", label="Tu pregunta")
                gr.Examples(examples=[[e] for e in EJEMPLOS], inputs=caja, label="Ejemplos")
                estado = gr.State(None)                              # memoria de la conversación del agente
                limpiar = gr.Button("🧹 Limpiar conversación")

        gr.Markdown("⚠️ *Herramienta analítica y académica: no es consejo médico, legal ni de seguridad. "
                    "La categoría 'Alerta' es del modelo (>37 µg/m³) y no es la declaratoria oficial del AMVA. "
                    "El modelo predice solo la categoría del día siguiente y su desempeño es similar al de la línea base "
                    "'mañana = hoy'.*")

        # --- Conexiones: qué función se ejecuta con cada acción del usuario ---
        entradas = [dd_est, dd_fecha, sl_vent]
        salidas = [tarjetas, plot_mapa, plot_serie, plot_comp, plot_pred, tabla_pred]
        dd_est.change(actualizar, entradas, salidas)
        dd_fecha.change(actualizar, entradas, salidas)
        sl_vent.release(actualizar, entradas, salidas)
        demo.load(actualizar, entradas, salidas)                     # al abrir la página, dibujar todo
        caja.submit(chatear, [caja, chatbot, estado], [caja, chatbot, estado])
        limpiar.click(lambda: ([], None), None, [chatbot, estado])
    return demo


def lanzar(share=False, puerto=7860, abrir_navegador=True):
    """Abre el dashboard. Por defecto en http://127.0.0.1:7860 (solo en este computador)."""
    import gradio as gr
    demo = crear_interfaz()
    demo.launch(server_name="127.0.0.1", server_port=puerto, share=share, inbrowser=abrir_navegador,
                theme=gr.themes.Soft(primary_hue="green"), css=CSS)
