import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
from groq import Groq

st.set_page_config(page_title="Mantenimiento Predictivo", page_icon="🏭", layout="wide")

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_mantenimiento.pkl")

modelo = cargar_modelo()

NOMBRES = {0: "Operación Normal", 1: "Mantenimiento Preventivo", 2: "Fallo Crítico"}

st.title("🏭 Sistema Predictivo de Mantenimiento Industrial")

# ---------- Controles laterales ----------
st.sidebar.header("Lecturas de sensores")
temperatura = st.sidebar.slider("Temperatura (°C)", 40.0, 120.0, 60.0, 0.5)
vibracion = st.sidebar.slider("Vibración (mm/s)", 0.0, 12.0, 1.5, 0.1)
horas = st.sidebar.number_input("Horas de operación", min_value=0, max_value=5000, value=300, step=50)
presion = st.sidebar.number_input("Presión (PSI)", min_value=80.0, max_value=160.0, value=92.0, step=0.5)

# ---------- Tarjetas ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperatura", f"{temperatura} °C")
col2.metric("Vibración", f"{vibracion} mm/s")
col3.metric("Operación", f"{horas} hrs")
col4.metric("Presión", f"{presion} PSI")

# ---------- Predicción ----------
entrada = pd.DataFrame(
    [[temperatura, vibracion, horas, presion]],
    columns=["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"],
)
estado = int(modelo.predict(entrada)[0])
probas = modelo.predict_proba(entrada)[0]
prob_estado = probas[list(modelo.classes_).index(estado)]

st.subheader("Diagnóstico del modelo")
mensaje = f"Estado {estado}: {NOMBRES[estado]} (probabilidad: {prob_estado * 100:.1f}%)"
if estado == 0:
    st.success(mensaje)
elif estado == 1:
    st.warning(mensaje)
else:
    st.error(mensaje)

with st.expander("Ver probabilidades de todas las clases"):
    for clase, p in zip(modelo.classes_, probas):
        st.write(f"**{NOMBRES[int(clase)]}**: {p * 100:.1f}%")

# ---------- Groq ----------
def generar_orden(estado, folio):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    # Detección dinámica: solo modelos de chat (sin audio, filtros de seguridad ni árabe)
    excluir = ["whisper", "guard", "tts", "playai", "orpheus", "embed", "allam", "qwen"]
    modelos_activos = [
        m.id for m in client.models.list().data
        if not any(x in m.id.lower() for x in excluir)
    ]
    preferidos = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    candidatos = [m for m in preferidos if m in modelos_activos]
    candidatos += [m for m in modelos_activos if m not in candidatos]

    system_prompt = (
        "Eres un Jefe de Mantenimiento Industrial con 20 años de experiencia. "
        "Responde SIEMPRE y SOLO en español, usando únicamente alfabeto latino. "
        "Sé técnico, claro y conciso. No inventes problemas que las lecturas no indican."
    )
    datos = (
        f"Temperatura: {temperatura} °C, Vibración: {vibracion} mm/s, "
        f"Horas de operación: {horas}, Presión: {presion} PSI."
    )
    if estado == 0:
        user_prompt = (
            f"Lecturas: {datos}\n"
            "El modelo de ML clasificó el equipo como OPERACIÓN NORMAL: todos los parámetros "
            "están dentro de rango estable. Redacta un dictamen breve de monitoreo preventivo "
            "en un solo párrafo de máximo 3 líneas. No uses listas ni sugieras fallas."
        )
    else:
        urgencia = "preventiva" if estado == 1 else "URGENTE con protocolo de paro inmediato"
        user_prompt = (
            f"Lecturas: {datos}\nEstado clasificado: {NOMBRES[estado]}.\n"
            f"Genera una Orden de Trabajo {urgencia} en formato Markdown con exactamente "
            "estas secciones:\n"
            f"- **Folio:** {folio}\n"
            "- **Diagnóstico**\n"
            "- **Acciones Técnicas** (exactamente 3, numeradas)\n"
            "- **Equipo de Seguridad (EPP)**"
        )

    ultimo_error = None
    for modelo_groq in candidatos:
        try:
            respuesta = client.chat.completions.create(
                model=modelo_groq,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_completion_tokens=1500,
                reasoning_effort="low",
            )
            texto = respuesta.choices[0].message.content
            if texto:
                st.caption(f"Modelo de Groq utilizado: {modelo_groq}")
                return texto
        except Exception as e:
            ultimo_error = e
    raise ultimo_error or RuntimeError("Sin modelos de chat disponibles")

st.subheader("Orden de Trabajo (IA Generativa)")
if st.button("📝 Generar Orden de Trabajo"):
    folio = f"OT-{datetime.now():%Y%m%d-%H%M%S}"
    try:
        with st.spinner("Consultando a Groq..."):
            st.markdown(generar_orden(estado, folio))
    except Exception:
        st.info(
            "⚠️ No se pudo generar la Orden con IA (revisa la API Key o el límite de peticiones). "
            f"Diagnóstico del modelo ML: **{NOMBRES[estado]}** ({prob_estado * 100:.1f}%)."
        )