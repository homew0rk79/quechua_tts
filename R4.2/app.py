"""R4.2 — Evaluación humana de transcripciones automáticas (quechua de Abancay).

La app NO ejecuta el modelo: muestra transcripciones precalculadas (items.csv) junto a su audio.

Variables de entorno (en Hugging Face Spaces van como "Secrets"):
  CODIGOS          códigos de acceso, formato "E1:clave1,E2:clave2,E3:clave3"
  HF_TOKEN         token con lectura de REPO_AUDIOS y escritura en REPO_RESPUESTAS
  REPO_AUDIOS      dataset privado con items.csv y la carpeta audios/   (si falta: modo local)
  REPO_RESPUESTAS  dataset privado donde se guardan las respuestas       (si falta: solo disco local)

Modo local (para probar): python app.py  → lee ./items/items.csv y ./audios/, guarda en ./respuestas/
"""
import hashlib, html, json, os, random, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from response_store import ResponseStore

import gradio as gr
import pandas as pd

# ------------------------------------------------------------------ textos a confirmar con el especialista
TITULO = "Evaluación de transcripciones automáticas en quechua de Abancay"

CONSENTIMIENTO = """
**Antes de empezar.** Esta evaluación forma parte de una tesis de pregrado de la Pontificia Universidad
Católica del Perú. Escuchará audios breves en quechua y calificará transcripciones escritas.
Sus respuestas se guardan con un código, sin su nombre, y solo se usarán con fines académicos.
Puede dejar la evaluación en cualquier momento.

*[REEMPLAZAR por el texto de consentimiento aprobado por el asesor o el comité de ética.]*
"""

# Regla clave: si no se define, cada evaluador aplicará su propio criterio.
INSTRUCCION_ORTOGRAFIA = (
    "Califique si el texto transmite lo que se dice en el audio. Las diferencias que son **solo de "
    "escritura** (por ejemplo, *q/k*, *h/j*, *e/i*, *o/u*) no deben bajar su calificación de inteligibilidad; "
    "márquelas en la pregunta 3. *[CONFIRMAR esta regla con el especialista.]*"
)

INSTRUCCIONES = f"""
### Cómo funciona

En cada pantalla encontrará un audio y una transcripción escrita. Algunas transcripciones fueron hechas por
un sistema automático y otras por personas; no se indica cuál es cuál.

1. Escuche el audio. Puede repetirlo las veces que necesite.
2. Lea la transcripción y responda las tres preguntas.
3. Si lo desea, corrija el texto en el recuadro final para que diga exactamente lo que se escucha.
   Si no cambia nada, se entiende que la transcripción le parece correcta.

{INSTRUCCION_ORTOGRAFIA}

Las dos primeras pantallas son de práctica y no se cuentan. La evaluación completa toma unos 40 minutos.
Sus respuestas se guardan en cada pantalla: si se interrumpe, puede volver a entrar con su código y
continuar donde se quedó.
"""

P1 = ("1. Leyendo solo este texto, ¿se entiende lo que dice el audio?",
      ["1 · No se entiende nada", "2 · Se entienden palabras sueltas", "3 · Se entiende la idea general con esfuerzo",
       "4 · Se entiende casi todo", "5 · Se entiende por completo"])
P2 = ("2. Si tuviera que transcribir este audio, este texto le serviría como…",
      ["1 · Me estorba, prefiero empezar de cero", "2 · Ayuda muy poco", "3 · Ahorra algo de trabajo",
       "4 · Solo necesita correcciones menores", "5 · Lo usaría tal cual"])
P3 = ("3. ¿Cuál es el tipo de error principal?",
      ["Ninguno", "Solo de escritura (la palabra es correcta, pero está mal escrita)",
       "Cambian el significado", "Faltan o sobran palabras"])
G1 = ("En general, ¿qué tan útiles le parecieron las transcripciones para su trabajo?",
      ["1 · Nada útiles", "2 · Poco útiles", "3 · Algo útiles", "4 · Bastante útiles", "5 · Muy útiles"])

# ------------------------------------------------------------------ datos
BASE = Path(__file__).parent
EN_RENDER = os.environ.get("RENDER") == "true"
if EN_RENDER:
    faltantes = [k for k in ("CODIGOS", "HF_TOKEN", "REPO_AUDIOS", "REPO_RESPUESTAS")
                 if not os.environ.get(k, "").strip()]
    if faltantes:
        raise RuntimeError("Faltan variables de Render: " + ", ".join(faltantes))
CODIGOS = dict(p.split(":", 1) for p in os.environ.get("CODIGOS", "E1:prueba1,E2:prueba2,E3:prueba3").split(","))
if any(not e or not c.strip() for e, c in CODIGOS.items()) or len(set(CODIGOS.values())) != len(CODIGOS):
    raise RuntimeError("Los códigos deben ser únicos y no estar vacíos.")
if EN_RENDER and any(c in {"prueba1", "prueba2", "prueba3"} for c in CODIGOS.values()):
    raise RuntimeError("Reemplaza los códigos de prueba antes de publicar en Render.")
TOKEN = os.environ.get("HF_TOKEN")
REPO_AUDIOS = os.environ.get("REPO_AUDIOS")
REPO_RESPUESTAS = os.environ.get("REPO_RESPUESTAS")
DIR_RESP = BASE / "respuestas"
DIR_RESP.mkdir(exist_ok=True)

if REPO_AUDIOS:
    from huggingface_hub import snapshot_download
    DIR_DATOS = Path(snapshot_download(REPO_AUDIOS, repo_type="dataset", token=TOKEN,
                                    local_dir=BASE / "datos",
                                    allow_patterns=["items.csv", "audios/*.wav"]))
    ITEMS_CSV, DIR_AUDIO = DIR_DATOS / "items.csv", DIR_DATOS / "audios"
else:
    ITEMS_CSV, DIR_AUDIO = BASE / "items" / "items.csv", BASE / "audios"

store = ResponseStore(DIR_RESP, repo_id=REPO_RESPUESTAS, token=TOKEN)
store.restore()

ITEMS = pd.read_csv(ITEMS_CSV).fillna("")
ITEMS = {r.item_id: r for r in ITEMS.itertuples()}
PRACTICA = [i for i, r in ITEMS.items() if r.tipo == "practica"]
EVALUABLES = [i for i, r in ITEMS.items() if r.tipo != "practica"]


def guardar(registro, archivo):
    try:
        store.save(registro, archivo)
    except Exception:
        raise gr.Error("No se pudo confirmar el guardado. Tus respuestas siguen en esta pantalla; "
                       "pulsa de nuevo para reintentar.") from None


def historial(evaluador):
    """Respuestas previas del evaluador (de cualquier sesión), para retomar."""
    regs = []
    for p in DIR_RESP.glob(f"{evaluador}_*.jsonl"):
        with open(p, encoding="utf-8") as f:
            regs += [json.loads(l) for l in f if l.strip()]
    return regs


def orden_items(evaluador):
    """Mismo orden aleatorio en cada ingreso del mismo evaluador; distinto entre evaluadores."""
    semilla = int(hashlib.sha256(evaluador.encode()).hexdigest(), 16) % 2**32
    resto = EVALUABLES.copy()
    random.Random(semilla).shuffle(resto)
    return PRACTICA + resto


def num(opcion):
    return int(opcion.split(" ·")[0]) if opcion else None


# ------------------------------------------------------------------ pantallas
def vista(nombre):
    """Visibilidad de las cinco pantallas, en el orden en que se declaran abajo."""
    return [gr.Column(visible=nombre == n) for n in ["ingreso", "perfil", "item", "cierre", "fin"]]


def mostrar_item(est):
    iid = est["orden"][est["idx"]]
    it = ITEMS[iid]
    n_prac, n_total = len(PRACTICA), len(EVALUABLES)
    if est["idx"] < n_prac:
        cab = f"### Práctica {est['idx'] + 1} de {n_prac}\nEsta pantalla no se cuenta en los resultados."
    else:
        cab = f"### Audio {est['idx'] - n_prac + 1} de {n_total}"
    est["inicio"] = time.time()
    audio = DIR_AUDIO / it.audio_path
    return (est, cab, str(audio) if audio.exists() else None, f'<p class="transcripcion">{html.escape(it.texto_mostrado)}</p>',
            None, None, None, it.texto_mostrado, "")


SALIDA_ITEM_VACIA = (gr.skip(),) * 8


def ingresar(codigo, est):
    evaluador = next((e for e, c in CODIGOS.items() if c == codigo.strip()), None)
    if not evaluador:
        return (est, "El código no es válido. Revíselo e intente de nuevo.", *vista("ingreso"), *SALIDA_ITEM_VACIA)
    regs = historial(evaluador)
    hechos = {r["item_id"] for r in regs if r["etapa"] == "item"}
    est = {"evaluador": evaluador, "archivo": str(DIR_RESP / f"{evaluador}_{uuid.uuid4().hex[:8]}.jsonl"),
           "orden": orden_items(evaluador)}
    if any(r["etapa"] == "cierre" for r in regs):
        return (est, "", *vista("fin"), *SALIDA_ITEM_VACIA)
    if not any(r["etapa"] == "perfil" for r in regs):
        return (est, "", *vista("perfil"), *SALIDA_ITEM_VACIA)
    pend = [k for k, i in enumerate(est["orden"]) if i not in hechos]
    if not pend:
        return (est, "", *vista("cierre"), *SALIDA_ITEM_VACIA)
    est["idx"] = pend[0]
    return (est, "", *vista("item"), *mostrar_item(est)[1:])


def guardar_perfil(acepta, variedad, experiencia, ortografia, est):
    if not acepta:
        return (est, "Para continuar, marque que acepta participar.", *vista("perfil"), *SALIDA_ITEM_VACIA)
    if not (variedad and experiencia and ortografia):
        return (est, "Responda las tres preguntas para continuar.", *vista("perfil"), *SALIDA_ITEM_VACIA)
    guardar({"etapa": "perfil", "evaluador": est["evaluador"], "variedad": variedad,
             "experiencia_escritura": experiencia, "ortografia": ortografia}, est["archivo"])
    est["idx"] = 0
    return (est, "", *vista("item"), *mostrar_item(est)[1:])


def guardar_item(p1, p2, p3, correccion, est):
    if not (p1 and p2 and p3):
        return (est, *vista("item"), *SALIDA_ITEM_VACIA[:7], "Responda las tres preguntas antes de seguir.")
    it = ITEMS[est["orden"][est["idx"]]]
    corr = " ".join(correccion.split())
    guardar({"etapa": "item", "evaluador": est["evaluador"], "item_id": it.item_id, "inteligibilidad": num(p1),
             "utilidad": num(p2), "tipo_error": p3, "correccion": corr,
             "corregido": corr != " ".join(it.texto_mostrado.split()),
             "segundos": round(time.time() - est.get("inicio", time.time()), 1)}, est["archivo"])
    hechos = {r["item_id"] for r in historial(est["evaluador"]) if r["etapa"] == "item"}
    pend = [k for k, i in enumerate(est["orden"]) if i not in hechos]
    if not pend:
        return (est, *vista("cierre"), *SALIDA_ITEM_VACIA[:7], "")
    est["idx"] = pend[0]
    s = mostrar_item(est)
    return (s[0], *vista("item"), *s[1:])


def guardar_cierre(g1, errores, comentarios, est):
    if not g1:
        return (est, "Responda la pregunta de utilidad general.", *vista("cierre"))
    guardar({"etapa": "cierre", "evaluador": est["evaluador"], "utilidad_general": num(g1),
             "errores_frecuentes": errores.strip(), "comentarios": comentarios.strip()}, est["archivo"])
    return (est, "", *vista("fin"))


# ------------------------------------------------------------------ interfaz
TEMA = gr.themes.Base(
    primary_hue=gr.themes.Color(c50="#eef3f1", c100="#d5e3de", c200="#abc7bd", c300="#80aa9b", c400="#568e7a",
                                c500="#2f6f5b", c600="#265a4a", c700="#1e4639", c800="#153229", c900="#0d1f19",
                                c950="#08130f"),
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Atkinson Hyperlegible"), "system-ui", "sans-serif"],
    radius_size="sm",
)
CSS = """
.gradio-container { max-width: 760px !important; margin: 0 auto; }
.transcripcion { font-size: 1.35rem; line-height: 1.7; padding: 1rem 1.25rem; margin: 0;
                 border-left: 4px solid var(--color-accent); background: var(--block-background-fill); }
.aviso { color: #a3361b; font-weight: 700; }
"""

with gr.Blocks(title=TITULO) as demo:
    est = gr.State({})
    gr.Markdown(f"## {TITULO}")

    with gr.Column(visible=True) as s_ingreso:
        gr.Markdown("Ingrese el código que recibió por correo.")
        codigo = gr.Textbox(label="Código de acceso", type="password", max_lines=1)
        b_ingresar = gr.Button("Ingresar", variant="primary")
        msg_ingreso = gr.Markdown(elem_classes="aviso")

    with gr.Column(visible=False) as s_perfil:
        gr.Markdown(CONSENTIMIENTO)
        acepta = gr.Checkbox(label="Leí la información anterior y acepto participar")
        gr.Markdown("### Sobre usted\nEstas preguntas ayudan a interpretar las calificaciones. No se pide su nombre.")
        variedad = gr.Radio(["Quechua de Abancay (Apurímac)", "Otra variedad chanka", "Quechua collao (Cusco)",
                             "Otra"], label="¿Qué variedad de quechua habla?")
        experiencia = gr.Radio(["Menos de 2 años", "De 2 a 5 años", "De 6 a 10 años", "Más de 10 años"],
                               label="¿Cuántos años tiene leyendo o escribiendo quechua?")
        ortografia = gr.Radio(["Alfabeto oficial de 3 vocales (a, i, u)", "5 vocales (a, e, i, o, u)",
                               "Uso ambos", "Otra"], label="¿Qué escritura usa habitualmente?")
        gr.Markdown(INSTRUCCIONES)
        b_perfil = gr.Button("Empezar", variant="primary")
        msg_perfil = gr.Markdown(elem_classes="aviso")

    with gr.Column(visible=False) as s_item:
        cab = gr.Markdown()
        audio = gr.Audio(label="Audio", interactive=False, type="filepath", buttons=[])
        gr.Markdown("**Transcripción**")
        texto = gr.HTML()
        p1 = gr.Radio(P1[1], label=P1[0])
        p2 = gr.Radio(P2[1], label=P2[0])
        p3 = gr.Radio(P3[1], label=P3[0])
        corr = gr.Textbox(label="4. Corrección (opcional): edite el texto para que diga exactamente lo que se escucha",
                          lines=3)
        b_item = gr.Button("Guardar y seguir", variant="primary")
        msg_item = gr.Markdown(elem_classes="aviso")

    with gr.Column(visible=False) as s_cierre:
        gr.Markdown("### Última pantalla")
        g1 = gr.Radio(G1[1], label=G1[0])
        errores = gr.Textbox(label="¿Qué errores vio repetirse con más frecuencia?", lines=3)
        comentarios = gr.Textbox(label="Comentarios (opcional)", lines=3)
        b_cierre = gr.Button("Terminar", variant="primary")
        msg_cierre = gr.Markdown(elem_classes="aviso")

    with gr.Column(visible=False) as s_fin:
        gr.Markdown("### Gracias\nSus respuestas quedaron guardadas. Ya puede cerrar esta ventana.")

    pantallas = [s_ingreso, s_perfil, s_item, s_cierre, s_fin]
    campos_item = [cab, audio, texto, p1, p2, p3, corr, msg_item]

    b_ingresar.click(ingresar, [codigo, est], [est, msg_ingreso, *pantallas, *campos_item])
    codigo.submit(ingresar, [codigo, est], [est, msg_ingreso, *pantallas, *campos_item])
    b_perfil.click(guardar_perfil, [acepta, variedad, experiencia, ortografia, est],
                   [est, msg_perfil, *pantallas, *campos_item])
    b_item.click(guardar_item, [p1, p2, p3, corr, est], [est, *pantallas, *campos_item]).then(
        None, js="() => window.scrollTo({top: 0})")
    b_cierre.click(guardar_cierre, [g1, errores, comentarios, est], [est, msg_cierre, *pantallas])

if __name__ == "__main__":
    demo.launch(theme=TEMA, css=CSS, allowed_paths=[str(DIR_AUDIO)],
                server_name="0.0.0.0" if EN_RENDER or os.environ.get("SPACE_ID") else "127.0.0.1",
                server_port=int(os.environ.get("PORT", "7860")))
