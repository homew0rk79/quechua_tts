"""R4.2 — Análisis de la evaluación humana y aplicación de la regla de decisión.

La regla está fijada en las constantes de abajo y debe quedar congelada ANTES de aplicar el cuestionario
(registrar la fecha en la bitácora). Cambiarla después de ver los datos invalida el veredicto.

Uso:
  python analizar_r42.py --respuestas respuestas/ --items items/items.csv --estratos items/estratos.json --out R4.2

Salidas en --out: resumen.json, por_item.csv, por_estrato.csv, post_edicion.csv
"""
import argparse, glob, json, os
import jiwer, krippendorff, numpy as np, pandas as pd
from scipy.stats import spearmanr

# ------------------------------------------------------------------ regla de decisión (congelar antes de aplicar)
UMBRAL_ACEPTABLE = 4         # un ítem es aceptable si la mediana de utilidad de los evaluadores es >= 4
SATISF_PCT, SATISF_MED = 60, 4
INSUF_PCT, INSUF_MED = 30, 2
CONTROL_MIN = 4              # validez: la mediana de utilidad de los controles debe ser >= 4
ALFA_MIN = 0.4               # validez: concordancia mínima (Krippendorff ordinal)

ap = argparse.ArgumentParser()
ap.add_argument("--respuestas", default="respuestas")
ap.add_argument("--items", default="items/items.csv")
ap.add_argument("--estratos", default="items/estratos.json")
ap.add_argument("--out", default="R4.2")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

regs = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.respuestas, "*.jsonl"))) for l in open(f, encoding="utf-8")
        if l.strip()]
items = pd.read_csv(a.items).fillna("")
pob = json.load(open(a.estratos, encoding="utf-8"))["poblacion_limpia"]

r = pd.DataFrame([x for x in regs if x["etapa"] == "item"])
r = r.sort_values("marca_tiempo").drop_duplicates(["evaluador", "item_id"], keep="last")  # última respuesta vale
r = r.merge(items, on="item_id")
r = r[r.tipo != "practica"].copy()
evaluadores = sorted(r.evaluador.unique())
print(f"Evaluadores: {evaluadores} | respuestas analizables: {len(r)}")
completos = r.groupby("evaluador").item_id.nunique()
if (completos < items[items.tipo != "practica"].shape[0]).any():
    print("AVISO: evaluación incompleta:\n" + completos.to_string())

# ------------------------------------------------------------------ por ítem
pi = r.groupby(["item_id", "tipo", "estrato", "fuente", "wer_item"]).agg(
    n_eval=("evaluador", "nunique"), inteligibilidad_med=("inteligibilidad", "median"),
    utilidad_med=("utilidad", "median"), utilidad_media=("utilidad", "mean"),
    corregido_por=("corregido", "sum"), segundos_med=("segundos", "median")).reset_index()
pi["aceptable"] = pi.utilidad_med >= UMBRAL_ACEPTABLE
pi.to_csv(os.path.join(a.out, "por_item.csv"), index=False)
mod, ctl = pi[pi.tipo == "modelo"], pi[pi.tipo == "control"]

# ------------------------------------------------------------------ por estrato y ponderado
pe = mod.groupby("estrato").agg(n=("item_id", "size"), pct_aceptables=("aceptable", "mean"),
                                utilidad_med=("utilidad_med", "median"),
                                inteligibilidad_med=("inteligibilidad_med", "median"),
                                wer_medio=("wer_item", "mean")).reindex(["bajo", "medio", "alto"])
pe["pct_aceptables"] *= 100
pe["peso_poblacion"] = [pob[e] / sum(pob.values()) for e in pe.index]
pe.round(2).to_csv(os.path.join(a.out, "por_estrato.csv"))
pct_ponderado = float((pe.pct_aceptables * pe.peso_poblacion).sum())
pct_simple = float(100 * mod.aceptable.mean())

# ------------------------------------------------------------------ post-edición (solo ítems del modelo)
rm = r[r.tipo == "modelo"]
pe_rows = []
for ev, sub in rm.groupby("evaluador"):
    o = jiwer.process_words(list(sub.texto_mostrado), list(sub.correccion.replace("", " ")))
    n = o.substitutions + o.deletions + o.hits
    pe_rows.append([ev, len(sub), int(sub.corregido.sum()),
                    100 * (o.substitutions + o.deletions + o.insertions) / n])
post = pd.DataFrame(pe_rows, columns=["evaluador", "items_modelo", "items_corregidos", "WER_post_edicion"]).round(2)
post.to_csv(os.path.join(a.out, "post_edicion.csv"), index=False)
o = jiwer.process_words(list(mod.merge(items, on="item_id").ref), list(mod.merge(items, on="item_id").hyp))
wer_auto = 100 * (o.substitutions + o.deletions + o.insertions) / (o.substitutions + o.deletions + o.hits)

# ------------------------------------------------------------------ concordancia y relación con el WER
def alfa(col):
    m = r.pivot(index="evaluador", columns="item_id", values=col).to_numpy(dtype=float)
    return float(krippendorff.alpha(reliability_data=m, level_of_measurement="ordinal")) if len(m) > 1 else float("nan")

alfa_util, alfa_intel = alfa("utilidad"), alfa("inteligibilidad")
rho, p_rho = spearmanr(mod.wer_item, mod.utilidad_med)
tipos = r[r.tipo == "modelo"].tipo_error.value_counts(normalize=True).mul(100).round(1).to_dict()

# ------------------------------------------------------------------ veredicto
med_util_modelo = float(mod.utilidad_med.median())
med_util_control = float(ctl.utilidad_med.median())
validez = []
if med_util_control < CONTROL_MIN:
    validez.append(f"los controles obtienen mediana {med_util_control} < {CONTROL_MIN}: revisar instrumento o referencias")
if not alfa_util >= ALFA_MIN:
    validez.append(f"concordancia baja en utilidad (alfa = {alfa_util:.2f} < {ALFA_MIN})")

if med_util_control < CONTROL_MIN:
    veredicto = "Sin veredicto (falla la condición de control)"
elif pct_ponderado >= SATISF_PCT and med_util_modelo >= SATISF_MED:
    veredicto = "Satisfactorio"
elif pct_ponderado < INSUF_PCT or med_util_modelo <= INSUF_MED:
    veredicto = "Insuficiente"
else:
    veredicto = "Parcialmente satisfactorio"
if validez and not veredicto.startswith("Sin"):
    veredicto += " (con cautela)"

resumen = {
    "veredicto": veredicto, "condiciones_de_validez_incumplidas": validez,
    "pct_aceptables_ponderado": round(pct_ponderado, 1), "pct_aceptables_simple": round(pct_simple, 1),
    "utilidad_mediana_modelo": med_util_modelo, "utilidad_mediana_control": med_util_control,
    "inteligibilidad_mediana_modelo": float(mod.inteligibilidad_med.median()),
    "inteligibilidad_mediana_control": float(ctl.inteligibilidad_med.median()),
    "brecha_utilidad_media_control_menos_modelo": round(float(ctl.utilidad_media.mean() - mod.utilidad_media.mean()), 2),
    "alfa_krippendorff_utilidad": round(alfa_util, 3), "alfa_krippendorff_inteligibilidad": round(alfa_intel, 3),
    "spearman_wer_vs_utilidad": {"rho": round(float(rho), 3), "p": float(f"{p_rho:.2g}")},
    "wer_automatico_items_modelo": round(wer_auto, 2),
    "wer_post_edicion_medio": round(float(post.WER_post_edicion.mean()), 2),
    "tipo_error_principal_pct": tipos,
    "regla": {"aceptable": f"utilidad mediana >= {UMBRAL_ACEPTABLE}",
              "satisfactorio": f">= {SATISF_PCT} % aceptables (ponderado) y utilidad mediana >= {SATISF_MED}",
              "insuficiente": f"< {INSUF_PCT} % aceptables o utilidad mediana <= {INSUF_MED}",
              "validez": f"control >= {CONTROL_MIN}; alfa >= {ALFA_MIN}"},
}
json.dump(resumen, open(os.path.join(a.out, "resumen.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("\nPor estrato (ítems del modelo):\n" + pe.round(2).to_string())
print("\nPost-edición:\n" + post.to_string(index=False))
print("\n" + json.dumps(resumen, ensure_ascii=False, indent=2))
