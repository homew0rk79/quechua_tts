"""R4.2 — Selección de ítems para la evaluación humana a partir de las predicciones de prueba de R4.1.

Diseño: 18 ítems del modelo C1 (6 por estrato de WER: bajo, medio, alto) + 6 controles
(transcripción humana de referencia, 2 por estrato) + 2 ítems de práctica que no se analizan.
Todos los audios salen del subconjunto limpio de prueba (sin fuentes contaminadas) y son distintos.

Uso:
  python preparar_items.py --pred pred_test.csv --out items --seed 42

Salidas en --out:
  items.csv          un ítem por fila (lo que lee la app y el análisis)
  estratos.json      puntos de corte y tamaño de cada estrato en la población limpia (para ponderar)
  lista_audios.txt   archivos de audio que hay que copiar desde el servidor
"""
import argparse, json, os
import jiwer, numpy as np, pandas as pd

EXCLUIDAS = ["01 TAQAQA RUNA", "02 TAQAQA RUNAS", "05_11-20 RUTA QUECHUA"]

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True, help="pred_test.csv de C1 (columnas audio_name, audio_path, fuente, gender, ref, hyp)")
ap.add_argument("--train_csv", default=None, help="opcional: train.csv para marcar fuentes no vistas")
ap.add_argument("--out", default="items")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n_modelo", type=int, default=6, help="ítems del modelo por estrato")
ap.add_argument("--n_control", type=int, default=2, help="controles por estrato")
ap.add_argument("--min_palabras", type=int, default=4)
ap.add_argument("--max_palabras", type=int, default=30)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)

d = pd.read_csv(a.pred).fillna("")
d = d[~d.fuente.isin(EXCLUIDAS)].copy()
d["wer_item"] = [100 * jiwer.wer(r, h) if r.strip() else np.nan for r, h in zip(d.ref, d.hyp)]
d["n_palabras"] = d.ref.str.split().str.len()
if a.train_csv:
    vistas = set(pd.read_csv(a.train_csv).fuente)
    d["vista_en_entrenamiento"] = d.fuente.isin(vistas)
else:  # las dos fuentes que R4.1 marcó como no vistas
    d["vista_en_entrenamiento"] = ~d.fuente.isin(["005a005989303ada8a755b7af6db785c", "05a8cb111e0791b950a61f9e579c9bae"])

# Estratos por terciles del WER por archivo, calculados sobre toda la población limpia
c1, c2 = np.percentile(d.wer_item, [100 / 3, 200 / 3])
d["estrato"] = np.where(d.wer_item <= c1, "bajo", np.where(d.wer_item <= c2, "medio", "alto"))
poblacion = d.estrato.value_counts().to_dict()

eleg = d[d.n_palabras.between(a.min_palabras, a.max_palabras)].copy()
print(f"Población limpia: {len(d)} | elegibles por longitud ({a.min_palabras}-{a.max_palabras} palabras): {len(eleg)}")
print(f"Cortes de WER por archivo: bajo <= {c1:.1f} < medio <= {c2:.1f} < alto | población: {poblacion}")

usados = set()

def elegir(pool, n, priorizar_no_vistas=False):
    """Elige n audios no usados, rotando entre fuentes para maximizar diversidad."""
    pool = pool[~pool.audio_name.isin(usados)].sample(frac=1, random_state=int(rng.integers(1e9)))
    fuentes = list(pool.fuente.unique())
    if priorizar_no_vistas:
        fuentes.sort(key=lambda f: bool(pool[pool.fuente == f].vista_en_entrenamiento.iloc[0]))
    elegidos = []
    while len(elegidos) < n and fuentes:
        for f in list(fuentes):
            cand = pool[(pool.fuente == f) & ~pool.audio_name.isin(usados)]
            if cand.empty:
                fuentes.remove(f)
                continue
            fila = cand.iloc[0]
            elegidos.append(fila)
            usados.add(fila.audio_name)
            if len(elegidos) == n:
                break
    if len(elegidos) < n:
        raise SystemExit(f"No hay suficientes audios elegibles ({len(elegidos)}/{n})")
    return pd.DataFrame(elegidos)

filas = []
for est in ["bajo", "medio", "alto"]:
    pool = eleg[eleg.estrato == est]
    m = elegir(pool, a.n_modelo, priorizar_no_vistas=True).assign(tipo="modelo", texto_mostrado=lambda x: x.hyp)
    c = elegir(pool, a.n_control).assign(tipo="control", texto_mostrado=lambda x: x.ref)
    filas += [m, c]
practica = pd.concat([
    elegir(eleg[eleg.estrato == "medio"], 1).assign(tipo="practica", texto_mostrado=lambda x: x.hyp),
    elegir(eleg[eleg.estrato == "bajo"], 1).assign(tipo="practica", texto_mostrado=lambda x: x.ref),
])

items = pd.concat(filas, ignore_index=True)
items = items.sample(frac=1, random_state=a.seed).reset_index(drop=True)  # que el código no revele el tipo
items["item_id"] = [f"I{i + 1:02d}" for i in range(len(items))]
practica["item_id"] = ["P1", "P2"]
items = pd.concat([practica, items], ignore_index=True)
items["estrato"] = np.where(items.tipo == "control", "control:" + items.estrato, items.estrato)

cols = ["item_id", "tipo", "estrato", "audio_name", "audio_path", "fuente", "gender",
        "vista_en_entrenamiento", "n_palabras", "wer_item", "texto_mostrado", "ref", "hyp"]
items = items[cols].round({"wer_item": 2})
items.to_csv(os.path.join(a.out, "items.csv"), index=False)
with open(os.path.join(a.out, "lista_audios.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(items.audio_path) + "\n")
with open(os.path.join(a.out, "estratos.json"), "w", encoding="utf-8") as f:
    json.dump({"corte_bajo_medio": round(c1, 2), "corte_medio_alto": round(c2, 2), "poblacion_limpia": poblacion,
               "semilla": a.seed}, f, ensure_ascii=False, indent=2)

an = items[items.tipo != "practica"]
print(f"\nÍtems: {len(an)} analizables + {len(practica)} de práctica")
print(an.groupby(["tipo", "estrato"]).agg(n=("item_id", "size"), wer_medio=("wer_item", "mean"),
                                           palabras=("n_palabras", "mean")).round(1).to_string())
print(f"Fuentes distintas: {an.fuente.nunique()} | de fuentes no vistas: {(~an.vista_en_entrenamiento).sum()} "
      f"| género F/M: {(an.gender == 'F').sum()}/{(an.gender == 'M').sum()}")
print(f"Salidas en {a.out}/")
