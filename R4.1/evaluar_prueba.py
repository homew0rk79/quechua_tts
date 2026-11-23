"""R4.1 — Comparación del modelo ajustado (C1) con la línea base (R2.3) sobre la partición de prueba.

No ejecuta ningún modelo: compara dos archivos de predicciones ya generados con eval_zeroshot.py.

Uso (desde ~/tesis/scripts):
  python evaluar_prueba.py --c1 ~/tesis/runs/C1/pred_test.csv \
      --base ~/tesis/runs/zs_romero_test.csv \
      --train_csv ~/tesis/data/particiones/train.csv --out ~/tesis/runs/R4.1

Salidas en --out:
  tabla_global.csv   WER/CER de ambos modelos por subconjunto, con y sin apóstrofo
  por_fuente.csv     WER/CER por fuente para ambos modelos, con marcas de excluida y no vista
  bootstrap.csv      reducción de WER y CER con intervalo de confianza del 95 % (bootstrap pareado)
"""
import argparse, os, sys
import jiwer, numpy as np, pandas as pd

EXCLUIDAS = ["01 TAQAQA RUNA", "02 TAQAQA RUNAS", "05_11-20 RUTA QUECHUA"]

ap = argparse.ArgumentParser()
ap.add_argument("--c1", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--train_csv", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_boot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
x = {k: os.path.expanduser(v) for k, v in vars(a).items() if isinstance(v, str)}
os.makedirs(x["out"], exist_ok=True)

# ------------------------------------------------------------ datos
c1 = pd.read_csv(x["c1"]).fillna("")
bs = pd.read_csv(x["base"]).fillna("")
d = c1[["audio_name", "fuente", "ref", "hyp"]].merge(
    bs[["audio_name", "ref", "hyp"]], on="audio_name", suffixes=("", "_base"), how="outer", indicator=True)
if (d._merge != "both").any():
    sys.exit(f"ERROR: los archivos no cubren los mismos audios: {d[d._merge != 'both'].audio_name.tolist()[:5]}")
if (d.ref != d.ref_base).any():
    sys.exit(f"ERROR: {(d.ref != d.ref_base).sum()} referencias difieren entre ambos archivos (¿otra normalización?)")
vistas = set(pd.read_csv(x["train_csv"]).fuente)
d["excluida"] = d.fuente.isin(EXCLUIDAS)
d["no_vista"] = ~d.fuente.isin(vistas)
print(f"Prueba: {len(d)} archivos | excluidas: {d.excluida.sum()} | "
      f"de fuentes no vistas en entrenamiento: {d.no_vista.sum()} ({d[d.no_vista].fuente.nunique()} fuentes)")
print("Fuentes no vistas:", sorted(d[d.no_vista].fuente.unique()))

sin_ap = lambda s: " ".join(s.replace("'", "").split())
for col in ["ref", "hyp", "hyp_base"]:
    d[col + "_sa"] = d[col].map(sin_ap)

# errores y longitudes por archivo (WER y CER de corpus = suma de errores / suma de referencia)
def conteos(ref, hyp, nivel):
    f = jiwer.process_words if nivel == "w" else jiwer.process_characters
    o = f(ref, hyp)
    return o.substitutions + o.deletions + o.insertions, o.substitutions + o.deletions + o.hits

for sa in ["", "_sa"]:
    for nivel in ["w", "c"]:
        for mod, h in [("c1", "hyp"), ("base", "hyp_base")]:
            e, n = zip(*[conteos(r, hh, nivel) for r, hh in zip(d["ref" + sa], d[h + sa])])
            d[f"e_{mod}_{nivel}{sa}"], d[f"n_{nivel}{sa}"] = e, n

def tasa(sub, mod, nivel, sa=""):
    return 100 * sub[f"e_{mod}_{nivel}{sa}"].sum() / sub[f"n_{nivel}{sa}"].sum()

subconjuntos = {"Completo": d, "Sin contaminadas": d[~d.excluida],
                "Solo contaminadas": d[d.excluida], "Fuentes no vistas": d[d.no_vista]}

# ------------------------------------------------------------ tabla global
filas = []
for nombre, sub in subconjuntos.items():
    for mod, etiqueta in [("base", "Sin ajuste (R2.3)"), ("c1", "Ajustado (C1)")]:
        filas.append([nombre, len(sub), etiqueta, tasa(sub, mod, "w"), tasa(sub, mod, "c"),
                      tasa(sub, mod, "w", "_sa"), tasa(sub, mod, "c", "_sa")])
tg = pd.DataFrame(filas, columns=["subconjunto", "archivos", "modelo", "WER", "CER",
                                  "WER_sin_apostrofo", "CER_sin_apostrofo"]).round(2)
tg.to_csv(os.path.join(x["out"], "tabla_global.csv"), index=False)
print("\n" + tg.to_string(index=False))

# ------------------------------------------------------------ por fuente
pf = []
for f, sub in d.groupby("fuente"):
    pf.append([f, len(sub), bool(sub.excluida.iloc[0]), bool(sub.no_vista.iloc[0]),
               tasa(sub, "base", "w"), tasa(sub, "c1", "w"), tasa(sub, "base", "c"), tasa(sub, "c1", "c")])
pf = pd.DataFrame(pf, columns=["fuente", "n", "excluida", "no_vista", "WER_base", "WER_c1", "CER_base", "CER_c1"])
pf = pf.sort_values("CER_c1", ascending=False).round(2)
pf.to_csv(os.path.join(x["out"], "por_fuente.csv"), index=False)
print("\nPor fuente (ordenado por CER de C1):\n" + pf.to_string(index=False))

# ------------------------------------------------------------ bootstrap pareado
rng = np.random.default_rng(a.seed)
bt = []
for nombre in ["Sin contaminadas", "Fuentes no vistas"]:
    sub = subconjuntos[nombre]
    idx = rng.integers(0, len(sub), size=(a.n_boot, len(sub)))
    for nivel, met in [("w", "WER"), ("c", "CER")]:
        eb, ec, n = (sub[k].to_numpy() for k in [f"e_base_{nivel}", f"e_c1_{nivel}", f"n_{nivel}"])
        red = 100 * (eb[idx].sum(1) - ec[idx].sum(1)) / n[idx].sum(1)
        obs = tasa(sub, "base", nivel) - tasa(sub, "c1", nivel)
        lo, hi = np.percentile(red, [2.5, 97.5])
        bt.append([nombre, len(sub), met, obs, lo, hi, (red <= 0).mean()])
bt = pd.DataFrame(bt, columns=["subconjunto", "archivos", "metrica", "reduccion_puntos",
                               "IC95_inf", "IC95_sup", "prop_remuestras_sin_mejora"]).round(3)
bt.to_csv(os.path.join(x["out"], "bootstrap.csv"), index=False)
print(f"\nBootstrap pareado ({a.n_boot} remuestras, semilla {a.seed}):\n" + bt.to_string(index=False))
print(f"\nSalidas en {x['out']}")
