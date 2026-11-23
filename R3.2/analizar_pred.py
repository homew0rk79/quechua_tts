"""Recalcula WER/CER desde un pred_val.csv: global, limpio y por fuente."""

import sys, jiwer, pandas as pd



EXCLUIDAS = ["01 TAQAQA RUNA", "02 TAQAQA RUNAS", "05_11-20 RUTA QUECHUA"]



df = pd.read_csv(sys.argv[1]).fillna("")



def met(d):

    r, h = d.ref.tolist(), d.hyp.tolist()

    return 100 * jiwer.wer(r, h), 100 * jiwer.cer(r, h)



limpio = df[~df.fuente.isin(EXCLUIDAS)]

for nombre, d in [("Completo", df), ("Limpio", limpio),

                  ("Solo excluidas", df[df.fuente.isin(EXCLUIDAS)])]:

    w, c = met(d)

    print(f"{nombre:15s} n={len(d):4d} | WER {w:6.2f} | CER {c:6.2f}")



filas = []

for f, d in df.groupby("fuente"):

    w, c = met(d)

    filas.append((f, len(d), w, c, f in EXCLUIDAS))

tabla = pd.DataFrame(filas, columns=["fuente", "n", "WER", "CER", "excluida"])

tabla = tabla.sort_values("CER", ascending=False).round(2)

print("\nPor fuente (ordenado por CER):")

print(tabla.to_string(index=False))

tabla.to_csv(sys.argv[1].replace(".csv", "_por_fuente.csv"), index=False)
