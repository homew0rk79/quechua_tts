"""R4.3 — Categorización semiautomática de errores del modelo ajustado (C1) sobre la partición de prueba.

Uso: python categorizar_errores.py --pred pred_test.csv --out R4.3
Usa el subconjunto sin fuentes contaminadas. Las reglas son heurísticas y se aplican en orden;
cada sustitución queda en la primera categoría cuya normalización iguala referencia e hipótesis.
"""
import argparse, os, re, collections
import jiwer, pandas as pd

EXCLUIDAS = ["01 TAQAQA RUNA", "02 TAQAQA RUNAS", "05_11-20 RUTA QUECHUA"]
ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--out", default="R4.3")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

d = pd.read_csv(a.pred).fillna("")
d = d[~d.fuente.isin(EXCLUIDAS)]

# Préstamo o palabra de origen castellano (aproximación): letras ajenas al alfabeto quechua oficial
# de tres vocales, o vocales tildadas.
CAST = re.compile(r"[eobdfgvzjáéíóú]|c(?!h)")
def castellano(w): return bool(CAST.search(w))

def sin_ap(w): return w.replace("'", "")
def sin_asp(w): return re.sub(r"(ch|p|t|k|q)h", r"\1", w)
def qk(w): return w.replace("q", "k")
def voc(w): return w.translate(str.maketrans("eoáéíóú", "iuaiiuu"))
def var_suf(w): return re.sub(r"chis\b", "chik", w)
REGLAS = [
    ("Glotalización (solo apóstrofo)", sin_ap),
    ("Aspiración (h tras oclusiva)", lambda w: sin_asp(sin_ap(w))),
    ("Alternancia q/k", lambda w: qk(sin_asp(sin_ap(w)))),
    ("Vocales e/i, o/u o tildes", lambda w: voc(qk(sin_asp(sin_ap(w))))),
    ("Variante del sufijo -chik/-chis", lambda w: var_suf(voc(qk(sin_asp(sin_ap(w)))))),
]
def lev(x, y):
    p = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        c = [i]
        for j, cy in enumerate(y, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (cx != cy)))
        p = c
    return p[-1]

cat = collections.Counter(); ejemplos = collections.defaultdict(list)
S = D = I = H = 0; seg_reg = 0; seg_pal = 0
pal_tot = collections.Counter(); pal_ok = collections.Counter()
for ref, hyp in zip(d.ref, d.hyp):
    rw, hw = ref.split(), hyp.split()
    o = jiwer.process_words(ref, hyp)
    S += o.substitutions; D += o.deletions; I += o.insertions; H += o.hits
    ch = o.alignments[0]
    # aciertos por tipo de palabra
    for c in ch:
        for k in range(c.ref_start_idx, c.ref_end_idx):
            t = "castellano" if castellano(rw[k]) else "quechua"
            pal_tot[t] += 1; pal_ok[t] += c.type == "equal"
    # regiones de error contiguas → segmentación
    i = 0
    while i < len(ch):
        if ch[i].type == "equal": i += 1; continue
        j = i
        while j < len(ch) and ch[j].type != "equal": j += 1
        r0 = min(c.ref_start_idx for c in ch[i:j]); r1 = max(c.ref_end_idx for c in ch[i:j])
        h0 = min(c.hyp_start_idx for c in ch[i:j]); h1 = max(c.hyp_end_idx for c in ch[i:j])
        rs, hs = rw[r0:r1], hw[h0:h1]
        if len(rs) != len(hs) and rs and hs and sin_ap("".join(rs)) == sin_ap("".join(hs)):
            seg_reg += 1; seg_pal += len(rs)
            if len(ejemplos["Segmentación"]) < 4: ejemplos["Segmentación"].append((" ".join(rs), " ".join(hs)))
        elif len(rs) == len(hs):  # sustituciones uno a uno
            for x, y in zip(rs, hs):
                if x == y: continue
                etiqueta = None
                for nombre, f in REGLAS:
                    if f(x) == f(y): etiqueta = nombre; break
                if etiqueta is None:
                    pref = len(os.path.commonprefix([x, y]))
                    if castellano(x): etiqueta = "Palabra de origen castellano"
                    elif pref >= 4 and pref >= min(len(x), len(y)) / 2:
                        etiqueta = "Misma raíz, sufijo o terminación distinta"
                    elif lev(x, y) == 1: etiqueta = "Otra, a un carácter de distancia"
                    else: etiqueta = "Otra sustitución"
                cat[etiqueta] += 1
                if len(ejemplos[etiqueta]) < 4: ejemplos[etiqueta].append((x, y))
        i = j

tot_err = S + D + I; N = S + D + H
print(f"Grabaciones: {len(d)} | palabras de referencia: {N} | WER {100*tot_err/N:.2f}")
ops = pd.DataFrame([["Sustituciones", S], ["Borrados", D], ["Inserciones", I]], columns=["operacion", "n"])
ops["pct_errores"] = (100 * ops.n / tot_err).round(1); print(ops.to_string(index=False))
tc = pd.DataFrame(cat.most_common(), columns=["categoria", "n"])
n_sub11 = tc.n.sum(); tc["pct_sust_1a1"] = (100 * tc.n / n_sub11).round(1)
tc["ejemplos (ref → hip)"] = [", ".join(f"{x} → {y}" for x, y in ejemplos[c][:3]) for c in tc.categoria]
print(f"\nSustituciones uno a uno clasificadas: {n_sub11}"); print(tc.to_string(index=False))
print(f"\nRegiones de segmentación: {seg_reg} (afectan {seg_pal} palabras de referencia). Ej.: {ejemplos['Segmentación']}")
tp = pd.DataFrame([[t, pal_tot[t], 100 * (1 - pal_ok[t] / pal_tot[t])] for t in ["quechua", "castellano"]],
                  columns=["tipo_palabra", "n_palabras", "pct_no_acertadas"]).round(1)
print("\n" + tp.to_string(index=False))
ops.to_csv(f"{a.out}/operaciones.csv", index=False); tc.to_csv(f"{a.out}/categorias_sustitucion.csv", index=False)
tp.to_csv(f"{a.out}/castellano_vs_quechua.csv", index=False)
pd.DataFrame([["regiones", seg_reg], ["palabras_ref", seg_pal]], columns=["segmentacion", "n"]).to_csv(f"{a.out}/segmentacion.csv", index=False)
