"""Curvas de entrenamiento desde trainer_state.json (evidencia de R3.1 y R3.2).

Uso (desde ~/tesis/scripts):
  python graficar_curvas.py C1 --linea_base 41.81 8.29   # modelo final + referencia zero-shot
  python graficar_curvas.py C1 C2 C3                     # una figura por corrida + comparación

Salidas en ~/tesis/runs/figuras/:
  curvas_<C>.png   pérdida (entrenamiento y validación) y WER/CER por época
  curvas_<C>.csv   los mismos datos por época, para tablas
  comparacion.png  WER y CER por época de todas las corridas (si se pasa más de una)

--linea_base solo tiene sentido para corridas que parten del checkpoint de Romero (C1, C3).
Las métricas de validación corresponden a la validación limpia (fuentes excluidas fuera).
"""
import argparse, glob, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, MaxNLocator
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("corridas", nargs="+")
ap.add_argument("--base", default="~/tesis/runs")
ap.add_argument("--salida", default="~/tesis/runs/figuras")
ap.add_argument("--linea_base", nargs=2, type=float, metavar=("WER", "CER"),
                help="WER y CER zero-shot de referencia sobre la validación limpia")
a = ap.parse_args()
base, salida = os.path.expanduser(a.base), os.path.expanduser(a.salida)
os.makedirs(salida, exist_ok=True)

plt.rcParams.update({"font.size": 10, "savefig.dpi": 300, "savefig.bbox": "tight"})
coma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))  # decimales con coma


def num_ckpt(ruta):
    return int(re.search(r"checkpoint-(\d+)", ruta).group(1))


def cargar(c):
    rutas = glob.glob(os.path.join(base, c, "checkpoint-*", "trainer_state.json"))
    if not rutas:
        raise SystemExit(f"ERROR: no hay checkpoint-*/trainer_state.json en {os.path.join(base, c)}")
    ruta = max(rutas, key=num_ckpt)  # el más reciente contiene todo el historial
    st = json.load(open(ruta, encoding="utf-8"))
    h = st["log_history"]
    tr = pd.DataFrame([e for e in h if "loss" in e])
    ev = pd.DataFrame([e for e in h if "eval_wer" in e])
    tr["epoca"] = np.ceil(tr.epoch - 1e-9).astype(int)
    ev["epoca"] = ev.epoch.round().astype(int)
    pe = ev[["epoca", "step", "eval_loss", "eval_wer", "eval_cer"]].merge(
        tr.groupby("epoca").loss.mean().rename("train_loss_media").reset_index(),
        on="epoca", how="left")
    mejor = st.get("best_model_checkpoint")
    fila = pe[pe.step == num_ckpt(mejor)] if mejor else pe.iloc[0:0]
    ep_mejor = int(fila.epoca.iloc[0]) if len(fila) else None
    return tr, pe, ep_mejor, ruta


def formato(ax, log=False):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(coma)
    ax.yaxis.set_major_formatter(coma)
    if log:
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.3)


datos = {}
for c in a.corridas:
    tr, pe, ep_mejor, ruta = cargar(c)
    datos[c] = pe
    pe.to_csv(os.path.join(salida, f"curvas_{c}.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(tr.epoch, tr.loss, color="0.75", lw=0.8, label="Entrenamiento (cada registro)")
    ax1.plot(pe.epoca, pe.train_loss_media, "o-", ms=3, label="Entrenamiento (media por época)")
    ax1.plot(pe.epoca, pe.eval_loss, "s-", ms=3, label="Validación limpia")
    ax1.set_yscale("log")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida CTC (escala logarítmica)")
    formato(ax1, log=True)
    ax1.legend(fontsize=8)

    ax2.plot(pe.epoca, pe.eval_wer, "o-", ms=3, color="C0", label="WER")
    ax2.plot(pe.epoca, pe.eval_cer, "s-", ms=3, color="C1", label="CER")
    if a.linea_base:
        ax2.axhline(a.linea_base[0], ls="--", lw=1, color="C0", label="WER zero-shot")
        ax2.axhline(a.linea_base[1], ls="--", lw=1, color="C1", label="CER zero-shot")
    if ep_mejor:
        ax2.axvline(ep_mejor, ls=":", lw=1, color="k", label=f"Mejor checkpoint (época {ep_mejor})")
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Error en validación limpia (%)")
    formato(ax2)
    ax2.legend(fontsize=8)

    fig.savefig(os.path.join(salida, f"curvas_{c}.png"))
    plt.close(fig)
    m = pe.loc[pe.epoca == ep_mejor].iloc[0] if ep_mejor else None
    print(f"{c}: {len(pe)} épocas | fuente {ruta}")
    if m is not None:
        print(f"    mejor: época {ep_mejor} | WER {m.eval_wer:.2f} | CER {m.eval_cer:.2f} | "
              f"pérdida val. {m.eval_loss:.3f}")

if len(datos) > 1:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, (c, pe) in enumerate(datos.items()):
        ax1.plot(pe.epoca, pe.eval_wer, marker="osd^v"[i % 5], ms=3, label=c)
        ax2.plot(pe.epoca, pe.eval_cer, marker="osd^v"[i % 5], ms=3, label=c)
    for ax, nombre in [(ax1, "WER"), (ax2, "CER")]:
        ax.set_ylim(0, 105)
        ax.set_xlabel("Época")
        ax.set_ylabel(f"{nombre} en validación limpia (%)")
        formato(ax)
        ax.legend(fontsize=8)
    fig.savefig(os.path.join(salida, "comparacion.png"))
    plt.close(fig)
    print("Comparación guardada.")

print(f"Salidas en {salida}")
