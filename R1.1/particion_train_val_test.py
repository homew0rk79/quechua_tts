"""
particion_train_val_test.py (v2)

Particiona los 2211 (verificar) archivos con Estado == "ok" en
train / val / test (80/10/10), estratificando por una "fuente" derivada
del nombre de archivo (audio_name), porque el corpus mezcla dos
estructuras de nombrado distintas:

  - Nombres tipo "01 TAQAQA RUNA-108"           -> fuente = "01 TAQAQA RUNA"
  - Nombres tipo "<hash>-35_P2" (conversacional) -> fuente = "<hash>"
                                                      participante = "P2"

Las fuentes con muy pocos archivos (por defecto < MIN_GROUP_SIZE) no
permiten una partición 80/10/10 confiable por sí solas, así que se
agrupan bajo "grupo_estratificacion" solo para decidir la partición;
la fuente original se conserva en el manifiesto final para trazabilidad.

No mueve ni copia audios: genera manifiestos (train.csv, val.csv,
test.csv) con audio_name, audio_path, transcripción y fuente.

AJUSTAR antes de ejecutar:
  - INPUT_FILE: ruta a tu qtts_trainable.xlsx (o su export a .csv)
  - AUDIO_DIR: carpeta local donde están los .wav, si quieres que el
    script calcule duración real con librosa (opcional; si no está
    disponible o AUDIO_DIR es None, el reporte omite duración)
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- CONFIGURACIÓN ----------
INPUT_FILE = "../qtts_trainable.xlsx"   # <-- ajustar (acepta .xlsx o .csv)
AUDIO_DIR = "../corpus_normalizado"                      # <-- p.ej. Path("corpus_normalizado"); None si no vas a calcular duración
OUTPUT_DIR = Path("particiones")


COLS = {
    "nombre": "audio_name",
    "ruta": "audio_path",
    "texto": "text_audio",
    "id": "audio_id",
    "genero": "gender",
    "estado": "Estado",
}
ESTADO_OK = "ok"

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10
SEED = 42
MIN_GROUP_SIZE = 10  # por debajo de esto, una fuente no se particiona sola

# Semillas candidatas, fijadas ANTES de ver ningún resultado (no ajustar
# esta lista después de correr el script para "mejorar" el resultado; si
# ninguna cumple los umbrales, el problema es el diseño, no la semilla).
CANDIDATE_SEEDS = [42, 1, 7, 123, 2024, 99, 17, 55, 8, 31]

# ---------- CARGA ----------
def cargar(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)

df = cargar(INPUT_FILE)
conservados = df[df[COLS["estado"]].astype(str).str.lower() == ESTADO_OK].copy()

print(f"Archivos con Estado == '{ESTADO_OK}': {len(conservados)} "
      f"(el Capítulo 4 reporta 2211; si difiere, revisar antes de continuar)")

# ---------- DERIVAR FUENTE Y PARTICIPANTE DESDE audio_name ----------
PARTICIPANTE_RE = re.compile(r"_(P\d+)$")
INDICE_RE = re.compile(r"-(\d+)$")

def derivar_fuente_participante(nombre):
    nombre = str(nombre).strip()
    participante = None
    m = PARTICIPANTE_RE.search(nombre)
    base = nombre
    if m:
        participante = m.group(1)
        base = nombre[: m.start()]
    m2 = INDICE_RE.search(base)
    if m2:
        fuente = base[: m2.start()].strip()
    else:
        fuente = base.strip()  # no siguió el patrón esperado -> anómalo
    anomalo = m2 is None
    return fuente, participante, anomalo

derivado = conservados[COLS["nombre"]].apply(derivar_fuente_participante)
conservados["fuente"] = derivado.apply(lambda t: t[0])
conservados["participante_local"] = derivado.apply(lambda t: t[1])
conservados["nombre_anomalo"] = derivado.apply(lambda t: t[2])

# ---------- REGISTRAR ANOMALÍAS (no bloquear el flujo) ----------
anomalos = conservados[conservados["nombre_anomalo"]]
if len(anomalos) > 0:
    anomalos[[COLS["nombre"]]].to_csv(OUTPUT_DIR / "casos_atipicos_particion.csv", index=False) \
        if OUTPUT_DIR.exists() else None
    print(f"AVISO: {len(anomalos)} nombre(s) no siguen el patrón esperado "
          f"'...-<indice>' o '...-<indice>_P<n>'. Revisar casos_atipicos_particion.csv.")

# ---------- DEFINIR LA UNIDAD DE PARTICIÓN (antes de decidir agrupamientos) ----------
# Regla (deliberada, no accidental): las entradas conversacionales (con
# participante_local, p.ej. "<hash>-35_P2") bloquean TODA la sesión (el hash
# completo) como UNA sola unidad, para no fugar contexto de diálogo entre
# splits. Las fuentes nombradas (sin participante_local) usan el ARCHIVO
# individual como unidad: con solo ~41 fuentes de tamaños muy desiguales,
# bloquear cada fuente nombrada entera hace que el split 80/10/10 sea
# matemáticamente inviable. Esto acepta que un mismo hablante/sesión de
# fuente nombrada pueda aparecer en train y test: es un trade-off forzado
# por la escasez y desigualdad de fuentes, y debe declararse como
# limitación explícita en R1.1.
def unidad_de_particion(row):
    if pd.notna(row["participante_local"]):
        return row["fuente"]           # unidad = sesión conversacional completa
    return row[COLS["nombre"]]         # unidad = archivo individual

conservados["unidad"] = conservados.apply(unidad_de_particion, axis=1)

# ---------- AGRUPAR FUENTES CON POCAS UNIDADES PARA LA ESTRATIFICACIÓN ----------
# CLAVE: el umbral se mide en UNIDADES de partición, no en archivos. Una
# fuente conversacional puede tener muchos archivos (turnos P1/P2/P3) pero
# UNA sola unidad (un solo hash de sesión) -> no se puede repartir 80/10/10
# por sí sola y debe agruparse, sin importar cuántos archivos tenga. Medir
# el umbral en archivos (como en la versión anterior) fue precisamente el
# error que dejó varias fuentes conversacionales grandes-en-archivos pero
# de una sola sesión como bloques indivisibles, forzados siempre a train.
unidades_por_fuente = conservados.groupby("fuente")["unidad"].nunique()
print(f"\nFuentes derivadas: {len(unidades_por_fuente)} distintas "
      f"({(unidades_por_fuente < MIN_GROUP_SIZE).sum()} con menos de {MIN_GROUP_SIZE} unidades de partición)")

fuentes_pequenas = set(unidades_por_fuente[unidades_por_fuente < MIN_GROUP_SIZE].index)
conservados["grupo_estratificacion"] = conservados["fuente"].where(
    ~conservados["fuente"].isin(fuentes_pequenas), "otras_fuentes_agrupadas"
)

objetivo = {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC}

def construir_particion(seed):
    def split_group(g):
        asign = {}
        # género mayoritario de cada unidad dentro de este grupo de fuente
        genero_unidad = g.groupby("unidad")[COLS["genero"]].agg(
            lambda s: s.astype(str).str.upper().mode().iloc[0]
        )
        for _, unidades_genero in genero_unidad.groupby(genero_unidad):
            unidades = unidades_genero.index.to_series().sample(frac=1, random_state=seed)
            n = len(unidades)
            n_train = int(round(n * TRAIN_FRAC))
            n_val = int(round(n * VAL_FRAC))
            for i, u in enumerate(unidades):
                if i < n_train:
                    asign[u] = "train"
                elif i < n_train + n_val:
                    asign[u] = "val"
                else:
                    asign[u] = "test"
        return g["unidad"].map(asign)

    return conservados.groupby("grupo_estratificacion", group_keys=False).apply(split_group)

def evaluar_particion(particion):
    tmp_particion = particion
    props = tmp_particion.value_counts(normalize=True)
    dev_prop = sum(abs(props.get(p, 0.0) - objetivo[p]) for p in ["train", "val", "test"])

    pct_fem_global = (conservados[COLS["genero"]].astype(str).str.upper() == "F").mean() * 100
    pct_fem_part = conservados.groupby(tmp_particion)[COLS["genero"]].apply(
        lambda s: (s.astype(str).str.upper() == "F").mean() * 100
    )
    dev_genero = sum(abs(pct_fem_part.get(p, pct_fem_global) - pct_fem_global) for p in ["train", "val", "test"])

    cobertura = conservados.groupby("fuente").apply(lambda g: tmp_particion.loc[g.index].nunique())
    fuentes_atrapadas = sum(
        1 for f in cobertura.index
        if cobertura[f] == 1 and unidades_por_fuente.get(f, 0) >= MIN_GROUP_SIZE
    )

    # Criterio de puntaje declarado de antemano: penaliza fuerte cualquier
    # fuente grande atrapada entera en una partición, y suma los desvíos de
    # proporción y de composición de género. Menor score = mejor.
    score = dev_prop * 100 + dev_genero + fuentes_atrapadas * 50
    return score, dev_prop, dev_genero, fuentes_atrapadas

print("\n--- Búsqueda de partición balanceada (semillas fijas de antemano) ---")
resultados = []
for seed in CANDIDATE_SEEDS:
    particion_candidata = construir_particion(seed)
    score, dev_prop, dev_genero, atrapadas = evaluar_particion(particion_candidata)
    resultados.append((seed, score, particion_candidata))
    print(f"  seed={seed:>5} | score={score:6.2f} | desvío proporción={dev_prop:.3f} "
          f"| desvío género(pp)={dev_genero:5.1f} | fuentes grandes atrapadas={atrapadas}")

mejor_seed, _, mejor_particion = min(resultados, key=lambda r: r[1])
print(f"\nSemilla elegida: {mejor_seed} (menor score de la tabla anterior; "
      f"las {len(CANDIDATE_SEEDS)} semillas y sus puntajes quedan arriba para trazabilidad "
      f"y deben citarse en el capítulo, no solo la elegida)")

SEED = mejor_seed
conservados["particion"] = mejor_particion

# ---------- VERIFICACIÓN AUTOMÁTICA (correr e interpretar antes de dar la partición por buena) ----------
print("\n--- Verificación de la partición ---")

# 1) Proporción de archivos por partición vs. el objetivo 80/10/10
props = conservados["particion"].value_counts(normalize=True)
print("Proporción de archivos por partición (objetivo 0.80 / 0.10 / 0.10):")
print(props.reindex(["train", "val", "test"]).round(3))
for p in ["train", "val", "test"]:
    real = props.get(p, 0.0)
    if abs(real - objetivo[p]) > 0.03:
        print(f"  AVISO: '{p}' se desvía {abs(real - objetivo[p]) * 100:.1f} pp del objetivo.")

# 2) Ninguna fuente con suficientes unidades para repartirse debería quedar
#    entera en una sola partición (si pasa, algo volvió a fallar).
cobertura = conservados.groupby("fuente")["particion"].nunique()
sospechosas = [f for f in cobertura.index if cobertura[f] == 1 and unidades_por_fuente.get(f, 0) >= MIN_GROUP_SIZE]
if sospechosas:
    print(f"  AVISO: estas fuentes tenían >= {MIN_GROUP_SIZE} unidades pero quedaron "
          f"enteras en una sola partición (revisar): {sospechosas}")
else:
    print(f"  OK: ninguna fuente con >= {MIN_GROUP_SIZE} unidades quedó entera en una sola partición.")

# 3) Composición de género por partición vs. el corpus completo
pct_femenino_global = (conservados[COLS["genero"]].astype(str).str.upper() == "F").mean() * 100
print(f"\n% femenino global del corpus curado: {pct_femenino_global:.1f}%")
pct_femenino_particion = conservados.groupby("particion")[COLS["genero"]].apply(
    lambda s: (s.astype(str).str.upper() == "F").mean() * 100
)
for p in ["train", "val", "test"]:
    val_p = pct_femenino_particion.get(p, float("nan"))
    if abs(val_p - pct_femenino_global) > 8:
        print(f"  AVISO: % femenino en '{p}' ({val_p:.1f}%) se desvía más de 8 pp del global.")

# ---------- (OPCIONAL) DURACIÓN REAL VÍA LIBROSA ----------
if AUDIO_DIR is not None:
    try:
        import librosa
        def duracion(row):
            try:
                return librosa.get_duration(path=str(Path(AUDIO_DIR) / row[COLS["ruta"]]))
            except Exception:
                return np.nan
        conservados["duracion_seg"] = conservados.apply(duracion, axis=1)
    except ImportError:
        print("AVISO: librosa no disponible; se omite el cálculo de duración.")
        conservados["duracion_seg"] = np.nan
else:
    conservados["duracion_seg"] = np.nan

# ---------- REPORTE (insumo para el párrafo pendiente de R1.1) ----------
reporte = (
    conservados.groupby("particion")
    .agg(
        n_archivos=(COLS["nombre"], "count"),
        duracion_total_seg=("duracion_seg", "sum"),
        pct_femenino=(COLS["genero"], lambda s: (s.astype(str).str.upper() == "F").mean() * 100),
        n_fuentes=("fuente", "nunique"),
    )
    .reindex(["train", "val", "test"])
)
print("\nResumen de la partición:\n", reporte)

# Trazabilidad: cuántas particiones distintas tocó cada fuente original
trazabilidad = conservados.groupby("fuente")["particion"].nunique()
fuentes_repartidas = trazabilidad[trazabilidad > 1]
print(f"\n{len(fuentes_repartidas)} fuente(s) originales quedaron repartidas "
      f"entre más de una partición (esperable para fuentes grandes; revisar "
      f"si alguna pequeña quedó dividida de forma no intencional).")

# ---------- EXPORTAR ----------
OUTPUT_DIR.mkdir(exist_ok=True)
cols_manifiesto = [COLS["nombre"], COLS["ruta"], COLS["texto"], COLS["genero"], "fuente"]
for particion in ["train", "val", "test"]:
    conservados[conservados["particion"] == particion][cols_manifiesto].to_csv(
        OUTPUT_DIR / f"{particion}.csv", index=False
    )
reporte.to_csv(OUTPUT_DIR / "reporte_particion.csv")
if len(anomalos) > 0:
    anomalos[[COLS["nombre"]]].to_csv(OUTPUT_DIR / "casos_atipicos_particion.csv", index=False)

print(f"\nManifiestos y reporte exportados a: {OUTPUT_DIR.resolve()}")
