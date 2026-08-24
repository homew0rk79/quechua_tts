"""
R1.2 - Normalización del corpus de audio
Quechua Abanquino - Tesis ASR PUCP

Parámetros objetivo:
  - Frecuencia de muestreo : 16 kHz
  - Canales               : mono
  - Amplitud              : normalización por pico
  - Formato               : WAV 16-bit PCM
"""

import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
EXCEL_PATH    = "audios/qtts_trainable.xlsx"   
AUDIO_DIR     = "audios/"     
OUTPUT_DIR    = "corpus_normalizado/"    
LOG_PATH      = "log_normalizacion.csv"
TARGET_SR     = 16000
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel(EXCEL_PATH)
resultados = []

print(f"Archivos a procesar: {len(df)}\n")

for i, fila in df.iterrows():
    nombre_archivo = fila["audio_path"]                      
    ruta_entrada   = os.path.join(AUDIO_DIR, nombre_archivo)
    ruta_salida    = os.path.join(OUTPUT_DIR, nombre_archivo)

    estado        = "OK"
    duracion_seg  = None
    sr_original   = None

    try:
        # Leer SR original sin cargar todo el audio (rápido)
        info        = sf.info(ruta_entrada)
        sr_original = info.samplerate

        # Cargar, resamplear a 16 kHz y convertir a mono
        audio, _ = librosa.load(ruta_entrada, sr=TARGET_SR, mono=True)

        if len(audio) == 0:
            raise ValueError("Archivo vacío")

        duracion_seg = round(len(audio) / TARGET_SR, 2)

        # Normalización de amplitud por pico
        pico = np.max(np.abs(audio))
        if pico > 0:
            audio = audio / pico

        # Guardar como WAV 16-bit PCM
        sf.write(ruta_salida, audio, TARGET_SR, subtype="PCM_16")

    except FileNotFoundError:
        estado = "ERROR: archivo no encontrado"
    except Exception as e:
        estado = f"ERROR: {e}"

    resultados.append({
        "audio_name"    : fila.get("audio_name", nombre_archivo),
        "audio_path"    : nombre_archivo,
        "sr_original"   : sr_original,
        "duracion_seg"  : duracion_seg,
        "estado"        : estado,
    })

    # Progreso cada 100 archivos
    if (i + 1) % 100 == 0:
        print(f"  Procesados: {i + 1}/{len(df)}")

# ─── LOG ──────────────────────────────────────────────────────────────────────
log_df = pd.DataFrame(resultados)
log_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

# ─── RESUMEN ──────────────────────────────────────────────────────────────────
ok      = (log_df["estado"] == "OK").sum()
errores = len(log_df) - ok

print(f"\n{'='*40}")
print(f"  Total procesados : {len(log_df)}")
print(f"  Exitosos         : {ok}")
print(f"  Con errores      : {errores}")

duraciones = log_df.loc[log_df["estado"] == "OK", "duracion_seg"]
if not duraciones.empty:
    print(f"  Duración mínima  : {duraciones.min():.2f} s")
    print(f"  Duración máxima  : {duraciones.max():.2f} s")
    print(f"  Duración media   : {duraciones.mean():.2f} s")
    print(f"  Duración total   : {duraciones.sum() / 3600:.2f} h")

print(f"  Log guardado en  : {LOG_PATH}")
print(f"{'='*40}")