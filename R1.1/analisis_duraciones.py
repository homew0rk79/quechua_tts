# -*- coding: utf-8 -*-
"""
analisis_duraciones.py
======================

Analiza la distribución de duraciones del corpus normalizado y la cruza con el
resultado de la revisión manual, con el fin de sustentar la reformulación del
indicador objetivamente verificable de duración del resultado R1.1.

Produce:
  1. Estadísticos descriptivos de duración (corpus completo y corpus curado).
  2. Histograma de duraciones (PNG).
  3. Tabla de escenarios de umbral máximo: cuántos archivos y cuántas horas
     quedan al filtrar por distintos topes de duración.
  4. Verificación de viabilidad de CTC: archivos cuya transcripción no cabe en
     los frames de entrada que produce el extractor de características.
  5. CSV con la duración de cada archivo, unida al estado de la revisión manual.

Uso básico:
    python analisis_duraciones.py --audios corpus_normalizado \
                                  --csv qtts_trainable_analisis_manual.csv \
                                  --salida salida_duraciones

Dependencias: pandas, numpy, soundfile, matplotlib
    pip install pandas numpy soundfile matplotlib
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parámetros del modelo (XLS-R / Wav2Vec2)
# ---------------------------------------------------------------------------
# El extractor convolucional de Wav2Vec2 reduce la señal por un factor de 320:
# con audio a 16 kHz, produce aproximadamente 50 frames por segundo.
FRAMES_POR_SEGUNDO = 50.0

# Escenarios de umbral máximo (segundos) a evaluar.
UMBRALES_MAX = [10, 12, 15, 20, 25, 30, None]

# Umbral mínimo a evaluar (segundos).
UMBRAL_MIN = 1.0


# ---------------------------------------------------------------------------
# Taxonomía de la revisión manual
# ---------------------------------------------------------------------------
def clasificar_estado(estado):
    """Agrupa el campo libre 'Estado' en las categorías reportadas en R1.1."""
    s = str(estado).strip().lower()
    if s == 'ok':
        return 'OK'
    if 'duplicado' in s:
        return 'Duplicado'
    if 'español' in s or 'no es quechua' in s:
        if 'cortar' in s or 'despues de' in s:
            return 'Recortable'
        return 'Idioma distinto'
    if 'cortar' in s or 'borrar' in s or 'crotar' in s:
        return 'Recortable'
    if 'no coincide' in s or 'coincide parcialmente' in s:
        return 'Desalineado'
    return 'Pendiente consulta'


# ---------------------------------------------------------------------------
# Lectura de duraciones
# ---------------------------------------------------------------------------
def duracion_desde_archivo(ruta):
    """Devuelve la duración en segundos leyendo solo la cabecera del WAV."""
    import soundfile as sf
    try:
        info = sf.info(ruta)
        return info.frames / float(info.samplerate), info.samplerate
    except Exception as exc:                                   # noqa: BLE001
        print('  [aviso] no se pudo leer %s: %s' % (ruta, exc))
        return np.nan, np.nan


def recolectar_duraciones(df, dir_audios, col_archivo):
    """Añade las columnas duracion_s y sample_rate al DataFrame."""
    duraciones, tasas, faltantes = [], [], []
    total = len(df)
    for i, nombre in enumerate(df[col_archivo].astype(str), start=1):
        ruta = os.path.join(dir_audios, nombre)
        if not os.path.exists(ruta) and not nombre.lower().endswith('.wav'):
            ruta = os.path.join(dir_audios, nombre + '.wav')
        if not os.path.exists(ruta):
            faltantes.append(nombre)
            duraciones.append(np.nan)
            tasas.append(np.nan)
        else:
            d, sr = duracion_desde_archivo(ruta)
            duraciones.append(d)
            tasas.append(sr)
        if i % 250 == 0 or i == total:
            print('  leidos %d/%d' % (i, total))
    df['duracion_s'] = duraciones
    df['sample_rate'] = tasas
    return df, faltantes


# ---------------------------------------------------------------------------
# Reportes
# ---------------------------------------------------------------------------
def fmt_horas(segundos):
    if not np.isfinite(segundos):
        return 'n/d'
    seg = int(round(segundos))
    return '%dh %02dm %02ds' % (seg // 3600, (seg % 3600) // 60, seg % 60)


def descriptivos(serie, etiqueta):
    s = serie.dropna()
    if s.empty:
        print('\n[%s] sin datos' % etiqueta)
        return
    print('\n--- %s ---' % etiqueta)
    print('  archivos            : %d' % len(s))
    print('  duracion total      : %s' % fmt_horas(s.sum()))
    print('  minimo              : %6.2f s' % s.min())
    print('  percentil 25        : %6.2f s' % s.quantile(0.25))
    print('  mediana             : %6.2f s' % s.median())
    print('  media               : %6.2f s' % s.mean())
    print('  percentil 75        : %6.2f s' % s.quantile(0.75))
    print('  percentil 90        : %6.2f s' % s.quantile(0.90))
    print('  percentil 95        : %6.2f s' % s.quantile(0.95))
    print('  percentil 99        : %6.2f s' % s.quantile(0.99))
    print('  maximo              : %6.2f s' % s.max())


def tabla_escenarios(serie, umbral_min=UMBRAL_MIN, umbrales=UMBRALES_MAX):
    """Cuántos archivos y horas sobreviven a cada tope máximo de duración."""
    s = serie.dropna()
    base_n, base_h = len(s), s.sum()
    filas = []
    for u in umbrales:
        if u is None:
            sel = s[s >= umbral_min]
            etiqueta = 'sin tope maximo'
        else:
            sel = s[(s >= umbral_min) & (s <= u)]
            etiqueta = '%.0f s' % u
        filas.append({
            'Tope maximo': etiqueta,
            'Archivos retenidos': len(sel),
            '% archivos': round(100.0 * len(sel) / base_n, 2) if base_n else 0.0,
            'Horas retenidas': round(sel.sum() / 3600.0, 2),
            '% horas': round(100.0 * sel.sum() / base_h, 2) if base_h else 0.0,
            'Archivos excluidos': base_n - len(sel),
        })
    return pd.DataFrame(filas)


def verificar_ctc(df, col_texto):
    """
    CTC exige frames_entrada >= longitud_objetivo. Si la transcripción tiene mas
    caracteres que frames produce el encoder, la perdida diverge a infinito.
    """
    d = df.dropna(subset=['duracion_s']).copy()
    d['n_caracteres'] = d[col_texto].astype(str).str.len()
    d['frames_estimados'] = (d['duracion_s'] * FRAMES_POR_SEGUNDO).astype(int)
    d['holgura'] = d['frames_estimados'] - d['n_caracteres']
    problemas = d[d['holgura'] <= 0]
    ajustados = d[(d['holgura'] > 0) & (d['holgura'] < 10)]
    return d, problemas, ajustados


def histograma(serie, ruta_png, umbral_actual=10):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n[aviso] matplotlib no disponible; se omite el histograma.')
        return
    s = serie.dropna()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(s, bins=60, color='#2f5f8a', edgecolor='white', linewidth=0.4)
    ax.axvline(umbral_actual, color='#d63b3b', linestyle='--', linewidth=1.4,
               label='Tope declarado actualmente (%d s)' % umbral_actual)
    ax.axvline(s.median(), color='#4a7c3f', linestyle='-', linewidth=1.4,
               label='Mediana (%.1f s)' % s.median())
    ax.set_xlabel('Duracion (segundos)')
    ax.set_ylabel('Numero de archivos')
    ax.set_title('Distribucion de duraciones del corpus curado')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=150)
    print('\nHistograma guardado en: %s' % ruta_png)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audios', required=True,
                    help='Directorio con los audios normalizados (corpus_normalizado)')
    ap.add_argument('--csv', required=True,
                    help='CSV de la revision manual (qtts_trainable_analisis_manual.csv)')
    ap.add_argument('--salida', default='salida_duraciones',
                    help='Directorio de salida')
    ap.add_argument('--sep', default=';', help='Separador del CSV (por defecto ;)')
    ap.add_argument('--col-archivo', default='audio_path',
                    help='Columna con el nombre de archivo')
    ap.add_argument('--col-texto', default='text_audio',
                    help='Columna con la transcripcion')
    ap.add_argument('--col-estado', default='Estado',
                    help='Columna con el estado de la revision manual')
    ap.add_argument('--umbral-min', type=float, default=UMBRAL_MIN,
                    help='Umbral minimo de duracion en segundos')
    args = ap.parse_args()

    if not os.path.isdir(args.audios):
        sys.exit('ERROR: no existe el directorio de audios: %s' % args.audios)
    if not os.path.isfile(args.csv):
        sys.exit('ERROR: no existe el CSV: %s' % args.csv)
    os.makedirs(args.salida, exist_ok=True)

    print('Leyendo CSV de revision manual...')
    df = pd.read_csv(args.csv, sep=args.sep, encoding='utf-8-sig')
    for col in (args.col_archivo, args.col_texto, args.col_estado):
        if col not in df.columns:
            sys.exit('ERROR: el CSV no tiene la columna "%s". Columnas: %s'
                     % (col, list(df.columns)))
    df['categoria'] = df[args.col_estado].apply(clasificar_estado)
    print('  %d filas leidas' % len(df))

    print('\nLeyendo duraciones de los audios...')
    df, faltantes = recolectar_duraciones(df, args.audios, args.col_archivo)
    if faltantes:
        print('\n[AVISO] %d archivos del CSV no se encontraron en el directorio.'
              % len(faltantes))
        print('        Primeros 10: %s' % faltantes[:10])

    sr = df['sample_rate'].dropna().unique()
    print('\nFrecuencias de muestreo encontradas: %s' % sorted(sr.tolist()))
    if len(sr) == 1 and int(sr[0]) == 16000:
        print('  OK: todo el corpus esta a 16 kHz (consistente con R1.2).')
    elif len(sr) > 1:
        print('  ATENCION: hay mas de una frecuencia de muestreo. '
              'Esto contradice el indicador de R1.2 y debe revisarse.')

    # ---- distribucion -----------------------------------------------------
    print('\n' + '=' * 70)
    print('1. DISTRIBUCION DE DURACIONES')
    print('=' * 70)
    descriptivos(df['duracion_s'], 'Corpus completo (todas las categorias)')
    curado = df[df['categoria'] == 'OK']
    descriptivos(curado['duracion_s'], 'Corpus curado (solo correspondencia verificada)')

    print('\n--- Duracion por categoria de la revision manual ---')
    resumen_cat = (df.groupby('categoria')['duracion_s']
                     .agg(archivos='count', total_s='sum', media='mean',
                          mediana='median', maximo='max')
                     .sort_values('archivos', ascending=False))
    resumen_cat['horas'] = (resumen_cat['total_s'] / 3600.0).round(2)
    print(resumen_cat[['archivos', 'horas', 'media', 'mediana', 'maximo']]
          .round(2).to_string())

    # ---- escenarios de umbral --------------------------------------------
    print('\n' + '=' * 70)
    print('2. ESCENARIOS DE UMBRAL MAXIMO (sobre el corpus curado)')
    print('   Umbral minimo aplicado: %.1f s' % args.umbral_min)
    print('=' * 70)
    esc = tabla_escenarios(curado['duracion_s'], umbral_min=args.umbral_min)
    print(esc.to_string(index=False))

    # ---- viabilidad CTC ---------------------------------------------------
    print('\n' + '=' * 70)
    print('3. VIABILIDAD DE CTC (frames de entrada vs. caracteres del objetivo)')
    print('   Supuesto: %.0f frames por segundo (downsampling 320x a 16 kHz)'
          % FRAMES_POR_SEGUNDO)
    print('=' * 70)
    d_ctc, problemas, ajustados = verificar_ctc(curado, args.col_texto)
    print('  archivos evaluados                  : %d' % len(d_ctc))
    print('  con holgura <= 0 (CTC inviable)     : %d' % len(problemas))
    print('  con holgura < 10 frames (al limite) : %d' % len(ajustados))
    if len(problemas):
        print('\n  Archivos con CTC inviable (primeros 15):')
        print(problemas[[args.col_archivo, 'duracion_s', 'n_caracteres',
                         'frames_estimados', 'holgura']]
              .head(15).round(2).to_string(index=False))
        print('\n  -> Estos archivos deben excluirse o habilitarse '
              'ctc_zero_infinity en el entrenamiento.')
    else:
        print('\n  OK: ningun archivo del corpus curado rompe la restriccion de CTC.')

    dur_min_segura = d_ctc['duracion_s'].min() if len(d_ctc) else np.nan
    print('\n  Duracion minima observada en el corpus curado: %.2f s' % dur_min_segura)

    # ---- salidas ----------------------------------------------------------
    print('\n' + '=' * 70)
    print('4. ARCHIVOS GENERADOS')
    print('=' * 70)

    ruta_detalle = os.path.join(args.salida, 'duraciones_por_archivo.csv')
    salida = d_ctc if len(d_ctc) else df
    cols = [args.col_archivo, args.col_texto, 'categoria', 'duracion_s',
            'sample_rate']
    cols += [c for c in ('n_caracteres', 'frames_estimados', 'holgura')
             if c in salida.columns]
    df_out = df.merge(
        salida[[args.col_archivo] + [c for c in ('n_caracteres', 'frames_estimados',
                                                 'holgura') if c in salida.columns]],
        on=args.col_archivo, how='left')
    df_out.to_csv(ruta_detalle, sep=';', index=False, encoding='utf-8-sig')
    print('  %s' % ruta_detalle)

    ruta_esc = os.path.join(args.salida, 'escenarios_umbral.csv')
    esc.to_csv(ruta_esc, sep=';', index=False, encoding='utf-8-sig')
    print('  %s' % ruta_esc)

    ruta_cat = os.path.join(args.salida, 'resumen_por_categoria.csv')
    resumen_cat.round(2).to_csv(ruta_cat, sep=';', encoding='utf-8-sig')
    print('  %s' % ruta_cat)

    ruta_png = os.path.join(args.salida, 'histograma_duraciones.png')
    histograma(curado['duracion_s'], ruta_png)

    print('\nListo.')


if __name__ == '__main__':
    main()