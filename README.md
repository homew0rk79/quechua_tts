# quechua_tts
 
Código y resultados de la tesis **Reconocimiento automático del habla (ASR) para la transcripción del quechua de Abancay**, Ingeniería Informática, PUCP (2026).
 
Este repositorio cubre los cuatro objetivos de la tesis: el corpus curado y su partición (O1), la selección del modelo y la línea base (O2), el ajuste fino (O3) y la evaluación objetiva y subjetiva del modelo resultante (O4).
 
> **Datos:** el corpus de audio es privado y **no se incluye** en el repositorio (`.gitignore`). Los CSV de particiones y de predicciones sí están, porque contienen las transcripciones y los resultados, no el audio. Los pesos del modelo ajustado tampoco se incluyen por su tamaño; se conservan en el servidor de cómputo institucional (`runs/C1/final`).
 
---
 
## 1. Resumen
 
| Objetivo | Resultado | Descripción | Carpeta | Estado |
|---|---|---|---|---|
| O1 | R1.1 | Corpus etiquetado, auditado y particionado (train/val/test) | `R1.1/` | Completado |
| O1 | R1.2 | Corpus normalizado (frecuencia de muestreo, amplitud, formato) | `R1.2/` | Completado |
| O2 | R2.1 | Comparación de modelos candidatos (zero-shot sobre validación) | `R2.1/` | Completado |
| O2 | R2.2 | Configuración de hiperparámetros del ajuste fino | `R2.2/` | Completado |
| O2 | R2.3 | Línea base zero-shot sobre prueba | `R2.3/` | Completado |
| O3 | R3.1 | Modelo ajustado (C1), entrenado hasta convergencia | `R3.1/` | Completado |
| O3 | R3.2 | Bitácora de experimentos (C1, C2, C3) y configuración final | `R3.2/` | Completado |
| O4 | R4.1 | WER y CER del modelo ajustado sobre prueba frente a la línea base | `R4.1/` | Completado |
| O4 | R4.2 | Evaluación subjetiva con hablantes nativos del quechua de Abancay | `R4.2/` | Completado |
| O4 | R4.3 | Categorización de errores y recomendaciones | `R4.3/` | Completado |
 
---
 
## 2. Estructura del repositorio
 
```
quechua_tts/
├── README.md
├── qtts_trainable.xlsx          # transcripciones y estado de cada audio
├── R1.1/                        # auditoría, análisis de duraciones y partición
│   ├── qtts_trainable_analisis_manual.csv   # registro de la revisión manual (2 312 archivos)
│   ├── analisis_duraciones.py
│   ├── particion_train_val_test.py
│   ├── particiones/             # train.csv, val.csv, test.csv, reporte_particion.csv
│   └── salida_duraciones/       # duraciones_por_archivo.csv, resumen_por_categoria.csv, histograma_duraciones.png, ...
├── R1.2/                        # normalización del corpus
│   ├── normalizacion.py
│   └── log_normalizacion.csv
├── R2.1/                        # predicciones zero-shot sobre validación
│   └── zs_romero_val.csv, zs_mms_quy_val.csv, zs_mms_quz_val.csv, zs_whisper_val.csv
├── R2.2/                        # normalización de texto, evaluador y configuraciones
│   ├── text_norm.py
│   ├── eval_zeroshot.py
│   └── configs/                 # C1.yaml, C2.yaml, C3.yaml
├── R2.3/                        # predicciones zero-shot sobre prueba (línea base)
│   └── zs_romero_test.csv
├── R3.1/                        # ajuste fino de C1 y prueba de carga e inferencia
│   ├── train.py, graficar_curvas.py
│   ├── trainer_state_C1.json, curvas_C1.csv, curvas_C1.png
│   └── pred_val_C1.csv
├── R3.2/                        # bitácora y resultados por configuración
│   ├── bitacora_R3.2.md
│   ├── analizar_pred.py, graficar_curvas.py, comparacion.png
│   └── C1/, C2/, C3/, C1_v0/, C2_cancelada/   # config_usada.yaml, metricas_val.json, predicciones, ...
├── R4.1/                        # evaluación final sobre prueba
│   ├── evaluar_prueba.py
│   ├── pred_test.csv, zs_romero_test.csv
│   └── tabla_global.csv, por_fuente.csv, bootstrap.csv
├── R4.2/                        # evaluación subjetiva
│   ├── preparar_items.py, app.py, response_store.py, analizar_r42.py
│   ├── items/                   # items.csv, estratos.json, lista_audios.txt
│   └── resultados/              # resumen.json, por_item.csv, por_estrato.csv, post_edicion.csv
│       └── recopilados/         # E2.jsonl, E4.jsonl, E6.jsonl, E7.jsonl
└── R4.3/                        # análisis de errores
    ├── categorizar_errores.py
    └── resultados/              # operaciones.csv, categorias_sustitucion.csv, castellano_vs_quechua.csv, segmentacion.csv
```
 
---
 
## 3. Entorno
 
| Componente | Versión |
|---|---|
| Python | 3.10 |
| torch / torchaudio | 2.11.0+cu128 |
| transformers | 5.18.0 |
| datasets | 5.0.1 |
| huggingface_hub | 1.33.0 |
| Otros | evaluate, jiwer, accelerate, soundfile, librosa, pandas, numpy, tensorboard |
 
Los experimentos se ejecutaron en un servidor con 2× NVIDIA RTX A6000 (48 GB), CUDA 12.8, usando solo una GPU (`CUDA_VISIBLE_DEVICES=1`) y en secuencia.
 
---
 
## 4. Objetivo 1: corpus curado y partición
 
### 4.1 El corpus
 
- 2 211 archivos de audio, 5 h 32 min.
- 41 fuentes: 17 con nombre descriptivo (agrupadas en 6 series) y 24 con código hexadecimal y sufijo `_P1`/`_P2`.
- La fuente se deriva del prefijo del nombre del archivo, antes del último guion (`01 TAQAQA RUNA-10.wav` → `01 TAQAQA RUNA`).
### 4.2 R1.2: normalización (`R1.2/normalizacion.py`)
 
Convierte los audios originales al formato de entrada de los modelos: **WAV, 16 kHz, mono**, con normalización de amplitud. Cada archivo procesado queda registrado en `log_normalizacion.csv`.
 
```bash
python R1.2/normalizacion.py
```
 
### 4.3 R1.1: auditoría, análisis de duraciones y partición
 
**Auditoría y revisión manual.** Se revisó manualmente la totalidad de los 2 312 archivos, escuchando cada grabación y contrastándola con su transcripción, para descartar audios que no fueran quechua o cuya transcripción no estuviera alineada. La auditoría automatizada detectó además archivos corruptos, duraciones anómalas y duplicados. El estado de cada audio está en `R1.1/qtts_trainable_analisis_manual.csv` y en `qtts_trainable.xlsx`.
 
**Análisis de duraciones** (`R1.1/analisis_duraciones.py`): calcula la duración de cada archivo y genera `salida_duraciones/` con el resumen por categoría, escenarios de umbral y el histograma de duraciones.
 
**Partición** (`R1.1/particion_train_val_test.py`): divide el corpus en entrenamiento, validación y prueba con **semilla 99**.
 
| Partición | Archivos | % (archivos) | Duración | Fuentes |
|---|---|---|---|---|
| Entrenamiento | 1 763 | 79,7 % | 4 h 23 min | 37 de 41 |
| Validación | 190 | 8,6 % | 27 min 54 s | 20 de 41 |
| Prueba | 258 | 11,7 % | 41 min 25 s | 20 de 41 |
 
Cuatro fuentes pequeñas aparecen solo en validación o prueba, para medir generalización a fuentes no vistas.
 
**Formato de los CSV de partición** (`R1.1/particiones/`):
 
| Columna | Contenido |
|---|---|
| `audio_name` | Nombre del archivo |
| `audio_path` | Solo el nombre del archivo; los scripts lo buscan recursivamente dentro de `corpus_normalizado/` |
| `text_audio` | Transcripción de referencia |
| `gender` | Género del hablante |
| `fuente` | Fuente derivada del nombre |

---
 
## 5. Objetivo 2: selección de modelo y línea base
 
### 5.1 Normalización de texto (`R2.2/text_norm.py`)
 
Función `normalizar(texto, sin_apostrofo=False)`, compartida por entrenamiento y evaluación:
 
- convierte a minúsculas;
- elimina puntuación (`¿ ? , . ; ( ) - “ ”`, etc.);
- unifica las variantes del apóstrofo (`’ ‘ ´`) en `'`;
- **conserva el apóstrofo**, salvo con `sin_apostrofo=True` (solo para la métrica complementaria).
### 5.2 Evaluador zero-shot (`R2.2/eval_zeroshot.py`)
 
Evalúa un modelo sobre una partición y guarda las predicciones.
 
```bash
python eval_zeroshot.py --kind ctc \
    --model <ruta_o_id_del_modelo> \
    --csv R1.1/particiones/val.csv \
    --audio_root <ruta>/corpus_normalizado \
    --out R2.1/zs_romero_val.csv
```
 
| `--kind` | Uso | Parámetros extra |
|---|---|---|
| `ctc` | Checkpoint Wav2Vec2-CTC ya ajustado | — |
| `mms` | MMS con adaptador de lengua | `--lang quy` (o `quz`) |
| `whisper` | Whisper seq2seq | `--lang` opcional |
 
Salida: WER y CER (con y sin apóstrofo), número de transcripciones vacías, 5 ejemplos, y un CSV con las columnas `hyp_raw`, `ref` y `hyp`.
 
### 5.3 R2.1: comparación de modelos (zero-shot, validación)
 
| Modelo | Parámetros | WER (%) | CER (%) |
|---|---|---|---|
| XLS-R-300M ajustado en quechua (Romero et al., 2024) | 300 M | **35,50** | **7,00** |
| MMS-1b-all, adaptador `quy` (Ayacucho) | 1 000 M | 40,52 | 8,71 |
| MMS-1b-all, adaptador `quz` (Cusco) | 1 000 M | 45,15 | 8,79 |
| Whisper large-v3 | 1 550 M | 161,39 | 70,71 |
 
- **Romero** obtuvo el mejor resultado y fue elegido como punto de partida del ajuste fino.
- **MMS** es competitivo en CER, pero peor en WER por diferencias de convención ortográfica.
- **Whisper** falló: bucles de repetición (23 archivos), confusión de lengua con salida en cirílico (14) y fragmentación de palabras aglutinantes. Sin los bucles: WER 112,08 % y CER 53,10 %.
- Los errores de Romero y MMS son principalmente de convención ortográfica (e/o frente a i/u, q/k, aspiradas, segmentación), no acústicos.
- El adaptador `quc` de MMS corresponde a k'iche' (lengua maya), no a quechua.
- XLS-R-300M no incluyó horas de quechua en su preentrenamiento (verificado).
### 5.4 R2.3: línea base (zero-shot, prueba)
 
Modelo de Romero sin ajuste fino sobre la partición de prueba (258 archivos): **WER 39,21 %, CER 7,64 %**.
  
### 5.5 R2.2: configuración del ajuste fino
 
**C1 (base)**, partiendo del checkpoint de Romero:
 
| Parámetro | Valor |
|---|---|
| Vocabulario | Nuevo, construido desde train normalizado; cabeza CTC reinicializada |
| Tasa de aprendizaje | 1×10⁻⁴, decaimiento lineal (reducida desde 3×10⁻⁴ por partir de un modelo ya ajustado) |
| Calentamiento | 170 pasos (~10 % de ~1 700) |
| Lote efectivo | 32 (16 por dispositivo × 2 de acumulación) |
| Épocas | 30; evaluación por época; mejor checkpoint por WER de validación |
| Dropout (atención / oculta / activación) | 0,1 |
| `feat_proj_dropout` / `layerdrop` | 0,0 / 0,1 |
| Enmascaramiento temporal | prob 0,05, longitud 10 |
| Extractor convolucional | Congelado |
| Pérdida | CTC, reducción `mean`, `ctc_zero_infinity=True` |
| Precisión y memoria | bf16, `gradient_checkpointing`, `group_by_length` |
| Otros | semilla 42, `dataloader_num_workers=4`, `save_total_limit=2` |
 
**Variaciones** (cada una cambia un solo factor respecto a C1):
 
- **C2:** igual a C1, pero partiendo de `facebook/wav2vec2-xls-r-300m` (mide el aporte del quechua previo).
- **C3:** la mejor entre C1 y C2, con dropout 0,2 en atención, oculta y `feat_proj`.

En la versión 5.18 de `transformers`, `group_by_length` se configura con `train_sampling_strategy: group_by_length` (cambio de nombre de la API, mismo comportamiento).

---

## 6. Objetivo 3: ajuste fino

### 6.1 R3.1: modelo ajustado (`R3.1/train.py`)

Lee la configuración YAML, aplica `text_norm.normalizar`, construye el vocabulario (37 símbolos) desde la partición de entrenamiento, reinicializa la cabeza CTC y entrena. Solo usa entrenamiento y validación.

```bash
python R3.1/train.py --config R2.2/configs/C1.yaml --prueba_memoria   # un paso con el lote más largo
python R3.1/train.py --config R2.2/configs/C1.yaml                    # entrenamiento completo
python R3.1/graficar_curvas.py C1 --linea_base 41.81 8.29              # curvas de pérdida y WER/CER
```

- 30 épocas (1 680 pasos); mejor checkpoint en la época 28; pérdida de validación 0,270.
- **Prueba de carga e inferencia:** el modelo final se cargó desde su directorio y se evaluó con `eval_zeroshot.py` sobre validación (`pred_val_C1.csv`): WER 25,94 % y CER 5,09 % en la partición completa, frente a 25,79 % y 5,12 % calculados durante el entrenamiento.

### 6.2 R3.2: comparación de configuraciones (validación)

Cada modelo final se evaluó con `eval_zeroshot.py`, y `R3.2/analizar_pred.py` desagrega las métricas por subconjunto y por fuente.

| Config. | Pérdida val. | WER sin contaminadas | CER sin contaminadas | WER completo | CER completo |
|---|---|---|---|---|---|
| C1 (Romero, base) | 0,270 | **27,88** | **5,54** | **25,94** | **5,09** |
| C2 (XLS-R-300M base) | 0,304 | 35,47 | 6,61 | 33,79 | 6,24 |
| C3 (C1 + dropout 0,2) | 0,290 | 32,33 | 6,20 | 30,27 | 5,72 |

**Configuración final: C1.** El razonamiento completo, junto con las corridas `C1_v0` (métrica corregida) y `C2_cancelada`, está en `R3.2/bitacora_R3.2.md`.

---

## 7. Objetivo 4: evaluación

Las fuentes contaminadas (`01 TAQAQA RUNA`, `02 TAQAQA RUNAS`, `05_11-20 RUTA QUECHUA`) se excluyen de la medida principal, porque el checkpoint de Romero las transcribe casi sin error.

### 7.1 R4.1: métricas sobre prueba (`R4.1/evaluar_prueba.py`)

La partición de prueba se evaluó **una sola vez** con el modelo final de R3.1. Las predicciones se generaron con `eval_zeroshot.py` (`pred_test.csv`); la línea base reutiliza las de R2.3. `evaluar_prueba.py` no ejecuta ningún modelo: compara ambos archivos y calcula un bootstrap pareado (10 000 remuestras, semilla 42).

```bash
python R4.1/evaluar_prueba.py --c1 R4.1/pred_test.csv --base R4.1/zs_romero_test.csv \
    --train_csv R1.1/particiones/train.csv --out R4.1
```

| Subconjunto | Archivos | WER línea base | WER ajustado | CER línea base | CER ajustado |
|---|---|---|---|---|---|
| **Sin contaminadas** | 221 | 42,64 | **30,84** | 8,43 | **5,82** |
| Completo | 258 | 39,21 | 29,84 | 7,64 | 5,58 |
| Fuentes no vistas | 92 | 46,32 | 34,90 | 9,97 | 6,90 |
| Solo contaminadas | 37 | 1,44 | 18,77 | 0,26 | 3,36 |

Reducción del WER sin contaminadas: **11,8 puntos (IC 95 %: 9,5 a 14,1)**.

### 7.2 R4.2: evaluación subjetiva (`R4.2/`)

- `preparar_items.py` selecciona 18 ítems del modelo (6 por estrato de WER) y 6 controles con la transcripción humana, del subconjunto de prueba sin contaminadas.
- `app.py` es la aplicación Gradio del cuestionario. No ejecuta el modelo: muestra las transcripciones precalculadas.
- `analizar_r42.py` aplica la regla de decisión, fijada antes de analizar las respuestas.

```bash
python R4.2/preparar_items.py --pred R4.1/pred_test.csv --out R4.2/items --seed 42
python R4.2/analizar_r42.py --respuestas R4.2/resultados/recopilados --items R4.2/items/items.csv \
    --estratos R4.2/items/estratos.json --out R4.2/resultados
```

Tres evaluadores completaron los 24 ítems. El 100 % de los ítems del modelo resultó aceptable, con una mediana de utilidad de 5. La concordancia (alfa de Krippendorff: 0,21 en utilidad y 0,29 en inteligibilidad) quedó por debajo de 0,4, por lo que el veredicto es **satisfactorio con cautela**. Las respuestas no contienen datos que identifiquen a los evaluadores.

### 7.3 R4.3: análisis de errores (`R4.3/categorizar_errores.py`)

```bash
python R4.3/categorizar_errores.py --pred R4.1/pred_test.csv --out R4.3/resultados
```

Sobre el subconjunto sin contaminadas (221 grabaciones, 3 051 palabras):

- 84,7 % de los errores son sustituciones.
- Las sustituciones se concentran en cambios de sufijo sobre una raíz correcta (28,5 %), en variaciones ortográficas y en palabras de origen castellano (14,7 %).
- Las palabras castellanas tienen más error que las quechuas: 36,2 % frente a 27,8 %.

---
 
## 8. Convenciones de trabajo
 
- Las decisiones se toman con **validación**; la prueba se usa una sola vez por modelo.
- **Misma normalización** (`text_norm.py`) en entrenamiento y evaluación.
- **Un solo factor por variación** experimental, para poder atribuir las diferencias.
- Las decisiones lingüísticas (ortografía, apóstrofo, corrección de referencias) las toma el especialista.
---