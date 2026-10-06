# qtts_trainable
 
Código y resultados de la tesis **Reconocimiento automático del habla (ASR) para la transcripción del quechua de Abancay**, Ingeniería Informática, PUCP (2026).
 
Este repositorio cubre el **Objetivo 1** (corpus curado y partición) y el **Objetivo 2** (selección de modelo y línea base). El ajuste fino (O3) y la evaluación final (O4) se agregarán después.
 
> **Datos:** el corpus de audio es privado y **no se incluye** en el repositorio (`.gitignore`). Los CSV de particiones y de predicciones sí están, porque contienen las transcripciones y los resultados, no el audio.
 
---
 
## 1. Resumen
 
| Objetivo | Resultado | Descripción | Carpeta | Estado |
|---|---|---|---|---|
| O1 | R1.1 | Corpus etiquetado, auditado y particionado (train/val/test) | `R1.1/` | Completado |
| O1 | R1.2 | Corpus normalizado (frecuencia de muestreo, amplitud, formato) | `R1.2/` | Completado |
| O2 | R2.1 | Comparación de modelos candidatos (zero-shot sobre validación) | `R2.1/` | Completado |
| O2 | R2.2 | Configuración de hiperparámetros del ajuste fino | `R2.2/` | Completado |
| O2 | R2.3 | Línea base zero-shot sobre prueba | `R2.3/` | Completado |
 
---
 
## 2. Estructura del repositorio
 
```
qtts_trainable/
├── README.md
├── requirements.txt
├── qtts_trainable.xlsx          # transcripciones y estado de cada audio
├── R1.1/                        # auditoría, análisis de duraciones y partición
│   ├── analisis_duraciones.py
│   ├── particion_train_val_test.py
│   ├── particiones/             # train.csv, val.csv, test.csv, reporte_particion.csv
│   └── salida_duraciones/       # duraciones_por_archivo.csv, resumen_por_categoria.csv, histograma_duraciones.png, ...
├── R1.2/                        # normalización del corpus
│   ├── normalizacion.py
│   └── log_normalizacion.csv
├── R2.1/                        # predicciones zero-shot sobre validación
│   └── zs_romero_val.csv, zs_mms_quy_val.csv, zs_mms_quz_val.csv
├── R2.2/                        # normalización de texto y evaluador, configuraciones
│   ├── text_norm.py
│   └── eval_zeroshot.py
└── R2.3/                        # predicciones zero-shot sobre prueba (línea base)
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
 
**Auditoría y revisión manual.** Se detectaron archivos corruptos, anómalos o vacíos, y se revisó manualmente una muestra para descartar audios que no fueran quechua o cuya transcripción no estuviera alineada. El estado de cada audio está en `qtts_trainable.xlsx`.
 
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
---
 
## 6. Convenciones de trabajo
 
- Las decisiones se toman con **validación**; la prueba se usa una sola vez por modelo.
- **Misma normalización** (`text_norm.py`) en entrenamiento y evaluación.
- **Un solo factor por variación** experimental, para poder atribuir las diferencias.
- Las decisiones lingüísticas (ortografía, apóstrofo, corrección de referencias) las toma el especialista.
---