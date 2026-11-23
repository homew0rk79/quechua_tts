# Bitácora de experimentos — R3.2

**Objetivo O3.** Adaptar mediante ajuste fino un modelo preentrenado con el corpus de quechua de Abancay.
**Resultado R3.2.** Registro estructurado de al menos tres configuraciones de ajuste fino, con la justificación de la configuración final.
**Estado de este documento:** C1, C2 y C3 completas y verificadas. **Configuración final: C1** (Entrada 12). Quedan pendientes menores (Entradas 6 y 7). Los campos marcados con `PENDIENTE` se completan al terminar cada paso; no se rellenan con estimaciones.

---

## 1. Convenciones

- **Referencia principal de comparación:** WER y CER sobre la validación limpia (152 archivos, sin las fuentes memorizadas por el checkpoint de Romero et al., 2024: `01 TAQAQA RUNA`, `02 TAQAQA RUNAS`, `05_11-20 RUTA QUECHUA`). La validación completa (190 archivos) se reporta como dato secundario.
- **Partición de prueba:** no se usa en O3. Se reserva para O4, una sola vez.
- **Un factor por variación:** C2 cambia solo el checkpoint de partida respecto a C1; C3 cambia solo el dropout respecto a la mejor de C1 y C2.
- **Criterio de diferencia relevante:** no se declara ganador por diferencias menores a ~1 punto de WER. La variación observada entre dos corridas idénticas de C1 (Entrada 5) fue de 0,7–0,9 puntos de WER, lo que respalda este umbral.
- **Normalización:** la misma en todo el flujo (`text_norm.normalizar`, conserva el apóstrofo).
- **Verificación independiente:** además de las métricas calculadas por `train.py`, cada modelo final se evalúa con `eval_zeroshot.py --kind ctc` sobre `val.csv`, y `analizar_pred.py` recalcula WER y CER por subconjunto y por fuente desde esas predicciones. Las dos medidas deben coincidir con diferencias de décimas.

## 2. Entorno

| Elemento | Valor |
|---|---|
| Servidor | `razorcrest` (acceso por servidor puente), Ubuntu 24.04 |
| GPU usada | NVIDIA RTX A6000 (48 GB), `CUDA_VISIBLE_DEVICES=1` |
| Software | Python 3.10.12, torch 2.11.0+cu128, transformers 5.18.0 |
| Semilla | 42 |
| Datos | Entrenamiento: 1 763 archivos (4,38 h). Validación: 190 archivos (152 limpios, 23,6 min) |
| Vocabulario CTC | 37 símbolos (35 del alfabeto normalizado + `[UNK]` + `[PAD]`) |

Nota sobre uso compartido: `nvidia-smi` mostró procesos ajenos en ambas GPU, incluida la GPU 1 (unos 6,2–6,7 GB ocupados por otros usuarios). La GPU 1 no es de uso exclusivo; hay margen de memoria suficiente (pico de la prueba de memoria: 13,7 GB).

Nota sobre los avisos de carga: al cargar los modelos finales aparecen avisos de que `bos_token_id` (37) y `eos_token_id` (38) están fuera del vocabulario. El tokenizador CTC agrega esos tokens por defecto; un modelo CTC nunca los emite. La carga y la inferencia funcionan correctamente (verificado en C1 y C2).

## 3. Resumen de configuraciones

| Config | Modelo de partida | Cambio respecto a C1 | Estado |
|---|---|---|---|
| C1 | `romero-quechua` (Romero et al., 2024) | — (base) | Completa y verificada (repetición tras corregir la métrica). **Configuración final** |
| C2 | `facebook/wav2vec2-xls-r-300m` | Solo el checkpoint de partida | Completa y verificada |
| C3 | `romero-quechua` (C1 resultó mejor) | Dropout de atención, capa oculta y proyección de características en 0,2 | Completa y verificada |

Hiperparámetros comunes (fuente: `config_usada.yaml` de cada corrida): tasa de aprendizaje 1×10⁻⁴ con decaimiento lineal; calentamiento de 170 pasos; lote efectivo de 32 (16 × 2 acumulaciones); 30 épocas (1 680 pasos); evaluación y guardado por época; dropout de 0,1 (feat_proj 0,0); layerdrop 0,1; enmascaramiento temporal de probabilidad 0,05 y longitud 10; extractor convolucional congelado; pérdida CTC con reducción `mean` y `zero_infinity`; bf16; gradient_checkpointing; muestreo `group_by_length`; mejor modelo por WER de validación limpia.

### Comparación de resultados (validación, medida independiente)

| Config | WER limpio | CER limpio | WER completo | CER completo | WER solo excluidas | CER solo excluidas |
|---|---|---|---|---|---|---|
| Romero zero-shot (R2.1) | 41,81 | 8,29 | 35,50 | 7,00 | — | — |
| C1 | **27,88** | **5,54** | **25,94** | **5,09** | 15,18 | 2,71 |
| C2 | 35,47 | 6,61 | 33,79 | 6,24 | 24,42 | 4,24 |
| C3 | 32,33 | 6,20 | 30,27 | 5,72 | 18,81 | 3,17 |

## 4. Registro cronológico

### Entrada 1 — Adaptación de `group_by_length` a transformers 5.18 (2026-11-01)

- **Hallazgo.** La primera ejecución de `train.py` se detuvo en la validación de claves: `group_by_length` ya no es un argumento de `TrainingArguments` en transformers 5.18.
- **Verificación.** Se inspeccionaron los campos de la clase. El reemplazo es `train_sampling_strategy`, cuyos valores válidos son `random`, `sequential`, `group_by_length` y `batch_rebalance`.
- **Decisión.** Reemplazar `group_by_length: true` por `train_sampling_strategy: group_by_length` en C1, C2 y C3.
- **Efecto sobre la metodología.** Ninguno: el comportamiento es equivalente y la decisión de R2.2 no cambia. Es un cambio de nombre de API. La tabla de R2.2 se mantiene como fue redactada antes del entrenamiento.
- **Comprobación posterior.** Todas las corridas ejecutaron sin errores con el nuevo argumento.

### Entrada 2 — Prueba de memoria de C1 (2026-11-01)

- **Procedimiento.** `train.py --prueba_memoria`: un paso hacia adelante y hacia atrás con el lote de los 16 audios más largos (30,1–36,4 s).
- **Resultado.** Memoria pico de 13,7 GB de 47,4 GB (29 %); pérdida de 14,157. La configuración cabe con amplio margen.
- **Observación.** La cabeza CTC se reinicializa por diferencia de tamaño de vocabulario (43 símbolos en el checkpoint de Romero frente a 37 en este corpus). Es el comportamiento previsto.

### Entrada 3 — Corrida C1_v0 (2026-11-01)

- **Ejecución.** 1 680 pasos en 41 min 59 s (`train_runtime` de 2 520 s). El mejor checkpoint fue `checkpoint-1568` (época 28).
- **Interrupción de conexión.** La sesión SSH se cayó hacia la época 16,6. El entrenamiento corría dentro de tmux y continuó sin necesidad de reanudar (el `train_runtime` es coherente con una corrida continua).
- **Comportamiento de la pérdida.** La pérdida de entrenamiento pasó de 25,4 al inicio a unos 10 al terminar la primera época. La evaluación de la época 1 dio WER y CER de 100 % (la cabeza recién reinicializada solo emitía el símbolo en blanco). Hacia el final, la pérdida de entrenamiento rondó 0,3–0,4 y la de validación 0,27.
- **Métricas reportadas por `train.py` (defectuosas, ver Entrada 4):** validación limpia WER 34,58 / CER 10,03; validación completa WER 33,18 / CER 9,98.
- **Métricas de la verificación independiente:** validación completa WER 26,60 / CER 5,13; validación limpia WER 28,77 / CER 5,59; solo excluidas WER 14,52 / CER 2,64.

### Entrada 4 — Defecto en el cálculo de métricas durante el entrenamiento (2026-11-01)

- **Detección.** La verificación independiente de C1_v0 dio WER 26,60 % y CER 5,13 % sobre la validación completa, frente a 33,18 % y 9,98 % reportados por `train.py` para el mismo modelo y los mismos archivos: diferencias de 6,6 y 4,9 puntos.
- **Causa identificada.** En `compute_metrics`, las predicciones (`argmax` de los logits) llegan rellenadas con `-100` cuando el `Trainer` concatena lotes de distinta longitud. Al decodificar, el tokenizador convierte ese valor en `[UNK]`, que queda añadido al final de la transcripción. Se confirmó el mecanismo decodificando un caso sintético: `batch_decode([[11,12,13,-100,-100]])` devuelve `'jkl[UNK]'`. Un `[UNK]` pegado a la última palabra agrega un error de palabra y cinco de carácter por audio afectado.
- **Corrección.** Reemplazar `-100` por el identificador de relleno antes de decodificar las predicciones (igual que ya se hacía con las etiquetas):
  `ids = np.where(pred.predictions == -100, tok.pad_token_id, pred.predictions)`
- **Comprobación de la corrección.** Tras corregir, las métricas de `train.py` y de la verificación independiente coinciden en décimas: C1, validación completa 25,79 / 5,12 frente a 25,94 / 5,09; C2, 33,84 / 6,29 frente a 33,79 / 6,24. **El defecto explicaba la totalidad de la diferencia.**
- **Decisión.** C1_v0 se conserva como registro (carpeta `runs/C1_v0`) pero no se usa para comparar configuraciones. C1 se repitió con la corrección, de modo que C1, C2 y C3 comparten el mismo criterio de selección.
- **Consecuencia sobre interpretaciones previas.** Durante la ejecución de C1_v0 se había observado un aparente aumento del CER respecto a la línea base zero-shot (10,0 frente a 8,3 en la validación limpia). Esa lectura se basaba en la métrica defectuosa y se descarta: con la medida correcta, C1 mejora tanto el WER como el CER.

### Entrada 5 — Repetición de C1 con la métrica corregida (2026-11-01)

- **Ejecución.** 1 680 pasos. El mejor checkpoint volvió a ser `checkpoint-1568` (época 28). Pérdida de validación final: 0,270.
- **Métricas de `train.py`:** validación limpia WER 27,76 / CER 5,57; validación completa WER 25,79 / CER 5,12.
- **Verificación independiente:** validación completa WER 25,94 / CER 5,09 (sin apóstrofo: 24,23 / 4,81); validación limpia WER 27,88 / CER 5,54; solo excluidas WER 15,18 / CER 2,71.
- **Frente a la línea base zero-shot (validación limpia):** el WER baja de 41,81 a 27,88 (−13,9 puntos) y el CER de 8,29 a 5,54 (−2,8 puntos).
- **Variación entre corridas idénticas.** C1_v0 y C1 comparten configuración y semilla; su medida independiente difiere en 0,66 puntos de WER (completa) y 0,89 (limpia). Es una sola observación, pero da un orden de magnitud del ruido entre corridas.
- **Convergencia.** El WER se mantuvo en 100 % durante las épocas 1 a 4 (el CER empezó a bajar en la época 4: 50,6 %), cayó con rapidez entre las épocas 5 y 9 (de 86,6 % a 32,9 %) y luego descendió lentamente hasta ~28 % en las épocas 26 a 30. En las últimas diez épocas mejoró alrededor de 1 punto: la curva se aplana, pero conserva una ligera tendencia descendente. El indicador de R3.1 se cumple por número de épocas cumplido. El mejor checkpoint (época 28, 27,76 %) queda ~0,5 puntos por debajo de sus épocas vecinas (28,29 y 28,59), lo que indica que la selección captura una fluctuación favorable; la cifra imparcial se obtendrá en la partición de prueba (O4).

| Época | 1–3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 15 | 20 | 25 | 28 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WER (%) | 100 | 100 | 86,60 | 44,13 | 40,04 | 35,41 | 32,92 | 32,44 | 31,26 | 29,42 | 28,94 | 27,76 | 28,23 |
| CER (%) | 100 | 50,61 | 15,32 | 8,28 | 7,29 | 6,68 | 6,27 | 6,36 | 6,08 | 5,84 | 5,73 | 5,57 | 5,64 |

### Entrada 6 — Análisis por fuente y archivo anómalo `…-52_P1` (2026-11-01)

- **Hallazgo.** En C1_v0, la fuente `01362597f974f4a90acff783f9c0842c` (20 archivos) tuvo el CER más alto entre las fuentes con más de dos archivos (10,97 %). El desglose por archivo mostró que el exceso proviene de un único archivo, `01362597f974f4a90acff783f9c0842c-52_P1`, con CER de 67,5 %; los otros 19 archivos de la fuente están entre 0,8 % y 9,9 %.
- **Diagnóstico.** El audio dura 6,0 s y la referencia tiene 274 caracteres (~46 caracteres/s), mientras que la predicción tiene 93 caracteres (~15,5 caracteres/s, un ritmo de habla normal). El modelo transcribió con fidelidad razonable el inicio de la referencia y nada del resto. La evidencia indica que la referencia contiene texto que no está en el audio (audio cortado o texto que corresponde a otro segmento). Es un problema de datos, no de reconocimiento.
- **Persistencia.** La fuente mantiene un CER alto en C1 (10,94 %) y en C2 (11,62 %), coherente con un problema de datos que afecta por igual a todas las configuraciones y, por tanto, no altera su comparación.
- **Pendientes.** Revisar la referencia de `…-52_P2` para comprobar si el texto sobrante pertenece a ese segmento; escuchar el audio; registrar el caso en `casos_pendientes_validacion.csv` para el especialista; calcular las métricas limpias con y sin el archivo para cuantificar su peso. El archivo **no** se excluye de la evaluación mientras el especialista no confirme el error; si se corrige, se reevalúan todas las configuraciones con la misma validación corregida.
- **Observación general.** Las fuentes con dos archivos (`18c6c9c0…`, `0d94caca…`) muestran valores extremos que no son interpretables por su tamaño.

### Entrada 7 — Corrida C2 cancelada (2026-11-01)

- Se lanzó C2 y se canceló manualmente tras la primera época (se guardó solo `checkpoint-56`, a las 21:33). Motivo de la cancelación: `PENDIENTE (completar)`.
- La carpeta se conservó como `runs/C2_cancelada` y C2 se repitió desde cero. No se usa para comparar.

### Entrada 8 — Corrida C2 (2026-11-02)

- **Ejecución.** 1 680 pasos en 31 min 55 s (`train_runtime` de 1 915 s). Mejor checkpoint: `checkpoint-1680` (época 30). Pérdida de validación final: 0,304.
- **Convergencia.** El WER se mantuvo en 100 % durante las épocas 1 a 9 (el CER empezó a bajar en la época 7: 43,6 %), cayó con rapidez entre las épocas 10 y 17 (de 69,5 % a 38,0 %) y se estabilizó entre 35,5 % y 36,3 % desde la época 22. Que el mejor checkpoint sea el último refleja una fluctuación dentro de esa meseta, no una mejora sostenida: el modelo convergió dentro del presupuesto de 30 épocas.
- **Métricas de `train.py`:** validación limpia WER 35,47 / CER 6,65; validación completa WER 33,84 / CER 6,29.
- **Verificación independiente:** validación completa WER 33,79 / CER 6,24 (sin apóstrofo: 32,18 / 5,97); validación limpia WER 35,47 / CER 6,61; solo excluidas WER 24,42 / CER 4,24.

| Época | 1–6 | 7 | 8 | 9 | 10 | 11 | 12 | 15 | 17 | 20 | 22 | 25 | 28 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WER (%) | 100 | 100 | 99,94 | 99,94 | 69,51 | 51,30 | 46,80 | 43,77 | 38,02 | 37,96 | 35,71 | 35,77 | 36,12 | 35,47 |
| CER (%) | 100 | 43,62 | 21,88 | 19,39 | 12,18 | 9,45 | 8,69 | 8,06 | 7,29 | 7,06 | 6,83 | 6,72 | 6,74 | 6,65 |

### Entrada 9 — Comparación de C1 y C2 y elección de la base de C3

- **Resultado.** Sobre la validación limpia (medida independiente), C1 supera a C2 por 7,6 puntos de WER (27,88 frente a 35,47) y 1,1 puntos de CER (5,54 frente a 6,61). La diferencia excede ampliamente el criterio de ~1 punto y la variación observada entre corridas idénticas.
- **Interpretación.** Con los mismos datos y los mismos hiperparámetros, partir del checkpoint ajustado previamente en quechua sureño (Romero et al., 2024) produce un mejor modelo que partir de XLS-R-300M base, aun cuando la cabeza CTC se reinicializa en ambos casos. Además, C1 sale de la fase en que solo emite blancos hacia la época 5 (WER por debajo de 50 % en la época 6), mientras que C2 lo hace hacia la época 10 (por debajo de 50 % recién en la época 12): el preentrenamiento en quechua también acelera la convergencia, en unas cinco épocas. *(Corrección: una versión anterior de esta bitácora afirmaba que C1 salía de esa fase tras la primera época; la afirmación no estaba respaldada y se corrigió al extraer la curva de C1.)*
- **Decisión.** C3 se construye sobre C1. Se verificó en `C3.yaml`: `checkpoint: ~/tesis/models/romero-quechua`; `attention_dropout`, `hidden_dropout` y `feat_proj_dropout` en 0,2; `activation_dropout` en 0,1 (sin cambio); `output_dir: ~/tesis/runs/C3`; `train_sampling_strategy: group_by_length`.
- **Sobre las fuentes excluidas.** La diferencia entre la validación limpia y las fuentes excluidas es similar en ambos modelos: 12,7 puntos de WER y 2,8 de CER en C1, y 11,1 puntos de WER y 2,4 de CER en C2. Como C2 nunca pasó por el checkpoint de Romero, la ventaja de esas fuentes se explica en su mayor parte por factores que comparten ambos modelos (fuentes más fáciles, o presencia de esas mismas fuentes en el entrenamiento), no por la memorización de Romero. El remanente atribuible a C1 (~1,6 puntos de WER) no es distinguible del ruido con 38 archivos. Esta es una inferencia: al estar las fuentes en el entrenamiento de ambos modelos, las dos causas no pueden separarse por completo.

### Entrada 10 — Regla de decisión para C3 (fijada antes de ejecutarla)

Medida: WER sobre la validación limpia, verificación independiente.

- Si C3 mejora el WER de C1 en ~1 punto o más, la configuración final es C3.
- Si la diferencia es menor a ~1 punto en cualquier sentido, no se demuestra efecto de la regularización y la configuración final es C1 (la configuración base, derivada directamente de la literatura en R2.2).
- Si C3 empeora en ~1 punto o más, la configuración final es C1.
- El CER se reporta en todos los casos y se usa para matizar la conclusión, no para cambiarla.

### Entrada 11 — Corrida C3

- **Ejecución.** 1 680 pasos en 32 min 7 s (`train_runtime` de 1 927 s). Mejor checkpoint: `checkpoint-1232` (época 22). Pérdida de validación final: 0,302. La pérdida de entrenamiento se mantuvo en ~0,5–0,6 al final, frente a ~0,3–0,4 en C1, coherente con la mayor regularización.
- **Convergencia.** El WER se mantuvo en 100 % durante las épocas 1 a 5 (el CER empezó a bajar en la época 4), cayó entre las épocas 6 y 15, y desde la época 15 osciló entre 32,3 % y 34,3 % sin tendencia clara. La curva es más irregular que la de C1. El modelo convergió dentro del presupuesto de 30 épocas a un nivel peor que C1.
- **Métricas de `train.py`:** validación limpia WER 32,33 / CER 6,22; validación completa WER 30,37 / CER 5,76.
- **Verificación independiente:** validación completa WER 30,27 / CER 5,72 (sin apóstrofo: 28,61 / 5,42); validación limpia WER 32,33 / CER 6,20; solo excluidas WER 18,81 / CER 3,17. Coincidencia con `train.py`: diferencias ≤ 0,10.

| Época | 1–3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 15 | 20 | 22 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WER (%) | 100 | 100 | 99,88 | 52,97 | 44,96 | 40,45 | 38,61 | 37,37 | 32,38 | 34,34 | 32,33 | 32,98 | 32,92 |
| CER (%) | 100 | 76,02 | 20,65 | 10,27 | 8,46 | 7,58 | 7,27 | 7,33 | 6,41 | 6,53 | 6,22 | 6,48 | 6,58 |

### Entrada 12 — Decisión final

- **Aplicación de la regla (Entrada 10).** Sobre la validación limpia (medida independiente), C3 empeora el WER de C1 en 4,45 puntos (32,33 frente a 27,88) y el CER en 0,66 (6,20 frente a 5,54). La diferencia supera el umbral de ~1 punto y la variación entre corridas idénticas (0,7–0,9). Corresponde al tercer caso de la regla: **la configuración final es C1**.
- **Interpretación.** C1 no mostraba sobreajuste: su pérdida de validación (0,27) fue menor que la de entrenamiento (~0,3–0,4) al final del entrenamiento. Sin sobreajuste que corregir, la regularización adicional redujo la capacidad de ajuste del modelo con 4,4 h de datos, sin mejorar la generalización.
- **Limitación.** C3 modifica tres dropouts a la vez (atención y capa oculta de 0,1 a 0,2; proyección de características de 0,0 a 0,2). El diseño trata la regularización como un solo factor, por lo que el deterioro no puede atribuirse a ninguno de ellos en particular.
- **Sin configuraciones adicionales.** No se agregaron configuraciones posteriores (por ejemplo, un dropout intermedio) para no ampliar la búsqueda sobre la misma validación, lo que haría más optimista la cifra de la configuración elegida. La exploración de valores intermedios queda como trabajo futuro.

## 5. Fichas por configuración

### C1 (repetición con métrica corregida)

| Campo | Valor |
|---|---|
| Parámetros | Ver `runs/C1/config_usada.yaml` |
| Épocas efectivas | 30 (1 680 pasos) |
| Mejor checkpoint | `checkpoint-1568` (época 28) |
| Pérdida de validación final | 0,270 |
| WER / CER validación limpia (`train.py`) | 27,76 / 5,57 |
| WER / CER validación completa (`train.py`) | 25,79 / 5,12 |
| WER / CER validación limpia (independiente) | 27,88 / 5,54 |
| WER / CER validación completa (independiente) | 25,94 / 5,09 |
| WER / CER solo excluidas (independiente) | 15,18 / 2,71 |
| Coincidencia entre ambas medidas | Sí (diferencias ≤ 0,15) |
| Curva por época | Entrada 5 |

### C2

| Campo | Valor |
|---|---|
| Parámetros | Ver `runs/C2/config_usada.yaml` |
| Épocas efectivas | 30 (1 680 pasos), 31 min 55 s |
| Mejor checkpoint | `checkpoint-1680` (época 30; meseta desde la época 22) |
| Pérdida de validación final | 0,304 |
| WER / CER validación limpia (`train.py`) | 35,47 / 6,65 |
| WER / CER validación completa (`train.py`) | 33,84 / 6,29 |
| WER / CER validación limpia (independiente) | 35,47 / 6,61 |
| WER / CER validación completa (independiente) | 33,79 / 6,24 |
| WER / CER solo excluidas (independiente) | 24,42 / 4,24 |
| Coincidencia entre ambas medidas | Sí (diferencias ≤ 0,05) |
| Curva por época | Entrada 8 |

### C3

| Campo | Valor |
|---|---|
| Parámetros | Ver `runs/C3/config_usada.yaml` |
| Épocas efectivas | 30 (1 680 pasos), 32 min 7 s |
| Mejor checkpoint | `checkpoint-1232` (época 22; oscilación sin tendencia desde la época 15) |
| Pérdida de validación final | 0,302 |
| WER / CER validación limpia (`train.py`) | 32,33 / 6,22 |
| WER / CER validación completa (`train.py`) | 30,37 / 5,76 |
| WER / CER validación limpia (independiente) | 32,33 / 6,20 |
| WER / CER validación completa (independiente) | 30,27 / 5,72 |
| WER / CER solo excluidas (independiente) | 18,81 / 3,17 |
| Coincidencia entre ambas medidas | Sí (diferencias ≤ 0,10) |
| Curva por época | Entrada 11 |

### C1_v0 (descartada para comparación, conservada como registro)

| Campo | Valor |
|---|---|
| Épocas efectivas | 30 (1 680 pasos) |
| Mejor checkpoint | `checkpoint-1568` (época 28) |
| WER / CER validación limpia (`train.py`, defectuoso) | 34,58 / 10,03 |
| WER / CER validación completa (`train.py`, defectuoso) | 33,18 / 9,98 |
| WER / CER validación limpia (independiente) | 28,77 / 5,59 |
| WER / CER validación completa (independiente) | 26,60 / 5,13 |
| WER / CER solo excluidas (independiente) | 14,52 / 2,64 |
| Observaciones | Ver Entrada 4 |

## 6. Observaciones lingüísticas preliminares

No se incluyen fragmentos del corpus en este documento por ser privado.

- El apóstrofo (marca de glotalización) cuesta 1,7 puntos de WER en C1 (25,94 con apóstrofo, 24,23 sin él), 1,6 en C2 (33,79 frente a 32,18) y 1,7 en C3 (30,27 frente a 28,61) sobre la validación completa. El efecto es similar en las tres configuraciones.
- Entre los errores observados en una muestra de ejemplos hay fusiones y divisiones de palabras aglutinantes y variaciones de ortografía en préstamos del español, coherentes con las 44 palabras de grafía inconsistente señaladas en el corpus. Es una cuestión para el especialista, no para el entrenamiento.
- Estas observaciones son cualitativas. El análisis cuantitativo por carácter está `PENDIENTE`.

## 7. Justificación de la configuración final

- **(a) C1 frente a C2:** resuelto (Entrada 9). C1 es mejor por 7,6 puntos de WER y 1,1 de CER en la validación limpia.
- **(b) Efecto de la regularización de C3:** resuelto (Entradas 11 y 12). Aumentar el dropout empeora el modelo en 4,45 puntos de WER y 0,66 de CER sobre la validación limpia.
- **(c) Configuración elegida y por qué:** **C1**. Obtiene el menor WER y CER de las tres configuraciones sobre la validación limpia (27,88 % y 5,54 %), con diferencias frente a C2 y C3 superiores al umbral fijado y a la variación entre corridas idénticas. La decisión se tomó con una regla fijada antes de ejecutar C3. La cifra de 27,88 % es ligeramente optimista porque el checkpoint se eligió sobre esa misma validación; la estimación imparcial se obtendrá una sola vez en la partición de prueba (O4).
- **(d) Estado del modelo final:** `runs/C1/final` carga y transcribe de forma independiente del entrenamiento (verificación de R3.1), con métricas coincidentes con las de `train.py` en décimas.

## 8. Archivos de evidencia

| Archivo | Origen |
|---|---|
| `config_usada.yaml` | `runs/<C>/` |
| `metricas_val.json` | `runs/<C>/` |
| `trainer_state.json` (curvas de pérdida y métricas por época) | `runs/<C>/checkpoint-*/` (el de número más alto) |
| `pred_val.csv` (predicciones de la verificación independiente) | `runs/<C>/` |
| `pred_val_por_fuente.csv` | salida de `analizar_pred.py` |
| Registros de C1_v0 y C2_cancelada | `runs/C1_v0/`, `runs/C2_cancelada/` |

Los checkpoints no se descargan al repositorio; se conservan en el servidor.
