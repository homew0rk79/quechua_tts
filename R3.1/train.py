"""Ajuste fino de Wav2Vec2-CTC para quechua de Abancay (R2.2 / R3.2).

Uso (desde ~/tesis/scripts, dentro de tmux):
  # 1) Prueba de memoria: un paso con el lote de mayor duración, sin entrenar
  CUDA_VISIBLE_DEVICES=1 python train.py --config ~/tesis/configs/C1.yaml --prueba_memoria
  # 2) Entrenamiento completo
  CUDA_VISIBLE_DEVICES=1 python train.py --config ~/tesis/configs/C1.yaml
  # 3) Reanudar desde el último checkpoint (si se interrumpió)
  CUDA_VISIBLE_DEVICES=1 python train.py --config ~/tesis/configs/C1.yaml --reanudar

Solo usa train y val. La partición de prueba se evalúa aparte (objetivo O4).
"""
import argparse, dataclasses, glob, json, os, shutil, sys, time
import jiwer, numpy as np, pandas as pd, soundfile as sf, torch, yaml
from transformers import (Trainer, TrainingArguments, Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor)
from text_norm import normalizar

SR = 16000

# ---------------------------------------------------------------- configuración
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--prueba_memoria", action="store_true")
ap.add_argument("--reanudar", action="store_true")
a = ap.parse_args()

def expandir(x):
    if isinstance(x, dict):
        return {k: expandir(v) for k, v in x.items()}
    return os.path.expanduser(x) if isinstance(x, str) and x.startswith("~") else x

ruta_cfg = os.path.expanduser(a.config)
cfg = expandir(yaml.safe_load(open(ruta_cfg, encoding="utf-8")))
m, dat, cm, ent = cfg["modelo"], cfg["datos"], cfg["config_modelo"], dict(cfg["entrenamiento"])

# Validar claves de entrenamiento antes de cargar nada (falla rápido)
validas = {f.name for f in dataclasses.fields(TrainingArguments)}
desconocidas = sorted(set(ent) - validas)
if desconocidas:
    sys.exit(f"ERROR: claves no reconocidas por TrainingArguments: {desconocidas}")
out_dir = ent["output_dir"]
os.makedirs(out_dir, exist_ok=True)
shutil.copy(ruta_cfg, os.path.join(out_dir, "config_usada.yaml"))  # trazabilidad

# ---------------------------------------------------------------- datos
indice = {}
for raiz, _, archivos in os.walk(dat["audio_root"]):
    for f in archivos:
        if f.lower().endswith(".wav"):
            indice.setdefault(f, os.path.join(raiz, f))

def cargar(csv):
    df = pd.read_csv(csv)
    faltan = [p for p in df.audio_path if p not in indice]
    if faltan:
        sys.exit(f"ERROR: {len(faltan)} audios de {csv} no encontrados. Ej.: {faltan[:3]}")
    df["texto"] = df.text_audio.map(normalizar)
    vacias = df[df.texto == ""]
    if len(vacias):
        sys.exit(f"ERROR: {len(vacias)} transcripciones vacías tras normalizar en {csv}")
    return df.reset_index(drop=True)

df_tr, df_va = cargar(dat["train_csv"]), cargar(dat["val_csv"])
excluidas = dat.get("fuentes_excluidas_eval", []) or []
desconocidas_f = sorted(set(excluidas) - set(df_va.fuente))
if desconocidas_f:
    sys.exit(f"ERROR: fuentes a excluir que no existen en val: {desconocidas_f}")
df_va_limpio = df_va[~df_va.fuente.isin(excluidas)].reset_index(drop=True)
print(f"Validación: {len(df_va)} archivos | sin fuentes excluidas: {len(df_va_limpio)}")

# ---------------------------------------------------------------- vocabulario
if m.get("vocabulario") != "nuevo":
    sys.exit("ERROR: solo está implementado vocabulario: nuevo")
chars = sorted(set("".join(df_tr.texto)))
fuera = set("".join(df_va.texto)) - set(chars)
if fuera:
    sys.exit(f"ERROR: caracteres de val ausentes en train: {fuera}")
vocab = {c: i for i, c in enumerate(c if c != " " else "|" for c in chars)}
vocab["[UNK]"], vocab["[PAD]"] = len(vocab), len(vocab) + 1
ruta_vocab = os.path.join(out_dir, "vocab.json")
json.dump(vocab, open(ruta_vocab, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Vocabulario: {len(vocab)} símbolos -> {ruta_vocab}")

tok = Wav2Vec2CTCTokenizer(ruta_vocab, unk_token="[UNK]", pad_token="[PAD]",
                           word_delimiter_token="|")
fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR, padding_value=0.0,
                              do_normalize=True, return_attention_mask=True)
proc = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)

# ---------------------------------------------------------------- dataset
class Corpus(torch.utils.data.Dataset):
    """Precarga los audios en RAM (~1 GB para train) y tokeniza las referencias."""
    def __init__(self, df):
        self.audio = [sf.read(indice[p], dtype="float32")[0] for p in df.audio_path]
        for p, x in zip(df.audio_path, self.audio):
            if x.ndim != 1:
                sys.exit(f"ERROR: {p} no es mono")
        self.labels = [tok(t).input_ids for t in df.texto]
        self.dur = np.array([len(x) / SR for x in self.audio])
    def __len__(self):
        return len(self.audio)
    def __getitem__(self, i):
        x = fe(self.audio[i], sampling_rate=SR).input_values[0]
        return {"input_values": x, "labels": self.labels[i]}

t0 = time.time()
ds_tr, ds_va, ds_va_completo = Corpus(df_tr), Corpus(df_va_limpio), Corpus(df_va)
print(f"Audios cargados en {time.time()-t0:.0f} s | train {len(ds_tr)} ({ds_tr.dur.sum()/3600:.2f} h)"
      f" | val {len(ds_va)} ({ds_va.dur.sum()/60:.1f} min)")

def collator(feats):
    x = fe.pad([{"input_values": f["input_values"]} for f in feats], padding=True,
               return_tensors="pt", return_attention_mask=True)
    y = tok.pad({"input_ids": [f["labels"] for f in feats]}, padding=True, return_tensors="pt")
    x["labels"] = y.input_ids.masked_fill(y.attention_mask.ne(1), -100)
    return x

# ---------------------------------------------------------------- modelo
model = Wav2Vec2ForCTC.from_pretrained(
    m["checkpoint"], vocab_size=len(vocab), pad_token_id=tok.pad_token_id,
    ignore_mismatched_sizes=True, **cm)
if m.get("reinicializar_cabeza_ctc", True):
    # Reinicialización explícita: no depender de que los tamaños de vocabulario difieran
    model.lm_head = torch.nn.Linear(model.lm_head.in_features, len(vocab))
    model.config.vocab_size = len(vocab)
if m.get("congelar_extractor_convolucional", True):
    model.freeze_feature_encoder()
entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Modelo: {m['checkpoint']} | parámetros entrenables: {entrenables/1e6:.1f} M")

# ---------------------------------------------------------------- prueba de memoria
if a.prueba_memoria:
    bs = ent["per_device_train_batch_size"]
    idx = np.argsort(-ds_tr.dur)[:bs]
    print(f"Prueba de memoria: lote de {bs} audios más largos "
          f"({ds_tr.dur[idx].min():.1f}–{ds_tr.dur[idx].max():.1f} s)")
    model.to("cuda").train()
    if ent.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    lote = {k: v.to("cuda") for k, v in collator([ds_tr[int(i)] for i in idx]).items()}
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(ent.get("bf16"))):
        perdida = model(**lote).loss
    perdida.backward()
    pico = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Pérdida: {perdida.item():.3f} | memoria pico: {pico:.1f} GB de {total:.1f} GB "
          f"({100*pico/total:.0f} %)")
    print("OK: la configuración cabe en memoria." if pico < 0.9 * total
          else "ATENCIÓN: margen menor al 10 %; reducir el lote por dispositivo.")
    sys.exit(0)

# ---------------------------------------------------------------- métricas
def preprocesar_logits(logits, labels):
    return logits.argmax(-1)  # evita acumular logits completos en memoria

def metricas(pred):
    ids = np.where(pred.predictions == -100, tok.pad_token_id, pred.predictions)
    lab = np.where(pred.label_ids == -100, tok.pad_token_id, pred.label_ids)
    hyp = tok.batch_decode(ids)
    ref = tok.batch_decode(lab, group_tokens=False)
    return {"wer": 100 * jiwer.wer(ref, hyp), "cer": 100 * jiwer.cer(ref, hyp)}
# ---------------------------------------------------------------- entrenamiento
args = TrainingArguments(**ent)
trainer = Trainer(model=model, args=args, data_collator=collator,
                  train_dataset=ds_tr, eval_dataset=ds_va, processing_class=proc,
                  compute_metrics=metricas, preprocess_logits_for_metrics=preprocesar_logits)

hay_ckpt = bool(glob.glob(os.path.join(out_dir, "checkpoint-*")))
if a.reanudar and not hay_ckpt:
    sys.exit(f"ERROR: --reanudar sin checkpoints en {out_dir}")
if not a.reanudar and hay_ckpt:
    sys.exit(f"ERROR: {out_dir} ya tiene checkpoints. Usa --reanudar o cambia output_dir.")

trainer.train(resume_from_checkpoint=True if a.reanudar else None)

# Guardar el mejor modelo (load_best_model_at_end) y su evaluación final en val
final = os.path.join(out_dir, "final")
trainer.save_model(final)
proc.save_pretrained(final)
res = trainer.evaluate()                                           # val sin fuentes excluidas
res_c = trainer.evaluate(eval_dataset=ds_va_completo, metric_key_prefix="eval_completo")
res.update(res_c)
res["mejor_checkpoint"] = trainer.state.best_model_checkpoint
json.dump(res, open(os.path.join(out_dir, "metricas_val.json"), "w"), indent=1)
print(f"\nMejor checkpoint: {res['mejor_checkpoint']}")
print(f"Validación sin excluidas — WER {res['eval_wer']:.2f} % | CER {res['eval_cer']:.2f} %")
print(f"Validación completa      — WER {res['eval_completo_wer']:.2f} % | CER {res['eval_completo_cer']:.2f} %")
print(f"Modelo final guardado en {final}")
