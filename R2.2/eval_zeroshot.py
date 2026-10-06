"""Evaluación zero-shot de modelos ASR sobre una partición del corpus.

Tipos de modelo (--kind):
  ctc      checkpoint Wav2Vec2-CTC ya ajustado (p. ej., Romero et al., 2024)
  mms      MMS con adaptador de lengua (requiere --lang, p. ej., quy)
  whisper  Whisper (seq2seq); --lang opcional (Whisper no incluye quechua)

Ejemplos:
  CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --kind ctc \
      --model ~/tesis/models/romero-quechua --csv ~/tesis/data/particiones/val.csv \
      --audio_root ~/tesis/data/corpus_normalizado --out ~/tesis/runs/zs_romero_val.csv
  CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --kind mms --lang quy \
      --model facebook/mms-1b-all ...
  CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py --kind whisper \
      --model openai/whisper-large-v3 ...
"""
import argparse, os, sys
import jiwer, pandas as pd, soundfile as sf, torch
from text_norm import normalizar

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["ctc", "mms", "whisper"], default="ctc")
ap.add_argument("--model", required=True, help="carpeta local o id de Hugging Face")
ap.add_argument("--lang", default=None)
ap.add_argument("--csv", required=True)
ap.add_argument("--audio_root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch_size", type=int, default=8)
a = ap.parse_args()
exp = os.path.expanduser
model_id = exp(a.model) if os.path.isdir(exp(a.model)) else a.model
if a.kind == "mms" and not a.lang:
    sys.exit("ERROR: --kind mms requiere --lang (código ISO 639-3 del adaptador).")

# ---------- datos
indice = {}
for raiz, _, archivos in os.walk(exp(a.audio_root)):
    for f in archivos:
        if f.lower().endswith(".wav"):
            indice.setdefault(f, os.path.join(raiz, f))
df = pd.read_csv(exp(a.csv))
faltan = [p for p in df.audio_path if p not in indice]
if faltan:
    sys.exit(f"ERROR: {len(faltan)} audios no encontrados. Ej.: {faltan[:3]}")
audios = {p: sf.read(indice[p], dtype="float32")[0] for p in df.audio_path}
df["dur"] = [len(audios[p]) / 16000 for p in df.audio_path]
orden = df.sort_values("dur").index

# ---------- modelo
if a.kind in ("ctc", "mms"):
    from transformers import (AutoModelForCTC, AutoProcessor, Wav2Vec2CTCTokenizer,
                              Wav2Vec2FeatureExtractor, Wav2Vec2Processor)
    try:
        proc = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"AutoProcessor falló ({type(e).__name__}); se usa procesador por defecto.")
        proc = Wav2Vec2Processor(
            feature_extractor=Wav2Vec2FeatureExtractor(sampling_rate=16000, do_normalize=True,
                                                       return_attention_mask=True),
            tokenizer=Wav2Vec2CTCTokenizer(os.path.join(model_id, "vocab.json")))
    if a.kind == "mms":
        proc.tokenizer.set_target_lang(a.lang)
        model = AutoModelForCTC.from_pretrained(model_id, target_lang=a.lang,
                                                ignore_mismatched_sizes=True)
    else:
        model = AutoModelForCTC.from_pretrained(model_id)
    model = model.to("cuda").eval()

    def transcribir(lote):
        x = proc(lote, sampling_rate=16000, return_tensors="pt", padding=True)
        x = {k: v.to("cuda") for k, v in x.items()}
        return proc.batch_decode(model(**x).logits.argmax(-1))
else:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16).to("cuda").eval()
    n_largos = int((df.dur > 30).sum())
    if n_largos:
        print(f"AVISO: {n_largos} audios superan 30 s; Whisper procesa solo los primeros 30 s.")

    def transcribir(lote):
        x = proc(lote, sampling_rate=16000, return_tensors="pt")
        feats = x.input_features.to("cuda", torch.bfloat16)
        kw = {"task": "transcribe", "max_new_tokens": 440}
        if a.lang:
            kw["language"] = a.lang
        ids = model.generate(feats, **kw)
        return proc.batch_decode(ids, skip_special_tokens=True)

# ---------- inferencia
hyp = {}
with torch.inference_mode():
    for i in range(0, len(orden), a.batch_size):
        idx = orden[i:i + a.batch_size]
        for j, t in zip(idx, transcribir([audios[df.at[j, "audio_path"]] for j in idx])):
            hyp[j] = t
df["hyp_raw"] = df.index.map(hyp)
df["ref"] = df.text_audio.map(normalizar)
df["hyp"] = df.hyp_raw.map(normalizar)

def metricas(r, h):
    return 100 * jiwer.wer(list(r), list(h)), 100 * jiwer.cer(list(r), list(h))

sa = lambda s: s.map(lambda t: t.replace("'", ""))
wer, cer = metricas(df.ref, df.hyp)
wer_na, cer_na = metricas(sa(df.ref), sa(df.hyp))
vacias = int((df.hyp == "").sum())
print(f"\nArchivos: {len(df)} | tipo: {a.kind} | modelo: {model_id} | lang: {a.lang}")
print(f"WER {wer:.2f} % | CER {cer:.2f} %")
print(f"WER sin apóstrofo {wer_na:.2f} % | CER sin apóstrofo {cer_na:.2f} %")
print(f"Transcripciones vacías: {vacias}")
print("\nEjemplos (ref -> hyp):")
for _, r in df.sample(5, random_state=0).iterrows():
    print(f"  REF: {r.ref}\n  HYP: {r.hyp}\n")
os.makedirs(os.path.dirname(exp(a.out)) or ".", exist_ok=True)
df.drop(columns="dur").to_csv(exp(a.out), index=False)
print(f"Predicciones guardadas en {a.out}")
