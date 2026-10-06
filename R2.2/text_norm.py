"""Normalización de transcripciones compartida por entrenamiento y evaluación."""
import re
import unicodedata

_APOS = {"\u2018": "'", "\u2019": "'", "`": "'", "\u00b4": "'"}
_NO_PERMITIDO = re.compile(r"[^\w' ]")  # conserva letras, apóstrofo y espacio


def normalizar(texto: str, sin_apostrofo: bool = False) -> str:
    t = unicodedata.normalize("NFC", str(texto)).lower()
    for k, v in _APOS.items():
        t = t.replace(k, v)
    t = _NO_PERMITIDO.sub(" ", t)          # puntuación -> espacio
    t = re.sub(r"[\d_]", " ", t)           # \w incluye dígitos y _: se eliminan
    t = re.sub(r"(^|\s)'+|'+(\s|$)", " ", t)  # apóstrofos sueltos (no glotalizados)
    if sin_apostrofo:
        t = t.replace("'", "")
    return re.sub(r"\s+", " ", t).strip()
