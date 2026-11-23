# R4.2 — Guía de uso y despliegue

## Despliegue actual: Render + datasets privados de Hugging Face

La app se aloja en un **Web Service gratuito de Render**. Los datasets `r42-audios`
y `r42-respuestas` permanecen privados en Hugging Face. No hace falta crear un Space.

1. Publicar únicamente el código en GitHub, conservando la carpeta `R4.2`.
   No incluir `items/items.csv`, `audios/`, `respuestas/`, `.env` ni el entorno Python.
2. En Render: **New → Web Service**, seleccionar el repositorio y la rama de despliegue.
3. Configurar **Root Directory** `R4.2`, Python, plan **Free**,
   **Build Command** `pip install --only-binary=:all: -r requirements.txt`
   y **Start Command** `python app.py`.
4. Añadir las variables de la tabla y desplegar. `render.yaml` contiene la misma
   configuración para un Blueprint (ruta del Blueprint: `R4.2/render.yaml`).

| Variable | Valor |
|---|---|
| `PYTHON_VERSION` | `3.11.11` |
| `GRADIO_ANALYTICS_ENABLED` | `False` |
| `REPO_AUDIOS` | `homew0rk79/r42-audios` |
| `REPO_RESPUESTAS` | `homew0rk79/r42-respuestas` |
| `HF_TOKEN` | Token privado limitado a los dos datasets, con lectura y escritura |
| `CODIGOS` | `E1:CLAVE_LARGA_1,E2:CLAVE_LARGA_2,E3:CLAVE_LARGA_3`; sustituir las claves |

Render define `RENDER=true` y `PORT` automáticamente. La app escucha en `0.0.0.0`
y rechaza el despliegue si faltan variables o se usan los códigos `prueba1/2/3`.
Los secretos se introducen en Render, nunca en GitHub.

Cada respuesta se sube de forma síncrona al dataset privado antes de avanzar;
si falla, se muestra un error y se puede reintentar. El nombre de cada respuesta
se mantiene al reintentar para evitar duplicados. Al reiniciar, se recupera el
historial; si no se puede descargar, la app no arranca con un historial vacío.
El análisis sigue leyendo archivos JSONL con el mismo esquema.

El plan gratuito se suspende tras 15 minutos sin tráfico y el disco es temporal.
Abrir el enlace antes de cada sesión y comprobar recuperación de respuestas.
Antes de enviar el enlace real, completar la sección **Antes de aplicar**.

Las instrucciones de Spaces de la sección 3 se conservan como alternativa de pago.

## Contenido

| Archivo | Para qué |
|---|---|
| `preparar_items.py` | Selecciona los 24 ítems + 2 de práctica desde `pred_test.csv` (ya ejecutado, semilla 42) |
| `items/items.csv` | Ítems seleccionados. Contiene transcripciones del corpus: **no subir a ningún repositorio público** |
| `items/estratos.json` | Cortes de WER y tamaño de cada estrato en la población limpia (se usa para ponderar) |
| `items/lista_audios.txt` | Los 26 audios que hay que traer del servidor |
| `app.py` | La app de evaluación (Gradio). No ejecuta el modelo |
| `analizar_r42.py` | Calcula los indicadores y aplica la regla de decisión |
| `space/README.md` | Cabecera que necesita el Space de Hugging Face |

## 1. Traer los audios del servidor (PowerShell en la PC)

```powershell
cd "$repo\R4.2"
scp items\lista_audios.txt razorcrest:~/lista_audios_r42.txt
ssh razorcrest "cd ~/tesis/data/corpus_normalizado && tar czf ~/audios_r42.tar.gz -T ~/lista_audios_r42.txt"
scp razorcrest:~/audios_r42.tar.gz .
mkdir audios; tar xzf audios_r42.tar.gz -C audios
(Get-ChildItem audios).Count   # debe dar 26
```

Si `corpus_normalizado` organiza los audios en subcarpetas, ajusta la ruta del `cd` antes de empaquetar.

## 2. Probar en local

```powershell
pip install -r requirements.txt
python app.py
```

Abre http://127.0.0.1:7860 e ingresa con `prueba1`. Recorre las dos prácticas y un par de ítems con audio
real. Las respuestas quedan en `respuestas\`. **Borra esa carpeta antes de la evaluación real.**

## 3. Hugging Face (comandos verificados con huggingface_hub 2.4)

```powershell
pip install -U huggingface_hub
hf auth login                      # pega el token en tu terminal, nunca en un chat

$u = "TU_USUARIO"
hf repos create "$u/r42-audios" --repo-type dataset --private
hf repos create "$u/r42-respuestas" --repo-type dataset --private
hf repos create "$u/r42-evaluacion" --repo-type space --space-sdk gradio

hf upload "$u/r42-audios" items\items.csv items.csv --repo-type dataset
hf upload "$u/r42-audios" audios audios --repo-type dataset

hf upload "$u/r42-evaluacion" app.py app.py --repo-type space
hf upload "$u/r42-evaluacion" requirements.txt requirements.txt --repo-type space
hf upload "$u/r42-evaluacion" space\README.md README.md --repo-type space
```

Luego, en la web del Space: **Settings → Variables and secrets**, agrega cuatro *secrets*:

| Secret | Valor |
|---|---|
| `HF_TOKEN` | Token *fine-grained* con lectura de `r42-audios` y escritura en `r42-respuestas` (y nada más) |
| `REPO_AUDIOS` | `TU_USUARIO/r42-audios` |
| `REPO_RESPUESTAS` | `TU_USUARIO/r42-respuestas` |
| `CODIGOS` | `E1:clave1,E2:clave2,E3:clave3` con claves largas y aleatorias, una por evaluador |

Notas:
- El Space puede ser público: sin código no se entra, y audios y respuestas viven en datasets privados.
  Revisa también la opción *protected* de los Spaces, que permite ocultar los archivos del repositorio.
- El hardware gratuito se suspende tras un tiempo sin uso y tarda en despertar. Ábrelo tú unos minutos
  antes de enviar los enlaces, y avisa a los evaluadores que la primera carga puede demorar.
- Las respuestas se suben al dataset antes de avanzar de pantalla; si el Space se reinicia, la app las recupera al arrancar.
- Haz una prueba completa en el Space con un código de prueba y borra luego esos archivos del dataset.

## 4. Antes de aplicar

- [ ] Reemplazar `CONSENTIMIENTO` por el texto aprobado.
- [ ] Confirmar `INSTRUCCION_ORTOGRAFIA` con el especialista.
- [ ] Revisar los 6 controles con el especialista (ver advertencia abajo).
- [ ] Piloto con una persona para medir el tiempo real.
- [ ] Congelar la regla de decisión de `analizar_r42.py` y anotar la fecha en la bitácora.

## 5. Analizar

```powershell
hf download "$u/r42-respuestas" --repo-type dataset --local-dir resultados
pip install -r requirements_analisis.txt
python analizar_r42.py --respuestas resultados\respuestas --out R4.2_resultados
```
