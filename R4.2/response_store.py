"""Guardado confirmado e idempotente de respuestas en disco y Hugging Face."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


class ResponseStore:
    def __init__(self, directory, repo_id=None, token=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.token = token
        self.lock = Lock()
        self.api = None
        if repo_id:
            from huggingface_hub import HfApi
            self.api = HfApi(token=token)

    def restore(self):
        if self.api is None:
            return
        from huggingface_hub import snapshot_download
        # No iniciar con un historial incompleto si falla la red o el token.
        if not self.api.repo_info(self.repo_id, repo_type="dataset").private:
            raise RuntimeError("El dataset de respuestas debe ser privado.")
        snapshot_download(self.repo_id, repo_type="dataset", token=self.token,
                          local_dir=self.directory.parent, allow_patterns="respuestas/*.jsonl")

    def save(self, record, session_file):
        # Un archivo por respuesta permite reintentar una subida sin duplicarla.
        key = record["etapa"] + ":" + str(record.get("item_id", ""))
        suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
        name = f"{Path(session_file).stem}_{suffix}.jsonl"
        destination = self.directory / name
        record = dict(record, marca_tiempo=datetime.now(timezone.utc).isoformat())
        payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        with self.lock:
            if self.api is not None:
                # No actualizar el historial ni avanzar hasta confirmar la subida.
                self.api.upload_file(path_or_fileobj=payload,
                                     path_in_repo=f"respuestas/{name}",
                                     repo_id=self.repo_id, repo_type="dataset",
                                     commit_message="Save evaluation response")
            temporary = destination.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(destination)
