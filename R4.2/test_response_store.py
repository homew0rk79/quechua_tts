"""Pruebas sin red para la persistencia en un servidor con disco temporal."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from response_store import ResponseStore


class ResponseStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "respuestas"
        self.store = ResponseStore(self.folder)
        self.store.repo_id = "test/respuestas"
        self.store.api = Mock()
        self.record = {"etapa": "item", "evaluador": "E1", "item_id": "item01", "utilidad": 4}

    def test_failed_upload_does_not_advance_local_history(self):
        self.store.api.upload_file.side_effect = ConnectionError("offline")
        with self.assertRaises(ConnectionError):
            self.store.save(self.record, "E1_session.jsonl")
        self.assertEqual(list(self.folder.glob("*.jsonl")), [])

    def test_retry_after_remote_timeout_uses_same_remote_path(self):
        remote = {}

        def upload(**kwargs):
            remote[kwargs["path_in_repo"]] = kwargs["path_or_fileobj"]
            if len(self.store.api.upload_file.call_args_list) == 1:
                raise TimeoutError("commit accepted, response lost")

        self.store.api.upload_file.side_effect = upload
        with self.assertRaises(TimeoutError):
            self.store.save(self.record, "E1_session.jsonl")
        self.store.save(self.record, "E1_session.jsonl")
        self.assertEqual(len(remote), 1)
        self.assertEqual(len(list(self.folder.glob("*.jsonl"))), 1)
        record = json.loads(next(iter(remote.values())))
        self.assertEqual(record["utilidad"], 4)
        self.assertIn("marca_tiempo", record)

    def test_restart_recovers_confirmed_answers(self):
        remote = {}
        self.store.api.upload_file.side_effect = lambda **k: remote.update({k["path_in_repo"]: k["path_or_fileobj"]})
        self.store.save(self.record, "E1_session.jsonl")
        restarted = ResponseStore(Path(self.temp.name) / "restart" / "respuestas")
        restarted.repo_id = "test/respuestas"
        restarted.api = Mock()
        restarted.api.repo_info.return_value = SimpleNamespace(private=True)

        def download(*args, **kwargs):
            for name, payload in remote.items():
                destination = Path(kwargs["local_dir"]) / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)

        with patch("huggingface_hub.snapshot_download", side_effect=download):
            restarted.restore()
        recovered = [json.loads(p.read_text()) for p in restarted.directory.glob("*.jsonl")]
        self.assertEqual([r["item_id"] for r in recovered], ["item01"])

    def test_restore_failure_prevents_empty_history_start(self):
        self.store.api.repo_info.return_value = SimpleNamespace(private=True)
        with patch("huggingface_hub.snapshot_download", side_effect=ConnectionError("offline")):
            with self.assertRaises(ConnectionError):
                self.store.restore()

    def test_public_response_dataset_is_rejected(self):
        self.store.api.repo_info.return_value = SimpleNamespace(private=False)
        with patch("huggingface_hub.snapshot_download") as download:
            with self.assertRaises(RuntimeError):
                self.store.restore()
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
