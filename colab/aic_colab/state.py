from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .persistence import verify_artifact


VALID_STATUSES = {"pending", "running", "done", "error"}


class StateStore:
    def __init__(
        self, path: Path, dataset_id: str, bundle_version: str, schema_version: int = 1
    ) -> None:
        self.path = path
        self.dataset_id = dataset_id
        self.bundle_version = bundle_version
        self.schema_version = schema_version

    @property
    def backup_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}.backup")

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_version": self.bundle_version,
            "dataset_id": self.dataset_id,
            "updated_at": _now(),
            "batches": {},
        }

    def _validate(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("dataset_id") != self.dataset_id:
            raise ValueError("state dataset does not match configuration")
        if state.get("bundle_version") != self.bundle_version:
            raise ValueError("state bundle version does not match")
        if state.get("schema_version") != self.schema_version:
            raise ValueError("state schema version does not match")
        if not isinstance(state.get("batches"), dict):
            raise ValueError("state batches must be an object")
        return state

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._new_state()
        try:
            return self._validate(json.loads(self.path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            if not self.backup_path.exists():
                raise ValueError("state JSON is corrupt and no backup exists") from None
            return self._validate(
                json.loads(self.backup_path.read_text(encoding="utf-8"))
            )

    def save(self, state: dict[str, Any]) -> None:
        self._validate(state)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            shutil.copy2(self.path, self.backup_path)
        state["updated_at"] = _now()
        partial = self.path.with_name(f"{self.path.name}.partial")
        with partial.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, self.path)

    def mark(
        self,
        state: dict[str, Any],
        batch_id: str,
        stage: str,
        shard_id: str,
        status: str,
        **metadata: Any,
    ) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status}")
        batch = state["batches"].setdefault(batch_id, {"stages": {}})
        stage_data = batch["stages"].setdefault(stage, {"shards": {}})
        shard = stage_data["shards"].setdefault(shard_id, {})
        shard.update(metadata)
        shard["status"] = status
        shard["updated_at"] = _now()

    def reset_stale_running(self, state: dict[str, Any]) -> int:
        reset = 0
        for batch in state["batches"].values():
            for stage in batch.get("stages", {}).values():
                for shard in stage.get("shards", {}).values():
                    if shard.get("status") == "running":
                        shard["status"] = "pending"
                        shard["updated_at"] = _now()
                        reset += 1
        return reset

    def write_heartbeat(self, path: Path, **details: Any) -> None:
        payload = {"dataset_id": self.dataset_id, "updated_at": _now(), **details}
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.partial")
        partial.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(partial, path)

    def rebuild_from_artifacts(self, results_root: Path) -> dict[str, Any]:
        state = self._new_state()
        for sidecar in sorted(results_root.rglob("*.sha256")):
            artifact = sidecar.with_name(sidecar.name.removesuffix(".sha256"))
            if verify_artifact(artifact):
                state.setdefault("recovered_artifacts", []).append(
                    artifact.relative_to(results_root).as_posix()
                )
        return state


def _now() -> str:
    return datetime.now(UTC).isoformat()
