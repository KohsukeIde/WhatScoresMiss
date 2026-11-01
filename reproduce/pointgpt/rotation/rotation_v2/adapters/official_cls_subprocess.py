"""Persistent subprocess adapter for pinned official classification code."""

from __future__ import annotations

import copy
import json
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from rotation_v2.snapshot_manifest import verify_snapshot

from .base import FrozenSystemAdapter
from .strict_loading import StrictLoadError, file_sha256


class OfficialClassificationSubprocessAdapter(FrozenSystemAdapter):
    """One immutable snapshot and model per persistent isolated interpreter."""

    def __init__(
        self,
        *,
        family: str,
        snapshot_name: str,
        config_path: Path,
        checkpoint_path: Path,
        snapshots_root: Path | None = None,
        device: str | torch.device = "cuda",
        forward_seed: int = 0,
        ready_timeout_seconds: float = 300.0,
    ) -> None:
        if family not in ("Point-MAE", "PCP-MAE"):
            raise ValueError("unsupported official classification family")
        if snapshot_name not in ("pointmae", "pcpmae"):
            raise ValueError("unsupported official snapshot name")
        if isinstance(forward_seed, bool) or int(forward_seed) < 0:
            raise ValueError("forward_seed must be a non-negative integer")
        if float(ready_timeout_seconds) <= 0.0:
            raise ValueError("ready timeout must be positive")
        self.family = family
        self.snapshot_name = snapshot_name
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.snapshots_root = (
            Path(snapshots_root) if snapshots_root is not None else None
        )
        self.device = torch.device(device)
        self.forward_seed = int(forward_seed)
        self.ready_timeout_seconds = float(ready_timeout_seconds)
        self._process: subprocess.Popen[str] | None = None
        self._stderr_handle: object | None = None
        self._rpc_directory: Path | None = None
        self._stderr_path: Path | None = None
        self._snapshot: dict[str, object] | None = None
        self._resolved_config: Path | None = None
        self._metadata: dict[str, object] | None = None
        self._request_id = 0
        self._forward_calls = 0

    def _stderr_tail(self) -> str:
        if self._stderr_path is None or not self._stderr_path.is_file():
            return ""
        try:
            text = self._stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-4000:]

    def _read_response(self, *, timeout: float) -> Mapping[str, object]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("classification worker is not running")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + float(timeout)
        try:
            while True:
                if process.poll() is not None:
                    line = process.stdout.readline()
                    if line:
                        break
                    raise RuntimeError(
                        "classification worker exited with status %s: %s"
                        % (process.returncode, self._stderr_tail())
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "classification worker response timed out: %s"
                        % self._stderr_tail()
                    )
                if selector.select(min(remaining, 1.0)):
                    line = process.stdout.readline()
                    if line:
                        break
        finally:
            selector.close()
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "classification worker emitted non-JSON protocol output"
            ) from error
        if not isinstance(response, Mapping):
            raise RuntimeError("classification worker response is not a mapping")
        return response

    def _rpc(
        self,
        command: str,
        payload: Mapping[str, object] | None = None,
        *,
        timeout: float = 300.0,
    ) -> Mapping[str, object]:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("classification worker is not running")
        if process.poll() is not None:
            raise RuntimeError(
                "classification worker already exited: %s" % self._stderr_tail()
            )
        self._request_id += 1
        request = {
            "command": str(command),
            "request_id": self._request_id,
            **({} if payload is None else dict(payload)),
        }
        process.stdin.write(
            json.dumps(
                request,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
        process.stdin.flush()
        response = self._read_response(timeout=timeout)
        if response.get("request_id") != self._request_id:
            raise RuntimeError("classification worker request ID mismatch")
        if response.get("status") == "ERROR":
            raise StrictLoadError(
                "classification worker rejected %s: %s: %s"
                % (
                    command,
                    response.get("error_type"),
                    response.get("error"),
                )
            )
        return response

    def _resolve_config(
        self,
        snapshot_path: Path,
        snapshot_record: Mapping[str, object],
    ) -> Path:
        candidate = (
            self.config_path
            if self.config_path.is_absolute()
            else snapshot_path / self.config_path
        ).resolve(strict=True)
        try:
            relative = candidate.relative_to(snapshot_path).as_posix()
        except ValueError as error:
            raise ValueError("classification config must be inside snapshot") from error
        files = snapshot_record.get("files")
        if not isinstance(files, list):
            raise ValueError("verified snapshot has no file inventory")
        matches = [
            record
            for record in files
            if isinstance(record, Mapping) and record.get("path") == relative
        ]
        if len(matches) != 1 or matches[0].get("sha256") != file_sha256(candidate):
            raise ValueError("classification config is not in snapshot inventory")
        return candidate

    def load_frozen_system(self) -> "OfficialClassificationSubprocessAdapter":
        if self._process is not None:
            metadata = self.metadata()
            if metadata.get("parameter_hash_invariant") is not True:
                raise StrictLoadError("classification worker state is not invariant")
            return self
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("classification checkpoint does not exist")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        snapshot = verify_snapshot(
            self.snapshot_name,
            snapshots_root=self.snapshots_root,
        )
        snapshot_path = Path(str(snapshot["path"]))
        config = self._resolve_config(snapshot_path, snapshot)
        checkpoint_hash = file_sha256(self.checkpoint_path)
        config_hash = file_sha256(config)
        worker_path = Path(__file__).with_name("_classification_worker.py")
        rpc_directory = Path(
            tempfile.mkdtemp(prefix="rotation_v2_cls_rpc_")
        ).resolve()
        stderr_path = rpc_directory / "worker_stderr.log"
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        self._stderr_handle = stderr_handle
        self._rpc_directory = rpc_directory
        self._stderr_path = stderr_path
        argv = [
            sys.executable,
            str(worker_path),
            "--family",
            self.snapshot_name,
            "--snapshot",
            str(snapshot_path),
            "--config",
            str(config),
            "--checkpoint",
            str(self.checkpoint_path.resolve()),
            "--device",
            str(self.device),
            "--rpc-root",
            str(rpc_directory),
        ]
        worker_environment = dict(os.environ)
        worker_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).resolve().parents[2]),
                env=worker_environment,
            )
            self._process = process
            response = self._read_response(timeout=self.ready_timeout_seconds)
            if response.get("status") != "READY":
                raise RuntimeError("classification worker did not become ready")
            worker_metadata = response.get("metadata")
            if not isinstance(worker_metadata, Mapping):
                raise RuntimeError("classification worker metadata is invalid")
            if file_sha256(self.checkpoint_path) != checkpoint_hash:
                raise StrictLoadError(
                    "classification checkpoint changed during worker startup"
                )
            if file_sha256(config) != config_hash:
                raise StrictLoadError(
                    "classification config changed during worker startup"
                )
            self._snapshot = snapshot
            self._resolved_config = config
            self._metadata = {
                "adapter_schema": "rotation_matrix_v2_official_cls_adapter_v1",
                "model_family": self.family,
                "task": "ScanObjectNN_classification",
                "complete_system_loaded": True,
                "snapshot": snapshot,
                "config_path": str(config),
                "config_sha256": config_hash,
                "checkpoint_path": str(self.checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "device": str(self.device),
                "consumed_features": ["xyz"],
                "input_schema": "B,N,3 float32 XYZ",
                "output_transport": "neutral JSON logits",
                "input_transport": "verified temporary no-pickle NPY referenced by JSON",
                "isolation": {
                    "mode": "persistent_dedicated_subprocess",
                    "one_snapshot_per_interpreter": True,
                    "worker_path": str(worker_path.resolve()),
                    "worker_sha256": file_sha256(worker_path),
                    "host_python": sys.executable,
                },
                "forward_seed_replayed_each_call": self.forward_seed,
                "official_runner_parity": {
                    "status": "BY_CONSTRUCTION",
                    "builder": "official tools.builder.model_builder",
                    "forward": "official model(points)",
                    "alternate_model_definition_used": False,
                },
                "worker": dict(worker_metadata),
            }
        except Exception:
            self.close(force=True)
            raise
        return self

    def forward_logits(self, points: torch.Tensor) -> torch.Tensor:
        if self._process is None or self._rpc_directory is None:
            raise RuntimeError("load_frozen_system() must be called first")
        if not torch.is_tensor(points):
            raise TypeError("points must be a torch tensor")
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("classification points must have shape (B, N, 3)")
        if points.dtype != torch.float32:
            raise TypeError("classification points must use float32")
        if not bool(torch.isfinite(points).all()):
            raise FloatingPointError("classification points contain a non-finite value")
        input_path = self._rpc_directory / (
            "input_%08d.npy" % (self._forward_calls + 1)
        )
        if input_path.exists():
            raise FileExistsError("classification RPC input path already exists")
        np.save(
            input_path,
            points.detach().cpu().contiguous().numpy(),
            allow_pickle=False,
        )
        try:
            response = self._rpc(
                "forward",
                {
                    "input_path": str(input_path),
                    "seed": self.forward_seed,
                },
            )
        finally:
            if input_path.exists():
                input_path.unlink()
        if response.get("status") != "OK":
            raise StrictLoadError("classification worker forward did not succeed")
        shape = response.get("shape")
        logits_value = response.get("logits")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[0] != points.shape[0]
            or not isinstance(logits_value, list)
        ):
            raise StrictLoadError("classification worker logits schema is invalid")
        logits_numpy = np.asarray(logits_value, dtype=np.float32)
        if list(logits_numpy.shape) != shape or not np.all(np.isfinite(logits_numpy)):
            raise StrictLoadError("classification worker logits payload is invalid")
        self._forward_calls += 1
        if response.get("forward_calls") != self._forward_calls:
            raise StrictLoadError("classification worker forward count diverged")
        return torch.from_numpy(logits_numpy)

    def metadata(self) -> Mapping[str, object]:
        if (
            self._metadata is None
            or self._snapshot is None
            or self._resolved_config is None
        ):
            raise RuntimeError("load_frozen_system() must be called first")
        response = self._rpc("metadata")
        if response.get("status") != "OK":
            raise StrictLoadError("classification metadata gate did not succeed")
        if response.get("parameter_hash_invariant") is not True:
            raise StrictLoadError("classification parameter hash changed")
        if response.get("forward_calls") != self._forward_calls:
            raise StrictLoadError("classification worker forward count diverged")
        current_snapshot = verify_snapshot(
            self.snapshot_name,
            snapshots_root=self.snapshots_root,
        )
        if (
            current_snapshot.get("manifest_sha256")
            != self._snapshot.get("manifest_sha256")
            or current_snapshot.get("content_tree_sha256")
            != self._snapshot.get("content_tree_sha256")
        ):
            raise StrictLoadError("classification source snapshot changed")
        if (
            file_sha256(self.checkpoint_path)
            != self._metadata["checkpoint_sha256"]
        ):
            raise StrictLoadError("classification checkpoint changed")
        if file_sha256(self._resolved_config) != self._metadata["config_sha256"]:
            raise StrictLoadError("classification config changed")
        result = copy.deepcopy(self._metadata)
        result.update(
            {
                "parameter_hash_post_evaluation": response[
                    "parameter_hash_post_evaluation"
                ],
                "parameter_hash_invariant": True,
                "forward_calls": self._forward_calls,
                "source_snapshot_post_evaluation_verified": True,
                "checkpoint_post_evaluation_verified": True,
                "config_post_evaluation_verified": True,
            }
        )
        return result

    def close(self, *, force: bool = False) -> None:
        process = self._process
        if process is not None:
            if process.poll() is None and not force:
                try:
                    response = self._rpc("close", timeout=30.0)
                    if response.get("status") != "CLOSED":
                        force = True
                except Exception:
                    force = True
            if process.poll() is None and force:
                process.terminate()
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        if self._stderr_handle is not None:
            self._stderr_handle.close()
        rpc_directory = self._rpc_directory
        self._process = None
        self._stderr_handle = None
        self._rpc_directory = None
        if rpc_directory is not None and rpc_directory.is_dir():
            shutil.rmtree(rpc_directory)

    def __enter__(self) -> "OfficialClassificationSubprocessAdapter":
        return self.load_frozen_system()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close(force=exc is not None)

    def __del__(self) -> None:
        try:
            self.close(force=True)
        except Exception:
            pass
