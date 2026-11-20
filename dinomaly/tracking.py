"""MLflow tracking shared by training and evaluation.

Common contract with the other B1375 repos (RF-DETR, classifiers):

- Tracking URI: ``$MLFLOW_TRACKING_URI`` or ``http://127.0.0.1:5001``.  HTTP
  servers are health-checked; if unreachable we fall back to a local file
  store in ``<repo>/mlruns`` and tag the run ``tracking_fallback=true``.
- Experiment: ``dinomaly/<face_group>`` (``project.name`` without the trailing
  ``-dinomaly``), overridable with ``tracking.experiment`` in the config.
- Metric keys: ``{split}/{level}/{metric}`` and
  ``{split}/{level}/class/{class_slug}/{metric}``.

Tracking is best-effort: every MLflow call is wrapped so a broken server can
never crash (or even interrupt) a long training run.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
import urllib.request
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger("dinomaly")

DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
MODEL_FAMILY = "dinomaly"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_FILE = "mlflow_run.json"

_MAX_PARAMS_PER_BATCH = 100
_MAX_METRICS_PER_BATCH = 500
_MAX_PARAM_LENGTH = 6000

_last_fallback = False


def slug(name: Any) -> str:
    """Lower-case, non-alphanumerics -> ``_``, no leading/trailing ``_``."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _health_ok(uri: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=timeout) as resp:
            return 200 <= int(resp.status) < 300
    except Exception:  # noqa: BLE001
        return False


def local_store_uri(repo_root: Path | None = None) -> str:
    path = ((repo_root or REPO_ROOT) / "mlruns").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()


def resolve_tracking_uri(repo_root: Path | None = None) -> str:
    """Return a usable tracking URI, falling back to ``<repo>/mlruns``."""
    global _last_fallback
    uri = os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    _last_fallback = False
    if uri.startswith(("http://", "https://")) and not _health_ok(uri):
        fallback = local_store_uri(repo_root)
        log.warning(
            f"MLflow server {uri} is not reachable; logging to the local file store {fallback}"
        )
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        _last_fallback = True
        return fallback
    if uri.startswith("file:"):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    return uri


def used_fallback() -> bool:
    return _last_fallback


def face_group(config: Mapping[str, Any]) -> str:
    name = str((config.get("project") or {}).get("name") or "default")
    return re.sub(r"-dinomaly$", "", name)


def experiment_name(config: Mapping[str, Any]) -> str:
    override = (config.get("tracking") or {}).get("experiment")
    return str(override) if override else f"{MODEL_FAMILY}/{face_group(config)}"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def git_info() -> dict[str, str]:
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_dirty": "unknown" if status is None else str(bool(status)).lower(),
    }


def dataset_tag(config: Mapping[str, Any], tiles_dir: str | os.PathLike | None = None) -> str:
    data = config.get("data") or {}
    roots = [str(s.get("root")) for s in data.get("sources") or []]
    tiles = str(tiles_dir or data.get("tiles_dir") or "")
    return f"tiles={tiles}; sources={','.join(roots)}"


def standard_tags(
    config: Mapping[str, Any],
    config_path: str | os.PathLike | None,
    stage: str,
    tiles_dir: str | os.PathLike | None = None,
) -> dict[str, str]:
    return {
        "model_family": MODEL_FAMILY,
        "face_group": face_group(config),
        "config_path": str(config_path or ""),
        **git_info(),
        "dataset": dataset_tag(config, tiles_dir),
        "tracking_fallback": str(used_fallback()).lower(),
        "stage": stage,
    }


def _iter_flat(d: Mapping[str, Any], prefix: str, sep: str) -> Iterator[tuple[str, Any]]:
    for key, value in d.items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _iter_flat(value, name, sep)
        else:
            yield name, value


def flatten(d: Mapping[str, Any], sep: str = "/") -> dict[str, Any]:
    return dict(_iter_flat(d, "", sep))


def sanitize_key(key: str) -> str:
    # MLflow allows alphanumerics, '_', '-', '.', ' ', ':' and '/'.
    key = re.sub(r"[^A-Za-z0-9_\-./ :]+", "_", str(key))
    return re.sub(r"/+", "/", key).strip("/")


def finite_metrics(d: Mapping[str, Any]) -> dict[str, float]:
    """Flatten nested dicts with ``/`` and keep finite numbers only."""
    out: dict[str, float] = {}
    for key, value in flatten(d).items():
        if isinstance(value, bool):
            value = float(value)
        if not isinstance(value, (int, float)):
            try:
                import numpy as np  # local: numpy scalars

                if isinstance(value, np.generic):
                    value = value.item()
                else:
                    continue
            except Exception:  # noqa: BLE001
                continue
        value = float(value)
        if math.isfinite(value):
            out[sanitize_key(key)] = value
    return out


def _mlflow():
    import mlflow

    return mlflow


def log_metrics_dict(d: Mapping[str, Any], step: int | None = None) -> int:
    """Log a nested metrics dict to the active run. Returns the number logged."""
    metrics = finite_metrics(d)
    if not metrics:
        return 0
    try:
        mlflow = _mlflow()
        if mlflow.active_run() is None:
            return 0
        items = list(metrics.items())
        for i in range(0, len(items), _MAX_METRICS_PER_BATCH):
            mlflow.log_metrics(dict(items[i : i + _MAX_METRICS_PER_BATCH]), step=step)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"MLflow: could not log metrics ({exc})")
        return 0
    return len(metrics)


def _safe_console() -> None:
    """MLflow prints emoji (e.g. 'View run ...'); a cp1252 console would raise."""
    for stream in (sys.stdout, sys.stderr):
        try:
            encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
            if encoding != "utf8" and hasattr(stream, "reconfigure"):
                stream.reconfigure(errors="backslashreplace")
        except Exception:  # noqa: BLE001
            pass


class Tracker:
    """Thin, failure-tolerant wrapper around one MLflow run."""

    def __init__(self) -> None:
        self.enabled = False
        self.run_id: str | None = None
        self.tracking_uri: str | None = None
        self.experiment: str | None = None
        self.resumed = False

    @property
    def active(self) -> bool:
        return self.enabled and self.run_id is not None

    def start(
        self,
        config: Mapping[str, Any],
        config_path: str | os.PathLike | None,
        run_name: str,
        stage: str,
        tiles_dir: str | os.PathLike | None = None,
        resume_run_id: str | None = None,
        extra_tags: Mapping[str, str] | None = None,
    ) -> Tracker:
        if (config.get("tracking") or {}).get("enabled", True) is False:
            log.info("MLflow tracking disabled (tracking.enabled=false)")
            return self
        try:
            _safe_console()
            mlflow = _mlflow()
            self.tracking_uri = resolve_tracking_uri()
            mlflow.set_tracking_uri(self.tracking_uri)
            self.experiment = experiment_name(config)
            mlflow.set_experiment(self.experiment)
            run = None
            if resume_run_id:
                try:
                    mlflow.tracking.MlflowClient().get_run(resume_run_id)
                    run = mlflow.start_run(run_id=resume_run_id)
                    self.resumed = True
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        f"MLflow: cannot resume run {resume_run_id} on {self.tracking_uri} ({exc}); "
                        "starting a new run"
                    )
            if run is None:
                run = mlflow.start_run(run_name=run_name)
            self.run_id = run.info.run_id
            self.enabled = True
            tags = standard_tags(config, config_path, stage, tiles_dir)
            if self.resumed:
                # Keep the tags of the original run; record what touched it last.
                tags = {
                    "last_stage": stage,
                    "last_git_commit": tags["git_commit"],
                    "last_git_dirty": tags["git_dirty"],
                }
            tags.update(extra_tags or {})
            mlflow.set_tags(tags)
            log.info(
                f"MLflow: {'resumed' if self.resumed else 'started'} run {self.run_id} "
                f"in experiment '{self.experiment}' ({self.tracking_uri})"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(f"MLflow disabled for this run: {exc}")
            self.enabled = False
            self.run_id = None
        return self

    def _call(self, what: str, fn, *args: Any, **kwargs: Any) -> Any:
        if not self.active:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"MLflow: could not {what} ({exc})")
            return None

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self.active:
            return

        def _do() -> None:
            mlflow = _mlflow()
            existing = mlflow.tracking.MlflowClient().get_run(self.run_id).data.params
            items = []
            for key, value in flatten(params, sep=".").items():
                key = sanitize_key(key)
                text = json.dumps(value, default=str) if isinstance(value, (list, tuple, dict)) else str(value)
                text = text[:_MAX_PARAM_LENGTH]
                # Params are immutable: on resume only add the missing ones.
                if key in existing:
                    continue
                items.append((key, text))
            for i in range(0, len(items), _MAX_PARAMS_PER_BATCH):
                mlflow.log_params(dict(items[i : i + _MAX_PARAMS_PER_BATCH]))

        self._call("log params", _do)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if self.active:
            log_metrics_dict(metrics, step=step)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        self._call("set tags", lambda: _mlflow().set_tags({k: str(v) for k, v in tags.items()}))

    def log_artifact(self, path: str | os.PathLike, artifact_path: str | None = None) -> None:
        path = Path(path)
        if not path.exists():
            return
        if path.is_dir():
            self._call("log artifacts", lambda: _mlflow().log_artifacts(str(path), artifact_path))
        else:
            self._call("log artifact", lambda: _mlflow().log_artifact(str(path), artifact_path))

    def log_dict(self, data: Any, artifact_file: str) -> None:
        self._call("log dict", lambda: _mlflow().log_dict(data, artifact_file))

    def save_run_file(self, run_dir: str | os.PathLike) -> None:
        if not self.active:
            return
        try:
            payload = {
                "run_id": self.run_id,
                "tracking_uri": self.tracking_uri,
                "experiment": self.experiment,
            }
            (Path(run_dir) / RUN_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"MLflow: could not write {RUN_FILE} ({exc})")

    def end(self, status: str = "FINISHED") -> None:
        if not self.active:
            return
        try:
            _mlflow().end_run(status=status)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"MLflow: end_run failed ({exc}); marking run {self.run_id} {status} directly")
            try:
                _mlflow().tracking.MlflowClient().set_terminated(self.run_id, status=status)
            except Exception as exc2:  # noqa: BLE001
                log.warning(f"MLflow: could not terminate run ({exc2})")
            try:
                while _mlflow().active_run() is not None:
                    _mlflow().end_run(status=status)
            except Exception:  # noqa: BLE001
                pass
        self.enabled = False


def read_run_file(run_dir: str | os.PathLike) -> str | None:
    """Run id stored by a previous training in ``run_dir`` (if any)."""
    path = Path(run_dir) / RUN_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("run_id")
    except Exception:  # noqa: BLE001
        return None
