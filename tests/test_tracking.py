import math
import os
import socket

import pytest

from dinomaly import tracking


def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_slug():
    assert tracking.slug("Glue on face") == "glue_on_face"
    assert tracking.slug("  Mark On-Face (x) ") == "mark_on_face_x"


def test_experiment_name_and_override():
    assert tracking.experiment_name({"project": {"name": "top-bottom-dinomaly"}}) == "dinomaly/top-bottom"
    assert tracking.experiment_name({"project": {"name": "left-right-dinomaly"}}) == "dinomaly/left-right"
    cfg = {"project": {"name": "x"}, "tracking": {"experiment": "dinomaly/smoke-test"}}
    assert tracking.experiment_name(cfg) == "dinomaly/smoke-test"


def test_finite_metrics_flattens_and_filters():
    flat = tracking.finite_metrics(
        {"val": {"image": {"auroc": 0.9, "bad": float("nan"), "inf": math.inf, "name": "x",
                            "class": {"glue_on_face": {"tp": 3}}}}}
    )
    assert flat == {"val/image/auroc": 0.9, "val/image/class/glue_on_face/tp": 3.0}


def test_fallback_to_file_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"http://127.0.0.1:{_unused_port()}")
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    uri = tracking.resolve_tracking_uri(tmp_path)
    assert uri == (tmp_path / "mlruns").resolve().as_uri()
    assert tracking.used_fallback()
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"


def test_tracker_end_to_end_on_file_store(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"http://127.0.0.1:{_unused_port()}")
    monkeypatch.setattr(tracking, "REPO_ROOT", tmp_path)
    cfg = {"project": {"name": "top-bottom-dinomaly"}, "train": {"lr": 0.1}}
    tracker = tracking.Tracker().start(cfg, "cfg.yaml", run_name="t", stage="training")
    try:
        assert tracker.active
        tracker.log_params(cfg)
        tracker.log_metrics({"val": {"image": {"auroc": 0.5}}}, step=3)
        tracker.save_run_file(tmp_path)
    finally:
        tracker.end()
    run_id = tracking.read_run_file(tmp_path)
    run = mlflow.tracking.MlflowClient().get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.metrics["val/image/auroc"] == 0.5
    assert run.data.params["train.lr"] == "0.1"
    assert run.data.tags["tracking_fallback"] == "true"
    assert run.data.tags["face_group"] == "top-bottom"
    assert run.data.tags["stage"] == "training"
    # Resuming (e.g. --resume or evaluate.py) continues the same run.
    again = tracking.Tracker().start(cfg, "cfg.yaml", run_name="t", stage="evaluation", resume_run_id=run_id)
    try:
        assert again.run_id == run_id and again.resumed
    finally:
        again.end()
    tags = mlflow.tracking.MlflowClient().get_run(run_id).data.tags
    assert tags["stage"] == "training" and tags["last_stage"] == "evaluation"
