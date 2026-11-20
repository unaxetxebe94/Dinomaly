import numpy as np
import pytest
import torch

from dinomaly.metrics import (
    AnomalyEvaluator,
    EvalConfig,
    aggregate_images,
    binary_report,
    per_class_reports,
    to_tracking_metrics,
)
from dinomaly.tracking import finite_metrics

DETECTION_KEYS = {
    "auroc", "ap", "precision", "recall", "f1", "accuracy",
    "tp", "fp", "fn", "tn", "threshold",
}


def test_binary_report_counts_at_given_threshold():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.6, 0.2, 0.7, 0.3, 0.1, 0.05])
    r = binary_report(y, s, threshold=0.5)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 3)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)
    assert r["accuracy"] == pytest.approx(5 / 7)
    assert r["threshold"] == 0.5
    # Without a threshold the best-F1 one of the split is used.
    best = binary_report(y, s)
    assert best["f1"] == pytest.approx(best["f1_best"])
    assert best["threshold"] == pytest.approx(best["best_threshold"])


def test_binary_report_threshold_is_inclusive():
    r = binary_report(np.array([1, 0]), np.array([0.5, 0.4]), threshold=0.5)
    assert (r["tp"], r["fp"]) == (1, 0)


def test_per_class_excludes_other_defect_classes():
    y = np.array([1, 1, 1, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.3])
    cats = [{1}, {2}, {1, 2}, set(), set()]
    out = per_class_reports(y, s, cats, threshold=0.5, name_of=lambda c: f"c{c}")
    assert set(out) == {"c1", "c2"}
    # class 1: samples 0, 2 vs normals 3, 4 -> all correct.
    assert (out["c1"]["tp"], out["c1"]["fn"], out["c1"]["tn"], out["c1"]["fp"]) == (2, 0, 2, 0)
    assert out["c1"]["negatives"] == 2
    # class 2: samples 1 (missed), 2 (hit); sample 0 (only class 1) excluded.
    assert (out["c2"]["tp"], out["c2"]["fn"]) == (1, 1)
    assert out["c2"]["positives"] + out["c2"]["negatives"] == 4


def test_aggregate_images_topk_and_categories():
    records = {
        "a": [(0.1, 0), (0.9, 1), (0.5, 0)],
        "b": [(0.2, 0), (0.3, 0)],
    }
    scores, labels, cats, uids = aggregate_images(records, topk=2, image_categories={"a": {3}})
    got = dict(zip(uids, zip(scores, labels)))
    assert got["a"][0] == pytest.approx(0.7) and got["a"][1] == 1
    assert got["b"][0] == pytest.approx(0.25) and got["b"][1] == 0
    assert cats[uids.index("a")] == {3} and cats[uids.index("b")] == set()


def _synthetic_evaluator() -> AnomalyEvaluator:
    ev = AnomalyEvaluator(
        classes={1: "Glue on face", 2: "Slip skin"},
        cfg=EvalConfig(pro_max_tiles=8, pro_size=16, image_topk=1),
    )
    gen = torch.Generator().manual_seed(0)
    for b in range(4):
        maps = torch.rand(4, 1, 32, 32, generator=gen) * 0.2
        masks = torch.zeros(4, 32, 32, dtype=torch.uint8)
        masks[0, 5:10, 5:10] = 1
        maps[0, 0, 5:10, 5:10] += 0.8
        if b % 2:
            masks[1, 1:4, 1:4] = 2
            maps[1, 0, 1:4, 1:4] += 0.5
        ev.update(maps, masks, [f"d{b}", f"d{b}", f"n{b}", f"n{b}"])
    return ev


def test_evaluator_tracking_keys_overall_and_per_class():
    result = _synthetic_evaluator().compute()
    flat = finite_metrics(to_tracking_metrics(result, "val"))
    for level in ("tile", "image"):
        for key in DETECTION_KEYS:
            assert f"val/{level}/{key}" in flat
            assert f"val/{level}/class/glue_on_face/{key}" in flat
            assert f"val/{level}/class/slip_skin/{key}" in flat
    for key in ("auroc", "ap", "f1", "aupro"):
        assert f"val/pixel/{key}" in flat
    assert "val/pixel/class/slip_skin/auroc" in flat
    assert flat["val/image/tp"] + flat["val/image/fn"] == 4  # 4 defective images
    assert flat["val/image/tn"] + flat["val/image/fp"] == 4  # 4 normal images
    assert flat["val/image/class/slip_skin/positives"] == 2
    assert flat["val/tile/tp"] + flat["val/tile/fn"] == 6


def test_evaluator_uses_given_thresholds_on_test():
    result = _synthetic_evaluator().compute(tile_threshold=10.0, image_threshold=10.0)
    for level in ("tile", "image"):
        node = result[level]
        assert node["threshold"] == 10.0
        assert node["tp"] == 0 and node["fp"] == 0
        assert node["f1_best"] == pytest.approx(1.0)
        assert node["per_category"]["Glue on face"]["threshold"] == 10.0
