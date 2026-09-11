import hashlib
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from triagedesk.config import settings
from triagedesk.ml import Classifier, clean_text, digest, load_classifier, route_reason

TRAIN = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/train.py"))


def test_splits_do_not_share_normalized_text():
    fit, calibration, validation, test, source = TRAIN["prepare_data"](Path("data"))
    parts = [{value for value, _ in rows} for rows in [fit, calibration, validation, test]]
    assert all(not parts[i] & parts[j] for i in range(4) for j in range(i + 1, 4))
    assert len(test) == 3080
    assert len({label for _, label in fit}) == 77
    assert source["official_train_rows"] == 10003


def test_saved_metrics_match_model_predictions():
    root = Path(settings.model_dir) / "banking77-v1"
    classifier = Classifier(root)
    report = json.loads((root / "evaluation.json").read_text())
    test = TRAIN["read_split"](Path("data/test.csv"))
    probabilities, _ = classifier.probabilities([value for value, _ in test])
    predictions = classifier.classes[probabilities.argmax(axis=1)]
    actual = f1_score([label for _, label in test], predictions, average="macro")
    assert actual == pytest.approx(report["models"][report["selected_model"]]["test"]["macro_f1"])
    assert actual > 0.8
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_model_tampering_is_detected(tmp_path):
    root = tmp_path / "banking77-v1"
    shutil.copytree(Path(settings.model_dir) / "banking77-v1", root)
    with (root / "weights.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        Classifier(root)
    with pytest.raises(ValueError, match="Manifest"):
        load_classifier(str(tmp_path), "banking77-v1", "invalid")
    with pytest.raises(ValueError, match="version"):
        load_classifier(str(tmp_path), "../other", "hash")


def test_redaction_and_routing_guards():
    assert clean_text(" A@Example.COM, +7 (999) 123-45-67 ") == "redacted_email , redacted_number"
    assert clean_text(clean_text("ＡＢＣ test")) == "abc test"
    assert route_reason("known text", "en", 0.3, 0.5, True) == "low_confidence"
    assert route_reason("known text", "en", 0.5, 0.5, True) is None
    assert route_reason("xyz", "en", 0.99, 0.5, False) == "no_known_terms"


def test_threshold_requires_enough_evidence():
    threshold, stats = TRAIN["choose_threshold"](
        ["a"] * 10, np.array([[0.99, 0.01]] * 10), np.array(["a", "b"])
    )
    assert threshold > 1 and stats["accepted"] == 0
    threshold, stats = TRAIN["choose_threshold"](
        ["a"] * 100, np.array([[0.99, 0.01]] * 100), np.array(["a", "b"])
    )
    assert threshold <= 0.99 and stats["wilson_lower_95"] >= 0.9


def test_feedback_excludes_holdouts_and_conflicting_labels(tmp_path):
    def row(ticket, value, intent, review_id):
        return {
            "ticket_id": ticket,
            "text": value,
            "intent": intent,
            "review_id": review_id,
            "language": "en",
            "text_sha256": hashlib.sha256(value.encode()).hexdigest(),
        }

    path = tmp_path / "feedback.json"
    rows = [
        row("1", "new issue", "a", 1),
        row("1", "new issue", "b", 2),
        row("2", "held issue", "a", 3),
        row("3", "base issue", "b", 4),
        row("4", "ambiguous", "a", 5),
        row("5", "ambiguous", "b", 6),
    ]
    path.write_text(json.dumps({"rows": rows}))
    fit, report = TRAIN["add_feedback"](
        [("base issue", "a"), ("other base", "b")], [[("held issue", "a")]], path
    )
    assert ("new issue", "b") in fit and ("new issue", "a") not in fit
    assert len(fit) == 3
    assert report["skipped"] == {
        "held_out_overlap": 1,
        "training_duplicate": 1,
        "conflicting_labels": 1,
    }


def test_data_checksum_rejects_modified_input(tmp_path):
    shutil.copytree("data", tmp_path / "data")
    (tmp_path / "data/train.csv").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        TRAIN["prepare_data"](tmp_path / "data")


def test_registered_manifest_loads_without_pickle():
    directory = Path(settings.model_dir) / "banking77-v1"
    classifier = load_classifier(
        settings.model_dir, "banking77-v1", digest(directory / "manifest.json")
    )
    assert classifier.predict("I forgot my passcode")["intent"] == "passcode_forgotten"
    assert set(classifier.manifest["files_sha256"]) == {"vectors.json", "weights.npz"}
