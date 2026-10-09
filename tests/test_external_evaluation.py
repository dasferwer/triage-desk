import json
import runpy
from pathlib import Path

import pytest

EVAL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/evaluate_external.py"))


def test_wilson_and_zero_denominator():
    rate = EVAL["rate"]
    assert rate(0, 0)["rate"] is None
    assert rate(0, 10)["wilson_95"][1] == pytest.approx(0.27753, abs=0.00001)
    assert rate(10, 10)["wilson_95"][0] == pytest.approx(0.72247, abs=0.00001)


def test_ood_automatic_is_error_and_manual_is_rejection():
    rows = [
        {"status": "auto_routed", "intent": "change_pin", "review_reason": None},
        {"status": "needs_review", "intent": "change_pin", "review_reason": "low_confidence"},
    ]
    result = EVAL["summarize"](rows, True)
    assert result["rejection_coverage"]["rate"] == 0.5
    assert result["automatic_error_all_rows"]["rate"] == 0.5
    assert result["automatic_error_conditional"]["rate"] == 1.0


def test_external_result_reproduces_saved_artifact(tmp_path):
    report = EVAL["evaluate"](tmp_path / "report.json")
    frozen = json.loads(Path("docs/external-evaluation.json").read_text())
    assert report == frozen
    assert report["remaining_family_overlap"] == 0
    assert report["groups"]["ood"]["rows"] == 1800
    assert report["groups"]["shift"]["rows"] == 90
    assert EVAL["family"]("HELLO,  world!") == EVAL["family"]("hello world")
