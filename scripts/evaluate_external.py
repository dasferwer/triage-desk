"""Независимая проверка замороженной модели; обучение и подбор порога запрещены."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from triagedesk.ml import Classifier, clean_text, digest

ROOT = Path(__file__).resolve().parents[1]


def family(value):
    return hashlib.sha256(" ".join(re.findall(r"\w+", clean_text(value))).encode()).hexdigest()


def rate(success, count):
    if not count:
        return {"count": 0, "success": success, "rate": None, "wilson_95": None}
    p, z = success / count, 1.959963984540054
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return {
        "count": count,
        "success": success,
        "rate": p,
        "wilson_95": [center - radius, center + radius],
    }


def summarize(rows, ood):
    automatic = [row for row in rows if row["status"] == "auto_routed"]
    errors = len(automatic) if ood else sum(row["intent"] != row["expected"] for row in automatic)
    return {
        "rows": len(rows),
        "automatic_coverage": rate(len(automatic), len(rows)),
        "rejection_coverage": rate(len(rows) - len(automatic), len(rows)),
        "automatic_error_all_rows": rate(errors, len(rows)),
        "automatic_error_conditional": rate(errors, len(automatic)),
        "reasons": dict(Counter(row["review_reason"] or "auto_routed" for row in rows)),
    }


def evaluate(output):
    directory = ROOT / "data/external/clinc150"
    source = json.loads((directory / "source.json").read_text())
    for name, expected in source["files_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"External checksum mismatch: {name}")
    model_dir = ROOT / "models/banking77-v1"
    artifact = json.loads((model_dir / "evaluation.json").read_text())
    if artifact["dataset"]["feedback"] is not None:
        raise ValueError("Feedback artifact requires an explicit feedback overlap corpus")
    model = Classifier(model_dir)
    domains = json.loads((directory / "domains.json").read_text())
    domain_by_intent = {intent: domain for domain, intents in domains.items() for intent in intents}
    payload = json.loads((directory / "data_full.json").read_text())
    banking_source = json.loads((ROOT / "data/source.json").read_text())
    for name, expected in banking_source["files_sha256"].items():
        if digest(ROOT / "data" / name) != expected:
            raise ValueError(f"BANKING77 checksum mismatch: {name}")
    blocked = set()
    for split in ["train", "test"]:
        with (ROOT / f"data/{split}.csv").open(newline="") as stream:
            blocked.update(family(row["text"]) for row in csv.DictReader(stream))
    selected, excluded, seen = [], Counter(), set()
    for index, (text, intent) in enumerate(payload["test"]):
        domain = domain_by_intent[intent]
        kind = "ood" if domain in source["ood_domains"] else "shift"
        if kind == "shift" and intent not in source["shift_mapping"]:
            continue
        key = family(text)
        if key in blocked:
            excluded["banking77_family_overlap"] += 1
            continue
        if key in seen:
            excluded["external_family_duplicate"] += 1
            continue
        seen.add(key)
        selected.append(
            {
                "id": index,
                "text": text,
                "source_intent": intent,
                "domain": domain,
                "kind": kind,
                "expected": None if kind == "ood" else source["shift_mapping"][intent],
                "family_sha256": key,
            }
        )
    # Замораживаем метки/порядок до вызова inference; никаких оптимизаций по ним.
    frozen_hash = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    predictions = [{**row, **model.predict(row["text"])} for row in selected]
    groups = defaultdict(list)
    for row in predictions:
        groups[row["kind"]].append(row)
        groups[row["kind"] + "/" + row["domain"]].append(row)
        groups[row["kind"] + "/intent/" + row["source_intent"]].append(row)
    report = {
        "source": source,
        "source_manifest_sha256": digest(directory / "source.json"),
        "model_manifest_sha256": digest(model_dir / "manifest.json"),
        "threshold": model.threshold,
        "frozen_selection_sha256": frozen_hash,
        "excluded": dict(excluded),
        "remaining_family_overlap": len(seen & blocked),
        "banking77_family_count": len(blocked),
        "selected_rows": len(selected),
        "groups": {
            key: summarize(rows, key.startswith("ood")) for key, rows in sorted(groups.items())
        },
        "limits": "Wilson intervals are row-binomial descriptive intervals; paraphrase families may remain dependent. Shift mapping covers only three intents, not all 77. OOD measures rejection, not 150-class accuracy.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "docs/external-evaluation.json")
    print(json.dumps(evaluate(parser.parse_args().output)["groups"], indent=2))
