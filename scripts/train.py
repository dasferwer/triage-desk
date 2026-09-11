"""Обучаем две модели и сохраняем результат вместе с данными о проверке качества."""

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn
from scipy.optimize import minimize_scalar
from scipy.sparse import hstack
from scipy.special import logsumexp, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from threadpoolctl import threadpool_limits

from triagedesk.ml import Classifier, clean_text, digest

ROOT = Path(__file__).resolve().parents[1]


def label(value):
    return value.lower().rstrip("?")


def read_split(path):
    with path.open(newline="") as stream:
        return [(clean_text(row["text"]), label(row["category"])) for row in csv.DictReader(stream)]


def prepare_data(directory):
    source = json.loads((directory / "source.json").read_text())
    for name, expected in source["files_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    train, test = read_split(directory / "train.csv"), read_split(directory / "test.csv")
    groups = defaultdict(set)
    for value, intent in train:
        groups[value].add(intent)
    # Одинаковые тексты не должны оказаться по обе стороны проверки качества.
    test_texts = {value for value, _ in test}
    clean = sorted(
        (value, next(iter(intents)))
        for value, intents in groups.items()
        if len(intents) == 1 and value not in test_texts
    )
    fit, held = train_test_split(
        clean, test_size=0.25, random_state=42, stratify=[y for _, y in clean]
    )
    calibration, validation = train_test_split(
        held, test_size=0.5, random_state=43, stratify=[y for _, y in held]
    )
    sets = [set(x for x, _ in split) for split in [fit, calibration, validation, test]]
    assert all(not (sets[i] & sets[j]) for i in range(4) for j in range(i + 1, 4))
    return (
        fit,
        calibration,
        validation,
        test,
        {
            **source,
            "official_train_rows": len(train),
            "official_test_rows": len(test),
            "removed_training_rows": len(train) - len(clean),
            "split_seed": [42, 43],
        },
    )


def add_feedback(fit, held, path):
    payload = json.loads(path.read_text())
    rows = payload["rows"] if isinstance(payload, dict) else payload
    if len(rows) > 10000:
        raise ValueError("Feedback file exceeds 10000 rows")
    latest = {}
    for row in rows:
        if row.get("review_id", 0) > latest.get(row["ticket_id"], {}).get("review_id", 0):
            latest[row["ticket_id"]] = row
    blocked = {value for split in held for value, _ in split}
    existing = {value for value, _ in fit}
    allowed = {intent for _, intent in fit}
    groups = defaultdict(set)
    skipped = Counter()
    for row in latest.values():
        value = clean_text(row["text"])
        if row.get("language") != "en" or row["intent"] not in allowed:
            skipped["unsupported"] += 1
        elif value in blocked:
            skipped["held_out_overlap"] += 1
        elif value in existing:
            skipped["training_duplicate"] += 1
        elif hashlib.sha256(value.encode()).hexdigest() != row["text_sha256"]:
            raise ValueError("Feedback text checksum mismatch")
        else:
            groups[value].add(row["intent"])
    added = [
        (value, next(iter(intents)))
        for value, intents in sorted(groups.items())
        if len(intents) == 1
    ]
    skipped["conflicting_labels"] += sum(len(intents) > 1 for intents in groups.values())
    return fit + added, {"file_sha256": digest(path), "added": len(added), "skipped": dict(skipped)}


def choose_threshold(y, probabilities, classes, target=0.90):
    correct = classes[probabilities.argmax(axis=1)] == np.array(y)
    confidence = probabilities.max(axis=1)
    # Проверяем нижнюю границу Wilson, чтобы не выбрать порог по нескольким удачным ответам.
    for threshold in np.arange(0, 1.001, 0.005):
        mask = confidence >= threshold
        count = int(mask.sum())
        if count < 50:
            continue
        p = float(correct[mask].mean())
        z = 1.96
        lower = (
            p + z * z / (2 * count) - z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
        ) / (1 + z * z / count)
        if lower >= target:
            return float(threshold), {
                "accepted": count,
                "accuracy": p,
                "wilson_lower_95": float(lower),
            }
    return 1.01, {"accepted": 0, "accuracy": None, "wilson_lower_95": None}


def metrics(y, probabilities, classes, threshold):
    prediction = classes[probabilities.argmax(axis=1)]
    accepted = probabilities.max(axis=1) >= threshold
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "automatic_coverage": float(accepted.mean()),
        "automatic_accuracy": float(accuracy_score(np.array(y)[accepted], prediction[accepted]))
        if accepted.any()
        else None,
        "automatic_count": int(accepted.sum()),
        "rows": len(y),
        "per_class": classification_report(y, prediction, output_dict=True, zero_division=0),
    }


def train(version, feedback=None):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", version):
        raise ValueError("Invalid version")
    directory = ROOT / "models" / version
    if directory.exists():
        raise ValueError("Model version already exists; choose a new version")
    started = time.monotonic()
    fit, calibration, validation, test, source = prepare_data(ROOT / "data")
    source["feedback"] = None
    if feedback:
        fit, source["feedback"] = add_feedback(fit, [calibration, validation, test], feedback)
    splits = [fit, calibration, validation, test]
    source["split_rows"] = dict(
        zip(["fit", "calibration", "validation", "test"], map(len, splits), strict=True)
    )
    source["split_sha256"] = {
        name: hashlib.sha256(json.dumps(split, ensure_ascii=False).encode()).hexdigest()
        for name, split in zip(source["split_rows"], splits, strict=True)
    }
    x, y = zip(*fit, strict=True)
    vector_specs = [
        dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True),
        dict(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=30000, sublinear_tf=True
        ),
    ]
    vectors = [TfidfVectorizer(**spec).fit(x) for spec in vector_specs]
    features = [[v.transform([value for value, _ in split]) for v in vectors] for split in splits]
    candidates = {}
    for name, estimator, used in [
        ("word_nb", ComplementNB(alpha=0.3), 1),
        ("word_char_lr", LogisticRegression(C=8, max_iter=400, solver="lbfgs", random_state=42), 2),
    ]:
        print("Training", name, flush=True)
        matrices = [hstack(parts[:used], format="csr") for parts in features]
        estimator.fit(matrices[0], y)
        weights = estimator.feature_log_prob_ if name == "word_nb" else estimator.coef_
        intercept = np.zeros(len(estimator.classes_)) if name == "word_nb" else estimator.intercept_
        scores = [matrix @ weights.T + intercept for matrix in matrices]
        assert np.array_equal(
            estimator.classes_[scores[1].argmax(axis=1)], estimator.predict(matrices[1])
        )
        targets = np.searchsorted(estimator.classes_, [intent for _, intent in calibration])

        def loss(log_temperature, scores=scores, targets=targets):
            scaled = scores[1] / np.exp(log_temperature)
            return float(
                (logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]).mean()
            )

        temperature = float(np.exp(minimize_scalar(loss, bounds=(-3, 3), method="bounded").x))
        probabilities = [softmax(score / temperature, axis=1) for score in scores]
        threshold, threshold_stats = choose_threshold(
            [intent for _, intent in validation], probabilities[2], estimator.classes_
        )
        candidates[name] = {
            "vectors": vectors[:used],
            "specs": vector_specs[:used],
            "weights": weights,
            "intercept": intercept,
            "classes": estimator.classes_,
            "temperature": temperature,
            "threshold": threshold,
            "threshold_selection": threshold_stats,
            "validation": metrics(
                [intent for _, intent in validation],
                probabilities[2],
                estimator.classes_,
                threshold,
            ),
            "test_probabilities": probabilities[3],
        }
    # Выбираем модель до расчёта тестовых метрик. Тест не участвует в выборе порога.
    winner = max(candidates, key=lambda name: candidates[name]["validation"]["macro_f1"])
    chosen = candidates[winner]
    report = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": source,
        "selected_model": winner,
        "selection_rule": "Highest validation macro-F1; temperature on calibration split, threshold on validation split",
        "target_validation_wilson_lower": 0.9,
        "models": {},
    }
    for name, candidate in candidates.items():
        report["models"][name] = {
            key: candidate[key]
            for key in ["temperature", "threshold", "threshold_selection", "validation"]
        }
        report["models"][name]["test"] = metrics(
            [intent for _, intent in test],
            candidate["test_probabilities"],
            candidate["classes"],
            candidate["threshold"],
        )
    directory.mkdir(parents=True)
    config = {
        "classes": chosen["classes"].tolist(),
        "vectorizers": [
            {
                "params": spec,
                "vocabulary": {key: int(value) for key, value in vector.vocabulary_.items()},
            }
            for spec, vector in zip(chosen["specs"], chosen["vectors"], strict=True)
        ],
    }
    (directory / "vectors.json").write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        directory / "weights.npz",
        weights=chosen["weights"],
        intercept=chosen["intercept"],
        **{f"idf_{i}": v.idf_ for i, v in enumerate(chosen["vectors"])},
    )
    manifest = {
        "format": 1,
        "version": version,
        "model": winner,
        "language": "en",
        "classes": chosen["classes"].tolist(),
        "temperature": chosen["temperature"],
        "threshold": chosen["threshold"],
        "sklearn_version": sklearn.__version__,
        "files_sha256": {
            name: digest(directory / name) for name in ["vectors.json", "weights.npz"]
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    restored = Classifier(directory)
    probabilities, _ = restored.probabilities([value for value, _ in test[:50]])
    assert np.allclose(probabilities, chosen["test_probabilities"][:50], atol=1e-10)
    report["serialization_parity"] = True
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "version": version,
                "selected": winner,
                "elapsed_seconds": report["elapsed_seconds"],
                "metrics": {
                    name: {k: v for k, v in item["test"].items() if k != "per_class"}
                    for name, item in report["models"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="banking77-v1")
    parser.add_argument("--feedback", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        train(args.version, args.feedback)
