import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", " redacted_email ", value)
    value = re.sub(r"(?<!\w)\+?\d[\d ()-]{5,}\d(?!\w)", " redacted_number ", value)
    return " ".join(value.split())


def department(intent: str) -> str:
    if any(word in intent for word in ("stolen", "compromised", "not_recognised")):
        return "security"
    if any(word in intent for word in ("identity", "verify", "source_of_funds")):
        return "verification"
    if any(word in intent for word in ("cash", "atm")):
        return "cash_operations"
    if any(word in intent for word in ("top_up", "topping_up", "balance_not_updated_after_cheque")):
        return "topups"
    if any(
        word in intent for word in ("transfer", "beneficiary", "receiving_money", "direct_debit")
    ):
        return "transfers"
    if any(word in intent for word in ("payment", "refund", "transaction", "exchange", "charge")):
        return "payments"
    if any(word in intent for word in ("card", "pin", "visa", "mastercard")):
        return "cards"
    return "accounts"


def route_reason(
    value: str, language: str, probability: float, threshold: float, known: bool
) -> str | None:
    if language != "en" or re.search(r"[\u0400-\u04ff]", value):
        return "unsupported_language"
    if not known:
        return "no_known_terms"
    if probability < threshold:
        return "low_confidence"
    return None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["format"] != 1 or set(manifest["files_sha256"]) != {"vectors.json", "weights.npz"}:
        raise ValueError("Unsupported model artifact")
    for name, expected in manifest["files_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Model checksum mismatch: {name}")
    return manifest


class Classifier:
    def __init__(self, directory: Path):
        self.manifest = read_manifest(directory)
        config = json.loads((directory / "vectors.json").read_text())
        self.classes = np.array(config["classes"])
        self.vectors = []
        # Загружаем только числа и словари; pickle для моделей здесь не нужен.
        with np.load(directory / "weights.npz", allow_pickle=False) as arrays:
            self.weights = arrays["weights"]
            self.intercept = arrays["intercept"]
            for i, spec in enumerate(config["vectorizers"]):
                spec["params"]["ngram_range"] = tuple(spec["params"]["ngram_range"])
                vector = TfidfVectorizer(**spec["params"], vocabulary=spec["vocabulary"])
                vector.idf_ = arrays[f"idf_{i}"]
                self.vectors.append(vector)
        self.temperature = float(self.manifest["temperature"])
        self.threshold = float(self.manifest["threshold"])

    def probabilities(self, values):
        texts = [clean_text(value) for value in values]
        matrices = [v.transform(texts) for v in self.vectors]
        features = hstack(matrices, format="csr")
        logits = features @ self.weights.T + self.intercept
        return softmax(logits / self.temperature, axis=1), np.asarray(
            matrices[0].getnnz(axis=1)
        ) > 0

    def predict(self, value: str, language="en") -> dict:
        probabilities, known = self.probabilities([value])
        order = np.argsort(probabilities[0])[::-1][:3]
        choices = [
            {
                "intent": str(self.classes[i]),
                "department": department(str(self.classes[i])),
                "score": float(probabilities[0, i]),
            }
            for i in order
        ]
        reason = route_reason(value, language, choices[0]["score"], self.threshold, bool(known[0]))
        return {
            "intent": choices[0]["intent"],
            "department": choices[0]["department"],
            "confidence": choices[0]["score"],
            "threshold": self.threshold,
            "top_choices": choices,
            "review_reason": reason,
            "status": "needs_review" if reason else "auto_routed",
        }


@lru_cache(maxsize=2)
def load_classifier(root: str, version: str, expected_manifest_hash: str) -> Classifier:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", version):
        raise ValueError("Invalid model version")
    directory = Path(root) / version
    if digest(directory / "manifest.json") != expected_manifest_hash:
        raise ValueError("Manifest differs from registered model")
    return Classifier(directory)
