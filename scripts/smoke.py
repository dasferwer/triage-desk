"""Проверяем через HTTP автоматическую сортировку, ручную правку и экспорт разметки."""

import json
import os
import time
import urllib.error
import urllib.request
from uuid import uuid4

BASE = os.environ.get("API_URL", "http://localhost:8000")


def request(path, token=None, method="GET", body=None, headers=None, expected=(200, 201, 202)):
    data = json.dumps(body).encode() if body is not None else None
    supplied = {"Content-Type": "application/json", **(headers or {})}
    if token:
        supplied["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=data, headers=supplied, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            status = response.status
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        status = error.code
        payload = json.load(error)
    assert status in expected, (path, status, payload)
    return payload


def login(email="admin@example.com"):
    return request(
        "/auth/login", method="POST", body={"email": email, "password": "TriageDeskDemo123!"}
    )["access_token"]


def submit(token, value, key=None, language="en"):
    return request(
        "/tickets",
        token,
        "POST",
        {"text": value, "language": language},
        {"Idempotency-Key": key or str(uuid4())},
    )


def wait_ticket(token, ticket_id, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = request("/tickets/" + ticket_id, token)
        if result["status"] in {"auto_routed", "needs_review", "reviewed"}:
            return result
        time.sleep(0.2)
    raise AssertionError("Ticket did not finish: " + ticket_id)


def main():
    admin, customer = login(), login("demo@example.com")
    key = str(uuid4())
    first = submit(customer, "My card has been stolen", key)
    assert submit(customer, "My card has been stolen", key)["id"] == first["id"]
    automatic = wait_ticket(customer, first["id"])
    assert automatic["status"] == "auto_routed"
    assert automatic["final_intent"] == "lost_or_stolen_card"
    assert automatic["final_department"] == "security"
    uncertain = wait_ticket(customer, submit(customer, "my money")["id"])
    assert uncertain["status"] == "needs_review" and uncertain["review_reason"] == "low_confidence"
    russian = wait_ticket(customer, submit(customer, "Не пришла карта", language="ru")["id"])
    assert russian["review_reason"] == "unsupported_language"
    corrected = request(
        "/tickets/" + uncertain["id"] + "/review",
        admin,
        "POST",
        {"intent": "receiving_money", "note": "Уточнил тему у клиента"},
        {"If-Match": str(uncertain["version"])},
    )
    assert corrected["status"] == "reviewed" and corrected["final_department"] == "transfers"
    request(
        "/tickets/" + uncertain["id"] + "/review",
        admin,
        "POST",
        {"intent": "card_arrival"},
        {"If-Match": str(uncertain["version"])},
        expected=(412,),
    )
    export = request("/feedback/export", admin)
    assert any(
        row["ticket_id"] == uncertain["id"] and row["intent"] == "receiving_money"
        for row in export["rows"]
    )
    request("/feedback/export", customer, expected=(403,))
    request(
        "/tickets/" + first["id"] + "/review",
        customer,
        "POST",
        {"intent": "card_arrival"},
        {"If-Match": str(automatic["version"])},
        expected=(403,),
    )
    models = request("/models", admin)
    assert any(row["active"] for row in models)
    print(
        json.dumps(
            {
                "ok": True,
                "automatic_intent": automatic["final_intent"],
                "uncertain_to_review": True,
                "unsupported_language_to_review": True,
                "operator_correction": True,
                "stale_review_rejected": True,
                "feedback_export": True,
                "ticket_id": first["id"],
            }
        )
    )


if __name__ == "__main__":
    main()
