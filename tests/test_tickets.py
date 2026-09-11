import asyncio
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from triagedesk.config import settings
from triagedesk.db import engine
from triagedesk.service import claim, complete, reserve_due
from triagedesk.worker import handle, infer


async def create(
    client, identities, value="My card has been stolen", key="ticket-1", language="en"
):
    response = await client.post(
        "/tickets",
        headers={**identities["alice"]["headers"], "Idempotency-Key": key},
        json={"text": value, "language": language},
    )
    assert response.status_code in {200, 202}, response.text
    return response.json()


async def process():
    payloads = await reserve_due()
    for payload in payloads:
        await handle(payload)
    return payloads


async def test_concurrent_idempotency(client, identities):
    rows = await asyncio.gather(*[create(client, identities) for _ in range(15)])
    assert len({row["id"] for row in rows}) == 1
    response = await client.post(
        "/tickets",
        headers={**identities["alice"]["headers"], "Idempotency-Key": "ticket-1"},
        json={"text": "Different request"},
    )
    assert response.status_code == 409
    async with engine.connect() as conn:
        assert await conn.scalar(text("SELECT count(*) FROM tickets")) == 1
        assert (
            await conn.scalar(text("SELECT count(*) FROM ticket_events WHERE type='created'")) == 1
        )


async def test_model_prediction_and_duplicate_message(client, identities):
    row = await create(client, identities)
    payload = (await process())[0]
    assert await handle(payload) is False
    response = await client.get("/tickets/" + row["id"], headers=identities["alice"]["headers"])
    result = response.json()
    assert result["status"] == "auto_routed"
    assert result["final_intent"] == "lost_or_stolen_card"
    assert result["final_department"] == "security"
    assert response.headers["ETag"] == '"2"'
    async with engine.connect() as conn:
        assert (
            await conn.scalar(text("SELECT count(*) FROM ticket_events WHERE type='classified'"))
            == 1
        )


@pytest.mark.parametrize(
    ("value", "language", "reason"),
    [
        ("Помогите вернуть карту", "ru", "unsupported_language"),
        ("Помогите вернуть карту", "en", "unsupported_language"),
        ("zxqv jzzqq xvvqq", "en", "no_known_terms"),
    ],
)
async def test_manual_review_guards(client, identities, value, language, reason):
    row = await create(client, identities, value, language=language)
    await process()
    result = (
        await client.get("/tickets/" + row["id"], headers=identities["alice"]["headers"])
    ).json()
    assert result["status"] == "needs_review"
    assert result["review_reason"] == reason
    assert result["final_intent"] is None


async def test_ownership_and_admin_access(client, identities):
    row = await create(client, identities)
    for suffix in ["", "/events"]:
        assert (
            await client.get("/tickets/" + row["id"] + suffix, headers=identities["bob"]["headers"])
        ).status_code == 404
    assert (await client.get("/tickets", headers=identities["bob"]["headers"])).json() == []
    assert (
        await client.get("/tickets/" + row["id"], headers=identities["admin"]["headers"])
    ).status_code == 200
    assert (
        await client.get("/feedback/export", headers=identities["alice"]["headers"])
    ).status_code == 403
    assert (await client.get("/models", headers=identities["alice"]["headers"])).status_code == 403


async def test_review_conflict_and_export(client, identities):
    row = await create(client, identities)
    await process()
    headers = {**identities["admin"]["headers"], "If-Match": '"2"'}
    url = "/tickets/" + row["id"] + "/review"
    responses = await asyncio.gather(
        *[
            client.post(url, headers=headers, json={"intent": intent})
            for intent in ["card_arrival", "change_pin"]
        ]
    )
    assert sorted(r.status_code for r in responses) == [200, 412]
    winner = next(r.json() for r in responses if r.status_code == 200)
    assert winner["predicted_intent"] == "lost_or_stolen_card"
    assert winner["status"] == "reviewed"
    exports = (await client.get("/feedback/export", headers=identities["admin"]["headers"])).json()
    assert len(exports["rows"]) == 1
    assert exports["rows"][0]["intent"] == winner["final_intent"]
    assert exports["rows"][0]["text_sha256"]
    assert (
        await client.post(
            url, headers={**headers, "If-Match": "3"}, json={"intent": "activate_my_card"}
        )
    ).status_code == 200
    page = (
        await client.get("/feedback/export?after=1&limit=1", headers=identities["admin"]["headers"])
    ).json()
    assert page["rows"][0]["intent"] == "activate_my_card"
    assert page["next_after"] == 2


async def test_review_requires_completed_ticket_and_known_label(client, identities):
    row = await create(client, identities)
    url = "/tickets/" + row["id"] + "/review"
    headers = identities["admin"]["headers"]
    assert (
        await client.post(url, headers=headers, json={"intent": "card_arrival"})
    ).status_code == 428
    assert (
        await client.post(
            url, headers={**headers, "If-Match": "1"}, json={"intent": "card_arrival"}
        )
    ).status_code == 409
    await process()
    assert (
        await client.post(url, headers={**headers, "If-Match": "2"}, json={"intent": "made_up"})
    ).status_code == 422
    assert (
        await client.post(
            url, headers={**headers, "If-Match": "bad"}, json={"intent": "card_arrival"}
        )
    ).status_code == 422


async def test_stale_worker_cannot_overwrite_new_generation(client, identities):
    row = await create(client, identities)
    old = (await reserve_due())[0]
    claimed = await claim(UUID(old["ticket_id"]), UUID(old["generation"]))
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE tickets SET lease_until=clock_timestamp()-interval '1 second'")
        )
    new = (await reserve_due())[0]
    assert old["generation"] != new["generation"]
    assert not await complete(UUID(old["ticket_id"]), UUID(old["generation"]), infer(claimed))
    assert await handle(new)
    result = (
        await client.get("/tickets/" + row["id"], headers=identities["alice"]["headers"])
    ).json()
    assert result["attempts"] == 2


async def test_result_and_event_commit_together(client, identities, monkeypatch):
    await create(client, identities)
    payload = (await reserve_due())[0]
    claimed = await claim(UUID(payload["ticket_id"]), UUID(payload["generation"]))
    original = AsyncConnection.execute

    async def fail(self, statement, *args, **kwargs):
        if "INSERT INTO ticket_events" in str(statement):
            raise RuntimeError("Injected event failure")
        return await original(self, statement, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(AsyncConnection, "execute", fail)
        with pytest.raises(RuntimeError):
            await complete(UUID(payload["ticket_id"]), UUID(payload["generation"]), infer(claimed))
    async with engine.connect() as conn:
        assert await conn.scalar(text("SELECT status FROM tickets")) == "processing"
        assert await conn.scalar(text("SELECT predicted_intent FROM tickets")) is None
    assert await complete(UUID(payload["ticket_id"]), UUID(payload["generation"]), infer(claimed))


async def test_failed_inference_goes_to_operator(client, identities, monkeypatch):
    await create(client, identities)
    monkeypatch.setattr(settings, "max_attempts", 1)
    payload = (await reserve_due())[0]
    await claim(UUID(payload["ticket_id"]), UUID(payload["generation"]))
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE tickets SET lease_until=clock_timestamp()-interval '1 second'")
        )
    assert await reserve_due() == []
    async with engine.connect() as conn:
        row = (await conn.execute(text("SELECT status,review_reason FROM tickets"))).one()
    assert row == ("needs_review", "inference_unavailable")


async def test_lost_publication_is_recoverable(client, identities):
    await create(client, identities)
    old = (await reserve_due())[0]
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE tickets SET lease_until=clock_timestamp()-interval '1 second'")
        )
    new = (await reserve_due())[0]
    assert await handle(old) is False
    assert await handle(new) is True


async def test_model_is_pinned_at_creation(client, identities):
    old = await create(client, identities)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO models SELECT 'banking77-v2',manifest_sha256,threshold,evaluation,created_at FROM models"
            )
        )
    response = await client.post(
        "/models/banking77-v2/activate", headers=identities["admin"]["headers"]
    )
    assert response.status_code == 200
    new = await create(client, identities, key="new-model")
    assert old["model_version"] == "banking77-v1"
    assert new["model_version"] == "banking77-v2"
    again = await create(client, identities)
    assert again["model_version"] == "banking77-v1"


async def test_redaction_before_storage_and_export(client, identities):
    row = await create(
        client, identities, "Please contact me at person@example.com about card 4242 4242 4242 4242"
    )
    assert "person@" not in row["text"] and "4242" not in row["text"]
    assert "redacted_email" in row["text"] and "redacted_number" in row["text"]
    async with engine.connect() as conn:
        assert "person@example.com" not in await conn.scalar(text("SELECT text FROM tickets"))


async def test_blank_and_missing_auth(client, identities):
    assert (
        await client.post(
            "/tickets", headers={"Idempotency-Key": "x"}, json={"text": "Hello there"}
        )
    ).status_code == 401
    assert (
        await client.post(
            "/tickets",
            headers={**identities["alice"]["headers"], "Idempotency-Key": "x"},
            json={"text": "   "},
        )
    ).status_code == 422


async def test_worker_health_uses_current_schema():
    import sys

    from triagedesk.service import heartbeat

    await heartbeat("classifier")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "scripts/worker_health.py",
        "classifier",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, error = await process.communicate()
    assert process.returncode == 0, error.decode()


async def test_intents_cover_all_departments(client, identities):
    rows = (await client.get("/intents", headers=identities["alice"]["headers"])).json()
    assert len(rows) == 77
    assert len({row["department"] for row in rows}) == 8


async def test_normalization_expansion_and_note_nul(client, identities):
    headers = {**identities["alice"]["headers"], "Idempotency-Key": "large"}
    response = await client.post("/tickets", headers=headers, json={"text": "\ufdfa" * 1000})
    assert response.status_code == 422
    row = await create(client, identities)
    await process()
    response = await client.post(
        "/tickets/" + row["id"] + "/review",
        headers={**identities["admin"]["headers"], "If-Match": "2"},
        json={"intent": "card_arrival", "note": "bad\u0000note"},
    )
    assert response.status_code == 422
