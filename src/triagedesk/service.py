import json
from datetime import timedelta
from uuid import uuid4

from sqlalchemy import text

from .config import settings
from .db import engine


async def event(conn, ticket_id, kind, details=None):
    await conn.execute(
        text(
            "INSERT INTO ticket_events(ticket_id,type,details) VALUES (:id,:type,CAST(:details AS jsonb))"
        ),
        {"id": ticket_id, "type": kind, "details": json.dumps(details or {})},
    )


async def heartbeat(name):
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO worker_heartbeats(name) VALUES (:name) ON CONFLICT(name) DO UPDATE SET updated_at=clock_timestamp()"
            ),
            {"name": name},
        )


async def reserve_due(limit=20):
    payloads = []
    async with engine.begin() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT * FROM tickets WHERE status='pending' OR (status IN ('queued','processing') AND lease_until<=clock_timestamp()) ORDER BY created_at LIMIT :limit FOR UPDATE SKIP LOCKED"
                    ),
                    {"limit": limit},
                )
            )
            .mappings()
            .all()
        )
        for row in rows:
            if row["attempts"] >= settings.max_attempts:
                await conn.execute(
                    text(
                        "UPDATE tickets SET status='needs_review',review_reason='inference_unavailable',lease_until=NULL,version=version+1,updated_at=clock_timestamp() WHERE id=:id"
                    ),
                    {"id": row["id"]},
                )
                await event(conn, row["id"], "sent_to_review", {"reason": "inference_unavailable"})
                continue
            generation = uuid4()
            now = await conn.scalar(text("SELECT clock_timestamp()"))
            await conn.execute(
                text(
                    "UPDATE tickets SET status='queued',generation=:token,lease_until=:until,updated_at=:now WHERE id=:id"
                ),
                {
                    "id": row["id"],
                    "token": generation,
                    "until": now + timedelta(seconds=settings.lease_seconds),
                    "now": now,
                },
            )
            await event(conn, row["id"], "queued" if row["status"] == "pending" else "recovered")
            payloads.append({"ticket_id": str(row["id"]), "generation": str(generation)})
    return payloads


async def claim(ticket_id, generation):
    async with engine.begin() as conn:
        row = (
            (
                await conn.execute(
                    text(
                        "UPDATE tickets SET status='processing',attempts=attempts+1,lease_until=clock_timestamp()+make_interval(secs=>:lease),updated_at=clock_timestamp() WHERE id=:id AND generation=:generation AND status='queued' RETURNING *"
                    ),
                    {"id": ticket_id, "generation": generation, "lease": settings.lease_seconds},
                )
            )
            .mappings()
            .first()
        )
        if row:
            await event(conn, ticket_id, "processing", {"attempt": row["attempts"]})
            manifest_hash = await conn.scalar(
                text("SELECT manifest_sha256 FROM models WHERE version=:version"),
                {"version": row["model_version"]},
            )
            return {**row, "manifest_sha256": manifest_hash}
    return None


async def complete(ticket_id, generation, result):
    async with engine.begin() as conn:
        # Проверяем поколение: старый воркер мог вернуться уже после повторной обработки.
        saved = await conn.scalar(
            text("""UPDATE tickets SET status=:status,predicted_intent=:intent,department=:department,
        confidence=:confidence,threshold=:threshold,top_choices=CAST(:choices AS jsonb),review_reason=:reason,
        final_intent=:final_intent,final_department=:final_department,lease_until=NULL,version=version+1,updated_at=clock_timestamp()
        WHERE id=:id AND generation=:generation AND status='processing' RETURNING id"""),
            {
                "id": ticket_id,
                "generation": generation,
                "status": result["status"],
                "intent": result["intent"],
                "department": result["department"],
                "confidence": result["confidence"],
                "threshold": result["threshold"],
                "choices": json.dumps(result["top_choices"]),
                "reason": result["review_reason"],
                "final_intent": result["intent"] if result["status"] == "auto_routed" else None,
                "final_department": result["department"]
                if result["status"] == "auto_routed"
                else None,
            },
        )
        if saved:
            await event(
                conn,
                ticket_id,
                "classified",
                {
                    "status": result["status"],
                    "model_version": await conn.scalar(
                        text("SELECT model_version FROM tickets WHERE id=:id"), {"id": ticket_id}
                    ),
                },
            )
        return bool(saved)
