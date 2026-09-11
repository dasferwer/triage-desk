import hashlib
import json
from typing import Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text

from .auth import User, admin_user, current_user, router
from .db import engine
from .ml import clean_text, department
from .observability import instrument
from .service import event

app = FastAPI(
    title="TriageDesk",
    version="0.1.0",
    description="Sort support tickets by topic, with human review for uncertain predictions.",
)
app.include_router(router)
instrument(app)

TICKET_FIELDS = "id,user_id,text,language,model_version,status,attempts,predicted_intent,department,confidence,threshold,top_choices,review_reason,final_intent,final_department,version,created_at,updated_at"


class TicketInput(BaseModel):
    text: str = Field(min_length=3, max_length=8000)
    language: str = Field(default="en", pattern=r"^[a-z]{2}$")

    @field_validator("text")
    @classmethod
    def meaningful(cls, value):
        if not 3 <= len(clean_text(value)) <= 8000 or "\x00" in value:
            raise ValueError("Normalized text must contain 3 to 8000 characters and no NUL")
        return value


class ReviewInput(BaseModel):
    intent: str = Field(min_length=1, max_length=100)
    note: str = Field(default="", max_length=1000)

    @field_validator("note")
    @classmethod
    def valid_note(cls, value):
        if "\x00" in value:
            raise ValueError("Note must not contain NUL")
        return value


async def get_ticket(conn, ticket_id, user, lock=False):
    row = (
        (
            await conn.execute(
                text(
                    f"SELECT {TICKET_FIELDS} FROM tickets WHERE id=:id"
                    + (" FOR UPDATE" if lock else "")
                ),
                {"id": ticket_id},
            )
        )
        .mappings()
        .first()
    )
    if not row or (row["user_id"] != user.id and user.role != "admin"):
        raise HTTPException(404, "Ticket not found")
    return dict(row)


def expected_version(value):
    if value is None:
        raise HTTPException(428, "If-Match is required")
    try:
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        number = int(value)
        if number < 1:
            raise ValueError
        return number
    except ValueError:
        raise HTTPException(422, "If-Match must contain a ticket version") from None


@app.get("/health", tags=["Operations"])
async def health():
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT name,extract(epoch FROM clock_timestamp()-updated_at) AS age FROM worker_heartbeats"
                    )
                )
            )
            .mappings()
            .all()
        )
        active = await conn.scalar(text("SELECT active_model FROM routing_config WHERE id=1"))
    return {
        "status": "ok",
        "database": "ok",
        "active_model": active,
        "workers": {r["name"]: float(r["age"]) for r in rows},
    }


@app.get("/intents", tags=["Classification"])
async def intents(user: User = Depends(current_user)):
    async with engine.connect() as conn:
        result = await conn.scalar(
            text(
                "SELECT evaluation FROM models JOIN routing_config ON version=active_model WHERE routing_config.id=1"
            )
        )
    classes = result["models"][result["selected_model"]]["test"]["per_class"]
    return [
        {"intent": intent, "department": department(intent)}
        for intent, stats in classes.items()
        if isinstance(stats, dict) and intent not in {"macro avg", "weighted avg"}
    ]


@app.post("/tickets", status_code=202, tags=["Tickets"])
async def create_ticket(
    data: TicketInput,
    response: Response,
    user: User = Depends(current_user),
    idempotency_key: str = Header(min_length=1, max_length=100),
):
    value = clean_text(data.text)
    fingerprint = hashlib.sha256(
        json.dumps({"text": value, "language": data.language}, sort_keys=True).encode()
    ).hexdigest()
    async with engine.begin() as conn:
        await conn.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key,0))"),
            {"key": f"{user.id}:{idempotency_key}"},
        )
        previous = (
            (
                await conn.execute(
                    text(
                        "SELECT id,request_hash FROM tickets WHERE user_id=:user AND idempotency_key=:key"
                    ),
                    {"user": user.id, "key": idempotency_key},
                )
            )
            .mappings()
            .first()
        )
        if previous:
            if previous["request_hash"] != fingerprint:
                raise HTTPException(409, "Idempotency key was used with another ticket")
            row = await get_ticket(conn, previous["id"], user)
            response.status_code = 200
        else:
            model = await conn.scalar(text("SELECT active_model FROM routing_config WHERE id=1"))
            if model is None:
                raise HTTPException(503, "No model is registered")
            ticket_id = uuid4()
            await conn.execute(
                text(
                    "INSERT INTO tickets(id,user_id,text,language,idempotency_key,request_hash,model_version) VALUES (:id,:user,:text,:language,:key,:hash,:model)"
                ),
                {
                    "id": ticket_id,
                    "user": user.id,
                    "text": value,
                    "language": data.language,
                    "key": idempotency_key,
                    "hash": fingerprint,
                    "model": model,
                },
            )
            await event(conn, ticket_id, "created", {"model_version": model})
            row = await get_ticket(conn, ticket_id, user)
    response.headers["ETag"] = f'"{row["version"]}"'
    return row


@app.get("/tickets", tags=["Tickets"])
async def list_tickets(
    status: Literal["pending", "queued", "processing", "auto_routed", "needs_review", "reviewed"]
    | None = None,
    after: UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    user: User = Depends(current_user),
):
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        f"SELECT {TICKET_FIELDS} FROM tickets WHERE (:admin OR user_id=:user) AND (CAST(:status AS text) IS NULL OR status=:status) AND (CAST(:after AS uuid) IS NULL OR id>:after) ORDER BY id LIMIT :limit"
                    ),
                    {
                        "admin": user.role == "admin",
                        "user": user.id,
                        "status": status,
                        "after": after,
                        "limit": limit,
                    },
                )
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


@app.get("/tickets/{ticket_id}", tags=["Tickets"])
async def ticket(ticket_id: UUID, response: Response, user: User = Depends(current_user)):
    async with engine.connect() as conn:
        row = await get_ticket(conn, ticket_id, user)
    response.headers["ETag"] = f'"{row["version"]}"'
    return row


@app.get("/tickets/{ticket_id}/events", tags=["Tickets"])
async def events(ticket_id: UUID, user: User = Depends(current_user)):
    async with engine.connect() as conn:
        await get_ticket(conn, ticket_id, user)
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT id,type,details,created_at FROM ticket_events WHERE ticket_id=:id ORDER BY id"
                    ),
                    {"id": ticket_id},
                )
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


@app.post("/tickets/{ticket_id}/review", tags=["Review"])
async def review(
    ticket_id: UUID,
    data: ReviewInput,
    response: Response,
    user: User = Depends(admin_user),
    if_match: str | None = Header(default=None),
):
    version = expected_version(if_match)
    async with engine.begin() as conn:
        row = await get_ticket(conn, ticket_id, user, lock=True)
        if row["version"] != version:
            raise HTTPException(412, "Ticket has changed; read it again")
        if row["status"] not in {"auto_routed", "needs_review", "reviewed"}:
            raise HTTPException(409, "Classification is still in progress")
        report = await conn.scalar(
            text("SELECT evaluation FROM models WHERE version=:model"),
            {"model": row["model_version"]},
        )
        classes = report["models"][report["selected_model"]]["test"]["per_class"]
        if data.intent not in classes or data.intent in {"accuracy", "macro avg", "weighted avg"}:
            raise HTTPException(422, "Unknown intent")
        await conn.execute(
            text(
                "UPDATE tickets SET status='reviewed',final_intent=:intent,final_department=:department,version=version+1,updated_at=clock_timestamp() WHERE id=:id"
            ),
            {"id": ticket_id, "intent": data.intent, "department": department(data.intent)},
        )
        review_id = await conn.scalar(
            text(
                "INSERT INTO reviews(ticket_id,reviewer_id,ticket_version,intent,note) VALUES (:ticket,:user,:version,:intent,:note) RETURNING id"
            ),
            {
                "ticket": ticket_id,
                "user": user.id,
                "version": version + 1,
                "intent": data.intent,
                "note": clean_text(data.note),
            },
        )
        await event(
            conn,
            ticket_id,
            "reviewed",
            {"review_id": review_id, "intent": data.intent, "reviewer_id": str(user.id)},
        )
        row = await get_ticket(conn, ticket_id, user)
    response.headers["ETag"] = f'"{row["version"]}"'
    return row


@app.get("/feedback/export", tags=["Review"])
async def export_feedback(
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=1000, ge=1, le=1000),
    user: User = Depends(admin_user),
):
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT r.id AS review_id,r.ticket_id,r.intent,r.reviewer_id,r.created_at,r.ticket_version,t.text,t.language,t.model_version,t.predicted_intent FROM reviews r JOIN tickets t ON t.id=r.ticket_id WHERE r.id>:after ORDER BY r.id LIMIT :limit"
                    ),
                    {"after": after, "limit": limit},
                )
            )
            .mappings()
            .all()
        )
    result = [
        {**row, "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()} for row in rows
    ]
    return {
        "rows": result,
        "next_after": result[-1]["review_id"] if result else after,
        "has_more": len(result) == limit,
    }


@app.get("/models", tags=["Models"])
async def models(user: User = Depends(admin_user)):
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT m.version,m.threshold,m.created_at,(c.active_model=m.version) AS active FROM models m CROSS JOIN routing_config c ORDER BY m.created_at"
                    )
                )
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


@app.get("/models/{version}/evaluation", tags=["Models"])
async def evaluation(version: str, user: User = Depends(admin_user)):
    async with engine.connect() as conn:
        report = await conn.scalar(
            text("SELECT evaluation FROM models WHERE version=:version"), {"version": version}
        )
    if not report:
        raise HTTPException(404, "Model not found")
    return report


@app.post("/models/{version}/activate", tags=["Models"])
async def activate(version: str, user: User = Depends(admin_user)):
    async with engine.begin() as conn:
        if not await conn.scalar(
            text("SELECT version FROM models WHERE version=:version"), {"version": version}
        ):
            raise HTTPException(404, "Register the model on the server first")
        await conn.execute(
            text("UPDATE routing_config SET active_model=:version WHERE id=1"), {"version": version}
        )
    return {"active_model": version, "applies_to": "new_tickets"}


@app.get("/statistics", tags=["Review"])
async def statistics(user: User = Depends(admin_user)):
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT model_version,status,count(*) AS count FROM tickets GROUP BY model_version,status ORDER BY model_version,status"
                    )
                )
            )
            .mappings()
            .all()
        )
        reviews_count = await conn.scalar(text("SELECT count(*) FROM reviews"))
        latest = (
            (
                await conn.execute(
                    text(
                        "SELECT count(*) AS total,count(*) FILTER(WHERE final_intent<>predicted_intent) AS corrected FROM tickets WHERE status='reviewed'"
                    )
                )
            )
            .mappings()
            .one()
        )
    return {
        "tickets": [dict(row) for row in rows],
        "review_events": reviews_count,
        "reviewed_tickets": latest["total"],
        "corrected_tickets": latest["corrected"],
        "note": "Reviews are a selected sample; correction rate is not overall model accuracy.",
    }
