import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import jwt
import pytest
from sqlalchemy import text

from triagedesk.config import settings
from triagedesk.db import engine
from triagedesk.main import app
from triagedesk.ml import digest

assert settings.testing and settings.database_url.endswith("_test"), (
    "Tests require an isolated *_test database"
)


@pytest.fixture(autouse=True)
async def clean():
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE users,models,routing_config,tickets,ticket_events,reviews,worker_heartbeats RESTART IDENTITY CASCADE"
            )
        )
        directory = Path(settings.model_dir) / "banking77-v1"
        manifest = json.loads((directory / "manifest.json").read_text())
        await conn.execute(
            text(
                "INSERT INTO models(version,manifest_sha256,threshold,evaluation) VALUES (:version,:hash,:threshold,CAST(:report AS jsonb))"
            ),
            {
                "version": manifest["version"],
                "hash": digest(directory / "manifest.json"),
                "threshold": manifest["threshold"],
                "report": (directory / "evaluation.json").read_text(),
            },
        )
        await conn.execute(text("INSERT INTO routing_config VALUES (1,'banking77-v1')"))
    yield


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as session:
        yield session


@pytest.fixture
async def identities():
    result = {}
    async with engine.begin() as conn:
        for name in ["admin", "alice", "bob"]:
            uid = uuid4()
            await conn.execute(
                text(
                    "INSERT INTO users(id,email,password_hash,role) VALUES (:id,:email,:hash,:role)"
                ),
                {
                    "id": uid,
                    "email": name + "@example.com",
                    "hash": "unused",
                    "role": "admin" if name == "admin" else "user",
                },
            )
            now = datetime.now(UTC)
            token = jwt.encode(
                {
                    "sub": str(uid),
                    "iat": now,
                    "exp": now + timedelta(hours=1),
                    "iss": "triagedesk",
                    "aud": "triagedesk",
                },
                settings.jwt_secret,
                algorithm="HS256",
            )
            result[name] = {"id": uid, "headers": {"Authorization": f"Bearer {token}"}}
    return result


@pytest.fixture(scope="session", autouse=True)
async def dispose_pool():
    yield
    await engine.dispose()
