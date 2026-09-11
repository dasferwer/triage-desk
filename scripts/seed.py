import asyncio
import json
from pathlib import Path
from uuid import UUID

from sqlalchemy import text

from triagedesk.auth import hasher
from triagedesk.config import settings
from triagedesk.db import engine
from triagedesk.ml import digest, read_manifest


async def seed():
    async with engine.begin() as conn:
        for name, role, suffix in [("admin", "admin", "1"), ("demo", "user", "2")]:
            await conn.execute(
                text(
                    "INSERT INTO users(id,email,password_hash,role) VALUES (:id,:email,:hash,:role) ON CONFLICT(email) DO NOTHING"
                ),
                {
                    "id": UUID("14000000-0000-0000-0000-00000000000" + suffix),
                    "email": name + "@example.com",
                    "hash": hasher.hash("TriageDeskDemo123!"),
                    "role": role,
                },
            )
        for directory in sorted(Path(settings.model_dir).iterdir()):
            if not directory.is_dir() or not (directory / "manifest.json").exists():
                continue
            manifest = read_manifest(directory)
            expected = digest(directory / "manifest.json")
            previous = await conn.scalar(
                text("SELECT manifest_sha256 FROM models WHERE version=:version"),
                {"version": manifest["version"]},
            )
            if previous is not None and previous != expected:
                raise ValueError("Registered model is immutable; create a new version")
            await conn.execute(
                text(
                    "INSERT INTO models(version,manifest_sha256,threshold,evaluation) VALUES (:version,:hash,:threshold,CAST(:report AS jsonb)) ON CONFLICT(version) DO NOTHING"
                ),
                {
                    "version": manifest["version"],
                    "hash": expected,
                    "threshold": manifest["threshold"],
                    "report": (directory / "evaluation.json").read_text(),
                },
            )
        await conn.execute(
            text(
                "INSERT INTO routing_config(id,active_model) VALUES (1,'banking77-v1') ON CONFLICT(id) DO NOTHING"
            )
        )
    await engine.dispose()
    print(json.dumps({"seed": "ready", "users": ["admin@example.com", "demo@example.com"]}))


if __name__ == "__main__":
    asyncio.run(seed())
