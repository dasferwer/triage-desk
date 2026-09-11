import asyncio
import json
import logging
import signal
from contextlib import suppress
from uuid import UUID

import aio_pika

from .broker import open_channel
from .config import settings
from .db import engine
from .ml import load_classifier
from .service import claim, complete, heartbeat


def infer(row):
    classifier = load_classifier(settings.model_dir, row["model_version"], row["manifest_sha256"])
    return classifier.predict(row["text"], row["language"])


async def handle(payload):
    ticket_id, generation = UUID(payload["ticket_id"]), UUID(payload["generation"])
    row = await claim(ticket_id, generation)
    if not row:
        return False
    if settings.inference_delay_seconds:
        await asyncio.sleep(settings.inference_delay_seconds)
    result = await asyncio.to_thread(infer, row)
    return await complete(ticket_id, generation, result)


async def main():
    logging.basicConfig(level=logging.INFO)
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    connection = await aio_pika.connect_robust(settings.amqp_url)

    async def consume(message):
        async with message.process(requeue=False):
            try:
                await handle(json.loads(message.body))
            except Exception as error:
                # Текст обращения в лог не попадает. Задание восстановится из базы после lease.
                logging.error("Classification failed: %s", type(error).__name__)

    try:
        channel, queue = await open_channel(connection)
        tag = await queue.consume(consume)
        while not stop.is_set():
            await heartbeat("classifier")
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), 2)
        await queue.cancel(tag)
    finally:
        await connection.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
