"""Останавливаем очередь и воркер этого проекта, затем проверяем сохранность обращений."""

import json
import os
import subprocess
import time

import smoke

smoke.BASE = "http://localhost:8140"


def compose(*args, delay="0"):
    subprocess.run(
        ["docker", "compose", *args],
        check=True,
        env={**os.environ, "INFERENCE_DELAY_SECONDS": delay},
    )


def main():
    token = smoke.login("demo@example.com")
    try:
        compose("stop", "rabbitmq")
        outage = smoke.submit(token, "I forgot my passcode")
        assert outage["status"] == "pending"
        compose("up", "-d", "--no-deps", "rabbitmq")
        recovered = smoke.wait_ticket(token, outage["id"], 120)
        assert recovered["final_intent"] == "passcode_forgotten"
        compose("up", "-d", "--no-deps", "--force-recreate", "worker", delay="5")
        interrupted = smoke.submit(token, "My card has been stolen")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = smoke.request("/tickets/" + interrupted["id"], token)
            if state["status"] == "processing":
                break
            time.sleep(0.1)
        else:
            raise AssertionError("Worker did not claim the ticket")
        compose("kill", "-s", "SIGKILL", "worker", delay="5")
        compose("up", "-d", "--no-deps", "--force-recreate", "worker")
        recovered = smoke.wait_ticket(token, interrupted["id"])
        assert recovered["status"] == "auto_routed" and recovered["attempts"] == 2
        events = smoke.request("/tickets/" + interrupted["id"] + "/events", token)
        assert sum(row["type"] == "classified" for row in events) == 1
        assert any(row["type"] == "recovered" for row in events)
        print(
            json.dumps(
                {
                    "ok": True,
                    "accepted_during_broker_outage": True,
                    "worker_sigkill_recovered": True,
                    "attempts": recovered["attempts"],
                    "classification_commits": 1,
                    "ticket_id": interrupted["id"],
                }
            )
        )
    finally:
        compose("up", "-d", "--no-deps", "rabbitmq", "worker")


if __name__ == "__main__":
    main()
