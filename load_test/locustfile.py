"""Locust load scenarios: mixed priorities with steady or bursty traffic shapes."""

from __future__ import annotations

import os
import random

from locust import HttpUser, LoadTestShape, between, task

INTERACTIVE_RATIO = float(os.environ.get("LOCUST_INTERACTIVE_RATIO", "0.7"))
LOCUST_SHAPE = os.environ.get("LOCUST_SHAPE", "bursty").lower()

SAMPLE_TEXTS = [
    "This product exceeded my expectations.",
    "The service was slow and unhelpful.",
    "Great value for the price point.",
    "I would not recommend this to anyone.",
    "Absolutely love the build quality.",
]


class InferenceUser(HttpUser):
    """Mixed interactive/batch /predict traffic."""

    wait_time = between(0.01, 0.15)

    @task(7)
    def predict_interactive(self) -> None:
        self._predict("interactive")

    @task(3)
    def predict_batch(self) -> None:
        self._predict("batch")

    @task(1)
    def predict_mixed_random(self) -> None:
        priority = "interactive" if random.random() < INTERACTIVE_RATIO else "batch"
        self._predict(priority)

    def _predict(self, priority: str) -> None:
        text = random.choice(SAMPLE_TEXTS) + f" sample={random.randint(0, 1_000_000)}"
        with self.client.post(
            "/predict",
            json={"text": text, "priority": priority},
            catch_response=True,
            name=f"/predict [{priority}]",
        ) as response:
            if response.status_code == 503:
                response.success()
            elif not response.ok:
                response.failure(f"unexpected status {response.status_code}")


class SteadyTrafficShape(LoadTestShape):
    """Flat concurrency for baseline throughput/latency."""

    steady_users = int(os.environ.get("LOCUST_STEADY_USERS", "30"))
    run_time_s = int(os.environ.get("LOCUST_RUN_TIME", "45"))

    def tick(self) -> tuple[int, float] | None:
        run_time = self.get_run_time()
        if run_time > self.run_time_s:
            return None
        return self.steady_users, float(os.environ.get("LOCUST_SPAWN_RATE", "5"))


class BurstyTrafficShape(LoadTestShape):
    """Ramp-up → spike → ramp-down to simulate bursty inference traffic."""

    stages = [
        {"duration": 15, "users": 5, "spawn_rate": 2},
        {"duration": 30, "users": 60, "spawn_rate": 12},
        {"duration": 20, "users": 80, "spawn_rate": 8},
        {"duration": 25, "users": 10, "spawn_rate": 10},
        {"duration": 10, "users": 0, "spawn_rate": 5},
    ]

    def tick(self) -> tuple[int, float] | None:
        run_time = self.get_run_time()
        elapsed = 0.0
        for stage in self.stages:
            elapsed += stage["duration"]
            if run_time < elapsed:
                return stage["users"], stage["spawn_rate"]
        return None


class PreemptionLoadShape(LoadTestShape):
    """Sustained load through a simulated Spot preemption window."""

    users = int(os.environ.get("LOCUST_PREEMPTION_USERS", "40"))
    spawn_rate = float(os.environ.get("LOCUST_SPAWN_RATE", "8"))
    run_time_s = int(os.environ.get("LOCUST_RUN_TIME", "35"))

    def tick(self) -> tuple[int, float] | None:
        if self.get_run_time() > self.run_time_s:
            return None
        return self.users, self.spawn_rate


_SHAPE_BY_NAME = {
    "steady": SteadyTrafficShape,
    "bursty": BurstyTrafficShape,
    "preemption": PreemptionLoadShape,
}


def _selected_shape() -> type[LoadTestShape] | None:
    if os.environ.get("LOCUST_USE_SHAPE", "1") not in ("1", "true", "yes"):
        return None
    return _SHAPE_BY_NAME.get(LOCUST_SHAPE, BurstyTrafficShape)


SelectedTrafficShape = _selected_shape()
