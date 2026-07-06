"""Locust load test scenarios. Implemented in a later phase."""

from locust import HttpUser, task


class InferenceUser(HttpUser):
    @task
    def health(self) -> None:
        self.client.get("/healthz")
