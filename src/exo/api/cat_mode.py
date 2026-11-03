"""One bounded, server-owned warm-up job, independent of dashboard lifetime."""

import time
from collections.abc import Awaitable, Callable
from typing import Literal

import anyio
from pydantic import BaseModel, ConfigDict

from exo.shared.types.common import ModelId
from exo.shared.types.worker.instances import InstanceId

Readiness = Literal["ready", "busy", "unavailable"]


class CatModeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: ModelId


class CatModeStatus(BaseModel):
    enabled: bool
    active: bool = False
    state: Literal["idle", "running", "stopped", "finished", "error"] = "idle"
    phase: Literal["waiting", "generating", "stopping"] | None = None
    model_id: ModelId | None = None
    instance_id: InstanceId | None = None
    started_at: float | None = None
    stop_at: float | None = None
    completed_requests: int = 0
    error: str | None = None


class CatModeController:
    def __init__(
        self,
        *,
        enabled: bool,
        generate: Callable[[ModelId, InstanceId, int], Awaitable[None]],
        readiness: Callable[[InstanceId], Readiness],
        spawn: Callable[[Callable[[], Awaitable[None]]], None],
        duration: float = 600,
        pause: float = 3,
    ) -> None:
        self.status = CatModeStatus(enabled=enabled)
        self._generate = generate
        self._readiness = readiness
        self._spawn = spawn
        self._duration = duration
        self._pause = pause
        self._scope: anyio.CancelScope | None = None
        self._stopping = False

    def start(self, model_id: ModelId, instance_id: InstanceId) -> CatModeStatus:
        if not self.status.enabled or self.status.active:
            raise ValueError("Cat mode is disabled or already active")
        now = time.time()
        self._stopping = False
        self.status = CatModeStatus(
            enabled=True,
            active=True,
            state="running",
            phase="waiting",
            model_id=model_id,
            instance_id=instance_id,
            started_at=now,
            stop_at=now + self._duration,
        )
        try:
            self._spawn(self._run)
        except Exception:
            self.status.active = False
            self.status.state = "error"
            self.status.phase = None
            self.status.error = "Cat mode controller is not running"
            raise
        return self.status

    def stop(self) -> CatModeStatus:
        if self.status.active:
            self._stopping = True
            self.status.phase = "stopping"
            if self._scope is not None:
                self._scope.cancel()
        return self.status

    async def _run(self) -> None:
        assert self.status.model_id is not None
        assert self.status.instance_id is not None
        try:
            with anyio.move_on_after(self._duration) as scope:
                self._scope = scope
                while not self._stopping:
                    readiness = self._readiness(self.status.instance_id)
                    if readiness == "unavailable":
                        raise RuntimeError("The selected model instance is unavailable")
                    self.status.phase = "waiting"
                    if readiness == "busy":
                        await anyio.sleep(self._pause)
                        continue
                    self.status.phase = "generating"
                    await self._generate(
                        self.status.model_id,
                        self.status.instance_id,
                        self.status.completed_requests + 1,
                    )
                    self.status.completed_requests += 1
                    self.status.phase = "waiting"
                    await anyio.sleep(self._pause)
            self.status.state = "stopped" if self._stopping else "finished"
        except anyio.get_cancelled_exc_class():
            self.status.state = "stopped"
            raise
        except Exception as exc:
            self.status.state = "error"
            self.status.error = str(exc)[:300]
        finally:
            self._scope = None
            self.status.active = False
            self.status.phase = None
