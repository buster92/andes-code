from __future__ import annotations

import os
from enum import Enum


class ExecutionMode(str, Enum):
    LOCAL = "LOCAL"
    REMOTE_INFERENCE = "REMOTE_INFERENCE"


_DEFAULT_MODE = ExecutionMode.LOCAL
_ENV_KEY = "ANDESCODE_EXECUTION_MODE"


def resolve_execution_mode(raw_value: str | None) -> ExecutionMode:
    if not raw_value:
        return _DEFAULT_MODE
    candidate = raw_value.strip().upper()
    if candidate == ExecutionMode.REMOTE_INFERENCE.value:
        return ExecutionMode.REMOTE_INFERENCE
    if candidate == ExecutionMode.LOCAL.value:
        return ExecutionMode.LOCAL
    return _DEFAULT_MODE


def get_execution_mode() -> ExecutionMode:
    return resolve_execution_mode(os.getenv(_ENV_KEY))


def execution_mode_env_key() -> str:
    return _ENV_KEY
