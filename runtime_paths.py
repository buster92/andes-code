from pathlib import Path

APP_DATA_DIR_NAME = "AndesCode"


def get_app_data_dir() -> Path:
    """Return AndesCode runtime data directory outside the repository."""
    return Path.home() / "Documents" / APP_DATA_DIR_NAME


def ensure_app_data_dir() -> Path:
    data_dir = get_app_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_runtime_log_path(name: str) -> Path:
    """Return a deterministic log file path in the app data directory."""
    if not name.endswith(".log"):
        name = f"{name}.log"
    return ensure_app_data_dir() / name
