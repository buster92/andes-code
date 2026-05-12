from pathlib import Path

import pytest

from conversation_export import sanitize_filename_part, write_conversation_export


def test_write_conversation_export_uses_title_and_date(tmp_path: Path) -> None:
    path = write_conversation_export(
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
        title="Auth: setup?",
        created_at="2026-05-12T10:11:12Z",
        export_dir=tmp_path,
    )

    assert path.parent == tmp_path
    assert path.name == "Auth- setup_2026-05-12_10-11-12.md"
    text = path.read_text(encoding="utf-8")
    assert "# Auth: setup?" in text
    assert "[USER]\nHello" in text
    assert "[ASSISTANT]\nHi" in text


def test_write_conversation_export_raises_for_empty_messages(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No conversation messages"):
        write_conversation_export([], export_dir=tmp_path)


def test_write_conversation_export_surfaces_filesystem_failure(tmp_path: Path) -> None:
    export_dir_file = tmp_path / "exports"
    export_dir_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_conversation_export(
            [{"role": "user", "content": "Hello"}],
            title="Failure",
            created_at="2026-05-12T10:11:12Z",
            export_dir=export_dir_file,
        )


def test_sanitize_filename_part_has_predictable_fallback() -> None:
    assert sanitize_filename_part("***", fallback="conversation") == "conversation"
