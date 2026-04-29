from pathlib import Path


def _ui_source() -> str:
    return Path("static/index.html").read_text(encoding="utf-8")


def test_previous_project_banner_uses_safe_dom_text_rendering():
    src = _ui_source()
    assert "document.getElementById('indexStatus').innerHTML" not in src
    assert "msg.textContent = `Previous project found:" in src
    assert "btn.textContent = 'Continue Previous Project'" in src


def test_chat_not_unlocked_during_scan_phase():
    src = _ui_source()
    assert "unlockChat(false)" not in src
    assert "Indexing in background — ask away" not in src


def test_chat_unlocks_only_after_done_or_restore():
    src = _ui_source()
    assert "setTimeout(() => unlockChat(true), 300);" in src
    assert "continuePreviousProject" in src
    assert "unlockChat(true);" in src
