from pathlib import Path

UI = Path("static/index.html").read_text(encoding="utf-8")
SERVER = Path("server.py").read_text(encoding="utf-8")
INDEXER = Path("indexer.py").read_text(encoding="utf-8")


def test_sidebar_has_scrollable_content_region() -> None:
    assert 'class="sidebar-scroll"' in UI
    assert 'overflow-y: auto' in UI
    assert 'min-height: 0' in UI


def test_reindex_button_forces_refresh() -> None:
    assert 'id="reindexBtn"' in UI
    assert 'async function reindexCodebase()' in UI
    assert 'indexCodebase(true)' in UI
    assert 'force_refresh: !!forceRefresh' in UI
    assert 'force_refresh = bool(body.get("force_refresh")' in SERVER
    assert 'Forced reindex requested; rebuilding vectors and graph artifacts' in INDEXER


def test_indexing_state_blocks_restore_and_competing_actions() -> None:
    assert 'let isIndexing = false' in UI
    assert 'setIndexingState(true' in UI
    assert 'clearRestorePrompt()' in UI
    assert '&& !serverIndexing' in UI
    assert 'if (isIndexing) return;' in UI


def test_export_uses_server_written_predictable_path() -> None:
    assert '/v1/conversation/export' in UI
    assert 'Exported to ${data.path}' in UI
    assert '@app.post("/v1/conversation/export")' in SERVER
    assert 'write_conversation_export' in SERVER
