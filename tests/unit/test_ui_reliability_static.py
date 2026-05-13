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
    assert 'CACHE.invalidate_repo(repo_fp, include_workspace=True)' in INDEXER


def test_indexing_state_blocks_restore_and_competing_actions() -> None:
    assert 'let isIndexing = false' in UI
    assert 'let localIndexRequestActive = false' in UI
    assert 'setIndexingState(true' in UI
    assert 'clearRestorePrompt()' in UI
    assert '&& !serverIndexing' in UI
    assert 'if (isIndexing) return;' in UI
    assert 'serverReportedIndexing' in UI
    assert 'manual_index_in_progress' in SERVER
    assert 'indexing_in_progress' in SERVER


def test_export_uses_server_written_predictable_path() -> None:
    assert '/v1/conversation/export' in UI
    assert 'Exported to ${data.path}' in UI
    assert '@app.post("/v1/conversation/export")' in SERVER
    assert 'write_conversation_export' in SERVER


def test_history_scroll_is_owned_by_sidebar_scroll_region() -> None:
    history_block = UI.split('.history-list {', 1)[1].split('}', 1)[0]
    assert 'flex: 1' not in history_block
    assert 'overflow-y' not in history_block


def test_message_text_and_debug_json_are_selectable() -> None:
    assert '.debug-json, .debug-json *' in UI
    assert 'user-select: text !important' in UI
    assert '-webkit-user-select: text !important' in UI
