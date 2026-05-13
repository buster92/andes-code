# UI Reliability QA — Indexing and Conversation Export

Use this checklist after changes to indexing or session export UI behavior.

## Fresh index
1. Start AndesCode and open `/ui`.
2. Select a project path with the folder picker or enter it manually.
3. Click **Index Project**.
4. Expected: project controls are disabled, only progress/status is shown for indexing, chat is disabled until indexing completes, and the project badge plus **Reindex Project** appear after success.

## Reindex existing project
1. With a project already indexed, click **Reindex Project**.
2. Expected: the current selected path is submitted with `force_refresh: true`, vectors are rebuilt, graph artifacts used by hybrid retrieval are regenerated, and no manual deletion of `index/` is required.

## Indexing progress state
1. While indexing is active, watch the sidebar and status polling.
2. Expected: project path, folder picker, Index, Reindex, Clear, chat send, and previous-project restore actions are unavailable; **Continue Previous Project** must not appear during active indexing.

## Sidebar scroll at small heights
1. Resize the app/window to a short height (for example 600px and 480px tall).
2. Expected: the header and model footer can remain fixed, but the main sidebar content scrolls vertically and lower panels/history remain reachable.

## Export conversation success
1. Send at least one message.
2. Click the export button in the top bar.
3. Expected: the app writes a Markdown file under `~/Documents/AndesCode/exports/` with the conversation title and UTC date in the filename, then shows a success toast containing the full file path.

## Export conversation failure
1. Temporarily make `~/Documents/AndesCode/exports/` unwritable or replace it with a file.
2. Click the export button with a non-empty conversation.
3. Expected: no silent failure; a visible error toast explains that export failed.
