from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.server]


def test_audit_log_no_code_content():
    log_path = Path(__file__).resolve().parents[2] / "audit.log"
    if not log_path.exists():
        pytest.skip("audit.log not found")

    lines = log_path.read_text().splitlines()
    suspicious = []
    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        entry = parts[1].strip()
        if any(x in entry for x in ["def ", "class ", "import ", "return "]):
            suspicious.append(line[:120])

    assert not suspicious, f"Potential code content in audit log:\n" + "\n".join(suspicious[:3])
