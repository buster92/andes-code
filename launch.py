#!/usr/bin/env python3
"""
AndesCode Launcher
Run this once to install everything, then every time to start AndesCode.
  python3 launch.py
"""
import os
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent.resolve()

# ── Colors ────────────────────────────────────────────────────────────────────
def _c(code, text): return f"\033[{code}m{text}\033[0m"
ok  = lambda t: print(_c("92", f"  ✓  {t}"))
err = lambda t: print(_c("91", f"  ✗  {t}"))
inf = lambda t: print(_c("94", f"  →  {t}"))
hdr = lambda t: print(_c("1",  f"\n{t}"))

# ── Python version ────────────────────────────────────────────────────────────
hdr("🏔️  AndesCode Launcher")
print()

v = sys.version_info
if v < (3, 10):
    err(f"Python 3.10+ required. You have {v.major}.{v.minor}.")
    err("Download from https://python.org")
    sys.exit(1)
ok(f"Python {v.major}.{v.minor}.{v.micro}")

# ── Install / upgrade pip quietly ─────────────────────────────────────────────
hdr("Checking dependencies...")
inf("Upgrading pip...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
    check=False
)

# ── Core packages needed to run the launcher itself ───────────────────────────
inf("Installing pywebview and certifi...")
subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "pywebview>=5.0", "certifi>=2024.0.0", "--quiet"],
    check=True
)
ok("pywebview + certifi ready")

# ── SSL fix for macOS Python before any network calls ─────────────────────────
import ssl, certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# ── Install requirements.txt ──────────────────────────────────────────────────
req_file = APP_DIR / "requirements.txt"
if req_file.exists():
    inf("Installing requirements.txt...")

    # Detect Metal (Apple Silicon) for llama-cpp-python
    import platform, subprocess as sp
    env = os.environ.copy()
    env["SSL_CERT_FILE"]    = certifi.where()
    env["REQUESTS_CA_BUNDLE"] = certifi.where()

    if platform.system() == "Darwin":
        try:
            arm = sp.check_output(
                ["sysctl", "-n", "hw.optional.arm64"], text=True
            ).strip()
            if arm == "1":
                env["CMAKE_ARGS"] = "-DGGML_METAL=on"
                env["FORCE_CMAKE"] = "1"
                inf("Apple Silicon detected — enabling Metal GPU")
        except Exception:
            pass
    elif platform.system() == "Windows":
        try:
            sp.check_output(["nvidia-smi"], capture_output=True)
            env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
            env["FORCE_CMAKE"] = "1"
            inf("NVIDIA GPU detected — enabling CUDA")
        except Exception:
            pass

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "-r", str(req_file), "--quiet", "--no-warn-script-location"],
        env=env
    )
    if result.returncode != 0:
        err("Dependency install failed.")
        err("If llama-cpp-python fails on macOS, run:")
        err("  xcode-select --install")
        err("Then run this launcher again.")
        sys.exit(1)
    ok("All dependencies installed")
else:
    err(f"requirements.txt not found in {APP_DIR}")
    sys.exit(1)

# ── Launch app ────────────────────────────────────────────────────────────────
hdr("Launching AndesCode...")
print()

app_py = APP_DIR / "app.py"
if not app_py.exists():
    err(f"app.py not found in {APP_DIR}")
    sys.exit(1)

try:
    os.execv(sys.executable, [sys.executable, str(app_py)])
except Exception as e:
    err(f"Failed to launch: {e}")
    sys.exit(1)
