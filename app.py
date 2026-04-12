"""
AndesCode — Native App Wrapper
Uses PyWebView to create a native macOS/Windows window.
Handles first-run setup: dependencies, model download, server start.
"""
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR   = Path(__file__).parent.resolve()
DATA_DIR  = Path.home() / "Documents" / "AndesCode"
MODEL_NAME= "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
LOCK_FILE = DATA_DIR / ".running"
LOG_FILE  = DATA_DIR / "app.log"

def _resolve_model_path() -> tuple[Path, Path]:
    """
    Find the best model directory.
    Priority:
      1. models/ next to app.py (developer setup, existing installs)
      2. ~/Documents/AndesCode/models/ (app bundle / standard install)
    Returns (model_dir, model_path).
    """
    local = APP_DIR / "models" / MODEL_NAME
    if local.exists() or (APP_DIR / "models").exists():
        return APP_DIR / "models", local
    return DATA_DIR / "models", DATA_DIR / "models" / MODEL_NAME

MODEL_DIR, MODEL_PATH = _resolve_model_path()

# HuggingFace model source
MODEL_REPO   = "lmstudio-community/gemma-4-26B-A4B-it-GGUF"
MODEL_HF_URL = (
    "https://huggingface.co/lmstudio-community/gemma-4-26B-A4B-it-GGUF"
    "/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)
MODEL_SIZE_GB  = 16.0   # approximate
MIN_DISK_GB    = 20.0
MIN_PYTHON_VER = (3, 10)

PORT = 8080

# ── Logging ───────────────────────────────────────────────────────────────────
DATA_DIR.mkdir(parents=True, exist_ok=True)
import logging
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("andescode.app")


# ── Setup status (shared between threads and JS) ──────────────────────────────
_status = {
    "step":     "init",
    "message":  "Starting...",
    "progress": 0,
    "detail":   "",
    "error":    None,
    "done":     False,
}
_status_lock = threading.Lock()

def set_status(step, message, progress=None, detail="", error=None):
    with _status_lock:
        _status["step"]    = step
        _status["message"] = message
        _status["detail"]  = detail
        _status["error"]   = error
        if progress is not None:
            _status["progress"] = progress
    log.info(f"[{step}] {message} {detail}")


# ── Minimal API — only folder picker (used after setup on main UI) ────────────
class AndesCodeAPI:
    """Exposed to JS via window.pywebview.api — only used post-setup."""

    def pick_folder(self):
        import webview
        result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
        return result[0] if result else None

    def quit(self):
        _cleanup()
        import webview
        webview.windows[0].destroy()


# ── Tiny status HTTP server (avoids pywebview JS bridge for setup) ────────────
# setup.html polls this with plain fetch() — no cross-thread JS calls needed

_setup_http_port = None

def _start_status_server() -> int:
    """
    Serve /status and /retry on a random localhost port.
    Returns the port number.
    """
    import http.server, json as _json

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_): pass  # silence access logs

        def do_GET(self):
            if self.path == "/status":
                with _status_lock:
                    data = dict(_status)
                body = _json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif self.path == "/retry":
                threading.Thread(target=_run_setup, daemon=True).start()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b"ok")

            else:
                self.send_response(404)
                self.end_headers()

    # Pick a free port
    import socketserver
    server = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    server.allow_reuse_address = True
    port = server.server_address[1]

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    log.info(f"Status server on port {port}")
    return port


# ── Single-instance lock ──────────────────────────────────────────────────────
def _acquire_lock() -> bool:
    """Return True if we're the only running instance."""
    try:
        if LOCK_FILE.exists():
            pid = int(LOCK_FILE.read_text().strip())
            # Check if that PID is still alive
            try:
                os.kill(pid, 0)
                return False   # process exists — already running
            except (ProcessLookupError, PermissionError):
                pass           # stale lock — process is gone
        LOCK_FILE.write_text(str(os.getpid()))
        return True
    except Exception:
        return True   # can't write lock — proceed anyway


def _release_lock():
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# ── Environment checks ────────────────────────────────────────────────────────
def _check_python():
    v = sys.version_info
    if v < MIN_PYTHON_VER:
        raise RuntimeError(
            f"Python {MIN_PYTHON_VER[0]}.{MIN_PYTHON_VER[1]}+ required. "
            f"You have {v.major}.{v.minor}. "
            f"Download from python.org"
        )


def _check_disk_space():
    free_gb = shutil.disk_usage(Path.home()).free / 1024**3
    if free_gb < MIN_DISK_GB:
        raise RuntimeError(
            f"Need at least {MIN_DISK_GB:.0f}GB free disk space. "
            f"You have {free_gb:.1f}GB. Free up space and try again."
        )


# ── Hardware detection ───────────────────────────────────────────────────────

def _detect_hardware() -> dict:
    """
    Detect CPU, GPU, RAM and acceleration capability.
    Returns a dict with keys: cpu, gpu, ram_gb, acceleration, compatible, warning
    """
    import platform as _platform
    system = _platform.system()
    result = {
        "system":        system,
        "cpu":           _platform.processor() or _platform.machine(),
        "gpu":           None,
        "ram_gb":        0,
        "acceleration":  "cpu",   # "metal", "cuda", "cpu"
        "compatible":    True,
        "warning":       None,
        "detail":        "",
    }

    # ── RAM ───────────────────────────────────────────────────────────────────
    try:
        import psutil
        result["ram_gb"] = psutil.virtual_memory().total / 1024**3
    except ImportError:
        # psutil not installed yet — estimate from platform
        if system == "Darwin":
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True
                ).strip()
                result["ram_gb"] = int(out) / 1024**3
            except Exception:
                result["ram_gb"] = 0
        elif system == "Windows":
            try:
                out = subprocess.check_output(
                    ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                    text=True
                )
                for line in out.splitlines():
                    line = line.strip()
                    if line.isdigit():
                        result["ram_gb"] = int(line) / 1024**3
                        break
            except Exception:
                result["ram_gb"] = 0

    # ── macOS ─────────────────────────────────────────────────────────────────
    if system == "Darwin":
        # Check Apple Silicon
        try:
            arm = subprocess.check_output(
                ["sysctl", "-n", "hw.optional.arm64"], text=True
            ).strip()
            is_apple_silicon = (arm == "1")
        except Exception:
            is_apple_silicon = False

        if is_apple_silicon:
            # Apple Silicon — get chip name
            try:
                chip = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                ).strip()
            except Exception:
                chip = "Apple Silicon"
            result["cpu"]          = chip
            result["gpu"]          = f"{chip} (Unified Memory)"
            result["acceleration"] = "metal"

            # Check memory — 26B Q4 needs ~16GB
            if result["ram_gb"] < 16:
                result["compatible"] = False
                result["warning"] = (
                    f"Insufficient memory: {result['ram_gb']:.0f}GB unified memory detected. "
                    f"Gemma 4 26B requires at least 16GB. "
                    f"Recommended: M2 Pro 16GB or better."
                )
            elif result["ram_gb"] < 24:
                result["warning"] = (
                    f"Low memory: {result['ram_gb']:.0f}GB unified memory. "
                    f"The model will run but system may be slow while AndesCode is active."
                )
            result["detail"] = (
                f"{chip} · {result['ram_gb']:.0f}GB unified · Metal GPU"
            )

        else:
            # Intel Mac — check for discrete GPU via Metal
            result["cpu"] = _platform.processor()
            try:
                gpu_info = subprocess.check_output(
                    ["system_profiler", "SPDisplaysDataType", "-json"],
                    text=True, timeout=10
                )
                import json as _json
                data = _json.loads(gpu_info)
                gpus = data.get("SPDisplaysDataType", [{}])[0]
                gpu_name = gpus.get("sppci_model", "Unknown GPU")
                result["gpu"] = gpu_name
                # Intel Macs with AMD GPU can use Metal
                if "amd" in gpu_name.lower() or "radeon" in gpu_name.lower():
                    result["acceleration"] = "metal"
                else:
                    result["acceleration"] = "cpu"
            except Exception:
                result["gpu"]          = "Unknown (Intel integrated)"
                result["acceleration"] = "cpu"

            if result["ram_gb"] < 32:
                result["compatible"] = False
                result["warning"] = (
                    f"Intel Mac with {result['ram_gb']:.0f}GB RAM detected. "
                    f"AndesCode requires 32GB RAM on Intel Macs (no unified memory). "
                    f"Apple Silicon M2 Pro or better is strongly recommended."
                )
            elif result["acceleration"] == "cpu":
                result["warning"] = (
                    f"No Metal GPU detected on Intel Mac. "
                    f"CPU-only inference will be very slow (5-10 min per response)."
                )
            result["detail"] = (
                f"Intel · {result['ram_gb']:.0f}GB RAM · "
                f"{result['acceleration'].upper()}"
            )

    # ── Windows ───────────────────────────────────────────────────────────────
    elif system == "Windows":
        # Check NVIDIA CUDA
        cuda_available = False
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=10
            ).strip().splitlines()

            if out:
                gpu_line  = out[0]
                parts     = gpu_line.split(",")
                gpu_name  = parts[0].strip()
                vram_mb   = int(parts[1].strip()) if len(parts) > 1 else 0
                vram_gb   = vram_mb / 1024

                result["gpu"]          = f"{gpu_name} ({vram_gb:.0f}GB VRAM)"
                result["acceleration"] = "cuda"
                cuda_available         = True

                if vram_gb < 8:
                    result["compatible"] = False
                    result["warning"] = (
                        f"GPU VRAM too low: {vram_gb:.0f}GB detected on {gpu_name}. "
                        f"Gemma 4 26B requires at least 10GB VRAM. "
                        f"Try a smaller model or upgrade your GPU."
                    )
                elif vram_gb < 12:
                    result["warning"] = (
                        f"Low VRAM: {vram_gb:.0f}GB on {gpu_name}. "
                        f"Model will partially offload to RAM — expect slower responses."
                    )
                result["detail"] = (
                    f"{gpu_name} · {vram_gb:.0f}GB VRAM · "
                    f"{result['ram_gb']:.0f}GB RAM · CUDA"
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if not cuda_available:
            # Check AMD (ROCm — experimental)
            try:
                subprocess.check_output(
                    ["rocm-smi", "--showid"], timeout=5
                )
                result["gpu"]          = "AMD GPU (ROCm — experimental)"
                result["acceleration"] = "cpu"
                result["warning"] = (
                    "AMD GPU detected. ROCm support in llama-cpp-python is "
                    "experimental on Windows. CPU fallback will be used. "
                    "Responses will be slow. NVIDIA GPU recommended."
                )
            except (FileNotFoundError, Exception):
                result["gpu"] = "No dedicated GPU detected"

            if result["ram_gb"] < 32:
                result["compatible"] = False
                result["warning"] = (
                    f"No NVIDIA GPU detected and only {result['ram_gb']:.0f}GB RAM. "
                    f"CPU-only inference of Gemma 4 26B requires 32GB RAM minimum. "
                    f"AndesCode will not run on this configuration."
                )
            else:
                if not result["warning"]:
                    result["warning"] = (
                        f"No NVIDIA GPU detected. CPU-only mode with "
                        f"{result['ram_gb']:.0f}GB RAM. "
                        f"Expect 3-5 minute response times. "
                        f"An NVIDIA RTX 3080 10GB+ is strongly recommended."
                    )
            result["detail"] = (
                f"CPU only · {result['ram_gb']:.0f}GB RAM · no GPU acceleration"
            )

    # ── Linux ─────────────────────────────────────────────────────────────────
    elif system == "Linux":
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=10
            ).strip().splitlines()
            if out:
                parts    = out[0].split(",")
                gpu_name = parts[0].strip()
                vram_mb  = int(parts[1].strip()) if len(parts) > 1 else 0
                vram_gb  = vram_mb / 1024
                result["gpu"]          = f"{gpu_name} ({vram_gb:.0f}GB)"
                result["acceleration"] = "cuda"
                result["detail"]       = f"{gpu_name} · {result['ram_gb']:.0f}GB RAM · CUDA"
                if vram_gb < 10:
                    result["compatible"] = False
                    result["warning"] = f"GPU VRAM too low: {vram_gb:.0f}GB. Need 10GB+."
        except (FileNotFoundError, Exception):
            if result["ram_gb"] < 32:
                result["compatible"] = False
                result["warning"] = (
                    f"No NVIDIA GPU and only {result['ram_gb']:.0f}GB RAM. "
                    f"Need 32GB RAM for CPU-only inference."
                )
            result["detail"] = f"CPU only · {result['ram_gb']:.0f}GB RAM"

    return result


def _check_hardware() -> dict:
    """Run hardware detection and raise if incompatible."""
    hw = _detect_hardware()
    if not hw["compatible"]:
        raise RuntimeError(hw["warning"])
    return hw


# ── Requirements install ──────────────────────────────────────────────────────
def _install_requirements(acceleration: str = "cpu"):
    req_file = APP_DIR / "requirements.txt"
    if not req_file.exists():
        raise RuntimeError(
            "requirements.txt not found. "
            "Make sure you're running from the AndesCode directory."
        )

    set_status("deps", "Installing dependencies...", 12,
               "This may take a few minutes on first run")

    # For Metal (macOS Apple Silicon), install llama-cpp-python with Metal support
    env = os.environ.copy()
    if acceleration == "metal":
        env["CMAKE_ARGS"]    = "-DGGML_METAL=on"
        env["FORCE_CMAKE"]   = "1"
    elif acceleration == "cuda":
        env["CMAKE_ARGS"]    = "-DGGML_CUDA=on"
        env["FORCE_CMAKE"]   = "1"

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file),
         "--quiet", "--no-warn-script-location"],
        capture_output=True, text=True, env=env
    )

    if result.returncode != 0:
        err = result.stderr.strip()

        # Specific: llama-cpp-python needs build tools
        if "llama" in err.lower() or "cmake" in err.lower():
            raise RuntimeError(
                "Building llama-cpp-python failed. "
                "Install Xcode Command Line Tools: "
                "xcode-select --install"
            )

        # Specific: Metal GPU support
        if "metal" in err.lower():
            raise RuntimeError(
                "Metal GPU support not available. "
                "Requires macOS 12+ with Apple Silicon or AMD GPU."
            )

        raise RuntimeError(
            f"Dependency install failed:\n{err[:400]}"
        )

    set_status("deps", "Dependencies ready", 15)


# ── Model download ────────────────────────────────────────────────────────────
def _check_model_integrity(path: Path) -> bool:
    """Quick sanity check — just verify file size is reasonable (>10GB)."""
    try:
        return path.stat().st_size > 10 * 1024**3
    except Exception:
        return False


def _download_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    partial = MODEL_PATH.with_suffix(".gguf.partial")

    # Check disk space before starting
    _check_disk_space()

    # Resume support — check if partial download exists
    resume_pos = 0
    if partial.exists():
        resume_pos = partial.stat().st_size
        set_status("download", "Resuming download...", 20,
                   f"Resuming from {resume_pos / 1024**3:.1f}GB")

    headers = {}
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"

    # macOS Python ships without system SSL certs.
    # Fix: try certifi first, then run Apple's cert installer, then fallback.
    import ssl as _ssl
    import platform as _platform
    ctx = None

    try:
        import certifi
        ctx = _ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass

    if ctx is None and _platform.system() == "Darwin":
        # Try to run Apple's certificate installer silently
        try:
            import glob
            matches = glob.glob(
                "/Applications/Python*/Install Certificates.command"
            )
            if matches:
                subprocess.run(["bash", matches[0]],
                               capture_output=True, timeout=30)
                ctx = _ssl.create_default_context()
        except Exception:
            pass

    if ctx is None:
        # Final fallback — disable verification for HuggingFace download
        ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode    = _ssl.CERT_NONE

    try:
        req = urllib.request.Request(MODEL_HF_URL, headers={
            **headers,
            "User-Agent": "AndesCode/1.0",
        })
        response = urllib.request.urlopen(req, timeout=30, context=ctx)
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to download server.\n"
            f"Check your internet connection and try again.\n{e}"
        )

    total_size   = int(response.headers.get("Content-Length", 0)) + resume_pos
    downloaded   = resume_pos
    chunk_size   = 1024 * 1024   # 1MB chunks
    t_start      = time.time()
    t_last_update= 0

    try:
        mode = "ab" if resume_pos > 0 else "wb"
        with open(partial, mode) as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - t_last_update > 0.5:   # update every 500ms
                    t_last_update = now
                    elapsed  = now - t_start + 0.001
                    speed    = (downloaded - resume_pos) / elapsed
                    pct      = int(downloaded / total_size * 100) if total_size > 0 else 0
                    # Map download to 20-90% of overall setup progress
                    prog     = 20 + int(pct * 0.7)
                    eta_secs = int((total_size - downloaded) / speed) if speed > 0 else 0
                    eta_str  = _fmt_time(eta_secs)
                    speed_str= f"{speed/1024**2:.1f} MB/s"
                    done_gb  = downloaded / 1024**3
                    total_gb = total_size / 1024**3

                    set_status(
                        "download",
                        f"Downloading model... {pct}%",
                        prog,
                        f"{done_gb:.1f} / {total_gb:.1f} GB  ·  {speed_str}  ·  {eta_str} remaining"
                    )

    except Exception as e:
        # Partial file is preserved — next run will resume
        raise RuntimeError(
            f"Download interrupted: {e}\n"
            f"Progress saved — reopen the app to resume."
        )

    # Rename partial → final
    partial.rename(MODEL_PATH)

    # Integrity check
    if not _check_model_integrity(MODEL_PATH):
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded file appears corrupted (too small). "
            "Delete the models folder and try again."
        )

    set_status("download", "Model downloaded", 90)


def _fmt_time(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds//60}m {seconds%60}s"
    return f"{seconds//3600}h {(seconds%3600)//60}m"


# ── Port check ────────────────────────────────────────────────────────────────
def _find_available_port(start: int = 8080, tries: int = 5) -> int:
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(
        f"Ports {start}-{start+tries-1} are all in use. "
        f"Close other applications and try again."
    )


# ── Server management ─────────────────────────────────────────────────────────
_server_proc = None

def _start_server(port: int):
    global _server_proc

    server_py = APP_DIR / "server.py"
    if not server_py.exists():
        raise RuntimeError(
            "server.py not found. "
            "Make sure you're running from the AndesCode directory."
        )

    env = os.environ.copy()
    env["PORT"]             = str(port)
    env["MODEL_PATH"]       = str(MODEL_PATH)
    env["ANDESCODE_APP_MODE"] = "1"   # tells server.py not to open browser

    _server_proc = subprocess.Popen(
        [sys.executable, str(server_py)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(APP_DIR),
    )

    # Stream server logs to our log file
    def _pipe_logs():
        for line in _server_proc.stdout:
            log.info(f"[server] {line.rstrip()}")

    threading.Thread(target=_pipe_logs, daemon=True).start()


def _wait_for_server(port: int, timeout: int = 60) -> bool:
    """Poll until server responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Check server process hasn't crashed
        if _server_proc and _server_proc.poll() is not None:
            raise RuntimeError(
                "Server process exited unexpectedly. "
                "Check ~/Documents/AndesCode/app.log for details."
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


# ── Full setup sequence ───────────────────────────────────────────────────────
def _run_setup():
    global PORT
    try:
        # Step 1: Python version
        set_status("python", "Checking Python version...", 2)
        _check_python()
        set_status("python", "Python OK", 4)

        # Step 2: Hardware compatibility
        set_status("hardware", "Checking hardware compatibility...", 5)
        hw = _detect_hardware()

        if not hw["compatible"]:
            # Incompatible — raise with full details
            raise RuntimeError(hw["warning"])

        accel = hw["acceleration"].upper()
        detail = hw.get("detail", "")
        if hw.get("warning"):
            # Compatible but with a warning — show it but continue
            set_status("hardware", f"Hardware OK ({accel}) ⚠ warning", 8, hw["warning"])
            time.sleep(3)   # pause so user can read the warning
        else:
            set_status("hardware", f"Hardware OK · {accel}", 8, detail)

        # Step 3: Disk space
        set_status("disk", "Checking disk space...", 9)
        _check_disk_space()
        set_status("disk", "Disk space OK", 10)

        # Step 4: Dependencies (pass acceleration hint for llama-cpp-python)
        set_status("deps", "Checking dependencies...", 11)
        _install_requirements(hw["acceleration"])

        # Step 4: Model
        if MODEL_PATH.exists() and _check_model_integrity(MODEL_PATH):
            set_status("model", "Model found", 90,
                       f"{MODEL_PATH.stat().st_size / 1024**3:.1f}GB")
        elif (MODEL_PATH.with_suffix(".gguf.partial")).exists():
            set_status("download", "Resuming model download...", 20)
            _download_model()
        else:
            set_status("download",
                       f"Downloading model ({MODEL_SIZE_GB:.0f}GB)...", 20,
                       "Only needed once — this will take a while")
            _download_model()

        # Step 5: Find port
        set_status("port", "Finding available port...", 91)
        PORT = _find_available_port(8080)
        set_status("port", f"Using port {PORT}", 92)

        # Step 6: Start server
        set_status("server", "Starting AndesCode server...", 93)
        _start_server(PORT)

        # Step 7: Wait for server to be ready
        set_status("server", "Loading model into memory...", 95,
                   "This takes 20-30 seconds on first start")

        if not _wait_for_server(PORT, timeout=120):
            raise RuntimeError(
                "Server took too long to start. "
                "Check ~/Documents/AndesCode/app.log"
            )

        # Done — tell UI to navigate to the main app
        set_status("done", "AndesCode is ready", 100, "", None)
        with _status_lock:
            _status["done"] = True
            _status["port"] = PORT

    except Exception as e:
        log.exception("Setup failed")
        set_status("error", "Setup failed", None, "", str(e))
        with _status_lock:
            _status["error"] = str(e)


# ── Cleanup on exit ───────────────────────────────────────────────────────────
def _cleanup():
    global _server_proc
    _release_lock()
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
    log.info("AndesCode exited cleanly")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not _acquire_lock():
        print("AndesCode is already running.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), sys.exit(0)))
    signal.signal(signal.SIGINT,  lambda *_: (_cleanup(), sys.exit(0)))

    try:
        import webview
    except ImportError:
        print("pywebview not installed. Run: pip install pywebview")
        sys.exit(1)

    # Start the tiny status HTTP server BEFORE creating the window
    # setup.html will poll this with plain fetch() — zero pywebview bridge usage
    status_port = _start_status_server()

    # Build setup URL with status port injected as query param
    setup_html_path = APP_DIR / "static" / "setup.html"
    if not setup_html_path.exists():
        print(f"setup.html not found at {setup_html_path}")
        sys.exit(1)

    setup_url = f"{setup_html_path.as_uri()}?statusPort={status_port}"

    api    = AndesCodeAPI()
    window = webview.create_window(
        title            = "AndesCode",
        url              = setup_url,
        js_api           = api,
        width            = 1200,
        height           = 800,
        min_size         = (800, 600),
        background_color = "#080b0f",
    )

    def _on_shown():
        # Start setup once window is visible — no JS calls, just background work
        threading.Thread(target=_run_setup, daemon=True).start()

        # Single background thread watches for completion and navigates once
        def _watch():
            while True:
                time.sleep(1)   # check once per second — no UI pressure
                with _status_lock:
                    done  = _status.get("done", False)
                    port  = _status.get("port", PORT)
                    error = _status.get("error")

                if done:
                    log.info(f"Setup done — navigating to http://localhost:{port}/ui")
                    time.sleep(0.8)   # let UI settle at 100% before navigating
                    # Use window.load_url() as the sole navigation mechanism.
                    # Do NOT rely on JS window.location.href from setup.html:
                    # file:// -> http:// cross-origin navigation is silently
                    # dropped by WKWebView, freezing the screen at 100%.
                    for attempt in range(5):
                        try:
                            window.load_url(f"http://localhost:{port}/ui")
                            log.info(f"Navigation triggered (attempt {attempt + 1})")
                            break
                        except Exception as e:
                            log.error(f"Navigation attempt {attempt + 1} failed: {e}")
                            time.sleep(1)
                    return   # stop watching

                if error:
                    return   # error shown in setup.html via status polling

        threading.Thread(target=_watch, daemon=True).start()

    window.events.shown += _on_shown

    try:
        webview.start(debug=False)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
