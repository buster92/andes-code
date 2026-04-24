"""
Answer Quality Eval Suite
==========================
Tests AndesCode's answer quality against a golden codebase fixture.
Default fixture: SecureCam Android (Kotlin / RxJava 3 / Hilt / Room / BLE).
Override with FIXTURE env var when additional fixtures are registered.

The Android fixture is intentionally the most demanding available — deep
RxJava chains, 4-level dependency graphs, hardware integration, and
framework-specific patterns stress-test retrieval and reasoning in ways
that simpler codebases would not. The eval framework itself is
stack-agnostic; Android is the current gold standard fixture.

Additional fixtures (FastAPI/SQLAlchemy, Tokio/Rust, etc.) can be added
to tests/fixtures/ and registered in eval_runner.py without changing
this file.

REQUIRES MODEL — the server must be running with a golden codebase indexed.

Setup:
    1. Start AndesCode: python3 server.py
    2. Index the golden codebase:
         python3 indexer.py <path/to/fixture>
       Or let eval_runner.py / AUTO_INDEX=1 handle it automatically.

Run:
    python3 tests/eval_runner.py --suite full        # recommended entrypoint
    python3 tests/test_answer_eval.py                # run directly

Scoring:
    Each eval case is scored 0–3:
        3  All required keywords present, no forbidden phrases
        2  ≥60% required keywords, no forbidden phrases
        1  ≥30% required keywords OR a forbidden phrase present
        0  <30% required keywords

    Categories: Code Understanding | Architecture | Nested Dependencies |
                Hardware | Framework | ReactiveX

    A final graded report is printed with per-category and overall scores.
"""

import json
import os
import re
import sys
import shutil
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL   = os.getenv("ANDESCODE_URL", "http://localhost:8080")
TIMEOUT    = 180   # seconds — model can be slow on first query
AUTO_INDEX = os.getenv("AUTO_INDEX", "0") == "1"

REPO_ROOT = Path(__file__).parent.parent.parent  # tests/eval/ → tests/ → repo root
sys.path.insert(0, str(REPO_ROOT))            # finds indexer.py, server.py
sys.path.insert(0, str(Path(__file__).parent)) # finds fixtures/

from fixtures.golden_android import write_golden_codebase

# ── API helpers ───────────────────────────────────────────────────────────────

def _ask(question: str, max_tokens: int = 600) -> str:
    """Ask the running AndesCode server and return the full response text."""
    body = json.dumps({
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    full_text = ""
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        for raw_line in r:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0]["delta"].get("content", "")
                full_text += delta
            except (json.JSONDecodeError, KeyError):
                continue
    return full_text


def _server_up() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/", timeout=5) as r:
            data = json.loads(r.read())
            return data.get("status") == "running"
    except Exception:
        return False


# ── Scoring ───────────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    id: str
    category: str
    question: str
    required_keywords: list[str]           # must appear in answer (case-insensitive)
    forbidden_phrases: list[str] = field(default_factory=list)   # must NOT appear
    notes: str = ""                        # why this matters

@dataclass
class EvalResult:
    case: EvalCase
    answer: str
    score: int        # 0–3
    hit_keywords: list[str]
    miss_keywords: list[str]
    hit_forbidden: list[str]
    elapsed_s: float


def score_answer(case: EvalCase, answer: str) -> EvalResult:
    answer_lower = answer.lower()
    hit = [k for k in case.required_keywords if k.lower() in answer_lower]
    miss = [k for k in case.required_keywords if k.lower() not in answer_lower]
    forbidden_hits = [f for f in case.forbidden_phrases if f.lower() in answer_lower]

    ratio = len(hit) / len(case.required_keywords) if case.required_keywords else 1.0

    if forbidden_hits:
        score = 0 if ratio < 0.3 else 1
    elif ratio >= 1.0:
        score = 3
    elif ratio >= 0.6:
        score = 2
    elif ratio >= 0.3:
        score = 1
    else:
        score = 0

    return EvalResult(
        case=case, answer=answer, score=score,
        hit_keywords=hit, miss_keywords=miss,
        hit_forbidden=forbidden_hits, elapsed_s=0.0
    )


# ── Eval cases ────────────────────────────────────────────────────────────────

EVAL_CASES: list[EvalCase] = [

    # ── CODE UNDERSTANDING ─────────────────────────────────────────────────

    EvalCase(
        id="cu-01",
        category="Code Understanding",
        question="What does the uploadVideo method in VideoRepository do, step by step?",
        required_keywords=[
            "UPLOADING", "updateSyncStatus",
            "multipart", "VideoApiService",
            "retryWhen", "markSynced",
            "FAILED", "onErrorResumeNext"
        ],
        forbidden_phrases=["I don't have", "no context", "unable to find", "not sure"],
        notes="Tests understanding of a multi-step RxJava chain with side effects"
    ),

    EvalCase(
        id="cu-02",
        category="Code Understanding",
        question="What does RetryWithDelay do and how does it decide whether to retry?",
        required_keywords=[
            "IOException", "HttpException", "5xx",
            "exponential", "backoff", "maxRetries",
            "flatMap", "timer", "4xx"
        ],
        forbidden_phrases=["I don't know", "not available"],
        notes="Tests understanding of the custom RxJava operator"
    ),

    EvalCase(
        id="cu-03",
        category="Code Understanding",
        question="Why does CameraViewModel call compositeDisposable.clear() in onCleared() instead of dispose()?",
        required_keywords=[
            "clear", "dispose",
            "resubscribe", "memory leak",
            "onCleared", "lifecycle"
        ],
        forbidden_phrases=["same thing", "equivalent", "doesn't matter"],
        notes="Tests nuanced RxJava lifecycle knowledge documented in CameraViewModel"
    ),

    EvalCase(
        id="cu-04",
        category="Code Understanding",
        question="What is the purpose of the SyncStatus enum and what values does it have?",
        required_keywords=[
            "PENDING", "UPLOADING", "SYNCED", "FAILED",
            "upload", "lifecycle", "status"
        ],
        notes="Tests entity-level code understanding"
    ),

    EvalCase(
        id="cu-05",
        category="Code Understanding",
        question="How does saveRecording() capture GPS data and what happens if no GPS fix is available?",
        required_keywords=[
            "GpsTracker", "getLastKnownLocation",
            "null", "latitude", "longitude",
            "gpsAccuracyMeters"
        ],
        notes="Tests understanding of null safety + hardware integration"
    ),

    # ── ARCHITECTURE ───────────────────────────────────────────────────────

    EvalCase(
        id="ar-01",
        category="Architecture",
        question="Describe the full data flow when a user taps 'upload' — from CameraFragment to the server.",
        required_keywords=[
            "CameraViewModel", "UploadVideoUseCase",
            "VideoRepository", "saveRecording",
            "uploadVideo", "VideoApiService",
            "Room", "GPS", "markSynced"
        ],
        forbidden_phrases=["I don't have enough context"],
        notes="Tests end-to-end architectural understanding across all layers"
    ),

    EvalCase(
        id="ar-02",
        category="Architecture",
        question="What are the architectural layers in this app and how do they communicate?",
        required_keywords=[
            "UI", "domain", "data",
            "ViewModel", "UseCase", "Repository",
            "LiveData", "RxJava", "Single"
        ],
        notes="Tests Clean Architecture understanding"
    ),

    EvalCase(
        id="ar-03",
        category="Architecture",
        question="Why is VideoRepository a Singleton and what problems would occur if it wasn't?",
        required_keywords=[
            "Singleton", "single instance",
            "ChromaDB", "Room", "Hilt",
            "multiple", "state"
        ],
        notes="Tests understanding of singleton scope implications"
    ),

    EvalCase(
        id="ar-04",
        category="Architecture",
        question="How does authentication flow from login to API requests?",
        required_keywords=[
            "AuthApiService", "UserRepository",
            "UserDao", "authToken",
            "AuthInterceptor", "Bearer",
            "NetworkModule", "OkHttp"
        ],
        notes="Tests cross-cutting concern (auth) spanning multiple layers"
    ),

    # ── NESTED DEPENDENCIES ────────────────────────────────────────────────

    EvalCase(
        id="nd-01",
        category="Nested Dependencies",
        question="List every direct and transitive dependency of UploadVideoUseCase.",
        required_keywords=[
            "VideoRepository",
            "VideoDao", "VideoApiService", "GpsTracker",
            "Room", "Retrofit", "FusedLocation"
        ],
        notes="Tests 3-level transitive dependency tracing"
    ),

    EvalCase(
        id="nd-02",
        category="Nested Dependencies",
        question="What components must be initialized before CameraViewModel can upload a video?",
        required_keywords=[
            "Hilt", "VideoRepository",
            "UploadVideoUseCase", "BluetoothController",
            "SchedulerProvider", "GpsTracker"
        ],
        notes="Tests full DI graph initialization understanding"
    ),

    EvalCase(
        id="nd-03",
        category="Nested Dependencies",
        question="Trace how a network auth token ends up in an HTTP request header.",
        required_keywords=[
            "UserRepository", "UserDao",
            "getActiveUser", "authToken",
            "AuthInterceptor", "Authorization",
            "Bearer", "OkHttpClient"
        ],
        notes="Tests cross-layer token flow across 4 components"
    ),

    EvalCase(
        id="nd-04",
        category="Nested Dependencies",
        question="How does GPS data travel from the hardware to the uploaded video on the server?",
        required_keywords=[
            "GpsTracker", "getLastKnownLocation",
            "VideoRepository", "saveRecording",
            "latitude", "longitude",
            "VideoApiService", "uploadVideo"
        ],
        notes="Tests hardware→data→network dependency tracing"
    ),

    # ── HARDWARE ───────────────────────────────────────────────────────────

    EvalCase(
        id="hw-01",
        category="Hardware",
        question="What hardware components does SecureCam use and what is each one responsible for?",
        required_keywords=[
            "CameraX", "CameraManager",
            "FusedLocationProvider", "GpsTracker",
            "Bluetooth", "BluetoothController",
            "trigger", "recording"
        ],
        notes="Tests enumeration of all three hardware layers"
    ),

    EvalCase(
        id="hw-02",
        category="Hardware",
        question="What happens when the Bluetooth trigger device sends a RECORD_TRIGGER signal?",
        required_keywords=[
            "BluetoothController", "triggerEvents",
            "CameraViewModel", "observeBluetoothTrigger",
            "startRecording", "RECORD_TRIGGER",
            "observable", "subscribe"
        ],
        notes="Tests BLE→ViewModel reactive event chain"
    ),

    EvalCase(
        id="hw-03",
        category="Hardware",
        question="What are the hardware limitations and constraints documented in CameraManager?",
        required_keywords=[
            "CameraX", "LifecycleOwner",
            "VideoCapture", "ImageCapture",
            "pipeline", "low-end",
            "CAMERA", "RECORD_AUDIO"
        ],
        notes="Tests reading of hardware constraints docs in comments"
    ),

    EvalCase(
        id="hw-04",
        category="Hardware",
        question="Why does the app need a foreground service for recording, and what types does it declare?",
        required_keywords=[
            "foreground service", "background",
            "camera", "microphone", "location",
            "RecordingService", "AndroidManifest"
        ],
        notes="Tests Android background execution constraint understanding"
    ),

    # ── FRAMEWORK ─────────────────────────────────────────────────────────

    EvalCase(
        id="fw-01",
        category="Framework",
        question="How does Hilt provide dependencies to ViewModels in this app?",
        required_keywords=[
            "HiltViewModel", "@Inject",
            "constructor", "SingletonComponent",
            "AppModule", "DatabaseModule", "NetworkModule"
        ],
        notes="Tests Hilt DI mechanics"
    ),

    EvalCase(
        id="fw-02",
        category="Framework",
        question="How does Room interact with RxJava in the VideoDao — what types are used and why?",
        required_keywords=[
            "Flowable", "Single", "Completable",
            "room-rxjava3", "invalidation tracker",
            "re-emits", "observe", "write"
        ],
        notes="Tests Room+RxJava integration understanding"
    ),

    EvalCase(
        id="fw-03",
        category="Framework",
        question="How does Retrofit convert API calls to RxJava Singles in this project?",
        required_keywords=[
            "RxJava3CallAdapterFactory", "addCallAdapterFactory",
            "Call", "Single", "NetworkModule",
            "adapter-rxjava3"
        ],
        notes="Tests Retrofit+RxJava adapter awareness"
    ),

    EvalCase(
        id="fw-04",
        category="Framework",
        question="What Room database migrations exist and what changes do they apply?",
        required_keywords=[
            "migration", "1", "2", "3",
            "upload_retries", "gps_accuracy_m",
            "ALTER TABLE", "ADD COLUMN"
        ],
        notes="Tests schema evolution knowledge in DatabaseModule"
    ),

    # ── REACTIVEX ──────────────────────────────────────────────────────────

    EvalCase(
        id="rx-01",
        category="ReactiveX",
        question="What RxJava schedulers are used in this app, and why is choosing the right one critical on Android?",
        required_keywords=[
            "io()", "mainThread", "computation()",
            "network", "disk", "UI thread",
            "ANR", "TrampolineSchedulerProvider",
            "SchedulerProvider"
        ],
        notes="Tests scheduler architecture knowledge including Android-specific concerns"
    ),

    EvalCase(
        id="rx-02",
        category="ReactiveX",
        question="Explain how the RetryWithDelay operator works inside a retryWhen call.",
        required_keywords=[
            "retryWhen", "Observable<Throwable>",
            "flatMap", "timer", "attempt",
            "exponential", "isRetryable",
            "IOException", "HttpException"
        ],
        notes="Tests deep RxJava operator internals"
    ),

    EvalCase(
        id="rx-03",
        category="ReactiveX",
        question="Why does SyncVideosUseCase use concatMapSingle instead of flatMap for sequential uploads?",
        required_keywords=[
            "concatMapSingle", "sequential",
            "parallel", "bandwidth", "flatMap",
            "one at a time", "concurrent"
        ],
        notes="Tests RxJava concurrency operator choice understanding"
    ),

    EvalCase(
        id="rx-04",
        category="ReactiveX",
        question="What is the difference between BehaviorSubject and PublishSubject, and which classes use each?",
        required_keywords=[
            "BehaviorSubject", "last value",
            "PublishSubject", "no initial",
            "GpsTracker", "CameraManager", "BluetoothController"
        ],
        notes="Tests Subject type understanding mapped to actual usage"
    ),

    EvalCase(
        id="rx-05",
        category="ReactiveX",
        question="How does the IOTransformer avoid repeating subscribeOn/observeOn boilerplate?",
        required_keywords=[
            "compose()", "transformer",
            "subscribeOn", "observeOn",
            "io()", "ui()", "SingleTransformer", "ObservableTransformer"
        ],
        notes="Tests compose() pattern knowledge"
    ),

    EvalCase(
        id="rx-06",
        category="ReactiveX",
        question="Trace the RxJava chain from GalleryViewModel.syncNow() all the way to a network call.",
        required_keywords=[
            "SyncVideosUseCase", "execute",
            "concatMapSingle", "VideoRepository",
            "uploadVideo", "VideoApiService",
            "retryWhen", "subscribeOn", "io()"
        ],
        notes="Tests full RxJava chain tracing across use case + repository + API"
    ),
]


# ── Test class ────────────────────────────────────────────────────────────────

class TestAnswerQuality(unittest.TestCase):
    """
    Runs all eval cases against the live AndesCode server.
    Each test method corresponds to one eval case.
    """
    tmpdir: str = ""
    results: list[EvalResult] = []

    @classmethod
    def setUpClass(cls):
        if not _server_up():
            raise unittest.SkipTest(
                "AndesCode server not running — start with: python3 server.py"
            )

        if AUTO_INDEX:
            cls.tmpdir = tempfile.mkdtemp(prefix="andescode_eval_")
            write_golden_codebase(cls.tmpdir)
            sys.path.insert(0, str(REPO_ROOT))
            from indexer import index_codebase
            result = index_codebase(cls.tmpdir)
            print(f"\n  Auto-indexed {result['indexed']} files from golden codebase")

        cls.results = []

    @classmethod
    def tearDownClass(cls):
        if cls.tmpdir:
            shutil.rmtree(cls.tmpdir, ignore_errors=True)
        _print_eval_report(cls.results)


def _make_test(case: EvalCase):
    def test_method(self):
        t0 = time.perf_counter()
        try:
            answer = _ask(case.question)
        except Exception as e:
            self.fail(f"Server error for [{case.id}]: {e}")

        elapsed = time.perf_counter() - t0
        result = score_answer(case, answer)
        result.elapsed_s = elapsed

        TestAnswerQuality.results.append(result)

        # Print compact summary
        stars = "★" * result.score + "☆" * (3 - result.score)
        print(f"\n  [{case.id}] {case.category} — {stars} ({result.score}/3) — {elapsed:.1f}s")
        if result.miss_keywords:
            print(f"    ✗ Missing: {result.miss_keywords}")
        if result.hit_forbidden:
            print(f"    ✗ Forbidden: {result.hit_forbidden}")

        # Fail test only on score 0
        self.assertGreater(
            result.score, 0,
            f"\n[{case.id}] Score=0\n"
            f"Question: {case.question}\n"
            f"Missing keywords: {result.miss_keywords}\n"
            f"Forbidden hits: {result.hit_forbidden}\n"
            f"Answer excerpt: {answer[:400]}"
        )

    test_method.__name__ = f"test_{case.id.replace('-', '_')}"
    test_method.__doc__ = f"[{case.category}] {case.question[:80]}"
    return test_method


# Dynamically attach test methods
for _case in EVAL_CASES:
    setattr(TestAnswerQuality, f"test_{_case.id.replace('-', '_')}", _make_test(_case))


# ── Report printer ─────────────────────────────────────────────────────────────

def _print_eval_report(results: list[EvalResult]):
    if not results:
        return

    categories = {}
    for r in results:
        cat = r.case.category
        categories.setdefault(cat, []).append(r)

    max_score = len(results) * 3
    total_score = sum(r.score for r in results)

    print(f"\n{'═'*65}")
    print(f"  ANDESCODE EVAL REPORT — SecureCam Android Golden Codebase")
    print(f"{'═'*65}")

    for cat, cat_results in categories.items():
        cat_total = sum(r.score for r in cat_results)
        cat_max   = len(cat_results) * 3
        pct       = cat_total / cat_max * 100 if cat_max else 0
        bar       = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"\n  {cat:<28} {bar}  {cat_total}/{cat_max}  ({pct:.0f}%)")
        for r in cat_results:
            stars = "★" * r.score + "☆" * (3 - r.score)
            flag  = " ← needs work" if r.score <= 1 else ""
            print(f"    {r.case.id:<8} {stars}  {r.case.question[:50]}...{flag}")

    overall_pct = total_score / max_score * 100 if max_score else 0
    big_bar = "█" * int(overall_pct / 5) + "░" * (20 - int(overall_pct / 5))
    print(f"\n{'─'*65}")
    print(f"  OVERALL SCORE:  {big_bar}  {total_score}/{max_score}  ({overall_pct:.1f}%)")

    grade = (
        "🟢 Production-ready"   if overall_pct >= 85 else
        "🟡 Good — polish edges" if overall_pct >= 70 else
        "🟠 Fair — retrieval work needed" if overall_pct >= 50 else
        "🔴 Needs significant work"
    )
    print(f"  Grade:          {grade}")

    avg_time = sum(r.elapsed_s for r in results) / len(results)
    print(f"  Avg response:   {avg_time:.1f}s")
    print(f"{'═'*65}\n")

    # Failure detail
    failures = [r for r in results if r.score <= 1]
    if failures:
        print(f"  Weak responses ({len(failures)} cases):")
        for r in failures:
            print(f"\n  [{r.case.id}] {r.case.question}")
            print(f"    Missing: {r.miss_keywords}")
            if r.hit_forbidden:
                print(f"    Forbidden: {r.hit_forbidden}")
            print(f"    Answer:  {r.answer[:300]}...")


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🏔️  AndesCode — Answer Quality Eval Suite")
    print(f"    Server: {BASE_URL}")
    print(f"    Cases:  {len(EVAL_CASES)} across 6 categories\n")

    if not _server_up():
        print("❌ Server is not running. Start with: python3 server.py")
        sys.exit(1)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAnswerQuality)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
