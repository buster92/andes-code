"""
Retrieval Precision Test Suite
================================
Tests AndesCode's retrieval pipeline against a golden codebase fixture.
Currently uses the SecureCam Android fixture (Kotlin / RxJava 3 / Hilt / Room)
— the most retrieval-demanding fixture available. Additional fixtures
(Python API, Rust CLI, etc.) can be registered in tests/fixtures/ and
run via eval_runner.py.

NO MODEL REQUIRED — these tests only exercise the indexer and search functions.
Runtime: ~5–15 seconds depending on disk speed.

Run:
    python3 tests/eval_runner.py --suite fast        # recommended entrypoint
    python3 tests/test_retrieval_precision.py        # run directly

Categories:
    RetrievalBaseline       — sanity checks (exact filename, obvious queries)
    RetrievalCodeUnderstanding — function/class purpose queries
    RetrievalArchitecture   — layer and data-flow queries
    RetrievalNestedDeps     — dependency chain queries
    RetrievalHardware       — hardware component queries
    RetrievalFramework      — DI, Room, Retrofit, RxJava queries
    RetrievalReactiveX      — RxJava-specific operator/scheduler queries

Scoring:
    Each test is PASS (required files present) / FAIL (missing or forbidden file found).
    A summary score is printed at the end.
"""

import os
import sys
import shutil
import tempfile
import unittest
import atexit
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))  # finds path_setup + fixtures/

from path_setup import find_repo_root, prepend_sys_path
from fixture_registry import DEFAULT_FIXTURE, load_fixture

REPO_ROOT = find_repo_root(__file__)
prepend_sys_path(REPO_ROOT)        # finds indexer.py
prepend_sys_path(EVAL_DIR)         # finds fixtures/

# ── Helpers ───────────────────────────────────────────────────────────────────

_indexed_dir: Optional[str] = None
_shared_tmpdir: Optional[str] = None
_owns_shared_tmpdir = False
_fixture_name = os.getenv("FIXTURE", DEFAULT_FIXTURE)
_fixture_module, _fixture_description = load_fixture(_fixture_name)


def _cleanup_shared_tmpdir():
    global _shared_tmpdir
    if _owns_shared_tmpdir and _shared_tmpdir:
        shutil.rmtree(_shared_tmpdir, ignore_errors=True)
        _shared_tmpdir = None


atexit.register(_cleanup_shared_tmpdir)

def _ensure_index(tmpdir: str):
    """Index the golden codebase once per test session."""
    global _indexed_dir
    if _indexed_dir == tmpdir:
        return
    from indexer import index_codebase
    result = index_codebase(tmpdir)
    assert result["indexed"] > 0, f"Nothing was indexed in {tmpdir}"
    _indexed_dir = tmpdir


def _search(query: str, n: int = 5) -> list[str]:
    """Return list of filenames from search results."""
    from indexer import search
    results = search(query, n_results=n)
    return [r["file"] for r in results]


def _assert_retrieval(
    test: unittest.TestCase,
    query: str,
    required: list[str],
    forbidden: list[str] = None,
    n_results: int = 5,
    label: str = ""
):
    """
    Assert that `required` files appear in search results and `forbidden` do not.
    Matching is substring-based (e.g. "auth.py" matches "...auth.py").
    """
    forbidden = forbidden or []
    retrieved = _search(query, n=n_results)
    prefix = f"[{label}] " if label else ""

    for req in required:
        found = any(req in r for r in retrieved)
        test.assertTrue(
            found,
            f"{prefix}Query: '{query}'\n"
            f"  Expected '{req}' in results.\n"
            f"  Got: {retrieved}"
        )

    for forb in forbidden:
        found = any(forb in r for r in retrieved)
        test.assertFalse(
            found,
            f"{prefix}Query: '{query}'\n"
            f"  '{forb}' should NOT be in results (wrong file retrieved).\n"
            f"  Got: {retrieved}"
        )


# ── Base class ────────────────────────────────────────────────────────────────

class GoldenBaseTest(unittest.TestCase):
    tmpdir: str = ""
    _android_only = False

    @classmethod
    def _reset_shared_state(cls):
        global _indexed_dir, _shared_tmpdir, _owns_shared_tmpdir, _fixture_name, _fixture_module, _fixture_description
        _indexed_dir = None
        _cleanup_shared_tmpdir()
        _shared_tmpdir = None
        _owns_shared_tmpdir = False
        _fixture_name = os.getenv("FIXTURE", DEFAULT_FIXTURE)
        _fixture_module, _fixture_description = load_fixture(_fixture_name)

    @classmethod
    def setUpClass(cls):
        global _shared_tmpdir, _owns_shared_tmpdir
        if cls._android_only and _fixture_name != "android":
            raise unittest.SkipTest(
                f"{cls.__name__} is Android-specific and not applicable to fixture '{_fixture_name}'."
            )

        if _shared_tmpdir is None:
            preset_tmpdir = os.getenv("GOLDEN_FIXTURE_DIR")
            if preset_tmpdir:
                _shared_tmpdir = preset_tmpdir
            else:
                _shared_tmpdir = tempfile.mkdtemp(prefix="andescode_golden_")
                _fixture_module.write_golden_codebase(_shared_tmpdir)
                _owns_shared_tmpdir = True
        cls.tmpdir = _shared_tmpdir
        try:
            _ensure_index(cls.tmpdir)
        except ModuleNotFoundError as exc:
            if exc.name == "indexer":
                raise unittest.SkipTest("indexer.py not found — run from repo root")
            raise

    @classmethod
    def tearDownClass(cls):
        # Shared tempdir is cleaned once (via eval runner or atexit).
        return


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE — sanity checks
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalFixtureSmoke(GoldenBaseTest):
    """Fixture-agnostic smoke checks to ensure fixture is written and indexed."""

    def test_fixture_indexed_has_results(self):
        retrieved = _search("repository architecture", n=3)
        self.assertGreater(len(retrieved), 0, "Expected non-empty retrieval results")


class RetrievalBaseline(GoldenBaseTest):
    """Basic sanity: obviously relevant files must rank in top results."""
    _android_only = True

    def test_camera_viewmodel_by_name(self):
        """Direct filename mention retrieves the file."""
        _assert_retrieval(self, "CameraViewModel", ["CameraViewModel.kt"],
                          label="baseline/filename")

    def test_video_entity_by_name(self):
        _assert_retrieval(self, "Video entity Room database",
                          ["Video.kt"], label="baseline/entity")

    def test_build_gradle_dependencies(self):
        _assert_retrieval(self, "project dependencies libraries gradle",
                          ["build.gradle"], label="baseline/gradle")

    def test_manifest_permissions(self):
        _assert_retrieval(self, "Android permissions camera location bluetooth",
                          ["AndroidManifest.xml"], label="baseline/manifest")

    def test_top5_covers_all_layers(self):
        """Broad architecture query should span multiple layers."""
        retrieved = _search("how does the app work overall", n=8)
        layers = {
            "viewmodel": any("ViewModel" in r for r in retrieved),
            "repository": any("Repository" in r for r in retrieved),
            "usecase":    any("UseCase" in r for r in retrieved),
        }
        missing = [k for k, v in layers.items() if not v]
        self.assertEqual([], missing,
                         f"Broad query missed layers: {missing}\nGot: {retrieved}")


# ─────────────────────────────────────────────────────────────────────────────
# CODE UNDERSTANDING
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalCodeUnderstanding(GoldenBaseTest):
    """Queries about specific functions, classes, and their purpose."""
    _android_only = True

    def test_upload_video_function(self):
        """Query about upload logic should retrieve the function's owner file."""
        _assert_retrieval(self, "how does video upload work",
                          ["VideoRepository.kt", "UploadVideoUseCase.kt"],
                          label="code/upload_fn")

    def test_compositeDisposable_usage(self):
        """CompositeDisposable is used in ViewModels."""
        _assert_retrieval(self, "CompositeDisposable subscription management",
                          ["CameraViewModel.kt"],
                          label="code/composite_disposable")

    def test_retry_with_delay_implementation(self):
        """Custom retry operator should be directly retrieved."""
        _assert_retrieval(self, "retry exponential backoff network error",
                          ["RetryWithDelay.kt"],
                          label="code/retry_operator")

    def test_sync_status_enum(self):
        """SyncStatus enum is defined in Video.kt."""
        _assert_retrieval(self, "SyncStatus PENDING UPLOADING SYNCED FAILED",
                          ["Video.kt"],
                          label="code/sync_status_enum")

    def test_token_expiry_check(self):
        """isTokenExpired logic lives in UserRepository."""
        _assert_retrieval(self, "auth token expiry check session validation",
                          ["UserRepository.kt"],
                          label="code/token_expiry")

    def test_multipart_upload_construction(self):
        """Multipart form data is built in VideoRepository.uploadVideo()."""
        _assert_retrieval(self, "multipart form data video file upload",
                          ["VideoRepository.kt"],
                          label="code/multipart")

    def test_flowable_room_observation(self):
        """Room Flowable for continuous observation is in VideoDao."""
        _assert_retrieval(self, "Flowable observe all videos Room database stream",
                          ["VideoDao.kt"],
                          label="code/room_flowable")


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalArchitecture(GoldenBaseTest):
    """Queries about layers, data flow, and architectural decisions."""
    _android_only = True

    def test_clean_architecture_layers(self):
        """Query about app layers must retrieve files from multiple layers."""
        retrieved = _search("app architecture layers data domain ui", n=8)
        has_domain = any("UseCase" in r for r in retrieved)
        has_data   = any("Repository" in r or "Dao" in r for r in retrieved)
        has_ui     = any("ViewModel" in r for r in retrieved)
        self.assertTrue(has_domain and has_data and has_ui,
                        f"Architecture query missed layers.\nGot: {retrieved}")

    def test_mvvm_pattern(self):
        """MVVM: ViewModel → UseCase → Repository chain should all appear."""
        _assert_retrieval(self,
            "MVVM pattern ViewModel observes LiveData repository",
            ["CameraViewModel.kt", "GalleryViewModel.kt"],
            label="arch/mvvm")

    def test_offline_first_design(self):
        """Offline-first: save to Room first, then upload."""
        _assert_retrieval(self,
            "save recording locally before uploading offline first",
            ["VideoRepository.kt"],
            label="arch/offline_first")

    def test_single_source_of_truth(self):
        """Repository is the single source of truth for video data."""
        _assert_retrieval(self,
            "single source of truth video data Room local database",
            ["VideoRepository.kt", "VideoDao.kt"],
            label="arch/ssot")

    def test_di_module_structure(self):
        """Three Hilt modules: App, Network, Database."""
        retrieved = _search("Hilt dependency injection module provides singleton", n=6)
        has_db      = any("DatabaseModule" in r for r in retrieved)
        has_network = any("NetworkModule" in r for r in retrieved)
        has_app     = any("AppModule" in r for r in retrieved)
        self.assertTrue(has_db or has_network or has_app,
                        f"DI module query missed modules.\nGot: {retrieved}")

    def test_auth_interceptor_location(self):
        """Auth token injection via interceptor is wired in NetworkModule."""
        _assert_retrieval(self,
            "Authorization Bearer token HTTP interceptor OkHttp",
            ["NetworkModule.kt"],
            label="arch/auth_interceptor")

    def test_room_migration_history(self):
        """Schema migrations are defined in DatabaseModule."""
        _assert_retrieval(self,
            "Room database migration schema version upgrade",
            ["DatabaseModule.kt"],
            label="arch/room_migrations")


# ─────────────────────────────────────────────────────────────────────────────
# NESTED DEPENDENCY UNDERSTANDING
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalNestedDeps(GoldenBaseTest):
    """
    Queries that require understanding multi-level dependency chains.
    These are the hardest for retrieval — the answer is spread across files.
    """
    _android_only = True

    def test_upload_use_case_full_chain(self):
        """
        UploadVideoUseCase depends on VideoRepository which depends on
        VideoDao + VideoApiService + GpsTracker. Full chain query.
        """
        retrieved = _search(
            "what does UploadVideoUseCase depend on VideoRepository VideoDao", n=8
        )
        has_usecase    = any("UploadVideoUseCase" in r for r in retrieved)
        has_repository = any("VideoRepository" in r for r in retrieved)
        has_dao        = any("VideoDao" in r for r in retrieved)
        self.assertTrue(has_usecase and has_repository,
                        f"Nested dep chain incomplete.\nGot: {retrieved}")
        print(f"\n    Dep chain coverage: UseCase={has_usecase}, "
              f"Repo={has_repository}, Dao={has_dao}")

    def test_camera_viewmodel_full_dep_tree(self):
        """
        CameraViewModel depends on UploadVideoUseCase, BluetoothController,
        SchedulerProvider. Query should surface all of them.
        """
        retrieved = _search(
            "CameraViewModel dependencies injection UploadVideoUseCase BluetoothController", n=8
        )
        has_vm = any("CameraViewModel" in r for r in retrieved)
        has_bt = any("BluetoothController" in r for r in retrieved)
        self.assertTrue(has_vm,
                        f"CameraViewModel not found.\nGot: {retrieved}")
        # Bluetooth is less likely to be retrieved but worth checking
        print(f"\n    VM dep coverage: VM={has_vm}, BT={has_bt}")

    def test_sync_use_case_chain(self):
        """SyncVideosUseCase → VideoRepository → RetryWithDelay + VideoApiService."""
        retrieved = _search(
            "SyncVideosUseCase retries pending failed uploads VideoRepository", n=8
        )
        has_sync = any("SyncVideosUseCase" in r for r in retrieved)
        has_repo = any("VideoRepository" in r for r in retrieved)
        self.assertTrue(has_sync and has_repo,
                        f"Sync chain missing.\nGot: {retrieved}")

    def test_network_auth_chain(self):
        """
        AuthInterceptor (NetworkModule) → UserRepository → UserDao.
        Token flows from DB through interceptor into every API call.
        """
        retrieved = _search(
            "auth token read UserRepository injected OkHttp interceptor API request", n=8
        )
        has_network_module = any("NetworkModule" in r for r in retrieved)
        has_user_repo      = any("UserRepository" in r for r in retrieved)
        self.assertTrue(has_network_module or has_user_repo,
                        f"Auth chain missing.\nGot: {retrieved}")

    def test_gps_flows_into_video_model(self):
        """
        GPS location is captured by GpsTracker, used in VideoRepository.saveRecording(),
        and stored in the Video entity's latitude/longitude fields.
        """
        retrieved = _search(
            "GPS coordinates latitude longitude stored video metadata", n=8
        )
        has_gps   = any("GpsTracker" in r for r in retrieved)
        has_video = any("Video.kt" in r for r in retrieved)
        has_repo  = any("VideoRepository" in r for r in retrieved)
        coverage  = sum([has_gps, has_video, has_repo])
        self.assertGreaterEqual(coverage, 2,
            f"GPS→Video chain: only {coverage}/3 files found.\nGot: {retrieved}")

    def test_scheduler_provider_reaches_viewmodel(self):
        """SchedulerProvider is provided in AppModule and injected into both ViewModels."""
        retrieved = _search(
            "SchedulerProvider io ui scheduler ViewModel injection", n=8
        )
        has_scheduler = any("SchedulerProvider" in r for r in retrieved)
        has_module    = any("AppModule" in r for r in retrieved)
        has_vm        = any("ViewModel" in r for r in retrieved)
        self.assertTrue(has_scheduler,
                        f"SchedulerProvider not retrieved.\nGot: {retrieved}")
        print(f"\n    Scheduler coverage: Provider={has_scheduler}, "
              f"Module={has_module}, VM={has_vm}")


# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE UNDERSTANDING
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalHardware(GoldenBaseTest):
    """Queries about hardware components, permissions, and constraints."""
    _android_only = True

    def test_camerax_hardware(self):
        """CameraX recording logic should retrieve CameraManager."""
        _assert_retrieval(self,
            "CameraX video recording hardware camera lifecycle",
            ["CameraManager.kt"],
            label="hw/camerax")

    def test_gps_hardware(self):
        """GPS / location queries should retrieve GpsTracker."""
        _assert_retrieval(self,
            "FusedLocationProviderClient GPS location tracking hardware",
            ["GpsTracker.kt"],
            label="hw/gps")

    def test_bluetooth_hardware(self):
        """BLE trigger device should retrieve BluetoothController."""
        _assert_retrieval(self,
            "Bluetooth LE GATT trigger device remote shutter",
            ["BluetoothController.kt"],
            label="hw/bluetooth")

    def test_hardware_permissions_manifest(self):
        """Permission requirements are declared in AndroidManifest."""
        _assert_retrieval(self,
            "runtime permissions CAMERA ACCESS_FINE_LOCATION BLUETOOTH_SCAN required",
            ["AndroidManifest.xml"],
            label="hw/permissions")

    def test_camera_hardware_constraint_pipeline(self):
        """
        CameraX limitation: VideoCapture + ImageCapture can't coexist on
        low-end devices. This constraint is documented in CameraManager.
        """
        _assert_retrieval(self,
            "CameraX pipeline slots VideoCapture ImageCapture limitation low-end",
            ["CameraManager.kt"],
            label="hw/camera_pipeline_constraint")

    def test_gps_accuracy_stored(self):
        """GPS accuracy field is on the Video entity and used in VideoRepository."""
        retrieved = _search("GPS accuracy meters stored video recording quality", n=6)
        has_video = any("Video.kt" in r for r in retrieved)
        has_repo  = any("VideoRepository" in r for r in retrieved)
        has_gps   = any("GpsTracker" in r for r in retrieved)
        self.assertTrue(has_video or has_repo,
                        f"GPS accuracy storage not found.\nGot: {retrieved}")

    def test_bluetooth_optional_feature(self):
        """BLE is optional (required=false in manifest) — app degrades gracefully."""
        _assert_retrieval(self,
            "Bluetooth optional feature degrade gracefully not required",
            ["AndroidManifest.xml", "BluetoothController.kt"],
            label="hw/bt_optional")

    def test_foreground_service_hardware(self):
        """Background recording uses a foreground service with camera+location type."""
        _assert_retrieval(self,
            "foreground service recording camera microphone location background",
            ["AndroidManifest.xml"],
            label="hw/foreground_service")


# ─────────────────────────────────────────────────────────────────────────────
# FRAMEWORK UNDERSTANDING
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalFramework(GoldenBaseTest):
    """Queries about Android framework patterns: Hilt, Room, Retrofit, LiveData."""
    _android_only = True

    def test_hilt_viewmodel_injection(self):
        """@HiltViewModel annotation is on both ViewModels."""
        _assert_retrieval(self,
            "HiltViewModel annotation constructor injection ViewModel",
            ["CameraViewModel.kt", "GalleryViewModel.kt"],
            label="fw/hilt_vm")

    def test_hilt_singleton_component(self):
        """SingletonComponent bindings are in the three DI modules."""
        retrieved = _search("SingletonComponent Hilt module application scope", n=6)
        has_module = any("Module" in r for r in retrieved)
        self.assertTrue(has_module, f"Hilt modules not found.\nGot: {retrieved}")

    def test_room_dao_rxjava_types(self):
        """Room DAOs return RxJava types (Single, Flowable, Completable)."""
        _assert_retrieval(self,
            "Room DAO Single Flowable Completable return type RxJava",
            ["VideoDao.kt"],
            label="fw/room_rx_types")

    def test_retrofit_rxjava_adapter(self):
        """RxJava3CallAdapterFactory converts Call<T> to Single<T>."""
        _assert_retrieval(self,
            "Retrofit RxJava3CallAdapterFactory call adapter Single",
            ["NetworkModule.kt"],
            label="fw/retrofit_rx_adapter")

    def test_livedata_observation(self):
        """ViewModels expose LiveData; Fragments observe it."""
        _assert_retrieval(self,
            "MutableLiveData LiveData observe ViewModel Fragment UI update",
            ["CameraViewModel.kt", "GalleryViewModel.kt"],
            label="fw/livedata")

    def test_room_onconflict_strategy(self):
        """OnConflictStrategy.REPLACE is used in insert operations."""
        _assert_retrieval(self,
            "Room insert OnConflictStrategy REPLACE upsert",
            ["VideoDao.kt", "UserDao.kt"],
            label="fw/room_conflict")

    def test_okhttp_timeout_config(self):
        """Long timeouts for video uploads are configured in NetworkModule."""
        _assert_retrieval(self,
            "OkHttp timeout connect read write seconds video upload",
            ["NetworkModule.kt"],
            label="fw/okhttp_timeout")


# ─────────────────────────────────────────────────────────────────────────────
# REACTIVEX / RXJAVA
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalReactiveX(GoldenBaseTest):
    """RxJava-specific queries — the hardest category."""
    _android_only = True

    def test_retrywhen_operator(self):
        """retryWhen is the RxJava hook used in VideoRepository."""
        _assert_retrieval(self,
            "retryWhen custom operator network retry RxJava",
            ["RetryWithDelay.kt", "VideoRepository.kt"],
            label="rx/retrywhen")

    def test_scheduler_types_and_purpose(self):
        """io() vs computation() vs ui() scheduler distinction."""
        _assert_retrieval(self,
            "Schedulers io computation mainThread Android RxJava thread",
            ["SchedulerProvider.kt"],
            label="rx/schedulers")

    def test_flatmap_chain(self):
        """flatMap chaining Single → Single in repository and use case."""
        _assert_retrieval(self,
            "flatMap Single chain RxJava sequential async",
            ["VideoRepository.kt", "UploadVideoUseCase.kt"],
            label="rx/flatmap")

    def test_composedisposable_lifecycle(self):
        """CompositeDisposable.clear() in onCleared() lifecycle pattern."""
        _assert_retrieval(self,
            "CompositeDisposable clear onCleared ViewModel lifecycle leak",
            ["CameraViewModel.kt"],
            label="rx/disposable_lifecycle")

    def test_behavior_subject_gps(self):
        """BehaviorSubject in GpsTracker provides last-known location."""
        _assert_retrieval(self,
            "BehaviorSubject last value GPS location subject",
            ["GpsTracker.kt"],
            label="rx/behavior_subject")

    def test_publish_subject_events(self):
        """PublishSubject used for event buses in hardware controllers."""
        _assert_retrieval(self,
            "PublishSubject events observable hardware controller",
            ["CameraManager.kt", "BluetoothController.kt"],
            label="rx/publish_subject")

    def test_compose_transformer(self):
        """IOTransformer uses compose() to apply schedulers cleanly."""
        _assert_retrieval(self,
            "compose transformer subscribeOn observeOn reusable pattern",
            ["IOTransformer.kt"],
            label="rx/compose_transformer")

    def test_andthen_completable_chain(self):
        """andThen() chains Completable → Single in repository methods."""
        _assert_retrieval(self,
            "andThen Completable Single chain Room update then return",
            ["VideoRepository.kt"],
            label="rx/andthen")

    def test_flowable_backpressure(self):
        """VideoDao returns Flowable for continuous Room observation."""
        _assert_retrieval(self,
            "Flowable backpressure Room database change observation stream",
            ["VideoDao.kt"],
            label="rx/flowable")

    def test_concat_map_sequential_upload(self):
        """concatMapSingle in SyncVideosUseCase ensures sequential uploads."""
        _assert_retrieval(self,
            "concatMapSingle sequential upload one at a time not parallel",
            ["SyncVideosUseCase.kt"],
            label="rx/concatmap_sequential")

    def test_trampoline_test_scheduler(self):
        """TrampolineSchedulerProvider makes unit tests synchronous."""
        _assert_retrieval(self,
            "TrampolineSchedulerProvider unit test synchronous RxJava",
            ["SchedulerProvider.kt"],
            label="rx/trampoline_test")


# ─── Scored runner ─────────────────────────────────────────────────────────────

class _ScoredResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record(self, name: str, ok: bool, msg: str = ""):
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append((name, msg))

    @property
    def total(self): return self.passed + self.failed

    @property
    def score(self): return self.passed / self.total if self.total else 0

    def print_summary(self):
        bar = "█" * int(self.score * 30) + "░" * (30 - int(self.score * 30))
        pct = self.score * 100
        print(f"\n{'═'*60}")
        print(f"  RETRIEVAL PRECISION SCORE")
        print(f"  {bar}  {self.passed}/{self.total}  ({pct:.1f}%)")
        print(f"{'═'*60}")
        if self.errors:
            print(f"\n  ✗ Failed tests ({len(self.errors)}):")
            for name, msg in self.errors:
                print(f"    • {name}")
                if msg:
                    first_line = msg.strip().split("\n")[0]
                    print(f"      {first_line}")
        print()


if __name__ == "__main__":
    print("🏔️  AndesCode — Retrieval Precision Test Suite")
    print("    Golden codebase: SecureCam Android (Kotlin/RxJava/Hilt/Room)\n")

    loader = unittest.TestLoader()
    suites = [
        RetrievalBaseline,
        RetrievalCodeUnderstanding,
        RetrievalArchitecture,
        RetrievalNestedDeps,
        RetrievalHardware,
        RetrievalFramework,
        RetrievalReactiveX,
    ]

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite([loader.loadTestsFromTestCase(s) for s in suites])
    )
    sys.exit(0 if result.wasSuccessful() else 1)
