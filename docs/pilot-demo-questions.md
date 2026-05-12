# AndesCode Pilot Demo Questions

These enterprise pilot questions are designed to demonstrate AndesCode on hard multi-file codebase understanding rather than single-file lookup. Use them alongside the model-free A/B eval reports: the eval measures whether retrieval found the right files, while these questions help humans judge whether the retrieved context supports a clear answer.

A strong answer should cite the files it used, explain the flow across components, identify likely files to change when applicable, and state uncertainty when retrieved context is incomplete.

## 1. Dependency tracing: upload orchestration

**Question:** In the Android fixture, what does `UploadVideoUseCase` depend on, and which lower-level files actually persist and upload the recording?

**A good answer should include:**
- The use case entry point and its direct repository dependency.
- The repository methods that save metadata locally and upload multipart video data.
- Supporting persistence/API/location files when available, such as DAO, API service, and GPS tracker context.

## 2. Auth/token flow: request authorization

**Question:** How does the app get an auth token from storage into outgoing API requests, and where would you change token refresh or expiry behavior?

**A good answer should include:**
- The token storage/repository code and any expiry-check logic.
- The network module/interceptor code that injects the `Authorization` header.
- The likely files to change for refresh, expiry, or request-header behavior.

## 3. Upload/data flow: offline-first recording sync

**Question:** Trace the flow from camera recording completion to local save, sync status update, and eventual upload.

**A good answer should include:**
- UI/ViewModel or manager entry points that initiate recording completion handling.
- Repository and entity fields involved in local persistence and sync state.
- Upload/sync use cases and retry behavior that move pending videos to the server.

## 4. Hardware/framework integration: camera, GPS, and BLE

**Question:** Which files coordinate CameraX recording, GPS metadata, and Bluetooth trigger events, and how do those hardware concerns reach app logic?

**A good answer should include:**
- Camera manager/framework integration points.
- GPS tracker/location stream and how its values are consumed.
- Bluetooth controller trigger/event flow and the ViewModel/use case code that responds.

## 5. Symbol definition plus usage flow: scheduler abstractions

**Question:** Where is `SchedulerProvider` defined, how is it provided for dependency injection, and where is it used to keep RxJava work off the UI thread?

**A good answer should include:**
- The scheduler abstraction/implementation file.
- The Hilt module/provider binding.
- ViewModels, repositories, transformers, or use cases that apply IO/UI schedulers.

## 6. Likely files to change: task creation in the Python API fixture

**Question:** If task creation should validate an additional business rule and enqueue a background notification, which files are likely involved?

**A good answer should include:**
- FastAPI route code that receives the create request.
- Service/repository/database-session code where validation and persistence happen.
- Worker/Celery task files or queue configuration for background notification behavior.

## 7. Dependency and runtime flow: Rust CLI worker scheduling

**Question:** In the Rust CLI fixture, trace how CLI arguments lead to pipeline execution and then worker scheduling/channel dispatch.

**A good answer should include:**
- CLI parsing/entry point files.
- Core pipeline/source discovery files.
- Worker scheduler and channel/backpressure files, plus the types passed between them.
