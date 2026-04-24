"""
Golden Android Codebase Fixture
================================
A realistic Android app ("SecureCam") used as ground truth for AndesCode eval.

Architecture: Clean Architecture (data / domain / ui / hardware / di)
Stack:        Kotlin, RxJava 3, Room, Retrofit, Hilt, CameraX
Hardware:     Camera2/CameraX, FusedLocationProvider (GPS), BluetoothLE
Complexity:   RxJava flatMap chains, custom operators, scheduler injection,
              4-level nested dependency graph, hardware lifecycle management.

Used by:
  - test_retrieval_precision.py  (indexes this, checks which files are retrieved)
  - test_answer_eval.py          (asks questions, scores keyword presence in answers)
"""

GOLDEN_FILES: dict[str, str] = {

# ─── BUILD & MANIFEST ─────────────────────────────────────────────────────────

"build.gradle": """\
plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
    id 'com.google.dagger.hilt.android'
    id 'kotlin-kapt'
}

android {
    compileSdk 34
    defaultConfig {
        applicationId "com.andestest.securecam"
        minSdk 26
        targetSdk 34
    }
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_17
        targetCompatibility JavaVersion.VERSION_17
    }
}

dependencies {
    // RxJava 3 + Android bindings
    implementation 'io.reactivex.rxjava3:rxjava:3.1.8'
    implementation 'io.reactivex.rxjava3:rxandroid:3.0.2'
    implementation 'io.reactivex.rxjava3:rxkotlin:3.0.1'

    // Room with RxJava support
    implementation "androidx.room:room-runtime:2.6.1"
    implementation "androidx.room:room-rxjava3:2.6.1"
    kapt "androidx.room:room-compiler:2.6.1"

    // Retrofit + RxJava adapter
    implementation 'com.squareup.retrofit2:retrofit:2.9.0'
    implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
    implementation 'com.squareup.retrofit2:adapter-rxjava3:2.9.0'

    // Hilt dependency injection
    implementation "com.google.dagger:hilt-android:2.50"
    kapt "com.google.dagger:hilt-compiler:2.50"

    // CameraX
    implementation "androidx.camera:camera-core:1.3.1"
    implementation "androidx.camera:camera-camera2:1.3.1"
    implementation "androidx.camera:camera-lifecycle:1.3.1"
    implementation "androidx.camera:camera-video:1.3.1"

    // Location
    implementation 'com.google.android.gms:play-services-location:21.1.0'

    // Bluetooth LE
    implementation 'no.nordicsemi.android:ble:2.7.2'

    // OkHttp logging interceptor
    implementation 'com.squareup.okhttp3:logging-interceptor:4.12.0'
}
""",

"AndroidManifest.xml": """\
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.andestest.securecam">

    <!-- Camera hardware — required, no fallback allowed -->
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-feature android:name="android.hardware.camera" android:required="true" />
    <uses-feature android:name="android.hardware.camera.autofocus" android:required="false" />

    <!-- Location for GPS metadata on recordings -->
    <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
    <uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE_LOCATION" />

    <!-- Bluetooth LE for wireless trigger device -->
    <uses-permission android:name="android.permission.BLUETOOTH_SCAN" android:usesPermissionFlags="neverForLocation" />
    <uses-permission android:name="android.permission.BLUETOOTH_CONNECT" />
    <uses-feature android:name="android.hardware.bluetooth_le" android:required="false" />

    <!-- Storage for video files -->
    <uses-permission android:name="android.permission.READ_MEDIA_VIDEO" />
    <uses-permission android:name="android.permission.INTERNET" />

    <application
        android:name=".SecureCamApp"
        android:label="SecureCam"
        android:allowBackup="false">

        <activity android:name=".ui.MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>

        <!-- Foreground service for background recording -->
        <service
            android:name=".service.RecordingService"
            android:foregroundServiceType="camera|microphone|location" />
    </application>
</manifest>
""",

# ─── DATA LAYER ───────────────────────────────────────────────────────────────

"app/src/main/java/com/andestest/securecam/data/model/Video.kt": """\
package com.andestest.securecam.data.model

import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.ColumnInfo

/**
 * Video entity stored in Room.
 * Represents a recorded video file with GPS metadata captured at recording time.
 * sync_status tracks upload lifecycle: PENDING → UPLOADING → SYNCED | FAILED.
 */
@Entity(tableName = "videos")
data class Video(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    @ColumnInfo(name = "file_path") val filePath: String,
    @ColumnInfo(name = "duration_ms") val durationMs: Long,
    @ColumnInfo(name = "size_bytes") val sizeBytes: Long,
    @ColumnInfo(name = "latitude") val latitude: Double?,
    @ColumnInfo(name = "longitude") val longitude: Double?,
    @ColumnInfo(name = "gps_accuracy_m") val gpsAccuracyMeters: Float?,
    @ColumnInfo(name = "recorded_at") val recordedAt: Long,    // epoch ms
    @ColumnInfo(name = "uploaded_at") val uploadedAt: Long?,
    @ColumnInfo(name = "remote_id") val remoteId: String?,
    @ColumnInfo(name = "sync_status") val syncStatus: SyncStatus = SyncStatus.PENDING,
    @ColumnInfo(name = "upload_retries") val uploadRetries: Int = 0
)

enum class SyncStatus { PENDING, UPLOADING, SYNCED, FAILED }
""",

"app/src/main/java/com/andestest/securecam/data/model/User.kt": """\
package com.andestest.securecam.data.model

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Authenticated user stored locally after login.
 * The authToken is used as a Bearer token in all API requests via AuthInterceptor.
 */
@Entity(tableName = "users")
data class User(
    @PrimaryKey val id: String,
    val email: String,
    val displayName: String,
    val authToken: String,
    val tokenExpiresAt: Long,    // epoch ms
    val organizationId: String
)
""",

"app/src/main/java/com/andestest/securecam/data/local/AppDatabase.kt": """\
package com.andestest.securecam.data.local

import androidx.room.Database
import androidx.room.RoomDatabase
import com.andestest.securecam.data.model.Video
import com.andestest.securecam.data.model.User
import com.andestest.securecam.data.local.dao.VideoDao
import com.andestest.securecam.data.local.dao.UserDao

/**
 * Room database for SecureCam.
 * Version history:
 *   1 → 2: Added upload_retries column to videos (migration provided)
 *   2 → 3: Added gps_accuracy_m column to videos (migration provided)
 *
 * Access via Hilt: @Inject AppDatabase — never instantiate directly.
 */
@Database(
    entities = [Video::class, User::class],
    version = 3,
    exportSchema = true
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun videoDao(): VideoDao
    abstract fun userDao(): UserDao
}
""",

"app/src/main/java/com/andestest/securecam/data/local/dao/VideoDao.kt": """\
package com.andestest.securecam.data.local.dao

import androidx.room.*
import com.andestest.securecam.data.model.Video
import com.andestest.securecam.data.model.SyncStatus
import io.reactivex.rxjava3.core.Completable
import io.reactivex.rxjava3.core.Flowable
import io.reactivex.rxjava3.core.Single

/**
 * Room DAO for video operations.
 *
 * RxJava return types:
 *   - Flowable<List<Video>>  : for UI observation (emits on every DB change)
 *   - Single<Video>          : for one-shot reads
 *   - Completable            : for writes (insert/update/delete)
 *
 * Room generates the RxJava adapter code at compile time via room-rxjava3 artifact.
 * Flowable is backed by SQLite's invalidation tracker — it re-emits whenever the
 * videos table is modified, making it suitable for LiveData-style observation.
 */
@Dao
interface VideoDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insert(video: Video): Single<Long>

    @Query("SELECT * FROM videos ORDER BY recorded_at DESC")
    fun observeAll(): Flowable<List<Video>>

    @Query("SELECT * FROM videos WHERE sync_status = :status ORDER BY recorded_at ASC")
    fun getByStatus(status: SyncStatus): Single<List<Video>>

    @Query("SELECT * FROM videos WHERE id = :id")
    fun getById(id: Long): Single<Video>

    @Query("UPDATE videos SET sync_status = :status, upload_retries = upload_retries + 1 WHERE id = :id")
    fun updateSyncStatus(id: Long, status: SyncStatus): Completable

    @Query("UPDATE videos SET remote_id = :remoteId, uploaded_at = :uploadedAt, sync_status = 'SYNCED' WHERE id = :id")
    fun markSynced(id: Long, remoteId: String, uploadedAt: Long): Completable

    @Query("DELETE FROM videos WHERE id = :id")
    fun delete(id: Long): Completable

    @Query("SELECT COUNT(*) FROM videos WHERE sync_status = 'PENDING' OR sync_status = 'FAILED'")
    fun countPendingUploads(): Flowable<Int>
}
""",

"app/src/main/java/com/andestest/securecam/data/local/dao/UserDao.kt": """\
package com.andestest.securecam.data.local.dao

import androidx.room.*
import com.andestest.securecam.data.model.User
import io.reactivex.rxjava3.core.Completable
import io.reactivex.rxjava3.core.Maybe

@Dao
interface UserDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insertOrReplace(user: User): Completable

    /** Returns Maybe.empty() if no authenticated user exists. */
    @Query("SELECT * FROM users LIMIT 1")
    fun getActiveUser(): Maybe<User>

    @Query("DELETE FROM users")
    fun clearAll(): Completable
}
""",

"app/src/main/java/com/andestest/securecam/data/remote/api/VideoApiService.kt": """\
package com.andestest.securecam.data.remote.api

import com.andestest.securecam.data.remote.dto.UploadResponseDto
import io.reactivex.rxjava3.core.Single
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.http.*

/**
 * Retrofit interface for video upload and management endpoints.
 *
 * All methods return RxJava Single — Retrofit's RxJava3 call adapter
 * (adapter-rxjava3) converts the Call<T> to Single<T> automatically.
 *
 * Authentication is handled by AuthInterceptor (added in NetworkModule),
 * which reads the stored token from UserRepository and injects the
 * Authorization: Bearer <token> header on every request.
 */
interface VideoApiService {

    /**
     * Multipart upload: video file + metadata as separate form parts.
     * lat/lng are optional — omitted when GPS fix was unavailable at record time.
     */
    @Multipart
    @POST("v1/videos/upload")
    fun uploadVideo(
        @Part file: MultipartBody.Part,
        @Part("duration_ms") durationMs: RequestBody,
        @Part("recorded_at") recordedAt: RequestBody,
        @Part("lat") lat: RequestBody?,
        @Part("lng") lng: RequestBody?
    ): Single<UploadResponseDto>

    @GET("v1/videos")
    fun listRemoteVideos(
        @Query("org_id") orgId: String,
        @Query("since") sinceEpochMs: Long
    ): Single<List<UploadResponseDto>>

    @DELETE("v1/videos/{id}")
    fun deleteRemoteVideo(@Path("id") videoId: String): Single<Unit>
}
""",

"app/src/main/java/com/andestest/securecam/data/remote/api/AuthApiService.kt": """\
package com.andestest.securecam.data.remote.api

import com.andestest.securecam.data.remote.dto.LoginRequestDto
import com.andestest.securecam.data.remote.dto.LoginResponseDto
import io.reactivex.rxjava3.core.Single
import retrofit2.http.Body
import retrofit2.http.POST

interface AuthApiService {
    @POST("v1/auth/login")
    fun login(@Body request: LoginRequestDto): Single<LoginResponseDto>

    @POST("v1/auth/refresh")
    fun refreshToken(@Body refreshToken: String): Single<LoginResponseDto>
}
""",

"app/src/main/java/com/andestest/securecam/data/remote/dto/VideoDto.kt": """\
package com.andestest.securecam.data.remote.dto

import com.google.gson.annotations.SerializedName

data class UploadResponseDto(
    @SerializedName("video_id") val videoId: String,
    @SerializedName("url") val url: String,
    @SerializedName("uploaded_at") val uploadedAt: Long
)

data class LoginRequestDto(val email: String, val password: String)

data class LoginResponseDto(
    @SerializedName("user_id") val userId: String,
    val email: String,
    @SerializedName("display_name") val displayName: String,
    @SerializedName("auth_token") val authToken: String,
    @SerializedName("expires_at") val expiresAt: Long,
    @SerializedName("org_id") val orgId: String
)
""",

"app/src/main/java/com/andestest/securecam/data/repository/VideoRepository.kt": """\
package com.andestest.securecam.data.repository

import com.andestest.securecam.data.local.dao.VideoDao
import com.andestest.securecam.data.model.Video
import com.andestest.securecam.data.model.SyncStatus
import com.andestest.securecam.data.remote.api.VideoApiService
import com.andestest.securecam.hardware.GpsTracker
import com.andestest.securecam.rx.operators.RetryWithDelay
import io.reactivex.rxjava3.core.Completable
import io.reactivex.rxjava3.core.Flowable
import io.reactivex.rxjava3.core.Single
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * VideoRepository is the single source of truth for video data.
 *
 * Dependency chain:
 *   VideoRepository
 *     ├── VideoDao          (Room — local persistence)
 *     ├── VideoApiService   (Retrofit — remote API)
 *     └── GpsTracker        (hardware — GPS coordinates at upload time)
 *
 * The repository coordinates between local and remote:
 *   - observeAll() always reads from Room (offline-first)
 *   - uploadVideo() writes to Room first (UPLOADING), calls API, then
 *     updates Room with the remote ID on success or marks FAILED on error
 *   - Network retries are handled via RetryWithDelay operator (3 attempts,
 *     exponential backoff: 2s, 4s, 8s)
 */
@Singleton
class VideoRepository @Inject constructor(
    private val videoDao: VideoDao,
    private val videoApiService: VideoApiService,
    private val gpsTracker: GpsTracker
) {
    fun observeAll(): Flowable<List<Video>> = videoDao.observeAll()

    fun countPendingUploads(): Flowable<Int> = videoDao.countPendingUploads()

    /**
     * Saves a newly recorded video to Room, attaches the current GPS fix,
     * and returns the persisted Video with its generated DB id.
     */
    fun saveRecording(filePath: String, durationMs: Long): Single<Video> {
        val location = gpsTracker.getLastKnownLocation()
        val video = Video(
            filePath = filePath,
            durationMs = durationMs,
            sizeBytes = File(filePath).length(),
            latitude = location?.latitude,
            longitude = location?.longitude,
            gpsAccuracyMeters = location?.accuracy,
            recordedAt = System.currentTimeMillis(),
            uploadedAt = null,
            remoteId = null
        )
        return videoDao.insert(video)
            .flatMap { id -> videoDao.getById(id) }
    }

    /**
     * Uploads a local video to the remote API.
     *
     * Flow:
     * 1. Mark video as UPLOADING in Room
     * 2. Build multipart request from file + metadata
     * 3. POST to VideoApiService.uploadVideo()
     * 4. On success: call markSynced() — sets remote_id + SYNCED status
     * 5. On error: mark FAILED, propagate error upstream
     *
     * Network retries: RetryWithDelay(maxRetries=3, delayMs=2000, backoffMultiplier=2.0)
     * Only retries on IOException and HTTP 5xx; propagates 4xx immediately.
     */
    fun uploadVideo(video: Video): Single<String> {
        val file = File(video.filePath)
        val filePart = MultipartBody.Part.createFormData(
            "file", file.name, file.asRequestBody("video/mp4".toMediaType())
        )
        return videoDao.updateSyncStatus(video.id, SyncStatus.UPLOADING)
            .andThen(
                videoApiService.uploadVideo(
                    file = filePart,
                    durationMs = video.durationMs.toString().toRequestBody(),
                    recordedAt = video.recordedAt.toString().toRequestBody(),
                    lat = video.latitude?.toString()?.toRequestBody(),
                    lng = video.longitude?.toString()?.toRequestBody()
                )
            )
            .retryWhen(RetryWithDelay(maxRetries = 3, initialDelayMs = 2000L, multiplier = 2.0))
            .flatMap { response ->
                videoDao.markSynced(video.id, response.videoId, response.uploadedAt)
                    .andThen(Single.just(response.videoId))
            }
            .onErrorResumeNext { error ->
                videoDao.updateSyncStatus(video.id, SyncStatus.FAILED)
                    .andThen(Single.error(error))
            }
    }

    fun getPendingVideos(): Single<List<Video>> =
        videoDao.getByStatus(SyncStatus.PENDING)

    fun getFailedVideos(): Single<List<Video>> =
        videoDao.getByStatus(SyncStatus.FAILED)

    fun deleteLocal(videoId: Long): Completable = videoDao.delete(videoId)
}
""",

"app/src/main/java/com/andestest/securecam/data/repository/UserRepository.kt": """\
package com.andestest.securecam.data.repository

import com.andestest.securecam.data.local.dao.UserDao
import com.andestest.securecam.data.model.User
import com.andestest.securecam.data.remote.api.AuthApiService
import com.andestest.securecam.data.remote.dto.LoginRequestDto
import io.reactivex.rxjava3.core.Completable
import io.reactivex.rxjava3.core.Maybe
import io.reactivex.rxjava3.core.Single
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Manages authentication state.
 * The stored User.authToken is consumed by AuthInterceptor (in NetworkModule)
 * to inject Authorization headers into every API request.
 */
@Singleton
class UserRepository @Inject constructor(
    private val userDao: UserDao,
    private val authApiService: AuthApiService
) {
    fun login(email: String, password: String): Single<User> =
        authApiService.login(LoginRequestDto(email, password))
            .map { dto ->
                User(
                    id = dto.userId,
                    email = dto.email,
                    displayName = dto.displayName,
                    authToken = dto.authToken,
                    tokenExpiresAt = dto.expiresAt,
                    organizationId = dto.orgId
                )
            }
            .flatMap { user ->
                userDao.insertOrReplace(user).andThen(Single.just(user))
            }

    fun getActiveUser(): Maybe<User> = userDao.getActiveUser()

    fun logout(): Completable = userDao.clearAll()

    fun isTokenExpired(): Single<Boolean> =
        userDao.getActiveUser()
            .map { user -> System.currentTimeMillis() > user.tokenExpiresAt }
            .toSingle(true)   // default: expired if no user
}
""",

# ─── DOMAIN LAYER ─────────────────────────────────────────────────────────────

"app/src/main/java/com/andestest/securecam/domain/usecase/UploadVideoUseCase.kt": """\
package com.andestest.securecam.domain.usecase

import com.andestest.securecam.data.model.Video
import com.andestest.securecam.data.repository.VideoRepository
import com.andestest.securecam.hardware.GpsTracker
import io.reactivex.rxjava3.core.Single
import javax.inject.Inject

/**
 * Orchestrates the full video upload pipeline.
 *
 * Full dependency tree:
 *   UploadVideoUseCase
 *     └── VideoRepository
 *           ├── VideoDao          (Room)
 *           ├── VideoApiService   (Retrofit + OkHttp + AuthInterceptor)
 *           └── GpsTracker        (FusedLocationProviderClient)
 *
 * Calling execute(filePath, durationMs):
 *   1. Calls VideoRepository.saveRecording() → persists locally with GPS snapshot
 *   2. Immediately calls VideoRepository.uploadVideo() with the saved Video
 *   3. Returns the remote video ID on success
 *
 * Schedulers: caller is responsible for subscribeOn/observeOn.
 * CameraViewModel applies schedulerProvider.io() / schedulerProvider.ui().
 */
class UploadVideoUseCase @Inject constructor(
    private val videoRepository: VideoRepository
) {
    data class Result(val videoId: String, val localId: Long)

    fun execute(filePath: String, durationMs: Long): Single<Result> =
        videoRepository.saveRecording(filePath, durationMs)
            .flatMap { video ->
                videoRepository.uploadVideo(video)
                    .map { remoteId -> Result(videoId = remoteId, localId = video.id) }
            }
}
""",

"app/src/main/java/com/andestest/securecam/domain/usecase/SyncVideosUseCase.kt": """\
package com.andestest.securecam.domain.usecase

import com.andestest.securecam.data.repository.VideoRepository
import com.andestest.securecam.rx.operators.RetryWithDelay
import io.reactivex.rxjava3.core.Observable
import io.reactivex.rxjava3.core.Single
import javax.inject.Inject

/**
 * Retries all PENDING and FAILED uploads.
 * Used by the background WorkManager job and the manual "sync now" button.
 *
 * Strategy:
 *   - Fetch PENDING + FAILED videos from Room
 *   - Upload each sequentially (not in parallel — avoids saturating upload bandwidth)
 *   - Collects per-video results; partial success is valid
 *   - Returns SyncResult with counts of succeeded and failed uploads
 *
 * Dependency chain:
 *   SyncVideosUseCase → VideoRepository → VideoDao + VideoApiService + GpsTracker
 *
 * Note: RetryWithDelay is applied inside VideoRepository.uploadVideo(), not here.
 * This use case handles the "which videos to retry" logic; the repository
 * handles the "how to retry each upload" logic.
 */
class SyncVideosUseCase @Inject constructor(
    private val videoRepository: VideoRepository
) {
    data class SyncResult(val succeeded: Int, val failed: Int, val total: Int)

    fun execute(): Single<SyncResult> {
        return Single.zip(
            videoRepository.getPendingVideos(),
            videoRepository.getFailedVideos()
        ) { pending, failed -> pending + failed }
            .flatMap { videos ->
                if (videos.isEmpty()) return@flatMap Single.just(SyncResult(0, 0, 0))

                Observable.fromIterable(videos)
                    .concatMapSingle { video ->
                        videoRepository.uploadVideo(video)
                            .map { SyncOutcome.SUCCESS }
                            .onErrorReturn { SyncOutcome.FAILURE }
                    }
                    .toList()
                    .map { outcomes ->
                        SyncResult(
                            succeeded = outcomes.count { it == SyncOutcome.SUCCESS },
                            failed    = outcomes.count { it == SyncOutcome.FAILURE },
                            total     = outcomes.size
                        )
                    }
            }
    }

    private enum class SyncOutcome { SUCCESS, FAILURE }
}
""",

"app/src/main/java/com/andestest/securecam/domain/usecase/AuthUseCase.kt": """\
package com.andestest.securecam.domain.usecase

import com.andestest.securecam.data.model.User
import com.andestest.securecam.data.repository.UserRepository
import io.reactivex.rxjava3.core.Completable
import io.reactivex.rxjava3.core.Single
import javax.inject.Inject

class AuthUseCase @Inject constructor(private val userRepository: UserRepository) {
    fun login(email: String, password: String): Single<User> =
        userRepository.login(email, password)

    fun logout(): Completable = userRepository.logout()

    fun requireValidSession(): Single<User> =
        userRepository.isTokenExpired()
            .flatMap { expired ->
                if (expired) Single.error(SessionExpiredException())
                else userRepository.getActiveUser().toSingle()
            }
}

class SessionExpiredException : Exception("Auth token expired — please log in again")
""",

# ─── RX LAYER ─────────────────────────────────────────────────────────────────

"app/src/main/java/com/andestest/securecam/rx/schedulers/SchedulerProvider.kt": """\
package com.andestest.securecam.rx.schedulers

import io.reactivex.rxjava3.android.schedulers.AndroidSchedulers
import io.reactivex.rxjava3.core.Scheduler
import io.reactivex.rxjava3.schedulers.Schedulers
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Abstracts RxJava schedulers for testability.
 *
 * Why this matters on Android:
 *   - AndroidSchedulers.mainThread() posts to the Android main looper.
 *     Using it directly in ViewModels makes unit tests require a Robolectric
 *     runtime. SchedulerProvider lets tests inject TrampolineSchedulerProvider
 *     (all work on the current thread, synchronous) instead.
 *   - Schedulers.io() is a shared, cached thread pool suitable for I/O work
 *     (network, disk). Never do I/O on Schedulers.computation() — that pool is
 *     sized to CPU cores and is meant for CPU-bound work only.
 *   - Schedulers.computation() is used for CPU-bound operators (e.g., map over
 *     large collections), not for database or network calls.
 *
 * All ViewModels and UseCases receive SchedulerProvider via Hilt injection.
 */
@Singleton
class SchedulerProvider @Inject constructor() {
    fun io(): Scheduler = Schedulers.io()
    fun ui(): Scheduler = AndroidSchedulers.mainThread()
    fun computation(): Scheduler = Schedulers.computation()
}

/** Test double — all work executes synchronously on the calling thread. */
class TrampolineSchedulerProvider : SchedulerProvider() {
    override fun io(): Scheduler = Schedulers.trampoline()
    override fun ui(): Scheduler = Schedulers.trampoline()
    override fun computation(): Scheduler = Schedulers.trampoline()
}
""",

"app/src/main/java/com/andestest/securecam/rx/operators/RetryWithDelay.kt": """\
package com.andestest.securecam.rx.operators

import io.reactivex.rxjava3.core.Observable
import io.reactivex.rxjava3.functions.Function
import java.io.IOException
import java.util.concurrent.TimeUnit
import retrofit2.HttpException

/**
 * Custom RxJava retryWhen operator with exponential backoff.
 *
 * Usage:
 *   observable.retryWhen(RetryWithDelay(maxRetries = 3, initialDelayMs = 2000L, multiplier = 2.0))
 *
 * Behaviour:
 *   - Retries on IOException (network errors) and HTTP 5xx (server errors)
 *   - Propagates immediately on HTTP 4xx (client errors — retrying won't help)
 *   - Delays between attempts: initialDelayMs * (multiplier ^ attemptNumber)
 *     e.g. 2s → 4s → 8s for initialDelayMs=2000, multiplier=2.0
 *   - After maxRetries exhausted, re-throws the original error
 *
 * The operator is a Function<Observable<Throwable>, Observable<*>> as required
 * by RxJava's retryWhen contract. Each emission from the input Observable
 * triggers a retry; an error or completion terminates the retry loop.
 *
 * Thread safety: stateless — safe to share across multiple subscriptions.
 */
class RetryWithDelay(
    private val maxRetries: Int = 3,
    private val initialDelayMs: Long = 1000L,
    private val multiplier: Double = 2.0
) : Function<Observable<Throwable>, Observable<*>> {

    override fun apply(errors: Observable<Throwable>): Observable<*> {
        var attempt = 0
        return errors.flatMap { error ->
            if (!isRetryable(error)) return@flatMap Observable.error<Any>(error)

            attempt++
            if (attempt > maxRetries) return@flatMap Observable.error<Any>(error)

            val delay = (initialDelayMs * Math.pow(multiplier, (attempt - 1).toDouble())).toLong()
            Observable.timer(delay, TimeUnit.MILLISECONDS)
        }
    }

    private fun isRetryable(error: Throwable): Boolean = when (error) {
        is IOException -> true
        is HttpException -> error.code() >= 500
        else -> false
    }
}
""",

"app/src/main/java/com/andestest/securecam/rx/transformers/IOTransformer.kt": """\
package com.andestest.securecam.rx.transformers

import com.andestest.securecam.rx.schedulers.SchedulerProvider
import io.reactivex.rxjava3.core.ObservableTransformer
import io.reactivex.rxjava3.core.SingleTransformer
import javax.inject.Inject

/**
 * Reusable RxJava transformers that apply standard scheduler assignments.
 *
 * compose() vs subscribeOn/observeOn directly:
 *   Using compose() with a transformer keeps the scheduler logic in one place
 *   and makes the call-site cleaner. It is equivalent to calling
 *   .subscribeOn(io).observeOn(ui) inline but avoids repeating that pattern.
 *
 * Example usage in a ViewModel:
 *   videoRepository.observeAll()
 *       .compose(ioTransformer.forFlowable())
 *       .subscribe(...)
 */
class IOTransformer @Inject constructor(private val schedulers: SchedulerProvider) {

    fun <T : Any> forSingle(): SingleTransformer<T, T> = SingleTransformer { upstream ->
        upstream
            .subscribeOn(schedulers.io())
            .observeOn(schedulers.ui())
    }

    fun <T : Any> forObservable(): ObservableTransformer<T, T> = ObservableTransformer { upstream ->
        upstream
            .subscribeOn(schedulers.io())
            .observeOn(schedulers.ui())
    }
}
""",

# ─── HARDWARE LAYER ───────────────────────────────────────────────────────────

"app/src/main/java/com/andestest/securecam/hardware/CameraManager.kt": """\
package com.andestest.securecam.hardware

import android.content.Context
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import dagger.hilt.android.qualifiers.ApplicationContext
import io.reactivex.rxjava3.core.Observable
import io.reactivex.rxjava3.subjects.PublishSubject
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Wraps CameraX for video recording.
 *
 * Hardware constraints:
 *   - CameraX requires a LifecycleOwner — recording is automatically stopped
 *     when the lifecycle enters STOPPED (e.g., app backgrounded)
 *   - Camera2 API is used under the hood by CameraX; direct Camera2 access is
 *     avoided because CameraX handles the complex state machine
 *   - VideoCapture<Recorder> is the CameraX use case for video; it cannot be
 *     combined with ImageCapture on low-end devices (limited pipeline slots)
 *   - Recording requires CAMERA + RECORD_AUDIO permissions at runtime
 *
 * The recordingEvents Observable emits RecordingEvent.STARTED, COMPLETED, or
 * ERROR, allowing ViewModels to react without coupling to CameraX directly.
 */
@Singleton
class CameraManager @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private val _recordingEvents = PublishSubject.create<RecordingEvent>()
    val recordingEvents: Observable<RecordingEvent> = _recordingEvents

    private var activeRecording: Recording? = null
    private var videoCapture: VideoCapture<Recorder>? = null

    fun bindToLifecycle(lifecycleOwner: LifecycleOwner, surfaceProvider: Preview.SurfaceProvider) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(surfaceProvider)
            }
            val recorder = Recorder.Builder()
                .setQualitySelector(QualitySelector.from(Quality.HD))
                .build()
            videoCapture = VideoCapture.withOutput(recorder)

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                videoCapture
            )
        }, ContextCompat.getMainExecutor(context))
    }

    fun startRecording(outputFile: File) {
        val fileOutputOptions = FileOutputOptions.Builder(outputFile).build()
        activeRecording = videoCapture?.output
            ?.prepareRecording(context, fileOutputOptions)
            ?.withAudioEnabled()
            ?.start(ContextCompat.getMainExecutor(context)) { event ->
                when (event) {
                    is VideoRecordEvent.Start -> _recordingEvents.onNext(RecordingEvent.STARTED)
                    is VideoRecordEvent.Finalize -> {
                        if (event.hasError()) {
                            _recordingEvents.onNext(RecordingEvent.ERROR(event.error.toString()))
                        } else {
                            _recordingEvents.onNext(RecordingEvent.COMPLETED(outputFile.absolutePath))
                        }
                    }
                    else -> {}
                }
            }
    }

    fun stopRecording() {
        activeRecording?.stop()
        activeRecording = null
    }
}

sealed class RecordingEvent {
    object STARTED : RecordingEvent()
    data class COMPLETED(val filePath: String) : RecordingEvent()
    data class ERROR(val message: String) : RecordingEvent()
}
""",

"app/src/main/java/com/andestest/securecam/hardware/GpsTracker.kt": """\
package com.andestest.securecam.hardware

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import com.google.android.gms.location.*
import dagger.hilt.android.qualifiers.ApplicationContext
import io.reactivex.rxjava3.core.Observable
import io.reactivex.rxjava3.subjects.BehaviorSubject
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Tracks device GPS location using FusedLocationProviderClient.
 *
 * Hardware integration notes:
 *   - FusedLocationProvider fuses GPS, Wi-Fi, and cell tower signals.
 *     GPS alone is used as fallback when Wi-Fi/cell are unavailable.
 *   - getLastKnownLocation() is synchronous and returns the cached fix —
 *     safe to call from any thread, including the RxJava IO scheduler.
 *   - The accuracy field (meters) is stored on Video so recipients can
 *     assess location reliability; a fix > 50m accuracy is flagged in the UI.
 *   - Location updates are only requested while the app is in the foreground
 *     (PRIORITY_HIGH_ACCURACY). The foreground recording service requests
 *     PRIORITY_BALANCED_POWER_ACCURACY to conserve battery during long sessions.
 *   - ACCESS_FINE_LOCATION permission is required for GPS-level accuracy.
 *     Without it, FusedLocationProvider falls back to cell/Wi-Fi only (coarse).
 *
 * The locationUpdates Observable is backed by a BehaviorSubject so new
 * subscribers immediately receive the last known location.
 */
@Singleton
class GpsTracker @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private val fusedClient = LocationServices.getFusedLocationProviderClient(context)
    private val _locationSubject = BehaviorSubject.create<Location>()
    val locationUpdates: Observable<Location> = _locationSubject

    private val locationRequest = LocationRequest.Builder(
        Priority.PRIORITY_HIGH_ACCURACY, 5_000L  // update interval: 5s
    ).setMinUpdateIntervalMillis(2_000L).build()

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            result.lastLocation?.let { _locationSubject.onNext(it) }
        }
    }

    @SuppressLint("MissingPermission")
    fun startTracking() {
        fusedClient.requestLocationUpdates(
            locationRequest, locationCallback, context.mainLooper
        )
    }

    fun stopTracking() {
        fusedClient.removeLocationUpdates(locationCallback)
    }

    /** Returns cached location — null if no fix has been obtained yet. */
    fun getLastKnownLocation(): Location? = _locationSubject.value
}
""",

"app/src/main/java/com/andestest/securecam/hardware/BluetoothController.kt": """\
package com.andestest.securecam.hardware

import android.bluetooth.*
import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import io.reactivex.rxjava3.core.Observable
import io.reactivex.rxjava3.subjects.PublishSubject
import no.nordicsemi.android.ble.BleManager
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Manages Bluetooth LE connection to a wireless trigger device (e.g., a remote shutter).
 *
 * Hardware design:
 *   - Uses Nordic Semiconductor's Android-BLE-Library (BleManager) which wraps
 *     the raw Android BluetoothGatt API with a cleaner state machine.
 *   - The trigger device advertises a custom GATT service (TRIGGER_SERVICE_UUID).
 *     When the characteristic TRIGGER_CHAR_UUID changes, the app interprets
 *     the value as a BluetoothEvent.
 *   - BLE scanning requires BLUETOOTH_SCAN permission (API 31+). On older APIs,
 *     BLUETOOTH_ADMIN + ACCESS_FINE_LOCATION are required instead.
 *   - BLE is optional — the app degrades gracefully if BT is unavailable or
 *     the feature flag is off (android.hardware.bluetooth_le required="false").
 *
 * CameraViewModel subscribes to triggerEvents() to start/stop recording
 * without the user touching the screen — important for mounted/remote use cases.
 */
@Singleton
class BluetoothController @Inject constructor(
    @ApplicationContext private val context: Context
) {
    companion object {
        val TRIGGER_SERVICE_UUID: UUID = UUID.fromString("0000ffe0-0000-1000-8000-00805f9b34fb")
        val TRIGGER_CHAR_UUID:    UUID = UUID.fromString("0000ffe1-0000-1000-8000-00805f9b34fb")
    }

    private val _events = PublishSubject.create<BluetoothEvent>()
    val triggerEvents: Observable<BluetoothEvent> = _events

    private var gatt: BluetoothGatt? = null

    fun connect(deviceAddress: String) {
        val adapter = (context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager).adapter
        val device = adapter.getRemoteDevice(deviceAddress)
        gatt = device.connectGatt(context, false, gattCallback)
    }

    fun disconnect() {
        gatt?.disconnect()
        gatt = null
    }

    private val gattCallback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {
                gatt.discoverServices()
                _events.onNext(BluetoothEvent.CONNECTED)
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                _events.onNext(BluetoothEvent.DISCONNECTED)
            }
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic, value: ByteArray
        ) {
            if (characteristic.uuid == TRIGGER_CHAR_UUID) {
                val event = when (value.firstOrNull()?.toInt()) {
                    0x01 -> BluetoothEvent.RECORD_TRIGGER
                    0x02 -> BluetoothEvent.STOP_TRIGGER
                    else -> BluetoothEvent.UNKNOWN
                }
                _events.onNext(event)
            }
        }
    }
}

enum class BluetoothEvent { CONNECTED, DISCONNECTED, RECORD_TRIGGER, STOP_TRIGGER, UNKNOWN }
""",

# ─── UI LAYER ─────────────────────────────────────────────────────────────────

"app/src/main/java/com/andestest/securecam/ui/viewmodel/CameraViewModel.kt": """\
package com.andestest.securecam.ui.viewmodel

import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.andestest.securecam.domain.usecase.UploadVideoUseCase
import com.andestest.securecam.hardware.BluetoothController
import com.andestest.securecam.hardware.BluetoothEvent
import com.andestest.securecam.rx.schedulers.SchedulerProvider
import dagger.hilt.android.lifecycle.HiltViewModel
import io.reactivex.rxjava3.disposables.CompositeDisposable
import javax.inject.Inject

/**
 * ViewModel for the camera screen.
 *
 * Dependency graph:
 *   CameraViewModel
 *     ├── UploadVideoUseCase          (domain)
 *     │     └── VideoRepository      (data)
 *     │           ├── VideoDao       (Room)
 *     │           ├── VideoApiService (Retrofit)
 *     │           └── GpsTracker    (hardware)
 *     ├── BluetoothController        (hardware)
 *     └── SchedulerProvider          (rx)
 *
 * CompositeDisposable pattern:
 *   All RxJava subscriptions are added to compositeDisposable.
 *   onCleared() calls clear() (not dispose()) so the ViewModel can be
 *   resubscribed if needed — dispose() permanently prevents new subscriptions.
 *
 * Bluetooth integration:
 *   The ViewModel subscribes to BluetoothController.triggerEvents() in init{}.
 *   A RECORD_TRIGGER event starts recording; STOP_TRIGGER stops it.
 *   This allows a paired BLE device to control recording hands-free.
 *
 * Upload state machine: Idle → Loading → Success(videoId) | Error(message)
 */
@HiltViewModel
class CameraViewModel @Inject constructor(
    private val uploadVideoUseCase: UploadVideoUseCase,
    private val bluetoothController: BluetoothController,
    private val schedulerProvider: SchedulerProvider
) : ViewModel() {

    private val _uploadState = MutableLiveData<UploadState>(UploadState.Idle)
    val uploadState: LiveData<UploadState> = _uploadState

    private val _isRecording = MutableLiveData(false)
    val isRecording: LiveData<Boolean> = _isRecording

    private val compositeDisposable = CompositeDisposable()

    init {
        observeBluetoothTrigger()
    }

    private fun observeBluetoothTrigger() {
        bluetoothController.triggerEvents
            .observeOn(schedulerProvider.ui())
            .subscribe(
                { event ->
                    when (event) {
                        BluetoothEvent.RECORD_TRIGGER -> startRecording()
                        BluetoothEvent.STOP_TRIGGER   -> stopRecording()
                        else -> {}
                    }
                },
                { error -> Log.e(TAG, "BT trigger error", error) }
            )
            .also { compositeDisposable.add(it) }
    }

    fun startRecording() { _isRecording.value = true }
    fun stopRecording()  { _isRecording.value = false }

    /**
     * Called by CameraFragment when CameraManager emits RecordingEvent.COMPLETED.
     * Delegates to UploadVideoUseCase which handles GPS tagging + upload.
     */
    fun onRecordingCompleted(filePath: String, durationMs: Long) {
        uploadVideoUseCase.execute(filePath, durationMs)
            .subscribeOn(schedulerProvider.io())
            .observeOn(schedulerProvider.ui())
            .doOnSubscribe { _uploadState.value = UploadState.Loading }
            .subscribe(
                { result -> _uploadState.value = UploadState.Success(result.videoId) },
                { error  -> _uploadState.value = UploadState.Error(error.message ?: "Upload failed") }
            )
            .also { compositeDisposable.add(it) }
    }

    override fun onCleared() {
        compositeDisposable.clear()
        super.onCleared()
    }

    companion object { private const val TAG = "CameraViewModel" }
}

sealed class UploadState {
    object Idle : UploadState()
    object Loading : UploadState()
    data class Success(val videoId: String) : UploadState()
    data class Error(val message: String) : UploadState()
}
""",

"app/src/main/java/com/andestest/securecam/ui/viewmodel/GalleryViewModel.kt": """\
package com.andestest.securecam.ui.viewmodel

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.andestest.securecam.data.model.Video
import com.andestest.securecam.data.repository.VideoRepository
import com.andestest.securecam.domain.usecase.SyncVideosUseCase
import com.andestest.securecam.rx.schedulers.SchedulerProvider
import dagger.hilt.android.lifecycle.HiltViewModel
import io.reactivex.rxjava3.disposables.CompositeDisposable
import javax.inject.Inject

/**
 * ViewModel for the video gallery screen.
 *
 * Observes VideoRepository.observeAll() — a Room Flowable that re-emits
 * whenever the videos table changes. This means the gallery auto-updates when:
 *   - A new recording is saved
 *   - An upload completes (sync_status changes to SYNCED)
 *   - A video is deleted
 *
 * The switchMap pattern is NOT used here because observeAll() is already
 * a continuous stream — we subscribe once and let it push updates.
 */
@HiltViewModel
class GalleryViewModel @Inject constructor(
    private val videoRepository: VideoRepository,
    private val syncVideosUseCase: SyncVideosUseCase,
    private val schedulerProvider: SchedulerProvider
) : ViewModel() {

    private val _videos = MutableLiveData<List<Video>>()
    val videos: LiveData<List<Video>> = _videos

    private val _syncState = MutableLiveData<SyncState>(SyncState.Idle)
    val syncState: LiveData<SyncState> = _syncState

    private val _pendingCount = MutableLiveData(0)
    val pendingCount: LiveData<Int> = _pendingCount

    private val compositeDisposable = CompositeDisposable()

    init {
        observeVideos()
        observePendingCount()
    }

    private fun observeVideos() {
        videoRepository.observeAll()
            .subscribeOn(schedulerProvider.io())
            .observeOn(schedulerProvider.ui())
            .subscribe(
                { videos -> _videos.value = videos },
                { /* DB errors are not recoverable; silently drop */ }
            )
            .also { compositeDisposable.add(it) }
    }

    private fun observePendingCount() {
        videoRepository.countPendingUploads()
            .subscribeOn(schedulerProvider.io())
            .observeOn(schedulerProvider.ui())
            .subscribe({ count -> _pendingCount.value = count }, {})
            .also { compositeDisposable.add(it) }
    }

    fun syncNow() {
        syncVideosUseCase.execute()
            .subscribeOn(schedulerProvider.io())
            .observeOn(schedulerProvider.ui())
            .doOnSubscribe { _syncState.value = SyncState.Syncing }
            .subscribe(
                { result -> _syncState.value = SyncState.Done(result.succeeded, result.failed) },
                { error  -> _syncState.value = SyncState.Error(error.message ?: "Sync failed") }
            )
            .also { compositeDisposable.add(it) }
    }

    override fun onCleared() {
        compositeDisposable.clear()
        super.onCleared()
    }
}

sealed class SyncState {
    object Idle : SyncState()
    object Syncing : SyncState()
    data class Done(val succeeded: Int, val failed: Int) : SyncState()
    data class Error(val message: String) : SyncState()
}
""",

# ─── DI LAYER ─────────────────────────────────────────────────────────────────

"app/src/main/java/com/andestest/securecam/di/DatabaseModule.kt": """\
package com.andestest.securecam.di

import android.content.Context
import androidx.room.Room
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.andestest.securecam.data.local.AppDatabase
import com.andestest.securecam.data.local.dao.VideoDao
import com.andestest.securecam.data.local.dao.UserDao
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {

    /**
     * Room migrations — schema changes must have explicit migrations.
     * Destructive fallback is disabled; missing migration = crash at startup.
     */
    private val MIGRATION_1_2 = object : Migration(1, 2) {
        override fun migrate(db: SupportSQLiteDatabase) {
            db.execSQL("ALTER TABLE videos ADD COLUMN upload_retries INTEGER NOT NULL DEFAULT 0")
        }
    }

    private val MIGRATION_2_3 = object : Migration(2, 3) {
        override fun migrate(db: SupportSQLiteDatabase) {
            db.execSQL("ALTER TABLE videos ADD COLUMN gps_accuracy_m REAL")
        }
    }

    @Provides @Singleton
    fun provideDatabase(@ApplicationContext context: Context): AppDatabase =
        Room.databaseBuilder(context, AppDatabase::class.java, "securecam.db")
            .addMigrations(MIGRATION_1_2, MIGRATION_2_3)
            .build()

    @Provides fun provideVideoDao(db: AppDatabase): VideoDao = db.videoDao()
    @Provides fun provideUserDao(db: AppDatabase): UserDao = db.userDao()
}
""",

"app/src/main/java/com/andestest/securecam/di/NetworkModule.kt": """\
package com.andestest.securecam.di

import com.andestest.securecam.data.remote.api.AuthApiService
import com.andestest.securecam.data.remote.api.VideoApiService
import com.andestest.securecam.data.repository.UserRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import hu.akarnokd.rxjava3.retrofit.RxJava3CallAdapterFactory
import okhttp3.Interceptor
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    private const val BASE_URL = "https://api.securecam.io/"

    /**
     * AuthInterceptor reads the stored user token from UserRepository and injects
     * Authorization: Bearer <token> into every outgoing request.
     * This is why UserRepository is a Singleton and injected here — the interceptor
     * needs access to the token at request time, not at construction time.
     */
    @Provides @Singleton
    fun provideAuthInterceptor(userRepository: UserRepository): Interceptor =
        Interceptor { chain ->
            val token = userRepository.getActiveUser().blockingGet()?.authToken
            val request = chain.request().newBuilder()
                .apply { if (token != null) addHeader("Authorization", "Bearer $token") }
                .build()
            chain.proceed(request)
        }

    @Provides @Singleton
    fun provideOkHttpClient(authInterceptor: Interceptor): OkHttpClient =
        OkHttpClient.Builder()
            .addInterceptor(authInterceptor)
            .addInterceptor(HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.HEADERS
            })
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)   // long timeout for video uploads
            .writeTimeout(120, TimeUnit.SECONDS)
            .build()

    @Provides @Singleton
    fun provideRetrofit(okHttpClient: OkHttpClient): Retrofit =
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .addCallAdapterFactory(RxJava3CallAdapterFactory.create())
            .build()

    @Provides fun provideVideoApiService(retrofit: Retrofit): VideoApiService =
        retrofit.create(VideoApiService::class.java)

    @Provides fun provideAuthApiService(retrofit: Retrofit): AuthApiService =
        retrofit.create(AuthApiService::class.java)
}
""",

"app/src/main/java/com/andestest/securecam/di/AppModule.kt": """\
package com.andestest.securecam.di

import com.andestest.securecam.rx.schedulers.SchedulerProvider
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * App-level Hilt bindings.
 * SingletonComponent = lives for the full application lifetime.
 *
 * SchedulerProvider is provided here (not auto-injected) so tests can
 * replace it with TrampolineSchedulerProvider without subclassing or
 * modifying the ViewModel code.
 */
@Module
@InstallIn(SingletonComponent::class)
object AppModule {
    @Provides @Singleton
    fun provideSchedulerProvider(): SchedulerProvider = SchedulerProvider()
}
""",

}


def write_golden_codebase(target_dir: str) -> list[str]:
    """
    Write all golden codebase files to target_dir.
    Returns list of written file paths.
    """
    import os
    written = []
    for rel_path, content in GOLDEN_FILES.items():
        full_path = os.path.join(target_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        written.append(full_path)
    return written
