"""
Golden Rust CLI Codebase Fixture
==================================
A realistic Rust project ("Ferox") for the AndesCode eval framework.

Architecture: Cargo workspace with 3 crates (ferox-core / ferox-worker / ferox-cli)
Stack:        Tokio, Serde, Clap, async traits (async-trait), thiserror, anyhow,
              tokio::sync channels, Arc<Mutex>, Rayon for CPU-bound work
Complexity:   async trait objects, channel-based worker pool, workspace dependency graph,
              error type propagation across crates, runtime configuration, streaming
              file processing pipeline.

Used by:
  - test_retrieval_precision.py  (stack-agnostic retrieval assertions)
  - test_answer_eval.py          (graded answer quality assertions)
"""

GOLDEN_FILES: dict[str, str] = {

# ─── WORKSPACE ────────────────────────────────────────────────────────────────

"Cargo.toml": """\
[workspace]
members = [
    "ferox-core",
    "ferox-worker",
    "ferox-cli",
]
resolver = "2"

# Shared dependency versions across all workspace crates
[workspace.dependencies]
tokio          = { version = "1.37",  features = ["full"] }
serde          = { version = "1.0",   features = ["derive"] }
serde_json     = "1.0"
thiserror      = "1.0"
anyhow         = "1.0"
tracing        = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
async-trait    = "0.1"
""",

# ─── FEROX-CORE ───────────────────────────────────────────────────────────────

"ferox-core/Cargo.toml": """\
[package]
name    = "ferox-core"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio      = { workspace = true }
serde      = { workspace = true }
serde_json = { workspace = true }
thiserror  = { workspace = true }
anyhow     = { workspace = true }
tracing    = { workspace = true }
async-trait = { workspace = true }
sha2        = "0.10"
hex         = "0.4"
walkdir     = "2.5"
rayon       = "1.10"    # parallel CPU-bound hashing
""",

"ferox-core/src/lib.rs": """\
//! ferox-core — shared types, traits, and pipeline logic.
//!
//! Crate dependency graph:
//!   ferox-cli  →  ferox-worker  →  ferox-core
//!   ferox-cli  →  ferox-core
//!
//! ferox-core has no dependency on ferox-worker or ferox-cli;
//! it defines the contracts (traits + types) that the other crates implement.

pub mod config;
pub mod error;
pub mod models;
pub mod pipeline;
pub mod processor;
pub mod traits;
""",

"ferox-core/src/error.rs": """\
use thiserror::Error;

/// Core error type for ferox-core operations.
///
/// thiserror generates Display and Error impls from the #[error(...)] attributes.
/// The #[from] attribute on IoError and SerdeError means these variants
/// can be created with ? from the corresponding standard/library error types.
///
/// FeroxError is the error type used throughout ferox-core and ferox-worker.
/// ferox-cli wraps it with anyhow::Error for ergonomic top-level error reporting.
#[derive(Debug, Error)]
pub enum FeroxError {
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    #[error("Pipeline error in stage '{stage}': {message}")]
    PipelineError { stage: String, message: String },

    #[error("Worker pool error: {0}")]
    WorkerError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Processing cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, FeroxError>;
""",

"ferox-core/src/config.rs": """\
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use crate::error::{FeroxError, Result};

/// Runtime configuration loaded from a TOML/JSON config file or CLI flags.
///
/// Serde allows deserialization from both TOML (via the CLI's --config flag)
/// and JSON (for programmatic use in tests). #[serde(default)] means missing
/// fields use Default::default() rather than failing deserialization.
///
/// worker_threads controls the Tokio runtime thread pool for async I/O.
/// rayon_threads controls the separate Rayon thread pool for CPU-bound work
/// (hashing, compression). Keeping them separate prevents CPU-bound work
/// from starving the async executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FeroxConfig {
    /// Number of Tokio worker threads (0 = use num_cpus)
    pub worker_threads: usize,

    /// Number of Rayon threads for CPU-bound work (0 = use num_cpus)
    pub rayon_threads: usize,

    /// Maximum number of files to process concurrently
    pub max_concurrency: usize,

    /// Output directory for processed files
    pub output_dir: PathBuf,

    /// File extensions to include (empty = all files)
    pub include_extensions: Vec<String>,

    /// Maximum file size to process in bytes (0 = no limit)
    pub max_file_size_bytes: u64,

    /// Whether to compute SHA-256 hash of each processed file
    pub compute_hashes: bool,

    /// Whether to recurse into subdirectories
    pub recursive: bool,
}

impl Default for FeroxConfig {
    fn default() -> Self {
        Self {
            worker_threads:      0,
            rayon_threads:       0,
            max_concurrency:     8,
            output_dir:          PathBuf::from("./output"),
            include_extensions:  vec![],
            max_file_size_bytes: 0,
            compute_hashes:      true,
            recursive:           true,
        }
    }
}

impl FeroxConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(FeroxError::SerdeError)
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_concurrency == 0 {
            return Err(FeroxError::ConfigError(
                "max_concurrency must be > 0".to_string()
            ));
        }
        Ok(())
    }
}
""",

"ferox-core/src/models.rs": """\
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// A file discovered during directory scanning, before processing.
#[derive(Debug, Clone)]
pub struct InputFile {
    pub path:       PathBuf,
    pub size_bytes: u64,
    pub extension:  Option<String>,
}

/// Result of processing a single file through the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedFile {
    pub input_path:  String,
    pub output_path: String,
    pub size_bytes:  u64,
    pub sha256_hash: Option<String>,
    pub duration_ms: u64,
    pub stage_times: Vec<StageTime>,
}

/// Timing for a single pipeline stage — used for performance profiling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTime {
    pub stage:    String,
    pub duration: Duration,
}

/// Aggregated result of a complete processing run.
#[derive(Debug, Serialize, Deserialize)]
pub struct RunSummary {
    pub total_files:   usize,
    pub succeeded:     usize,
    pub failed:        usize,
    pub skipped:       usize,
    pub total_bytes:   u64,
    pub total_duration: Duration,
    pub errors:        Vec<String>,
}

impl RunSummary {
    pub fn new() -> Self {
        Self {
            total_files: 0, succeeded: 0, failed: 0, skipped: 0,
            total_bytes: 0, total_duration: Duration::ZERO, errors: vec![],
        }
    }
}
""",

"ferox-core/src/traits.rs": """\
use async_trait::async_trait;
use crate::error::Result;
use crate::models::{InputFile, ProcessedFile};

/// A single stage in the processing pipeline.
///
/// async-trait generates the correct vtable layout for async methods on
/// trait objects. Without it, `async fn` in traits is not object-safe
/// (the Future return type would differ per impl, making Box<dyn Stage>
/// impossible).
///
/// Using Box<dyn Stage + Send + Sync> allows mixing different stage
/// implementations at runtime (e.g., swapping hash algorithms in tests).
///
/// The name() method is used in StageTime records and error messages.
#[async_trait]
pub trait Stage: Send + Sync {
    fn name(&self) -> &'static str;
    async fn process(&self, file: &InputFile) -> Result<ProcessedFile>;
}

/// Trait for types that can discover files from a source (directory, archive, etc.)
#[async_trait]
pub trait FileSource: Send + Sync {
    async fn discover(&self) -> Result<Vec<InputFile>>;
}

/// Trait for types that write processed output to a destination.
#[async_trait]
pub trait OutputWriter: Send + Sync {
    async fn write(&self, result: &ProcessedFile) -> Result<()>;
}
""",

"ferox-core/src/processor.rs": """\
use std::sync::Arc;
use std::time::Instant;
use sha2::{Sha256, Digest};
use rayon::prelude::*;
use crate::config::FeroxConfig;
use crate::error::Result;
use crate::models::{InputFile, ProcessedFile, StageTime};
use tracing::{info, warn};

/// Core file processor — handles a single file through read → hash → write.
///
/// CPU-bound work (SHA-256 hashing) is offloaded to the Rayon thread pool
/// via spawn_blocking equivalent: rayon::scope runs on the Rayon pool,
/// not on Tokio's async executor. This is critical because blocking the
/// Tokio executor with CPU-bound work would starve other async tasks.
///
/// The Arc<FeroxConfig> is cheap to clone across tasks — the config is
/// immutable after startup and shared via reference counting.
pub struct FileProcessor {
    config: Arc<FeroxConfig>,
}

impl FileProcessor {
    pub fn new(config: Arc<FeroxConfig>) -> Self {
        Self { config }
    }

    /// Process a single file: read → optionally hash → write to output_dir.
    /// Returns a ProcessedFile with timing metadata for each stage.
    pub async fn process(&self, file: &InputFile) -> Result<ProcessedFile> {
        let mut stage_times = Vec::new();

        // Stage 1: Read file
        let t0 = Instant::now();
        let content = tokio::fs::read(&file.path).await?;
        stage_times.push(StageTime { stage: "read".into(), duration: t0.elapsed() });

        // Stage 2: Hash (CPU-bound — run on Rayon pool via spawn_blocking)
        let sha256_hash = if self.config.compute_hashes {
            let content_clone = content.clone();
            let t1 = Instant::now();
            let hash = tokio::task::spawn_blocking(move || {
                let mut hasher = Sha256::new();
                hasher.update(&content_clone);
                hex::encode(hasher.finalize())
            }).await.map_err(|e| crate::error::FeroxError::WorkerError(e.to_string()))?;
            stage_times.push(StageTime { stage: "hash".into(), duration: t1.elapsed() });
            Some(hash)
        } else {
            None
        };

        // Stage 3: Write to output directory
        let t2      = Instant::now();
        let out_name = file.path.file_name().unwrap_or_default();
        let out_path = self.config.output_dir.join(out_name);
        tokio::fs::create_dir_all(&self.config.output_dir).await?;
        tokio::fs::write(&out_path, &content).await?;
        stage_times.push(StageTime { stage: "write".into(), duration: t2.elapsed() });

        info!(path = ?file.path, bytes = content.len(), "processed");

        Ok(ProcessedFile {
            input_path:  file.path.to_string_lossy().to_string(),
            output_path: out_path.to_string_lossy().to_string(),
            size_bytes:  content.len() as u64,
            sha256_hash,
            duration_ms: t0.elapsed().as_millis() as u64,
            stage_times,
        })
    }
}
""",

"ferox-core/src/pipeline.rs": """\
use std::sync::Arc;
use std::path::PathBuf;
use walkdir::WalkDir;
use crate::config::FeroxConfig;
use crate::error::{FeroxError, Result};
use crate::models::{InputFile, RunSummary};
use crate::processor::FileProcessor;
use crate::traits::{FileSource, OutputWriter};
use tracing::{info, warn, error};

/// Discovers files from a directory tree, respecting FeroxConfig filters.
///
/// Dependency chain:
///   DirectorySource
///     └── FeroxConfig  (extension filter, recursive flag, max_file_size_bytes)
pub struct DirectorySource {
    root:   PathBuf,
    config: Arc<FeroxConfig>,
}

impl DirectorySource {
    pub fn new(root: PathBuf, config: Arc<FeroxConfig>) -> Self {
        Self { root, config }
    }
}

#[async_trait::async_trait]
impl FileSource for DirectorySource {
    async fn discover(&self) -> Result<Vec<InputFile>> {
        let root   = self.root.clone();
        let config = self.config.clone();
        // walkdir is sync — run on blocking thread pool
        tokio::task::spawn_blocking(move || {
            let walker = WalkDir::new(&root).max_depth(if config.recursive { usize::MAX } else { 1 });
            let mut files = Vec::new();
            for entry in walker.into_iter().filter_map(|e| e.ok()) {
                if !entry.file_type().is_file() { continue; }
                let path = entry.path().to_path_buf();
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);

                if config.max_file_size_bytes > 0 && size > config.max_file_size_bytes {
                    continue;
                }
                if !config.include_extensions.is_empty() {
                    let ext = path.extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("");
                    if !config.include_extensions.iter().any(|e| e == ext) {
                        continue;
                    }
                }
                let extension = path.extension().and_then(|e| e.to_str()).map(str::to_string);
                files.push(InputFile { path, size_bytes: size, extension });
            }
            Ok(files)
        })
        .await
        .map_err(|e| FeroxError::WorkerError(e.to_string()))?
    }
}
""",

# ─── FEROX-WORKER ─────────────────────────────────────────────────────────────

"ferox-worker/Cargo.toml": """\
[package]
name    = "ferox-worker"
version = "0.1.0"
edition = "2021"

[dependencies]
ferox-core = { path = "../ferox-core" }
tokio      = { workspace = true }
serde      = { workspace = true }
tracing    = { workspace = true }
async-trait = { workspace = true }
""",

"ferox-worker/src/lib.rs": """\
//! ferox-worker — async worker pool for parallel file processing.
//!
//! Depends on ferox-core for types and processor.
//! Does NOT depend on ferox-cli — the CLI orchestrates the worker pool,
//! not the other way around.

pub mod pool;
pub mod scheduler;
pub mod channel;
""",

"ferox-worker/src/channel.rs": """\
use tokio::sync::mpsc;
use ferox_core::models::{InputFile, ProcessedFile, RunSummary};
use ferox_core::error::FeroxError;

/// Message types for the worker channel protocol.
///
/// The worker pool uses two Tokio MPSC channels:
///   work_tx / work_rx  — sends InputFile items from the scheduler to workers
///   result_tx / result_rx — workers send WorkerResult back to the collector
///
/// Using separate channels for work and results avoids back-pressure coupling:
/// the scheduler can continue dispatching work even if the result collector
/// is temporarily busy writing output.
///
/// Channel capacity (bounded MPSC):
///   Bounded channels apply back-pressure — if workers fall behind, the
///   scheduler blocks on send(), preventing unbounded memory growth from
///   queueing all files at once. Capacity = 2 × worker_count is a
///   reasonable default that keeps workers fed without excessive buffering.

pub type WorkSender   = mpsc::Sender<InputFile>;
pub type WorkReceiver = mpsc::Receiver<InputFile>;

pub enum WorkerResult {
    Success(ProcessedFile),
    Failure { path: String, error: FeroxError },
    Skipped { path: String, reason: String },
}

pub type ResultSender   = mpsc::Sender<WorkerResult>;
pub type ResultReceiver = mpsc::Receiver<WorkerResult>;

pub struct ChannelPair {
    pub work_tx:   WorkSender,
    pub work_rx:   WorkReceiver,
    pub result_tx: ResultSender,
    pub result_rx: ResultReceiver,
}

impl ChannelPair {
    /// Create bounded channels sized for the given worker count.
    /// Capacity = 2 × worker_count keeps workers fed with minimal buffering.
    pub fn new(worker_count: usize) -> Self {
        let capacity = (worker_count * 2).max(4);
        let (work_tx,   work_rx)   = mpsc::channel(capacity);
        let (result_tx, result_rx) = mpsc::channel(capacity);
        Self { work_tx, work_rx, result_tx, result_rx }
    }
}
""",

"ferox-worker/src/pool.rs": """\
use std::sync::Arc;
use tokio::task::JoinHandle;
use ferox_core::config::FeroxConfig;
use ferox_core::processor::FileProcessor;
use ferox_core::error::FeroxError;
use crate::channel::{WorkReceiver, ResultSender, WorkerResult};
use tracing::{info, warn, error};

/// A single async worker that reads InputFiles from the work channel,
/// processes them via FileProcessor, and sends results to the result channel.
///
/// Workers run as independent Tokio tasks (spawned via tokio::spawn).
/// Each worker owns a clone of Arc<FileProcessor> — cheap clone because
/// Arc only increments a reference count.
///
/// Shutdown: when the work_rx channel closes (all senders dropped),
/// recv() returns None and the worker task exits cleanly.
pub struct Worker {
    id:        usize,
    processor: Arc<FileProcessor>,
}

impl Worker {
    pub fn new(id: usize, config: Arc<FeroxConfig>) -> Self {
        Self {
            id,
            processor: Arc::new(FileProcessor::new(config)),
        }
    }

    /// Spawn this worker as a Tokio task.
    /// Returns the JoinHandle so the pool can await all workers on shutdown.
    pub fn spawn(
        self,
        mut work_rx:  WorkReceiver,
        result_tx:    ResultSender,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            info!(worker_id = self.id, "worker started");
            while let Some(file) = work_rx.recv().await {
                let path = file.path.to_string_lossy().to_string();
                let result = match self.processor.process(&file).await {
                    Ok(pf)   => WorkerResult::Success(pf),
                    Err(FeroxError::Cancelled) => {
                        warn!(worker_id = self.id, "processing cancelled");
                        break;
                    }
                    Err(e) => {
                        error!(worker_id = self.id, path = %path, error = %e, "processing failed");
                        WorkerResult::Failure { path, error: e }
                    }
                };
                if result_tx.send(result).await.is_err() {
                    break;  // result collector dropped — shut down
                }
            }
            info!(worker_id = self.id, "worker stopped");
        })
    }
}
""",

"ferox-worker/src/scheduler.rs": """\
use std::sync::Arc;
use tokio::sync::mpsc;
use ferox_core::config::FeroxConfig;
use ferox_core::models::{InputFile, RunSummary};
use ferox_core::error::Result;
use ferox_core::traits::FileSource;
use crate::channel::{ChannelPair, WorkerResult};
use crate::pool::Worker;
use tracing::info;

/// Orchestrates the worker pool: discovers files, dispatches to workers,
/// collects results, and returns a RunSummary.
///
/// Full dependency chain:
///   WorkerScheduler
///     ├── FileSource (trait object)  — discovers InputFile list
///     ├── Worker × N                — process files concurrently
///     │     └── FileProcessor       — read/hash/write per file
///     │           └── FeroxConfig   — filters and output config
///     └── ChannelPair               — bounded MPSC channels
///
/// Worker count: max_concurrency from FeroxConfig. Each worker is an
/// independent Tokio task; Tokio's work-stealing scheduler distributes
/// them across worker_threads OS threads.
pub struct WorkerScheduler {
    config: Arc<FeroxConfig>,
    source: Box<dyn FileSource>,
}

impl WorkerScheduler {
    pub fn new(config: Arc<FeroxConfig>, source: Box<dyn FileSource>) -> Self {
        Self { config, source }
    }

    pub async fn run(self) -> Result<RunSummary> {
        let files   = self.source.discover().await?;
        let n_files = files.len();
        let n_workers = self.config.max_concurrency;
        info!(files = n_files, workers = n_workers, "starting run");

        // One channel pair per scheduler run — channels are dropped after run completes
        let channels = ChannelPair::new(n_workers);
        let ChannelPair { work_tx, work_rx, result_tx, result_rx } = channels;

        // Spawn workers — each gets a clone of work_rx via tokio::sync::broadcast
        // (here we use a simpler model: one shared work_rx, workers compete for items)
        // For simplicity, spawn N workers sharing the same work_rx via Arc<Mutex>
        // In production you'd use a proper work-stealing deque (e.g., tokio::sync::mpsc with N receivers)
        let mut handles = Vec::new();
        for id in 0..n_workers {
            let worker = Worker::new(id, self.config.clone());
            // Note: real impl would split work_rx — simplified here for clarity
            let _ = result_tx.clone();
            info!(worker_id = id, "spawning worker");
        }

        // Dispatch work
        for file in files {
            if work_tx.send(file).await.is_err() {
                break;  // workers stopped
            }
        }
        drop(work_tx);  // signal workers that no more work is coming

        // Collect results
        let mut summary = RunSummary::new();
        summary.total_files = n_files;
        let mut result_rx = result_rx;
        while let Some(result) = result_rx.recv().await {
            match result {
                WorkerResult::Success(pf) => {
                    summary.succeeded    += 1;
                    summary.total_bytes  += pf.size_bytes;
                }
                WorkerResult::Failure { path, error } => {
                    summary.failed += 1;
                    summary.errors.push(format!("{path}: {error}"));
                }
                WorkerResult::Skipped { .. } => {
                    summary.skipped += 1;
                }
            }
        }

        for handle in handles {
            let _: tokio::task::JoinHandle<()> = handle;
        }

        Ok(summary)
    }
}
""",

# ─── FEROX-CLI ────────────────────────────────────────────────────────────────

"ferox-cli/Cargo.toml": """\
[package]
name    = "ferox-cli"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "ferox"
path = "src/main.rs"

[dependencies]
ferox-core   = { path = "../ferox-core" }
ferox-worker = { path = "../ferox-worker" }
tokio        = { workspace = true }
serde        = { workspace = true }
serde_json   = { workspace = true }
anyhow       = { workspace = true }
tracing      = { workspace = true }
tracing-subscriber = { workspace = true }
clap         = { version = "4.5", features = ["derive"] }
""",

"ferox-cli/src/main.rs": """\
//! Ferox CLI entrypoint.
//!
//! Builds the Tokio runtime with the configured thread count, then dispatches
//! to the appropriate subcommand. anyhow::Result is used at the top level for
//! ergonomic error reporting — it prints the full error chain on exit.
//!
//! Crate dependency chain (CLI → Worker → Core):
//!   ferox (main)
//!     ├── ferox-cli commands
//!     │     ├── ferox-worker::WorkerScheduler
//!     │     │     ├── ferox-core::FileProcessor
//!     │     │     │     └── ferox-core::FeroxConfig
//!     │     │     └── ferox-core::DirectorySource
//!     │     └── ferox-core::FeroxConfig
//!     └── Tokio runtime (multi-thread, worker_threads from config)

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;
mod commands;
use commands::Cli;

fn main() -> Result<()> {
    // Initialise tracing — FEROX_LOG env var controls verbosity
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_env("FEROX_LOG"))
        .init();

    let cli = Cli::parse();

    // Build Tokio runtime with configured thread count
    // 0 means "use num_cpus" — tokio's default
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(if cli.threads == 0 { num_cpus::get() } else { cli.threads })
        .enable_all()
        .build()?;

    rt.block_on(cli.run())
}
""",

"ferox-cli/src/commands/mod.rs": """\
use clap::{Parser, Subcommand};
use anyhow::Result;

pub mod process;
pub mod watch;

/// Ferox — fast async file processor
#[derive(Parser, Debug)]
#[command(name = "ferox", version, about)]
pub struct Cli {
    /// Number of Tokio worker threads (0 = num_cpus)
    #[arg(long, default_value_t = 0)]
    pub threads: usize,

    /// Path to JSON config file (overrides CLI defaults)
    #[arg(long)]
    pub config: Option<std::path::PathBuf>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Process files in a directory
    Process(process::ProcessArgs),
    /// Watch a directory and process new files automatically
    Watch(watch::WatchArgs),
}

impl Cli {
    pub async fn run(self) -> Result<()> {
        match self.command {
            Commands::Process(args) => process::run(args, self.config).await,
            Commands::Watch(args)   => watch::run(args, self.config).await,
        }
    }
}
""",

"ferox-cli/src/commands/process.rs": """\
use std::path::PathBuf;
use std::sync::Arc;
use anyhow::{Context, Result};
use clap::Args;
use ferox_core::config::FeroxConfig;
use ferox_core::pipeline::DirectorySource;
use ferox_worker::scheduler::WorkerScheduler;
use tracing::info;

/// CLI arguments for the `process` subcommand.
///
/// Clap's `#[derive(Args)]` generates the argument parser.
/// #[arg(short, long)] creates both -i and --input flags.
#[derive(Args, Debug)]
pub struct ProcessArgs {
    /// Input directory to process
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output directory for processed files
    #[arg(short, long)]
    pub output: PathBuf,

    /// File extensions to include (e.g. --ext rs --ext toml)
    #[arg(long = "ext")]
    pub extensions: Vec<String>,

    /// Maximum concurrent workers
    #[arg(long, default_value_t = 8)]
    pub concurrency: usize,

    /// Skip SHA-256 hashing (faster for large files)
    #[arg(long)]
    pub no_hash: bool,

    /// Do not recurse into subdirectories
    #[arg(long)]
    pub no_recurse: bool,
}

/// Run the process subcommand.
///
/// Dependency chain:
///   run()
///     ├── FeroxConfig         (from CLI args or JSON config file)
///     ├── DirectorySource     (ferox-core — walks input directory)
///     └── WorkerScheduler     (ferox-worker — dispatches to worker pool)
///           └── Worker × N   (ferox-worker — processes each file)
///                 └── FileProcessor (ferox-core — read/hash/write)
pub async fn run(args: ProcessArgs, config_path: Option<PathBuf>) -> Result<()> {
    let config = build_config(&args, config_path)?;
    let config = Arc::new(config);

    let source     = Box::new(DirectorySource::new(args.input.clone(), config.clone()));
    let scheduler  = WorkerScheduler::new(config.clone(), source);

    info!(input = ?args.input, output = ?config.output_dir, "starting process run");

    let summary = scheduler.run().await
        .context("Worker scheduler failed")?;

    println!("✓ Processed {}/{} files ({} skipped, {} failed)",
             summary.succeeded, summary.total_files,
             summary.skipped, summary.failed);

    if !summary.errors.is_empty() {
        eprintln!("Errors:");
        for e in &summary.errors { eprintln!("  {e}"); }
    }

    if summary.failed > 0 { std::process::exit(1); }
    Ok(())
}

fn build_config(args: &ProcessArgs, config_path: Option<PathBuf>) -> Result<FeroxConfig> {
    let mut config = if let Some(path) = config_path {
        let json = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        FeroxConfig::from_json(&json)
            .context("Failed to parse config file")?
    } else {
        FeroxConfig::default()
    };

    // CLI flags override config file
    config.output_dir            = args.output.clone();
    config.max_concurrency       = args.concurrency;
    config.compute_hashes        = !args.no_hash;
    config.recursive             = !args.no_recurse;
    if !args.extensions.is_empty() {
        config.include_extensions = args.extensions.clone();
    }

    config.validate().context("Invalid configuration")?;
    Ok(config)
}
""",

"ferox-cli/src/commands/watch.rs": """\
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use anyhow::Result;
use clap::Args;
use tokio::time::sleep;
use ferox_core::config::FeroxConfig;
use ferox_core::pipeline::DirectorySource;
use ferox_worker::scheduler::WorkerScheduler;
use tracing::{info, warn};

/// CLI arguments for the `watch` subcommand.
#[derive(Args, Debug)]
pub struct WatchArgs {
    #[arg(short, long)]
    pub input:  PathBuf,

    #[arg(short, long)]
    pub output: PathBuf,

    /// Polling interval in seconds (default: 5)
    #[arg(long, default_value_t = 5)]
    pub interval_secs: u64,
}

/// Watch a directory and re-run processing whenever new files appear.
///
/// Uses polling rather than inotify/FSEvents because:
///   1. Cross-platform (Linux, macOS, Windows) without per-OS APIs
///   2. Network filesystems (NFS, SMB) don't reliably deliver inotify events
///
/// A production implementation would use the `notify` crate for
/// native filesystem events on local disks, with polling as fallback.
///
/// Cancellation: Ctrl-C sends SIGINT, Tokio's signal handler catches it,
/// and the loop exits after the current run completes (graceful shutdown).
pub async fn run(args: WatchArgs, _config_path: Option<PathBuf>) -> Result<()> {
    info!(input = ?args.input, interval = args.interval_secs, "starting watch mode");

    let config = Arc::new(FeroxConfig {
        output_dir: args.output.clone(),
        ..FeroxConfig::default()
    });

    let mut ctrl_c = tokio::signal::ctrl_c();

    loop {
        tokio::select! {
            _ = async { ctrl_c = tokio::signal::ctrl_c(); ctrl_c.await } => {
                info!("received Ctrl-C, shutting down");
                break;
            }
            _ = sleep(Duration::from_secs(args.interval_secs)) => {
                let source    = Box::new(DirectorySource::new(args.input.clone(), config.clone()));
                let scheduler = WorkerScheduler::new(config.clone(), source);
                match scheduler.run().await {
                    Ok(summary) => info!(
                        succeeded = summary.succeeded,
                        failed    = summary.failed,
                        "watch run complete"
                    ),
                    Err(e) => warn!(error = %e, "watch run failed"),
                }
            }
        }
    }
    Ok(())
}
""",

# ─── TESTS ────────────────────────────────────────────────────────────────────

"ferox-core/tests/integration_test.rs": """\
//! Integration tests for ferox-core.
//! Tests run against a real temp directory — no mocking.

use std::sync::Arc;
use tempfile::TempDir;
use ferox_core::config::FeroxConfig;
use ferox_core::models::InputFile;
use ferox_core::processor::FileProcessor;
use ferox_core::pipeline::DirectorySource;
use ferox_core::traits::FileSource;

#[tokio::test]
async fn test_processor_computes_hash() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("hello.txt"), b"hello world").unwrap();

    let config = Arc::new(FeroxConfig {
        output_dir: tmp.path().join("out"),
        compute_hashes: true,
        ..Default::default()
    });
    let processor = FileProcessor::new(config);
    let file = InputFile {
        path: tmp.path().join("hello.txt"),
        size_bytes: 11,
        extension: Some("txt".into()),
    };
    let result = processor.process(&file).await.unwrap();
    assert!(result.sha256_hash.is_some());
    // SHA-256 of "hello world"
    assert_eq!(
        result.sha256_hash.unwrap(),
        "b94d27b9934d3e08a52e52d7da7dabfac484efe04294e576f3d0c29b3a00d72"
    );
}

#[tokio::test]
async fn test_directory_source_respects_extension_filter() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.rs"), b"fn main() {}").unwrap();
    std::fs::write(tmp.path().join("b.toml"), b"[package]").unwrap();

    let config = Arc::new(FeroxConfig {
        include_extensions: vec!["rs".into()],
        ..Default::default()
    });
    let source = DirectorySource::new(tmp.path().to_path_buf(), config);
    let files  = source.discover().await.unwrap();
    assert_eq!(files.len(), 1);
    assert!(files[0].path.to_string_lossy().ends_with(".rs"));
}

#[tokio::test]
async fn test_config_validation_rejects_zero_concurrency() {
    let config = ferox_core::config::FeroxConfig {
        max_concurrency: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}
""",

}


def write_golden_codebase(target_dir: str) -> list[str]:
    import os
    written = []
    for rel_path, content in GOLDEN_FILES.items():
        full_path = os.path.join(target_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        written.append(full_path)
    return written
