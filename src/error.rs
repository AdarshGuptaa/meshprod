//! Named pipeline failures.
//!
//! Every failure carries a stable machine code, a human description and
//! actionable hints derived from the measured failure catalogue (see the
//! README's troubleshooting table). The CLI writes `<output>_error.json` so the
//! web UI can display a named, actionable error instead of a stack of logs.

use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Pipeline stage most recently entered (1..=9), for failure attribution.
static STAGE: AtomicUsize = AtomicUsize::new(0);

/// Record the stage currently running.
pub fn set_stage(stage: usize) {
    STAGE.store(stage, Ordering::Relaxed);
}

/// Stage recorded by the last [`set_stage`].
pub fn current_stage() -> usize {
    STAGE.load(Ordering::Relaxed)
}

/// A named, actionable pipeline failure.
#[derive(Debug)]
pub enum PipelineError {
    /// CLI or config value rejected before the pipeline starts.
    InvalidArgs(String),
    /// One of `--fx/--fy/--cx/--cy` missing or non-positive.
    MissingIntrinsics,
    /// Video could not be decoded (empty, truncated, audio-only, codec).
    VideoDecode(String),
    /// Fewer than two usable frames after subsampling.
    NotEnoughFrames {
        /// Frames actually kept.
        got: usize,
    },
    /// SfM could not establish an initial pair.
    BootstrapFailed(String),
    /// Too few views registered for a useful reconstruction.
    LowRegistration {
        /// Views registered.
        registered: usize,
        /// Views attempted.
        total: usize,
    },
    /// The dense stage produced no depth maps.
    NoDepthMaps,
    /// The dense cloud is empty after plane removal / cropping.
    EmptyCloud,
    /// TSDF produced no triangles.
    EmptyMesh,
    /// Cleanup removed every patch.
    MeshPruned,
    /// `--cuda` requested but no device is usable.
    CudaUnavailable(String),
    /// A stage failed for a reason without a more specific classification.
    Processing(String),
    /// Filesystem failure.
    Io {
        /// Path involved.
        path: String,
        /// Underlying message.
        message: String,
    },
}

impl PipelineError {
    /// Stable machine-readable code.
    pub fn code(&self) -> &'static str {
        match self {
            PipelineError::InvalidArgs(_) => "invalid-args",
            PipelineError::MissingIntrinsics => "missing-intrinsics",
            PipelineError::VideoDecode(_) => "video-decode",
            PipelineError::NotEnoughFrames { .. } => "not-enough-frames",
            PipelineError::BootstrapFailed(_) => "bootstrap-failed",
            PipelineError::LowRegistration { .. } => "low-registration",
            PipelineError::NoDepthMaps => "no-depth-maps",
            PipelineError::EmptyCloud => "empty-cloud",
            PipelineError::EmptyMesh => "empty-mesh",
            PipelineError::MeshPruned => "mesh-pruned",
            PipelineError::CudaUnavailable(_) => "cuda-unavailable",
            PipelineError::Processing(_) => "processing-error",
            PipelineError::Io { .. } => "io-error",
        }
    }

    /// Actionable hints, ordered by likely payoff.
    pub fn hints(&self) -> Vec<&'static str> {
        match self {
            PipelineError::InvalidArgs(_) => {
                vec!["Check ranges: --frame-step >= 1, --match-window >= 1, --dense-planes >= 2."]
            }
            PipelineError::MissingIntrinsics => vec![
                "fx = f_35mm / 36 * frame_width (about 895 px for 1280x720 phone video).",
                "cx/cy are the image centre for a phone lens.",
            ],
            PipelineError::VideoDecode(_) => vec![
                "Confirm the file plays (audio-only and truncated files decode no frames).",
                "Install GStreamer plugins: good, libav (H.264/H.265 decoders).",
                "The pinned kornia-io reports end-of-stream; older builds time out at 30 s.",
            ],
            PipelineError::NotEnoughFrames { .. } => vec![
                "Reduce --frame-step so more frames are kept.",
                "Check that the clip is longer than a couple of seconds.",
            ],
            PipelineError::BootstrapFailed(_) => vec![
                "SfM needs translation (parallax); panning in place cannot bootstrap.",
                "Trim to a segment with real camera motion.",
                "Reject blurry frames: increase shutter speed / light.",
            ],
            PipelineError::LowRegistration { .. } => vec![
                "Use the `registered source frames` log to find the usable seconds.",
                "Markers or a textured background give matches across larger baselines.",
                "Avoid moving the object during the capture.",
            ],
            PipelineError::NoDepthMaps => vec![
                "Lower --ncc-min, or try --cost sgm.",
                "Textureless surfaces need physical texture (stickers, matte spray).",
            ],
            PipelineError::EmptyCloud => vec![
                "Disable --remove-plane or --auto-crop for this capture.",
                "Check that dense depth is not empty (depth vs sparse log line).",
            ],
            PipelineError::EmptyMesh => vec![
                "Lower --ncc-min or raise --voxel-resolution.",
                "Too few registered views is the usual cause; improve the capture.",
            ],
            PipelineError::MeshPruned => vec![
                "Lower --min-component-faces.",
                "The surface is fragmented; try --cost sgm or a better capture.",
            ],
            PipelineError::CudaUnavailable(_) => vec![
                "Run without --cuda (the CPU path is fully supported).",
                "Install an NVIDIA driver and the CUDA runtime if the GPU is intended.",
            ],
            PipelineError::Processing(_) => vec![
                "Read the log line before the error for the failing stage.",
                "If it looks like a bug, attach the run config and log to an issue.",
            ],
            PipelineError::Io { .. } => vec![
                "Check that the path exists and the disk has free space.",
                "Windows paths with spaces must be quoted.",
            ],
        }
    }

    /// JSON for the web UI / `_error.json`.
    pub fn to_json(&self, stage: usize) -> serde_json::Value {
        serde_json::json!({
            "code": self.code(),
            "message": self.to_string(),
            "hints": self.hints(),
            "stage": stage,
        })
    }
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::InvalidArgs(m) => write!(f, "invalid arguments: {m}"),
            PipelineError::MissingIntrinsics => {
                write!(f, "camera intrinsics are required (--fx --fy --cx --cy)")
            }
            PipelineError::VideoDecode(m) => write!(f, "video decode failed: {m}"),
            PipelineError::NotEnoughFrames { got } => {
                write!(f, "only {got} frame(s) kept; at least 2 are required")
            }
            PipelineError::BootstrapFailed(m) => write!(f, "SfM bootstrap failed: {m}"),
            PipelineError::LowRegistration { registered, total } => write!(
                f,
                "only {registered}/{total} views registered — not enough for a reliable mesh"
            ),
            PipelineError::NoDepthMaps => write!(f, "no dense depth maps were produced"),
            PipelineError::EmptyCloud => write!(f, "the dense cloud is empty after filtering"),
            PipelineError::EmptyMesh => {
                write!(f, "the TSDF produced no surface (empty zero level set)")
            }
            PipelineError::MeshPruned => write!(f, "cleanup removed every mesh patch"),
            PipelineError::CudaUnavailable(m) => write!(f, "CUDA is unavailable: {m}"),
            PipelineError::Processing(m) => write!(f, "pipeline stage failed: {m}"),
            PipelineError::Io { path, message } => write!(f, "I/O error at {path}: {message}"),
        }
    }
}

impl std::error::Error for PipelineError {}

/// The full failure catalogue, for the web UI's diagnostics panel.
pub fn catalogue() -> Vec<serde_json::Value> {
    vec![
        PipelineError::InvalidArgs(String::new()),
        PipelineError::MissingIntrinsics,
        PipelineError::VideoDecode(String::new()),
        PipelineError::NotEnoughFrames { got: 0 },
        PipelineError::BootstrapFailed(String::new()),
        PipelineError::LowRegistration {
            registered: 0,
            total: 0,
        },
        PipelineError::NoDepthMaps,
        PipelineError::EmptyCloud,
        PipelineError::EmptyMesh,
        PipelineError::MeshPruned,
        PipelineError::CudaUnavailable(String::new()),
        PipelineError::Processing(String::new()),
        PipelineError::Io {
            path: String::new(),
            message: String::new(),
        },
    ]
    .iter()
    .map(|e| {
        serde_json::json!({
            "code": e.code(),
            "description": e.to_string(),
            "hints": e.hints(),
        })
    })
    .collect()
}

/// Emit a machine-readable stage progress line.
///
/// The web UI parses `PROGRESS <stage> <done> <total>` from stderr to drive
/// per-stage loading bars. Stages that cannot report partial work are simply
/// emitted as `1/1` when they finish.
pub fn progress(stage: usize, done: usize, total: usize) {
    eprintln!("PROGRESS {stage} {done} {}", total.max(1));
}

/// Write `<output>_error.json` next to the requested output.
pub fn write_error_file(output: &Path, error: &PipelineError) {
    let path = sibling_json(output, "_error");
    if let Ok(json) = serde_json::to_string_pretty(&error.to_json(current_stage())) {
        let _ = std::fs::write(path, json);
    }
}

/// `<stem><suffix>.json` next to `output`.
pub fn sibling_json(output: &Path, suffix: &str) -> std::path::PathBuf {
    let stem = output
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    output.with_file_name(format!("{stem}{suffix}.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_variants() -> Vec<PipelineError> {
        vec![
            PipelineError::InvalidArgs("x".into()),
            PipelineError::MissingIntrinsics,
            PipelineError::VideoDecode("x".into()),
            PipelineError::NotEnoughFrames { got: 1 },
            PipelineError::BootstrapFailed("x".into()),
            PipelineError::LowRegistration {
                registered: 2,
                total: 100,
            },
            PipelineError::NoDepthMaps,
            PipelineError::EmptyCloud,
            PipelineError::EmptyMesh,
            PipelineError::MeshPruned,
            PipelineError::CudaUnavailable("x".into()),
            PipelineError::Processing("x".into()),
            PipelineError::Io {
                path: "x".into(),
                message: "y".into(),
            },
        ]
    }

    #[test]
    fn every_error_has_a_code_description_and_hints() {
        let mut codes = std::collections::HashSet::new();
        for error in all_variants() {
            assert!(!error.code().is_empty());
            assert!(
                codes.insert(error.code()),
                "duplicate code {}",
                error.code()
            );
            assert!(!error.to_string().is_empty());
            assert!(!error.hints().is_empty());
            let json = error.to_json(7);
            assert_eq!(json["stage"], 7);
            assert_eq!(json["code"], error.code());
        }
    }

    #[test]
    fn error_file_is_written_with_the_code() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("mesh.ply");
        set_stage(4);
        write_error_file(
            &output,
            &PipelineError::LowRegistration {
                registered: 3,
                total: 82,
            },
        );
        let path = sibling_json(&output, "_error");
        let text = std::fs::read_to_string(path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["code"], "low-registration");
        assert_eq!(json["stage"], 4);
        assert!(json["hints"].as_array().unwrap().len() >= 2);
    }
}
