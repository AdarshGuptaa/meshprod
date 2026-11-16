//! Local web UI backend: runs the CLI pipeline as a child process and exposes
//! progress, metrics and mesh artifacts over HTTP.
//!
//! The pipeline itself is untouched — the server streams the child's stderr,
//! parses the stage/metric lines the CLI already prints, and serves the PLY
//! outputs plus a small JSON API consumed by the frontend in `web/`.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use axum::{
    extract::{DefaultBodyLimit, Multipart, Path as UrlPath, State},
    http::StatusCode,
    response::{Json, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};

use crate::error;

/// Every CLI knob, mirrored for the web form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub detector: String,
    pub frame_step: usize,
    pub n_features: usize,
    pub match_window: usize,
    pub ratio: f32,
    pub geo_verify: bool,
    pub geo_threshold: f64,
    pub geo_min_inliers: usize,
    pub wide_baseline: usize,
    pub sprt: bool,
    pub refine_intrinsics: bool,
    pub auto_config: bool,
    pub remove_plane: bool,
    pub dense_scale: f64,
    pub dense_planes: usize,
    pub dense_window: usize,
    pub dense_neighbors: usize,
    pub ncc_min: f32,
    pub lr_required: usize,
    pub cost: String,
    pub sgm_paths: usize,
    pub voxel_resolution: usize,
    pub trunc_voxels: f64,
    pub volume_smoothing: usize,
    pub hole_fill_iterations: usize,
    pub smooth_iterations: usize,
    pub bilateral_iterations: usize,
    pub needle_aspect: f64,
    pub target_faces: usize,
    pub min_component_faces: usize,
    /// In-scene Tag36h11 edge length in metres (0 = up-to-scale).
    pub scale_tag_m: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            fx: 895.0,
            fy: 895.0,
            cx: 640.0,
            cy: 360.0,
            detector: "sift".into(),
            frame_step: 10,
            n_features: 500,
            match_window: 5,
            ratio: 0.8,
            geo_verify: true,
            geo_threshold: 3.0,
            geo_min_inliers: 15,
            wide_baseline: 0,
            sprt: true,
            refine_intrinsics: false,
            auto_config: true,
            remove_plane: false,
            dense_scale: 0.5,
            dense_planes: 96,
            dense_window: 5,
            dense_neighbors: 3,
            ncc_min: 0.4,
            lr_required: 1,
            cost: "ncc".into(),
            sgm_paths: 4,
            voxel_resolution: 256,
            trunc_voxels: 3.0,
            volume_smoothing: 2,
            hole_fill_iterations: 0,
            smooth_iterations: 8,
            bilateral_iterations: 2,
            needle_aspect: 0.0,
            target_faces: 100_000,
            min_component_faces: 200,
            scale_tag_m: 0.0,
        }
    }
}

/// Metrics parsed from the pipeline log.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Stats {
    pub texture: Option<f64>,
    pub cost: Option<String>,
    pub views: Option<usize>,
    pub points: Option<usize>,
    pub rmse: Option<f64>,
    pub depth_median: Option<f64>,
    pub depth_p90: Option<f64>,
    pub raw_faces: Option<usize>,
    pub faces: Option<usize>,
    pub patches: Option<usize>,
    pub euler: Option<i64>,
    pub closed: Option<bool>,
    pub seconds: Option<f64>,
}

/// One finished run, kept for the dashboard charts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub name: String,
    pub views: usize,
    pub points: usize,
    pub rmse: f64,
    pub depth_median: f64,
    pub faces: usize,
    pub patches: usize,
    pub euler: i64,
    pub closed: bool,
    pub cost: String,
}

#[derive(Debug, Clone, Default, Serialize)]
struct ProgressView {
    stage: usize,
    done: usize,
    total: usize,
}

struct Job {
    status: String,
    stage: usize,
    progress: Option<ProgressView>,
    log: Vec<String>,
    stats: Stats,
    error: Option<serde_json::Value>,
    report: Option<serde_json::Value>,
    video: Option<PathBuf>,
    output: Option<PathBuf>,
}

#[derive(Clone)]
struct App {
    job: Arc<Mutex<Job>>,
    dir: PathBuf,
    history: Arc<Mutex<Vec<RunSummary>>>,
}

/// Start the local web UI on `port` (binds 127.0.0.1 only).
pub fn serve(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from("web_runs");
    std::fs::create_dir_all(&dir)?;

    let history_path = dir.join("history.json");
    let mut seeded: Vec<RunSummary> = if history_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&history_path)?).unwrap_or_default()
    } else {
        Vec::new()
    };
    if seeded.is_empty() {
        seeded = seed_history();
        let _ = std::fs::write(&history_path, serde_json::to_string_pretty(&seeded)?);
    }

    let app = App {
        job: Arc::new(Mutex::new(Job {
            status: "idle".into(),
            stage: 0,
            progress: None,
            log: Vec::new(),
            stats: Stats::default(),
            error: None,
            report: None,
            video: None,
            output: None,
        })),
        dir: dir.clone(),
        history: Arc::new(Mutex::new(seeded)),
    };

    println!("meshprod UI: http://127.0.0.1:{port}");
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move {
        let router = Router::new()
            .route("/api/defaults", get(defaults))
            .route("/api/status", get(status))
            .route("/api/history", get(history))
            .route("/api/diagnostics", get(diagnostics))
            .route("/api/run", post(run))
            .route("/api/calibrate", post(calibrate))
            .route("/api/mesh/{kind}", get(mesh))
            .fallback_service(tower_http::services::ServeDir::new("web"))
            .layer(DefaultBodyLimit::max(4 * 1024 * 1024 * 1024))
            .with_state(app);
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
        axum::serve(listener, router).await?;
        Ok::<(), Box<dyn std::error::Error>>(())
    })?;
    Ok(())
}

fn seed_history() -> Vec<RunSummary> {
    vec![
        RunSummary {
            name: "swiss_knife".into(),
            views: 22,
            points: 1175,
            rmse: 2.086,
            depth_median: 4.0,
            faces: 86971,
            patches: 33,
            euler: -97,
            closed: false,
            cost: "ncc".into(),
        },
        RunSummary {
            name: "lip_balm".into(),
            views: 66,
            points: 1622,
            rmse: 2.05,
            depth_median: 14.8,
            faces: 19581,
            patches: 41,
            euler: -196,
            closed: false,
            cost: "sgm".into(),
        },
        RunSummary {
            name: "remote".into(),
            views: 50,
            points: 1048,
            rmse: 1.951,
            depth_median: 4.2,
            faces: 768,
            patches: 2,
            euler: 2,
            closed: false,
            cost: "sgm".into(),
        },
        RunSummary {
            name: "phone_stand".into(),
            views: 140,
            points: 2656,
            rmse: 2.012,
            depth_median: 9.1,
            faces: 21925,
            patches: 20,
            euler: -233,
            closed: false,
            cost: "sgm".into(),
        },
        RunSummary {
            name: "keyboard".into(),
            views: 63,
            points: 1077,
            rmse: 2.03,
            depth_median: 4.5,
            faces: 4880,
            patches: 4,
            euler: -9,
            closed: false,
            cost: "sgm".into(),
        },
    ]
}

async fn defaults() -> Json<Config> {
    Json(Config::default())
}

async fn diagnostics() -> Json<Vec<serde_json::Value>> {
    Json(error::catalogue())
}

async fn history(State(app): State<App>) -> Json<Vec<RunSummary>> {
    Json(app.history.lock().unwrap().clone())
}

#[derive(Serialize)]
struct StatusView {
    status: String,
    stage: usize,
    log: Vec<String>,
    stats: Stats,
    progress: Option<ProgressView>,
    error: Option<serde_json::Value>,
    report: Option<serde_json::Value>,
    video: Option<String>,
    kinds: Vec<String>,
}

async fn status(State(app): State<App>) -> Json<StatusView> {
    let job = app.job.lock().unwrap();
    let mut kinds = Vec::new();
    if job.output.is_some() {
        kinds = ["mesh", "sparse", "cloud", "oriented"]
            .iter()
            .map(|s| s.to_string())
            .collect();
    }
    let log_start = job.log.len().saturating_sub(300);
    Json(StatusView {
        status: job.status.clone(),
        stage: job.stage,
        log: job.log[log_start..].to_vec(),
        stats: job.stats.clone(),
        progress: job.progress.clone(),
        error: job.error.clone(),
        report: job.report.clone(),
        video: job
            .video
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned()),
        kinds,
    })
}

async fn run(
    State(app): State<App>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    {
        let job = app.job.lock().unwrap();
        if job.status == "running" {
            return Err((StatusCode::CONFLICT, "a run is already in progress".into()));
        }
    }
    let mut filename = "input.mp4".to_string();
    let mut config = Config::default();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let stem = format!("run-{stamp}");
    let video_path = app.dir.join(format!("{stem}.mp4"));
    let mut video_written = false;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        match field.name() {
            Some("video") => {
                if let Some(name) = field.file_name() {
                    filename = name.to_string();
                }
                // Stream straight to disk: a 1 GB clip must not also live in RAM.
                let mut file = tokio::fs::File::create(&video_path)
                    .await
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                let mut field = field;
                while let Some(chunk) = field
                    .chunk()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
                {
                    tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
                        .await
                        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                }
                tokio::io::AsyncWriteExt::flush(&mut file)
                    .await
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                video_written = true;
            }
            Some("config") => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
                config = serde_json::from_str(&text)
                    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            }
            _ => {}
        }
    }

    if !video_written {
        return Err((StatusCode::BAD_REQUEST, "no video uploaded".into()));
    }
    let output = app.dir.join(format!("{stem}.ply"));
    let size_mb = std::fs::metadata(&video_path)
        .map(|m| m.len() / 1_048_576)
        .unwrap_or(0);

    {
        let mut job = app.job.lock().unwrap();
        *job = Job {
            status: "running".into(),
            stage: 0,
            progress: None,
            log: vec![format!("uploaded {filename} ({size_mb} MB)")],
            stats: Stats::default(),
            error: None,
            report: None,
            video: Some(video_path.clone()),
            output: None,
        };
    }

    let job_state = app.job.clone();
    let history_state = app.history.clone();
    let history_path = app.dir.join("history.json");
    let run_name = filename.clone();
    let cfg = config.clone();
    let out_path = output.clone();
    std::thread::spawn(move || {
        let exe = std::env::current_exe().expect("current exe");
        let mut cmd = Command::new(exe);
        cmd.arg(&video_path).arg("--output").arg(&out_path);
        cmd.args(build_args(&cfg));
        // stdout is unused; leaving it piped would block the child once full.
        cmd.stdout(Stdio::null()).stderr(Stdio::piped());

        let started = std::time::Instant::now();
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let mut j = job_state.lock().unwrap();
                j.status = "error".into();
                j.log.push(format!("spawn failed: {e}"));
                return;
            }
        };
        if let Some(stderr) = child.stderr.take() {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let mut j = job_state.lock().unwrap();
                parse_line(&line, &mut j);
                j.log.push(line);
                if j.log.len() > 2000 {
                    j.log.drain(0..500);
                }
            }
        }
        let code = child.wait().ok().and_then(|s| s.code());
        let mut j = job_state.lock().unwrap();
        j.stats.seconds = Some(started.elapsed().as_secs_f64());
        if code == Some(0) && out_path.exists() {
            j.status = "done".into();
            j.output = Some(out_path.clone());
            // Prefer the machine-readable report over log parsing.
            let report_path = error::sibling_json(&out_path, "_report");
            if let Ok(text) = std::fs::read_to_string(&report_path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(stats) = json.get("stats") {
                        let get = |key: &str| stats.get(key).cloned().unwrap_or_default();
                        j.stats.views = get("views").as_u64().map(|v| v as usize);
                        j.stats.points = get("points").as_u64().map(|v| v as usize);
                        j.stats.rmse = get("rmse_px").as_f64();
                        j.stats.depth_median = get("depth_median_pct").as_f64();
                        j.stats.depth_p90 = get("depth_p90_pct").as_f64();
                        j.stats.raw_faces = get("raw_faces").as_u64().map(|v| v as usize);
                        j.stats.faces = get("faces").as_u64().map(|v| v as usize);
                        j.stats.cost = get("cost").as_str().map(|s| s.to_string());
                        j.stats.texture = get("texture").as_f64();
                    }
                    if let Some(q) = json.get("mesh_quality") {
                        j.stats.patches = q
                            .get("patches")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize);
                        j.stats.euler = q.get("euler").and_then(|v| v.as_i64());
                        j.stats.closed = q.get("closed").and_then(|v| v.as_bool());
                    }
                    j.report = Some(json);
                }
            }
            let stats = j.stats.clone();
            let summary = RunSummary {
                name: run_name,
                views: stats.views.unwrap_or(0),
                points: stats.points.unwrap_or(0),
                rmse: stats.rmse.unwrap_or(0.0),
                depth_median: stats.depth_median.unwrap_or(0.0),
                faces: stats.faces.unwrap_or(0),
                patches: stats.patches.unwrap_or(0),
                euler: stats.euler.unwrap_or(0),
                closed: stats.closed.unwrap_or(false),
                cost: stats.cost.clone().unwrap_or_else(|| "ncc".into()),
            };
            let mut hist = history_state.lock().unwrap();
            hist.push(summary);
            let excess = hist.len().saturating_sub(50);
            if excess > 0 {
                hist.drain(0..excess);
            }
            let _ = std::fs::write(&history_path, serde_json::to_string_pretty(&*hist).unwrap());
        } else {
            j.status = "error".into();
            let error_path = error::sibling_json(&out_path, "_error");
            if let Ok(text) = std::fs::read_to_string(&error_path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(stage) = json.get("stage").and_then(|v| v.as_u64()) {
                        j.stage = stage as usize;
                    }
                    j.error = Some(json);
                }
            }
            j.log.push(format!("pipeline exited with code {code:?}"));
        }
    });

    Ok(Json(serde_json::json!({ "started": true })))
}

/// Estimate the focal length from a printed Tag36h11 tag.
///
/// Uploads a short clip, runs the CLI's `--calibrate-only` mode and returns its
/// JSON estimate; the UI fills `fx/fy/cx/cy` with it. The clip is temporary.
async fn calibrate(
    State(app): State<App>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = app.dir.join(format!("calib-{stamp}.mp4"));
    let mut tag_size_m = 0.1_f64;
    let mut written = false;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        match field.name() {
            Some("video") => {
                let mut file = tokio::fs::File::create(&path)
                    .await
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                let mut field = field;
                while let Some(chunk) = field
                    .chunk()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
                {
                    tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
                        .await
                        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                }
                tokio::io::AsyncWriteExt::flush(&mut file)
                    .await
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                written = true;
            }
            Some("tag_size_m") => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
                tag_size_m = text
                    .trim()
                    .parse()
                    .map_err(|_| (StatusCode::BAD_REQUEST, "invalid tag size".into()))?;
            }
            _ => {}
        }
    }
    if !written {
        return Err((StatusCode::BAD_REQUEST, "no video uploaded".into()));
    }

    let exe =
        std::env::current_exe().map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let video = path.clone();
    let output = tokio::task::spawn_blocking(move || {
        Command::new(exe)
            .arg(&video)
            .args(["--estimate-focal", &tag_size_m.to_string()])
            .args(["--calibrate-only", "--frame-step", "5"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let _ = std::fs::remove_file(&path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err((
            StatusCode::BAD_REQUEST,
            // main() prints "error: <message>" first, then code + hints; the
            // first line is the actual failure, the last is a hint.
            stderr
                .lines()
                .next()
                .unwrap_or("calibration failed")
                .trim_start_matches("error: ")
                .to_string(),
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let estimate: serde_json::Value = serde_json::from_str(text.trim())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(estimate))
}

async fn mesh(
    State(app): State<App>,
    UrlPath(kind): UrlPath<String>,
) -> Result<Response, (StatusCode, String)> {
    let output = {
        let job = app.job.lock().unwrap();
        job.output
            .clone()
            .ok_or((StatusCode::NOT_FOUND, "no output yet".into()))?
    };
    let path = match kind.as_str() {
        "mesh" => output.clone(),
        "sparse" => sibling(&output, "_sparse"),
        "cloud" => sibling(&output, "_cloud"),
        "oriented" => sibling(&output, "_oriented"),
        _ => return Err((StatusCode::BAD_REQUEST, "unknown kind".into())),
    };
    let bytes = std::fs::read(&path)
        .map_err(|_| (StatusCode::NOT_FOUND, format!("missing {}", path.display())))?;
    Ok(Response::builder()
        .header("content-type", "application/octet-stream")
        .body(axum::body::Body::from(bytes))
        .unwrap())
}

fn sibling(output: &Path, suffix: &str) -> PathBuf {
    let stem = output
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "run".to_string());
    output.with_file_name(format!("{stem}{suffix}.ply"))
}

fn build_args(c: &Config) -> Vec<String> {
    let mut a: Vec<String> = Vec::new();
    let mut opt = |k: &str, v: String| {
        a.push(k.to_string());
        a.push(v);
    };
    opt("--cache-dir", "web_runs/cache".into());
    opt("--fx", c.fx.to_string());
    opt("--fy", c.fy.to_string());
    opt("--cx", c.cx.to_string());
    opt("--cy", c.cy.to_string());
    opt("--detector", c.detector.clone());
    opt("--frame-step", c.frame_step.to_string());
    opt("--n-features", c.n_features.to_string());
    opt("--match-window", c.match_window.to_string());
    opt("--ratio", c.ratio.to_string());
    opt("--geo-threshold", c.geo_threshold.to_string());
    opt("--geo-min-inliers", c.geo_min_inliers.to_string());
    opt("--wide-baseline", c.wide_baseline.to_string());
    opt("--dense-scale", c.dense_scale.to_string());
    opt("--dense-planes", c.dense_planes.to_string());
    opt("--dense-window", c.dense_window.to_string());
    opt("--dense-neighbors", c.dense_neighbors.to_string());
    opt("--ncc-min", c.ncc_min.to_string());
    opt("--lr-required", c.lr_required.to_string());
    opt("--cost", c.cost.clone());
    opt("--sgm-paths", c.sgm_paths.to_string());
    opt("--voxel-resolution", c.voxel_resolution.to_string());
    opt("--trunc-voxels", c.trunc_voxels.to_string());
    opt("--volume-smoothing", c.volume_smoothing.to_string());
    opt("--hole-fill-iterations", c.hole_fill_iterations.to_string());
    opt("--smooth-iterations", c.smooth_iterations.to_string());
    opt("--bilateral-iterations", c.bilateral_iterations.to_string());
    opt("--needle-aspect", c.needle_aspect.to_string());
    opt("--target-faces", c.target_faces.to_string());
    opt("--min-component-faces", c.min_component_faces.to_string());
    if c.scale_tag_m > 0.0 {
        opt("--scale-tag", c.scale_tag_m.to_string());
    }
    for (on, flag) in [
        (c.geo_verify, "--geo-verify"),
        (c.sprt, "--sprt"),
        (c.refine_intrinsics, "--refine-intrinsics"),
        (c.auto_config, "--auto-config"),
        (c.remove_plane, "--remove-plane"),
    ] {
        if on {
            a.push(flag.to_string());
        }
    }
    a
}

fn parse_line(line: &str, job: &mut Job) {
    if let Some(rest) = line.strip_prefix("PROGRESS ") {
        let mut parts = rest.split_whitespace();
        let (Some(stage), Some(done), Some(total)) = (parts.next(), parts.next(), parts.next())
        else {
            return;
        };
        if let (Ok(stage), Ok(done), Ok(total)) = (stage.parse(), done.parse(), total.parse()) {
            job.progress = Some(ProgressView { stage, done, total });
        }
        return;
    }
    if let Some(rest) = line.strip_prefix('[') {
        if let Some(slash) = rest.find("/9]") {
            if let Ok(n) = rest[..slash].trim().parse::<usize>() {
                job.stage = n;
            }
        }
    }
    if let Some(idx) = line.find("reconstructed ") {
        let rest = &line[idx + "reconstructed ".len()..];
        let parts: Vec<&str> = rest.split_whitespace().collect();
        if let Some(v) = parts.first() {
            job.stats.points = v.parse().ok();
        }
        if let Some(i) = rest.find("across ") {
            let tail = &rest[i + 7..];
            job.stats.views = tail.split_whitespace().next().and_then(|s| s.parse().ok());
        }
        if let Some(i) = rest.find("rmse ") {
            let tail = &rest[i + 5..];
            job.stats.rmse = tail.split_whitespace().next().and_then(|s| s.parse().ok());
        }
    }
    if let Some(idx) = line.find("depth vs sparse:") {
        let rest = &line[idx..];
        if let Some(i) = rest.find("median ") {
            let tail = &rest[i + 7..];
            job.stats.depth_median = tail.split('%').next().and_then(|s| s.trim().parse().ok());
        }
        if let Some(i) = rest.find("p90 ") {
            let tail = &rest[i + 4..];
            job.stats.depth_p90 = tail.split('%').next().and_then(|s| s.trim().parse().ok());
        }
    }
    if let Some(idx) = line.find("raw mesh:") {
        let rest = &line[idx..];
        if let Some(i) = rest.find(", ") {
            let after = &rest[i + 2..];
            job.stats.raw_faces = after.split_whitespace().next().and_then(|s| s.parse().ok());
        }
    }
    if let Some(idx) = line.find("wrote mesh (") {
        let rest = &line[idx..];
        if let Some(i) = rest.find(" faces, patches=") {
            let head = &rest[..i];
            if let Some(v) = head.split_whitespace().last() {
                job.stats.faces = v.parse().ok();
            }
            let tail = &rest[i + " faces, patches=".len()..];
            if let Some(v) = tail.split([',', ')']).next() {
                job.stats.patches = v.trim().parse().ok();
            }
        }
        if let Some(i) = rest.find("euler=") {
            let tail = &rest[i + 6..];
            job.stats.euler = tail
                .split([',', ')'])
                .next()
                .and_then(|s| s.trim().parse().ok());
        }
        if let Some(i) = rest.find("closed=") {
            let tail = &rest[i + 7..];
            job.stats.closed = tail.split([',', ')']).next().map(|s| s.trim() == "true");
        }
    }
    if let Some(idx) = line.find("[auto] texture score ") {
        let rest = &line[idx + "[auto] texture score ".len()..];
        let mut parts = rest.split_whitespace();
        job.stats.texture = parts.next().and_then(|s| s.parse().ok());
        if let Some(i) = rest.find("-> cost ") {
            let tail = &rest[i + "-> cost ".len()..];
            job.stats.cost = tail.split([',', ' ']).next().map(|s| s.to_string());
        }
    }
}
