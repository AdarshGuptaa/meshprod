//! Camera focal-length calibration from a printed AprilTag.
//!
//! The tag is planar, so one view gives a board→image homography and Zhang's
//! orthonormality constraint recovers `f` ([`kornia_calib::estimate_focal`]).
//! Each sampled frame yields one estimate; the reported focal is the median and
//! the spread is a robust (MAD) dispersion, which catches a tag seen only at
//! shallow angles. The principal point is assumed to be the image centre —
//! a single tag cannot separate it from the focal.

use std::collections::HashMap;
use std::path::Path;

use kornia_algebra::Vec2F64;
use kornia_apriltag::{family::TagFamilyKind, AprilTagDecoder, DecodeTagsConfig};
use kornia_calib::{estimate_focal, TagObservation};
use kornia_image::Image;
use serde::Serialize;

use crate::{error::PipelineError, video};

/// Frames sampled across the clip; detections on these set the estimate.
const SAMPLE_FRAMES: usize = 40;

/// One calibration result.
#[derive(Debug, Clone, Serialize)]
pub struct FocalEstimate {
    /// Estimated focal length in pixels (`fx = fy`).
    pub fx: f64,
    /// Principal point assumed by the estimate (image centre X).
    pub cx: f64,
    /// Principal point assumed by the estimate (image centre Y).
    pub cy: f64,
    /// Frames where a tag was detected.
    pub detected: usize,
    /// Frames sampled.
    pub sampled: usize,
    /// Robust relative spread of the per-frame estimates, in percent.
    pub spread_pct: f64,
}

/// Decode `path`, detect a Tag36h11 tag and estimate the focal length.
///
/// Returns the estimate together with the decoded frames so the caller can run
/// the pipeline on them instead of decoding twice.
pub fn calibrate(
    path: &Path,
    frame_step: usize,
    tag_size_m: f64,
) -> Result<(FocalEstimate, video::VideoFrames), PipelineError> {
    if !tag_size_m.is_finite() || tag_size_m <= 0.0 {
        return Err(PipelineError::InvalidArgs(
            "--estimate-focal takes the tag's edge length in metres (e.g. 0.1)".into(),
        ));
    }
    let frames = video::read_frames(path, frame_step)
        .map_err(|e| PipelineError::VideoDecode(e.to_string()))?;
    let (_, gray) = &frames;
    let Some(first) = gray.first() else {
        return Err(PipelineError::NotEnoughFrames { got: 0 });
    };
    let size = first.size();
    let (cx, cy) = (size.width as f64 * 0.5, size.height as f64 * 0.5);

    let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11])
        .map_err(|e| PipelineError::CalibrationFailed(format!("decoder config: {e}")))?;
    let mut decoder = AprilTagDecoder::new(config, size)
        .map_err(|e| PipelineError::CalibrationFailed(format!("decoder setup: {e}")))?;

    let stride = (gray.len() / SAMPLE_FRAMES).max(1);
    let sampled = gray.iter().step_by(stride).count();
    let half = tag_size_m * 0.5;
    // Quad corner order is [TR, BR, BL, TL] (image y-down); use the same order
    // on the board so the homography is consistent.
    let board = [
        Vec2F64::new(half, -half),
        Vec2F64::new(half, half),
        Vec2F64::new(-half, half),
        Vec2F64::new(-half, -half),
    ];

    let mut focals = Vec::new();
    let mut detected = 0;
    for frame in gray.iter().step_by(stride) {
        let Ok(detections) = decoder.decode(frame) else {
            continue;
        };
        let Some(detection) = detections
            .iter()
            .max_by(|a, b| a.decision_margin.total_cmp(&b.decision_margin))
        else {
            continue;
        };
        detected += 1;
        let image: Vec<Vec2F64> = detection
            .quad
            .corners
            .iter()
            .map(|p| Vec2F64::new(p.x as f64, p.y as f64))
            .collect();
        if let Some(focal) = estimate_focal(&board, &image, cx, cy) {
            focals.push(focal);
        }
    }

    if focals.is_empty() {
        return Err(PipelineError::CalibrationFailed(format!(
            "no Tag36h11 tag found in {sampled} sampled frames"
        )));
    }
    let fx = median(&mut focals);
    let mad = median(&mut focals.iter().map(|f| (f - fx).abs()).collect::<Vec<_>>()) * 1.4826;
    Ok((
        FocalEstimate {
            fx,
            cx,
            cy,
            detected,
            sampled,
            spread_pct: if fx > 0.0 { 100.0 * mad / fx } else { 0.0 },
        },
        frames,
    ))
}

/// Detect every Tag36h11 tag across the decoded frames and collect each tag's
/// per-camera corner observations in aruco winding for metric scale anchoring.
///
/// The detector returns corners in `[TR, BR, BL, TL]` order (image y-down);
/// [`TagObservation`] wants aruco winding `[TL, TR, BR, BL]`. Each distinct tag
/// id becomes one observation carrying every camera that saw it, so
/// `kornia_calib::reconstruct` can pick the widest-baseline pair and triangulate
/// the corners against the known side length.
pub fn detect_tags_for_scale(gray: &[Image<u8, 1>]) -> Result<Vec<TagObservation>, PipelineError> {
    let Some(first) = gray.first() else {
        return Ok(Vec::new());
    };
    let size = first.size();
    let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11])
        .map_err(|e| PipelineError::CalibrationFailed(format!("decoder config: {e}")))?;
    let mut decoder = AprilTagDecoder::new(config, size)
        .map_err(|e| PipelineError::CalibrationFailed(format!("decoder setup: {e}")))?;

    let mut by_id: HashMap<u16, Vec<(usize, [Vec2F64; 4])>> = HashMap::new();
    for (camera, frame) in gray.iter().enumerate() {
        let Ok(detections) = decoder.decode(frame) else {
            continue;
        };
        for detection in detections {
            let corners = detection.quad.corners;
            // [TR, BR, BL, TL] -> aruco [TL, TR, BR, BL].
            let aruco = [
                Vec2F64::new(corners[3].x as f64, corners[3].y as f64),
                Vec2F64::new(corners[0].x as f64, corners[0].y as f64),
                Vec2F64::new(corners[1].x as f64, corners[1].y as f64),
                Vec2F64::new(corners[2].x as f64, corners[2].y as f64),
            ];
            by_id.entry(detection.id).or_default().push((camera, aruco));
        }
    }

    Ok(by_id
        .into_iter()
        .map(|(tag_id, per_camera)| TagObservation { tag_id, per_camera })
        .collect())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::median;

    #[test]
    fn median_handles_even_and_odd_counts() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&mut [7.0]), 7.0);
    }
}
