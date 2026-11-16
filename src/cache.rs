//! On-disk cache for the sparse stage: decoded frames and multi-view tracks.
//!
//! Decoding a clip is the single most expensive step of a run, and the web UI
//! re-runs the pipeline every time a dense/mesh slider moves. Both artifacts
//! are keyed by the video's SHA-256 and by the parameters that produced them,
//! so a stale entry is never read: change `--frame-step` and the frames miss;
//! change a matching knob and the tracks miss.
//!
//! Frames are stored raw (gray then RGB) because that is what the dense stage
//! consumes; a long clip can therefore use a few hundred MB on disk.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use kornia_algebra::Vec2F64;
use kornia_calib::FeatureTrack;
use kornia_image::{Image, ImageSize};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Magic + version of the frames cache file.
const FRAMES_MAGIC: &[u8; 5] = b"MPFR1";

/// Short, stable hex digest of a list of parameters.
pub fn key(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.as_bytes());
        hasher.update([0u8]);
    }
    let digest = format!("{:x}", hasher.finalize());
    digest[..16].to_string()
}

/// Decoded frames: RGB (colours, texturing) and matching grayscale.
pub type CachedVideoFrames = (Vec<Image<u8, 3>>, Vec<Image<u8, 1>>);

/// Cache handle for one video at one frame step.
pub struct SparseCache {
    dir: PathBuf,
    video: String,
    frame_step: usize,
    tracks_key: String,
}

impl SparseCache {
    /// Create a handle. `video_hash` is the hex SHA-256 of the video file;
    /// `tracks_key` hashes the feature/matching parameters.
    pub fn new(dir: PathBuf, video_hash: &str, frame_step: usize, tracks_key: String) -> Self {
        let short = video_hash.chars().take(16).collect();
        Self {
            dir,
            video: short,
            frame_step,
            tracks_key,
        }
    }

    fn frames_path(&self) -> PathBuf {
        self.dir
            .join(format!("frames-{}-{}.bin", self.video, self.frame_step))
    }

    fn tracks_path(&self) -> PathBuf {
        self.dir.join(format!(
            "tracks-{}-{}-{}.json",
            self.video, self.frame_step, self.tracks_key
        ))
    }

    /// Load decoded frames, if a valid cache entry exists.
    pub fn load_frames(&self) -> Option<CachedVideoFrames> {
        let path = self.frames_path();
        let file = File::open(path).ok()?;
        let mut reader = BufReader::new(file);
        let mut magic = [0u8; 5];
        reader.read_exact(&mut magic).ok()?;
        if &magic != FRAMES_MAGIC {
            return None;
        }
        let width = read_u32(&mut reader)? as usize;
        let height = read_u32(&mut reader)? as usize;
        let count = read_u32(&mut reader)? as usize;
        // Guard against a corrupt header triggering a huge allocation.
        if width == 0
            || height == 0
            || count == 0
            || width > 8192
            || height > 8192
            || count > 20_000
        {
            return None;
        }
        let size = ImageSize { width, height };
        let plane = width * height;

        // Layout: all gray frames (plane bytes each) followed by all RGB frames
        // (plane * 3 bytes each) — the mirror of `store_frames`.
        let mut gray = Vec::with_capacity(count);
        for _ in 0..count {
            let mut buffer = vec![0u8; plane];
            reader.read_exact(&mut buffer).ok()?;
            gray.push(Image::new(size, buffer).ok()?);
        }
        let mut rgb = Vec::with_capacity(count);
        for _ in 0..count {
            let mut buffer = vec![0u8; plane * 3];
            reader.read_exact(&mut buffer).ok()?;
            rgb.push(Image::new(size, buffer).ok()?);
        }
        Some((rgb, gray))
    }

    /// Persist decoded frames (atomically: a partial write never becomes a hit).
    pub fn store_frames(&self, rgb: &[Image<u8, 3>], gray: &[Image<u8, 1>]) -> std::io::Result<()> {
        let path = self.frames_path();
        fs::create_dir_all(&self.dir)?;
        let tmp = path.with_extension("bin.tmp");
        {
            let mut writer = BufWriter::new(File::create(&tmp)?);
            writer.write_all(FRAMES_MAGIC)?;
            let (width, height) = gray
                .first()
                .map(|f| (f.width(), f.height()))
                .unwrap_or((0, 0));
            writer.write_all(&(width as u32).to_le_bytes())?;
            writer.write_all(&(height as u32).to_le_bytes())?;
            writer.write_all(&(gray.len() as u32).to_le_bytes())?;
            for frame in gray {
                writer.write_all(frame.as_slice())?;
            }
            for frame in rgb {
                writer.write_all(frame.as_slice())?;
            }
            writer.flush()?;
        }
        fs::rename(&tmp, &path)
    }

    /// Load multi-view tracks, if a valid cache entry exists.
    pub fn load_tracks(&self) -> Option<Vec<FeatureTrack>> {
        let text = fs::read_to_string(self.tracks_path()).ok()?;
        let file: TracksFile = serde_json::from_str(&text).ok()?;
        Some(
            file.tracks
                .into_iter()
                .map(|track| FeatureTrack {
                    obs: track
                        .obs
                        .into_iter()
                        .map(|(camera, [x, y])| {
                            (camera, Vec2F64::new(f64::from_bits(x), f64::from_bits(y)))
                        })
                        .collect(),
                })
                .collect(),
        )
    }

    /// Persist multi-view tracks.
    pub fn store_tracks(&self, tracks: &[FeatureTrack]) -> std::io::Result<()> {
        let path = self.tracks_path();
        fs::create_dir_all(&self.dir)?;
        let file = TracksFile {
            tracks: tracks
                .iter()
                .map(|track| TrackRecord {
                    obs: track
                        .obs
                        .iter()
                        .map(|&(camera, pixel)| (camera, [pixel.x.to_bits(), pixel.y.to_bits()]))
                        .collect(),
                })
                .collect(),
        };
        let text = serde_json::to_string(&file).map_err(std::io::Error::other)?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, text)?;
        fs::rename(&tmp, &path)
    }
}

#[derive(Serialize, Deserialize)]
struct TracksFile {
    tracks: Vec<TrackRecord>,
}

#[derive(Serialize, Deserialize)]
struct TrackRecord {
    /// Pixel coordinates as raw `f64` bits. `serde_json` parses decimal floats
    /// off by one ULP for f32-promoted coordinates, which silently perturbs the
    /// whole reconstruction on a cache hit; bits round-trip exactly.
    obs: Vec<(usize, [u64; 2])>,
}

fn read_u32(reader: &mut impl Read) -> Option<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).ok()?;
    Some(u32::from_le_bytes(bytes))
}

/// Hash a file with SHA-256, returning lowercase hex.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    use std::io::Read as _;
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: usize, height: usize, seed: u8) -> Image<u8, 3> {
        let mut data = vec![0u8; width * height * 3];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_add(seed);
        }
        Image::new(ImageSize { width, height }, data).unwrap()
    }

    #[test]
    fn frames_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cache = SparseCache::new(dir.path().to_path_buf(), "abc", 5, "cfg".into());
        // Three frames so an interleaved-vs-blocked layout mismatch cannot
        // accidentally align on the final frame (a two-frame case masked the bug).
        let rgb = vec![frame(4, 3, 0), frame(4, 3, 7), frame(4, 3, 13)];
        let gray: Vec<Image<u8, 1>> = rgb
            .iter()
            .map(|f| {
                let data = f.as_slice().iter().step_by(3).copied().collect::<Vec<_>>();
                Image::new(f.size(), data).unwrap()
            })
            .collect();
        cache.store_frames(&rgb, &gray).unwrap();
        let (rgb2, gray2) = cache.load_frames().unwrap();
        assert_eq!(rgb2.len(), 3);
        assert_eq!(gray2.len(), 3);
        for i in 0..3 {
            assert_eq!(rgb2[i].as_slice(), rgb[i].as_slice());
            assert_eq!(gray2[i].as_slice(), gray[i].as_slice());
        }
    }

    #[test]
    fn tracks_roundtrip_and_keying() {
        let dir = tempfile::tempdir().unwrap();
        let cache = SparseCache::new(dir.path().to_path_buf(), "abc", 5, "cfg".into());
        let tracks = vec![FeatureTrack {
            obs: vec![
                (0, Vec2F64::new(1.5, 2.5)),
                // f32-promoted coordinate: `serde_json` decimal round-trip is
                // off by 1 ULP here, so this guards the bit-pattern encoding.
                (3, Vec2F64::new(42.525307f32 as f64, 350.49282f32 as f64)),
            ],
        }];
        cache.store_tracks(&tracks).unwrap();
        let loaded = cache.load_tracks().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].obs.len(), 2);
        assert_eq!(loaded[0].obs[1].0, 3);
        assert_eq!(loaded[0].obs[1].1.x, 42.525307f32 as f64);
        assert_eq!(loaded[0].obs[1].1.y, 350.49282f32 as f64);

        // A different matching config must not hit the same file.
        let other = SparseCache::new(dir.path().to_path_buf(), "abc", 5, "other".into());
        assert!(other.load_tracks().is_none());
    }

    #[test]
    fn missing_cache_is_a_miss_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let cache = SparseCache::new(dir.path().to_path_buf(), "gone", 1, "cfg".into());
        assert!(cache.load_frames().is_none());
        assert!(cache.load_tracks().is_none());
    }
}
