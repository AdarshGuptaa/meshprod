# Preset matrix (measured)

How the four Studio presets behave on the reference captures. Each preset is a
**complete** config (it replaces, not merges); `fast`/`balanced`/`quality` pin
explicit dense parameters and therefore run with `auto_config` **off**, while
`auto` lets the pipeline pick dense parameters from the measured frame texture.

Reproduce (values are the UI presets; intrinsics from the decoded frame):

```bash
sfm2mesh <video> --output out.ply \
  --fx <0.7*max(w,h)> --fy <same> --cx <w/2> --cy <h/2> \
  --detector sift --n-features 500 --match-window 5 --ratio 0.8 \
  --geo-verify --geo-threshold 3.0 --geo-min-inliers 15 \
  --sprt --cost ncc --dense-window 5 --dense-neighbors 3 --ncc-min 0.4 \
  <preset flags below>
```

| Preset | frame\_step | dense\_scale | dense\_planes | voxel\_res | target\_faces | auto\_config |
|---|---|---|---|---|---|---|
| fast | 20 | 0.35 | 64 | 192 | 20 000 | off |
| balanced | 10 | 0.5 | 96 | 256 | 100 000 | off |
| quality | 5 | 0.75 | 128 | 320 | 80 000 | off |
| auto | 10 | 0.5 | 96 | 256 | 100 000 | **on** |

## Results

### `swiss_knife.mp4` (1280×720, 628 frames — the golden capture)

| Preset | Registered | RMSE | Depth med | Faces | Wall clock |
|---|---|---|---|---|---|
| fast | 8 / 32 | 1.58 px | 2.0 % | 19 999 | **32 s** |
| balanced | 19 / 63 | 2.09 px | 3.9 % | 99 999 | 77 s |
| quality | 40 / 126 | 2.44 px | 3.3 % | 79 999 | 340 s |
| auto | 19 / 63 | 2.09 px | 3.9 % | 77 479 | 76 s |

All four complete. `auto` reproduces the historical golden result (≈22 views,
≈87k faces, 4 % depth); `fast` is ~10× cheaper and still hits its 20k-face
budget; `quality` is ~4× the `balanced` cost for more views but a smaller
simplified mesh.

### `remote.mp4` (854×480, 50 s — dark remote on a plain red table, known hard)

| Preset | Registered | RMSE | Depth med | Faces | Verdict |
|---|---|---|---|---|---|
| fast | 12 / 75 | 2.1 px | — | — | **fail: TSDF no surface** |
| balanced | 20 / 151 | 2.04 px | 2.8 % | 284 | ok (≈empty mesh) |
| quality | 35 / 301 | 2.15 px | **41.3 %** | 48 742 | ok, **self-assessment "low"** (0.12) |
| auto | 20 / 151 | 2.1 px | 2.6 % | — | **fail: TSDF no surface** |

This capture is the documented hard case: the object is dark and small in a
texture-poor scene. `fast`/`auto` cannot resolve a TSDF surface; `balanced`
registers but reconstructs almost no surface; `quality` builds a large mesh
from **41 % depth error**, which the self-assessment correctly flags as low
confidence (0.12) instead of reporting a good mesh. The failures are named
(`empty-mesh`) and carry hints.

## Takeaways

- The four presets are genuinely distinct (verified: `fast`/`quality` logs show
  no `[auto]` override and use their explicit `scale`/`planes`).
- `self-assessment` correctly demotes the garbage `quality` mesh on `remote`.
- Significantly improving `remote` needs the capture-side fixes (frame the
  object larger, add light/texture) and loop closure (UP-1), not preset tuning.
