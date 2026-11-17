# cbct-jaw-print

A dental cone-beam CT to a life-size 3D print of both jaws: the bone in wood-coloured filament, every tooth in white, printed in its socket and snapped out, on a black hinged stand that opens on M4 screws. The bone is the scan's real surface, unidealised, ending where the scan ends. An AI model separates the teeth; crowns follow the scan's edges, roots the model's outline.

Not a medical device: for display and education, not diagnosis, treatment planning or surgical guides.

## Outputs (`out/`)

| File | |
|---|---|
| `lower-jaw.3mf`, `upper-jaw.3mf` | Bone and teeth as one coloured object, in print orientation (upper jaw upside down on its cut top) |
| `stand.3mf` | Cradle, two arms, post and base, each oriented for least support |
| `coupon.3mf` | One premolar at four detent strengths (labelled 10–40, hundredths of a mm) to tune the snap fit |
| `assembly.glb`, `assembly.png`, `jaw-views.png` | The assembly closed and open; each jaw from three sides |
| `check-*.json`, `report-*.json` | Overhangs, bridges, floating starts, thin walls; per-tooth gaps, detents, removal directions |

## Pipeline

| Step | Script | |
|---|---|---|
| Ingest | `ingest_jpeg.py`, `to_nifti.py` | Slice images to one RAS volume |
| Segment | TotalSegmentator `teeth` | FDI teeth 11–48, jawbones, nerve canals, sinuses |
| Refine | `scripts/refine2.py` | Full-resolution bone by hysteresis threshold; crowns at the scan's half-max edge, roots from the model grown by a crown-calibrated offset; marrow filled, canals kept |
| Landmarks | `scripts/landmarks.py` | Midline, occlusal plane, hinge axis by Bonwill's triangle and Balkwill's angle |
| Print parts | `scripts/print_parts.py` | Signed-distance fields; each tooth's removal direction; sockets as the tooth swept out plus a gap; detent bumps between neighbours; thin bone thickened; Taubin smoothing |
| Stand | `scripts/stand.py`, `stand_orient.py`, `stand_print.py` | Hinge and stand fitted to the scan's cut faces with crush-ribbed tenons and pins; print orientations |
| Check | `scripts/check_print.py`, `jaw_views.py`, `render_assembly.py` | Support analysis, thin walls, renders |

## Run

Linux, Python 3.12, an NVIDIA GPU (CUDA 12.8) for speed; without one SciPy takes over. About 25 minutes on an RTX 4090.

```sh
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
./reproduce.sh SCAN [SEG] [DIRS]
```

`SCAN` is a folder of slice images or an NRRD. Segmentation (GPU) and the removal-direction search are not bit-exact between runs; pass a saved `seg_teeth.nii.gz` and `out/dirs.json` to rebuild a run exactly, and check it with `python verify.py out MANIFEST` (`--write` makes a manifest). `NAMEPLATE="text"` adds a wood nameplate to the base.

TotalSegmentator sends usage statistics unless `"send_usage_stats": false` is set in `~/.totalsegmentator/config.json`.

## Print

Tuned on a Bambu Lab H2D in PLA: 0.12 mm layers, 3 walls, tree supports at 30° in the bone's filament, outer brim on the lower jaw.

Print the coupon first and set `DETENT` in `scripts/print_parts.py` to the strength that holds without a fight (teeth sit `GAP`, 0.15 mm, clear of their sockets). Hinge: 2× M4×20 socket cap screws, 2× M4 nylon-insert lock nuts, 4× M4 washers. Post: 2× M4×16 flat-head screws, 2× M4 hex nuts. Nothing is glued.

## Limits

Built on one scan; these still assume it:

- Input: JPEG slices named `FILE_*.jpg`, voxel size (0.1403 mm) and orientation fixed in code; no DICOM.
- Tissue thresholds are that scan's 8-bit grey levels (bone 68/85, crown edge 77.3), not derived per scan.
- Several sizes are in voxels; the scan's cylinder and flat top are typed in. The stand mounts on those cut faces, so it suits a cropped dental CBCT, not a full-skull CT.
- All 32 teeth are assumed present.
- Gap, detent and crush-rib fit are tuned to one printer and filament.

## Credits

[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (Wasserthal et al., *Radiology: AI* 2023, [doi:10.1148/ryai.230024](https://doi.org/10.1148/ryai.230024)); its `teeth` task (Apache-2.0) is trained on ToothFairy3 (Bolelli et al., [*Segmenting Maxillofacial Structures in CBCT Volumes*](https://openaccess.thecvf.com/content/CVPR2025/html/Bolelli_Segmenting_Maxillofacial_Structures_in_CBCT_Volumes_CVPR_2025_paper.html), CVPR 2025).

MIT licence.
