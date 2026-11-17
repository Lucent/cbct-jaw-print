"""Full-resolution (0.1403 mm) anatomy from the scan: real bone surface + separated teeth.

Bone: hysteresis threshold (weak 68 / strong 85 on sigma-1 smoothed data; soft tissue is 54.5 +- 7.5),
light opening to drop noise fuzz, cut cleanly just inside the scan cylinder and end slices.
Marrow is filled only inside the model's jawbone labels (never sinuses or nasal air) and, below the
model's truncated lower-jaw label, per axial slice (the mandible body has no air spaces).
Teeth: crowns follow the scan's half-max surface at full resolution; roots (no visible ligament gap
against bone) are the model label grown by the crown-calibrated offset. Pulp is filled; contacts go
to the nearest tooth.

Output refined2.nii.gz: 1 lower jaw, 2 upper jaw, 3/4 inferior alveolar canals, 11-48 teeth (FDI).
"""
import time
import nibabel as nib, numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.segmentation import expand_labels

WEAK, STRONG = 68, 85
GROW = 1                        # voxels the model teeth are grown (calibrated on crowns, calib.py)
HALF_MAX = 77.3                 # crown edge: midway between enamel/dentin (100) and soft tissue (54.5)
FOV_R, Z_TRIM = 300, 3          # clean cut faces: cylinder radius (of 314) and end slices dropped
ORDER = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28,
         31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
FDI = {11 + i: n for i, n in enumerate(ORDER)}          # model label -> FDI

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
def bbox(mask, pad, shape):
    sl = ndi.find_objects(mask.astype(np.int8))[0]
    return tuple(slice(max(0, s.start - pad), min(n, s.stop + pad)) for s, n in zip(sl, shape))

img = nib.load("jaw.nii.gz")
raw = np.asarray(img.dataobj).astype(np.float32)
lab = np.asarray(nib.load("seg_teeth.nii.gz").dataobj).astype(np.int16)
sh = raw.shape
sm1 = ndi.gaussian_filter(raw, 1.0)
sm05 = ndi.gaussian_filter(raw, 0.5)
log("loaded")

teeth_lab = np.isin(lab, list(FDI))
jaw_lab = (lab == 1) | (lab == 2)
air_lab = np.isin(lab, [5, 6, 7])
x, y = np.mgrid[:sh[0], :sh[1]]
fov = (((x - (sh[0] - 1) / 2) ** 2 + (y - (sh[1] - 1) / 2) ** 2) < FOV_R ** 2)[..., None] & \
      (np.arange(sh[2]) >= Z_TRIM)[None, None, :] & (np.arange(sh[2]) < sh[2] - Z_TRIM)[None, None, :]

# --- hard tissue as it appears in the scan. Bone is grown from bone seeds with the teeth masked out,
# so faint noise touching only a crown can't join; teeth (bright) are added back separately.
# Keep only pieces that overlap the jaw/teeth labels (drops hyoid/spine; keeps the lower-jaw strip
# the model truncated, which a distance limit would cut off).
teeth_zone = ndi.binary_dilation(teeth_lab, iterations=2) & (sm1 > WEAK)
# spatial prior: bone lies within ~2 mm of the model's jawbone labels; below the model's truncated
# lower-jaw label, within the lower jaw's outline carried straight down
z0 = np.nonzero((lab == 1).any((0, 1)))[0].min()
foot = ndi.binary_dilation((lab[:, :, z0:z0 + 25] == 1).any(2), iterations=25)
near = ndi.binary_dilation(jaw_lab, iterations=14)
near[:, :, :z0 + 10] |= foot[..., None]
hard = apply_hysteresis_threshold(np.where(teeth_zone | ~near, 0, sm1), WEAK, STRONG) & fov
hard = ndi.binary_opening(hard, iterations=1) | ((sm1 > STRONG) & fov & near & ~teeth_zone)
hard |= teeth_zone & fov
cc, n = ndi.label(hard)
ov = np.bincount(cc[jaw_lab | teeth_lab], minlength=n + 1); ov[0] = 0
hard = ov[cc] > 1000; del cc
log("hard", hard.sum(), "comps kept", int((ov > 1000).sum()), "of", n)

# --- marrow fill
interior = ndi.binary_erosion(jaw_lab, iterations=2) & ~air_lab & fov
z_lower_label = np.nonzero((lab == 1).any((0, 1)))[0].min()
below = np.zeros(sh, bool)
closed = ndi.binary_closing(hard, iterations=2)
for z in range(Z_TRIM, z_lower_label + 15):          # mandible body below the model's truncated label
    below[:, :, z] = ndi.binary_fill_holes(closed[:, :, z])
tissue = hard | interior | (below & fov)
tissue &= ~(teeth_zone & ~hard)
log("filled; lower-label bottom z", z_lower_label)

# --- which jaw owns each hard voxel: nearest model jawbone label
_, idx = ndi.distance_transform_edt(~jaw_lab, return_indices=True)
owner = lab[idx[0], idx[1], idx[2]]; del idx
log("owner map")

# --- teeth: crowns from the scan's half-max surface, roots from the model label grown by the
# crown-calibrated offset (calib.py: model teeth sit ~1 voxel inside the half-max edge; roots show no edge)
bb = bbox(teeth_lab, 20, sh)
labb, smb = lab[bb], sm05[bb]
model = np.zeros(labb.shape, np.int16)
for l, f in FDI.items():
    m = ndi.binary_fill_holes(labb == l)                  # pulp is part of the solid tooth
    model[m & (model == 0)] = f
grown = expand_labels(model, GROW)                         # calibrated root surface
reach = expand_labels(model, 7)                            # crowns may follow the scan up to ~1 mm out (model clips cusp tips)
crown_zone = ~ndi.binary_dilation(tissue[bb] & ~ndi.binary_dilation(model > 0, iterations=2), iterations=3)
crown = (smb > HALF_MAX) & crown_zone & (reach > 0)
out_b = np.where(crown_zone, np.where(crown, reach, 0), grown).astype(np.int16)
for f in FDI.values():
    tb = bbox(out_b == f, 3, out_b.shape) if (out_b == f).any() else None
    if tb is None: continue
    t = out_b[tb] == f
    t = ndi.binary_fill_holes(ndi.binary_closing(t, iterations=1))
    c2, k = ndi.label(t)
    if k > 1: t = c2 == (np.argmax(np.bincount(c2.ravel())[1:]) + 1)
    view = out_b[tb]; view[(view == f) & ~t] = 0; view[t & (view == 0)] = f
log("teeth")
out = np.zeros(sh, np.int16); out[bb] = out_b
teeth = out > 0
for b in (1, 2):
    bone = tissue & ~teeth & (owner == b)
    c2, k = ndi.label(bone); sz = np.bincount(c2.ravel()); sz[0] = 0
    bone = sz[c2] > 0.002 * sz.max()                       # drop crumbs, keep real fragments (e.g. thin walls)
    out[bone & (out == 0)] = b
    log("bone", b, "comps", k, "kept", int((sz > 0.002 * sz.max()).sum()))
for c in (3, 4):
    out[(lab == c) & (out == 1)] = c                       # canals: tunnels through the lower jaw

nib.save(nib.Nifti1Image(out, img.affine), "refined2.nii.gz")
vox = 0.1403 ** 3
for l, f in FDI.items():
    print(f"  tooth {f}: model {np.count_nonzero(lab == l) * vox:6.0f} mm3  refined {np.count_nonzero(out == f) * vox:6.0f} mm3")
log("saved refined2.nii.gz")
