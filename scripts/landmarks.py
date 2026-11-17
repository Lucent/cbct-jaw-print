"""Anatomical landmarks for the hinge and stand, in RAS mm (+X patient left, +Y anterior, +Z up; the meshes'
frame): the midsagittal plane, the occlusal plane, the lower incisal point, the hinge axis by Bonwill's triangle
(each condyle BONWILL mm from the incisal point, INTERCONDYLAR apart) tilted BALKWILL degrees up from the
occlusal plane, the rami's cut faces at the scan's edge, and the upper jaw's flat top.
  python scripts/landmarks.py  ->  out/landmarks.json"""
import json
from pathlib import Path

import nibabel as nib, numpy as np
from scipy import ndimage as ndi

S = 0.1403
BONWILL, INTERCONDYLAR, BALKWILL = 100.0, 100.0, 26.0
ref = np.asarray(nib.load("refined2.nii.gz").dataobj).astype(np.int16)
ras = lambda p: np.asarray(p, float) * S * np.array([-1.0, -1.0, 1.0])

cen = {t: ras(ndi.center_of_mass(ref == t)) for t in list(range(11, 19)) + list(range(21, 29)) + list(range(31, 39)) + list(range(41, 49))}
pairs = [(a, a + 10) for a in range(11, 19)] + [(a + 10, a) for a in range(31, 39)]   # (patient right, left): 1x/4x are right
mid0 = np.mean([(cen[a] + cen[b]) / 2 for a, b in pairs], 0)
lat = np.mean([cen[b] - cen[a] for a, b in pairs], 0); lat /= np.linalg.norm(lat)        # toward patient left
if lat[0] < 0: lat = -lat

def top(t):
    p = np.argwhere(ref == t); return ras(p[p[:, 2] >= p[:, 2].max() - 2].mean(0))
tips = np.array([top(t) for t in range(31, 49) if t % 10 and (ref == t).any()])
c = tips.mean(0); n = np.linalg.svd(tips - c)[2][2]
if n[2] < 0: n = -n                                                                   # occlusal normal, up
lat = lat - (lat @ n) * n; lat /= np.linalg.norm(lat)                                # level with the occlusal plane
incisal = (top(31) + top(41)) / 2
fwd = np.cross(lat, n); fwd /= np.linalg.norm(fwd)                                  # anterior, in the occlusal plane
if fwd[1] < 0: fwd = -fwd
half = INTERCONDYLAR / 2
reach = np.sqrt(BONWILL ** 2 - half ** 2)                                           # incisal point to the axis midpoint
b = np.radians(BALKWILL)
axis_mid = incisal - fwd * reach * np.cos(b) + n * reach * np.sin(b)
axis_mid = axis_mid - lat * ((axis_mid - mid0) @ lat)                               # on the midline
condyles = [axis_mid - lat * half, axis_mid + lat * half]

# rami: lower-jaw bone at the scan's cylinder edge, behind the last molars
lowb = ref == 1
xg, yg = np.mgrid[:ref.shape[0], :ref.shape[1]]
rr = np.sqrt((xg - (ref.shape[0] - 1) / 2) ** 2 + (yg - (ref.shape[1] - 1) / 2) ** 2)
edge = (rr > 296)[..., None] & lowb
rami = {}
for side, sgn in (("right", -1), ("left", 1)):
    pts = ras(np.argwhere(edge))
    pts = pts[np.sign((pts - mid0) @ lat) == sgn]
    pts = pts[pts[:, 1] < min(cen[37 if sgn > 0 else 47][1], cen[38 if sgn > 0 else 48][1])]   # behind the molars
    if len(pts): rami[side] = {"centroid": pts.mean(0).round(2).tolist(), "z_range": [round(float(pts[:, 2].min()), 1), round(float(pts[:, 2].max()), 1)], "n": int(len(pts))}

upb = ref == 2
ztop = max(z for z in range(ref.shape[2]) if upb[:, :, z].any())
foot = ras(np.column_stack([np.argwhere(upb[:, :, ztop - 2]), np.full(int(upb[:, :, ztop - 2].sum()), ztop - 2)]))
out = {"midline_point": mid0.round(2).tolist(), "lateral": lat.round(4).tolist(), "occlusal_normal": n.round(4).tolist(),
       "anterior": fwd.round(4).tolist(), "incisal_point": incisal.round(2).tolist(),
       "hinge_axis_mid": axis_mid.round(2).tolist(), "condyles": [c.round(2).tolist() for c in condyles],
       "rami_cut": rami, "upper_top_z": round(ztop * S, 2),
       "upper_top_footprint_bbox": [foot.min(0).round(1).tolist(), foot.max(0).round(1).tolist()],
       "lower_bottom_z": round(float(ras(np.argwhere(lowb))[:, 2].min()), 2)}
Path("out").mkdir(exist_ok=True); Path("out/landmarks.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
