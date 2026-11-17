"""Each jaw with its teeth in place, from the front, the side and three-quarter (checks/look.py shots), anatomical
orientation: for judging whether roots show through the bone.
  python scripts/jaw_views.py  ->  out/jaw-views.png"""
import sys
from pathlib import Path

import numpy as np, trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "checks"))
import look  # noqa: E402

look.SIZE = 600
COL = {"bone": (0.71, 0.51, 0.35), "tooth": (0.95, 0.94, 0.92), "filling": (0.17, 0.17, 0.17)}
rows = []
for jaw in ("upper-jaw", "lower-jaw"):
    V, T, C = [], [], []
    for p in sorted(Path("out").glob(f"{jaw}--*.ply")):
        m = trimesh.load(p)
        if jaw == "upper-jaw": m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        k = "filling" if "filling" in p.stem else "bone" if "bone" in p.stem else "tooth"
        T.append(np.asarray(m.faces) + sum(len(v) for v in V)); V.append(np.asarray(m.vertices)); C.append(np.tile(COL[k], (len(m.faces), 1)))
    V, T, C = np.vstack(V), np.vstack(T), np.vstack(C)
    c, d = (V.min(0) + V.max(0)) / 2, np.linalg.norm(V.max(0) - V.min(0))
    rows.append([look.shot(V, T, C, c + d * np.array(e)) for e in ((0, 1, 0.05), (-1, 0.3, 0.05), (1, 0.6, 0.3))])
W = max(s.shape[1] for r in rows for s in r); H = max(s.shape[0] for r in rows for s in r)
out = np.ones((len(rows) * H, 3 * W, 3))
for i, r in enumerate(rows):
    for j, s in enumerate(r): out[i * H:i * H + s.shape[0], j * W:j * W + s.shape[1]] = s
Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save("out/jaw-views.png")
