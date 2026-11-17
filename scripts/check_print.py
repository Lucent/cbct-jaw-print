"""Printability of out/<piece>--*.ply in print orientation, with checks/ (supports.py,
thin.py) at millimetres, 1:1:
  floats  (start in the air: a slicer must support them), leans (overhangs past 30 deg), bridges (and their long
  parts), per layer; and the area of bone thinner than one and two nozzle lines.
  python scripts/check_print.py lower-jaw upper-jaw  ->  out/check.json and printed totals
"""
import json, sys
from pathlib import Path

import manifold3d as m3
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "checks"))
import supports as SUP  # noqa: E402
import thin  # noqa: E402

LAYER = 0.12
SUP.SCALE, SUP.LAYER, SUP.LINE, SUP.BRIDGE, SUP.SPECK = 1.0, LAYER, 0.4, 10.0, 0.01
SUP.STEP = LAYER / np.tan(np.radians(SUP.STEEP))
OUT = Path("out")


def manifold(m):
    return m3.Manifold(m3.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))


res = {}
for piece in sys.argv[1:]:
    parts = {p.name.split("--")[1][:-4]: trimesh.load(p) for p in sorted(OUT.glob(f"{piece}--*.ply"))}
    z0 = min(m.bounds[0][2] for m in parts.values())
    ms = {k: manifold(m.copy().apply_translation([0, 0, -z0])) for k, m in parts.items()}
    bad = {k: str(v.status()) for k, v in ms.items() if v.status() != m3.Error.NoError}
    whole = m3.Manifold.batch_boolean([v for v in ms.values() if v.status() == m3.Error.NoError], m3.OpType.Add)
    area, _, places, _ = SUP.supports(whole)
    tri, _, t = thin.thickness(ms["bone"])
    a = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    teeth = {k: v for k, v in ms.items() if k.startswith("tooth") and "filling" not in k}
    to_bone = {k: round(v.min_gap(ms["bone"], 1.0), 3) for k, v in teeth.items()}
    names = sorted(teeth)
    to_tooth = min(((teeth[a].min_gap(teeth[b], 1.0), a, b) for i, a in enumerate(names) for b in names[i + 1:]), default=(1.0, "-", "-"))
    res[piece] = {"tooth_to_bone_gap_min": min(to_bone.values()), "tooth_to_bone_worst": sorted(to_bone.items(), key=lambda kv: kv[1])[:4],
                  "tooth_to_tooth_gap_min": [round(to_tooth[0], 3), to_tooth[1], to_tooth[2]],
                  "not_manifold": bad, "support_mm2": {k: round(v, 1) for k, v in area.items()},
                  "largest": places[:8],
                  "bone_mm2_thinner_than": {"0.4": round(float(a[t < 0.4].sum()), 1), "0.8": round(float(a[t < 0.8].sum()), 1)},
                  "bone_mm2": round(float(a.sum()), 1)}
    print(piece, json.dumps(res[piece], indent=1))
(OUT / f"check-{'-'.join(sys.argv[1:])}.json").write_text(json.dumps(res, indent=1))
