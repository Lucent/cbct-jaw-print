"""Best print orientation for each stand part: the six axis-aligned ones, scored by the support check (checks/supports.py)
(floats, overhang leans, bridges), least support first.  python scripts/stand_orient.py  ->  printed table"""
import sys
from pathlib import Path
import manifold3d as m3, numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "checks"))
import supports as SUP  # noqa: E402
SUP.SCALE, SUP.LAYER, SUP.LINE, SUP.BRIDGE, SUP.SPECK = 1.0, 0.3, 0.4, 10.0, 0.01
SUP.STEP = SUP.LAYER / np.tan(np.radians(SUP.STEEP))
ROT = {"as modelled": np.eye(3), "upside down": np.diag([1.0, -1, -1]),
       "on +x": trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])[:3, :3], "on -x": trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0])[:3, :3],
       "on +y": trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])[:3, :3], "on -y": trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])[:3, :3]}
for p in [] if __name__ != "__main__" else sorted(Path("out").glob(f"stand-{sys.argv[1] if len(sys.argv) > 1 else '*'}--*.ply")):
    if "nameplate" in p.name: continue
    m = trimesh.load(p); res = []
    for k, R in ROT.items():
        v = np.asarray(m.vertices) @ R.T; v[:, 2] -= v[:, 2].min()
        M = m3.Manifold(m3.Mesh(vert_properties=v.astype(np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
        area, _, _, _ = SUP.supports(M)
        res.append((area["float"] + area["lean"] + area["bridge"], k, {a: round(b) for a, b in area.items()}))
    res.sort(key=lambda r: r[0])
    print(p.stem, "| best:", res[0][1], res[0][2], "| next:", res[1][1], res[1][2])
