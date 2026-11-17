"""Pictures of out/assembly.glb (checks/look.py's ray-cast shots): closed from four sides, and open OPEN degrees.
  python scripts/render_assembly.py  ->  out/assembly.png"""
import json, sys
from pathlib import Path

import numpy as np, trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "checks"))
import look  # noqa: E402

OPEN = 35.0
L = json.loads(Path("out/landmarks.json").read_text())
u, mid = np.array(L["lateral"]), np.array(L["hinge_axis_mid"])
scene = trimesh.load("out/assembly.glb")
def gather(opened):
    V, T, C = [], [], []
    for node in scene.graph.nodes_geometry:
        tf, g = scene.graph[node]
        m = scene.geometry[g].copy(); m.apply_transform(tf)
        if opened and (node.startswith("lower-jaw") or node.startswith("arm")):
            m.apply_transform(trimesh.transformations.rotation_matrix(-np.radians(OPEN), u, mid))
        T.append(np.asarray(m.faces) + sum(len(v) for v in V)); V.append(np.asarray(m.vertices))
        C.append(np.repeat(np.asarray(m.visual.face_colors)[:1, :3] / 255.0, len(m.faces), 0))
    return np.vstack(V), np.vstack(T), np.vstack(C)
look.SIZE = 650
shots = []
V, T, C = gather(False)
c, d = (V.min(0) + V.max(0)) / 2, np.linalg.norm(V.max(0) - V.min(0))
for e in ((-0.7, 1.0, 0.45), (-1.0, 0.0, 0.15), (0.6, -1.0, 0.5)):
    shots.append(look.shot(V, T, C, c + d * np.array(e)))
V, T, C = gather(True)
shots.append(look.shot(V, T, C, c + d * np.array((-1.0, 0.0, 0.15))))
W, H = max(s.shape[1] for s in shots), max(s.shape[0] for s in shots)
out = np.ones((2 * H, 2 * W, 3))
for i, s in enumerate(shots): out[(i // 2) * H:(i // 2) * H + s.shape[0], (i % 2) * W:(i % 2) * W + s.shape[1]] = s
Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save("out/assembly.png")
