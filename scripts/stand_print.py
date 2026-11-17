"""out/stand.3mf rewritten with each part in its best print orientation (from stand_orient.py's out/orient-*.txt),
on the bed (z from 0); the base keeps its nameplate inlay, if any.  python scripts/stand_print.py"""
import re, sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as PP  # noqa: E402
sys.argv = sys.argv[:1]
from stand_orient import ROT  # noqa: E402  (the same six orientations)
objs = []
for name in ("cradle", "arm_right", "arm_left", "post", "base"):
    key = re.search(r"best: (.+?) \{", Path(f"out/orient-{name}.txt").read_text()).group(1)
    R = ROT[key]; parts = []
    for p in sorted(Path("out").glob(f"stand-{name}--*.ply")):
        m = trimesh.load(p); m.vertices = np.asarray(m.vertices) @ R.T; parts.append((p.stem.split("--")[1], "wood" if "nameplate" in p.stem else "black", m))
    z0 = min(m.bounds[0][2] for _, _, m in parts)
    for _, _, m in parts: m.apply_translation([0, 0, -z0])
    objs.append((name.replace("_", " "), parts)); print(name, key)
PP.write_3mf(Path("out/stand.3mf"), objs)
