"""The scan's JPEG slices (FILE_0001.jpg ... in slice order, inferior to superior) as one 8-bit volume:
jaw_0p1403mm.nrrd, 0.1403 mm isotropic (the JFIF 181 DPI tag, checked against tooth sizes); axes x patient
right->left, y anterior->posterior, z inferior->superior (radiological, as the slices read).
  python ingest_jpeg.py SCAN_DIR  ->  jaw_0p1403mm.nrrd"""
import glob, sys
import numpy as np
from PIL import Image

S = 0.1403
fs = sorted(glob.glob(sys.argv[1].rstrip("/") + "/FILE_*.jpg"))
v = np.stack([np.asarray(Image.open(f)) for f in fs])           # (z, y, x)
hdr = (f"NRRD0004\ntype: uint8\ndimension: 3\nspace: left-posterior-superior\nsizes: {v.shape[2]} {v.shape[1]} {v.shape[0]}\n"
       f"space directions: ({S},0,0) (0,{S},0) (0,0,{S})\nkinds: domain domain domain\nendian: little\nencoding: raw\n"
       "space origin: (0,0,0)\n\n")
open("jaw_0p1403mm.nrrd", "wb").write(hdr.encode() + np.ascontiguousarray(v).tobytes())
print(len(fs), "slices", v.shape[::-1], "->", [round(n * S, 1) for n in v.shape[::-1]], "mm")
