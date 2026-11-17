"""Is a rebuild the same print? Hashes each 3MF's mesh XML (the zip's own timestamps vary run to run) and each JSON
report (parsed) in OUT_DIR against MANIFEST.
  python verify.py OUT_DIR MANIFEST            ->  prints each mismatch, exits 1 if any
  python verify.py OUT_DIR MANIFEST --write    ->  writes MANIFEST from OUT_DIR"""
import hashlib, json, sys, zipfile
from pathlib import Path

out, man = Path(sys.argv[1]), Path(sys.argv[2])
def digest(f):
    data = zipfile.ZipFile(f).read("3D/3dmodel.model") if f.suffix == ".3mf" else json.dumps(json.loads(f.read_text()), sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()
if "--write" in sys.argv:
    man.write_text("".join(f"{digest(f)}  {f.name}\n" for f in sorted(out.glob("*.3mf")) + sorted(out.glob("*.json"))))
    sys.exit(0)
bad = 0
for line in man.read_text().splitlines():
    h, name = line.split("  ", 1)
    f = out / name
    ok = f.exists() and digest(f) == h
    if not ok: print("DIFFERS" if f.exists() else "MISSING", name); bad += 1
print(f"{'SAME' if not bad else 'NOT THE SAME'}: {bad} of {len(man.read_text().splitlines())} differ"); sys.exit(1 if bad else 0)
