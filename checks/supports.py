"""Where a slicer would want supports under a piece, found the way it finds them: the piece sliced into the
printer's layers, each held against the one under it. What of a layer has nothing under it, past what a layer may
step out (STEEP, the slicer's own threshold), is a float (touching nothing held in its layer: it starts in the air),
a lean (held along one side, or a strip narrower than a nozzle line: an overhang wanting support) or a bridge
(spanned between what holds it); of a bridge, what is further than half of BRIDGE from what holds it is long.
Lengths are in model units, SCALE printed mm each; callers set SCALE, LAYER, LINE, BRIDGE and SPECK."""
import manifold3d as m3
import numpy as np

SCALE = 10.0   # (printed mm per model unit; callers set it with the units below)
LAYER = 0.02   # (model units: a 0.2 mm layer)
STEEP = 30.0   # (degrees from level: a face steeper than this prints on the layer under it, Bambu Studio's default)
STEP = LAYER / np.tan(np.radians(STEEP))  # (so a layer may stand out this far past the one under it)
LINE = 0.04    # (a 0.4 mm nozzle's line)
BRIDGE = 1.0   # (10 mm printed, the span a bridge prints clean over)
SPECK = 0.005  # (narrower than this is the booleans' noise)
KINDS = ("float", "lean", "bridge", "long")


def plate(section, z):
    """A layer's region as a thin plate's triangles under the layer at z, to be seen from below."""
    mesh = m3.Manifold.extrude(section, 0.004).translate((0, 0, z - LAYER / 2 - 0.008)).to_mesh()
    return np.asarray(mesh.vert_properties, np.float64)[:, :3][np.asarray(mesh.tri_verts)]


def supports(m):
    """({kind: mm² printed}, {kind: triangles (n, 3, 3) of thin plates where they are}, [(mm², kind, (x, y, z))]
    largest first, {kind: [the regions, CrossSections]}) for a piece lying on its lowest face."""
    b = m.bounding_box()
    area, plates, places, regions = dict.fromkeys(KINDS, 0.0), {k: [] for k in KINDS}, [], {k: [] for k in KINDS}
    below = m.slice(b[2] + LAYER / 2)
    for z in np.arange(b[2] + 1.5 * LAYER, b[5], LAYER):
        layer = m.slice(z)
        hung = (layer - below.offset(STEP, m3.JoinType.Square)).offset(-SPECK, m3.JoinType.Square).offset(SPECK, m3.JoinType.Square)
        below = layer
        if hung.is_empty():
            continue
        held = layer - hung
        for u in hung.decompose():
            touch = u.offset(2 * SPECK, m3.JoinType.Square) ^ held
            if touch.is_empty():
                kind, bad = "float", u
            elif u.offset(-LINE / 2, m3.JoinType.Square).is_empty():
                kind, bad = "lean", u
            elif (u - touch.hull().offset(2 * SPECK, m3.JoinType.Square)).area() < 0.1 * u.area():
                kind, bad = "bridge", u
                far = u - touch.offset(BRIDGE / 2, m3.JoinType.Round)
                area["long"] += far.area() * SCALE ** 2
                if far.area() * SCALE ** 2 > 0.01:
                    plates["long"].append(plate(far, z))
                    regions["long"].append(far)
            else:
                kind, bad = "lean", u
            if bad.area() * SCALE ** 2 < 0.01:
                continue
            area[kind] += bad.area() * SCALE ** 2
            c = bad.bounds()
            places.append((round(bad.area() * SCALE ** 2, 2), kind, (round((c[0] + c[2]) / 2, 2), round((c[1] + c[3]) / 2, 2), round(float(z), 2))))
            plates[kind].append(plate(bad, z))
            regions[kind].append(bad)
    return ({k: round(v) for k, v in area.items()},
            {k: np.concatenate(v) if v else np.zeros((0, 3, 3)) for k, v in plates.items()}, sorted(places, reverse=True), regions)
