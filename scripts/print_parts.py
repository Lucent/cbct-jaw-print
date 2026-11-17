"""Print geometry for the jaws, from refined2.nii.gz, as signed-distance fields at the scan's 0.1403 mm.

Each jaw prints as one object: its bone (wood) with every tooth (white) printed in place in its socket, to be
snapped out. A tooth's socket is the tooth swept along its own removal direction (the one, within CONE of its
long axis, that cuts least bone and clears its neighbours) plus GAP all round, so it pops straight out.
Neighbouring teeth are shaved CONTACT apart (same filament: touching would fuse). Bone thinner than ~0.85 mm is
thickened to print; the scan's top and cylinder edge are cut flat; the inferior alveolar canals are tunnels at
least CANAL_D wide for a 1.75 mm cord. FILL's fillings (none by default) print as black inlays.

  python scripts/print_parts.py [lower] [upper] [coupon]  -> out/lower-jaw.3mf, out/upper-jaw.3mf, out/coupon.3mf, out/report-*.json
  (each piece can run as its own process; JAW_OUT overrides out/)
"""
import json, os, sys, time
from pathlib import Path

import manifold3d as m3, nibabel as nib, numpy as np, trimesh
from skimage.measure import marching_cubes
from skimage.morphology import ball, skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import S, Z_TOP, FOV_R, UPPER, LOWER, rotation_to_z, write_3mf, place, text  # noqa: E402
from gpu import ndi, sweep as gpu_sweep  # noqa: E402  (scipy.ndimage's interface, on the GPU if there is one)

GAP = 0.15                      # mm, tooth to socket (the first coupon: every gap released; 0.15 the tightest)
CONTACT = 0.30                  # mm, tooth to tooth (same filament)
MARGIN = 0.08                   # mm added to the socket gap in the fields: simplification and the teeth's smoothing eat ~0.05
CONTACT_MARGIN = 0.12           # mm added at contacts: smoothing rounds the shaved faces back ~0.1 toward each other
RIM = 3.0                       # mm: within this of a socket, bone too thin to print is removed, not thickened
PLATE = 0.5                     # mm: least bone kept between a socket and the jaw's outer surface (roots are shaved for it)
REACH = 2.0                     # mm round a root within which the scan's hard-tissue outline is bone (below the crest)
WEAK, STRONG = 68, 85           # the outline: hysteresis on sigma-1 data, as refine2's bone, but with the teeth as seeds
SECTORS = 16                    # round a tooth, for its scalloped crest
BONE_R, TOOTH_R = 3, 2          # voxels: noise bumps narrower than ~2r are removed from bone (0.84 mm) and teeth (0.56 mm)
CREST_DROP = 1.5                # mm: no sector's crest lies further below the highest bone round its tooth
SHAVE_MAX = 1.0                 # mm of root at most shaved for the plate; beyond it the plate thins (reported)
COUPON_GAP = 0.15               # mm: the first coupon's tightest gap (every tooth still slid out freely)
COUPON_DETENTS = (0.10, 0.20, 0.30, 0.40)   # mm the bump reaches past the gap into the tooth's way out
COUPON_TOOTH = 45
DETENT = 0.25                   # mm interference for the jaws (the detent coupon: between its 20 and 30)
DETENT_R = 0.75                 # mm, bump radius (1.5 mm across)
DETENT_BACKING = 0.5            # least share of a bump's outer half already solid bone, or it is left off
DETENT_DEPTH = 1.5              # mm below the crest on its side: the bump's centre
CONE = 25.0                     # degrees a removal direction may lean from the tooth's long axis
CANAL_D = 2.8                   # mm, least canal tunnel diameter
THIN_R = 3                      # voxels: an opening of this radius finds bone thinner than ~0.85 mm
THICKEN_R = 2                   # voxels (0.28 mm each side) added to thin bone
OUT = Path(os.environ.get("JAW_OUT", "out"))

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def box(mask, pad, shape, padz=None):
    sl = ndi.find_objects(mask.astype(np.int8))[0]
    p = (pad, pad, pad if padz is None else padz)
    return tuple(slice(max(0, s.start - q), min(n, s.stop + q)) for s, q, n in zip(sl, p, shape))


def sdf(mask, sigma=0.7):
    """Signed distance in mm (negative inside), surface midway between voxels, lightly smoothed."""
    if not mask.any(): return np.full(mask.shape, 1e3, np.float32)
    d = np.where(mask, -(ndi.distance_transform_edt(mask) - 0.5), ndi.distance_transform_edt(~mask) - 0.5) * S
    return ndi.gaussian_filter(d.astype(np.float32), sigma)


def sweep(mask, d):
    """Every voxel at or beyond a voxel of mask in direction d (the volume the shape passes through leaving along d)."""
    R = rotation_to_z(d); c = (np.array(mask.shape) - 1) / 2
    n = int(np.ceil(np.linalg.norm(mask.shape))) + 2; co = (np.array([n, n, n]) - 1) / 2
    return gpu_sweep(mask, R, c, co, n)


def tooth_axis(tooth, crown_dir):
    pts = np.argwhere(tooth).astype(float); pts -= pts.mean(0)
    axis = np.linalg.svd(pts, full_matrices=False)[2][0]
    if axis @ crown_dir < 0: axis = -axis
    if np.degrees(np.arccos(np.clip(axis @ crown_dir, -1, 1))) > 40: axis = crown_dir     # flat molars: use the arch's
    return axis


def removal_direction(tooth, bone, others, crown_dir, band):
    """Within CONE of the tooth's long axis (oriented to its crown), the direction whose sweep cuts least bone (the
    scan's outline included, so the outer plate counts), hits no neighbour, and needs least root shaved to keep the
    plate (band: root too near the outer surface); searched on a 2x coarser grid.
    Returns (direction, bone voxels cut, neighbour voxels hit)."""
    ds = lambda m: m[::2, ::2, ::2]
    t, b, o, bd = ds(tooth), ds(bone), ds(others), ds(band)
    axis = tooth_axis(tooth, crown_dir)
    R = rotation_to_z(axis)
    best = None
    for tilt in (0, 8, 16, CONE):
        for az in (np.arange(0, 360, 45) if tilt else [0]):
            a, z = np.radians(az), np.radians(tilt)
            local = np.array([np.sin(z) * np.cos(a), np.sin(z) * np.sin(a), np.cos(z)])
            d = R.T @ local
            sw = sweep(t, d) & ~t
            cut, hit = int((sw & b).sum()), int((sw & o).sum())
            shave = int((sweep(bd, -d) & t).sum()) if bd.any() else 0
            score = cut + 50 * hit + shave
            if best is None or score < best[0]: best = (score, d, cut * 8, hit * 8)
    return best[1], best[2], best[3]


def mesh(field, offset, tol=0.02):
    """Marching cubes of field <= 0 (padded so the surface closes) in RAS mm, simplified by manifold3d within tol mm
    (which keeps it a closed manifold), wound outward."""
    f = np.pad(field, 1, constant_values=1e3)
    if f.min() >= 0: return None
    v, fc, _, _ = marching_cubes(f, 0.0)
    v = (v - 1 + np.array(offset)) * S * np.array([-1.0, -1.0, 1.0])   # voxel (L, P, S) -> RAS mm: a rotation
    t = trimesh.Trimesh(v, fc, process=True)
    M = m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
    if M.status() != m3.Error.NoError: raise RuntimeError(f"not manifold: {M.status()}")
    if M.volume() < 0:
        t.invert()
        M = m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
    M = M.simplify(tol)
    o = M.to_mesh()
    t = trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=False)
    return t


SMOOTH_TEETH, SMOOTH_BONE = (0, 0) if os.environ.get("NO_SMOOTH") else (15, 25)   # Taubin passes: the scan's noise texture out (95% of the surface moves < 0.13 / 0.19 mm)


def smooth(t, iterations, pin=None, crop=False):
    """Taubin smoothing (volume-preserving, no shrink) with vertices pinned: the cut faces (the scan's top plane and
    cylinder: the upper jaw's bed face, the rami's glue faces; a crop's box faces) and any passed in."""
    v0 = np.asarray(t.vertices, float).copy(); v = v0.copy()
    cyl = np.array([-(SH[0] - 1) / 2 * S, -(SH[1] - 1) / 2 * S])
    fixed = (np.abs(v[:, 2] - Z_TOP * S) < 0.03) | (np.abs(np.hypot(v[:, 0] - cyl[0], v[:, 1] - cyl[1]) - FOV_R * S) < 0.03)
    if crop:
        lo, hi = v.min(0), v.max(0); fixed |= ((v - lo) < 0.15).any(1) | ((hi - v) < 0.15).any(1)
    if pin is not None: fixed |= pin
    L = trimesh.smoothing.laplacian_calculation(t)
    for _ in range(iterations):
        for k in (0.5, -0.53):
            v = v + k * (L.dot(v) - v); v[fixed] = v0[fixed]
    return trimesh.Trimesh(v, np.asarray(t.faces), process=False)


# ---------------------------------------------------------------- inputs
img = nib.load("refined2.nii.gz")
ref = np.asarray(img.dataobj).astype(np.int16)
raw = np.asarray(nib.load("jaw.nii.gz").dataobj).astype(np.float32)
sm = ndi.gaussian_filter(raw, 0.8)
SM1 = ndi.gaussian_filter(raw, 1.0)
SH = ref.shape
log("loaded")

cen = {t: np.array(ndi.center_of_mass(ref == t)) for t in UPPER + LOWER}
FILL = {}                                                     # {tooth: filling mask} to print as black inlays


report = {"gap": GAP, "contact": CONTACT, "teeth": {}}


def denoise(mask, r, keep_sheets=300):
    """The scan's noise texture out of a mask: opening then closing with a ball of r voxels removes bumps and pits
    narrower than ~2r without shrinking larger shapes; thin parts the opening would delete are kept when they are
    large connected sheets (sinus walls, plates), since noise bumps are small isolated blobs."""
    opened = ndi.binary_opening(mask, ball(r))
    thin = mask & ~opened
    lab, n = ndi.label(thin)
    if n:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        opened |= sz[lab] >= keep_sheets
    return ndi.binary_closing(opened, ball(r))


def crest_field(mt, bone_t, dn):
    """(H, CR): each voxel's height along dn (mm, from the tooth's centre) and the crest height above it: the top of
    the bone round the tooth in SECTORS sectors, a sector whose bone is missing or far below the rest (the outer
    plate the scan can't resolve) interpolated from its neighbours, so the crest scallops like a real one."""
    c0 = np.argwhere(mt).mean(0)
    rel = np.indices(mt.shape, dtype=np.float32).reshape(3, -1).T - c0
    H = (rel @ dn * S).reshape(mt.shape).astype(np.float32)
    e1 = np.cross(dn, [1.0, 0, 0] if abs(dn[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(dn, e1)
    sec = (((np.arctan2(rel @ e2, rel @ e1) + np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS).reshape(mt.shape)
    ring = ndi.binary_dilation(mt, ball(8)) & ~ndi.binary_dilation(mt, ball(1)) & bone_t
    tops = np.full(SECTORS, np.nan)
    for k in range(SECTORS):
        sel = ring & (sec == k)
        if sel.any(): tops[k] = H[sel].max()
    ok = np.isfinite(tops)
    if not ok.any(): return H, np.full(mt.shape, -1e3, np.float32)
    k = np.arange(SECTORS)
    filled = np.interp(k, k[ok], tops[ok], period=SECTORS)
    filled = np.maximum(filled, np.nanmax(tops) - CREST_DROP)       # the outer plate the scan loses can't sink the crest
    return H, filled[sec].astype(np.float32)


def neighbours(t):
    """The teeth either side of t in its arch (FDI): mesial (the other central across the midline) and distal."""
    q, n = divmod(t, 10)
    out = [n - 1 + 10 * q if n > 1 else 10 * {1: 2, 2: 1, 3: 4, 4: 3}[q] + 1, t + 1 if n < 8 else None]
    return [x for x in out if x in cen]


def detents(t, tshape, T, H, CR, dn, d, gap, inter, origin):
    """Snap detents for tooth t: on its two interproximal walls (facing its neighbours, hidden in the septa), a
    bump of radius DETENT_R, centred DETENT_DEPTH below that side's crest, reaching `inter` mm into the way out of
    the tooth below it (what must pass the bump: the sweep of the tooth below the dimple), and a dimple in the
    tooth round each bump keeping the in-place gap. Returns (dimpled T, [bump centres, tb voxels], info)."""
    rd = DETENT_R + gap + MARGIN
    idx = np.indices(tshape.shape, dtype=np.float32)
    c0 = np.argwhere(tshape).mean(0)
    rel = np.moveaxis(idx, 0, -1) - c0
    sides = []
    for nb in neighbours(t):
        e = cen[nb] - (origin + c0); e -= (e @ dn) * dn; sides.append(e / np.linalg.norm(e))
    if len(sides) == 1: sides.append(-sides[0])                                # the last molar: its distal wall
    surf = tshape & ~ndi.binary_erosion(tshape)
    centres, info = [], []
    for e in sides:
        proj = rel @ e
        face = surf & (proj > 0.7 * proj[surf].max())
        hc = float(np.median(CR[face])) - DETENT_DEPTH
        below = tshape & (H < hc - rd)
        Sb = sdf(sweep(below, d))
        cand = (np.abs(Sb) < 0.1) & (np.abs(H - hc) < 0.1)
        if not cand.any(): info.append({"side": e.round(2).tolist(), "skipped": "no wall"}); continue
        q = np.argwhere(cand)[np.argmax(proj[cand])]
        g = np.array([np.gradient(Sb[tuple(slice(max(0, v - 2), v + 3) for v in q)], axis=a)[tuple(min(2, v) for v in q)] for a in range(3)])
        g -= (g @ dn) * dn; n = g / np.linalg.norm(g)
        qs = q - n * Sb[tuple(q)] / S                                          # onto the surface
        c = qs + n * (DETENT_R - inter) / S
        centres.append(c)
        info.append({"side": e.round(2).tolist(), "height_mm": round(hc, 2), "crest_mm": round(hc + DETENT_DEPTH, 2)})
    for c in centres:
        Dc = np.sqrt(((idx - c[:, None, None, None]) ** 2).sum(0)) * S
        T = np.maximum(T, rd - Dc)
    return T, centres, info


def build_jaw(jaw, teeth, gap, crop=None, detent=None):
    """(bone trimesh, [(tooth, white trimesh, black trimesh|None)]) for a jaw (1 lower, 2 upper) at this gap;
    crop: (box slices) keeps only that box and the teeth wholly in it."""
    upper = jaw == 2
    allm = np.isin(ref, [jaw, 3, 4] + teeth)
    bb = box(allm, 12, SH)
    if crop is not None:                                                        # work in the crop (plus room)
        bb = tuple(slice(max(a.start, c.start - 25), min(a.stop, c.stop + 25)) for a, c in zip(bb, crop))
    off = [s.start for s in bb]
    R = ref[bb]
    bone = denoise(R == jaw, BONE_R)
    toothmask = np.isin(R, teeth)
    bone_open = ndi.binary_opening(R == jaw, ball(2))                         # (the crest is measured without the gum-line whiskers)
    # thin bone thickened (never into the teeth: their sockets are carved after)
    opened = ndi.binary_opening(bone, ball(THIN_R))
    thin = bone & ~opened
    bone |= ndi.binary_dilation(thin, ball(THICKEN_R)) & ~ndi.binary_dilation(toothmask, ball(1))
    B = sdf(bone)
    # O: the scan's hard-tissue outline (bone, plate and root together), nothing opened away: over the roots this is
    # the jaw's real outer surface, which the scan does resolve
    smb = SM1[bb]
    prior = ndi.binary_dilation(np.isin(R, [jaw] + teeth), iterations=14)
    weak = (smb > WEAK) & prior
    lab, _ = ndi.label(weak)
    seeds = np.unique(lab[(smb > STRONG) & weak]); seeds = seeds[seeds > 0]
    hard = np.isin(lab, seeds) | toothmask | bone; del lab, weak, prior
    # only the true exterior counts as outside: the dark ligament band and marrow enclosed in the jaw are holes in
    # the threshold, not its outer surface (treating them as outer surface shaved roots on every side)
    out_lab, _ = ndi.label(~hard)
    edge = np.unique(np.concatenate([out_lab[0].ravel(), out_lab[-1].ravel(), out_lab[:, 0].ravel(), out_lab[:, -1].ravel(),
                                     out_lab[:, :, 0].ravel(), out_lab[:, :, -1].ravel()]))
    exterior = ~denoise(~np.isin(out_lab, edge[edge > 0]), BONE_R); del out_lab
    O = sdf(~exterior, 0.5)
    # teeth: shave contacts, find removal directions, carve sockets
    crown_dir = np.array([0, 0, -1.0]) if upper else np.array([0, 0, 1.0])
    W = np.full(B.shape, 1e3, np.float32)
    NUBS, WT = [], []
    PZ = np.full(B.shape, 1e3, np.float32)                                     # the outline round the roots, below their crests
    out = []
    keep = teeth
    if crop is not None:
        cb = tuple(slice(c.start - o, c.stop - o) for c, o in zip(crop, off))
        keep = [t for t in teeth if (R == t).any() and not ((R == t) & ~_inbox(R.shape, cb)).any()]
    for t in teeth:
        m = R == t
        if not m.any(): continue
        tb = box(m, 30, R.shape, padz=95)
        mt = denoise(m[tb], TOOTH_R, keep_sheets=10**9) | ndi.binary_erosion(m[tb], ball(TOOTH_R)); others = toothmask[tb] & ~m[tb]
        T = np.maximum(sdf(mt), (CONTACT + CONTACT_MARGIN) / 2 - sdf(others)) if others.any() else sdf(mt)
        tshape = T <= 0
        Ot = O[tb]
        near = sdf(mt) < REACH
        band_of = lambda Hh, CRr, sh: near & (Ot > -(gap + MARGIN + PLATE)) & (Hh < CRr)   # the plate's room: near the outer surface, below the crest
        if t not in DIRS:
            a0 = tooth_axis(tshape, crown_dir)
            H0, CR0 = crest_field(mt, bone_open[tb], a0)
            P0 = np.maximum(np.maximum(Ot, H0 - CR0), sdf(mt) - REACH) <= 0
            DIRS[t] = removal_direction(tshape, bone[tb] | P0, others, crown_dir, band_of(H0, CR0, tshape))
        d, cut, hit = DIRS[t]
        dn = d / np.linalg.norm(d)
        H, CR = crest_field(mt, bone_open[tb], dn)
        # the plate: shave the root, along its way out, wherever it comes within gap + PLATE of the outer surface
        band = band_of(H, CR, tshape)
        shave = (sweep(band, -dn) & tshape) if band.any() else np.zeros_like(tshape)
        Dm = sdf(mt)
        thin_plate = int((shave & (Dm < -SHAVE_MAX)).sum())                    # where the plate must thin instead
        shave &= Dm >= -SHAVE_MAX
        if shave.any():
            depth = float((-Dm)[shave].max())
            # the shaved tooth's own distance field, recomputed: max(T, -sdf(shave)) measures, inside the shave, the
            # distance to the shave's edge (~0.3 mm), not to the tooth, so the socket carved the plate it made room for
            tshape = (T <= 0) & ~shave; T = sdf(tshape)
        else:
            depth = 0.0
        above = float(((shave) & (H >= CR)).sum() * S ** 3)
        report["teeth"][t] = {"direction": np.round(dn, 3).tolist(), "bone_cut_vox": cut, "neighbour_hit_vox": hit,
                              "shaved_above_crest_mm3": round(above, 3),
                              "root_shaved_mm3": round(float(shave.sum() * S ** 3), 2), "root_shaved_max_mm": round(depth, 2),
                              "plate_thinned_mm3": round(thin_plate * S ** 3, 2)}
        pin_c = []
        if detent is not None and t in keep:
            T, cs, dinfo = detents(t, T <= 0, T, H, CR, dn, d, gap, detent, np.array(off) + [s.start for s in tb])
            tshape = T <= 0
            report["teeth"][t]["detents"] = dinfo
            for c in cs:                                                        # the bump's field, in a window round it
                lo = np.maximum(0, np.floor(c - 12).astype(int)); hi = np.minimum(T.shape, np.ceil(c + 13).astype(int))
                w = tuple(slice(a, b) for a, b in zip(lo, hi))
                Dc = np.sqrt(((np.indices(hi - lo) + (lo - c)[:, None, None, None]) ** 2).sum(0)) * S
                N = np.maximum(Dc - DETENT_R, T[w] - (gap + MARGIN + 0.6))      # (only its own wall: never into a neighbour's socket)
                wall = (Dc < DETENT_R) & (T[w] > gap + MARGIN + 0.1)
                NUBS.append((tuple(slice(a + s.start, b + s.start) for a, b, s in zip(lo, hi, tb)), N, wall, t))
                pin_c.append((c + [s.start for s in tb] + off) * S * np.array([-1.0, -1.0, 1.0]))
        Wt = np.minimum(T, sdf(sweep(tshape, d)))                              # the tooth's own exact surface, the sweep beyond it
        W[tb] = np.minimum(W[tb], Wt)
        if detent is not None: WT.append((tb, Wt, t))                          # (each bump keeps out of the other teeth's sockets)
        # bone round the root below its crest: the scan's outline, within REACH of the root
        PZ[tb] = np.minimum(PZ[tb], np.maximum(np.maximum(Ot, H - CR), sdf(mt) - REACH))
        if os.environ.get("DEBUG_TOOTH") == str(t):
            DEBUG.update(t=t, tb=tb, off=off, T=T, H=H, CR=CR, Ot=Ot, band=band, shave=shave, Wt=Wt, mt=mt,
                         P=np.maximum(np.maximum(Ot, H - CR), sdf(mt) - REACH))
        if t in keep:
            toff = [o + s.start for o, s in zip(off, tb)]
            if t in FILL:
                Fm = FILL[t][bb][tb]
                F = sdf(Fm, 0.5)
                out.append((t, mesh(np.maximum(T, -F), toff), mesh(np.maximum(T, F), toff)))
            else:
                tm = mesh(T, toff)
                pin = None
                if pin_c:                                                       # the dimples keep their exact gap
                    v = np.asarray(tm.vertices)
                    pin = np.min([np.linalg.norm(v - c, axis=1) for c in pin_c], 0) < DETENT_R + gap + MARGIN + 0.3
                out.append((t, smooth(tm, SMOOTH_TEETH, pin), None))
    for f, Fm in FILL.items():                                                  # a filled surface was exposed in life
        if f in teeth: PZ = np.maximum(PZ, 1.5 - sdf(Fm[bb], 0.5))
    B = np.minimum(B, PZ)
    plate_zone = PZ <= 0
    zg = np.arange(bb[2].start, bb[2].stop)[None, None, :]
    xg = np.arange(bb[0].start, bb[0].stop)[:, None, None]; yg = np.arange(bb[1].start, bb[1].stop)[None, :, None]
    B = np.maximum(B, (zg - Z_TOP) * S)                                         # flat top (the scan's)
    B = np.maximum(B, (np.sqrt((xg - (SH[0] - 1) / 2) ** 2 + (yg - (SH[1] - 1) / 2) ** 2) - FOV_R) * S)   # cylinder
    if not upper:
        canal = np.isin(R, [3, 4])
        line = skeletonize(canal)
        C = sdf(canal | ndi.binary_dilation(line, ball(int(round(CANAL_D / 2 / S)))))
        B = np.maximum(B, -C)
    B = np.maximum(B, gap + MARGIN - W)
    # near the sockets, bone too thin to print (crest whiskers, slivers between sockets) is removed
    solid = B <= 0
    sliver = solid & ~ndi.binary_opening(solid, ball(2)) & (W < RIM) & ~plate_zone   # (the plate is kept)
    if sliver.any(): B = np.maximum(B, -sdf(sliver, 0.5))
    for w, N, wall, t in NUBS:                                                  # detent bumps (after the sliver pass, which would nibble them)
        Wo = np.full(N.shape, 1e3, np.float32)
        for tb2, Wt2, t2 in WT:
            if t2 == t: continue
            lo = [max(a.start, b.start) for a, b in zip(w, tb2)]; hi = [min(a.stop, b.stop) for a, b in zip(w, tb2)]
            if any(h <= l for l, h in zip(lo, hi)): continue
            dst = tuple(slice(l - a.start, h - a.start) for l, h, a in zip(lo, hi, w))
            Wo[dst] = np.minimum(Wo[dst], Wt2[tuple(slice(l - b.start, h - b.start) for l, h, b in zip(lo, hi, tb2))])
        N = np.maximum(N, gap + MARGIN + 0.1 - Wo)
        backing = round(float((B[w][wall] <= 0).mean()), 2) if wall.any() else 0.0
        report["teeth"][t].setdefault("detent_backing", []).append(backing)
        if backing < DETENT_BACKING: continue                                   # no septum there (crowded front teeth): no bump
        B[w] = np.minimum(B[w], N)
    # one piece: what the scan's cylinder cut loose (coronoid tips) or never joined (the hyoid) is dropped
    lab, k = ndi.label(B <= 0)
    if k > 1:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        B = np.maximum(B, -sdf((lab > 0) & (lab != sz.argmax()), 0.5))
    log("jaw", jaw, "slivers removed", int(sliver.sum()), "loose pieces dropped", k - 1)
    if crop is not None:
        cb = tuple(slice(c.start - o, c.stop - o) for c, o in zip(crop, off))
        inside = _inbox(R.shape, cb)
        B = np.maximum(B, np.where(inside, -1e3, 1e3).astype(np.float32))       # crisp box faces
    if DEBUG: np.savez(OUT / f"debug-{DEBUG['t']}.npz", B=B[DEBUG["tb"]], W=W[DEBUG["tb"]], PZ=PZ[DEBUG["tb"]],
                       **{k: v for k, v in DEBUG.items() if k not in ("tb", "off", "t")})
    bone_m = mesh(B, off, 0.03)
    # the socket walls stay as the fields made them (the gaps are measured there); the rest is smoothed
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    for _, w, _ in out: sc.add_triangles(o3d.core.Tensor(np.asarray(w.vertices, np.float32)), o3d.core.Tensor(np.asarray(w.faces, np.uint32)))
    near_teeth = sc.compute_distance(o3d.core.Tensor(np.asarray(bone_m.vertices, np.float32))).numpy() < gap + MARGIN + 0.15 if out else None   # socket walls only (the plate's outer face is >= 0.5 mm further)
    bone_m = smooth(bone_m, SMOOTH_BONE, near_teeth, crop is not None)
    return bone_m, out


def label_underside(m, label, depth=0.6):
    """Deboss label into the flat bottom face of a piece, mirrored to read from below, along the face's long axis."""
    M = m3.Manifold(m3.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
    z0 = M.bounding_box()[2]
    sec = M.slice(z0 + 0.05)
    pts = np.vstack([np.asarray(p) for p in sec.to_polygons()])
    c = pts.mean(0); ax = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    w = np.ptp(pts @ np.array([-ax[1], ax[0]]))                    # the face's narrow width
    t = text(label, min(6.0, 0.6 * w), c[0], c[1], z0 - 0.01, z0 + depth, mirror=True, angle=float(np.degrees(np.arctan2(ax[1], ax[0]))))
    tp = np.asarray(t.to_mesh().vert_properties)[:, :2] - c
    log(f"label {label}: face {np.ptp(pts @ ax):.1f} x {w:.1f} mm, text {np.ptp(tp @ ax):.1f} along x {np.ptp(tp @ np.array([-ax[1], ax[0]])):.1f} across")
    o = (M - t).to_mesh()
    return trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=False)


def _inbox(shape, sl):
    m = np.zeros(shape, bool); m[sl] = True; return m


DEBUG = {}
DIRS_FILE = OUT / "dirs.json"
DIRS = {int(k): (np.array(v[0]), v[1], v[2]) for k, v in json.loads(DIRS_FILE.read_text()).items()} if DIRS_FILE.exists() else {}
OUT.mkdir(exist_ok=True)
PIECES = sys.argv[1:] or ["lower", "upper", "coupon"]


def save_dirs():
    """Merge this process's removal directions into the cache (other pieces may be writing theirs)."""
    old = json.loads(DIRS_FILE.read_text()) if DIRS_FILE.exists() else {}
    old.update({str(t): [np.asarray(d).tolist(), int(c), int(h)] for t, (d, c, h) in DIRS.items()})
    DIRS_FILE.write_text(json.dumps(old))


for jaw, teeth, name in ((1, LOWER, "lower-jaw"), (2, UPPER, "upper-jaw")):
    if name.split("-")[0] not in PIECES: continue
    bone_m, teeth_m = build_jaw(jaw, teeth, GAP, detent=DETENT)
    upper = jaw == 2
    parts = [("bone", "wood", place(bone_m, upper))]
    for t, w, b in teeth_m:
        parts.append((f"tooth {t}", "white", place(w, upper)))
        if b is not None: parts.append((f"tooth {t} filling", "black", place(b, upper)))
    write_3mf(OUT / f"{name}.3mf", [(name, parts)])
    report[name] = {"bone_faces": len(bone_m.faces), "bone_watertight": bool(bone_m.is_watertight),
                    "teeth_watertight": all(w.is_watertight for _, w, _ in teeth_m),
                    "size_mm": np.round(parts[0][2].extents, 1).tolist()}
    for p, f, m in parts: m.export(OUT / f"{name}--{p.replace(' ', '_')}.ply")
    parts[0][2].export(OUT / f"{name}-bare-bone.ply")                       # (stand.py cuts the mount's holes from this)
    save_dirs()
    log(name, report[name])

if "coupon" in PIECES:
    # coupon: one premolar in its bone at the tightest gap, once per detent strength, labelled underneath
    cbox = box(ref == COUPON_TOOTH, 14, SH)
    cbox = (cbox[0], cbox[1], slice(max(0, cbox[2].start - 10), cbox[2].stop))
    coupon = []
    for i, k in enumerate(COUPON_DETENTS):
        bone_m, teeth_m = build_jaw(1, LOWER, COUPON_GAP, crop=cbox, detent=k)
        bone_m = label_underside(bone_m, f"{round(k * 100):d}")
        shift = [i * 25.0, 0, 0]
        parts = [("bone", "wood", bone_m.copy().apply_translation(shift))]
        for t, w, b in teeth_m:
            parts.append((f"tooth {t}", "white", w.copy().apply_translation(shift)))
        coupon.append((f"coupon detent {k:.2f} mm", parts))
        for p, f, m in parts: m.export(OUT / f"detent-{k:.2f}--{p.replace(' ', '_')}.ply")
        report[f"detent-{k:.2f}"] = {t: report["teeth"][t].get("detents") for t, _, _ in teeth_m} | \
                                    {f"{t} backing": report["teeth"][t].pop("detent_backing", None) for t, _, _ in teeth_m}
        log("coupon detent", k, report[f"detent-{k:.2f}"])
    write_3mf(OUT / "coupon.3mf", coupon)
    save_dirs()
(OUT / f"report-{'-'.join(PIECES)}.json").write_text(json.dumps(report, indent=1, default=int))
log("done", PIECES)
