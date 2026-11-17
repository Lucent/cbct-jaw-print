"""The hinge and display stand, from out/landmarks.json and the jaw meshes (RAS mm: +X patient left, +Y anterior,
+Z up). All black, but for an optional wood nameplate.

  cradle  a plate over the upper jaw's flat cut top (a 1 mm lipped pocket the shape of the bone's cut footprint
          locates it; tenons shaped to the solid bone under the top push into mortises), a crossbar behind, and a
          tab at each condyle carrying the pivot and a stop peg
  arms    (2) each a pad against its ramus's cut face (two pins push into teardrop holes in the ramus), a bar up
          to a disc at the condyle; the disc's arc slot rides on the tab's peg: closed at the scanned bite, open
          OPEN degrees; an M4x20 bolt, two washers and a nyloc nut through tab and disc is the friction hinge
  post    from the base up into the crossbar, behind the jaw; an M4x16 flat-head screw up through the base and one
          down through the cradle, each into an M4 nut slid into a slot in the post's back
  base    no bigger than the pivots, post and chin need, rounded; with NAMEPLATE set, a wood nameplate in black
          lettering along its front edge
Nothing is glued: tenons and pins are FIT clear of their holes with crush ribs RIB into the walls (they hold
whatever the printer's tolerance; the lower jaw is held between the arms anyway). The jaws get the mortises and
pin holes (out/upper-jaw.3mf and out/lower-jaw.3mf rewritten, from print_parts' bare bone in out/*-bare-bone.ply).
Hardware (metric, Ace/Hillman): 2 M4-0.7x20 socket cap screws (black), 2 M4 nylon-insert lock nuts, 4 M4 flat
washers, 2 M4-0.7x16 flat-head machine screws, 2 M4 hex nuts.

  python scripts/stand.py  ->  out/stand.3mf, out/assembly.glb, out/assembly.png
"""
import json, os, sys
from pathlib import Path

import manifold3d as m3, nibabel as nib, numpy as np, trimesh
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from scipy import ndimage as ndi
from shapely.geometry import Polygon, MultiPolygon, Point, box as sbox
from shapely.affinity import translate as stranslate
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as PP  # noqa: E402
from common import text  # noqa: E402

S = PP.S
OPEN = 35.0                    # degrees the jaw opens to its stop
T_DISC, R_DISC, R_PEG, R_SLOT = 5.0, 12.0, 1.5, 8.0
R_BOLT = 2.2                   # M4 clearance
CLEAR = 0.4                    # mm between moving parts
PLATE_T, POCKET_D = 8.0, 1.0
POST = 16.0
BASE_T, BASE_GAP = 10.0, 12.0  # base thickness; clearance under the open chin
MARGIN_XY, CORNER = 6.0, 8.0     # base edge beyond what stands on it; corner radius
PAD_T = 4.0
FIT, RIB, WALL = 0.15, 0.25, 1.0   # mm: tenon/pin clearance; crush ribs' reach past it; least bone round a mortise
TENON_D, PIN_R, PIN_D = 6.0, 2.0, 7.0   # upper tenon depth; ramus pin radius and depth
NUT_AF, NUT_H, CSK_R = 7.0 + 0.25, 3.2 + 0.3, 4.1    # M4 hex nut trap (across flats, height); flat-head countersink radius
NAME = os.environ.get("NAMEPLATE", "")   # nameplate text; empty: no nameplate
L = json.loads(Path("out/landmarks.json").read_text())
u = np.array(L["lateral"]); n = np.array(L["occlusal_normal"]); fwd = np.array(L["anterior"])
CR, CL = (np.array(c) for c in L["condyles"])            # right (patient), left
axis_mid = np.array(L["hinge_axis_mid"])
Z_TOP = PP.Z_TOP * S


def M(mesh):
    return m3.Manifold(m3.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32), tri_verts=np.asarray(mesh.faces, np.uint32)))


def T(m):
    o = m.to_mesh(); return trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=False)


def frame(z_to):
    """3x4 transform taking +z to unit vector z_to (about the origin)."""
    R = PP.rotation_to_z(z_to).T
    return np.hstack([R, np.zeros((3, 1))])


def disc(c, r, t, axis=u, seg=96):
    """Cylinder of radius r, thickness t along axis, centred on c."""
    return m3.Manifold.cylinder(t, r, r, seg, True).transform(frame(axis)).translate(tuple(c))


def rod(a, b, r, seg=24):
    d = np.asarray(b) - np.asarray(a); h = np.linalg.norm(d)
    return m3.Manifold.cylinder(h, r, r, seg, False).transform(frame(d / h)).translate(tuple(a))


def poly_xs(geom):
    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    rings = []
    for p in polys:
        rings.append(np.asarray(p.exterior.coords)[:-1]); rings += [np.asarray(i.coords)[:-1] for i in p.interiors]
    return m3.CrossSection([r.tolist() for r in rings], m3.FillRule.EvenOdd)


def to_frame(origin, dirn):
    """(R, 3x4) taking origin to 0 and dirn to +z."""
    R = PP.rotation_to_z(np.asarray(dirn, float)); return R, np.hstack([R, (-R @ np.asarray(origin, float))[:, None]])


def from_frame(R, origin):
    return np.hstack([R.T, np.asarray(origin, float)[:, None]])


def solid_through(Mf, A, z0, z1):
    """The cross-section (frame A's xy) solid at every depth z0..z1 of frame A's z."""
    Mt = Mf.transform(A); cs = None
    for z in np.arange(z0, z1 + 1e-6, 0.4):
        sl = Mt.slice(z); cs = sl if cs is None else cs ^ sl
    return cs


def ribs(ring_pts, inside, k, z0, z1, r=0.6):
    """k crush ribs (cylinders along z, z0..z1) spaced round a tenon's outline, reaching FIT + RIB outside it."""
    P = np.asarray(ring_pts, float); seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
    out = []
    for j in range(k):
        t = cum[-1] * (j + 0.5) / k; i = int(np.searchsorted(cum, t) - 1); i = min(i, len(P) - 1)
        a, b = P[i], P[(i + 1) % len(P)]; p = a + (b - a) * (t - cum[i]) / max(seg[i], 1e-9)
        tg = (b - a) / max(seg[i], 1e-9); nr = np.array([tg[1], -tg[0]])
        if inside(p + nr * 0.3): nr = -nr
        c = p + nr * (FIT + RIB - r)
        out.append(m3.Manifold.cylinder(z1 - z0, r, r, 16).translate((c[0], c[1], z0)))
    return out


def pixels_polygon(mask2d, xs, ys):
    """Union of the mask's pixels as a shapely geometry in RAS mm (pixel (i, j) -> (-i S, -j S))."""
    sq = [sbox(-(i + 0.5) * S, -(j + 0.5) * S, -(i - 0.5) * S, -(j - 0.5) * S) for i, j in zip(*np.nonzero(mask2d))]
    return unary_union(sq).buffer(0.02).buffer(-0.02).simplify(0.03)


ref = np.asarray(nib.load("refined2.nii.gz").dataobj).astype(np.int16)
SH = ref.shape
CYL = np.array([-(SH[0] - 1) / 2 * S, -(SH[1] - 1) / 2 * S]); R_CYL = PP.FOV_R * S

# ------------------------------------------------------------------ cradle
top = pixels_polygon(ref[:, :, int(PP.Z_TOP) - 1] == 2, None, None)
plate2d = top.convex_hull.buffer(3.0)
upper_bare = trimesh.load("out/upper-jaw-bare-bone.ply")
upper_bare.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))       # (printed upside down) back to anatomy
UB = M(upper_bare)
band = UB.slice(Z_TOP - 0.05)
for zb in np.arange(Z_TOP - POCKET_D - 0.3, Z_TOP - 0.05, 0.1): band = band + UB.slice(zb)
pocket_cs = band.offset(0.15, m3.JoinType.Round)                                 # (the bone flares under its cut top: the lip clears it)
cradle = poly_xs(plate2d).extrude(PLATE_T + POCKET_D).translate((0, 0, Z_TOP - POCKET_D))
cradle -= pocket_cs.extrude(POCKET_D + 0.01).translate((0, 0, Z_TOP - POCKET_D - 0.01))
y_back = min(CR[1], CL[1]) - 20.0
tabs, pegs, tab_x = [], [], {}
for c, inward in ((CR, u), (CL, -u)):
    tc = c + inward * (T_DISC / 2 + CLEAR + T_DISC / 2)          # tab centre, inboard of the arm's disc
    tab = disc(tc, R_DISC, T_DISC)
    tab = m3.Manifold.batch_hull([tab, m3.Manifold.cube((T_DISC, 22, PLATE_T), True).translate((tc[0], c[1] - 5, Z_TOP + PLATE_T / 2))])
    peg_dir = (fwd + n) / np.linalg.norm(fwd + n)                   # anterior-up: away from the arm's bar
    pc = c + peg_dir * R_SLOT
    peg = rod(pc + inward * (T_DISC / 2 + CLEAR + 2), pc - inward * (T_DISC / 2 - 0.3), R_PEG)   # from inside the tab to 0.3 short of the arm's far face
    tabs.append(tab); pegs.append(peg); tab_x[tuple(c)] = tc
crossbar = m3.Manifold.batch_hull([m3.Manifold.cube((1, 26, PLATE_T), True).translate((tab_x[tuple(CR)][0], y_back + 13 + 2, Z_TOP + PLATE_T / 2)),
                                   m3.Manifold.cube((1, 26, PLATE_T), True).translate((tab_x[tuple(CL)][0], y_back + 13 + 2, Z_TOP + PLATE_T / 2))])
link = m3.Manifold.batch_hull([poly_xs(plate2d).extrude(PLATE_T).translate((0, 0, Z_TOP)),
                               m3.Manifold.cube((POST + 8, 1, PLATE_T), True).translate((axis_mid[0], y_back + 14, Z_TOP + PLATE_T / 2))])
cradle = m3.Manifold.batch_boolean([cradle, crossbar, link - pocket_cs.extrude(POCKET_D).translate((0, 0, Z_TOP))] + tabs + pegs, m3.OpType.Add)
cradle -= pocket_cs.extrude(POCKET_D + 0.01).translate((0, 0, Z_TOP - POCKET_D - 0.01))
# tenons: the solid bone under the upper jaw's top, WALL in, TENON_D deep
o_up = np.array([0.0, 0.0, Z_TOP]); Ru, Au = to_frame(o_up, [0, 0, -1.0])
holes = [c for c in solid_through(UB, Au, 0.3, TENON_D + WALL).offset(-WALL, m3.JoinType.Round).decompose() if c.area() >= 8.0]
upper_mortise = m3.Manifold.batch_boolean([h.extrude(TENON_D + 1).translate((0, 0, -1)) for h in holes], m3.OpType.Add).transform(from_frame(Ru, o_up))
tenons = []
for h in holes:
    ring = max(h.to_polygons(), key=len); pg = Polygon(ring)
    t_cs = h.offset(-FIT, m3.JoinType.Round)
    tenons.append(t_cs.extrude(TENON_D - 0.4 + PLATE_T / 2).translate((0, 0, -PLATE_T / 2)))
    tenons += ribs(ring, lambda q: pg.contains(Point(q)), 4, 1.0, TENON_D - 0.4)
cradle += m3.Manifold.batch_boolean(tenons, m3.OpType.Add).transform(from_frame(Ru, o_up))
print("upper tenons:", [round(h.area(), 1) for h in holes], "mm2")
for c in (CR, CL):
    cradle -= rod(c - u * 20, c + u * 20, R_BOLT)
post_xy = (axis_mid[0], y_back + 2 + POST / 2)
cradle -= m3.Manifold.cube((POST + 0.3, POST + 0.3, PLATE_T / 2), True).translate((post_xy[0], post_xy[1], Z_TOP + PLATE_T / 4 - 0.01))

# ------------------------------------------------------------------ arms
lower_bone = trimesh.load("out/lower-jaw-bare-bone.ply")
LB = M(lower_bone)
import open3d as o3d
OSC = o3d.t.geometry.RaycastingScene()
OSC.add_triangles(o3d.core.Tensor(np.asarray(lower_bone.vertices, np.float32)), o3d.core.Tensor(np.asarray(lower_bone.faces, np.uint32)))
xg, yg = np.mgrid[:SH[0], :SH[1]]
rr = np.sqrt((xg - (SH[0] - 1) / 2) ** 2 + (yg - (SH[1] - 1) / 2) ** 2)
th = np.arctan2(-yg * S - CYL[1], -xg * S - CYL[0])
arms, pin_holes, arm_pins, inwards = [], [], [], []
for c, side in ((CR, -1), (CL, 1)):
    cut = (rr > PP.FOV_R - 1.5)[..., None] & (ref == 1)
    pts = np.argwhere(cut)
    ras_pts = pts * S * np.array([-1.0, -1.0, 1.0])
    keep = (np.sign((ras_pts - np.array(L["midline_point"])) @ u) == side) & (ras_pts[:, 1] < np.array(L["rami_cut"]["right" if side < 0 else "left"]["centroid"])[1] + 15)
    pts, ras_pts = pts[keep], ras_pts[keep]
    ang = th[pts[:, 0], pts[:, 1]]
    a0 = np.median(ang)
    sz = np.stack([(ang - a0) * R_CYL, ras_pts[:, 2]], 1)              # the cut face unrolled: arc length, height
    face = unary_union([sbox(a - S * 0.8, z - S * 0.8, a + S * 0.8, z + S * 0.8) for a, z in sz[::3]]).buffer(0.2).buffer(-0.2)
    face = max(face.geoms, key=lambda g: g.area) if isinstance(face, MultiPolygon) else face
    slab = poly_xs(face).extrude(PAD_T).refine(4)                     # (s, z, t): t = outward from the cylinder
    pad = slab.warp_batch(lambda V: np.stack([CYL[0] + (R_CYL + 0.15 + V[:, 2]) * np.cos(a0 + V[:, 0] / R_CYL),
                                              CYL[1] + (R_CYL + 0.15 + V[:, 2]) * np.sin(a0 + V[:, 0] / R_CYL), V[:, 1]], 1))   # (0.15 off the cut face: seats without glue)
    pc = np.array(face.centroid.coords[0]); p_out = np.array([CYL[0] + (R_CYL + PAD_T / 2) * np.cos(a0 + pc[0] / R_CYL),
                                                              CYL[1] + (R_CYL + PAD_T / 2) * np.sin(a0 + pc[0] / R_CYL), pc[1]])
    down = -(n - (n @ u) * u); down /= np.linalg.norm(down)
    p1 = c + down * (R_DISC + 4)
    bar = m3.Manifold.batch_hull([disc(c, 5, T_DISC), disc(p1, 5, T_DISC)]) + m3.Manifold.batch_hull([disc(p1, 5, T_DISC), disc(p_out, 5, T_DISC)])
    arm = m3.Manifold.batch_boolean([pad, disc(c, R_DISC, T_DISC), bar], m3.OpType.Add)
    arm -= rod(c - u * 20, c + u * 20, R_BOLT)
    peg_dir = (fwd + n) / np.linalg.norm(fwd + n)
    slot = []
    for a in np.linspace(0, OPEN, 15):                                # the peg as seen from the arm while it opens
        R = trimesh.transformations.rotation_matrix(np.radians(a), u)[:3, :3]
        q = c + R @ (peg_dir * R_SLOT)
        slot.append(rod(q - u * 10, q + u * 10, R_PEG + CLEAR / 2, 16))
    arm -= m3.Manifold.batch_boolean([m3.Manifold.batch_hull(slot[i:i + 2]) for i in range(len(slot) - 1)], m3.OpType.Add)
    # two pins from the pad's inner face into the ramus, where the bone is solid PIN_D deep (WALL round them)
    a = a0 + pc[0] / R_CYL; radial = np.array([np.cos(a), np.sin(a), 0.0])
    o_f = np.array([CYL[0] + R_CYL * np.cos(a), CYL[1] + R_CYL * np.sin(a), pc[1]])
    Rf, Af = to_frame(o_f, -side * u)                                          # along the hinge axis: the arm pushes on sideways, onto the pins and the tab's peg together
    # candidate pin lines (frame xy) over the cut face; each traced through the bone: it must enter through the
    # cut face (where the pad covers it) and stay solid, WALL round the hole, PIN_D deep
    fv = np.asarray(lower_bone.vertices); rv = np.hypot(fv[:, 0] - CYL[0], fv[:, 1] - CYL[1])
    fv = fv[(np.abs(rv - R_CYL) < 0.3) & (np.sign((fv - np.array(L["midline_point"])) @ u) == side)]
    fxy = (fv @ Rf.T + Af[:, 3])[:, :2]
    gx, gy = np.meshgrid(np.arange(fxy[:, 0].min(), fxy[:, 0].max(), 0.5), np.arange(fxy[:, 1].min(), fxy[:, 1].max(), 0.5))
    G = np.stack([gx.ravel(), gy.ravel()], 1)
    zs = np.arange(-10, PIN_D + 8, 0.2)
    r_ring = PIN_R + FIT + WALL
    offs = np.vstack([[0, 0]] + [[r_ring * np.cos(t), r_ring * np.sin(t)] for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)])
    pts = np.zeros((len(G), len(offs), len(zs), 3))
    pts[..., :2] = (G[:, None, :] + offs[None])[:, :, None, :]; pts[..., 2] = zs
    world = pts.reshape(-1, 3) @ Rf + o_f                                          # frame -> RAS (R^T x + o)
    occ = OSC.compute_occupancy(o3d.core.Tensor(world.astype(np.float32)), nsamples=3).numpy().reshape(len(G), len(offs), len(zs)) > 0.5
    good = []
    for gi in range(len(G)):
        c = occ[gi, 0]
        if not c.any(): continue
        e = int(np.argmax(c)); ze = zs[e]                                      # where the centre line enters bone
        we = world.reshape(len(G), len(offs), len(zs), 3)[gi, 0, e]
        if abs(np.hypot(we[0] - CYL[0], we[1] - CYL[1]) - R_CYL) > 0.4: continue   # (enters through the outer cortex: no)
        span = (zs >= ze + 1.5) & (zs <= ze + PIN_D + 0.5)
        if occ[gi][:, span].all(): good.append((G[gi], ze))
    if not good: raise SystemExit(f"no room for a pin in the {'right' if side < 0 else 'left'} ramus")
    P_ = np.array([g for g, _ in good]); D2 = ((P_[:, None] - P_[None]) ** 2).sum(-1); i, j = np.unravel_index(D2.argmax(), D2.shape)
    sites = [good[i], good[j]] if D2[i, j] > (2 * (PIN_R + FIT) + WALL) ** 2 else [good[i]]
    centres = [g for g, _ in sites]
    up = (Rf @ np.array([0, 0, 1.0]))[:2]; up /= np.linalg.norm(up)
    for c2, ze in sites:
        rh = PIN_R + FIT
        tear = m3.CrossSection.batch_hull([m3.CrossSection.circle(rh, 32).translate(tuple(c2)),
                                           m3.CrossSection.circle(0.05, 8).translate(tuple(c2 + up * rh * np.sqrt(2)))])   # (prints sideways: a 45 deg roof)
        pin_holes.append(tear.extrude(PIN_D + 3).translate((0, 0, ze - 3)).transform(from_frame(Rf, o_f)))   # (out through the cut face, into the pad)
        ring = [tuple(c2 + PIN_R * np.array([np.cos(t), np.sin(t)])) for t in np.linspace(0, 2 * np.pi, 48, endpoint=False)]
        pin = [m3.CrossSection.circle(PIN_R, 48).translate(tuple(c2)).extrude(PIN_D - 0.4 + 1.2).translate((0, 0, ze - 1.2))]   # (from just inside the pad)
        pin += ribs(ring, lambda q, c2=c2: np.linalg.norm(np.asarray(q) - c2) < PIN_R, 3, ze + 1.0, ze + PIN_D - 0.4)
        arm_pins.append((len(arms), m3.Manifold.batch_boolean(pin, m3.OpType.Add).transform(from_frame(Rf, o_f))))
    print("ramus", "right" if side < 0 else "left", "pins", len(sites), "of", len(good), "possible lines; entries (frame xy, z):", [(np.round(c2, 1).tolist(), round(float(ze), 1)) for c2, ze in sites])
    arms.append(arm); inwards.append(-side * u)

# each arm clears the bone all the way along its sliding fit (0.2 mm round), then gets its pins
e1 = np.cross(u, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
for i, inward in enumerate(inwards):
    lo, hi = np.array(arms[i].bounding_box()[:3]) - 12, np.array(arms[i].bounding_box()[3:]) + 12
    near = LB ^ m3.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))
    moves = [near.translate(tuple(inward * k + off)) for k in np.arange(0, PIN_D + 1.0, 0.5)
             for off in (np.zeros(3), 0.2 * e1, -0.2 * e1, 0.2 * e2, -0.2 * e2)]
    arms[i] = arms[i] - m3.Manifold.batch_boolean(moves, m3.OpType.Add)
    cyl_in = m3.Manifold.cylinder(400, R_CYL + 0.15, R_CYL + 0.15, 256).translate((CYL[0], CYL[1], -200))
    stub = m3.Manifold.batch_boolean([p_ for j, p_ in arm_pins if j == i], m3.OpType.Add) - arms[i] - cyl_in
    sv = np.asarray(stub.to_mesh().vert_properties)[:, :3]
    print(f"arm {i}: pin volume outside both pad and bone side: {stub.volume():.2f} mm3; its radius from the scan axis "
          f"{np.hypot(sv[:, 0] - CYL[0], sv[:, 1] - CYL[1]).min() - R_CYL:.2f}..{np.hypot(sv[:, 0] - CYL[0], sv[:, 1] - CYL[1]).max() - R_CYL:.2f} mm past the cut (pad {PAD_T + 0.15:.2f})" if len(sv) else f"arm {i}: no stub")
    arms[i] = m3.Manifold.batch_boolean([arms[i]] + [p_ for j, p_ in arm_pins if j == i], m3.OpType.Add)

# ------------------------------------------------------------------ the chin's lowest point while opening
lowest = np.inf
for a in np.linspace(0, OPEN, 8):
    R = trimesh.transformations.rotation_matrix(-np.radians(a), u, axis_mid)
    lowest = min(lowest, (lower_bone.vertices @ R[:3, :3].T + R[:3, 3])[:, 2].min())
base_top = lowest - BASE_GAP

# ------------------------------------------------------------------ post and base
# The base is only as big as what stands on it: the pivots side to side, the post to just ahead of the chin, with
# the nameplate (if any) along the front edge.
SOCKET = BASE_T * 0.6
post = m3.Manifold.cube((POST, POST, Z_TOP + PLATE_T / 2 - base_top + SOCKET), False).translate((post_xy[0] - POST / 2, post_xy[1] - POST / 2, base_top - SOCKET))
post_z0, post_z1 = base_top - SOCKET, Z_TOP + PLATE_T / 2
for zc in (post_z0 + 9.0, post_z1 - 9.0):                                       # M4 nut traps, slid in from the back
    hexa = m3.CrossSection.circle(NUT_AF / 2 / np.cos(np.pi / 6), 6).rotate(30).translate(post_xy)
    slot = m3.CrossSection.square((NUT_AF, POST)).translate((post_xy[0] - NUT_AF / 2, post_xy[1] - POST))
    post -= (hexa + slot).extrude(NUT_H).translate((0, 0, zc - NUT_H / 2))
post -= m3.Manifold.cylinder(post_z1 - post_z0 + 2, R_BOLT, R_BOLT, 24).translate((post_xy[0], post_xy[1], post_z0 - 1))
chin_y = lower_bone.vertices[:, 1].max()
plate_h = 11.0 if NAME else 0.0
x0, x1 = CR[0] - MARGIN_XY, CL[0] + MARGIN_XY
y0, y1 = post_xy[1] - POST / 2 - MARGIN_XY, chin_y + plate_h + (2 if NAME else 1) * MARGIN_XY
outline = m3.CrossSection.square((x1 - x0, y1 - y0)).translate((x0, y0)).offset(-CORNER, m3.JoinType.Square).offset(CORNER, m3.JoinType.Round)
base = outline.extrude(BASE_T).translate((0, 0, base_top - BASE_T))
base -= m3.Manifold.cube((POST + 0.3, POST + 0.3, SOCKET + 0.01)).translate((post_xy[0] - POST / 2 - 0.15, post_xy[1] - POST / 2 - 0.15, base_top - SOCKET))
if NAME:
    plate_w = min(90.0, x1 - x0 - 2 * MARGIN_XY)
    inlay = m3.Manifold.cube((plate_w, plate_h, 2.0)).translate(((x0 + x1) / 2 - plate_w / 2, y1 - MARGIN_XY - plate_h, base_top - 2.0))
    letters = text(NAME, 6.0, (x0 + x1) / 2, y1 - MARGIN_XY - plate_h / 2, base_top - 2.0, base_top)
    wood = inlay - letters
    base -= wood
base -= m3.Manifold.cylinder(BASE_T + 1, R_BOLT, R_BOLT, 24).translate((post_xy[0], post_xy[1], base_top - BASE_T - 0.5))   # flat-head M4 up into the post
base -= m3.Manifold.cylinder(CSK_R - R_BOLT + 0.01, CSK_R, R_BOLT, 32).translate((post_xy[0], post_xy[1], base_top - BASE_T - 0.01))
cradle -= m3.Manifold.cylinder(PLATE_T + 1, R_BOLT, R_BOLT, 24).translate((post_xy[0], post_xy[1], Z_TOP - 0.5))         # and one down into its top
cradle -= m3.Manifold.cylinder(CSK_R - R_BOLT + 0.01, R_BOLT, CSK_R, 32).translate((post_xy[0], post_xy[1], Z_TOP + PLATE_T - (CSK_R - R_BOLT)))

# ------------------------------------------------------------------ the jaws' mortises and pin holes; outputs
bone = LB - m3.Manifold.batch_boolean(pin_holes, m3.OpType.Add)
ubone = UB - upper_mortise
for jaw, bm, upper in (("lower-jaw", bone, False), ("upper-jaw", ubone, True)):
    placed = PP.place(T(bm), upper)
    placed.export(f"out/{jaw}--bone.ply")
    parts = [("bone", "wood", placed)]
    for p in sorted(Path("out").glob(f"{jaw}--tooth*.ply")):
        nm = p.stem.split("--")[1].replace("_", " ")
        parts.append((nm, "black" if "filling" in nm else "white", trimesh.load(p)))
    PP.write_3mf(Path(f"out/{jaw}.3mf"), [(jaw, parts)])
print("mortise / pin-hole volume outside the bare bone (mouths only expected): upper %.1f of %.1f mm3, lower %.1f of %.1f mm3"
      % ((upper_mortise - UB).volume(), upper_mortise.volume(), (m3.Manifold.batch_boolean(pin_holes, m3.OpType.Add) - LB).volume(),
         m3.Manifold.batch_boolean(pin_holes, m3.OpType.Add).volume()))

stand = [("cradle", [("cradle", "black", T(cradle))]), ("arm right", [("arm", "black", T(arms[0]))]),
         ("arm left", [("arm", "black", T(arms[1]))]), ("post", [("post", "black", T(post))]),
         ("base", [("base", "black", T(base))] + ([("nameplate", "wood", T(wood))] if NAME else []))]
PP.write_3mf(Path("out/stand.3mf"), stand)
for old in Path("out").glob("stand-*--*.ply"): old.unlink()                     # (a part no longer made must not linger)
for name, ps in stand:
    for pn, f, m in ps: m.export(f"out/stand-{name.replace(' ', '_')}--{pn.replace(' ', '_')}.ply")

# assembly preview, closed and open
scene = trimesh.Scene()
col = {"wood": [181, 131, 90, 255], "white": [242, 240, 235, 255], "black": [43, 43, 43, 255]}
for name, ps in stand:
    for pn, f, m in ps:
        mm = m.copy(); mm.visual.face_colors = col[f]; scene.add_geometry(mm, node_name=f"{name} {pn}")
for jaw in ("upper-jaw", "lower-jaw"):
    for p in sorted(Path("out").glob(f"{jaw}--*.ply")):
        m = trimesh.load(p); f = "wood" if "bone" in p.stem else "black" if "filling" in p.stem else "white"
        if jaw == "upper-jaw": m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))   # back to anatomy
        m.visual.face_colors = col[f]; scene.add_geometry(m, node_name=p.stem)
scene.export("out/assembly.glb")
report = {"base_top_z": round(float(base_top), 1), "chin_lowest_open_z": round(float(lowest), 1),
          "base_mm": [round(x1 - x0, 1), round(y1 - y0, 1), BASE_T],
          "status": {n: str(m.status()) for n, m in (("cradle", cradle), ("arm_r", arms[0]), ("arm_l", arms[1]), ("post", post), ("base", base), ("bone", bone))}}
Path("out/stand.json").write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))

# ------------------------------------------------------------------ fit check: the mount touches the bone only at its crush ribs
for nm, part, bm in (("cradle/upper", cradle, ubone), ("arm r/lower", arms[0], bone), ("arm l/lower", arms[1], bone)):
    ov = (part ^ bm).volume()
    print(f"{nm}: overlap {ov:.2f} mm3 (crush ribs only: ~{RIB:.2f} mm deep), clearance elsewhere {part.min_gap(bm - (part ^ bm).offset(0) if False else bm, 1.0):.3f}" if False else f"{nm}: overlap {ov:.2f} mm3")
for nm, part, bm in (("cradle/upper", cradle, ubone), ("arm r/lower", arms[0], bone), ("arm l/lower", arms[1], bone)):
    for piece in (part ^ bm).decompose():
        if piece.volume() > 0.5:
            bb = np.array(piece.bounding_box()); print(f"  {nm}: {piece.volume():.1f} mm3 at x {bb[0]:.1f}..{bb[3]:.1f} y {bb[1]:.1f}..{bb[4]:.1f} z {bb[2]:.1f}..{bb[5]:.1f}")
print("Z_TOP", Z_TOP)
