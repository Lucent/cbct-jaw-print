"""What the print scripts share: the scan's voxel size and cut faces, the teeth, the filaments, the 3MF writer and
the print orientation."""
import zipfile
from xml.sax.saxutils import quoteattr

import manifold3d as m3
import numpy as np
import trimesh
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

S = 0.1403                      # mm per voxel
Z_TOP, FOV_R = 616.0, 299.5     # the scan's flat top (voxel z) and cylinder radius (voxels): cut faces
COLOURS = {"wood": "#B5835AFF", "white": "#F2F0EBFF", "black": "#2B2B2BFF"}
UPPER = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]


def rotation_to_z(d):
    """Rotation matrix taking unit vector d to +z."""
    d = d / np.linalg.norm(d); z = np.array([0.0, 0, 1])
    v = np.cross(d, z); c = d @ z
    if np.linalg.norm(v) < 1e-9: return np.eye(3) if c > 0 else np.diag([1.0, -1, -1])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def write_3mf(path, objects):
    """3MF of named objects [(name, [(part name, filament, trimesh)])], each part coloured from COLOURS, so Bambu
    Studio loads each object as one with its parts and gives each part its filament."""
    names = list(COLOURS)
    body = ['<basematerials id="1">' + "".join(f'<base name="{c}" displaycolor="{COLOURS[c]}"/>' for c in names) + '</basematerials>']
    nid, items = 2, []
    # on the H2D's plate: each object's lowest point at z 0, the whole layout centred on the 350 x 320 bed
    lo = np.min([m.bounds[0] for _, parts in objects for _, _, m in parts], 0); hi = np.max([m.bounds[1] for _, parts in objects for _, _, m in parts], 0)
    sx, sy = 175.0 - (lo[0] + hi[0]) / 2, 160.0 - (lo[1] + hi[1]) / 2
    for name, parts in objects:
        z0 = min(m.bounds[0][2] for _, _, m in parts)
        comps = []
        for pname, fil, m in parts:
            verts = "".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in m.vertices)
            tris = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in m.faces)
            body.append(f'<object id="{nid}" name={quoteattr(pname)} type="model" pid="1" pindex="{names.index(fil)}">'
                        f'<mesh><vertices>{verts}</vertices><triangles>{tris}</triangles></mesh></object>')
            comps.append(nid); nid += 1
        body.append(f'<object id="{nid}" name={quoteattr(name)} type="model"><components>'
                    + "".join(f'<component objectid="{c}"/>' for c in comps) + '</components></object>')
        items.append((nid, f"1 0 0 0 1 0 0 0 1 {sx:.4f} {sy:.4f} {-z0:.4f}")); nid += 1
    model = ('<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             f'<resources>{"".join(body)}</resources><build>{"".join(f"<item objectid={chr(34)}{i}{chr(34)} transform={chr(34)}{t}{chr(34)}/>" for i, t in items)}</build></model>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr("3D/3dmodel.model", model)


def place(m, upper):
    """Print orientation: the upper jaw upside down on its flat cut top, the lower jaw on its lower edge; z from 0."""
    m = m.copy()
    if upper: m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    return m


def text(s, h, x, y, z0, z1, anchor="center", mirror=False, angle=0.0):
    """Extruded text, height h mm, centred at (x, y), from z0 to z1."""
    tp = TextPath((0, 0), s, size=h, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    polys = [np.asarray(p) for p in tp.to_polygons() if len(p) > 2]
    xs = m3.CrossSection([p.tolist() for p in polys], m3.FillRule.EvenOdd)
    b = xs.bounds(); cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    xs = xs.translate((-cx, -cy))
    if mirror: xs = xs.mirror((1, 0))                  # reads right way round from the other side (an underside)
    return xs.rotate(angle).translate((x, y)).extrude(z1 - z0).translate((0, 0, z0))
