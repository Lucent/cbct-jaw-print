"""Ray-cast pictures of a mesh, with no renderer: orthographic, lit by which way each face looks."""
import numpy as np
import open3d as o3d

SIZE = 900


def shot(V, T, colour, eye, up=(0.0, 0.0, 1.0), points=False):
    """The mesh seen from eye toward its middle, orthographic, SIZE px wide: (h, w, 3) floats; with points, also
    the point each pixel sees ((h, w, 3), nan where none)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(T.astype(np.uint32)))
    mid = (V.min(0) + V.max(0)) / 2
    f = mid - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    P = (V - mid) @ np.stack([r, u], 1)
    lo, hi = P.min(0), P.max(0)
    px = (hi - lo).max() / SIZE
    w, h = int(np.ceil((hi[0] - lo[0]) / px)) + 8, int(np.ceil((hi[1] - lo[1]) / px)) + 8
    X, Y = np.meshgrid(lo[0] + (np.arange(w) - 3.5) * px, hi[1] - (np.arange(h) - 3.5) * px)
    far = np.linalg.norm(V.max(0) - V.min(0))
    org = mid - f * far + X[..., None] * r + Y[..., None] * u
    rays = np.concatenate([org, np.broadcast_to(f, org.shape)], -1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    t, ids = hit["t_hit"].numpy(), hit["primitive_ids"].numpy()
    ok = np.isfinite(t)
    n = hit["primitive_normals"].numpy()
    # (lit from over the eye's left shoulder; a face turned away from the eye is a back, dark)
    light = -f + 0.5 * u - 0.4 * r
    light /= np.linalg.norm(light)
    shade = 0.25 + 0.75 * np.clip(np.abs(n @ light), 0, 1)
    depth = np.where(ok, 1.0 - 0.25 * (t - t[ok].min()) / max(np.ptp(t[ok]), 1e-6), 0)
    img = np.ones((h, w, 3))
    img[ok] = colour[ids[ok]] * (shade[ok] * depth[ok])[:, None]
    return (img, np.where(ok[..., None], org + t[..., None] * f, np.nan)) if points else img
