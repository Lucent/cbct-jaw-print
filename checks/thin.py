"""How thick a piece is behind each face: a ray cast inward from each face to the far side."""
import numpy as np
import open3d as o3d

def thickness(m):
    """(triangles (n, 3, 3), inward normals (n, 3), each face's thickness behind it) of m."""
    mesh = m.to_mesh()
    V, T = np.asarray(mesh.vert_properties, np.float32)[:, :3], np.asarray(mesh.tri_verts, np.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V), o3d.core.Tensor(T))
    tri = V[T].astype(np.float64)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a = np.linalg.norm(n, axis=1)
    inward = -n / (a[:, None] + 1e-12)
    start = tri.mean(1) + inward * 5e-4
    t = scene.cast_rays(o3d.core.Tensor(np.hstack([start, inward]).astype(np.float32)))["t_hit"].numpy() + 5e-4
    return tri, inward, np.where(a > 1e-8, t, np.inf)
