"""scipy.ndimage's interface, run on the GPU (CuPy) where CuPy has the function: numpy in,
numpy out, so the build scripts change only their import. What CuPy lacks (find_objects, center_of_mass on
huge masks is fine either way) falls through to scipy."""
import numpy as np
from scipy import ndimage as _sp

try:
    import cupy as cp
    import cupyx.scipy.ndimage as _cu
    GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:                         # (no CuPy, or no CUDA device: plain scipy)
    GPU = False

_ON_GPU = ("distance_transform_edt", "gaussian_filter", "binary_dilation", "binary_erosion", "binary_opening",
           "binary_closing", "binary_fill_holes", "label", "affine_transform", "uniform_filter")


def _to(x):
    return cp.asarray(x) if isinstance(x, np.ndarray) else x


def _back(x):
    if isinstance(x, tuple): return tuple(_back(v) for v in x)
    return cp.asnumpy(x) if GPU and isinstance(x, cp.ndarray) else x


class _NDI:
    def __getattr__(self, name):
        if GPU and name in _ON_GPU:
            f = getattr(_cu, name)
            def run(*a, **k):
                a = [_to(v) for v in a]; k = {kk: _to(v) for kk, v in k.items()}
                if name.startswith("binary_") and k.get("iterations", 1) != 1:     # (CuPy iterates only by brute force)
                    k["brute_force"] = True
                if name == "label" and len(a) == 1 and "structure" not in k:      # (CuPy needs it spelled out)
                    k["structure"] = cp.asarray(_sp.generate_binary_structure(a[0].ndim, 1))
                return _back(f(*a, **k))
            return run
        return getattr(_sp, name)


ndi = _NDI()


def sweep(mask, R, c, co, n):
    """GPU sweep: mask rotated by R (input = R^T (o - co) + c) into an n^3 frame, every voxel at or beyond a set one
    along +z, rotated back to mask's frame."""
    if not GPU:
        rot = _sp.affine_transform(mask.astype(np.float32), R.T, offset=c - R.T @ co, output_shape=(n, n, n), order=1)
        rot = np.maximum.accumulate(rot > 0.5, axis=2)
        return _sp.affine_transform(rot.astype(np.float32), R, offset=co - R @ c, output_shape=mask.shape, order=1) > 0.5
    g = cp.asarray(mask, dtype=cp.float32)
    rot = _cu.affine_transform(g, cp.asarray(R.T), offset=cp.asarray(c - R.T @ co), output_shape=(n, n, n), order=1)
    rot = (cp.cumsum(rot > 0.5, axis=2) > 0).astype(cp.float32)
    back = _cu.affine_transform(rot, cp.asarray(R), offset=cp.asarray(co - R @ c), output_shape=mask.shape, order=1)
    return cp.asnumpy(back > 0.5)
