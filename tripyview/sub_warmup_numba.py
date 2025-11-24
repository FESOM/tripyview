"""
Warm-up JIT compilation for Numba-accelerated kernels in tripyview.

This module is imported automatically and triggers one lightweight call
to each Numba @njit function with a tiny dummy mesh. This eliminates
slow first-call warm-up costs and ensures that all kernels are compiled
and ready at full speed when the user needs them.

Numba caching (`cache=True`) must be enabled on every @njit function.
"""

import numpy as np

# Import all Numba kernels you want to warm-up
from .sub_mesh import (
    njit_compute_boundary_edges,
    njit_compute_nINe, 
    njit_compute_eINe,
    njit_compute_nINn, 
    njit_grid_rotmat, 
    njit_grid_cart3d, 
    njit_grid_g2r, 
    njit_grid_r2g,
    njit_vec_r2g_0d,
    njit_vec_r2g_123d,
    njit_grid_rotmat,
    )


def warmup_compute_xINx():
    """
    Pre-compile all Numba kernels once using a tiny dummy mesh.
    This ensures that subsequent calls run instantly at full speed.
    """

    # Dummy triangle mesh with shape (1,3)
    # Must be int32 and C-contiguous
    dummy_e = np.ascontiguousarray(
        np.array([[0, 1, 2]], dtype=np.int32)
    )

    # ---- Trigger compilation of each kernel ----
    try:
        njit_compute_nINe(dummy_e)
    except Exception as e:
        print("Warning: nod_in_elem warmup failed:", e)

    try:
        njit_compute_eINe(dummy_e)
    except Exception as e:
        print("Warning: elem_in_elem warmup failed:", e)

    try:
        njit_compute_nINn(dummy_e)
    except Exception as e:
        print("Warning: nod_in_nod warmup failed:", e)

    try:
        njit_compute_boundary_edges(dummy_e)
    except Exception as e:
        print("Warning: boundary_edges warmup failed:", e)




def warmup_grid_kernels():
    rmat = njit_grid_rotmat(30., 40., 50.)
    x,y,z = njit_grid_cart3d(np.array([0.1]), np.array([0.2]), 1.0)

    # these need arrays
    lon = np.array([10.0])
    lat = np.array([20.0])
    _   = njit_grid_g2r(30.,40.,50., lon, lat)
    _   = njit_grid_r2g(30.,40.,50., lon, lat)

 

 
def warmup_vec_r2g_kernels():
    print("Warming up Numba kernels for vec_r2g...")

    # ---------------------------------------------------------
    # 1. create small dummy inputs
    # ---------------------------------------------------------
    npts = 4
    nd   = 2
    nt   = 2

    lon_rad  = np.linspace(0, 1, npts)
    lat_rad  = np.linspace(0, 1, npts)
    rlon_rad = np.linspace(0, 1, npts)
    rlat_rad = np.linspace(0, 1, npts)

    u1 = np.ones(npts)
    v1 = np.ones(npts)

    u2 = np.ones((nd, npts))
    v2 = np.ones((nd, npts))

    u3 = np.ones((nt, nd, npts))
    v3 = np.ones((nt, nd, npts))

    # Rotation matrix
    rmat = np.eye(3, dtype=np.float64)

    # ---------------------------------------------------------
    # 2. warm up kernels
    # ---------------------------------------------------------

    # warm 0d kernel
    _ = njit_vec_r2g_0d(
        rmat,
        np.sin(lat_rad), np.cos(lat_rad),
        np.sin(lon_rad), np.cos(lon_rad),
        np.sin(rlat_rad), np.cos(rlat_rad),
        np.sin(rlon_rad), np.cos(rlon_rad),
        u1, v1,
    )

    # warm 1D path
    _ = njit_vec_r2g_123d(
        rmat, lon_rad, lat_rad, rlon_rad, rlat_rad, u1, v1
    )

    # warm 2D path
    _ = njit_vec_r2g_123d(
        rmat, lon_rad, lat_rad, rlon_rad, rlat_rad, u2, v2
    )

    # warm 3D path
    _ = njit_vec_r2g_123d(
        rmat, lon_rad, lat_rad, rlon_rad, rlat_rad, u3, v3
    )

    print(" → Numba warm-up complete.")


# Execute warm-up at module import
warmup_compute_xINx()
warmup_grid_kernels()
warmup_vec_r2g_kernels()
