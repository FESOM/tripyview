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
from .sub_mesh import *


def warmup_compute_x_nghbr_x():
    """
    Pre-compile all Numba kernels once using a tiny dummy mesh.
    This ensures that subsequent calls run instantly at full speed.
    """

    # Dummy triangle mesh with shape (1,3)
    # Must be int32 and C-contiguous
    print(" -> Warming up Numba neighborhood connectivity")
    dummy_e = np.ascontiguousarray(
        np.array([[0, 1, 2]], dtype=np.int32)
    )

    # ---- Trigger compilation of each kernel ----
    try:
        _, _ = njit_compute_n_nghbr_e(dummy_e)
    except Exception as e:
        print("Warning: nod_in_elem warmup failed:", e)

    try:
        _, _ = njit_compute_e_nghbr_e(dummy_e)
    except Exception as e:
        print("Warning: elem_in_elem warmup failed:", e)

    try:
        _, _ = njit_compute_n_nghbr_n(dummy_e)
    except Exception as e:
        print("Warning: nod_in_nod warmup failed:", e)

    try:
        _ = njit_compute_boundary_edges(dummy_e)
    except Exception as e:
        print("Warning: boundary_edges warmup failed:", e)




def warmup_grid_kernels():
    print(" -> Warming up Numba grid kernel")
    rmat = njit_grid_rotmat(30., 40., 50.)
    x,y,z = njit_grid_cart3d(np.array([0.1]), np.array([0.2]), 1.0)

    # these need arrays
    lon = np.array([10.0])
    lat = np.array([20.0])
    _   = njit_grid_g2r(30.,40.,50., lon, lat)
    _   = njit_grid_r2g(30.,40.,50., lon, lat)

 

 
def warmup_vec_r2g_kernels():
    print(" -> Warming up Numba kernels for vec_r2g")

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


def warmup_lsmask():
    """
    Warm-up for lsmask Numba kernels:
    - njit_compute_boundary_edges
    - njit_build_adjacency
    - njit_trace_loops
    - njit_build_polygons

    Only compiles kernels; does not compute real polygons.
    """
    print(" -> Warming up Numba lsmask compute")
    n2dn = 6

    # Node coordinates (simple 2x3 grid)
    n_x = np.array([0.0, 1.0, 2.0,   0.0, 1.0, 2.0], dtype=np.float32)
    n_y = np.array([0.0, 0.0, 0.0,   1.0, 1.0, 1.0], dtype=np.float32)

    # Triangles (4 elems) defined by (node1, node2, node3)
    # Here a simple quad split into triangles
    #
    #  (0)----(1)----(2)
    #   |   /  |    / |
    #   |  /   |   /  |
    #  (3)----(4)----(5)
    #
    e_i = np.array([
        [0, 1, 4],   # triangle 0
        [0, 4, 3],   # triangle 1
        [1, 2, 5],   # triangle 2
        [1, 5, 4],   # triangle 3
    ], dtype=np.int32)

    # 1. boundary edges (this kernel is already warm in many cases)
    bnde = njit_compute_boundary_edges(e_i)
    bnde_nodes = np.unique(bnde.ravel())
    nbnde_nodes = bnde_nodes.size
    mapping = -np.ones(n_x.size, dtype=np.int32)
    for ii, gg in enumerate(bnde_nodes): mapping[gg] = ii
    
    # 2. adjacency builder
    adj = njit_lsmask_build_adjacency(bnde, mapping, nbnde_nodes)

    # 3. loop tracer
    loops = njit_lsmask_trace_loops(adj)
    
    # 4. warmup find_period_crossings
    n_x = np.array([-180, 180, 75], dtype=np.float32)
    _ = njit_find_period_crossings(n_x, 180.0)


def warmup_smoothing_kernels():
    print(" -> Warming up Numba smoothers")

    # Tiny dummy node setup
    n2dn = 5
    n_x = np.linspace(0.0, 1.0, n2dn)
    n_y = np.linspace(0.0, 1.0, n2dn)

    # Node neighbours (ring)
    n_nghbr_n = np.array([ [1, 4, -1], [0, 2, -1], [1, 3, -1], [2, 4, -1], [3, 0, -1] ], dtype=np.int32)

    nINn_num = np.array([2, 2, 2, 2, 2], dtype=np.int32)

    data_n = np.random.rand(n2dn).astype(np.float64)

    # Weak boxes: none
    weak_boxes_empty = np.zeros((0, 5), dtype=np.float64)

    # Node smoothing warm-up
    _ = njit_node_smoothing(n2dn, n_x, n_y, n_nghbr_n, nINn_num,  data_n,
                            1, weak_boxes_empty, 1.0)

    # Tiny dummy element setup
    n2de = 4

    # Element centroids
    elem_x = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
    elem_y = np.array([0.1, 0.3, 0.7, 0.9], dtype=np.float64)

     
    e_nghbr_e = np.array([
                        [1, 3, -1],   # elem 0 neighbors: 1 and 3
                        [0, 2, -1],   # elem 1 neighbors: 0 and 2
                        [1, 3, -1],   # elem 2 neighbors: 1 and 3
                        [0, 2, -1],   # elem 3 neighbors: 0 and 2
                        ], dtype=np.int32)

    eINe_num = np.array([2, 2, 2, 2], dtype=np.int32)

    data_e = np.random.rand(n2de).astype(np.float64)

    # Element smoothing warm-up
    _ = njit_elem_smoothing(n2de, elem_x, elem_y, e_nghbr_e, eINe_num, data_e,
                            1, weak_boxes_empty, 1.0)



# Execute warm-up at module import
warmup_compute_x_nghbr_x()
warmup_grid_kernels()
warmup_vec_r2g_kernels()
warmup_lsmask()
warmup_smoothing_kernels()
