"""
pyvista_interactive_view_with_rotation_history.py

Requirements:
    pip install pyvista numpy

Usage:
    python pyvista_interactive_view_with_rotation_history.py

Description:
    1) Loads voxel_grid.bin (written by your C++ code).
    2) Interprets the 3D array shape as (Z, Y, X).
    3) Extracts top percentile of brightness.
    4) Applies an additional Euler rotation to the entire cloud (user-defined).
    5) Displays them interactively in a PyVista window,
       so you can orbit, zoom, and pan with the mouse.
    6) (Screenshots disabled by default)
"""

import os
import re
import math
import json
import argparse
from pathlib import Path
import numpy as np
import pyvista as pv
import requests


def load_voxel_grid(filename):
    """
    Reads a voxel grid from a binary file with the following layout:
      1) int32: N (size of the NxNxN grid)
      2) float32: voxel_size
      3) N*N*N float32: the voxel data in row-major order
    Returns:
       voxel_grid (N x N x N),
       voxel_size
    """
    with open(filename, "rb") as f:
        # read N
        raw = f.read(4)
        N = np.frombuffer(raw, dtype=np.int32)[0]

        # read voxel_size
        raw = f.read(4)
        voxel_size = np.frombuffer(raw, dtype=np.float32)[0]

        # read the voxel data
        count = N*N*N
        raw = f.read(count*4)
        data = np.frombuffer(raw, dtype=np.float32)
        voxel_grid = data.reshape((N, N, N))

    return voxel_grid, voxel_size


def extract_top_percentile_z_up(voxel_grid, voxel_size, grid_center,
                                percentile=99.5, use_hard_thresh=False, hard_thresh=700):
    """
    Extract the top 'percentile' bright voxels (or above 'hard_thresh').
    We interpret the array shape as (Z, Y, X).

    index: voxel[z, y, x]

    We'll produce an Nx3 array of points in (x, y, z).
    Then we'll produce intensities as a separate array.
    """
    N = voxel_grid.shape[0]  # assume shape is (N,N,N)
    half_side = (N * voxel_size) * 0.5
    grid_min = grid_center - half_side

    # Flatten to find threshold
    flat_vals = voxel_grid.ravel()
    if use_hard_thresh:
        thresh = hard_thresh
    else:
        thresh = np.percentile(flat_vals, percentile)

    coords = np.argwhere(voxel_grid > thresh)
    if coords.size == 0:
        print(f"No voxels above threshold {thresh}. Nothing to display.")
        return None, None

    intensities = voxel_grid[coords[:, 0], coords[:, 1], coords[:, 2]]

    # Because we're now treating 0 -> z, 1 -> y, 2 -> x:
    z_idx = coords[:, 0] + 0.5
    y_idx = coords[:, 1] + 0.5
    x_idx = coords[:, 2] + 0.5

    # Convert to world coords
    x_world = grid_min[0] + x_idx * voxel_size
    y_world = grid_min[1] + y_idx * voxel_size
    z_world = grid_min[2] + z_idx * voxel_size

    points = np.column_stack((x_world, y_world, z_world))
    return points, intensities


def rotation_matrix_xyz(rx_deg, ry_deg, rz_deg):
    """
    Build a rotation matrix (3x3) for Euler angles (rx, ry, rz) in degrees,
    applied in X->Y->Z order. That is:
      R = Rz(rz) * Ry(ry) * Rx(rx)
    so we rotate first by rx around X, then ry around Y, then rz around Z.
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # Rx
    Rx = np.array([
        [1,   0,   0],
        [0,  cx, -sx],
        [0,  sx,  cx]
    ], dtype=np.float32)

    # Ry
    Ry = np.array([
        [ cy,  0,  sy],
        [  0,  1,   0],
        [-sy,  0,  cy]
    ], dtype=np.float32)

    # Rz
    Rz = np.array([
        [ cz, -sz,  0],
        [ sz,  cz,  0],
        [  0,   0,  1]
    ], dtype=np.float32)

    # Combined: Rz * Ry * Rx
    Rtemp = Rz @ Ry
    Rfinal = Rtemp @ Rx
    return Rfinal


def _fetch_elevation_grid_opentopo(lat_center, lon_center, lat_span, lon_span, grid_size=128, dataset="mapzen"):
    """
    Fetch a grid_size x grid_size elevation grid (meters) from OpenTopoData around
    (lat_center, lon_center) covering the provided lat/lon spans.

    dataset options: 'mapzen' (global), 'srtm30m' (US), 'eudem25m' (EU) etc.
    """
    # Build query locations row-major from top-left to bottom-right for human sanity
    lat_min = lat_center - lat_span * 0.5
    lat_max = lat_center + lat_span * 0.5
    lon_min = lon_center - lon_span * 0.5
    lon_max = lon_center + lon_span * 0.5

    lats = np.linspace(lat_max, lat_min, grid_size)  # top to bottom
    lons = np.linspace(lon_min, lon_max, grid_size)  # left to right

    locations = [f"{lat:.6f},{lon:.6f}" for lat in lats for lon in lons]

    elevations = []
    for i in range(0, len(locations), 100):
        chunk = locations[i:i+100]
        url = f"https://api.opentopodata.org/v1/{dataset}?locations={'|'.join(chunk)}"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"OpenTopoData HTTP {r.status_code}")
            data = r.json()
            # Some results may have null elevation; coerce to 0
            elevations.extend([float(res.get("elevation") or 0.0) for res in data.get("results", [])])
        except Exception as e:
            raise RuntimeError(f"Failed fetching OpenTopoData: {e}")

    if len(elevations) != grid_size * grid_size:
        raise RuntimeError("Unexpected elevation sample count")

    grid = np.array(elevations, dtype=np.float32).reshape((grid_size, grid_size))
    # Replace NaNs just in case
    grid = np.nan_to_num(grid, nan=0.0)
    return grid


def add_terrain_overlay_fit_grid(plotter, grid_min, grid_max, lat, lon,
                                 grid_size=128, opacity=0.5, vertical_scale=1.0,
                                 base_z=0.0, cmap="terrain", dataset="mapzen"):
    """
    Fetch elevation around (lat, lon) and render a terrain mesh that fits exactly
    within [grid_min.x..grid_max.x] x [grid_min.y..grid_max.y].

    - grid_min, grid_max: np.array([x,y,z]) bounds of the voxel grid in world units
    - vertical_scale: multiply elevation meters to match your scene Z scale
    - base_z: world Z offset at which to place elevation=0
    """
    width_m = float(grid_max[0] - grid_min[0])
    height_m = float(grid_max[1] - grid_min[1])

    # Convert XY extents (meters) to approximate lat/lon spans (degrees)
    # 1 degree latitude ~ 111 km, longitude scaled by cos(latitude)
    lat_span_deg = max(1e-6, (height_m / 1000.0) / 111.0)
    lon_span_deg = max(1e-6, (width_m / 1000.0) / (111.0 * max(1e-6, math.cos(math.radians(lat)))))

    elev_grid = _fetch_elevation_grid_opentopo(lat, lon, lat_span_deg, lon_span_deg,
                                               grid_size=grid_size, dataset=dataset)

    # Build a PyVista UniformGrid matching XY extents
    spacing_x = width_m / max(1, grid_size - 1)
    spacing_y = height_m / max(1, grid_size - 1)
    terrain = pv.UniformGrid(
        dimensions=(grid_size, grid_size, 1),
        spacing=(spacing_x, spacing_y, 1.0),
        origin=(float(grid_min[0]), float(grid_min[1]), float(base_z))
    )

    # Apply elevation as scalars and warp
    terrain["elevation"] = (elev_grid.T.flatten(order="C") * float(vertical_scale))
    # Warp along +Z by the elevation scalar
    warped = terrain.warp_by_scalar("elevation", factor=1.0)

    # Color by elevation (colormap); smooth shading for nicer look
    plotter.add_mesh(
        warped,
        scalars="elevation",
        cmap=cmap,
        opacity=float(opacity),
        smooth_shading=True,
        show_edges=False,
    )

    min_e, max_e = float(elev_grid.min()), float(elev_grid.max())
    print(f"[terrain] OpenTopoData loaded {grid_size}x{grid_size} (range {min_e:.1f}..{max_e:.1f} m)")


def get_next_image_index(folder, prefix="voxel_", suffix=".png"):
    """
    Scan 'folder' for files named like 'voxel_XXXX.png'.
    Find the largest XXXX as int, return that + 1.
    If none found, return 1.
    """
    if not os.path.exists(folder):
        return 1

    pattern = re.compile(rf"^{prefix}(\d+){suffix}$")
    max_index = 0
    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if match:
            idx = int(match.group(1))
            if idx > max_index:
                max_index = idx
    return max_index + 1


def main():
    ap = argparse.ArgumentParser(description="View voxel_grid.bin with PyVista")
    ap.add_argument("--bin", dest="bin_path", default="voxel_grid.bin", help="Path to voxel_grid.bin to view")
    ap.add_argument("--json", dest="meta_json", default=None, help="Optional voxel_grid.json for grid_center and metadata")
    ap.add_argument("--center", dest="center", default=None, help="Grid center as 'x,y,z' (overrides JSON)")
    ap.add_argument("--percentile", type=float, default=99.9, help="Top percentile to show (e.g., 99.9)")
    ap.add_argument("--rotate", default="90,270,0", help="Euler rotation degrees 'rx,ry,rz' applied to points")
    # Terrain overlay options (off by default; enable with --terrain)
    ap.add_argument("--terrain", default=None, help="Terrain center as 'lat,lon' (e.g., '37.7749,-122.4194' for SF)")
    ap.add_argument("--terrain-grid", type=int, default=128, help="Terrain grid resolution N (NxN), default 128")
    ap.add_argument("--terrain-opacity", type=float, default=0.45, help="Terrain opacity (0..1)")
    ap.add_argument("--terrain-scale", type=float, default=1.0, help="Vertical scale factor for elevation (meters -> world units)")
    ap.add_argument("--terrain-z", type=float, default=0.0, help="World Z offset for terrain base (elevation=0)")
    ap.add_argument("--terrain-cmap", default="terrain", help="Matplotlib colormap name for terrain colors")
    ap.add_argument("--terrain-dataset", default="mapzen", help="OpenTopoData dataset (e.g., mapzen, eudem25m, srtm30m)")
    args = ap.parse_args()

    bin_path = Path(args.bin_path)
    if not bin_path.exists():
        raise SystemExit(f"voxel grid not found: {bin_path}")

    # 1) Load the voxel grid
    voxel_grid, vox_size = load_voxel_grid(str(bin_path))
    print("Loaded voxel grid:", voxel_grid.shape, "voxel_size=", vox_size)
    print("Max voxel value:", float(voxel_grid.max()))

    # 2) Resolve grid center
    grid_center = None
    if args.meta_json:
        jp = Path(args.meta_json)
        if not jp.exists():
            raise SystemExit(f"metadata json not found: {jp}")
        meta = json.load(open(jp))
        if isinstance(meta.get("grid_center"), (list, tuple)) and len(meta["grid_center"]) == 3:
            grid_center = np.array(meta["grid_center"], dtype=np.float32)
    elif (bin_path.parent / "voxel_grid.json").exists():
        meta = json.load(open(bin_path.parent / "voxel_grid.json"))
        if isinstance(meta.get("grid_center"), (list, tuple)) and len(meta["grid_center"]) == 3:
            grid_center = np.array(meta["grid_center"], dtype=np.float32)
    if args.center:
        vals = tuple(float(x) for x in args.center.split(","))
        if len(vals) == 3:
            grid_center = np.array(vals, dtype=np.float32)
    if grid_center is None:
        grid_center = np.array([0, 0, 300], dtype=np.float32)
    print("Grid center:", grid_center.tolist())

    # 3) Extract top percentile (Z-up)
    percentile_to_show = float(args.percentile)
    points, intensities = extract_top_percentile_z_up(
        voxel_grid,
        voxel_size=vox_size,
        grid_center=grid_center,
        percentile=percentile_to_show,
        use_hard_thresh=False,
        hard_thresh=700
    )
    if points is None:
        return  # nothing to show

    # 4) Optional rotation
    # e.g. rotate to fix orientation
    rx_deg, ry_deg, rz_deg = (float(x) for x in args.rotate.split(","))

    R = rotation_matrix_xyz(rx_deg, ry_deg, rz_deg)  # shape (3,3)
    points_rot = points @ R.T
    # 5) PyVista Plotter (interactive)
    plotter = pv.Plotter(off_screen=True,)
    plotter.set_background("white")
    plotter.enable_terrain_style()


    # Convert to PolyData with scalars
    cloud = pv.PolyData(points_rot)
    cloud["intensity"] = intensities

    # Add points
    plotter.add_points(
        cloud,
        scalars="intensity",
        cmap="hot",
        point_size=4.0,
        render_points_as_spheres=True,
        opacity=0.1,
    )

    plotter.add_scalar_bar(
        title="Brightness",
        n_labels=5
    )

    # Show interactive window (no auto-screenshot)
    plotter.show(window_size=[1920, 1080], auto_close=False)

    plotter = pv.Plotter(off_screen=False, )
    plotter.set_background("white")
    plotter.enable_terrain_style()

    # Convert to PolyData with scalars
    cloud = pv.PolyData(points_rot)
    cloud["intensity"] = intensities

    # Add points
    plotter.add_points(
        cloud,
        scalars="intensity",
        cmap="hot",
        point_size=4.0,
        render_points_as_spheres=True,
        opacity=0.05,
    )

    # Optionally add a terrain overlay that fits the grid XY extents
    if args.terrain:
        try:
            lat_str, lon_str = str(args.terrain).split(",")
            lat = float(lat_str.strip())
            lon = float(lon_str.strip())
            # Compute voxel grid bounds in world units
            N = int(voxel_grid.shape[0])
            half_side = (N * vox_size) * 0.5
            grid_min = grid_center - half_side
            grid_max = grid_center + half_side
            add_terrain_overlay_fit_grid(
                plotter,
                grid_min=grid_min,
                grid_max=grid_max,
                lat=lat,
                lon=lon,
                grid_size=max(16, int(args.terrain_grid)),
                opacity=float(args.terrain_opacity),
                vertical_scale=float(args.terrain_scale),
                base_z=float(args.terrain_z),
                cmap=str(args.terrain_cmap),
                dataset=str(args.terrain_dataset),
            )
        except Exception as e:
            print(f"[terrain] Error: {e}")

    # plotter.add_scalar_bar(
    #     title="Brightness",
    #     n_labels=5
    # )
    print("[Done]")
    plotter.show(window_size=[1920, 1080], auto_close=False)




if __name__ == "__main__":
    main()
