from pathlib import Path
from wgpu.gui.auto import WgpuCanvas, run
import numpy as np
import pygfx as gfx


npz_dir = Path("data/npz")


def lat_lon_alt_to_cartesian(lat_lon_alt, ref_lat_lon):
    # Constants
    a = 6378137.0  # Semi-major axis of the Earth (meters)
    e2 = 6.69437999014e-3  # Square of eccentricity

    # Split the input array into lat, lon, and alt
    lat = lat_lon_alt[:, 0]
    lon = lat_lon_alt[:, 1]
    alt = lat_lon_alt[:, 2]

    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_lat_rad = np.radians(ref_lat_lon[0])
    ref_lon_rad = np.radians(ref_lat_lon[1])

    # Calculate N, the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

    # Cartesian coordinates in ECEF (Earth-Centered, Earth-Fixed)
    X_ecef = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    Y_ecef = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    Z_ecef = (N * (1 - e2) + alt) * np.sin(lat_rad)

    # Reference point in ECEF
    N_ref = a / np.sqrt(1 - e2 * np.sin(ref_lat_rad)**2)
    X_ref = (N_ref + 0) * np.cos(ref_lat_rad) * np.cos(ref_lon_rad)
    Y_ref = (N_ref + 0) * np.cos(ref_lat_rad) * np.sin(ref_lon_rad)
    Z_ref = (N_ref * (1 - e2) + 0) * np.sin(ref_lat_rad)

    # Translate ECEF coordinates to reference point
    dX = X_ecef - X_ref
    dY = Y_ecef - Y_ref
    dZ = Z_ecef - Z_ref

    # Rotation matrix for ECEF to ENU (East-North-Up) conversion
    sin_lat_ref = np.sin(ref_lat_rad)
    cos_lat_ref = np.cos(ref_lat_rad)
    sin_lon_ref = np.sin(ref_lon_rad)
    cos_lon_ref = np.cos(ref_lon_rad)

    R = np.array([
        [-sin_lon_ref, cos_lon_ref, 0],
        [-sin_lat_ref*cos_lon_ref, -sin_lat_ref*sin_lon_ref, cos_lat_ref],
        [cos_lat_ref*cos_lon_ref, cos_lat_ref*sin_lon_ref, sin_lat_ref]
    ])

    enu_coords = np.dot(R, np.array([dX, dY, dZ]))

    # Convert to kilometers
    result = 0.001 * enu_coords.T
    return result.astype(np.float32)


class PointCloudVis:
    def __init__(self):
        self.canvas = WgpuCanvas(size=(1200, 800))
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()

        # self.axes = gfx.AxesHelper(size=1, thickness=2)
        # self.scene.add(self.axes)

        self.camera = gfx.PerspectiveCamera(fov=60, aspect=1, depth_range=(0.1, 1000.0))
        # self.camera.show_object(self.axes, up=(0, 0, 1))
        self.controller = gfx.OrbitController(
            self.camera, register_events=self.renderer
        )

        self.ambient_light = gfx.AmbientLight()
        self.scene.add(self.ambient_light)

    def add_points(self, points):
        n_points = points.shape[0]
        assert points.shape == (n_points, 3)
        geometry = gfx.Geometry(positions=points)
        material = gfx.PointsMaterial(color=(1, 0, 0), size=0.75)
        points = gfx.Points(geometry, material)
        self.scene.add(points)
        self.camera.show_object(points, up=(0, 0, 1))

    def animate(self):
        self.renderer.render(self.scene, self.camera)

    def run(self):
        self.canvas.request_draw(self.animate)
        run()


def main():
    print("Loading points...")
    points = np.load(npz_dir / "bc_092b084_3_4_2_xyes_8_utm10_2019.npz")["points"]
    n_points = points.shape[0]
    # Switch from lon/lat to lat/lon
    points = points[:, [1, 0, 2]]
    print(f"Loaded {n_points} points.")
    print(points.dtype)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    print(mins)
    print(maxs)
    print("")

    print("Collecting a random subset of points...")
    n_points_subset = 10000000
    indices = np.arange(points.shape[0])
    indices = np.random.choice(indices, size=n_points_subset, replace=False)
    points_subset = points[indices]
    print("")

    print("Converting to XYZ...")
    lat_lon_center = 0.5 * (mins[:2] + maxs[:2])
    points_subset_xyz = lat_lon_alt_to_cartesian(points_subset, lat_lon_center)
    mins = np.min(points_subset_xyz, axis=0)
    maxs = np.max(points_subset_xyz, axis=0)
    print(mins)
    print(maxs)

    vis = PointCloudVis()
    vis.add_points(points_subset_xyz)
    vis.run()


if __name__ == "__main__":
    main()
