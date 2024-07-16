from pathlib import Path
from wgpu.gui.auto import WgpuCanvas, run
import numpy as np
import pygfx as gfx


npz_dir = Path("data/npz")


class PointCloudVis:
    def __init__(self):
        self.canvas = WgpuCanvas(size=(1200, 800))
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()

        self.axes = gfx.AxesHelper(size=1, thickness=2)
        self.scene.add(self.axes)

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
    # ll = lower left, lr = lower right, ul = upper left, ur = upper right
    points_ll = np.load(npz_dir / "bc_092b084_3_2_3_xyes_8_utm10_2019.npz")["points"]
    points_lr = np.load(npz_dir / "bc_092b084_3_2_4_xyes_8_utm10_2019.npz")["points"]
    points_ul = np.load(npz_dir / "bc_092b084_3_4_1_xyes_8_utm10_2019.npz")["points"]
    points_ur = np.load(npz_dir / "bc_092b084_3_4_2_xyes_8_utm10_2019.npz")["points"]
    points = np.concatenate([points_ll, points_lr, points_ul, points_ur])
    n_points = points.shape[0]
    print(f"Loaded {n_points} points.")
    print(points.dtype)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    print(mins)
    print(maxs)
    print("")

    print("Collecting a random subset of points...")
    n_points_subset = min(10000000, n_points)
    indices = np.arange(points.shape[0])
    indices = np.random.choice(indices, size=n_points_subset, replace=False)
    points_subset = points[indices]
    print("")

    vis = PointCloudVis()
    vis.add_points(points_subset)
    vis.run()


if __name__ == "__main__":
    main()
