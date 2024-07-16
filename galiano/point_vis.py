from pathlib import Path
from wgpu.gui.auto import WgpuCanvas, run
import numpy as np
import pygfx as gfx

from lat_lon import lat_lon_alt_to_cartesian


# npz_dir = Path("data/npz")
npz_small_dir = Path("data/npz_small")


class PointCloudVis:
    def __init__(self, ground_points, tree_points):
        self.canvas = WgpuCanvas(size=(1200, 800))
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()

        # This will be used to position the view rendering camera.
        # ref_lat_lon = np.array([48.875, -123.325])
        # viewer_lat_lon_alt = np.array([48.875, -123.325, 1.0]).reshape(1, 3)
        # viewer_xyz = lat_lon_alt_to_cartesian(viewer_lat_lon_alt, ref_lat_lon)
        # self.axes = gfx.AxesHelper(size=1, thickness=2)
        # self.axes.world.position = viewer_xyz
        # self.scene.add(self.axes)

        self.camera = gfx.PerspectiveCamera(fov=60, aspect=1, depth_range=(0.1, 1000.0))
        self.controller = gfx.OrbitController(
            self.camera, register_events=self.renderer
        )

        self.ambient_light = gfx.AmbientLight()
        self.scene.add(self.ambient_light)

        self.add_points(ground_points, (0.0, 0.0, 1.0))
        self.add_points(tree_points, (0.0, 1.0, 0.0))

        self.grid = gfx.Grid(
            None,
            gfx.GridMaterial(
                major_step=0.1,
                thickness_space="world",
                major_thickness=0.002,
                infinite=True,
            ),
            orientation="xy",
        )
        self.scene.add(self.grid)


    def add_points(self, points, color):
        n_points = points.shape[0]
        assert points.shape == (n_points, 3)
        geometry = gfx.Geometry(positions=points)
        material = gfx.PointsMaterial(color=color, size=0.75)
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
    tree_points=np.load(npz_small_dir / "tree_points.npz")["points"]
    ground_points = np.load(npz_small_dir / "ground_points.npz")["points"]
    n_tree_points = tree_points.shape[0]
    n_ground_points = ground_points.shape[0]
    print(f"{n_tree_points} tree points")
    print(f"{n_ground_points} ground points")

    vis = PointCloudVis(ground_points, tree_points)
    vis.run()


if __name__ == "__main__":
    main()
