from pathlib import Path
from wgpu.gui.auto import WgpuCanvas, run
import numpy as np
import pygfx as gfx
import jax
import jax.numpy as jnp

from lat_lon import lat_lon_alt_to_cartesian
from quantization import make_bin_def, get_bin_index, dequantize
from spatial_sort import make_spatial_sort, insert_points_dense, find_nearest_neighbor


# npz_dir = Path("data/npz")
npz_small_dir = Path("data/npz_small")


def build_spatial_sort(points, dx=0.01):
    min_vals = jnp.min(points, axis=0)
    max_vals = jnp.max(points, axis=0)
    bounds = jnp.stack([min_vals, max_vals], axis=0)
    bin_def = make_bin_def(bounds, jnp.array(dx, dtype=jnp.float32), force_even=True)
    n_points = points.shape[0]
    spatial_sort = make_spatial_sort(int(bin_def.n_total_bins), n_points)
    spatial_sort = insert_points_dense(spatial_sort, bin_def, points, n_points)
    return bin_def, spatial_sort


def distance_point_to_line(point, line_point, line_direction):
    """
    Computes the distance between a line and a point in 2D.

    Parameters:
    - point: A numpy array representing the point (e.g., np.array([x, y])).
    - line_point: A numpy array representing a point on the line (e.g., np.array([x, y])).
    - line_direction: A numpy array representing the direction of the line (e.g., np.array([dx, dy])).

    Returns:
    - The distance between the point and the line.
    """
    # Normalize the line direction
    line_direction = line_direction / jnp.linalg.norm(line_direction)

    # Compute the vector from the line point to the point
    point_vector = point - line_point

    # Compute the projection of the point_vector onto the line_direction
    projection_length = jnp.einsum("...i,i->...", point_vector, line_direction)
    projection_vector = projection_length[..., None] * line_direction

    # Compute the perpendicular vector
    perpendicular_vector = point_vector - projection_vector

    # Compute the distance as the norm of the perpendicular vector
    distance = jnp.linalg.norm(perpendicular_vector, axis=-1)

    return distance


class PointCloudVis:
    def __init__(self, ground_points, tree_points):
        print("Building spatial sort...")
        bin_def, spatial_sort = build_spatial_sort(ground_points[:, :2])
        print(f"n bins: {bin_def.n_bins}")
        print(f"bin size: {bin_def.bin_size}")
        print(f"max bin occupancy: {jnp.max(spatial_sort.bin_counts)}")
        print("")

        viewer_xy = jnp.array([-0.54, -0.21])
        viewer_dir = jnp.array([1.0, 0.0])
        index, distance = find_nearest_neighbor(spatial_sort, bin_def, viewer_xy)
        viewer_xyz = ground_points[index] + np.array([0.0, 0.0, 0.005])

        print("grid")
        ranges = [np.arange(1, n) for n in bin_def.n_bins]
        mesh = np.meshgrid(*ranges, indexing='ij')
        corner_ij = jnp.array(np.stack(mesh, axis=-1).reshape(-1, bin_def.n_bins.size))
        print(corner_ij.shape)

        offsets = np.array([[-1, -1], [0, -1], [-1, 0], [0, 0]])
        corner_bin_ij = corner_ij[:, np.newaxis, :] + offsets[np.newaxis, :, :]
        corner_bin_idx = get_bin_index(corner_bin_ij, bin_def)
        print("corner bin idx")
        print(corner_bin_idx.shape)
        print(corner_bin_idx)

        print("dequantize")
        corner_xyz = dequantize(corner_ij, bin_def)
        print(corner_xyz)

        print("corner_distances args")
        print(corner_xyz.shape)
        print(viewer_xy.shape)
        print(viewer_dir.shape)
        corner_distances = distance_point_to_line(corner_xyz, viewer_xy, viewer_dir)
        print("corner_distances")
        print(corner_distances.shape)
        print(corner_distances)
        print("")

        mask = corner_distances < 0.5 * jnp.max(bin_def.bin_size)
        print("mask count", jnp.sum(mask))

        self.canvas = WgpuCanvas(size=(1200, 800))
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()

        self.axes = gfx.AxesHelper(size=0.1, thickness=2)
        self.axes.world.position = viewer_xyz
        self.scene.add(self.axes)

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
        material = gfx.PointsMaterial(color=color, size=1.0)
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
    print("")

    vis = PointCloudVis(ground_points, tree_points)
    vis.run()


if __name__ == "__main__":
    main()
