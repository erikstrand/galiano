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


def signed_distance_point_to_line(point, line_point, line_direction):
    """
    Computes the distance between a line and a point in 2D.

    Parameters:
    - point: A numpy array representing the point (e.g., np.array([x, y])).
    - line_point: A numpy array representing a point on the line (e.g., np.array([x, y])).
    - line_direction: A numpy array representing the direction of the line (e.g., np.array([dx, dy])).

    Returns:
    - The distance between the point and the line.
    """
    # Rotate the line direction by 90 degrees counter-clockwise to get the perpendicular direction
    perp_dir = jnp.array([line_direction[1], -line_direction[0]])

    # Normalize the perpendicular direction
    perp_dir = perp_dir / jnp.linalg.norm(perp_dir)

    # Compute the vector from the line point to the point
    point_vector = point - line_point

    # Compute the dot product of the point vector and the perpendicular direction
    signed_distance = jnp.einsum("...i,i->...", point_vector, perp_dir)
    return signed_distance


class PointCloudVis:
    def __init__(self, ground_points, tree_points):
        n_ground_points = ground_points.shape[0]
        print("Building spatial sort for ground points...")
        bin_def, spatial_sort = build_spatial_sort(ground_points[:, :2])
        print(f"n bins: {bin_def.n_bins}, total bins: {bin_def.n_total_bins}")
        print(f"bin size: {bin_def.bin_size}")
        print(f"max bin occupancy: {jnp.max(spatial_sort.bin_counts)}")
        print(spatial_sort.positions.shape)
        print(spatial_sort.bins.shape)
        print("")

        # Define the viewer position.
        viewer_xy = jnp.array([-0.54, -0.21])
        index, distance = find_nearest_neighbor(spatial_sort, bin_def, viewer_xy)
        viewer_xyz = ground_points[index] + np.array([0.0, 0.0, 0.005])

        # Define the viewer field of view.
        viewer_dir = jnp.array([1.0, 0.0])
        viewer_fov = 50.0 * (jnp.pi / 180.0)
        viewer_ray_left = jnp.array([jnp.cos(viewer_fov / 2), jnp.sin(viewer_fov / 2)])
        viewer_ray_right = jnp.array([jnp.cos(-viewer_fov / 2), jnp.sin(-viewer_fov / 2)])

        # Compute the integer indices and XY coordinates of the corners of all bins.
        # (Ok, almost all corners. We exclude corners on the boundary.)
        ranges = [np.arange(1, n) for n in bin_def.n_bins]
        mesh = np.meshgrid(*ranges, indexing='ij')
        corner_ij = jnp.array(np.stack(mesh, axis=-1).reshape(-1, bin_def.n_bins.size))
        corner_xy = dequantize(corner_ij, bin_def)
        n_corners = corner_ij.shape[0]
        assert corner_ij.shape == (n_corners, 2)
        assert corner_xy.shape == (n_corners, 2)

        # Compute the bin coords of the bins that are adjacent to each corner.
        # (This is why we excluded boundary corners. They would have less than four neighbors.)
        offsets = np.array([[-1, -1], [0, -1], [-1, 0], [0, 0]])
        corner_bin_ij = corner_ij[:, np.newaxis, :] + offsets[np.newaxis, :, :]
        assert corner_bin_ij.shape == (n_corners, 4, 2)
        corner_bin_idx = get_bin_index(corner_bin_ij, bin_def)
        assert corner_bin_idx.shape == (n_corners, 4)

        # Find bins with corners that lie between the rays.
        corner_distances_left = signed_distance_point_to_line(corner_xy, viewer_xy, viewer_ray_left)
        corner_distances_right = signed_distance_point_to_line(corner_xy, viewer_xy, viewer_ray_right)
        mask = np.logical_and(corner_distances_left > 0.0, corner_distances_right < 0.0)
        assert mask.shape == (n_corners,)
        visible_bins = jnp.unique(corner_bin_idx[mask])
        n_visible_bins = visible_bins.shape[0]
        bin_mask = np.zeros(spatial_sort.n_total_bins, dtype=bool)
        bin_mask[visible_bins] = True

        # Find the points in the visible bins.
        # Generate the list of indices i such that visible_bins[spatial_sort.bins[i]] is True.
        tmp = np.asarray(bin_mask[spatial_sort.bins])
        assert tmp.shape == (n_ground_points,)
        visible_point_idx = tmp.nonzero()[0]
        n_visible_points = visible_point_idx.shape[0]
        print("{n_visible_points} visible points")

        # Subset points
        ground_points = ground_points[visible_point_idx]

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
    # bins = np.array([0, 0, 1, 1, 1, 3, 4, 4])
    # mask = np.array([True, False, True, True, False])

    # # Generate the list of indices i such that mask[bins[i]] is True
    # tmp = np.asarray(mask[bins])
    # print(tmp)
    # indices = tmp.nonzero()
    # print(indices)

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
