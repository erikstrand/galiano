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


def render_splats(points, viewer_xyz, viewer_frame, viewer_fov_x, viewer_fov_y, near, far):
    # Convert points to camera space.
    ground_points_camera = jnp.einsum("ij,kj->ki", viewer_frame, points - viewer_xyz)

    # Convert points to projective space.
    zs = ground_points_camera[:, 2][:, None]
    ground_points_proj = jnp.concatenate([ground_points_camera[:, :2] / zs, zs], axis=1)

    # Clip.
    fov_bound_x = jnp.tan(0.5 * viewer_fov_x)
    fov_bound_y = jnp.tan(0.5 * viewer_fov_y)
    mask = jnp.logical_and(
        ground_points_proj[:, 0] >= -fov_bound_x,
        ground_points_proj[:, 0] <= fov_bound_x,
    )
    mask = jnp.logical_and(mask, ground_points_proj[:, 1] >= -fov_bound_y)
    mask = jnp.logical_and(mask, ground_points_proj[:, 1] <= fov_bound_y)
    mask = jnp.logical_and(mask, ground_points_proj[:, 2] >= near)
    mask = jnp.logical_and(mask, ground_points_proj[:, 2] <= far)

    return mask


class PointCloudVis:
    def __init__(self, ground_points, tree_points):
        self.canvas = WgpuCanvas(size=(1200, 800))
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()

        self.axes = gfx.AxesHelper(size=0.1, thickness=2)
        # self.axes.world.position = viewer_xyz
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
        points = points.astype(np.float32)
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

    # Define the viewer orientation. X is left, Y is up, Z is forward. Note that it is a left
    # handed system. Here the first index identifies the vector of the frame, and the second is
    # the coordinate value in world space.
    viewer_frame = jnp.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])

    image_size = (300, 200)
    aspect_ratio = image_size[1] / image_size[0]
    viewer_fov_x = 60.0 * (jnp.pi / 180.0)
    viewer_fov_y = viewer_fov_x * aspect_ratio

    near = 0.001
    far = 10.0
    mask = render_splats(ground_points, viewer_xyz, viewer_frame, viewer_fov_x, viewer_fov_y, near, far)

    ground_points = ground_points[mask]
    vis = PointCloudVis(ground_points, tree_points)
    vis.run()


if __name__ == "__main__":
    main()
