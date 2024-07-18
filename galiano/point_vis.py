from pathlib import Path
from functools import partial
from wgpu.gui.auto import WgpuCanvas, run
import numpy as np
import pygfx as gfx
import jax
import jax.numpy as jnp

from lat_lon import lat_lon_alt_to_cartesian
from quantization import make_bin_def, get_bin_index, dequantize
from spatial_sort import make_spatial_sort, insert_points_dense, find_nearest_neighbor
from image_io import write_array_as_image


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


@partial(jax.jit, static_argnums=(0,))
def splat_tile(
    tile_size: tuple[int, int], # 16 x 16 in Kerbl et. al "3D Gaussian Splatting..." (2023)
    tile_bounds: jax.Array,  # [[x_min, y_min], [x_max, y_max]] in the projection plane
    means: jax.Array,  # shape (max_splats, 2), in projected space
    variances: jax.Array,  # shape (max_splats,), in projected space
):
    n_splats = means.shape[0]
    assert means.shape == (n_splats, 2)
    assert variances.shape == (n_splats,)
    assert tile_bounds.shape == (2, 2)

    # Construct pixel positions in the projection plane (z = 1).
    pixel_size = (tile_bounds[1] - tile_bounds[0]) / jnp.array(tile_size)
    pixel_x = jnp.linspace(
        tile_bounds[0, 0], tile_bounds[1, 0], tile_size[0], endpoint=False
    )
    pixel_y = jnp.linspace(
        tile_bounds[0, 1], tile_bounds[1, 1], tile_size[1], endpoint=False
    )
    pixel_x = pixel_x + 0.5 * pixel_size[0]
    pixel_y = pixel_y + 0.5 * pixel_size[1]
    pixel_x, pixel_y = jnp.meshgrid(pixel_x, pixel_y, indexing="ij")
    pixel_xy = jnp.stack([pixel_x, pixel_y], axis=-1)
    assert pixel_xy.shape == (tile_size[0], tile_size[1], 2)

    # Initialize buffers.
    rgba_buffer = jnp.zeros(tile_size + (4,), dtype=jnp.float32)

    base_color = jnp.array([0.5, 0.2, 0.1, 1.0])

    def body(state):
        idx, rgba_buffer = state

        mean = means[idx]
        variance = variances[idx]

        # Compute the distance from the mean to each pixel.
        distance = jnp.linalg.norm(pixel_xy - mean, axis=-1)
        assert distance.shape == tile_size

        # Compute the alpha for each pixel.
        # Clip to zero after 2 standard deviations.
        alpha = jnp.exp(-0.5 * distance**2 / variance)
        alpha = jnp.where(distance < 2.0 * jnp.sqrt(variance), alpha, 0.0)
        assert alpha.shape == tile_size

        # Premultiply the color by the alpha.
        color = base_color * alpha[:, :, None]
        assert color.shape == tile_size + (4,)

        # Update the buffer. We use the formula R = S + D * (1 - S_A), where R is RGBA for the
        # result, S is RGBA for the source (top layer), and D is RGBA for the destination (bottom
        # layer). S_A refers to the alpha channel of the source. This formula assumes that the RGB
        # components have been multiplied by the alpha channel.
        s_alpha = rgba_buffer[..., 3, None]
        rgba_buffer = rgba_buffer + color * (1.0 - s_alpha)

        return (idx + 1, rgba_buffer)

    def cond(state):
        # Continue as long as there are more splats to process and some pixel isn't opaque.
        idx, rgba_buffer = state
        return jnp.logical_and(
            idx < n_splats,
            jnp.any(rgba_buffer[..., 3] < 0.999),
        )

    # Work through the splats from front to back.
    state = (jnp.array(0, dtype=jnp.int32), rgba_buffer)
    n_processed_splats, rgba_buffer = jax.lax.while_loop(cond, body, state)

    # TODO Do I divide by alpha?
    return rgba_buffer


def render_splats(
    tile_size: tuple[int, int],
    n_tiles: tuple[int, int],
    points,
    viewer_xyz,
    viewer_frame,
    viewer_fov_x,
    viewer_fov_y,
    near,
    far,
    splat_radius,
):
    # Compute tile bounds (in the image plane i.e. z = 1).
    fov_bound_x = jnp.tan(0.5 * viewer_fov_x)
    fov_bound_y = jnp.tan(0.5 * viewer_fov_y)
    tile_view_bounds = jnp.array(
        [
            [-fov_bound_x, -fov_bound_y],
            [fov_bound_x, fov_bound_y],
        ]
    )
    tile_plane_x = jnp.linspace(tile_view_bounds[0, 0], tile_view_bounds[1, 0], n_tiles[0] + 1)
    tile_plane_y = jnp.linspace(tile_view_bounds[0, 1], tile_view_bounds[1, 1], n_tiles[1] + 1)

    # Convert points to camera space.
    points = jnp.einsum("ij,kj->ki", viewer_frame, points - viewer_xyz)

    # Clip based on near and far planes.
    print(f"Starting with {points.shape[0]} points")
    print("Clipping points based on near and far planes...")
    mask = jnp.logical_and(points[:, 2] >= near, points[:, 2] <= far)
    # points = points[mask]
    # print(f"{points.shape[0]} points remaining")

    # Clip based on field of view.
    # We interpret splat radius as the standard deviation of a 2D Gaussian.
    print("Clipping points based on field of view...")
    two_sigma = 2 * splat_radius
    inverse_zs = 1.0 / points[:, 2, None]
    assert inverse_zs.shape == (points.shape[0], 1)
    print("a")
    mask = jnp.logical_and(mask, (points[:, 0] + two_sigma) * inverse_zs >= -fov_bound_x)
    print("a")
    mask = jnp.logical_and(mask, (points[:, 0] - two_sigma) * inverse_zs <= fov_bound_x)
    print("a")
    mask = jnp.logical_and(mask, (points[:, 1] + two_sigma) * inverse_zs >= -fov_bound_y)
    print("a")
    mask = jnp.logical_and(mask, (points[:, 1] - two_sigma) * inverse_zs <= fov_bound_y)
    print("a")
    points = points[mask]
    print(f"{points.shape[0]} points remaining")

    # Project the remaining points to the image plane.
    zs = points[:, 2, None]
    points_proj = jnp.concatenate([points[:, :2] / zs, zs], axis=1)

    # Compute splat size in the image plane. For now I'm assuming the splats are still circular
    # after projection to the screen, which is approximately true if they are all aligned with the
    # view direction. Later I should model the screen space splats as ellipses.
    screen_splat_radii = splat_radius / points_proj[:, 2]

    # Globally sort splats by z.
    order = jnp.argsort(points_proj[:, 2])
    points_proj = points_proj[order]

    # Assign splats to tiles.
    # Ideally I'd spawn a thread per splat, and add its index to the arrays for relevant tiles.
    # But this isn't doable in JAX. For now I'll make a giant boolean array and cumsum it to get
    # id slots. But this will use a lot of memory so I might have to do it in chunks.
    # ...

    # n_subset = 16
    # means = points_proj[:n_subset, :2]
    # variances = screen_splat_radii[:n_subset]**2

    # Per-tile sort would be here... Probably not necessary as long as my splats are all aligned.

    # Render each tile. Vmap or loop?
    means = points_proj[:, :2]
    variances = screen_splat_radii**2
    image_rgba = splat_tile(tile_size, tile_view_bounds, means, variances)
    print("image rgba", image_rgba)
    image_rgb = image_rgba[..., :3]

    return mask, image_rgb


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
    tree_points = np.load(npz_small_dir / "tree_points.npz")["points"]
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
    viewer_frame = jnp.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    # For now I just have one tile.
    image_size = (16, 16)
    aspect_ratio = image_size[1] / image_size[0]
    viewer_fov_x = 60.0 * (jnp.pi / 180.0)
    viewer_fov_y = viewer_fov_x * aspect_ratio

    print("Rendering image...")
    near = 0.001
    far = 10.0
    splat_size = 0.001
    tile_size = (32, 32)
    n_tiles = (2, 2)
    mask, image = render_splats(
        tile_size,
        n_tiles,
        ground_points,
        viewer_xyz,
        viewer_frame,
        viewer_fov_x,
        viewer_fov_y,
        near,
        far,
        splat_size,
    )
    print(image.shape)
    print(jnp.min(image, axis=(0, 1)))
    print(jnp.max(image, axis=(0, 1)))
    image_path = "test.png"
    print(f"Saving to {image_path}")
    write_array_as_image(image, image_path)

    # ground_points = ground_points[mask]
    # vis = PointCloudVis(ground_points, tree_points)
    # vis.run()


if __name__ == "__main__":
    main()
