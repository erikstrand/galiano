from functools import partial
from pathlib import Path
from wgpu.gui.auto import WgpuCanvas, run
import jax
import jax.numpy as jnp
import numpy as np
import pygfx as gfx
import time

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


def collect_visible_splats(
    cutoff: float,
    splat_capacity,
    points,
    viewer_xyz,
    viewer_frame,
    viewer_fov_x,
    viewer_fov_y,
    near,
    far,
    splat_radius,
):
    """
    This method returns the number of visible splats and the ids of the visible splats.

    In order for all arrays to be statically shaped, the list of splat ids always has shape
    (splat_capacity,). All visible splat ids are at the beginning of the array (the rest of the
    array is filled with zeros).
    """

    n_points = points.shape[0]

    # Convert points to camera space.
    points = jnp.einsum("ij,kj->ki", viewer_frame, points - viewer_xyz)

    # Project points to the image plane.
    zs = points[:, 2][:, None]
    points_proj = jnp.concatenate([points[:, :2] / zs, zs], axis=1)

    # Compute splat size in the image plane. For now I'm assuming the splats are still circular
    # after projection to the screen, which is approximately true if they are all aligned with the
    # view direction. Later I should model the screen space splats as ellipses.
    splat_radii_proj = splat_radius / points_proj[:, 2]

    # Clip based on z.
    z_mask = jnp.logical_and(points_proj[:, 2] >= near, points_proj[:, 2] <= far)

    # Clip based on visibility.
    fov_bound_x = jnp.tan(0.5 * viewer_fov_x)
    fov_bound_y = jnp.tan(0.5 * viewer_fov_y)
    offset = cutoff * splat_radii_proj
    x_mask = jnp.logical_and(
        points_proj[:, 0] >= -fov_bound_x - offset,
        points_proj[:, 0] <= fov_bound_x + offset,
    )
    y_mask = jnp.logical_and(
        points_proj[:, 1] >= -fov_bound_y - offset,
        points_proj[:, 1] <= fov_bound_y + offset,
    )
    xy_mask = jnp.logical_and(x_mask, y_mask)

    # Combine masks.
    mask = jnp.logical_and(z_mask, xy_mask)
    assert mask.shape == (n_points,)

    # Collect ids of visible splats.
    splat_slots = jnp.cumsum(mask, axis=0)
    n_visible_splats = splat_slots[-1]
    splat_slots = jnp.where(mask, splat_slots - 1, splat_capacity)
    splat_ids = jnp.zeros(n_points, dtype=jnp.uint32)
    splat_ids = splat_ids.at[splat_slots].set(
        jnp.arange(n_points, dtype=jnp.uint32), mode="drop"
    )
    return n_visible_splats, splat_ids


def identify_splats_that_intersect_tile(
    cutoff: float,
    points,  # point xyz in world space
    viewer_xyz,
    viewer_frame,
    splat_radius,
    n_visible_splats,  # scalar, the number of valid splat ids in visible_splat_ids
    visible_splat_ids,  # shape (max_visible_splats,), ids of visible splats
    tile_bounds,  # shape (2, 2): [[x_min, y_min], [x_max, y_max]]
    n_tile_splats,  # scalar, the number of splats already intersected with this tile
    tile_splats,  # shape (capacity,), ids of splats in this tile (output, should be donated)
):
    n_points = points.shape[0]
    max_visible_splats = visible_splat_ids.shape[0]
    max_splats_per_tile = tile_splats.shape[0]

    # Convert points to camera space.
    splat_xyz = points[visible_splat_ids]
    splat_xyz = jnp.einsum("ij,kj->ki", viewer_frame, splat_xyz - viewer_xyz)

    # Project points to the image plane.
    zs = splat_xyz[:, 2][:, None]
    splat_xyz_proj = jnp.concatenate([splat_xyz[:, :2] / zs, zs], axis=1)

    # Compute splat size in the image plane. For now I'm assuming the splats are still circular
    # after projection to the screen, which is approximately true if they are all aligned with the
    # view direction. Later I should model the screen space splats as ellipses.
    splat_radii_proj = splat_radius / splat_xyz_proj[:, 2]

    # Determine which splats intersect this tile.
    # We express this as a boolean mask for visible_splat_ids.
    offset = cutoff * splat_radii_proj
    x_min_cond = splat_xyz_proj[:, 0] >= tile_bounds[0, 0] - offset
    x_max_cond = splat_xyz_proj[:, 0] <= tile_bounds[1, 0] + offset
    y_min_cond = splat_xyz_proj[:, 1] >= tile_bounds[0, 1] - offset
    y_max_cond = splat_xyz_proj[:, 1] <= tile_bounds[1, 1] + offset
    x_overlap = jnp.logical_and(x_min_cond, x_max_cond)
    y_overlap = jnp.logical_and(y_min_cond, y_max_cond)
    mask = jnp.logical_and(x_overlap, y_overlap)
    assert mask.shape == (max_visible_splats,)

    # Collect the ids of splats that intersect this tile.
    splat_slots = jnp.cumsum(mask)
    n_intersecting_splats = splat_slots[-1]
    splat_slots = jnp.where(mask, splat_slots + n_tile_splats - 1, max_splats_per_tile)
    splat_slots = jnp.where(
        jnp.arange(n_points) < n_visible_splats, splat_slots, max_splats_per_tile
    )
    tile_splats = tile_splats.at[splat_slots].set(visible_splat_ids, mode="drop")
    return n_intersecting_splats, tile_splats


pair_splats_and_tiles = jax.vmap(
    identify_splats_that_intersect_tile,
    in_axes=(None, None, None, None, None, None, None, 0, 0, 0),
    out_axes=(0, 0),
)


def splat_tile(
    cutoff: float,
    # 16 x 16 in Kerbl et. al "3D Gaussian Splatting..." (2023)
    tile_size: tuple[int, int],
    tile_bounds: jax.Array,  # [[x_min, y_min], [x_max, y_max]] in the projection plane
    points: jax.Array,  # shape (max_splats, 2), in world space
    colors: jax.Array,  # shape (max_splats, 4), RGBA
    viewer_xyz: jax.Array,
    viewer_frame: jax.Array,
    splat_radius: float,  # scalar
    n_splats: int,  # scalar
    splat_ids: jax.Array,
):
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

    # Convert points to camera space.
    splat_xyz = points[splat_ids]
    splat_xyz = jnp.einsum("ij,kj->ki", viewer_frame, splat_xyz - viewer_xyz)

    # Project points to the image plane.
    zs = splat_xyz[:, 2][:, None]
    splat_xyz_proj = jnp.concatenate([splat_xyz[:, :2] / zs, zs], axis=1)

    # Compute splat size in the image plane. For now I'm assuming the splats are still circular
    # after projection to the screen, which is approximately true if they are all aligned with the
    # view direction. Later I should model the screen space splats as ellipses.
    splat_radii_proj = splat_radius / splat_xyz_proj[:, 2]

    # Sort the splats by z.
    sort_order = jnp.argsort(splat_xyz_proj[:, 2])
    means = splat_xyz_proj[sort_order, :2]
    splat_radii_proj = splat_radii_proj[sort_order]

    # Load colors
    splat_colors = colors[splat_ids]
    splat_colors = splat_colors[sort_order]

    # Initialize the tile.
    rgba_buffer = jnp.zeros(tile_size + (4,), dtype=jnp.float32)

    def body(state):
        idx, rgba_buffer = state

        mean = means[idx]
        std_dev = splat_radii_proj[idx]
        color = splat_colors[idx]

        # Compute the distance from the mean to each pixel.
        distance = jnp.linalg.norm(pixel_xy - mean, axis=-1)
        assert distance.shape == tile_size

        # Compute the alpha for each pixel.
        # Clip to zero after the specified number of standard deviations.
        alpha = color[3] * jnp.exp(-0.5 * distance**2 / std_dev**2)
        alpha = jnp.where(distance < cutoff * std_dev, alpha, 0.0)
        assert alpha.shape == tile_size

        # Premultiply the color by the alpha.
        color = color * alpha[:, :, None]
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
    return n_processed_splats, rgba_buffer


splat_tiles = jax.vmap(
    splat_tile,
    in_axes=(None, None, 0, None, None, None, None, None, 0, 0),
    out_axes=(0, 0),
)


def merge_tiles(n_tiles, tile_rgba):
    tile_x = np.arange(n_tiles[0])
    tile_y = np.arange(n_tiles[1])
    tile_x, tile_y = np.meshgrid(tile_x, tile_y, indexing="ij")
    tile_xy = np.stack([tile_x, tile_y], axis=-1)
    assert tile_xy.shape == (n_tiles[0], n_tiles[1], 2)
    tile_xy = np.reshape(tile_xy, (-1, 2))

    tile_size = tile_rgba.shape[1:3]
    n_pixels_x = tile_size[0] * n_tiles[0]
    n_pixels_y = tile_size[1] * n_tiles[1]
    result = np.zeros((n_pixels_x, n_pixels_y, 4), dtype=np.float32)
    for idx, (x, y) in enumerate(tile_xy):
        x_start = x * tile_size[0]
        x_end = x_start + tile_size[0]
        y_start = y * tile_size[1]
        y_end = y_start + tile_size[1]
        result[x_start:x_end, y_start:y_end] = tile_rgba[idx]

    return result


def render_splats(
    tile_size: tuple[int, int],
    n_tiles: tuple[int, int],
    cutoff: float,  # in units of standard deviations
    points,
    colors,
    viewer_xyz,
    viewer_frame,
    viewer_fov_x,
    viewer_fov_y,
    near,
    far,
    splat_radius,
):
    """
    Gaussian splat rendering.

    We divide the image into 16x16 tiles.

    We start with a very large array of splats. First we figure out which splats are potentially
    visible. This involves some basic math on every splat, and produces a smaller array of splat
    ids.

    Next we figure out which tiles each splat intersects. This involves some additional math on all
    the splats in the previous buffer. It produces a buffer of splat ids for each tile.

    Finally we render each tile. For each tile, this involves a lot of math for each splat in the
    tile's buffer.
    """
    start_time = time.time()
    points = jnp.asarray(points)

    print("Collecting visible splats...")
    visible_splat_capacity = 500_000

    collect_visible_splats_jit = jax.jit(
        partial(collect_visible_splats, cutoff), static_argnums=(0,)
    )
    while True:
        n_visible_splats, splat_ids = collect_visible_splats_jit(
            visible_splat_capacity,
            points,
            viewer_xyz,
            viewer_frame,
            viewer_fov_x,
            viewer_fov_y,
            near,
            far,
            splat_radius,
        )
        n_visible_splats = int(n_visible_splats)
        print(f"{n_visible_splats} visible splats: {splat_ids[:n_visible_splats]}")
        if n_visible_splats <= visible_splat_capacity:
            break
        else:
            print("Too many visible splats; allocating a bigger buffer.")
            visible_splat_capacity = n_visible_splats
            continue
    print("")

    # Compute tile bounds (in the image plane i.e. z = 1).
    fov_bound_x = jnp.tan(0.5 * viewer_fov_x)
    fov_bound_y = jnp.tan(0.5 * viewer_fov_y)
    tile_view_bounds = jnp.array(
        [
            [-fov_bound_x, -fov_bound_y],
            [fov_bound_x, fov_bound_y],
        ]
    )
    tile_bounds_x = jnp.linspace(
        tile_view_bounds[0, 0], tile_view_bounds[1, 0], n_tiles[0] + 1
    )
    tile_bounds_y = jnp.linspace(
        tile_view_bounds[0, 1], tile_view_bounds[1, 1], n_tiles[1] + 1
    )
    low_x, low_y = jnp.meshgrid(tile_bounds_x[:-1], tile_bounds_y[:-1], indexing="ij")
    high_x, high_y = jnp.meshgrid(tile_bounds_x[1:], tile_bounds_y[1:], indexing="ij")
    low = jnp.stack([low_x, low_y], axis=-1)
    high = jnp.stack([high_x, high_y], axis=-1)
    tile_bounds = jnp.stack([low, high], axis=2)
    # The last two axes are min/max, x/y.
    assert tile_bounds.shape == (n_tiles[0], n_tiles[1], 2, 2)
    tile_bounds = jnp.reshape(tile_bounds, (-1, 2, 2))

    print("Pairing splats and tiles...")
    n_tiles_total = n_tiles[0] * n_tiles[1]
    n_tile_splats = jnp.zeros((n_tiles_total,), dtype=jnp.uint32)
    max_splats_per_tile = 200_000
    tile_splats = jnp.zeros((n_tiles_total, max_splats_per_tile), dtype=jnp.uint32)
    pair_splats_and_tiles_jit = jax.jit(
        partial(pair_splats_and_tiles, cutoff), donate_argnums=(8,)
    )
    n_tile_splats, tile_splats = pair_splats_and_tiles_jit(
        points,
        viewer_xyz,
        viewer_frame,
        splat_radius,
        n_visible_splats,
        splat_ids,
        tile_bounds,
        n_tile_splats,
        tile_splats,
    )
    # print(f"n intersecting splats: {n_tile_splats}")
    print("")

    # Render each tile.
    print("Rendering tiles...")

    splat_tiles_jit = jax.jit(partial(splat_tiles, cutoff, tile_size))
    n_processed_splats, image_rgb = splat_tiles_jit(
        tile_bounds,  # [[x_min, y_min], [x_max, y_max]] in the projection plane
        points,  # shape (max_splats, 2), in world space
        colors,
        viewer_xyz,
        viewer_frame,
        splat_radius,  # scalar
        n_tile_splats,  # scalar
        tile_splats,
    )
    # print(f"n processed splats: {n_processed_splats}")
    print("")

    print("Merging tiles...")
    image = merge_tiles(n_tiles, image_rgb)
    print("")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")
    return image, n_visible_splats, splat_ids


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

    print("Building splats...")
    ground_color = np.array([0.5, 0.2, 0.1, 1.0])
    tree_color = np.array([0.1, 0.6, 0.2, 1.0])
    all_points = np.concatenate([ground_points, tree_points], axis=0)
    colors = np.concatenate(
        [
            np.tile(ground_color[None, :], (ground_points.shape[0], 1)),
            np.tile(tree_color[None, :], (tree_points.shape[0], 1)),
        ],
        axis=0,
    )

    # hack
    # all_points = tree_points
    # colors = jnp.tile(tree_color[None, :], (tree_points.shape[0], 1))

    # double hack
    # all_points = jnp.array([
    #     [0.0, 0.0, 0.004]
    # ])
    # colors = np.tile(tree_color[None, :], (all_points.shape[0], 1))
    # viewer_xyz = jnp.array([0.0, 0.0, 0.0])
    # viewer_frame = jnp.eye(3)

    print("")

    print("Rendering image...")
    near = 0.001
    far = 10.0
    splat_size = 0.001
    tile_size = (16, 16)
    n_tiles = (32, 32)
    cutoff = 3.0
    image, n_visible_splats, splat_ids = render_splats(
        tile_size,
        n_tiles,
        cutoff,
        all_points,
        colors,
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

    # ground_points = ground_points[splat_ids]
    # vis = PointCloudVis(ground_points, tree_points)
    # vis.run()


if __name__ == "__main__":
    main()
