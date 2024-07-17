from typing import NamedTuple, Generator
import jax
import jax.numpy as jnp
import numpy as np

from quantization import BinDef, get_bin_index, quantize


class SpatialSortData(NamedTuple):
    n_points: jnp.ndarray  # () int
    positions: jnp.ndarray  # (max_points, n_dims) float
    bins: jnp.ndarray  # (max_points, n_dims) int
    bin_counts: jnp.ndarray  # (n_total_bins,) int
    bin_starts: jnp.ndarray  # (n_total_bins,) int
    bin_contents: jnp.ndarray  # (max_points,) int

    @property
    def point_capacity(self) -> int:
        return self.positions.shape[0]

    @property
    def n_total_bins(self) -> int:
        return self.bin_counts.shape[0]

    @property
    def n_dims(self) -> int:
        return self.positions.shape[1]


def make_spatial_sort(n_bins: int, max_points: int) -> SpatialSortData:
    dim = 2
    return SpatialSortData(
        n_points=jnp.array(0, dtype=jnp.int32),
        positions=jnp.zeros((max_points, dim), dtype=jnp.float32),
        bins=jnp.full((max_points,), n_bins, dtype=jnp.int32),
        bin_counts=jnp.zeros(n_bins, dtype=jnp.int32),
        bin_starts=jnp.zeros(n_bins + 1, dtype=jnp.int32),
        bin_contents=jnp.zeros(max_points, dtype=jnp.int32),
    )


def insert_points_dense(
    spatial_sort: SpatialSortData,
    bin_def: BinDef,
    positions_to_insert: jnp.ndarray,
    n_points_to_insert: int,
) -> SpatialSortData:
    """
    This method inserts a batch of points into the spatial sort data structure.

    Only the first n_points_to_insert points are inserted.
    """

    # Update n_points.
    old_n_points = spatial_sort.n_points
    new_n_points = old_n_points + n_points_to_insert

    # Figure out where to store the new points.
    max_points_to_insert = positions_to_insert.shape[0]
    dst_indices = jnp.arange(max_points_to_insert)
    dst_indices = jnp.where(
        dst_indices < n_points_to_insert,
        old_n_points + dst_indices,
        spatial_sort.point_capacity,
    )

    # Update positions.
    new_positions = spatial_sort.positions.at[dst_indices].set(
        positions_to_insert, mode="drop"
    )

    # Determine the bin of each new point. We use n_bins to mark empty slots.
    n_bins = bin_def.n_total_bins
    inserted_bins = quantize(positions_to_insert, bin_def)
    inserted_bin_indices = jnp.where(
        jax.lax.iota(jnp.int32, max_points_to_insert) < n_points_to_insert,
        get_bin_index(inserted_bins, bin_def),
        n_bins,
    )

    # Update point bins.
    new_bins = spatial_sort.bins.at[dst_indices].set(inserted_bin_indices, mode="drop")

    # Update bin counts.
    new_bin_counts = spatial_sort.bin_counts.at[inserted_bin_indices].add(
        1, mode="drop"
    )

    # Update bin starts.
    new_bin_starts = spatial_sort.bin_starts.at[1:].set(jnp.cumsum(new_bin_counts))

    # Update bin contents.
    new_bin_contents = jnp.argsort(new_bins)

    spatial_sort = SpatialSortData(
        n_points=new_n_points,
        positions=new_positions,
        bins=new_bins,
        bin_counts=new_bin_counts,
        bin_starts=new_bin_starts,
        bin_contents=new_bin_contents,
    )
    return spatial_sort


def insert_points_sparse(
    spatial_sort: SpatialSortData,
    bin_def: BinDef,
    positions_to_insert: jnp.ndarray,
    position_mask: jnp.ndarray,
) -> SpatialSortData:
    """
    This method inserts a batch of points into the spatial sort data structure.

    Only the points identified by position_mask are inserted.
    """

    # Update n_points.
    old_n_points = spatial_sort.n_points
    n_points_to_insert = jnp.sum(position_mask)
    new_n_points = old_n_points + n_points_to_insert

    # Figure out where to store the new points.
    dst_indices = (spatial_sort.n_points - 1) + jnp.cumsum(position_mask)
    dst_indices = jnp.where(position_mask, dst_indices, spatial_sort.point_capacity)

    # Update positions.
    new_positions = spatial_sort.positions.at[dst_indices].set(
        positions_to_insert, mode="drop"
    )

    # Determine the bin of each new point. We use n_bins to mark empty slots.
    n_bins = bin_def.n_total_bins
    inserted_bins = quantize(positions_to_insert, bin_def)
    inserted_bin_indices = jnp.where(
        position_mask,
        get_bin_index(inserted_bins, bin_def),
        n_bins,
    )

    # Update point bins.
    new_bins = spatial_sort.bins.at[dst_indices].set(inserted_bin_indices, mode="drop")

    # Update bin counts.
    new_bin_counts = spatial_sort.bin_counts.at[inserted_bin_indices].add(
        1, mode="drop"
    )

    # Update bin starts.
    new_bin_starts = spatial_sort.bin_starts.at[1:].set(jnp.cumsum(new_bin_counts))

    # Update bin contents.
    new_bin_contents = jnp.argsort(new_bins)

    return SpatialSortData(
        n_points=new_n_points,
        positions=new_positions,
        bins=new_bins,
        bin_counts=new_bin_counts,
        bin_starts=new_bin_starts,
        bin_contents=new_bin_contents,
    )


def reduce_over_cell(
    spatial_sort: SpatialSortData,
    cell_idx: jnp.ndarray,  # () int
    reduce_fun,  # a, int, position -> a
    init_val,  # a
):
    start = spatial_sort.bin_starts[cell_idx]
    end = spatial_sort.bin_starts[cell_idx + 1]
    state = {"idx": start, "end": end, "data": init_val}

    def cond_fun(state):
        return state["idx"] < state["end"]

    def body_fun(state):
        point_idx = spatial_sort.bin_contents[state["idx"]]
        position = spatial_sort.positions[point_idx]
        new_data = reduce_fun(state["data"], point_idx, position)
        return {"idx": state["idx"] + 1, "end": state["end"], "data": new_data}

    return jax.lax.while_loop(cond_fun, body_fun, state)["data"]


def bin_neighbor_offsets(dimension: int) -> Generator[np.ndarray, None, None]:
    """
    Iterates over all offsets to neighboring bins (including no offset, i.e. this bin).
    """

    for dindex in np.ndindex((3,) * dimension):
        yield np.array(dindex, dtype=np.int32) - 1


def reduce_over_neighboring_cells(
    spatial_sort: SpatialSortData,
    bin_def: BinDef,
    cell_coords: jnp.ndarray,  # (n_dims,) int
    reduce_fun,  # a, int, position -> a
    init_val,  # a
):
    state = init_val
    for offset in bin_neighbor_offsets(spatial_sort.n_dims):
        neighbor_coords = cell_coords + offset
        neighbor_idx = get_bin_index(neighbor_coords, bin_def)
        state = jax.lax.cond(
            jnp.logical_and(
                jnp.all(neighbor_coords >= 0), jnp.all(neighbor_coords < bin_def.n_bins)
            ),
            lambda: reduce_over_cell(spatial_sort, neighbor_idx, reduce_fun, state),
            lambda: state,
        )
    return state


def find_nearest_neighbor_impl(
    spatial_sort: SpatialSortData,
    bin_def: BinDef,
    position: jnp.ndarray,  # (n_dims,) float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns the index and position of the nearest neighbor to the given position.
    """

    def reduce_fun(data, neighbor_idx, neighbor_position):
        neighbor_distance = jnp.linalg.norm(neighbor_position - data["position"])
        is_new_best = neighbor_distance < data["min_distance"]
        return {
            "position": data["position"],
            "min_distance": jax.lax.select(
                is_new_best, neighbor_distance, data["min_distance"]
            ),
            "min_index": jax.lax.select(is_new_best, neighbor_idx, data["min_index"]),
        }

    init_state = {
        "position": position,
        "min_distance": jnp.array(jnp.inf, dtype=jnp.float32),
        "min_index": jnp.array(spatial_sort.point_capacity, dtype=jnp.int32),
    }

    cell_coords = quantize(position, bin_def)
    result = reduce_over_neighboring_cells(
        spatial_sort, bin_def, cell_coords, reduce_fun, init_state
    )
    return result["min_index"], result["min_distance"]


find_nearest_neighbor_vmap = jax.vmap(
    find_nearest_neighbor_impl, in_axes=(None, None, 0)
)


def find_nearest_neighbor(
    spatial_sort: SpatialSortData,
    bin_def: BinDef,
    position: jnp.ndarray,  # (n_dims,) or (n_query, n_dims,) float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    assert len(position.shape) in (1, 2)
    if len(position.shape) == 1:
        return find_nearest_neighbor_impl(spatial_sort, bin_def, position)
    else:
        return find_nearest_neighbor_vmap(spatial_sort, bin_def, position)
