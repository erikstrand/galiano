from typing import NamedTuple
import jax
import jax.numpy as jnp


class BinDef(NamedTuple):
    n_total_bins: jnp.ndarray  # () int
    n_bins: jnp.ndarray  # (n_dims,) int
    strides: jnp.ndarray  # (n_dims,) int
    bounds: jnp.ndarray  # (2, n_dims) float
    bin_size: jnp.ndarray  # (n_dims,) float

    @property
    def dim(self):
        return self.n_bins.shape[0]

    def __getitem__(self, key) -> "BinDef":
        if isinstance(key, int) or isinstance(key, slice):
            return BinDef(
                n_total_bins=self.n_bins[key].prod(),
                n_bins=self.n_bins[key],
                strides=self.strides[key],
                bounds=self.bounds[:, key],
                bin_size=self.bin_size[key],
            )
        else:
            raise ValueError(
                f"{type(self).__name__} indices must be integers or slices, not {type(key).__name__}"
            )


def make_bin_def(
    bounds: jnp.ndarray,  # (2, n_dims) float
    min_bin_size: jnp.ndarray,  # (1,) or (n_dims,) float
    force_even: bool,
    int_type: jnp.dtype = jnp.int32,
) -> BinDef:
    """
    Constructs a 1D bin definition from the given bounds and minimum bin size.

    Our goal is to make n_bins as small as possible while respecting min_bin_size.
    """

    if len(bounds.shape) == 1:
        bounds = bounds[:, None]
    assert len(bounds.shape) == 2
    n_dims = bounds.shape[1]
    assert bounds.shape == (2, n_dims)

    if isinstance(min_bin_size, float):
        min_bin_size = jnp.array(min_bin_size, dtype=jnp.float32)
    if min_bin_size.shape == ():
        min_bin_size = min_bin_size[None]
    assert min_bin_size.shape == (1,) or min_bin_size.shape == (n_dims,)

    extent = bounds[1] - bounds[0]
    n_bins = jnp.floor(extent / min_bin_size).astype(int_type)
    if force_even:
        n_bins = jnp.where(n_bins % 2 == 0, n_bins, n_bins - 1)
    bin_size = extent / n_bins
    assert jnp.all(min_bin_size <= bin_size)
    n_total_bins = jnp.prod(n_bins)
    strides = jnp.cumprod(
        jnp.concatenate([jnp.array([1], dtype=int_type), n_bins[:0:-1]])
    )[::-1]
    final_bounds = jnp.stack([bounds[0], bounds[0] + n_bins * bin_size], axis=0)
    return BinDef(
        n_total_bins=n_total_bins,
        n_bins=n_bins,
        strides=strides,
        bounds=final_bounds,
        bin_size=bin_size,
    )


def get_bin_index(bin_coords: jnp.ndarray, bin_def: BinDef) -> jnp.ndarray:
    """
    Converts a n-dimensional bin coordinate vector into a scalar bin index.

    This is useful for indexing into flattened bin arrays.
    """

    assert bin_coords.shape[-1] == bin_def.strides.shape[0]
    return jnp.einsum("...i,i->...", bin_coords, bin_def.strides)


def quantize(x: jnp.ndarray, bin_def: BinDef) -> jnp.ndarray:
    """
    Quantizes a floating point value into one of an array of equally sized bins.

    If quantize_1d(x) = i, then bins.f_min + i * bins.bin_size <= x < bins.f_min + (i + 1) *
    bins.bin_size. With the exception that the first and last bin contain all outliers as well.
    """

    assert (
        len(x.shape) == 0
        or (len(x.shape) == 1 and bin_def.dim == 1)
        or x.shape[-1] == bin_def.n_bins.shape[0]
    )

    # This may be off by one due to rounding error.
    bin_idx = jnp.floor((x - bin_def.bounds[0]) / bin_def.bin_size)

    # Correct for rounding error.
    bin_idx = jnp.where(
        x < bin_def.bounds[0] + bin_idx * bin_def.bin_size, bin_idx - 1, bin_idx
    )
    bin_idx = jnp.where(
        x >= bin_def.bounds[0] + (bin_idx + 1) * bin_def.bin_size, bin_idx + 1, bin_idx
    )

    # Convert to an integer and clip.
    bin_idx = bin_idx.astype(bin_def.n_bins.dtype)
    return jnp.clip(bin_idx, a_min=0, a_max=bin_def.n_bins - 1)


def dequantize(bin_coords: jnp.ndarray, bin_def: BinDef) -> jnp.ndarray:
    """
    Dequantizes a bin index into a floating point value.

    If you give an integral value for bin_coords, you'll get the lower left corner. You can add a
    fractional offset to get e.g. the center of the bin.
    """

    assert (
        len(bin_coords.shape) == 0
        or (len(bin_coords.shape) == 1 and bin_def.dim == 1)
        or bin_coords.shape[-1] == bin_def.n_bins.shape[0]
    )
    return bin_def.bounds[0] + bin_coords * bin_def.bin_size
