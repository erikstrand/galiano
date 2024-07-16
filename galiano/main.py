import numpy as np
import laspy
import jax
import jax.numpy as jnp


def load_laz_points(laz_file):
    with laspy.open(laz_file) as fh:
        las = fh.read()
        points = jnp.array([las.X, las.Y, las.Z]).T
        print(points.shape)
    return points


if __name__ == "__main__":
    load_laz_points("data/bc_092b084_3_4_2_xyes_8_utm10_2019.laz")
