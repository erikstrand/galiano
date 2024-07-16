import numpy as np
import laspy
import jax
import jax.numpy as jnp
import pyproj


def load_laz_points(laz_file):
    with laspy.open(laz_file) as fh:
        las = fh.read()
        header = las.header
        raw_points = jnp.array([las.X, las.Y, las.Z]).T

        # Apply scale factors and offsets
        real_x = raw_points[:, 0] * header.x_scale + header.x_offset
        real_y = raw_points[:, 1] * header.y_scale + header.y_offset
        real_z = raw_points[:, 2] * header.z_scale + header.z_offset
        points = jnp.vstack((real_x, real_y, real_z)).T

        print("Coordinate Reference System (CRS):", header.vlrs)
        print("X Offset:", header.x_offset)
        print("Y Offset:", header.y_offset)
        print("Z Offset:", header.z_offset)
        print("X Scale Factor:", header.x_scale)
        print("Y Scale Factor:", header.y_scale)
        print("Z Scale Factor:", header.z_scale)

        # Extracting WKT from the WktCoordinateSystemVlr
        crs_vlr = None
        for vlr in header.vlrs:
            if isinstance(vlr, laspy.vlrs.known.WktCoordinateSystemVlr):
                crs_vlr = vlr
                break

        if crs_vlr:
            print("WKT CRS Information:", crs_vlr.string)

    return points


if __name__ == "__main__":
    points = load_laz_points("data/bc_092b084_3_4_2_xyes_8_utm10_2019.laz")
    print("points", points.shape)
    print("min", jnp.min(points, axis=0))
    print("max", jnp.max(points, axis=0))


    # Define the UTM Zone 10N CRS
    utm_crs = pyproj.CRS("EPSG:3157")  # NAD83(CSRS) / UTM zone 10N
    wgs84_crs = pyproj.CRS("EPSG:4326")  # WGS84 (latitude and longitude)

    # Define a transformer to convert UTM to WGS84
    transformer = pyproj.Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)

    # Example UTM coordinates (take a sample point)
    print("eastings, northings, elevations", points[0])
    utm_x, utm_y = points[0][:2]

    # Convert to latitude and longitude
    lat, lon = transformer.transform(utm_x, utm_y)
    print(f"Latitude: {lat}, Longitude: {lon}")
