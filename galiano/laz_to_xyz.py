from tqdm import tqdm
import click
import laspy
import numpy as np
import pyproj


def load_laz_points(laz_file, verbose):
    """
    Extracts XYZ data from a LAS/LAZ file. The coordinate system depends on the file.
    """
    with laspy.open(laz_file) as fh:
        las = fh.read()
        header = las.header

    if verbose:
        # Print coordinate system data.
        print("Coordinate Reference System (CRS):", header.vlrs)
        # Extracting WKT from the WktCoordinateSystemVlr
        crs_vlr = None
        for vlr in header.vlrs:
            if isinstance(vlr, laspy.vlrs.known.WktCoordinateSystemVlr):
                crs_vlr = vlr
                break
        if crs_vlr:
            print("WKT CRS Information:", crs_vlr.string)

        print("X Offset:", header.x_offset)
        print("Y Offset:", header.y_offset)
        print("Z Offset:", header.z_offset)
        print("X Scale Factor:", header.x_scale)
        print("Y Scale Factor:", header.y_scale)
        print("Z Scale Factor:", header.z_scale)

        # Print point format data.
        print("Available dimensions in the LAZ file:")
        point_format = las.point_format
        for dimension in point_format.dimension_names:
            print(dimension)
        print("")

    # Apply scale factors and offsets
    raw_points = np.array([las.X, las.Y, las.Z]).T
    real_x = raw_points[:, 0] * header.x_scale + header.x_offset
    real_y = raw_points[:, 1] * header.y_scale + header.y_offset
    real_z = raw_points[:, 2] * header.z_scale + header.z_offset
    points = np.vstack((real_x, real_y, real_z)).T

    return points


def utm_to_lon_lat(utm_points):
    """
    Converts from NAD83(CSRS) UTM zone 10N coordinates to WGS84 longitude and latitude.

    Height is passed through unaltered.
    """
    # Define the UTM Zone 10N CRS
    utm_crs = pyproj.CRS("EPSG:3157")  # NAD83(CSRS) / UTM zone 10N
    wgs84_crs = pyproj.CRS("EPSG:4326")  # WGS84 (latitude and longitude)

    # Define a transformer to convert UTM to WGS84
    transformer = pyproj.Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)

    lon_lat_points = np.zeros_like(utm_points)
    for i in tqdm(range(utm_points.shape[0])):
        utm_x, utm_y = utm_points[i, :2]
        lon_lat_points[i, :2] = transformer.transform(utm_x, utm_y)
    lon_lat_points[:, 2] = utm_points[:, 2]
    return lon_lat_points.astype(np.float32)


def lat_lon_alt_to_cartesian(lat_lon_alt, ref_lat_lon):
    """
    Converts lat/lon coordinates to Cartesian coordinates relative to a reference point.

    The reference point is specified by a lat/lon pair. In the output coordinate system, X is East,
    Y is North, and Z is Up (at the reference point).
    """
    # Constants
    a = 6378137.0  # Semi-major axis of the Earth (meters)
    e2 = 6.69437999014e-3  # Square of eccentricity

    # Split the input array into lat, lon, and alt
    lat = lat_lon_alt[:, 0]
    lon = lat_lon_alt[:, 1]
    alt = lat_lon_alt[:, 2]

    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_lat_rad = np.radians(ref_lat_lon[0])
    ref_lon_rad = np.radians(ref_lat_lon[1])

    # Calculate N, the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

    # Cartesian coordinates in ECEF (Earth-Centered, Earth-Fixed)
    X_ecef = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    Y_ecef = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    Z_ecef = (N * (1 - e2) + alt) * np.sin(lat_rad)

    # Reference point in ECEF
    N_ref = a / np.sqrt(1 - e2 * np.sin(ref_lat_rad)**2)
    X_ref = (N_ref + 0) * np.cos(ref_lat_rad) * np.cos(ref_lon_rad)
    Y_ref = (N_ref + 0) * np.cos(ref_lat_rad) * np.sin(ref_lon_rad)
    Z_ref = (N_ref * (1 - e2) + 0) * np.sin(ref_lat_rad)

    # Translate ECEF coordinates to reference point
    dX = X_ecef - X_ref
    dY = Y_ecef - Y_ref
    dZ = Z_ecef - Z_ref

    # Rotation matrix for ECEF to ENU (East-North-Up) conversion
    sin_lat_ref = np.sin(ref_lat_rad)
    cos_lat_ref = np.cos(ref_lat_rad)
    sin_lon_ref = np.sin(ref_lon_rad)
    cos_lon_ref = np.cos(ref_lon_rad)

    R = np.array([
        [-sin_lon_ref, cos_lon_ref, 0],
        [-sin_lat_ref*cos_lon_ref, -sin_lat_ref*sin_lon_ref, cos_lat_ref],
        [cos_lat_ref*cos_lon_ref, cos_lat_ref*sin_lon_ref, sin_lat_ref]
    ])

    enu_coords = np.dot(R, np.array([dX, dY, dZ]))

    # Convert to kilometers
    result = 0.001 * enu_coords.T
    return result.astype(np.float32)


@click.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True), help="Input laz file")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output npz file")
@click.option("--compress", "-c", is_flag=True, help="Compress the output npz file")
@click.option("--ref-lat", "-rl", type=float, default=48.875, help="Reference latitude")
@click.option("--ref-lon", "-rlon", type=float, default=-123.325, help="Reference longitude")
@click.option("--verbose", "-v", is_flag=True, help="Print verbose output")
@click.option("--max-points", "-m", type=int, default=None, help="Maximum number of points to process")
def main(input, output, compress, ref_lat, ref_lon, verbose, max_points):
    print(f"Loading {input}...")
    utm_points = load_laz_points(input, verbose)
    print(f"Loaded {utm_points.shape[0]} points.")
    print("")

    if max_points is not None:
        print(f"Using only the first {max_points} points.")
        utm_points = utm_points[:max_points]

    print("Converting UTM to Longitude and Latitude...")
    lon_lat_points = utm_to_lon_lat(utm_points)
    print("")

    print("Converting lon/lat/height to XYZ...")
    lat_lon_points = lon_lat_points[:, [1, 0, 2]]
    ref_lat_lon = np.array([ref_lat, ref_lon])
    xyz = lat_lon_alt_to_cartesian(lat_lon_points, ref_lat_lon)

    print("Converting from column-major to row-major...")
    print(xyz.shape)
    xyz = np.ascontiguousarray(xyz)
    print(xyz.shape)

    print(f"Saving to {output}...")
    if compress:
        np.savez_compressed(output, points=xyz)
    else:
        np.savez(output, points=xyz)
    print("Done.")


if __name__ == "__main__":
    main()
