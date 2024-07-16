from tqdm import tqdm
import click
import laspy
import numpy as np
import pyproj


def load_laz_points(laz_file):
    with laspy.open(laz_file) as fh:
        las = fh.read()
        header = las.header
        raw_points = np.array([las.X, las.Y, las.Z]).T

        # Apply scale factors and offsets
        real_x = raw_points[:, 0] * header.x_scale + header.x_offset
        real_y = raw_points[:, 1] * header.y_scale + header.y_offset
        real_z = raw_points[:, 2] * header.z_scale + header.z_offset
        points = np.vstack((real_x, real_y, real_z)).T

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


def utm_to_lat_long(utm_points):
    # Define the UTM Zone 10N CRS
    utm_crs = pyproj.CRS("EPSG:3157")  # NAD83(CSRS) / UTM zone 10N
    wgs84_crs = pyproj.CRS("EPSG:4326")  # WGS84 (latitude and longitude)

    # Define a transformer to convert UTM to WGS84
    transformer = pyproj.Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)

    lat_lon_points = np.zeros_like(utm_points)
    for i in tqdm(range(utm_points.shape[0])):
        utm_x, utm_y = utm_points[i, :2]
        lat_lon_points[i, :2] = transformer.transform(utm_x, utm_y)
    lat_lon_points[:, 2] = utm_points[:, 2]
    return lat_lon_points


@click.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True), help="Input laz file")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output npz file")
def main(input, output):
    print(f"Loading {input}...")
    utm_points = load_laz_points(input)
    print(f"Loaded {utm_points.shape[0]} points.")
    print("")

    print("Converting UTM to Latitude and Longitude...")
    lat_lon_points = utm_to_lat_long(utm_points)
    print("")

    print(f"Saving to {output}...")
    np.savez(output, points=lat_lon_points)
    print("Done.")


if __name__ == "__main__":
    main()
