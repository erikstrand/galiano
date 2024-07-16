import click
import laspy
import numpy as np


def load_laz_classifications(laz_file, verbose):
    """
    Extracts classification data from a LAS/LAZ file.
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

    return np.array(las.classification, dtype=np.uint8)


@click.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True), help="Input laz file")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output npz file")
@click.option("--compress", "-c", is_flag=True, help="Compress the output npz file")
@click.option("--verbose", "-v", is_flag=True, help="Print verbose output")
@click.option("--max-points", "-m", type=int, default=None, help="Maximum number of points to process")
def main(input, output, compress, verbose, max_points):
    print(f"Loading {input}...")
    classifications = load_laz_classifications(input, verbose)
    print(classifications)
    print(f"Loaded {classifications.shape[0]} points.")
    print("")

    if verbose:
        unique, counts = np.unique(classifications, return_counts=True)
        classification_counts = dict(zip(unique, counts))
        print("Classification counts:")
        for key, value in classification_counts.items():
            print(f"{key}: {value}")

    if max_points is not None:
        print(f"Using only the first {max_points} points.")
        classifications = classifications[:max_points]

    print(f"Saving to {output}...")
    if compress:
        np.savez_compressed(output, classifications=classifications)
    else:
        np.savez(output, classifications=classifications)
    print("Done.")


if __name__ == "__main__":
    main()
