import numpy as np


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
