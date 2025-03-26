from typing import List, Tuple, Dict, NamedTuple
import numpy as np
from scipy.spatial import transform


# transforming latitudes and longitudes to spherical coordinates
# note that the spherical coordinates are in radians
# latitudes are in the range of -90 to 90, transform to the range of 0 to 180 first
def lat_lon_deg_to_spherical(node_lat: np.ndarray,
                             node_lon: np.ndarray,
                             ) -> Tuple[np.ndarray, np.ndarray]:
    phi = np.deg2rad(node_lon)
    theta = np.deg2rad(90 - node_lat)
    return phi, theta


def spherical_to_lat_lon(phi: np.ndarray,
                         theta: np.ndarray,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.mod(np.rad2deg(phi), 360)
    lat = 90 - np.rad2deg(theta)
    return lat, lon


# transforming between cartesian and spherical coordinates, for unit radius in 3D
def cartesian_to_spherical(x: np.ndarray,
                           y: np.ndarray,
                           z: np.ndarray,
                           ) -> Tuple[np.ndarray, np.ndarray]:
    phi = np.arctan2(y, x)
    with np.errstate(invalid="ignore"):  # circumventing b/253179568
        theta = np.arccos(z)  # Assuming unit radius.
    return phi, theta


def spherical_to_cartesian(
    phi: np.ndarray, theta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Assuming unit radius.
    return (np.cos(phi)*np.sin(theta),
            np.sin(phi)*np.sin(theta),
            np.cos(theta))


# transforming between latitudes and longitudes to cartesian coordinates
def lat_lon_deg_to_cartesian(node_lat: np.ndarray,
                             node_lon: np.ndarray,
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi, theta = lat_lon_deg_to_spherical(node_lat, node_lon)
    return spherical_to_cartesian(phi, theta)


def cartesian_to_lat_lon_deg(x: np.ndarray,
                             y: np.ndarray,
                             z: np.ndarray,
                             ) -> Tuple[np.ndarray, np.ndarray]:
    phi, theta = cartesian_to_spherical(x, y, z)
    return spherical_to_lat_lon(phi, theta)


def get_rotation_matrices_to_local_coordinates(
        reference_phi: np.ndarray,
        reference_theta: np.ndarray,
        rotate_latitude: bool,
        rotate_longitude: bool) -> np.ndarray:
    """Returns a rotation matrix to rotate to a point based on a reference vector.

    The rotation matrix is build such that, a vector in the
    same coordinate system at the reference point that points towards the pole
    before the rotation, continues to point towards the pole after the rotation.

    Args:
      reference_phi: [leading_axis] Polar angles of the reference.
      reference_theta: [leading_axis] Azimuthal angles of the reference.
      rotate_latitude: Whether to produce a rotation matrix that would rotate
          R^3 vectors to zero latitude.
      rotate_longitude: Whether to produce a rotation matrix that would rotate
          R^3 vectors to zero longitude.

    Returns:
      Matrices of shape [leading_axis] such that when applied to the reference
          position with `rotate_with_matrices(rotation_matrices, reference_pos)`

          * phi goes to 0. if "rotate_longitude" is True.

          * theta goes to np.pi / 2 if "rotate_latitude" is True.

          The rotation consists of:
          * rotate_latitude = False, rotate_longitude = True:
              Latitude preserving rotation.
          * rotate_latitude = True, rotate_longitude = True:
              Latitude preserving rotation, followed by longitude preserving
              rotation.
          * rotate_latitude = True, rotate_longitude = False:
              Latitude preserving rotation, followed by longitude preserving
              rotation, and the inverse of the latitude preserving rotation. Note
              this is computationally different from rotating the longitude only
              and is. We do it like this, so the polar geodesic curve, continues
              to be aligned with one of the axis after the rotation.

    """

    if rotate_longitude and rotate_latitude:

        # We first rotate around the z axis "minus the azimuthal angle", to get the
        # point with zero longitude
        azimuthal_rotation = - reference_phi

        # One then we will do a polar rotation (which can be done along the y
        # axis now that we are at longitude 0.), "minus the polar angle plus 2pi"
        # to get the point with zero latitude.
        polar_rotation = - reference_theta + np.pi/2

        return transform.Rotation.from_euler(
            "zy", np.stack([azimuthal_rotation, polar_rotation],
                           axis=1)).as_matrix()
    elif rotate_longitude:
        # Just like the previous case, but applying only the azimuthal rotation.
        azimuthal_rotation = - reference_phi
        return transform.Rotation.from_euler("z", -reference_phi).as_matrix()
    elif rotate_latitude:
        # Just like the first case, but after doing the polar rotation, undoing
        # the azimuthal rotation.
        azimuthal_rotation = - reference_phi
        polar_rotation = - reference_theta + np.pi/2

        return transform.Rotation.from_euler(
            "zyz", np.stack(
                [azimuthal_rotation, polar_rotation, -azimuthal_rotation], axis=1)).as_matrix()
    else:
        raise ValueError(
            "At least one of longitude and latitude should be rotated.")
