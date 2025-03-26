import build_mesh as build_mesh
import numpy as np
import scipy
import geometry as geometry
import trimesh


def radius_query_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: build_mesh.TriangularMesh,
        radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns mesh-grid edge indices for radius query.

    Args:
      grid_latitude: Latitude values for the grid [num_lat_points]
      grid_longitude: Longitude values for the grid [num_lon_points]
      mesh: Mesh object.
      radius: Radius of connectivity in R3. for a sphere of unit radius.

    Returns:
      tuple with `grid_indices` and `mesh_indices` indicating edges between the
      grid and the mesh such that the distances in a straight line (not geodesic)
      are smaller than or equal to `radius`.
      * grid_indices: Indices of shape [num_edges], that index into a
        [num_lat_points, num_lon_points] grid, after flattening the leading axes.
      * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
    """

    # [num_lat_points * num_lon_points, 3]
    grid_positions = geometry.lat_lon_deg_to_cartesian(
        grid_latitude, grid_longitude)
    grid_positions = np.stack(grid_positions, axis=-1)

    # [num_mesh_points, 3]
    mesh_positions = mesh.vertices
    kd_tree = scipy.spatial.cKDTree(mesh_positions)

    # [num_grid_points, num_mesh_points_per_grid_point]
    # Note `num_mesh_points_per_grid_point` is not constant, so this is a list
    # of arrays, rather than a 2d array.
    query_indices = kd_tree.query_ball_point(x=grid_positions, r=radius)

    grid_edge_indices = []
    mesh_edge_indices = []
    for grid_index, mesh_neighbors in enumerate(query_indices):
        grid_edge_indices.append(np.repeat(grid_index, len(mesh_neighbors)))
        mesh_edge_indices.append(mesh_neighbors)

    # [num_edges]
    grid_edge_indices = np.concatenate(grid_edge_indices, axis=0).astype(int)
    mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0).astype(int)

    return grid_edge_indices, mesh_edge_indices


def in_mesh_triangle_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
        mesh: build_mesh.TriangularMesh) -> tuple[np.ndarray, np.ndarray]:
    """Returns mesh-grid edge indices for grid points contained in mesh triangles.

    Args:
      grid_latitude: Latitude values for the grid [num_lat_points]
      grid_longitude: Longitude values for the grid [num_lon_points]
      mesh: Mesh object.

    Returns:
      tuple with `grid_indices` and `mesh_indices` indicating edges between the
      grid and the mesh vertices of the triangle that contain each grid point.
      The number of edges is always num_lat_points * num_lon_points * 3
      * grid_indices: Indices of shape [num_edges], that index into a
        [num_lat_points, num_lon_points] grid, after flattening the leading axes.
      * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
    """

    # [num_grid_points=num_lat_points * num_lon_points, 3]
    grid_positions = geometry.lat_lon_deg_to_cartesian(
        grid_latitude, grid_longitude)
    grid_positions = np.stack(grid_positions, axis=-1)

    mesh_trimesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    # [num_grid_points] with mesh face indices for each grid point.
    _, _, query_face_indices = trimesh.proximity.closest_point(
        mesh_trimesh, grid_positions)

    # [num_grid_points, 3] with mesh node indices for each grid point.
    mesh_edge_indices = mesh.faces[query_face_indices]

    # [num_grid_points, 3] with grid node indices, where every row simply contains
    # the row (grid_point) index.
    grid_indices = np.arange(grid_positions.shape[0])
    grid_edge_indices = np.tile(grid_indices.reshape([-1, 1]), [1, 3])

    # Flatten to get a regular list.
    # [num_edges=num_grid_points*3]
    mesh_edge_indices = mesh_edge_indices.reshape([-1])
    grid_edge_indices = grid_edge_indices.reshape([-1])

    return grid_edge_indices, mesh_edge_indices
