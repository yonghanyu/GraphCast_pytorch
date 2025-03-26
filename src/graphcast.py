import torch
import torch.nn as nn
import numpy as np
import build_mesh as build_mesh
import mesh_grid as mesh_grid
import geometry as geometry
from constant import *
import model_utils as model_utils
from encoder import Grid2MeshEncoder
from decoder import *
from processor import MeshProcessor
import xarray as xr


# TODO: resolve import with relative path


def _get_max_edge_distance(mesh):
    senders, receivers = build_mesh.faces_to_edges(mesh.faces)
    edge_distances = np.linalg.norm(
        mesh.vertices[senders] - mesh.vertices[receivers], axis=-1)
    return edge_distances.max()


class GraphBase(nn.Module):
    def __init__(self,
                 latitudes,
                 longitudes,
                 mesh_size: int = 6,
                 radius_query_fraction_edge_length: float = 0.6,):

        super().__init__()
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.mesh_size = mesh_size
        self.radius_query_fraction_edge_length = radius_query_fraction_edge_length

        self.meshes = None  # list of TriangularMesh objects
        self.num_mesh_nodes = None
        self.mesh_nodes_latitudes = None
        self.mesh_nodes_longitudes = None
        self.num_grid_nodes = None
        self.grid_latitudes = None
        self.grid_longitudes = None
        self._init_meshes()
        self._init_grids()

        self.query_radius = (_get_max_edge_distance(self.finest_mesh) *
                             self.radius_query_fraction_edge_length)

    @property
    def finest_mesh(self):
        return self.meshes[-1]

    def _init_meshes(self):
        # nodes of fine mesh are super set of coarse mesh,
        # so we use the nodes of the finest mesh with implied coarse edges
        # mesh nodes are in cartesian coordinates, with unit radius centered at (0, 0, 0)
        self.meshes = build_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
            self.mesh_size)
        self.num_mesh_nodes = self.finest_mesh.vertices.shape[0]
        (mesh_nodes_latitudes,
         mesh_nodes_longitudes) = geometry.cartesian_to_lat_lon_deg(
            self.finest_mesh.vertices[:, 0],
            self.finest_mesh.vertices[:, 1],
            self.finest_mesh.vertices[:, 2],
        )

        self.mesh_nodes_latitudes = mesh_nodes_latitudes.astype(np.float32)
        self.mesh_nodes_longitudes = mesh_nodes_longitudes.astype(np.float32)

    # TODO: consider remove latitude at north pole/south pole

    def _init_grids(self):
        self.num_grid_nodes = len(self.latitudes) * len(self.longitudes)

        grid_latitudes, grid_longitudes = np.meshgrid(
            self.latitudes, self.longitudes
        )

        self.grid_latitudes = grid_latitudes.flatten().astype(np.float32)
        self.grid_longitudes = grid_longitudes.flatten().astype(np.float32)

    def _init_grid2mesh_graph(self):
        # [N_edges]
        grid_nodes_idx, mesh_nodes_idx = mesh_grid.radius_query_indices(
            grid_latitude=self.grid_latitudes,
            grid_longitude=self.grid_longitudes,
            mesh=self.finest_mesh,
            radius=self.query_radius
        )

        # src_geo_nodes_features: [N_src_nodes, 3] (sine of latitude, sine and cosine of latitude)
        # dst_geo_nodes_features: [N_dst_nodes, 3]
        # edge_features: [N_edges, 4] (length of the edge, and the vector difference between the 3d positions of the sender
        (grid_nodes_geo_features,
         mesh_nodes_geo_features,
         edge_features) = model_utils.get_bipartite_graph_spatial_features(
             senders_node_lat=self.grid_latitudes,
             senders_node_lon=self.grid_longitudes,
             receivers_node_lat=self.mesh_nodes_latitudes,
             receivers_node_lon=self.mesh_nodes_longitudes,
             senders=grid_nodes_idx,
             receivers=mesh_nodes_idx,
             add_node_positions=False,
             add_node_latitude=True,
             add_node_longitude=True,
             add_relative_positions=True,
             relative_latitude_local_coordinates=True,
             relative_longitude_local_coordinates=True,
             edge_normalization_factor=None
        )

        grid_nodes_geo_features = torch.tensor(
            grid_nodes_geo_features, dtype=torch.float32)
        mesh_nodes_geo_features = torch.tensor(
            mesh_nodes_geo_features, dtype=torch.float32)
        grid2mesh_edge_features = torch.tensor(
            edge_features, dtype=torch.float32)
        grid2mesh_edge_index = torch.tensor(
            np.array([grid_nodes_idx, mesh_nodes_idx]), dtype=torch.long)
        return grid_nodes_geo_features, mesh_nodes_geo_features, grid2mesh_edge_features, grid2mesh_edge_index

    def _init_mesh_graphs(self):
        # build the graph that connects the mesh nodes based on the mesh edges, on all refinement levels
        #                        0
        #                       / \
        #                      /   \
        #                     /     \
        #                    /       \
        #                   /         \
        #                  /           \
        #                 1 ----------- 2
        # edges: (0, 1), (1, 2), (2, 0)
        merged_meshes = build_mesh.merge_meshes(self.meshes)
        # sender has shape [num_edges=3*num_faces]
        sender_idx, receiver_idx = build_mesh.faces_to_edges(
            merged_meshes.faces)

        _, edge_features = model_utils.get_graph_spatial_features(
            node_lat=self.mesh_nodes_latitudes,
            node_lon=self.mesh_nodes_longitudes,
            senders=sender_idx,
            receivers=receiver_idx,
            add_node_positions=False,
            add_node_latitude=True,
            add_node_longitude=True,
            add_relative_positions=True,
            relative_latitude_local_coordinates=True,
            relative_longitude_local_coordinates=True,
        )

        mesh2mesh_edge_features = torch.tensor(
            edge_features, dtype=torch.float32)
        mesh2mesh_edge_index = torch.tensor(
            np.array([sender_idx, receiver_idx]), dtype=torch.long)

        return mesh2mesh_edge_features, mesh2mesh_edge_index

    def _init_mesh2grid_graph(self):
        grid_nodes_idx, mesh_nodes_idx = mesh_grid.in_mesh_triangle_indices(
            grid_latitude=self.grid_latitudes,
            grid_longitude=self.grid_longitudes,
            mesh=self.finest_mesh
        )

        (_, _, edge_feature) = model_utils.get_bipartite_graph_spatial_features(
            senders_node_lat=self.mesh_nodes_latitudes,
            senders_node_lon=self.mesh_nodes_longitudes,
            receivers_node_lat=self.grid_latitudes,
            receivers_node_lon=self.grid_longitudes,
            senders=mesh_nodes_idx,
            receivers=grid_nodes_idx,
            add_node_positions=False,
            add_node_latitude=True,
            add_node_longitude=True,
            add_relative_positions=True,
            relative_latitude_local_coordinates=True,
            relative_longitude_local_coordinates=True,
            edge_normalization_factor=None  # FIXME: settle this
        )

        mesh2grid_edge_features = torch.tensor(
            edge_feature, dtype=torch.float32)
        mesh2grid_edge_index = torch.tensor(
            np.array([mesh_nodes_idx, grid_nodes_idx]), dtype=torch.long)
        return mesh2grid_edge_features, mesh2grid_edge_index


class GraphCastBase(GraphBase):
    def __init__(self,
                 latitude,
                 longitude,
                 in_dim: int = 363,
                 out_dim: int = 363,
                 mesh_size: int = 6,
                 latent_size: int = 512,
                 radius_query_fraction_edge_length: float = 0.6,
                 use_checkpoint=True,
                 residual=True,
                 embed_geo_feature=True) -> None:
        super().__init__(latitude, longitude, mesh_size, radius_query_fraction_edge_length)
        """__init__ mesh node feature is precomputed (cosine of latitude, sine and cosine of longitude), num of features = 3
                    edge feature is precomputed (length of the edge, and the vector difference between the 3d positions of the sender
                    node and the receiver node computed in a local coordinate system of the receiver. The local
                    coordinate system of the receiver is computed by applying a rotation that changes the azimuthal
                    angle until that receiver node lies at longitude 0, followed by a rotation that changes the polar
                    angle until the receiver also lies at latitude 0.) num of features = 4

                    this is a bipartite graph
                    edge index in COO format (2, num_edges)
                    graph structure is fixed, only the feature is updated

        """
        self.residual = residual

        self.mesh_size = mesh_size
        self.latent_size = latent_size

        self.radius_query_fraction_edge_length = radius_query_fraction_edge_length
        self.use_checkpoint = use_checkpoint
        self.embed_geo_feature = embed_geo_feature

        (grid_nodes_geo_features,
         mesh_nodes_geo_features,
         grid2mesh_edge_features,
         grid2mesh_edge_index) = self._init_grid2mesh_graph()

        self.encoder = Grid2MeshEncoder(
            in_grid_node_dim=in_dim + 3,
            latent_grid_node_dim=self.latent_size,
            in_mesh_node_dim=3,
            latent_mesh_node_dim=self.latent_size,
            in_edge_dim=4,
            latent_edge_dim=self.latent_size,
            n_processing_layers=1,
            use_checkpoint=True,
            use_layer_norm=True,
        )
        self.encoder.register_buffer(
            'grid_nodes_geo_features', grid_nodes_geo_features)
        self.encoder.register_buffer(
            'mesh_nodes_geo_features', mesh_nodes_geo_features)
        self.encoder.register_buffer(
            'grid2mesh_edge_features', grid2mesh_edge_features)
        self.encoder.register_buffer(
            'grid2mesh_edge_index', grid2mesh_edge_index)

        (mesh2grid_edge_features,
         mesh2grid_edge_index) = self._init_mesh2grid_graph()
        self.decoder = Mesh2GridDecoder(
            latent_grid_node_dim=self.latent_size,
            out_grid_node_dim=out_dim,
            latent_mesh_node_dim=self.latent_size,
            latent_edge_dim=self.latent_size,
            n_processing_layers=1,
            use_checkpoint=True,
            use_layer_norm=False,
        )
        self.decoder.register_buffer(
            'mesh2grid_edge_features', mesh2grid_edge_features)
        self.decoder.register_buffer(
            'mesh2grid_edge_index', mesh2grid_edge_index)

    def restore_shape(self, predictions):
        """restore_shape _summary_

        Args:
            predictions (_type_): [b,  n_lat * n_lon,  n_channel]

        Returns:
            _type_:  [b, n_channel, n_lat, n_lon, ]
        """
        predictions = torch.reshape(predictions, (predictions.shape[0], len(self.latitudes), len(self.longitudes), -1))
        predictions = einops.rearrange(predictions, 'b lat lon c -> b c lat lon')
        return predictions

    def encode(self, grid_nodes_features):
        # grid_nodes_features: [batch_size, n_lat, n_lon, n_channel]
        b, n_lat, n_lon, _ = grid_nodes_features.shape
        grid_nodes_features = torch.reshape(
            grid_nodes_features, (b, n_lat*n_lon, -1))

        (
            grid_nodes_features,
            mesh_nodes_features,
        ) = self.encoder(grid_nodes_features)
        return grid_nodes_features, mesh_nodes_features

    def decode(self, mesh_nodes_features, grid_nodes_features):
        grid_nodes_features = self.decoder(
            mesh_nodes_features,
            grid_nodes_features,
        )
        return grid_nodes_features

    def forward(self, grid_nodes_features):
        # grid_nodes_features: [batch_size, n_lat, n_lon, n_channel]
        grid_nodes_features, mesh_nodes_features = self.encode(
            grid_nodes_features)
        grid_nodes_features = self.decode(
            mesh_nodes_features, grid_nodes_features)
        return self.restore_shape(grid_nodes_features)

    def prediction_to_xr_variables(self, predictions, dims=('latitude', 'longitude', 'channels')):
        """Converts a prediction tensor to xarray variables.

        Args:
            prediction: A torch.Tensor of shape [batch_size, n_lat * n_lon, n_channel].
            dims: A tuple of strings representing the dimensions of the output xarray variables.

        Returns:
            A dictionary of xarray variables.
        """
        predictions = list(predictions.detach().cpu().numpy())
        # predictions = np.reshape(predictions, (predictions.shape[0], len(self.latitudes), len(self.longitudes), -1))
        # predictions = list(predictions)

        xr_predictions = []
        for prediction in predictions:
            xr_prediction = xr.Variable(
                data=prediction,
                dims=dims,
            )
            xr_predictions.append(xr_prediction)

        return xr_predictions


class GraphCast(GraphCastBase):
    def __init__(self,
                 latitude,
                 longitude,
                 in_dim: int = 363,
                 out_dim: int = 363,
                 mesh_size: int = 6,
                 latent_size: int = 512,
                 radius_query_fraction_edge_length: float = 0.6,
                 nround_mp: int = 3,
                 use_checkpoint=True,
                 residual=True
                 ) -> None:
        super().__init__(latitude, longitude, in_dim, out_dim, mesh_size, latent_size,
                         radius_query_fraction_edge_length, use_checkpoint, residual, False)
        self.n_round_mp = nround_mp  # number of message passing rounds

        mesh2mesh_edge_features, mesh2mesh_edge_index = self._init_mesh_graphs()
        self.processor = MeshProcessor(
            in_mesh_edge_dim=4,
            latent_size=self.latent_size,
            n_processing_layers=nround_mp,
            use_checkpoint=True,
            use_layer_norm=True
        )
        self.processor.register_buffer(
            'mesh2mesh_edge_features', mesh2mesh_edge_features)
        self.processor.register_buffer(
            'mesh2mesh_edge_index', mesh2mesh_edge_index)

    def forward(self, grid_nodes_features):
        # grid_nodes_features: [batch_size, n_channel, n_lat, n_lon]
        grid_nodes_features = einops.rearrange(
            grid_nodes_features, 'b c lat lon  -> b lat lon c')
        
        def forward_(grid_nodes_features):

            grid_nodes_features, mesh_nodes_features = self.encode(
                grid_nodes_features)
            mesh_nodes_features = self.processor(
                mesh_nodes_features
            )

            grid_nodes_features = self.decode(
                mesh_nodes_features, grid_nodes_features)

            return self.restore_shape(grid_nodes_features)

        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(forward_, grid_nodes_features, use_reentrant=False)
        else:
            return forward_(grid_nodes_features)



