from typing import Tuple
import einops
import torch
import numpy as np
import torch.nn as nn

from graphnet import MLP, GraphMLP


# TODO: setup default number of features
class Grid2MeshEncoder(nn.Module):
    def __init__(self,
                 in_grid_node_dim: int = 512,
                 latent_grid_node_dim: int = 512,
                 in_mesh_node_dim: int = 3,
                 latent_mesh_node_dim: int = 512,
                 in_edge_dim: int = 4,
                 latent_edge_dim: int = 512,
                 n_processing_layers: int = 1,
                 use_layer_norm: bool = True,
                 use_checkpoint: bool = False,
                 ) -> None:
        """__init__ _summary_

        Args:
            Note: in default, we have a bipartite graph, with grid nodes and mesh nodes, and edges from grid to mesh nodes
            We first embed nodes features and mesh features. Note that by default: 
            - grid nodes have 474 features
            - mesh nodes have 3 features
            - grid to mesh edges have 4 features 

            graph (data.Data): _description_
            in_grid_node_dim (int, optional): _description_. Defaults to 128.
            hidden_grid_node_dim (int, optional): _description_. Defaults to 128.
            latent_grid_node_dim (int, optional): _description_. Defaults to 128.
            in_mesh_node_dim (int, optional): _description_. Defaults to 128.
            hidden_mesh_node_dim (int, optional): _description_. Defaults to 128.
            latent_mesh_node_dim (int, optional): _description_. Defaults to 128.
            in_edge_dim (int, optional): _description_. Defaults to 128.
            hidden_edge_dim (int, optional): _description_. Defaults to 128.
            latent_edge_dim (int, optional): _description_. Defaults to 128.
            n_node_layers (int, optional): _description_. Defaults to 1.
            n_edge_layers (int, optional): _description_. Defaults to 1.
        """

        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.in_grid_node_dim = in_grid_node_dim
        self.latent_grid_node_dim = latent_grid_node_dim
        self.in_mesh_node_dim = in_mesh_node_dim
        self.latent_mesh_node_dim = latent_mesh_node_dim
        self.in_edge_dim = in_edge_dim
        self.latent_edge_dim = latent_edge_dim
        self.n_processing_layers = n_processing_layers
        self.use_checkpoint = use_checkpoint

        self.grid_node_embedder = MLP(
            in_dim=in_grid_node_dim,
            out_dim=self.latent_grid_node_dim,
            hidden_dim=self.latent_grid_node_dim,
            hidden_layers=1,
            use_layer_norm=True,
            use_checkpoint=use_checkpoint
        )

        self.mesh_node_embedder = MLP(
            in_dim=in_mesh_node_dim,
            out_dim=self.latent_mesh_node_dim,
            hidden_dim=self.latent_mesh_node_dim,
            hidden_layers=1,
            use_layer_norm=True,
            use_checkpoint=use_checkpoint
        )

        self.grid2mesh_edge_embedder = MLP(
            in_dim=in_edge_dim,
            out_dim=self.latent_edge_dim,
            hidden_dim=self.latent_edge_dim,
            hidden_layers=1,
            use_layer_norm=True,
            use_checkpoint=use_checkpoint
        )

        self.graph_processor = GraphMLP(
            src_node_dim=latent_grid_node_dim,
            dest_node_dim=latent_mesh_node_dim,
            edge_dim=latent_edge_dim,
            node_hidden_dim=latent_grid_node_dim,
            node_hidden_layers=1,
            edge_hidden_dim=latent_edge_dim,
            edge_hidden_layers=1,
            message_passing_steps=n_processing_layers,
            use_layer_norm=use_layer_norm,
            use_checkpoint=use_checkpoint
        )

        self.grid_node_mlp = MLP(
            in_dim=latent_grid_node_dim,
            out_dim=latent_grid_node_dim,
            hidden_dim=latent_grid_node_dim,
            hidden_layers=1,
            use_layer_norm=use_layer_norm,
            use_checkpoint=use_checkpoint
        )

    def forward(self,
                grid_nodes_features: torch.Tensor
                ):
        """forward _summary_

        Args:
            grid_nodes_features (torch.Tensor): B, n_grid_nodes, n_features
            mesh_nodes_features (torch.Tensor): n_mesh_nodes, n_features
            grid2mesh_edge_feature (torch.Tensor): n_edges, n_features
            edge_index (torch.Tensor): 2, n_edges
        """
        def forward_(grid_nodes_features):
            b, _, _ = grid_nodes_features.shape

            grid_nodes_geofeatures = einops.repeat(
                self.grid_nodes_geo_features, 'n f -> b n f', b=b)  # [B, n_grid_nodes, 3]
            grid_nodes_features = torch.cat(
                [grid_nodes_features, grid_nodes_geofeatures], dim=-1)

            # embedding
            grid_nodes_features = self.grid_node_embedder(grid_nodes_features)
            mesh_nodes_features = self.mesh_node_embedder(self.mesh_nodes_geo_features)
            mesh_nodes_features = einops.repeat(
                mesh_nodes_features, 'n_nodes n_features -> b n_nodes n_features', b=b).contiguous()

            grid2mesh_edge_feature = self.grid2mesh_edge_embedder(
                self.grid2mesh_edge_features)

            # update e^G2N', V^M'
            # v^M = v^M + v^M'

            mesh_nodes_features, grid2mesh_edge_feature = self.graph_processor(
                grid_nodes_features,
                mesh_nodes_features,
                grid2mesh_edge_feature,
                self.grid2mesh_edge_index
            )  # 7216

            del grid2mesh_edge_feature
            torch.cuda.empty_cache()

            # v^G = MLP(v^G) + v^G
            out = self.grid_node_mlp(
                grid_nodes_features) + grid_nodes_features
            del grid_nodes_features
            torch.cuda.empty_cache()
            return out.float(), mesh_nodes_features.float()

        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(forward_,
                                                     grid_nodes_features,
                                                     use_reentrant=False)
        else:
            return forward_(grid_nodes_features)
