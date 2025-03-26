from typing import Tuple
import einops
import torch
import torch.nn as nn
import numpy as np
import torch.utils.checkpoint


from graphnet import MLP, GraphMLP


class Mesh2GridDecoder(nn.Module):
    def __init__(self,
                 latent_grid_node_dim: int = 512,
                 out_grid_node_dim: int = 471,
                 latent_mesh_node_dim: int = 512,
                 latent_edge_dim: int = 512,
                 n_processing_layers: int = 1,
                 use_checkpoint: bool = False,
                 use_layer_norm: bool = False,
                 ) -> None:
        super().__init__()

        self.latent_grid_node_dim = latent_grid_node_dim
        self.out_grid_node_dim = out_grid_node_dim

        self.latent_mesh_node_dim = latent_mesh_node_dim
        self.latent_edge_dim = latent_edge_dim
        self.n_processing_layers = n_processing_layers
        self.use_layer_norm = use_layer_norm
        self.use_checkpoint = use_checkpoint

        self.mesh2grid_edge_embedder = MLP(
            in_dim=4,
            out_dim=self.latent_edge_dim,
            hidden_dim=self.latent_edge_dim,
            hidden_layers=1,
            use_layer_norm=use_layer_norm,
            use_checkpoint=use_checkpoint
        )

        self.graph_processor = GraphMLP(
            src_node_dim=latent_mesh_node_dim,
            dest_node_dim=latent_grid_node_dim,
            edge_dim=latent_edge_dim,
            node_hidden_dim=latent_mesh_node_dim,
            node_hidden_layers=1,
            edge_hidden_dim=latent_edge_dim,
            edge_hidden_layers=1,
            message_passing_steps=n_processing_layers,
            use_layer_norm=use_layer_norm,
            use_checkpoint=use_checkpoint,
            nested_checkpoint=True
        )

        self.out_mlp = MLP(
            in_dim=latent_grid_node_dim,
            out_dim=out_grid_node_dim,
            hidden_dim=latent_grid_node_dim,
            hidden_layers=1,
            use_layer_norm=False,
            use_checkpoint=use_checkpoint,
        )

    def forward(self,
                mesh_node_features: torch.Tensor,
                grid_node_features: torch.Tensor
                ):
        """forward _summary_

        Args:
            mesh_node_features (torch.Tensor): B, n_mesh_nodes, n_features
            grid_node_features (torch.Tensor): B, n_grid_nodes, n_features


        Returns:
            _type_: _description_
        """

        def forward_(mesh_node_features,
                     grid_node_features):
            b, _, _ = grid_node_features.shape #3274

            # e^M2G = MLP(e^M2G)
            mesh2grid_edge_features = self.mesh2grid_edge_embedder(
                self.mesh2grid_edge_features) #9358

            # update e^M2G', V^G'
            # V^G = V^G + V^G'
            grid_node_features, mesh2grid_edge_features = self.graph_processor(
                mesh_node_features,
                grid_node_features,
                mesh2grid_edge_features,
                self.mesh2grid_edge_index
            )

            #mesh2grid_edge_features.detach()
            del mesh2grid_edge_features
            torch.cuda.empty_cache()
            grid_node_features = self.out_mlp(grid_node_features)
            return grid_node_features.float()

        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(forward_,
                                                     mesh_node_features,
                                                     grid_node_features,
                                                     use_reentrant=False)
        else:
            return forward_(mesh_node_features,
                            grid_node_features)
