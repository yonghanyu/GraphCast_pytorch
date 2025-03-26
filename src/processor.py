from typing import Dict, Optional
import einops
import torch
import torch.nn as nn
import numpy as np
import torch.utils.checkpoint


from graphnet import GraphMLP, MLP


class MeshProcessor(nn.Module):
    def __init__(self,
                 in_mesh_edge_dim: int = 4,
                 latent_size: int = 512,
                 n_processing_layers: int = 10,
                 use_checkpoint: bool = True,
                 use_layer_norm: bool = True,
                 ) -> None:
        super().__init__()

        self.latent_size = latent_size
        self.use_checkpoint = use_checkpoint

        self.mesh_edge_embedder = MLP(
            in_dim=in_mesh_edge_dim,
            out_dim=latent_size,
            hidden_dim=latent_size,
            hidden_layers=1,
            use_layer_norm=use_layer_norm,
            use_checkpoint=use_checkpoint
        )

        self.processor = GraphMLP(
            src_node_dim=self.latent_size,
            dest_node_dim=self.latent_size,
            edge_dim=self.latent_size,
            node_hidden_dim=self.latent_size,
            node_hidden_layers=1,
            edge_hidden_dim=self.latent_size,
            edge_hidden_layers=1,
            message_passing_steps=n_processing_layers,
            use_layer_norm=use_layer_norm,
            use_checkpoint=use_checkpoint,
        )

    def forward(self,
                mesh_nodes_features: torch.Tensor,
                ) -> torch.Tensor:
        """forward _summary_

        Args:
            mesh_nodes_features (torch.Tensor): B, n_mesh_nodes, n_features
            mesh2mesh_edge_features (torch.Tensor): n_edges, n_features
            edge_index (torch.Tensor): 2, n_edges

        Returns:
            torch.Tensor: _description_
        """

        def forward_(mesh_nodes_features: torch.Tensor,
                     ) -> torch.Tensor: 
            b, _, _ = mesh_nodes_features.shape
            mesh2mesh_edge_features = self.mesh_edge_embedder(
                self.mesh2mesh_edge_features) 

            mesh_nodes_features, mesh2mesh_edge_features = self.processor(
                    mesh_nodes_features, 
                    mesh_nodes_features, 
                    mesh2mesh_edge_features, 
                    self.mesh2mesh_edge_index)
            
            return mesh_nodes_features, mesh2mesh_edge_features


        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(forward_, 
                                                     mesh_nodes_features, 
                                                     use_reentrant=False)
        else: 
            return forward_(mesh_nodes_features)
        
