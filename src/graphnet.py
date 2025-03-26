from typing import Dict, Optional
import torch.distributed as dist
import torch
from torch import nn
from torch_scatter import scatter_add
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import einops
from attention import TemporalAttentionBlock

class MLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 hidden_layers: int,
                 use_layer_norm: bool = True,
                 use_checkpoint: bool = False,
                 activation = nn.SiLU,) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers - 1
        self.use_checkpoint = use_checkpoint


        _linear_type = nn.Linear

        models = [_linear_type(self.in_dim, self.hidden_dim), activation()]
        for _ in range(self.hidden_layers):
            models.extend(
                [_linear_type(self.hidden_dim, self.hidden_dim), activation()])
        models.append(_linear_type(self.hidden_dim, self.out_dim))
        if use_layer_norm:
            models.append(nn.LayerNorm(self.out_dim))
        self.model = nn.Sequential(*models)

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint: 
            return checkpoint(self.model, x, use_reentrant=False)
        else:
            return self.model(x)
        


class ConcatFreeLinear(nn.Module):
    def __init__(self, in_dim, out_dim, checkpoint=False):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        #self.linear3 = nn.Linear(in_dim, out_dim)
        self.checkpoint = checkpoint
    
    def forward(self, x1, x2):

        def forward_(x1, x2):
            x1 = self.linear1(x1)
            x2 = self.linear2(x2)
            
            return x1 + x2
        return checkpoint(forward_, x1, x2, use_reentrant=False)



class ConcatFreeMLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 hidden_layers: int,
                 use_layer_norm: bool = True,
                 use_checkpoint: bool = False,
                 nested_checkpoint: bool = False,
                 activation = nn.SiLU) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers - 1
        self.use_checkpoint = use_checkpoint
        self.nested_checkpoint = nested_checkpoint

        self.linear1 = ConcatFreeLinear(self.in_dim, self.hidden_dim, checkpoint=self.nested_checkpoint)
        self.act = activation()
        self.out = nn.Linear(self.hidden_dim, self.out_dim)
        self.layer_norm = nn.LayerNorm(self.out_dim) if use_layer_norm else nn.Identity()


    def forward(self, x1: torch.Tensor, x2: torch.Tensor):#, x3: torch.Tensor):
        def forward_(x1, x2):
            combined = self.linear1(x1, x2)
            del x1, x2
            torch.cuda.empty_cache()
            combined = self.act(combined)
            combined = self.out(combined)
            return self.layer_norm(combined) 
        if self.use_checkpoint:
            return checkpoint(forward_, x1, x2, use_reentrant=False)
        else:
            return forward_(x1, x2)


# this class update edge feature with src and dst node feature, with/without residual connection
class EdgeMLP(nn.Module):
    def __init__(self,
                 in_src_node_dim,
                 in_dest_node_dim,
                 edge_dim,
                 out_dim=512,
                 hidden_dim=512,
                 hidden_layers=1,
                 residual=False, 
                 use_layer_norm=True,
                 use_checkpoint=False,
                 nested_checkpoint=False
                 ) -> None:
        super().__init__()
        self.in_src_node_dim = in_src_node_dim
        self.in_dest_node_dim = in_dest_node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.residual = residual
        self.out_dim = out_dim
        self.use_checkpoint = use_checkpoint

        mlp_type = ConcatFreeMLP
        self.edge_mlp = mlp_type(self.in_src_node_dim,
                                 self.out_dim,
                                 self.hidden_dim,
                                 self.hidden_layers,
                                 use_layer_norm=use_layer_norm,
                                 use_checkpoint=use_checkpoint,
                                 )

    def forward(self,
                src: torch.Tensor,
                #dest: torch.Tensor,
                edge_attr: torch.Tensor,):
        """forward batch first  

        Args:
            src (torch.Tensor): [B, n_edges, node_dim]
            dest (torch.Tensor): [B, n_edges, node_dim]
            edge (torch.Tensor): [B, n_edges, edge_dim]

        Returns:
            _type_: _description_
        """
        #edge_attr = torch.cat([src, edge_attr], dim=-1)
        #edge_attr = self.edge_mlp(edge_attr)
        edge_attr = self.edge_mlp(src,  edge_attr)
        return edge_attr



# this class update node feature with a single round of message passing
class NodeMLP(nn.Module):
    def __init__(self,
                 node_dim,
                 edge_dim,
                 hidden_dim=512,
                 hidden_layers=1,
                 out_dim=512,
                 residual=False,
                 use_layer_norm=True,
                 use_checkpoint=False) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.hidden_layers = hidden_layers
        self.residual = residual
        self.use_checkpoint = use_checkpoint

        mlp_type = MLP

        self.node_mlp = mlp_type(self.node_dim + self.edge_dim,
                                 self.out_dim,
                                 self.hidden_dim,
                                 self.hidden_layers,
                                 use_layer_norm=use_layer_norm,
                                 use_checkpoint=use_checkpoint,)

    def forward(self,
                node: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor):
        """forward batch first  

        Args:
            node (torch.Tensor): [B, n_nodes, node_dim]
            edge_index (torch.Tensor): [2, n_edges]
            edge (torch.Tensor): [B, n_edges, edge_dim]

        Returns:
            _type_: _description_
        """
        b, n_nodes, node_dim = node.shape 
        _, receiver = edge_index

        # [B, n_node, edge_dim]
        edge_sum = scatter_add(edge_attr, receiver, dim=1, dim_size=n_nodes)
        
        node = torch.cat([node, edge_sum], dim=-1)
        return self.node_mlp(node)
        

class MessagePassing(nn.Module):
    def __init__(self,
                 src_node_dim,
                 dest_node_dim,
                 edge_dim,
                 node_hidden_dim=512,
                 node_hidden_layers=1,
                 edge_hidden_dim=512,
                 edge_hidden_layers=1,
                 out_dim=None,
                 use_layer_norm=True,
                 use_checkpoint=True,
                 ) -> None:
        super().__init__()
        self.out_dim = out_dim if out_dim is not None else dest_node_dim
        self.src_node_dim = src_node_dim
        self.dest_node_dim = dest_node_dim
        self.edge_dim = edge_dim
        self.node_hidden_dim = node_hidden_dim
        self.node_hidden_layers = node_hidden_layers
        self.edge_hidden_dim = edge_hidden_dim
        self.edge_hidden_layers = edge_hidden_layers
        self.use_layer_norm = use_layer_norm
        self.use_checkpoint = use_checkpoint

        self.edge_mlp = EdgeMLP(
            in_src_node_dim=self.src_node_dim,
            in_dest_node_dim=self.dest_node_dim,
            edge_dim=self.edge_dim,
            out_dim=self.edge_dim,
            hidden_dim=self.edge_hidden_dim,
            hidden_layers=self.edge_hidden_layers,
            use_layer_norm=self.use_layer_norm,
            use_checkpoint=self.use_checkpoint,
            nested_checkpoint=False
        )

        self.node_mlp = NodeMLP(
            node_dim=self.dest_node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.node_hidden_dim,
            hidden_layers=self.node_hidden_layers,
            out_dim=self.out_dim,
            use_layer_norm=self.use_layer_norm,
            use_checkpoint=self.use_checkpoint,

        )

    
    def forward(self,
                src_node: torch.Tensor,
                dest_node: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_index: torch.Tensor
                ):
        """forward batch first  

        Args:
            x (torch.Tensor): [B, n_nodes, node_dim]
            edge_index (torch.Tensor): [2, n_edges]
            edge_attr (torch.Tensor): [B, n_edges, edge_dim] or [n_edges, edge_dim]

        Returns:
            _type_: _description_
        """

        sender, _ = edge_index
        edge_attr_ = self.edge_mlp(src_node[:, sender, :], edge_attr)
        dest_node = dest_node + self.node_mlp(dest_node, edge_index, edge_attr_)
        edge_attr = edge_attr + edge_attr_
        del edge_attr_
        return dest_node, edge_attr
                


class GraphMLP(nn.Module):
    def __init__(self,
                 src_node_dim,
                 dest_node_dim,
                 edge_dim,
                 node_hidden_dim=512,
                 node_hidden_layers=1,
                 edge_hidden_dim=512,
                 edge_hidden_layers=1,
                 out_dim=None,
                 message_passing_steps=1,
                 use_layer_norm=True,
                 use_checkpoint=True,
                 ) -> None:
        super().__init__()
        self.out_dim = out_dim if out_dim is not None else dest_node_dim
        self.src_node_dim = src_node_dim
        self.dest_node_dim = dest_node_dim
        self.edge_dim = edge_dim
        self.node_hidden_dim = node_hidden_dim
        self.node_hidden_layers = node_hidden_layers
        self.edge_hidden_dim = edge_hidden_dim
        self.edge_hidden_layers = edge_hidden_layers
        self.message_passing_steps = message_passing_steps
        self.use_layer_norm = use_layer_norm
        self.use_checkpoint = use_checkpoint
        self.model = []

        for _ in range(message_passing_steps):
            self.model.extend(
                [
                    MessagePassing(
                        self.src_node_dim,
                        self.dest_node_dim,
                        self.edge_dim,
                        self.node_hidden_dim,
                        self.node_hidden_layers,
                        self.edge_hidden_dim,
                        self.edge_hidden_layers,
                        out_dim=self.out_dim,
                        use_layer_norm=self.use_layer_norm,
                        use_checkpoint=self.use_checkpoint
                    )
                ]
            )
        
        self.model = nn.ModuleList(self.model)

    def forward(self,
                src_node: torch.Tensor,
                dest_node: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_index: torch.Tensor
                ):
        """forward batch first  

        Args:
            x (torch.Tensor): [B, n_nodes, node_dim]
            edge_index (torch.Tensor): [2, n_edges]
            edge_attr (torch.Tensor): [B, n_edges, edge_dim] or [n_edges, edge_dim]

        Returns:
            _type_: _description_
        """

        b, _, _ = src_node.shape
        if len(edge_attr.shape) == 2:
            edge_attr = einops.repeat(
                edge_attr, 'n_edges edge_dim -> b n_edges edge_dim', b=b
            ).contiguous()
        
        for i, layer in enumerate(self.model):
            if self.use_checkpoint:
                dest_node, edge_attr = checkpoint(layer, src_node, dest_node, edge_attr, edge_index, use_reentrant=False)
            else:
                dest_node, edge_attr = layer(src_node, dest_node, edge_attr, edge_index)
            src_node = dest_node
        return dest_node, edge_attr
        


