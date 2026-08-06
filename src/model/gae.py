import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import to_hetero

from src.model.encoder import GNNEncoder
from src.model.decoder import InnerProductDecoder
from src.data.graph_builder import GraphBuilder
from src.utils.config import Config


class GraphAutoEncoder(nn.Module):
    def __init__(self, config: Config, metadata, graph_builder: GraphBuilder):
        """
        Args:
            config:         hyperparameter config
            metadata:       (node_types, edge_types) from HeteroData.metadata()
            graph_builder:  provides sensor/value_node index mappings per sensor type
        """
        super().__init__()
        encoder = GNNEncoder(
            config.hidden_channels,
            config.latent_channels,
            config.num_layers,
            config.encoder_type,
        )
        self.encoder = to_hetero(encoder, metadata, aggr=config.aggr)
        self.decoder = InnerProductDecoder()
        self.gb = graph_builder

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """Full autoencoder pass for inference: encode → decode → softmax.

        Returns softmax probability distributions per sensor type, ready for
        thresholding during rule extraction.
        Use encode() + decode() directly in the trainer to get raw logits for the loss.
        """
        z_dict = self.encode(x_dict, edge_index_dict)
        logits = self.decode(z_dict)
        return {st: F.softmax(l, dim=-1) for st, l in logits.items()}

    def encode(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """Return per-node-type embedding dicts."""
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, z_dict: dict) -> dict:
        """Return raw logits over value nodes per sensor type.

        For each sensor type, computes logits [n_sensors_of_type, n_bins_of_type]
        against all value nodes of that type via inner product.
        Pass to F.cross_entropy directly in the trainer (applies softmax internally).
        Call forward() instead to get softmax probabilities for rule extraction.

        Returns:
            dict mapping sensor_type -> logits [n_sensors_of_type, n_bins_of_type]
        """
        z_sensor = z_dict['sensor']
        z_value  = z_dict['value_node']
        logits = {}
        for st in self.gb.SENSOR_TYPES:
            sensor_indices = torch.tensor(
                [idx for (stype, _), idx in self.gb.sensor_idx.items() if stype == st],
                device=z_sensor.device,
            )
            value_indices = torch.tensor(
                [idx for (vtype, _), idx in self.gb.value_node_idx.items() if vtype == st],
                device=z_sensor.device,
            )
            logits[st] = self.decoder(z_sensor[sensor_indices], z_value[value_indices])
        return logits
