from pathlib import Path

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler

from src.model.gae import GraphAutoEncoder
from src.data.dataset import LeakDBDataset
from src.training.masking import Masker
from src.utils.config import Config


class Trainer:
    def __init__(self, model: GraphAutoEncoder, dataset: LeakDBDataset, config: Config, device: torch.device):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device

        torch.manual_seed(config.random_seed)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        self.train_loader = DataLoader(
            dataset, batch_size=config.batch_size,
            sampler=SubsetRandomSampler(dataset.train_indices),
        )
        self.val_loader = DataLoader(
            dataset, batch_size=config.batch_size,
            sampler=SubsetRandomSampler(dataset.val_indices),
        )

        Path("checkpoints").mkdir(parents=True, exist_ok=True)
        self.masker = Masker(model.gb, config)
        

    def train(self) -> dict:
        """Run training loop with early stopping. Returns loss history."""
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train': [], 'val': []}

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate(epoch)

            history['train'].append(train_loss)
            history['val'].append(val_loss)
            print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                if patience_counter == self.config.patience:
                    print(f"Early stopping at epoch {epoch} (patience={self.config.patience})")
                    break

        return history
    

    def _loss(self, z_dict: dict, measured_by_edge_index: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss over sensor-to-value-node assignments.

        For each sensor type, computes cross-entropy between the decoded logits
        and the true bin each sensor connects to at this timestamp.
        No negative sampling needed — softmax over all bins provides implicit contrast.
        Supports batch_size > 1: logits and true_bins are [n_graphs * n_sensors_of_type].
        """
        gb = self.model.gb
        device = z_dict['sensor'].device
        n_graphs = z_dict['sensor'].shape[0] // gb.n_sensors

        # scatter measured_by edges into a flat lookup: batched_sensor_idx -> batched_value_node_idx
        # edge direction is value_node -> sensor, so [0]=value_node, [1]=sensor
        value_of_sensor = torch.empty(n_graphs * gb.n_sensors, dtype=torch.long, device=device)
        value_of_sensor[measured_by_edge_index[1]] = measured_by_edge_index[0]

        logits_dict = self.model.decode(z_dict)
        total_loss = torch.tensor(0.0, device=device)

        for st in gb.SENSOR_TYPES:
            sensor_indices = gb.sensor_indices_by_type[st].to(device)
            bin_offset = gb.value_indices_by_type[st][0].item()

            if n_graphs == 1:
                true_bins = value_of_sensor[sensor_indices] - bin_offset
            else:
                g_idx = torch.arange(n_graphs, device=device)
                # gather each graph's sensor rows: [n_graphs, n_sensors_of_type]
                s_idx_all = sensor_indices.unsqueeze(0) + (g_idx * gb.n_sensors).unsqueeze(1)
                raw_value_idx = value_of_sensor[s_idx_all]
                # subtract per-graph value_node offset and type bin_offset to get local bin
                v_off = (g_idx * gb.n_value_nodes).unsqueeze(1)
                true_bins = (raw_value_idx - v_off - bin_offset).flatten()

            total_loss = total_loss + F.cross_entropy(logits_dict[st], true_bins)

        return total_loss / len(gb.SENSOR_TYPES)
    

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        progress = tqdm(self.train_loader, desc=f" train epoch {epoch}", colour="green", leave=False)
        for data in progress:
            data = data.to(self.device)
            measured_by_ei = data[('value_node', 'measured_by', 'sensor')].edge_index
            if self.config.use_masking:
                data = self.masker.apply(data)
            self.optimizer.zero_grad()
            z_dict = self.model.encode(data.x_dict, data.edge_index_dict)
            loss = self._loss(z_dict, measured_by_ei)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)
    

    def _validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            progress = tqdm(self.val_loader, desc=f"  val epoch {epoch}", colour="blue", leave=False)
            for data in progress:
                data = data.to(self.device)
                z_dict = self.model.encode(data.x_dict, data.edge_index_dict)
                measured_by_ei = data[('value_node', 'measured_by', 'sensor')].edge_index
                loss = self._loss(z_dict, measured_by_ei)
                total_loss += loss.item()
                progress.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.val_loader)
    

    def _save_checkpoint(self, epoch: int, val_loss: float):
        torch.save({
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, 'checkpoints/best_model.pt')


    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']
