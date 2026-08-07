from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler

from src.model.gae import GraphAutoEncoder
from src.data.dataset import LeakDBDataset
from src.utils.config import Config


class Trainer:
    def __init__(self, model: GraphAutoEncoder, dataset: LeakDBDataset, config: Config, device: torch.device):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device

        torch.manual_seed(config.random_seed)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # batch_size=1 avoids node index offset handling across graphs in the loss
        self.train_loader = DataLoader(
            dataset, batch_size=1,
            sampler=SubsetRandomSampler(dataset.train_indices),
        )
        self.val_loader = DataLoader(
            dataset, batch_size=1,
            sampler=SubsetRandomSampler(dataset.val_indices),
        )

        Path("checkpoints").mkdir(parents=True, exist_ok=True)

    def train(self) -> dict:
        """Run training loop with early stopping. Returns loss history."""
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train': [], 'val': []}

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._validate()

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

    def _loss(self, z_dict: dict, has_measure_edge_index: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss over sensor-to-value-node assignments.

        For each sensor type, computes cross-entropy between the decoded logits
        and the true bin each sensor connects to at this timestamp.
        No negative sampling needed — softmax over all bins provides implicit contrast.
        """
        gb = self.model.gb
        device = z_dict['sensor'].device

        # scatter has_measure edges into a tensor: sensor_idx -> value_node_idx
        # valid because each sensor has exactly one has_measure edge per timestamp
        value_of_sensor = torch.empty(gb.n_sensors, dtype=torch.long, device=device)
        value_of_sensor[has_measure_edge_index[0]] = has_measure_edge_index[1]

        logits_dict = self.model.decode(z_dict)
        total_loss = torch.tensor(0.0, device=device)

        for st in gb.SENSOR_TYPES:
            sensor_indices = gb.sensor_indices_by_type[st].to(device)
            value_indices = gb.value_indices_by_type[st].to(device)

            # value_node_idx is built with contiguous offsets per type, so subtracting
            # the first bin's global index converts to local 0-based bin indices
            true_bins = value_of_sensor[sensor_indices] - value_indices[0]

            # cross_entropy applies softmax per row independently, so each sensor gets
            # its own probability distribution over its bins — not one softmax over all sensors
            total_loss = total_loss + F.cross_entropy(logits_dict[st], true_bins)

        return total_loss / len(gb.SENSOR_TYPES)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            z_dict = self.model.encode(data.x_dict, data.edge_index_dict)
            has_measure_ei = data[('sensor', 'has_measure', 'value_node')].edge_index
            loss = self._loss(z_dict, has_measure_ei)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                z_dict = self.model.encode(data.x_dict, data.edge_index_dict)
                has_measure_ei = data[('sensor', 'has_measure', 'value_node')].edge_index
                loss = self._loss(z_dict, has_measure_ei)
                total_loss += loss.item()

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
