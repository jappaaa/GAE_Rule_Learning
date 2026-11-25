"""Builds the right Dataset class for config.dataset_name.

LeakDBDataset, DiTECWDNDataset and LBNLDataset all share the same constructor
signature (config, transform=None) and expose the same interface (gb, train_indices,
val_indices, __len__/__getitem__), so every caller that currently hardcodes one of
them can switch to build_dataset(config) instead without any other changes.
"""
from src.data.ditec_wdn.dataset import DiTECWDNDataset
from src.data.lbnl_b59.dataset import LBNLDataset
from src.data.leakdb.dataset import LeakDBDataset

_DATASETS = {
    'leakdb': LeakDBDataset,
    'ditec_wdn': DiTECWDNDataset,
    'lbnl_b59': LBNLDataset,
}


def build_dataset(config, transform=None):
    """Instantiate the Dataset class matching config.dataset_name."""
    try:
        dataset_cls = _DATASETS[config.dataset_name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset_name '{config.dataset_name}'; expected one of {sorted(_DATASETS)}"
        )
    return dataset_cls(config, transform=transform)
