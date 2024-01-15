
from pathlib import Path
from typing import Optional, Tuple
import torch

from torch.utils.data import DataLoader
from functools import partial
import random

import glob

from TinyLlama.lit_gpt.packed_dataset import CombinedDataset, PackedDataset
# from TinyLlama.pretrain.tinyllama import create_dataloaders

train_data_config = [
    ("train_ind", 1.0),
]

val_data_config = [
    ("train_ind", 1.0),
]



def get_default_supported_precision(training: bool, tpu: bool = False) -> str:
    """Return default precision that is supported by the hardware.

    Args:
        training: `-mixed` or `-true` version of the precision to use
        tpu: whether TPU device is used

    Returns:
        default precision that is suitable for the task and is supported by the hardware
    """
    return "32-true"

    if tpu:
        return "32-true"
    if not torch.cuda.is_available() or torch.cuda.is_bf16_supported():
        return "bf16-mixed" if training else "bf16-true"
    return "16-mixed" if training else "16-true"



def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader, val_dataloader
