from pathlib import Path
from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from data_pipline.sample_store import SampleSampler
from utils.nestedlora import NestedLoRA

from torch import nn

def train_svd_network(model: NestedLoRA, device : str, dataloader, optimizer, scheduler, epochs, checkpoint_saving : Callable[[nn.Module], None] | None = None):
    """Train on batches of h (B,H), f_concat_z (B,Z,F+1), scores (B,Z)."""
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for h, f_concat_z, scores in dataloader:
            h, f_concat_z, scores = [
                tensor.to(device) for tensor in (h, f_concat_z, scores)
            ]
            optimizer.zero_grad()

            left, right = model(h, f_concat_z.flatten(0, 1))
            right = right.reshape(h.shape[0], scores.shape[1], -1)
            # Each kernel row belongs to its own sampled (h, f) pair.
            loss = torch.stack([
                model.compute_loss_kernel(
                    left[i:i + 1], right[i], scores[i:i + 1]
                )[0]
                for i in range(h.shape[0])
            ]).mean()
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        
        if checkpoint_saving is not None and (epoch + 1) % 100 == 0:
            checkpoint_saving(model)
        train_loss /= len(dataloader)
        print(f"Epoch {epoch}: Train loss: {train_loss:.4f}")

    return model


class _SampleStoreDataset(Dataset):
    def __init__(
        self, store_dir: str | Path, num_samples: int, seed: int, tokens_per_pair: int
    ):
        self.sampler = SampleSampler(store_dir, seed=seed)
        self.num_samples = num_samples
        self.tokens_per_pair = tokens_per_pair
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        h, f, logits = self.sampler.sample()
        token_ids = torch.randint(
            logits.shape[0], (self.tokens_per_pair,), generator=self.generator
        )
        f_concat_z = torch.cat(
            (f.unsqueeze(0).expand(self.tokens_per_pair, -1), token_ids.unsqueeze(-1)),
            dim=-1,
        )
        return h, f_concat_z, logits[token_ids]


def load_data(
    store_dir: str | Path,
    batch_size: int,
    batches_per_epoch: int,
    *,
    seed: int = 0,
    tokens_per_pair: int = 32,
) -> DataLoader:
    """Sample h (B,H), f_concat_z (B,Z,F+1), scores (B,Z), with Z=tokens_per_pair."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batches_per_epoch <= 0:
        raise ValueError("batches_per_epoch must be positive")
    if tokens_per_pair <= 0:
        raise ValueError("tokens_per_pair must be positive")
    dataset = _SampleStoreDataset(
        store_dir,
        num_samples=batch_size * batches_per_epoch,
        seed=seed,
        tokens_per_pair=tokens_per_pair,
    )
    return DataLoader(dataset, batch_size=batch_size)
