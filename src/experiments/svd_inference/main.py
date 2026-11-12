from pathlib import Path

import torch

from .model import get_nested_lora_network
from .train import load_data, train_svd_network


def load_model_from_checkpoint(checkpoint_path, neigs, sequential=False):
    """
    Load a NestedLoRA model from a checkpoint.
    """
    model = get_nested_lora_network(neigs=neigs, sequential=sequential)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def get_checkpoint_saver(checkpoint_path: Path):
    def save_checkpoint(model):
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    return save_checkpoint


def main():
    store_dir = "data/sample_stores/wikitext-103-train__gpt2-small__h64-f32"
    checkpoint_path = Path("trained_model.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    neigs = 128
    batch_size = 32
    tokens_per_pair = 16
    batches_per_epoch = 100
    epochs = 100000
    lr = 3e-4

    torch.manual_seed(seed)
    dataloader = load_data(
        store_dir, batch_size, batches_per_epoch,
        seed=seed, tokens_per_pair=tokens_per_pair,
    )
    if checkpoint_path.exists():
        model = load_model_from_checkpoint(checkpoint_path, neigs=neigs)
    else:
        model = get_nested_lora_network(neigs=neigs)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {device}")
    train_svd_network(model=model,
                      device=device,
                      dataloader=dataloader,
                      optimizer=optimizer,
                      scheduler=None,
                      epochs=epochs,
                      checkpoint_saving=get_checkpoint_saver(checkpoint_path)
                      )
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)


if __name__ == "__main__":
    main()
