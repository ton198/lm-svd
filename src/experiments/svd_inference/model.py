from safetensors import safe_open
import torch
from torch import nn

from utils.nestedlora import NestedLoRA


VOCAB_SIZE = 50257

def load_gpt_embedding(model_path):
    with safe_open(
        f"{model_path}/model.safetensors",
        framework="pt",
        device="cpu",
    ) as checkpoint:
        weight = checkpoint.get_tensor("wte.weight")

    return nn.Embedding.from_pretrained(weight, freeze=False)

class SingularNetwork(nn.Module):
    def __init__(self, hidden_dim, layers, neigs, max_seq_len=512, use_pretrained_embedding=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.neigs = neigs

        if use_pretrained_embedding:
            self.token_embedding = nn.Sequential(
                load_gpt_embedding("data/sample_stores/wikitext-103-train__gpt2-small__h64-f32"),
                nn.Linear(768, hidden_dim),
            )
        else:
            self.token_embedding = nn.Embedding(VOCAB_SIZE, hidden_dim)
        

        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        nhead = next(heads for heads in (4, 2, 1) if hidden_dim % heads == 0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=2 * hidden_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_layer = nn.Linear(hidden_dim, neigs)

    def forward(self, input_ids):
        """Map unpadded token IDs (..., T) to function values (..., neigs)."""
        batch_shape = input_ids.shape[:-1]
        seq_len = input_ids.shape[-1]
        positions = torch.arange(seq_len, device=input_ids.device)
        tokens = self.token_embedding(input_ids.reshape(-1, seq_len))
        tokens = tokens + self.position_embedding(positions)
        features = self.transformer(tokens).mean(dim=1)
        return self.output_layer(features).reshape(*batch_shape, self.neigs)


class LoRANetwork(nn.Module):
    def __init__(self, neigs):
        super().__init__()
        self.neigs = neigs
        self.left_singular_function = SingularNetwork(
            hidden_dim=128,
            layers=2,
            neigs=neigs,
            max_seq_len=512
        )
        self.right_singular_function = SingularNetwork(
            hidden_dim=1024,
            layers=2,
            neigs=neigs,
            max_seq_len=512
        )

    def forward(self, h, f_concat_z):
        """
        h: (B, L)
        f_concat_z: (B, L)
        """
        left_singular_values = self.left_singular_function(h)  # (B, neigs)
        right_singular_values = self.right_singular_function(f_concat_z)  # (B, neigs)
        return left_singular_values, right_singular_values


def get_nested_lora_network(neigs, sequential=False):
    """
    Wraps the base model with nested LoRA Layers.
    """
    model = NestedLoRA(
        model=LoRANetwork(neigs=neigs),
        neigs=neigs,
        sequential=sequential
    )
    return model
