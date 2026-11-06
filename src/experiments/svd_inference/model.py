import torch
from torch import nn
from utils.nestedlora import NestedLoRA

class ParallelMLP(nn.Module):
    """
    For every singular function f_i, we construct a separate MLP.
    U = [f_1, f_2, ..., f_n] is the matrix of singular functions.
    """
    def __init__(self, input_dim, hidden_dim, number_of_models):
        super().__init__()
        models = []
        for _ in range(number_of_models):
            model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            models.append(model)
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.stack(outputs, dim=-1)

def get_nested_lora_model(base_model, input_dim, hidden_dim, neigs, sequential=False):
    """
    Wraps the base model with nested LoRA Layers.
    """
    model = NestedLoRA(
        model=base_model,
        neigs=neigs,
        sequential=sequential
    )
    return model