import torch
from data_pipline import SampleSampler

sampler = SampleSampler(
    "data/sample_stores/wikitext-103-train__gpt2-small__h64-f32",
    seed=0,
)
sampler.wait_until_ready()

# 单条样本，返回的都是 CPU tensor
h, f, logits = sampler.sample()

# 组一个 batch，例如 batch size = 32
samples = [sampler.sample() for _ in range(32)]
h, f, logits = [torch.stack(items) for items in zip(*samples)]

# 按训练设备转移
h, f, logits = [x.to("cuda") for x in (h, f, logits)]

