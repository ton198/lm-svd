EXPERIMENT = "exp7_independent_subnets"
SUBNETS = ("scratch", "roberta", "gpt2")
# Kinds that start from pretrained weights: encoders at encoder_lr, heads at lr, no warmup.
PRETRAINED = ("roberta", "gpt2")
# exp4's table directory -> the name exp7 records it under.
TABLE_KEYS = {"train": "selection", "val": "report"}
WARMUP_STEPS = 1000
