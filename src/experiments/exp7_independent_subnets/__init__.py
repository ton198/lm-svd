EXPERIMENT = "exp7_independent_subnets"
SUBNETS = ("scratch", "roberta", "gpt2")
# Kinds that start from pretrained weights: encoders at encoder_lr, heads at lr, no warmup.
PRETRAINED = ("roberta", "gpt2")
# exp4's table directory -> the name exp7 records it under.
TABLE_KEYS = {"train": "selection", "val": "report"}
WARMUP_STEPS = 1000
# Default --lr of each kind. On a finite table the left encoder of a scratch mode stopped
# reading its input at 3e-4 (three runs of three) and never did at 1e-4.
DEFAULT_LR = {"scratch": 1e-4, "roberta": 3e-4, "gpt2": 3e-4}
