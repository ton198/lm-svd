EXPERIMENT = "exp5_gpt2_mlp_table"
TRAIN_DIR = "train"
# --lr-schedule: cosine lowers every learning rate to 0 over the run.
LR_SCHEDULES = ("constant", "cosine")
# A stored header token may sit this far below the recomputed top-k boundary.
HEADER_TOLERANCE = 1e-3
