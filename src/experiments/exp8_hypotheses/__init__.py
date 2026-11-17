EXPERIMENT = "exp8_hypotheses"
# What is removed from a stored table before it is analysed or compared with.
# g: the centered logit of each header token after its future alone (exp5-exp7).
# column-mean: each column's mean over the table's own histories.
BASELINES = ("g", "column-mean")
