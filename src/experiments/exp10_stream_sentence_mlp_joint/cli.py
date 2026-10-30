from experiments.exp9_stream_sentence_mlp.cli import run_cli


EXPERIMENT = "exp10_stream_sentence_mlp_joint"


def main(argv: list[str] | None = None) -> None:
    run_cli(argv, prog="exp10", experiment=EXPERIMENT, nesting="joint")
