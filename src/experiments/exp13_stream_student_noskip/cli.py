from experiments.exp11_stream_student.cli import run_cli


EXPERIMENT = "exp13_stream_student_noskip"


def main(argv: list[str] | None = None) -> None:
    run_cli(argv, prog="exp13", experiment=EXPERIMENT, distilled=True, skip=False)
