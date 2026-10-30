from experiments.exp11_stream_student.cli import run_cli


EXPERIMENT = "exp12_stream_student_nodistill"


def main(argv: list[str] | None = None) -> None:
    run_cli(argv, prog="exp12", experiment=EXPERIMENT, distilled=False, skip=True)
