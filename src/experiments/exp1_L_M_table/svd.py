import argparse
import json
from pathlib import Path

import numpy as np

from utils import matrix_svd

from .table import _save_numpy, _write_atomic


INPUT_NAME = "matrix.npy"
OUTPUT_NAMES = ("svd_U.npy", "svd_S.npy", "svd_Vt.npy", "svd_metadata.json")


def compute_svd(
    *, input_dir: Path, output_dir: Path, overwrite: bool
) -> dict[str, int | float | str]:
    matrix_path = input_dir / INPUT_NAME
    if not matrix_path.is_file():
        raise ValueError(f"matrix file does not exist: {matrix_path}")
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"output already exists: {path}; pass --overwrite to replace it"
            )

    matrix = np.load(matrix_path)
    result = matrix_svd(matrix)
    data = matrix.astype(np.float64)
    frobenius_norm = float(np.linalg.norm(data))
    reconstruction_relative_error = (
        float(np.linalg.norm(data - (result.U * result.S) @ result.Vt) / frobenius_norm)
        if frobenius_norm != 0
        else 0.0
    )
    m, n = matrix.shape
    table_metadata_path = input_dir / "metadata.json"
    metadata = {
        "input_path": str(matrix_path),
        "matrix_shape": [m, n],
        "input_dtype": str(matrix.dtype),
        "compute_dtype": "float64",
        "centering": "none (SVD of the stored matrix)",
        "svd": "thin: matrix = U @ diag(S) @ Vt",
        "rank": result.rank,
        "rank_tolerance": result.rank_tolerance,
        "rank_tolerance_rule": "S.max() * max(matrix.shape) * finfo(input_dtype).eps",
        "singular_value_max": float(result.S[0]),
        "singular_value_min": float(result.S[-1]),
        "frobenius_norm": frobenius_norm,
        "reconstruction_relative_error": reconstruction_relative_error,
        "table_metadata": (
            json.loads(table_metadata_path.read_text(encoding="utf-8"))
            if table_metadata_path.exists()
            else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, factor in zip(OUTPUT_NAMES[:3], (result.U, result.S, result.Vt)):
        _write_atomic(output_dir / name, lambda tmp: _save_numpy(tmp, factor))
    _write_atomic(
        output_dir / OUTPUT_NAMES[3],
        lambda tmp: tmp.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        ),
    )
    return {
        "output_dir": str(output_dir),
        "rows": m,
        "columns": n,
        "rank": result.rank,
        "singular_value_max": float(result.S[0]),
        "singular_value_min": float(result.S[-1]),
        "reconstruction_relative_error": reconstruction_relative_error,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.exp1_L_M_table.svd",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/experiment_results/exp1_L_M_table/gpt2-small"),
        help="Directory containing matrix.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; use the input directory when unset",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    args = parser.parse_args(argv)
    output_dir = args.input_dir if args.output_dir is None else args.output_dir
    try:
        summary = compute_svd(
            input_dir=args.input_dir, output_dir=output_dir, overwrite=args.overwrite
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(
        f"output_dir: {summary['output_dir']}\n"
        f"matrix: {summary['rows']} x {summary['columns']}\n"
        f"rank: {summary['rank']}\n"
        f"singular values: max {summary['singular_value_max']:.6g}, "
        f"min {summary['singular_value_min']:.6g}; "
        f"reconstruction relative error {summary['reconstruction_relative_error']:.3e}"
    )


if __name__ == "__main__":
    main()
