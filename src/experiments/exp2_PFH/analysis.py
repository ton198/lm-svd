import json
from pathlib import Path

import numpy as np

from utils.analysis import (
    CENTERING_FORMULAS,
    RANK_TOLERANCE_RULE,
    compute_svd,
    format_log_log_fit,
    visualize_svd,
)
from utils.common import save_json, save_numpy, write_atomic
from utils.experiment_paths import ANALYSIS_DIR, INFERENCE_DIR


MATRIX_NAME = "feature_log_prob.npy"
FACTOR_NAMES = {"U": "svd_U.npy", "S": "svd_S.npy", "Vt": "svd_Vt.npy"}
METADATA_NAME = "metadata.json"
PLOT_NAME = "svd_spectrum.png"


def run_analysis(*, result_dir: Path, centering: str, device: str, overwrite: bool) -> str:
    inference_dir = result_dir / INFERENCE_DIR
    matrix_path = inference_dir / MATRIX_NAME
    if not matrix_path.is_file():
        raise ValueError(f"matrix file does not exist: {matrix_path}")
    output_dir = result_dir / ANALYSIS_DIR / centering
    factor_paths = {key: output_dir / name for key, name in FACTOR_NAMES.items()}
    metadata_path = output_dir / METADATA_NAME
    plot_path = output_dir / PLOT_NAME
    if not overwrite:
        for path in (*factor_paths.values(), metadata_path, plot_path):
            if path.exists():
                raise FileExistsError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )
    inference_metadata_path = inference_dir / "metadata.json"
    inference_metadata = (
        json.loads(inference_metadata_path.read_text(encoding="utf-8"))
        if inference_metadata_path.is_file()
        else None
    )
    matrix = np.load(matrix_path)
    m, n = matrix.shape
    svd, svd_summary = compute_svd(matrix, centering=centering, device=device)
    spectrum, loglog_fit = visualize_svd(
        svd.S, rank_tolerance=svd.rank_tolerance, matrix_shape=(m, n), plot_path=plot_path
    )
    for key, factor in (("U", svd.U), ("S", svd.S), ("Vt", svd.Vt)):
        write_atomic(factor_paths[key], lambda tmp: save_numpy(tmp, factor))
    metadata = {
        "input": {
            "matrix_path": str(matrix_path),
            "matrix_shape": [m, n],
            "matrix_dtype": str(matrix.dtype),
            "inference_metadata": inference_metadata,
        },
        "centering": {"strategy": centering, "formula": CENTERING_FORMULAS[centering]},
        "svd": {"factors": dict(FACTOR_NAMES), **svd_summary},
        "rank": {
            "value": svd.rank,
            "tolerance": svd.rank_tolerance,
            "tolerance_rule": RANK_TOLERANCE_RULE,
        },
        "spectrum": spectrum,
        "loglog_fit": loglog_fit,
    }
    write_atomic(metadata_path, lambda tmp: save_json(tmp, metadata))
    return (
        f"output_dir: {output_dir}\n"
        f"centering: {centering}\n"
        f"matrix: {m} x {n}\n"
        f"singular values: max {svd_summary['singular_value_max']:.6g}, "
        f"min {svd_summary['singular_value_min']:.6g}; "
        f"reconstruction relative error {svd_summary['reconstruction_relative_error']:.3e}\n"
        f"rank: {svd.rank} (tolerance {svd.rank_tolerance:.6g})\n"
        f"loglog fit: {format_log_log_fit(loglog_fit)}\n"
        f"metadata: {metadata_path}\n"
        f"plot: {plot_path}"
    )
