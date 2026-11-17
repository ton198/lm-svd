"""Print the last evaluation of every exp8 run under a results directory.

    python scripts/exp8_summary.py [data/experiment_results/exp8_hypotheses/<dataset>__<model>]

Reads each run's metadata.json and eval_history.jsonl; needs no model and no GPU.
"""

import json
import sys
from pathlib import Path

DEFAULT = Path("data/experiment_results/exp8_hypotheses/wikitext-large__gpt2-small")


def history(run: Path) -> list[dict]:
    path = run / "eval_history.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def numbers(values: list[float]) -> str:
    return " ".join(f"{value:.3f}" for value in values)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    print("mean runs (study C): energy on each table minus g")
    for run in sorted(root.glob("mean-*")):
        records = history(run)
        if not records:
            continue
        steps = json.loads((run / "metadata.json").read_text())["steps"]
        last = records[-1]
        cells = "  ".join(
            f"{name}: {last[name]['energy']:.4f} (ceiling {last[name]['ceiling']:.4f}, "
            f"exact rank 1 {last[name]['exact_rank1']:.4f})" for name in ("val", "train"))
        print(f"  {run.name}: step {last['step']}/{steps}  {cells}")
    print("center and control runs (studies B and D)")
    for run in sorted([*root.glob("center-*"), *root.glob("minus-g-*")]):
        records = history(run)
        if not records:
            continue
        metadata = json.loads((run / "metadata.json").read_text())
        last = records[-1]
        print(f"  {run.name}: step {last['step']}/{metadata['steps']}, "
              f"{last['seconds_per_step']:.2f} s/step, "
              f"{metadata['trainable_parameters'] / 1e6:.1f}M trained parameters")
        for name in ("val", "train"):
            comparison = last[name]
            print(f"    {name:5s} ranks  {comparison['ranks']}")
            print(f"          energy {numbers(comparison['captured_energy'])}"
                  f" | exact {numbers(comparison['exact_captured_energy'])}")
            print(f"          span energy left {numbers(comparison['left_span_energy'])}"
                  f" | right {numbers(comparison['right_span_energy'])}")
            if "minus_g_energy" in comparison:
                print(f"          table minus g, mean + modes {numbers(comparison['minus_g_energy'])}"
                      f" | exact {numbers(comparison['minus_g_exact_energy'])}"
                      f" | mean alone {comparison['mean_energy']:.4f}")


if __name__ == "__main__":
    main()
