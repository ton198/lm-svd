"""Lower the learning rates when a table's captured energy stops rising.

The rule reads only the evaluation history, so a continued run decides as the
uninterrupted one would: among the evaluations made at the current rates, the
mean of the last `window` is compared with the mean of the `window` before
them, and every rate is multiplied by `factor` when it is not `gain` (a
fraction of the earlier mean) above it. After that the rule waits for
2 * window evaluations at the new rates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# The settings of the rule in a run's config; absent from a constant-rate run.
CONFIG_KEY = "lr_plateau"
# The multiple of the first rates an evaluation's steps were trained with.
SCALE_KEY = "lr_scale"


@dataclass(frozen=True)
class Plateau:
    """The table whose top-rank captured energy is watched, and the rule's numbers."""

    table: str
    window: int = 20
    gain: float = 0.01
    factor: float = 0.3

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"plateau window must be at least 1, got {self.window}")
        if self.gain < 0:
            raise ValueError(f"plateau gain must not be negative, got {self.gain}")
        if not 0 < self.factor < 1:
            raise ValueError(f"plateau factor must be between 0 and 1, got {self.factor}")

    def config(self) -> dict:
        return asdict(self)

    def scale(self, history: list[dict]) -> float:
        """The multiple of the first rates for the steps after the last record of history."""
        if not history:
            return 1.0
        current = history[-1].get(SCALE_KEY, 1.0)
        energies = []
        for record in reversed(history):
            if record.get(SCALE_KEY, 1.0) != current:
                break
            energies.append(record[self.table]["captured_energy"][-1])
        if len(energies) < 2 * self.window:
            return current
        recent = sum(energies[:self.window]) / self.window
        before = sum(energies[self.window:2 * self.window]) / self.window
        return current * self.factor if recent - before < self.gain * abs(before) else current

    def describe(self) -> str:
        return (
            f"every learning rate is multiplied by {self.factor:g} when the mean {self.table} "
            f"energy of the last {self.window} evaluations is less than {self.gain:.2%} above "
            f"the mean of the {self.window} before")


def continues(saved: dict, config: dict) -> bool:
    """Whether a run of settings `config` continues a checkpoint of settings `saved`.

    The settings are the same, or `saved` is the same run at constant rates: such
    a run may take up the plateau rule, and is a plateau run from then on.
    """
    return saved == config or saved == {
        key: value for key, value in config.items() if key != CONFIG_KEY}
