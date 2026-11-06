"""Data preparation and sampling. Heavy dependencies load only when needed."""

__all__ = ["SampleSampler"]


def __getattr__(name):
    if name == "SampleSampler":
        from .sample_store import SampleSampler

        return SampleSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
