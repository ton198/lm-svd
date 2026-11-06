"""Sample generation, persistence and sampling; imports remain lightweight."""

__all__ = ["SampleSampler"]


def __getattr__(name):
    if name == "SampleSampler":
        from .store import SampleSampler

        return SampleSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
