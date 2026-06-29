"""Try ONNX execution providers in caller preference order."""

from typing import TypeVar
from collections.abc import Callable, Sequence

T = TypeVar("T")

DEFAULT_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


def load_with_providers(providers: Sequence[str], available: Sequence[str], load: Callable[[str], T]) -> tuple[T, str]:
    if isinstance(providers, str) or not providers or any(not isinstance(p, str) or not p for p in providers):
        raise ValueError("providers must be a non-empty sequence of provider names")

    failures: list[tuple[str, Exception]] = []
    for provider in dict.fromkeys(providers):
        if provider not in available:
            continue
        try:
            return load(provider), provider
        except Exception as exc:
            failures.append((provider, exc))

    if failures:
        names = ", ".join(name for name, _ in failures)
        raise RuntimeError(f"could not initialize an ONNX provider ({names})") from failures[-1][1]
    raise ValueError(f"no requested ONNX provider is available: {', '.join(providers)}")
