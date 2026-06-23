"""Synchronous recognition contracts for independent bounded audio runs.

Input is ordered mono 16 kHz PCM S16LE. Each feed has one uniform speech label;
recognizers consume all samples regardless of that label. Calls block the caller;
use a worker outside an asyncio transport loop. Serialize calls to each run.
"""

from enum import Enum
from typing import Protocol
from dataclasses import dataclass

SAMPLE_RATE = 16_000


class RunOutcome(str, Enum):
    """Terminal reason for a run; None means it is still accepting input."""

    FINISHED = "finished"
    ABORTED = "aborted"
    FAILED = "failed"
    OVERFLOW = "overflow"


@dataclass(frozen=True)
class Transcript:
    """Complete text so far; T-one only publishes completed phrases."""

    text: str
    revision: int
    sample_end: int
    final: bool
    phrase_start: float | None = None
    phrase_end: float | None = None


class RecognitionRun(Protocol):
    """A single isolated, ordered and bounded input segment."""

    @property
    def outcome(self) -> RunOutcome | None:
        """Terminal reason, or None while the run is active."""
        ...

    def feed(self, pcm: bytes, *, speech: bool = True) -> tuple[Transcript, ...]:
        """Consume contiguous PCM with a uniform speech label; return observations."""
        ...

    def finish(self) -> tuple[Transcript, ...]:
        """Flush the final incomplete window and publish a final transcript."""
        ...

    def abort(self) -> None:
        """Discard an interrupted run; no final transcript is published."""
        ...


class Recognizer(Protocol):
    """Reusable loaded model; implementation owns its native audio conversion."""

    def new_run(self, *, max_samples: int) -> RecognitionRun:
        """Create a fresh run with a caller-chosen bound in 16-kHz samples."""
        ...


class _RunBase:
    def __init__(self, max_samples: int) -> None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0:
            raise ValueError("max_samples must be a positive integer")
        self.max_samples = max_samples
        self.sample_end = 0
        self.revision = 0
        self._closed = False
        self._outcome: RunOutcome | None = None

    @property
    def outcome(self) -> RunOutcome | None:
        return self._outcome

    def _terminate(self, outcome: RunOutcome) -> None:
        self._closed = True
        self._outcome = outcome

    def _accept(self, pcm: bytes, speech: bool = True) -> None:
        if self._closed:
            raise RuntimeError("run already finished or aborted")
        if not isinstance(pcm, bytes) or len(pcm) % 2:
            raise ValueError("expected complete mono PCM S16LE samples as bytes")
        if not isinstance(speech, bool):
            raise ValueError("speech must be a bool")
        if self.sample_end + len(pcm) // 2 > self.max_samples:
            self._terminate(RunOutcome.OVERFLOW)
            raise OverflowError("run audio bound exceeded")
        self.sample_end += len(pcm) // 2

    def _finish(self) -> None:
        if self._closed:
            raise RuntimeError("run already finished or aborted")
        # Close admission before running the blocking final inference.
        self._closed = True

    def _complete(self) -> None:
        self._terminate(RunOutcome.FINISHED)

    def _fail(self) -> None:
        self._terminate(RunOutcome.FAILED)

    def _output(
        self, text: str, *, final: bool, phrase_start: float | None = None, phrase_end: float | None = None
    ) -> Transcript:
        self.revision += 1
        return Transcript(
            text=text,
            revision=self.revision,
            sample_end=self.sample_end,
            final=final,
            phrase_start=phrase_start,
            phrase_end=phrase_end,
        )

    def abort(self) -> None:
        if self._closed:
            raise RuntimeError("run already finished or aborted")
        self._terminate(RunOutcome.ABORTED)


def feed_pcm(run: RecognitionRun, pcm: bytes, *, chunk_samples: int = 1600) -> tuple[Transcript, ...]:
    """Feed a complete PCM buffer in uniform speech=True chunks, without finishing.

    This is a convenience for recordings without VAD. It does not decode a file
    format; callers supply raw PCM. An empty buffer produces no feed calls.
    """
    if isinstance(chunk_samples, bool) or not isinstance(chunk_samples, int) or chunk_samples <= 0:
        raise ValueError("chunk_samples must be a positive integer")
    if not isinstance(pcm, bytes) or len(pcm) % 2:
        raise ValueError("expected complete mono PCM S16LE samples as bytes")
    events: list[Transcript] = []
    for offset in range(0, len(pcm), chunk_samples * 2):
        events.extend(run.feed(pcm[offset : offset + chunk_samples * 2], speech=True))
    return tuple(events)
