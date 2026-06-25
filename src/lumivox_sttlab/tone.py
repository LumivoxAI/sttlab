"""Stateful T-one ONNX CTC with completed-phrase greedy output.

The implementation follows the acoustic padding, splitter and time correction
of voicekit-team/T-one (Apache-2.0, revision 3c5b6c015038173840e62cea99e10cdb1c759116).
It does not emit unfinished within-phrase partial text or require KenLM/pyctcdecode.
"""

from pathlib import Path
from itertools import groupby

import numpy as np
import numpy.typing as npt

from .recognition import Transcript, _RunBase

_LABELS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "
_WINDOW = 2400  # 300 ms at 8 kHz


class Tone:
    """Reusable T-one acoustic ONNX session; mutable state belongs to each run."""

    def __init__(self, path: str | Path, *, provider: str = "CPUExecutionProvider") -> None:
        import onnxruntime as ort

        if provider not in ort.get_available_providers():
            raise ValueError(f"unavailable ONNX provider: {provider}")
        model = Path(path)
        if not model.is_file():
            raise FileNotFoundError(model)
        self._session = ort.InferenceSession(str(model), providers=[provider])
        expected_inputs = {"signal": ("tensor(int32)", [1, 2400, 1]), "state": ("tensor(float16)", [1, 219729])}
        expected_outputs = {"logprobs": ("tensor(float)", [10, 35]), "state_next": ("tensor(float16)", [219729])}
        for inp in self._session.get_inputs():
            if inp.name not in expected_inputs:
                raise ValueError(f"unexpected T-one input: {inp.name}")
            dtype, shape = expected_inputs[inp.name]
            if inp.type != dtype or inp.shape[1:] != shape[1:]:
                raise ValueError(f"incompatible T-one input: {inp.name}")
        if {inp.name for inp in self._session.get_inputs()} != set(expected_inputs):
            raise ValueError("incomplete T-one input contract")
        if {out.name: (out.type, out.shape[1:]) for out in self._session.get_outputs()} != expected_outputs:
            raise ValueError("incompatible T-one output contract")

    def new_run(self, *, max_samples: int) -> "ToneRun":
        """Allocate fresh resampler, acoustic state and phrase splitter state."""
        return ToneRun(self, max_samples)


class ToneRun(_RunBase):
    def __init__(self, runtime: Tone, max_samples: int) -> None:
        super().__init__(max_samples)
        import soxr

        self._runtime = runtime
        self._resampler = soxr.ResampleStream(16_000, 8_000, 1, dtype="int16")
        self._pcm = np.zeros(_WINDOW, dtype=np.int32)  # upstream left padding
        self._state = np.zeros((1, 219729), dtype=np.float16)
        self._past = np.empty((0, 35), dtype=np.float32)
        self._offset = 0
        self._phrases: list[str] = []

    def feed(self, pcm: bytes, *, speech: bool = True) -> tuple[Transcript, ...]:
        try:
            self._accept(pcm, speech)
        except OverflowError:
            self._release()
            raise
        try:
            if pcm:
                samples = np.frombuffer(pcm, dtype="<i2")
                self._append(self._resampler.resample_chunk(samples))
            events: list[Transcript] = []
            while len(self._pcm) >= _WINDOW:
                events.extend(self._window(is_last=False))
            return tuple(events)
        except Exception:
            self._fail()
            self._release()
            raise

    def finish(self) -> tuple[Transcript, ...]:
        self._finish()
        try:
            if self.sample_end == 0:
                result = (self._output("", final=True),)
                self._complete()
                return result
            self._append(self._resampler.resample_chunk(np.empty(0, dtype=np.int16), last=True))
            self._append(np.zeros(_WINDOW, dtype=np.int16))  # upstream right padding
            if len(self._pcm) % _WINDOW:
                self._append(np.zeros(-len(self._pcm) % _WINDOW, dtype=np.int16))
            events: list[Transcript] = []
            while len(self._pcm) >= _WINDOW:
                events.extend(self._window(is_last=len(self._pcm) == _WINDOW))
            events.append(self._output(" ".join(self._phrases), final=True))
            self._complete()
            return tuple(events)
        except Exception:
            self._fail()
            raise
        finally:
            self._release()

    def abort(self) -> None:
        super().abort()
        self._release()

    def _release(self) -> None:
        self._pcm = np.empty(0, dtype=np.int32)
        self._past = np.empty((0, 35), dtype=np.float32)
        self._state = np.empty((0, 0), dtype=np.float16)
        self._phrases.clear()

    def _append(self, pcm: npt.NDArray[np.int16]) -> None:
        if len(pcm):
            self._pcm = np.concatenate((self._pcm, pcm.astype(np.int32)))

    def _window(self, *, is_last: bool) -> list[Transcript]:
        chunk, self._pcm = self._pcm[:_WINDOW], self._pcm[_WINDOW:]
        logprobs, self._state = self._runtime._session.run(
            ["logprobs", "state_next"], {"signal": chunk[None, :, None], "state": self._state}
        )
        return self._split(logprobs[0], is_last=is_last)

    def _split(self, logprobs: npt.NDArray[np.float32], *, is_last: bool) -> list[Transcript]:
        history = np.concatenate((self._past, logprobs))
        is_speech = np.exp(history[:, -2:]).sum(axis=-1) <= 0.9
        padded = np.pad(is_speech, (20, 20 if is_last else 0))
        changes = np.diff(np.pad(~padded, (1, 1)).astype(np.int32))
        starts = np.flatnonzero(changes == 1) - 20
        ends = np.flatnonzero(changes == -1) - 20
        valid = ends - starts >= 20
        starts, ends = starts[valid], ends[valid]
        boundaries: list[tuple[int, int]] = []
        for i, (start, end) in enumerate(zip(ends.tolist(), starts.tolist()[1:] + [len(history)])):
            while end - start >= 2000:
                boundaries.append((start, start + 2000))
                start += 2000
            if i < len(ends) - 1:
                boundaries.append((start, end))
        events: list[Transcript] = []
        last = 0
        for start, end in boundaries:
            labels = history[max(0, start - 3) : end + 3].argmax(axis=-1).tolist()
            text = "".join(_LABELS[token] for token, _ in groupby(labels) if token < len(_LABELS)).strip()
            if text:
                self._phrases.append(text)
                phrase_start = max(0.0, round((start + self._offset) * 0.03 - 0.63, 2))
                phrase_end = max(phrase_start, round((end + self._offset) * 0.03 - 0.63, 2))
                events.append(
                    self._output(" ".join(self._phrases), final=False, phrase_start=phrase_start, phrase_end=phrase_end)
                )
            last = end
        if not np.any(is_speech[last:]):
            last = max(last, len(history) - 3)
        self._offset += last
        self._past = history[last:]
        return events
