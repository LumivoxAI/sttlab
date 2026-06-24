"""Bounded batch GigaAM through onnx-asr; no upstream GigaAM import."""

from pathlib import Path

import numpy as np

from .recognition import Transcript, _RunBase

_MODELS = frozenset(
    {
        "gigaam-v3-ctc",
        "gigaam-v3-rnnt",
        "gigaam-v3-e2e-ctc",
        "gigaam-v3-e2e-rnnt",
        "gigaam-multilingual-ctc",
        "gigaam-multilingual-large-ctc",
    }
)


class Gigaam:
    """Loaded batch recognizer. No network access or automatic model downloads.

    Install the ``gigaam`` extra, provide an onnx-asr compatible model directory,
    and select the CPU or CUDA execution provider explicitly. An ONNX session may
    be shared by independent runs; entry points block the calling thread.
    """

    def __init__(
        self,
        model: str,
        path: str | Path,
        *,
        quantization: str | None = None,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        if model not in _MODELS:
            raise ValueError(f"unsupported GigaAM model: {model}")
        directory = Path(path)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        import onnx_asr
        import onnxruntime as ort

        if provider not in ort.get_available_providers():
            raise ValueError(f"unavailable ONNX provider: {provider}")
        self.model = model
        self._asr = onnx_asr.load_model(model, directory, quantization=quantization, providers=[provider])

    def new_run(self, *, max_samples: int) -> "GigaamRun":
        """Allocate a run; max_samples bounds retained PCM until final inference."""
        return GigaamRun(self, max_samples)


class GigaamRun(_RunBase):
    def __init__(self, runtime: Gigaam, max_samples: int) -> None:
        super().__init__(max_samples)
        self._runtime = runtime
        self._chunks: list[bytes] = []

    def feed(self, pcm: bytes, *, speech: bool = True) -> tuple[Transcript, ...]:
        try:
            self._accept(pcm, speech)
        except OverflowError:
            self._chunks.clear()
            raise
        try:
            if pcm:
                self._chunks.append(pcm)
        except Exception:
            self._fail()
            self._chunks.clear()
            raise
        return ()

    def finish(self) -> tuple[Transcript, ...]:
        self._finish()
        try:
            if not self._chunks:
                text = ""
            else:
                audio = np.frombuffer(b"".join(self._chunks), dtype="<i2").astype(np.float32) / 32768
                text = self._runtime._asr.recognize(audio, sample_rate=16_000)
            result = (self._output(text, final=True),)
            self._complete()
            return result
        except Exception:
            self._fail()
            raise
        finally:
            self._chunks.clear()

    def abort(self) -> None:
        super().abort()
        self._chunks.clear()
