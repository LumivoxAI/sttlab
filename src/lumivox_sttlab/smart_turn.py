"""Pinned smart-turn v3.1 ONNX scoring over explicit bounded audio.

This experimental component is deliberately not an endpoint detector: callers
own audio retention, VAD/silence gating, invocation policy and score thresholds.
"""

from pathlib import Path
from collections.abc import Sequence

import numpy as np

from .turn import AcousticScore, ScoreProvenance
from ._providers import DEFAULT_PROVIDERS, load_with_providers

_SAMPLE_RATE = 16_000
_WINDOW_SAMPLES = 8 * _SAMPLE_RATE
_INPUT_SHAPE = (1, 80, 800)
_OUTPUT_SHAPE = (1, 1)


def _position(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _compatible_shape(shape: list[int | str | None], tail: tuple[int, ...]) -> bool:
    return len(shape) == len(tail) + 1 and (shape[0] == 1 or not isinstance(shape[0], int)) and shape[1:] == list(tail)


class SmartTurn:
    """Reusable blocking smart-turn v3.1 ONNX score component.

    Install the ``smart-turn`` extra and one ONNX Runtime extra. The caller must
    serialize calls unless the selected runtime has separately been shown safe
    for concurrent inference. Input bytes are converted to an owned float array
    before inference, so the session retains no reference to caller PCM.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        source: str,
        model: str = "smart-turn-v3.1",
        configuration: str | None = None,
        providers: Sequence[str] = DEFAULT_PROVIDERS,
    ) -> None:
        artifact = Path(path)
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        # Validate source and provenance before allocating native resources.
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string")
        self.source = source
        self.provenance = ScoreProvenance(model=model, configuration=configuration)

        def load(provider: str) -> ort.InferenceSession:
            options = ort.SessionOptions()
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(str(artifact), sess_options=options, providers=[provider])
            if provider not in session.get_providers():
                raise RuntimeError(f"ONNX Runtime fell back from {provider}")
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if len(inputs) != 1 or inputs[0].name != "input_features":
                raise ValueError("incompatible smart-turn input name")
            if inputs[0].type != "tensor(float)" or not _compatible_shape(inputs[0].shape, _INPUT_SHAPE[1:]):
                raise ValueError("incompatible smart-turn input tensor")
            if len(outputs) != 1 or outputs[0].name != "logits":
                raise ValueError("incompatible smart-turn output name")
            if outputs[0].type != "tensor(float)" or not _compatible_shape(outputs[0].shape, _OUTPUT_SHAPE[1:]):
                raise ValueError("incompatible smart-turn output tensor")
            return session

        self._session, self.provider = load_with_providers(providers, ort.get_available_providers(), load)
        self._extractor = WhisperFeatureExtractor(chunk_length=8)  # type: ignore[no-untyped-call]

    def score(self, pcm: bytes, *, sample_start: int, sample_end: int, event_id: int) -> AcousticScore:
        """Score the latest eight seconds and attribute its consumed range.

        ``sample_start`` and ``sample_end`` describe all supplied PCM. Overlong
        input is truncated from the left; shorter input is left-zero-padded for
        the model, without extending the returned real-audio range.
        """
        if not isinstance(pcm, bytes) or len(pcm) % 2:
            raise ValueError("expected complete mono PCM S16LE samples as bytes")
        _position(sample_start, "sample_start")
        _position(sample_end, "sample_end")
        if sample_end < sample_start or sample_end - sample_start != len(pcm) // 2:
            raise ValueError("sample range length must match supplied PCM")
        if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id <= 0:
            raise ValueError("event_id must be a positive integer")

        consumed_start = max(sample_start, sample_end - _WINDOW_SAMPLES)
        audio = np.frombuffer(pcm, dtype="<i2")[-_WINDOW_SAMPLES:].astype(np.float32) / 32768.0
        if len(audio) < _WINDOW_SAMPLES:
            audio = np.pad(audio, (_WINDOW_SAMPLES - len(audio), 0))
        features = self._extractor(
            audio,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=_WINDOW_SAMPLES,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        if features.shape != _INPUT_SHAPE:
            raise RuntimeError(f"feature extractor returned incompatible shape: {features.shape}")
        output = np.asarray(self._session.run(["logits"], {"input_features": features})[0])
        if output.shape != _OUTPUT_SHAPE:
            raise RuntimeError(f"smart-turn returned incompatible shape: {output.shape}")
        return AcousticScore(
            source=self.source,
            event_id=event_id,
            sample_start=consumed_start,
            sample_end=sample_end,
            value=float(output[0, 0]),
            provenance=self.provenance,
        )
