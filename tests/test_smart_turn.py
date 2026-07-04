"""Contract and pinned-artifact checks for smart-turn scoring."""

import sys
import wave
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from pathlib import Path

import numpy as np
import pytest
import onnxruntime as ort

from lumivox_sttlab.smart_turn import SmartTurn

DATA = Path(__file__).resolve().parents[1] / "data"


class _Extractor:
    windows: list[np.ndarray[Any, Any]] = []

    def __init__(self, *, chunk_length: int) -> None:
        assert chunk_length == 8

    def __call__(self, audio: np.ndarray[Any, Any], **kwargs: object) -> SimpleNamespace:
        assert kwargs == {
            "sampling_rate": 16_000,
            "return_tensors": "np",
            "padding": "max_length",
            "max_length": 128_000,
            "truncation": True,
            "do_normalize": True,
        }
        self.windows.append(audio.copy())
        return SimpleNamespace(input_features=np.zeros((1, 80, 800), dtype=np.float32))


class _Session:
    attempts: list[str] = []

    def __init__(self, path: str, *, sess_options: object, providers: list[str]) -> None:
        assert Path(path).is_file()
        assert sess_options is not None
        self.provider = providers[0]
        self.attempts.append(self.provider)
        if self.provider == "CUDAExecutionProvider":
            raise RuntimeError("CUDA unavailable")

    def get_providers(self) -> list[str]:
        return [self.provider]

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input_features", type="tensor(float)", shape=["batch", 80, 800])]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="logits", type="tensor(float)", shape=["batch", 1])]

    def run(self, outputs: list[str], inputs: dict[str, object]) -> list[np.ndarray[Any, Any]]:
        assert outputs == ["logits"] and list(inputs) == ["input_features"]
        return [np.array([[0.75]], dtype=np.float32)]


@pytest.fixture
def fake_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SmartTurn:
    model = tmp_path / "smart-turn.onnx"
    model.touch()
    transformers = ModuleType("transformers")
    transformers.WhisperFeatureExtractor = _Extractor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", _Session)
    _Extractor.windows.clear()
    _Session.attempts.clear()
    return SmartTurn(model, source="acoustic", configuration="v3.1-cpu-f766f81")


def test_provider_fallback_preprocessing_and_attribution(fake_runtime: SmartTurn) -> None:
    assert fake_runtime.provider == "CPUExecutionProvider"
    assert _Session.attempts == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    pcm = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
    score = fake_runtime.score(pcm, sample_start=100, sample_end=103, event_id=4)
    assert score.source == "acoustic" and score.event_id == 4
    assert (score.sample_start, score.sample_end) == (100, 103)
    assert score.value == pytest.approx(0.75)
    assert score.provenance.model == "smart-turn-v3.1"
    assert score.provenance.configuration == "v3.1-cpu-f766f81"
    assert len(_Extractor.windows[0]) == 128_000
    assert np.count_nonzero(_Extractor.windows[0][:-3]) == 0
    assert _Extractor.windows[0][-3:].tolist() == [-1.0, 0.0, pytest.approx(32767 / 32768)]


def test_exact_overlong_empty_and_independent_calls(fake_runtime: SmartTurn) -> None:
    exact = np.arange(128_000, dtype=np.int16).tobytes()
    first = fake_runtime.score(exact, sample_start=10, sample_end=128_010, event_id=1)
    assert (first.sample_start, first.sample_end) == (10, 128_010)
    overlong = b"\x01\x00" * 7 + exact
    second = fake_runtime.score(overlong, sample_start=20, sample_end=128_027, event_id=2)
    assert (second.sample_start, second.sample_end) == (27, 128_027)
    assert np.array_equal(_Extractor.windows[-1], np.frombuffer(exact, dtype="<i2").astype(np.float32) / 32768)
    empty = fake_runtime.score(b"", sample_start=500, sample_end=500, event_id=3)
    assert (empty.sample_start, empty.sample_end) == (500, 500)
    assert np.count_nonzero(_Extractor.windows[-1]) == 0
    assert first.event_id == 1 and second.event_id == 2


@pytest.mark.parametrize(
    ("pcm", "start", "end", "event_id"),
    [
        (b"\0", 0, 0, 1),
        (bytearray(b"\0\0"), 0, 1, 1),
        (b"\0\0", 0, 2, 1),
        (b"\0\0", 2, 1, 1),
        (b"\0\0", -1, 0, 1),
        (b"\0\0", 0, 1, 0),
    ],
)
def test_invalid_pcm_ranges_and_event_ids(
    fake_runtime: SmartTurn, pcm: object, start: int, end: int, event_id: int
) -> None:
    with pytest.raises(ValueError):
        fake_runtime.score(pcm, sample_start=start, sample_end=end, event_id=event_id)  # type: ignore[arg-type]


def test_missing_artifact_is_rejected_before_optional_import(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SmartTurn(tmp_path / "missing.onnx", source="acoustic")


def test_incompatible_model_tensor_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "smart-turn.onnx"
    model.touch()
    transformers = ModuleType("transformers")
    transformers.WhisperFeatureExtractor = _Extractor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    class BadSession(_Session):
        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="input_features", type="tensor(float)", shape=[1, 79, 800])]

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", BadSession)
    with pytest.raises(RuntimeError, match="could not initialize") as error:
        SmartTurn(model, source="acoustic", providers=["CPUExecutionProvider"])
    assert isinstance(error.value.__cause__, ValueError)


def _pcm(path: Path) -> bytes:
    with wave.open(str(path)) as source:
        return cast(bytes, source.readframes(source.getnframes()))


@pytest.mark.model
def test_pinned_smart_turn_parity() -> None:
    model = DATA / "smart-turn/smart-turn-v3.1-cpu.onnx"
    audio_path = DATA / "gigaam/example.wav"
    if not model.exists() or not audio_path.exists():
        pytest.skip("pinned smart-turn model or GigaAM sample missing under data/")
    audio = _pcm(audio_path)
    scorer = SmartTurn(model, source="smart-turn", providers=["CPUExecutionProvider"])
    score = scorer.score(audio, sample_start=0, sample_end=len(audio) // 2, event_id=1)
    assert score.value == pytest.approx(0.6561, abs=0.0001)
