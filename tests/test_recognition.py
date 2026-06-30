"""Contract and pinned-artifact checks for the first recognition backends."""

import wave
from types import SimpleNamespace
from typing import cast
from pathlib import Path

import pytest
import onnxruntime as ort

from lumivox_sttlab.tone import Tone, ToneRun
from lumivox_sttlab.gigaam import Gigaam
from lumivox_sttlab._providers import DEFAULT_PROVIDERS, load_with_providers
from lumivox_sttlab.recognition import (
    Recognizer,
    RunOutcome,
    Transcript,
    RecognitionRun,
    IntermediateOutput,
    RecognitionCapabilities,
    _RunBase,
    feed_pcm,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def test_recognizer_capabilities_reject_incompatible_requirements() -> None:
    tone: Recognizer = object.__new__(Tone)
    gigaam: Recognizer = object.__new__(Gigaam)
    assert tone.capabilities.intermediate is IntermediateOutput.COMPLETED_PHRASES
    assert not tone.capabilities.requires_complete_segment
    assert gigaam.capabilities.intermediate is IntermediateOutput.NONE
    assert gigaam.capabilities.requires_complete_segment
    for backend in (tone, gigaam):
        backend.capabilities.require()
        assert backend.capabilities.final_flush
    tone.capabilities.require(intermediate=IntermediateOutput.COMPLETED_PHRASES)
    with pytest.raises(ValueError, match="completed_phrases"):
        gigaam.capabilities.require(intermediate=IntermediateOutput.COMPLETED_PHRASES)
    with pytest.raises(ValueError, match="revisable_partials"):
        tone.capabilities.require(intermediate=IntermediateOutput.REVISABLE_PARTIALS)
    with pytest.raises(ValueError, match="flush"):
        RecognitionCapabilities(IntermediateOutput.NONE, True, False).require()


def test_revisable_partial_recognizer_contract_on_uneven_chunks_and_eof() -> None:
    class PartialRun(_RunBase):
        def feed(self, pcm: bytes, *, speech: bool = True) -> tuple[Transcript, ...]:
            self._accept(pcm, speech)
            return (self._output("draft" if self.sample_end < 3 else "corrected", final=False),)

        def finish(self) -> tuple[Transcript, ...]:
            self._finish()
            result = (self._output("corrected final", final=True),)
            self._complete()
            return result

    class PartialRecognizer:
        capabilities = RecognitionCapabilities(IntermediateOutput.REVISABLE_PARTIALS, False, True)

        def new_run(self, *, max_samples: int) -> PartialRun:
            return PartialRun(max_samples)

    backend: Recognizer = PartialRecognizer()
    backend.capabilities.require(intermediate=IntermediateOutput.REVISABLE_PARTIALS)
    with pytest.raises(ValueError, match="completed_phrases"):
        backend.capabilities.require(intermediate=IntermediateOutput.COMPLETED_PHRASES)
    a, b = (backend.new_run(max_samples=3) for _ in range(2))
    first = a.feed(b"\0\0")
    assert b.feed(b"\0\0\0\0\0\0")[0].sample_end == 3
    second = a.feed(b"\0\0\0\0")
    tail = a.finish()
    assert [(e.text, e.revision, e.sample_end, e.final) for e in first + second + tail] == [
        ("draft", 1, 1, False),
        ("corrected", 2, 3, False),
        ("corrected final", 3, 3, True),
    ]
    assert b.finish()[0].revision == 2
    assert a.outcome is b.outcome is RunOutcome.FINISHED
    aborted = backend.new_run(max_samples=1)
    aborted.abort()
    assert aborted.outcome is RunOutcome.ABORTED
    with pytest.raises(RuntimeError):
        aborted.finish()


def test_provider_preference_and_initialization_fallback() -> None:
    attempts: list[str] = []

    def load(name: str) -> str:
        attempts.append(name)
        if name == "CUDAExecutionProvider":
            raise RuntimeError("CUDA libraries missing")
        return name

    cpu = "CPUExecutionProvider"
    cuda = "CUDAExecutionProvider"
    assert load_with_providers(DEFAULT_PROVIDERS, [cpu], load) == (cpu, cpu)
    assert attempts == [cpu]
    attempts.clear()
    assert load_with_providers([cuda, cpu], [cpu, cuda], load) == (cpu, cpu)
    assert attempts == [cuda, cpu]
    attempts.clear()
    assert load_with_providers([cpu, cuda], [cpu, cuda], load) == (cpu, cpu)
    assert attempts == [cpu]
    with pytest.raises(RuntimeError, match="could not initialize") as error:
        load_with_providers([cuda], [cuda, cpu], load)
    assert isinstance(error.value.__cause__, RuntimeError)
    with pytest.raises(ValueError, match="no requested ONNX provider"):
        load_with_providers([cuda], [cpu], load)
    with pytest.raises(ValueError, match="non-empty"):
        load_with_providers([], [cpu], load)


def test_tone_retries_when_onnx_runtime_silently_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "tone.onnx"
    model.touch()
    attempts: list[str] = []

    class Session:
        def __init__(self, path: str, *, providers: list[str]) -> None:
            assert path == str(model)
            attempts.extend(providers)
            self.provider = providers[0]

        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

        def get_inputs(self) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(name="signal", type="tensor(int32)", shape=[1, 2400, 1]),
                SimpleNamespace(name="state", type="tensor(float16)", shape=[1, 219729]),
            ]

        def get_outputs(self) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(name="logprobs", type="tensor(float)", shape=[1, 10, 35]),
                SimpleNamespace(name="state_next", type="tensor(float16)", shape=[1, 219729]),
            ]

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", Session)
    runtime = Tone(model)
    assert attempts == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert runtime.provider == "CPUExecutionProvider"


def test_gigaam_retries_when_acoustic_session_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import onnx_asr

    attempts: list[str] = []

    def load_model(*args: object, providers: list[str], **kwargs: object) -> SimpleNamespace:
        attempts.extend(providers)
        session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
        return SimpleNamespace(asr=SimpleNamespace(_model=session))

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(onnx_asr, "load_model", load_model)
    runtime = Gigaam("gigaam-v3-ctc", tmp_path)
    assert attempts == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert runtime.provider == "CPUExecutionProvider"


class _FakeAsr:
    def recognize(self, audio: object, *, sample_rate: int) -> str:
        assert sample_rate == 16_000
        return "sample"


def test_batch_run_boundaries_and_empty_input() -> None:
    runtime = object.__new__(Gigaam)
    runtime._asr = _FakeAsr()  # type: ignore[assignment]
    backend: Recognizer = runtime
    first: RecognitionRun = backend.new_run(max_samples=2)
    assert backend.new_run(max_samples=1).outcome is None
    assert first.feed(b"\x00\x01") == ()
    assert first.feed(b"\x00\x01") == ()
    result = first.finish()
    assert first.outcome is RunOutcome.FINISHED
    assert [(item.text, item.revision, item.sample_end, item.final) for item in result] == [("sample", 1, 2, True)]
    with pytest.raises(RuntimeError):
        first.feed(b"")

    second = backend.new_run(max_samples=2)
    assert second.finish()[0].text == ""
    interrupted = backend.new_run(max_samples=2)
    interrupted.feed(b"\x00\x00")
    interrupted.abort()
    assert interrupted.outcome is RunOutcome.ABORTED
    with pytest.raises(RuntimeError):
        interrupted.finish()
    overflow = backend.new_run(max_samples=1)
    with pytest.raises(OverflowError):
        overflow.feed(b"\0\0\0\0")
    assert overflow.outcome is RunOutcome.OVERFLOW
    with pytest.raises(RuntimeError):
        overflow.finish()
    with pytest.raises(ValueError):
        backend.new_run(max_samples=0)
    with pytest.raises(ValueError):
        backend.new_run(max_samples=True)
    with pytest.raises(ValueError):
        backend.new_run(max_samples=2).feed(b"\0")
    labeled = backend.new_run(max_samples=2)
    with pytest.raises(ValueError):
        labeled.feed(b"\0\0", speech=1)  # type: ignore[arg-type]
    assert labeled.feed(b"\0\0", speech=False) == ()
    assert labeled.feed(b"", speech=False) == ()  # a paused producer supplies no audio time
    assert labeled.feed(b"\0\0", speech=True) == ()
    assert labeled.finish()[0].sample_end == 2


def test_batch_failure_releases_audio_and_keeps_original_exception() -> None:
    class BrokenAsr:
        def recognize(self, audio: object, *, sample_rate: int) -> str:
            raise LookupError("inference failed")

    runtime = object.__new__(Gigaam)
    runtime._asr = BrokenAsr()  # type: ignore[assignment]
    failed = runtime.new_run(max_samples=3)
    failed.feed(b"\0\0")
    with pytest.raises(LookupError, match="inference failed"):
        failed.finish()
    assert failed.outcome is RunOutcome.FAILED
    assert failed._chunks == []
    with pytest.raises(RuntimeError):
        failed.feed(b"")
    with pytest.raises(RuntimeError):
        failed.abort()
    assert failed.outcome is RunOutcome.FAILED

    surviving = runtime.new_run(max_samples=1)
    assert surviving.finish()[0].final
    assert surviving.outcome is RunOutcome.FINISHED


def test_tone_feed_and_flush_failures_release_state(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = object.__new__(Tone)
    run = ToneRun(runtime, 10)

    def fail(*args: object, **kwargs: object) -> None:
        raise LookupError("native inference failed")

    monkeypatch.setattr(run, "_window", fail)
    with pytest.raises(LookupError, match="native inference failed"):
        run.feed(b"\0\0")
    assert run.outcome is RunOutcome.FAILED
    assert len(run._pcm) == 0 and len(run._past) == 0 and len(run._state) == 0
    with pytest.raises(RuntimeError):
        run.finish()

    flush = ToneRun(runtime, 10)

    def consume_window(*, is_last: bool) -> list[Transcript]:
        flush._pcm = flush._pcm[2400:]
        return []

    monkeypatch.setattr(flush, "_window", consume_window)
    flush.feed(b"\0\0")
    monkeypatch.setattr(flush, "_window", fail)
    with pytest.raises(LookupError, match="native inference failed"):
        flush.finish()
    assert flush.outcome is RunOutcome.FAILED
    assert len(flush._pcm) == 0 and len(flush._state) == 0


def test_feed_pcm_chunks_without_vad_and_leaves_finalization_to_caller() -> None:
    class Run:
        def __init__(self) -> None:
            self.chunks: list[tuple[bytes, bool]] = []
            self.outcome: RunOutcome | None = None

        def feed(self, pcm: bytes, *, speech: bool = True) -> tuple[Transcript, ...]:
            self.chunks.append((pcm, speech))
            return (
                Transcript(str(len(self.chunks)), len(self.chunks), sum(len(c) // 2 for c, _ in self.chunks), False),
            )

        def finish(self) -> tuple[Transcript, ...]:
            return (Transcript("final", len(self.chunks) + 1, sum(len(c) // 2 for c, _ in self.chunks), True),)

        def abort(self) -> None:
            pass

    run = Run()
    assert feed_pcm(run, b"") == ()
    pcm = b"\0\0\1\0\2\0\3\0\4\0"
    updates = feed_pcm(run, pcm, chunk_samples=2)
    assert run.chunks == [(pcm[:4], True), (pcm[4:8], True), (pcm[8:], True)]
    assert [update.sample_end for update in updates] == [2, 4, 5]
    assert run.finish()[0].final
    for invalid in (b"\0", bytearray(b"\0\0")):
        with pytest.raises(ValueError):
            feed_pcm(run, invalid)  # type: ignore[arg-type]
    for size in (0, -1, True):
        with pytest.raises(ValueError):
            feed_pcm(run, pcm, chunk_samples=size)
    assert len(run.chunks) == 3


def _pcm(path: Path) -> bytes:
    with wave.open(str(path)) as source:
        assert source.getnchannels() == 1 and source.getframerate() == 16000
        return cast(bytes, source.readframes(source.getnframes()))


@pytest.mark.model
def test_gigaam_batch_russian_and_english() -> None:
    if not (DATA / "gigaam/v3/v3_ctc.int8.onnx").exists():
        pytest.skip("pinned model missing under data/gigaam")
    russian = _pcm(DATA / "gigaam/example.wav")
    ctc = Gigaam("gigaam-v3-ctc", DATA / "gigaam/v3", quantization="int8")
    assert ctc.provider == "CPUExecutionProvider"
    run = ctc.new_run(max_samples=len(russian) // 2)
    for offset in range(0, len(russian), 1994):
        assert not run.feed(russian[offset : offset + 1994])
    assert run.finish()[0].text.startswith("ничьих не требуя похвал")
    if (DATA / "gigaam/v3/v3_e2e_rnnt_encoder.int8.onnx").exists():
        rnnt = Gigaam("gigaam-v3-e2e-rnnt", DATA / "gigaam/v3", quantization="int8")
        second_run = rnnt.new_run(max_samples=len(russian) // 2)
        second_run.feed(russian)
        assert second_run.finish()[0].text == (
            "Ничьих не требуя похвал, Счастлив уж я надеждой сладкой, "
            "Что дева с трепетом любви Посмотрит, может быть, украдкой "
            "На песни грешные мои. У лукоморья дуб зелёный."
        )
    if (DATA / "gigaam/jfk.wav").exists() and (DATA / "gigaam/multilingual/multilingual_ctc.int8.onnx").exists():
        english = _pcm(DATA / "gigaam/jfk.wav")
        multi = Gigaam("gigaam-multilingual-ctc", DATA / "gigaam/multilingual", quantization="int8")
        second = multi.new_run(max_samples=len(english) // 2)
        second.feed(english)
        assert "ask not what your country can do for you" in second.finish()[0].text


@pytest.mark.model
def test_tone_interleaved_runs_and_eof() -> None:
    if not (DATA / "tone/model.onnx").exists() or not (DATA / "tone/audio_short_16k.wav").exists():
        pytest.skip("pinned T-one model/sample missing under data/tone")
    audio = _pcm(DATA / "tone/audio_short_16k.wav")
    runtime = Tone(DATA / "tone/model.onnx")
    assert runtime.provider == "CPUExecutionProvider"
    a, b = (runtime.new_run(max_samples=len(audio) // 2) for _ in range(2))
    events_a: list[Transcript] = []
    events_b: list[Transcript] = []
    for offset in range(0, len(audio), 1994):
        chunk = audio[offset : offset + 1994]
        events_a.extend(a.feed(chunk))
        events_b.extend(b.feed(chunk))
    events_a.extend(a.finish())
    events_b.extend(b.finish())
    assert events_a == events_b
    assert events_a[-1].text == "ну сейчас к тебе приедет бригада давай давай я жду"
    assert events_a[-1].final
    phrases = [(e.phrase_start, e.phrase_end) for e in events_a if e.phrase_start is not None]
    assert len(phrases) == 2
    for observed, reference in zip(phrases, [(0.03, 2.91), (5.76, 6.21)], strict=True):
        assert observed[0] == pytest.approx(reference[0], abs=0.031)
        assert observed[1] == pytest.approx(reference[1], abs=0.031)
    with pytest.raises(RuntimeError):
        a.feed(b"")
    interrupted = runtime.new_run(max_samples=len(audio) // 2)
    interrupted.feed(audio[:1994])
    interrupted.abort()
    with pytest.raises(RuntimeError):
        interrupted.finish()
    assert runtime.new_run(max_samples=1).finish()[0].text == ""
