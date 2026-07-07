"""Deterministic traces for the first explicit synchronous recipes."""

from collections.abc import Callable

import pytest

from lumivox_sttlab.turn import (
    ScoreRule,
    TextScore,
    SilenceRule,
    LifecycleSignal,
    ScoreComparison,
    ScoreProvenance,
    LifecycleObservation,
)
from lumivox_sttlab.composition import (
    ActionEvent,
    RuntimeFailure,
    ActionVisibility,
    ObservationEvent,
    RecognitionEvent,
    SilenceEndpointPipeline,
    TextEndpointBatchPipeline,
)
from lumivox_sttlab.recognition import (
    RunOutcome,
    Transcript,
    IntermediateOutput,
    RecognitionCapabilities,
)


class FakeRun:
    def __init__(
        self,
        max_samples: int,
        *,
        intermediate: bool,
        label: str,
        finish_error: Exception | None = None,
    ) -> None:
        self.max_samples = max_samples
        self.intermediate = intermediate
        self.label = label
        self.finish_error = finish_error
        self.sample_end = 0
        self.revision = 0
        self.pcm = bytearray()
        self.outcome: RunOutcome | None = None

    def _transcript(self, text: str, *, final: bool) -> Transcript:
        self.revision += 1
        return Transcript(text, self.revision, self.sample_end, final)

    def feed(self, pcm: bytes, *, speech: bool = True) -> tuple[Transcript, ...]:
        if self.outcome is not None:
            raise RuntimeError("closed")
        if self.sample_end + len(pcm) // 2 > self.max_samples:
            self.outcome = RunOutcome.OVERFLOW
            raise OverflowError
        self.sample_end += len(pcm) // 2
        self.pcm.extend(pcm)
        if self.intermediate and pcm:
            return (self._transcript(f"{self.label}-{self.revision + 1}", final=False),)
        return ()

    def finish(self) -> tuple[Transcript, ...]:
        if self.outcome is not None:
            raise RuntimeError("closed")
        if self.finish_error is not None:
            raise self.finish_error
        self.outcome = RunOutcome.FINISHED
        return (self._transcript(f"{self.label}-final", final=True),)

    def abort(self) -> None:
        if self.outcome is not None:
            raise RuntimeError("closed")
        self.outcome = RunOutcome.ABORTED
        self.pcm.clear()


class FakeRecognizer:
    def __init__(
        self,
        capabilities: RecognitionCapabilities,
        *,
        label: str,
        finish_error: Exception | None = None,
    ) -> None:
        self.capabilities = capabilities
        self.label = label
        self.finish_error = finish_error
        self.runs: list[FakeRun] = []

    def new_run(self, *, max_samples: int) -> FakeRun:
        run = FakeRun(
            max_samples,
            intermediate=self.capabilities.intermediate is not IntermediateOutput.NONE,
            label=self.label,
            finish_error=self.finish_error,
        )
        self.runs.append(run)
        return run


class FakeTextScorer:
    source = "text-turn"

    def __init__(self, value: float = 0.9, mutate: Callable[[TextScore], TextScore] | None = None) -> None:
        self.value = value
        self.mutate = mutate

    def score(
        self,
        transcript: Transcript,
        *,
        transcript_source: str,
        event_id: int,
        sample_end: int,
    ) -> TextScore:
        score = TextScore(
            source=self.source,
            event_id=event_id,
            sample_end=sample_end,
            value=self.value,
            provenance=ScoreProvenance(model="fake-text-turn"),
            transcript_source=transcript_source,
            transcript_revision=transcript.revision,
            transcript_sample_end=transcript.sample_end,
        )
        return score if self.mutate is None else self.mutate(score)


STREAMING = RecognitionCapabilities(IntermediateOutput.COMPLETED_PHRASES, False, True)
BATCH = RecognitionCapabilities(IntermediateOutput.NONE, True, True)


def _text_pipeline(
    *,
    score: float = 0.9,
    batch_error: Exception | None = None,
    scorer: FakeTextScorer | None = None,
) -> tuple[TextEndpointBatchPipeline, FakeRecognizer, FakeRecognizer]:
    streaming = FakeRecognizer(STREAMING, label="stream")
    batch = FakeRecognizer(BATCH, label="batch", finish_error=batch_error)
    producer = scorer or FakeTextScorer(score)
    pipeline = TextEndpointBatchPipeline(
        streaming,
        batch,
        producer,
        streaming_source="streaming",
        batch_source="batch",
        input_source="input",
        rule=ScoreRule(
            source="text-turn",
            score_type=TextScore,
            threshold=0.7,
            comparison=ScoreComparison.AT_LEAST,
            action="request-endpoint",
        ),
    )
    return pipeline, streaming, batch


def test_silence_recipe_orders_events_without_fabricating_eof() -> None:
    recognizer = FakeRecognizer(STREAMING, label="stream")
    pipeline = SilenceEndpointPipeline(
        recognizer,
        recognizer_source="streaming",
        silence_source="vad",
        rule=SilenceRule(source="vad", threshold_samples=4, action="request-endpoint"),
    )
    run = pipeline.new_run(max_samples=8)

    first = run.feed(b"\0\0" * 2, speech=False)
    second = run.feed(b"\0\0" * 2, speech=False)
    assert [type(event) for event in first] == [RecognitionEvent, ObservationEvent]
    assert [type(event) for event in second] == [RecognitionEvent, ObservationEvent, ActionEvent]
    action = second[-1]
    assert isinstance(action, ActionEvent)
    assert action.visibility is ActionVisibility.EXTERNAL
    assert action.selection.action == "request-endpoint"
    assert action.selection.sample_position == 4
    assert run.outcome is None

    # The caller may still supply audio before establishing end-of-input.
    assert run.feed(b"\0\0", speech=True)
    finished = run.finish()
    assert isinstance(finished[0], ObservationEvent)
    lifecycle = finished[0].observation
    assert isinstance(lifecycle, LifecycleObservation)
    assert lifecycle.signal is LifecycleSignal.END_OF_INPUT
    final = finished[-1]
    assert isinstance(final, RecognitionEvent)
    assert final.transcript.final and final.selected_final
    assert run.outcome is RunOutcome.FINISHED


def test_text_rule_activates_batch_only_after_explicit_finish_and_supersedes_streaming() -> None:
    pipeline, _, batch = _text_pipeline()
    run = pipeline.new_run(max_samples=8)
    pcm_a = b"\1\0" * 2
    pcm_b = b"\2\0"

    activated = run.feed(pcm_a)
    assert [type(event) for event in activated] == [RecognitionEvent, ObservationEvent, ActionEvent, ActionEvent]
    internal, external = activated[-2:]
    assert isinstance(internal, ActionEvent) and isinstance(external, ActionEvent)
    assert internal.visibility is ActionVisibility.INTERNAL
    assert internal.selection.action == TextEndpointBatchPipeline.INTERNAL_BATCH_ACTION
    assert external.visibility is ActionVisibility.EXTERNAL
    assert external.selection.action == "request-endpoint"
    assert not batch.runs
    assert run.outcome is None

    # A request is not EOF; continued input is retained and processed.
    continued = run.feed(pcm_b)
    assert [type(event) for event in continued] == [RecognitionEvent, ObservationEvent]
    assert run.retained_samples == 3
    finished = run.finish()
    assert isinstance(finished[0], ObservationEvent)
    recognition = [event for event in finished if isinstance(event, RecognitionEvent)]
    assert [event.source for event in recognition] == ["streaming", "batch"]
    assert not recognition[0].selected_final
    assert recognition[1].selected_final
    assert recognition[1].supersedes == recognition[0].reference
    assert bytes(batch.runs[0].pcm) == pcm_a + pcm_b
    assert run.retained_samples == 0
    assert run.outcome is RunOutcome.FINISHED


def test_unmatched_text_rule_skips_batch_and_selects_streaming_final() -> None:
    pipeline, _, batch = _text_pipeline(score=0.2)
    run = pipeline.new_run(max_samples=2)
    assert [type(event) for event in run.feed(b"\0\0")] == [RecognitionEvent, ObservationEvent]
    finished = run.finish()
    final = finished[-1]
    assert isinstance(final, RecognitionEvent)
    assert final.source == "streaming" and final.selected_final
    assert not batch.runs


def test_composed_bounds_abort_and_interleaved_runs_are_isolated() -> None:
    pipeline, streaming, _ = _text_pipeline()
    first = pipeline.new_run(max_samples=2)
    second = pipeline.new_run(max_samples=2)
    first.feed(b"\1\0")
    second.feed(b"\2\0" * 2)
    aborted = first.abort()
    assert first.outcome is RunOutcome.ABORTED and first.retained_samples == 0
    assert isinstance(aborted[0], ObservationEvent)
    assert second.retained_samples == 2
    with pytest.raises(OverflowError):
        second.feed(b"\3\0")
    assert second.outcome is RunOutcome.OVERFLOW and second.retained_samples == 0
    assert [run.outcome for run in streaming.runs] == [RunOutcome.ABORTED, RunOutcome.ABORTED]


def test_runtime_failure_is_propagated_and_marked_with_failing_stage() -> None:
    failure = RuntimeFailure("shared session failed")
    pipeline, streaming, _ = _text_pipeline(batch_error=failure)
    run = pipeline.new_run(max_samples=2)
    run.feed(b"\0\0")
    with pytest.raises(RuntimeFailure) as caught:
        run.finish()
    assert caught.value is failure
    assert run.outcome is RunOutcome.FAILED and run.retained_samples == 0
    assert run.failure is not None
    assert run.failure.stage == "batch" and run.failure.shared_runtime
    assert run.failure.exception is failure
    assert streaming.runs[0].outcome is RunOutcome.FINISHED


def test_build_time_and_runtime_provenance_validation() -> None:
    streaming = FakeRecognizer(STREAMING, label="stream")
    batch = FakeRecognizer(BATCH, label="batch")
    scorer = FakeTextScorer()
    rule = ScoreRule(
        source="text-turn",
        score_type=TextScore,
        threshold=0.5,
        comparison=ScoreComparison.AT_LEAST,
        action="endpoint",
    )
    with pytest.raises(ValueError, match="distinct"):
        TextEndpointBatchPipeline(
            streaming,
            batch,
            scorer,
            streaming_source="same",
            batch_source="batch",
            input_source="same",
            rule=rule,
        )
    with pytest.raises(ValueError, match="streaming recognizer"):
        SilenceEndpointPipeline(
            batch,
            recognizer_source="batch",
            silence_source="vad",
            rule=SilenceRule(source="vad", threshold_samples=1, action="endpoint"),
        )

    def wrong_revision(score: TextScore) -> TextScore:
        return TextScore(
            source=score.source,
            event_id=score.event_id,
            sample_end=score.sample_end,
            value=score.value,
            provenance=score.provenance,
            transcript_source=score.transcript_source,
            transcript_revision=score.transcript_revision + 1,
            transcript_sample_end=score.transcript_sample_end,
        )

    invalid, invalid_streaming, _ = _text_pipeline(scorer=FakeTextScorer(mutate=wrong_revision))
    run = invalid.new_run(max_samples=1)
    with pytest.raises(ValueError, match="provenance"):
        run.feed(b"\0\0")
    assert run.outcome is RunOutcome.FAILED
    assert run.failure is not None and run.failure.stage == "text_score"
    assert not run.failure.shared_runtime
    assert invalid_streaming.runs[0].outcome is RunOutcome.ABORTED
