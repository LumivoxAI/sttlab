"""Experimental synchronous composition recipes.

This module intentionally contains explicit typed orchestration rather than a
general graph API. Calls block, return events in deterministic order, and never
invoke application callbacks.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol
from dataclasses import dataclass
from collections.abc import Sequence

from .turn import (
    RuleState,
    ScoreRule,
    TextScore,
    SilenceRule,
    SilenceTracker,
    ActionSelection,
    LifecycleSignal,
    TurnObservation,
    LifecycleObservation,
)
from .recognition import Recognizer, RunOutcome, Transcript, RecognitionRun, IntermediateOutput


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _name(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


class RuntimeFailure(Exception):
    """Marker for a failure affecting a reusable shared model runtime."""


@dataclass(frozen=True)
class StageFailure:
    """Failure metadata retained while the original exception propagates."""

    stage: str
    shared_runtime: bool
    exception: Exception


@dataclass(frozen=True)
class TranscriptRef:
    """Exact identity of a transcript in one attributed recognizer run."""

    source: str
    revision: int
    sample_end: int


@dataclass(frozen=True)
class RecognitionEvent:
    """An attributed transcript and its recipe-level selection semantics."""

    source: str
    transcript: Transcript
    selected_final: bool = False
    supersedes: TranscriptRef | None = None

    @property
    def sample_start(self) -> int:
        return 0

    @property
    def sample_end(self) -> int:
        return self.transcript.sample_end

    @property
    def reference(self) -> TranscriptRef:
        return TranscriptRef(self.source, self.transcript.revision, self.transcript.sample_end)


@dataclass(frozen=True)
class ObservationEvent:
    """A source-attributed turn observation emitted by a connected stage."""

    observation: TurnObservation


class ActionVisibility(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"


@dataclass(frozen=True)
class ActionEvent:
    """A rule-selected action; external actions are for caller dispatch."""

    selection: ActionSelection
    visibility: ActionVisibility
    transcript: TranscriptRef | None = None


PipelineEvent = RecognitionEvent | ObservationEvent | ActionEvent


class TextScoreProducer(Protocol):
    """Blocking producer of a score for one exact transcript revision."""

    @property
    def source(self) -> str: ...

    def score(
        self,
        transcript: Transcript,
        *,
        transcript_source: str,
        event_id: int,
        sample_end: int,
    ) -> TextScore: ...


def _transcript_events(
    source: str,
    transcripts: Sequence[Transcript],
    *,
    selected_final: bool = False,
    supersedes: TranscriptRef | None = None,
) -> tuple[RecognitionEvent, ...]:
    last_final = max((index for index, item in enumerate(transcripts) if item.final), default=None)
    return tuple(
        RecognitionEvent(
            source=source,
            transcript=item,
            selected_final=selected_final and index == last_final,
            supersedes=supersedes if selected_final and index == last_final else None,
        )
        for index, item in enumerate(transcripts)
    )


def _validate_transcripts(transcripts: Sequence[Transcript], *, sample_end: int, require_final: bool = False) -> None:
    if any(not isinstance(item, Transcript) or item.sample_end > sample_end for item in transcripts):
        raise ValueError("recognizer returned invalid transcript provenance")
    if require_final and not any(item.final and item.sample_end == sample_end for item in transcripts):
        raise ValueError("recognizer did not return a final transcript at end-of-input")


def _safe_abort(run: RecognitionRun) -> None:
    if run.outcome is None:
        try:
            run.abort()
        except Exception:
            pass


def _safe_abort_silence(tracker: SilenceTracker) -> None:
    if tracker.outcome is None:
        try:
            tracker.abort()
        except Exception:
            pass


class _ComposedRun:
    def __init__(self, max_samples: int) -> None:
        _positive_int(max_samples, "max_samples")
        self.max_samples = max_samples
        self.sample_end = 0
        self._outcome: RunOutcome | None = None
        self._failure: StageFailure | None = None

    @property
    def outcome(self) -> RunOutcome | None:
        return self._outcome

    @property
    def failure(self) -> StageFailure | None:
        return self._failure

    def _validate_feed(self, pcm: bytes, speech: bool) -> int:
        self._require_open()
        if not isinstance(pcm, bytes) or len(pcm) % 2:
            raise ValueError("expected complete mono PCM S16LE samples as bytes")
        if not isinstance(speech, bool):
            raise ValueError("speech must be a bool")
        next_sample = self.sample_end + len(pcm) // 2
        if next_sample > self.max_samples:
            raise OverflowError("run audio bound exceeded")
        return next_sample

    def _require_open(self) -> None:
        if self._outcome is not None:
            raise RuntimeError("composed run already finished or aborted")

    def _failed(self, stage: str, exception: Exception) -> None:
        self._outcome = RunOutcome.FAILED
        self._failure = StageFailure(stage, isinstance(exception, RuntimeFailure), exception)


class SilenceEndpointPipeline:
    """Streaming recognition with endpoint actions from labelled silence."""

    def __init__(
        self, recognizer: Recognizer, *, recognizer_source: str, silence_source: str, rule: SilenceRule
    ) -> None:
        _name(recognizer_source, "recognizer_source")
        _name(silence_source, "silence_source")
        if recognizer_source == silence_source:
            raise ValueError("recognizer and silence sources must be distinct")
        if not isinstance(rule, SilenceRule) or rule.source != silence_source:
            raise ValueError("silence rule must consume the configured silence source")
        recognizer.capabilities.require(final_flush=True)
        if recognizer.capabilities.requires_complete_segment:
            raise ValueError("silence recipe requires a streaming recognizer")
        self.recognizer = recognizer
        self.recognizer_source = recognizer_source
        self.silence_source = silence_source
        self.rule = rule

    def new_run(self, *, max_samples: int) -> SilenceEndpointRun:
        return SilenceEndpointRun(self, max_samples)


class SilenceEndpointRun(_ComposedRun):
    def __init__(self, pipeline: SilenceEndpointPipeline, max_samples: int) -> None:
        super().__init__(max_samples)
        self._pipeline = pipeline
        self._recognition = pipeline.recognizer.new_run(max_samples=max_samples)
        self._silence = SilenceTracker(source=pipeline.silence_source, max_samples=max_samples)
        self._rule_state = RuleState()

    def feed(self, pcm: bytes, *, speech: bool = True) -> tuple[PipelineEvent, ...]:
        try:
            next_sample = self._validate_feed(pcm, speech)
        except OverflowError:
            self._outcome = RunOutcome.OVERFLOW
            _safe_abort(self._recognition)
            _safe_abort_silence(self._silence)
            raise
        try:
            transcripts = self._recognition.feed(pcm, speech=speech)
        except Exception as error:
            self._failed("streaming", error)
            _safe_abort(self._recognition)
            _safe_abort_silence(self._silence)
            raise
        try:
            observations = self._silence.feed(pcm, speech=speech)
            _validate_transcripts(transcripts, sample_end=next_sample)
            events: list[PipelineEvent] = list(_transcript_events(self._pipeline.recognizer_source, transcripts))
            for observation in observations:
                events.append(ObservationEvent(observation))
                evaluation = self._pipeline.rule.evaluate(observation, self._rule_state)
                self._rule_state = evaluation.state
                events.extend(ActionEvent(item, ActionVisibility.EXTERNAL) for item in evaluation.actions)
            self.sample_end = next_sample
            return tuple(events)
        except Exception as error:
            self._failed("silence_route", error)
            _safe_abort(self._recognition)
            _safe_abort_silence(self._silence)
            raise

    def finish(self) -> tuple[PipelineEvent, ...]:
        self._require_open()
        try:
            lifecycle = self._silence.finish()
            transcripts = self._recognition.finish()
            _validate_transcripts(transcripts, sample_end=self.sample_end, require_final=True)
            events: list[PipelineEvent] = [ObservationEvent(item) for item in lifecycle]
            events.extend(_transcript_events(self._pipeline.recognizer_source, transcripts, selected_final=True))
            self._outcome = RunOutcome.FINISHED
            return tuple(events)
        except Exception as error:
            stage = "streaming_finish" if self._silence.outcome is RunOutcome.FINISHED else "silence_finish"
            self._failed(stage, error)
            _safe_abort(self._recognition)
            _safe_abort_silence(self._silence)
            raise

    def abort(self) -> tuple[PipelineEvent, ...]:
        self._require_open()
        try:
            if self._recognition.outcome is None:
                self._recognition.abort()
            lifecycle = self._silence.abort()
            self._outcome = RunOutcome.ABORTED
            return tuple(ObservationEvent(item) for item in lifecycle)
        except Exception as error:
            self._failed("abort", error)
            raise


class TextEndpointBatchPipeline:
    """Streaming text rule with a conditional all-audio batch pass after EOF."""

    INTERNAL_BATCH_ACTION = "activate_batch_after_finish"

    def __init__(
        self,
        streaming: Recognizer,
        batch: Recognizer,
        scorer: TextScoreProducer,
        *,
        streaming_source: str,
        batch_source: str,
        input_source: str,
        rule: ScoreRule,
    ) -> None:
        for value, name in (
            (streaming_source, "streaming_source"),
            (batch_source, "batch_source"),
            (input_source, "input_source"),
        ):
            _name(value, name)
        if len({streaming_source, batch_source, input_source, scorer.source}) != 4:
            raise ValueError("streaming, batch, input and text-score sources must be distinct")
        if not isinstance(rule, ScoreRule) or rule.score_type is not TextScore or rule.source != scorer.source:
            raise ValueError("text rule must consume TextScore from the configured scorer")
        streaming.capabilities.require(final_flush=True)
        if (
            streaming.capabilities.intermediate is IntermediateOutput.NONE
            or streaming.capabilities.requires_complete_segment
        ):
            raise ValueError("text endpoint recipe requires intermediate streaming transcripts")
        batch.capabilities.require(intermediate=IntermediateOutput.NONE, final_flush=True)
        if not batch.capabilities.requires_complete_segment:
            raise ValueError("conditional final pass requires a complete-segment batch recognizer")
        self.streaming = streaming
        self.batch = batch
        self.scorer = scorer
        self.streaming_source = streaming_source
        self.batch_source = batch_source
        self.input_source = input_source
        self.rule = rule

    def new_run(self, *, max_samples: int) -> TextEndpointBatchRun:
        return TextEndpointBatchRun(self, max_samples)


class TextEndpointBatchRun(_ComposedRun):
    def __init__(self, pipeline: TextEndpointBatchPipeline, max_samples: int) -> None:
        super().__init__(max_samples)
        self._pipeline = pipeline
        self._streaming = pipeline.streaming.new_run(max_samples=max_samples)
        self._retained = bytearray()
        self._rule_state = RuleState()
        self._next_score_event_id = 1
        self._batch_requested = False
        self._latest_streaming: TranscriptRef | None = None

    @property
    def retained_samples(self) -> int:
        return len(self._retained) // 2

    def _release(self) -> None:
        self._retained.clear()

    def _terminate_overflow(self) -> None:
        self._outcome = RunOutcome.OVERFLOW
        self._release()
        _safe_abort(self._streaming)

    def _terminate_failure(self, stage: str, error: Exception) -> None:
        self._failed(stage, error)
        self._release()
        _safe_abort(self._streaming)

    def feed(self, pcm: bytes, *, speech: bool = True) -> tuple[PipelineEvent, ...]:
        try:
            next_sample = self._validate_feed(pcm, speech)
        except OverflowError:
            self._terminate_overflow()
            raise
        try:
            transcripts = self._streaming.feed(pcm, speech=speech)
            _validate_transcripts(transcripts, sample_end=next_sample)
        except Exception as error:
            self._terminate_failure("streaming", error)
            raise
        self.sample_end = next_sample
        self._retained.extend(pcm)
        events: list[PipelineEvent] = []
        for transcript in transcripts:
            recognition = RecognitionEvent(self._pipeline.streaming_source, transcript)
            self._latest_streaming = recognition.reference
            events.append(recognition)
            try:
                event_id = self._next_score_event_id
                score = self._pipeline.scorer.score(
                    transcript,
                    transcript_source=self._pipeline.streaming_source,
                    event_id=event_id,
                    sample_end=self.sample_end,
                )
                self._next_score_event_id += 1
                if (
                    not isinstance(score, TextScore)
                    or score.source != self._pipeline.scorer.source
                    or score.event_id != event_id
                    or score.sample_end != self.sample_end
                    or score.transcript_source != self._pipeline.streaming_source
                    or score.transcript_revision != transcript.revision
                    or score.transcript_sample_end != transcript.sample_end
                ):
                    raise ValueError("text scorer returned mismatched transcript provenance")
            except Exception as error:
                self._terminate_failure("text_score", error)
                raise
            events.append(ObservationEvent(score))
            try:
                evaluation = self._pipeline.rule.evaluate(score, self._rule_state)
                self._rule_state = evaluation.state
                for selection in evaluation.actions:
                    if selection.action != self._pipeline.rule.action:
                        raise ValueError("text rule returned an unknown action")
                    if not self._batch_requested:
                        self._batch_requested = True
                        internal = ActionSelection(
                            action=self._pipeline.INTERNAL_BATCH_ACTION,
                            source=selection.source,
                            event_id=selection.event_id,
                            sample_position=selection.sample_position,
                        )
                        events.append(ActionEvent(internal, ActionVisibility.INTERNAL, recognition.reference))
                    events.append(ActionEvent(selection, ActionVisibility.EXTERNAL, recognition.reference))
            except Exception as error:
                self._terminate_failure("text_rule", error)
                raise
        return tuple(events)

    def _lifecycle(self, signal: LifecycleSignal) -> LifecycleObservation:
        return LifecycleObservation(
            source=self._pipeline.input_source,
            event_id=1,
            sample_end=self.sample_end,
            signal=signal,
        )

    def finish(self) -> tuple[PipelineEvent, ...]:
        self._require_open()
        batch_run: RecognitionRun | None = None
        try:
            events: list[PipelineEvent] = [ObservationEvent(self._lifecycle(LifecycleSignal.END_OF_INPUT))]
            streaming = self._streaming.finish()
            _validate_transcripts(streaming, sample_end=self.sample_end, require_final=True)
            streaming_events = _transcript_events(
                self._pipeline.streaming_source,
                streaming,
                selected_final=not self._batch_requested,
            )
            events.extend(streaming_events)
            for event in streaming_events:
                self._latest_streaming = event.reference
        except Exception as error:
            self._terminate_failure("streaming_finish", error)
            raise

        if self._batch_requested:
            try:
                batch_run = self._pipeline.batch.new_run(max_samples=self.max_samples)
                unexpected = batch_run.feed(bytes(self._retained), speech=True)
                if unexpected:
                    raise ValueError("batch recognizer emitted intermediate transcripts")
                batch = batch_run.finish()
                _validate_transcripts(batch, sample_end=self.sample_end, require_final=True)
                events.extend(
                    _transcript_events(
                        self._pipeline.batch_source,
                        batch,
                        selected_final=True,
                        supersedes=self._latest_streaming,
                    )
                )
            except Exception as error:
                self._terminate_failure("batch", error)
                if batch_run is not None:
                    _safe_abort(batch_run)
                raise
        self._release()
        self._outcome = RunOutcome.FINISHED
        return tuple(events)

    def abort(self) -> tuple[PipelineEvent, ...]:
        self._require_open()
        try:
            if self._streaming.outcome is None:
                self._streaming.abort()
            self._release()
            self._outcome = RunOutcome.ABORTED
            return (ObservationEvent(self._lifecycle(LifecycleSignal.ABORTED)),)
        except Exception as error:
            self._terminate_failure("abort", error)
            raise
