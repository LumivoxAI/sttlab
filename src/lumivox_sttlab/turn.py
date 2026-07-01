"""Experimental synchronous turn observations and deterministic rules.

The types in this module are composition building blocks, not a stable public
trigger DSL. Rules are side-effect free: callers pass explicit evaluator state
and receive the next state plus zero or more action selections.
"""

from enum import Enum
from math import isfinite
from dataclasses import dataclass

from .recognition import RunOutcome


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _position(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _name(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, kw_only=True)
class TurnObservation:
    """One source-attributed event at an absolute 16-kHz sample position.

    ``event_id`` is strictly increasing within one source and run. Identity is
    the pair ``(source, event_id)``; provenance fields do not double as event
    identity.
    """

    source: str
    event_id: int
    sample_end: int

    def __post_init__(self) -> None:
        _name(self.source, "source")
        _positive_int(self.event_id, "event_id")
        _position(self.sample_end, "sample_end")

    @property
    def identity(self) -> tuple[str, int]:
        return (self.source, self.event_id)


@dataclass(frozen=True, kw_only=True)
class SilenceObservation(TurnObservation):
    """A homogeneous speech-labelled, non-empty sample interval.

    For non-speech, ``silence_start`` identifies the start of the current
    uninterrupted silence and may precede ``sample_start``. A rule can thus
    locate a threshold crossing inside a large input chunk exactly. It is
    ``None`` for speech, which resets silence-based rules.
    """

    sample_start: int
    speech: bool
    silence_start: int | None

    def __post_init__(self) -> None:
        super().__post_init__()
        _position(self.sample_start, "sample_start")
        if self.sample_start >= self.sample_end:
            raise ValueError("a silence observation interval must be non-empty")
        if not isinstance(self.speech, bool):
            raise ValueError("speech must be a bool")
        if self.speech:
            if self.silence_start is not None:
                raise ValueError("speech observations cannot have a silence_start")
        else:
            if self.silence_start is None:
                raise ValueError("non-speech observations require silence_start")
            _position(self.silence_start, "silence_start")
            if self.silence_start > self.sample_start:
                raise ValueError("silence_start cannot follow sample_start")


@dataclass(frozen=True, kw_only=True)
class ScoreProvenance:
    """Model identity and an optional caller-readable configuration revision."""

    model: str
    configuration: str | None = None

    def __post_init__(self) -> None:
        _name(self.model, "model")
        if self.configuration is not None:
            _name(self.configuration, "configuration")


def _validate_score(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError("value must be a finite score")


@dataclass(frozen=True, kw_only=True)
class AcousticScore(TurnObservation):
    """A model-specific score over an explicit bounded audio range."""

    sample_start: int
    value: float
    provenance: ScoreProvenance

    def __post_init__(self) -> None:
        super().__post_init__()
        _position(self.sample_start, "sample_start")
        if self.sample_start > self.sample_end:
            raise ValueError("sample_start cannot follow sample_end")
        _validate_score(self.value)
        if not isinstance(self.provenance, ScoreProvenance):
            raise ValueError("provenance must be ScoreProvenance")


@dataclass(frozen=True, kw_only=True)
class TextScore(TurnObservation):
    """A model-specific score attributed to one exact transcript revision."""

    value: float
    provenance: ScoreProvenance
    transcript_source: str
    transcript_revision: int
    transcript_sample_end: int

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_score(self.value)
        if not isinstance(self.provenance, ScoreProvenance):
            raise ValueError("provenance must be ScoreProvenance")
        _name(self.transcript_source, "transcript_source")
        _positive_int(self.transcript_revision, "transcript_revision")
        _position(self.transcript_sample_end, "transcript_sample_end")
        if self.transcript_sample_end > self.sample_end:
            raise ValueError("a transcript cannot end after its text score")


@dataclass(frozen=True, kw_only=True)
class BooleanObservation(TurnObservation):
    """A named producer's boolean condition without probability semantics."""

    value: bool

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.value, bool):
            raise ValueError("value must be a bool")


class LifecycleSignal(str, Enum):
    END_OF_INPUT = "end_of_input"
    ABORTED = "aborted"


@dataclass(frozen=True, kw_only=True)
class LifecycleObservation(TurnObservation):
    """An input boundary, distinct from successful recognizer inference."""

    signal: LifecycleSignal

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.signal, LifecycleSignal):
            raise ValueError("signal must be a LifecycleSignal")


class SilenceTracker:
    """Bounded run-local silence state over mono PCM S16LE feeds."""

    def __init__(self, *, source: str, max_samples: int) -> None:
        _name(source, "source")
        _positive_int(max_samples, "max_samples")
        self.source = source
        self.max_samples = max_samples
        self.sample_end = 0
        self._silence_start: int | None = None
        self._next_event_id = 1
        self._outcome: RunOutcome | None = None

    @property
    def outcome(self) -> RunOutcome | None:
        return self._outcome

    def feed(self, pcm: bytes, *, speech: bool = True) -> tuple[SilenceObservation, ...]:
        self._require_open()
        if not isinstance(pcm, bytes) or len(pcm) % 2:
            raise ValueError("expected complete mono PCM S16LE samples as bytes")
        if not isinstance(speech, bool):
            raise ValueError("speech must be a bool")
        samples = len(pcm) // 2
        if self.sample_end + samples > self.max_samples:
            self._outcome = RunOutcome.OVERFLOW
            self._silence_start = None
            raise OverflowError("run audio bound exceeded")
        if samples == 0:
            return ()
        sample_start = self.sample_end
        self.sample_end += samples
        if speech:
            self._silence_start = None
        elif self._silence_start is None:
            self._silence_start = sample_start
        observation = SilenceObservation(
            source=self.source,
            event_id=self._take_event_id(),
            sample_start=sample_start,
            sample_end=self.sample_end,
            speech=speech,
            silence_start=self._silence_start,
        )
        return (observation,)

    def finish(self) -> tuple[LifecycleObservation, ...]:
        self._require_open()
        self._outcome = RunOutcome.FINISHED
        self._silence_start = None
        return (
            LifecycleObservation(
                source=self.source,
                event_id=self._take_event_id(),
                sample_end=self.sample_end,
                signal=LifecycleSignal.END_OF_INPUT,
            ),
        )

    def abort(self) -> tuple[LifecycleObservation, ...]:
        self._require_open()
        self._outcome = RunOutcome.ABORTED
        self._silence_start = None
        return (
            LifecycleObservation(
                source=self.source,
                event_id=self._take_event_id(),
                sample_end=self.sample_end,
                signal=LifecycleSignal.ABORTED,
            ),
        )

    def _require_open(self) -> None:
        if self._outcome is not None:
            raise RuntimeError("tracker run already finished or aborted")

    def _take_event_id(self) -> int:
        event_id = self._next_event_id
        self._next_event_id += 1
        return event_id


class FiringMode(str, Enum):
    EDGE = "edge"
    LEVEL = "level"


class ScoreComparison(str, Enum):
    AT_LEAST = "at_least"
    AT_MOST = "at_most"


@dataclass(frozen=True)
class RuleState:
    """Complete bounded state for one rule in one run."""

    last_event_id: int | None = None
    active: bool = False
    next_sample: int | None = None


@dataclass(frozen=True)
class ActionSelection:
    """An opaque action selected by one attributable observation."""

    action: str
    source: str
    event_id: int
    sample_position: int


@dataclass(frozen=True)
class RuleEvaluation:
    state: RuleState
    actions: tuple[ActionSelection, ...] = ()


def _rule_config(source: str, action: str, firing: FiringMode) -> None:
    _name(source, "source")
    _name(action, "action")
    if not isinstance(firing, FiringMode):
        raise ValueError("firing must be a FiringMode")


def _new_event(observation: TurnObservation, state: RuleState) -> bool:
    if state.last_event_id is None:
        return True
    if observation.event_id < state.last_event_id:
        raise ValueError("event_id must increase monotonically within a source and run")
    return observation.event_id != state.last_event_id


def _selection(action: str, observation: TurnObservation, sample_position: int | None = None) -> ActionSelection:
    return ActionSelection(
        action=action,
        source=observation.source,
        event_id=observation.event_id,
        sample_position=observation.sample_end if sample_position is None else sample_position,
    )


def _condition_evaluation(
    *,
    observation: TurnObservation,
    state: RuleState,
    condition: bool,
    action: str,
    firing: FiringMode,
) -> RuleEvaluation:
    if not _new_event(observation, state):
        return RuleEvaluation(state)
    actions: tuple[ActionSelection, ...] = ()
    if condition and (firing is FiringMode.LEVEL or not state.active):
        actions = (_selection(action, observation),)
    return RuleEvaluation(RuleState(last_event_id=observation.event_id, active=condition), actions)


@dataclass(frozen=True, kw_only=True)
class SilenceRule:
    source: str
    threshold_samples: int
    action: str
    firing: FiringMode = FiringMode.EDGE
    repeat_samples: int | None = None

    def __post_init__(self) -> None:
        _rule_config(self.source, self.action, self.firing)
        _positive_int(self.threshold_samples, "threshold_samples")
        if self.firing is FiringMode.LEVEL:
            if self.repeat_samples is None:
                raise ValueError("level silence rules require repeat_samples")
            _positive_int(self.repeat_samples, "repeat_samples")
        elif self.repeat_samples is not None:
            raise ValueError("edge silence rules cannot have repeat_samples")

    def evaluate(self, observation: TurnObservation, state: RuleState = RuleState()) -> RuleEvaluation:
        if not isinstance(observation, SilenceObservation) or observation.source != self.source:
            return RuleEvaluation(state)
        if not _new_event(observation, state):
            return RuleEvaluation(state)
        if observation.speech:
            return RuleEvaluation(RuleState(last_event_id=observation.event_id))
        assert observation.silence_start is not None
        threshold_position = observation.silence_start + self.threshold_samples
        if self.firing is FiringMode.EDGE:
            condition = observation.sample_end >= threshold_position
            actions: tuple[ActionSelection, ...] = ()
            if condition and not state.active:
                actions = (_selection(self.action, observation, threshold_position),)
            return RuleEvaluation(RuleState(last_event_id=observation.event_id, active=condition), actions)

        assert self.repeat_samples is not None
        next_sample = state.next_sample if state.next_sample is not None else threshold_position
        selected: list[ActionSelection] = []
        while next_sample <= observation.sample_end:
            selected.append(_selection(self.action, observation, next_sample))
            next_sample += self.repeat_samples
        return RuleEvaluation(
            RuleState(
                last_event_id=observation.event_id,
                active=observation.sample_end >= threshold_position,
                next_sample=next_sample,
            ),
            tuple(selected),
        )


ScoreType = type[AcousticScore] | type[TextScore]


@dataclass(frozen=True, kw_only=True)
class ScoreRule:
    source: str
    score_type: ScoreType
    threshold: float
    comparison: ScoreComparison
    action: str
    firing: FiringMode = FiringMode.EDGE

    def __post_init__(self) -> None:
        _rule_config(self.source, self.action, self.firing)
        if self.score_type not in (AcousticScore, TextScore):
            raise ValueError("score_type must be AcousticScore or TextScore")
        _validate_score(self.threshold)
        if not isinstance(self.comparison, ScoreComparison):
            raise ValueError("comparison must be a ScoreComparison")

    def evaluate(self, observation: TurnObservation, state: RuleState = RuleState()) -> RuleEvaluation:
        if not isinstance(observation, self.score_type) or observation.source != self.source:
            return RuleEvaluation(state)
        condition = (
            observation.value >= self.threshold
            if self.comparison is ScoreComparison.AT_LEAST
            else observation.value <= self.threshold
        )
        return _condition_evaluation(
            observation=observation,
            state=state,
            condition=condition,
            action=self.action,
            firing=self.firing,
        )


@dataclass(frozen=True, kw_only=True)
class BooleanRule:
    source: str
    expected: bool
    action: str
    firing: FiringMode = FiringMode.EDGE

    def __post_init__(self) -> None:
        _rule_config(self.source, self.action, self.firing)
        if not isinstance(self.expected, bool):
            raise ValueError("expected must be a bool")

    def evaluate(self, observation: TurnObservation, state: RuleState = RuleState()) -> RuleEvaluation:
        if not isinstance(observation, BooleanObservation) or observation.source != self.source:
            return RuleEvaluation(state)
        return _condition_evaluation(
            observation=observation,
            state=state,
            condition=observation.value is self.expected,
            action=self.action,
            firing=self.firing,
        )


@dataclass(frozen=True, kw_only=True)
class LifecycleRule:
    source: str
    signal: LifecycleSignal
    action: str
    firing: FiringMode = FiringMode.EDGE

    def __post_init__(self) -> None:
        _rule_config(self.source, self.action, self.firing)
        if not isinstance(self.signal, LifecycleSignal):
            raise ValueError("signal must be a LifecycleSignal")

    def evaluate(self, observation: TurnObservation, state: RuleState = RuleState()) -> RuleEvaluation:
        if not isinstance(observation, LifecycleObservation) or observation.source != self.source:
            return RuleEvaluation(state)
        return _condition_evaluation(
            observation=observation,
            state=state,
            condition=observation.signal is self.signal,
            action=self.action,
            firing=self.firing,
        )
