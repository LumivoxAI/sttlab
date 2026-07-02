"""Table-driven contracts for turn observations and pure trigger rules."""

from collections.abc import Iterable

import pytest

from lumivox_sttlab.turn import (
    RuleState,
    ScoreRule,
    TextScore,
    FiringMode,
    BooleanRule,
    SilenceRule,
    AcousticScore,
    LifecycleRule,
    SilenceTracker,
    LifecycleSignal,
    ScoreComparison,
    ScoreProvenance,
    BooleanObservation,
    SilenceObservation,
    LifecycleObservation,
)
from lumivox_sttlab.recognition import RunOutcome

PROVENANCE = ScoreProvenance(model="fake-turn-v1", configuration="threshold-study-a")


def _pcm(samples: int) -> bytes:
    return b"\0\0" * samples


def test_silence_tracker_boundaries_reset_empty_lifecycle_and_isolation() -> None:
    first = SilenceTracker(source="vad", max_samples=8)
    second = SilenceTracker(source="vad", max_samples=4)

    assert first.feed(b"", speech=False) == ()
    silence_a = first.feed(_pcm(2), speech=False)[0]
    silence_b = first.feed(_pcm(3), speech=False)[0]
    speech = first.feed(_pcm(1), speech=True)[0]
    silence_c = first.feed(_pcm(1), speech=False)[0]
    assert [
        (event.event_id, event.sample_start, event.sample_end, event.silence_start) for event in (silence_a, silence_b)
    ] == [
        (1, 0, 2, 0),
        (2, 2, 5, 0),
    ]
    assert (speech.event_id, speech.sample_start, speech.sample_end, speech.silence_start) == (3, 5, 6, None)
    assert (silence_c.event_id, silence_c.silence_start) == (4, 6)
    eof = first.finish()[0]
    assert (eof.event_id, eof.sample_end, eof.signal) == (5, 7, LifecycleSignal.END_OF_INPUT)
    assert first.outcome is RunOutcome.FINISHED
    with pytest.raises(RuntimeError):
        first.feed(b"")

    assert second.sample_end == 0
    aborted = second.abort()[0]
    assert (aborted.event_id, aborted.sample_end, aborted.signal) == (1, 0, LifecycleSignal.ABORTED)
    assert second.outcome is RunOutcome.ABORTED


def test_silence_tracker_validates_input_and_closes_on_overflow() -> None:
    tracker = SilenceTracker(source="vad", max_samples=1)
    with pytest.raises(ValueError, match="PCM"):
        tracker.feed(b"\0")
    with pytest.raises(ValueError, match="speech"):
        tracker.feed(_pcm(1), speech=1)  # type: ignore[arg-type]
    with pytest.raises(OverflowError):
        tracker.feed(_pcm(2), speech=False)
    assert tracker.outcome is RunOutcome.OVERFLOW
    with pytest.raises(RuntimeError):
        tracker.finish()


def _silence_actions(chunks: Iterable[tuple[int, bool]], rule: SilenceRule) -> list[int]:
    tracker = SilenceTracker(source="vad", max_samples=20)
    state = RuleState()
    positions: list[int] = []
    for samples, speech in chunks:
        for observation in tracker.feed(_pcm(samples), speech=speech):
            result = rule.evaluate(observation, state)
            state = result.state
            positions.extend(action.sample_position for action in result.actions)
    return positions


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        ([(2, False), (5, False), (1, True), (4, False)], [4, 12]),
        ([(1, False), (1, False), (1, False), (1, False), (3, False), (1, True), (2, False), (2, False)], [4, 12]),
    ],
)
def test_edge_silence_rule_is_chunk_boundary_invariant(chunks: list[tuple[int, bool]], expected: list[int]) -> None:
    rule = SilenceRule(source="vad", threshold_samples=4, action="endpoint")
    assert _silence_actions(chunks, rule) == expected


@pytest.mark.parametrize(
    "chunks",
    [
        [(10, False)],
        [(1, False), (2, False), (4, False), (3, False)],
    ],
)
def test_level_silence_rule_uses_sample_cadence_not_chunk_cadence(chunks: list[tuple[int, bool]]) -> None:
    rule = SilenceRule(
        source="vad",
        threshold_samples=4,
        action="still-silent",
        firing=FiringMode.LEVEL,
        repeat_samples=3,
    )
    assert _silence_actions(chunks, rule) == [4, 7, 10]


def _acoustic(event_id: int, value: float, *, sample_end: int | None = None) -> AcousticScore:
    end = event_id if sample_end is None else sample_end
    return AcousticScore(
        source="acoustic",
        event_id=event_id,
        sample_start=0,
        sample_end=end,
        value=value,
        provenance=PROVENANCE,
    )


def test_score_rule_edge_rearming_duplicate_suppression_and_explicit_state() -> None:
    rule = ScoreRule(
        source="acoustic",
        score_type=AcousticScore,
        threshold=0.7,
        comparison=ScoreComparison.AT_LEAST,
        action="endpoint",
    )
    state = RuleState()
    rows = [
        (_acoustic(1, 0.6), []),
        (_acoustic(2, 0.7), [2]),
        (_acoustic(2, 0.7), []),
        (_acoustic(3, 0.9), []),
        (_acoustic(4, 0.5), []),
        (_acoustic(5, 0.8), [5]),
    ]
    for observation, positions in rows:
        result = rule.evaluate(observation, state)
        assert [action.sample_position for action in result.actions] == positions
        state = result.state

    # Rules retain no hidden run state: replaying from the same state is exact.
    first = rule.evaluate(_acoustic(1, 0.8), RuleState())
    assert first == rule.evaluate(_acoustic(1, 0.8), RuleState())
    with pytest.raises(ValueError, match="monotonically"):
        rule.evaluate(_acoustic(4, 0.8), state)


def test_level_score_rule_fires_for_each_distinct_matching_observation() -> None:
    rule = ScoreRule(
        source="acoustic",
        score_type=AcousticScore,
        threshold=0.3,
        comparison=ScoreComparison.AT_MOST,
        action="continue",
        firing=FiringMode.LEVEL,
    )
    state = RuleState()
    fired: list[int] = []
    for observation in (_acoustic(1, 0.2), _acoustic(2, 0.1), _acoustic(3, 0.4), _acoustic(4, 0.3)):
        result = rule.evaluate(observation, state)
        state = result.state
        fired.extend(action.event_id for action in result.actions)
    assert fired == [1, 2, 4]


def test_text_score_carries_exact_transcript_revision_and_rules_are_typed() -> None:
    score = TextScore(
        source="text-turn",
        event_id=3,
        sample_end=8000,
        value=1.25,
        provenance=PROVENANCE,
        transcript_source="tone",
        transcript_revision=2,
        transcript_sample_end=7600,
    )
    text_rule = ScoreRule(
        source="text-turn",
        score_type=TextScore,
        threshold=1.0,
        comparison=ScoreComparison.AT_LEAST,
        action="endpoint",
    )
    assert text_rule.evaluate(score).actions[0].event_id == 3
    acoustic_rule = ScoreRule(
        source="text-turn",
        score_type=AcousticScore,
        threshold=1.0,
        comparison=ScoreComparison.AT_LEAST,
        action="wrong-type",
    )
    assert acoustic_rule.evaluate(score).actions == ()


def test_boolean_and_lifecycle_rules_have_independent_edge_state() -> None:
    boolean = BooleanRule(source="gate", expected=True, action="activate")
    state = RuleState()
    for event_id, value, expected_count in [(1, False, 0), (2, True, 1), (3, True, 0), (4, False, 0), (5, True, 1)]:
        result = boolean.evaluate(
            BooleanObservation(source="gate", event_id=event_id, sample_end=event_id, value=value), state
        )
        assert len(result.actions) == expected_count
        state = result.state

    lifecycle = LifecycleRule(source="input", signal=LifecycleSignal.END_OF_INPUT, action="flush")
    abort = LifecycleObservation(source="input", event_id=1, sample_end=5, signal=LifecycleSignal.ABORTED)
    eof = LifecycleObservation(source="input", event_id=2, sample_end=5, signal=LifecycleSignal.END_OF_INPUT)
    state = lifecycle.evaluate(abort).state
    assert lifecycle.evaluate(eof, state).actions[0].action == "flush"


@pytest.mark.parametrize(
    "build",
    [
        lambda: SilenceObservation(
            source="vad", event_id=1, sample_start=2, sample_end=2, speech=True, silence_start=None
        ),
        lambda: SilenceObservation(
            source="vad", event_id=1, sample_start=0, sample_end=1, speech=False, silence_start=None
        ),
        lambda: AcousticScore(
            source="acoustic", event_id=1, sample_start=2, sample_end=1, value=0.5, provenance=PROVENANCE
        ),
        lambda: AcousticScore(
            source="acoustic", event_id=1, sample_start=0, sample_end=1, value=float("nan"), provenance=PROVENANCE
        ),
        lambda: TextScore(
            source="text",
            event_id=1,
            sample_end=2,
            value=0.5,
            provenance=PROVENANCE,
            transcript_source="tone",
            transcript_revision=1,
            transcript_sample_end=3,
        ),
    ],
)
def test_observations_reject_invalid_ranges_and_provenance(build: object) -> None:
    with pytest.raises(ValueError):
        build()  # type: ignore[operator]


def test_rule_configuration_rejects_incompatible_values() -> None:
    with pytest.raises(ValueError, match="repeat_samples"):
        SilenceRule(source="vad", threshold_samples=2, action="x", firing=FiringMode.LEVEL)
    with pytest.raises(ValueError, match="cannot have"):
        SilenceRule(source="vad", threshold_samples=2, action="x", repeat_samples=1)
    with pytest.raises(ValueError, match="score_type"):
        ScoreRule(
            source="score",
            score_type=BooleanObservation,  # type: ignore[arg-type]
            threshold=0.5,
            comparison=ScoreComparison.AT_LEAST,
            action="x",
        )
