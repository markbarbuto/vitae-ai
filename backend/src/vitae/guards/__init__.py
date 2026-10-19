"""The verdict vocabulary every input guard speaks, and the ordered stage list."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, NewType

from vitae.contract import RefusalReason
from vitae.profile import Profile
from vitae.text import Turn

StageName = Literal[
    "pii", "injection_rules", "injection_classifier", "moderation", "blocked_topic", "availability", "scope"
]
RuleId = NewType("RuleId", str)
"""Stable, low-cardinality id of what fired: "pii.card", "injection.classifier", "topic.family".
Built from engine constants and profile slugs only, never from visitor text. Safe to log."""


@dataclass(frozen=True, slots=True)
class Pass:
    pass


PASS: Final = Pass()


@dataclass(frozen=True, slots=True)
class Refuse:
    reason: RefusalReason
    """The copy comes from profile.refusals[reason] at the edge, never from here."""
    rule: RuleId


@dataclass(frozen=True, slots=True)
class Availability:
    rule: RuleId
    """The answer and action live in the profile."""


@dataclass(frozen=True, slots=True)
class Skipped:
    """A slot with no backing model (Llama Guard in phase 2). Logged as skipped, never as a pass."""

    why: Literal["no_model"]


@dataclass(frozen=True, slots=True)
class Down:
    """A model stage could not answer (phase 4: Llama Guard server down)."""

    rule: RuleId


Verdict = Pass | Skipped | Refuse | Availability | Down


@dataclass(frozen=True, slots=True)
class Stage:
    name: StageName
    check: Callable[[Turn], Verdict]
    """Sync and side-effect free. May be CPU-heavy (the classifier). Runs off the event loop."""


@dataclass(frozen=True, slots=True)
class InputStages:
    stages: tuple[Stage, ...]
    """Pipeline order. Security verdicts come before topic verdicts, so an attack that also
    happens to be off-topic is logged and scored as the attack."""

    def for_reason(self, reason: RefusalReason) -> "InputStages":
        """The subset whose stage can return that reason, in the same order. For the benchmark."""
        return InputStages(tuple(stage for stage in self.stages if STAGE_REASONS[stage.name] == reason))


STAGE_REASONS: Final[Mapping[StageName, RefusalReason | None]] = {
    "pii": "pii",
    "injection_rules": "injection",
    "injection_classifier": "injection",
    "moderation": "unsafe",
    "blocked_topic": "blocked_topic",
    "availability": None,
    "scope": "unrelated",
}
"""One table. for_reason, the threat model's stage-to-OWASP map and the log field derive from it."""

# The stage modules import the vocabulary above, so they can only be imported once it exists.
from vitae.guards.injection import (
    Calibration,
    InjectionClassifier,
    injection_classifier_stage,
    injection_rule_stage,
)
from vitae.guards.moderation import Moderator, moderation_stage
from vitae.guards.pii import pii_stage
from vitae.guards.topics import availability_stage, blocked_stage, scope_stage


def build_input_stages(
    profile: Profile, *, classifier: InjectionClassifier, moderator: Moderator, calibration: Calibration
) -> InputStages:
    return InputStages(
        (
            pii_stage(),
            injection_rule_stage(),
            injection_classifier_stage(classifier, calibration.injection_threshold),
            moderation_stage(moderator),
            blocked_stage(profile.topics),
            availability_stage(profile.availability),
            scope_stage(profile.topics, profile.assistant.addressing),
        )
    )
