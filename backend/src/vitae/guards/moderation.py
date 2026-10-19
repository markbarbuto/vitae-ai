"""The Llama Guard slot. Phase 2 wires NO_MODERATION, which every turn logs as skipped."""

from typing import Final, Protocol

from vitae.guards import Skipped, Stage, Verdict
from vitae.text import Turn


class Moderator(Protocol):
    def check(self, turn: Turn) -> Verdict:
        """Refuse("unsafe", ...) for a harmful request, Down when the model cannot answer."""
        ...


class _NoModeration:
    def check(self, turn: Turn) -> Verdict:
        return SKIPPED


SKIPPED: Final = Skipped("no_model")
NO_MODERATION: Final[Moderator] = _NoModeration()


def moderation_stage(moderator: Moderator) -> Stage:
    return Stage("moderation", moderator.check)
