"""Injection rules, the two-classifier ONNX cascade (ADR 0001, ADR 0002), and the committed calibration it reads."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, Self, cast

import numpy as np
import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from tokenizers import Tokenizer

from vitae.guards import PASS, Refuse, RuleId, Stage, Verdict
from vitae.text import Payload, Turn

CALIBRATION_PATH: Final = Path(__file__).with_name("calibration.json")
"""One writer: `python -m bench.guards calibrate --split dev`. Hashed by the pre-registered held-out run."""

WINDOW_TOKENS: Final = 512
WINDOW_STRIDE: Final = 128
BENIGN_LABELS: Final = frozenset({"BENIGN", "SAFE", "LEGIT", "LEGITIMATE"})


_SHA256: Final = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ModelFile:
    dir: str
    """The model's directory relative to VITAE_INJECTION_MODEL_DIR, POSIX separators."""
    onnx: str
    """The ONNX file inside dir."""
    sha256: str
    """sha256 of that ONNX file. A different file needs its own threshold."""

    def problem(self, models_dir: Path) -> str | None:
        model = models_dir / self.dir / self.onnx
        if not model.is_file():
            return f"No injection model at {model}. Set VITAE_INJECTION_MODEL_DIR or run `python -m bench.fetch`."
        if hashlib.sha256(model.read_bytes()).hexdigest() != self.sha256:
            return f"{model} is not the model {CALIBRATION_PATH} was calibrated on (sha256 {self.sha256})."
        return None


@dataclass(frozen=True, slots=True)
class Calibration:
    injection_threshold: float
    """Cascade score at or above which a message is refused as injection."""
    calibrated: bool
    """False until bench.guards has written a threshold measured on the dev split."""
    cascade: tuple[ModelFile, ModelFile]
    """The two models the threshold was measured on, in the order the cascade runs them (ADR 0002)."""

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> Self:
        match json.loads(path.read_bytes()):
            case {
                "injection_threshold": float() | int() as threshold,
                "calibrated": bool() as calibrated,
                "cascade": [
                    {"dir": str() as first_dir, "onnx": str() as first_onnx, "sha256": str() as first_sha},
                    {"dir": str() as second_dir, "onnx": str() as second_onnx, "sha256": str() as second_sha},
                ],
            } if (
                not isinstance(threshold, bool)
                and 0.0 <= threshold <= 1.0
                and _SHA256.fullmatch(first_sha)
                and _SHA256.fullmatch(second_sha)
            ):
                first = ModelFile(first_dir, first_onnx, first_sha)
                return cls(float(threshold), calibrated, (first, ModelFile(second_dir, second_onnx, second_sha)))
            case raw:
                raise ValueError(
                    f"{path}: needs injection_threshold, a number from 0 to 1, calibrated, true or false, and "
                    f"cascade, two models each with dir, onnx and a lowercase hex sha256. Got {raw!r}"
                )

    def model_problem(self, models_dir: Path) -> str | None:
        """Why the cascade under models_dir cannot use this threshold, or None when it can: the threshold
        must be calibrated, and each model there must be the very file it was measured on."""
        if not self.calibrated:
            return f"{CALIBRATION_PATH} is uncalibrated. Run `python -m bench.guards calibrate --split dev` first."
        for model in self.cascade:
            if (problem := model.problem(models_dir)) is not None:
                return problem
        return None


class TextClassifier(Protocol):
    def score(self, text: str) -> float:
        """Probability that text is an injection or jailbreak, the max over overlapping windows. At most 1."""
        ...


class _Input(Protocol):
    @property
    def name(self) -> str: ...


class _Session(Protocol):
    """The part of onnxruntime.InferenceSession used here. onnxruntime ships no type stubs."""

    def get_inputs(self) -> Sequence[_Input]: ...
    def run(self, output_names: None, input_feed: Mapping[str, NDArray[np.int64]]) -> Sequence[NDArray[np.float32]]: ...


def _open_session(path: Path, threads: int) -> _Session:
    options = ort.SessionOptions()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])  # pyright: ignore[reportUnknownArgumentType]
    return cast(_Session, session)


def load_tokenizer(model_dir: Path) -> Tokenizer:
    """model_dir's tokenizer.json, unpadded, cutting longer input into windows of WINDOW_TOKENS that
    overlap by WINDOW_STRIDE."""
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=WINDOW_TOKENS, stride=WINDOW_STRIDE)  # pyright: ignore[reportUnknownMemberType]
    return tokenizer


class OnnxInjectionClassifier:
    """onnxruntime and tokenizers only, no transformers. Windows longer input and keeps the max score."""

    def __init__(self, session: _Session, tokenizer: Tokenizer, pad_id: int, benign_index: int) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._pad_id = pad_id
        self._benign_index = benign_index
        self._inputs = frozenset(node.name for node in session.get_inputs())

    @classmethod
    def load(cls, model_dir: Path, *, onnx_file: str = "model.onnx", threads: int = 1) -> Self:
        """model_dir holds config.json, tokenizer.json and special_tokens_map.json; onnx_file is relative to it.
        threads is onnxruntime's intra-op thread count for this session."""
        config = json.loads((model_dir / "config.json").read_bytes())
        labels = {int(index): str(name).upper() for index, name in config["id2label"].items()}
        benign = [index for index, name in labels.items() if name in BENIGN_LABELS]
        if len(benign) != 1 or len(labels) < 2:
            raise ValueError(f"{model_dir}/config.json: expected exactly one benign label, got {labels}")
        special = json.loads((model_dir / "special_tokens_map.json").read_bytes())
        pad = cast(object, special["pad_token"])
        pad_token = str(cast(dict[str, object], pad)["content"]) if isinstance(pad, dict) else str(pad)
        tokenizer = load_tokenizer(model_dir)
        pad_id = tokenizer.token_to_id(pad_token)
        if pad_id is None:
            raise ValueError(f"{model_dir}: pad token {pad_token!r} is not in tokenizer.json")
        return cls(_open_session(model_dir / onnx_file, threads), tokenizer, pad_id, benign[0])

    def score(self, text: str) -> float:
        encoding = self._tokenizer.encode(text)
        windows = [encoding.ids, *(overflow.ids for overflow in encoding.overflowing)]
        width = max(len(window) for window in windows)
        ids = np.full((len(windows), width), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(windows), width), dtype=np.int64)
        for row, window in enumerate(windows):
            ids[row, : len(window)] = window
            mask[row, : len(window)] = 1
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        (logits,) = self._session.run(None, {name: value for name, value in feed.items() if name in self._inputs})
        return float((1.0 - _softmax(logits)[:, self._benign_index]).max())


def _softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (shifted / shifted.sum(axis=1, keepdims=True)).astype(np.float32)


_NOT_A_MENTION: Final = (
    r"(?![\"'”’»]?\s*(?:attacks?|injections?|style|type|examples?|phrases?|prompts?|jailbreaks?|attempts?|tricks?)\b"
    r"|[\"'”’»]\s*[?)])"
)
"""The phrase used, not discussed: "what are 'ignore previous instructions' attacks?" is a fair question.
A rule that steps aside for a mention leaves the message to the classifier."""

_CLAUSE_START: Final = (
    r"(?:^|[.!?:;,\n(\"'“‘«]\s*|\b(?:please|now|just|and|then|so|first|also|simply|kindly"
    r"|you\s+(?:must|should|will|can|may|need\s+to|have\s+to)|i\s+(?:want|need)\s+you\s+to)\s+)"
)
"""An imperative, not a verb in someone's story: "ignore the rules" starts a clause; "users who ignore the rules" doesn't."""

_RULE_NOUNS: Final = (
    r"(?:instructions?|rules|guidelines|restrictions|constraints|directives|programming|training|safeguards"
    r"|guardrails|policies|prompts?|orders|commands|sentences|context)"
)

_WORD_RULES: Final[tuple[tuple[RuleId, re.Pattern[str]], ...]] = (
    (
        RuleId("injection.override"),
        re.compile(
            r"\b(?:ignore|disregard|forget|skip|override|bypass)\b.{0,40}?"
            r"\b(?:previous|prior|above|earlier|preceding|all|any|your)\b.{0,30}?"
            r"\b(?:instructions?|prompts?|rules|directions|guidelines|commands?|directives?|guidance|programming)\b"
            + _NOT_A_MENTION
        ),
    ),
    (
        RuleId("injection.override"),
        re.compile(
            _CLAUSE_START + r"(?:ignore|disregard|forget|abandon|drop|neglect)\s+(?:all\s+)?(?:of\s+)?"
            r"(?:your|the|these|those|previous|prior|earlier|above|any|all)\s+(?:\w+\s+)?"
            + _RULE_NOUNS
            + r"\b"
            + _NOT_A_MENTION
        ),
    ),
    (
        RuleId("injection.override"),
        re.compile(
            _CLAUSE_START + r"(?:forget|ignore|disregard)\s+(?:about\s+)?(?:everything|anything)\s+"
            r"(?:you|that|above|before|previously|prior|said|else)\b"
        ),
    ),
    (
        RuleId("injection.override"),
        re.compile(
            r"\b(?:ignoriere|ignorieren\s+sie|vergiss|vergessen\s+sie|missachte)\b.{0,40}?"
            r"\b(?:alles|alle|vorherig\w*|obig\w*|bisherig\w*|anweisung\w*|instruktionen|auftrag\w*|befehl\w*"
            r"|vorgaben|regeln)\b"
            r"|\b(?:obigen|vorherigen|bisherigen|vorangegangenen)\s+"
            r"(?:ausfuhrungen|anweisungen|auftrage|befehle|instruktionen|informationen)\s+"
            r"(?:ignorieren|vergessen|missachten)\b"
            r"|\b(?:ignorez|oubliez|oublie)\b.{0,40}?\b(?:instructions|consignes|regles)\b"
            r"|\b(?:ignora|olvida)\b.{0,40}?\b(?:instrucciones|reglas|indicaciones)\b"
        ),
    ),
    (
        RuleId("injection.leak"),
        re.compile(
            r"\b(?:reveal|print|repeat|output|dump|leak|spell\s+out|write\s+out|recite)\s+(?:back\s+)?"
            r"(?:(?:me|us|all|of|the|your|entire|full|whole|complete|exact|original|initial|hidden|secret)\s+){1,4}"
            r"(?:system\s+prompt|system\s+message|instructions\s+above|prompt\s+above|text\s+above|password)\b"
            + _NOT_A_MENTION
        ),
    ),
    (
        RuleId("injection.persona"),
        re.compile(r"\bdo\s+anything\s+now\b|\bdeveloper\s+mode\s+(?:enabled|output)\b|\bdan\s+mode\s+enabled\b"),
    ),
)
"""Run on Turn.folded and every Turn.decoded payload. Only phrasings that stayed at zero false
positives on the dev benign corpora, recruiter questions included, and on the 105,346 Gretel PII dev
documents. A broad role-play rule flagged benign "You are a ..." prompts, so persona setups are left
to the classifier."""

_SQUASHED_RULES: Final[tuple[tuple[RuleId, re.Pattern[str]], ...]] = (
    (
        RuleId("injection.override"),
        re.compile(
            r"(?:ignore|disregard|forget)(?:all|any|the|your)*(?:previous|prior|above|earlier)"
            r"(?:instructions?|prompts?|rules|directions)(?!s?(?:attack|injection|style|type|example))"
        ),
    ),
)
"""Run on Turn.squashed, for the same phrase broken up by spaces or punctuation."""

_READABLE_OVERRIDE: Final = re.compile(
    r"\b(?:ignore|disregard|forget)\s+(?:(?:all|any|the|your)\s+)*(?:previous|prior|above|earlier)\s+"
    r"(?:instructions?|prompts?|rules|directions)\b"
)
"""The squashed phrase written as plain words. The word rules have already judged it, mention or not,
so squashing it again would refuse a quoted question about the attack."""


def _match_rule(turn: Turn) -> RuleId | None:
    for text in (turn.folded, *turn.decoded):
        for rule, pattern in _WORD_RULES:
            if pattern.search(text):
                return rule
    for readable in (turn.folded, *turn.decoded):
        if _READABLE_OVERRIDE.search(readable):
            continue
        squashed = "".join(char for char in readable if char.isalnum())
        for rule, pattern in _SQUASHED_RULES:
            if pattern.search(squashed):
                return rule
    return None


def injection_rule_stage() -> Stage:
    def check(turn: Turn) -> Verdict:
        rule = _match_rule(turn)
        return PASS if rule is None else Refuse("injection", rule)

    return Stage("injection_rules", check)


_ENCODED: Final = frozenset({"base64", "hex", "url"})
_COMMON_WORDS: Final = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on", "at", "by", "for", "with",
    "from", "as", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
    "not", "no", "yes", "you", "your", "yours", "i", "me", "my", "we", "us", "our", "it", "its", "this",
    "that", "these", "those", "what", "which", "who", "whom", "how", "why", "when", "where", "all", "any",
    "some", "can", "could", "will", "would", "should", "may", "might", "must", "shall", "about", "into",
    "over", "under", "again", "only", "just", "also", "than", "so", "very", "now", "here", "there", "please",
    "tell", "say", "said", "write", "give", "show", "print", "ignore", "instructions", "previous", "above",
    "below", "password", "secret", "prompt", "system",
})  # fmt: skip
_WORDS: Final = re.compile(r"[a-z]+")
_RISING_CASE: Final = re.compile(r"[a-z][A-Z]")
_LETTER_RUN: Final = re.compile(r"[A-Za-z]{3,}")
LANGUAGE_MARGIN: Final = 0.05
LANGUAGE_FLOOR: Final = 0.15
LANGUAGE_WORDS: Final = 3
IRREGULAR_CASE: Final = 0.25


def _common_words(text: str) -> tuple[int, int]:
    """(common English words, all words) in text."""
    words = _WORDS.findall(text.lower())
    return sum(word in _COMMON_WORDS for word in words), len(words)


def _reads_as_language(payload: Payload, message: str) -> bool:
    """A leet or rot13 reading that is more English than the message it came from: its share of common
    words is clearly higher. A rot13 reading also counts when it has several common words and at least
    half as many as the message, for a rot13 attack behind a plain request. rot13 scrambles every word,
    so its common words are all new; leet keeps most words, so that count would pass plain English."""
    common, words = _common_words(payload.text)
    plain, plain_words = _common_words(message)
    share = common / words if words else 0.0
    plain_share = plain / plain_words if plain_words else 0.0
    if share >= max(plain_share + LANGUAGE_MARGIN, LANGUAGE_FLOOR):
        return True
    return payload.kind == "rot13" and common >= LANGUAGE_WORDS and 2 * common >= plain


def _irregular_case_share(text: str) -> float:
    words = _LETTER_RUN.findall(text)
    # Names like "JavaScript" or "iPhone" rise from lower to upper case once; random casing rises often.
    return sum(len(_RISING_CASE.findall(word)) >= 2 for word in words) / len(words) if words else 0.0


def classifier_views(turn: Turn) -> tuple[str, ...]:
    """What the classifier scores: the text, every explicitly encoded payload, a leet or rot13 reading
    only when it reads as language, the folded text when the letters are randomly cased, and the same
    again for each line of a message or decoding written on several lines. The rot13 reading of an
    ordinary message is noise, and scoring it would double the classifier's cost on every turn."""
    views = [turn.text]
    for payload in turn.payloads:
        if payload.kind in _ENCODED or _reads_as_language(payload, turn.folded):
            views += [payload.text, *payload.lines]
    if _irregular_case_share(turn.text) >= IRREGULAR_CASE:
        views.append(turn.folded)
    for line in turn.lines:
        views += classifier_views(line)
    return tuple(dict.fromkeys(views))


def classifier_score(classifier: TextClassifier, turn: Turn) -> float:
    return max(classifier.score(view) for view in classifier_views(turn))


@dataclass(frozen=True, slots=True)
class CascadeClassifier:
    """ADR 0002. A message's score is the mean of the two classifiers' scores, each the max over
    classifier_views. The second classifier runs only when the mean could still reach the threshold,
    so the cascade refuses exactly the messages the full mean refuses."""

    first: TextClassifier
    second: TextClassifier

    @classmethod
    def load(cls, models_dir: Path, calibration: Calibration, *, threads: int = 1) -> Self:
        first, second = (
            OnnxInjectionClassifier.load(models_dir / model.dir, onnx_file=model.onnx, threads=threads)
            for model in calibration.cascade
        )
        return cls(first, second)

    def mean(self, turn: Turn) -> float:
        """Both classifiers, always: what the threshold is calibrated on."""
        return (classifier_score(self.first, turn) + classifier_score(self.second, turn)) / 2

    def refuses(self, turn: Turn, threshold: float) -> bool:
        views = classifier_views(turn)
        first = max(self.first.score(view) for view in views)
        # ADR 0002's "first < 2t - 1", written as the highest mean a second score of at most 1 could give.
        # Rounding is monotonic, so this form is exact where the rearranged one could be off by one ulp.
        if (first + 1.0) / 2 < threshold:
            return False
        return (first + max(self.second.score(view) for view in views)) / 2 >= threshold


InjectionClassifier = CascadeClassifier | TextClassifier
"""What the classifier stage runs: the cascade in production, or one classifier on its own, as the
tests' stand-ins and the benchmark's single-model baseline do."""

CLASSIFIER_RULE: Final = RuleId("injection.classifier")


def injection_classifier_stage(classifier: InjectionClassifier, threshold: float) -> Stage:
    def check(turn: Turn) -> Verdict:
        if isinstance(classifier, CascadeClassifier):
            refused = classifier.refuses(turn, threshold)
        else:
            refused = classifier_score(classifier, turn) >= threshold
        return Refuse("injection", CLASSIFIER_RULE) if refused else PASS

    return Stage("injection_classifier", check)
