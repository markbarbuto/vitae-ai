"""PII recognisers with checksums. The input guard refuses structured identifiers and street
addresses; the output rule withholds any hit not in the profile's allowed contacts.

Rules only, and precision over recall: a structured kind is reported only when its format and,
where one exists, its checksum hold. Names are found only after a cue (a label, an honorific, a
salutation, "my name is"), never from capitalisation alone.
"""

import ipaddress
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never

from vitae.guards import PASS, Refuse, RuleId, Stage, Verdict
from vitae.text import Turn


class PiiKind(StrEnum):
    """Every kind T2 reports. Dispatch on it with match and assert_never, so a kind added here
    without a recogniser fails pyright."""

    EMAIL = "email"
    PHONE = "phone"
    CARD = "card"
    IBAN = "iban"
    NATIONAL_ID = "national_id"
    IP = "ip"
    STREET_ADDRESS = "street_address"
    NAME = "name"


REFUSED_AT_INPUT: Final = frozenset(PiiKind) - {PiiKind.NAME}
"""Owner decision 2026-10-18: a recruiter introducing themselves, or a question naming the owner,
is not refused. Names are handled at output."""

PRIORITY: Final = (
    PiiKind.IBAN,
    PiiKind.CARD,
    PiiKind.NATIONAL_ID,
    PiiKind.EMAIL,
    PiiKind.IP,
    PiiKind.PHONE,
    PiiKind.STREET_ADDRESS,
    PiiKind.NAME,
)
"""Where two recognisers claim overlapping text, the earlier kind keeps it. Checksummed kinds come
first, because a Luhn-valid card number also has the shape of a long phone number."""


@dataclass(frozen=True, slots=True)
class PiiHit:
    kind: PiiKind
    start: int
    end: int
    """Offsets into the text searched, so the output rule can compare the hit with allowed contacts."""


@dataclass(frozen=True, slots=True)
class _Clues:
    """What the recognisers' regexes need before they can match, computed once per text. An answer
    sentence with a year in it skips every recogniser that needs six digits or more."""

    digits: int
    words: frozenset[str]
    """Casefolded runs of letters."""


_DIGIT: Final = re.compile(r"\d")
_LETTER_RUN: Final = re.compile(r"[^\W\d_]+")


def find_pii(text: str) -> tuple[PiiHit, ...]:
    clues = _Clues(len(_DIGIT.findall(text)), frozenset(_LETTER_RUN.findall(text.casefold())))
    hits: list[PiiHit] = []
    for kind in PRIORITY:
        for start, end in _spans(kind, text, clues):
            if not any(hit.start < end and start < hit.end for hit in hits):
                hits.append(PiiHit(kind, start, end))
    return tuple(sorted(hits, key=lambda hit: (hit.start, hit.end)))


def pii_stage() -> Stage:
    return Stage("pii", _check)


def _check(turn: Turn) -> Verdict:
    for hit in find_pii(turn.text):
        if hit.kind in REFUSED_AT_INPUT:
            return Refuse("pii", RuleId(f"pii.{hit.kind}"))
    for view in turn.decoded:
        for hit in find_pii(view):
            # A decoding carries the message's digits through unchanged (rot13 does) while it scrambles
            # the cue words around them. Only a hit the decoding revealed is new evidence.
            if hit.kind in REFUSED_AT_INPUT and view[hit.start : hit.end] not in turn.folded:
                return Refuse("pii", RuleId(f"pii.{hit.kind}"))
    return PASS


def _spans(kind: PiiKind, text: str, clues: _Clues) -> Iterable[tuple[int, int]]:
    match kind:
        case PiiKind.EMAIL:
            return _emails(text) if "@" in text else ()
        case PiiKind.PHONE:
            return _phones(text) if clues.digits >= 7 else ()
        case PiiKind.CARD:
            return _cards(text) if clues.digits >= 12 else ()
        case PiiKind.IBAN:
            return _ibans(text) if clues.digits >= 8 else ()
        case PiiKind.NATIONAL_ID:
            return _national_ids(text) if clues.digits >= 6 else ()
        case PiiKind.IP:
            return _ips(text) if clues.digits >= 4 or ":" in text else ()
        case PiiKind.STREET_ADDRESS:
            return _street_addresses(text, clues.words) if clues.digits else ()
        case PiiKind.NAME:
            return _names(text, clues.words) if clues.words & _NAME_TRIGGERS else ()
        case _:
            assert_never(kind)


def luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


_EMAIL: Final = re.compile(
    r"(?<![\w.%+-])[\w.%+-]{1,64}@[^\W_](?:[\w-]*[^\W_])?(?:\.[^\W_](?:[\w-]*[^\W_])?)*\.[^\W\d_]{2,24}\b"
)


def _emails(text: str) -> Iterator[tuple[int, int]]:
    for match in _EMAIL.finditer(text):
        local = match.group().partition("@")[0]
        if not (local.startswith(".") or local.endswith(".") or ".." in local):
            yield match.span()


_CARD: Final = re.compile(r"(?<![\w-])\d(?:[ -]?\d){11,18}(?![\w-])")
_CARD_PREFIX: Final = re.compile(
    r"4|5[1-5]|2(?:22[1-9]|2[3-9]\d|[3-6]\d\d|7[01]\d|720)|3[47]|3(?:0[0-5]|[68])|35"
    r"|6(?:011|4[4-9]|5|2)|5(?:018|020|038|[6-8])|6(?:304|7)|2131|1800"
)
"""Visa, Mastercard (51-55 and 2221-2720), Amex, Diners, JCB (35, and 2131 or 1800 for 15 digits),
Discover, UnionPay, Maestro."""
_MAESTRO: Final = re.compile(r"5(?:018|020|038|[6-8])|6(?:304|759|76[1-3])")
"""Maestro alone issues 12-digit numbers."""
_IBAN_HEAD: Final = re.compile(r"[A-Za-z]{2}\d{2} $")


def _cards(text: str) -> Iterator[tuple[int, int]]:
    for match in _CARD.finditer(text):
        raw = match.group()
        digits = re.sub(r"\D", "", raw)
        groups = re.split(r"[ -]", raw)
        if len(set(re.findall(r"[ -]", raw))) > 1 or (len(groups) > 1 and min(map(len, groups[:-1])) < 4):
            continue
        if _IBAN_HEAD.search(text, max(0, match.start() - 5), match.start()):
            continue
        issuer = _CARD_PREFIX.match(digits) if len(digits) >= 13 else _MAESTRO.match(digits)
        if issuer and len(digits) <= 19 and luhn_ok(digits):
            yield match.span()


IBAN_LENGTHS: Final = {
    "AD": 24,
    "AE": 23,
    "AL": 28,
    "AT": 20,
    "AZ": 28,
    "BA": 20,
    "BE": 16,
    "BG": 22,
    "BH": 22,
    "BI": 27,
    "BR": 29,
    "BY": 28,
    "CH": 21,
    "CR": 22,
    "CY": 28,
    "CZ": 24,
    "DE": 22,
    "DJ": 27,
    "DK": 18,
    "DO": 28,
    "EE": 20,
    "EG": 29,
    "ES": 24,
    "FI": 18,
    "FK": 18,
    "FO": 18,
    "FR": 27,
    "GB": 22,
    "GE": 22,
    "GI": 23,
    "GL": 18,
    "GR": 27,
    "GT": 28,
    "HN": 28,
    "HR": 21,
    "HU": 28,
    "IE": 22,
    "IL": 23,
    "IQ": 23,
    "IS": 26,
    "IT": 27,
    "JO": 30,
    "KW": 30,
    "KZ": 20,
    "LB": 28,
    "LC": 32,
    "LI": 21,
    "LT": 20,
    "LU": 20,
    "LV": 21,
    "LY": 25,
    "MC": 27,
    "MD": 24,
    "ME": 22,
    "MK": 19,
    "MN": 20,
    "MR": 27,
    "MT": 31,
    "MU": 30,
    "NI": 28,
    "NL": 18,
    "NO": 15,
    "OM": 23,
    "PK": 24,
    "PL": 28,
    "PS": 29,
    "QA": 29,
    "RO": 24,
    "RS": 22,
    "RU": 33,
    "SA": 24,
    "SC": 31,
    "SD": 18,
    "SE": 24,
    "SI": 19,
    "SK": 24,
    "SM": 27,
    "SO": 23,
    "ST": 25,
    "SV": 28,
    "TL": 23,
    "TN": 24,
    "TR": 26,
    "UA": 29,
    "VA": 22,
    "VG": 24,
    "XK": 20,
    "YE": 30,
}
"""The SWIFT IBAN registry's length per country."""

_IBAN: Final = re.compile(r"(?<![^\W_])([A-Za-z]{2})(\d{2})((?: ?[A-Za-z0-9]){11,30})")


def iban_ok(compact: str) -> bool:
    compact = compact.upper()
    if IBAN_LENGTHS.get(compact[:2]) != len(compact) or not compact.isalnum():
        return False
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(char, 36)) for char in rearranged)) % 97 == 1


def _ibans(text: str) -> Iterator[tuple[int, int]]:
    for match in _IBAN.finditer(text):
        length = IBAN_LENGTHS.get(match.group(1).upper())
        if length is None:
            continue
        end = match.start(3)
        wanted = length - 4
        for char_end, char in enumerate(match.group(3), start=match.start(3) + 1):
            if char != " ":
                wanted -= 1
                if wanted == 0:
                    end = char_end
                    break
        if wanted or (end < len(text) and text[end].isalnum()):
            continue
        if iban_ok(text[match.start() : end].replace(" ", "")):
            yield match.start(), end


_ID_CUE: Final = re.compile(
    r"(?i:\b(?:ssn|social security|social insurance|national (?:id|identity|insurance)|national identification"
    r"|tax (?:id|identification)|burgerservicenummer|sofinummer|personnummer|identity (?:number|card)"
    r"|id (?:number|no)|steuer-?id|identifikationsnummer|codice fiscale|num[eé]ro de s[eé]curit[eé] sociale))"
    r"|\b(?:SIN|TIN|BSN|DNI|NIE|NIF|NIR|NINO)\b"
)
"""Acronyms only in capitals: "sin" and "nie" are ordinary Spanish and German words."""
_CUE_WINDOW: Final = 40

_SSN: Final = re.compile(r"(?<![\w-])(\d{3})([- ])(\d{2})\2(\d{4})(?![\w-])")
_SIN: Final = re.compile(r"(?<![\w-])\d{3}([- ])\d{3}\1\d{3}(?![\w-])")
_NINE: Final = re.compile(r"(?<![\w-])\d{9}(?![\w-])")
_NINO: Final = re.compile(r"(?i)(?<![^\W_])([A-Z]{2}) ?(\d{2}) ?(\d{2}) ?(\d{2}) ?([A-Z])(?![^\W_])")
_NINO_BAD_PREFIX: Final = frozenset({"BG", "GB", "KN", "NK", "NT", "TN", "ZZ"})
_CODICE_FISCALE: Final = re.compile(
    r"(?i)(?<![^\W_])[A-Z]{6}[\dLMNPQRSTUV]{2}[ABCDEHLMPRST][\dLMNPQRSTUV]{2}[A-Z][\dLMNPQRSTUV]{3}[A-Z](?![^\W_])"
)
_NIR: Final = re.compile(r"(?<![\w-])([12]) ?(\d{2}) ?(\d{2}) ?(\d{2}|2[AB]) ?(\d{3}) ?(\d{3}) ?(\d{2})(?![\w-])")
_PERSONNUMMER: Final = re.compile(r"(?<![\w-])(?:19|20)?(\d{2})(\d{2})(\d{2})[-+](\d{4})(?![\w-])")
_DNI: Final = re.compile(r"(?i)(?<![^\W_])([XYZ]-?)?(\d{7,8})-?([A-Z])(?![^\W_])")
_DNI_LETTERS: Final = "TRWAGMYFPDXBNJZSQVHLCKE"


_PHONE_HEAD: Final = re.compile(r"(?:\+\d{1,4}|\(\d{1,5}\)|\d) $")


def _in_phone(text: str, start: int) -> bool:
    """A 3-3-3 or 3-2-4 group after "+61 " or "(001) " is the tail of a phone number."""
    return _PHONE_HEAD.search(text, max(0, start - 8), start) is not None


def _cued(text: str, start: int) -> bool:
    return _ID_CUE.search(text, max(0, start - _CUE_WINDOW), start) is not None


def _ssn_ok(area: str, group: str, serial: str) -> bool:
    return area not in {"000", "666"} and area[0] != "9" and group != "00" and serial != "0000"


def _sin_ok(digits: str) -> bool:
    return digits[0] not in "08" and luhn_ok(digits)


_CF_ODD: Final = dict(
    zip(
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        (1, 0, 5, 7, 9, 13, 15, 17, 19, 21) * 2 + (2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23),
        strict=True,
    )
)
"""The Agenzia delle Entrate table for characters in odd positions. Even positions count 0 to 9 for
digits and 0 to 25 for letters."""


def _codice_fiscale_ok(code: str) -> bool:
    total = sum(
        _CF_ODD[char] if index % 2 == 0 else int(char) if char.isdigit() else ord(char) - ord("A")
        for index, char in enumerate(code[:15])
    )
    return chr(ord("A") + total % 26) == code[15]


def _nir_ok(match: re.Match[str]) -> bool:
    sex, year, month, department, commune, order, key = match.groups()
    if not (1 <= int(month) <= 12 or 20 <= int(month) <= 42 or int(month) >= 50):
        return False
    number = int(f"{sex}{year}{month}{department.upper().replace('2A', '19').replace('2B', '18')}{commune}{order}")
    if department.upper() == "2A":
        number -= 1_000_000
    elif department.upper() == "2B":
        number -= 2_000_000
    return 97 - number % 97 == int(key)


def _personnummer_ok(match: re.Match[str]) -> bool:
    year, month, day, tail = match.groups()
    return 1 <= int(month) <= 12 and 1 <= int(day) % 60 <= 31 and luhn_ok(f"{year}{month}{day}{tail}")


def _dni_ok(match: re.Match[str]) -> bool:
    prefix, digits, letter = match.groups()
    if prefix:
        if len(digits) != 7:
            return False
        digits = str("XYZ".index(prefix[0].upper())) + digits
    elif len(digits) != 8:
        return False
    return _DNI_LETTERS[int(digits) % 23] == letter.upper()


def _nino_ok(match: re.Match[str]) -> bool:
    prefix, suffix = match.group(1).upper(), match.group(5).upper()
    return (
        prefix[0] not in "DFIQUV" and prefix[1] not in "DFIOQUV" and prefix not in _NINO_BAD_PREFIX and suffix in "ABCD"
    )


_SIX_DIGITS: Final = re.compile(r"\d(?:[ -]?\d){5}")
"""Every national ID but the codice fiscale has six digits in a row, allowing single separators."""


def _national_ids(text: str) -> Iterator[tuple[int, int]]:
    """After a cue such as "SSN:" the shape is enough, because the writer said what the number is.
    Without one, a number must pass its issuer's range rules or check digit."""
    for match in _CODICE_FISCALE.finditer(text):
        if _codice_fiscale_ok(match.group().upper()):
            yield match.span()
    if _SIX_DIGITS.search(text) is None:
        return
    for match in _SSN.finditer(text):
        if _cued(text, match.start()) or (
            match.group(2) == "-"
            and not _in_phone(text, match.start())
            and _ssn_ok(match.group(1), match.group(3), match.group(4))
        ):
            yield match.span()
    for match in _SIN.finditer(text):
        if _cued(text, match.start()) or (
            not _in_phone(text, match.start()) and _sin_ok(re.sub(r"\D", "", match.group()))
        ):
            yield match.span()
    for match in _NINE.finditer(text):
        if _cued(text, match.start()):
            yield match.span()
    for match in _NINO.finditer(text):
        if _cued(text, match.start()) or _nino_ok(match):
            yield match.span()
    for match in _NIR.finditer(text):
        if _nir_ok(match):
            yield match.span()
    for match in _PERSONNUMMER.finditer(text):
        if _personnummer_ok(match):
            yield match.span()
    for match in _DNI.finditer(text):
        if _dni_ok(match):
            yield match.span()


_IPV4: Final = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w]|\.\d)")
_IPV6: Final = re.compile(r"(?<!\w)(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")


def _ips(text: str) -> Iterator[tuple[int, int]]:
    for match in _IPV4.finditer(text) if text.count(".") >= 3 else ():
        try:
            ipaddress.IPv4Address(match.group())
        except ValueError:
            continue
        yield match.span()
    for match in _IPV6.finditer(text) if text.count(":") >= 2 else ():
        candidate = match.group()
        groups = candidate.split(":")
        if len([group for group in groups if group]) < 4 and len(groups[0]) != 4:
            continue
        try:
            ipaddress.IPv6Address(candidate)
        except ValueError:
            continue
        yield match.span()


_PHONE: Final = re.compile(r"(?<![\w+(.,/-])(\+ ?)?[(\d][\d ().-]{5,24}\d(?: ?(?:x|ext\.?) ?\d{1,6})?(?![\w]|[.,/-]\d)")
_PHONE_CUE: Final = re.compile(r"(?i)\b(?:phone|tel|telephone|mobile|cell|call|fax|contact|whatsapp|text me|ring)\b")
_CURRENCY: Final = re.compile(r"[$€£¥₹]|\b(?:usd|eur|gbp|sek|chf|kr|dollars?|euros?|pounds?)\b", re.IGNORECASE)
_YEAR: Final = re.compile(r"(?:19|20)\d\d")
_EIN_CUE: Final = re.compile(r"\b(?:EIN|FEIN)(?::| is)? ?$")
"""A US employer number (12-3456789) names a business, not a person, and has a phone's shape."""


_DATE: Final = re.compile(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})|(\d{1,2})[-./](\d{1,2})[-./](\d{2}|\d{4})(?!\d)")
_DECIMAL: Final = re.compile(r"\d\.\d{2}(?:\s|$)")
"""An amount such as 10000.00 in a column of figures."""
_YEAR_SERIAL: Final = re.compile(r"(?:19|20)\d\d-\d{3,6}")
"""Ticket and invoice numbers such as 2023-001234."""


def _has_date(body: str) -> bool:
    for match in _DATE.finditer(body):
        first, second = (match.group(2), match.group(3)) if match.group(1) else (match.group(4), match.group(5))
        if 1 <= int(first) <= 31 and 1 <= int(second) <= 31 and min(int(first), int(second)) <= 12:
            return True
    return False


def _phone_ok(candidate: str, text: str, start: int, end: int) -> bool:
    body = re.split(r" ?(?:x|ext\.?) ?(?=\d+$)", candidate)[0]
    digits = re.sub(r"\D", "", body)
    if body.count("(") != body.count(")") or not 7 <= len(digits) <= 15:
        return False
    after = end + 1 if end < len(text) and text[end] == " " else end
    if _CURRENCY.search(text, max(0, start - 4), start) or _CURRENCY.match(text, after):
        return False
    if body.startswith("+"):
        return len(digits) >= 8
    cued = _PHONE_CUE.search(text, max(0, start - 30), start) is not None
    if _EIN_CUE.search(text, max(0, start - 12), start):
        return False
    groups = [group for group in re.split(r"[ ().-]+", body) if group]
    if len(groups) == 1:
        return len(digits) in (10, 11) and digits[:2] != "00" and digits[0] == "0" or 8 <= len(digits) <= 11 and cued
    if len(digits) <= 8 and not cued:
        return False
    if any(len(group) > 5 for group in groups[:-1]) or _has_date(body) or _YEAR_SERIAL.fullmatch(body):
        return False
    if _DECIMAL.search(body):
        return False
    if len(groups) == 2 and all(_YEAR.fullmatch(group) for group in groups):
        return False
    if [len(group) for group in groups] == [3, 2, 4]:
        return False
    separators = set(re.findall(r"[ ().-]", body))
    if separators == {"."}:
        return len(groups) == 3 and len(groups[0]) >= 2 or cued
    if separators == {" "} and len(groups[0]) <= 3 and all(len(group) == 3 for group in groups[1:]):
        return len(digits) >= 9 or cued
    return True


def _phones(text: str) -> Iterator[tuple[int, int]]:
    for match in _PHONE.finditer(text):
        start, end = match.span()
        candidate = match.group().rstrip(" .-")
        if candidate.startswith("(") is False and candidate.endswith(")"):
            candidate = candidate[:-1]
        end = start + len(candidate)
        if _phone_ok(candidate, text, start, end):
            yield start, end


_UPPER: Final = "A-ZÀ-ÖØ-ÞĀ-Ž"
_LOWER: Final = "a-zß-öø-ÿā-ž"
_WORD: Final = rf"[{_UPPER}][{_LOWER}'’-]*"
_STREET_SUFFIXES: Final = (
    "street|st|avenue|ave|av|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|courts|place|pl|square|sq|terrace"
    "|way|crescent|close|parkway|pkwy|highway|hwy|circle|cir|trail|trl|alley|row|mews|grove|gardens|hill|hills"
    "|walk|crossing|crossroad|crossway|crest|corner|corners|common|commons|courtyard|burg|burgs|cove|creek"
    "|estate|estates|expressway|extension|falls|ferry|field|fields|flat|flats|ford|forest|forge|fork|forks"
    "|freeway|garden|gateway|glen|glens|green|harbor|harbour|haven|heights|hollow|inlet|island|isle|junction"
    "|knoll|knolls|lake|lakes|landing|lodge|loop|mall|manor|meadow|meadows|mill|mills|mission|motorway|mount"
    "|mountain|orchard|oval|overpass|park|pass|passage|path|pike|pine|pines|plaza|point|points|port|ports"
    "|prairie|ranch|rapid|rapids|ridge|river|route|run|shoal|shore|shores|skyway|spring|springs|spur|station"
    "|stravenue|stream|summit|throughway|trace|track|trafficway|tunnel|turnpike|underpass|union|valley|byway"
    "|viaduct|view|views|village|ville|vista|wall|wells|wharf|bypass|causeway|cliff|cliffs|club|curve|dale"
    "|dam|divide|fort|hbr|ldg|plains|radial|ramp|rest|vlg|neck|brook|brooks|keys|lock|locks|parade|rise"
    "|embankment|approach|broadway|esplanade|quay|wynd|yard|drove|end|fold|gate|hey|meander|nook|side|marg|nagar|chowk"
)
_FRENCH_WORDS: Final = "rue|avenue|av|boulevard|bd|chemin|allée|allee|place|impasse|quai|route|cours"
_STREET_EN: Final = re.compile(
    rf"(?<![\w/-])\d{{1,5}}[A-Za-z]? (?!The )(?:{_WORD} ){{1,3}}(?i:(?:{_STREET_SUFFIXES})s?)\b"
    rf"|(?<![\w-])\d{{1,5}},? (?i:{_FRENCH_WORDS})\.?"
    rf" (?:(?:de|du|des|la|le|l'|d')\s?)*{_WORD}"
)
_COMPOUND: Final = (
    r"(?:stra(?:ß|ss)e|str\.|weg|gasse|allee|platz|ring|damm|ufer|straat|laan|plein|gracht|singel|kade|dijk"
    r"|steeg|hof|gatan|v[äa]gen|gr[äa]nd|torget|backen|stigen)"
)
_COMPOUND_AFTER_NUMBER: Final = (
    r"(?:stra(?:ß|ss)e|weg|gasse|allee|platz|straat|laan|plein|gracht|singel|kade|dijk|steeg|gatan|v[äa]gen)"
)
"""Without "ring" and "hof", which end too many ordinary words to follow a bare number."""
_COMPOUND_NAME: Final = rf"(?:[{_UPPER}][{_LOWER}]*-)*[{_UPPER}][{_LOWER}-]*{_COMPOUND}"
_STREET_COMPOUND: Final = re.compile(
    rf"(?<![\w-]){_COMPOUND_NAME} \d{{1,4}}[a-zA-Z]?(?![\w-])"
    rf"|(?<![\w-])\d{{1,4}}[a-zA-Z]? (?:[{_UPPER}][{_LOWER}]*-)*[{_UPPER}][{_LOWER}-]*{_COMPOUND_AFTER_NUMBER}(?![\w-])"
)
_LEADING: Final = (
    r"(?:Calle|C/|Avenida|Avda\.?|Paseo|Plaza|Carrer|Camino|Ronda|Via|Viale|Piazza|Piazzale|Corso|Largo|Vicolo"
    r"|Strada|Rua|Alameda|Rambla|Traves[ií]a|Ca[ñn]ada|Glorieta|Callej[oó]n|Pasaje|Cuesta|Urbanizaci[oó]n)"
    rf" (?:(?:de|del|de la|della|delle|dei|di|degli) )?{_WORD}(?: {_WORD}){{0,2}}"
)
"""Spanish, Catalan, Italian and Portuguese street words that come before the name."""
_STREET_LEADING: Final = re.compile(
    rf"(?<![\w-]){_LEADING},? (?:n\.?º? ?)?\d{{1,4}}(?![\w-])|(?<![\w/-])\d{{1,5}},? {_LEADING}"
)


_SUFFIX_WORDS: Final = frozenset(_STREET_SUFFIXES.split("|")) | frozenset(_FRENCH_WORDS.split("|"))
_COMPOUND_ENDINGS: Final = (
    "straße",
    "strasse",
    "str",
    "weg",
    "gasse",
    "allee",
    "platz",
    "ring",
    "damm",
    "ufer",
    "straat",
    "laan",
    "plein",
    "gracht",
    "singel",
    "kade",
    "dijk",
    "steeg",
    "hof",
    "gatan",
    "vägen",
    "vagen",
    "gränd",
    "grand",
    "torget",
    "backen",
    "stigen",
)
_LEADING_WORDS: Final = frozenset(
    [
        "calle",
        "c",
        "avenida",
        "avda",
        "paseo",
        "plaza",
        "carrer",
        "camino",
        "ronda",
        "via",
        "viale",
        "piazza",
        "piazzale",
        "corso",
        "largo",
        "vicolo",
        "strada",
        "rua",
        "alameda",
        "rambla",
        "travesía",
        "travesia",
        "cañada",
        "canada",
        "glorieta",
        "callejón",
        "callejon",
        "pasaje",
        "cuesta",
        "urbanización",
        "urbanizacion",
    ]
)
"""The words each street pattern cannot match without, casefolded, so most sentences skip the regexes."""


_NUMBER_START: Final = re.compile(r"(?<!\d)\d")


def _street_addresses(text: str, words: frozenset[str]) -> Iterator[tuple[int, int]]:
    if words & _SUFFIX_WORDS or {word[:-1] for word in words if word.endswith("s")} & _SUFFIX_WORDS:
        # Every _STREET_EN alternative starts with the house number, so it is tried only where one starts.
        for number in _NUMBER_START.finditer(text):
            if match := _STREET_EN.match(text, number.start()):
                yield match.span()
    patterns: list[re.Pattern[str]] = []
    if any(word.endswith(_COMPOUND_ENDINGS) for word in words):
        patterns.append(_STREET_COMPOUND)
    if words & _LEADING_WORDS:
        patterns.append(_STREET_LEADING)
    for pattern in patterns:
        for match in pattern.finditer(text):
            yield match.span()


_NAME_RUN: Final = rf"({_WORD}(?: (?:(?:van der|van den|van|von|de la|de|del|della|di|da|du|le|bin|al) )?(?:[{_UPPER}]\.|{_WORD})){{0,3}})"
_NOT_NAMES: Final = frozenset(
    [
        "The",
        "A",
        "An",
        "This",
        "That",
        "There",
        "Here",
        "Team",
        "All",
        "Everyone",
        "Sir",
        "Madam",
        "Hiring",
        "Customer",
        "Valued",
        "Client",
        "Friend",
        "Friends",
        "Interested",
        "Looking",
        "Writing",
        "Reaching",
        "Not",
        "Sorry",
        "Happy",
        "Glad",
        "Available",
        "Currently",
        "Also",
        "Just",
        "Still",
        "Very",
        "So",
        "Sure",
        "Ready",
        "Back",
        "New",
        "Good",
        "Great",
        "I",
        "We",
        "You",
        "He",
        "She",
        "It",
        "They",
        "Our",
        "Your",
        "My",
        "His",
        "Her",
        "Their",
        "In",
        "On",
        "At",
        "For",
        "With",
        "From",
        "To",
        "Of",
        "And",
        "Or",
        "But",
        "If",
        "As",
        "By",
        "Please",
        "Thanks",
        "Thank",
        "Hi",
        "Hello",
        "Dear",
        "Yes",
        "No",
        "Ok",
        "Okay",
        "Re",
        "Fwd",
        "Subject",
        "Date",
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "Prof",
        "Name",
        "Address",
        "Phone",
        "Email",
        "Applicant",
        "Patient",
        "Borrower",
        "Lender",
        "Account",
        "Bank",
        "Company",
        "Inc",
        "Ltd",
        "LLC",
        "Department",
        "Manager",
        "Director",
        "Officer",
        "Recruiter",
        "Engineer",
        "Developer",
    ]
)


def _cue(phrases: str, separator: str, *, ignore_case: bool) -> tuple[frozenset[str], re.Pattern[str]]:
    """A name pattern: one of the phrases, the separator, then a run of capitalised words. Also returns
    each phrase's first word, so a sentence without any of them skips the pattern."""
    alternatives = phrases.split("|")
    cue = "|".join(re.escape(phrase) for phrase in alternatives)
    cue = f"(?i:{cue})" if ignore_case else f"(?:{cue})"
    triggers = frozenset(_LETTER_RUN.findall(phrase.casefold())[0] for phrase in alternatives)
    return triggers, re.compile(rf"\b{cue}{separator}{_NAME_RUN}")


_NAMES: Final = (
    _cue(
        "my name is|my name's|name is|i am|i'm|i’m|this is|call me|named|signed by|me llamo|soy|je m'appelle"
        "|je suis|ich bin|ich heiße|mi chiamo|sono|jag heter|jag är|ik ben|ik heet",
        " ",
        ignore_case=True,
    ),
    _cue(
        "name|full name|patient|patient name|client|client name|customer|customer name|employee|employee name"
        "|applicant|applicant name|borrower|borrower name|lender|insured|beneficiary|account holder|cardholder"
        "|contact|contact person|signed|signature|attn|attention|prepared by|author|witness|guarantor|tenant"
        "|landlord|physician|doctor|nombre|nom|nome|namn|naam",
        r"\s?: ?",
        ignore_case=True,
    ),
    _cue(
        "Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Sir|Madam|Herr|Frau|Mme|Mlle|Sr|Sra|Srta|Sig|Sig.ra|Dott|Dhr|Mevr|Fru",
        r"\.? ",
        ignore_case=False,
    ),
    _cue(
        "Dear|Hi|Hello|Hey|Hola|Bonjour|Hallo|Ciao|Hej|Beste|Estimado|Estimada|Querido|Querida|Cher|Chère|Liebe"
        "|Lieber|Sehr geehrte|Sehr geehrter|Gentile|Caro|Cara|Bästa",
        " ",
        ignore_case=False,
    ),
    _cue(
        "sincerely|regards|best regards|kind regards|warm regards|best wishes|yours truly|cheers|cordialement"
        "|saludos|atentamente|mit freundlichen grüßen|distinti saluti|cordiali saluti|vänliga hälsningar"
        "|met vriendelijke groet",
        ",? ",
        ignore_case=True,
    ),
)
_NAME_TRIGGERS: Final = frozenset[str]().union(*(triggers for triggers, _ in _NAMES))
_TRIGGER_WORDS: Final = {
    word: re.compile(rf"(?i)(?<![^\W\d_]){re.escape(word)}(?![^\W\d_])") for word in _NAME_TRIGGERS
}


def _names(text: str, words: frozenset[str]) -> Iterator[tuple[int, int]]:
    """Each cue pattern is tried only where one of its first words occurs, not scanned across the text.
    Spans come out in text order, so overlap resolution never depends on set iteration order."""
    yield from sorted(set(_name_spans(text, words)))


def _name_spans(text: str, words: frozenset[str]) -> Iterator[tuple[int, int]]:
    lowered = text.lower()
    for word in words & _NAME_TRIGGERS:
        if len(lowered) == len(text):
            starts = _occurrences(lowered, word)
        else:
            starts = [token.start() for token in _TRIGGER_WORDS[word].finditer(text)]
        for start in starts:
            for triggers, pattern in _NAMES:
                match = pattern.match(text, start) if word in triggers else None
                if match is None:
                    continue
                run = match.group(1).split(" ")
                while run and run[-1].rstrip(".") in _NOT_NAMES:
                    run.pop()
                if not run or run[0].rstrip(".") in _NOT_NAMES or len(run[0]) < 2:
                    continue
                yield match.start(1), match.start(1) + len(" ".join(run).rstrip(".,"))


def _occurrences(lowered: str, word: str) -> list[int]:
    """Where word starts as a whole word. str.find is far cheaper than a case-insensitive regex."""
    starts: list[int] = []
    position = lowered.find(word)
    while position != -1:
        end = position + len(word)
        before = lowered[position - 1] if position else " "
        after = lowered[end] if end < len(lowered) else " "
        if not (before.isalpha() or after.isalpha()):
            starts.append(position)
        position = lowered.find(word, position + 1)
    return starts
