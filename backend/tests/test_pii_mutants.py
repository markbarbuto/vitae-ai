import pytest

from vitae.guards import InputStages, Refuse, RuleId
from vitae.guards.pii import _cue, find_pii, iban_ok, luhn_ok, pii_stage  # pyright: ignore[reportPrivateUsage]
from vitae.text import normalize


def found(text: str) -> list[tuple[str, str]]:
    return [(hit.kind.value, text[hit.start : hit.end]) for hit in find_pii(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("card 4222222222222", [("card", "4222222222222")]),
        ("card 4111111111111111110", [("card", "4111111111111111110")]),
        ("card 5555 5555 5555 4444", [("card", "5555 5555 5555 4444")]),
        ("ref 501823766545.", [("card", "501823766545")]),
        ("ref 411111111117", []),
        ("Call 2131084024224", [("card", "2131084024224")]),
        ("6526756486365187447.", [("card", "6526756486365187447")]),
        ("411 1111111111111", []),
        ("4111 11 1111111111", []),
        ("4111 1111 11 111111", []),
        ("4111-1111 1111 1111 or 4111 1111 1111 1111", [("card", "4111 1111 1111 1111")]),
        ("AB12 4111 1111 1111 1111", []),
        ("AB12 4111 1111 1111 1111 and 4111 1111 1111 1111", [("card", "4111 1111 1111 1111")]),
        ("IBAN AB12 6259 384961 35381 kr", []),
    ],
)
def test_a_card_needs_one_kind_of_separator_full_groups_an_issuer_and_luhn(text: str, expected: list[tuple[str, str]]):
    assert found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("IBAN GB11 WEST ABCD EFGH 1234 56", [("iban", "GB11 WEST ABCD EFGH 1234 56")]),
        (
            "ZZ12 3456 7890 1234, GB82 WEST 1234 5698 7654 32",
            [("phone", "3456 7890 1234"), ("iban", "GB82 WEST 1234 5698 7654 32")],
        ),
        (
            "GB82 WEST 1234 5698 76, GB82 WEST 1234 5698 7654 32",
            [("phone", "1234 5698 76"), ("iban", "GB82 WEST 1234 5698 7654 32")],
        ),
        ("IBAN DE89370400440532013000X", []),
    ],
)
def test_an_iban_is_cut_at_its_country_length_and_not_glued_to_more_text(text: str, expected: list[tuple[str, str]]):
    assert found(text) == expected


def test_iban_ok_rejects_a_valid_checksum_at_the_wrong_length():
    assert iban_ok("GB82WEST12345698765432") is True
    assert iban_ok("GB08WEST12345698765") is False


def test_luhn_ok_doubles_every_second_digit_from_the_right():
    assert luhn_ok("5555555555554444") is True
    assert luhn_ok("5555555555554445") is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SSN 912-34-5678", [("national_id", "912-34-5678")]),
        ("ref 912-34-5678 is my SSN", []),
        ("ref XA 12 34 56 A", [("national_id", "XA 12 34 56 A")]),
        ("ref AX 12 34 56 A", [("national_id", "AX 12 34 56 A")]),
        ("tel DA 58 23 67 A today.", []),
        ("IBAN AB12 AD 28 69 69 B", []),
        ("costs $CE 29 92 88 A dollars", [("national_id", "CE 29 92 88 A")]),
        ("SIN +969371648 and 10.0.0.1", [("national_id", "969371648"), ("ip", "10.0.0.1")]),
        ("Call 777-469-271 kr", [("national_id", "777-469-271")]),
        ("441-095-752 kr", []),
        ("€971 847 406 and 10.0.0.1", [("ip", "10.0.0.1")]),
        ("Our EIN is 894-563-899", []),
        ("+61 046 454 286", [("phone", "+61 046 454 286")]),
        ("Call +61 046-454-286", [("phone", "+61 046-454-286")]),
        ("(001) 356-13-7255", [("phone", "(001) 356-13-7255")]),
        ("My SSN is 0055402343 today.", []),
        ("My SSN is 37117-175415-8 and 10.0.0.1", [("ip", "10.0.0.1")]),
        ("costs $1 36 42 2B 727 456 22 and 10.0.0.1", [("national_id", "1 36 42 2B 727 456 22"), ("ip", "10.0.0.1")]),
        ("ref 237422A54109884 EUR", [("national_id", "237422A54109884")]),
        ("286207543555765 and 10.0.0.1", [("national_id", "286207543555765"), ("ip", "10.0.0.1")]),
        ("ref 1 96 01 75 799 751 41 today.", [("national_id", "1 96 01 75 799 751 41")]),
        ("SIN 123501361507748 today.", [("national_id", "123501361507748")]),
        ("+61 184121335491944 dollars", [("national_id", "184121335491944")]),
        ("costs $208432B92943178 and 10.0.0.1", [("ip", "10.0.0.1")]),
        ("Contact 277132A72739627 kr", []),
        ("19990101+9803 EUR", [("national_id", "19990101+9803")]),
        ("IBAN AB12 401261+8981.", [("national_id", "401261+8981")]),
        ("ref 330691+8800 kr", [("national_id", "330691+8800")]),
        ("My SSN is 19851391+1993 and 10.0.0.1", [("ip", "10.0.0.1")]),
        ("€Z6041718Z kr", [("national_id", "Z6041718Z")]),
        ("€1914257J dollars", []),
        ("My SSN is Y-99402576M EUR", []),
    ],
)
def test_a_national_id_passes_its_issuer_rules_or_follows_a_cue(text: str, expected: list[tuple[str, str]]):
    assert found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("host 1.2.3.4", [("ip", "1.2.3.4")]),
        ("999.1.1.1 or 10.0.0.1", [("ip", "10.0.0.1")]),
        ("1:2:3 or 2001:db8::1", [("ip", "2001:db8::1")]),
        ("1::2::3:4 or 2001:db8::1", [("ip", "2001:db8::1")]),
        ("+61 fe80::1.", [("ip", "fe80::1")]),
        ("phone: abcd:ef:: dollars", [("ip", "abcd:ef::")]),
        ("(001) db8:85a3::8a2e:370.", [("ip", "db8:85a3::8a2e:370")]),
        ("tel 300.19.275.157 and 10.0.0.1", [("phone", "300.19.275.157"), ("ip", "10.0.0.1")]),
    ],
)
def test_an_ip_address_must_parse(text: str, expected: list[tuple[str, str]]):
    assert found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail .jane@example.com", []),
        ("mail jane.@example.com", []),
        ("mail ja..ne@example.com", []),
        ("€jane.@example.com and 10.0.0.1", [("ip", "10.0.0.1")]),
    ],
)
def test_an_email_local_part_cannot_start_end_or_double_a_dot(text: str, expected: list[tuple[str, str]]):
    assert found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Call 555-0134", [("phone", "555-0134")]),
        ("on 0207299991", [("phone", "0207299991")]),
        ("Call 051552611330,", []),
        ("Call 35142341805.", [("phone", "35142341805")]),
        ("Total 0412345678€", []),
        ("Fee USD: 0412345678", [("phone", "0412345678")]),
        ("Room 1234 5678, call me", []),
        ("Call me any time; desk is 1234 5678", [("phone", "1234 5678")]),
        ("Call me any time; desk line is 1234 5678", []),
        ("EIN 12-3456789", []),
        ("Call 1999 2000 2001", [("phone", "1999 2000 2001")]),
        ("Call 2015 2019", []),
        ("Call 2021-12-31", []),
        ("Call 2021-12-32", [("phone", "2021-12-32")]),
        ("Call 1-28-2021", []),
        ("Call 19-1-2015", []),
        ("tel 31-10-2021,", []),
        ("(001) 21-12-23", []),
        ("(001) 2021-23-13 today.", [("phone", "(001) 2021-23-13")]),
        ("phone: 32.8.2021,", [("phone", "32.8.2021")]),
        ("phone: 2015.14.21.", []),
        ("ref 1289.0441.818.891,", []),
        ("(001) 085.34048.9474", [("phone", "(001) 085.34048.9474")]),
        ("IBAN AB12 70.717.490689.", [("phone", "70.717.490689")]),
        ("IBAN AB12 422 628 935 today.", [("phone", "422 628 935")]),
        ("ref 1800 15748 385 2,", [("phone", "1800 15748 385 2")]),
        ("on 799-63784-63-1", [("phone", "799-63784-63-1")]),
        ("Contact 628961 0114-9660 dollars", []),
        ("(001) 381232+2091 kr", [("phone", "(001) 381232")]),
        ("tel 19700191+6472x", [("phone", "19700191")]),
        ("SIN + (524)  510-00.", [("phone", "+ (524)  510-00")]),
        ("phone: (343) 7 80ext3.", []),
        ("on 0536 09x", []),
        ("+61 06433427Mx", []),
    ],
)
def test_a_phone_is_told_apart_from_dates_years_prices_and_serials(text: str, expected: list[tuple[str, str]]):
    assert found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I live at 12 Water Lanes.", [("street_address", "12 Water Lanes")]),
        ("Hi, my name is Bo Chen.", [("name", "Bo Chen")]),
        ("Dear Team Alpha,", []),
        ("My name is Priya A.", [("name", "Priya")]),
        ("Dear Priya Team,", [("name", "Priya")]),
        ("Dear Priya Raman", [("name", "Priya Raman")]),
        ("Hi all. Hi Priya. Hi", [("name", "Priya")]),
        ("İlkay: my name is Priya Raman.", [("name", "Priya Raman")]),
    ],
)
def test_streets_and_names_are_found_after_their_cue_words(text: str, expected: list[tuple[str, str]]):
    assert found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Name: Priya-GB82WEST12345698765432", [("name", "Priya-"), ("iban", "GB82WEST12345698765432")]),
        ("abcd:ef::+61 2 9876 5432", [("ip", "abcd:ef::"), ("phone", "+61 2 9876 5432")]),
    ],
)
def test_hits_that_touch_without_overlapping_are_both_kept(text: str, expected: list[tuple[str, str]]):
    assert found(text) == expected


def test_the_pii_stage_is_selected_for_pii_refusals():
    stage = pii_stage()
    assert InputStages((stage,)).for_reason("pii").stages == (stage,)
    assert stage.check(normalize("My card is 5555 5555 5555 4444")) == Refuse("pii", RuleId("pii.card"))


def test_a_name_cue_is_built_from_its_phrases_its_first_words_and_its_case_rule():
    # The cues are built once at import; calling the builder is the only way a mutant of it can run.
    triggers, pattern = _cue("my name is|i am|call me", " ", ignore_case=True)
    assert triggers == frozenset({"my", "i", "call"})
    match = pattern.match("I AM Priya Raman")
    assert match is not None and match.group(1) == "Priya Raman"
    triggers, pattern = _cue("Dr|Prof", r"\.? ", ignore_case=False)
    assert triggers == frozenset({"dr", "prof"})
    assert pattern.match("dr. Smith") is None
    match = pattern.match("Dr. Smith")
    assert match is not None and match.group(1) == "Smith"
