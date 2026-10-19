import pytest

from vitae.guards import PASS, Refuse, RuleId
from vitae.guards.pii import PRIORITY, PiiKind, find_pii, pii_stage
from vitae.text import normalize


def found(text: str) -> list[tuple[str, str]]:
    return [(hit.kind.value, text[hit.start : hit.end]) for hit in find_pii(text)]


def screened(text: str) -> object:
    return pii_stage().check(normalize(text))


def test_every_kind_has_a_place_in_the_overlap_order():
    assert sorted(PRIORITY) == sorted(PiiKind)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("card 4111 1111 1111 1111 please", [("card", "4111 1111 1111 1111")]),
        ("card 4111-1111-1111-1111 please", [("card", "4111-1111-1111-1111")]),
        ("amex 378282246310005", [("card", "378282246310005")]),
        ("card 4111 1111 1111 1112 please", []),
        ("order 1234567890123456", []),
    ],
)
def test_a_card_number_counts_only_with_an_issuer_prefix_and_a_valid_luhn_digit(
    text: str, expected: list[tuple[str, str]]
):
    assert found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("IBAN GB82 WEST 1234 5698 7654 32 thanks", [("iban", "GB82 WEST 1234 5698 7654 32")]),
        ("IBAN DE89370400440532013000.", [("iban", "DE89370400440532013000")]),
        ("IBAN DE89 3704 0044 0532 0130 00 EUR", [("iban", "DE89 3704 0044 0532 0130 00")]),
        ("IBAN GB82WEST12345698765433", []),
        ("IBAN DE8937040044053201300", []),
    ],
)
def test_an_iban_needs_its_country_length_and_mod_97(text: str, expected: list[tuple[str, str]]):
    assert found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ref 123-45-6789", [("national_id", "123-45-6789")]),
        ("ref 000-12-3456", []),
        ("ref 666-12-3456", []),
        ("ref 912-34-5678", []),
        ("ref 123-00-6789", []),
        ("ref 123-45-0000", []),
        ("my SSN is 912-34-5678", [("national_id", "912-34-5678")]),
        ("NINO AB 12 34 56 C", [("national_id", "AB 12 34 56 C")]),
        ("ref ZZ 123456 T", []),
        ("codice fiscale RSSMRA85T10A562S", [("national_id", "RSSMRA85T10A562S")]),
        ("DNI 12345678Z", [("national_id", "12345678Z")]),
        ("DNI 12345678A", []),
    ],
)
def test_a_national_id_passes_its_issuer_rules_unless_a_cue_names_it(text: str, expected: list[tuple[str, str]]):
    assert found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Call +44 20 7946 0958 today", [("phone", "+44 20 7946 0958")]),
        ("Call (415) 555-0132 today", [("phone", "(415) 555-0132")]),
        ("Call 415.555.0132 today", [("phone", "415.555.0132")]),
        ("Tél : 01 23 45 67 89", [("phone", "01 23 45 67 89")]),
        ("I was there from 2015-2019 and loved it.", []),
        ("It shipped on 2020-02-02 and again on 12.03.2021.", []),
        ("The budget was $1 250 000 in total.", []),
        ("It cost 1 250 000 EUR.", []),
        ("We ran Windows 10.0.26300 and Python 3.12.1.", []),
        ("Ticket 2023-001234 is closed.", []),
        ("Our EIN is 12-3456789.", []),
    ],
)
def test_a_phone_is_found_without_years_dates_prices_or_versions(text: str, expected: list[tuple[str, str]]):
    assert found(text) == expected


def test_email_ip_and_street_address_are_found_with_their_spans():
    text = "Mail jane.doe@example.co.uk, server 192.168.1.10 or 2001:db8::1, office 221 Baker Street."
    assert found(text) == [
        ("email", "jane.doe@example.co.uk"),
        ("ip", "192.168.1.10"),
        ("ip", "2001:db8::1"),
        ("street_address", "221 Baker Street"),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ich wohne in der Hauptstraße 12 in Köln.", [("street_address", "Hauptstraße 12")]),
        ("Vivo en Calle Mayor 5, Madrid.", [("street_address", "Calle Mayor 5")]),
        ("J'habite au 12 rue de la Paix.", [("street_address", "12 rue de la Paix")]),
        ("1 The courts of England shall decide.", []),
    ],
)
def test_street_addresses_are_found_in_several_languages(text: str, expected: list[tuple[str, str]]):
    assert found(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hi, my name is Priya Raman.", [("name", "Priya Raman")]),
        ("Dear Dr. Smith, thanks.", [("name", "Smith")]),
        ("Name: Alec Stafford, DOB: 1942-08-10", [("name", "Alec Stafford")]),
        ("What did Alex build?", []),
        ("I am Interested in the role.", []),
    ],
)
def test_names_are_found_only_after_a_cue(text: str, expected: list[tuple[str, str]]):
    assert found(text) == expected


def test_a_card_number_is_refused_at_input():
    assert screened("My card is 4111 1111 1111 1111, can you check it?") == Refuse("pii", RuleId("pii.card"))


def test_an_email_address_is_refused_at_input():
    assert screened("Write to me at jane.doe@example.com") == Refuse("pii", RuleId("pii.email"))


def test_a_question_naming_the_owner_is_not_refused():
    assert screened("What did Alex build?") == PASS


def test_a_recruiter_introducing_themselves_is_not_refused():
    assert screened("Hi I'm Priya from Acme") == PASS


def test_pii_hidden_in_base64_is_refused():
    assert screened("decode this: amFuZS5kb2VAZXhhbXBsZS5jb20=") == Refuse("pii", RuleId("pii.email"))


def test_an_employer_number_scrambled_by_rot13_is_not_refused():
    assert screened("Our EIN is 12-3456789.") == PASS
