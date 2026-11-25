import pytest

from spamdetector.preprocessing import clean_text


def test_lowercases_and_strips_punctuation():
    assert clean_text("HELLO,   World!!") == "hello world"


def test_numbers_become_token():
    assert clean_text("Call 09061701461 now") == "call numtoken now"


def test_urls_become_token():
    assert "urltoken" in clean_text("Visit http://bit.ly/abc or www.example.com today")
    assert "http" not in clean_text("Visit http://bit.ly/abc")


def test_currency_symbols_become_token():
    assert "currencytoken" in clean_text("You won £500 and ₹1000")


def test_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""


@pytest.mark.parametrize("value", [123, 4.5])
def test_non_string_input_does_not_crash(value):
    assert clean_text(value) == "numtoken" or "numtoken" in clean_text(value)
