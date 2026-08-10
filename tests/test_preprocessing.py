from src.preprocessing import normalize_text


def test_normalize_text_replaces_structured_values() -> None:
    result = normalize_text("WIN £100 at https://example.com or call 12345")
    assert "currency" in result
    assert "url" in result
    assert "phone_number" in result


def test_normalize_text_handles_whitespace() -> None:
    assert normalize_text("  Hello   WORLD  ") == "hello world"
