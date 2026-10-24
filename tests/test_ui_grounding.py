"""Unit tests for UI citation grounding chip rendering in AuralQ Studio."""

import html
import re
import pytest

from app.minecraft_theme import render_grounded_answer_html
from src.agent.schemas import CitationMatch, GroundingReport


class DummyTurn:
    """Mock TurnRecord for testing render_grounded_answer_html."""
    def __init__(self, verified_tokens=None, hallucinated_tokens=None):
        verified = [
            CitationMatch(raw_token=tok, parsed_value=float(tok) if tok.replace(".", "", 1).replace("-", "", 1).isdigit() else 0.0)
            for tok in (verified_tokens or [])
        ]
        self.grounding_report = GroundingReport(
            is_fully_grounded=True,
            verified_citations=verified,
            hallucinated_numbers=hallucinated_tokens or [],
        )


def strip_html_tags(text: str) -> str:
    """Helper to strip all HTML tags from a rendered string."""
    return re.sub(r"<[^>]+>", "", text)


def test_no_nesting_and_no_split_digits():
    """Both 0.0 and 0.0001 present in the verified list -> separate non-nested chips."""
    content = "The noise floor was 0.0 and the minimum RMS was 0.0001 across frames."
    turn = DummyTurn(verified_tokens=["0.0", "0.0001"])
    rendered = render_grounded_answer_html(content, turn)

    # Acceptance checks
    assert "<span class='citation-verified'>0.0</span>" in rendered
    assert "<span class='citation-verified'>0.0001</span>" in rendered
    assert "[ [" not in rendered
    assert "<span class='citation-verified'><span" not in rendered
    assert "001" not in re.sub(r"<span[^>]*>0\.0001</span>", "", rendered), "0.0001 was split into 0.0 and 001!"

    # Plain text stripped equals original
    plain = html.unescape(strip_html_tags(rendered))
    assert plain == content


def test_value_at_end_of_sentence():
    """Number followed by a period at the end of a sentence -> period is outside the chip."""
    content = "The measured RMS energy was 0.0001."
    turn = DummyTurn(verified_tokens=["0.0001"])
    rendered = render_grounded_answer_html(content, turn)

    assert "<span class='citation-verified'>0.0001</span>." in rendered
    assert "<span class='citation-verified'>0.0001.</span>" not in rendered

    plain = html.unescape(strip_html_tags(rendered))
    assert plain == content


def test_value_inside_longer_unverified_number():
    """Number inside a larger unverified number (10.05 when only 0.05 verified) must NOT match."""
    content = "The duration was 10.05 seconds and total samples was 1005."
    turn = DummyTurn(verified_tokens=["0.05", "05"])
    rendered = render_grounded_answer_html(content, turn)

    # Neither 10.05 nor 1005 should be matched or wrapped
    assert "<span class='citation-verified'>" not in rendered
    plain = html.unescape(strip_html_tags(rendered))
    assert plain == content


def test_empty_verified_list():
    """Empty verified list -> text returned unmodified (except safe HTML escaping)."""
    content = "The audio signal is clear and tonal."
    turn = DummyTurn(verified_tokens=[])
    rendered = render_grounded_answer_html(content, turn)

    assert "<span" not in rendered
    assert rendered == content


def test_text_containing_html_entities():
    """Text with < and & characters -> escaped correctly without double-escaping chips."""
    content = "Check: dynamic range < 0.0001 & centroid > 0.0 Hz."
    turn = DummyTurn(verified_tokens=["0.0001", "0.0"])
    rendered = render_grounded_answer_html(content, turn)

    assert "&lt;" in rendered
    assert "&amp;" in rendered
    assert "&gt;" in rendered
    assert "<span class='citation-verified'>0.0001</span>" in rendered
    assert "<span class='citation-verified'>0.0</span>" in rendered

    plain = html.unescape(strip_html_tags(rendered))
    assert plain == content


def test_acceptance_criteria_combined():
    """Full acceptance check: no chip inside another, no split digits, tag stripping returns original text."""
    content = "Spectral flatness of 0.0 confirms pure tone, whereas 0.0001 indicates negligible noise."
    turn = DummyTurn(verified_tokens=["0.0001", "0.0"])
    rendered = render_grounded_answer_html(content, turn)

    assert "<span class='citation-verified'>" in rendered
    assert "[ [" not in rendered
    assert "<span class='citation-verified'><span" not in rendered
    assert html.unescape(strip_html_tags(rendered)) == content

