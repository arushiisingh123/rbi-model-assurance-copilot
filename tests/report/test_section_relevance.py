"""Phase C1: the section relevance gate was retuned, not weakened.

The gate's keyword lists grew when the corpus changed from one 2014 excerpt to
six current RBI Directions, because RBI does not legislate in machine-learning
vocabulary. Growing a gate is exactly how a gate quietly stops gating, so these
tests exist to prove it still tells the sections apart and still rejects text
that is merely regulatory-sounding.
"""
import pytest

from app.report.generate import (
    SECTION_QUERIES,
    SECTION_RELEVANCE_KEYWORDS,
    _retrieve_section_evidence,
)

SECTIONS = sorted(SECTION_QUERIES)


def _retrieves(text):
    """A retrieval double that always returns ``text``."""

    def fn(*, query):
        return {
            "query": query,
            "retrieved_text": text,
            "source": "test-double",
            "chunk_index": 0,
        }

    return fn


def _passes(section, text):
    evidence = _retrieve_section_evidence(
        section_key=section, query=SECTION_QUERIES[section], retrieval_fn=_retrieves(text)
    )
    return evidence.evidence_status == "RETRIEVED"


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_every_section_has_a_query_and_a_keyword_list():
    assert set(SECTION_QUERIES) == set(SECTION_RELEVANCE_KEYWORDS)
    assert len(SECTIONS) == 5


def test_no_keyword_is_a_single_common_word():
    """A bare "audit" or "policy" is on every page of every Direction.

    Allowing one would turn the gate into a rubber stamp. Every entry must be
    either a multi-word phrase or a term specific enough to be meaningless
    outside its own domain.
    """
    specific_single_words = {
        "explainability", "interpretability", "shap", "lime", "npa",
        "prudential", "supervisory", "advances", "provisioning",
        "non-performing", "discrimination",
    }
    for section, keywords in SECTION_RELEVANCE_KEYWORDS.items():
        for keyword in keywords:
            if " " in keyword or "-" in keyword:
                continue
            assert keyword in specific_single_words, (
                f"{section}: {keyword!r} is a bare common word; the gate would "
                "stop discriminating"
            )


# ---------------------------------------------------------------------------
# The gate still rejects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("section", SECTIONS)
def test_generic_regulatory_prose_clears_no_section(section):
    """Text that sounds regulatory but says nothing specific must fail.

    This is the sentence a weakened gate would let through: full of the words
    that appear in every RBI Direction, and evidence for nothing.
    """
    generic = (
        "The regulated entity shall put in place a Board approved policy and "
        "shall ensure that the same is placed before the Board periodically, "
        "as required under the applicable provisions."
    )
    assert not _passes(section, generic)


@pytest.mark.parametrize("section", SECTIONS)
def test_unrelated_text_clears_no_section(section):
    unrelated = "Quantum chromodynamics photosynthesis basketball tournament schedule."
    assert not _passes(section, unrelated)


@pytest.mark.parametrize("section", SECTIONS)
def test_an_empty_retrieval_clears_no_section(section):
    assert not _passes(section, "   ")


# ---------------------------------------------------------------------------
# The gate still discriminates BETWEEN sections
# ---------------------------------------------------------------------------

# Real sentences from the indexed Directions, one per section. Each must clear
# its own section's bar and must NOT clear most others'.
SECTION_EVIDENCE = {
    "explainability": (
        "RE shall provide a Key Fact Statement (KFS) to the borrower before "
        "the execution of the contract."
    ),
    "fairness": (
        "RE shall conduct enhanced due diligence taking into account the LSP's "
        "fairness in conduct with borrowers and past records of conduct."
    ),
    "drift": (
        "The operational risk management policy shall incorporate periodic "
        "assessment of IT-related risks, both inherent and potential."
    ),
    "compliance": (
        "The RE shall ensure compliance with all applicable statutory and "
        "regulatory requirements in respect of the outsourced activity."
    ),
}


@pytest.mark.parametrize("section", sorted(SECTION_EVIDENCE))
def test_each_section_accepts_its_own_evidence(section):
    assert _passes(section, SECTION_EVIDENCE[section]), (
        f"{section} rejected a real clause about its own subject"
    )


def test_section_evidence_is_not_interchangeable():
    """A clause relevant to one section must not satisfy every other.

    The gate is allowed some overlap -- RBI writes about grievance redressal
    and customer protection in the same breath -- but a sentence must not
    clear the bar for every section, which is what a rubber stamp looks like.
    """
    for section, text in SECTION_EVIDENCE.items():
        accepted = {other for other in SECTIONS if _passes(other, text)}
        assert section in accepted
        assert len(accepted) < len(SECTIONS), (
            f"evidence for {section!r} cleared EVERY section's bar: {accepted}"
        )


def test_analytics_vocabulary_still_clears_its_own_section():
    """The ML half of each list was kept, not replaced.

    A future ML-specific RBI source (the draft Model Risk Management
    directions, say) must still be recognisable, and the 2014 fixture must
    still clear the compliance bar.
    """
    assert _passes("explainability", "SHAP feature importance was computed.")
    assert _passes("fairness", "The disparate impact ratio was measured.")
    assert _passes("drift", "The population stability index was computed.")
    assert _passes("model", "Independent model validation was performed.")
    assert _passes(
        "compliance",
        "A non performing asset is an advance where interest remains overdue.",
    )


# ---------------------------------------------------------------------------
# Phase F: the heading counts towards relevance, and precision is preserved
# ---------------------------------------------------------------------------


def _retrieves_with_title(text, section_title):
    """A retrieval double carrying a PDF-native heading, as the retriever does."""

    def fn(*, query):
        return {
            "query": query,
            "retrieved_text": text,
            "source": "test-double",
            "chunk_index": 0,
            "provenance": {"section_title": section_title},
        }

    return fn


def _status(section, retrieval_fn):
    return _retrieve_section_evidence(
        section_key=section,
        query=SECTION_QUERIES[section],
        retrieval_fn=retrieval_fn,
    ).evidence_status


# The real false negative this change exists for: continuation text of clause 8
# of the 2025 Digital Lending Directions, whose heading IS a section keyword
# while the window itself abbreviates to "KFS".
CLAUSE_8_CONTINUATION = (
    "that digitally signed documents (on the letter head of the RE) viz., KFS, "
    "summary of loan product, sanction letter, terms and conditions, account "
    "statements, privacy policies"
)


def test_the_brittle_wording_failure_is_reproduced_without_the_heading():
    """Text alone still fails -- the vocabulary was not broadened."""
    assert _status("explainability", _retrieves_with_title(CLAUSE_8_CONTINUATION, None)) == (
        "NOT_FOUND"
    )


def test_the_same_text_is_accepted_under_its_own_heading():
    """The document's heading says what the passage is about."""
    assert _status(
        "explainability",
        _retrieves_with_title(CLAUSE_8_CONTINUATION, "Disclosures to borrowers"),
    ) == "RETRIEVED"


def test_an_off_topic_heading_does_not_rescue_irrelevant_text():
    """A heading admits a passage only for the section it is on-topic for."""
    fn = _retrieves_with_title(CLAUSE_8_CONTINUATION, "Disclosures to borrowers")

    assert _status("fairness", fn) == "NOT_FOUND"
    assert _status("drift", fn) == "NOT_FOUND"
    assert _status("model", fn) == "NOT_FOUND"


@pytest.mark.parametrize("section", SECTIONS)
def test_generic_prose_is_still_rejected_whatever_its_heading(section):
    """The heading must not become a way in for governance boilerplate."""
    generic = (
        "The regulated entity shall put in place a Board approved policy and "
        "shall ensure that the same is placed before the Board periodically, "
        "as required under the applicable provisions."
    )
    assert _status(section, _retrieves_with_title(generic, "Preamble")) == "NOT_FOUND"
    assert _status(section, _retrieves_with_title(generic, "Definitions")) == "NOT_FOUND"
    assert _status(section, _retrieves_with_title(generic, None)) == "NOT_FOUND"


@pytest.mark.parametrize("section", SECTIONS)
def test_unrelated_text_is_still_rejected_whatever_its_heading(section):
    unrelated = "Quantum chromodynamics photosynthesis basketball tournament."
    assert _status(section, _retrieves_with_title(unrelated, "Short title")) == "NOT_FOUND"


def test_a_missing_heading_is_handled_like_any_absent_metadata():
    """Retrieval doubles and text sources carry no provenance at all."""

    def no_provenance(*, query):
        return {
            "query": query,
            "retrieved_text": "SHAP feature importance was computed.",
            "source": "s",
            "chunk_index": 0,
        }

    assert _status("explainability", no_provenance) == "RETRIEVED"


def test_the_heading_is_matched_with_the_same_keyword_list():
    """No separate, looser vocabulary exists for headings.

    A heading that is not in the section's own keyword list cannot admit
    anything -- the change is which TEXT is searched, never what is searched
    for.
    """
    fn = _retrieves_with_title("Some entirely neutral sentence.", "Exit strategy")
    for section in SECTIONS:
        assert _status(section, fn) == "NOT_FOUND"
