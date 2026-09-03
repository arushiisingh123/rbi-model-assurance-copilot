"""Provenance and disclaimer for the RBI rule set (owner: Nidhi).

Phase 1: the rule *engine* is real, but the rules themselves are
illustrative and are NOT verified against current, binding RBI
regulation. Nothing here may be presented as regulatory evidence
(CLAUDE.md sections 6, 12, 21).
"""

RULE_SET_VERSION = "0.1.0-phase1"

RULE_SET_DISCLAIMER = (
    "These rules are ILLUSTRATIVE sample rules created to build and test "
    "the compliance rule engine during Phase 1. They are not a verified "
    "RBI rule repository, they do not cite specific binding RBI clauses, "
    "and their thresholds are placeholders chosen for demonstration. They "
    "must not be presented as real regulatory requirements or as "
    "compliance evidence. Verified rule-to-clause mapping and any real "
    "RBI text handling is later-phase work; see docs/decisions.md and "
    "docs/rbi-rules.md."
)

# The one real RBI document currently in the repo (used only by the
# Phase 0 RAG smoke test). Recorded here for reference; no Phase 1 rule
# is mapped to a specific clause of it. See docs/decisions.md,
# "Phase 0 RAG smoke-test source selected".
RAG_SMOKE_TEST_SOURCE = {
    "title": (
        "Master Circular - Prudential Norms on Income Recognition, Asset "
        "Classification and Provisioning pertaining to Advances"
    ),
    "circular_reference": "RBI/2014-15/74; DBOD.No.BP.BC.9/21.04.048/2014-15",
    "issue_date": "2014-07-01",
    "local_excerpt": (
        "data/rbi_sources/RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt"
    ),
}
