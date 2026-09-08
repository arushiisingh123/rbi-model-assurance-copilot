"""Phase 3 Report Generation: generate_report() and supporting helpers.

Owner: Khushi (API / Integration)
Provider: Groq (openai/gpt-oss-120b) via Groq Python SDK
Safety: Strictly enforced in Python (regulatory_basis forced, post-gen scan-and-strip)
"""
from datetime import datetime, timezone
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import uuid

from app.api.schemas import (
    Citation,
    EvidenceCoverage,
    LLMInterpretation,
    ReportResult,
    ReportSection,
    RetrievedEvidence,
    TechnicalFinding,
)
from app.compliance.technical_findings import build_technical_findings

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
INTERIM_SOURCE_LABEL = "INTERIM SINGLE-DOC: RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt"

# Regulatory claim detection regex
REGULATORY_CLAIM_PATTERN = re.compile(
    r"(?i)\b(rbi|regulation[s]?|requirement[s]?|clause[s]?|must comply|circular[s]?|mandate[s]?|mandated|statutory|governing rule)\b|§"
)

SAFE_FALLBACK_TEXT = (
    "The generated response for this section was withheld because no supporting evidence was retrieved."
)

SAFE_RETRIEVED_FALLBACK_TEMPLATE = (
    "Technical finding for {heading} evaluates with status {status}. "
    "Detailed regulatory interpretation was withheld because generated text went beyond the retrieved evidence excerpt."
)

# Domain keywords required for retrieved text to clear genuine Python relevance bar
SECTION_RELEVANCE_KEYWORDS = {
    "model": ["model risk", "model validation", "governance framework", "algorithm validation", "scoring model"],
    "explainability": ["explainability", "interpretability", "shap", "lime", "feature importance"],
    "fairness": ["disparate impact", "demographic parity", "protected group", "discrimination", "adverse impact"],
    "drift": ["population stability index", "distribution shift", "data drift", "kolmogorov-smirnov", "ks statistic"],
    "compliance": ["non-performing", "non performing", "npa", "advances", "classification", "provisioning", "prudential"],
}

SECTION_QUERIES = {
    "model": "model risk management governance documentation validation records credit scoring",
    "explainability": "model explainability global feature importance transparency decisions shap lime",
    "fairness": "fairness non-discrimination disparate impact protected demographic parity selection rate",
    "drift": "data drift population stability index distribution shift ks statistic",
    "compliance": "What is a non performing asset prudential norms classification advances",
}


class ReportGenerationError(Exception):
    """Base exception for report generation errors."""
    pass


class ReportGenerationUnavailable(ReportGenerationError):
    """Raised when live report generation cannot proceed (e.g. missing API key)."""
    pass


class IsolatedRAGRetriever:
    """Isolated, thread-safe RAG retriever for a single report generation run.

    Builds an isolated collection with a unique UUID name in ChromaDB's EphemeralClient,
    runs section queries against it, and cleans up strictly its own collection on completion.
    Never touches smoke_test.py's shared 'rbi_smoke_test' collection, avoiding concurrent race conditions.
    """

    def __init__(self):
        import chromadb
        from app.rag.smoke_test import (
            CHUNK_SIZE_WORDS,
            DOCUMENT_PATH,
            HashingEmbeddingFunction,
            chunk_text,
            load_document_text,
        )

        self.client = chromadb.EphemeralClient()
        self.collection_name = f"rbi_rep_{uuid.uuid4().hex}"
        self.document_path = DOCUMENT_PATH

        text = load_document_text(self.document_path)
        chunks = chunk_text(text, chunk_size_words=CHUNK_SIZE_WORDS)
        self.chunks = chunks

        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=HashingEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        self.collection.add(
            documents=chunks,
            ids=[f"chunk-{i}" for i in range(len(chunks))],
            metadatas=[{"source": self.document_path.name, "chunk_index": i} for i in range(len(chunks))],
        )

    def query(self, query_text: str) -> Dict[str, Any]:
        result = self.collection.query(query_texts=[query_text], n_results=1)
        return {
            "query": query_text,
            "retrieved_text": result["documents"][0][0],
            "source": result["metadatas"][0][0]["source"],
            "chunk_index": result["metadatas"][0][0]["chunk_index"],
            "num_chunks_indexed": len(self.chunks),
        }

    def close(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as exc:
            logger.debug("Failed to delete isolated collection %s: %s", self.collection_name, exc)


def _is_retrieved_text_grounded_in_citation(
    text: str,
    citation_quote: str,
    citation_locator: str = "",
) -> Tuple[bool, str]:
    """Verify that regulatory-claim language in a RETRIEVED section is reasonably grounded.

    Returns (is_grounded, reason).
    """
    if not REGULATORY_CLAIM_PATTERN.search(text):
        return True, ""

    quote_lower = (citation_quote or "").lower()
    locator_lower = (citation_locator or "").lower()

    # 1. Specific identifier check: circular codes, clause numbers, section symbols
    specific_id_patterns = [
        r"\b(?:circular|circular\s+no\.?|notification)\s+([0-9a-zA-Z\.\-_/]+)",
        r"\b(?:clause|rule)\s+([0-9a-zA-Z\.\-_/]+)",
        r"§\s*([0-9a-zA-Z\.\-_/]+)",
        r"\b(rbi[/\-_][0-9a-zA-Z\.\-_/]+)\b",
    ]
    for pat in specific_id_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        for match in matches:
            norm_match = match.lower().strip()
            if norm_match not in quote_lower and norm_match not in locator_lower:
                return False, f"Fabricated regulatory identifier '{match}' not present in citation quote or locator."

    # 2. Strong mandate check: absolute legal/statutory obligation claims
    mandate_patterns = [
        r"\bmust comply\b",
        r"\bstatutory\s+(?:mandate|requirement|obligation)\b",
        r"\blegally\s+mandated\b",
        r"\bpunitive\b",
    ]
    for pat in mandate_patterns:
        if re.search(pat, text, re.IGNORECASE):
            if not re.search(r"\b(shall|must|required|mandated)\b", quote_lower):
                return False, "Absolute compliance mandate asserted but unsupported by citation text."

    # 3. Substantive overlap check: ensure the claim shares key words with the quote
    quote_tokens = set(re.findall(r"\b[a-z]{4,}\b", quote_lower))
    stopwords = {"this", "that", "with", "from", "have", "been", "were", "what", "where", "when", "your", "their", "into"}
    substantive_quote_tokens = quote_tokens - stopwords
    if substantive_quote_tokens:
        text_tokens = set(re.findall(r"\b[a-z]{4,}\b", text.lower())) - stopwords
        overlap = substantive_quote_tokens.intersection(text_tokens)
        if not overlap:
            return False, "Regulatory claim shares no substantive concepts with retrieved citation."

    return True, ""


def _extract_model_finding(model: Dict[str, Any]) -> TechnicalFinding:
    """Extract Layer 1 technical finding for model evaluation."""
    meta = model.get("model_metadata", {})
    val = {
        "model_type": meta.get("model_type", "logistic_regression"),
        "version": meta.get("version", "0.1.0"),
        "sample_predictions": model.get("predictions", [])[:3],
        "sample_probabilities": [round(float(p), 4) for p in model.get("probabilities", [])[:3]],
    }
    return TechnicalFinding(
        ref="model.model_metadata",
        value=val,
        status="PASS",
        source_module="app.models",
        provenance="mock" if model.get("is_mock", False) else "observed",
    )


def _extract_explainability_finding(explainability: Dict[str, Any]) -> TechnicalFinding:
    """Extract Layer 1 technical finding for feature explainability."""
    exp_method = explainability.get("method", "shap")
    top_feature = None
    top_importance = None
    global_imp = explainability.get("global_importance", {})
    if isinstance(global_imp, dict) and global_imp:
        sorted_features = sorted(global_imp.items(), key=lambda x: abs(x[1]), reverse=True)
        top_feature, top_importance = sorted_features[0]

    val = {
        "method": exp_method,
        "top_feature": top_feature,
        "top_importance": round(float(top_importance), 4) if top_importance is not None else None,
    }
    return TechnicalFinding(
        ref="explainability.global_importance",
        value=val,
        status="PASS",
        source_module="app.explainability",
        provenance="mock" if explainability.get("is_mock", False) else "observed",
    )


def _extract_fairness_finding(fairness: Dict[str, Any]) -> TechnicalFinding:
    """Extract Layer 1 technical finding for fairness evaluation."""
    val = {
        "protected_attribute": fairness.get("protected_attribute", "personal_status_and_sex"),
        "disparate_impact_ratio": fairness.get("disparate_impact_ratio"),
        "demographic_parity_diff": fairness.get("demographic_parity_diff"),
    }
    return TechnicalFinding(
        ref="fairness.disparate_impact_ratio",
        value=val,
        status=fairness.get("status", "PASS"),
        source_module="app.fairness",
        provenance="mock" if fairness.get("is_mock", False) else "observed",
    )


def _extract_drift_finding(drift: Dict[str, Any]) -> TechnicalFinding:
    """Extract Layer 1 technical finding for drift detection."""
    val = {
        "psi": drift.get("psi"),
        "ks_statistic": drift.get("ks_statistic"),
        "features_evaluated": drift.get("features_evaluated", []),
    }
    return TechnicalFinding(
        ref="drift.psi",
        value=val,
        status=drift.get("status", "PASS"),
        source_module="app.drift",
        provenance="mock" if drift.get("is_mock", False) else "observed",
    )


def _extract_compliance_finding(compliance: Dict[str, Any]) -> TechnicalFinding:
    """Extract Layer 1 technical finding for compliance mapping."""
    findings = compliance.get("findings", [])
    evaluated_statuses = sorted(list(set(f.get("status") for f in findings if f.get("status"))))
    val = {
        "total_findings": len(findings),
        "evaluated_statuses": evaluated_statuses,
    }
    overall_status = (
        "FAIL" if any(f.get("status") == "FAIL" for f in findings)
        else "WARNING" if any(f.get("status") == "WARNING" for f in findings)
        else "PASS"
    )
    return TechnicalFinding(
        ref="compliance.findings",
        value=val,
        status=overall_status,
        source_module="app.compliance",
        provenance="mock" if compliance.get("is_mock", False) else "observed",
    )


def _retrieve_section_evidence(
    section_key: str,
    query: str,
    retrieval_fn: Optional[Any] = None,
) -> RetrievedEvidence:
    """Retrieve evidence using RAG smoke test with Python relevance gating."""
    if retrieval_fn is None:
        return RetrievedEvidence(evidence_status="NOT_FOUND", citations=[])

    try:
        outcome = retrieval_fn(query=query)
    except Exception as exc:
        logger.warning("Retrieval failed for query '%s': %s", query, exc)
        return RetrievedEvidence(evidence_status="NOT_FOUND", citations=[])

    retrieved_text = outcome.get("retrieved_text", "")
    text_lower = retrieved_text.lower()

    # Apply genuine Python relevance bar
    keywords = SECTION_RELEVANCE_KEYWORDS.get(section_key, [])
    is_relevant = any(kw in text_lower for kw in keywords)

    if not is_relevant or not retrieved_text.strip():
        return RetrievedEvidence(evidence_status="NOT_FOUND", citations=[])

    source_doc = outcome.get("source", "RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt")
    chunk_idx = outcome.get("chunk_index", 0)

    citation = Citation(
        source=f"INTERIM SINGLE-DOC: {source_doc}",
        locator=f"chunk #{chunk_idx}",
        quote=retrieved_text.strip(),
        provenance="interim_single_document",
    )
    return RetrievedEvidence(evidence_status="RETRIEVED", citations=[citation])


def _build_llm_prompt(
    findings: Dict[str, TechnicalFinding],
    evidence: Dict[str, RetrievedEvidence],
) -> str:
    """Construct prompt for the single Groq LLM call."""
    sections_prompt_parts = []
    for key in ["model", "explainability", "fairness", "drift", "compliance"]:
        tf = findings[key]
        ev = evidence[key]
        if ev.evidence_status == "RETRIEVED" and ev.citations:
            ev_desc = f"RETRIEVED CITATION: \"{ev.citations[0].quote}\" (Locator: {ev.citations[0].locator})"
        else:
            ev_desc = "NONE RETRIEVED"

        part = (
            f"Section: {key}\n"
            f"Technical Finding Ref: {tf.ref}\n"
            f"Status: {tf.status}\n"
            f"Value: {json.dumps(tf.value)}\n"
            f"Retrieved Evidence: {ev_desc}\n"
        )
        sections_prompt_parts.append(part)

    all_sections = "\n---\n".join(sections_prompt_parts)

    return (
        "You are an AI Model Risk & Assurance Copilot reporting assistant.\n"
        "Generate a natural-language interpretation for each of the following 5 evaluation sections.\n\n"
        "CRITICAL RULES:\n"
        "1. For any section where retrieved evidence is 'NONE RETRIEVED', you must NOT state or imply any "
        "regulatory requirement, mandate, or governing rule — explain only the technical finding in plain language.\n"
        "2. Do not invent or recalculate any metrics. Use only the provided technical finding values.\n"
        "3. Output MUST be valid JSON with exactly the following structure:\n"
        "{\n"
        "  \"model\": \"...\",\n"
        "  \"explainability\": \"...\",\n"
        "  \"fairness\": \"...\",\n"
        "  \"drift\": \"...\",\n"
        "  \"compliance\": \"...\"\n"
        "}\n\n"
        f"SECTIONS DATA:\n{all_sections}"
    )


def _call_groq_llm(
    prompt: str,
    llm_client: Optional[Any] = None,
) -> Dict[str, str]:
    """Call Groq chat completions API once and parse JSON result."""
    client = llm_client
    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or not api_key.strip():
            raise ReportGenerationUnavailable(
                "GROQ_API_KEY environment variable is not set. Live LLM report generation unavailable."
            )
        try:
            from groq import Groq
            client = Groq(api_key=api_key.strip())
        except Exception as exc:
            raise ReportGenerationUnavailable(f"Failed to initialize Groq client: {exc}") from exc

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assurance reporting assistant. You strictly output valid JSON "
                        "mapping section keys ('model', 'explainability', 'fairness', 'drift', 'compliance') "
                        "to concise 1-3 sentence interpretations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()
    except ReportGenerationError:
        raise
    except Exception as exc:
        raise ReportGenerationError(f"Groq API call failed: {exc}") from exc

    # Parse JSON from response
    try:
        if "```" in content:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if match:
                content = match.group(1).strip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response did not contain a JSON object.")
        return parsed
    except Exception as exc:
        logger.warning("Failed to parse LLM JSON response: %s. Response content: %s", exc, content)
        raise ReportGenerationError(f"Failed to parse LLM response as JSON: {exc}") from exc


def generate_report(
    *,
    model: Dict[str, Any],
    explainability: Dict[str, Any],
    fairness: Dict[str, Any],
    drift: Dict[str, Any],
    compliance: Dict[str, Any],
    evidence_records: Optional[List[Dict[str, Any]]] = None,
    llm_client: Optional[Any] = None,
    retrieval_fn: Optional[Any] = None,
    skip_live: bool = False,
) -> Dict[str, Any]:
    """Generate a structured, three-layer assurance report (validating against ReportResult).

    Parameters
    ----------
    model : Dict[str, Any]
        Real model prediction dictionary from predict_batch().
    explainability : Dict[str, Any]
        Explainability dictionary from explain().
    fairness : Dict[str, Any]
        Fairness report dictionary from fairness_report().
    drift : Dict[str, Any]
        Drift report dictionary from drift_report().
    compliance : Dict[str, Any]
        Compliance evaluation dictionary from evaluate_compliance().
    evidence_records : Optional[List[Dict[str, Any]]], optional
        Optional evidence records from Phase 3 modules (e.g. fairness_evidence).
    llm_client : Optional[Any], optional
        Injectable Groq client or test double. If None, builds Groq client using GROQ_API_KEY.
    retrieval_fn : Optional[Any], optional
        Injectable retrieval function or test double. If None, uses an isolated RAG retriever.
    skip_live : bool, optional
        If True, immediately raises ReportGenerationUnavailable without running any pipeline work.

    Returns
    -------
    Dict[str, Any]
        Full ReportResult dictionary payload.

    Raises
    ------
    ReportGenerationUnavailable
        If GROQ_API_KEY is not set when live generation is attempted or skip_live is True.
    ReportGenerationError
        If Groq API call fails or response parsing fails.
    """
    # 0. EARLY CHECK: Fast failure if API key is missing or live generation is explicitly skipped
    if skip_live:
        raise ReportGenerationUnavailable("Live generation explicitly skipped (skip_live=True).")

    if llm_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or not api_key.strip():
            raise ReportGenerationUnavailable(
                "GROQ_API_KEY environment variable is not set. Live LLM report generation unavailable."
            )

    # 1. ASSEMBLE: build per-section technical findings (Layer 1) verbatim
    _ = build_technical_findings(
        model=model,
        explainability=explainability,
        fairness=fairness,
        drift=drift,
    )

    findings: Dict[str, TechnicalFinding] = {
        "model": _extract_model_finding(model),
        "explainability": _extract_explainability_finding(explainability),
        "fairness": _extract_fairness_finding(fairness),
        "drift": _extract_drift_finding(drift),
        "compliance": _extract_compliance_finding(compliance),
    }

    # 2. RETRIEVE: RAG retrieval per section (Layer 2) with isolated collection & genuine relevance gate
    retriever = None
    active_retrieval_fn = retrieval_fn
    if active_retrieval_fn is None:
        try:
            retriever = IsolatedRAGRetriever()
            active_retrieval_fn = retriever.query
        except Exception as exc:
            logger.warning("Failed to initialize IsolatedRAGRetriever: %s", exc)
            active_retrieval_fn = None

    evidence: Dict[str, RetrievedEvidence] = {}
    try:
        for key in ["model", "explainability", "fairness", "drift", "compliance"]:
            evidence[key] = _retrieve_section_evidence(
                section_key=key,
                query=SECTION_QUERIES[key],
                retrieval_fn=active_retrieval_fn,
            )
    finally:
        if retriever is not None:
            retriever.close()

    # 3. CALL THE LLM ONCE: single Groq call
    prompt = _build_llm_prompt(findings, evidence)
    llm_texts = _call_groq_llm(prompt, llm_client=llm_client)

    # 4. ENFORCE SAFETY IN PYTHON (Critical, non-negotiable)
    section_configs = [
        ("model", "Credit Scoring Model Evaluation", ["model.model_metadata", "model.version"]),
        ("explainability", "Feature Explainability (SHAP)", ["explainability.global_importance"]),
        ("fairness", "Fairness Evaluation", ["fairness.disparate_impact_ratio", "fairness.status"]),
        ("drift", "Data & Prediction Drift Detection", ["drift.psi", "drift.ks_statistic", "drift.status"]),
        ("compliance", "RBI Compliance Rules Mapping", ["compliance.findings", "fairness.status"]),
    ]

    report_sections: List[ReportSection] = []
    retrieved_count = 0
    not_found_count = 0

    for key, heading, grounded_refs in section_configs:
        tf = findings[key]
        ev = evidence[key]
        raw_text = str(llm_texts.get(key, f"Evaluation for {heading} completed with status {tf.status}."))

        if ev.evidence_status == "RETRIEVED":
            retrieved_count += 1
            # Regulatory basis forced to illustrative_rule_only for interim single-doc
            reg_basis = "illustrative_rule_only"
            grounded_in = grounded_refs + [cit.locator for cit in ev.citations]

            # Post-generation scan on RETRIEVED sections: ensure claims are grounded in quote
            cit_quote = ev.citations[0].quote if ev.citations else ""
            cit_loc = ev.citations[0].locator if ev.citations else ""
            is_grounded, reason = _is_retrieved_text_grounded_in_citation(
                text=raw_text,
                citation_quote=cit_quote,
                citation_locator=cit_loc,
            )
            if not is_grounded:
                logger.warning(
                    "Safety enforcement: overreaching regulatory claim detected in RETRIEVED section '%s' (%s). "
                    "Replacing with safe fallback.",
                    key,
                    reason,
                )
                final_text = SAFE_RETRIEVED_FALLBACK_TEMPLATE.format(
                    heading=heading,
                    status=tf.status,
                )
            else:
                final_text = raw_text
        else:
            not_found_count += 1
            # NOT_FOUND forces regulatory_basis = "none"
            reg_basis = "none"
            grounded_in = grounded_refs

            # Post-generation scan-and-strip on NOT_FOUND sections: detect regulatory claim language
            if REGULATORY_CLAIM_PATTERN.search(raw_text):
                logger.warning(
                    "Safety enforcement: regulatory claim language detected in NOT_FOUND section '%s'. "
                    "Replacing with safe fallback.",
                    key,
                )
                final_text = SAFE_FALLBACK_TEXT
            else:
                final_text = raw_text

        interpretation = LLMInterpretation(
            text=final_text,
            grounded_in=grounded_in,
            regulatory_basis=reg_basis,
            is_mock=False,
        )

        report_sections.append(
            ReportSection(
                heading=heading,
                technical_finding=tf,
                retrieved_evidence=ev,
                llm_interpretation=interpretation,
            )
        )

    # Summary and Disclaimers
    coverage = EvidenceCoverage(
        retrieved=retrieved_count,
        not_found=not_found_count,
        total=len(section_configs),
    )

    disclaimers = [
        "LLM-generated report text produced via Groq (openai/gpt-oss-120b). Technical findings are calculated by Python analytical modules and passed verbatim.",
        "Retrieved evidence is sourced from the Phase 0 interim single-document excerpt (RBI IRAC Advances 2014), not a full verified RBI regulatory corpus.",
        "Regulatory basis is restricted to illustrative_rule_only or none; full cited_evidence requires the upcoming verified regulatory corpus.",
        "For sections without supporting evidence (NOT_FOUND), text is restricted to technical explanation with no regulatory claims.",
    ]

    model_version = model.get("model_metadata", {}).get("version", "0.1.0")

    result = ReportResult(
        report_id=f"rep-{uuid.uuid4().hex[:8]}",
        generated_at=datetime.now(timezone.utc).isoformat(),
        model_version=model_version,
        sections=report_sections,
        disclaimers=disclaimers,
        evidence_coverage=coverage,
        is_mock=False,
    )

    return result.model_dump()
