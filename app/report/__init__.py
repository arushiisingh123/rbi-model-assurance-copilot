"""Phase 3 LLM Report Generation package (Owner: Khushi).

Produces structured, three-layer assurance reports (ReportResult) combining:
- Layer 1: Verbatim technical findings from analytical modules
- Layer 2: Grounded regulatory evidence from RAG retrieval
- Layer 3: LLM synthesis via Groq (openai/gpt-oss-120b) with Python-enforced safety
"""
from app.report.generate import (
    ReportGenerationError,
    ReportGenerationUnavailable,
    generate_report,
)

__all__ = [
    "ReportGenerationError",
    "ReportGenerationUnavailable",
    "generate_report",
]
