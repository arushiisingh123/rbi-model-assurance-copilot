"""Continuous monitoring layer (owner: Arushi).

Coordinates the analytical modules that already own each calculation --
``app.drift`` (feature and prediction drift) and ``app.fairness`` -- over a
reference/current window pair, and reports one model-aware monitoring status.

This package calculates no metric, defines no threshold, and introduces no
second model-identity or evidence mechanism. See ``monitor.py`` for the
channel separation and ``evidence.py`` for the report-routing constraint that
currently applies to monitoring evidence records.
"""

from app.monitoring.evidence import MONITORING_EVIDENCE_TYPES, monitoring_evidence
from app.monitoring.monitor import MONITORING_CHANNELS, monitor_run
from app.monitoring.orchestration import (
    build_monitoring_window,
    resolve_protected_attribute,
    run_monitoring,
)
from app.monitoring.windows import (
    WINDOW_PROVENANCE_VALUES,
    MonitoringWindow,
    window_from_model_output,
)

__all__ = [
    "MONITORING_CHANNELS",
    "MONITORING_EVIDENCE_TYPES",
    "WINDOW_PROVENANCE_VALUES",
    "MonitoringWindow",
    "build_monitoring_window",
    "monitor_run",
    "monitoring_evidence",
    "resolve_protected_attribute",
    "run_monitoring",
    "window_from_model_output",
]
