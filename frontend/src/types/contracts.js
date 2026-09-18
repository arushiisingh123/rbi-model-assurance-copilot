/**
 * Frontend mirror of the backend's Pydantic response models.
 *
 * JSDoc typedefs rather than TypeScript: the app is plain JSX and a student
 * team maintains it, so this documents the contract and gives editor
 * autocomplete without adding a compiler to the toolchain.
 *
 * FIELD NAMES ARE THE BACKEND'S, VERBATIM (snake_case). Nothing is renamed on
 * the way in. Renaming would mean the UI and the API disagree about what a
 * field is called, and the first person to debug a missing value would have
 * to reverse-engineer the mapping. Sourced from app/api/schemas.py.
 *
 * @see app/api/schemas.py
 */

/**
 * GET /models -> array of these (ModelAdapter.metadata()).
 * @typedef {Object} ModelRegistryEntry
 * @property {string} model_id
 * @property {string} model_version
 * @property {string} model_type
 * @property {string} integration_type
 * @property {Object<string, boolean>} capabilities
 */

/**
 * @typedef {Object} ModelMetadata
 * @property {string} model_type
 * @property {string} version
 * @property {string} trained_on
 * @property {string[]} feature_names
 * @property {Object|null} [label_semantics]
 */

/**
 * @typedef {Object} ModelMetrics
 * @property {number} accuracy
 * @property {number} precision
 * @property {number} recall
 * @property {number} f1
 * @property {number|null} roc_auc
 * @property {"computed"|"unavailable_no_probabilities"} roc_auc_status
 * @property {number} n_test_samples
 * @property {boolean} is_mock
 */

/**
 * GET /model
 * @typedef {Object} ModelResult
 * @property {number[]} predictions
 * @property {number[]} probabilities
 * @property {string[]} instance_ids
 * @property {Object[]} feature_matrix
 * @property {ModelMetadata} model_metadata
 * @property {boolean} is_mock
 * @property {ModelMetrics|null} [model_metrics]  null = not applicable
 */

/**
 * GET /explainability
 *
 * `scale`, `explainer` and `fidelity` are authoritative and must be read from
 * here -- they are NOT derivable from `method`.
 *
 * @typedef {Object} ExplainabilityResult
 * @property {"shap"|"lime"} method
 * @property {{row_index: number, contributions: Object<string, number>}[]} per_instance
 * @property {Object<string, number>} global_importance
 * @property {boolean} is_mock
 * @property {string|null} [model_id]
 * @property {string|null} [model_type]
 * @property {string|null} [integration_type]
 * @property {boolean|null} [available]   false = structured unavailable
 * @property {string|null} [explainer]    LinearExplainer | TreeExplainer | KernelExplainer | LimeTabularExplainer
 * @property {string|null} [scale]        "log_odds" | "probability"
 * @property {string|null} [fidelity]     "exact" | "approximate" | "surrogate"
 * @property {string[]|null} [feature_space]
 * @property {string[]|null} [limitations]
 */

/**
 * @typedef {Object} FairnessGroup
 * @property {string} group
 * @property {number} count
 * @property {number} favorable_count
 * @property {number} selection_rate
 */

/**
 * protected_attribute === "none_declared" with status "PENDING" is the
 * structured unavailable case. Never render it as a pass.
 *
 * @typedef {Object} FairnessResult
 * @property {string} protected_attribute
 * @property {number} demographic_parity_diff
 * @property {number} disparate_impact_ratio
 * @property {"PASS"|"WARNING"|"FAIL"|"PENDING"} status
 * @property {boolean} is_mock
 * @property {FairnessGroup[]} groups
 */

/**
 * @typedef {Object} DriftResult
 * @property {string[]} features_evaluated
 * @property {number} psi
 * @property {number} ks_statistic
 * @property {"PASS"|"WARNING"|"FAIL"|"PENDING"} status
 * @property {boolean} is_mock
 * @property {string|null} [note]
 * @property {{feature: string, psi: number, ks_statistic: number}[]} per_feature
 */

/** GET /fairness-drift
 * @typedef {Object} FairnessDriftResult
 * @property {FairnessResult} fairness
 * @property {DriftResult} drift
 * @property {string|null} [note]
 */

/**
 * Prediction/output drift -- distinct from feature drift.
 * @typedef {Object} PredictionDriftResult
 * @property {number} label_psi
 * @property {string} label_status
 * @property {Array<string|number>} classes_evaluated
 * @property {Object[]} per_class
 * @property {number} [score_psi]
 * @property {number} [score_ks_statistic]
 * @property {string} [score_status]
 * @property {"computed"|"unavailable_no_scores"} score_availability
 * @property {string} status
 * @property {boolean} is_mock
 */

/**
 * @typedef {Object} AssuranceRunContext
 * @property {string} model_id
 * @property {string} model_version
 * @property {string} assurance_run_id
 * @property {string|null} [adapter_id]
 */

/**
 * GET /monitoring -> { result, evidence, protected_attribute }
 *
 * A channel present with status PENDING was not measured; a channel absent
 * (null) did not run at all. Neither is "no drift".
 *
 * @typedef {Object} MonitoringResult
 * @property {AssuranceRunContext} context
 * @property {string} reference_window_id
 * @property {string} current_window_id
 * @property {{reference: Object, current: Object}} windows
 * @property {DriftResult|null} feature_drift
 * @property {PredictionDriftResult|null} prediction_drift
 * @property {FairnessResult|null} fairness
 * @property {Object<string, string>} channel_status
 * @property {string} monitoring_status
 * @property {{channel: string, status: string, detail: Object}[]} alerts
 * @property {boolean} is_mock
 */

/**
 * @typedef {Object} MonitoringAssuranceResult
 * @property {MonitoringResult} result
 * @property {Object[]} evidence
 * @property {string|null} protected_attribute
 */

/**
 * @typedef {Object} ComplianceFinding
 * @property {string} rule_id
 * @property {string} rule_description
 * @property {string} technical_finding_ref
 * @property {"PASS"|"WARNING"|"FAIL"|"PENDING"} status
 * @property {string[]} evidence_chunks
 * @property {string|null} [model_id]
 * @property {string|null} [assurance_run_id]
 */

/** GET /compliance
 * @typedef {Object} ComplianceResult
 * @property {ComplianceFinding[]} findings
 * @property {boolean} is_mock
 * @property {string|null} [model_id]
 * @property {string|null} [assurance_run_id]
 */

/**
 * GET /assurance-result
 *
 * monitoring === null means it did not run, and
 * monitoring_unavailable_reason then states why.
 *
 * @typedef {Object} AssuranceResult
 * @property {ModelResult} model
 * @property {ExplainabilityResult} explainability
 * @property {FairnessDriftResult} fairness_drift
 * @property {ComplianceResult} compliance
 * @property {string} note
 * @property {string|null} [model_id]
 * @property {string|null} [assurance_run_id]
 * @property {MonitoringAssuranceResult|null} [monitoring]
 * @property {string|null} [monitoring_unavailable_reason]
 */

/**
 * One report section's three layers.
 * @typedef {Object} ReportSection
 * @property {string} heading
 * @property {Object} technical_finding      Layer 1: computed
 * @property {{evidence_status: "RETRIEVED"|"NOT_FOUND", citations: Object[]}} retrieved_evidence  Layer 2
 * @property {Object} llm_interpretation     Layer 3: narrative
 * @property {Object[]} [supporting_evidence]
 */

/** GET /report
 * @typedef {Object} ReportResult
 * @property {string} report_id
 * @property {string} generated_at
 * @property {string} model_version
 * @property {ReportSection[]} sections
 * @property {string[]} disclaimers
 * @property {{retrieved: number, not_found: number, total: number}} evidence_coverage
 * @property {boolean} is_mock
 * @property {string|null} [model_id]
 * @property {string|null} [assurance_run_id]
 */

export {};
