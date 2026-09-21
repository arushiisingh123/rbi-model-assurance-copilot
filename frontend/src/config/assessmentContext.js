/**
 * DECLARED ASSESSMENT CONTEXTS — who is being assessed, stated by the caller.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * RBI applicability depends on facts about the ORGANISATION, not about the
 * model: what kind of regulated entity it is, whether it lends digitally,
 * whether a third party serves the model. A dataset cannot tell you any of
 * these, and the backend deliberately refuses to guess — an undeclared
 * dimension stays APPLICABILITY_UNCLEAR rather than being assumed either way.
 *
 * So somebody has to declare them. This file is that declaration, written
 * down in one readable place instead of being buried in a request builder.
 * It is a DEMO context for a DEMO model. It is not a claim about any real
 * institution, and the UI says so on screen.
 *
 * WHAT THIS FILE IS NOT
 * ---------------------
 * Not a backend default. Nothing here is compiled into the RBI engine; if
 * this file were deleted, the engine would go back to reporting everything
 * UNCLEAR, which is the correct behaviour for an undeclared caller. Nothing
 * here alters a requirement, a clause, a source URL or a regulatory status.
 *
 * DECLARED vs CONSUMED
 * --------------------
 * The context below records eight dimensions because that is how a reviewer
 * describes an assessment. The applicability engine currently consumes five
 * of them (see ENGINE_DIMENSIONS). The other three are recorded for the
 * reader and sent nowhere. That gap is shown in the UI rather than hidden:
 * a dimension that influences no decision must not look like one that does.
 */

/** The dimensions the backend applicability engine actually reads. */
export const ENGINE_DIMENSIONS = [
  "entity_type",
  "nbfc_layer",
  "digital_lending",
  "microfinance",
  "uses_external_model_vendor",
];

/**
 * The synthetic bank demo, as declared by whoever set up this demo.
 *
 * `declared` is the reviewer-facing description. `profile` is the same facts
 * expressed in the backend's vocabulary — written out explicitly, one line
 * per dimension, so the translation can be audited rather than inferred.
 *
 * Dimensions genuinely absent are OMITTED, never filled with a plausible
 * guess. `nbfc_layer` is absent because the declared entity is a bank, and
 * `microfinance` because this demo context does not state it. Both therefore
 * reach the backend undeclared, which is what makes UNCLEAR still reachable.
 */
const SYNTHETIC_BANK_CONTEXT = {
  label: "Synthetic bank — declared demo assessment context",
  note:
    "Declared for this demo, not observed from the model or its data. " +
    "Applicability below follows from these declarations; change them and " +
    "the applicable requirements change with them.",

  declared: [
    { key: "regulated_entity_type", value: "BANK", consumed: true },
    { key: "model_use_case", value: "CREDIT_SCORING", consumed: false },
    { key: "product", value: "RETAIL_LENDING", consumed: false },
    { key: "digital_vs_physical", value: "DIGITAL", consumed: true },
    { key: "third_party_dependency", value: "TRUE", consumed: true },
    { key: "data_type", value: "CREDIT_DATA", consumed: false },
    { key: "customer_population", value: "RETAIL", consumed: false },
    { key: "loan_type", value: "PERSONAL", consumed: false },
  ],

  // Declared context -> backend dimensions. Each line is a deliberate
  // statement, not a derivation:
  //
  //   regulated_entity_type = BANK      -> entity_type: "BANK"
  //   digital_vs_physical   = DIGITAL   -> digital_lending: true
  //   third_party_dependency= TRUE      -> uses_external_model_vendor: true
  //
  // nbfc_layer and microfinance are intentionally absent -- see above.
  profile: {
    entity_type: "BANK",
    digital_lending: true,
    uses_external_model_vendor: true,
  },
};

/**
 * The German Credit demo models, as declared by whoever set up this demo.
 *
 * Both `german-credit-logistic-regression` and `german-credit-random-forest`
 * share this context: same declared institution, two candidate model types
 * evaluated in-house. `uses_external_model_vendor: false` is literally true
 * for these two -- both run in-process (scikit-learn), not via RESTAdapter,
 * unlike the synthetic bank's HTTP-served model.
 *
 * Dimensions genuinely absent are OMITTED, never filled with a plausible
 * guess -- `microfinance` is absent because this demo context does not state
 * it, matching the synthetic bank context above.
 */
const GERMAN_CREDIT_CONTEXT = {
  label: "German Credit demo models — declared demo assessment context",
  note:
    "Declared for this demo, not observed from the model or its data. " +
    "Applicability below follows from these declarations; change them and " +
    "the applicable requirements change with them.",

  declared: [
    { key: "regulated_entity_type", value: "NBFC", consumed: true },
    { key: "nbfc_layer", value: "MIDDLE", consumed: true },
    { key: "model_use_case", value: "CREDIT_SCORING", consumed: false },
    { key: "product", value: "RETAIL_LENDING", consumed: false },
    { key: "digital_vs_physical", value: "DIGITAL", consumed: true },
    { key: "third_party_dependency", value: "FALSE", consumed: true },
    { key: "data_type", value: "CREDIT_DATA", consumed: false },
    { key: "customer_population", value: "RETAIL", consumed: false },
    { key: "loan_type", value: "PERSONAL", consumed: false },
  ],

  // Declared context -> backend dimensions:
  //
  //   regulated_entity_type = NBFC      -> entity_type: "NBFC"
  //   nbfc_layer            = MIDDLE    -> nbfc_layer: "Middle"
  //   digital_vs_physical   = DIGITAL   -> digital_lending: true
  //   third_party_dependency= FALSE     -> uses_external_model_vendor: false
  //
  // microfinance is intentionally absent -- see above.
  profile: {
    entity_type: "NBFC",
    nbfc_layer: "Middle",
    digital_lending: true,
    uses_external_model_vendor: false,
  },
};

/**
 * Declared contexts by model id. A model with no entry here has no declared
 * context, and its assessment stays UNCLEAR. That is the default, and it is
 * the honest one.
 */
export const ASSESSMENT_CONTEXTS = {
  "synthetic-bank-credit-v1": SYNTHETIC_BANK_CONTEXT,
  "german-credit-logistic-regression": GERMAN_CREDIT_CONTEXT,
  "german-credit-random-forest": GERMAN_CREDIT_CONTEXT,
};

/** The declared context for a model, or null when none was declared. */
export function getAssessmentContext(modelId) {
  if (!modelId) return null;
  return ASSESSMENT_CONTEXTS[modelId] || null;
}

/**
 * The backend query parameters for a declared context, or an empty object.
 *
 * Returns only what was declared. It never invents a dimension, and never
 * substitutes a default for an absent one.
 */
export function toEntityProfileParams(context) {
  if (!context || !context.profile) return {};
  return { ...context.profile };
}
