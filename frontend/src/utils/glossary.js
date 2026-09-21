/**
 * Plain-language meanings for the vocabulary this platform reports.
 *
 * WHY THIS FILE EXISTS
 *     The backend speaks precisely: PASS, EVIDENCE_MISSING,
 *     APPLICABILITY_UNCLEAR, "disparate impact ratio", "population stability
 *     index". Every one of those is the correct technical term, and none of
 *     them means anything to a business reader seeing the dashboard for the
 *     first time.
 *
 *     This is the single place those terms are translated. Pages import from
 *     here rather than writing their own wording, so the same status never
 *     gets two different explanations on two different screens.
 *
 * WHAT THIS FILE IS NOT
 *     It is display text only. Nothing here computes, re-derives, rounds,
 *     re-classifies or re-orders a value. A status arrives already decided by
 *     the backend; this only says what that decision means in English.
 *
 *     The technical token is never thrown away either -- every helper keeps it
 *     available, because a model-risk reviewer needs the precise term and a
 *     business stakeholder needs the plain one. They get both.
 */
import statusMeaningData from "./statusMeaning.json";

/**
 * One entry per status token the backend can emit.
 *
 *   label    what a non-technical reader sees
 *   meaning  one sentence answering "what does this tell me?"
 *   glyph    a text symbol, so status is never carried by colour alone
 *   tone     which visual treatment applies
 *
 * The data itself lives in statusMeaning.json, NOT inline here. That file is
 * the single source of truth for status labels across both renderers this
 * platform has -- this web app AND the PDF assurance report
 * (app/report/status_meaning.py loads the same JSON file directly). A label
 * change belongs in the JSON only; changing it here would silently leave
 * the PDF on the old wording.
 */
export const STATUS_MEANING = statusMeaningData;

/** Visual treatment per tone. Colour is a reinforcement, never the only signal. */
const TONE_STYLES = {
  pass: "badge-success",
  warning: "badge-warning",
  fail: "badge-error",
  info: "badge-info",
  neutral: "badge-ghost border-base-300",
};

/**
 * Everything a badge needs for one status token.
 *
 * An unrecognised token is passed through rather than hidden: showing an
 * unknown value plainly is safer than silently rendering it as neutral and
 * letting a reader assume it is benign.
 */
export function describeStatus(status) {
  if (status === null || status === undefined || status === "") {
    return {
      token: null,
      label: "Unknown",
      meaning: "No status was reported for this item.",
      glyph: "?",
      style: TONE_STYLES.neutral,
    };
  }
  const token = String(status);
  const entry = STATUS_MEANING[token] || STATUS_MEANING[token.toUpperCase()];
  if (!entry) {
    return {
      token,
      label: token.replace(/_/g, " ").toLowerCase(),
      meaning: `Reported status: ${token}.`,
      glyph: "•",
      style: TONE_STYLES.neutral,
    };
  }
  return {
    token,
    label: entry.label,
    meaning: entry.meaning,
    glyph: entry.glyph,
    style: TONE_STYLES[entry.tone] || TONE_STYLES.neutral,
  };
}

/** Whether a status is one a person should act on. */
export function needsAttention(status) {
  const token = String(status ?? "").toUpperCase();
  return token === "FAIL" || token === "WARNING";
}

/**
 * Plain-language definitions for the domain terms shown on screen.
 *
 * Each entry pairs the precise term with what it actually tells you. The
 * technical name stays on screen -- these are shown through a help control
 * next to it, not as a replacement.
 */
export const TERM_GLOSSARY = {
  explainability:
    "Which pieces of information the model leaned on when it made its decisions, and how strongly each one pushed the answer one way or the other.",
  fairness:
    "Whether the model's favourable decisions are spread evenly across different groups of people, using the group attribute your organisation declared as protected.",
  "feature drift":
    "Whether the kind of applicants coming through has changed compared with an earlier reference period. This is about the incoming data, not the model.",
  "prediction drift":
    "Whether the model's own answers have shifted compared with an earlier period — for example, flagging a noticeably higher share of applicants as high risk.",
  monitoring:
    "Comparing a recent period against an earlier reference period to see whether anything has moved: the incoming data, the model's answers, or the fairness of its outcomes.",
  "protected attribute":
    "The characteristic that fairness is measured across — for example region or an applicant-status field. Your organisation declares this; the platform never guesses it.",
  "disparate impact ratio":
    "Compares the group least likely to get a favourable decision with the group most likely to. A value of 1.00 means every group is treated equally; the further below 1.00, the larger the gap.",
  "demographic parity difference":
    "The simple gap between the highest and lowest favourable-decision rates across groups. Reported for information; no pass or fail is attached to it on its own.",
  "population stability index":
    "A standard banking measure of how far a distribution has moved between two periods. Larger numbers mean a bigger shift.",
  psi: "A standard banking measure of how far a distribution has moved between two periods. Larger numbers mean a bigger shift.",
  "ks statistic":
    "Another measure of how far two distributions differ. It is reported for information only — no pass or fail threshold is attached to it.",
  accuracy: "Out of all the cases tested, the share the model got right.",
  precision:
    "When the model flags a case as high risk, how often it turns out to be right.",
  recall:
    "Out of all the genuinely high-risk cases, the share the model successfully flagged.",
  f1: "A single score balancing precision and recall, useful when you care about both.",
  "roc auc":
    "How well the model separates high-risk from low-risk cases overall. 1.0 would be perfect separation; 0.5 would be no better than guessing.",
  "assurance run":
    "One complete assessment of one model at one point in time. Every finding on these pages belongs to the run identified here, so results from different runs or different models can never be confused.",
  provenance:
    "Where the data being examined came from — real observed activity, generated test data, or unstated. If it was not stated, we show it as unstated rather than assuming it was real.",
  attestation:
    "A requirement that can only be evidenced by your organisation's own records. The platform cannot verify it from model behaviour, so it reports it as needing evidence rather than passing or failing it.",
  "reference window":
    "The earlier period used as the baseline for comparison.",
  "current window": "The more recent period being checked against the baseline.",
};

/** Look up a glossary term, case- and spacing-insensitively. */
export function defineTerm(term) {
  if (!term) return null;
  const key = String(term).toLowerCase().replace(/_/g, " ").trim();
  return TERM_GLOSSARY[key] || null;
}
