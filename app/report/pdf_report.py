"""PDF assurance evidence report (Owner: Khushi).

Renders the SAME result object build_assurance_result() already produces
(model, explainability, fairness_drift, compliance incl. verified_requirements)
as a downloadable PDF. No new computation happens here -- every number in
the document is read straight from that dict, exactly as the JSON endpoints
already serve it. This module is purely a second, additive rendering of
existing data; GET /report and its JSON shape are untouched.

Status display labels (e.g. EVIDENCE_MISSING -> "Awaiting Organizational
Evidence") are read from app/report/status_meaning.py, which loads the SAME
frontend/src/utils/statusMeaning.json the web app's glossary.js imports --
one authored file, two readers, so a label only ever changes in one place.
"""
import io
import os
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.report.status_meaning import describe_status

# ---------------------------------------------------------------------------
# Fonts -- DejaVu Sans, bundled with matplotlib (a direct dependency below),
# zero new system deps. Also matplotlib's own default sans-serif font, so
# chart text and document text match without extra configuration. Registered
# once at import time (pdfmetrics' font registry is process-global).
# ---------------------------------------------------------------------------
_FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVuSans", os.path.join(_FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", os.path.join(_FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSansMono", os.path.join(_FONT_DIR, "DejaVuSansMono.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSansMono-Bold", os.path.join(_FONT_DIR, "DejaVuSansMono-Bold.ttf")))

# ---------------------------------------------------------------------------
# Palette (shared between ReportLab flowables and matplotlib charts)
# ---------------------------------------------------------------------------
INK = colors.HexColor("#1e293b")
MUTED = colors.HexColor("#64748b")
BORDER = colors.HexColor("#cbd5e1")
CARD_BG = colors.HexColor("#f8fafc")
ACCENT_BLUE = colors.HexColor("#1d4ed8")
ACCENT_GRAY = colors.HexColor("#94a3b8")
STATUS_TEXT_COLOR = {
    "pass": colors.HexColor("#15803d"),
    "warning": colors.HexColor("#b45309"),
    "fail": colors.HexColor("#b91c1c"),
    "neutral": MUTED,
    "info": ACCENT_BLUE,
}
MPL_INK = "#1e293b"
MPL_MUTED = "#64748b"
MPL_BLUE = "#1d4ed8"
MPL_GRID = "#e2e8f0"

plt.rcParams["font.family"] = "DejaVu Sans"

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = {
    "DocTitle": ParagraphStyle(
        "DocTitle", fontName="DejaVuSans-Bold", fontSize=25, textColor=INK,
        leading=30, spaceAfter=6,
    ),
    "DocSubtitle": ParagraphStyle(
        "DocSubtitle", fontName="DejaVuSans", fontSize=12, textColor=MUTED,
        leading=17, spaceAfter=4,
    ),
    "CoverLabel": ParagraphStyle(
        "CoverLabel", fontName="DejaVuSans-Bold", fontSize=7.5, textColor=MUTED,
        leading=10,
    ),
    "CoverValue": ParagraphStyle(
        "CoverValue", fontName="DejaVuSans", fontSize=9.5, textColor=INK,
        leading=13,
    ),
    "SectionHeading": ParagraphStyle(
        "SectionHeading", fontName="DejaVuSans-Bold", fontSize=16,
        textColor=INK, spaceAfter=4,
    ),
    "SectionSubtitle": ParagraphStyle(
        "SectionSubtitle", fontName="DejaVuSans", fontSize=9.5,
        textColor=MUTED, leading=13, spaceAfter=14,
    ),
    "SubHeading": ParagraphStyle(
        "SubHeading", fontName="DejaVuSans-Bold", fontSize=11,
        textColor=INK, spaceBefore=14, spaceAfter=6,
    ),
    "BodyMuted": ParagraphStyle(
        "BodyMuted", fontName="DejaVuSans", fontSize=8.3, textColor=MUTED, leading=12,
    ),
    "TableHeader": ParagraphStyle(
        "TableHeader", fontName="DejaVuSans-Bold", fontSize=7.3,
        textColor=MUTED, leading=9,
    ),
    "TableCell": ParagraphStyle(
        "TableCell", fontName="DejaVuSans", fontSize=8.3, textColor=INK, leading=11.5,
    ),
    "TableCellMono": ParagraphStyle(
        "TableCellMono", fontName="DejaVuSansMono", fontSize=7.6, textColor=INK, leading=11,
    ),
    "StatNumber": ParagraphStyle(
        "StatNumber", fontName="DejaVuSans-Bold", fontSize=20, textColor=ACCENT_BLUE,
        leading=24,
    ),
    "StatLabel": ParagraphStyle(
        "StatLabel", fontName="DejaVuSans-Bold", fontSize=7.3, textColor=MUTED,
        leading=10, spaceAfter=2,
    ),
    "NoteBox": ParagraphStyle(
        "NoteBox", fontName="DejaVuSans-Oblique", fontSize=8.5, textColor=MUTED,
        leading=12.5,
    ),
    "ReqID": ParagraphStyle(
        "ReqID", fontName="DejaVuSansMono-Bold", fontSize=9, textColor=INK,
    ),
    "Pill": ParagraphStyle(
        "Pill", fontName="DejaVuSans-Bold", fontSize=7.5, textColor=colors.white,
        alignment=1, leading=9,
    ),
    "PillNeutral": ParagraphStyle(
        "PillNeutral", fontName="DejaVuSans-Bold", fontSize=7.5, textColor=INK,
        alignment=1, leading=9,
    ),
    "ReqTitle": ParagraphStyle(
        "ReqTitle", fontName="DejaVuSans-Bold", fontSize=11, textColor=INK,
        leading=14, spaceBefore=6, spaceAfter=6,
    ),
    "MetaLabel": ParagraphStyle(
        "MetaLabel", fontName="DejaVuSans-Bold", fontSize=6.7, textColor=MUTED, leading=9,
    ),
    "MetaValue": ParagraphStyle(
        "MetaValue", fontName="DejaVuSans", fontSize=8.5, textColor=INK, leading=11,
    ),
    "Quote": ParagraphStyle(
        "Quote", fontName="DejaVuSans-Oblique", fontSize=8.7, textColor=INK,
        leading=12.5, leftIndent=10, spaceBefore=10, spaceAfter=10,
    ),
    "FieldLabelNeutral": ParagraphStyle(
        "FieldLabelNeutral", fontName="DejaVuSans-Bold", fontSize=7, textColor=MUTED,
        leading=9, spaceBefore=10, spaceAfter=3,
    ),
    "FieldLabelAccent": ParagraphStyle(
        "FieldLabelAccent", fontName="DejaVuSans-Bold", fontSize=7, textColor=ACCENT_BLUE,
        leading=9, spaceBefore=10, spaceAfter=3,
    ),
    "FieldText": ParagraphStyle(
        "FieldText", fontName="DejaVuSans", fontSize=8.7, textColor=INK, leading=12.5,
    ),
    "FieldTextMuted": ParagraphStyle(
        "FieldTextMuted", fontName="DejaVuSans", fontSize=8.3, textColor=MUTED, leading=12,
    ),
    "SourceLink": ParagraphStyle(
        "SourceLink", fontName="DejaVuSansMono", fontSize=7, textColor=ACCENT_BLUE,
        leading=10, spaceBefore=8,
    ),
}

PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 40 * mm  # left+right margins of 20mm each


# ---------------------------------------------------------------------------
# Header / footer
# ---------------------------------------------------------------------------
def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVuSans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, PAGE_H - 13 * mm, "AI Model Risk & Assurance Copilot")
    canvas.drawRightString(PAGE_W - 20 * mm, PAGE_H - 13 * mm, "Assurance Evidence Report")
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, PAGE_H - 15 * mm, PAGE_W - 20 * mm, PAGE_H - 15 * mm)

    canvas.line(20 * mm, 14 * mm, PAGE_W - 20 * mm, 14 * mm)
    canvas.drawString(
        20 * mm, 9 * mm,
        "Illustrative technical checks and attestation-only RBI requirements — "
        "not a regulatory compliance certificate.",
    )
    canvas.drawRightString(PAGE_W - 20 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Chart helpers (matplotlib -> PNG -> ReportLab Image)
# ---------------------------------------------------------------------------
def _finish_chart(fig, ax, width_in=6.3):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MPL_GRID)
    ax.spines["bottom"].set_color(MPL_GRID)
    ax.tick_params(colors=MPL_INK, labelsize=8.5)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    aspect = img.imageHeight / float(img.imageWidth)
    img.drawWidth = width_in * 72
    img.drawHeight = img.drawWidth * aspect
    return img


def _chart_horizontal_bar(labels, values, xlabel):
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(labels) + 0.6))
    y = range(len(labels))
    ax.barh(list(y), values, color=MPL_BLUE, height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5, color=MPL_INK)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=8.5, color=MPL_MUTED)
    ax.xaxis.grid(True, color=MPL_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return _finish_chart(fig, ax)


def _chart_group_bar(labels, values, ylabel):
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    x = range(len(labels))
    ax.bar(list(x), values, color=MPL_BLUE, width=0.55)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9, color=MPL_INK)
    ax.set_ylabel(ylabel, fontsize=8.5, color=MPL_MUTED)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, color=MPL_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=8, color=MPL_INK)
    return _finish_chart(fig, ax, width_in=6.6)


def _zebra_table(header, rows, col_widths, header_style="TableHeader", cell_style="TableCell"):
    data = [[Paragraph(h, styles[header_style]) for h in header]]
    for row in rows:
        data.append([c if isinstance(c, Paragraph) else Paragraph(str(c), styles[cell_style]) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, BORDER),
        ("LINEBELOW", (0, -1), (-1, -1), 0.75, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), CARD_BG))
    t.setStyle(TableStyle(style))
    return t


def _stat_card(number_text, label_text, width):
    t = Table(
        [[Paragraph(number_text, styles["StatNumber"])], [Paragraph(label_text, styles["StatLabel"])]],
        colWidths=[width],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.75, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (0, 0), 12),
        ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ("TOPPADDING", (0, 1), (0, 1), 0),
        ("BOTTOMPADDING", (0, 1), (0, 1), 12),
    ]))
    return t


def _pill(text, bg, style_key="Pill"):
    t = Table([[Paragraph(text, styles[style_key])]], colWidths=[None])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _status_pill_text(token):
    meaning = describe_status(token)
    return f"{meaning['glyph']}  {meaning['label']}"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _cover_page(data):
    story = [Spacer(1, 60 * mm)]
    story.append(Paragraph("AI Model Assurance", styles["DocTitle"]))
    story.append(Paragraph("Evidence Report", styles["DocTitle"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Technical assurance results and attestation-only RBI regulatory requirements, "
        "generated directly from computed pipeline outputs.",
        styles["DocSubtitle"],
    ))
    story.append(Spacer(1, 22))

    model_meta = data["model"]["model_metadata"]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = [
        ("MODEL", model_meta.get("model_type", "unknown")),
        ("MODEL VERSION", model_meta.get("version", "unknown")),
        ("TRAINED ON", model_meta.get("trained_on", "unknown")),
        ("ASSURANCE RUN ID", data.get("assurance_run_id", "unknown")),
        ("GENERATED", generated_at),
    ]
    cover_rows = [[Paragraph(k, styles["CoverLabel"]), Paragraph(str(v), styles["CoverValue"])] for k, v in rows]
    t = Table(cover_rows, colWidths=[45 * mm, CONTENT_W - 45 * mm])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(t)
    story.append(PageBreak())
    return story


def _model_meta_label_line(data):
    sem = data["model"]["model_metadata"].get("label_semantics", {})
    return f"{sem.get('0', 'class 0')} · {sem.get('1', 'class 1')} — probabilities represent {sem.get('probabilities_represent', 'n/a')}"


def _model_overview_section(data):
    metrics = data["model"]["model_metrics"]
    story = [
        Paragraph("Model Overview", styles["SectionHeading"]),
        Paragraph(
            "Held-out evaluation metrics computed directly on the model's test split. "
            "Reported as unavailable, never substituted, for any model where this split has no meaning.",
            styles["SectionSubtitle"],
        ),
    ]
    if metrics:
        header = ["METRIC", "VALUE"]
        rows = [
            ("Accuracy", f"{metrics['accuracy']:.2%}"),
            ("Precision", f"{metrics['precision']:.2%}"),
            ("Recall", f"{metrics['recall']:.2%}"),
            ("F1 score", f"{metrics['f1']:.2%}"),
            ("ROC AUC", f"{metrics['roc_auc']:.3f}" if metrics.get("roc_auc_status") == "computed" else "unavailable"),
            ("Test sample count", str(metrics["n_test_samples"])),
        ]
        story.append(_zebra_table(header, rows, [90 * mm, CONTENT_W - 90 * mm]))
    else:
        story.append(Paragraph("Model metrics unavailable for this model's schema.", styles["BodyMuted"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Label semantics: " + _model_meta_label_line(data), styles["BodyMuted"]))
    return story


def _explainability_section(data):
    exp = data["explainability"]
    model = data["model"]
    story = [
        Paragraph("Explainability", styles["SectionHeading"]),
        Paragraph(
            f"Global and per-instance feature attribution via {exp['method'].upper()}, computed on "
            f"{len(exp['per_instance'])} held-out instances.",
            styles["SectionSubtitle"],
        ),
    ]

    gi = sorted(exp["global_importance"].items(), key=lambda kv: kv[1], reverse=True)[:12]
    labels = [k for k, _ in gi][::-1]
    values = [v for _, v in gi][::-1]
    story.append(Paragraph("Global feature importance (top 12, mean |contribution|)", styles["SubHeading"]))
    story.append(_chart_horizontal_bar(labels, values, "Mean |SHAP contribution|"))

    if exp["per_instance"]:
        story.append(Paragraph("Sampled instances (predicted-probability extremes)", styles["SubHeading"]))
        story.append(Paragraph(
            f"5 of {len(exp['per_instance'])} held-out instances, selected by predicted-probability "
            "extremes: the 2 most confidently predicted GOOD, the 2 most confidently predicted BAD, "
            "and the 1 closest to the decision boundary. Labeled by instance ID; not a random or "
            "representative sample.",
            styles["BodyMuted"],
        ))
        story.append(Spacer(1, 6))

        probs = model["probabilities"]
        preds = model["predictions"]
        ids = model["instance_ids"]
        per_instance = exp["per_instance"]
        order = sorted(range(len(probs)), key=lambda i: probs[i])
        selected = list(dict.fromkeys(
            order[:2] + order[-2:] + [min(range(len(probs)), key=lambda i: abs(probs[i] - 0.5))]
        ))[:5]

        header = ["INSTANCE", "PRED.", "P(BAD)", "TOP CONTRIBUTING FEATURES"]
        rows = []
        for idx in selected:
            contrib = per_instance[idx]["contributions"]
            top3 = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            top3_text = "; ".join(f"{k} ({v:+.2f})" for k, v in top3)
            rows.append([
                Paragraph(ids[idx], styles["TableCellMono"]),
                Paragraph("BAD" if preds[idx] == 1 else "GOOD", styles["TableCell"]),
                Paragraph(f"{probs[idx]:.2f}", styles["TableCell"]),
                Paragraph(top3_text, styles["TableCell"]),
            ])
        story.append(_zebra_table(header, rows, [28 * mm, 16 * mm, 20 * mm, CONTENT_W - 64 * mm]))
    return story


def _fairness_section(data):
    fairness = data["fairness_drift"]["fairness"]
    story = [
        Paragraph("Fairness", styles["SectionHeading"]),
        Paragraph(
            f"Group fairness computed against the declared protected attribute "
            f"'{fairness['protected_attribute']}'.",
            styles["SectionSubtitle"],
        ),
    ]
    stats_row = Table(
        [[
            _stat_card(f"{fairness['demographic_parity_diff']:.2f}", "DEMOGRAPHIC PARITY DIFF.", CONTENT_W / 2 - 4),
            _stat_card(f"{fairness['disparate_impact_ratio']:.2f}", "DISPARATE IMPACT RATIO", CONTENT_W / 2 - 4),
        ]],
        colWidths=[CONTENT_W / 2, CONTENT_W / 2],
    )
    stats_row.setStyle(TableStyle([("LEFTPADDING", (1, 0), (1, 0), 8)]))
    story.append(stats_row)
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Overall status: {fairness['status']}", styles["BodyMuted"]))
    story.append(Spacer(1, 10))

    groups = fairness["groups"]
    if groups:
        story.append(Paragraph("Selection rate by group", styles["SubHeading"]))
        story.append(_chart_group_bar([g["group"] for g in groups], [g["selection_rate"] for g in groups], "Selection rate"))

        header = ["GROUP", "COUNT", "FAVOURABLE COUNT", "SELECTION RATE"]
        rows = [[g["group"], str(g["count"]), str(g["favorable_count"]), f"{g['selection_rate']:.2%}"] for g in groups]
        story.append(_zebra_table(header, rows, [30 * mm, 30 * mm, 40 * mm, CONTENT_W - 100 * mm]))
    return story


def _drift_section(data):
    drift = data["fairness_drift"]["drift"]
    story = [
        Paragraph("Drift", styles["SectionHeading"]),
        Paragraph(
            "Feature/data drift compares the training reference distribution to the held-out "
            "current split.",
            styles["SectionSubtitle"],
        ),
    ]
    per_feature = sorted(drift["per_feature"], key=lambda f: f["psi"], reverse=True)
    if per_feature:
        story.append(Paragraph("Feature drift (PSI, all evaluated features)", styles["SubHeading"]))
        story.append(_chart_horizontal_bar(
            [f["feature"] for f in per_feature][::-1], [f["psi"] for f in per_feature][::-1],
            "Population Stability Index (PSI)",
        ))

        header = ["FEATURE", "PSI", "KS STATISTIC"]
        rows = [[f["feature"], f"{f['psi']:.4f}", f"{f['ks_statistic']:.4f}"] for f in per_feature]
        story.append(_zebra_table(header, rows, [60 * mm, 40 * mm, CONTENT_W - 100 * mm]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"Overall: PSI {drift['psi']:.4f}, KS {drift['ks_statistic']:.4f} — status {drift['status']}",
            styles["BodyMuted"],
        ))

    story.append(Spacer(1, 14))
    story.append(Paragraph("Prediction / output drift", styles["SubHeading"]))
    reason = data.get("monitoring_unavailable_reason")
    monitoring = data.get("monitoring")
    if reason or not monitoring:
        note_text = reason or "Prediction drift unavailable for this run."
        note = Table([[Paragraph(note_text, styles["NoteBox"])]], colWidths=[CONTENT_W])
        note.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
            ("BOX", (0, 0), (-1, -1), 0.75, BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(note)
    else:
        # Available when the request identified a specific model_id -- an
        # adapter is required to score reference/current windows. See
        # _monitoring_for_assurance() in app/api/orchestration.py.
        pd = monitoring["result"]["prediction_drift"]
        story.append(Paragraph(
            "Reference vs. current window comparison of the model's own outputs -- "
            "available because this run identifies a specific model.",
            styles["BodyMuted"],
        ))
        story.append(Spacer(1, 6))
        stats_row = Table(
            [[
                _stat_card(f"{pd['label_psi']:.4f}", "PREDICTED-LABEL PSI", CONTENT_W / 3 - 5),
                _stat_card(f"{pd['score_psi']:.4f}", "SCORE PSI", CONTENT_W / 3 - 5),
                _stat_card(f"{pd['score_ks_statistic']:.4f}", "SCORE KS STATISTIC", CONTENT_W / 3 - 5),
            ]],
            colWidths=[CONTENT_W / 3, CONTENT_W / 3, CONTENT_W / 3],
        )
        stats_row.setStyle(TableStyle([
            ("LEFTPADDING", (1, 0), (1, 0), 8),
            ("LEFTPADDING", (2, 0), (2, 0), 8),
        ]))
        story.append(stats_row)
        story.append(Spacer(1, 6))
        header = ["PREDICTED CLASS", "REFERENCE RATE", "CURRENT RATE"]
        rows = [[str(c["class"]), f"{c['reference_rate']:.2%}", f"{c['current_rate']:.2%}"] for c in pd["per_class"]]
        story.append(_zebra_table(header, rows, [50 * mm, 50 * mm, CONTENT_W - 100 * mm]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"Overall status: {pd['status']}", styles["BodyMuted"]))
    return story


def _technical_checks_section(data):
    findings = data["compliance"]["findings"]
    story = [
        Paragraph("Technical Assurance Checks", styles["SectionHeading"]),
        Paragraph(
            "Illustrative sample technical checks derived from computed pipeline results "
            "(is_mock=True rule text; not verified RBI regulatory language).",
            styles["SectionSubtitle"],
        ),
    ]
    if findings:
        header = ["RULE ID", "DESCRIPTION", "STATUS"]
        rows = []
        for f in findings:
            meaning = describe_status(f["status"])
            status_style = ParagraphStyle(
                f"Status_{f['status']}", parent=styles["TableCell"], fontName="DejaVuSans-Bold",
                textColor=STATUS_TEXT_COLOR.get(meaning["tone"], INK),
            )
            rows.append([
                Paragraph(f["rule_id"], styles["TableCellMono"]),
                Paragraph(f["rule_description"], styles["TableCell"]),
                Paragraph(meaning["label"], status_style),
            ])
        story.append(_zebra_table(header, rows, [32 * mm, CONTENT_W - 32 * mm - 26 * mm, 26 * mm]))
    return story


def _requirement_card(req):
    applicability_meaning = describe_status(req["applicability"])
    accent = ACCENT_BLUE if applicability_meaning["tone"] == "info" else ACCENT_GRAY
    applicability_pill = _pill(
        _status_pill_text(req["applicability"]), accent,
        "Pill" if applicability_meaning["tone"] == "info" else "PillNeutral",
    )
    status_pill = _pill(_status_pill_text(req["status"]), colors.HexColor("#e2e8f0"), "PillNeutral")

    header_row = Table(
        [[Paragraph(req["requirement_id"], styles["ReqID"]), applicability_pill, status_pill, ""]],
        colWidths=[145, None, None, None],
    )
    header_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 8),
        ("RIGHTPADDING", (1, 0), (1, 0), 6),
    ]))

    meta = Table(
        [
            [Paragraph("INSTRUMENT", styles["MetaLabel"]), Paragraph("CLAUSE", styles["MetaLabel"]), Paragraph("ASSESSMENT MODE", styles["MetaLabel"])],
            [Paragraph(req["document_title"], styles["MetaValue"]), Paragraph(req["clause"], styles["MetaValue"]), Paragraph(req["assessment_mode"], styles["MetaValue"])],
        ],
        colWidths=[240, 60, 130],
    )
    meta.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ("TOPPADDING", (0, 1), (-1, 1), 0),
    ]))

    # IMPORTANT: one flowable per ROW, not a list of flowables nested inside
    # one cell -- ReportLab cannot reliably compute the height of a table
    # cell that itself contains many stacked flowables, which under/over-
    # estimates during pagination. A single-column table with one row per
    # element (background/border/accent applied across the whole row span
    # via TableStyle) lets ReportLab size and split each row correctly.
    rows = [[header_row], [Spacer(1, 6)], [Paragraph(req["requirement"], styles["ReqTitle"])], [meta]]
    if req.get("quote"):
        rows.append([Paragraph(f"“{req['quote']}”", styles["Quote"])])
    rows.append([Paragraph("WHY", styles["FieldLabelNeutral"])])
    rows.append([Paragraph(req["reason"], styles["FieldText"])])
    if req.get("suggestion"):
        rows.append([Paragraph("NEXT STEP", styles["FieldLabelAccent"])])
        rows.append([Paragraph(req["suggestion"], styles["FieldText"])])
    if req.get("limitation"):
        rows.append([Paragraph("LIMITATION", styles["FieldLabelNeutral"])])
        rows.append([Paragraph(req["limitation"], styles["FieldTextMuted"])])
    if req.get("source_url"):
        rows.append([Paragraph(req["source_url"], styles["SourceLink"])])

    card = Table(rows, colWidths=[470])
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.75, BORDER),
        ("LINEBEFORE", (0, 0), (0, -1), 3, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 12),
    ]))
    # KeepTogether prevents a card splitting mid-paragraph across a page
    # boundary; safe here because the height-miscalculation bug (above) is
    # already fixed by the one-flowable-per-row layout.
    return [KeepTogether([card, Spacer(1, 14)])]


def _rbi_section(requirements):
    story = [
        Paragraph("RBI Regulatory Requirements", styles["SectionHeading"]),
    ]
    if not requirements:
        story.append(Paragraph(
            "A curated set of verified RBI requirements, each read from an official RBI instrument.",
            styles["SectionSubtitle"],
        ))
        note = Table([[Paragraph(
            "No RBI requirements assessed for this request. The regulatory layer is assessed "
            "against a declared entity profile — entity type, NBFC layer, digital lending, "
            "external vendor use. None was declared for this request, and applicability is "
            "never guessed from a model or its data.",
            styles["NoteBox"],
        )]], colWidths=[CONTENT_W])
        note.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
            ("BOX", (0, 0), (-1, -1), 0.75, BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(note)
        return story

    total = len(requirements)
    applies = sum(1 for r in requirements if r["applicability"] == "APPLIES")
    not_applicable = sum(1 for r in requirements if r["applicability"] == "NOT_APPLICABLE")
    unclear = total - applies - not_applicable
    story.append(Paragraph(
        "A curated set of verified RBI requirements, each read from an official RBI instrument "
        "and carrying its exact clause reference and source link. These are not compliance "
        "results — every requirement here requires evidence held by the regulated entity, so "
        "none can be satisfied by model analytics and none returns a pass. "
        f"{total} requirements assessed: {applies} apply, {not_applicable} do not apply, "
        f"{unclear} applicability unclear.",
        styles["SectionSubtitle"],
    ))
    for req in requirements:
        story.extend(_requirement_card(req))
    return story


def build_pdf_report(data: dict) -> bytes:
    """Render an assurance-result dict (build_assurance_result()'s shape,
    with compliance.verified_requirements attached) as a PDF, returning the
    raw bytes. No computation happens here -- purely presentation of data
    already computed by the same pipeline GET /assurance-result serves.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=22 * mm, bottomMargin=20 * mm, leftMargin=20 * mm, rightMargin=20 * mm,
    )
    story = []
    story += _cover_page(data)
    story += _model_overview_section(data)
    story.append(PageBreak())
    story += _explainability_section(data)
    story.append(PageBreak())
    story += _fairness_section(data)
    story.append(PageBreak())
    story += _drift_section(data)
    story.append(PageBreak())
    story += _technical_checks_section(data)
    story.append(PageBreak())
    story += _rbi_section(data["compliance"].get("verified_requirements", []))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return buf.getvalue()
