"""CLI entrypoint for running the end-to-end model assurance pipeline.

Run via:
    python run_assurance.py
"""
import argparse
import json
import sys
from typing import List, Optional

from app.api.orchestration import (
    build_assurance_result,
    summarize,
)


def main(argv: Optional[List[str]] = None) -> int:
    """Execute the end-to-end assurance pipeline and print a console report."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end AI model risk and assurance evaluation for RBI compliance."
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        metavar="PATH",
        help="Optional path to export full assurance result as JSON.",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Run a live LR-vs-RF drift comparison and print both models' results.",
    )
    args = parser.parse_args(argv)

    if args.compare_models:
        from app.api.orchestration import build_drift_comparison

        print("=" * 60)
        print("AI Model Risk & Assurance Copilot - Cross-Model Drift Comparison")
        print("=" * 60)
        try:
            comparison = build_drift_comparison()
            print(f"\nComparability: {comparison['comparability']}")
            if comparison.get("reason"):
                print(f"Reason: {comparison['reason']}")
            for label, envelope in (
                ("Model A", comparison["drift_a"]),
                ("Model B", comparison["drift_b"]),
            ):
                ctx = envelope["context"]
                result = envelope["result"]
                print(f"\n{label}: {ctx['model_id']} (v{ctx['model_version']})")
                print(f"  Status: {result['status']}")
                print(f"  PSI: {result['psi']:.4f}  KS: {result['ks_statistic']:.4f}")
            print("=" * 60)
            return 0
        except Exception as exc:
            print(f"[ERROR] Drift comparison failed: {exc}", file=sys.stderr)
            return 1

    print("=" * 60)
    print("AI Model Risk & Assurance Copilot - Assurance Report")
    print("Owner: Khushi (API / Dashboard / Integration)")
    print("=" * 60)

    try:
        result = build_assurance_result()
        summary = summarize(result)

        report_status = "UNAVAILABLE (fallback to mock)"
        try:
            from app.report import generate_report

            _ = generate_report(
                model=result["model"],
                explainability=result["explainability"],
                fairness=result["fairness_drift"]["fairness"],
                drift=result["fairness_drift"]["drift"],
                compliance=result["compliance"],
            )
            report_status = "GENERATED"
        except Exception:
            report_status = "UNAVAILABLE (fallback to mock)"

        print("\nPer-Domain Status:")
        print(f"  Model:          {summary['model']}")
        print(f"  Explainability: {summary['explainability']}")
        print(f"  Fairness:       {summary['fairness']}")
        print(f"  Drift:          {summary['drift']}")
        print(f"  Compliance:     {summary['compliance']}")
        print(f"  Report:         {report_status}")

        print("\n" + "-" * 60)
        print("Disclaimers & Scenario Notes:")
        print("  Drift Detection:")
        print(
            "    Drift compares the development training split (reference) with the "
            "held-out test split (current); this is not production monitoring data."
        )
        print("\n  Compliance Evaluation:")
        print(
            "    Compliance findings are evaluated against illustrative sample RBI "
            "rules, not verified regulatory text (is_mock: True)."
        )
        print("=" * 60)

        if args.json_path:
            with open(args.json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"\nAssurance result JSON exported to: {args.json_path}")

        return 0
    except Exception as exc:
        print(f"[ERROR] Assurance evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
