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
    args = parser.parse_args(argv)

    print("=" * 60)
    print("AI Model Risk & Assurance Copilot - Assurance Report")
    print("Owner: Khushi (API / Dashboard / Integration)")
    print("=" * 60)

    try:
        result = build_assurance_result()
        summary = summarize(result)

        print("\nPer-Domain Status:")
        print(f"  Model:          {summary['model']}")
        print(f"  Explainability: {summary['explainability']}")
        print(f"  Fairness:       {summary['fairness']}")
        print(f"  Drift:          {summary['drift']}")
        print(f"  Compliance:     {summary['compliance']}")

        print("\n" + "-" * 60)
        print("Disclaimers & Scenario Notes:")
        print("  Drift Scenario:")
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
