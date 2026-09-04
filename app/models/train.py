"""CLI training entry point for credit risk model (owner: Namitha).

Run via:
    python -m app.models.train
"""

import sys
from app.models.model import train, DEFAULT_DATASET_PATH, DEFAULT_MODEL_ARTIFACT_PATH


def main() -> int:
    """Execute training pipeline and display results."""
    print("=" * 60)
    print("AI Model Risk & Assurance Copilot - Model Training")
    print("Owner: Namitha (Model / Data)")
    print(f"Dataset:  {DEFAULT_DATASET_PATH}")
    print(f"Artifact: {DEFAULT_MODEL_ARTIFACT_PATH}")
    print("=" * 60)

    try:
        result = train(
            dataset_path=DEFAULT_DATASET_PATH,
            save_path=DEFAULT_MODEL_ARTIFACT_PATH,
            random_state=42,
        )
    except Exception as e:
        print(f"\n[ERROR] Model training failed: {e}", file=sys.stderr)
        return 1

    metrics = result["metrics"]
    print("\nTraining completed successfully!")
    print(f"Train samples: {result['n_train_samples']}")
    print(f"Test samples:  {result['n_test_samples']}")
    print("\nEvaluation Metrics (Held-out Test Set):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nArtifact saved to: {DEFAULT_MODEL_ARTIFACT_PATH}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
