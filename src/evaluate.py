"""
Evaluation module for the ESG Controversy Detection System.

Produces comprehensive evaluation reports including:
    - Confusion matrix across all 9 categories
    - Per-class precision, recall, and F1
    - Macro and weighted F1 scores
    - Comparison against TF-IDF + Logistic Regression baseline
    - Misclassification analysis with example articles per error type
    - All outputs saved to reports/ directory

Usage:
    python -m src.evaluate                    # Evaluate with synthetic data
    python -m src.evaluate --model-dir PATH   # Evaluate specific model
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class ESGEvaluator:
    """Comprehensive evaluation suite for the ESG controversy classifier.

    Generates confusion matrices, per-class metrics, baseline comparisons,
    and misclassification analyses. All outputs are saved to the reports directory.
    """

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or config.REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Full Evaluation Pipeline
    # ------------------------------------------------------------------
    def run_full_evaluation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        texts: list[str] | None = None,
        confidences: np.ndarray | None = None,
        train_texts: list[str] | None = None,
        train_labels: np.ndarray | None = None,
    ) -> dict:
        """Run the complete evaluation pipeline.

        Args:
            y_true: True labels (integer category IDs).
            y_pred: Predicted labels (integer category IDs).
            texts: Optional list of original texts for misclassification analysis.
            confidences: Optional prediction confidence scores.
            train_texts: Optional training texts for baseline comparison.
            train_labels: Optional training labels for baseline comparison.

        Returns:
            Dictionary containing all evaluation results.
        """
        results = {}

        # 1. Core metrics
        logger.info("Computing core metrics...")
        results["metrics"] = self.compute_metrics(y_true, y_pred)

        # 2. Confusion matrix
        logger.info("Generating confusion matrix...")
        results["confusion_matrix"] = self.plot_confusion_matrix(y_true, y_pred)

        # 3. Per-class report
        logger.info("Generating classification report...")
        results["classification_report"] = self.generate_classification_report(y_true, y_pred)

        # 4. Baseline comparison
        if train_texts is not None and train_labels is not None:
            logger.info("Training TF-IDF baseline...")
            test_texts = texts if texts else None
            results["baseline"] = self.train_baseline(
                train_texts, train_labels, test_texts, y_true
            )
        else:
            logger.info("Skipping baseline (no training data provided)")

        # 5. Misclassification analysis
        if texts is not None:
            logger.info("Analysing misclassifications...")
            results["misclassifications"] = self.analyse_misclassifications(
                y_true, y_pred, texts, confidences
            )

        # Save combined report
        self._save_summary_report(results)
        logger.info("Full evaluation complete. Reports saved to %s", self.output_dir)

        return results

    # ------------------------------------------------------------------
    # Core Metrics
    # ------------------------------------------------------------------
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute macro/weighted precision, recall, and F1.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary of aggregate metrics.
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        logger.info("Core Metrics:")
        for key, value in metrics.items():
            logger.info("  %s: %.4f", key, value)

        return metrics

    # ------------------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Generate and save a confusion matrix heatmap.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Confusion matrix as numpy array.
        """
        cm = confusion_matrix(y_true, y_pred)
        labels = [config.ID_TO_CATEGORY.get(i, str(i)) for i in range(config.NUM_CATEGORIES)]

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(14, 12))

            # Normalise for display
            cm_normalised = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm_normalised = np.nan_to_num(cm_normalised)

            sns.heatmap(
                cm_normalised,
                annot=cm,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                square=True,
                cbar_kws={"label": "Normalised proportion"},
            )

            ax.set_xlabel("Predicted Category", fontsize=12, fontweight="bold")
            ax.set_ylabel("True Category", fontsize=12, fontweight="bold")
            ax.set_title("ESG Controversy Classification — Confusion Matrix", fontsize=14, fontweight="bold")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            output_path = self.output_dir / "confusion_matrix.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Confusion matrix saved to %s", output_path)

        except ImportError as exc:
            logger.warning("Could not generate confusion matrix plot: %s", exc)

        return cm

    # ------------------------------------------------------------------
    # Classification Report
    # ------------------------------------------------------------------
    def generate_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict:
        """Generate a per-class classification report.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Classification report as dict.
        """
        labels = list(range(config.NUM_CATEGORIES))
        target_names = [config.ID_TO_CATEGORY.get(i, str(i)) for i in labels]

        report_dict = classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        # Save as JSON
        output_path = self.output_dir / "classification_report.json"
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        # Also save human-readable version
        report_str = classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        )
        text_path = self.output_dir / "classification_report.txt"
        with open(text_path, "w") as f:
            f.write("ESG Controversy Classifier — Per-Class Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(report_str)

        logger.info("Classification report saved to %s", output_path)
        return report_dict

    # ------------------------------------------------------------------
    # TF-IDF + Logistic Regression Baseline
    # ------------------------------------------------------------------
    def train_baseline(
        self,
        train_texts: list[str],
        train_labels: np.ndarray,
        test_texts: list[str] | None = None,
        test_labels: np.ndarray | None = None,
    ) -> dict:
        """Train a TF-IDF + Logistic Regression baseline for comparison.

        Args:
            train_texts: Training text corpus.
            train_labels: Training labels.
            test_texts: Optional test text corpus.
            test_labels: Optional test labels.

        Returns:
            Dictionary with baseline metrics and comparison.
        """
        logger.info("Training TF-IDF + LogReg baseline (max_features=10000)...")

        # TF-IDF vectorisation
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        X_train = vectorizer.fit_transform(train_texts)

        # Logistic Regression with balanced class weights
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="saga",
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X_train, train_labels)

        results = {"model": "TF-IDF + Logistic Regression"}

        if test_texts is not None and test_labels is not None:
            X_test = vectorizer.transform(test_texts)
            y_pred_baseline = clf.predict(X_test)

            results.update({
                "accuracy": float(accuracy_score(test_labels, y_pred_baseline)),
                "macro_f1": float(f1_score(test_labels, y_pred_baseline, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(test_labels, y_pred_baseline, average="weighted", zero_division=0)),
            })

            logger.info("Baseline Results:")
            for key, value in results.items():
                if isinstance(value, float):
                    logger.info("  %s: %.4f", key, value)

        # Save baseline report
        output_path = self.output_dir / "baseline_comparison.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    # ------------------------------------------------------------------
    # Misclassification Analysis
    # ------------------------------------------------------------------
    def analyse_misclassifications(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        texts: list[str],
        confidences: np.ndarray | None = None,
        examples_per_error: int = 5,
    ) -> dict:
        """Analyse misclassified articles with examples per error type.

        Groups misclassifications by (true_category, predicted_category) pairs
        and extracts sample articles for each error type.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            texts: Original article texts.
            confidences: Prediction confidence scores.
            examples_per_error: Number of example articles per error pair.

        Returns:
            Dict with error analysis grouped by (true, predicted) pairs.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Find misclassified indices
        misclassified_mask = y_true != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]

        total_errors = len(misclassified_indices)
        error_rate = total_errors / len(y_true) if len(y_true) > 0 else 0
        logger.info("Total misclassifications: %d (%.1f%%)", total_errors, error_rate * 100)

        # Group by error type
        error_groups: dict[str, list] = {}
        for idx in misclassified_indices:
            true_cat = config.ID_TO_CATEGORY.get(int(y_true[idx]), str(y_true[idx]))
            pred_cat = config.ID_TO_CATEGORY.get(int(y_pred[idx]), str(y_pred[idx]))
            key = f"{true_cat} → {pred_cat}"

            if key not in error_groups:
                error_groups[key] = []

            example = {
                "index": int(idx),
                "text": texts[idx][:500] if idx < len(texts) else "",
                "true_category": true_cat,
                "predicted_category": pred_cat,
            }
            if confidences is not None and idx < len(confidences):
                example["confidence"] = float(confidences[idx])

            error_groups[key].append(example)

        # Limit examples per error type and sort by frequency
        analysis = {
            "total_misclassifications": total_errors,
            "error_rate": round(error_rate, 4),
            "error_types": {},
        }

        sorted_errors = sorted(error_groups.items(), key=lambda x: -len(x[1]))
        for error_type, examples in sorted_errors:
            analysis["error_types"][error_type] = {
                "count": len(examples),
                "examples": examples[:examples_per_error],
            }

        # Save analysis
        output_path = self.output_dir / "misclassification_analysis.json"
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        # Log top error types
        logger.info("Top misclassification patterns:")
        for error_type, data in list(analysis["error_types"].items())[:5]:
            logger.info("  %s: %d occurrences", error_type, data["count"])

        return analysis

    # ------------------------------------------------------------------
    # Summary Report
    # ------------------------------------------------------------------
    def _save_summary_report(self, results: dict) -> None:
        """Save a combined evaluation summary report.

        Args:
            results: Dictionary containing all evaluation results.
        """
        report_lines = [
            "=" * 70,
            "ESG Controversy Detection System — Evaluation Summary",
            "=" * 70,
            "",
        ]

        # Metrics
        if "metrics" in results:
            report_lines.append("CORE METRICS")
            report_lines.append("-" * 40)
            for key, value in results["metrics"].items():
                report_lines.append(f"  {key:25s}: {value:.4f}")
            report_lines.append("")

        # Baseline comparison
        if "baseline" in results:
            report_lines.append("BASELINE COMPARISON (TF-IDF + LogReg)")
            report_lines.append("-" * 40)
            for key, value in results["baseline"].items():
                if isinstance(value, float):
                    report_lines.append(f"  {key:25s}: {value:.4f}")
            report_lines.append("")

            # Delta vs baseline
            if "metrics" in results:
                model_f1 = results["metrics"].get("macro_f1", 0)
                baseline_f1 = results["baseline"].get("macro_f1", 0)
                delta = model_f1 - baseline_f1
                report_lines.append(f"  FinBERT vs Baseline ΔF1: {delta:+.4f}")
                report_lines.append("")

        # Misclassification summary
        if "misclassifications" in results:
            misclass = results["misclassifications"]
            report_lines.append("MISCLASSIFICATION SUMMARY")
            report_lines.append("-" * 40)
            report_lines.append(f"  Total errors: {misclass['total_misclassifications']}")
            report_lines.append(f"  Error rate:   {misclass['error_rate']:.2%}")
            report_lines.append("")

        output_path = self.output_dir / "evaluation_summary.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(report_lines))

        logger.info("Evaluation summary saved to %s", output_path)


# ============================================================================
# CLI entry point
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESG Controversy Evaluation")
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.DATA_PROCESSED_DIR / "processed_articles.csv"),
        help="Path to preprocessed CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise SystemExit(1)

    df = pd.read_csv(input_path)

    # Filter to ESG-relevant articles only
    esg_df = df[df["label_id"] >= 0].reset_index(drop=True)

    # Simulate predictions for demonstration (in production, use actual model predictions)
    np.random.seed(42)
    y_true = esg_df["label_id"].values
    # Add realistic noise: 85% correct, 15% random errors
    y_pred = y_true.copy()
    error_mask = np.random.random(len(y_pred)) < 0.15
    y_pred[error_mask] = np.random.randint(0, config.NUM_CATEGORIES, size=error_mask.sum())

    texts = esg_df["masked_text"].tolist() if "masked_text" in esg_df.columns else esg_df["text"].tolist()
    confidences = np.where(y_true == y_pred, np.random.uniform(0.7, 0.99, len(y_true)), np.random.uniform(0.3, 0.7, len(y_true)))

    evaluator = ESGEvaluator()
    results = evaluator.run_full_evaluation(
        y_true=y_true,
        y_pred=y_pred,
        texts=texts,
        confidences=confidences,
        train_texts=texts,
        train_labels=y_true,
    )

    print(f"\n{'='*50}")
    print("Evaluation complete! Reports saved to:", config.REPORTS_DIR)
    print(f"  Accuracy:    {results['metrics']['accuracy']:.4f}")
    print(f"  Macro F1:    {results['metrics']['macro_f1']:.4f}")
    print(f"  Weighted F1: {results['metrics']['weighted_f1']:.4f}")
