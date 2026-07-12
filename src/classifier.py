"""
Stage 2: Controversy Classifier for the ESG Controversy Detection System.

Fine-tunes ProsusAI/finbert for 9-class sequence classification across
ESG controversy categories. Uses weighted cross-entropy loss, AdamW
optimiser with linear warmup scheduler, and saves the best model by
macro-F1 score.

Categories:
    0: environmental_violation
    1: carbon_fraud
    2: labour_dispute
    3: supply_chain_abuse
    4: data_breach
    5: privacy_violation
    6: bribery_corruption
    7: board_misconduct
    8: community_impact

Usage:
    python -m src.classifier              # Train on synthetic data
    python -m src.classifier --evaluate   # Evaluate saved model
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


# ============================================================================
# Weighted Cross-Entropy Trainer for Multi-class
# ============================================================================
class WeightedMultiClassTrainer(Trainer):
    """Custom Trainer with class-weighted cross-entropy loss for 9-class classification.

    Handles class imbalance by applying inverse-frequency weights to the
    cross-entropy loss function.
    """

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute weighted cross-entropy loss.

        Args:
            model: The transformer model.
            inputs: Batch of tokenised inputs with labels.
            return_outputs: Whether to return model outputs alongside loss.

        Returns:
            Loss tensor, optionally with model outputs.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# Controversy Classifier
# ============================================================================
class ControversyClassifier:
    """9-class ESG controversy classifier built on ProsusAI/FinBERT.

    Fine-tunes FinBERT for classifying ESG-relevant articles into
    specific controversy categories. Uses AdamW with linear warmup
    and saves the best checkpoint by macro-F1.
    """

    def __init__(
        self,
        model_name: str | None = None,
        model_dir: Path | None = None,
        max_seq_length: int | None = None,
    ):
        self.model_name = model_name or config.CLASSIFIER_MODEL_NAME
        self.model_dir = model_dir or config.CLASSIFIER_MODEL_DIR
        self.max_seq_length = max_seq_length or config.CLASSIFIER_HPARAMS["max_seq_length"]
        self.tokenizer = None
        self.model = None
        self.num_labels = config.NUM_CATEGORIES
        self.id2label = config.ID_TO_CATEGORY
        self.label2id = config.CATEGORY_TO_ID

    def prepare_data(
        self,
        df: pd.DataFrame,
        text_column: str = "masked_text",
        label_column: str = "label_id",
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> tuple[Dataset, Dataset, Dataset]:
        """Prepare tokenised HuggingFace Datasets for training.

        Filters to ESG-relevant articles only (label_id >= 0), then
        splits into train/val/test sets with stratification.

        Args:
            df: Preprocessed DataFrame with text and label columns.
            text_column: Column containing preprocessed text.
            label_column: Column containing integer category labels (0–8).
            test_size: Fraction reserved for test set.
            val_size: Fraction of remaining data for validation.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        # Filter to ESG-relevant only
        df = df.copy()
        df = df[df[label_column] >= 0].reset_index(drop=True)
        df[label_column] = df[label_column].astype(int)

        logger.info("Preparing classifier data: %d ESG-relevant articles", len(df))

        # Verify all categories present
        unique_labels = sorted(df[label_column].unique())
        logger.info("Categories present: %s", unique_labels)

        # Split data
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df[label_column], random_state=42
        )
        train_df, val_df = train_test_split(
            train_df, test_size=val_size, stratify=train_df[label_column], random_state=42
        )

        logger.info(
            "Data split — Train: %d, Val: %d, Test: %d",
            len(train_df), len(val_df), len(test_df),
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Convert to datasets
        train_ds = self._df_to_dataset(train_df, text_column, label_column)
        val_ds = self._df_to_dataset(val_df, text_column, label_column)
        test_ds = self._df_to_dataset(test_df, text_column, label_column)

        return train_ds, val_ds, test_ds

    def _df_to_dataset(self, df: pd.DataFrame, text_col: str, label_col: str) -> Dataset:
        """Convert DataFrame to tokenised HuggingFace Dataset.

        Args:
            df: DataFrame with text and label columns.
            text_col: Name of the text column.
            label_col: Name of the label column.

        Returns:
            Tokenised Dataset ready for training.
        """
        dataset = Dataset.from_pandas(
            df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}),
            preserve_index=False,
        )

        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
            )

        dataset = dataset.map(tokenize_fn, batched=True, desc="Tokenizing")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return dataset

    def compute_class_weights(self, dataset: Dataset) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced loss.

        Args:
            dataset: Training dataset with 'label' column.

        Returns:
            Tensor of class weights with shape (num_labels,).
        """
        labels = np.array(dataset["label"])
        counts = np.bincount(labels, minlength=self.num_labels)
        total = len(labels)
        # Inverse frequency weighting with smoothing
        weights = total / (self.num_labels * counts.clip(min=1))
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        logger.info("Class weights:")
        for i, (cat, w) in enumerate(zip(config.CONTROVERSY_CATEGORIES, weights)):
            logger.info("  %s: %.3f (count: %d)", cat, w, counts[i])

        return weights_tensor

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        class_weights: torch.Tensor | None = None,
    ) -> Trainer:
        """Fine-tune the controversy classifier.

        Uses AdamW optimiser with linear warmup scheduler, weighted
        cross-entropy loss, and saves the best model by macro-F1.

        Args:
            train_dataset: Tokenised training dataset.
            val_dataset: Tokenised validation dataset.
            class_weights: Optional pre-computed class weights.

        Returns:
            Trained Trainer instance.
        """
        hparams = config.CLASSIFIER_HPARAMS

        # Load model with mismatched size handling (FinBERT has 3 labels, we need 9)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.model_dir / "checkpoints"),
            num_train_epochs=hparams["num_train_epochs"],
            per_device_train_batch_size=hparams["per_device_train_batch_size"],
            per_device_eval_batch_size=hparams["per_device_eval_batch_size"],
            learning_rate=hparams["learning_rate"],
            warmup_ratio=hparams["warmup_ratio"],
            weight_decay=hparams["weight_decay"],
            optim="adamw_torch",
            lr_scheduler_type="linear",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=2,
            logging_steps=50,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        # Compute class weights if not provided
        if class_weights is None:
            class_weights = self.compute_class_weights(train_dataset)

        # Create trainer
        trainer = WeightedMultiClassTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
        )

        # Train
        logger.info("Starting controversy classifier training...")
        trainer.train()

        # Save best model
        self.model_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self.model_dir))
        self.tokenizer.save_pretrained(str(self.model_dir))
        logger.info("Best model saved to %s", self.model_dir)

        return trainer

    def evaluate(self, trainer: Trainer, test_dataset: Dataset) -> dict:
        """Evaluate the model on the test set.

        Args:
            trainer: Trained Trainer instance.
            test_dataset: Tokenised test dataset.

        Returns:
            Dictionary of evaluation metrics.
        """
        results = trainer.evaluate(test_dataset)
        logger.info("Test set results:")
        for key, value in results.items():
            logger.info("  %s: %.4f", key, value)
        return results

    def load_model(self) -> None:
        """Load a previously saved model from disk.

        Raises:
            FileNotFoundError: If the model directory does not exist.
        """
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.eval()
        logger.info("Loaded controversy classifier from %s", self.model_dir)

    def predict(self, texts: list[str]) -> list[dict]:
        """Run inference on a list of texts.

        Args:
            texts: List of preprocessed text strings.

        Returns:
            List of dicts with keys: category, confidence, probabilities, label_id.
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        device = next(self.model.parameters()).device
        results = []

        # Process in batches
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

            for pred, prob in zip(preds, probs):
                pred_id = pred.item()
                confidence = prob[pred_id].item()
                prob_dict = {
                    config.ID_TO_CATEGORY[j]: prob[j].item()
                    for j in range(self.num_labels)
                }
                results.append({
                    "category": config.ID_TO_CATEGORY[pred_id],
                    "label_id": pred_id,
                    "confidence": confidence,
                    "probabilities": prob_dict,
                })

        return results

    def classify_controversy(self, texts: list[str]) -> list[dict]:
        """Public API for controversy classification.

        Alias for predict() that follows the specification naming.

        Args:
            texts: List of preprocessed text strings.

        Returns:
            List of prediction dicts.
        """
        return self.predict(texts)

    @staticmethod
    def _compute_metrics(eval_pred) -> dict:
        """Compute evaluation metrics for the Trainer.

        Calculates accuracy, macro-F1, weighted-F1, and per-class F1.

        Args:
            eval_pred: EvalPrediction with predictions and label_ids.

        Returns:
            Dictionary of metrics.
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
            "macro_precision": precision_score(labels, predictions, average="macro", zero_division=0),
            "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
        }

        # Per-class F1
        per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
        for i, f1_val in enumerate(per_class_f1):
            if i < len(config.CONTROVERSY_CATEGORIES):
                cat = config.CONTROVERSY_CATEGORIES[i]
                metrics[f"f1_{cat}"] = f1_val

        return metrics


# ============================================================================
# CLI entry point
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESG Controversy Classifier Training")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved model only")
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.DATA_PROCESSED_DIR / "processed_articles.csv"),
        help="Path to preprocessed CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s. Run preprocessing first.", input_path)
        raise SystemExit(1)

    df = pd.read_csv(input_path)
    classifier = ControversyClassifier()

    if args.evaluate:
        classifier.load_model()
        sample_texts = df[df["label_id"] >= 0]["masked_text"].head(10).tolist()
        predictions = classifier.predict(sample_texts)
        for text, pred in zip(sample_texts[:5], predictions[:5]):
            print(f"Text: {text[:80]}...")
            print(f"  → {pred['category']} (confidence: {pred['confidence']:.3f})")
            print()
    else:
        train_ds, val_ds, test_ds = classifier.prepare_data(df)
        class_weights = classifier.compute_class_weights(train_ds)
        trainer = classifier.train(train_ds, val_ds, class_weights)
        results = classifier.evaluate(trainer, test_ds)
        print(f"\nTest Results:")
        for k, v in sorted(results.items()):
            print(f"  {k}: {v:.4f}")
