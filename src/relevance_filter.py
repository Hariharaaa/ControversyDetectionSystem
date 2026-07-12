"""
Stage 1: ESG Relevance Filter for the ESG Controversy Detection System.

Fine-tunes distilbert-base-uncased as a binary classifier to separate
ESG-relevant articles from non-ESG content. Uses the HuggingFace
Transformers Trainer API with weighted loss, early stopping, and
checkpoint saving.

Usage:
    python -m src.relevance_filter             # Train on synthetic data
    python -m src.relevance_filter --evaluate   # Evaluate saved model
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
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


# ============================================================================
# Weighted Cross-Entropy Trainer
# ============================================================================
class WeightedCETrainer(Trainer):
    """Custom Trainer with class-weighted cross-entropy loss.

    Handles imbalanced binary classification by upweighting the
    minority class in the loss computation.
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
# Relevance Filter
# ============================================================================
class ESGRelevanceFilter:
    """Binary classifier to filter ESG-relevant articles.

    Fine-tunes distilbert-base-uncased to distinguish between
    ESG-relevant and non-ESG articles. Designed as the first stage
    of the two-stage classification pipeline.
    """

    LABEL_MAP = {0: "not_esg", 1: "esg_relevant"}

    def __init__(
        self,
        model_name: str | None = None,
        model_dir: Path | None = None,
        max_seq_length: int | None = None,
    ):
        self.model_name = model_name or config.RELEVANCE_MODEL_NAME
        self.model_dir = model_dir or config.RELEVANCE_MODEL_DIR
        self.max_seq_length = max_seq_length or config.RELEVANCE_HPARAMS["max_seq_length"]
        self.tokenizer = None
        self.model = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        text_column: str = "masked_text",
        label_column: str = "is_esg",
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> tuple[Dataset, Dataset, Dataset]:
        """Prepare HuggingFace Datasets from a preprocessed DataFrame.

        Creates binary labels (ESG-relevant=1, not=0) and splits into
        train, validation, and test sets.

        Args:
            df: Preprocessed DataFrame with text and ESG label columns.
            text_column: Column containing preprocessed text.
            label_column: Column containing binary ESG labels (0/1).
            test_size: Fraction reserved for test set.
            val_size: Fraction of remaining data for validation.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        # Ensure labels are integers
        df = df.copy()
        df[label_column] = df[label_column].astype(int)

        # Split: train/test, then train/val
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

        # Convert to HuggingFace Datasets
        train_ds = self._df_to_dataset(train_df, text_column, label_column)
        val_ds = self._df_to_dataset(val_df, text_column, label_column)
        test_ds = self._df_to_dataset(test_df, text_column, label_column)

        return train_ds, val_ds, test_ds

    def _df_to_dataset(self, df: pd.DataFrame, text_col: str, label_col: str) -> Dataset:
        """Convert DataFrame to tokenised HuggingFace Dataset."""
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
            Tensor of class weights [weight_class_0, weight_class_1].
        """
        labels = np.array(dataset["label"])
        counts = np.bincount(labels, minlength=2)
        total = len(labels)
        weights = total / (2.0 * counts)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        logger.info("Class weights — Not ESG: %.3f, ESG: %.3f", weights[0], weights[1])
        return weights_tensor

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        class_weights: torch.Tensor | None = None,
    ) -> Trainer:
        """Fine-tune the relevance filter model.

        Args:
            train_dataset: Tokenised training dataset.
            val_dataset: Tokenised validation dataset.
            class_weights: Optional pre-computed class weights.

        Returns:
            Trained Trainer instance.
        """
        hparams = config.RELEVANCE_HPARAMS

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label=self.LABEL_MAP,
            label2id={v: k for k, v in self.LABEL_MAP.items()},
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
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=hparams["metric_for_best_model"],
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
        trainer = WeightedCETrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=hparams["early_stopping_patience"])],
        )

        # Train
        logger.info("Starting relevance filter training...")
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
        """Load a previously saved model from disk."""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.eval()
        logger.info("Loaded relevance filter model from %s", self.model_dir)

    def predict(self, texts: list[str]) -> list[dict]:
        """Run inference on a list of texts.

        Args:
            texts: List of preprocessed text strings.

        Returns:
            List of dicts with keys: label, confidence, is_esg.
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
                results.append({
                    "label": self.LABEL_MAP[pred_id],
                    "confidence": confidence,
                    "is_esg": pred_id == 1,
                })

        return results

    def filter_esg_relevant(self, texts: list[str]) -> list[bool]:
        """Filter texts to only ESG-relevant ones.

        Args:
            texts: List of preprocessed text strings.

        Returns:
            List of boolean flags (True = ESG-relevant).
        """
        predictions = self.predict(texts)
        return [p["is_esg"] for p in predictions]

    @staticmethod
    def _compute_metrics(eval_pred) -> dict:
        """Compute evaluation metrics for the Trainer.

        Args:
            eval_pred: EvalPrediction with predictions and label_ids.

        Returns:
            Dictionary of metrics: accuracy, precision, recall, f1.
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average="binary", zero_division=0),
            "recall": recall_score(labels, predictions, average="binary", zero_division=0),
            "f1": f1_score(labels, predictions, average="binary", zero_division=0),
        }


# ============================================================================
# CLI entry point
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESG Relevance Filter Training")
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
    filter_model = ESGRelevanceFilter()

    if args.evaluate:
        filter_model.load_model()
        # Quick test
        sample_texts = df["masked_text"].head(10).tolist()
        predictions = filter_model.predict(sample_texts)
        for text, pred in zip(sample_texts[:3], predictions[:3]):
            print(f"Text: {text[:80]}...")
            print(f"  → {pred['label']} (confidence: {pred['confidence']:.3f})")
    else:
        train_ds, val_ds, test_ds = filter_model.prepare_data(df)
        class_weights = filter_model.compute_class_weights(train_ds)
        trainer = filter_model.train(train_ds, val_ds, class_weights)
        results = filter_model.evaluate(trainer, test_ds)
        print(f"\nTest Results: {results}")
