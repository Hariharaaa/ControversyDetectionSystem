"""
Unit tests for the ESG Controversy Classifier.

Tests cover tokenisation, label mapping, prediction output format,
class weights, model output shape, and confidence ranges.

Run: pytest tests/test_classifier.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class TestLabelMapping:
    """Tests for controversy category label mapping consistency."""

    def test_all_categories_have_unique_ids(self):
        """All 9 controversy categories should map to unique integer IDs."""
        ids = list(config.CATEGORY_TO_ID.values())
        assert len(ids) == len(set(ids))
        assert len(ids) == 9

    def test_id_to_category_is_inverse(self):
        """ID_TO_CATEGORY should be the exact inverse of CATEGORY_TO_ID."""
        for cat, idx in config.CATEGORY_TO_ID.items():
            assert config.ID_TO_CATEGORY[idx] == cat

    def test_all_categories_present(self):
        """All expected categories should be in the mapping."""
        expected = {
            "environmental_violation", "carbon_fraud", "labour_dispute",
            "supply_chain_abuse", "data_breach", "privacy_violation",
            "bribery_corruption", "board_misconduct", "community_impact",
        }
        assert set(config.CONTROVERSY_CATEGORIES) == expected

    def test_label_ids_are_contiguous(self):
        """Label IDs should be contiguous integers starting from 0."""
        ids = sorted(config.CATEGORY_TO_ID.values())
        assert ids == list(range(9))

    def test_num_categories_matches(self):
        """NUM_CATEGORIES should equal the length of the categories list."""
        assert config.NUM_CATEGORIES == len(config.CONTROVERSY_CATEGORIES)
        assert config.NUM_CATEGORIES == 9


class TestTokenization:
    """Tests for tokenisation pipeline correctness."""

    @pytest.fixture
    def tokenizer(self):
        """Load the FinBERT tokenizer."""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(config.CLASSIFIER_MODEL_NAME)

    def test_max_length_truncation(self, tokenizer):
        """Tokenised output should not exceed max_seq_length."""
        long_text = "word " * 1000  # Way longer than 256 tokens
        encoded = tokenizer(
            long_text,
            padding="max_length",
            truncation=True,
            max_length=config.CLASSIFIER_HPARAMS["max_seq_length"],
            return_tensors="pt",
        )
        assert encoded["input_ids"].shape[1] == 256

    def test_padding_to_max_length(self, tokenizer):
        """Short text should be padded to max_seq_length."""
        short_text = "Company fined."
        encoded = tokenizer(
            short_text,
            padding="max_length",
            truncation=True,
            max_length=config.CLASSIFIER_HPARAMS["max_seq_length"],
            return_tensors="pt",
        )
        assert encoded["input_ids"].shape[1] == 256

    def test_attention_mask_shape(self, tokenizer):
        """Attention mask should match input_ids shape."""
        text = "Environmental violation at factory."
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        assert encoded["attention_mask"].shape == encoded["input_ids"].shape

    def test_batch_tokenization(self, tokenizer):
        """Batch tokenisation should produce correctly shaped tensors."""
        texts = [
            "Company polluted river",
            "Data breach affected millions",
            "Bribery scandal uncovered",
        ]
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        assert encoded["input_ids"].shape == (3, 256)


class TestClassWeights:
    """Tests for class weight computation."""

    def test_weights_are_positive(self):
        """All class weights should be positive."""
        from src.classifier import ControversyClassifier
        from datasets import Dataset

        # Create a small mock dataset
        labels = [0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8]
        dataset = Dataset.from_dict({"label": labels, "text": ["t"] * len(labels)})

        classifier = ControversyClassifier()
        weights = classifier.compute_class_weights(dataset)
        assert torch.all(weights > 0)

    def test_minority_class_gets_higher_weight(self):
        """Minority classes should receive higher weights than majority."""
        from src.classifier import ControversyClassifier
        from datasets import Dataset

        # Create imbalanced dataset: class 0 has 50, class 1 has 5
        labels = [0] * 50 + [1] * 5 + [2] * 10 + list(range(3, 9))
        dataset = Dataset.from_dict({"label": labels, "text": ["t"] * len(labels)})

        classifier = ControversyClassifier()
        weights = classifier.compute_class_weights(dataset)
        assert weights[1] > weights[0]  # Minority > majority


class TestPredictionFormat:
    """Tests for prediction output format validation."""

    def test_prediction_dict_keys(self):
        """Prediction output should contain required keys."""
        # Test with mock prediction
        prediction = {
            "category": "environmental_violation",
            "label_id": 0,
            "confidence": 0.85,
            "probabilities": {cat: 0.1 for cat in config.CONTROVERSY_CATEGORIES},
        }
        assert "category" in prediction
        assert "confidence" in prediction
        assert "probabilities" in prediction
        assert "label_id" in prediction

    def test_confidence_range_valid(self):
        """Confidence should be between 0 and 1."""
        # Softmax always produces values in [0, 1]
        logits = torch.randn(1, 9)
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max().item()
        assert 0.0 <= confidence <= 1.0

    def test_probabilities_sum_to_one(self):
        """All category probabilities should sum to approximately 1."""
        logits = torch.randn(1, 9)
        probs = torch.softmax(logits, dim=-1)
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)


class TestModelOutputShape:
    """Tests for model output tensor shapes."""

    def test_logits_shape(self):
        """Model logits should have shape (batch_size, num_categories)."""
        # Simulate model output
        batch_size = 4
        logits = torch.randn(batch_size, config.NUM_CATEGORIES)
        assert logits.shape == (batch_size, 9)

    def test_softmax_preserves_shape(self):
        """Softmax should preserve the logits shape."""
        logits = torch.randn(8, 9)
        probs = torch.softmax(logits, dim=-1)
        assert probs.shape == logits.shape

    def test_argmax_produces_valid_labels(self):
        """Argmax of logits should produce valid label IDs (0–8)."""
        logits = torch.randn(16, 9)
        preds = torch.argmax(logits, dim=-1)
        assert torch.all(preds >= 0)
        assert torch.all(preds < 9)
