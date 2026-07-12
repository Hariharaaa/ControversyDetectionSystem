"""
Central configuration for the ESG Controversy Detection System.

All project paths, API credentials, model identifiers, controversy categories,
source credibility weights, scoring thresholds, and training hyperparameters
are defined here. API keys are loaded from a .env file when available.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
import os

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("esg_controversy")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
APP_DIR = ROOT_DIR / "app"
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
TESTS_DIR = ROOT_DIR / "tests"

# Ensure directories exist
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, NOTEBOOKS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API Keys (loaded from .env, graceful fallback to None)
# ---------------------------------------------------------------------------
NEWSAPI_KEY: str | None = os.getenv("NEWSAPI_KEY")
GUARDIAN_API_KEY: str | None = os.getenv("GUARDIAN_API_KEY")
SEC_USER_AGENT: str = os.getenv("SEC_USER_AGENT", "ESGControversySystem admin@example.com")

# ---------------------------------------------------------------------------
# Model Identifiers
# ---------------------------------------------------------------------------
RELEVANCE_MODEL_NAME = "distilbert-base-uncased"
CLASSIFIER_MODEL_NAME = "ProsusAI/finbert"

RELEVANCE_MODEL_DIR = MODELS_DIR / "relevance_filter"
CLASSIFIER_MODEL_DIR = MODELS_DIR / "controversy_classifier"

# ---------------------------------------------------------------------------
# Controversy Categories (9 classes)
# ---------------------------------------------------------------------------
CONTROVERSY_CATEGORIES: list[str] = [
    "environmental_violation",
    "carbon_fraud",
    "labour_dispute",
    "supply_chain_abuse",
    "data_breach",
    "privacy_violation",
    "bribery_corruption",
    "board_misconduct",
    "community_impact",
]

CATEGORY_TO_ID: dict[str, int] = {cat: idx for idx, cat in enumerate(CONTROVERSY_CATEGORIES)}
ID_TO_CATEGORY: dict[int, str] = {idx: cat for idx, cat in enumerate(CONTROVERSY_CATEGORIES)}
NUM_CATEGORIES: int = len(CONTROVERSY_CATEGORIES)

# ---------------------------------------------------------------------------
# Category severity weights (used in scoring engine)
# ---------------------------------------------------------------------------
CATEGORY_SEVERITY: dict[str, float] = {
    "environmental_violation": 9.0,
    "carbon_fraud": 8.5,
    "labour_dispute": 7.0,
    "supply_chain_abuse": 8.0,
    "data_breach": 8.0,
    "privacy_violation": 7.5,
    "bribery_corruption": 10.0,
    "board_misconduct": 9.0,
    "community_impact": 6.5,
}

# ---------------------------------------------------------------------------
# Source Credibility Weights
# ---------------------------------------------------------------------------
SOURCE_CREDIBILITY: dict[str, float] = {
    "reuters": 1.0,
    "bloomberg": 1.0,
    "bbc": 0.9,
    "financial times": 0.85,
    "the guardian": 0.85,
    "wall street journal": 0.9,
    "associated press": 0.95,
    "cnbc": 0.8,
    "sec filing": 1.0,
}
DEFAULT_SOURCE_CREDIBILITY: float = 0.6

# ---------------------------------------------------------------------------
# Scoring Thresholds
# ---------------------------------------------------------------------------
RISK_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Low": (0.0, 25.0),
    "Medium": (25.0, 50.0),
    "High": (50.0, 75.0),
    "Critical": (75.0, 100.0),
}

SPIKE_THRESHOLD: float = 15.0  # 7-day delta above this triggers alert
ROLLING_WINDOW_DAYS: int = 30
RECENCY_DECAY_LAMBDA: float = 0.05  # exp(-λ * days) → ~22% at 30 days

# ---------------------------------------------------------------------------
# Training Hyperparameters — Relevance Filter (Stage 1)
# ---------------------------------------------------------------------------
RELEVANCE_HPARAMS = {
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_seq_length": 256,
    "early_stopping_patience": 2,
    "metric_for_best_model": "accuracy",
}

# ---------------------------------------------------------------------------
# Training Hyperparameters — Controversy Classifier (Stage 2)
# ---------------------------------------------------------------------------
CLASSIFIER_HPARAMS = {
    "learning_rate": 2e-5,
    "num_train_epochs": 4,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_seq_length": 256,
    "metric_for_best_model": "eval_macro_f1",
}

# ---------------------------------------------------------------------------
# Synthetic Data Configuration
# ---------------------------------------------------------------------------
SYNTHETIC_NUM_ARTICLES: int = 1000
SYNTHETIC_OUTPUT_PATH: Path = DATA_RAW_DIR / "synthetic_articles.csv"

# ---------------------------------------------------------------------------
# Seeded Companies for Dashboard Demo
# ---------------------------------------------------------------------------
SEEDED_COMPANIES: list[dict[str, str]] = [
    {"name": "Apple", "ticker": "AAPL", "sector": "Technology"},
    {"name": "Shell", "ticker": "SHEL", "sector": "Energy"},
    {"name": "Meta", "ticker": "META", "sector": "Technology"},
    {"name": "Nestlé", "ticker": "NSRGY", "sector": "Consumer Staples"},
    {"name": "JPMorgan", "ticker": "JPM", "sector": "Financials"},
]

# ---------------------------------------------------------------------------
# ESG Keywords for API Searches
# ---------------------------------------------------------------------------
ESG_KEYWORDS: list[str] = [
    "pollution", "emissions", "carbon", "climate", "environmental",
    "labour", "labor", "workers", "safety", "supply chain",
    "data breach", "privacy", "cybersecurity",
    "bribery", "corruption", "fraud", "misconduct",
    "community", "human rights", "discrimination",
    "governance", "board", "executive", "whistleblower",
    "violation", "fine", "penalty", "lawsuit", "investigation",
]
