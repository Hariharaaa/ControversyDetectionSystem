"""
Scoring engine for the ESG Controversy Detection System.

Aggregates article-level predictions into rolling 30-day controversy
scores per company on a 0–100 scale. Applies source credibility
weights, recency decay, computes weekly deltas, and benchmarks
against sector averages.

Usage:
    python -m src.scorer   # Demo with synthetic data
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class ControversyScorer:
    """Compute and aggregate ESG controversy scores at the company level.

    Scoring Pipeline:
        1. Compute base article score from prediction confidence × category severity
        2. Apply source credibility weight
        3. Apply recency decay (exponential)
        4. Aggregate all weighted scores into a company-level 0–100 score
        5. Compute rolling 30-day time series
        6. Detect 7-day spikes (delta > threshold)
        7. Benchmark against sector peers
    """

    def __init__(
        self,
        severity_weights: dict[str, float] | None = None,
        source_credibility: dict[str, float] | None = None,
        default_credibility: float | None = None,
        decay_lambda: float | None = None,
        spike_threshold: float | None = None,
        rolling_window: int | None = None,
    ):
        self.severity_weights = severity_weights or config.CATEGORY_SEVERITY
        self.source_credibility = source_credibility or config.SOURCE_CREDIBILITY
        self.default_credibility = default_credibility or config.DEFAULT_SOURCE_CREDIBILITY
        self.decay_lambda = decay_lambda or config.RECENCY_DECAY_LAMBDA
        self.spike_threshold = spike_threshold or config.SPIKE_THRESHOLD
        self.rolling_window = rolling_window or config.ROLLING_WINDOW_DAYS

    # ------------------------------------------------------------------
    # Article-Level Scoring
    # ------------------------------------------------------------------
    def compute_article_score(self, prediction: dict) -> float:
        """Compute a raw score for a single article prediction.

        Score = confidence × category_severity_weight

        Args:
            prediction: Dict with keys 'category' and 'confidence'.

        Returns:
            Raw article score (before source/recency adjustments).
        """
        category = prediction.get("category", "")
        confidence = prediction.get("confidence", 0.0)
        severity = self.severity_weights.get(category, 5.0)
        return confidence * severity

    def apply_source_weight(self, score: float, source: str) -> float:
        """Apply source credibility weighting to an article score.

        Args:
            score: Raw article score.
            source: Name of the news source.

        Returns:
            Source-weighted score.
        """
        source_lower = source.lower().strip() if source else ""
        credibility = self.default_credibility

        for known_source, weight in self.source_credibility.items():
            if known_source in source_lower:
                credibility = weight
                break

        return score * credibility

    def apply_recency_decay(
        self,
        score: float,
        article_date: datetime,
        reference_date: datetime | None = None,
    ) -> float:
        """Apply exponential recency decay to an article score.

        Score decreases exponentially with age:
            decayed_score = score × exp(−λ × days_old)

        With default λ=0.05, a 30-day-old article retains ~22% weight.

        Args:
            score: Source-weighted article score.
            article_date: Publication date of the article.
            reference_date: Reference date for age calculation (default: now).

        Returns:
            Recency-decayed score.
        """
        reference_date = reference_date or datetime.now()
        days_old = max(0, (reference_date - article_date).days)
        decay_factor = math.exp(-self.decay_lambda * days_old)
        return score * decay_factor

    # ------------------------------------------------------------------
    # Company-Level Aggregation
    # ------------------------------------------------------------------
    def compute_company_score(
        self,
        articles: list[dict],
        reference_date: datetime | None = None,
    ) -> float:
        """Aggregate article scores into a single company controversy score.

        Applies the full scoring pipeline (base → source → recency → aggregate)
        and normalises to a 0–100 scale.

        Args:
            articles: List of dicts with keys: category, confidence, source,
                     published_at (str or datetime).
            reference_date: Date for recency calculation (default: now).

        Returns:
            Company controversy score on a 0–100 scale.
        """
        if not articles:
            return 0.0

        reference_date = reference_date or datetime.now()
        weighted_scores = []

        for article in articles:
            # Base score
            base_score = self.compute_article_score(article)

            # Source weighting
            source = article.get("source", "")
            weighted = self.apply_source_weight(base_score, source)

            # Recency decay
            pub_date = self._parse_date(article.get("published_at", ""))
            if pub_date:
                decayed = self.apply_recency_decay(weighted, pub_date, reference_date)
            else:
                decayed = weighted * 0.5  # Penalise unknown dates

            weighted_scores.append(decayed)

        # Aggregate: sum with diminishing returns (log scale)
        if not weighted_scores:
            return 0.0

        total = sum(weighted_scores)
        # Normalise to 0–100 using a sigmoid-like transformation
        # Calibrated so that 10 high-severity articles ≈ 75 score
        raw_score = 100 * (1 - math.exp(-total / 30))
        return min(100.0, max(0.0, round(raw_score, 2)))

    def compute_rolling_score(
        self,
        articles: list[dict],
        end_date: datetime | None = None,
        num_days: int | None = None,
    ) -> pd.DataFrame:
        """Compute a daily rolling controversy score time series.

        For each day in the window, computes the company score using
        all articles published within the preceding `rolling_window` days.

        Args:
            articles: List of article dicts with published_at dates.
            end_date: End of the time window (default: now).
            num_days: Number of days to compute (default: rolling_window).

        Returns:
            DataFrame with columns: date, score.
        """
        end_date = end_date or datetime.now()
        num_days = num_days or self.rolling_window

        daily_scores = []
        for day_offset in range(num_days, -1, -1):
            ref_date = end_date - timedelta(days=day_offset)
            window_start = ref_date - timedelta(days=self.rolling_window)

            # Filter articles within window
            window_articles = []
            for art in articles:
                pub_date = self._parse_date(art.get("published_at", ""))
                if pub_date and window_start <= pub_date <= ref_date:
                    window_articles.append(art)

            score = self.compute_company_score(window_articles, reference_date=ref_date)
            daily_scores.append({"date": ref_date.date(), "score": score})

        return pd.DataFrame(daily_scores)

    # ------------------------------------------------------------------
    # Spike Detection
    # ------------------------------------------------------------------
    def compute_weekly_delta(self, rolling_scores: pd.DataFrame) -> dict:
        """Compute the 7-day score delta and detect spikes.

        Args:
            rolling_scores: DataFrame from compute_rolling_score with date and score.

        Returns:
            Dict with keys: current_score, previous_score, delta,
            is_spike, spike_threshold.
        """
        if len(rolling_scores) < 8:
            return {
                "current_score": rolling_scores["score"].iloc[-1] if len(rolling_scores) > 0 else 0.0,
                "previous_score": 0.0,
                "delta": 0.0,
                "is_spike": False,
                "spike_threshold": self.spike_threshold,
            }

        current_score = rolling_scores["score"].iloc[-1]
        previous_score = rolling_scores["score"].iloc[-8]  # 7 days back
        delta = current_score - previous_score

        return {
            "current_score": current_score,
            "previous_score": previous_score,
            "delta": round(delta, 2),
            "is_spike": delta > self.spike_threshold,
            "spike_threshold": self.spike_threshold,
        }

    # ------------------------------------------------------------------
    # Sector Benchmarking
    # ------------------------------------------------------------------
    def benchmark_sector(
        self,
        company_score: float,
        sector_scores: list[float],
    ) -> dict:
        """Benchmark a company score against its sector peers.

        Args:
            company_score: The company's controversy score.
            sector_scores: List of scores for all companies in the sector.

        Returns:
            Dict with: sector_mean, sector_median, percentile_rank,
            deviation_from_mean, risk_level.
        """
        if not sector_scores:
            return {
                "sector_mean": 0.0,
                "sector_median": 0.0,
                "percentile_rank": 0.0,
                "deviation_from_mean": company_score,
                "risk_level": self.get_risk_level(company_score),
            }

        all_scores = sorted(sector_scores)
        sector_mean = float(np.mean(all_scores))
        sector_median = float(np.median(all_scores))

        # Percentile rank
        below_count = sum(1 for s in all_scores if s < company_score)
        equal_count = sum(1 for s in all_scores if s == company_score)
        percentile = ((below_count + 0.5 * equal_count) / len(all_scores)) * 100

        return {
            "sector_mean": round(sector_mean, 2),
            "sector_median": round(sector_median, 2),
            "percentile_rank": round(percentile, 1),
            "deviation_from_mean": round(company_score - sector_mean, 2),
            "risk_level": self.get_risk_level(company_score),
        }

    # ------------------------------------------------------------------
    # Risk Level Classification
    # ------------------------------------------------------------------
    @staticmethod
    def get_risk_level(score: float) -> str:
        """Classify a controversy score into a risk level.

        Args:
            score: Controversy score (0–100).

        Returns:
            Risk level string: 'Low', 'Medium', 'High', or 'Critical'.
        """
        for level, (low, high) in config.RISK_THRESHOLDS.items():
            if low <= score < high:
                return level
        return "Critical"  # score == 100

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_date(date_str: str | datetime) -> datetime | None:
        """Parse a date string into a datetime object.

        Handles multiple common date formats.

        Args:
            date_str: Date string or datetime object.

        Returns:
            Parsed datetime or None if parsing fails.
        """
        if isinstance(date_str, datetime):
            return date_str

        if not date_str or not isinstance(date_str, str):
            return None

        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        logger.debug("Could not parse date: %s", date_str)
        return None


# ============================================================================
# CLI entry point
# ============================================================================
if __name__ == "__main__":
    # Demo with sample data
    scorer = ControversyScorer()

    # Create sample article predictions
    sample_articles = [
        {
            "category": "environmental_violation",
            "confidence": 0.92,
            "source": "Reuters",
            "published_at": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        {
            "category": "bribery_corruption",
            "confidence": 0.85,
            "source": "Bloomberg",
            "published_at": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        {
            "category": "labour_dispute",
            "confidence": 0.78,
            "source": "BBC",
            "published_at": (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        {
            "category": "data_breach",
            "confidence": 0.65,
            "source": "Unknown Blog",
            "published_at": (datetime.now() - timedelta(days=28)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    ]

    # Company score
    score = scorer.compute_company_score(sample_articles)
    risk = scorer.get_risk_level(score)
    print(f"Company Controversy Score: {score:.1f}/100 ({risk})")

    # Rolling scores
    rolling = scorer.compute_rolling_score(sample_articles)
    print(f"\nRolling scores (last 5 days):")
    print(rolling.tail().to_string(index=False))

    # Weekly delta
    delta = scorer.compute_weekly_delta(rolling)
    print(f"\n7-Day Delta: {delta['delta']:+.1f}")
    print(f"Spike Alert: {'⚠️ YES' if delta['is_spike'] else '✓ No'}")

    # Sector benchmark
    sector = scorer.benchmark_sector(score, [15.0, 22.5, 35.0, 48.0, 55.0, 12.0])
    print(f"\nSector Mean: {sector['sector_mean']:.1f}")
    print(f"Percentile Rank: {sector['percentile_rank']:.0f}th")
