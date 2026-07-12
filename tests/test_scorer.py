"""
Unit tests for the ESG Controversy Scoring Engine.

Tests cover source credibility weighting, recency decay, company score
computation, spike detection, sector benchmarking, and edge cases.

Run: pytest tests/test_scorer.py -v
"""

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.scorer import ControversyScorer


@pytest.fixture
def scorer():
    """Create a fresh ControversyScorer instance for each test."""
    return ControversyScorer()


@pytest.fixture
def sample_articles():
    """Generate a set of sample articles for testing."""
    now = datetime.now()
    return [
        {
            "category": "environmental_violation",
            "confidence": 0.90,
            "source": "Reuters",
            "published_at": (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        {
            "category": "bribery_corruption",
            "confidence": 0.85,
            "source": "Bloomberg",
            "published_at": (now - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        {
            "category": "labour_dispute",
            "confidence": 0.70,
            "source": "BBC",
            "published_at": (now - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        {
            "category": "data_breach",
            "confidence": 0.60,
            "source": "Unknown Blog",
            "published_at": (now - timedelta(days=25)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    ]


class TestSourceCredibilityWeights:
    """Tests for source credibility weighting."""

    def test_reuters_has_full_credibility(self, scorer):
        """Reuters should receive credibility weight of 1.0."""
        result = scorer.apply_source_weight(10.0, "Reuters")
        assert result == 10.0

    def test_bloomberg_has_full_credibility(self, scorer):
        """Bloomberg should receive credibility weight of 1.0."""
        result = scorer.apply_source_weight(10.0, "Bloomberg News")
        assert result == 10.0

    def test_bbc_has_reduced_credibility(self, scorer):
        """BBC should receive credibility weight of 0.9."""
        result = scorer.apply_source_weight(10.0, "BBC News")
        assert result == pytest.approx(9.0, abs=0.01)

    def test_unknown_source_gets_default(self, scorer):
        """Unknown sources should receive the default credibility weight (0.6)."""
        result = scorer.apply_source_weight(10.0, "Random Blog")
        assert result == pytest.approx(6.0, abs=0.01)

    def test_empty_source_gets_default(self, scorer):
        """Empty source string should receive the default credibility weight."""
        result = scorer.apply_source_weight(10.0, "")
        assert result == pytest.approx(6.0, abs=0.01)

    def test_case_insensitive_matching(self, scorer):
        """Source matching should be case-insensitive."""
        result = scorer.apply_source_weight(10.0, "REUTERS")
        assert result == 10.0


class TestRecencyDecay:
    """Tests for recency decay weighting."""

    def test_recent_article_scores_higher(self, scorer):
        """A recent article should score higher than an old article."""
        ref_date = datetime.now()
        recent_score = scorer.apply_recency_decay(10.0, ref_date - timedelta(days=1), ref_date)
        old_score = scorer.apply_recency_decay(10.0, ref_date - timedelta(days=30), ref_date)
        assert recent_score > old_score

    def test_today_article_has_near_full_weight(self, scorer):
        """An article from today should retain nearly full weight."""
        ref_date = datetime.now()
        result = scorer.apply_recency_decay(10.0, ref_date, ref_date)
        assert result == pytest.approx(10.0, abs=0.01)

    def test_30_day_old_article_decayed(self, scorer):
        """A 30-day-old article should retain approximately 22% weight (default lambda)."""
        ref_date = datetime.now()
        article_date = ref_date - timedelta(days=30)
        result = scorer.apply_recency_decay(10.0, article_date, ref_date)
        expected = 10.0 * math.exp(-0.05 * 30)  # ~2.23
        assert result == pytest.approx(expected, abs=0.1)


class TestCompanyScore:
    """Tests for company-level score computation."""

    def test_score_in_valid_range(self, scorer, sample_articles):
        """Company score should always be between 0 and 100."""
        score = scorer.compute_company_score(sample_articles)
        assert 0.0 <= score <= 100.0

    def test_empty_articles_returns_zero(self, scorer):
        """No articles should produce a score of 0."""
        score = scorer.compute_company_score([])
        assert score == 0.0

    def test_more_articles_increase_score(self, scorer, sample_articles):
        """More articles should generally increase the score."""
        score_few = scorer.compute_company_score(sample_articles[:2])
        score_many = scorer.compute_company_score(sample_articles)
        assert score_many >= score_few

    def test_higher_confidence_increases_score(self, scorer):
        """Higher confidence predictions should produce higher scores."""
        now = datetime.now()
        low_conf = [{
            "category": "bribery_corruption",
            "confidence": 0.3,
            "source": "Reuters",
            "published_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }]
        high_conf = [{
            "category": "bribery_corruption",
            "confidence": 0.95,
            "source": "Reuters",
            "published_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }]
        assert scorer.compute_company_score(high_conf) > scorer.compute_company_score(low_conf)


class TestWeeklyDelta:
    """Tests for 7-day spike detection."""

    def test_spike_detection_above_threshold(self, scorer):
        """Delta above threshold should trigger a spike alert."""
        import pandas as pd

        # Create scores with a clear spike
        dates = [datetime.now().date() - timedelta(days=i) for i in range(30, -1, -1)]
        scores = [10.0] * 24 + [10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 50.0]  # Spike at end
        rolling = pd.DataFrame({"date": dates, "score": scores})

        delta = scorer.compute_weekly_delta(rolling)
        assert delta["is_spike"] == True
        assert delta["delta"] > scorer.spike_threshold

    def test_no_spike_when_stable(self, scorer):
        """Stable scores should not trigger a spike."""
        import pandas as pd

        dates = [datetime.now().date() - timedelta(days=i) for i in range(30, -1, -1)]
        scores = [25.0] * 31
        rolling = pd.DataFrame({"date": dates, "score": scores})

        delta = scorer.compute_weekly_delta(rolling)
        assert delta["is_spike"] == False
        assert delta["delta"] == pytest.approx(0.0, abs=0.1)


class TestSectorBenchmark:
    """Tests for sector percentile ranking."""

    def test_percentile_rank_calculation(self, scorer):
        """Percentile rank should correctly position a score."""
        sector_scores = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = scorer.benchmark_sector(45.0, sector_scores)
        # 45 is above 4 out of 5 scores
        assert result["percentile_rank"] > 50.0

    def test_sector_mean_correct(self, scorer):
        """Sector mean should be computed correctly."""
        sector_scores = [10.0, 20.0, 30.0]
        result = scorer.benchmark_sector(25.0, sector_scores)
        assert result["sector_mean"] == pytest.approx(20.0, abs=0.01)

    def test_empty_sector_graceful(self, scorer):
        """Empty sector scores should not raise errors."""
        result = scorer.benchmark_sector(50.0, [])
        assert result["sector_mean"] == 0.0
        assert result["percentile_rank"] == 0.0

    def test_risk_level_classification(self, scorer):
        """Risk levels should match configured thresholds."""
        assert scorer.get_risk_level(10.0) == "Low"
        assert scorer.get_risk_level(35.0) == "Medium"
        assert scorer.get_risk_level(60.0) == "High"
        assert scorer.get_risk_level(85.0) == "Critical"
