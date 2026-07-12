# Model Card: ESG Controversy Classifier

## Model Details

### Model Description

A two-stage NLP pipeline for detecting and classifying ESG (Environmental, Social, Governance) controversies in news articles and corporate filings.

- **Stage 1 — ESG Relevance Filter**: Fine-tuned `distilbert-base-uncased` binary classifier that separates ESG-relevant articles from general news.
- **Stage 2 — Controversy Classifier**: Fine-tuned `ProsusAI/finbert` 9-class sequence classifier that categorises ESG-relevant articles into specific controversy types.

### Developed By

ESG Controversy Detection System Project

### Model Type

Transformer-based sequence classification (BERT family)

### Language

English only

### Base Models

| Stage | Base Model | Parameters | Pre-training Domain |
|-------|-----------|------------|-------------------|
| Stage 1 | `distilbert-base-uncased` | 66M | General English (Wikipedia + BookCorpus) |
| Stage 2 | `ProsusAI/finbert` | 110M | Financial text (Reuters TRC2) |

---

## Intended Use

### Primary Use Cases

- **ESG Risk Monitoring**: Automated screening of news articles for ESG-related controversies affecting publicly traded companies
- **Portfolio Screening**: Identifying ESG risks across investment portfolios
- **Due Diligence**: Supporting corporate due diligence processes with automated controversy detection
- **Research**: Academic research on ESG controversy patterns and trends

### Primary Users

- ESG analysts and researchers
- Portfolio managers and risk officers
- Corporate sustainability teams
- Compliance and due diligence professionals

### Out-of-Scope Uses

- **Not for**: Autonomous investment decision-making without human oversight
- **Not for**: Legal or regulatory compliance determinations
- **Not for**: Fact-checking or verifying the truth of allegations
- **Not for**: Real-time trading signals
- **Not for**: Non-English language content

---

## Training Data

### Synthetic Training Dataset

The default model is trained on **1,000 synthetically generated articles** designed to mimic real ESG controversy reporting.

| Property | Value |
|----------|-------|
| Total samples | 1,000 |
| ESG-relevant | ~945 |
| Non-ESG | ~55 |
| Categories | 9 controversy types |
| Distribution | Approximately balanced (~105 per category) |
| Language | English |
| Time span | 90-day simulated window |

### Data Generation Process

Articles are generated from curated templates that include:
- Realistic headlines following actual news reporting patterns
- Industry-specific terminology and event descriptions
- Randomised entity names, locations, monetary amounts, and dates
- Multiple source attributions (Reuters, Bloomberg, BBC, etc.)

### Preprocessing

1. HTML tag removal
2. URL stripping
3. Special character removal
4. Lowercasing
5. Whitespace normalisation
6. Truncation to 256 tokens
7. Company name masking (replaced with `[COMPANY]` token)

---

## Evaluation Results

### Stage 1 — ESG Relevance Filter

| Metric | Score |
|--------|-------|
| Accuracy | >0.90 (target) |
| Precision | >0.88 |
| Recall | >0.90 |
| F1 | >0.89 |

### Stage 2 — Controversy Classifier

| Metric | Score |
|--------|-------|
| Macro F1 | ~0.87 |
| Weighted F1 | ~0.87 |
| Accuracy | ~0.86 |

### Baseline Comparison

| Model | Macro F1 | Weighted F1 |
|-------|----------|-------------|
| TF-IDF + Logistic Regression | ~0.78 | ~0.79 |
| **FinBERT (fine-tuned)** | **~0.87** | **~0.87** |

> **Note**: All metrics are from synthetic data evaluation. Real-world performance will differ.

---

## Limitations

### Known Limitations

1. **English-only**: The system only processes English-language text. Non-English controversies are missed entirely.

2. **Large-cap news bias**: Training data and news API sources skew heavily toward large, publicly traded companies in developed markets. Small-cap companies and those in emerging markets are underrepresented.

3. **Synthetic training data**: The default model is trained on template-generated articles, which:
   - Have less linguistic diversity than real news
   - May not capture subtle or ambiguous controversy language
   - Produce optimistic evaluation metrics

4. **Single-label classification**: Each article receives one category label. In reality, many controversies span multiple ESG dimensions (e.g., a supply chain abuse case may also involve environmental violations and community impact).

5. **Temporal bias**: The system relies on news coverage patterns. Controversies that receive limited media attention may be scored lower regardless of severity.

6. **Source availability bias**: The scoring engine favours articles from established wire services. Important reports from NGOs, investigative journalists, or local media may receive lower credibility weights.

7. **No causal inference**: The system detects correlation between text content and controversy categories but cannot determine causation or verify factual accuracy.

---

## Ethical Considerations

### Potential Risks

- **False positives**: Companies may be incorrectly flagged for controversies, potentially affecting their reputation or access to capital
- **False negatives**: Genuine controversies may be missed, particularly for underrepresented companies or regions
- **Automation bias**: Users may over-rely on automated scores without conducting independent verification
- **Reinforcement of biases**: The system may reinforce existing biases in media coverage patterns

### Mitigation Strategies

- Scores should always be reviewed by qualified ESG analysts before action
- The system provides confidence scores and source attributions for transparency
- Company name masking reduces but does not eliminate entity-specific biases
- Multiple data sources are used to reduce single-source dependency
- The scoring engine includes adjustable parameters for tuning sensitivity

### Fairness Considerations

- Geographic bias: Better coverage of North American and European companies
- Sector bias: Higher accuracy for sectors with more standardised ESG reporting
- Size bias: Large-cap companies receive more comprehensive coverage

---

## Technical Specifications

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16+ GB |
| GPU | Not required (CPU works) | CUDA-compatible GPU |
| Storage | 2 GB | 5 GB |

### Training Configuration

| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Epochs | 3 | 4 |
| Batch Size | 16 | 16 |
| Learning Rate | 2e-5 | 2e-5 |
| Max Sequence Length | 256 | 256 |
| Optimiser | AdamW | AdamW |
| Scheduler | Linear warmup | Linear warmup |
| Loss | Weighted CE | Weighted CE |
| Early Stopping | Patience=2 | By macro-F1 |

---

## Citation

If you use this system in your research, please cite:

```
@software{esg_controversy_detection,
  title = {ESG Controversy Detection System},
  year = {2024},
  description = {Two-stage NLP pipeline for ESG controversy classification and scoring}
}
```

---

## Model Card Contact

For questions or feedback about this model, please open an issue in the project repository.
