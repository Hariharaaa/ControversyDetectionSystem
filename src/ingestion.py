"""
Data ingestion module for the ESG Controversy Detection System.

Provides three live ingestion backends (NewsAPI, The Guardian, SEC EDGAR)
and a synthetic data generator fallback that produces 1,000 realistic
labelled articles across all 9 controversy categories.

Usage:
    python -m src.ingestion          # Generate synthetic data
    python -m src.ingestion --live   # Attempt live ingestion (requires API keys)
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


# ============================================================================
# NewsAPI Ingester
# ============================================================================
class NewsAPIIngester:
    """Fetch ESG-related news articles via NewsAPI.

    Requires a valid NEWSAPI_KEY in .env. Searches for company names
    combined with ESG-related keywords.
    """

    BASE_URL = "https://newsapi.org/v2/everything"
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2.0

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or config.NEWSAPI_KEY
        if not self.api_key:
            logger.warning("NewsAPI key not configured. Live ingestion unavailable.")

    def fetch_articles(
        self,
        company: str,
        keywords: list[str] | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        page_size: int = 100,
        max_pages: int = 5,
    ) -> list[dict]:
        """Fetch articles for a company + ESG keyword combinations.

        Args:
            company: Company name to search for.
            keywords: ESG keywords to combine with company name. Defaults to config.ESG_KEYWORDS.
            from_date: Start date in YYYY-MM-DD format.
            to_date: End date in YYYY-MM-DD format.
            page_size: Number of results per page (max 100 for NewsAPI).
            max_pages: Maximum number of pages to fetch.

        Returns:
            List of article dicts with keys: title, text, source, published_at, url, company.
        """
        if not self.api_key:
            logger.error("Cannot fetch from NewsAPI: no API key configured.")
            return []

        keywords = keywords or config.ESG_KEYWORDS
        articles = []

        for keyword in keywords[:10]:  # Limit to avoid rate limits
            query = f'"{company}" AND "{keyword}"'
            for page in range(1, max_pages + 1):
                params = {
                    "q": query,
                    "apiKey": self.api_key,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": page_size,
                    "page": page,
                }
                if from_date:
                    params["from"] = from_date
                if to_date:
                    params["to"] = to_date

                response = self._request_with_retry(params)
                if response is None:
                    break

                data = response.json()
                if data.get("status") != "ok":
                    logger.warning("NewsAPI error: %s", data.get("message", "Unknown"))
                    break

                raw_articles = data.get("articles", [])
                if not raw_articles:
                    break

                for art in raw_articles:
                    articles.append({
                        "title": art.get("title", ""),
                        "text": art.get("description", "") or art.get("content", ""),
                        "source": art.get("source", {}).get("name", "Unknown"),
                        "published_at": art.get("publishedAt", ""),
                        "url": art.get("url", ""),
                        "company": company,
                    })

                if len(raw_articles) < page_size:
                    break

        logger.info("NewsAPI: fetched %d articles for '%s'", len(articles), company)
        return articles

    def _request_with_retry(self, params: dict) -> requests.Response | None:
        """Execute HTTP request with exponential backoff retry."""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                if response.status_code == 429:
                    wait = self.BACKOFF_FACTOR ** attempt
                    logger.warning("NewsAPI rate limit hit. Waiting %.1fs...", wait)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                logger.error("NewsAPI request failed (attempt %d): %s", attempt + 1, exc)
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.BACKOFF_FACTOR ** attempt)
        return None


# ============================================================================
# The Guardian API Ingester
# ============================================================================
class GuardianIngester:
    """Fetch ESG-related articles from The Guardian Open Platform API.

    Requires a GUARDIAN_API_KEY in .env. The free tier allows 12 calls/second
    and 5,000 calls/day.
    """

    BASE_URL = "https://content.guardianapis.com/search"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or config.GUARDIAN_API_KEY
        if not self.api_key:
            logger.warning("Guardian API key not configured. Live ingestion unavailable.")

    def fetch_articles(
        self,
        query: str,
        from_date: str | None = None,
        to_date: str | None = None,
        section: str | None = None,
        page_size: int = 50,
        max_pages: int = 5,
    ) -> list[dict]:
        """Search The Guardian for articles matching a query.

        Args:
            query: Search query (supports AND, OR, NOT operators).
            from_date: Start date in YYYY-MM-DD format.
            to_date: End date in YYYY-MM-DD format.
            section: Guardian section filter (e.g., 'environment', 'business').
            page_size: Results per page (max 200).
            max_pages: Maximum pages to fetch.

        Returns:
            List of article dicts.
        """
        if not self.api_key:
            logger.error("Cannot fetch from Guardian: no API key configured.")
            return []

        articles = []

        for page in range(1, max_pages + 1):
            params = {
                "q": query,
                "api-key": self.api_key,
                "show-fields": "bodyText,headline,byline",
                "page-size": page_size,
                "page": page,
                "order-by": "relevance",
            }
            if from_date:
                params["from-date"] = from_date
            if to_date:
                params["to-date"] = to_date
            if section:
                params["section"] = section

            try:
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                results = data.get("response", {}).get("results", [])
                if not results:
                    break

                for art in results:
                    fields = art.get("fields", {})
                    articles.append({
                        "title": fields.get("headline", art.get("webTitle", "")),
                        "text": fields.get("bodyText", ""),
                        "source": "The Guardian",
                        "published_at": art.get("webPublicationDate", ""),
                        "url": art.get("webUrl", ""),
                        "company": "",  # Populated downstream
                    })

                if len(results) < page_size:
                    break

            except requests.RequestException as exc:
                logger.error("Guardian API request failed (page %d): %s", page, exc)
                break

        logger.info("Guardian: fetched %d articles for query '%s'", len(articles), query)
        return articles


# ============================================================================
# SEC EDGAR Full-Text Search Ingester
# ============================================================================
class SECEdgarIngester:
    """Search SEC EDGAR full-text search index for 10-K and 10-Q filings.

    Uses the EFTS (EDGAR Full-Text Search) endpoint. No API key required,
    but a proper User-Agent header is mandatory per SEC fair access policy.
    """

    SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    FILING_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

    def __init__(self, user_agent: str | None = None):
        self.user_agent = user_agent or config.SEC_USER_AGENT
        self.headers = {"User-Agent": self.user_agent}

    def search_filings(
        self,
        query: str,
        forms: list[str] | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
        max_results: int = 50,
    ) -> list[dict]:
        """Search EDGAR full-text index for filing content.

        Args:
            query: Text query to search for in filings.
            forms: Filing types to search (default: ['10-K', '10-Q']).
            date_start: Start date in YYYY-MM-DD format.
            date_end: End date in YYYY-MM-DD format.
            max_results: Maximum number of results to return.

        Returns:
            List of filing dicts with keys: title, text, source, published_at, url, company.
        """
        forms = forms or ["10-K", "10-Q"]
        params = {
            "q": query,
            "forms": ",".join(forms),
            "from": 0,
            "size": min(max_results, 50),
        }
        if date_start:
            params["startdt"] = date_start
        if date_end:
            params["enddt"] = date_end

        try:
            response = requests.get(
                self.SEARCH_URL, params=params, headers=self.headers, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            filings = []
            for hit in data.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                filings.append({
                    "title": source.get("file_description", source.get("display_names", [""])[0]),
                    "text": source.get("file_description", ""),
                    "source": "SEC Filing",
                    "published_at": source.get("file_date", ""),
                    "url": source.get("file_url", ""),
                    "company": ", ".join(source.get("display_names", [])),
                })

            logger.info("SEC EDGAR: fetched %d filings for query '%s'", len(filings), query)
            return filings

        except requests.RequestException as exc:
            logger.error("SEC EDGAR search failed: %s", exc)
            return []


# ============================================================================
# Synthetic Data Generator
# ============================================================================
class SyntheticDataGenerator:
    """Generate realistic synthetic ESG controversy articles for development.

    Produces labelled articles across all 9 controversy categories plus
    a non-ESG category, using industry-specific templates with randomised
    company names, dates, locations, and monetary amounts.
    """

    # Template building blocks
    COMPANIES = [
        "Acme Corp", "GlobalTech Industries", "Pacific Energy Holdings",
        "Continental Mining Group", "Atlas Pharmaceuticals", "Pinnacle Financial",
        "Nexus Semiconductor", "Vanguard Chemical Corp", "Sterling Bank Group",
        "Meridian Oil & Gas", "Apex Consumer Goods", "Horizon Telecom",
        "Titan Automotive", "Crestview Capital", "Solaris Energy",
        "OceanView Foods", "Ironclad Manufacturing", "Summit Healthcare",
        "Cascade Logistics", "Zenith Technology", "Nordic Shipping Ltd",
        "Emerald Agriculture", "Quantum Data Systems", "Regency Hotels Group",
        "Citadel Defense Corp",
    ]

    LOCATIONS = [
        "Houston, Texas", "London, UK", "Frankfurt, Germany", "Lagos, Nigeria",
        "Mumbai, India", "São Paulo, Brazil", "Beijing, China", "Sydney, Australia",
        "Singapore", "Toronto, Canada", "Tokyo, Japan", "Johannesburg, South Africa",
        "Dubai, UAE", "Mexico City, Mexico", "Jakarta, Indonesia",
    ]

    SOURCES = [
        "Reuters", "Bloomberg", "BBC", "Financial Times", "The Guardian",
        "Wall Street Journal", "Associated Press", "CNBC", "Al Jazeera",
        "The New York Times", "The Washington Post", "Forbes",
    ]

    # Per-category article templates (title_template, body_template)
    TEMPLATES: dict[str, list[tuple[str, str]]] = {
        "environmental_violation": [
            (
                "{company} fined ${amount}M for illegal waste dumping in {location}",
                "{company} has been ordered to pay ${amount} million in penalties after environmental "
                "regulators discovered the company had been illegally dumping toxic industrial waste near "
                "residential areas in {location}. The investigation, which began {months} months ago, found "
                "contaminated groundwater levels exceeding safe limits by {factor}x. Local residents have "
                "reported increased health issues including respiratory problems. The company's CEO issued "
                "a statement promising to implement corrective measures and cooperate fully with authorities."
            ),
            (
                "{company} faces EPA enforcement action over air quality violations",
                "The Environmental Protection Agency has initiated enforcement proceedings against {company} "
                "for persistent violations of Clean Air Act standards at its {location} facility. Monitoring "
                "data shows emissions of nitrogen oxides and particulate matter exceeded permitted levels on "
                "{count} separate occasions over the past {months} months. The company could face fines of "
                "up to ${amount} million per violation day. Environmental groups have called for the facility's "
                "operating permit to be revoked pending a full environmental impact assessment."
            ),
            (
                "Oil spill at {company} facility contaminates {location} waterway",
                "An estimated {amount} thousand barrels of crude oil leaked from a storage facility operated "
                "by {company} in {location}, contaminating a major waterway and threatening local ecosystems. "
                "Emergency response teams have been deployed, but environmental experts warn the cleanup could "
                "take {months} months. Wildlife rescuers report finding oil-covered birds and marine life in the "
                "affected area. Government officials have demanded {company} cover all cleanup costs."
            ),
        ],
        "carbon_fraud": [
            (
                "{company} investigated for fabricating carbon offset certificates",
                "Federal investigators have launched a probe into {company} after whistleblower documents "
                "suggested the firm fabricated carbon offset certificates worth ${amount} million. The alleged "
                "scheme involved creating fictitious reforestation projects in {location} that never materialised. "
                "If proven, the fraud would represent one of the largest carbon market manipulations in history. "
                "The company's shares dropped {pct}% following the announcement. Industry experts warn this could "
                "undermine confidence in voluntary carbon markets globally."
            ),
            (
                "{company} accused of greenwashing with inflated emissions reduction claims",
                "A forensic analysis by climate researchers has revealed that {company}'s reported emissions "
                "reductions were overstated by approximately {pct}% over the past {months} months. The company "
                "allegedly used outdated baselines and excluded significant scope 3 emissions from its calculations. "
                "Investors who purchased green bonds issued by {company} totalling ${amount} million are now "
                "considering legal action. Regulatory bodies in {location} have opened an investigation."
            ),
            (
                "Carbon credit trading scandal implicates {company} executives",
                "Senior executives at {company} have been implicated in a carbon credit trading scandal involving "
                "the systematic inflation of verified emission reduction units. Documents obtained by investigators "
                "show that the company's {location} subsidiary generated approximately ${amount} million in revenue "
                "from credits linked to projects that delivered less than {pct}% of their claimed environmental "
                "benefits. Three senior managers have been placed on administrative leave pending the investigation."
            ),
        ],
        "labour_dispute": [
            (
                "Workers at {company} {location} plant launch indefinite strike",
                "Over {count} workers at {company}'s manufacturing plant in {location} have begun an indefinite "
                "strike, demanding higher wages, improved safety conditions, and better healthcare benefits. "
                "Union representatives say management has refused to negotiate in good faith for the past {months} "
                "months. The strike is expected to disrupt production worth ${amount} million per week. Reports of "
                "unsafe working conditions, including exposure to hazardous chemicals without proper equipment, "
                "have been cited as key grievances."
            ),
            (
                "{company} hit with class-action lawsuit over wage theft allegations",
                "A class-action lawsuit filed on behalf of {count} current and former employees alleges that "
                "{company} systematically underpaid workers at its {location} operations by misclassifying them "
                "as independent contractors. The lawsuit seeks ${amount} million in back wages and damages. "
                "Plaintiffs allege the company avoided paying overtime, health insurance contributions, and "
                "retirement benefits. Labour advocates say this is part of a broader pattern in the industry."
            ),
            (
                "OSHA investigation finds {company} failed to report {count} workplace injuries",
                "An investigation by the Occupational Safety and Health Administration has found that {company} "
                "failed to report {count} serious workplace injuries at its {location} facility over the past "
                "{months} months. The violations include unreported fractures, chemical burns, and repetitive "
                "strain injuries. OSHA has proposed penalties of ${amount} million. The company disputes the "
                "findings, claiming that all incidents were properly logged in internal systems."
            ),
        ],
        "supply_chain_abuse": [
            (
                "Investigation reveals forced labour in {company} supply chain in {location}",
                "An investigative report has uncovered evidence of forced labour practices in the supply chain "
                "of {company}, specifically at supplier facilities in {location}. Workers were allegedly subjected "
                "to confiscation of identity documents, restricted movement, and wage withholding. The company "
                "sources approximately ${amount} million worth of raw materials annually from the region. Human "
                "rights organisations are calling for immediate supply chain audits and remediation. {company} "
                "has pledged to investigate the allegations and terminate contracts with non-compliant suppliers."
            ),
            (
                "{company} supplier accused of employing child workers in {location}",
                "A major supplier to {company} has been accused of employing children as young as {count} years "
                "old at its production facilities in {location}. The allegations emerged from an undercover "
                "investigation conducted over {months} months. {company}, which has an annual procurement budget "
                "of ${amount} billion, has faced sharp criticism from child welfare organisations. The company "
                "says it has zero tolerance for child labour and is conducting an immediate audit."
            ),
            (
                "Audit reveals hazardous conditions at {company} supplier factories in {location}",
                "An independent audit of {company}'s tier-2 supplier factories in {location} has revealed "
                "widespread safety violations including blocked fire exits, inadequate ventilation, and exposure "
                "to toxic chemicals without protective equipment. {count} workers were found to be working shifts "
                "exceeding {months}0 hours per week. The audit covered {count} facilities responsible for "
                "approximately ${amount} million in annual production for {company}."
            ),
        ],
        "data_breach": [
            (
                "{company} confirms massive data breach affecting {count} million customers",
                "{company} has disclosed a major cybersecurity breach that exposed personal data of approximately "
                "{count} million customers. The compromised information includes names, email addresses, phone "
                "numbers, and in some cases, encrypted payment card details. The breach, which occurred at the "
                "company's {location} data centre, went undetected for {months} months. Cybersecurity experts "
                "estimate the total cost of the breach, including regulatory fines and remediation, could reach "
                "${amount} million. The company has engaged forensic investigators and is offering affected "
                "customers free identity theft monitoring."
            ),
            (
                "Ransomware attack on {company} systems disrupts operations for {months} weeks",
                "A sophisticated ransomware attack has crippled {company}'s IT infrastructure, forcing the "
                "company to shut down operations at its {location} headquarters and multiple regional offices. "
                "The attackers are demanding a ransom of ${amount} million in cryptocurrency. Internal documents "
                "suggest the attack exploited a known vulnerability that had not been patched for {months} months. "
                "{count} employees have been unable to access work systems during the outage."
            ),
            (
                "{company} employee database leaked on dark web forums",
                "Sensitive employee records from {company}, including Social Security numbers, salary information, "
                "and performance reviews for {count} thousand staff members, have appeared on dark web forums. "
                "The company's {location} office discovered the leak during a routine security scan. Investigation "
                "indicates the data was exfiltrated over a period of {months} months through a compromised "
                "administrative account. {company} faces potential fines of up to ${amount} million under "
                "data protection regulations."
            ),
        ],
        "privacy_violation": [
            (
                "{company} sued for secretly tracking users without consent",
                "A lawsuit filed in {location} alleges that {company} secretly collected location data, browsing "
                "habits, and app usage patterns from {count} million users without obtaining proper consent. The "
                "complaint claims the company sold aggregated user data to third-party advertisers for ${amount} "
                "million in revenue. Privacy advocates say this represents a systematic violation of data protection "
                "regulations. {company} maintains its data practices comply with its terms of service."
            ),
            (
                "Regulator fines {company} ${amount}M for GDPR violations in {location}",
                "Data protection authorities in {location} have imposed a fine of ${amount} million on {company} "
                "for multiple violations of the General Data Protection Regulation. The investigation found that "
                "the company processed personal data of {count} million EU residents without a valid legal basis, "
                "failed to respond to data subject access requests within the required timeframe, and lacked "
                "adequate technical measures to protect user data. The ruling sets a precedent for enforcement "
                "against technology firms in the region."
            ),
            (
                "{company} whistleblower reveals internal surveillance of employees",
                "A former employee of {company} has come forward with evidence that the company deployed "
                "invasive surveillance software on {count} thousand employee devices at its {location} offices "
                "without informing staff. The software reportedly captured screenshots, logged keystrokes, and "
                "monitored email communications. Labour unions have demanded an immediate investigation and "
                "potential damages of ${amount} million. The whistleblower alleges management used the data "
                "to identify and penalise union organisers."
            ),
        ],
        "bribery_corruption": [
            (
                "{company} executives indicted on foreign bribery charges in {location}",
                "A federal grand jury has indicted three senior executives of {company} on charges of violating "
                "the Foreign Corrupt Practices Act. The indictment alleges the executives authorised payments "
                "totalling ${amount} million to government officials in {location} to secure lucrative contracts "
                "worth over ${amount}0 million. The scheme allegedly operated for {months} years through a network "
                "of shell companies and intermediaries. If convicted, the executives face up to {count}0 years "
                "in prison. {company}'s board has suspended the individuals pending the outcome of the trial."
            ),
            (
                "Anti-corruption probe targets {company} operations in {location}",
                "Authorities in {location} have launched a major anti-corruption investigation into {company}'s "
                "local operations. Prosecutors allege the company made illicit payments of ${amount} million to "
                "expedite permits and circumvent environmental regulations. Seized documents reportedly show a "
                "systematic pattern of payments to {count} government officials over {months} months. The company's "
                "shares fell {pct}% on the news. Industry analysts say a guilty verdict could bar {company} "
                "from future government contracts in the region."
            ),
            (
                "{company} reaches ${amount}M settlement in kickback scheme investigation",
                "{company} has agreed to pay ${amount} million to settle allegations that its employees engaged "
                "in a kickback scheme with procurement officials. The Department of Justice investigation found "
                "that the company's {location} division paid inflated commissions to agents who then funnelled "
                "funds to decision-makers at client organisations. The settlement includes a deferred prosecution "
                "agreement requiring {company} to implement enhanced compliance monitoring for {months} years."
            ),
        ],
        "board_misconduct": [
            (
                "{company} board member resigns amid insider trading investigation",
                "A senior board member of {company} has resigned after securities regulators in {location} opened "
                "an investigation into suspected insider trading. The director allegedly traded ${amount} million "
                "worth of company shares in the days before a major earnings announcement, generating profits of "
                "approximately ${amount} million. The trades occurred across {count} separate brokerage accounts. "
                "{company}'s board has engaged independent counsel to review the matter and has pledged full "
                "cooperation with regulators."
            ),
            (
                "Shareholder revolt at {company} over excessive executive compensation",
                "Shareholders of {company} have voted against the company's executive compensation package in "
                "a rare say-on-pay defeat. The rejected package proposed ${amount} million in total compensation "
                "for the CEO, representing a {pct}% increase despite the company's stock declining {pct}% over "
                "the past {months} months. Proxy advisory firms had recommended voting against the plan, citing "
                "a disconnect between pay and performance. The vote, while non-binding, sends a strong signal "
                "of investor dissatisfaction with the board's governance practices."
            ),
            (
                "{company} CEO fired for undisclosed conflicts of interest",
                "The board of directors at {company} has terminated its CEO after an internal investigation "
                "revealed undisclosed conflicts of interest. The CEO allegedly directed ${amount} million in "
                "company contracts to firms owned by family members without board approval. The investigation, "
                "which covered transactions over {months} months, also found that the CEO failed to disclose "
                "board positions at {count} companies that had commercial relationships with {company}. The "
                "company has appointed an interim CEO and is reviewing all related-party transactions."
            ),
        ],
        "community_impact": [
            (
                "Residents protest {company} factory expansion in {location}",
                "Hundreds of residents in {location} have staged protests against {company}'s planned factory "
                "expansion, citing concerns over air pollution, noise, and loss of green space. The ${amount} "
                "million project would increase the facility's production capacity by {pct}% but residents say "
                "the environmental impact assessment failed to consider effects on {count} nearby schools and "
                "residential areas. Local council members have called for a public hearing before approving "
                "the expansion permits. Environmental groups have filed objections with the planning authority."
            ),
            (
                "{company} mining project threatens indigenous lands in {location}",
                "Indigenous communities in {location} are opposing a major mining project proposed by {company}, "
                "which would affect {count} thousand hectares of ancestral land. The ${amount} million project "
                "has drawn criticism from human rights organisations who argue that the company failed to conduct "
                "adequate free, prior, and informed consent processes. Water testing in the area has already "
                "shown elevated levels of heavy metals linked to exploratory drilling. Community leaders are "
                "seeking a court injunction to halt all activities on the site."
            ),
            (
                "Health study links {company} operations to increased disease rates in {location}",
                "A peer-reviewed health study has found statistically significant increases in respiratory "
                "disease and cancer rates among communities living within {count} kilometres of {company}'s "
                "processing plant in {location}. Researchers tracked health outcomes of {count} thousand "
                "residents over {months} years and found a {pct}% increase in lung disease diagnoses. {company} "
                "disputes the study's methodology but has agreed to fund an independent review costing ${amount} "
                "million. Local authorities are considering tightening emissions standards for the facility."
            ),
        ],
    }

    # Non-ESG templates for negative examples
    NON_ESG_TEMPLATES = [
        (
            "{company} reports strong Q{quarter} earnings, beating analyst estimates",
            "{company} has reported quarterly revenue of ${amount} billion, exceeding Wall Street estimates "
            "by {pct}%. The company attributed the strong performance to increased demand for its products "
            "in {location} and successful cost optimisation measures. CEO stated that the company remains "
            "well-positioned for continued growth. Shares rose {pct}% in after-hours trading."
        ),
        (
            "{company} announces new product line targeting {location} market",
            "{company} unveiled its latest product lineup designed for the growing {location} market at "
            "an industry event today. The new products represent an investment of approximately ${amount} "
            "million in research and development over {months} months. Analysts expect the launch to "
            "contribute significantly to revenue growth in the coming quarters."
        ),
        (
            "{company} completes acquisition of startup for ${amount}M",
            "{company} has completed its acquisition of a technology startup for ${amount} million in cash "
            "and stock. The deal, first announced {months} months ago, is expected to enhance {company}'s "
            "capabilities in artificial intelligence and cloud computing. Approximately {count} employees "
            "from the acquired firm will join {company}'s {location} research centre."
        ),
        (
            "Analysts upgrade {company} stock to Buy, cite growth potential",
            "Investment bank analysts have upgraded {company}'s stock from Hold to Buy with a price target "
            "of ${amount}, representing a {pct}% upside from current levels. The upgrade cites the company's "
            "strong market position in {location}, improving margins, and a robust pipeline of new products. "
            "Trading volume surged following the recommendation."
        ),
        (
            "{company} partners with university for AI research initiative",
            "{company} has announced a ${amount} million research partnership with a leading university in "
            "{location} to advance artificial intelligence applications. The {months}-year collaboration "
            "will focus on natural language processing and computer vision technologies. The initiative "
            "will create {count} new research positions and fund {count} doctoral scholarships."
        ),
    ]

    def __init__(self, num_articles: int | None = None, seed: int = 42):
        self.num_articles = num_articles or config.SYNTHETIC_NUM_ARTICLES
        self.seed = seed

    def generate(self) -> pd.DataFrame:
        """Generate a DataFrame of synthetic ESG controversy articles.

        Returns:
            DataFrame with columns: title, text, source, published_at, url, company,
            category, is_esg, label_id
        """
        random.seed(self.seed)
        records = []

        # Allocate articles: ~105 per ESG category + ~55 non-ESG
        esg_per_category = (self.num_articles - 55) // len(config.CONTROVERSY_CATEGORIES)
        non_esg_count = self.num_articles - esg_per_category * len(config.CONTROVERSY_CATEGORIES)

        # Generate ESG articles
        for category in config.CONTROVERSY_CATEGORIES:
            templates = self.TEMPLATES[category]
            for i in range(esg_per_category):
                template = random.choice(templates)
                record = self._fill_template(template, category)
                record["category"] = category
                record["is_esg"] = 1
                record["label_id"] = config.CATEGORY_TO_ID[category]
                records.append(record)

        # Generate non-ESG articles
        for i in range(non_esg_count):
            template = random.choice(self.NON_ESG_TEMPLATES)
            record = self._fill_template(template, "none")
            record["category"] = "none"
            record["is_esg"] = 0
            record["label_id"] = -1  # Not a controversy category
            records.append(record)

        # Shuffle
        random.shuffle(records)
        df = pd.DataFrame(records)

        # Save to CSV
        output_path = config.SYNTHETIC_OUTPUT_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(
            "Generated %d synthetic articles (%d ESG, %d non-ESG). Saved to %s",
            len(df), df["is_esg"].sum(), len(df) - df["is_esg"].sum(), output_path,
        )

        return df

    def _fill_template(self, template: tuple[str, str], category: str) -> dict:
        """Fill a template with random values to create a realistic article."""
        company = random.choice(self.COMPANIES)
        location = random.choice(self.LOCATIONS)
        source = random.choice(self.SOURCES)

        # Random values for placeholders
        fill_values = {
            "company": company,
            "location": location,
            "amount": random.randint(1, 500),
            "count": random.randint(3, 5000),
            "months": random.randint(2, 24),
            "pct": random.randint(2, 45),
            "factor": random.randint(2, 20),
            "quarter": random.randint(1, 4),
        }

        title = template[0].format(**fill_values)
        body = template[1].format(**fill_values)

        # Random date within last 90 days
        days_ago = random.randint(0, 90)
        pub_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "title": title,
            "text": body,
            "source": source,
            "published_at": pub_date,
            "url": f"https://example.com/articles/{random.randint(10000, 99999)}",
            "company": company,
        }


# ============================================================================
# Unified ingestion interface
# ============================================================================
def ingest_all(
    companies: list[str] | None = None,
    use_synthetic: bool = True,
) -> pd.DataFrame:
    """Run the full ingestion pipeline.

    Attempts live APIs when keys are available, falls back to synthetic data.

    Args:
        companies: List of company names to search. Defaults to seeded companies.
        use_synthetic: If True, generate synthetic data as fallback.

    Returns:
        Combined DataFrame of all ingested articles.
    """
    companies = companies or [c["name"] for c in config.SEEDED_COMPANIES]
    all_articles = []

    # Try live APIs
    if config.NEWSAPI_KEY:
        ingester = NewsAPIIngester()
        for company in companies:
            articles = ingester.fetch_articles(company)
            all_articles.extend(articles)
            logger.info("NewsAPI: %d articles for %s", len(articles), company)

    if config.GUARDIAN_API_KEY:
        ingester = GuardianIngester()
        for company in companies:
            for keyword in config.ESG_KEYWORDS[:5]:
                query = f"{company} {keyword}"
                articles = ingester.fetch_articles(query)
                all_articles.extend(articles)

    # SEC EDGAR (no key required, just User-Agent)
    try:
        edgar = SECEdgarIngester()
        for company in companies:
            filings = edgar.search_filings(f"{company} environmental social governance")
            all_articles.extend(filings)
    except Exception as exc:
        logger.warning("SEC EDGAR ingestion failed: %s", exc)

    # Fallback to synthetic data if no live articles or explicitly requested
    if not all_articles or use_synthetic:
        logger.info("Using synthetic data generator (fallback mode)")
        generator = SyntheticDataGenerator()
        df = generator.generate()
        return df

    df = pd.DataFrame(all_articles)
    output_path = config.DATA_RAW_DIR / "live_articles.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved %d live articles to %s", len(df), output_path)
    return df


# ============================================================================
# CLI entry point
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESG Controversy Data Ingestion")
    parser.add_argument("--live", action="store_true", help="Attempt live API ingestion")
    parser.add_argument("--synthetic-only", action="store_true", help="Only generate synthetic data")
    args = parser.parse_args()

    if args.synthetic_only or not args.live:
        generator = SyntheticDataGenerator()
        df = generator.generate()
        print(f"\nSynthetic data generated: {len(df)} articles")
        print(f"Category distribution:\n{df['category'].value_counts().to_string()}")
    else:
        df = ingest_all(use_synthetic=True)
        print(f"\nIngested {len(df)} articles total")
