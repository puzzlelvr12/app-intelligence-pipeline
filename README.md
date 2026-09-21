# App Store Product Intelligence & Feedback Telemetry Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![SQLite 3 WAL](https://img.shields.io/badge/storage-SQLite3_WAL-003B57.svg)](https://www.sqlite.org/wal.html)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Code Style: Enterprise](https://img.shields.io/badge/code%20standard-Apple%20Engineering-black.svg)]()

An end-to-end data engineering and product telemetry pipeline designed to transform massive volumes of unstructured App Store and Google Play customer reviews into actionable release intelligence, regression signals, and executive dashboards.

---

## 1. Executive Summary & Problem Context

At enterprise scale (such as the **Apple App Store**, **Apple Music**, or **iOS Core Applications**), software engineering organizations receive millions of user reviews each week. The challenge is not data volume—it is **signal-to-noise ratio**:
- Critical release regressions (e.g., audio pipeline crashes or crossfade bugs after a minor update) often get buried under generic praise or localized billing complaints.
- Traditional LLM-based categorization pipelines are computationally expensive, slow, non-deterministic, and suffer from latency bottlenecks when triaging high-velocity release streams.
- Product teams need an instantaneous, deterministic, and auditable telemetry engine that categorizes customer feedback into canonical triage buckets (`Bug Report`, `Feature Request`, `UI Friction`, `Praise`), scores emotional polarity, tracks release version regressions, and indexes records for sub-millisecond analytical querying.

This repository demonstrates a **production-ready reference architecture** that solves this problem end-to-end.

---

## 2. System Architecture

The pipeline follows a clean decoupled ETL architecture: Ingestion &rarr; Validation & NLP Transformation &rarr; Relational Core &rarr; Executive Telemetry Dashboard.

```
+---------------------------------------------------------------------------------------+
|                                DATA INGESTION ENGINE                                  |
|                                     (scraper.py)                                      |
|  +---------------------------+   Exponential Backoff   +---------------------------+  |
|  | Google Play / App Store   | ----------------------> |  Paginator & Sanitizer    |  |
|  | Public Telemetry Streams  |      (Max 3 retries)    |  - Strip null bytes       |  |
|  +---------------------------+                         |  - Normalize whitespace   |  |
|                                                        |  - Metadata standardization| |
+---------------------------------------------------------------------------------------+
                                           |
                                           v  UTF-8 Encoded CSV
+---------------------------------------------------------------------------------------+
|                               ETL & NLP SEMANTIC ENGINE                               |
|                                     (pipeline.py)                                     |
|                                                                                       |
|   +-------------------------------------------------------------------------------+   |
|   | 1. Record Validation & Schema Hygiene                                         |   |
|   |    - Missing field checks, deduplication by review_id, rating clipping (1-5)  |   |
|   +-------------------------------------------------------------------------------+   |
|                                           |                                           |
|   +---------------------------------------+---------------------------------------+   |
|   |                                                                               |   |
|   v                                                                               v   |
|  +-------------------------------------+   +-------------------------------------+    |
|  |    150+ Keyword Mobile Taxonomy     |   |       VADER Sentiment Engine        |    |
|  |     (Pre-compiled Regex Engine)     |   |      (Valence Polarity Scoring)     |    |
|  |  Bug / Feature / Friction / Praise  |   |     [-1.000 to +1.000, Compound]    |    |
|  +-------------------------------------+   +-------------------------------------+    |
|   |                                                                               |   |
|   +---------------------------------------+---------------------------------------+   |
|                                           |                                           |
|                                           v                                           |
|   +-------------------------------------------------------------------------------+   |
|   | 2. Fallback Heuristic Classifier (Zero Unclassified Records)                   |   |
|   |    Multi-tier resolution: Keyword Density -> Polarity -> Star Rating Vectors  |   |
|   +-------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------+
                                           |
                                           v  Guaranteed Context Manager (Zero Leaks)
+---------------------------------------------------------------------------------------+
|                             RELATIONAL TELEMETRY STORAGE                              |
|                                     (reviews.db)                                      |
|                                                                                       |
|   - SQLite 3 with Write-Ahead Logging (WAL Mode) & Normal Synchronization             |
|   - Strict Constraints: CHECK (rating 1-5), CHECK (sentiment -1.0 to 1.0)             |
|   - Secondary B-Tree Indexes: app_name, category, sentiment, review_date, app_version |
|   - Composite Index: idx_reviews_app_cat_ver (app_name, category, app_version)       |
|   - Idempotent Atomic Upserts: ON CONFLICT(review_id) DO UPDATE                       |
+---------------------------------------------------------------------------------------+
                                           |
                                           v  Sub-millisecond Indexed Reads
+---------------------------------------------------------------------------------------+
|                            EXECUTIVE INTELLIGENCE DASHBOARD                           |
|                                       (app.py)                                        |
|                                                                                       |
|   - Executive KPI Grid: Volume, Bug Rate %, Net Sentiment Score, Praise CSAT %        |
|   - Semantic Release Regression Telemetry: Semantic version sorting & spike alerts    |
|   - Interactive Plotly Visualizations: Category mix, sentiment polarization, ratings  |
|   - Real-Time Inference Playground: Live customer review triage & entity extraction   |
+---------------------------------------------------------------------------------------+
```

---

## 3. Technology Stack & Architectural Decisions

| Layer | Technology | Engineering Rationale |
| :--- | :--- | :--- |
| **Ingestion Engine** | `google-play-scraper`, `pandas` | High-throughput, resilient API harvesting with exponential backoff and rate-limit pacing. |
| **Data Cleaning** | Python `typing`, Regex, UTF-8 Normalization | Defends against null bytes, surrogate encoding corruption on Windows, and malformed timestamps. |
| **NLP & Sentiment** | `vaderSentiment`, Pre-compiled Regex | Sub-millisecond deterministic inference. Unlike external LLM APIs, executes locally at zero marginal cost with 100% auditability and reproducibility. |
| **Relational Storage**| SQLite 3 in WAL Mode | Zero-config, serverless, embeddable relational engine. WAL mode allows concurrent reads during writes; compound B-Tree indexes ensure `<2ms` query execution. |
| **Visualization UI** | `streamlit` (v1.64+), `plotly` | Reactive executive dashboard utilizing Apple Human Interface-inspired dark mode styling, responsive CSS cards, and interactive SVG/WebGL charts. |

---

## 4. Key Business Insights Extracted: Release Regression Case Study

Using **1,000+ real-world production reviews from Spotify (`com.spotify.music`)**, the pipeline extracted significant product intelligence:

```
Total Ingested Reviews:  1,000
Praise / CSAT:           72.9% (729 records)
Bug Report Defect Rate:  15.4% (154 records)
UI Friction Issues:       6.0% (60 records)
Feature Requests:         5.7% (57 records)
Net Average Sentiment:   +0.428 (Positive Baseline)
```

### Critical Version Regression Finding (v9.1.80 &rarr; v9.1.82)
1. **The Regression Spike**: In version `9.1.80`, the bug report incidence was maintained at a baseline of **~12.3%**. Upon rolling out version `9.1.82.2161`, the bug incidence spiked dramatically to **21.8%** among versioned complaints.
2. **Root Cause Extraction via Semantic Entities**:
   - Querying the database for keyword co-occurrences in version `9.1.82` revealed a severe regression in the **Audio Crossfade & Auto-mix engine**:
   > *"Not happy with the recent update. Playlists and Albums automatically Crossfade, even though Crossfade is turned off (0 seconds) and auto mix is turned off. FIX YOUR APP."*
   - Rather than waiting for monthly aggregated App Store ratings to drop, the pipeline's **Release Regression Telemetry** triggers an automated regression alert immediately upon deployment, allowing release engineers to halt phased rollouts before mass user churn.

---

## 5. Production SQL Analytical Queries

The SQLite database is pre-indexed for high-performance business intelligence. Below are 5 analytical queries used by engineering and product leadership:

### Query 1: Release Version Regression Severity Analysis
*Identifies which app version has the highest proportion of reported defects to isolate faulty rollouts.*
```sql
SELECT
    app_version,
    COUNT(*) AS total_reviews,
    SUM(CASE WHEN category = 'Bug Report' THEN 1 ELSE 0 END) AS bug_count,
    ROUND(CAST(SUM(CASE WHEN category = 'Bug Report' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) AS bug_rate_pct,
    ROUND(AVG(rating), 2) AS avg_rating,
    ROUND(AVG(sentiment_score), 3) AS avg_sentiment
FROM reviews
WHERE app_version != 'Unknown'
GROUP BY app_version
HAVING total_reviews >= 10
ORDER BY bug_rate_pct DESC;
```

### Query 2: High-Impact Feature Requests Ranked by Community Engagement
*Surfaces the most requested product capabilities weighted by upvotes (`thumbs_up`).*
```sql
SELECT
    review_text,
    user_name,
    rating,
    thumbs_up,
    review_date
FROM reviews
WHERE category = 'Feature Request'
ORDER BY thumbs_up DESC, rating ASC
LIMIT 10;
```

### Query 3: Net Sentiment Score (NSS) & Churn Risk Segmentation
*Calculates net sentiment score (Positive % minus Negative %) grouped by application.*
```sql
SELECT
    app_name,
    COUNT(*) AS total_sample,
    ROUND(SUM(CASE WHEN sentiment_label = 'Positive' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) AS positive_pct,
    ROUND(SUM(CASE WHEN sentiment_label = 'Negative' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) AS negative_pct,
    ROUND((SUM(CASE WHEN sentiment_label = 'Positive' THEN 1.0 ELSE 0 END) - 
           SUM(CASE WHEN sentiment_label = 'Negative' THEN 1.0 ELSE 0 END)) / COUNT(*) * 100, 1) AS net_sentiment_score,
    ROUND(AVG(rating), 2) AS avg_star_rating
FROM reviews
GROUP BY app_name;
```

### Query 4: UI Friction & Usability Complaints on Recent Builds
*Filters negative sentiment reviews specifically mentioning navigation, layouts, or redesign issues.*
```sql
SELECT
    app_version,
    rating,
    sentiment_score,
    review_text
FROM reviews
WHERE category = 'UI Friction'
  AND rating <= 2
ORDER BY review_date DESC
LIMIT 15;
```

### Query 5: Developer Response Effectiveness Analysis
*Measures whether developer replies correlate with higher-star ratings or are deployed exclusively to 1-star reviews.*
```sql
SELECT
    CASE WHEN dev_reply IS NOT NULL THEN 'Developer Replied' ELSE 'No Reply' END AS reply_status,
    COUNT(*) AS total_reviews,
    ROUND(AVG(rating), 2) AS avg_rating,
    ROUND(AVG(sentiment_score), 3) AS avg_sentiment,
    ROUND(SUM(CASE WHEN category = 'Bug Report' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) AS bug_ratio_pct
FROM reviews
GROUP BY reply_status;
```

---

## 6. Local Setup & Execution Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12+
- Git

### 1. Clone the Repository & Create Virtual Environment
```powershell
# Windows PowerShell
git clone <YOUR_REPO_URL>
cd app-intelligence-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
```bash
# macOS / Linux
git clone <YOUR_REPO_URL>
cd app-intelligence-pipeline
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Ingest Fresh App Store Reviews (Optional — Pre-seeded dataset included)
```powershell
# Default: Harvests 1,000 Spotify reviews
python scraper.py

# Or target other major applications
python scraper.py --app duolingo --count 1000
python scraper.py --app netflix --count 1000
```

### 4. Execute the Production ETL Pipeline
```powershell
python pipeline.py
```
*Processes raw data, executes NLP classification, performs VADER sentiment scoring, creates relational indexes, and executes batch upserts.*

### 5. Launch the Executive Dashboard
```powershell
streamlit run app.py
```
The dashboard will open automatically at **`http://localhost:8501`**.

---

## 7. Reliability, Resilience & Production Standards

1. **Guaranteed Zero Connection Leaks**: All database interactions in `pipeline.py` and `app.py` utilize Python context managers (`get_db`) with strict `try ... finally: conn.close()` guarantees.
2. **High-Performance WAL Mode**: Enables concurrent read transactions alongside background pipeline writes, preventing SQLite `database is locked` exceptions under concurrent load.
3. **Idempotent Pipeline Upserts**: Data ingestion is strictly idempotent. Repeated execution of `pipeline.py` safely updates records via `ON CONFLICT(review_id) DO UPDATE` without creating duplicate entries.
4. **Resilient Network Harvesting**: `scraper.py` implements an exponential backoff loop (up to 3 retries) to survive transient network drops or Google Play throttling.
5. **Encoding Purity**: Enforces explicit UTF-8 encoding across all CSV file handles and database writes, eliminating Windows ANSI/cp1252 corruption.

---

## 8. Author & Portfolio Contact

- **Project**: App Store Product Intelligence & Feedback Telemetry Pipeline
- **Target Role**: Data Engineer / Software Engineer (Apple Services, Apple Music, App Store)
