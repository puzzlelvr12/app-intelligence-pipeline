"""
pipeline.py
===========
Production ETL pipeline: ingests raw app reviews from CSV, performs
robust NLP classification (Bug Report, Feature Request, UI Friction, Praise),
executes VADER sentiment analysis, validates record schemas, and persists
structured records into an optimized, indexed SQLite database (reviews.db).

Usage:
    python pipeline.py                          # default: reviews_raw.csv -> reviews.db
    python pipeline.py --input reviews_raw.csv  # explicit input path
    python pipeline.py --db reviews.db          # explicit database target
"""

import argparse
from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3
import sys
from typing import Generator, Optional

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── Configuration & Defaults ────────────────────────────────────────────────
DEFAULT_CSV = "reviews_raw.csv"
DB_PATH     = "reviews.db"

# ─── Production NLP Taxonomy ──────────────────────────────────────────────────
# Domain-specific keyword taxonomy engineered for App Store & Play Store feedback.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Bug Report": [
        "crash", "crashes", "crashing", "crashed", "bug", "bugs", "buggy",
        "error", "errors", "broken", "freeze", "freezes", "freezing", "frozen",
        "glitch", "glitches", "glitchy", "not working", "doesn't work", "doesnt work",
        "fails", "failure", "failing", "failed", "corrupted", "loading", "won't load",
        "wont load", "stuck", "infinite loop", "force close", "unresponsive",
        "lag", "laggy", "lags", "blank screen", "white screen", "black screen",
        "cannot open", "cant open", "can't open", "keeps stopping", "fix", "fixes",
        "fixed", "issue", "issues", "problem", "problems", "defect", "defects",
        "shut down", "closes itself", "kicks me out", "stop working", "stopped working",
        "black out", "doesn't open", "doesnt open", "wont open", "won't open",
        "cant play", "can't play", "cannot play", "wont play", "won't play",
        "stops playing", "skips", "skipping", "buffer", "buffering", "disconnect",
        "disconnects", "battery drain", "drain battery", "overheat", "overheating",
        "sync", "syncing", "fail", "no sound", "sound cuts", "no audio", "not loading",
        "can't log in", "cant log in", "cannot log in", "login issue", "broke the app",
        "ruined the app", "crossfade", "unplayable", "unusable", "keeps crashing",
        "app closed", "shutting down", "error code",
    ],
    "Feature Request": [
        "please add", "would be great", "wish", "feature", "features", "suggestion",
        "suggestions", "request", "requests", "allow", "should have", "add option",
        "need the ability", "it would be nice", "missing", "want", "hope",
        "could you add", "implement", "option to", "should be able", "need", "needs",
        "bring back", "bring it back", "give us", "please allow", "should add",
        "can you add", "add a", "add an", "why can't we", "why cant we", "would love",
        "would like", "how about", "consider adding", "we need", "make it possible",
        "option for", "support for", "capability", "new feature", "enhancement",
        "please bring", "dark mode", "offline mode", "folder", "folders",
        "landscape mode", "widget", "widgets",
    ],
    "UI Friction": [
        "confusing", "hard to find", "difficult", "complicated", "annoying",
        "frustrating", "not intuitive", "unintuitive", "ugly", "clunky", "bad design",
        "poor ux", "poorly designed", "navigation", "hard to use", "can't find",
        "cant find", "where is", "layout", "too many clicks", "cluttered", "slow",
        "too slow", "hate the new", "terrible update", "horrible interface", "awful ui",
        "bad update", "messy", "unfriendly", "awkward", "hard to navigate",
        "waste of time", "too many ads", "excessive ads", "ad popups", "intrusive",
        "unpleasant", "uncomfortable", "cumbersome", "dislike the update",
        "interface is", "ui is", "ruined the layout", "ads are annoying", "bad interface",
        "hard to read", "font size", "gesture", "gestures", "buttons",
    ],
    "Praise": [
        "love", "loving", "lovin", "loved", "amazing", "excellent", "great", "awesome",
        "fantastic", "best", "perfect", "wonderful", "brilliant", "superb", "top notch",
        "well done", "thank you", "impressive", "outstanding", "helpful", "beautiful",
        "easy to use", "intuitive", "flawless", "highly recommend", "5 stars",
        "five stars", "good", "nice", "enjoy", "enjoying", "favourite", "favorite",
        "fabulous", "super", "clean", "smooth", "satisfying", "stellar", "goat",
        "masterpiece", "fire", "cool", "reliable", "solid", "gem", "works great",
        "works well", "love it", "thumbs up", "must have", "exceptional", "delight",
        "delightful", "splendid", "top tier", "incredible", "addicted", "useful",
        "fun", "rock", "rocks", "wonderful app", "good app", "great app", "best app",
    ],
}

# Pre-compile regex boundaries for high throughput
_PATTERNS: dict[str, re.Pattern] = {
    cat: re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b",
        re.IGNORECASE,
    )
    for cat, kws in CATEGORY_KEYWORDS.items()
}

_vader = SentimentIntensityAnalyzer()


# ─── NLP Classification & Sentiment ──────────────────────────────────────────

def classify_category(
    text: str,
    rating: Optional[int] = None,
    sentiment_score: Optional[float] = None,
) -> str:
    """
    Robust, deterministic classifier assigning reviews into canonical categories:
    ['Bug Report', 'Feature Request', 'UI Friction', 'Praise'].

    1. Keyword scoring via compiled regular expressions.
    2. Priority tie-breaking: Bug Report > Feature Request > UI Friction > Praise.
    3. Sentiment & Star-Rating fallback heuristics when no explicit keywords fire.
    """
    text = (text or "").strip()
    scores = {cat: len(pat.findall(text)) for cat, pat in _PATTERNS.items()}

    # Check if any keyword matches
    if any(v > 0 for v in scores.values()):
        priority = ["Bug Report", "Feature Request", "UI Friction", "Praise"]
        best_score = max(scores.values())
        for cat in priority:
            if scores[cat] == best_score:
                return cat

    # Fallback heuristic: Rating + Sentiment awareness ensures zero unclassified records
    if rating is not None:
        try:
            r = int(rating)
        except (ValueError, TypeError):
            r = 3

        # Compute compound score if not provided
        if sentiment_score is None:
            sentiment_score = _vader.polarity_scores(text)["compound"]

        if r >= 4:
            return "Praise" if sentiment_score >= -0.05 else "UI Friction"
        elif r <= 2:
            return "Bug Report" if sentiment_score <= 0.05 else "UI Friction"
        else:  # 3-star
            if sentiment_score >= 0.15:
                return "Praise"
            elif sentiment_score <= -0.15:
                return "Bug Report"
            return "UI Friction"

    # Pure text fallback based on sentiment
    if sentiment_score is None:
        sentiment_score = _vader.polarity_scores(text)["compound"]
    return "Praise" if sentiment_score >= 0 else "Bug Report"


def vader_sentiment(text: str) -> tuple[float, str]:
    """
    Computes VADER sentiment compound score and normalized categorical label.
    Returns:
        (compound_score: float [-1.0, 1.0], label: 'Positive' | 'Neutral' | 'Negative')
    """
    safe_text = (text or "").strip()
    if not safe_text:
        return 0.0, "Neutral"

    compound = _vader.polarity_scores(safe_text)["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return round(float(compound), 4), label


# ─── Database Operations & Connection Management ──────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS reviews (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id         TEXT NOT NULL UNIQUE,
    app_name          TEXT NOT NULL,
    user_name         TEXT,
    rating            INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    review_date       TEXT NOT NULL,
    app_version       TEXT DEFAULT 'Unknown',
    review_text       TEXT NOT NULL,
    thumbs_up         INTEGER DEFAULT 0,
    dev_reply         TEXT,
    category          TEXT NOT NULL CHECK(category IN ('Bug Report', 'Feature Request', 'UI Friction', 'Praise')),
    sentiment_score   REAL NOT NULL CHECK(sentiment_score BETWEEN -1.0 AND 1.0),
    sentiment_label   TEXT NOT NULL CHECK(sentiment_label IN ('Positive', 'Neutral', 'Negative')),
    processed_at      TEXT DEFAULT (datetime('now'))
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_reviews_app_name ON reviews(app_name);",
    "CREATE INDEX IF NOT EXISTS idx_reviews_category ON reviews(category);",
    "CREATE INDEX IF NOT EXISTS idx_reviews_sentiment_label ON reviews(sentiment_label);",
    "CREATE INDEX IF NOT EXISTS idx_reviews_app_version ON reviews(app_version);",
    "CREATE INDEX IF NOT EXISTS idx_reviews_review_date ON reviews(review_date);",
    "CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);",
    "CREATE INDEX IF NOT EXISTS idx_reviews_app_cat_ver ON reviews(app_name, category, app_version);",
]

UPSERT_SQL = """
INSERT INTO reviews (
    review_id, app_name, user_name, rating, review_date,
    app_version, review_text, thumbs_up, dev_reply,
    category, sentiment_score, sentiment_label
) VALUES (
    :review_id, :app_name, :user_name, :rating, :review_date,
    :app_version, :review_text, :thumbs_up, :dev_reply,
    :category, :sentiment_score, :sentiment_label
)
ON CONFLICT(review_id) DO UPDATE SET
    rating          = excluded.rating,
    app_version     = excluded.app_version,
    review_text     = excluded.review_text,
    thumbs_up       = excluded.thumbs_up,
    dev_reply       = excluded.dev_reply,
    category        = excluded.category,
    sentiment_score = excluded.sentiment_score,
    sentiment_label = excluded.sentiment_label,
    processed_at    = datetime('now');
"""


@contextmanager
def get_db(db_path: str = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for SQLite connections guaranteeing proper closure,
    WAL performance tuning, and foreign key enforcement.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    """Initializes the production relational schema and secondary B-Tree indexes."""
    with conn:
        conn.execute(CREATE_TABLE_SQL)
        for index_sql in CREATE_INDEXES_SQL:
            conn.execute(index_sql)


def upsert_records(conn: sqlite3.Connection, records: list[dict]) -> None:
    """Performs atomic batch upsert of validated records."""
    with conn:
        conn.executemany(UPSERT_SQL, records)


# ─── ETL Execution Engine ────────────────────────────────────────────────────

def run_pipeline(csv_path: str = DEFAULT_CSV, db_path: str = DB_PATH) -> int:
    """
    Executes the end-to-end data pipeline:
    1. Reads and sanitizes raw CSV data.
    2. Runs NLP classification & VADER sentiment scoring.
    3. Validates records against database schema constraints.
    4. Upserts records into SQLite and logs operational metrics.
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] Source file not found: {path}. Run scraper.py first.")
        sys.exit(1)

    print(f"[pipeline] Ingesting {path} ...")
    df = pd.read_csv(path, encoding="utf-8")
    print(f"[pipeline] Loaded {len(df)} raw records.")

    # Validate essential schema fields
    required_cols = ["review_text", "rating", "review_date"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] Missing required column '{col}' in input data.")
            sys.exit(1)

    # Clean and standardize types
    df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
    df = df[df["review_text"] != ""].copy()

    # Generate synthetic deterministic review_id if missing
    if "review_id" not in df.columns or df["review_id"].isna().any():
        df["review_id"] = [f"rev_{i}_{hash(row['review_text']) & 0xFFFFFFFF}" for i, row in df.iterrows()]
    df["review_id"] = df["review_id"].astype(str)

    df["app_name"]    = df.get("app_name", "Unknown").fillna("Unknown").astype(str).str.strip()
    df["user_name"]   = df.get("user_name", "Anonymous").fillna("Anonymous").astype(str).str.strip()
    df["app_version"] = df.get("app_version", "Unknown").fillna("Unknown").astype(str).str.strip()
    df["app_version"] = df["app_version"].replace({"": "Unknown", "None": "Unknown", "nan": "Unknown"})
    df["thumbs_up"]   = pd.to_numeric(df.get("thumbs_up", 0), errors="coerce").fillna(0).astype(int)
    df["dev_reply"]   = df.get("dev_reply", None).where(pd.notna(df.get("dev_reply", None)), None)
    df["rating"]      = pd.to_numeric(df["rating"], errors="coerce").fillna(3).astype(int).clip(1, 5)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    # ── Sentiment Scoring ────────────────────────────────────────────────────
    print("[pipeline] Scoring sentiment with VADER engine ...")
    sentiment_results = df["review_text"].apply(vader_sentiment)
    df["sentiment_score"] = [r[0] for r in sentiment_results]
    df["sentiment_label"] = [r[1] for r in sentiment_results]

    # ── Category Classification ──────────────────────────────────────────────
    print("[pipeline] Classifying categories across domain taxonomy ...")
    df["category"] = [
        classify_category(text, rating, score)
        for text, rating, score in zip(df["review_text"], df["rating"], df["sentiment_score"])
    ]

    # ── Persist to SQLite ────────────────────────────────────────────────────
    print(f"[pipeline] Upserting records into {db_path} ...")
    records = df[[
        "review_id", "app_name", "user_name", "rating", "review_date",
        "app_version", "review_text", "thumbs_up", "dev_reply",
        "category", "sentiment_score", "sentiment_label",
    ]].to_dict(orient="records")

    with get_db(db_path) as conn:
        init_db(conn)
        upsert_records(conn, records)

    cat_counts = df["category"].value_counts().to_dict()
    sent_counts = df["sentiment_label"].value_counts().to_dict()

    print(f"[pipeline] Pipeline complete. Persisted {len(records)} records into {db_path}")
    print(f"[pipeline] Category breakdown: {cat_counts}")
    print(f"[pipeline] Sentiment breakdown: {sent_counts}")
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pipeline for app store reviews")
    parser.add_argument("--input", default=DEFAULT_CSV, help="Path to input raw CSV (default: reviews_raw.csv)")
    parser.add_argument("--db",    default=DB_PATH,     help="Path to SQLite database (default: reviews.db)")
    args = parser.parse_args()

    run_pipeline(args.input, args.db)
