"""
scraper.py
==========
Production-grade ingestion engine: scrapes public Google Play reviews
for target applications with exponential backoff retries, metadata
extraction, text sanitization, and UTF-8 serialization.

Usage:
    python scraper.py                  # defaults to Spotify (1,000 reviews)
    python scraper.py --app duolingo   # scrapes Duolingo
    python scraper.py --app spotify --count 1000
"""

import argparse
import sys
import time
from typing import Optional
import pandas as pd
from google_play_scraper import reviews, Sort

# ─── App registry ────────────────────────────────────────────────────────────
APP_IDS = {
    "spotify":  "com.spotify.music",
    "duolingo": "com.duolingo",
    "netflix":  "com.netflix.mediaclient",
    "youtube":  "com.google.android.youtube",
}

OUTPUT_FILE = "reviews_raw.csv"
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0  # seconds


def sanitize_text(text: Optional[str]) -> str:
    """Normalize whitespace and strip null bytes / non-printable control chars."""
    if not text:
        return ""
    # Remove null bytes and control chars except standard whitespace
    cleaned = "".join(ch for ch in str(text) if ch == "\t" or ch == "\n" or ch >= " ")
    return " ".join(cleaned.split())


def scrape(app_name: str, count: int = 1000) -> pd.DataFrame:
    """
    Scrapes up to `count` reviews for `app_name` from Google Play.
    Handles pagination, transient rate limits, and network retries.
    """
    app_id = APP_IDS.get(app_name.lower())
    if app_id is None:
        print(f"[ERROR] Unknown app '{app_name}'. Choose from: {list(APP_IDS)}")
        sys.exit(1)

    print(f"[scraper] Fetching up to {count} reviews for '{app_name}' ({app_id}) ...")

    all_reviews: list[dict] = []
    continuation_token = None
    batch_size = 200  # API page size

    while len(all_reviews) < count:
        remaining = count - len(all_reviews)
        fetch_count = min(batch_size, remaining)
        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result, continuation_token = reviews(
                    app_id,
                    lang="en",
                    country="us",
                    sort=Sort.NEWEST,
                    count=fetch_count,
                    continuation_token=continuation_token,
                )
                success = True
                break
            except Exception as e:
                backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
                print(f"[scraper] Warning: batch request failed (attempt {attempt}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES:
                    print(f"[scraper] Retrying in {backoff:.1f}s ...")
                    time.sleep(backoff)
                else:
                    print(f"[ERROR] Maximum retries exceeded for '{app_name}'.")

        if not success or not result:
            if not success:
                print("[scraper] Ingestion aborted due to repeated network errors.")
            else:
                print("[scraper] No more reviews available from Google Play.")
            break

        all_reviews.extend(result)
        print(f"[scraper]  -> fetched {len(all_reviews)} / {count} records")

        if continuation_token is None:
            break

        time.sleep(0.5)  # Polite pacing between batches

    if not all_reviews:
        print("[ERROR] No reviews were scraped. Check network connectivity or app ID.")
        sys.exit(1)

    df = pd.DataFrame(all_reviews)

    # ── Normalise columns ────────────────────────────────────────────────────
    keep = {
        "reviewId":        "review_id",
        "userName":        "user_name",
        "score":           "rating",
        "at":              "review_date",
        "content":         "review_text",
        "appVersion":      "app_version",
        "thumbsUpCount":   "thumbs_up",
        "replyContent":    "dev_reply",
    }
    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)

    # Guarantee schema columns
    for col in ["app_version", "dev_reply", "user_name"]:
        if col not in df.columns:
            df[col] = None

    df["app_name"] = app_name.capitalize()
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    df["review_text"] = df["review_text"].apply(sanitize_text)
    df["app_version"] = df["app_version"].fillna("Unknown").astype(str).str.strip()
    df["app_version"] = df["app_version"].replace({"": "Unknown", "None": "Unknown", "nan": "Unknown"})

    # Filter out empty or whitespace-only review text
    df = df[df["review_text"] != ""].drop_duplicates(subset=["review_id"]).reset_index(drop=True)

    # Write clean UTF-8 CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"[scraper] Ingestion complete. Saved {len(df)} validated reviews -> {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Play review scraper")
    parser.add_argument("--app",   default="spotify", help="App to scrape (spotify|duolingo|netflix|youtube)")
    parser.add_argument("--count", default=1000, type=int, help="Number of reviews to fetch (default: 1000)")
    args = parser.parse_args()

    scrape(args.app, args.count)

