"""Extract Bible text from wol.jw.org (New World Translation).

This module does TEXT EXTRACTION ONLY — no embeddings, no vector store.
The goal is to verify the extraction works before spending Jina tokens.

Usage:
    python -u -m assistants.bible.loader                    # test John 3
    python -u -m assistants.bible.loader --book 1 --chapter 1
    python -u -m assistants.bible.loader --full             # all 66 books
    python -u -m assistants.bible.loader --full --delay-min 3 --delay-max 8
"""

import re
import time
import json
import os
import random
import argparse
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup


# ============================================================
# REFERENCE DATA
# ============================================================
BASE_URL = "https://wol.jw.org/en/wol/b/r1/lp-e/nwtsty"

# Canonical order: 1-66. Index 0 is unused so book_number == list index.
BOOKS = [
    None,
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
    "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
    "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah",
    "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
    "Haggai", "Zechariah", "Malachi",
    "Matthew", "Mark", "Luke", "John", "Acts", "Romans",
    "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
    "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John",
    "Jude", "Revelation",
]

# Chapter counts per book (1-indexed, so CHAPTERS[1] = 50 for Genesis)
CHAPTERS = [
    0,
    50, 40, 27, 36, 34, 24, 21, 4, 31, 24,
    22, 25, 29, 36, 10, 13, 10, 42, 150, 31,
    12, 8, 66, 52, 5, 48, 12, 14, 3, 9,
    1, 4, 7, 3, 3, 3, 2, 14, 4,
    28, 16, 24, 21, 28, 16, 16, 13, 6, 6,
    4, 4, 5, 3, 6, 4, 3, 1, 13,
    5, 5, 3, 5, 1, 1, 1, 22,
]


# ============================================================
# POLITE RATE LIMITING
# ============================================================
# Jittered delay between requests — looks less like a scraper
DELAY_MIN        = 2.0   # seconds
DELAY_MAX        = 5.0   # seconds
LONG_PAUSE_EVERY = 75    # chapters
LONG_PAUSE_MIN   = 20.0  # seconds
LONG_PAUSE_MAX   = 45.0  # seconds


def _polite_sleep(chapter_index: int) -> None:
    """Sleep a random amount. Occasionally take a longer break."""
    if chapter_index > 0 and chapter_index % LONG_PAUSE_EVERY == 0:
        pause = random.uniform(LONG_PAUSE_MIN, LONG_PAUSE_MAX)
        print(f"   ☕ taking a {pause:.0f}s break after {chapter_index} chapters")
        time.sleep(pause)
    else:
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


# ============================================================
# HTML PARSING
# ============================================================
# Two verse-ID formats exist in the wild:
#   WOL:    v40-24-14-1  (book-chapter-verse-part)
#   jw.org: v40024014    (book 2-digit + chapter 2-digit + verse 3-digit)
_VERSE_ID_WOL = re.compile(r"^v(\d+)-(\d+)-(\d+)(?:-(\d+))?$")
_VERSE_ID_JW  = re.compile(r"^v(\d{2})(\d{2})(\d{3})$")


def _parse_verse_id(raw_id: str):
    """Return (book, chapter, verse, part) or None if not a verse element."""
    m = _VERSE_ID_WOL.match(raw_id)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4) or 1)

    m = _VERSE_ID_JW.match(raw_id)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), 1

    return None


def _clean_verse_text(text: str) -> str:
    """Remove footnote markers, study-note refs, and redundant whitespace."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[*†‡]+", "", text)
    text = re.sub(r"^\s*\d+\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_chapter_html(html: str) -> List[Dict]:
    """
    Parse a wol.jw.org chapter page into a list of verse dicts.

    Returns:
        [{"verse": 1, "text": "In the beginning..."}, ...]
    """
    soup = BeautifulSoup(html, "html.parser")

    verse_spans = []
    for span in soup.find_all("span", id=True):
        parsed = _parse_verse_id(span["id"])
        if parsed:
            book, chapter, verse, part = parsed
            verse_spans.append((book, chapter, verse, part, span))

    if not verse_spans:
        return []

    merged: Dict[tuple, str] = {}
    for book, chapter, verse, part, span in verse_spans:
        # Remove footnotes / study-note anchors before getting text
        for unwanted in span.find_all(["sup", "a"], class_=True):
            unwanted.decompose()
        chunk = _clean_verse_text(span.get_text(" ", strip=True))
        if not chunk:
            continue
        key = (book, chapter, verse)
        merged[key] = (merged.get(key, "") + " " + chunk).strip()

    return [
        {"verse": v, "text": txt}
        for (_, _, v), txt in sorted(merged.items(), key=lambda kv: kv[0][2])
    ]


# ============================================================
# FETCH
# ============================================================
def fetch_chapter(book_num: int, chapter: int, retries: int = 4) -> Optional[str]:
    """Fetch a single chapter. Returns raw HTML or None on failure."""
    url = f"{BASE_URL}/{book_num}/{chapter}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://wol.jw.org/",
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=20)

            # Rate-limited? Back off much harder.
            if resp.status_code == 429:
                wait = 30 * attempt + random.uniform(0, 15)
                print(f"  ⏸ 429 rate limit — sleeping {wait:.0f}s "
                      f"(attempt {attempt}/{retries})")
                time.sleep(wait)
                continue

            # 403/503 usually means "you're being too aggressive"
            if resp.status_code in (403, 503):
                wait = 60 * attempt
                print(f"  ⏸ HTTP {resp.status_code} — sleeping {wait}s "
                      f"(attempt {attempt}/{retries})")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.text

        except Exception as e:
            print(f"  ⚠️ {url} attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt + random.uniform(0, 2))

    return None


def extract_chapter(book_num: int, chapter: int) -> Optional[Dict]:
    """Fetch + parse a single chapter. Returns structured dict or None."""
    html = fetch_chapter(book_num, chapter)
    if not html:
        return None

    verses = parse_chapter_html(html)
    if not verses:
        return None

    full_text = "\n".join(f"{v['verse']}. {v['text']}" for v in verses)

    return {
        "book_number": book_num,
        "book_name":   BOOKS[book_num],
        "chapter":     chapter,
        "verses":      verses,
        "full_text":   full_text,
        "url":         f"{BASE_URL}/{book_num}/{chapter}",
        "verse_count": len(verses),
    }


# ============================================================
# TEST / SCRAPE ALL
# ============================================================
def _save_json(data, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(data)} chapters to {path}")


def test_single_chapter(book_num: int = 43, chapter: int = 3) -> None:
    """Extract one chapter and print a preview. No JSON written."""
    print("=" * 68)
    print(f"🧪 TEST: {BOOKS[book_num]} {chapter}")
    print("=" * 68)

    result = extract_chapter(book_num, chapter)
    if not result:
        print("❌ Extraction returned no verses.")
        return

    print(f"   URL          : {result['url']}")
    print(f"   Book         : {result['book_name']} ({result['book_number']})")
    print(f"   Chapter      : {result['chapter']}")
    print(f"   Verse count  : {result['verse_count']}")
    print(f"   Total chars  : {len(result['full_text']):,}")
    print(f"   Total words  : {len(result['full_text'].split()):,}")
    print(f"   Est. tokens  : {len(result['full_text']) // 4:,}")
    print()
    print("   --- First 5 verses ---")
    for v in result["verses"][:5]:
        preview = v["text"][:120] + ("..." if len(v["text"]) > 120 else "")
        print(f"   {v['verse']}. {preview}")
    print()
    print("   --- Last 2 verses ---")
    for v in result["verses"][-2:]:
        preview = v["text"][:120] + ("..." if len(v["text"]) > 120 else "")
        print(f"   {v['verse']}. {preview}")


def scrape_all_books(output_path: str = "data/bible/bible_nwt.json",
                     delay_min: float = DELAY_MIN,
                     delay_max: float = DELAY_MAX) -> None:
    """Scrape every chapter of every book. Takes ~2-3 hours. No embeddings."""
    global DELAY_MIN, DELAY_MAX
    DELAY_MIN, DELAY_MAX = delay_min, delay_max

    print("=" * 68)
    print("📚 Full Bible extraction")
    print(f"   delay: {DELAY_MIN:.1f}-{DELAY_MAX:.1f}s "
          f"+ long pause every {LONG_PAUSE_EVERY} chapters")
    print("=" * 68)

    all_chapters = []
    total_planned = sum(CHAPTERS[1:])
    done = 0
    t0 = time.time()

    for book_num in range(1, 67):
        n_chapters = CHAPTERS[book_num]
        print(f"\n📖 {BOOKS[book_num]} ({n_chapters} chapters)")

        for ch in range(1, n_chapters + 1):
            result = extract_chapter(book_num, ch)
            if result:
                all_chapters.append(result)
            done += 1

            if done % 25 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (total_planned - done) / rate if rate else 0
                print(f"   [{done}/{total_planned}] "
                      f"{rate:.2f} ch/s, ETA {eta/60:.1f} min")

            # Jittered pause; occasionally a longer break
            _polite_sleep(done)

    _save_json(all_chapters, output_path)

    total_chars = sum(len(c["full_text"]) for c in all_chapters)
    total_verses = sum(c["verse_count"] for c in all_chapters)
    print()
    print("=" * 68)
    print("📊 Extraction summary")
    print("=" * 68)
    print(f"   Chapters     : {len(all_chapters):,}")
    print(f"   Verses       : {total_verses:,}")
    print(f"   Characters   : {total_chars:,}")
    print(f"   Est. tokens  : {total_chars // 4:,}")
    print(f"   Wall time    : {(time.time()-t0)/60:.1f} min")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Bible text from wol.jw.org")
    parser.add_argument("--full", action="store_true",
                        help="Scrape all 66 books (takes ~2-3 hours)")
    parser.add_argument("--book", type=int, default=43,
                        help="Book number for test mode (default: 43 = John)")
    parser.add_argument("--chapter", type=int, default=3,
                        help="Chapter number for test mode (default: 3)")
    parser.add_argument("--out", default="data/bible/bible_nwt.json",
                        help="Output JSON path for --full mode")
    parser.add_argument("--delay-min", type=float, default=DELAY_MIN,
                        help=f"Min seconds between requests (default: {DELAY_MIN})")
    parser.add_argument("--delay-max", type=float, default=DELAY_MAX,
                        help=f"Max seconds between requests (default: {DELAY_MAX})")
    args = parser.parse_args()

    if args.full:
        if args.delay_min >= args.delay_max:
            print(f"❌ --delay-min ({args.delay_min}) must be < --delay-max ({args.delay_max})")
            raise SystemExit(2)
        scrape_all_books(
            output_path=args.out,
            delay_min=args.delay_min,
            delay_max=args.delay_max,
        )
    else:
        test_single_chapter(args.book, args.chapter)