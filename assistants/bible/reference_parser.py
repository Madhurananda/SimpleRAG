"""Parse Bible references out of natural-language queries.

Examples:
    "John 3:16"           -> ("John", 3, 16, 16)
    "Psalm 23"            -> ("Psalms", 23, None, None)
    "1 Corinthians 13"    -> ("1 Corinthians", 13, None, None)
    "Genesis 1:1-5"       -> ("Genesis", 1, 1, 5)
    "What does John 3:16 mean?" -> ("John", 3, 16, 16)
"""

import re
from typing import Optional, Tuple

# Canonical book names — MUST match metadata stored in Qdrant
_BOOKS = [
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

_ALIASES = {
    "psalm":             "Psalms",
    "psalms":            "Psalms",
    "song of songs":     "Song of Solomon",
    "song of solomon":   "Song of Solomon",
    "canticles":         "Song of Solomon",
    "revelations":       "Revelation",
    "apocalypse":        "Revelation",
    "1 cor":             "1 Corinthians",
    "2 cor":             "2 Corinthians",
    "1 corinth":         "1 Corinthians",
    "2 corinth":         "2 Corinthians",
    "1 thess":           "1 Thessalonians",
    "2 thess":           "2 Thessalonians",
    "1 tim":             "1 Timothy",
    "2 tim":             "2 Timothy",
    "1 pet":             "1 Peter",
    "2 pet":             "2 Peter",
    "1 sam":             "1 Samuel",
    "2 sam":             "2 Samuel",
    "1 chr":             "1 Chronicles",
    "2 chr":             "2 Chronicles",
}

# Matches: [optional 1/2/3] Book chapter[:verse[-verse]]
_REF_PATTERN = re.compile(
    r"\b([123]\s+)?"
    r"([A-Za-z][A-Za-z]+(?:\s+(?:of\s+)?[A-Za-z]+)?)"
    r"\s+(\d{1,3})"
    r"(?::(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?)?",
    re.IGNORECASE,
)


def _normalize_book(name: str) -> Optional[str]:
    """Return canonical book name if the input matches (or aliases) one."""
    if not name:
        return None
    key = re.sub(r"\s+", " ", name.strip().lower())
    if key in _ALIASES:
        return _ALIASES[key]
    for book in _BOOKS:
        if book.lower() == key:
            return book
    return None


def parse_reference(query: str) -> Optional[Tuple[str, int, Optional[int], Optional[int]]]:
    """
    Parse the FIRST Bible reference in `query`.

    Returns (book_name, chapter, verse_start, verse_end) or None.
    `verse_start`/`verse_end` are None when only a chapter is given.
    """
    if not query:
        return None

    for m in _REF_PATTERN.finditer(query):
        prefix, book, chapter, v_start, v_end = m.groups()

        book_query = book.strip()
        if prefix:
            book_query = f"{prefix.strip()} {book_query}"

        canonical = _normalize_book(book_query)
        if not canonical:
            continue

        v_s = int(v_start) if v_start else None
        v_e = int(v_end) if v_end else v_s
        return (canonical, int(chapter), v_s, v_e)

    return None


# ----- quick self-test -----
if __name__ == "__main__":
    tests = [
        "John 3:16",
        "What does John 3:16 mean?",
        "Psalm 23",
        "psalms 23:1",
        "1 Corinthians 13",
        "Genesis 1:1-5",
        "the ten commandments in Exodus",
        "Song of Songs 2:1",
        "how should we treat our enemies?",
    ]
    for t in tests:
        print(f"{t!r:45} -> {parse_reference(t)}")