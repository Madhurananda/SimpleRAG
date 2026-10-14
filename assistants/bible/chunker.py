"""Chunk the extracted Bible JSON into metadata-rich Documents.

Run to preview chunks WITHOUT embedding:
    python -u -m assistants.bible.chunker
    python -u -m assistants.bible.chunker --book 43 --chapter 3
    python -u -m assistants.bible.chunker --write     # save chunks to JSON
"""

import json
import os
import argparse
from typing import List, Dict

from langchain_core.documents import Document


# ============================================================
# CONFIG
# ============================================================
INPUT_JSON  = "data/bible/bible_nwt.json"
OUTPUT_JSON = "data/bible/bible_chunks.json"

# Rough token estimation: NWT English averages ~4 chars/token
CHARS_PER_TOKEN  = 4
TARGET_TOKENS    = 400    # target chunk size
OVERLAP_TOKENS   = 40     # carryover between chunks
MIN_TOKENS       = 60     # don't emit a chunk smaller than this


# ============================================================
# CHUNKING
# ============================================================
def _verse_tokens(text: str) -> int:
    """Estimate token count for a verse."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def chunk_chapter(chapter_data: Dict) -> List[Document]:
    """
    Split one chapter's verses into overlapping chunks.

    Each chunk:
      - Targets ~TARGET_TOKENS tokens
      - Overlaps the previous chunk by ~OVERLAP_TOKENS tokens
      - Never splits a verse across chunks
      - Records verse_start / verse_end for citation
    """
    book_name   = chapter_data["book_name"]
    book_number = chapter_data["book_number"]
    chapter     = chapter_data["chapter"]
    verses      = chapter_data["verses"]           # [{"verse": 1, "text": "..."}, ...]

    if not verses:
        return []

    chunks: List[Document] = []
    i = 0
    while i < len(verses):
        # Grow the chunk until we hit TARGET_TOKENS
        chunk_verses = []
        chunk_tokens = 0
        j = i
        while j < len(verses) and chunk_tokens < TARGET_TOKENS:
            v = verses[j]
            v_tokens = _verse_tokens(v["text"])
            # Always include at least one verse, even if it's huge (e.g., Psalm 119:176)
            if chunk_verses and chunk_tokens + v_tokens > TARGET_TOKENS * 1.4:
                break
            chunk_verses.append(v)
            chunk_tokens += v_tokens
            j += 1

        # Skip tiny trailing chunks unless they're the only chunk
        if chunk_tokens < MIN_TOKENS and chunks:
            # Merge into previous chunk instead of creating a stub
            last = chunks[-1]
            for v in chunk_verses:
                last.page_content += f"\n{v['verse']}. {v['text']}"
                last.metadata["verse_end"] = v["verse"]
            i = j
            continue

        verse_start = chunk_verses[0]["verse"]
        verse_end   = chunk_verses[-1]["verse"]

        # Format: "Genesis 1:1-8\n1. In the beginning...\n2. ..."
        body = "\n".join(f"{v['verse']}. {v['text']}" for v in chunk_verses)
        content = f"{book_name} {chapter}:{verse_start}-{verse_end}\n{body}"

        chunks.append(Document(
            page_content=content,
            metadata={
                "book_name":   book_name,
                "book_number": book_number,
                "chapter":     chapter,
                "verse_start": verse_start,
                "verse_end":   verse_end,
                "testament":   "OT" if book_number <= 39 else "NT",
                "citation":    f"{book_name} {chapter}:{verse_start}-{verse_end}",
                "token_est":   chunk_tokens,
                "source_type": "bible_nwt",
            },
        ))

        # Advance with overlap: back up ~OVERLAP_TOKENS worth of verses
        if j >= len(verses):
            break

        overlap_tokens = 0
        i = j
        while i > 0 and overlap_tokens < OVERLAP_TOKENS:
            i -= 1
            overlap_tokens += _verse_tokens(verses[i]["text"])
        i = max(i, 1) if j - i < 1 else i   # ensure forward progress

    return chunks


def build_all_chunks(input_json: str = INPUT_JSON) -> List[Document]:
    """Chunk the entire Bible."""
    with open(input_json, encoding="utf-8") as f:
        chapters = json.load(f)

    all_chunks: List[Document] = []
    for ch in chapters:
        all_chunks.extend(chunk_chapter(ch))

    return all_chunks


# ============================================================
# INSPECTION
# ============================================================
def print_chunk_stats(chunks: List[Document]) -> None:
    """Summary of chunks so you can sanity-check before embedding."""
    import statistics

    token_counts = [c.metadata["token_est"] for c in chunks]
    char_counts  = [len(c.page_content) for c in chunks]

    print("=" * 68)
    print("📊 Chunk statistics")
    print("=" * 68)
    print(f"   Total chunks       : {len(chunks):,}")
    print(f"   Avg tokens/chunk   : {statistics.mean(token_counts):.0f}")
    print(f"   Median tokens/chunk: {statistics.median(token_counts):.0f}")
    print(f"   Min / Max tokens   : {min(token_counts)} / {max(token_counts)}")
    print(f"   Total chars        : {sum(char_counts):,}")
    print(f"   Total est. tokens  : {sum(token_counts):,}")
    print()
    print(f"   By testament:")
    ot = [c for c in chunks if c.metadata["testament"] == "OT"]
    nt = [c for c in chunks if c.metadata["testament"] == "NT"]
    print(f"      OT chunks      : {len(ot):,}")
    print(f"      NT chunks      : {len(nt):,}")
    print()
    print(f"   Estimated Jina cost (1 embed pass): "
          f"{sum(token_counts):,} tokens")
    print()


def print_sample_chunks(chunks: List[Document], n: int = 3) -> None:
    """Show a few chunks so you can eyeball the format."""
    print("=" * 68)
    print(f"📄 Sample chunks (first {n})")
    print("=" * 68)
    for i, c in enumerate(chunks[:n], 1):
        m = c.metadata
        print(f"\n[Chunk {i}] citation={m['citation']}  "
              f"tokens≈{m['token_est']}")
        print("-" * 68)
        preview = c.page_content[:400] + ("..." if len(c.page_content) > 400 else "")
        print(preview)
    print()


def save_chunks(chunks: List[Document], path: str = OUTPUT_JSON) -> None:
    """Serialize Document objects to JSON on disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = [
        {"page_content": c.page_content, "metadata": c.metadata}
        for c in chunks
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"✅ Wrote {len(chunks):,} chunks to {path}")
    print(f"   File size: {os.path.getsize(path) / 1024 / 1024:.1f} MB")



# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Chunk Bible JSON into Documents")
    parser.add_argument("--write", action="store_true",
                        help=f"Write chunks to {OUTPUT_JSON}")
    parser.add_argument("--book", type=int,
                        help="Preview a single book (numbered 1-66)")
    parser.add_argument("--chapter", type=int,
                        help="Preview a single chapter (requires --book)")
    args = parser.parse_args()

    # ---- Single-chapter preview ----
    if args.book and args.chapter:
        with open(INPUT_JSON, encoding="utf-8") as f:
            chapters = json.load(f)
        match = [c for c in chapters
                 if c["book_number"] == args.book and c["chapter"] == args.chapter]
        if not match:
            print(f"❌ No data for book {args.book} chapter {args.chapter}")
            return
        chunks = chunk_chapter(match[0])
        print(f"📖 {match[0]['book_name']} {args.chapter} → {len(chunks)} chunks\n")
        print_sample_chunks(chunks, n=len(chunks))
        print_chunk_stats(chunks)
        if args.write:
            save_chunks(chunks, OUTPUT_JSON)
        return

    # ---- Full Bible ----
    print("Building chunks from the full Bible...")
    chunks = build_all_chunks()
    print_chunk_stats(chunks)
    print_sample_chunks(chunks, n=3)

    # Show a mid-Bible sample too
    mid = len(chunks) // 2
    print("=" * 68)
    print(f"📄 Mid-Bible sample (chunk {mid})")
    print("=" * 68)
    m = chunks[mid].metadata
    print(f"citation={m['citation']}  tokens≈{m['token_est']}")
    print("-" * 68)
    print(chunks[mid].page_content[:400])
    print()

    if args.write:
        save_chunks(chunks, OUTPUT_JSON)
    else:
        print(f"(tip: pass --write to save chunks to {OUTPUT_JSON})")


if __name__ == "__main__":
    main()