"""Create payload indexes on the bible_nwt collection.

Qdrant Cloud requires an index on any field used in a filter. Without
these, metadata filters return HTTP 400.

Run once:
    python -u scripts/create_bible_payload_indexes.py
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qdrant_client import QdrantClient
from qdrant_client.http import models

from core import config

COLLECTION = "bible_nwt"

# (field_name, schema_type) — for our filters we need these three
INDEXES = [
    ("metadata.book_name",   models.PayloadSchemaType.KEYWORD),
    ("metadata.chapter",     models.PayloadSchemaType.INTEGER),
    ("metadata.verse_start", models.PayloadSchemaType.INTEGER),
    ("metadata.verse_end",   models.PayloadSchemaType.INTEGER),
    # Optional but useful — lets you filter by testament for other queries
    ("metadata.testament",   models.PayloadSchemaType.KEYWORD),
]


def main():
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=60,
    )

    print("=" * 68)
    print(f"📇 Creating payload indexes on '{COLLECTION}'")
    print("=" * 68)

    for field, schema in INDEXES:
        try:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=schema,
                wait=True,
            )
            print(f"   ✅ {field:<28} ({schema})")
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg:
                print(f"   ℹ️  {field:<28} already indexed")
            else:
                print(f"   ❌ {field:<28} failed: {e}")

    # Verify
    print()
    info = client.get_collection(COLLECTION)
    payload_schema = getattr(info, "payload_schema", {}) or {}
    print("=" * 68)
    print("📊 Indexed payload fields")
    print("=" * 68)
    for field, meta in sorted(payload_schema.items()):
        print(f"   {field:<28} {getattr(meta, 'data_type', meta)}")
    print()
    print("   ✅ Done. Reference filtering is now enabled.")


if __name__ == "__main__":
    main()