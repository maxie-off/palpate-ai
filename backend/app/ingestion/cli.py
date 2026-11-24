"""Ingest one PDF from the command line (development helper).

    uv run python -m app.ingestion.cli ../data/private/note.pdf --email me@example.tw --no-embed

Creates the dev user if missing, ingests the file and, unless --no-embed, embeds it through
Bedrock (needs AWS credentials and model access).
"""

import argparse
from pathlib import Path
from typing import get_args

from app.core.config import get_settings, load_domain_config
from app.core.db import connect
from app.ingestion.store import DocType, embed_document, ingest_document


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--email", required=True)
    ap.add_argument("--doc-type", choices=get_args(DocType), default="note")
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    config = load_domain_config(settings.config_path)
    with connect(settings.database_url) as conn:
        row = conn.execute(
            "INSERT INTO users (email, school_domain) VALUES (%s, %s) "
            "ON CONFLICT (email) DO UPDATE SET last_seen_at = now() RETURNING id",
            (args.email, args.email.split("@")[-1]),
        ).fetchone()
        owner_id = str(row[0])
        conn.commit()
        result = ingest_document(
            conn, owner_id=owner_id, path=args.pdf, doc_type=args.doc_type, config=config
        )
        print(result)
        if args.no_embed or result.status == "embedded":
            return
        from app.core.llm import BedrockLLMClient

        llm = BedrockLLMClient(settings.aws_region, settings.embedding_model_id)
        n = embed_document(
            conn,
            llm,
            owner_id=owner_id,
            document_id=result.document_id,
            embed_model=settings.embedding_model_id,
        )
        print(f"embedded {n} sections")


if __name__ == "__main__":
    main()
