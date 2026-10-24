"""Database helper. psycopg is imported lazily so unit tests need no database driver."""

from typing import Any


def connect(database_url: str) -> Any:
    import psycopg

    return psycopg.connect(database_url)
