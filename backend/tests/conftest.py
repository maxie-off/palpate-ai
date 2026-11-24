"""Database fixtures. DB tests run only when PALPATE_TEST_DATABASE_URL is set.

The URL must point at a database with db/migrations applied, connecting as the non-superuser
``palpate_app`` role so row-level security is really enforced.
"""

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from app.core.db import connect


@pytest.fixture
def db() -> Iterator[Any]:
    url = os.environ.get("PALPATE_TEST_DATABASE_URL")
    if not url:
        pytest.skip("PALPATE_TEST_DATABASE_URL not set")
    conn = connect(url)
    created: list[str] = []
    conn.new_user = lambda: _new_user(conn, created)
    try:
        yield conn
    finally:
        conn.rollback()
        for user_id in created:
            conn.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        conn.close()


def _new_user(conn: Any, created: list[str]) -> str:
    user_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO users (id, email, school_domain) VALUES (%s, %s, 'test.tw')",
        (user_id, f"{user_id}@test.tw"),
    )
    conn.commit()
    created.append(user_id)
    return user_id
