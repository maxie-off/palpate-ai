.PHONY: install up down lint fmt typecheck test test-db check

install:
	cd backend && uv sync --extra dev

up:
	docker compose up --build

down:
	docker compose down

lint:
	cd backend && uv run ruff check . && uv run ruff format --check .

fmt:
	cd backend && uv run ruff check --fix . && uv run ruff format .

typecheck:
	cd backend && uv run mypy app

test:
	cd backend && uv run pytest

# DB-backed tests: needs `make up` (or any Postgres with db/migrations applied)
test-db:
	cd backend && PALPATE_TEST_DATABASE_URL=postgresql://palpate_app:palpate@localhost:5432/palpate uv run pytest

check: lint typecheck test
