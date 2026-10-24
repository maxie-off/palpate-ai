"""FastAPI application factory."""

from fastapi import FastAPI

from app import __version__
from app.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="Palpate", version=__version__)
    app.include_router(router)
    return app


app = create_app()
