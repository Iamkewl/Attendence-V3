"""Worker integration package exposing Celery app utilities."""

from app.worker.celery_app import celery_app, get_celery_app


__all__ = ["celery_app", "get_celery_app"]
