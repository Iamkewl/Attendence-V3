SHELL := /bin/bash
COMPOSE := docker compose -f docker-compose.dev.yml

.PHONY: demo up down clean migrate seed logs test

## One-command reviewer bootstrap: build, start, migrate, seed, and print URL.
demo:
	@echo "==> Building and starting the demo stack..."
	$(COMPOSE) up -d --build
	@echo "==> Waiting for API to be ready..."
	@bash scripts/wait_for_api.sh
	@echo "==> Running Alembic migrations..."
	$(COMPOSE) exec api alembic upgrade head
	@echo "==> Seeding demo data..."
	$(COMPOSE) exec api python /app/backend/scripts/seed_demo_data.py
	@echo ""
	@echo "============================================================"
	@echo "  Open http://localhost:5173"
	@echo "  Login: admin@demo.local / DemoAdmin1!"
	@echo "  API docs: http://localhost:8000/docs"
	@echo "============================================================"

## Start all services in detached mode (build if needed).
up:
	$(COMPOSE) up -d --build

## Stop all services.
down:
	$(COMPOSE) down

## Stop services and remove the persistent Postgres volume.
clean:
	$(COMPOSE) down -v

## Run Alembic migrations inside the running api container.
migrate:
	$(COMPOSE) exec api alembic upgrade head

## Run the demo seed script inside the running api container.
seed:
	$(COMPOSE) exec api python /app/backend/scripts/seed_demo_data.py

## Tail logs for all services (Ctrl-C to stop).
logs:
	$(COMPOSE) logs -f

## Run the pytest suite inside the api container.
test:
	$(COMPOSE) exec api pytest -v --tb=short -m "not slow"
