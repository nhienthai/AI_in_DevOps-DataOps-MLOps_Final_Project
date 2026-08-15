.PHONY: install lint format typecheck test up down smoke

install:
	pip install -e . && pip install -r requirements-dev.txt

lint:
	flake8 src tests scripts
	black --check --diff src tests scripts
	isort --check-only --diff src tests scripts

format:
	black src tests scripts
	isort src tests scripts

typecheck:
	mypy src scripts

test:
	pytest -m "not slow and not integration" --cov=sentiment --cov-report=term-missing --cov-fail-under=80

up:
	docker compose up -d

down:
	docker compose down -v

smoke:
	pytest tests/integration -m integration -v --no-cov
