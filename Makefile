.PHONY: install test lint fmt typecheck ci clean notebook

## Install package in editable mode with all dev deps
install:
	pip install -e ".[all]"

## Run full test suite with coverage
test:
	pytest tests/ -v --cov=models --cov=backtesting --cov=ml --cov-report=term-missing

## Lint with ruff
lint:
	ruff check .

## Format with ruff
fmt:
	ruff format .

## Type-check with mypy
typecheck:
	mypy models/ backtesting/ ml/

## Full CI pipeline (what GitHub Actions runs)
ci: lint typecheck test

## Remove caches and build artefacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/

## Launch Jupyter for research notebooks
notebook:
	jupyter lab research/
