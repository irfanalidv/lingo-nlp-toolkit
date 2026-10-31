.PHONY: help install install-dev install-full install-gpu test test-cov lint format clean demo examples build build-check publish-test publish install-deps check ci

help:  ## Show this help message
	@echo "Lingo NLP Toolkit - Development Commands"
	@echo "========================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in development mode
	pip install -e .

install-dev:  ## Install with development dependencies
	pip install -e ".[dev]"

install-full:  ## Install with all optional dependencies
	pip install -e ".[full]"

install-gpu:  ## Install with GPU extras
	pip install -e ".[gpu]"

test:  ## Run tests
	python -m pytest tests/ -v

test-cov:  ## Run tests with coverage
	python -m pytest tests/ --cov=lingo --cov-report=html --cov-report=term

lint:  ## Run linting checks
	ruff check .
	ruff format --check .

format:  ## Format code
	ruff check --fix .
	ruff format .

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

demo:  ## Run the demo script
	python demo.py

examples:  ## Run the basic usage examples
	python examples/basic_usage.py

build: clean  ## Build sdist and wheel
	python -m build

build-check: build  ## Build and check package metadata
	twine check dist/*

publish-test: build-check  ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: build-check  ## Publish to PyPI
	twine upload dist/*

install-deps:  ## Download spaCy/NLTK data and common models
	lingo setup

check: lint test  ## Run all checks (lint + test)

ci: check  ## Run CI checks (alias for check)
