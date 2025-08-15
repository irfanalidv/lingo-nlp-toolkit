.PHONY: help install install-dev install-full test lint format clean demo examples docs build publish test-publish

help:  ## Show this help message
	@echo "Lingo NLP Toolkit - Development Commands"
	@echo "========================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in development mode
	pip install -e .

install-dev:  ## Install with development dependencies
	pip install -e .[dev]

install-full:  ## Install with all optional dependencies
	pip install -e .[full]

install-gpu:  ## Install with GPU support
	pip install -e .[gpu]

test:  ## Run tests
	python -m pytest tests/ -v

test-cov:  ## Run tests with coverage
	python -m pytest tests/ --cov=lingo --cov-report=html --cov-report=term

lint:  ## Run linting checks
	flake8 lingo/ tests/
	black --check lingo/ tests/
	isort --check-only lingo/ tests/

format:  ## Format code
	black lingo/ tests/
	isort lingo/ tests/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

demo:  ## Run the demo script
	python demo.py

examples:  ## Run the basic usage examples
	python examples/basic_usage.py

docs:  ## Build documentation
	cd docs && make html

build: clean  ## Build package for distribution
	python setup.py sdist bdist_wheel
	python -m build

build-check: build  ## Build and check package
	twine check dist/*

publish-test: build-check  ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: build-check  ## Publish to PyPI
	twine upload dist/*

test-publish: publish-test  ## Test publish to TestPyPI first

release: clean build publish  ## Full release process

# Removed duplicate build command

publish:  ## Publish to PyPI (requires authentication)
	twine upload dist/*

install-deps:  ## Install required system dependencies
	# Use Lingo's built-in setup command
	lingo setup

# Documentation
docs-serve: docs  ## Build and serve documentation
	cd docs/_build/html && python -m http.server 8000

# Testing variations
test-unit:  ## Run unit tests only
	pytest tests/ -v -m "not integration and not slow"

test-integration:  ## Run integration tests only
	pytest tests/ -v -m "integration"

test-slow:  ## Run slow tests only
	pytest tests/ -v -m "slow"

# Performance testing
benchmark:  ## Run performance benchmarks
	python examples/enterprise_nlp.py

# Package verification
verify: build-check  ## Verify package integrity
	twine check dist/*
	python -m pip install dist/*.whl --force-reinstall
	python -c "import lingo; print('Package verification successful!')"

setup-dev: install-dev install-deps  ## Setup development environment
	@echo "Development environment setup complete!"
	@echo "You can now run: make test, make demo, or make examples"

check: lint test  ## Run all checks (lint + test)

pre-commit:  ## Install pre-commit hooks
	pre-commit install

ci: check  ## Run CI checks (alias for check)

.PHONY: requirements
requirements:  ## Generate requirements files
	pip-compile setup.py --output-file requirements.txt
	pip-compile setup.py --extra=dev --output-file requirements-dev.txt
	pip-compile setup.py --extra=full --output-file requirements-full.txt
	pip-compile setup.py --extra=gpu --output-file requirements-gpu.txt
