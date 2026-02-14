.PHONY: help install clean test run-notebook

help:
	@echo "Available commands:"
	@echo "  install       Install dependencies"
	@echo "  clean         Clean cache files"
	@echo "  test          Run tests"
	@echo "  run-notebook  Start Jupyter notebook"
	@echo "  setup-env     Create virtual environment"

install:
	pip install -r requirements.txt
	pip install -e .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pkl" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

test:
	pytest tests/ -v

run-notebook:
	jupyter notebook notebooks/

setup-env:
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows)"