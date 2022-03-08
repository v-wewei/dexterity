SHELL := /bin/bash
N_CPU := $$(python -c "import os; print(os.cpu_count())")

.PHONY: help check format test explore
.DEFAULT: help

help:
	@echo "Usage: make <target>"
	@echo
	@echo "Available targets:"
	@echo "  check: Run type checking and code styling *without* modifying files"
	@echo "  format: Run type checking and code styling inplace"
	@echo "  test: Run all tests"
	@echo "  explore: Run the inhand_manipulation application"

check:
	isort --check .
	black --check .
	flake8 --show-source .
	mypy .

format:
	isort .
	black .
	flake8 .
	mypy .

test:
	pytest --durations=10 -n "${N_CPU}" .

explore:
	python shadow_hand/tasks/inhand_manipulation/explore.py
