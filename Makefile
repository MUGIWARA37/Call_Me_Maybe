FUNCTIONS := data/input/functions_definition.json
INPUT     := data/input/function_calling_tests.json
OUTPUT    := data/output/function_calling_results.json

TORCH_TARGET := /home/rhlou/goinfre/torch-packages

.PHONY: all install run debug lint lint-strict clean help

all: run

install:
	uv sync
	mkdir -p $(TORCH_TARGET)
	pip install torch \
		--index-url https://download.pytorch.org/whl/cpu \
		--target $(TORCH_TARGET)
	pip install transformers huggingface_hub \
		--target $(TORCH_TARGET)

run:
	uv run python -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug:
	uv run python -m pdb -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

lint:
	uv run flake8 src
	uv run mypy src

lint-strict:
	uv run flake8 src
	uv run mypy src --strict

clean:
	find . -type d -name __pycache__  -exec rm -rf {} +
	find . -type d -name .mypy_cache  -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo ""
	@echo "  make install      Install project dependencies"
	@echo "  make run          Run the full pipeline"
	@echo "  make debug        Run the pipeline under pdb (Python debugger)"
	@echo "  make lint         flake8 + mypy (required flags from subject)"
	@echo "  make lint-strict  flake8 + mypy --strict"
	@echo "  make clean        Remove __pycache__, .mypy_cache, and .pyc files"
	@echo ""