# ──────────────────────────────────────────────────────────────────────────────
#  Call Me Maybe — Makefile
# ──────────────────────────────────────────────────────────────────────────────

FUNCTIONS := data/input/functions_definition.json
INPUT     := data/input/function_calling_tests.json
OUTPUT    := data/output/function_calling_results.json

TORCH_TARGET := /home/rhlou/goinfre/torch-packages

.PHONY: all install run debug lint lint-strict clean help

# ── Default ───────────────────────────────────────────────────────────────────
all: run

# ── Install dependencies ──────────────────────────────────────────────────────
install:
	uv sync
	mkdir -p $(TORCH_TARGET)
	pip install torch \
		--index-url https://download.pytorch.org/whl/cpu \
		--target $(TORCH_TARGET)
	pip install transformers huggingface_hub \
		--target $(TORCH_TARGET)

# ── Run the full pipeline ─────────────────────────────────────────────────────
run:
	uv run python -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

# ── Run in debug mode (pdb) ───────────────────────────────────────────────────
debug:
	uv run python -m pdb -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

# ── Lint: flake8 + mypy with required subject flags ───────────────────────────
lint:
	uv run flake8 .
	uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

# ── Strict lint: flake8 + mypy --strict ──────────────────────────────────────
lint-strict:
	uv run flake8 .
	uv run mypy . --strict

# ── Clean caches (NOT the output file) ───────────────────────────────────────
clean:
	find . -type d -name __pycache__  -exec rm -rf {} +
	find . -type d -name .mypy_cache  -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  make install      Install project dependencies"
	@echo "  make run          Run the full pipeline"
	@echo "  make debug        Run the pipeline under pdb (Python debugger)"
	@echo "  make lint         flake8 + mypy (required flags from subject)"
	@echo "  make lint-strict  flake8 + mypy --strict"
	@echo "  make clean        Remove __pycache__, .mypy_cache, and .pyc files"
	@echo ""