# ──────────────────────────────────────────────────────────────────────────────
#  Call Me Maybe — Makefile
# ──────────────────────────────────────────────────────────────────────────────

FUNCTIONS := data/input/functions_definition.json
INPUT     := data/input/function_calling_tests.json
OUTPUT    := data/output

TORCH_TARGET := /home/rhlou/goinfre/torch-packages

.PHONY: all run install lint type-check check clean test help

# ── Default ───────────────────────────────────────────────────────────────────
all: run

# ── Run the full pipeline ─────────────────────────────────────────────────────
run:
	uv run python -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

# ── Install torch + transformers into goinfre (once per session) ──────────────
install:
	mkdir -p $(TORCH_TARGET)
	pip install torch \
		--index-url https://download.pytorch.org/whl/cpu \
		--target $(TORCH_TARGET)
	pip install transformers huggingface_hub \
		--target $(TORCH_TARGET)

# ── Run decoder tests only ────────────────────────────────────────────────────
test:
	uv run python test.py

# ── Linting ───────────────────────────────────────────────────────────────────
lint:
	uv run flake8 src/ \
		--max-line-length 99 \
		--extend-ignore E203,W503

# ── Type checking ─────────────────────────────────────────────────────────────
type-check:
	uv run mypy src/ \
		--ignore-missing-imports \
		--strict

# ── Run both checks ───────────────────────────────────────────────────────────
check: lint type-check

# ── Clean generated output ────────────────────────────────────────────────────
clean:
	rm -rf $(OUTPUT)
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  make install      Install torch + transformers into goinfre (once per session)"
	@echo "  make run          Run the full pipeline"
	@echo "  make test         Run decoder tests only"
	@echo "  make lint         flake8 check"
	@echo "  make type-check   mypy check"
	@echo "  make check        lint + type-check"
	@echo "  make clean        Remove output files and __pycache__"
	@echo ""