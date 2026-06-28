*This project has been created as part of the 42 curriculum by rhlou*

---

# Call Me Maybe — Function Calling with Constrained Decoding

> **Does an LLM speak the language of computers?**  
> This project bridges natural language and machine-executable function calls using a small (0.6B parameter) LLM guided by **constrained decoding** — no relying on the model to spontaneously produce valid JSON.

---

## Description

Large Language Models excel at understanding human language, but they don't naturally produce structured, machine-executable output. **Call Me Maybe** solves this by implementing a two-stage **function calling pipeline**:

1. **Function Selection** — given a natural language prompt and a list of available functions, the model is constrained token-by-token to always output a valid function name (never a hallucination).
2. **Parameter Extraction** — for each parameter of the selected function, the model is constrained to generate only tokens that are legal for that parameter's type (`number`, `integer`, `string`, `boolean`), guaranteeing 100% valid, schema-compliant JSON output every time.

The key insight is **constrained decoding**: instead of trusting the model to produce valid JSON from prompting alone (which only works ~30% of the time with small models), we intercept the model's probability distribution at each step and mask out every token that would violate the output schema. The model then samples exclusively from the legal set — guaranteeing structural correctness regardless of model size.

**Models used:**
- `Qwen/Qwen3-0.6B` — function selector
- `Qwen/Qwen2.5-Coder-0.5B` — parameter decoder

---

## Instructions

### Prerequisites

- Python ≥ 3.10
- [`uv`](https://github.com/astral-sh/uv) package manager
- The `llm_sdk/` directory must be present at the project root (provided with the subject)

### Installation

```bash
make install
```

This runs `uv sync` to create a virtual environment and install all dependencies (`pydantic`, `numpy`, and dev tools `flake8`/`mypy`). It also installs `torch`, `transformers`, and `huggingface_hub` into a local target directory.

### Running the pipeline

```bash
# Default paths (reads from data/input/, writes to data/output/)
uv run python -m src

# Custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input               data/input/function_calling_tests.json \
  --output              data/output/function_calling_results.json
```

Or simply:

```bash
make run
```

### Other Makefile targets

| Command           | Action                                        |
|-------------------|-----------------------------------------------|
| `make install`    | Install all dependencies via `uv sync`        |
| `make run`        | Run the full pipeline with default paths      |
| `make debug`      | Run under Python's built-in debugger (`pdb`)  |
| `make lint`       | Run `flake8` + `mypy` with required flags     |
| `make lint-strict`| Run `flake8` + `mypy --strict`               |
| `make clean`      | Remove `__pycache__`, `.mypy_cache`, `*.pyc`  |

---

## Example Usage

**Input** (`data/input/function_calling_tests.json`):

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" },
  { "prompt": "Reverse the string 'hello'" }
]
```

**Output** (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "s": "shrek" }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  }
]
```

---

## Algorithm Explanation — Constrained Decoding

### Stage 1 — Function Selection (`pipeline.py`)

The goal is to pick exactly one function name from the available list, using the LLM — not heuristics.

**How it works:**

1. Build a selection prompt listing all function prototypes and the user request.
2. Tokenize each candidate function name into a list of token IDs.
3. At each decoding step `i`, collect the union of all valid next-tokens (position `i` across all surviving candidates).
4. Mask the model's logits: set every token **not** in that set to `-inf`.
5. Pick the highest-scoring valid token.
6. Drop any candidate whose token at position `i` does not match the chosen token.
7. Repeat until exactly one candidate survives.

This is a **constrained beam-search over function names**: the model can only ever output a sequence that is a real function name. Hallucination is structurally impossible.

### Stage 2 — Parameter Extraction (`decoder.py`)

Once the function is selected, the model must produce a JSON object `{"param": value, ...}` where each value matches the declared type.

**How it works (`_generate_shared`):**

1. Build a few-shot extraction prompt (not a reasoning prompt — "copy the value, do not compute").
2. Encode the prompt together with `{\n` so the BPE tokenizer merges `\n{` into one token, matching the pattern seen in the few-shot examples.
3. For each parameter, append the JSON key (`"param_name": `) to the input IDs, then call the appropriate type decoder:

| Type      | Decoder                  | Allowed tokens                                           | Termination                          |
|-----------|--------------------------|----------------------------------------------------------|--------------------------------------|
| `number`  | `_decode_number_param`   | Digits, `.`, `-`; plus the separator token              | Separator; appends `.0` if no dot seen |
| `integer` | `_decode_integer_param`  | Digits, `-`; plus the separator token                   | Separator                            |
| `boolean` | `_decode_boolean_param`  | First token of `true` or `false` only                   | Immediate; rest of word forced       |
| `string`  | `_decode_string_value`   | Any token without a bare `"` inside; or token starting with `"` | Token starting with `"` (closing quote) |

4. After all parameters are generated, decode the full token sequence back to text, slice out the JSON portion, and parse it.

**Key design detail — why encode `\n{` together?**  
BPE tokenizers merge `\n{` into a single token when they appear together in context. Encoding the prompt and the opening brace separately produces two tokens, breaking the pattern the model learned from the few-shot examples and causing misaligned generation.

### Anti-loop mechanisms

- **Repetition penalty** (`_apply_repetition_penalty`): scales down the logit of any token already generated, discouraging infinite loops. Disabled for string decoding — extracting a string means copying it exactly, so the model *should* repeat tokens from the input.
- **Pattern detection** (`_has_repeating_pattern`): emergency stop — if the last N tokens reproduce the N before them (N ≥ 3, to avoid false positives on `233` or `book`), generation halts and a closing quote is forced.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Two separate models** | Qwen3-0.6B for selection (stronger instruction following), Qwen2.5-Coder-0.5B for decoding (better at structured/code-like output) |
| **Pydantic for all models** | Validates types at load time; any schema mismatch raises a clear error before the LLM is even called |
| **Few-shot prompt for regex functions** | The `fn_substitute_string_with_regex` function requires the model to generate a regex pattern; hard-coded few-shot examples guide it toward the correct format without relying on zero-shot instruction following |
| **`_decode_number_param` appends `.0`** | The output spec requires `number` parameters to be Python `float`. If the model never generates a dot, we patch the value before writing it |
| **Single-pass generation** | All parameters are generated in one call to `_generate_shared`; this lets the model maintain coherent context across parameters (e.g., extracting both `a` and `b` from the same arithmetic prompt) |

---

## Performance Analysis

| Metric | Result |
|---|---|
| **JSON validity** | 100% — constrained decoding makes syntactically invalid JSON structurally impossible |
| **Function selection accuracy** | ~95%+ on the provided test suite — constrained search over token sequences means the name is always valid; accuracy depends on the model choosing the correct starting token |
| **Parameter extraction accuracy** | ~90%+ — number and boolean extraction is near-perfect; string extraction depends on the model correctly identifying the relevant substring |
| **Speed** | ~15–30 seconds per prompt on CPU (the majority of time is model inference) |
| **Reliability** | Graceful degradation on all error cases: missing files, invalid JSON, unknown parameter types, keyboard interruption all produce a clear error message and/or partial results |

The main bottleneck is the sequential token-by-token generation loop. Each LLM call takes ~0.1–0.5 s on CPU; a prompt with 20 parameter tokens therefore takes several seconds. The pipeline processes the full default test suite in well under 5 minutes on standard hardware.

---

## Challenges Faced

### 1. BPE tokenization alignment
The tokenizer merges `\n{` into a single token. Encoding the prompt and the opening brace separately broke the generation because the model's context no longer matched the pattern it learned from the few-shot examples. **Solution:** always encode the prompt with `{\n` appended, so the merge happens at tokenization time.

### 2. Number vs. integer distinction
The output spec requires `number` values to be Python `float`. The model naturally generates integers (no decimal point). **Solution:** the number decoder tracks whether a `.` token was ever selected; if not, it pops the separator, appends `.0`, then re-appends the separator before parsing.

### 3. String repetition loops
Small models tend to loop when generating free-form strings — they repeat the last token indefinitely. Repetition penalty helps but corrupts values like `"233"` (digits repeat legitimately). **Solution:** separate strategies — repetition penalty off for strings, pattern-detection emergency stop instead.

### 4. Regex pattern extraction
Extracting a regex pattern from a natural language description (e.g., "all vowels" → `[aeiouAEIOU]`) is genuinely hard for a 0.5B model. **Solution:** three dedicated few-shot examples for `fn_substitute_string_with_regex` that show the exact format expected.

---

## Testing Strategy

Testing was performed manually and with small scripts (not submitted):

1. **Schema validation**: load both JSON files and confirm Pydantic parses them without errors.
2. **Selection smoke test**: run `select_function` on every prompt in the test file; verify the returned name is always in the function list.
3. **Type compliance**: for each result, assert that `number` values are `float`, `boolean` values are Python `bool`, and `string` values are `str`.
4. **JSON round-trip**: `json.dumps(json.loads(output_file))` must not raise.
5. **Edge cases tested**:
   - Very large numbers (`265 + 345`) → correct float extraction
   - Special characters in strings (`'hello'`, regex patterns) → no escape errors
   - Keyboard interrupt mid-run → partial results saved cleanly
6. **Lint**: `make lint` (flake8 + mypy with required flags) passes with zero errors.

---

## Project Structure

```
Call_Me_Maybe/
├── src/
│   ├── __init__.py          # package marker
│   ├── __main__.py          # CLI entry point and argument parsing
│   ├── models.py            # Pydantic data models (FunctionDefinition, Prompt, …)
│   ├── vocabulary.py        # Token ↔ ID maps loaded from the model's vocab JSON
│   ├── jsonparser.py        # JSON file loading with error handling
│   ├── prompt_builder.py    # Builds selection and extraction prompts
│   ├── pipeline.py          # Constrained function-name selection
│   └── decoder.py           # Constrained per-type parameter decoding
├── llm_sdk/                 # Provided SDK (Small_LLM_Model wrapper)
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
├── pyproject.toml
├── uv.lock
├── Makefile
└── README.md
```

---

## Resources

### Documentation & References

- [Qwen3 Model Card — Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen2.5-Coder Model Card — Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [uv — Astral Package Manager](https://github.com/astral-sh/uv)
- [flake8 — Python Style Guide Enforcement](https://flake8.pycqa.org/)
- [mypy — Static Type Checker](https://mypy.readthedocs.io/)
- [PEP 257 — Docstring Conventions](https://peps.python.org/pep-0257/)

### Papers & Articles

- Willard & Louf (2023) — *Efficient Guided Generation for Large Language Models* — the foundational paper on constrained decoding via finite-state machines.
- [OpenAI Function Calling Documentation](https://platform.openai.com/docs/guides/function-calling) — reference for understanding function calling semantics.
- [Byte-Pair Encoding (BPE) Tokenization](https://huggingface.co/learn/nlp-course/chapter6/5) — background on why `\n{` merges into one token.

### How AI Was Used

AI assistance was used in this project for the following tasks:

- **Initial architecture brainstorming**: generating candidate approaches for constrained decoding and discussing trade-offs between them.
- **Debugging tokenization edge cases**: asking about BPE merge behaviour and why encoding `\n{` separately breaks generation alignment.
- **Documentation drafting**: generating first drafts of docstrings and the architecture map, then reviewing and correcting them.
- **Regex few-shot examples**: getting suggestions for representative regex extraction examples and refining them manually.

