# Call Me Maybe — Architecture Map

> One-page reference: every function, what it does, and why it exists.

---

## How a prompt becomes a function call

```
User prompt (plain text)
        │
        ▼
┌───────────────────┐
│   JsonParser      │  loads functions + prompts from JSON files
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  PromptBuilder    │  turns data into text the LLM can read
│  build_selection  │──────────────────────────────────────────┐
│  build_parameters │                                          │
└───────┬───────────┘                                          │
        │                                                      │
        ▼                                                      ▼
┌───────────────────┐                              ┌───────────────────────┐
│  select_function  │  picks the right function    │  Decoder.generate     │
│  (pipeline.py)    │  token by token              │  (decoder.py)         │
└───────────────────┘                              └───────────────────────┘
                                                            │
                                    ┌───────────────────────┼───────────────────────┐
                                    ▼                       ▼                       ▼
                        ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
                        │ _decode_integer  │   │ _decode_number   │   │ _decode_string   │
                        │ _decode_boolean  │   │                  │   │ _decode_value    │
                        └──────────────────┘   └──────────────────┘   └──────────────────┘
                                    │                       │                       │
                                    └───────────────────────┴───────────────────────┘
                                                            │
                                                            ▼
                                                ┌───────────────────┐
                                                │  _mask_logits     │  used by every decoder
                                                └───────────────────┘
```

---

## File by file

### `models.py` — data shapes

| Class | What it holds | Why it exists |
|---|---|---|
| `ParameterSpec` | one parameter: just its `type` | validates that the type is a known one |
| `ReturnSpec` | the return type of a function | same validation as above |
| `FunctionDefinition` | full function: name, description, parameters, return | central schema shared by all files |
| `Prompt` | the raw user text | wraps the string so Pydantic can validate it |

---

### `vocabulary.py` — token ↔ ID maps

| Function | What it does | Why it exists |
|---|---|---|
| `Vocabulary.from_model` | reads the tokenizer JSON file and builds two dicts: `token→id` and `id→token` | the decoder needs to find the IDs of `"`, `.`, digits, etc. to build its allowed-token sets |

---

### `jsonparser.py` — file loading

| Function | What it does | Why it exists |
|---|---|---|
| `read_json_file` | opens the file and parses JSON | single place for all file/JSON errors |
| `load_functions` | loads function definitions | prepares the candidate list for the selector |
| `load_prompts` | loads user prompts | validates every prompt before it enters the pipeline |

---

### `prompt_builder.py` — text prompt construction

| Function | What it does | Why it exists |
|---|---|---|
| `build_selection` | lists all functions and asks the model to name one | the selector needs a text prompt before it can constrain tokens |
| `build_parameters` | builds the few-shot extraction prompt | the decoder needs examples in the exact same format it will produce, so its tokenisation matches |

> **Why few-shot examples?** A 0.5B model needs to see examples of the exact output format. Without them it may evaluate math instead of copying values.

---

### `pipeline.py` — function selection

| Function | What it does | Why it exists |
|---|---|---|
| `select_function` | picks the right function using token-level beam search | guarantees the output is always a valid function name (no hallucination) |

**Step by step inside `select_function`:**

```
1. Tokenise every function name → list of token IDs per candidate
2. At step i, ask the model for logits
3. Mask: only keep tokens that appear at position i in a surviving candidate
4. Pick the best token → discard any candidate that doesn't match
5. Repeat until one candidate is left
```

> **Why constrained search instead of free generation?** Free generation could produce a function name that doesn't exist. Constraining to real token sequences makes selection 100% valid.

---

### `decoder.py` — parameter extraction

#### Helpers

| Function | What it does | Why it exists |
|---|---|---|
| `_mask_logits` | sets every disallowed token to `-inf` | forces the model to only generate tokens valid for the current field type |
| `_apply_repetition_penalty` | lowers the score of already-used tokens | prevents the model from looping (disabled for strings — see below) |
| `_has_repeating_pattern` | detects if the last N tokens repeat the N before them | emergency stop for runaway loops; starts at length 3 to avoid false positives on `233` or `book` |

#### Type decoders

| Function | What it generates | Key constraint |
|---|---|---|
| `_decode_integer_param` | an integer | only digit/sign tokens allowed |
| `_decode_number_param` | a float | digit/dot/sign tokens; appends `.0` if no dot seen |
| `_decode_boolean_param` | `true` or `false` | only the first token of each word is offered |
| `_decode_string_value` | a quoted string | any token without a bare `"` inside; stops on token starting with `"` |

> **Why no repetition penalty for strings?**
> Extracting a string means copying it exactly from the prompt.
> The penalty would lower the score of tokens the model already saw, which corrupts repeated words or digits like `"233"`.

#### Main logic

| Function | What it does | Why it exists |
|---|---|---|
| `_generate_shared` | builds the full JSON object in one pass | one prompt for all parameters is faster and lets the model use the few-shot examples |
| `generate` | public entry point | clean API — callers only need one method |

**Step by step inside `_generate_shared`:**

```
1. Build the few-shot prompt
2. Encode prompt + '{\n' together (keeps '\n{' as one merged token)
3. For each parameter:
   a. Append the key  →  "param_name": 
   b. Call the right type decoder
   c. Append separator  →  ,\n  or  \n}  for the last one
4. Decode the full token list back to text
5. Slice out everything after the prompt  →  that's the JSON
6. Parse and return the dict
```

> **Why encode `'\n{'` together?** BPE tokenisers merge `\n{` into one token. If you encode the prompt and the `{` separately you get two tokens instead of one, which breaks the pattern the model learned from the few-shot examples.
