# Call Me Maybe: Constrained Decoding Optimisations

This document outlines the specific changes made to the `decoder.py` and `prompt_builder.py` pipeline to enable accurate parameter extraction using a 0.5B parameter model. Small language models are highly susceptible to tokenization mismatches, local context bias, and constrained-masking traps. The following optimisations were implemented to cure these issues.

## 1. Shared Prompt Strategy Over Per-Parameter Strategy
- **The Problem:** The pipeline was separating string parameters into individual prompts. This discarded the single shared prompt approach (which is much faster) and failed to leverage the complex few-shot examples effectively.
- **The Implementation:** Modified `Decoder.generate` to universally use `_generate_shared`, completely removing the `_generate_per_param` and `build_single_parameter` functions.
- **The Why:** This simplifies the codebase and allows the model to extract all parameters in a single pass while utilizing the robust few-shot examples defined in `build_parameters`.

## 2. Disabling Repetition Penalty for Substring Extraction
- **The Problem:** Extracted strings like `"233"` or `"The cat sat on the mat with another cat"` were being heavily distorted (truncated to `23` or hallucinating uppercase words like `CATT`).
- **The Implementation:** Hardcoded `use_repetition_penalty=False` when calling `_decode_string_value`.
- **The Why:** The repetition penalty aggressively alters natural text by artificially lowering the probability of previously generated tokens. Because parameter extraction requires *exact substring copying* from the prompt, applying a repetition penalty is actively harmful and forces the model to hallucinate "creative" bypasses.

## 3. Adjusting N-Gram Loop Prevention (`_has_repeating_pattern`)
- **The Problem:** The safety net designed to prevent infinite loops was checking for patterns of length 1 (a single token repeating). This caused numbers like `233` (two `3` tokens in a row) to trigger the safety net, which immediately and prematurely terminated the string.
- **The Implementation:** Changed the loop to check for patterns starting at length 3: `range(3, max_pattern_len + 1)`.
- **The Why:** This safely allows natural double-letters (`book`) and double-digits (`233`) to be generated without interruption, while still successfully catching genuine runaway model loops (which are typically multi-token sequences, like `\d+\s+`).

## 4. Eliminating Tokenization Mismatches
- **The Problem:** The few-shot examples were formatted as `User request: ...\n{"source_string"...`. In standard BPE tokenizers, the sequence `\n{` is naturally merged into a single token. However, the decoder was encoding the prompt (ending in `\n`) and then manually appending the token for `{`. This mismatch (two tokens vs one merged token) caused the model to completely fail to recognise the few-shot pattern, leading it to evaluate math (e.g., outputting `4.0` instead of `16.0`) instead of extracting values.
- **The Implementation:** 
  - Added `Example:\n` to the final prompt template to perfectly mirror the prefix of the few-shot examples.
  - Changed `decoder.py` to encode `built_prompt + "{\n"` as a single continuous string to allow natural token merging.
  - Sliced the final generated text by the character length of `built_prompt` rather than relying on an exact token count index.
- **The Why:** This guarantees that the final test prompt is tokenized identically to the few-shot examples, ensuring the model snaps cleanly into the expected copying pattern.

## 5. Multiline JSON and Safe String Termination
- **The Problem:** The model hallucinated massive story continuations (e.g., `"The cat sat..., and the dog sat..."`) and looped on regex generations. This occurred because the few-shot examples used `",\n` to end strings, but the constrained decoder's mask explicitly blocked *all* tokens containing `"` (except for the single isolated `"` token). Because the model's preferred token (`",\n`) was blocked, it was forced to pick its second-best guess (like `, ` or `\s+`), leading to run-away text.
- **The Implementation:** 
  - Formatted the few-shot examples in `prompt_builder.py` to use multiline JSON for cleaner parameter separation.
  - Modified `_decode_string_value` to explicitly allow any token in the vocabulary that *starts* with `"` (e.g. `",\n` or `",`). 
  - If the model predicts one of these tokens, the decoder safely breaks the loop and manually appends the `,\n` separator.
- **The Why:** This aligns the constrained mask with the model's natural predictions. The model can now select its preferred end-of-string token without being artificially blocked, completely curing the runaway hallucinations and significantly easing the cognitive load on the 0.5B model.
