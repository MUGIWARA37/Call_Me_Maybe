from pydantic import BaseModel
from collections import Counter
import json
from llm_sdk import Small_LLM_Model
from .vocabulary import Vocabulary
from .models import FunctionDefinition
from .prompt_builder import PromptBuilder, Prompt
from typing import Any, cast
import numpy as np
import os
import sys

os.environ["HF_HOME"] = "/home/rhlou/goinfre/huggingface"
sys.path.insert(0, "/home/rhlou/goinfre/torch-packages")
sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe/llm_sdk")


class Decoder(BaseModel):
    """Drives the model token by token to produce a valid JSON object."""

    model: Small_LLM_Model
    vocabulary: Vocabulary

    model_config = {"arbitrary_types_allowed": True}

    # ── Logit helpers ──────────────────────────────────────────────────────

    def _mask_logits(
        self, logits: list[Any], valid_ids: set[int]
    ) -> list[Any]:
        """Set every logit to -inf except the ones in valid_ids.

        This forces the model to only pick tokens we allow.
        """
        arr = np.array(logits)
        mask = np.full(len(arr), float('-inf'))
        mask[list(valid_ids)] = arr[list(valid_ids)]
        return mask.tolist()

    def _apply_repetition_penalty(
        self,
        logits: list[Any],
        generated_tokens: list[int],
        penalty: float = 1.5
    ) -> list[Any]:
        """Lower the score of tokens the model already used, to avoid loops.

        Disabled for strings because exact copying needs no discouragement.
        """
        arr = np.array(logits, dtype=float)
        for token_id, count in Counter(generated_tokens).items():
            factor = penalty ** count
            if arr[token_id] > 0:
                arr[token_id] /= factor
            else:
                arr[token_id] *= factor
        return cast(list[Any], arr.tolist())

    def _has_repeating_pattern(
        self, tokens: list[int], max_pattern_len: int = 10
    ) -> bool:
        """Return True if the last N tokens repeat the N tokens before them.

        Starts at pattern length 3 to avoid false positives on double digits
        like '233' or double letters like 'book'.
        """
        for pattern_len in range(3, max_pattern_len + 1):
            if len(tokens) >= pattern_len * 2:
                tail = tokens[-pattern_len:]
                prev = tokens[-pattern_len * 2:-pattern_len]
                if tail == prev:
                    return True
        return False

    # ── One decoder per parameter type ────────────────────────────────────

    def _decode_integer_param(
        self, input_ids: list[int], sep_tokens: list[int]
    ) -> None:
        """Generate an integer value token by token (modifies input_ids)."""
        # Only digit/sign tokens are allowed, plus the separator to end.
        numeric_ids = {
            v for k, v in self.vocabulary.token_to_id.items()
            if k and all(ch in "0123456789-" for ch in k)
        }
        allowed = numeric_ids | {sep_tokens[0]}

        for _ in range(20):
            logits = self.model.get_logits_from_input_ids(input_ids)
            logits = self._mask_logits(logits, allowed)
            token_id = logits.index(max(logits))
            input_ids.append(token_id)
            if token_id == sep_tokens[0]:
                input_ids.extend(sep_tokens[1:])
                return

        input_ids.extend(sep_tokens)  # safety: force separator if 20 steps

    def _decode_number_param(
        self, input_ids: list[int], sep_tokens: list[int]
    ) -> None:
        """Generate a float value token by token (modifies input_ids).

        If the model never generates a '.', we append '.0' to keep it valid.
        """
        numeric_ids = {
            v for k, v in self.vocabulary.token_to_id.items()
            if k and all(ch in "0123456789.-" for ch in k)
        }
        dot_id = self.vocabulary.token_to_id.get(".")
        allowed = numeric_ids | {sep_tokens[0]}
        seen_dot = False

        for _ in range(20):
            logits = self.model.get_logits_from_input_ids(input_ids)
            logits = self._mask_logits(logits, allowed)
            token_id = logits.index(max(logits))
            input_ids.append(token_id)
            if dot_id is not None and token_id == dot_id:
                seen_dot = True
            if token_id == sep_tokens[0]:
                if not seen_dot:
                    # Replace the separator with ".0" then re-append it.
                    input_ids.pop()
                    input_ids.extend(
                        self.model.encode(".0")[0].numpy().tolist()
                    )
                    input_ids.extend(sep_tokens)
                else:
                    input_ids.extend(sep_tokens[1:])
                return

        if not seen_dot:
            input_ids.extend(self.model.encode(".0")[0].numpy().tolist())
        input_ids.extend(sep_tokens)

    def _decode_boolean_param(
        self, input_ids: list[int], sep_tokens: list[int]
    ) -> None:
        """Generate 'true' or 'false' by forcing the first token of each."""
        true_tokens = self.model.encode("true")[0].numpy().tolist()
        false_tokens = self.model.encode("false")[0].numpy().tolist()
        logits = self.model.get_logits_from_input_ids(input_ids)
        logits = self._mask_logits(
            logits, {true_tokens[0], false_tokens[0]}
        )
        token_id = logits.index(max(logits))
        input_ids.append(token_id)
        chosen = true_tokens if token_id == true_tokens[0] else false_tokens
        input_ids.extend(chosen[1:])   # append the rest of 'true'/'false'
        input_ids.extend(sep_tokens)

    def _decode_string_value(
        self,
        input_ids: list[int],
        use_repetition_penalty: bool = True
    ) -> str:
        """Generate a quoted string value (modifies input_ids in-place).

        - Repetition penalty is off by default for strings because
          exact copying needs no discouragement (e.g. '233', repeated words).
        - Stops when the model picks a token starting with '"' (closing quote),
          or a repeating pattern is detected, or 100 tokens are reached.
        """
        quote_id = self.vocabulary.token_to_id['"']
        input_ids.append(quote_id)  # force opening '"'

        # Allow any token that has no bare '"' inside it,
        # plus tokens that START with '"' (used as the closing quote signal).
        allowed_ids = (
            {v for k, v in self.vocabulary.token_to_id.items()
             if k and '"' not in k}
            | {v for k, v in self.vocabulary.token_to_id.items()
               if k and k.startswith('"')}
        )

        generated_tokens: list[int] = []
        collected: list[int] = []

        while len(generated_tokens) < 100:
            logits = self.model.get_logits_from_input_ids(input_ids)
            if use_repetition_penalty:
                logits = self._apply_repetition_penalty(
                    logits, generated_tokens
                )
            logits = self._mask_logits(logits, allowed_ids)
            token_id = logits.index(max(logits))
            token_str = self.model.decode([token_id])

            if token_str.startswith('"'):
                # Model chose a closing-quote token — end the string here.
                input_ids.append(quote_id)
                break

            input_ids.append(token_id)
            collected.append(token_id)
            generated_tokens.append(token_id)

            if self._has_repeating_pattern(generated_tokens):
                input_ids.append(quote_id)
                break
        else:
            input_ids.append(quote_id)  # 100-token limit reached

        raw = self.model.decode(collected)
        # JSON strings double-escape backslashes; json.loads fixes that.
        try:
            return cast(str, json.loads(f'"{raw}"'))
        except json.JSONDecodeError:
            return raw

    # ── Main generation logic ──────────────────────────────────────────────

    def _generate_shared(
        self, prompt: str, function: FunctionDefinition
    ) -> dict[str, Any]:
        """Generate the full JSON object for all parameters in one pass.

        How it works:
        1. Build the few-shot prompt and encode it together with '{' so the
           tokeniser sees '\\n{' as one merged token (same as in examples).
        2. For each parameter, append the key and call the right type decoder.
        3. Decode the full token sequence back to text, slice out the JSON.
        """
        builder = PromptBuilder(functions=[function])
        built_prompt = builder.build_parameters(
            Prompt(prompt=prompt), function
        )

        # Encode prompt + '{' together so '\n{' merges into one token,
        # matching how it appears in the few-shot examples.
        input_ids = self.model.encode(
            built_prompt + "{\n"
        )[0].numpy().tolist()

        last_param = list(function.parameters.keys())[-1]

        for param_name, param_spec in function.parameters.items():
            key_tokens = self.model.encode(
                f'"{param_name}": '
            )[0].numpy().tolist()
            input_ids.extend(key_tokens)

            is_last = (param_name == last_param)
            sep_tokens = (
                self.model.encode('\n}')[0].numpy().tolist() if is_last
                else self.model.encode(',\n')[0].numpy().tolist()
            )

            if param_spec.type == "integer":
                self._decode_integer_param(input_ids, sep_tokens)
            elif param_spec.type == "number":
                self._decode_number_param(input_ids, sep_tokens)
            elif param_spec.type == "boolean":
                self._decode_boolean_param(input_ids, sep_tokens)
            elif param_spec.type == "string":
                self._decode_string_value(
                    input_ids, use_repetition_penalty=False
                )
                input_ids.extend(sep_tokens)

        full_text = self.model.decode(input_ids)
        generated = full_text[len(built_prompt):]
        if not generated.startswith("{"):
            generated = "{" + generated

        result = cast(
            dict[str, Any], json.JSONDecoder().raw_decode(generated)[0]
        )

        # Ensure number params are Python floats (not int).
        for param_name, param_spec in function.parameters.items():
            if param_spec.type == "number" and param_name in result:
                result[param_name] = float(result[param_name])

        return result

    def generate(
        self, prompt: str, function: FunctionDefinition
    ) -> dict[str, Any]:
        """Entry point — extract all parameter values from the user prompt."""
        return self._generate_shared(prompt, function)
