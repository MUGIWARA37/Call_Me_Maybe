import os
os.environ["HF_HOME"] = "/home/rhlou/goinfre/huggingface"

import sys
sys.path.insert(0, "/home/rhlou/goinfre/torch-packages")
sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe/llm_sdk")



from pydantic import BaseModel
import json
from llm_sdk import Small_LLM_Model
from .vocabulary import Vocabulary
from .models import FunctionDefinition
from .prompt_builder import PromptBuilder, Prompt
from typing import Any
import numpy as np


class Decoder(BaseModel):
    model: Small_LLM_Model
    vocabulary: Vocabulary

    model_config = {"arbitrary_types_allowed": True}

    def _mask_logits(self, logits: list, valid_ids: set) -> list:
        """Zero out all token logits except the ones in valid_ids."""
        logits = np.array(logits)
        mask = np.full(len(logits), float('-inf'))
        mask[list(valid_ids)] = logits[list(valid_ids)]
        return mask.tolist()

    def _apply_repetition_penalty(
        self, logits: list, generated_tokens: list[int], penalty: float = 1.5
    ) -> list:
        from collections import Counter
        logits = np.array(logits, dtype=float)
        counts = Counter(generated_tokens)
        for token_id, count in counts.items():
            factor = penalty ** count
            if logits[token_id] > 0:
                logits[token_id] /= factor
            else:
                logits[token_id] *= factor
        return logits.tolist()

    def _has_repeating_pattern(
        self, tokens: list[int], max_pattern_len: int = 10
    ) -> bool:
        for pattern_len in range(3, max_pattern_len + 1):
            if len(tokens) >= pattern_len * 2:
                if tokens[-pattern_len:] == tokens[-pattern_len * 2:-pattern_len]:
                    return True
        return False

    # ── Type-specific decoders ─────────────────────────────────────────────────

    def _decode_integer_param(
        self, input_ids: list[int], param_name: str, sep_tokens: list[int]
    ) -> None:
        """Constrained decode of an integer; modifies input_ids in-place."""
        numeric_ids = [
            v for k, v in self.vocabulary.token_to_id.items()
            if k and all(ch in "0123456789-" for ch in k)
        ]
        for _ in range(20):
            logits = self.model.get_logits_from_input_ids(input_ids)
            logits = self._mask_logits(logits, set(numeric_ids + [sep_tokens[0]]))
            token_id = logits.index(max(logits))
            input_ids.append(token_id)
            print(f"  [{param_name}] int: {repr(self.model.decode([token_id]))}", flush=True)  # debug
            if token_id == sep_tokens[0]:
                input_ids.extend(sep_tokens[1:])
                return
        input_ids.extend(sep_tokens)  # safety: force separator after max tokens

    def _decode_number_param(
        self, input_ids: list[int], param_name: str, sep_tokens: list[int]
    ) -> None:
        """Constrained decode of a float; modifies input_ids in-place."""
        numeric_ids = [
            v for k, v in self.vocabulary.token_to_id.items()
            if k and all(ch in "0123456789.-" for ch in k)
        ]
        dot_id = self.vocabulary.token_to_id.get(".")
        seen_dot = False
        for _ in range(20):
            logits = self.model.get_logits_from_input_ids(input_ids)
            allowed = set(numeric_ids) | {sep_tokens[0]}  # separator always allowed
            logits = self._mask_logits(logits, allowed)
            token_id = logits.index(max(logits))
            input_ids.append(token_id)
            print(f"  [{param_name}] num: {repr(self.model.decode([token_id]))}", flush=True)  # debug
            if dot_id is not None and token_id == dot_id:
                seen_dot = True
            if token_id == sep_tokens[0]:
                if not seen_dot:
                    input_ids.pop()
                    input_ids.extend(self.model.encode(".0")[0].numpy().tolist())
                    input_ids.extend(sep_tokens)
                else:
                    input_ids.extend(sep_tokens[1:])
                return
        # safety: max tokens reached
        if not seen_dot:
            input_ids.extend(self.model.encode(".0")[0].numpy().tolist())
        input_ids.extend(sep_tokens)

    def _decode_boolean_param(
        self, input_ids: list[int], param_name: str, sep_tokens: list[int]
    ) -> None:
        """Constrained decode of a boolean; modifies input_ids in-place."""
        true_tokens = self.model.encode("true")[0].numpy().tolist()
        false_tokens = self.model.encode("false")[0].numpy().tolist()
        logits = self.model.get_logits_from_input_ids(input_ids)
        logits = self._mask_logits(logits, {true_tokens[0], false_tokens[0]})
        token_id = logits.index(max(logits))
        input_ids.append(token_id)
        chosen = true_tokens if token_id == true_tokens[0] else false_tokens
        input_ids.extend(chosen[1:])
        input_ids.extend(sep_tokens)

    def _decode_string_value(
        self, input_ids: list[int], param_name: str, use_repetition_penalty: bool = True
    ) -> str:
        """Decode a string; appends tokens to input_ids and returns the decoded string.

        Args:
            use_repetition_penalty: Set False for per-parameter prompting where
                the focused prompt already prevents looping, and penalising
                repeated tokens would incorrectly truncate strings like '233'.
        """
        quote_id = self.vocabulary.token_to_id['"']
        input_ids.append(quote_id)  # force opening quote

        string_ids = [v for k, v in self.vocabulary.token_to_id.items() if k and '"' not in k]
        quote_starting_ids = [v for k, v in self.vocabulary.token_to_id.items() if k and k.startswith('"')]
        allowed_ids = set(string_ids + quote_starting_ids)

        generated_tokens: list[int] = []
        collected: list[int] = []

        while len(generated_tokens) < 100:
            logits = self.model.get_logits_from_input_ids(input_ids)
            if use_repetition_penalty:
                logits = self._apply_repetition_penalty(logits, generated_tokens)
            
            logits = self._mask_logits(logits, allowed_ids)
            token_id = logits.index(max(logits))
            
            token_str = self.model.decode([token_id])
            if token_str.startswith('"'):
                input_ids.append(quote_id)
                break

            input_ids.append(token_id)
            print(f"  [{param_name}] str: {repr(token_str)}", flush=True)  # debug

            collected.append(token_id)
            generated_tokens.append(token_id)

            if self._has_repeating_pattern(generated_tokens):
                input_ids.append(quote_id)
                break
        else:
            input_ids.append(quote_id)

        raw = self.model.decode(collected)
        # The model generates JSON-context strings, so backslashes are doubled.
        # Unescape one layer so \\d+ becomes \d+ (correct Python regex).
        try:
            return json.loads(f'"{raw}"')
        except json.JSONDecodeError:
            return raw

    # ── Generation strategies ──────────────────────────────────────────────────

    def _generate_shared(self, prompt: str, function: FunctionDefinition) -> dict[str, Any]:
        """One shared prompt; builds the full JSON object token by token.

        Used when there is at most one string parameter, so cross-parameter
        attention pollution is not a concern.
        """
        builder = PromptBuilder(functions=[function])
        built_prompt = builder.build_parameters(Prompt(prompt=prompt), function)
        
        # Encode the prompt and the opening brace together to avoid tokenization mismatches
        # at the boundary (e.g. \n{ merging into a single token).
        input_ids = self.model.encode(built_prompt + "{\n")[0].numpy().tolist()

        for param_name, param_spec in function.parameters.items():
            param_name_ids = self.model.encode(f'"{param_name}": ')[0].numpy().tolist()
            input_ids.extend(param_name_ids)

            is_last = list(function.parameters.keys())[-1] == param_name
            sep_tokens = (
                self.model.encode('\n}')[0].numpy().tolist() if is_last
                else self.model.encode(',\n')[0].numpy().tolist()
            )

            if param_spec.type == "integer":
                self._decode_integer_param(input_ids, param_name, sep_tokens)
            elif param_spec.type == "number":
                self._decode_number_param(input_ids, param_name, sep_tokens)
            elif param_spec.type == "boolean":
                self._decode_boolean_param(input_ids, param_name, sep_tokens)
            elif param_spec.type == "string":
                self._decode_string_value(input_ids, param_name, use_repetition_penalty=False)
                input_ids.extend(sep_tokens)

        full_generated = self.model.decode(input_ids)
        # Extract everything after the original prompt
        generated = full_generated[len(built_prompt):]
        if not generated.startswith("{"):
            generated = "{" + generated
            
        print(f"generated: {generated}")
        result, _ = json.JSONDecoder().raw_decode(generated)

        for param_name, param_spec in function.parameters.items():
            if param_spec.type == "number" and param_name in result:
                result[param_name] = float(result[param_name])

        return result

    def generate(self, prompt: str, function: FunctionDefinition) -> dict[str, Any]:
        """Extract parameter values from a natural-language prompt.

        Uses the shared strategy where the model decodes the entire JSON 
        object in one pass, guided by few-shot examples.
        """
        return self._generate_shared(prompt, function)



# if __name__ == "__main__":
#     import os
#     os.environ["HF_HOME"] = "/home/rhlou/goinfre/huggingface"
#     import sys
#     sys.path.insert(0, "/home/rhlou/goinfre/torch-packages")
#     sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe/llm_sdk")
#     sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe")

#     from llm_sdk import Small_LLM_Model
#     from src.models import FunctionDefinition, ParameterSpec, ReturnSpec
#     from src.vocabulary import Vocabulary
#     from src.decoder import Decoder

#     model = Small_LLM_Model()
#     vocabulary = Vocabulary.from_model(model)
#     decoder = Decoder(model=model, vocabulary=vocabulary)

#     functions = [
#         FunctionDefinition(
#             name="fn_add_numbers",
#             description="Add two numbers together and return their sum.",
#             parameters={
#                 "a": ParameterSpec(type="number"),
#                 "b": ParameterSpec(type="number")
#             },
#             returns=ReturnSpec(type="number")
#         ),
#         FunctionDefinition(
#             name="fn_greet",
#             description="Generate a greeting message for a person by name.",
#             parameters={
#                 "name": ParameterSpec(type="string")
#             },
#             returns=ReturnSpec(type="string")
#         ),
#     ]

#     tests = [
#         ("What is the sum of -2885.55 and ---3214.11?", functions[0]),
#         ("Greet John", functions[1]),
#     ]

#     for prompt, function in tests:
#         result = decoder.generate(prompt, function)
#         print(f"prompt: {prompt}")
#         print(f"result: {result}")
#         print()