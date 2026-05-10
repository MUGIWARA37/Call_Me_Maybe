from pydantic import BaseModel, model_validator
from enum import Enum
import numpy as np
import json
import sys
sys.path.insert(0, "/home/rhlou/goinfre/torch-packages")
sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe/llm_sdk")

from llm_sdk import Small_LLM_Model
from vocabulary import Vocabulary
from models import FunctionDefinition


class DecoderState(Enum):
    START        = "start"
    NAME_KEY     = "name_key"
    NAME_VALUE   = "name_value"
    PARAMS_KEY   = "params_key"
    PARAM_NAME   = "param_name"
    PARAM_COLON  = "param_colon"   # ← new
    PARAM_VALUE  = "param_value"
    PARAM_STRING = "param_string"
    PARAM_SEP    = "param_sep"     # ← new
    END          = "end"
    DONE         = "done"


class Decoder(BaseModel):
    model:      Small_LLM_Model
    vocabulary: Vocabulary

    name_key_tokens:    list[int] = []
    params_key_tokens:  list[int] = []
    param_colon_tokens: list[int] = []
    param_sep_tokens:   list[int] = []
    end_tokens:         list[int] = []

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode='after')
    def precompute_tokens(self) -> 'Decoder':
        self.name_key_tokens    = self.model.encode('"name": "').tolist()[0]
        self.params_key_tokens  = self.model.encode('", "parameters": {"').tolist()[0]
        self.param_colon_tokens = self.model.encode('": ').tolist()[0]
        self.param_sep_tokens = self.model.encode('"').tolist()[0]
        self.end_tokens = self.model.encode("}").tolist()[0]
        return self

    def get_valid_tokens(self, state: DecoderState, function_def: FunctionDefinition,
                     current_prm: str = "", state_tokens: int = 0,
                     params_list: list[str] = [], param_index: int = 0,
                     param_name_tokens: dict[str, list[int]] = {},
                     filled_params: int = 0) -> list[int]:

        if state == DecoderState.START:
            return [self.vocabulary.token_to_id["{"]]

        elif state == DecoderState.NAME_KEY:
            idx = min(state_tokens, len(self.name_key_tokens) - 1)
            return [self.name_key_tokens[idx]]

        elif state == DecoderState.NAME_VALUE:
            return self.model.encode(function_def.name).tolist()[0]

        elif state == DecoderState.PARAMS_KEY:
            idx = min(state_tokens, len(self.params_key_tokens) - 1)
            return [self.params_key_tokens[idx]]

        elif state == DecoderState.PARAM_NAME:
            current_param_name = params_list[param_index]
            tokens = param_name_tokens[current_param_name]
            idx = min(state_tokens, len(tokens) - 1)
            return [tokens[idx]]

        elif state == DecoderState.PARAM_COLON:
            idx = min(state_tokens, len(self.param_colon_tokens) - 1)
            return [self.param_colon_tokens[idx]]

        elif state == DecoderState.PARAM_SEP:
            idx = min(state_tokens, len(self.param_sep_tokens) - 1)
            return [self.param_sep_tokens[idx]]

        elif state == DecoderState.PARAM_VALUE:
            param_type = function_def.parameters[current_prm].type

            if param_type in ["number", "integer"]:
                number_tokens = self.model.encode('0123456789.-').tolist()[0]
                if filled_params >= len(function_def.parameters) - 1:
                    # last param — only allow } to terminate
                    separator_tokens = self.model.encode('}').tolist()[0]
                else:
                    # not last param — allow , to terminate
                    separator_tokens = self.model.encode(',').tolist()[0]
                return number_tokens + separator_tokens

            elif param_type == "boolean":
                return [
                    self.vocabulary.token_to_id.get("true", -1),
                    self.vocabulary.token_to_id.get("false", -1)
                ]

            elif param_type == "string":
                return [self.vocabulary.token_to_id.get('"', -1)]

        elif state == DecoderState.PARAM_STRING:
            return list(range(len(self.vocabulary.id_to_token)))

        elif state == DecoderState.END:
            idx = min(state_tokens, len(self.end_tokens) - 1)
            return [self.end_tokens[idx]]

        elif state == DecoderState.DONE:
            return []

        return []

    def mask_logits(self, logits: list[float], valid_ids: list[int]) -> list[float]:
        masked = list(logits)
        valid_set = set(valid_ids)
        for i in range(len(masked)):
            if i not in valid_set:
                masked[i] = float('-inf')
        return masked

    def update_state(self, state: DecoderState, next_token: int, function_def: FunctionDefinition,
                     current_prm: str, state_tokens: int, filled_params: int,
                     function_name_tokens: list[int],
                     param_name_tokens: dict[str, list[int]],
                     params_list: list[str],
                     param_index: int) -> tuple[DecoderState, str]:

        if state == DecoderState.START:
            return DecoderState.NAME_KEY, current_prm

        elif state == DecoderState.NAME_KEY:
            if state_tokens >= len(self.name_key_tokens):
                return DecoderState.NAME_VALUE, current_prm
            return DecoderState.NAME_KEY, current_prm

        elif state == DecoderState.NAME_VALUE:
            if state_tokens >= len(function_name_tokens):
                return DecoderState.PARAMS_KEY, current_prm
            return DecoderState.NAME_VALUE, current_prm

        elif state == DecoderState.PARAMS_KEY:
            if state_tokens >= len(self.params_key_tokens):
                return DecoderState.PARAM_NAME, current_prm
            return DecoderState.PARAMS_KEY, current_prm

        elif state == DecoderState.PARAM_NAME:
            current_param_name = params_list[param_index]
            tokens = param_name_tokens[current_param_name]
            if state_tokens >= len(tokens):
                return DecoderState.PARAM_COLON, current_param_name
            return DecoderState.PARAM_NAME, current_prm

        elif state == DecoderState.PARAM_COLON:
            if state_tokens >= len(self.param_colon_tokens):
                return DecoderState.PARAM_VALUE, current_prm
            return DecoderState.PARAM_COLON, current_prm

        elif state == DecoderState.PARAM_VALUE:
            param_type = function_def.parameters[current_prm].type

            if param_type in ["number", "integer"]:
                decoded = self.vocabulary.id_to_token.get(next_token, "")
                is_number_token = any(c in decoded for c in '0123456789.-')
                if not is_number_token:
                    if filled_params >= len(function_def.parameters) - 1:
                        return DecoderState.END, current_prm
                    return DecoderState.PARAM_SEP, current_prm
                return DecoderState.PARAM_VALUE, current_prm

            elif param_type == "boolean":
                if state_tokens >= 1:
                    if filled_params >= len(function_def.parameters) - 1:
                        return DecoderState.END, current_prm
                    return DecoderState.PARAM_SEP, current_prm
                return DecoderState.PARAM_VALUE, current_prm

            elif param_type == "string":
                return DecoderState.PARAM_STRING, current_prm

        elif state == DecoderState.PARAM_STRING:
            decoded = self.vocabulary.id_to_token.get(next_token, "")
            if '"' in decoded:
                if filled_params >= len(function_def.parameters) - 1:
                    return DecoderState.END, current_prm
                return DecoderState.PARAM_SEP, current_prm
            return DecoderState.PARAM_STRING, current_prm

        elif state == DecoderState.PARAM_SEP:
            if state_tokens >= len(self.param_sep_tokens):
                return DecoderState.PARAM_NAME, current_prm
            return DecoderState.PARAM_SEP, current_prm

        elif state == DecoderState.END:
            if state_tokens >= len(self.end_tokens):
                return DecoderState.DONE, current_prm
            return DecoderState.END, current_prm

        return DecoderState.DONE, current_prm

    def generate(self, prompt: str, function_def: FunctionDefinition) -> dict:

        input_ids            = self.model.encode(prompt).tolist()[0]
        function_name_tokens = self.model.encode(function_def.name).tolist()[0]
        param_name_tokens = {
            p: self.model.encode(p).tolist()[0]
            for p in function_def.parameters.keys()
        }
        params_list  = list(function_def.parameters.keys())
        param_index  = 0

        state         = DecoderState.START
        generated     = []
        current_prm   = ""
        state_tokens  = 0
        filled_params = 0

        while state != DecoderState.DONE:

            logits    = self.model.get_logits_from_input_ids(input_ids + generated)
            valid_ids = self.get_valid_tokens(
                state, function_def, current_prm,
                state_tokens, params_list, param_index,
                param_name_tokens, filled_params
            )
            masked     = self.mask_logits(logits, valid_ids)
            next_token = int(np.argmax(masked))

            decoded_token = self.vocabulary.id_to_token.get(next_token, "?")

            generated.append(next_token)
            state_tokens += 1

            new_state, current_prm = self.update_state(
                state, next_token, function_def, current_prm,
                state_tokens, filled_params,
                function_name_tokens, param_name_tokens,
                params_list, param_index
            )
            
            if state in [DecoderState.PARAM_VALUE, DecoderState.PARAM_STRING] and new_state == DecoderState.PARAM_SEP:
                filled_params += 1
                param_index   += 1

            if new_state != state:
                state_tokens = 0

            state = new_state

        json_str = self.model.decode(generated)
        print(f"Generated string: {repr(json_str)}")
        decoder_json = json.JSONDecoder()
        result, _ = decoder_json.raw_decode(json_str.strip())
        return result
    
    
    
# if __name__ == "__main__":
#     from pathlib import Path
#     from prompt_builder import PromptBuilder
#     from jsonparser import JsonParser

#     print("Loading model...")
#     model = Small_LLM_Model()

#     print("Loading vocabulary...")
#     vocabulary = Vocabulary.from_model(model)

#     print("Loading input files...")
#     fn_parser = JsonParser(path=Path("data/input"), name="functions_definition.json")
#     pr_parser = JsonParser(path=Path("data/input"), name="function_calling_tests.json")

#     functions = fn_parser.load_functions()
#     prompts   = pr_parser.load_prompts()

#     print("Building decoder...")
#     decoder = Decoder(model=model, vocabulary=vocabulary)

#     builder = PromptBuilder(functions=functions)

#     # expected function for each prompt
#     expected = [
#         "fn_add_numbers",            # What is the sum of 2 and 3?
#         "fn_add_numbers",            # What is the sum of 265 and 345?
#         "fn_greet",                  # Greet shrek
#         "fn_greet",                  # Greet john
#         "fn_reverse_string",         # Reverse the string 'hello'
#         "fn_reverse_string",         # Reverse the string 'world'
#         "fn_get_square_root",        # What is the square root of 16?
#         "fn_get_square_root",        # Calculate the square root of 144
#         "fn_substitute_string_with_regex",  # Replace all numbers...
#         "fn_substitute_string_with_regex",  # Replace all vowels...
#         "fn_substitute_string_with_regex",  # Substitute the word...
#     ]

#     results  = []
#     correct  = 0

#     for i, prompt in enumerate(prompts):
#         print(f"\n{'='*50}")
#         print(f"Prompt {i+1}/{len(prompts)}: {prompt.prompt}")
#         print(f"Expected function: {expected[i]}")

#         try:
#             prompt_str = builder.build(prompt)

#             # find the expected function definition
#             fn = next(f for f in functions if f.name == expected[i])

#             result = decoder.generate(prompt_str, fn)

#             print(f"Result: {result}")
#             results.append({
#                 "prompt":     prompt.prompt,
#                 "name":       result["name"],
#                 "parameters": result["parameters"]
#             })

#             if result["name"] == expected[i]:
#                 correct += 1
#                 print("✓ Correct function")
#             else:
#                 print(f"✗ Wrong function — got {result['name']}")

#         except Exception as e:
#             print(f"Error: {e}")
#             results.append({
#                 "prompt":     prompt.prompt,
#                 "name":       "error",
#                 "parameters": {}
#             })

#     print(f"\n{'='*50}")
#     print(f"Accuracy: {correct}/{len(prompts)} correct")
#     print(f"\nAll results:")
#     for r in results:
#         print(r)