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


class Decoder(BaseModel):
    model: Small_LLM_Model
    vocabulary: Vocabulary
    
    model_config = {"arbitrary_types_allowed": True}
    
    def generate(self, prompt: str, function: FunctionDefinition) -> dict[str, Any]:
        builder = PromptBuilder(functions=[function])
        built_prompt = builder.build_parameters(Prompt(prompt=prompt), function)

        input_ids = [int(x) for x in self.model.encode(built_prompt)[0]]
        prompt_length = len(input_ids)

        # force {
        logits = self.model.get_logits_from_input_ids(input_ids)
        for i in range(len(logits)):
            if i != self.vocabulary.token_to_id['{']:
                logits[i] = float('-inf')
        token_id = logits.index(max(logits))
        input_ids.append(token_id)

        for param_name, param_spec in function.parameters.items():
            # force param name
            param_name_ids = [int(x) for x in self.model.encode(f'"{param_name}": ')[0]]
            for token in param_name_ids:
                logits = self.model.get_logits_from_input_ids(input_ids)
                for i in range(len(logits)):
                    if i != token:
                        logits[i] = float('-inf')
                token_id = logits.index(max(logits))
                input_ids.append(token_id)

            is_last = list(function.parameters.keys())[-1] == param_name
            sep_tokens = [int(x) for x in self.model.encode('}')[0]] if is_last else [int(x) for x in self.model.encode(', "')[0]]

            if param_spec.type in ("number", "integer"):
                numeric_ids = [
                    v for k, v in self.vocabulary.token_to_id.items()
                    if k and all(ch in "0123456789.-" for ch in k)
                ]

                while True:
                    logits = self.model.get_logits_from_input_ids(input_ids)
                    allowed = numeric_ids + [sep_tokens[0]]
                    for i in range(len(logits)):
                        if i not in allowed:
                            logits[i] = float('-inf')
                    token_id = logits.index(max(logits))
                    input_ids.append(token_id)
                    if token_id == sep_tokens[0]:
                        break

            elif param_spec.type == "boolean":
                true_tokens = [int(x) for x in self.model.encode("true")[0]]
                false_tokens = [int(x) for x in self.model.encode("false")[0]]
                # first token: only allow true[0] or false[0]
                logits = self.model.get_logits_from_input_ids(input_ids)
                for i in range(len(logits)):
                    if i not in {true_tokens[0], false_tokens[0]}:
                        logits[i] = float('-inf')
                token_id = logits.index(max(logits))
                input_ids.append(token_id)
                # force the rest of the chosen boolean
                chosen = true_tokens if token_id == true_tokens[0] else false_tokens
                for token in chosen[1:]:
                    logits = self.model.get_logits_from_input_ids(input_ids)
                    for i in range(len(logits)):
                        if i != token:
                            logits[i] = float('-inf')
                    token_id = logits.index(max(logits))
                    input_ids.append(token_id)
                # force separator
                for token in sep_tokens:
                    logits = self.model.get_logits_from_input_ids(input_ids)
                    for i in range(len(logits)):
                        if i != token:
                            logits[i] = float('-inf')
                    token_id = logits.index(max(logits))
                    input_ids.append(token_id)

            elif param_spec.type == "string":
                # force opening "
                quote_id = self.vocabulary.token_to_id['"']
                logits = self.model.get_logits_from_input_ids(input_ids)
                for i in range(len(logits)):
                    if i != quote_id:
                        logits[i] = float('-inf')
                token_id = logits.index(max(logits))
                input_ids.append(token_id)
                # allow any token except "
                string_ids = [v for k, v in self.vocabulary.token_to_id.items() if '"' not in k]
                while True:
                    logits = self.model.get_logits_from_input_ids(input_ids)
                    allowed = string_ids + [quote_id]
                    for i in range(len(logits)):
                        if i not in allowed:
                            logits[i] = float('-inf')
                    token_id = logits.index(max(logits))
                    input_ids.append(token_id)
                    if token_id == quote_id:
                        break
                # force separator
                for token in sep_tokens:
                    logits = self.model.get_logits_from_input_ids(input_ids)
                    for i in range(len(logits)):
                        if i != token:
                            logits[i] = float('-inf')
                    token_id = logits.index(max(logits))
                    input_ids.append(token_id)

        # decode and return
        generated = self.model.decode(input_ids[prompt_length:])
        result, _ = json.JSONDecoder().raw_decode(generated)
        return result




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