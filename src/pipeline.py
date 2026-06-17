from .models import FunctionDefinition
from .prompt_builder import PromptBuilder, Prompt
from llm_sdk import Small_LLM_Model



def select_function(
                    prompt: str,
                    functions: list[FunctionDefinition],
                    model: Small_LLM_Model
                    ) -> FunctionDefinition:
    builder = PromptBuilder(functions=functions)
    builded_prompt = builder.build_selection(Prompt(prompt=prompt))
    
    
    input_ids = [int(x) for x in model.encode(builded_prompt)[0]]
    
    candidates = []
    
    
    for fn in functions:
        full = [int(x) for x in model.encode(builded_prompt + fn.name)[0]]
        # the function name tokens are the difference
        name_tokens = full[len(input_ids):]
        candidates.append((fn, name_tokens))  
    
    i = 0
    while len(candidates) > 1:
        logits = model.get_logits_from_input_ids(input_ids)

        valid_ids = {tokens[i] for _, tokens in candidates if i < len(tokens)}


        for idx in range(len(logits)):
            if idx not in valid_ids:
                logits[idx] = float('-inf')

        token_id = logits.index(max(logits))
        

        candidates = [
            (fn, tokens)
            for fn, tokens in candidates
            if i < len(tokens) and tokens[i] == token_id
        ]
        

        if len(candidates) == 0:
            raise ValueError(f"No matching function found for prompt: '{prompt}'")

        input_ids.append(token_id)
        i += 1
        

    return candidates[0][0]





# if __name__ == "__main__":
#     import os
#     os.environ["HF_HOME"] = "/home/rhlou/goinfre/huggingface"
#     import sys
#     sys.path.insert(0, "/home/rhlou/goinfre/torch-packages")
#     sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe/llm_sdk")
#     sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe")

#     from llm_sdk import Small_LLM_Model
#     from src.models import FunctionDefinition, ParameterSpec, ReturnSpec

#     # replace relative imports at top of file temporarily
#     from src.prompt_builder import PromptBuilder, Prompt

#     model = Small_LLM_Model()

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
#         )
#     ]

#     prompts = [
#         "What is the sum of 2 and 3?",
#         "Greet John",
#     ]

#     for prompt in prompts:
#         result = select_function(prompt, functions, model)
#         print(f"prompt: {prompt}")
#         print(f"selected: {result.name}")
#         print()