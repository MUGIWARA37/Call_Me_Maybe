import os
os.environ["HF_HOME"] = "/home/rhlou/goinfre/huggingface"
import sys
sys.path.insert(0, "/home/rhlou/goinfre/torch-packages")
sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe/llm_sdk")








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
    
    
    input_ids = list(model.encode(builded_prompt)[0])
    
    candidates = [
        (fn, [int(x) for x in model.encode(fn.name)[0]])
        for fn in functions
    ]
    
    
    print(f"builded prompt: {builded_prompt}")
    print(f"candidates tokens:")
    for fn, tokens in candidates:
        print(f"  {fn.name}: {tokens}")
    
    i = 0
    while len(candidates) > 1:
        logits = model.get_logits_from_input_ids(input_ids)

        valid_ids = {
            token
            for _, tokens in candidates
            for token in tokens
        }
        print(f"i={i} valid_ids={valid_ids}")

        for idx in range(len(logits)):
            if idx not in valid_ids:
                logits[idx] = float('-inf')

        token_id = logits.index(max(logits))
        print(f"i={i} chosen token_id={token_id}")

        candidates = [
            (fn, tokens)
            for fn, tokens in candidates
            if token_id in tokens
        ]

        if len(candidates) == 0:
            raise ValueError(f"No matching function found for prompt: '{prompt}'")

        input_ids.append(token_id)
        i += 1
        
        print(f"i={i} remaining candidates={[fn.name for fn, _ in candidates]}")

    return candidates[0][0]





if __name__ == "__main__":
    import os
    os.environ["HF_HOME"] = "/home/rhlou/goinfre/huggingface"
    import sys
    sys.path.insert(0, "/home/rhlou/goinfre/torch-packages")
    sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe/llm_sdk")
    sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe")

    from llm_sdk import Small_LLM_Model
    from src.models import FunctionDefinition, ParameterSpec, ReturnSpec

    # replace relative imports at top of file temporarily
    from src.prompt_builder import PromptBuilder, Prompt

    model = Small_LLM_Model()

    functions = [
        FunctionDefinition(
            name="fn_add_numbers",
            description="Add two numbers together and return their sum.",
            parameters={
                "a": ParameterSpec(type="number"),
                "b": ParameterSpec(type="number")
            },
            returns=ReturnSpec(type="number")
        ),
        FunctionDefinition(
            name="fn_greet",
            description="Generate a greeting message for a person by name.",
            parameters={
                "name": ParameterSpec(type="string")
            },
            returns=ReturnSpec(type="string")
        )
    ]

    prompts = [
        "What is the sum of 2 and 3?",
        "Greet John",
    ]

    for prompt in prompts:
        result = select_function(prompt, functions, model)
        print(f"prompt: {prompt}")
        print(f"selected: {result.name}")
        print()