from .models import FunctionDefinition
from .prompt_builder import PromptBuilder, Prompt
from llm_sdk import Small_LLM_Model
import numpy as np


def select_function(
    prompt: str,
    functions: list[FunctionDefinition],
    model: Small_LLM_Model
) -> FunctionDefinition:
    """Pick the right function for a prompt using constrained token search.

    How it works:
    1. Build a selection prompt listing all available functions.
    2. Tokenise each function name to get its token sequence.
    3. At each step, only allow tokens that appear next in at least one
       candidate.
    4. Keep only candidates whose token matches the chosen token.
    5. Stop when one candidate remains.
    """
    builder = PromptBuilder(functions=functions)
    base_prompt = builder.build_selection(Prompt(prompt=prompt))
    base_ids = model.encode(base_prompt)[0].numpy().tolist()

    # Build (function, name_token_list) pairs.
    candidates = []
    for fn in functions:
        full = model.encode(base_prompt + fn.name)[0].numpy().tolist()
        name_tokens = full[len(base_ids):]
        candidates.append((fn, name_tokens))

    input_ids = list(base_ids)
    step = 0

    while len(candidates) > 1:
        logits = model.get_logits_from_input_ids(input_ids)

        # Only allow tokens that appear at position `step` in a live candidate.
        valid_ids = {
            tokens[step] for _, tokens in candidates if step < len(tokens)
        }

        mask = np.full(len(logits), float('-inf'))
        mask[list(valid_ids)] = np.array(logits)[list(valid_ids)]
        best_token = int(np.argmax(mask))

        # Drop candidates that diverge from the chosen token.
        candidates = [
            (fn, tokens)
            for fn, tokens in candidates
            if step < len(tokens) and tokens[step] == best_token
        ]

        if not candidates:
            raise ValueError(
                f"No matching function found for prompt: '{prompt}'"
            )

        input_ids.append(best_token)
        step += 1

    return candidates[0][0]
