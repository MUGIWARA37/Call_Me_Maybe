import os
os.environ["HF_HOME"] = "/home/rhlou/goinfre/huggingface"

import sys
sys.path.insert(0, "/home/rhlou/goinfre/torch-packages")
sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe/llm_sdk")
sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe")

from pathlib import Path

from llm_sdk import Small_LLM_Model
from src.vocabulary import Vocabulary
from src.decoder import Decoder
from src.jsonparser import JsonParser
from src.prompt_builder import PromptBuilder
# from src.pipeline import select_function


def main():
    fn_parser = JsonParser(
        path=Path("data/input"),
        name="functions_definition.json"
    )
    
    prompt_parser = JsonParser(
        path=Path("data/input"),
        name="function_calling_tests.json"
    )

    functions = fn_parser.load_functions()
    prompts   = prompt_parser.load_prompts()

    print(f"[info] loaded {len(functions)} functions")
    print(f"[info] loaded {len(prompts)} prompts")

    # ── init model, vocabulary, decoder, prompt builder ───────────────────────
    print("[info] loading model...")
    model      = Small_LLM_Model()
    vocabulary = Vocabulary.from_model(model)
    decoder    = Decoder(model=model, vocabulary=vocabulary)
    builder    = PromptBuilder(functions=functions)

    # ── run each prompt ────────────────────────────────────────────────────────
    passed = 0
    failed = 0

    for i, prompt in enumerate(prompts):
        built_prompt = builder.build(prompt)

        # reset decoder state between runs
        decoder.token_position = 0
        decoder.filled_params  = 0

        function = select_function(prompt.prompt, functions)

        try:
            result = decoder.generate(built_prompt, function)
            print(f"[{i+1:02d}] ✓  {prompt.prompt}")
            print(f"        → {result}")
            passed += 1
        except Exception as e:
            print(f"[{i+1:02d}] ✗  {prompt.prompt}")
            print(f"        → error: {e}")
            failed += 1

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print(f"results: {passed}/{passed + failed} passed")


if __name__ == "__main__":
    main()
