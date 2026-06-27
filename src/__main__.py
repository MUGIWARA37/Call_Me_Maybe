import os
import sys

os.environ["HF_HOME"] = "/home/rhlou/goinfre/huggingface"
sys.path.insert(0, "/home/rhlou/goinfre/torch-packages")
sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe/llm_sdk")

import argparse  # noqa: E402
import json  # noqa: E402
from llm_sdk import Small_LLM_Model  # noqa: E402
from .vocabulary import Vocabulary  # noqa: E402
from .decoder import Decoder  # noqa: E402
from .pipeline import select_function  # noqa: E402
from .jsonparser import JsonParser  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Function-calling pipeline"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json"
    )
    args = parser.parse_args()

    # Load input files.
    try:
        functions = JsonParser(
            filepath=args.functions_definition
        ).load_functions()
        prompts = JsonParser(filepath=args.input).load_prompts()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load models.
    model_selector = Small_LLM_Model()
    model_decoder = Small_LLM_Model("Qwen/Qwen2.5-Coder-0.5B")
    vocabulary = Vocabulary.from_model(model_decoder)
    decoder = Decoder(model=model_decoder, vocabulary=vocabulary)

    # Run the pipeline on every prompt.
    results = []
    print("Processing prompts...")
    try:
        for prompt in prompts:
            function = select_function(
                prompt.prompt, functions, model_selector
            )
            parameters = decoder.generate(prompt.prompt, function)
            results.append({
                "prompt": prompt.prompt,
                "name": function.name,
                "parameters": parameters
            })
            print(f"  User request   : {prompt.prompt}")
            print(f"  Function choosed  : {function.name}")
            print(f"  Parameters found: {parameters}\n")
    except KeyboardInterrupt:
        print(
            f"\nInterrupted — saving {len(results)} partial result(s)..."
        )

    # Write whatever results were collected (full run or partial).
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Done — {len(results)} result(s) written to '{args.output}'.")


if __name__ == "__main__":
    main()
