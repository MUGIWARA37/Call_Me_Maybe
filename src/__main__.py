import os
os.environ["HF_HOME"] = "/home/rhlou/goinfre/huggingface"
import sys
sys.path.insert(0, "/home/rhlou/goinfre/torch-packages")
sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe/llm_sdk")

import argparse
import json
from llm_sdk import Small_LLM_Model
from .vocabulary import Vocabulary
from .decoder import Decoder
from .pipeline import select_function
from .jsonparser import JsonParser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    args = parser.parse_args()

    # load data
    functions = JsonParser(filepath=args.functions_definition).load_functions()
    prompts = JsonParser(filepath=args.input).load_prompts()

    # init model and tools
    model = Small_LLM_Model()
    vocabulary = Vocabulary.from_model(model)
    decoder = Decoder(model=model, vocabulary=vocabulary)

    # process each prompt
    results = []
    for prompt in prompts:
        function = select_function(prompt.prompt, functions, model)
        parameters = decoder.generate(prompt.prompt, function)
        results.append({
            "prompt": prompt.prompt,
            "name": function.name,
            "parameters": parameters
        })
        print(f"user request: {prompt}\nfunction selected: {function.name}\nParameteres choosed: {parameters}")

    # write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Done! {len(results)} results written to {args.output}")


if __name__ == "__main__":
    main()