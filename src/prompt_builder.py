from .models import FunctionDefinition, Prompt as Prompt
from pydantic import BaseModel
from typing import List


class PromptBuilder(BaseModel):
    """Builds the text prompts that are fed into the LLM."""

    functions: List[FunctionDefinition]

    def build_selection(self, prompt: Prompt) -> str:
        """Build a prompt that asks the model to pick a function name."""
        functions_str = "\n".join(str(fn) for fn in self.functions)
        return (
            "You are a function calling assistant.\n\n"
            f"Available functions:\n{functions_str}\n\n"
            f"User request: '''{prompt.prompt}'''\n\n"
            "Reply with only the function name:"
        )

    def build_parameters(
        self, prompt: Prompt, function: FunctionDefinition
    ) -> str:
        """Build a prompt that asks the model to extract parameter values."""
        signature = (
            f"{function.name}"
            "("
            + ", ".join(
                f"{n}: {s.type}"
                for n, s in function.parameters.items()
            )
            + f") -> {function.returns.type}"
        )

        few_shot = ""
        if "regex" in function.parameters:
            few_shot = (
                "Example:\n"
                "User request: '''Replace all numbers in "
                "\"Hello 34 I'm 233 years old\" with NUMBERS'''\n"
                "{\n"
                '"source_string": "Hello 34 I\'m 233 years old",\n'
                '"regex": "\\\\d+",\n'
                '"replacement": "NUMBERS"\n'
                "}\n\n"
                "Example:\n"
                "User request: '''Replace all vowels in "
                "'Programming is fun' with asterisks'''\n"
                "{\n"
                '"source_string": "Programming is fun",\n'
                '"regex": "[aeiouAEIOU]",\n'
                '"replacement": "*"\n'
                "}\n\n"
                "Example:\n"
                "User request: '''Substitute the word 'cat' with 'dog' "
                "in 'The cat sat on the mat with another cat''''  \n"
                "{\n"
                '"source_string": "The cat sat on the mat with another cat",\n'
                '"regex": "cat",\n'
                '"replacement": "dog"\n'
                "}\n\n"
            )

        return (
            f"You are a parameter extractor.\n\n"
            f"Function: {signature}\n\n"
            f"Rules:\n"
            f"- Extract the raw input values exactly as they appear.\n"
            f"- Do NOT compute, solve, evaluate or execute the function.\n"
            f"- Preserve the exact spelling and casing of all strings.\n\n"
            f"{few_shot}"
            f"Example:\n"
            f"User request: '''{prompt.prompt}'''\n"
        )
