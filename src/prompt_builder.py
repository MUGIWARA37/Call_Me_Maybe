from .models import FunctionDefinition, Prompt
from pydantic import BaseModel
from typing import List

class PromptBuilder(BaseModel):
    functions: List[FunctionDefinition]
    
    
    def build_selection(self,prompt: Prompt) -> str:
        """Build a prompt for function selection only.

        Args:
            prompt: The user request.

        Returns:
            A prompt string asking the LLM to pick a function name.
        """
        functions_str = "\n".join(str(fn) for fn in self.functions)
        
        return (
            "You are a function calling assistant.\n\n"
            f"Available functions:\n{functions_str}\n\n"
            f"User request: '''{prompt.prompt}'''\n\n"
            "Reply with only the function name:"
        )
        
        
    def build_parameters(self, prompt: Prompt, function: FunctionDefinition) -> str:
        """Build a prompt that extracts parameter values from a user request.

        Adds few-shot examples only when the function has a 'regex' parameter,
        to guide the model toward concise regex patterns.

        Args:
            prompt: The user request.
            function: The function whose parameters must be extracted.

        Returns:
            A prompt string ready to feed into the LLM decoder.
        """
        signature = (
            f"{function.name}"
            f"({', '.join(f'{n}: {s.type}' for n, s in function.parameters.items())})"
            f" -> {function.returns.type}"
        )
        schema = (
            "{"
            + ", ".join(
                f'"{n}": <{s.type}>' for n, s in function.parameters.items()
            )
            + "}"
        )

        few_shot = ""
        if function.name == "fn_get_square_root":
            few_shot = (
                "Example:\n"
                "User request: '''What is the square root of 16?'''\n"
                "{\n"
                '"a": 16.0\n'
                "}\n\n"
                "Example:\n"
                "User request: '''Calculate the square root of 144'''\n"
                "{\n"
                '"a": 144.0\n'
                "}\n\n"
            )
        elif "regex" in function.parameters:
            few_shot = (
                "Example:\n"
                "User request: '''Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS'''\n"
                "{\n"
                '"source_string": "Hello 34 I\'m 233 years old",\n'
                '"regex": "[0-9]+",\n'
                '"replacement": "NUMBERS"\n'
                "}\n\n"
                "Example:\n"
                "User request: '''Replace all vowels in 'Programming is fun' with asterisks'''\n"
                "{\n"
                '"source_string": "Programming is fun",\n'
                '"regex": "[aeiouAEIOU]",\n'
                '"replacement": "*"\n'
                "}\n\n"
                "Example:\n"
                "User request: '''Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat''''\n"
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
            f"- Extract the raw input values exactly as they appear in the user request.\n"
            f"- Do NOT compute, solve, evaluate or execute the function.\n"
            f"- Preserve the exact spelling and casing of all string values.\n\n"
            f"{few_shot}"
            f"Example:\n"
            f"User request: '''{prompt.prompt}'''\n"
        )



        
        
        


# if __name__ == "__main__":
    # from models import FunctionDefinition, ParameterSpec, ReturnSpec, Prompt
# 
    # functions = [
        # FunctionDefinition(
            # name="fn_add_numbers",
            # description="Add two numbers together and return their sum.",
            # parameters={
                # "a": ParameterSpec(type="number"),
                # "b": ParameterSpec(type="number")
            # },
            # returns=ReturnSpec(type="number")
        # ),
        # FunctionDefinition(
            # name="fn_greet",
            # description="Generate a greeting message for a person by name.",
            # parameters={
                # "name": ParameterSpec(type="string")
            # },
            # returns=ReturnSpec(type="string")
        # )
    # ]
# 
    # prompt = Prompt(prompt="What is the sum of 40 and 2?")
# 
    # builder = PromptBuilder(functions=functions)
    # print(builder.build_selection(prompt))