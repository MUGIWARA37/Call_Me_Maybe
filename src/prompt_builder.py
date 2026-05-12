from models import FunctionDefinition, Prompt
from pydantic import BaseModel
from typing import List

class PromptBuilder(BaseModel):
    functions: List[FunctionDefinition]
    
    
    def build(self,prompt: Prompt) -> str:
        functions_str = "\n".join(str(fn) for fn in self.functions)
        
        return (
            f"You are a function calling assistant.\n\n"
            f"Available functions:\n{functions_str}\n\n"
            f"User request: {prompt.prompt}\n\n"
            f"Respond with only a JSON object in this exact format:\n"
            f'{{"name": "function_name", "parameters": {{"arg1": value1}}}}'
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
    # print(builder.build(prompt))