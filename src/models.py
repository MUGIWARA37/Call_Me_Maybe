from pydantic import BaseModel, model_validator
from typing import Dict


VALID_TYPES = ["number", "string", "boolean", "integer"]


class ParameterSpec(BaseModel):
    """One parameter of a function (just its type)."""
    type: str

    @model_validator(mode='after')
    def check_type(self) -> 'ParameterSpec':
        if self.type not in VALID_TYPES:
            raise ValueError(f"Invalid type: '{self.type}'. Allowed: {VALID_TYPES}")
        return self


class ReturnSpec(BaseModel):
    """The return type of a function."""
    type: str

    @model_validator(mode='after')
    def check_type(self) -> 'ReturnSpec':
        if self.type not in VALID_TYPES:
            raise ValueError(f"Invalid type: '{self.type}'. Allowed: {VALID_TYPES}")
        return self


class FunctionDefinition(BaseModel):
    """Everything we know about one callable function."""
    name: str
    description: str
    parameters: Dict[str, ParameterSpec]
    returns: ReturnSpec

    @model_validator(mode='after')
    def check_name(self) -> 'FunctionDefinition':
        if not self.name:
            raise ValueError("Function name cannot be empty.")
        return self

    def __str__(self) -> str:
        params = ", ".join(f"{k}: {v.type}" for k, v in self.parameters.items())
        return f"prototype: def {self.name}({params}) -> {self.returns.type}"


class Prompt(BaseModel):
    """A raw user request."""
    prompt: str

    def __str__(self) -> str:
        return self.prompt