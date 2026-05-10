from pydantic import BaseModul, Field, ValidationError, ModulValidator
from typing import Dict, Any


class FunctionDefinition(BaseModul):
    name: str = Field(...)
    description: str = Field(...)
    parameters: Dict[str, Dict[str, str]]= Field(...)
    returns:  Dict[str, Dict[str, str]] = Field(...)
    
    @ModulValidator(mode="after")