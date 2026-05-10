from pydantic import BaseModel, model_validator
from typing import Dict


valid_data_type = ["number", "string"]

class ParameterSpec(BaseModel):
    type: str
    
    @model_validator(mode='after')
    def param_validator(self) -> 'ParameterSpec':
        if not self.type in valid_data_type:
            raise ValueError(f"Error: {self.type} is Invalide data type !!")
        
        return self

class ReturnSpec(BaseModel):
    type: str
    
    @model_validator(mode='after')
    def return_validator(self) -> 'ReturnSpec':
        if not self.type in valid_data_type:
            raise ValueError(f"Error: {self.type} is Invalide data type !!")
        
        return self



class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterSpec]
    returns:  ReturnSpec
    
    
class Prompt(BaseModel):
    prompt: str