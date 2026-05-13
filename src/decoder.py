from pydantic import BaseModel, model_validator
from enum import Enum
from llm_sdk import Small_LLM_Model
from .vocabulary import Vocabulary
from .models import FunctionDefinition


class DecoderState(Enum):
    start = 'start'
    name_key = 'name_key'
    name_value = 'name_value'
    param_key = 'param_key'
    param_name = 'param_name'
    param_colon = 'param_colon'
    param_value = 'param_value'
    param_string = 'param_string'
    param_sep = 'param_sep'
    end = 'end'
    done = 'done'

class Decoder(BaseModel):
    model: Small_LLM_Model
    vocabulary: Vocabulary
    
    token_psotion: int = 0
    name_key_tokens: list[int] = []
    params_key_tokens: list[int] = []
    param_colon_tokens: list[int] = []
    param_sep_tokens: list[int] = []
    end_tokens: list[int] = []
    
    
    model_config = {"arbitrary_types_allowed": True}
    
    @model_validator(mode='after')
    def precompute_tokens(self) -> 'Decoder':
        self.name_key_tokens = list(self.model.encode('"name": ')[0])
        self.params_key_tokens = list(self.model.encode(', "parameters": {')[0])
        self.param_colon_tokens = list(self.model.encode(': ')[0])
        self.param_sep_tokens = list(self.model.encode(', ')[0])
        self.end_tokens = list(self.model.encode('}')[0])
        
        return self
    
    
    def get_allowed_tokens(self ,state: DecoderState, function: FunctionDefinition) -> list[int]:
        