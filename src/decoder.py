from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Tuple
import numpy as np
import json

from llm_sdk import Small_LLM_Model
from vocabulary import Vocabulary
from models import FunctionDefinition


class DecoderState(Enum):
    START        = "start"
    NAME_KEY     = "name_key"
    NAME_VALUE   = "name_value"
    PARAMS_KEY   = "params_key"
    PARAM_NAME   = "param_name"
    PARAM_VALUE  = "param_value"
    END          = "end"
    DONE         = "done"


class Decoder(BaseModel):
    model: Small_LLM_Model
    vocabulary: Vocabulary
    
    model_config = {"arbitrary_types_allowed": True}
    
    def get_valid_tokens(self, state: str, function_def: FunctionDefinition, current_param: str = "") -> list[int]:
    
        if state == DecoderState.START:
            return [self.vocabulary.token_to_id["{"]]

        elif state == DecoderState.NAME_KEY:
            return self.model.encode('"name": "').tolist()[0]

        elif state == DecoderState.NAME_VALUE:
            return [self.vocabulary.token_to_id[name] for name in [function_def.name]]

        elif state == DecoderState.PARAMS_KEY:
            return self.model.encode('", "parameters": {').tolist()[0]

        elif state == DecoderState.PARAM_NAME:
            return [self.vocabulary.token_to_id[f'"{p}"'] for p in function_def.parameters.keys()]

        elif state == DecoderState.PARAM_VALUE:
            param_type = function_def.parameters[current_param].type
    
            if param_type in ["number", "integer"]:
                return self.model.encode('0123456789.-').tolist()[0]

            elif param_type == "boolean":
                return [
                    self.vocabulary.token_to_id["true"],
                    self.vocabulary.token_to_id["false"]
                ]

            elif param_type == "string":
                return [i for i in range(len(self.vocabulary.id_to_token))]

        elif state == DecoderState.END:
            return self.model.encode("}}").tolist()[0]

        elif state == DecoderState.DONE:
            return []

        return []


    def mask_logits(self, logits: List[float], valid_ids: List[int]) -> List[float]:
        masked = list(logits)
        valid_set = set(valid_ids)
        for i in range(len(masked)):
            if i not in valid_set:
                masked[i] = float('-inf')
        return masked
    
    
    def generate(self, prompt: str, function_def: FunctionDefinition) -> Dict:
        
        input_id = self.model.encode(prompt)
        
        
        state = DecoderState.START
        
        generated = []
        
        current_prm = ""
        
        
        while state != DecoderState.DONE:
            
            logits = self.model.get_logits_from_input_ids(input_id + generated)
            
            valid_ids = self.get_valid_tokens(state, function_def, current_param)
            
            masked = self.mask_logits(logits, valid_ids)
            
            next_token = int(np.argmax(masked))
            
            generated.append(next_token)
            
            
            state, current_param = self.update_state(state, next_token, function_def, current_prm)
            
            
        json_str = self.model.decode(generated)
        
        return json.loads(json_str)