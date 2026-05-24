import os
os.environ["HF_HOME"] = "/home/rhlou/goinfre/huggingface"

import sys
sys.path.insert(0, "/home/rhlou/goinfre/torch-packages")
sys.path.insert(0, "/home/rhlou/Desktop/1337/Call_Me_Maybe/llm_sdk")



from pydantic import BaseModel, model_validator
from enum import Enum
import json
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
    
    token_position: int = 0
    filled_params: int = 0
    name_key_tokens: list[int] = []
    params_key_tokens: list[int] = []
    param_colon_tokens: list[int] = []
    param_sep_tokens: list[int] = []
    end_tokens: list[int] = []
    numeric_token_ids: list[int] = []
    all_token_ids: list[int] = []
    string_token_ids: list[int] = []
    quote_token_id: int | None = None
    true_tokens: list[int] = []
    false_tokens: list[int] = []
    boolean_tokens: list[int] = []
    numeric_has_digit: bool = False
    string_token_count: int = 0
    max_string_tokens: int = 64
    
    
    model_config = {"arbitrary_types_allowed": True}
    
    @model_validator(mode='after')
    def precompute_tokens(self) -> 'Decoder':
        self.name_key_tokens = [int(t) for t in self.model.encode('"name": ')[0]]
        self.params_key_tokens = [int(t) for t in self.model.encode(', "parameters": {')[0]]
        self.param_colon_tokens = [int(t) for t in self.model.encode(': ')[0]]
        self.param_sep_tokens = [int(t) for t in self.model.encode(', ')[0]]
        self.end_tokens = [int(t) for t in self.model.encode('}')[0]]
        self.numeric_token_ids = [
            v for k, v in self.vocabulary.token_to_id.items()
            if k and all(ch in "0123456789.-" for ch in k)
        ]
        self.all_token_ids = list(self.vocabulary.token_to_id.values())
        self.string_token_ids = [
            v for k, v in self.vocabulary.token_to_id.items()
            if '"' not in k
        ]
        self.quote_token_id = self.vocabulary.token_to_id.get('"')
        if self.quote_token_id is not None:
            self.string_token_ids.append(self.quote_token_id)
        self.true_tokens = [int(t) for t in self.model.encode("true")[0]]
        self.false_tokens = [int(t) for t in self.model.encode("false")[0]]
        
        
        return self
    
    def _get_sep_tokens(self, function: FunctionDefinition) -> list[int]:
        if self.filled_params < len(function.parameters) - 1:
            return self.param_sep_tokens
        return self.end_tokens
    
    def get_allowed_tokens(self, state: DecoderState, function: FunctionDefinition) -> list[int]:
        
        if state == DecoderState.start:
            return [self.vocabulary.token_to_id['{']]

        elif state == DecoderState.name_key:
            return [self.name_key_tokens[self.token_position]]
        
        elif state == DecoderState.name_value:
            function_name_token = [int(t) for t in self.model.encode(f'"{function.name}"')[0]]
            return [function_name_token[self.token_position]]

        elif state == DecoderState.param_key:
            return [self.params_key_tokens[self.token_position]]
        
        elif state == DecoderState.param_colon:
            return [self.param_colon_tokens[self.token_position]]
        
        elif state == DecoderState.param_name:
            param_name = list(function.parameters.keys())[self.filled_params]
            tokens = [int(t) for t in self.model.encode(f'"{param_name}"')[0]]
            return [tokens[self.token_position]]
        
        elif state == DecoderState.param_value:
            param_name = list(function.parameters.keys())[self.filled_params]
            param_type = function.parameters[param_name].type
            
            if param_type in ["number", "integer"]:
                allowed = list(self.numeric_token_ids)
                if self.numeric_has_digit:
                    allowed.append(self._get_sep_tokens(function)[0])
                return allowed
            
            elif param_type == "boolean":
                if self.boolean_tokens:
                    return [self.boolean_tokens[self.token_position]]
                return list({self.true_tokens[0], self.false_tokens[0]})
            
            elif param_type == "string":
                return [self.vocabulary.token_to_id['"']]
            
        elif state == DecoderState.param_string:
            if self.quote_token_id is not None and self.string_token_count >= self.max_string_tokens:
                return [self.quote_token_id]
            return self.string_token_ids
        
        elif state == DecoderState.param_sep:
            sep_tokens = self._get_sep_tokens(function)
            return [sep_tokens[self.token_position]]
        
        elif state == DecoderState.end:
            return [self.end_tokens[self.token_position]]
        
        elif state == DecoderState.done:
            return []
        
    def get_next_state(self, state: DecoderState, function: FunctionDefinition, last_token: str) -> DecoderState:
        
        if state == DecoderState.start:
            return DecoderState.name_key

        elif state == DecoderState.name_key:
            if self.token_position < len(self.name_key_tokens):
                return DecoderState.name_key
            return DecoderState.name_value

        elif state == DecoderState.name_value:
            tokens = [int(t) for t in self.model.encode(f'"{function.name}"')[0]]
            if self.token_position < len(tokens):
                return DecoderState.name_value
            return DecoderState.param_key
        
        elif state == DecoderState.param_key:
            if self.token_position < len(self.params_key_tokens):
                return DecoderState.param_key
            return DecoderState.param_name

        elif state == DecoderState.param_name:
            param_name = list(function.parameters.keys())[self.filled_params]
            if self.token_position < len([int(t) for t in self.model.encode(f'"{param_name}"')[0]]):
                return DecoderState.param_name
            return DecoderState.param_colon
        
        elif state == DecoderState.param_colon:
            if self.token_position < len(self.param_colon_tokens):
                return DecoderState.param_colon
            return DecoderState.param_value
        
        elif state == DecoderState.param_string:
            if last_token == '"':
                return DecoderState.param_sep
            return DecoderState.param_string
        
        elif state == DecoderState.param_value:
            param_name = list(function.parameters.keys())[self.filled_params]
            param_type = function.parameters[param_name].type
            if param_type == 'string':
                return DecoderState.param_string
            if param_type == 'boolean' and self.boolean_tokens:
                if self.token_position < len(self.boolean_tokens):
                    return DecoderState.param_value
            return DecoderState.param_value
        
        elif state == DecoderState.param_sep:
            sep_tokens = self._get_sep_tokens(function)
            if self.token_position < len(sep_tokens):
                return DecoderState.param_sep
            if self.filled_params < len(function.parameters) - 1:
                return DecoderState.param_name
            return DecoderState.end
        
        elif state == DecoderState.end:
            if self.token_position < len(self.end_tokens):
                return DecoderState.end
            return DecoderState.done
        
    def generate(self, prompt: str, function: FunctionDefinition) -> dict:
        encoded_prompt = list(self.model.encode(prompt)[0])
        generated_tokens: list[int] = []
        
        decod_state = DecoderState.start
        
        while decod_state != DecoderState.done:
            
            allowed_tokens = self.get_allowed_tokens(decod_state, function)
            
            logits = self.model.get_logits_from_input_ids(encoded_prompt)
             
            for i in range(len(logits)):
                if i not in allowed_tokens:
                    logits[i] = float('-inf')
                     
                     
            token_id = logits.index(max(logits))
            print(f"[debug] token: {repr(self.vocabulary.id_to_token[token_id])} state: {decod_state.value}")
            token = self.vocabulary.id_to_token[token_id]
            
            self.token_position += 1
            if decod_state == DecoderState.param_value:
                param_name = list(function.parameters.keys())[self.filled_params]
                param_type = function.parameters[param_name].type
                if param_type == "boolean" and not self.boolean_tokens:
                    if token_id == self.true_tokens[0]:
                        self.boolean_tokens = self.true_tokens
                    elif token_id == self.false_tokens[0]:
                        self.boolean_tokens = self.false_tokens
                    else:
                        raise ValueError("Invalid boolean token emitted")
                if param_type in ["number", "integer"] and any(ch.isdigit() for ch in token):
                    self.numeric_has_digit = True

            if decod_state == DecoderState.param_string:
                if self.quote_token_id is None or token_id != self.quote_token_id:
                    self.string_token_count += 1

            new_state = self.get_next_state(decod_state, function, token)
            sep_started = False
            sep_completed = False
            if decod_state == DecoderState.param_value:
                param_name = list(function.parameters.keys())[self.filled_params]
                param_type = function.parameters[param_name].type
                if param_type in ["number", "integer"]:
                    sep_tokens = self._get_sep_tokens(function)
                    if self.numeric_has_digit and token_id == sep_tokens[0]:
                        if len(sep_tokens) == 1:
                            if self.filled_params < len(function.parameters) - 1:
                                new_state = DecoderState.param_name
                            else:
                                new_state = DecoderState.end
                            sep_completed = True
                        else:
                            new_state = DecoderState.param_sep
                            sep_started = True
                elif param_type == "boolean" and self.boolean_tokens:
                    if self.token_position >= len(self.boolean_tokens):
                        new_state = DecoderState.param_sep
            
            if new_state != decod_state:
                if new_state == DecoderState.param_sep and sep_started:
                    self.token_position = 1
                else:
                    self.token_position = 0
            
            if decod_state == DecoderState.param_sep and new_state == DecoderState.param_name:
                self.filled_params += 1
            if sep_completed and new_state == DecoderState.param_name:
                self.filled_params += 1
            if decod_state != DecoderState.param_value and new_state == DecoderState.param_value:
                self.boolean_tokens = []
                self.numeric_has_digit = False
            if decod_state == DecoderState.param_value and new_state != DecoderState.param_value:
                self.boolean_tokens = []
                self.numeric_has_digit = False
            if decod_state != DecoderState.param_string and new_state == DecoderState.param_string:
                self.string_token_count = 0
            if decod_state == DecoderState.param_string and new_state != DecoderState.param_string:
                self.string_token_count = 0
                
            decod_state = new_state
            encoded_prompt.append(token_id)
            generated_tokens.append(token_id)
            

        output = self.model.decode(generated_tokens)
        print(f"[debug] output: {repr(output)}")
        result, _ = json.JSONDecoder().raw_decode(output)
        return result
