from pydantic import BaseModel
from typing import List, Dict
from pathlib import Path
from .models import FunctionDefinition, Prompt
from llm_sdk import Small_LLM_Model
from .decoder import Decoder
from .vocabulary import Vocabulary
from .prompt_builder import PromptBuilder

class Pipline(BaseModel):
    
    model: Small_LLM_Model
    vocabulary: Vocabulary
    decoder: Decoder
    functions: List[FunctionDefinition]
    prompts: List[Prompt]
    output_path: Path
    
    
    model_config = {"arbetry_types_allowed": True}
    
    
    def select_function(self, prompt: Prompt) -> FunctionDefinition:
        prmt_builder = PromptBuilder(functions=self.functions)
        prompt_str = prmt_builder.build(prompt)
        tokens = self.model.encode(prompt_str).tolist()[0]
        logits = self.model.get_logits_from_input_ids(tokens)
        
                
    def run(self) -> List[Dict]:
        pass
    
    def write_output(self) -> None:
        pass