from pydantic import BaseModel
from typing import List, Dict
from pathlib import Path
from .models import FunctionDefinition, Prompt
from llm_sdk import Small_LLM_Model
from .decoder import Decoder
from .vocabulary import Vocabulary
from .prompt_builder import PromptBuilder
import json

class Pipline(BaseModel):
    
    model: Small_LLM_Model
    vocabulary: Vocabulary
    decoder: Decoder
    functions: List[FunctionDefinition]
    prompts: List[Prompt]
    output_path: Path
    
    
    model_config = {"arbitry_types_allowed": True}
    
    
    def select_function(self, prompt: Prompt) -> FunctionDefinition:
        prmt_builder = PromptBuilder(functions=self.functions)
        prompt_str = prmt_builder.build(prompt)
        tokens = self.model.encode(prompt_str).tolist()[0]
        logits = self.model.get_logits_from_input_ids(tokens)
        
        best_function = None
        best_score    = float('-inf')
        for function in self.functions:
            func_name_token = self.model.encode(function.name).tolist()[0][0]
            score = logits[func_name_token]
            if score > best_score:
                best_score    = score
                best_function = function
            
        return best_function
                
    def run(self) -> List[Dict]:
        resulte: List[Dict] = []
        prmt_builder = PromptBuilder(functions=self.functions)
    
        for prompt in self.prompts:
            try:
                prmpt_str = prmt_builder.build(prompt)
                prmpt_func = self.select_function(prompt)
                decode = self.decoder.generate(prmpt_str, prmpt_func)
                resulte.append({
                    "prompt":     prompt.prompt,
                    "name":       decode["name"],
                    "parameters": decode["parameters"]
                })
            except Exception as e:
                print(f"Error processing  prompt '{prompt.prompt}': {e}")
        
        return resulte
    
    def write_output(self, results: list[dict]) -> None:
        try:
            self.output_path.mkdir(parents=True, exist_ok=True)
            output_file = self.output_path / "function_calling_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except PermissionError as e:
            raise PermissionError(f"Error: You don't have the permission to write to {self.output_path}") from e
        except OSError as e:
            raise OSError(f"Error: could not write output file to {self.output_path}") from e