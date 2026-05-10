from pydantic import BaseModel, ValidationError
from typing import Any, List, Dict
from pathlib import Path
import json
from .models import FunctionDefinition, Prompt, ParameterSpec, ReturnSpec


class JsonParser (BaseModel):
    path: Path
    name: str
    
    
    def read_json_file(self) ->  List[Any] | Dict[str, Any]:
        
        
        try:
            with open(self.path / self.name, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: the path {self.path} or the name {self.name} is inccorect !!") from e
        except PermissionError as e:
            raise PermissionError(f"Error: you don't have the right to read from the file {self.path / self.name}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Error: Invalide JSON format in the file {self.name}") from e

    def load_functions(self) -> list[FunctionDefinition]:
        try:
            data = self.read_json_file()
            return [FunctionDefinition.model_validate(definition) for definition in data]
        except ValidationError as e:
            raise ValueError(f"Error: invalid function definition structure: {e}") from e







# if __name__ == "__main__":
# json_file = JsonParser(path=Path("data/input"), name="functions_definition.json")
# print(json_file.read_json_file())