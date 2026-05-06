from typing import List, Dict, Any
from pydantic import BaseModel, ValidationError
import json


class Parametere(BaseModel):
    type: str
    

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameteres: Dict[str, Parametere]
    returns: Dict[str, str]


class PromptItem(BaseModel):
    prompt: str
    
    
def parse_json_file(path: str) -> Any:
    """
    open the jason file in the directory provided by the path
    
    read the json file in return a python loaded version of it
    
    raise FileNotFoundError in case the path is incorrect and JSONDecodeError in case 
    of an invalide json formate
    
    """
    
    try:
        with  open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"File not fount in the provided path ({path})")
    except PermissionError:
        raise RuntimeError("You don't have the permission to read from the JSON file")
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON in file: {path}")
    
def 