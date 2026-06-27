from pydantic import BaseModel
from llm_sdk import Small_LLM_Model
from typing import Dict
import json


class Vocabulary(BaseModel):
    """Maps tokens to IDs and back, loaded once from the model's vocab file."""

    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    @classmethod
    def from_model(cls, model: Small_LLM_Model) -> 'Vocabulary':
        """Read the vocab JSON from the model's tokenizer and build maps."""
        vocab_path = model.get_path_to_vocab_file()
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                token_to_id = json.load(f)
            id_to_token = {v: k for k, v in token_to_id.items()}
            return cls(token_to_id=token_to_id, id_to_token=id_to_token)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Vocab file not found: {vocab_path}"
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in vocab file: {vocab_path}"
            ) from e
