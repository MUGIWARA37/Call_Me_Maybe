from pydantic import BaseModel, ValidationError
from typing import Any, List, Dict, cast
import json
from .models import FunctionDefinition, Prompt


class JsonParser(BaseModel):
    """Reads a JSON file and returns validated model objects."""

    filepath: str

    def read_json_file(self) -> List[Any] | Dict[str, Any]:
        """Open the file and return the raw parsed JSON."""
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return cast(List[Any] | Dict[str, Any], json.load(f))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"File not found: {self.filepath}"
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read file: {self.filepath}"
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in file: {self.filepath}"
            ) from e

    def load_functions(self) -> list[FunctionDefinition]:
        """Load function definitions."""
        try:
            data = self.read_json_file()
            if not isinstance(data, list):
                raise ValueError(
                    f"Expected a JSON array in: {self.filepath}"
                )
            return [FunctionDefinition.model_validate(d) for d in data]
        except ValidationError as e:
            raise ValueError(f"Invalid function definition: {e}") from e

    def load_prompts(self) -> List[Prompt]:
        """Load user prompts from the JSON file."""
        try:
            data = self.read_json_file()
            return [Prompt.model_validate(p) for p in data]
        except ValidationError as e:
            raise ValueError(f"Invalid prompt structure: {e}") from e
