from __future__ import annotations

from typing import Iterable

from .models import FunctionDefinition


def _find_by_keywords(functions: Iterable[FunctionDefinition], keywords: Iterable[str]) -> FunctionDefinition | None:
    lowered_keywords = [kw.lower() for kw in keywords]
    for function in functions:
        haystack = f"{function.name} {function.description}".lower()
        if any(keyword in haystack for keyword in lowered_keywords):
            return function
    return None


def _best_match(prompt: str, functions: Iterable[FunctionDefinition]) -> FunctionDefinition | None:
    prompt_terms = prompt.lower().replace('"', " ").replace("'", " ").split()
    best_function = None
    best_score = 0
    for function in functions:
        haystack = f"{function.name} {function.description}".lower()
        score = sum(1 for term in prompt_terms if term in haystack)
        if score > best_score:
            best_score = score
            best_function = function
    if best_score > 0:
        return best_function
    return None


def select_function(prompt: str, functions: list[FunctionDefinition]) -> FunctionDefinition:
    if not functions:
        raise ValueError("No functions available for selection")

    prompt_lowered = prompt.lower()
    rules = [
        (["square root", "sqrt"], ["square_root", "square root"]),
        (["reverse"], ["reverse"]),
        (["greet", "hello", "hi"], ["greet"]),
        (["replace", "substitute", "regex"], ["substitute", "regex", "replace"]),
        (["sum", "add", "plus", "total"], ["add", "sum"]),
    ]

    for prompt_keywords, function_keywords in rules:
        if any(keyword in prompt_lowered for keyword in prompt_keywords):
            match = _find_by_keywords(functions, function_keywords)
            if match is not None:
                return match

    best_match = _best_match(prompt, functions)
    if best_match is not None:
        return best_match

    return functions[0]
