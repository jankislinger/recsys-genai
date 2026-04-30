"""Utilities for interacting with LLMs via Ollama.

This module provides simple wrappers for calling Ollama models
for text generation and embeddings.
"""

import json
import os
from typing import Any

import ollama

_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_client = ollama.Client(host=_OLLAMA_HOST)


def retry(times: int, exceptions: tuple[type[BaseException]], *, verbose: bool = False):
    def decorator(func):
        def new_fn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if verbose:
                        print(
                            "Exception thrown when attempting to run %s, attempt "
                            "%d of %d" % (func, attempt, times)
                        )
                    attempt += 1
            return func(*args, **kwargs)

        return new_fn

    return decorator


def ollama_generate(
    prompt: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> str:
    """Generate text using Ollama.

    Args:
        prompt: The input prompt for the model
        model: Name of the Ollama model to use (required)
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum number of tokens to generate

    Returns:
        Generated text from the model

    Example:
        >>> response = ollama_generate("What is 2+2?", model="ministral-3:3b")
        >>> isinstance(response, str)
        True

    Environment Variables:
        OLLAMA_HOST: Base URL for Ollama API (default: http://localhost:11434)
    """
    response = _client.generate(
        model=model,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    )
    return response["response"]


@retry(times=3, exceptions=(json.JSONDecodeError,))
def ollama_generate_json(
    prompt: str,
    model: str,
    temperature: float = 0.3,
) -> dict[str, Any]:
    """Generate JSON-formatted output using Ollama.

    Args:
        prompt: The input prompt for the model (should request JSON output)
        model: Name of the Ollama model to use (required)
        temperature: Sampling temperature (lower for more structured output)

    Returns:
        Parsed JSON object as a dictionary

    Example:
        >>> prompt = "Generate JSON: {'name': 'example', 'value': 42}"
        >>> result = ollama_generate_json(prompt, model="ministral-3:3b")
        >>> isinstance(result, dict)
        True

    Environment Variables:
        OLLAMA_HOST: Base URL for Ollama API (default: http://localhost:11434)
    """
    response_text = ollama_generate(prompt, model=model, temperature=temperature, max_tokens=500)

    # Try to extract JSON from the response
    # Sometimes models wrap JSON in markdown code blocks
    text = response_text.strip()

    # Remove markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    return json.loads(text)


def ollama_embed(
    text: str | list[str],
    model: str,
) -> list[float] | list[list[float]]:
    """Generate embeddings using Ollama.

    Args:
        text: Single text string or list of text strings to embed
        model: Name of the Ollama embedding model to use (required)

    Returns:
        Embedding vector (single text) or list of embedding vectors (multiple texts)

    Example:
        >>> embedding = ollama_embed("Hello world", model="nomic-embed-text-v2-moe")
        >>> isinstance(embedding, list)
        True
        >>> isinstance(embedding[0], float)
        True

    Environment Variables:
        OLLAMA_HOST: Base URL for Ollama API (default: http://localhost:11434)
    """
    is_single = isinstance(text, str)
    texts = [text] if is_single else text

    response = _client.embed(model=model, input=texts)

    if is_single:
        return response["embeddings"][0]
    return response["embeddings"]


def check_ollama_available(model: str) -> bool:
    """Check if Ollama is running and the model is available.

    Args:
        model: Name of the model to check (required)

    Returns:
        True if Ollama is running and model is available, False otherwise

    Environment Variables:
        OLLAMA_HOST: Base URL for Ollama API (default: http://localhost:11434)
    """
    models_list = _client.list()
    for m in models_list.get("models", []):
        print(m)
        if model == m["model"] or model == m["model"].split(":")[0]:
            return True
    return False
