"""Tests for LLM utilities module."""

import pytest

from recsys_genai.llm_utils import (
    check_ollama_available,
    ollama_embed,
    ollama_generate,
    ollama_generate_json,
)


def test_check_ollama_available():
    """Test checking if Ollama is available."""
    # This test just checks that the function runs without error
    # The result depends on whether Ollama is actually running
    result = check_ollama_available("ministral-3:3b")
    assert isinstance(result, bool)


@pytest.mark.skipif(
    not check_ollama_available("ministral-3:3b"),
    reason="Ollama not available or model not pulled",
)
def test_ollama_generate():
    """Test text generation with Ollama."""
    response = ollama_generate(
        "What is 2+2? Answer with just the number.",
        model="ministral-3:3b",
        temperature=0.0,
        max_tokens=10,
    )

    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.skipif(
    not check_ollama_available("ministral-3:3b"),
    reason="Ollama not available or model not pulled",
)
def test_ollama_generate_json():
    """Test JSON generation with Ollama."""
    prompt = """Generate a JSON object with these fields:
- name: "example"
- value: 42

Output ONLY the JSON object, nothing else."""

    result = ollama_generate_json(prompt, model="ministral-3:3b", temperature=0.0)

    assert isinstance(result, dict)
    # The exact keys may vary, but we should get a dictionary


@pytest.mark.skipif(
    not check_ollama_available("nomic-embed-text-v2-moe"),
    reason="Ollama not available or embedding model not pulled",
)
def test_ollama_embed_single():
    """Test embedding a single text."""
    embedding = ollama_embed("Hello world", model="nomic-embed-text-v2-moe")

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert isinstance(embedding[0], float)


@pytest.mark.skipif(
    not check_ollama_available("nomic-embed-text-v2-moe"),
    reason="Ollama not available or embedding model not pulled",
)
def test_ollama_embed_multiple():
    """Test embedding multiple texts."""
    texts = ["Hello world", "Goodbye world", "Testing embeddings"]
    embeddings = ollama_embed(texts, model="nomic-embed-text-v2-moe")

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) > 0 for emb in embeddings)
    assert all(isinstance(emb[0], float) for emb in embeddings)
