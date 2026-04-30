import inspect
import json

from IPython.display import Markdown


def show_source(obj):
    block = f"```python\n{inspect.getsource(obj)}\n```"
    return Markdown(block)


def show_prompt(prompt: str):
    md_lines = [
        "**Prompt:**",
        "```{.prompt}",
        prompt,
        "```",
    ]
    return Markdown("\n".join(md_lines))


def show_response(response: str | dict | list):
    if not isinstance(response, str):
        response = json.dumps(response, indent=2)

    md_lines = [
        "**LLM Response:**",
        "```{.response}",
        response,
        "```",
    ]
    return Markdown("\n".join(md_lines))


def ollama_model_link(model_name: str) -> str:
    """Generate a Markdown link to the Ollama model page.

    Args:
        model_name: Name of the Ollama model (e.g., "ministral-3:3b" or "ministral-3")

    Returns:
        Markdown formatted link with emoji

    Examples:
        >>> ollama_model_link("ministral-3:3b")
        '[ministral-3:3b 🔗](https://ollama.com/library/ministral-3)'
        >>> ollama_model_link("nomic-embed-text-v2-moe")
        '[nomic-embed-text-v2-moe 🔗](https://ollama.com/library/nomic-embed-text-v2-moe)'
    """
    base_model = model_name.split(":")[0]
    url = f"https://ollama.com/library/{base_model}"
    return f"[{model_name} 🔗]({url})"


def tmdb_images(paths: list[str]):
    base_url = "https://media.themoviedb.org/t/p/w154"
    lines = [f"![]({base_url}{path})" for path in paths]
    return Markdown("\n".join(lines))
