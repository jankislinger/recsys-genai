import inspect

from IPython.display import Markdown


def show_source(obj):
    block = f"```python\n{inspect.getsource(obj)}\n```"
    return Markdown(block)
