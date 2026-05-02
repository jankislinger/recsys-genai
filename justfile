_default:
    @just --list

# Start Jupyter Lab server
lab:
    uv run --env-file .env jupyter lab \
        --notebook-dir=notebooks

# Serve rendered website from _side/docs
serve port='8000':
    uv run -m http.server {{port}} -d _site/docs

# Start interactive preview using Quarto
preview:
    uv run --env-file .env quarto preview

# Render single file using Quarto
render file='index.qmd':
    uv run --env-file .env quarto render {{file}}

# Render entire website
render-all:
    uv run --env-file .env quarto render

# Render fast files (no heavy execution cells)
render-fast:
    @just render index.qmd
    @just render references.qmd
    @just render slides

# Format & lint python code
format:
    uv run ruff format
    uv run ruff check --fix --unsafe-fixes

# Run unit tests
test:
    uv run pytest
