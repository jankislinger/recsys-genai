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

# Format a single notebook: convert qmd -> py, run ruff, convert back, fix "# |" -> "#|"
format-notebook file:
    #!/usr/bin/env bash
    set -euo pipefail

    qmd_file="{{file}}"
    py_file="${qmd_file%.qmd}.py"

    # Check for uncommitted changes in this specific file
    if ! git diff --quiet "$qmd_file" || ! git diff --cached --quiet "$qmd_file"; then
        echo "Error: $qmd_file has uncommitted changes. Commit or stash them first."
        exit 1
    fi

    echo "Processing $qmd_file..."
    uv run jupytext --to py:percent "$qmd_file"
    uv run ruff format "$py_file"
    uv run jupytext --to qmd "$py_file"
    rm -f "$py_file"

    # Fix '# |' -> '#|' (removing space after hash)
    sed -i 's/^# |/#|/g' "$qmd_file"

    echo "✓ Formatted $qmd_file"

# Format all notebooks
format-notebooks:
    #!/usr/bin/env bash
    set -euo pipefail

    # Check for uncommitted changes in notebooks directory
    if ! git diff --quiet notebooks/ || ! git diff --cached --quiet notebooks/; then
        echo "Error: notebooks/ has uncommitted changes. Commit or stash them first."
        exit 1
    fi

    for qmd_file in notebooks/*.qmd; do
        just format-notebook "$qmd_file"
    done

    echo "✓ All notebooks formatted successfully!"
