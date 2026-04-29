# Recommender Systems in the Age of Generative AI

**ML Prague 2026 Workshop** | May 4, 2026, 14:00 – 17:30

## Overview

This workshop explores modern recommender systems enhanced with generative AI techniques.
All examples are provided in Jupyter notebooks for hands-on learning.

## Setup

To prepare for the workshop, there are three main components to download in advance:
1. **Python libraries** - Dependencies for the workshop code
2. **Data** - MovieLens datasets for experiments
3. **LLMs** - Language models via Ollama for local inference

### Local Machine Setup

1. **Install uv** (if not already installed):
   ```shell
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   For other installation methods (e.g., Windows, Homebrew, Cargo), see https://docs.astral.sh/uv/getting-started/installation/

2. **Clone the repository and install dependencies**:
   ```shell
   uv python install 3.14
   uv sync
   ```

## Dataset

We will use the MovieLens dataset for the experiments.

- **Latest**: Contains approximately 33M ratings applied to 86K movies by 330K users
- **Small**: A smaller dataset useful if experiments are too large for your machine

### Download and Prepare Data

Download both datasets:
```shell
uv run recsys-genai download-data
```

Prepare the dataset for use (convert CSV files to Parquet format):
```shell
# For the latest (large) dataset
uv run recsys-genai prepare-data --dataset latest

# For the small dataset
uv run recsys-genai prepare-data --dataset small
```

The prepared Parquet files will be saved in the `data/` directory and include:
- `ratings.parquet` - User ratings for movies
- `movies.parquet` - Movie metadata
- `links.parquet` - Links to external movie databases
- `tags.parquet` - User-generated movie tags

## Language Models Setup

### Install Ollama

Ollama provides CPU-optimized local inference with automatic quantization, perfect for running models on laptops and workstations without GPU requirements.

1. **Install Ollama**:

   **macOS/Linux**:
   ```shell
   curl -fsSL https://ollama.com/install.sh | sh
   ```

   **Windows**:
   - Download from [https://ollama.com/download](https://ollama.com/download)

   **Alternative (manual)**:
   - Visit [https://ollama.com/download](https://ollama.com/download) for other installation methods

2. **Verify installation**:
   ```shell
   ollama --version
   ```

### Pull Models

Download the models needed for the workshop.
These are CPU-optimized and will run efficiently on consumer laptops.

```shell
# Embedding models
ollama pull nomic-embed-text-v2-moe
ollama pull qwen3-embedding:0.6b

# Text generation models
ollama pull ministral-3:3b
ollama pull ministral-3:8b
ollama pull gemma3:4b
ollama pull gpt-oss:20b
```

### Running the Models in Cloud (Optional)

**This section is optional.**
All workshop examples will work with the small local models listed above.

Cloud models enable running larger models without downloading them locally.
Ollama provides cloud-hosted variants that run inference on Ollama's servers.
This is useful if you want to experiment with larger models but have limited local storage or RAM.

**Setup requirements:**

1. **Sign up for an Ollama account**:
   - Visit [ollama.com](https://ollama.com) and create an account

2. **Authenticate via CLI**:
   ```shell
   ollama login
   ```
   - Enter your credentials when prompted

3. **Pull cloud models**:
   ```shell
   ollama pull gpt-oss:20b-cloud
   ollama pull gpt-oss:120b-cloud
   ```
