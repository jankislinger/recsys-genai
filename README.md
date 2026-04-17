# Recommender Systems in the Age of Generative AI

**ML Prague 2026 Workshop** | May 4, 2026, 14:00 – 17:30

## Overview

This workshop explores modern recommender systems enhanced with generative AI techniques.
All examples are provided in Jupyter notebooks for hands-on learning.

## Setup

To prepare for the workshop, there are three main components to download in advance:
1. **Python libraries** - Dependencies for the workshop code
2. **Data** - MovieLens datasets for experiments
3. **LLMs** - Pre-trained models from Hugging Face

### Local Machine Setup

1. **Install uv** (if not already installed):
   ```shell
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

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

## Pre-caching Models

### Setup Hugging Face Access

1. **Create a Hugging Face account** (if you don't have one already):
   - Visit [https://huggingface.co/join](https://huggingface.co/join)

2. **Generate a read-only access token**:
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Select "Read" access type
   - Give it a name (e.g., "recsys-genai-workshop")
   - Copy the token

3. **Agree to model terms**:
   - Some models require accepting terms of use
   - When prompted by the CLI, open the respective model page and agree to the terms
   - The CLI will provide the URL if needed

### Cache Models

Cache all models:
```shell
HF_TOKEN=<your access token> uv run recsys-genai cache-models
```

Models that will be cached:
- **google/gemma-3-4b-pt**: 4B parameter pre-trained model (~8 GB)
- **google/gemma-3-12b-pt**: 12B parameter pre-trained model (~24 GB)
- **mistralai/Ministral-8B-Instruct-2410**: 8B parameter instruction-tuned model (~16 GB)
- **nomic-ai/nomic-embed-text-v2-moe**: Multilingual MoE text embedding model (~2 GB)
- **Qwen/Qwen3-Embedding-0.6B**: Smaller efficient embedding model (~1.2 GB)
