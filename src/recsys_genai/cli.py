"""CLI for downloading and preparing MovieLens datasets."""

import re
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import click
import polars as pl

DATASETS = {
    "small": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "name": "ml-latest-small",
    },
    "latest": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
        "name": "ml-latest",
    },
}

# CSV files to process
FILES_TO_PROCESS = ["ratings", "movies", "links", "tags"]


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    # Insert underscore before uppercase letters and convert to lowercase
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


@click.group()
def cli():
    """MovieLens dataset management tool."""
    pass


@cli.command("download-data")
@click.option(
    "--dataset",
    type=click.Choice(["small", "latest", "both"]),
    default="both",
    help="Which dataset to download (default: both)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="data/raw",
    help="Output directory for downloaded datasets (default: data/raw)",
)
def download_data(dataset: str, output_dir: Path):
    """Download MovieLens datasets.

    Downloads the specified MovieLens dataset(s) from GroupLens.
    It's recommended to download both datasets.

    Examples:
        recsys-genai download-data                    # downloads both
        recsys-genai download-data --dataset both     # downloads both
        recsys-genai download-data --dataset small
        recsys-genai download-data --dataset latest
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_download = list(DATASETS.keys()) if dataset == "both" else [dataset]

    for ds_name in datasets_to_download:
        ds_config = DATASETS[ds_name]
        extract_dir = output_dir / ds_config["name"]

        # Check if already extracted
        if extract_dir.exists():
            click.echo(f"✓ Dataset {ds_name} already exists at {extract_dir}, skipping download")
            continue

        zip_path = output_dir / f"{ds_config['name']}.zip"

        click.echo(f"Downloading {ds_name} dataset from {ds_config['url']}...")

        try:
            urlretrieve(ds_config["url"], zip_path)
            click.echo(f"✓ Downloaded to {zip_path}")

            # Extract the zip file
            click.echo(f"Extracting to {extract_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
            click.echo(f"✓ Extracted to {extract_dir}")

            # Clean up zip file
            zip_path.unlink()
            click.echo(f"✓ Cleaned up {zip_path}")

        except Exception as e:
            click.echo(f"✗ Error downloading {ds_name}: {e}", err=True)
            raise


@cli.command("prepare-data")
@click.option(
    "--dataset",
    type=click.Choice(["small", "latest"]),
    required=True,
    help="Which dataset to prepare (user must choose one)",
)
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path),
    default="data/raw",
    help="Input directory with extracted datasets (default: data/raw)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="data",
    help="Output directory for parquet files (default: data)",
)
def prepare_data(dataset: str, input_dir: Path, output_dir: Path):
    """Prepare datasets into parquet format.

    Converts all CSV files to parquet format with unified snake_case schema.
    Processes common files (ratings, movies, links, tags) for both datasets.

    Examples:
        recsys-genai prepare-data --dataset small
        recsys-genai prepare-data --dataset latest
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_config = DATASETS[dataset]
    dataset_dir = input_dir / ds_config["name"]

    if not dataset_dir.exists():
        click.echo(
            f"✗ Dataset {dataset} not found at {dataset_dir}. Run 'download' first.",
            err=True,
        )
        return

    click.echo(f"Processing {dataset} dataset from {dataset_dir}...")
    click.echo(f"Files to process: {', '.join(FILES_TO_PROCESS)}")

    total_processed = 0
    for file_name in FILES_TO_PROCESS:
        csv_path = dataset_dir / f"{file_name}.csv"
        # Use consistent filename without dataset prefix
        parquet_path = output_dir / f"{file_name}.parquet"

        if not csv_path.exists():
            click.echo(f"  ⚠ Skipping {file_name}.csv (not found)")
            continue

        try:
            # Read CSV with polars
            df = pl.read_csv(csv_path)

            # Convert column names from camelCase to snake_case
            df = df.rename({col: camel_to_snake(col) for col in df.columns})

            # Show basic stats
            n_rows = df.height
            n_cols = len(df.columns)
            click.echo(f"  {file_name}.csv: {n_rows:,} rows, {n_cols} columns")
            click.echo(f"    Columns: {', '.join(df.columns)}")

            # Write to parquet
            df.write_parquet(parquet_path)
            click.echo(f"    ✓ Saved to {parquet_path}")
            total_processed += 1

        except Exception as e:
            click.echo(f"    ✗ Error processing {file_name}: {e}", err=True)
            raise

    click.echo(f"\n✓ Successfully processed {total_processed} files for {dataset} dataset")


if __name__ == "__main__":
    cli()
