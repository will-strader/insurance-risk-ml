"""
Load the Kaggle Allstate Claims Severity CSV,
apply basic cleaning / type-casting, and save as compressed Parquet.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import argparse
from utils.logger import get_logger

logger = get_logger(__name__)


def _cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast all object columns to category via str; cont* → float32."""
    obj_cols  = df.select_dtypes("object").columns
    cont_cols = [c for c in df.columns if c.startswith("cont")]

    df[obj_cols]  = df[obj_cols].astype(str).astype("category")
    df[cont_cols] = df[cont_cols].astype("float32")
    return df


def load_raw(csv_path: Path) -> pd.DataFrame:
    logger.info("Loading raw data from %s …", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning: type-cast and clip extreme outliers."""
    df = _cast_columns(df)

    # Drop extreme loss outliers (> 99.5th percentile) to stabilise training
    if "loss" in df.columns:
        thresh = df["loss"].quantile(0.995)
        n_drop = (df["loss"] > thresh).sum()
        if n_drop:
            logger.info("Dropping %d extreme rows (loss > %.0f)", n_drop, thresh)
            df = df[df["loss"] <= thresh]

    df.reset_index(drop=True, inplace=True)
    return df


def save_processed(df: pd.DataFrame, out_path: Path, sample_rows: int | None = 10) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_path, index=False)
    logger.info("Saved processed file → %s (rows=%d)", out_path, len(df))

    # Optional small sample for quick tests
    if sample_rows:
        sample_path = out_path.with_name(out_path.stem + "_sample.parquet")
        df.head(sample_rows).to_parquet(sample_path, index=False)
        logger.info("Saved %d-row sample → %s", sample_rows, sample_path)


# CLI entry-point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", help="YAML config path")
    args = parser.parse_args()

    cfg        = yaml.safe_load(Path(args.config).read_text())
    raw_path   = Path(cfg["data"]["raw"])
    proc_path  = Path(cfg["data"]["processed"])

    df_raw     = load_raw(raw_path)
    df_cleaned = clean_df(df_raw)
    save_processed(df_cleaned, proc_path)