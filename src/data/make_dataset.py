"""
Load the Kaggle Allstate Claims Severity CSV,
clean minimal issues, and save as compressed Parquet.
"""

from pathlib import Path
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

def load_raw(csv_path: Path) -> pd.DataFrame:
    logger.info("Loading raw data …")
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df


def save_processed(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert all object columns to string (fixes mixed-type issues)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    df.to_parquet(out_path, index=False)
    logger.info("Saved cleaned file → %s", out_path)


if __name__ == "__main__":
    import yaml, argparse

    print("make_dataset.py is running...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    raw = Path(cfg["data"]["raw"])
    processed = Path(cfg["data"]["processed"])

    df = load_raw(raw)
    # (Add real cleaning here: cast categoricals, handle missing)
    save_processed(df, processed)