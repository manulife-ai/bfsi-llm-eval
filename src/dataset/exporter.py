"""Export dataset to Parquet and JSONL formats."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class DatasetExporter:
    """Export assembled records to Parquet + JSONL."""

    def __init__(self, output_dir: str = "data/final"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, records: list[dict]) -> tuple[Path, Path]:
        """Export records to both Parquet and JSONL. Returns (parquet_path, jsonl_path)."""
        parquet_path = self._export_parquet(records)
        jsonl_path = self._export_jsonl(records)
        logger.info("Exported %d records to %s and %s", len(records), parquet_path, jsonl_path)
        return parquet_path, jsonl_path

    def _export_parquet(self, records: list[dict]) -> Path:
        """Export to Parquet format."""
        path = self.output_dir / "dataset.parquet"

        # Flatten pass_criteria to JSON string for Parquet compatibility
        flat_records = []
        for r in records:
            flat = dict(r)
            if isinstance(flat.get("pass_criteria"), dict):
                flat["pass_criteria"] = json.dumps(flat["pass_criteria"])
            if isinstance(flat.get("expected_failure_modes"), list):
                flat["expected_failure_modes"] = json.dumps(flat["expected_failure_modes"])
            if isinstance(flat.get("linked_prompt_ids"), list):
                flat["linked_prompt_ids"] = json.dumps(flat["linked_prompt_ids"])
            flat_records.append(flat)

        df = pd.DataFrame(flat_records)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)
        logger.info("Wrote Parquet: %s (%d rows)", path, len(df))
        return path

    def _export_jsonl(self, records: list[dict]) -> Path:
        """Export to JSONL format (one JSON object per line)."""
        path = self.output_dir / "dataset.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote JSONL: %s (%d records)", path, len(records))
        return path

    def verify_parquet(self) -> bool:
        """Verify Parquet file is loadable by HuggingFace datasets."""
        path = self.output_dir / "dataset.parquet"
        if not path.exists():
            logger.error("Parquet file not found: %s", path)
            return False
        try:
            from datasets import load_dataset
            ds = load_dataset("parquet", data_files=str(path))
            logger.info("Parquet verification OK: %d rows", len(ds["train"]))
            return True
        except Exception as e:
            logger.error("Parquet verification failed: %s", e)
            return False
