#!/usr/bin/env python3
"""
Pipeline execution script — spawned as a subprocess by the API.

Usage:
    python run_pipeline.py <config_path>

The config JSON has the shape:
{
    "name": "my-pipeline",
    "input_path": "/workspace/curator/data/input.jsonl",
    "output_path": "/workspace/curator/data/output.jsonl",
    "text_field": "text",
    "stages": [
        {"type": "ScoreFilter", "params": {...}},
        ...
    ]
}
"""

import json
import sys
import time
import traceback
from pathlib import Path

from loguru import logger


def build_pipeline(config: dict):
    """Construct a NeMo Curator Pipeline from config."""
    from nemo_curator.pipeline.pipeline import Pipeline
    from nemo_curator.stages.base import _STAGE_REGISTRY

    # Trigger text stage registration
    from api.stage_registry import _load_text_stages
    _load_text_stages()

    pipeline = Pipeline(
        name=config.get("name", "curation-job"),
        description=config.get("description"),
    )

    # Build IO stages
    input_path = config["input_path"]
    output_path = config["output_path"]
    text_field = config.get("text_field", "text")

    # Add reader stage
    if input_path.endswith(".jsonl"):
        from nemo_curator.stages.text.io.reader.jsonl import JsonlReaderStage
        from nemo_curator.stages.file_partitioning import FilePartitioningStage

        pipeline.add_stage(FilePartitioningStage(
            file_paths=input_path,
        ))
        pipeline.add_stage(JsonlReaderStage(
            fields=[text_field] if text_field else None,
        ))
    elif input_path.endswith(".parquet"):
        from nemo_curator.stages.text.io.reader.parquet import ParquetReaderStage
        from nemo_curator.stages.file_partitioning import FilePartitioningStage

        pipeline.add_stage(FilePartitioningStage(
            input_path=input_path,
            filetype="parquet",
        ))
        pipeline.add_stage(ParquetReaderStage(
            fields=[text_field] if text_field else None,
        ))
    else:
        raise ValueError(f"Unsupported input format: {input_path}")

    # Add user-configured processing stages
    for stage_config in config.get("stages", []):
        stage_type = stage_config["type"]
        stage_params = stage_config.get("params", {})

        cls = _STAGE_REGISTRY.get(stage_type)
        if cls is None:
            raise ValueError(f"Unknown stage type: {stage_type}")

        try:
            stage_instance = cls(**stage_params)
        except TypeError as e:
            raise ValueError(f"Failed to instantiate {stage_type}: {e}") from e

        pipeline.add_stage(stage_instance)

    # Add writer stage
    import os
    output_format = config.get("output_format", "jsonl")
    os.makedirs(output_path, exist_ok=True)

    if output_format == "jsonl":
        from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
        pipeline.add_stage(JsonlWriter(path=output_path))
    elif output_format == "parquet":
        from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
        pipeline.add_stage(ParquetWriter(path=output_path))
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


    return pipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: run_pipeline.py <config_path>", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    config = json.loads(config_path.read_text())

    logger.info(f"Starting curation pipeline: {config.get('name', 'unnamed')}")
    logger.info(f"Input: {config.get('input_path')}")
    logger.info(f"Output: {config.get('output_path')}")
    logger.info(f"Stages: {len(config.get('stages', []))}")

    start_time = time.time()

    try:
        pipeline = build_pipeline(config)
        logger.info(f"Pipeline built: {pipeline}")

        # Use RayDataExecutor for local execution
        from nemo_curator.backends.ray_data.executor import RayDataExecutor

        executor = RayDataExecutor()
        result = pipeline.run(executor=executor)

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed:.1f}s")
        if result:
            logger.info(f"Output tasks: {len(result)}")

    except Exception:
        elapsed = time.time() - start_time
        logger.error(f"Pipeline failed after {elapsed:.1f}s")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
