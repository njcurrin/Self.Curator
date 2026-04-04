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
import os
import sys
import time
import traceback
from pathlib import Path

from loguru import logger

from nemo_curator.backends.ray_data.executor import RayDataExecutor

from nemo_curator.stages.text.filters.heuristic.string import (
    WordCountFilter,
    NonAlphaNumericFilter,
    UrlsFilter,
    NumbersFilter,
    WhiteSpaceFilter,
    BulletsFilter,
    LongWordFilter,
    BoilerPlateStringFilter,
    SymbolsToWordsFilter,
    ParenthesesFilter,
    MeanWordLengthFilter,
    PunctuationFilter,
    EllipsisFilter,
    CommonEnglishWordsFilter,
    WordsWithoutAlphabetsFilter,
    PornographicUrlsFilter,
    SubstringFilter,
)
from nemo_curator.stages.text.filters.heuristic.repetition.repetition import (
    RepeatedLinesFilter,
    RepeatedParagraphsFilter,
    RepeatedLinesByCharFilter,
    RepeatedParagraphsByCharFilter,
    RepeatingTopNGramsFilter,
    RepeatingDuplicateNGramsFilter,
)
from nemo_curator.stages.text.filters.heuristic.code.code import (
    PythonCommentToCodeFilter,
    GeneralCommentToCodeFilter,
    NumberOfLinesOfCodeFilter,
    XMLHeaderFilter,
    AlphaFilter,
    HTMLBoilerplateFilter,
    TokenizerFertilityFilter,
    PerExtensionFilter,
)
from nemo_curator.stages.text.filters.fasttext.fasttext_filters import (
    FastTextQualityFilter,
    FastTextLangId,
)
from nemo_curator.stages.text.modifiers.fasttext.fasttext_label import FastTextLabelModifier
from nemo_curator.stages.text.filters.histogram.histogram import HistogramFilter
from nemo_curator.stages.text.filters.token.token_count import TokenCountFilter
from nemo_curator.stages.text.filters.score_filter import ScoreFilter

from nemo_curator.stages.text.modifiers.string.c4 import BoilerPlateStringModifier
from nemo_curator.stages.text.modifiers.string.quotation_remover import QuotationRemover
from nemo_curator.stages.text.modifiers.string.markdown_remover import MarkdownRemover
from nemo_curator.stages.text.modifiers.string.newline_normalizer import NewlineNormalizer
from nemo_curator.stages.text.modifiers.string.slicer import Slicer
from nemo_curator.stages.text.modifiers.string.line_remover import LineRemover
from nemo_curator.stages.text.modifiers.string.url_remover import UrlRemover
from nemo_curator.stages.text.modifiers.unicode.unicode_reformatter import UnicodeReformatter
from nemo_curator.stages.text.modifiers.modifier import Modify
from nemo_curator.stages.text.classifiers.quality import QualityClassifier
from nemo_curator.stages.text.classifiers.domain import DomainClassifier, MultilingualDomainClassifier
from nemo_curator.stages.text.classifiers.content_type import ContentTypeClassifier
from nemo_curator.stages.text.classifiers.fineweb_edu import (
    FineWebEduClassifier,
    FineWebMixtralEduClassifier,
    FineWebNemotronEduClassifier,
)
from nemo_curator.stages.text.classifiers.prompt_task_complexity import PromptTaskComplexityClassifier

_FILTER_CLASS_REGISTRY = {
    'WordCountFilter': WordCountFilter,
    'NonAlphaNumericFilter': NonAlphaNumericFilter,
    'UrlsFilter': UrlsFilter,
    'NumbersFilter': NumbersFilter,
    'WhiteSpaceFilter': WhiteSpaceFilter,
    'BulletsFilter': BulletsFilter,
    'LongWordFilter': LongWordFilter,
    'BoilerPlateStringFilter': BoilerPlateStringFilter,
    'SymbolsToWordsFilter': SymbolsToWordsFilter,
    'ParenthesesFilter': ParenthesesFilter,
    'MeanWordLengthFilter': MeanWordLengthFilter,
    'PunctuationFilter': PunctuationFilter,
    'EllipsisFilter': EllipsisFilter,
    'CommonEnglishWordsFilter': CommonEnglishWordsFilter,
    'WordsWithoutAlphabetsFilter': WordsWithoutAlphabetsFilter,
    'PornographicUrlsFilter': PornographicUrlsFilter,
    'SubstringFilter': SubstringFilter,
    'RepeatedLinesFilter': RepeatedLinesFilter,
    'RepeatedParagraphsFilter': RepeatedParagraphsFilter,
    'RepeatedLinesByCharFilter': RepeatedLinesByCharFilter,
    'RepeatedParagraphsByCharFilter': RepeatedParagraphsByCharFilter,
    'RepeatingTopNGramsFilter': RepeatingTopNGramsFilter,
    'RepeatingDuplicateNGramsFilter': RepeatingDuplicateNGramsFilter,
    # Code quality filters
    'PythonCommentToCodeFilter': PythonCommentToCodeFilter,
    'GeneralCommentToCodeFilter': GeneralCommentToCodeFilter,
    'NumberOfLinesOfCodeFilter': NumberOfLinesOfCodeFilter,
    'XMLHeaderFilter': XMLHeaderFilter,
    'AlphaFilter': AlphaFilter,
    'HTMLBoilerplateFilter': HTMLBoilerplateFilter,
    'TokenizerFertilityFilter': TokenizerFertilityFilter,
    'PerExtensionFilter': PerExtensionFilter,
    # FastText filters
    'FastTextQualityFilter': FastTextQualityFilter,
    'FastTextLangId': FastTextLangId,
    # Language / token filters
    'HistogramFilter': HistogramFilter,
    'TokenCountFilter': TokenCountFilter,
}

_SCORE_FILTER_WRAPPER_KEYS = {'text_field', 'score_field', 'invert'}

_MODIFIER_CLASS_REGISTRY = {
    'BoilerPlateStringModifier': BoilerPlateStringModifier,
    'QuotationRemover': QuotationRemover,
    'MarkdownRemover': MarkdownRemover,
    'NewlineNormalizer': NewlineNormalizer,
    'Slicer': Slicer,
    'LineRemover': LineRemover,
    'UrlRemover': UrlRemover,
    'UnicodeReformatter': UnicodeReformatter,
    'FastTextLabelModifier': FastTextLabelModifier,
}

_CLASSIFIER_CLASS_REGISTRY = {
    'QualityClassifier': QualityClassifier,
    'DomainClassifier': DomainClassifier,
    'MultilingualDomainClassifier': MultilingualDomainClassifier,
    'ContentTypeClassifier': ContentTypeClassifier,
    'FineWebEduClassifier': FineWebEduClassifier,
    'FineWebMixtralEduClassifier': FineWebMixtralEduClassifier,
    'FineWebNemotronEduClassifier': FineWebNemotronEduClassifier,
    'PromptTaskComplexityClassifier': PromptTaskComplexityClassifier,
}

_DEDUP_TYPES = { 'ExactDedup', 'FuzzyDedup' }


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

        if stage_type in _FILTER_CLASS_REGISTRY:
            filter_cls = _FILTER_CLASS_REGISTRY[stage_type]
            filter_params = {k: v for k, v in stage_params.items() if k not in _SCORE_FILTER_WRAPPER_KEYS and v is not None}
            wrapper_params = {k: stage_params[k] for k in _SCORE_FILTER_WRAPPER_KEYS if k in stage_params}
            if wrapper_params.get('score_field') == '':
                wrapper_params['score_field'] = None
            stage_instance = ScoreFilter(filter_obj=filter_cls(**filter_params), **wrapper_params)
        elif stage_type in _MODIFIER_CLASS_REGISTRY:
            modifier_cls = _MODIFIER_CLASS_REGISTRY[stage_type]
            text_field = stage_params.pop('text_field', 'text')
            stage_instance = Modify(modifier_fn=modifier_cls(**stage_params), input_fields=text_field)
        elif stage_type in _CLASSIFIER_CLASS_REGISTRY:
            classifier_cls = _CLASSIFIER_CLASS_REGISTRY[stage_type]
            if not stage_params.get('filter_by'):
                stage_params['filter_by'] = None
            if stage_params.get('score_field') == '':
                stage_params['score_field'] = None
            stage_instance = classifier_cls(**stage_params)
        else:
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


def _detect_filetype(directory: str) -> str:
    """Infer file format from what's actually in the output directory."""
    import glob
    if glob.glob(os.path.join(directory, "*.jsonl")):
        return "jsonl"
    return "parquet"


def run_exact_dedup(input_path: str, output_path: str, cache_path: str,
                    text_field: str = "text", assign_id: bool = True) -> None:
    """Two-phase exact dedup: ID identification + document removal."""
    import os
    from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
    from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
    from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor

    input_filetype = _detect_filetype(input_path)
    ids_path = os.path.join(cache_path, "exact_ids_to_remove")
    os.makedirs(ids_path, exist_ok=True)

    executor = RayActorPoolExecutor()

    # Phase A: identify duplicates → write IDs to ids_path
    id_workflow = ExactDeduplicationWorkflow(
        output_path=ids_path,
        input_path=input_path,
        input_filetype=input_filetype,
        text_field=text_field,
        assign_id=assign_id,
        perform_removal=False,
    )
    id_result = id_workflow.run(executor=executor)
    id_generator_path = id_result.metadata.get("id_generator_path") if assign_id else None

    # Phase B: remove duplicates → write clean data to output_path
    removal_workflow = TextDuplicatesRemovalWorkflow(
        input_path=input_path,
        ids_to_remove_path=ids_path,
        output_path=output_path,
        input_filetype=input_filetype,
        output_filetype=input_filetype,
        id_generator_path=id_generator_path,
    )
    removal_workflow.run(executor=executor)
    logger.info(f"ExactDedup complete → {output_path}")


def run_fuzzy_dedup(input_path: str, output_path: str, cache_path: str,
                    text_field: str = "text", assign_id: bool = True,
                    char_ngrams: int = 24, num_bands: int = 20,
                    minhashes_per_band: int = 13, use_64_bit_hash: bool = False,
                    seed: int = 42) -> None:
    """Multi-phase fuzzy (MinHash/LSH) dedup: minhash → LSH → connected components → removal."""
    import os
    from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
    from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
    from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor

    input_filetype = _detect_filetype(input_path)
    fuzzy_cache = os.path.join(cache_path, "fuzzy_cache")
    ids_path = os.path.join(cache_path, "fuzzy_ids_to_remove")
    os.makedirs(fuzzy_cache, exist_ok=True)
    os.makedirs(ids_path, exist_ok=True)

    executor = RayActorPoolExecutor()

    # Phase A: MinHash → LSH → connected components → identify duplicate IDs
    fuzzy_workflow = FuzzyDeduplicationWorkflow(
        cache_path=fuzzy_cache,
        output_path=ids_path,
        input_path=input_path,
        input_filetype=input_filetype,
        text_field=text_field,
        perform_removal=False,
        char_ngrams=char_ngrams,
        num_bands=num_bands,
        minhashes_per_band=minhashes_per_band,
        use_64_bit_hash=use_64_bit_hash,
        seed=seed,
    )
    fuzzy_result = fuzzy_workflow.run(executor=executor)
    id_generator_path = fuzzy_result.metadata.get("id_generator_path")

    # Phase B: remove near-duplicates
    removal_workflow = TextDuplicatesRemovalWorkflow(
        input_path=input_path,
        ids_to_remove_path=ids_path,
        output_path=output_path,
        input_filetype=input_filetype,
        output_filetype=input_filetype,
        id_generator_path=id_generator_path,
    )
    removal_workflow.run(executor=executor)
    logger.info(f"FuzzyDedup complete → {output_path}")


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
    # Split dedup stages (whole-dataset, run post-streaming) from streaming stages
        all_stages = config.get("stages", [])
        dedup_stages = [s for s in all_stages if s["type"] in _DEDUP_TYPES]
        stream_stages = [s for s in all_stages if s["type"] not in _DEDUP_TYPES]

        if dedup_stages:
            # Two-phase: stream → intermediate, then dedup → final output
            intermediate_path = config["output_path"] + "_pre_dedup"
            stream_input = intermediate_path if stream_stages else config["input_path"]

            if stream_stages:
                stream_config = {**config, "output_path": intermediate_path, "stages": stream_stages}
                pipeline = build_pipeline(stream_config)
                logger.info(f"Pipeline built: {pipeline}")
                executor = RayDataExecutor()
                pipeline.run(executor=executor)

            # Run the dedup workflow
            dedup_cfg = dedup_stages[0]
            params = dict(dedup_cfg.get("params", {}))
            text_field = params.pop("text_field", config.get("text_field", "text"))
            cache_path = config["output_path"] + "_dedup_cache"
            os.makedirs(cache_path, exist_ok=True)

            if dedup_cfg["type"] == "ExactDedup":
                run_exact_dedup(
                    input_path=stream_input,
                    output_path=config["output_path"],
                    cache_path=cache_path,
                    text_field=text_field,
                    **params,
                )
            elif dedup_cfg["type"] == "FuzzyDedup":
                run_fuzzy_dedup(
                    input_path=stream_input,
                    output_path=config["output_path"],
                    cache_path=cache_path,
                    text_field=text_field,
                    **params,
                )

            # Clean up intermediate
            if stream_stages and os.path.exists(intermediate_path):
                import shutil
                shutil.rmtree(intermediate_path)

        else:
            # Normal streaming-only pipeline (no dedup)
            pipeline = build_pipeline(config)
            logger.info(f"Pipeline built: {pipeline}")
            executor = RayDataExecutor()
            result = pipeline.run(executor=executor)

    except Exception:
        elapsed = time.time() - start_time
        logger.error(f"Pipeline failed after {elapsed:.1f}s")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
