# modality: text
"""
End-to-end registry tests for built-in pipeline nodes.

Migrated from tests/api/test_pipeline_nodes.py to tests/nodes/ (T-106).
All tests marked fast — no Ray cluster, no disk I/O, no subprocess.

These tests verify that each node type the UI can produce (filter or modifier)
can be instantiated through the same registry path used by run_pipeline.py, and
that the resulting stage transforms data correctly when .process() is called.
"""

import importlib.util

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

from nemo_curator.stages.text.filters.score_filter import ScoreFilter
from nemo_curator.stages.text.modifiers.modifier import Modify
from nemo_curator.tasks import DocumentBatch

from run_pipeline import (
    _FILTER_CLASS_REGISTRY,
    _MODIFIER_CLASS_REGISTRY,
    _SCORE_FILTER_WRAPPER_KEYS,
    _CLASSIFIER_CLASS_REGISTRY,
    _DEDUP_TYPES,
    _detect_filetype,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_batch(texts: list[str], col: str = "text") -> DocumentBatch:
    return DocumentBatch(
        task_id="test",
        dataset_name="test_ds",
        data=pd.DataFrame({col: texts}),
    )


def run_filter(stage_type: str, params: dict) -> DocumentBatch:
    """Instantiate a ScoreFilter via the registry and run it."""
    docs = params.pop("_docs")
    filter_cls = _FILTER_CLASS_REGISTRY[stage_type]
    filter_params = {k: v for k, v in params.items() if k not in _SCORE_FILTER_WRAPPER_KEYS}
    wrapper_params = {k: params[k] for k in _SCORE_FILTER_WRAPPER_KEYS if k in params}
    stage = ScoreFilter(filter_obj=filter_cls(**filter_params), **wrapper_params)
    stage.setup()
    return stage.process(make_batch(docs))


def run_modifier(stage_type: str, params: dict, docs: list[str]) -> DocumentBatch:
    """Instantiate a Modify stage via the registry and run it."""
    modifier_cls = _MODIFIER_CLASS_REGISTRY[stage_type]
    text_field = params.pop("text_field", "text")
    stage = Modify(modifier_fn=modifier_cls(**params), input_fields=text_field)
    stage.setup()
    return stage.process(make_batch(docs))


def kept_texts(batch: DocumentBatch) -> list[str]:
    return batch.data["text"].tolist()


# ---------------------------------------------------------------------------
# Filter node tests
# ---------------------------------------------------------------------------

class TestWordCountFilterNode:
    def test_keeps_docs_in_range(self):
        docs = [
            "one two three four five",           # 5 words — below min, drop
            " ".join(["word"] * 20),             # 20 words — keep
            " ".join(["word"] * 200),            # 200 words — keep
            " ".join(["word"] * 150001),         # over max, drop
        ]
        batch = make_batch(docs)
        stage = ScoreFilter(
            filter_obj=_FILTER_CLASS_REGISTRY["WordCountFilter"](min_words=10, max_words=150000)
        )
        stage.setup()
        result = kept_texts(stage.process(batch))
        assert len(result) == 2
        assert all("word" in t for t in result)

    def test_invert_keeps_short_docs(self):
        docs = ["one two three", " ".join(["word"] * 20)]
        batch = make_batch(docs)
        stage = ScoreFilter(
            filter_obj=_FILTER_CLASS_REGISTRY["WordCountFilter"](min_words=10),
            invert=True,
        )
        stage.setup()
        result = kept_texts(stage.process(batch))
        assert result == ["one two three"]


class TestNonAlphaNumericFilterNode:
    def test_drops_high_symbol_ratio(self):
        docs = [
            "hello world this is normal text",           # low ratio — keep
            "!@#$%^&*() !@#$%^&*() !@#$%^&*()",        # all symbols — drop
        ]
        batch = make_batch(docs)
        stage = ScoreFilter(
            filter_obj=_FILTER_CLASS_REGISTRY["NonAlphaNumericFilter"](max_non_alpha_numeric_to_text_ratio=0.25)
        )
        stage.setup()
        result = kept_texts(stage.process(batch))
        assert result == ["hello world this is normal text"]


class TestUrlsFilterNode:
    def test_drops_url_heavy_docs(self):
        docs = [
            "Visit https://nvidia.com for more info.",
            "https://a.com https://b.com https://c.com https://d.com https://e.com",
        ]
        batch = make_batch(docs)
        stage = ScoreFilter(
            filter_obj=_FILTER_CLASS_REGISTRY["UrlsFilter"](max_url_to_text_ratio=0.5)
        )
        stage.setup()
        result = kept_texts(stage.process(batch))
        assert len(result) == 1
        assert "Visit" in result[0]


class TestNumbersFilterNode:
    def test_drops_number_heavy_docs(self):
        docs = [
            "The answer is 42 and the result is 7.",     # low ratio — keep
            "1234567890 1234567890 1234567890",           # all numbers — drop
        ]
        batch = make_batch(docs)
        stage = ScoreFilter(
            filter_obj=_FILTER_CLASS_REGISTRY["NumbersFilter"](max_number_to_text_ratio=0.15)
        )
        stage.setup()
        result = kept_texts(stage.process(batch))
        assert len(result) == 1
        assert "answer" in result[0]


class TestWhiteSpaceFilterNode:
    def test_drops_whitespace_heavy_docs(self):
        docs = [
            "Normal text without excessive spacing.",
            "a   " * 50,   # mostly whitespace — drop
        ]
        batch = make_batch(docs)
        stage = ScoreFilter(
            filter_obj=_FILTER_CLASS_REGISTRY["WhiteSpaceFilter"](max_white_space_ratio=0.25)
        )
        stage.setup()
        result = kept_texts(stage.process(batch))
        assert len(result) == 1
        assert "Normal" in result[0]


class TestBulletsFilterNode:
    def test_drops_bullet_heavy_docs(self):
        bullet_doc = "\n".join(["• item"] * 20)
        normal_doc = "This is a normal paragraph with actual sentences."
        batch = make_batch([normal_doc, bullet_doc])
        stage = ScoreFilter(
            filter_obj=_FILTER_CLASS_REGISTRY["BulletsFilter"](max_bullet_lines_ratio=0.9)
        )
        stage.setup()
        result = kept_texts(stage.process(batch))
        assert result == [normal_doc]


class TestLongWordFilterNode:
    def test_drops_docs_with_long_words(self):
        docs = [
            "The quick brown fox jumps over the lazy dog.",
            "This contains a verylongwordthatexceedsthethreshold_" + "x" * 50,
        ]
        batch = make_batch(docs)
        stage = ScoreFilter(
            filter_obj=_FILTER_CLASS_REGISTRY["LongWordFilter"](max_word_length=40)
        )
        stage.setup()
        result = kept_texts(stage.process(batch))
        assert len(result) == 1
        assert "quick" in result[0]


class TestBoilerPlateStringFilterNode:
    def test_drops_docs_with_boilerplate(self):
        docs = [
            "This is a perfectly normal document about science.",
            "Please read our terms of use before continuing.",
        ]
        batch = make_batch(docs)
        stage = ScoreFilter(
            filter_obj=_FILTER_CLASS_REGISTRY["BoilerPlateStringFilter"]()
        )
        stage.setup()
        result = kept_texts(stage.process(batch))
        assert result == ["This is a perfectly normal document about science."]


# ---------------------------------------------------------------------------
# Modifier node tests
# ---------------------------------------------------------------------------

class TestBoilerPlateStringModifierNode:
    def test_removes_boilerplate_paragraphs(self):
        # Modifier removes boilerplate only when it's at the top or bottom
        doc = "terms of use\n\nGood content here.\n\nMore good content."
        result = run_modifier("BoilerPlateStringModifier", {}, [doc])
        text = kept_texts(result)[0]
        assert "terms of use" not in text
        assert "Good content" in text


class TestQuotationRemoverNode:
    def test_removes_surrounding_quotes(self):
        docs = ['"Hello, world!"', 'No quotes here']
        result = kept_texts(run_modifier("QuotationRemover", {}, docs))
        assert result[0] == "Hello, world!"
        assert result[1] == "No quotes here"


class TestMarkdownRemoverNode:
    def test_removes_markdown(self):
        docs = ["This is **bold** and *italic* text.", "Plain text."]
        result = kept_texts(run_modifier("MarkdownRemover", {}, docs))
        assert result[0] == "This is bold and italic text."
        assert result[1] == "Plain text."


class TestNewlineNormalizerNode:
    def test_collapses_excess_newlines(self):
        docs = ["line one\n\n\n\nline two", "normal\ntext"]
        result = kept_texts(run_modifier("NewlineNormalizer", {}, docs))
        assert result[0] == "line one\n\nline two"
        assert result[1] == "normal\ntext"


class TestSlicerNode:
    def test_slices_by_character_index(self):
        docs = ["Hello, world!", "0123456789"]
        result = kept_texts(run_modifier("Slicer", {"left": 7, "right": 12}, docs))
        assert result[0] == "world"
        assert result[1] == "789"

    def test_default_strips_whitespace(self):
        docs = ["   trimmed   "]
        result = kept_texts(run_modifier("Slicer", {}, docs))
        assert result[0] == "trimmed"


class TestLineRemoverNode:
    def test_removes_matching_lines(self):
        docs = [
            "Keep this line\nRemove me\nKeep this too",
            "No match here",
        ]
        result = kept_texts(run_modifier("LineRemover", {"patterns": ["Remove me"]}, docs))
        assert result[0] == "Keep this line\nKeep this too"
        assert result[1] == "No match here"


class TestUrlRemoverNode:
    def test_removes_urls(self):
        docs = [
            "Check out https://nvidia.com for details.",
            "No URLs here.",
        ]
        result = kept_texts(run_modifier("UrlRemover", {}, docs))
        assert "https" not in result[0]
        assert result[1] == "No URLs here."


class TestUnicodeReformatterNode:
    def test_fixes_mojibake(self):
        docs = [
            "The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.",
            "Clean text already.",
        ]
        result = kept_texts(run_modifier("UnicodeReformatter", {}, docs))
        # ftfy fixes the mojibake — "doesn't" with a curly apostrophe is the correct output
        assert "doesn" in result[0] and "t have eyebrows" in result[0]
        assert result[1] == "Clean text already."

# ---------------------------------------------------------------------------
# Classifier node tests
# ---------------------------------------------------------------------------

class TestQualityClassifierNode:
    def test_registry_entry_exists(self):
        assert 'QualityClassifier' in _CLASSIFIER_CLASS_REGISTRY

    def test_instantiates_with_defaults(self):
        classifier = _CLASSIFIER_CLASS_REGISTRY['QualityClassifier']()
        assert classifier.label_field == 'quality_pred'
        assert classifier.text_field == 'text'
        assert classifier.score_field is None
        assert classifier.filter_by is None

    def test_custom_params_flow_through(self):
        classifier = _CLASSIFIER_CLASS_REGISTRY['QualityClassifier'](
            label_field='my_label',
            text_field='content',
        )
        assert classifier.label_field == 'my_label'
        assert classifier.text_field == 'content'

    def test_filter_by_empty_list_coercion(self):
        # Mirrors the run_pipeline.py coercion: if not stage_params.get('filter_by') → None
        stage_params = {'filter_by': []}
        if not stage_params.get('filter_by'):
            stage_params['filter_by'] = None
        classifier = _CLASSIFIER_CLASS_REGISTRY['QualityClassifier'](**stage_params)
        assert classifier.filter_by is None
        from nemo_curator.stages.text.filters import Filter
        assert not any(isinstance(s, Filter) for s in classifier.stages)

    def test_score_field_empty_string_coercion(self):
        # Mirrors the run_pipeline.py coercion: score_field='' → None
        stage_params = {'score_field': ''}
        if stage_params.get('score_field') == '':
            stage_params['score_field'] = None
        classifier = _CLASSIFIER_CLASS_REGISTRY['QualityClassifier'](**stage_params)
        assert classifier.score_field is None

    def test_filter_by_with_labels_adds_filter_stage(self):
        from nemo_curator.stages.text.filters import Filter
        classifier = _CLASSIFIER_CLASS_REGISTRY['QualityClassifier'](
            filter_by=['High', 'Medium']
        )
        assert classifier.filter_by == ['High', 'Medium']
        assert any(isinstance(s, Filter) for s in classifier.stages)


class TestDomainClassifierNode:
    def test_registry_entry_exists(self):
        assert 'DomainClassifier' in _CLASSIFIER_CLASS_REGISTRY

    def test_instantiates_with_defaults(self):
        classifier = _CLASSIFIER_CLASS_REGISTRY['DomainClassifier']()
        assert classifier.label_field == 'domain_pred'
        assert classifier.text_field == 'text'
        assert classifier.score_field is None
        assert classifier.filter_by is None

    def test_filter_by_with_labels_adds_filter_stage(self):
        from nemo_curator.stages.text.filters import Filter
        classifier = _CLASSIFIER_CLASS_REGISTRY['DomainClassifier'](filter_by=['News', 'Finance'])
        assert any(isinstance(s, Filter) for s in classifier.stages)

    def test_filter_by_empty_list_coercion(self):
        from nemo_curator.stages.text.filters import Filter
        stage_params = {'filter_by': []}
        if not stage_params.get('filter_by'):
            stage_params['filter_by'] = None
        classifier = _CLASSIFIER_CLASS_REGISTRY['DomainClassifier'](**stage_params)
        assert classifier.filter_by is None
        assert not any(isinstance(s, Filter) for s in classifier.stages)


class TestMultilingualDomainClassifierNode:
    def test_registry_entry_exists(self):
        assert 'MultilingualDomainClassifier' in _CLASSIFIER_CLASS_REGISTRY

    def test_instantiates_with_defaults(self):
        classifier = _CLASSIFIER_CLASS_REGISTRY['MultilingualDomainClassifier']()
        assert classifier.label_field == 'multilingual_domain_pred'
        assert classifier.text_field == 'text'
        assert classifier.score_field is None
        assert classifier.filter_by is None

    def test_filter_by_with_labels_adds_filter_stage(self):
        from nemo_curator.stages.text.filters import Filter
        classifier = _CLASSIFIER_CLASS_REGISTRY['MultilingualDomainClassifier'](filter_by=['News'])
        assert any(isinstance(s, Filter) for s in classifier.stages)


class TestContentTypeClassifierNode:
    def test_registry_entry_exists(self):
        assert 'ContentTypeClassifier' in _CLASSIFIER_CLASS_REGISTRY

    def test_instantiates_with_defaults(self):
        classifier = _CLASSIFIER_CLASS_REGISTRY['ContentTypeClassifier']()
        assert classifier.label_field == 'content_pred'
        assert classifier.text_field == 'text'
        assert classifier.score_field is None
        assert classifier.filter_by is None

    def test_filter_by_with_labels_adds_filter_stage(self):
        from nemo_curator.stages.text.filters import Filter
        classifier = _CLASSIFIER_CLASS_REGISTRY['ContentTypeClassifier'](filter_by=['Article'])
        assert any(isinstance(s, Filter) for s in classifier.stages)


class TestFineWebEduClassifierNode:
    def test_registry_entry_exists(self):
        assert 'FineWebEduClassifier' in _CLASSIFIER_CLASS_REGISTRY

    def test_instantiates_with_defaults(self):
        classifier = _CLASSIFIER_CLASS_REGISTRY['FineWebEduClassifier']()
        assert classifier.label_field == 'fineweb-edu-score-label'
        assert classifier.float_score_field == 'fineweb-edu-score-float'
        assert classifier.int_score_field == 'fineweb-edu-score-int'
        assert classifier.text_field == 'text'
        assert classifier.filter_by is None

    def test_filter_by_with_labels_adds_filter_stage(self):
        from nemo_curator.stages.text.filters import Filter
        classifier = _CLASSIFIER_CLASS_REGISTRY['FineWebEduClassifier'](filter_by=['high_quality'])
        assert any(isinstance(s, Filter) for s in classifier.stages)

    def test_filter_by_empty_list_coercion(self):
        from nemo_curator.stages.text.filters import Filter
        stage_params = {'filter_by': []}
        if not stage_params.get('filter_by'):
            stage_params['filter_by'] = None
        classifier = _CLASSIFIER_CLASS_REGISTRY['FineWebEduClassifier'](**stage_params)
        assert classifier.filter_by is None
        assert not any(isinstance(s, Filter) for s in classifier.stages)


class TestFineWebMixtralEduClassifierNode:
    def test_registry_entry_exists(self):
        assert 'FineWebMixtralEduClassifier' in _CLASSIFIER_CLASS_REGISTRY

    def test_instantiates_with_defaults(self):
        classifier = _CLASSIFIER_CLASS_REGISTRY['FineWebMixtralEduClassifier']()
        assert classifier.label_field == 'fineweb-mixtral-edu-score-label'
        assert classifier.float_score_field == 'fineweb-mixtral-edu-score-float'
        assert classifier.int_score_field == 'fineweb-mixtral-edu-score-int'
        assert classifier.filter_by is None

    def test_filter_by_with_labels_adds_filter_stage(self):
        from nemo_curator.stages.text.filters import Filter
        classifier = _CLASSIFIER_CLASS_REGISTRY['FineWebMixtralEduClassifier'](filter_by=['high_quality'])
        assert any(isinstance(s, Filter) for s in classifier.stages)


class TestFineWebNemotronEduClassifierNode:
    def test_registry_entry_exists(self):
        assert 'FineWebNemotronEduClassifier' in _CLASSIFIER_CLASS_REGISTRY

    def test_instantiates_with_defaults(self):
        classifier = _CLASSIFIER_CLASS_REGISTRY['FineWebNemotronEduClassifier']()
        assert classifier.label_field == 'fineweb-nemotron-edu-score-label'
        assert classifier.float_score_field == 'fineweb-nemotron-edu-score-float'
        assert classifier.int_score_field == 'fineweb-nemotron-edu-score-int'
        assert classifier.filter_by is None

    def test_filter_by_with_labels_adds_filter_stage(self):
        from nemo_curator.stages.text.filters import Filter
        classifier = _CLASSIFIER_CLASS_REGISTRY['FineWebNemotronEduClassifier'](filter_by=['high_quality'])
        assert any(isinstance(s, Filter) for s in classifier.stages)


class TestPromptTaskComplexityClassifierNode:
    def test_registry_entry_exists(self):
        assert 'PromptTaskComplexityClassifier' in _CLASSIFIER_CLASS_REGISTRY

    def test_instantiates_with_defaults(self):
        classifier = _CLASSIFIER_CLASS_REGISTRY['PromptTaskComplexityClassifier']()
        assert classifier.text_field == 'text'
        assert classifier.filter_by is None

    def test_custom_text_field_flows_through(self):
        classifier = _CLASSIFIER_CLASS_REGISTRY['PromptTaskComplexityClassifier'](text_field='prompt')
        assert classifier.text_field == 'prompt'

    def test_filter_by_raises_at_instantiation(self):
        with pytest.raises(NotImplementedError):
            _CLASSIFIER_CLASS_REGISTRY['PromptTaskComplexityClassifier'](filter_by=['high'])


# ---------------------------------------------------------------------------
# Dedup node tests
# ---------------------------------------------------------------------------


class TestDedupRegistry:
    def test_exact_dedup_in_dedup_types(self):
        assert 'ExactDedup' in _DEDUP_TYPES

    def test_fuzzy_dedup_in_dedup_types(self):
        assert 'FuzzyDedup' in _DEDUP_TYPES

    def test_dedup_types_not_in_filter_registry(self):
        from run_pipeline import _FILTER_CLASS_REGISTRY
        for t in _DEDUP_TYPES:
            assert t not in _FILTER_CLASS_REGISTRY

    def test_dedup_types_not_in_classifier_registry(self):
        for t in _DEDUP_TYPES:
            assert t not in _CLASSIFIER_CLASS_REGISTRY


class TestDetectFiletype:
    def test_detects_jsonl(self, tmp_path):
        (tmp_path / "output.jsonl").write_text('{"text": "hello"}\n')
        assert _detect_filetype(str(tmp_path)) == "jsonl"

    def test_falls_back_to_parquet(self, tmp_path):
        (tmp_path / "output.parquet").write_bytes(b"PAR1")
        assert _detect_filetype(str(tmp_path)) == "parquet"

    def test_empty_dir_returns_parquet(self, tmp_path):
        # No files — default fallback is parquet
        assert _detect_filetype(str(tmp_path)) == "parquet"

    def test_jsonl_takes_priority_over_parquet(self, tmp_path):
        (tmp_path / "a.jsonl").write_text('{"text": "hello"}\n')
        (tmp_path / "b.parquet").write_bytes(b"PAR1")
        assert _detect_filetype(str(tmp_path)) == "jsonl"

    # R12 regression tests: file-path branch (previously untested — fix
    # was trivially revertible without breaking directory-only tests).

    def test_detects_jsonl_file_path(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "x"}\n')
        assert _detect_filetype(str(f)) == "jsonl"

    def test_detects_parquet_file_path(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_bytes(b"PAR1")
        assert _detect_filetype(str(f)) == "parquet"

    def test_nonexistent_path_raises(self, tmp_path):
        """Finding F-005: silently returning 'parquet' for missing paths
        produced confusing downstream errors. Now raises FileNotFoundError."""
        import pytest
        with pytest.raises(FileNotFoundError):
            _detect_filetype(str(tmp_path / "does_not_exist.jsonl"))


class TestDedupStageSplitting:
    """Verify the main() routing logic: dedup stages are split from stream stages."""

    def _split(self, stages):
        """Mirror the split logic from main()."""
        dedup = [s for s in stages if s["type"] in _DEDUP_TYPES]
        stream = [s for s in stages if s["type"] not in _DEDUP_TYPES]
        return dedup, stream

    def test_pure_stream_pipeline_has_no_dedup(self):
        stages = [{"type": "WordCountFilter", "params": {}}, {"type": "MarkdownRemover", "params": {}}]
        dedup, stream = self._split(stages)
        assert dedup == []
        assert len(stream) == 2

    def test_pure_dedup_pipeline_has_no_stream(self):
        stages = [{"type": "ExactDedup", "params": {"text_field": "text"}}]
        dedup, stream = self._split(stages)
        assert len(dedup) == 1
        assert stream == []

    def test_mixed_pipeline_splits_correctly(self):
        stages = [
            {"type": "WordCountFilter", "params": {}},
            {"type": "ExactDedup", "params": {}},
        ]
        dedup, stream = self._split(stages)
        assert len(dedup) == 1
        assert len(stream) == 1
        assert dedup[0]["type"] == "ExactDedup"
        assert stream[0]["type"] == "WordCountFilter"

    def test_fuzzy_dedup_is_routed_as_dedup(self):
        stages = [{"type": "FuzzyDedup", "params": {}}]
        dedup, stream = self._split(stages)
        assert len(dedup) == 1
        assert stream == []


class TestDedupParamsCopy:
    """Verify text_field pop doesn't mutate the original config."""

    def test_params_copy_preserves_original(self):
        dedup_cfg = {"type": "ExactDedup", "params": {"text_field": "content", "assign_id": True}}
        original_params = dedup_cfg["params"]

        params = dict(dedup_cfg.get("params", {}))
        text_field = params.pop("text_field", "text")

        assert text_field == "content"
        assert "text_field" in original_params, "original params dict should be unmodified"
        assert "text_field" not in params, "working copy should have text_field removed"


@pytest.mark.skipif(importlib.util.find_spec("cudf") is None, reason="requires deduplication_cuda12 (cudf not installed)")
class TestExactDeduplicationWorkflowParams:
    def test_instantiates_with_required_params(self, tmp_path):
        from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
        wf = ExactDeduplicationWorkflow(
            output_path=str(tmp_path),
            input_path=str(tmp_path),
            text_field="text",
            assign_id=True,
            perform_removal=False,
        )
        assert wf.text_field == "text"
        assert wf.assign_id is True
        assert wf.perform_removal is False

    def test_perform_removal_raises(self, tmp_path):
        from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
        with pytest.raises(NotImplementedError):
            ExactDeduplicationWorkflow(
                output_path=str(tmp_path),
                input_path=str(tmp_path),
                perform_removal=True,
            )


@pytest.mark.skipif(importlib.util.find_spec("cudf") is None, reason="requires deduplication_cuda12 (cudf not installed)")
class TestFuzzyDeduplicationWorkflowParams:
    def test_instantiates_with_defaults(self, tmp_path):
        from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
        wf = FuzzyDeduplicationWorkflow(
            cache_path=str(tmp_path / "cache"),
            output_path=str(tmp_path / "out"),
            input_path=str(tmp_path),
            text_field="text",
            perform_removal=False,
        )
        assert wf.char_ngrams == 24
        assert wf.num_bands == 20
        assert wf.minhashes_per_band == 13
        assert wf.use_64_bit_hash is False
        assert wf.seed == 42

    def test_custom_params_flow_through(self, tmp_path):
        from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
        wf = FuzzyDeduplicationWorkflow(
            cache_path=str(tmp_path / "cache"),
            output_path=str(tmp_path / "out"),
            input_path=str(tmp_path),
            char_ngrams=5,   # below 20 triggers a warning but not an error
            num_bands=10,
            minhashes_per_band=5,
            use_64_bit_hash=True,
            seed=99,
            perform_removal=False,
        )
        assert wf.num_bands == 10
        assert wf.minhashes_per_band == 5
        assert wf.use_64_bit_hash is True
        assert wf.seed == 99
        assert wf.num_hashes == 50  # num_bands * minhashes_per_band

    def test_perform_removal_raises(self, tmp_path):
        from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
        with pytest.raises(NotImplementedError):
            FuzzyDeduplicationWorkflow(
                cache_path=str(tmp_path / "cache"),
                output_path=str(tmp_path / "out"),
                input_path=str(tmp_path),
                perform_removal=True,
            )

    def test_bands_per_iteration_out_of_range_raises(self, tmp_path):
        from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
        with pytest.raises(ValueError):
            FuzzyDeduplicationWorkflow(
                cache_path=str(tmp_path / "cache"),
                output_path=str(tmp_path / "out"),
                input_path=str(tmp_path),
                num_bands=10,
                bands_per_iteration=99,  # > num_bands
                perform_removal=False,
            )


# ---------------------------------------------------------------------------
# Additional String Filters
# ---------------------------------------------------------------------------

class TestSymbolsToWordsFilterNode:
    def test_removes_symbol_heavy_doc(self):
        # 10 hash symbols + 1 word → ratio 10.0, well above 0.1
        result = run_filter('SymbolsToWordsFilter', {
            '_docs': ['Hello world this is normal text', '# # # # # # # # # # one'],
            'max_symbol_to_word_ratio': 0.1,
        })
        assert kept_texts(result) == ['Hello world this is normal text']

    def test_keeps_clean_doc(self):
        result = run_filter('SymbolsToWordsFilter', {
            '_docs': ['The quick brown fox jumps over the lazy dog'],
            'max_symbol_to_word_ratio': 0.1,
        })
        assert len(kept_texts(result)) == 1


class TestParenthesesFilterNode:
    def test_removes_parentheses_heavy_doc(self):
        # Almost entirely parenthesised text
        result = run_filter('ParenthesesFilter', {
            '_docs': ['Normal text here.', '(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)'],
            'max_parentheses_ratio': 0.1,
        })
        assert kept_texts(result) == ['Normal text here.']

    def test_keeps_doc_under_threshold(self):
        result = run_filter('ParenthesesFilter', {
            '_docs': ['Hello (world) this is fine text with very few parentheses overall.'],
            'max_parentheses_ratio': 0.1,
        })
        assert len(kept_texts(result)) == 1


class TestMeanWordLengthFilterNode:
    def test_removes_doc_with_very_long_words(self):
        # avg word length >> 10
        result = run_filter('MeanWordLengthFilter', {
            '_docs': ['Hi there friend.', 'superlongword extraordinarilylong incomprehensibilities'],
            'min_mean_word_length': 3,
            'max_mean_word_length': 10,
        })
        assert kept_texts(result) == ['Hi there friend.']

    def test_keeps_doc_in_range(self):
        result = run_filter('MeanWordLengthFilter', {
            '_docs': ['The quick brown fox'],
            'min_mean_word_length': 3,
            'max_mean_word_length': 10,
        })
        assert len(kept_texts(result)) == 1


class TestPunctuationFilterNode:
    def test_removes_doc_without_end_punctuation(self):
        # All sentences lack end marks
        result = run_filter('PunctuationFilter', {
            '_docs': ['Good sentence here.', 'no punct\nno punct\nno punct\nno punct\nno punct'],
            'max_num_sentences_without_endmark_ratio': 0.5,
        })
        assert kept_texts(result) == ['Good sentence here.']

    def test_keeps_punctuated_doc(self):
        result = run_filter('PunctuationFilter', {
            '_docs': ['This is fine. It has punctuation. Really!'],
            'max_num_sentences_without_endmark_ratio': 0.85,
        })
        assert len(kept_texts(result)) == 1


class TestEllipsisFilterNode:
    def test_removes_doc_with_many_ellipsis_lines(self):
        result = run_filter('EllipsisFilter', {
            '_docs': ['Normal line.\nAnother line.', 'Line one...\nLine two...\nLine three...\nLine four...'],
            'max_num_lines_ending_with_ellipsis_ratio': 0.3,
        })
        assert kept_texts(result) == ['Normal line.\nAnother line.']

    def test_keeps_doc_under_threshold(self):
        result = run_filter('EllipsisFilter', {
            '_docs': ['Just one line.'],
            'max_num_lines_ending_with_ellipsis_ratio': 0.3,
        })
        assert len(kept_texts(result)) == 1


class TestCommonEnglishWordsFilterNode:
    def test_removes_doc_without_common_words(self):
        # stop_at_false=False so all words are counted, not just up to first non-common word
        result = run_filter('CommonEnglishWordsFilter', {
            '_docs': ['the is and in of to a', 'xkzj qwprf bvnmt lkwq'],
            'min_num_common_words': 2,
            'stop_at_false': False,
        })
        assert kept_texts(result) == ['the is and in of to a']

    def test_keeps_doc_with_enough_common_words(self):
        result = run_filter('CommonEnglishWordsFilter', {
            '_docs': ['the is and in of to a'],
            'min_num_common_words': 2,
            'stop_at_false': False,
        })
        assert len(kept_texts(result)) == 1


class TestWordsWithoutAlphabetsFilterNode:
    def test_removes_doc_with_few_alpha_words(self):
        # "123 456 789 abc" → 1/4 words have alphabets = 0.25 < 0.8
        result = run_filter('WordsWithoutAlphabetsFilter', {
            '_docs': ['hello world good text', '123 456 789 abc'],
            'min_words_with_alphabets': 0.8,
        })
        assert kept_texts(result) == ['hello world good text']

    def test_keeps_doc_with_mostly_alpha_words(self):
        result = run_filter('WordsWithoutAlphabetsFilter', {
            '_docs': ['The quick brown fox jumps'],
            'min_words_with_alphabets': 0.8,
        })
        assert len(kept_texts(result)) == 1


class TestPornographicUrlsFilterNode:
    def test_removes_doc_with_porn_url(self):
        result = run_filter('PornographicUrlsFilter', {
            '_docs': ['Clean document with no URLs.', 'Visit http://www.porn.com for adult content'],
        })
        assert kept_texts(result) == ['Clean document with no URLs.']

    def test_keeps_clean_doc(self):
        result = run_filter('PornographicUrlsFilter', {
            '_docs': ['Visit http://www.example.com for more info'],
        })
        assert len(kept_texts(result)) == 1


class TestSubstringFilterNode:
    def test_prefix_match(self):
        result = run_filter('SubstringFilter', {
            '_docs': ['Hello world', 'Goodbye world'],
            'substring': 'Hello',
            'position': 'prefix',
        })
        assert kept_texts(result) == ['Hello world']

    def test_suffix_match(self):
        result = run_filter('SubstringFilter', {
            '_docs': ['Hello world', 'Hello everyone'],
            'substring': 'world',
            'position': 'suffix',
        })
        assert kept_texts(result) == ['Hello world']

    def test_any_match(self):
        result = run_filter('SubstringFilter', {
            '_docs': ['The quick brown fox', 'The lazy dog'],
            'substring': 'brown',
            'position': 'any',
        })
        assert kept_texts(result) == ['The quick brown fox']


# ---------------------------------------------------------------------------
# Repetition Filters
# ---------------------------------------------------------------------------

class TestRepeatedLinesFilterNode:
    def test_removes_doc_with_repeated_lines(self):
        repeated = '\n'.join(['same line'] * 20)
        result = run_filter('RepeatedLinesFilter', {
            '_docs': ['Line one.\nLine two.\nLine three.', repeated],
            'max_repeated_line_fraction': 0.7,
        })
        assert kept_texts(result) == ['Line one.\nLine two.\nLine three.']

    def test_keeps_unique_lines_doc(self):
        result = run_filter('RepeatedLinesFilter', {
            '_docs': ['Alpha.\nBeta.\nGamma.\nDelta.'],
            'max_repeated_line_fraction': 0.7,
        })
        assert len(kept_texts(result)) == 1


class TestRepeatedParagraphsFilterNode:
    def test_removes_doc_with_repeated_paragraphs(self):
        repeated = '\n\n'.join(['Same paragraph text here.'] * 10)
        result = run_filter('RepeatedParagraphsFilter', {
            '_docs': ['Para one.\n\nPara two.\n\nPara three.', repeated],
            'max_repeated_paragraphs_ratio': 0.7,
        })
        assert kept_texts(result) == ['Para one.\n\nPara two.\n\nPara three.']

    def test_keeps_unique_paragraphs_doc(self):
        result = run_filter('RepeatedParagraphsFilter', {
            '_docs': ['First paragraph.\n\nSecond paragraph.\n\nThird paragraph.'],
            'max_repeated_paragraphs_ratio': 0.7,
        })
        assert len(kept_texts(result)) == 1


class TestRepeatedLinesByCharFilterNode:
    def test_removes_doc_with_high_repeated_char_ratio(self):
        repeated = '\n'.join(['abcdefghij'] * 20)
        result = run_filter('RepeatedLinesByCharFilter', {
            '_docs': ['Short unique line.\nAnother unique line.', repeated],
            'max_repeated_lines_char_ratio': 0.8,
        })
        assert kept_texts(result) == ['Short unique line.\nAnother unique line.']

    def test_keeps_doc_under_threshold(self):
        result = run_filter('RepeatedLinesByCharFilter', {
            '_docs': ['Unique line alpha.\nUnique line beta.\nUnique line gamma.'],
            'max_repeated_lines_char_ratio': 0.8,
        })
        assert len(kept_texts(result)) == 1


class TestRepeatedParagraphsByCharFilterNode:
    def test_removes_doc_with_high_repeated_para_char_ratio(self):
        repeated = '\n\n'.join(['Repeated paragraph content here.'] * 10)
        result = run_filter('RepeatedParagraphsByCharFilter', {
            '_docs': ['Unique para one.\n\nUnique para two.', repeated],
            'max_repeated_paragraphs_char_ratio': 0.8,
        })
        assert kept_texts(result) == ['Unique para one.\n\nUnique para two.']

    def test_keeps_unique_paragraphs(self):
        result = run_filter('RepeatedParagraphsByCharFilter', {
            '_docs': ['First unique paragraph.\n\nSecond unique paragraph.'],
            'max_repeated_paragraphs_char_ratio': 0.8,
        })
        assert len(kept_texts(result)) == 1


class TestRepeatingTopNGramsFilterNode:
    def test_removes_doc_dominated_by_top_ngram(self):
        # "the cat" repeated many times dominates by char count
        repeated = ' '.join(['the cat'] * 50)
        result = run_filter('RepeatingTopNGramsFilter', {
            '_docs': ['The quick brown fox jumps over the lazy dog near the river bank.', repeated],
            'n': 2,
            'max_repeating_ngram_ratio': 0.2,
        })
        assert kept_texts(result) == ['The quick brown fox jumps over the lazy dog near the river bank.']

    def test_keeps_diverse_doc(self):
        result = run_filter('RepeatingTopNGramsFilter', {
            '_docs': ['Lions and tigers and bears roam across the savanna under the hot sun.'],
            'n': 2,
            'max_repeating_ngram_ratio': 0.2,
        })
        assert len(kept_texts(result)) == 1


class TestRepeatingDuplicateNGramsFilterNode:
    def test_removes_doc_with_many_duplicate_ngrams(self):
        repeated = ' '.join(['hello world'] * 30)
        result = run_filter('RepeatingDuplicateNGramsFilter', {
            '_docs': ['The fox jumped over the fence into the field.', repeated],
            'n': 2,
            'max_repeating_duplicate_ngram_ratio': 0.2,
        })
        assert kept_texts(result) == ['The fox jumped over the fence into the field.']

    def test_keeps_low_duplication_doc(self):
        result = run_filter('RepeatingDuplicateNGramsFilter', {
            '_docs': ['Every word in this sentence is different from the others around it.'],
            'n': 2,
            'max_repeating_duplicate_ngram_ratio': 0.2,
        })
        assert len(kept_texts(result)) == 1


# ---------------------------------------------------------------------------
# Code Quality Filters
# ---------------------------------------------------------------------------

class TestPythonCommentToCodeFilterNode:
    def test_removes_doc_with_no_comments(self):
        # No comments → score 0, below min_comment_to_code_ratio=0.01 → filtered
        no_comment_code = "x = 1\ny = 2\nresult = x + y\nz = result * 2\ntotal = z + result\n"
        result = run_filter('PythonCommentToCodeFilter', {
            '_docs': [no_comment_code, 'def add(a, b):\n    # compute the sum\n    return a + b\n'],
            'min_comment_to_code_ratio': 0.01,
            'max_comment_to_code_ratio': 0.85,
        })
        assert kept_texts(result) == ['def add(a, b):\n    # compute the sum\n    return a + b\n']

    def test_keeps_doc_with_moderate_comment_ratio(self):
        result = run_filter('PythonCommentToCodeFilter', {
            '_docs': ['def foo():\n    # this computes a value\n    return 42\n'],
            'min_comment_to_code_ratio': 0.01,
            'max_comment_to_code_ratio': 0.85,
        })
        assert len(kept_texts(result)) == 1


class TestGeneralCommentToCodeFilterNode:
    def test_removes_doc_with_no_comments(self):
        # C code with no comments → score 0 < min=0.01 → filtered
        no_comment = "int add(int a, int b) {\n    return a + b;\n}\nint main() {\n    return add(1, 2);\n}\n"
        commented = "// Adds two numbers\nint add(int a, int b) {\n    return a + b;\n}\nint main() {\n    return add(1, 2);\n}\n"
        result = run_filter('GeneralCommentToCodeFilter', {
            '_docs': [no_comment, commented],
            'language': 'text/x-c',
            'min_comment_to_code_ratio': 0.01,
            'max_comment_to_code_ratio': 0.85,
        })
        assert kept_texts(result) == [commented]

    def test_keeps_code_with_moderate_comments(self):
        result = run_filter('GeneralCommentToCodeFilter', {
            '_docs': ['// Process input\nvoid process(int x) {\n    x = x + 1;\n}\n'],
            'language': 'text/x-c',
            'min_comment_to_code_ratio': 0.01,
            'max_comment_to_code_ratio': 0.85,
        })
        assert len(kept_texts(result)) == 1


class TestNumberOfLinesOfCodeFilterNode:
    def test_removes_doc_with_too_few_lines(self):
        # 3 lines < min_lines=10
        short = "x = 1\ny = 2\nz = 3"
        long_code = "\n".join(f"x_{i} = {i}" for i in range(15))  # 15 lines
        result = run_filter('NumberOfLinesOfCodeFilter', {
            '_docs': [short, long_code],
            'min_lines': 10,
            'max_lines': 20000,
        })
        assert kept_texts(result) == [long_code]

    def test_keeps_doc_within_range(self):
        code = "\n".join(f"line_{i} = {i}" for i in range(12))  # 12 lines
        result = run_filter('NumberOfLinesOfCodeFilter', {
            '_docs': [code],
            'min_lines': 10,
            'max_lines': 20000,
        })
        assert len(kept_texts(result)) == 1


class TestXMLHeaderFilterNode:
    def test_removes_doc_with_xml_header(self):
        xml_doc = '<?xml version="1.0" encoding="UTF-8"?>\n<root><item>value</item></root>'
        normal_code = 'def hello():\n    return "world"\n'
        result = run_filter('XMLHeaderFilter', {
            '_docs': [xml_doc, normal_code],
        })
        assert kept_texts(result) == [normal_code]

    def test_keeps_doc_without_xml_header(self):
        result = run_filter('XMLHeaderFilter', {
            '_docs': ['def foo():\n    pass\n'],
        })
        assert len(kept_texts(result)) == 1


class TestAlphaFilterNode:
    def test_removes_doc_with_low_alpha_ratio(self):
        # All digits/dots/spaces — zero alphabetic characters
        numeric_only = "1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0"
        normal_text = "The quick brown fox jumps over the lazy dog near the river bank."
        result = run_filter('AlphaFilter', {
            '_docs': [numeric_only, normal_text],
            'min_alpha_ratio': 0.25,
        })
        assert kept_texts(result) == [normal_text]

    def test_keeps_doc_with_high_alpha_ratio(self):
        result = run_filter('AlphaFilter', {
            '_docs': ['Hello world this is normal text with plenty of alphabetic characters.'],
            'min_alpha_ratio': 0.25,
        })
        assert len(kept_texts(result)) == 1


class TestHTMLBoilerplateFilterNode:
    def test_removes_html_that_is_mostly_script(self):
        # Script dominates — visible text is just "Hi" (2 chars) out of ~500+
        script_heavy = (
            '<html><body>'
            '<script>' + ('var x = 1; var y = 2; x += y; console.log(x); ' * 20) + '</script>'
            '<p>Hi</p>'
            '</body></html>'
        )
        content_rich = (
            '<html><body>'
            '<p>This is a lovely paragraph about science and nature with lots of '
            'meaningful content for humans to read and understand deeply.</p>'
            '</body></html>'
        )
        result = run_filter('HTMLBoilerplateFilter', {
            '_docs': [script_heavy, content_rich],
            'min_lang_content_ratio': 0.2,
            'min_lang_content_num_chars': 100,
        })
        assert kept_texts(result) == [content_rich]

    def test_keeps_html_with_sufficient_content(self):
        result = run_filter('HTMLBoilerplateFilter', {
            '_docs': [
                '<html><body><p>' + 'The quick brown fox. ' * 10 + '</p></body></html>'
            ],
            'min_lang_content_ratio': 0.2,
            'min_lang_content_num_chars': 100,
        })
        assert len(kept_texts(result)) == 1


# ---------------------------------------------------------------------------
# FastText + remaining model-based node tests
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch

import numpy as np

from nemo_curator.stages.text.filters.fasttext.fasttext_filters import (
    FastTextQualityFilter,
    FastTextLangId,
)
from nemo_curator.stages.text.filters.histogram.histogram import HistogramFilter
from nemo_curator.stages.text.filters.token.token_count import TokenCountFilter
from nemo_curator.stages.text.filters.heuristic.code.code import (
    TokenizerFertilityFilter,
    PerExtensionFilter,
)


class TestFastTextLabelModifierNode:
    def test_prepends_label(self):
        result = run_modifier('FastTextLabelModifier', {'label': '__label__hq'}, ['hello world'])
        assert kept_texts(result) == ['__label__hq hello world']

    def test_prepends_custom_label(self):
        result = run_modifier('FastTextLabelModifier', {'label': '__label__low'}, ['some text here'])
        assert kept_texts(result) == ['__label__low some text here']


def _ft_score_mock(value):
    """Build a fake fasttext score object that supports [0][0].item()."""
    score_val = MagicMock()
    score_val.item.return_value = value
    return score_val


class TestFastTextQualityFilterNode:
    def test_registered(self):
        assert 'FastTextQualityFilter' in _FILTER_CLASS_REGISTRY

    def test_keeps_high_quality_doc(self):
        filter_obj = FastTextQualityFilter(model_path='/fake/model.bin')
        mock_model = MagicMock()
        # predict returns (labels, scores) where scores[0][0].item() == 0.95
        mock_model.predict.return_value = ([['__label__hq']], [[_ft_score_mock(0.95)]])
        filter_obj._fasttext_quality_filter_model = mock_model

        with patch.object(filter_obj, 'load_model'):
            stage = ScoreFilter(filter_obj=filter_obj)
            stage.setup()

        # Patch pareto so keep_document is deterministic: pareto=0.99 → keep when 0.99 > 1-0.95=0.05
        with patch('nemo_curator.stages.text.filters.fasttext.fasttext_filters.np.random.pareto', return_value=0.99):
            result = stage.process(make_batch(['top quality document']))

        assert len(kept_texts(result)) == 1

    def test_removes_low_quality_doc(self):
        filter_obj = FastTextQualityFilter(model_path='/fake/model.bin')
        mock_model = MagicMock()
        # score 0.1 → keep requires pareto(3) > 0.9; patch pareto=0.0 → always drop
        mock_model.predict.return_value = ([['__label__hq']], [[_ft_score_mock(0.1)]])
        filter_obj._fasttext_quality_filter_model = mock_model

        with patch.object(filter_obj, 'load_model'):
            stage = ScoreFilter(filter_obj=filter_obj)
            stage.setup()

        with patch('nemo_curator.stages.text.filters.fasttext.fasttext_filters.np.random.pareto', return_value=0.0):
            result = stage.process(make_batch(['low quality garbage']))

        assert len(kept_texts(result)) == 0


class TestFastTextLangIdNode:
    def test_keeps_high_confidence_language(self):
        filter_obj = FastTextLangId(model_path='/fake/lid.176.bin', min_langid_score=0.3)
        mock_model = MagicMock()
        # confidence 0.9, language EN → above threshold
        mock_model.predict.return_value = ([['__label__EN']], [[_ft_score_mock(0.9)]])
        filter_obj._fasttext_langid_model = mock_model

        with patch.object(filter_obj, 'load_model'):
            stage = ScoreFilter(filter_obj=filter_obj)
            stage.setup()

        result = stage.process(make_batch(['This is an English sentence.']))
        assert len(kept_texts(result)) == 1

    def test_removes_low_confidence_doc(self):
        filter_obj = FastTextLangId(model_path='/fake/lid.176.bin', min_langid_score=0.3)
        mock_model = MagicMock()
        # confidence 0.1 → below threshold 0.3
        mock_model.predict.return_value = ([['__label__ZH']], [[_ft_score_mock(0.1)]])
        filter_obj._fasttext_langid_model = mock_model

        with patch.object(filter_obj, 'load_model'):
            stage = ScoreFilter(filter_obj=filter_obj)
            stage.setup()

        result = stage.process(make_batch(['ambiguous text']))
        assert len(kept_texts(result)) == 0


class TestHistogramFilterNode:
    def _make_filter(self):
        # Bypass download and file read; manually set histogram state after
        with patch('nemo_curator.stages.text.filters.histogram.histogram.os.path.isdir', return_value=True), \
             patch.object(HistogramFilter, '_read_hist'):
            f = HistogramFilter(lang='en', threshold=0.8)
        # Set histogram to basic ASCII letters + space (mimics English)
        f._histogram = set('abcdefghijklmnopqrstuvwxyz ')
        return f

    def test_keeps_text_matching_histogram(self):
        filter_obj = self._make_filter()
        stage = ScoreFilter(filter_obj=filter_obj)
        stage.setup()
        # All chars in the histogram → coverage 1.0 > 0.8 → kept
        result = stage.process(make_batch(['hello world this is english']))
        assert len(kept_texts(result)) == 1

    def test_removes_text_not_matching_histogram(self):
        filter_obj = self._make_filter()
        stage = ScoreFilter(filter_obj=filter_obj)
        stage.setup()
        # Only digits and punctuation — none in histogram → coverage 0 < 0.8 → removed
        result = stage.process(make_batch(['12345 67890 !@#$%']))
        assert len(kept_texts(result)) == 0


class TestTokenCountFilterNode:
    def _make_filter(self, min_tokens=5, max_tokens=20):
        filter_obj = TokenCountFilter.__new__(TokenCountFilter)
        # Bypass __init__ entirely; set required state manually
        filter_obj._min_tokens = min_tokens
        filter_obj._max_tokens = max_tokens
        filter_obj._name = 'token_count'
        mock_tokenizer = MagicMock()
        filter_obj._token_count_filter_tokenizer = mock_tokenizer
        return filter_obj, mock_tokenizer

    def test_keeps_doc_within_token_range(self):
        filter_obj, mock_tok = self._make_filter(min_tokens=5, max_tokens=20)
        mock_tok.encode.return_value = list(range(10))  # 10 tokens — in range
        with patch.object(filter_obj, 'load_tokenizer', lambda: None):
            stage = ScoreFilter(filter_obj=filter_obj)
            stage.setup()
        result = stage.process(make_batch(['a reasonable length document']))
        assert len(kept_texts(result)) == 1

    def test_removes_doc_below_min_tokens(self):
        filter_obj, mock_tok = self._make_filter(min_tokens=5, max_tokens=20)
        mock_tok.encode.return_value = list(range(2))  # 2 tokens — below min
        with patch.object(filter_obj, 'load_tokenizer', lambda: None):
            stage = ScoreFilter(filter_obj=filter_obj)
            stage.setup()
        result = stage.process(make_batch(['hi']))
        assert len(kept_texts(result)) == 0


class TestTokenizerFertilityFilterNode:
    def test_keeps_high_ratio_code(self):
        mock_sp = MagicMock()
        # 30 chars / 3 tokens = 10.0 ratio → above default threshold 2.5
        mock_sp.encode_as_pieces.return_value = ['token1', 'token2', 'token3']

        with patch('nemo_curator.stages.text.filters.heuristic.code.code.sentencepiece.SentencePieceProcessor', return_value=mock_sp):
            filter_obj = TokenizerFertilityFilter(path_to_tokenizer='/fake/tokenizer.model')

        stage = ScoreFilter(filter_obj=filter_obj)
        stage.setup()
        result = stage.process(make_batch(['def foo(): return bar']))
        assert len(kept_texts(result)) == 1

    def test_removes_low_ratio_code(self):
        mock_sp = MagicMock()
        # 5 chars / 10 tokens = 0.5 ratio → below threshold 2.5
        mock_sp.encode_as_pieces.return_value = list(range(10))  # 10 "tokens"

        with patch('nemo_curator.stages.text.filters.heuristic.code.code.sentencepiece.SentencePieceProcessor', return_value=mock_sp):
            filter_obj = TokenizerFertilityFilter(path_to_tokenizer='/fake/tokenizer.model')

        stage = ScoreFilter(filter_obj=filter_obj)
        stage.setup()
        result = stage.process(make_batch(['hello']))  # 5 chars / 10 tokens = 0.5
        assert len(kept_texts(result)) == 0


class TestPerExtensionFilterNode:
    @staticmethod
    def _csv_path():
        from pathlib import Path
        import nemo_curator
        return str(Path(nemo_curator.__file__).parent / 'utils' / 'code_meta.csv')

    def test_keeps_clean_python_code(self):
        result = run_filter('PerExtensionFilter', {
            '_docs': ['def greet(name):\n    return f"Hello, {name}"\n'],
            'lang': 'python',
            'extension': 'py',
            'metadata_file': self._csv_path(),
        })
        assert len(kept_texts(result)) == 1

    def test_removes_python_with_extreme_line_length(self):
        # A single extremely long line (10000 chars) should exceed Long_line_threshold=1000
        long_line = 'x' * 10000 + '\n'
        result = run_filter('PerExtensionFilter', {
            '_docs': [long_line],
            'lang': 'python',
            'extension': 'py',
            'metadata_file': self._csv_path(),
        })
        assert len(kept_texts(result)) == 0
