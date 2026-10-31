"""Tests for Pipeline, model wrappers, CLI and utilities.

Heavy Hugging Face / sentence-transformers models are replaced with fakes so the
suite runs offline and fast.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

import lingo
from lingo import cli
from lingo.core import Pipeline
from lingo.models import BaseModel, EmbeddingModel, QuestionAnsweringModel, TextClassifier
from lingo.utils import (
    clean_text,
    evaluate_classification,
    evaluate_ner,
    get_available_models,
    load_model,
    save_model,
)

# --------------------------------------------------------------------------- fakes


class FakeModel(BaseModel):
    """Records inputs and returns one dict per input text."""

    loaded_from: list[str] = []

    def _load_model(self) -> None:
        FakeModel.loaded_from.append(self.model_name)

    def __call__(self, inputs: Any, **kwargs: Any) -> list[dict[str, Any]]:
        if isinstance(inputs, str):
            inputs = [inputs]
        self.last_inputs = list(inputs)
        return [{"label": "POSITIVE", "text": t} for t in inputs]


class FakeQAModel(BaseModel):
    def _load_model(self) -> None:
        pass

    def __call__(self, question: str, context: str, **kwargs: Any) -> dict[str, Any]:
        return {"answer": context.split()[0], "question": question}


@pytest.fixture
def fake_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = dict(Pipeline.SUPPORTED_TASKS)
    tasks["sentiment-analysis"] = FakeModel
    tasks["question-answering"] = FakeQAModel
    monkeypatch.setattr(Pipeline, "SUPPORTED_TASKS", tasks)
    FakeModel.loaded_from = []


# ------------------------------------------------------------------------ package


def test_version_matches_metadata() -> None:
    from importlib.metadata import version

    assert lingo.__version__ == version("lingo-nlp-toolkit")


def test_public_api_exports() -> None:
    for name in lingo.__all__:
        assert hasattr(lingo, name), name


# ----------------------------------------------------------------------- pipeline


def test_unsupported_task_raises() -> None:
    with pytest.raises(ValueError, match="not supported"):
        Pipeline(task="translation", model="x", device="cpu")


def test_single_input_returns_single_result(fake_tasks: None) -> None:
    nlp = Pipeline(task="sentiment-analysis", model="fake", device="cpu")
    result = nlp("Hello World")
    assert result == {"label": "POSITIVE", "text": "Hello World"}


def test_pipeline_does_not_lowercase_by_default(fake_tasks: None) -> None:
    nlp = Pipeline(task="sentiment-analysis", model="fake", device="cpu")
    nlp("Apple Inc.   is in   Cupertino")
    # Casing is preserved (important for cased NER models); whitespace is normalized.
    assert nlp.model.last_inputs == ["Apple Inc. is in Cupertino"]


def test_pipeline_respects_custom_preprocessing(fake_tasks: None) -> None:
    nlp = Pipeline(
        task="sentiment-analysis",
        model="fake",
        device="cpu",
        config={"preprocessing": {"lowercase": True, "remove_punctuation": True}},
    )
    nlp("Hello, World!")
    assert nlp.model.last_inputs == ["hello world"]


def test_list_input_returns_list(fake_tasks: None) -> None:
    nlp = Pipeline(task="sentiment-analysis", model="fake", device="cpu")
    results = nlp(["a", "b"])
    assert [r["text"] for r in results] == ["a", "b"]


def test_single_item_list_returns_list(fake_tasks: None) -> None:
    nlp = Pipeline(task="sentiment-analysis", model="fake", device="cpu")
    results = nlp(["only one"])
    assert isinstance(results, list) and len(results) == 1


def test_batch_predict_handles_partial_last_batch(fake_tasks: None) -> None:
    nlp = Pipeline(task="sentiment-analysis", model="fake", device="cpu")
    texts = [f"t{i}" for i in range(5)]
    results = nlp.batch_predict(texts, batch_size=2)
    assert [r["text"] for r in results] == texts


def test_batch_predict_rejects_bad_batch_size(fake_tasks: None) -> None:
    nlp = Pipeline(task="sentiment-analysis", model="fake", device="cpu")
    with pytest.raises(ValueError):
        nlp.batch_predict(["a"], batch_size=0)


def test_inputs_required(fake_tasks: None) -> None:
    nlp = Pipeline(task="sentiment-analysis", model="fake", device="cpu")
    with pytest.raises(ValueError, match="Inputs are required"):
        nlp()


def test_question_answering_requires_question_and_context(fake_tasks: None) -> None:
    qa = Pipeline(task="question-answering", model="fake", device="cpu")
    with pytest.raises(ValueError):
        qa(question="Who?")
    assert qa(question="Who?", context="Guido created Python")["answer"] == "Guido"


def test_get_info_and_repr(fake_tasks: None) -> None:
    nlp = Pipeline(task="sentiment-analysis", model="fake", device="cpu")
    info = nlp.get_info()
    assert info["task"] == "sentiment-analysis"
    assert "embedding" in info["supported_tasks"]
    assert "sentiment-analysis" in repr(nlp)


def test_save_and_load_roundtrip_uses_saved_weights(fake_tasks: None, tmp_path) -> None:
    nlp = Pipeline(
        task="sentiment-analysis",
        model="fake-model",
        device="cpu",
        config={"preprocessing": {"lowercase": True}},
    )
    save_model(nlp, tmp_path / "pipe")
    assert (tmp_path / "pipe" / "pipeline_config.json").exists()
    assert (tmp_path / "pipe" / "model" / "model_config.json").exists()

    FakeModel.loaded_from = []
    loaded = load_model(tmp_path / "pipe")
    assert isinstance(loaded, Pipeline)
    # Weights are loaded from the saved directory, not re-fetched by name
    assert FakeModel.loaded_from == [str(tmp_path / "pipe" / "model")]
    assert loaded.model_name == "fake-model"
    assert loaded.preprocessor.config["lowercase"] is True


def test_load_model_errors(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing")
    with pytest.raises(ValueError):
        load_model(tmp_path)


# ------------------------------------------------------------------------- models


class _FakeHFPipeline:
    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        if isinstance(inputs, list):
            return [{"label": "LABEL_0", "score": 0.9} for _ in inputs]
        return {"label": "LABEL_0", "score": 0.9}


def test_hf_wrapper_always_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TextClassifier, "_load_model", lambda self: None)
    clf = TextClassifier(model_name="fake")
    clf.pipeline = _FakeHFPipeline()
    assert clf("one") == [{"label": "LABEL_0", "score": 0.9}]
    assert len(clf(["a", "b"])) == 2


class _FakeEncoding(dict):
    """Mimics a fast-tokenizer BatchEncoding for one window: [CLS] q [SEP] ctx... [SEP]."""

    def __init__(self, context: str) -> None:
        import torch

        self.words = context.split()
        offsets, pos = [(0, 0), (0, 0), (0, 0)], 0  # CLS, question, SEP
        for w in self.words:
            start = context.index(w, pos)
            offsets.append((start, start + len(w)))
            pos = start + len(w)
        offsets.append((0, 0))  # final SEP
        n = len(offsets)
        super().__init__(
            input_ids=torch.zeros((1, n), dtype=torch.long),
            attention_mask=torch.ones((1, n), dtype=torch.long),
            offset_mapping=torch.tensor([offsets]),
        )
        self.n = n

    def sequence_ids(self, i: int) -> list[int | None]:
        return [None, 0, None] + [1] * len(self.words) + [None]


class _FakeQAModelOutput:
    def __init__(self, start_logits: Any, end_logits: Any) -> None:
        self.start_logits = start_logits
        self.end_logits = end_logits


def test_qa_model_extracts_best_span(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    context = "Python was created by Guido van Rossum in 1991"
    enc = _FakeEncoding(context)
    # Token positions: 3 + word index. "Guido"=word 4 -> 7, "Rossum"=word 6 -> 9
    start = torch.full((1, enc.n), -5.0)
    end = torch.full((1, enc.n), -5.0)
    start[0, 7] = 8.0
    end[0, 9] = 8.0
    start[0, 0] = end[0, 0] = 20.0  # CLS must be ignored (not part of context)

    monkeypatch.setattr(QuestionAnsweringModel, "_load_model", lambda self: None)
    qa = QuestionAnsweringModel(model_name="fake")
    qa.tokenizer = lambda *a, **kw: enc
    qa.model = lambda **kw: _FakeQAModelOutput(start, end)
    qa.model.device = "cpu"

    result = qa("Who created Python?", context)
    assert result["answer"] == "Guido van Rossum"
    assert context[result["start"] : result["end"]] == "Guido van Rossum"
    assert 0.9 < result["score"] <= 1.0

    with pytest.raises(ValueError):
        qa("", context)


def test_summarization_uses_prefix_and_generation_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from lingo.models import SummarizationModel

    calls: dict[str, Any] = {}

    class FakeTokenizer:
        model_max_length = 512

        def __call__(self, texts: list[str], **kw: Any) -> dict[str, Any]:
            calls["texts"] = texts
            return {"input_ids": torch.zeros((len(texts), 3), dtype=torch.long)}

        def batch_decode(self, ids: Any, **kw: Any) -> list[str]:
            return [f" summary {i} " for i in range(len(ids))]

    class FakeSeq2Seq:
        device = "cpu"

        class config:
            task_specific_params = {"summarization": {"prefix": "summarize: ", "num_beams": 4}}

        def generate(self, **kw: Any) -> Any:
            calls["generate"] = kw
            return kw["input_ids"]

    monkeypatch.setattr(SummarizationModel, "_load_model", lambda self: None)
    model = SummarizationModel(model_name="fake")
    model.tokenizer = FakeTokenizer()
    model.model = FakeSeq2Seq()

    out = model(["long text a", "long text b"], max_length=20)
    assert out == [{"summary_text": "summary 0"}, {"summary_text": "summary 1"}]
    assert calls["texts"] == ["summarize: long text a", "summarize: long text b"]
    assert calls["generate"]["num_beams"] == 4
    assert calls["generate"]["max_length"] == 20


class _FakeSentenceTransformer:
    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        vocab = ["cat", "mat", "weather"]
        return np.array([[float(w in t.lower()) for w in vocab] for t in texts], dtype=np.float32)


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> EmbeddingModel:
    monkeypatch.setattr(EmbeddingModel, "_load_model", lambda self: None)
    model = EmbeddingModel(model_name="fake")
    model.model = _FakeSentenceTransformer()
    return model


def test_embedding_shapes(embedder: EmbeddingModel) -> None:
    single = embedder("the cat")
    assert single == [1.0, 0.0, 0.0]
    batch = embedder(["the cat", "the weather"])
    assert batch == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert all(isinstance(x, float) for x in single)


def test_embedding_similarity(embedder: EmbeddingModel) -> None:
    assert embedder.similarity("cat on mat", "a cat on the mat") == pytest.approx(1.0)
    assert embedder.similarity("cat", "weather") == pytest.approx(0.0)


def test_embedding_pipeline_single_text_returns_vector(
    embedder: EmbeddingModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = dict(Pipeline.SUPPORTED_TASKS)
    tasks["embedding"] = lambda **kw: embedder
    monkeypatch.setattr(Pipeline, "SUPPORTED_TASKS", tasks)
    nlp = Pipeline(task="embedding", model="fake", device="cpu")
    assert nlp("The cat") == [1.0, 0.0, 0.0]


# -------------------------------------------------------------------------- utils


def test_clean_text_normalizes_curly_quotes() -> None:
    assert clean_text("“Hi”  it’s   ok") == '"Hi" it\'s ok'


def test_evaluate_classification() -> None:
    y_true = ["pos", "neg", "pos", "neu"]
    y_pred = ["pos", "neg", "pos", "pos"]
    metrics = evaluate_classification(y_true, y_pred, labels=["pos", "neg", "neu"])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert set(metrics["per_class"]) == {"pos", "neg", "neu"}
    assert metrics["per_class"]["neg"]["precision"] == pytest.approx(1.0)
    assert len(metrics["confusion_matrix"]) == 3


def test_evaluate_ner() -> None:
    gold = [[{"entity": "ORG", "start": 0, "end": 5}, {"entity": "LOC", "start": 10, "end": 15}]]
    pred = [[{"entity": "ORG", "start": 0, "end": 5}]]
    m = evaluate_ner(gold, pred)
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    assert m["f1"] == pytest.approx(2 / 3)


def test_available_models_cover_all_tasks() -> None:
    models = get_available_models()
    assert set(models) == set(Pipeline.SUPPORTED_TASKS)
    assert all(models.values())


def test_nltk_resource_paths_are_valid() -> None:
    from lingo.setup_utils import NLTK_RESOURCES

    assert NLTK_RESOURCES["stopwords"] == "corpora/stopwords"
    assert NLTK_RESOURCES["punkt"] == "tokenizers/punkt"


# ---------------------------------------------------------------------------- CLI


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert lingo.__version__ in capsys.readouterr().out


def test_cli_list_models(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list-models"]) == 0
    assert "SENTIMENT-ANALYSIS" in capsys.readouterr().out


def test_cli_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()


class _CliFakePipeline:
    def __init__(self, task: str, **kwargs: Any) -> None:
        self.task = task

    def __call__(self, text: str | None = None, **kwargs: Any) -> Any:
        if self.task == "question-answering":
            return {"answer": kwargs["context"].split()[0], "score": np.float32(0.5)}
        return {"label": "POSITIVE", "score": np.float32(0.25)}


def test_cli_run_serializes_numpy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "Pipeline", _CliFakePipeline)
    code = cli.main(["run", "sentiment-analysis", "--model", "m", "--text", "great"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"label": "POSITIVE", "score": 0.25}


def test_cli_run_question_answering(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    monkeypatch.setattr(cli, "Pipeline", _CliFakePipeline)
    out_file = tmp_path / "out.json"
    code = cli.main(
        [
            "run",
            "question-answering",
            "--model",
            "m",
            "--text",
            "Guido created Python",
            "--question",
            "Who?",
            "--output-file",
            str(out_file),
        ]
    )
    assert code == 0
    assert json.loads(out_file.read_text())["answer"] == "Guido"


def test_cli_question_answering_requires_question(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "Pipeline", _CliFakePipeline)
    assert cli.main(["run", "question-answering", "--model", "m", "--text", "ctx"]) == 1
