"""Additional offline tests: model loading/saving, preprocessing options, utilities,
setup helpers and CLI commands, plus regression tests for bugs fixed in 2.5.0.

Hugging Face, spaCy, NLTK downloads and HTTP are all replaced with fakes.
"""

from __future__ import annotations

import io
import json
import signal
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

import lingo
from lingo import cli, models, setup_utils, utils
from lingo.core import Pipeline
from lingo.models import EmbeddingModel, NERModel, SentimentAnalyzer, TextClassifier
from lingo.preprocessing import TextPreprocessor

# ------------------------------------------------------------------- fakes


class _FakeTokenizer:
    saved_to: list[Path] = []

    def save_pretrained(self, path: Path) -> None:
        _FakeTokenizer.saved_to.append(Path(path))


class _FakeHFModel:
    def __init__(self) -> None:
        self.device_moves: list[str] = []

    def to(self, device: str) -> _FakeHFModel:
        self.device_moves.append(device)
        return self

    def save_pretrained(self, path: Path) -> None:
        (Path(path) / "weights.bin").write_text("w")


@pytest.fixture
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the transformers entry points used by ``models._load_hf_pipeline``."""
    calls: dict[str, Any] = {"sources": []}

    class _Auto:
        @staticmethod
        def from_pretrained(source: str, cache_dir: str | None = None) -> Any:
            calls["sources"].append(source)
            return _FakeHFModel()

    class _Tok:
        @staticmethod
        def from_pretrained(source: str, cache_dir: str | None = None) -> Any:
            return _FakeTokenizer()

    def _pipeline(task: str, **kwargs: Any) -> Any:
        calls["task"] = task
        calls["pipeline_kwargs"] = kwargs
        return lambda inputs, **kw: [{"label": "X", "text": t} for t in inputs]

    monkeypatch.setattr(models, "AutoTokenizer", _Tok)
    monkeypatch.setattr(models, "AutoModelForSequenceClassification", _Auto)
    monkeypatch.setattr(models, "AutoModelForTokenClassification", _Auto)
    monkeypatch.setattr(models, "pipeline", _pipeline)
    return calls


# --------------------------------------------------------------- exceptions


def test_exceptions_are_backward_compatible() -> None:
    assert issubclass(lingo.UnsupportedTaskError, ValueError)
    assert issubclass(lingo.InvalidInputError, ValueError)
    assert issubclass(lingo.UnsupportedTaskError, lingo.LingoError)
    with pytest.raises(lingo.UnsupportedTaskError, match="not supported"):
        Pipeline(task="translation", model="x", device="cpu")


# ------------------------------------------------------------------- models


def test_hf_models_load_forward_kwargs_and_move_to_device(
    fake_transformers: dict[str, Any],
) -> None:
    ner = NERModel("dslim/bert-base-NER", device="cpu", aggregation_strategy="simple")
    assert fake_transformers["task"] == "ner"
    assert fake_transformers["pipeline_kwargs"]["aggregation_strategy"] == "simple"
    assert fake_transformers["pipeline_kwargs"]["device"] == "cpu"
    assert ner.model.device_moves == ["cpu"]
    assert ner(["Paris"]) == [{"label": "X", "text": "Paris"}]

    SentimentAnalyzer("s", device="cpu")
    assert fake_transformers["task"] == "sentiment-analysis"


def test_hf_model_load_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Broken:
        @staticmethod
        def from_pretrained(source: str, cache_dir: str | None = None) -> Any:
            raise OSError(f"{source} is not a valid model identifier")

    monkeypatch.setattr(models, "AutoTokenizer", _Broken)
    with pytest.raises(OSError, match="not a valid model"):
        TextClassifier("does/not-exist")


def test_model_save_and_load_roundtrip(fake_transformers: dict[str, Any], tmp_path: Path) -> None:
    clf = TextClassifier("org/model", config={"k": 1}, device="cpu")
    clf.save(tmp_path / "saved")
    saved = json.loads((tmp_path / "saved" / "model_config.json").read_text(encoding="utf-8"))
    assert saved["model_name"] == "org/model"
    assert (tmp_path / "saved" / "weights.bin").exists()

    clf.load(tmp_path / "saved")
    assert fake_transformers["sources"][-1] == str(tmp_path / "saved")
    assert clf.model_name == "org/model"
    assert clf.config == {"k": 1}


def test_embedding_model_load_save_and_zero_vector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _ST:
        def __init__(self, name: str, cache_folder: str | None = None) -> None:
            self.name = name

        def to(self, device: str) -> None:
            self.device = device

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            return np.array([[0.0, 0.0] if t == "" else [1.0, 2.0] for t in texts])

        def save(self, path: str) -> None:
            (Path(path) / "st.bin").write_text("x")

    fake_module = mock.MagicMock(SentenceTransformer=_ST)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    emb = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    assert emb.model.device == "cpu"
    assert emb("hi") == [1.0, 2.0]
    # A zero vector used to produce NaN through a division by zero
    assert emb.similarity("", "hi") == 0.0
    emb.save(tmp_path / "emb")
    assert (tmp_path / "emb" / "st.bin").exists()
    assert json.loads((tmp_path / "emb" / "model_config.json").read_text())["model_name"]


def test_pipeline_load_can_override_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    created: list[str] = []

    class _Fake(models.BaseModel):
        def _load_model(self) -> None:
            created.append(self.device)

        def __call__(self, inputs: Any, **kwargs: Any) -> Any:
            return inputs

        def save(self, path: str | Path) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    tasks = dict(Pipeline.SUPPORTED_TASKS)
    tasks["sentiment-analysis"] = _Fake
    monkeypatch.setattr(Pipeline, "SUPPORTED_TASKS", tasks)

    nlp = Pipeline(task="sentiment-analysis", model="m", device="cuda")
    nlp.save(tmp_path / "p")
    Pipeline.load(tmp_path / "p", device="cpu")
    assert created == ["cuda", "cpu"]
    assert Pipeline.load(tmp_path / "p").device == "cuda"


# ------------------------------------------------------------- preprocessing


# TextPreprocessor fetches missing NLTK data on first use (as it does for users).


def test_all_preprocessing_options() -> None:
    pre = TextPreprocessor(
        config={
            "use_spacy": False,
            "lowercase": True,
            "remove_numbers": True,
            "remove_special_chars": True,
            "expand_slang": True,
            "correct_spelling": True,
        }
    )
    assert pre("BTW the  price is $42!!") == "by the way the price is"
    assert pre(["A1", "B2"]) == ["a", "b"]
    assert pre("") == ""


def test_expand_contractions_handles_irregular_forms() -> None:
    pre = TextPreprocessor(config={"use_spacy": False, "lowercase": False})
    assert pre.expand_contractions("I won't go, I can't.") == "I will not go, I cannot."
    assert pre.expand_contractions("They're here") == "They are here"


def test_contractions_are_expanded_before_punctuation_is_removed() -> None:
    pre = TextPreprocessor(
        config={"use_spacy": False, "expand_contractions": True, "remove_punctuation": True}
    )
    # Used to give "i cant go" because the apostrophe was stripped first
    assert pre("I can't go!") == "i cannot go"
    assert pre("It\u2019s fine") == "it is fine"


def test_subword_tokenization_warns(caplog: pytest.LogCaptureFixture) -> None:
    TextPreprocessor(config={"use_spacy": False, "use_subword_tokenization": True})
    assert "not implemented" in caplog.text


def test_missing_spacy_model_falls_back_to_nltk(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(name: str) -> Any:
        raise OSError(name)

    monkeypatch.setattr("spacy.load", _missing)
    pre = TextPreprocessor(config={"use_spacy": True, "spacy_model": "xx_missing"})
    assert pre.spacy_model is None
    assert pre.tokenize_words("Hello world") == ["Hello", "world"]


def test_stemming_and_unknown_stopword_language() -> None:
    pre = TextPreprocessor(
        config={"use_spacy": False, "stem": True, "remove_stopwords": True},
        language="klingon",
    )
    assert pre.stopwords == set()
    assert pre.remove_stopwords(["a", "b"]) == ["a", "b"]
    result = pre.get_preprocessing_pipeline("Running runners ran")
    assert result["stemmed"] == ["run", "runner", "ran"]


def test_preprocessor_load_roundtrip(tmp_path: Path) -> None:
    pre = TextPreprocessor(config={"use_spacy": False, "lowercase": True}, language="english")
    pre.save(tmp_path)
    loaded = TextPreprocessor.load(tmp_path)
    assert loaded.config == pre.config
    assert loaded("ABC") == "abc"


# -------------------------------------------------------------------- utils


def test_chunk_text_terminates_when_separator_is_close() -> None:
    text = "Hello world.\nThis is a second line that is long.\nEnd."

    def _timeout(*_: Any) -> None:
        raise TimeoutError("chunk_text did not terminate")

    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(5)
    try:
        chunks = utils.chunk_text(text, max_length=20, overlap=15)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    assert chunks[0] == "Hello world."
    assert chunks[-1].endswith("End.")
    assert all(len(c) <= 20 for c in chunks)


def test_chunk_text_validates_arguments() -> None:
    with pytest.raises(ValueError, match="max_length"):
        utils.chunk_text("abc", max_length=0)
    with pytest.raises(ValueError, match="overlap"):
        utils.chunk_text("abc", overlap=-1)


def test_flesch_reading_ease_uses_syllables() -> None:
    easy = utils.get_text_statistics("The cat sat on the mat. The dog ran.")
    hard = utils.get_text_statistics(
        "Institutional considerations necessitate comprehensive organizational restructuring."
    )
    # 9 words, 2 sentences, 9 syllables: 206.835 - 1.015*4.5 - 84.6*1.0 = 117.7 -> clamped
    assert easy["flesch_reading_ease"] == 100.0
    assert easy["syllables"] == 9
    assert hard["flesch_reading_ease"] < 10
    assert hard["reading_level"] == "Very Difficult"
    assert utils.get_text_statistics("")["flesch_reading_ease"] == 0.0


@pytest.mark.parametrize(
    ("score", "level"),
    [
        (95, "Very Easy"),
        (85, "Easy"),
        (75, "Fairly Easy"),
        (65, "Standard"),
        (55, "Fairly Difficult"),
        (35, "Difficult"),
    ],
)
def test_reading_levels(score: float, level: str) -> None:
    assert utils._get_reading_level(score) == level


def test_download_file_uses_timeout_and_closes_response(tmp_path: Path) -> None:
    response = mock.MagicMock()
    response.__enter__.return_value = response
    response.headers = {"content-length": "6"}
    response.iter_content.return_value = [b"abc", b"def"]
    with mock.patch.object(utils.requests, "get", return_value=response) as get:
        out = utils.download_file("https://example.com/f", tmp_path / "f.bin", show_progress=True)
    assert get.call_args.kwargs["timeout"] == 30.0
    assert Path(out).read_bytes() == b"abcdef"
    response.__exit__.assert_called_once()

    response.headers = {}
    with mock.patch.object(utils.requests, "get", return_value=response):
        utils.download_file("https://example.com/f", str(tmp_path / "g.bin"), timeout=5)
    assert (tmp_path / "g.bin").read_bytes() == b"abcdef"


def test_download_model_returns_cache_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", mock.MagicMock())
    monkeypatch.setattr(transformers, "AutoModel", mock.MagicMock())
    path = utils.download_model("org/name", cache_dir=str(tmp_path))
    assert path == str(tmp_path / "models--org--name")
    assert utils.download_model("bert").endswith("models--bert")


def test_save_model_and_standalone_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="save method"):
        utils.save_model(object(), tmp_path)
    (tmp_path / "model_config.json").write_text("{}")
    with pytest.raises(NotImplementedError):
        utils.load_model(tmp_path)


def test_batch_texts() -> None:
    assert utils.batch_texts(["a", "b", "c"], batch_size=2) == [["a", "b"], ["c"]]


# -------------------------------------------------------------- setup utils


def test_download_nltk_data_only_fetches_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import nltk

    downloaded: list[str] = []

    def _find(path: str) -> None:
        if path.endswith("wordnet"):
            raise LookupError(path)

    monkeypatch.setattr(nltk.data, "find", _find)
    monkeypatch.setattr(nltk, "download", lambda pkg, quiet: downloaded.append(pkg))
    assert setup_utils.download_nltk_data() is True
    assert downloaded == ["wordnet"]


def test_download_spacy_model_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import spacy

    monkeypatch.setattr(spacy, "load", lambda name: object())
    assert setup_utils.download_spacy_model("en_core_web_sm") is True

    attempts = {"n": 0}

    def _load_after_download(name: str) -> Any:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("missing")
        return object()

    monkeypatch.setattr(spacy, "load", _load_after_download)
    with mock.patch.object(setup_utils.subprocess, "run") as run:
        assert setup_utils.download_spacy_model("en_core_web_sm") is True
    assert run.call_args.args[0][-1] == "en_core_web_sm"

    def _always_missing(name: str) -> Any:
        raise OSError("missing")

    monkeypatch.setattr(spacy, "load", _always_missing)
    with mock.patch.object(setup_utils.subprocess, "run"):
        assert setup_utils.download_spacy_model("en_core_web_sm") is False


def test_download_transformers_data_tolerates_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    tok = mock.MagicMock()
    tok.from_pretrained.side_effect = [None, OSError("offline"), None, None]
    monkeypatch.setattr(transformers, "AutoTokenizer", tok)
    monkeypatch.setattr(transformers, "AutoModel", mock.MagicMock())
    assert setup_utils.download_transformers_data() is True
    assert tok.from_pretrained.call_count == 4


def test_setup_environment_writes_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(setup_utils.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(setup_utils, "download_spacy_model", lambda: True)
    monkeypatch.setattr(setup_utils, "download_nltk_data", lambda: True)
    monkeypatch.setattr(setup_utils, "download_transformers_data", lambda: False)
    assert setup_utils.setup_lingo_environment() is False
    marker = (tmp_path / ".lingo_cache" / "setup_complete.txt").read_text(encoding="utf-8")
    assert "Transformers: failed" in marker
    assert setup_utils.check_lingo_environment() is True


# ---------------------------------------------------------------------- CLI


class _EchoPipeline:
    def __init__(self, task: str, **kwargs: Any) -> None:
        self.task = task

    def __call__(self, text: str | None = None, **kwargs: Any) -> Any:
        return {"echo": text}


def test_cli_reads_input_file_and_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "Pipeline", _EchoPipeline)
    src = tmp_path / "in.txt"
    src.write_text("  from file  ", encoding="utf-8")
    assert cli.main(["run", "ner", "--model", "m", "--input-file", str(src)]) == 0
    assert json.loads(capsys.readouterr().out) == {"echo": "from file"}

    monkeypatch.setattr(sys, "stdin", io.StringIO("from stdin\n"))
    assert cli.main(["run", "ner", "--model", "m"]) == 0
    assert json.loads(capsys.readouterr().out) == {"echo": "from stdin"}


def test_cli_errors_go_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "Pipeline", _EchoPipeline)
    monkeypatch.setattr(sys, "stdin", io.StringIO("   "))
    assert cli.main(["run", "ner", "--model", "m"]) == 1
    captured = capsys.readouterr()
    assert "No input text" in captured.err
    assert captured.out == ""

    def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("model missing")

    monkeypatch.setattr(cli, "Pipeline", _boom)
    assert cli.main(["run", "ner", "--model", "m", "--text", "x", "-v"]) == 1
    assert "model missing" in capsys.readouterr().err


def test_cli_setup_and_download(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(cli.Path, "home", lambda: tmp_path)
    marker = tmp_path / ".lingo_cache" / "setup_complete.txt"
    marker.parent.mkdir()
    marker.write_text("old")
    monkeypatch.setattr(cli, "setup_lingo_environment", lambda: True)
    assert cli.main(["setup", "--force"]) == 0
    assert not marker.exists()
    monkeypatch.setattr(cli, "setup_lingo_environment", lambda: False)
    assert cli.main(["setup"]) == 1

    monkeypatch.setattr(cli, "download_model", lambda model, cache_dir: f"/cache/{model}")
    assert cli.main(["download-model", "--model", "bert"]) == 0
    assert "/cache/bert" in capsys.readouterr().out

    def _fail(model: str, cache_dir: str | None) -> str:
        raise OSError("offline")

    monkeypatch.setattr(cli, "download_model", _fail)
    assert cli.main(["download-model", "--model", "bert"]) == 1
    assert "offline" in capsys.readouterr().err
