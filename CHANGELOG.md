# Changelog

All notable changes to the Lingo NLP Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.5.0] - 2026-10-31

### Added

- `lingo.exceptions` with `LingoError`, `UnsupportedTaskError` and `InvalidInputError`
  (both also `ValueError`, as raised before), exported from `lingo`.
- `Pipeline.load(path, device=None)` can override the saved device.
- `download_file(..., timeout=30.0)`.
- Four runnable use-case examples replace the previous simulated scripts.
- CI runs mypy and enforces 90% coverage; CodeQL workflow; pre-commit config.

### Fixed

- `chunk_text` could loop forever when a separator shortened a chunk below `overlap`.
- `get_text_statistics` computed Flesch Reading Ease from characters per word instead of
  syllables per word, so nearly every text scored 0.
- With `expand_contractions` and `remove_punctuation` both enabled, contractions were never
  expanded (punctuation removal ran first). Typographic apostrophes are handled, and
  `won't`/`can't`/`shan't` expand correctly.
- `TextPreprocessor(remove_stopwords=True)` crashed with `OSError` for languages without an NLTK
  stopword list.
- `download_file` had no timeout and leaked the streamed response.
- `EmbeddingModel.similarity` returned NaN for zero vectors.
- JSON config files are read and written as UTF-8 on every platform.
- CLI errors are printed to stderr.

### Changed

- Ruff rule set broadened (`SIM`, `C4`, `RUF`, `PTH`, `RET`, `PIE`); stricter mypy settings.
- Removed QUICKSTART.md, INSTALLATION.md, CAPABILITIES.md, PYPI_RELEASE.md and demo.py; the README
  is the single user guide.

## [2.4.0] - 2026-10-31

### Fixed

- **Question answering and summarization on transformers 5**: transformers 5 removed the
  `question-answering` and `summarization` pipelines, so both tasks raised `KeyError`.
  They are now implemented directly on `AutoModelForQuestionAnswering` /
  `AutoModelForSeq2SeqLM` and work on transformers 4.x and 5.x (long QA contexts are
  split into overlapping windows; T5-style prefixes are applied automatically).
- **Pipeline lowercased all input by default**, which badly degraded cased models
  (e.g. `dslim/bert-base-NER` returned word-piece fragments). Pipelines now only normalize
  Unicode and whitespace unless `config["preprocessing"]` says otherwise.
- **Embedding pipeline on a single string returned one float** instead of the vector.
- **`batch_predict` crashed** when the last batch had a single item.
- **`Pipeline.load` ignored saved weights** (it re-downloaded the model by name) and
  discarded the saved preprocessor config. It now loads from the saved directory.
- **`evaluate_classification` per-class metrics were mismatched** with the given labels
  (sklearn sorted labels alphabetically).
- `clean_text` curly-quote normalization was a no-op; it now converts curly quotes to ASCII.
- `lingo setup` re-downloaded NLTK stopwords/wordnet every run (wrong lookup path) and used
  `os.system("python ...")` instead of the current interpreter.
- CLI output failed on numpy scalars (`json` serialization); `--verbose` before the
  subcommand was ignored.
- `get_available_models()` listed `microsoft/DialoGPT-medium` (a chat model) for question
  answering; replaced with `deepset/minilm-uncased-squad2`.
- Constructor keyword arguments (e.g. `aggregation_strategy="simple"` for NER) are now
  forwarded to the Hugging Face pipeline.

### Added

- `lingo run question-answering --question ...` and `lingo --version`.
- `lingo.__version__` is read from package metadata.
- Offline test suite (42 tests) covering Pipeline, model wrappers, save/load, CLI and utilities.
- GitHub Actions CI (lint + tests on Python 3.10-3.13), Ruff config, community files
  (CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue/PR templates, Dependabot, CITATION.cff).

### Changed

- Packaging consolidated into `pyproject.toml` (PEP 639 `license = "MIT"`); removed
  `setup.py`, the `setup_*.py` variants, `requirements.txt` and committed build artifacts.
- Requires Python 3.10+ (3.10-3.13 supported).
- Dropped unused core dependencies (`fastapi`, `uvicorn`, `pandas`, `scipy`, `pydantic`,
  `beautifulsoup4`, `lxml`, `tokenizers` pin). FastAPI/uvicorn are now in the `api` extra and
  pandas in the `examples` extra; unused heavy packages were removed from `full`.
- Linting switched from black/isort/flake8 to Ruff (`make lint` / `make format`).
- README: corrected example outputs to verified values; removed claims for features that do
  not exist (fine-tuning, Hub export, mixed precision, async batching) and the unverified
  benchmark table.

## [2.3] - 2025-01-15

### Added

- **PyPI Downloads Badge**: Added download statistics badge to show package popularity
- **PyPI Package Links**: Added direct links to PyPI package page with #files section across all documentation

### Fixed

- **PyPI Badge**: Corrected PyPI version badge to point to `lingo-nlp-toolkit` instead of `lingo`
- **Documentation URLs**: Updated all documentation links from `lingo.readthedocs.io` to GitHub repository
- **Download URLs**: Fixed all PyPI download links to include #files section for better user experience
- **Package Consistency**: Ensured all setup files and documentation use consistent package names and URLs

### Documentation

- **README.md**: Updated PyPI badges and package links
- **All Documentation Files**: Added consistent PyPI package links with #files section
- **Setup Files**: Synchronized all version numbers and URLs

## [0.2.2] - 2025-01-15

### Fixed

- **Summarization Output Format**: Corrected README examples to use `summary['summary_text']` instead of `summary[0]['summary_text']`
- **Documentation Consistency**: Ensured all examples match actual pipeline output behavior
- **Version Consistency**: Updated all setup files to use version 0.2.2
- **Example Files**: Enhanced examples to handle both list and dict output formats robustly

### Documentation

- **README.md**: Fixed summarization example to match actual pipeline output
- **Examples**: Updated to handle different output formats gracefully
- **Setup Files**: Synchronized all version numbers across setup files

## [0.2.1] - 2025-01-15

### Fixed

- **Documentation Updates**: Fixed all remaining old package name references (`lingo` → `lingo-nlp-toolkit`)
- **Installation Instructions**: Updated all pip install commands throughout documentation
- **Repository References**: Ensured all GitHub URLs point to the new repository
- **Setup Files**: Updated all setup files to use correct package name and version
- **Requirements**: Updated requirements.txt with correct package references

## [0.2.0] - 2025-01-15

### Added

- **Repository Migration**: Moved to new GitHub repository at [https://github.com/irfanalidv/lingo-nlp-toolkit](https://github.com/irfanalidv/lingo-nlp-toolkit)
- **Updated Documentation**: All documentation now references the new repository URL
- **PyPI Package**: Successfully published as `lingo-nlp-toolkit` on PyPI

### Changed

- **Package Name**: Changed from `lingo` to `lingo-nlp-toolkit` to avoid PyPI conflicts
- **Installation Command**: Users now install with `pip install lingo-nlp-toolkit`
- **Repository URLs**: All GitHub links updated to new repository

### Fixed

- **Metadata Issues**: Resolved PyPI metadata conflicts
- **Documentation Consistency**: All files now have consistent repository references

## [Unreleased]

### Added

- Advanced use case examples for enterprise applications
- Document intelligence system with comprehensive analytics
- Content recommendation engine using embeddings
- Social media content analysis capabilities
- Customer feedback analysis with priority scoring
- Performance benchmarking and optimization tools
- Comprehensive functionality showcase
- Enterprise-grade NLP pipeline with job queuing
- Batch processing with performance monitoring
- Advanced text preprocessing configurations

### Changed

- Improved text preprocessing with multiple configuration profiles
- Enhanced error handling and logging throughout the toolkit
- Better performance optimization for GPU and CPU usage
- More comprehensive documentation and examples

### Fixed

- Resolved embedding pipeline return type handling
- Improved NLTK data download and initialization
- Better error handling for missing dependencies

## [0.1.0] - 2024-01-15

### Added

- Core NLP pipeline with unified interface
- Text preprocessing with comprehensive cleaning options
- Sentiment analysis using transformer models
- Named Entity Recognition (NER) capabilities
- Text classification with custom models
- Text embeddings and similarity search
- Question answering with context support
- Text summarization (extractive and abstractive)
- Auto-setup for NLTK, spaCy, and transformer models
- Command-line interface (CLI) for all operations
- Configuration management with YAML support
- Pipeline persistence (save/load functionality)
- Batch processing for multiple texts
- GPU support with automatic device detection
- Comprehensive documentation and examples

### Features

- **Text Preprocessing**: Unicode normalization, contraction expansion, slang expansion, stopword removal, lemmatization, stemming
- **NLP Tasks**: Sentiment analysis, NER, text classification, embeddings, QA, summarization
- **Model Management**: Hugging Face integration, automatic model downloading, caching
- **Performance**: GPU acceleration, batch processing, memory optimization
- **Developer Experience**: Pythonic API, comprehensive examples, CLI tools

### Technical Details

- Python 3.9+ support
- PyTorch and Transformers integration
- spaCy and NLTK integration
- Comprehensive error handling and logging
- Type hints throughout the codebase
- Modular and extensible architecture

---

## Version History

- **2.3**: **CURRENT** - Comprehensive documentation updates, PyPI badges, and package link fixes
- **0.2.2**: Documentation consistency fixes and summarization output format corrections
- **0.2.1**: Documentation fixes and package name consistency updates
- **0.2.0**: Repository migration and PyPI package updates
- **0.1.0**: Initial release with core NLP capabilities
- **Unreleased**: Advanced enterprise features and optimizations

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
