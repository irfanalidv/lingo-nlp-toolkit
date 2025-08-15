# Changelog

All notable changes to the Lingo NLP Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

- **0.2.1**: Documentation fixes and package name consistency updates
- **0.2.0**: Repository migration and PyPI package updates
- **0.1.0**: Initial release with core NLP capabilities
- **Unreleased**: Advanced enterprise features and optimizations

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
