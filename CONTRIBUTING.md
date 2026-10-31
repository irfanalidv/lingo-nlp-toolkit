# Contributing to lingo-nlp-toolkit

Thanks for helping improve `lingo-nlp-toolkit`!

## Development setup

```bash
git clone https://github.com/irfanalidv/lingo-nlp-toolkit.git
cd lingo-nlp-toolkit
uv venv && source .venv/bin/activate   # or: python -m venv .venv
uv pip install -e ".[dev]"             # or: pip install -e ".[dev]"
pre-commit install                     # run Ruff on every commit
```

## Before opening a pull request

```bash
ruff check .                               # lint
ruff format --check .                      # formatting
mypy lingo                                 # type-check
pytest --cov=lingo --cov-fail-under=90     # tests (offline; models are faked)
```

- Add or update tests for any behaviour change.
- Add a line to `CHANGELOG.md` under **Unreleased**.
- Keep pull requests focused: one fix or feature per PR.

## Reporting bugs and requesting features

Use the [issue templates](https://github.com/irfanalidv/lingo-nlp-toolkit/issues/new/choose).
