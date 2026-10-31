"""
Command-line interface for Lingo NLP toolkit.
Provides easy access to NLP tasks from the command line.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .core import Pipeline
from .setup_utils import setup_lingo_environment
from .utils import download_model, get_available_models


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def list_models() -> None:
    """List available models for different tasks."""
    models = get_available_models()

    print("Available models for different NLP tasks:\n")

    for task, model_list in models.items():
        print(f"📋 {task.upper()}:")
        for model in model_list:
            print(f"   • {model}")
        print()


def _json_default(obj: Any) -> Any:
    """Make numpy scalars/arrays returned by HF pipelines JSON serializable."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def run_pipeline(args: argparse.Namespace) -> int:
    """Run a specific NLP pipeline."""
    try:
        # Create pipeline
        pipeline = Pipeline(
            task=args.task,
            model=args.model,
            device=args.device,
            cache_dir=args.cache_dir,
        )

        # Process input
        if args.input_file:
            with open(args.input_file, encoding="utf-8") as f:
                text = f.read().strip()
        elif args.text:
            text = args.text
        else:
            # Read from stdin
            text = sys.stdin.read().strip()

        if not text:
            print("Error: No input text provided")
            return 1

        # Run inference
        if args.task == "question-answering":
            if not args.question:
                print(
                    "Error: question-answering requires --question (the input text is the context)"
                )
                return 1
            result = pipeline(question=args.question, context=text)
        else:
            result = pipeline(text)

        # Output results
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=_json_default)
            print(f"Results saved to {args.output_file}")
        else:
            print(json.dumps(result, indent=2, default=_json_default))

        return 0

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def setup_cmd(args: argparse.Namespace) -> int:
    """Set up Lingo environment."""
    try:
        if args.force:
            # Remove completion marker to force re-setup
            cache_dir = Path.home() / ".lingo_cache"
            completion_file = cache_dir / "setup_complete.txt"
            if completion_file.exists():
                completion_file.unlink()

        print("Setting up Lingo NLP environment...")
        print("This will download required NLP data and models.")
        print("This may take a few minutes on first run.")

        success = setup_lingo_environment()

        if success:
            print("✅ Lingo environment setup completed successfully!")
            return 0
        else:
            print("⚠️  Setup completed with some warnings. Check the output above.")
            return 1

    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return 1


def download_model_cmd(args: argparse.Namespace) -> int:
    """Download a model from Hugging Face Hub."""
    try:
        print(f"Downloading model: {args.model}")
        cache_path = download_model(args.model, args.cache_dir)
        print(f"Model downloaded successfully to: {cache_path}")
        return 0
    except Exception as e:
        print(f"Error downloading model: {e}")
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Lingo: Advanced NLP Toolkit - Lightweight, Fast, and Transformer-Ready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set up Lingo environment (first time)
  lingo setup

  # Sentiment analysis
  lingo run sentiment-analysis --model cardiffnlp/twitter-roberta-base-sentiment-latest --text "I love this product!"

  # Text classification from file
  lingo run text-classification --model bert-base-uncased --input-file input.txt --output-file results.json

  # NER from stdin
  echo "Apple Inc. is headquartered in Cupertino, California." | lingo run ner --model dslim/bert-base-NER

  # List available models
  lingo list-models

  # Download a model
  lingo download-model --model bert-base-uncased
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List models command
    subparsers.add_parser("list-models", help="List available models")

    # Setup command
    setup_parser = subparsers.add_parser(
        "setup", help="Set up Lingo environment (download required data)"
    )
    setup_parser.add_argument("--force", action="store_true", help="Force re-download of all data")

    # Download model command
    download_parser = subparsers.add_parser("download-model", help="Download a model")
    download_parser.add_argument("--model", required=True, help="Model name to download")
    download_parser.add_argument("--cache-dir", help="Directory to cache the model")

    # Run pipeline command
    run_parser = subparsers.add_parser("run", help="Run an NLP pipeline")
    run_parser.add_argument(
        "task",
        choices=[
            "text-classification",
            "ner",
            "sentiment-analysis",
            "embedding",
            "question-answering",
            "summarization",
        ],
        help="NLP task to perform",
    )
    run_parser.add_argument("--model", required=True, help="Model to use")
    run_parser.add_argument("--text", help="Input text")
    run_parser.add_argument(
        "--question", help="Question to answer (question-answering task; input text is the context)"
    )
    run_parser.add_argument("--input-file", help="Input file path")
    run_parser.add_argument("--output-file", help="Output file path")
    run_parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device to run on",
    )
    run_parser.add_argument("--cache-dir", help="Directory to cache models")
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Verbose output",
    )

    # Global options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args(argv)

    # Setup logging
    setup_logging(args.verbose)

    # Handle commands
    if args.command == "list-models":
        list_models()
        return 0
    elif args.command == "setup":
        return setup_cmd(args)
    elif args.command == "download-model":
        return download_model_cmd(args)
    elif args.command == "run":
        return run_pipeline(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
