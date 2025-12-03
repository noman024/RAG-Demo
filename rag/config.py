"""
Central configuration for paths and model settings.

This keeps environment handling and default values in one place so the rest
of the codebase can depend on simple constants.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_store"

# Model names are defined here so they can be overridden via environment
# variables if needed, e.g. to experiment with different OpenAI models.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Vision model for image-based understanding (OCR/tables, screenshots, etc.).
# Defaults to gpt-4o which supports multimodal input.
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    # We avoid failing at import-time so that tooling (linters, IDEs) still work.
    # The ingest/query modules will raise a clearer error if the key is missing
    # when they try to actually call OpenAI.
    pass
