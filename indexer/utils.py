"""
Indexer utilities: hashing, state persistence, logging setup,
and deterministic candidate ID generation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure structured stdout logging for the indexer."""
    # Windows consoles default to cp1252, which cannot encode the '→' and '✓'
    # used in the progress messages below. Every such line raised
    # UnicodeEncodeError inside the logging handler — including the "no CVs
    # found" hint, so the one message explaining an empty run was the one that
    # blew up. Harmless no-op where stdout is already UTF-8 (Docker, Linux).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def compute_md5(path: str | Path) -> str:
    """Return the MD5 hex digest of a file. Reads in 64 KB chunks."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_candidate_id(path: str | Path) -> str:
    """
    Generate a deterministic UUID from the file's resolved path.
    Re-indexing the same file always produces the same candidate_id,
    which allows Qdrant payloads to be associated correctly.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(Path(path).resolve())))


def load_state(state_path: str | Path) -> dict:
    """
    Load the index state JSON file.
    Returns an empty dict if the file doesn't exist or is corrupt.
    """
    p = Path(state_path)
    if not p.exists():
        logger.info("No existing state file at %s — starting fresh", p)
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info("Loaded state for %d files from %s", len(state), p)
        return state
    except Exception as e:
        logger.error("Failed to read state file %s: %s — starting fresh", p, e)
        return {}


def save_state(state_path: str | Path, state: dict) -> None:
    """
    Atomically write the state dict to JSON.
    Uses a temp-file + rename pattern to avoid corruption on interrupted runs.
    """
    p = Path(state_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=p.parent,
            delete=False,
            suffix=".tmp.json",
        ) as tmp:
            json.dump(state, tmp, indent=2, ensure_ascii=False)
            tmp_path = tmp.name

        os.replace(tmp_path, p)
        logger.debug("State saved to %s (%d entries)", p, len(state))
    except Exception as e:
        if tmp_path and Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        logger.error("Failed to save state file %s: %s", p, e)
        raise
