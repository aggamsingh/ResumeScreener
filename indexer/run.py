"""
CV Indexer — Main Entrypoint

Scans the CV folder, hashes each file, skips unchanged files,
parses and embeds new/modified CVs, and upserts them into Qdrant.

Usage:
    docker compose run indexer          # recommended
    python -m indexer.run               # local dev (needs Qdrant running)

Never crashes on a bad CV file — catches per-file exceptions,
logs them, and continues. Writes index_state.json atomically on exit.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl  # POSIX only (Linux/macOS) — available inside Docker
    _HAVE_FCNTL = True
except ImportError:
    _HAVE_FCNTL = False  # Windows local dev — lock is skipped with a warning

from dotenv import load_dotenv

# Load .env before importing modules that read env vars at import time
load_dotenv()

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from indexer.embedder import embed_and_upsert, ensure_collection
from indexer.parser import SUPPORTED_EXTENSIONS, parse_file
from indexer.utils import (
    compute_md5,
    get_candidate_id,
    load_state,
    save_state,
    setup_logging,
)
from qdrant_client.http.models import FieldCondition, Filter, FilterSelector, MatchValue

# ── Configuration ──────────────────────────────────────────────────────────────

CV_FOLDER      = Path(os.getenv("CV_FOLDER_PATH", "./cvs"))
STATE_PATH     = Path(os.getenv("INDEX_STATE_PATH", "./data/index_state.json"))
QDRANT_HOST    = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION     = os.getenv("QDRANT_COLLECTION", "resumes")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("CV Indexer — Starting")
    logger.info("CV folder : %s", CV_FOLDER.resolve())
    logger.info("Qdrant    : %s:%d / collection='%s'", QDRANT_HOST, QDRANT_PORT, COLLECTION)
    logger.info("=" * 60)

    # ── Concurrent indexer protection ─────────────────────────────
    # Prevents two simultaneous indexer runs from creating duplicate
    # vectors (each run generates new uuid4 point IDs).
    lock_path = STATE_PATH.parent / ".indexer.lock"
    lock_fd = None
    if _HAVE_FCNTL:
        try:
            lock_fd = open(lock_path, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.debug("Acquired indexer lock: %s", lock_path)
        except (IOError, BlockingIOError):
            logger.error(
                "Another indexer instance is already running. "
                "If this is wrong, delete %s and retry.",
                lock_path,
            )
            sys.exit(1)
    else:
        logger.warning(
            "fcntl not available (Windows local dev) — concurrent indexer "
            "protection is disabled. Do not run two indexers simultaneously."
        )

    # ── Validate CV folder ─────────────────────────────────────────
    if not CV_FOLDER.exists():
        logger.error(
            "CV folder not found: %s\n"
            "  → Set CV_FOLDER_PATH in your .env file to the correct path.",
            CV_FOLDER.resolve(),
        )
        sys.exit(1)

    # ── Load existing state ────────────────────────────────────────
    state = load_state(STATE_PATH)

    # ── Connect to Qdrant ──────────────────────────────────────────
    logger.info("Connecting to Qdrant...")
    try:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3)
        qdrant.get_collections()  # connection test
        logger.info("Qdrant server connected")
    except Exception as e:
        logger.warning(
            "Could not connect to Qdrant at %s:%d (%s). Falling back to embedded local storage at ./data/qdrant_db",
            QDRANT_HOST,
            QDRANT_PORT,
            e,
        )
        qdrant = QdrantClient(path="./data/qdrant_db")
        logger.info("Embedded local Qdrant database initialized")

    # ── Load embedding model ───────────────────────────────────────
    # Model must be loaded before ensure_collection so we can pass
    # the actual vector dimension (model.get_sentence_embedding_dimension())
    # instead of a hardcoded constant.
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector_dim = model.get_sentence_embedding_dimension()
    logger.info("Embedding model ready (dim=%d)", vector_dim)

    ensure_collection(qdrant, COLLECTION, vector_dim)


    # ── Discover CV files ──────────────────────────────────────────
    all_files = [
        f for f in CV_FOLDER.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    all_files.sort()

    if not all_files:
        logger.warning(
            "No .pdf or .docx files found in %s\n"
            "  → Drop your CV files in that folder and run the indexer again.",
            CV_FOLDER.resolve(),
        )
        sys.exit(0)

    logger.info("Found %d CV files to process", len(all_files))

    # ── Indexing loop ──────────────────────────────────────────
    stats = {"scanned": 0, "indexed": 0, "skipped": 0, "failed": 0}
    started_at = datetime.now(timezone.utc)
    _CHECKPOINT_EVERY = 50  # Save state after every N files indexed

    try:
        for file_path in all_files:
            stats["scanned"] += 1
            file_key = str(file_path.resolve())

            # Compute file hash (skip unreadable files)
            try:
                current_hash = compute_md5(file_path)
            except Exception as e:
                logger.error("Cannot read '%s': %s", file_path.name, e)
                stats["failed"] += 1
                continue

            # Skip unchanged files
            stored = state.get(file_key, {})
            if stored.get("hash") == current_hash:
                logger.debug("Skip (unchanged): %s", file_path.name)
                stats["skipped"] += 1
                continue

            # Parse → embed → upsert
            try:
                candidate_id = get_candidate_id(file_path)
                parsed = parse_file(file_path, candidate_id)

                if parsed is None:
                    # Unsupported extension or empty — already warned in parser
                    stats["skipped"] += 1
                    continue

                n_chunks = embed_and_upsert(qdrant, model, parsed, COLLECTION)

                # Update state entry
                state[file_key] = {
                    "hash":         current_hash,
                    "candidate_id": candidate_id,
                    "name":         parsed.name,
                    "chunks":       n_chunks,
                    "indexed_at":   datetime.now(timezone.utc).isoformat(),
                }
                stats["indexed"] += 1
                logger.info("✓  %s  →  '%s'  (%d chunks)", file_path.name, parsed.name, n_chunks)

                # Periodic checkpoint — saves progress so a crash doesn't
                # force a full re-index from scratch on the next run.
                if stats["indexed"] % _CHECKPOINT_EVERY == 0:
                    save_state(STATE_PATH, state)
                    logger.info(
                        "[Checkpoint] State saved — %d indexed so far",
                        stats["indexed"],
                    )

            except Exception as e:
                # Per-file fault isolation: log and continue
                logger.error("✗  %s  →  %s: %s", file_path.name, type(e).__name__, e)
                stats["failed"] += 1
                continue

        # ── Stale file cleanup ───────────────────────────────────
        # If a CV file was deleted from the folder, its Qdrant vectors
        # and state entry must be removed so it no longer appears in results.
        discovered_keys = {str(f.resolve()) for f in all_files}
        stale_keys = [k for k in list(state.keys()) if k not in discovered_keys]

        if stale_keys:
            logger.info("Cleaning up %d removed CV file(s)...", len(stale_keys))
            for key in stale_keys:
                cid = state[key].get("candidate_id")
                name = state[key].get("name", key)
                if cid:
                    try:
                        qdrant.delete(
                            collection_name=COLLECTION,
                            points_selector=FilterSelector(
                                filter=Filter(
                                    must=[
                                        FieldCondition(
                                            key="candidate_id",
                                            match=MatchValue(value=cid),
                                        )
                                    ]
                                )
                            ),
                            wait=True,
                        )
                        logger.info("Removed from Qdrant: '%s' (candidate_id=%s)", name, cid)
                    except Exception as e:
                        logger.warning("Could not remove Qdrant points for '%s': %s", name, e)
                del state[key]
            stats["cleaned"] = len(stale_keys)
        else:
            stats["cleaned"] = 0

    finally:
        # Always save state on exit — whether normal, crash, or Ctrl+C.
        # This ensures partial progress is never lost.
        save_state(STATE_PATH, state)
        if lock_fd is not None:
            try:
                lock_fd.close()
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass  # Best-effort lock release

    # ── Print summary ─────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    logger.info("=" * 60)
    logger.info("Indexing Complete  (%.1fs)", elapsed)
    logger.info("  Scanned  : %d", stats["scanned"])
    logger.info("  Indexed  : %d  (new or modified)", stats["indexed"])
    logger.info("  Skipped  : %d  (unchanged or unsupported)", stats["skipped"])
    logger.info("  Failed   : %d", stats["failed"])
    logger.info("  Cleaned  : %d  (removed CV files purged from Qdrant)", stats.get("cleaned", 0))
    logger.info("=" * 60)

    if stats["failed"] > 0:
        logger.warning(
            "%d file(s) failed to index. Check the logs above for details.",
            stats["failed"],
        )

    if stats["indexed"] == 0 and stats["failed"] == 0:
        logger.info("Nothing new to index — all CVs are already up to date.")


if __name__ == "__main__":
    main()
