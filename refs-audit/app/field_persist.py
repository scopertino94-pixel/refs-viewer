"""Persist decoded GRIB field cache (.npz files) across container restarts.

On cpu-basic Spaces every restart wipes /tmp, forcing a full GRIB-fetch +
decode cycle (3-5 s/field) on every cold start.  This module adds a second
level of cache backed by a private HF Dataset so decoded fields survive
restarts, factory reboots, and sleep/wake cycles.

Key design decisions
--------------------
* **Size-based key, not mtime-based.**  The local field cache keys by mtime so
  partial-GRIB self-heal appends invalidate naturally.  Across restarts the
  same S3 file is re-downloaded with a new mtime, making every local cache key
  invalid even though the content is identical.  The remote key uses file size
  instead of mtime — size is stable across re-downloads and still invalidates
  when self-heal grows the file.

* **No LFS bloat.**  The dataset is initialised with a .gitattributes that opts
  .npz out of LFS tracking.  Files ≤100 MB are stored as regular git objects
  and git deduplicates by content hash — re-uploading an unchanged field costs
  zero additional storage.

* **Best-effort only.**  Every network call is wrapped; a failure never blocks
  a render.  The local disk cache is always written first and takes priority.

* **Instant revert.**  Set the Space variable  PERSIST_FIELDS=0  (or unset it)
  and this module becomes a no-op.  No code change required.

Env vars
--------
  PERSIST_FIELDS          "1" to enable (default: off)
  FIELDS_REPO             HF dataset repo id (default: <whoami>/refs-fields)
  HF_TOKEN                write token — set as a Space secret
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("refs-viewer.field_persist")

# How many fields to batch-download on startup (most-recent run only).
_WARM_LIMIT = 300
# Remote operations timeout (seconds).
_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def enabled() -> bool:
    return os.environ.get("PERSIST_FIELDS", "").strip() == "1"


def _token() -> Optional[str]:
    return (os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or None)


def _repo_id() -> Optional[str]:
    repo = os.environ.get("FIELDS_REPO", "").strip()
    if repo:
        return repo
    try:
        from huggingface_hub import HfApi
        who = HfApi().whoami(token=_token())
        name = who.get("name")
        if name:
            return f"{name}/refs-fields"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Stable cache key (size-based, survives container restarts)
# ---------------------------------------------------------------------------

_STABLE_VERSION = 1


def stable_key(filepath: Path, spec: dict, step, thresh, below) -> Optional[str]:
    """Return a hex key that is stable across container restarts for the same
    GRIB content.  Returns None if the file size cannot be read (uncacheable)."""
    try:
        size = os.path.getsize(str(filepath))
    except OSError:
        return None
    raw = repr((
        _STABLE_VERSION,
        filepath.name,   # encodes date/run/fhr in the filename
        size,
        sorted((k, str(v)) for k, v in spec.items()),
        step, thresh, below,
    ))
    return hashlib.sha1(raw.encode()).hexdigest()[:24]


def remote_path(key: str) -> str:
    """Path inside the dataset repo for a field file."""
    return f"fields/{key}.npz"


def remote_grid_path(grid_key: str) -> str:
    return f"fields/grid_{grid_key}.npz"


# ---------------------------------------------------------------------------
# Dataset initialisation (run once on startup)
# ---------------------------------------------------------------------------

def _ensure_repo(api, repo: str) -> bool:
    """Create the dataset repo if it doesn't exist; return True on success."""
    try:
        api.create_repo(repo_id=repo, repo_type="dataset",
                        private=True, exist_ok=True, token=_token())
    except Exception as e:
        log.warning("field_persist: create_repo failed: %s", e)
        return False

    # Upload .gitattributes once to opt .npz out of LFS tracking.
    # Without this, files >10 MB are stored as LFS blobs and every upload
    # adds a new blob — even if content is identical.  As regular git objects,
    # deduplication is by SHA-1 content hash so re-uploading unchanged fields
    # costs nothing.
    try:
        from huggingface_hub.utils import EntryNotFoundError
        try:
            api.hf_hub_download(repo_id=repo, repo_type="dataset",
                                filename=".gitattributes", token=_token())
        except EntryNotFoundError:
            attrs = "*.npz filter=lfs=false diff=lfs merge=lfs -text\n"
            api.upload_file(
                path_or_fileobj=attrs.encode(),
                path_in_repo=".gitattributes",
                repo_id=repo, repo_type="dataset",
                token=_token(),
                commit_message="disable LFS for .npz (content-addressed, deduped by git)",
            )
    except Exception as e:
        log.warning("field_persist: .gitattributes init failed: %s", e)
    return True


# ---------------------------------------------------------------------------
# Startup warm: download fields for the current run into local cache
# ---------------------------------------------------------------------------

def warm_on_startup(fields_dir: Path) -> int:
    """Download cached fields from the dataset into `fields_dir`.
    Returns the number of files restored.  Best-effort."""
    if not enabled():
        return 0
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    except Exception as e:
        log.warning("field_persist: huggingface_hub unavailable: %s", e)
        return 0

    repo = _repo_id()
    if not repo:
        log.warning("field_persist: could not resolve repo id")
        return 0

    api = HfApi()
    if not _ensure_repo(api, repo):
        return 0

    fields_dir.mkdir(parents=True, exist_ok=True)
    restored = 0
    try:
        items = list(api.list_repo_files(
            repo_id=repo, repo_type="dataset", token=_token()
        ))
    except Exception as e:
        log.warning("field_persist: list_repo_files failed: %s", e)
        return 0

    # Filter to .npz files not already on disk, bounded by _WARM_LIMIT.
    candidates = [
        f for f in items
        if f.startswith("fields/") and f.endswith(".npz")
        and not (fields_dir / os.path.basename(f)).exists()
    ][:_WARM_LIMIT]

    t0 = time.monotonic()
    for remote in candidates:
        local_name = os.path.basename(remote)
        local_path = fields_dir / local_name
        if local_path.exists():
            continue
        try:
            dl = hf_hub_download(
                repo_id=repo, repo_type="dataset",
                filename=remote, token=_token(),
            )
            import shutil
            shutil.copy2(dl, local_path)
            restored += 1
        except (EntryNotFoundError, RepositoryNotFoundError):
            continue
        except Exception as e:
            log.debug("field_persist: warm download failed for %s: %s", remote, e)
            continue

    elapsed = time.monotonic() - t0
    log.info("field_persist: warmed %d/%d fields in %.1fs from %s",
             restored, len(candidates), elapsed, repo)
    return restored


# ---------------------------------------------------------------------------
# Per-field remote read (L2 cache lookup before GRIB decode)
# ---------------------------------------------------------------------------

def try_load_remote(key: str, fields_dir: Path) -> bool:
    """Attempt to download a single field from the dataset into fields_dir.
    Returns True if the file now exists locally (either already there or just
    downloaded).  Never raises."""
    local_path = fields_dir / f"{key}.npz"
    if local_path.exists():
        return True
    if not enabled():
        return False

    repo = _repo_id()
    if not repo:
        return False
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
        dl = hf_hub_download(
            repo_id=repo, repo_type="dataset",
            filename=remote_path(key), token=_token(),
        )
        import shutil
        shutil.copy2(dl, local_path)
        return True
    except (EntryNotFoundError, RepositoryNotFoundError):
        return False
    except Exception as e:
        log.debug("field_persist: remote load failed for %s: %s", key, e)
        return False


# ---------------------------------------------------------------------------
# Per-field remote write (async, after local save)
# ---------------------------------------------------------------------------

async def push_field_async(key: str, local_path: Path,
                           grid_key: Optional[str] = None,
                           grid_path: Optional[Path] = None) -> None:
    """Upload a field (and optionally its grid file) to the dataset.
    Runs in a thread so it never blocks the event loop.  Best-effort."""
    if not enabled():
        return
    repo = _repo_id()
    if not repo:
        return

    async def _upload(local: Path, repo_path: str) -> None:
        if not local.exists():
            return
        try:
            await asyncio.to_thread(_upload_sync, local, repo_path, repo)
        except Exception as e:
            log.debug("field_persist: push failed %s: %s", repo_path, e)

    await _upload(local_path, remote_path(key))
    if grid_key and grid_path:
        await _upload(grid_path, remote_grid_path(grid_key))


def _upload_sync(local: Path, repo_path: str, repo: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    _ensure_repo(api, repo)
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=repo_path,
        repo_id=repo, repo_type="dataset",
        token=_token(),
        commit_message=f"field cache: {local.name}",
    )


# ---------------------------------------------------------------------------
# Cleanup: prune fields older than 48 h from the dataset
# ---------------------------------------------------------------------------

async def prune_old_fields_async(max_age_h: int = 48) -> None:
    """Delete dataset files whose names suggest they are from old runs.
    Best-effort, runs in background.  Only acts when PERSIST_FIELDS=1."""
    if not enabled():
        return
    try:
        await asyncio.to_thread(_prune_sync, max_age_h)
    except Exception as e:
        log.debug("field_persist: prune failed: %s", e)


def _prune_sync(max_age_h: int) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError
    repo = _repo_id()
    if not repo:
        return
    api = HfApi()
    cutoff = time.time() - max_age_h * 3600
    try:
        files = list(api.list_repo_files(
            repo_id=repo, repo_type="dataset", token=_token()
        ))
    except RepositoryNotFoundError:
        return
    except Exception as e:
        log.debug("field_persist: prune list failed: %s", e)
        return

    # Use commit timestamps to identify stale files.
    deleted = 0
    for f in files:
        if not f.startswith("fields/") or not f.endswith(".npz"):
            continue
        try:
            commits = list(api.list_repo_commits(
                repo_id=repo, repo_type="dataset",
                revision="main", token=_token(),
            ))
            # The first commit touching this file is most recent; skip if young.
            if commits and commits[0].created_at.timestamp() > cutoff:
                continue
            api.delete_file(
                path_in_repo=f, repo_id=repo, repo_type="dataset",
                token=_token(), commit_message=f"prune: {f}",
            )
            deleted += 1
        except Exception:
            continue
    if deleted:
        log.info("field_persist: pruned %d stale field files", deleted)
