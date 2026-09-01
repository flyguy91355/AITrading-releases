"""Applying a downloaded release to a live install: what's safe to touch,
and (Task 4) how the archive gets extracted and copied. Deny-list always
wins over allow-list, per the spec's 'never touch instance data' rule.
See docs/superpowers/specs/2026-08-11-update-available-feature-design.md."""

import logging
import ntpath
import shutil
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Deny-list entries. A trailing "/" means "this directory and everything
# under it"; a bare entry means that exact path (or anything nested under
# it), never a longer sibling name -- see _matches_path_entry (GitHub #146:
# "config/settings.yaml" used to be a raw string-prefix match, the odd one
# out among its directory-style siblings).
DENIED_PATH_PREFIXES = (
    ".env",
    "data/",
    "config/settings.yaml",
    "certs/",
)

# Deliberately looser than the boundary rule above, and deliberately kept
# that way: ".env.local"/".env.production" are real credential files, so any
# file whose basename starts with ".env" is denied outright no matter where
# it sits. A deny-list erring wide costs nothing (nothing here is ever meant
# to be updatable); erring narrow leaks credentials.
DENIED_FILENAME_PREFIXES = (".env",)

ALLOWED_PATH_PREFIXES = (
    "src/",
    "web/",
    "scripts/",
    "requirements.txt",
    "requirements-lock.txt",
)


def _matches_path_entry(normalized: str, entry: str) -> bool:
    """Directory-boundary-aware match for one allow/deny entry.

    A trailing-slash entry ("src/") matches anything inside that directory.
    A bare entry ("requirements.txt") matches that exact path, or something
    genuinely nested under it -- never a longer sibling name such as
    "requirements.txt.bak", which a plain .startswith() would have accepted
    into the allow-list (GitHub #145)."""
    if entry.endswith("/"):
        return normalized.startswith(entry)
    return normalized == entry or normalized.startswith(entry + "/")


def _is_traversal_unsafe(normalized: str) -> bool:
    """True for any path that isn't a plain relative path inside the install
    root: empty, absolute (POSIX, Windows drive, or UNC), or containing a
    ".." component.

    This is deliberately duplicated defense (GitHub #127) rather than
    relying solely on tarfile's filter="data" in extract_release_archive():
    a future switch to a zip archive, a hand-built source tree, or a Python
    fallback that drops the filter= kwarg would otherwise silently remove
    every traversal protection this module has."""
    if not normalized:
        return True
    if normalized.startswith("/"):
        # POSIX absolute, and "//server/share" UNC.
        return True
    if ntpath.splitdrive(normalized)[0]:
        # "C:/..." drive-qualified (ntpath, not os.path, so this behaves
        # identically on the Linux production box and Windows local dev).
        return True
    return any(part == ".." for part in normalized.split("/"))


def is_path_updatable(relative_path: str) -> bool:
    """True iff an update is allowed to overwrite this path (relative to
    the install root). Checked in order: traversal/absolute-path rejection,
    then the deny-list (always wins over the allow-list), then the explicit
    allow-list, then a fallback rule for a bare top-level *.py file (e.g.
    start.py)."""
    normalized = relative_path.replace("\\", "/")

    if _is_traversal_unsafe(normalized):
        return False

    basename = normalized.rsplit("/", 1)[-1]
    for denied_name in DENIED_FILENAME_PREFIXES:
        if basename.startswith(denied_name):
            return False

    for denied in DENIED_PATH_PREFIXES:
        if _matches_path_entry(normalized, denied):
            return False

    for allowed in ALLOWED_PATH_PREFIXES:
        if _matches_path_entry(normalized, allowed):
            return True

    if "/" not in normalized and normalized.endswith(".py"):
        return True

    return False


def requirements_changed(old_content: str, new_content: str) -> bool:
    """True iff the two requirements.txt contents differ, ignoring leading/
    trailing whitespace (a trailing-newline-only diff shouldn't trigger a
    real pip install)."""
    return old_content.strip() != new_content.strip()


def extract_release_archive(archive_path: str, dest_dir: str) -> str:
    """Extracts a .tar.gz release archive into dest_dir and returns the
    path to its single top-level directory (GitHub's auto-generated
    release tarballs always have exactly one)."""
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir, filter="data")
    dest_path = Path(dest_dir)
    top_level_dirs = [entry for entry in dest_path.iterdir() if entry.is_dir()]
    if len(top_level_dirs) != 1:
        raise ValueError(
            f"Expected exactly one top-level directory in the release archive, "
            f"found {len(top_level_dirs)}"
        )
    return str(top_level_dirs[0])


def _safe_destination(target_root: Path, relative: str) -> Path | None:
    """Where `relative` may actually be written under target_root, or None
    if writing there wouldn't land where the allow-list thinks it does.

    shutil.copy2 FOLLOWS a symlink at the destination and writes straight
    through it, so a symlink planted at an allow-listed path (by a prior
    release, or a compromised one) pointing at .env or config/settings.yaml
    would overwrite that denied file while every string-level check still
    reported "src/... — allowed" (GitHub #128). Three refusals, covering
    both the symlinked file itself and a symlinked parent directory:
    a destination that IS a symlink, one whose real location escapes the
    install root entirely, and one whose real location inside the root is
    no longer an allow-listed path."""
    destination = target_root / relative
    if destination.is_symlink():
        return None
    try:
        resolved_root = target_root.resolve()
        # strict=False: the destination usually doesn't exist yet, but any
        # symlink in the parent chain that DOES exist still gets resolved.
        real_relative = destination.resolve().relative_to(resolved_root).as_posix()
    except (OSError, ValueError):
        return None
    if not is_path_updatable(real_relative):
        return None
    return destination


def copy_updatable_files(source_dir: str, target_dir: str) -> list[str]:
    """Walks source_dir, copies every file whose path (relative to
    source_dir) passes is_path_updatable() into the same relative path
    under target_dir, creating parent directories as needed. Returns the
    sorted list of relative paths actually copied. Never touches anything
    outside that allow-list, even if the source tree contains a denied
    path (e.g. a stray data/ directory in a malformed archive) — the deny
    check in is_path_updatable() is authoritative regardless of what's on
    disk in target_dir already. A destination that would resolve somewhere
    other than where its relative path says (a symlink, or a symlinked
    parent directory) is skipped and logged, never written through — see
    _safe_destination()."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    copied: list[str] = []

    for file_path in source_path.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(source_path).as_posix()
        if not is_path_updatable(relative):
            continue
        destination = _safe_destination(target_path, relative)
        if destination is None:
            logger.warning(
                "Update: refusing to write %s — the destination is a symlink or "
                "resolves outside the allow-listed install path",
                relative,
            )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination)
        copied.append(relative)

    return sorted(copied)
