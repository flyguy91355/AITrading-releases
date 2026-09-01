"""GitHub Releases API client for the update-distribution repo, and
release-notes parsing. The HTTP call is injectable (http_get param) so
tests never hit the real network — mirrors this codebase's existing
injected-callable pattern (see classify_ticker() in
src/analytics/benchmark_store.py).
See docs/superpowers/specs/2026-08-11-update-available-feature-design.md."""

import re

import requests

_DEFAULT_TIMEOUT_SECS = 10

# A release tag we're willing to interpolate into a download URL: an
# optional "v", a dotted numeric version, and an optional pre-release/build
# suffix built only from characters that can't change the URL's shape.
# Anything with "/", "\", ":", whitespace, or ".." is refused outright.
_VALID_TAG_PATTERN = re.compile(r"v?\d+(?:\.\d+)*[0-9A-Za-z._+-]*")


def _validated_tag(tag) -> str:
    """Returns the stripped tag if it looks like a real release tag, else
    raises ValueError (GitHub #147).

    tag_name arrives from the GitHub API — a trust boundary, even for a
    self-controlled releases repo — and is interpolated straight into the
    URL a live production apply then downloads and unpacks over the running
    install. A tag containing "/" or ".." could point that URL at an
    entirely different path. Callers already treat a raise here the same as
    any other release-fetch failure ("no update info available")."""
    if not isinstance(tag, str):
        raise ValueError(f"Release tag_name is not a string: {tag!r}")
    stripped = tag.strip()
    if ".." in stripped or not _VALID_TAG_PATTERN.fullmatch(stripped):
        raise ValueError(f"Refusing to build a download URL for release tag: {tag!r}")
    return stripped


def parse_release_notes(body: str) -> tuple[str, str]:
    """Splits a release body into (severity, notes). The first line, if it
    matches 'severity: critical' or 'severity: routine' (case-insensitive),
    sets severity and is stripped from the notes; otherwise severity
    defaults to 'routine' and the whole body is the notes."""
    lines = body.split("\n", 1)
    first_line = lines[0].strip().lower()
    if first_line.startswith("severity:"):
        severity = first_line.split(":", 1)[1].strip()
        rest = lines[1] if len(lines) > 1 else ""
        notes = rest.strip()
        return severity, notes
    return "routine", body.strip()


def fetch_latest_release(repo: str, http_get=None) -> dict:
    """Fetches the latest release from the given public repo (e.g.
    'owner/AITrading-releases') via GitHub's public Releases API — no
    credential needed since the distribution repo is public. Raises on any
    HTTP error (caller decides how to handle — see web/app.py's
    /api/update-status, which treats a failure as 'no update info
    available' rather than crashing). Also raises ValueError if the
    returned tag_name doesn't validate — see _validated_tag."""
    getter = http_get or requests.get
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    response = getter(url, timeout=_DEFAULT_TIMEOUT_SECS)
    response.raise_for_status()
    data = response.json()
    severity, notes = parse_release_notes(data.get("body", ""))
    tag = _validated_tag(data["tag_name"])
    # Use the direct github.com archive URL rather than the API tarball_url —
    # api.github.com/tarball/... counts against the unauthenticated 60 req/hr
    # rate limit; the github.com/archive URL bypasses it entirely for public repos.
    download_url = f"https://github.com/{repo}/archive/refs/tags/{tag}.tar.gz"
    return {
        "tag_name": tag,
        "severity": severity,
        "notes": notes,
        "tarball_url": data["tarball_url"],
        "download_url": download_url,
    }


def fetch_recent_releases(repo: str, limit: int = 15, http_get=None) -> list[dict]:
    """Fetches the `limit` most recent releases from the given public repo (e.g.
    'owner/AITrading-releases') via GitHub's public Releases API's list endpoint —
    no credential needed, same as fetch_latest_release. Backs the About panel's
    Version History section (2026-08-20, owner request) — deliberately reuses the
    exact same release data Apply Update already produces, nothing new to author or
    maintain. GitHub's list endpoint already returns releases newest-first, so this
    preserves that order rather than re-sorting. Raises on any HTTP error, same
    caller-decides-how-to-handle contract as fetch_latest_release."""
    getter = http_get or requests.get
    url = f"https://api.github.com/repos/{repo}/releases"
    response = getter(url, timeout=_DEFAULT_TIMEOUT_SECS, params={"per_page": limit})
    response.raise_for_status()
    releases = []
    for data in response.json():
        severity, notes = parse_release_notes(data.get("body", ""))
        releases.append({
            "tag_name": data["tag_name"],
            "severity": severity,
            "notes": notes,
            "published_at": data.get("published_at"),
        })
    return releases
