#!/bin/bash
# Cuts a new release on the AITrading-releases distribution repo.
# Usage: scripts/cut_release.sh <version-tag> <critical|routine> <notes-file>
# Example: scripts/cut_release.sh v1.5.0 routine /tmp/release-notes.txt
#
# The notes file's content becomes the release body, with a machine-readable
# "severity: <level>" line prepended -- src/update/release_client.py's
# parse_release_notes() reads this line to set the severity shown on any
# install's dashboard.

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <version-tag> <critical|routine> <notes-file>"
    exit 1
fi

VERSION_TAG="$1"
SEVERITY="$2"
NOTES_FILE="$3"

if [ "$SEVERITY" != "critical" ] && [ "$SEVERITY" != "routine" ]; then
    echo "Severity must be 'critical' or 'routine', got: $SEVERITY"
    exit 1
fi

if [ ! -f "$NOTES_FILE" ]; then
    echo "Notes file not found: $NOTES_FILE"
    exit 1
fi

RELEASES_REPO=$(grep -A2 "^update:" config/settings.yaml | grep "releases_repo:" | sed 's/.*releases_repo:\s*"\?\([^"]*\)"\?/\1/')

if [ -z "$RELEASES_REPO" ]; then
    echo "Could not read update.releases_repo from config/settings.yaml"
    exit 1
fi

BODY_FILE=$(mktemp)
echo "severity: $SEVERITY" > "$BODY_FILE"
echo "" >> "$BODY_FILE"
cat "$NOTES_FILE" >> "$BODY_FILE"

gh release create "$VERSION_TAG" \
    --repo "$RELEASES_REPO" \
    --title "$VERSION_TAG" \
    --notes-file "$BODY_FILE"

rm -f "$BODY_FILE"

echo "Released $VERSION_TAG ($SEVERITY) to $RELEASES_REPO"
