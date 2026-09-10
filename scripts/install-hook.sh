#!/usr/bin/env sh
set -eu

HOOK_SRC="scripts/pre-commit.sh"
HOOK_DEST=".git/hooks/pre-commit"

if [ ! -d .git ]; then
  echo ".git directory not found. Run this from repository root."
  exit 1
fi

cp "$HOOK_SRC" "$HOOK_DEST"
chmod +x "$HOOK_DEST"
echo "Installed pre-commit hook to $HOOK_DEST"
