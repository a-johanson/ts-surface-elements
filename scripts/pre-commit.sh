#!/usr/bin/env sh
set -eu

echo "Running TypeScript check (tsc)..."
if ! npx -y tsc --noEmit; then
  echo "TypeScript errors detected. Commit aborted."
  exit 1
fi

echo "Running biome check (format + lint)..."
if ! npx -y biome check .; then
  echo "Formatting/linting errors detected by biome. Commit aborted."
  exit 1
fi

echo "Pre-commit checks passed."
exit 0
