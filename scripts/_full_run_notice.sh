#!/usr/bin/env bash
set -euo pipefail

label="$1"
shift

missing=()
for var in "$@"; do
  if [[ -z "${!var:-}" ]]; then
    missing+=("$var")
  fi
done

if (( ${#missing[@]} )); then
  printf 'Full external %s run not launched; missing: %s\n' "$label" "${missing[*]}"
else
  printf 'Full external %s prerequisites are set. Use the configs and reproduce/ entry points documented in docs/REPRODUCIBILITY.md.\n' "$label"
fi
