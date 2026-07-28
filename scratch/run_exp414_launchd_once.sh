#!/bin/bash
set -euo pipefail

trap 'launchctl remove com.openai.exp414.full >/dev/null 2>&1 || true' EXIT

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

"$@"
