#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <google-drive-package-dir> [dataset-dir]" >&2
  exit 2
fi

package_dir="$1"
dataset_dir="${2:-/Users/minho/Documents/Dataset}"
database_archive="$package_dir/database/univariate_ts.db.zst"
artifact_archive="$package_dir/artifacts/exp403_408_required_artifacts.tar.zst"
checksum_file="$package_dir/SHA256SUMS.txt"

for required in "$database_archive" "$artifact_archive" "$checksum_file"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing handoff file: $required" >&2
    exit 1
  fi
done

command -v zstd >/dev/null 2>&1 || {
  echo "zstd is required. Install it with: brew install zstd" >&2
  exit 1
}

mkdir -p "$dataset_dir"

(
  cd "$package_dir"
  shasum -a 256 -c SHA256SUMS.txt
)

database_target="$dataset_dir/univariate_ts.db"
database_temp="$dataset_dir/.univariate_ts.db.restore.tmp"
rm -f "$database_temp"
zstd -dc "$database_archive" > "$database_temp"
sqlite3 "$database_temp" "PRAGMA quick_check;" | grep -qx "ok"
mv "$database_temp" "$database_target"

tar --use-compress-program=unzstd -xf "$artifact_archive" -C "$dataset_dir"

echo "Restore complete"
echo "Database: $database_target"
echo "Artifacts: $dataset_dir"
