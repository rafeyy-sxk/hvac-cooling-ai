#!/usr/bin/env bash
# Build infra/build/lambda.zip: this package plus its runtime dependencies as Linux arm64 wheels
# for the python3.12 Lambda runtime (Amazon Linux 2023, glibc 2.34, so manylinux_2_28 wheels load).
# Run from anywhere; needs pip and zip.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/infra/build"
PKG="$BUILD/pkg"
PYTHON="${PYTHON:-python3}"

rm -rf "$PKG" "$BUILD/lambda.zip"
mkdir -p "$PKG"

# Runtime dependencies only (the anthropic SDK is not needed: the function serves the tools, not the LLM).
"$PYTHON" -m pip install --quiet --target "$PKG" \
  --platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64 --implementation cp --python-version 3.12 --only-binary=:all: \
  "numpy>=1.26" "scikit-learn==1.9.1" "joblib>=1.3"
"$PYTHON" -m pip install --quiet --target "$PKG" --no-deps "$ROOT"

find "$PKG" -type d -name "__pycache__" -prune -exec rm -rf {} +
(cd "$PKG" && zip -q -r -9 "$BUILD/lambda.zip" .)

# Lambda's limit for a zip package, unzipped, is 250 MB (262,144,000 bytes).
LIMIT=262144000
UNZIPPED=$(unzip -l "$BUILD/lambda.zip" | tail -1 | awk '{print $1}')
ZIPPED=$(wc -c < "$BUILD/lambda.zip" | tr -d ' ')
echo "unzipped: $UNZIPPED bytes (Lambda limit $LIMIT)"
echo "zipped:   $ZIPPED bytes -> $BUILD/lambda.zip (over 50 MB, so Terraform uploads it via S3)"
if [ "$UNZIPPED" -gt "$LIMIT" ]; then
  echo "package is over the Lambda unzipped size limit" >&2
  exit 1
fi
