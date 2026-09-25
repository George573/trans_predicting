#!/usr/bin/env bash
# libcatboostmodel для cgo. Версия как у catboost в pip (1.2.10), .cbm совместим.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p lib
V=1.2.10
if [[ $(uname -s) == Linux ]]; then
  curl -fsSL -o lib/libcatboostmodel.so "https://github.com/catboost/catboost/releases/download/v$V/libcatboostmodel-linux-x86_64-$V.so"
else
  curl -fsSL -o lib/libcatboostmodel.dylib "https://github.com/catboost/catboost/releases/download/v$V/libcatboostmodel-darwin-universal2-$V.dylib"
fi
ls -la lib
