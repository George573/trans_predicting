#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p lib
V=1.2.10
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) A=linux-x86_64; F=libcatboostmodel.so ;;
  Linux-aarch64) A=linux-aarch64; F=libcatboostmodel.so ;;
  Darwin-*) A=darwin-universal2; F=libcatboostmodel.dylib ;;
  *) echo "unsupported platform $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
curl -fsSL -o "lib/$F" "https://github.com/catboost/catboost/releases/download/v$V/libcatboostmodel-$A-$V.${F##*.}"
ls -la lib
