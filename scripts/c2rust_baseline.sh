#!/usr/bin/env bash
# Generate a disposable C2Rust parity baseline for one vendored IMOD C source.
# The output is reference material only; integrate functions into src/imod/ by
# source mapping rather than committing this generated crate.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 IMOD/path/to/source.c output-directory" >&2
  exit 2
fi

source_file=$1
output_directory=$2
config_directory=$(mktemp -d /tmp/imod-rs-c2rust-config.XXXXXX)
trap 'rm -rf "$config_directory"' EXIT
ln -s "$PWD/IMOD/sysdep/win/VC-imodconfig.h" "$config_directory/imodconfig.h"

c2rust transpile --emit-modules --preserve-unused-functions \
  --output-dir "$output_directory" "$source_file" -- \
  -D__int64='long long' -I"$config_directory" -IIMOD/include \
  -IIMOD/libiimod -IIMOD/libcfshr
