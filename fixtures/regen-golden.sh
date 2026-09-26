#!/bin/bash
# Regenerate fixtures/<suite>/golden.manifest from the native reference.
#
#   fixtures/regen-golden.sh <suite> [args for the make script]
#   KEEP=1 fixtures/regen-golden.sh <suite>     # leave the native files
#   RECORD_ONLY=1 fixtures/regen-golden.sh <suite>
#                         # skip the make script: record from native files
#                         # already under fixtures/<suite>/golden/
#   DIFF=1 fixtures/regen-golden.sh <suite>
#                         # exhaustive native differential instead: the make
#                         # script also runs the `#full` rows pruned from the
#                         # ordinary suite (FULL=1), the tests run every row
#                         # against those native files (IMOD_RS_GOLDEN_NATIVE=1,
#                         # IMOD_RS_FULL_CASES=1) and nothing is recorded; the
#                         # native files are left in place
#
# Three steps (fixtures/README.md, "Goldens"):
#  1. fixtures/make-<suite>-goldens.sh runs the native programs and writes their
#     full outputs to fixtures/<suite>/golden/ (git-ignored), exactly as before;
#  2. the suite's tests run with IMOD_RS_GOLDEN_RECORD=1, which makes every
#     comparison read those native files, compare exactly as the test always
#     did, and record the expected bytes (whole, or as a SHA-256 after the
#     suite's masking) into fixtures/<suite>/golden.manifest;
#  3. the native files are deleted, unless KEEP=1 — keep them to debug a
#     mismatch (diff them against the output of the failing case).
#
# `defined/` expectations (upstream bugs fixed in translation, produced by our
# own build) are kept from the old manifest unless a make script with a
# `defined` mode regenerates them (`make-<suite>-goldens.sh defined`).
#
# Suite "." is the loose files at the top of fixtures/ (fixtures/golden.manifest).
set -e
suite=${1:?usage: fixtures/regen-golden.sh <suite> [make args]}
shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# Make script and test targets for each suite whose names do not follow
# make-<suite>-goldens.sh and <suite>_cli.
case $suite in
  densmatch|findcontrast) make=fixtures/make-densmatch-findcontrast-goldens.sh ;;
  .) make= ;;
  *) make=fixtures/make-$suite-goldens.sh ;;
esac
# (The tests listed are those whose goldens live in fixtures/<suite>.  Some
# suites also read another's *inputs* -- trimvol and findcontrast read
# densmatch's, mrcinfo reads mrcx's, refinematch findwarp's, warpvol
# matchvol's mb.mrc, findsection imodtrans's multi.mod -- so after changing a
# suite's inputs, regenerate those suites too.)
case $suite in
  xftransforms) tests="xftoxg_cli xfproduct_cli" ;;
  xml) tests="xml_corpus" ;;
  comrun) tests="comrun" ;;
  .) tests="newstack_warp newstack_warp_chunked newstack_reduce newstack_replace \
newstack_mixed_inputs newstack_mdoc mrcsec_sections model_chunk_fixture usage_text" ;;
  *) tests="${suite}_cli" ;;
esac

if [ -z "$RECORD_ONLY" ]; then
  if [ -z "$make" ] || [ ! -e "$make" ]; then
    echo "no make script for $suite: put the native files under fixtures/$suite/golden/" >&2
    echo "yourself and rerun with RECORD_ONLY=1" >&2
    exit 1
  fi
  [ -n "$DIFF" ] && export FULL=1
  case $make in *.py) python3 "$make" "$@" ;; *) bash "$make" "$@" ;; esac
  # A suite with a defined.list has a `defined` mode that writes defined/
  # from our own build (target/release/imod) for the cases it names.
  if [ -f "fixtures/$suite/defined.list" ] && grep -q '= defined' "$make"; then
    bash "$make" defined
  fi
fi

if [ -n "$DIFF" ]; then
  status=0
  for t in $tests; do
    IMOD_RS_GOLDEN_NATIVE=1 IMOD_RS_FULL_CASES=1 \
      cargo test --release -q --test "$t" -- --test-threads=1 || status=1
  done
  echo "native differential for $suite done (status $status); native files in fixtures/$suite/golden/"
  exit $status
fi

for t in $tests; do
  IMOD_RS_GOLDEN_RECORD=1 cargo test --release -q --test "$t" -- --test-threads=1
done

if [ -z "$KEEP" ]; then
  [ "$suite" = . ] || rm -rf "fixtures/$suite/golden" "fixtures/$suite/defined"
fi
echo "fixtures/$suite/golden.manifest recorded"
