#!/usr/bin/env bash
# Rebuild Phase 0 CCC evidence; this is an audit tool, not translated IMOD code.
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ccc=${CCC_RS:-/data/henriksson/github/claude/code-complexity-comparator/target/release/ccc-rs}

if [[ ! -x $ccc ]]; then
  echo "CCC_RS must name an executable ccc-rs binary; got: $ccc" >&2
  exit 2
fi

cd "$root"
mkdir -p audit/source audit/order

"$ccc" analyze IMOD/clip -l cpp --recurse -o audit/source/clip.cpp.json
"$ccc" analyze IMOD/flib/image/newstack.f90 -l fortran -o audit/source/newstack.fortran.json
"$ccc" analyze IMOD/flib/image/binvol.f90 -l fortran -o audit/source/binvol.fortran.json
"$ccc" analyze IMOD/flib/image/header.f90 -l fortran -o audit/source/header.fortran.json
"$ccc" analyze IMOD/flib/image/alterheader.f90 -l fortran -o audit/source/alterheader.fortran.json
"$ccc" analyze IMOD/flib/model/convertmod.f -l fortran -o audit/source/convertmod.fortran.json
"$ccc" analyze IMOD/mrc/tif2mrc.c -l c -o audit/source/tif2mrc.c.json
"$ccc" analyze IMOD/qttools/mrc2tif/mrc2tif.cpp -l cpp -o audit/source/mrc2tif.cpp.json
"$ccc" analyze IMOD/imodutil/imodinfo.cpp -l cpp -o audit/source/imodinfo.cpp.json
"$ccc" analyze IMOD/imodutil/imodjoin.c -l c -o audit/source/imodjoin.c.json
"$ccc" analyze IMOD/imodutil/wmod2imod.c -l c -o audit/source/wmod2imod.c.json
"$ccc" analyze IMOD/libiimod/mrcfiles.c -l c -o audit/source/mrcfiles.c.json
"$ccc" analyze IMOD/libiimod/iimage.c -l c -o audit/source/iimage.c.json
"$ccc" analyze IMOD/libiimod/iimrc.c -l c -o audit/source/iimrc.c.json
"$ccc" analyze IMOD/libiimod/halffloat.c -l c -o audit/source/halffloat.c.json
"$ccc" analyze IMOD/libiimod/mrcsec.c -l c -o audit/source/mrcsec.c.json
"$ccc" analyze IMOD/libcfshr/b3dutil.c -l c -o audit/source/b3dutil.c.json
"$ccc" analyze IMOD/libcfshr/ilist.c -l c -o audit/source/ilist.c.json
"$ccc" analyze IMOD/libcfshr/islice.c -l c -o audit/source/islice.c.json
"$ccc" analyze src/imod -l rust --recurse -o audit/rust-all.json
"$ccc" analyze src/imod/clip -l rust --recurse -o audit/rust-clip.json

"$ccc" order IMOD/clip -l cpp --recurse --strict -o audit/order/clip.csv
"$ccc" order IMOD/flib/image/newstack.f90 -l fortran --strict -o audit/order/newstack.csv
"$ccc" order IMOD/mrc/tif2mrc.c -l c --strict -o audit/order/tif2mrc.csv
"$ccc" order IMOD/qttools/mrc2tif/mrc2tif.cpp -l cpp --strict -o audit/order/mrc2tif.csv
"$ccc" order IMOD/imodutil/imodinfo.cpp -l cpp --strict -o audit/order/imodinfo.csv
"$ccc" order IMOD/libiimod/mrcfiles.c -l c --strict -o audit/order/mrcfiles.csv
"$ccc" order IMOD/libiimod/iimage.c -l c --strict -o audit/order/iimage.csv
"$ccc" order IMOD/libiimod/iimrc.c -l c --strict -o audit/order/iimrc.csv
"$ccc" order IMOD/libiimod/halffloat.c -l c --strict -o audit/order/halffloat.csv
"$ccc" order IMOD/libiimod/mrcsec.c -l c --strict -o audit/order/mrcsec.csv
"$ccc" order IMOD/libcfshr/ilist.c -l c --strict -o audit/order/ilist.csv
"$ccc" order IMOD/libcfshr/islice.c -l c --strict -o audit/order/islice.csv
"$ccc" missing audit/rust-clip.json audit/source/clip.cpp.json --mapping ccc_mapping.toml --format json > audit/clip-missing.json
"$ccc" compare-structs audit/rust-clip.json audit/source/clip.cpp.json --format json > audit/clip-structs.json
"$ccc" analyze src/imod/libiimod/mrcfiles.rs -l rust -o audit/rust-mrcfiles.json
"$ccc" analyze src/imod/libiimod/iimage.rs -l rust -o audit/rust-iimage.json
"$ccc" analyze src/imod/libiimod/iimrc.rs -l rust -o audit/rust-iimrc.json
"$ccc" analyze src/imod/libiimod/halffloat.rs -l rust -o audit/rust-halffloat.json
"$ccc" analyze src/imod/libiimod/mrcsec.rs -l rust -o audit/rust-mrcsec.json
"$ccc" analyze src/imod/libcfshr/b3dutil.rs -l rust -o audit/rust-b3dutil.json
"$ccc" analyze src/imod/libcfshr/ilist.rs -l rust -o audit/rust-ilist.json
"$ccc" analyze src/imod/libcfshr/islice.rs -l rust -o audit/rust-islice.json
"$ccc" missing audit/rust-mrcfiles.json audit/source/mrcfiles.c.json --mapping ccc_mapping.toml --format json > audit/mrcfiles-missing.json
"$ccc" missing audit/rust-iimage.json audit/source/iimage.c.json --mapping ccc_mapping.toml --format json > audit/iimage-missing.json
"$ccc" missing audit/rust-iimrc.json audit/source/iimrc.c.json --mapping ccc_mapping.toml --format json > audit/iimrc-missing.json
"$ccc" missing audit/rust-halffloat.json audit/source/halffloat.c.json --mapping ccc_mapping.toml --format json > audit/halffloat-missing.json
"$ccc" missing audit/rust-mrcsec.json audit/source/mrcsec.c.json --mapping ccc_mapping.toml --format json > audit/mrcsec-missing.json
"$ccc" missing audit/rust-ilist.json audit/source/ilist.c.json --mapping ccc_mapping.toml --format json > audit/ilist-missing.json
"$ccc" missing audit/rust-islice.json audit/source/islice.c.json --mapping ccc_mapping.toml --format json > audit/islice-missing.json

# Enrich bottom-up orders with the source/Rust counterpart selected by the
# explicit mapping (or documented normalization).  `translated` remains a
# reviewer-maintained progress bit and is carried forward by `order --merge`.
for unit in clip newstack tif2mrc mrc2tif imodinfo mrcfiles iimage iimrc halffloat mrcsec ilist islice; do
  case "$unit" in
    clip) source=audit/source/clip.cpp.json ;;
    newstack) source=audit/source/newstack.fortran.json ;;
    tif2mrc) source=audit/source/tif2mrc.c.json ;;
    mrc2tif) source=audit/source/mrc2tif.cpp.json ;;
    imodinfo) source=audit/source/imodinfo.cpp.json ;;
    mrcfiles) source=audit/source/mrcfiles.c.json ;;
    iimage) source=audit/source/iimage.c.json ;;
    iimrc) source=audit/source/iimrc.c.json ;;
    halffloat) source=audit/source/halffloat.c.json ;;
    mrcsec) source=audit/source/mrcsec.c.json ;;
    ilist) source=audit/source/ilist.c.json ;;
    islice) source=audit/source/islice.c.json ;;
  esac
  "$ccc" order-annotate "audit/order/$unit.csv" --source "$source" --rust audit/rust-all.json \
    --mapping ccc_mapping.toml -o "audit/order/$unit.annotated.csv"
done
