//! Native-golden coverage for `imodchopconts` (`IMOD/imodutil/imodchopconts.cpp`).
//!
//! Every row of `fixtures/imodchopconts/cases.tsv` was run through the native
//! reference program by `fixtures/make-imodchopconts-goldens.sh`, with
//! standard output captured through a pipe.  `golden/<case>.rc` is the exit
//! status, `.stdout` the standard output and `.mod` the output model, when
//! native wrote one.  The input `col.mod` carries surface numbers, per-point
//! colour changes and object-level per-contour, per-surface and min/max data,
//! so length-based chopping, both `-colors` modes and the transfer of
//! fine-grained data are all exercised.  Models are compared byte for byte,
//! with native's heap residue in `Imod.name` reconciled (`BUGS.md` §2).

mod common;

use std::path::{Path, PathBuf};

const PROGRAM: &str = "imodchopconts";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/imodchopconts")
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(fixture_dir().join("col.mod"), dir.join("col.mod")).unwrap();
    dir
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, args) = (fields[0], fields[1]);
        let dir = scratch(name);
        let args: Vec<&str> = args.split_whitespace().collect();
        let output = common::imod_cmd(PROGRAM)
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args(&args)
            .output()
            .unwrap();
        count += 1;
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}\n{}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let stdout = common::golden::expect(&golden.join(format!("{name}.stdout")));
        if !stdout.matches(&output.stdout) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                stdout.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let expected = common::golden::load(&golden.join(format!("{name}.mod")));
        let written = std::fs::read(dir.join("o.mod")).ok();
        match (expected, written) {
            (None, None) => {}
            (Some(g), Some(w)) if g.matches_reconciled(&w, common::golden::identity) => {}
            (g, w) => failures.push(format!(
                "{name}: o.mod differs (native {:?}, ours {:?} bytes)",
                g.map(|b| b.describe()),
                w.map(|b| b.len())
            )),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 16, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// BUGS.md (imodchopconts): with `-colors 1` native recovers each surface
/// from its map key with `B3DNINT(color + surf * 2^25)`, an out-of-range
/// `int` conversion for surface numbers of 64 and up.  Defined behaviour: the
/// rounding is done in double, so a model whose surfaces are renumbered from
/// 64 splits exactly as the original does (golden `colorsurf`: 7 contours
/// become 17).
#[test]
fn colors_by_surface_handles_large_surface_numbers() {
    let dir = scratch("bigsurf");
    let mut bytes = std::fs::read(dir.join("col.mod")).unwrap();
    // CONT chunk: id, psize, flags, time, surf (big-endian ints).
    let mut at = 0;
    while let Some(k) = bytes[at..].windows(4).position(|w| w == b"CONT") {
        let surf = at + k + 16;
        let value = i32::from_be_bytes(bytes[surf..surf + 4].try_into().unwrap()) + 63;
        bytes[surf..surf + 4].copy_from_slice(&value.to_be_bytes());
        at = surf + 4;
    }
    std::fs::write(dir.join("big.mod"), &bytes).unwrap();
    let output = common::imod_cmd(PROGRAM)
        .current_dir(&dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args(["-colors", "1", "big.mod", "o.mod"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    let native = common::golden::read_to_string(&fixture_dir().join("golden/colorsurf.stdout"));
    assert_eq!(String::from_utf8_lossy(&output.stdout), native);
    let _ = std::fs::remove_dir_all(&dir);
}
