//! Native-golden coverage for `imodtrans` (`IMOD/imodutil/imodtrans.c`).
//!
//! Every case in `fixtures/imodtrans/cases.tsv` was run through the native
//! reference `imodtrans` by `fixtures/make-imodtrans-goldens.sh`; the exit
//! status is in the table and the output model, when native leaves one, is
//! `fixtures/imodtrans/golden/<case>.mod`.  The inputs are native-authored:
//! `multi.mod` by `wmod2imod` from `multi.wimp`, `meshed.mod` by `imodmesh`,
//! `flipped.mod` by `imodtrans -T`, and `img1/img2.mrc` by `raw2mrc` plus
//! `alterheader`; `BBa_erase.fid` and `BBa.xf` are read from the vendored
//! tree.
//!
//! Pruned 2026-09-26: 32 of 37 rows kept (dropped the scale-rotate-shift mix beside `rall`/`sxyz`, `-R 0` beside `-R 1`, `-R` and `-2 -S` repeated on the larger `BBa_erase.fid`, and `-tq`, the same invalid-option exit as `-q`); the rest stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).

mod common;

use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/imodtrans")
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-imodtrans-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() && path.file_name().unwrap() != "golden.manifest" {
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
    }
    let bb = Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD/Etomo/uitestData/BB");
    for name in ["BBa_erase.fid", "BBa.xf"] {
        std::fs::copy(bb.join(name), dir.join(name)).unwrap();
    }
    dir
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, input, args, rc) = (fields[0], fields[1], fields[2], fields[3]);
        let rc: i32 = rc.parse().unwrap();
        let dir = scratch(name);
        std::fs::copy(dir.join(input), dir.join("in")).unwrap();
        let output = common::imod_cmd("imodtrans")
            .current_dir(&dir)
            .args(args.split_whitespace())
            .args(["in", "out.mod"])
            .output()
            .unwrap();
        count += 1;
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}; stdout {}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let golden =
            common::golden::load(&fixture_dir().join("golden").join(format!("{name}.mod")));
        let written = std::fs::read(dir.join("out.mod")).ok();
        match (golden, written) {
            (None, None) => {}
            // `MINX` `oscale`/`orot` are native residue when `-i` supplies no
            // `IrefImage` (`imodtrans.c:93`, `:404`); reconciled against the
            // defined identity (`BUGS.md` §2).
            (Some(g), Some(w)) if g.matches_reconciled(&w, common::golden::identity) => {}
            (g, w) => failures.push(format!(
                "{name}: output differs (native {:?}, ours {:?} bytes)",
                g.map(|b| b.describe()),
                w.map(|b| b.len())
            )),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 30, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The embedded-value spelling (`-tx5.5`) and the separate one (`-tx 5.5`) are
/// the same transform; a failed `-2` transform exits after the output has
/// already been created empty (`imodtrans.c:316` opens it before
/// `filetrans` runs).
#[test]
fn embedded_values_and_failed_transform_leave_native_state() {
    let golden = fixture_dir().join("golden");
    assert!(
        common::golden::expect(&golden.join("tx.mod"))
            .same_as(&common::golden::expect(&golden.join("txembed.mod")))
    );
    assert_eq!(common::golden::expect(&golden.join("x2lfar.mod")).len(), 0);
}

/// An existing output is backed up to `name~` before being overwritten
/// (`imodtrans.c:313`), and the usage paths exit 3.
#[test]
fn backup_and_usage_exits() {
    let dir = scratch("backup");
    std::fs::copy(dir.join("meshed.mod"), dir.join("out.mod")).unwrap();
    let status = common::imod_cmd("imodtrans")
        .current_dir(&dir)
        .args([
            "-tx",
            "5.5",
            "-ty",
            "-3",
            "-tz",
            "2",
            "multi.mod",
            "out.mod",
        ])
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(0));
    assert_eq!(
        std::fs::read(dir.join("out.mod~")).unwrap(),
        std::fs::read(dir.join("meshed.mod")).unwrap()
    );
    assert!(
        common::golden::expect(&fixture_dir().join("golden/tx.mod"))
            .matches(&std::fs::read(dir.join("out.mod")).unwrap())
    );
    for argv in [&[][..], &["only.mod"][..]] {
        let output = common::imod_cmd("imodtrans")
            .current_dir(&dir)
            .args(argv)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(3));
        assert!(String::from_utf8_lossy(&output.stdout).contains("Usage: imodtrans"));
    }
    let output = common::imod_cmd("imodtrans")
        .current_dir(&dir)
        .args(["nosuch.mod", "o.mod"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let _ = std::fs::remove_dir_all(&dir);
}

/// BUGS.md, fixed in translation: the `-R` conflict message names `-Y`,
/// the option that exists, not the source's nonexistent `-F`.  (The other
/// `imodtrans` fix, `-tx`/`-ty` applied once with `-2 file -l N`, is the
/// `x2lt` row of `cases.tsv`, whose golden is native run with half the shift.)
#[test]
fn r_conflict_message_names_y() {
    let dir = scratch("r-message");
    let output = common::imod_cmd("imodtrans")
        .current_dir(&dir)
        .args(["-R", "1", "-Y", "multi.mod", "out2.mod"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let text = String::from_utf8_lossy(&output.stdout).into_owned()
        + &String::from_utf8_lossy(&output.stderr);
    assert!(text.contains("-R with either -Y or -T"), "{text}");
    let _ = std::fs::remove_dir_all(&dir);
}
