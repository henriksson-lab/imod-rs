//! Native-golden coverage for `xfmodel` (`IMOD/flib/model/xfmodel.f90`).
//!
//! Every row of `fixtures/xfmodel/cases.tsv` was run through the native
//! reference program by `fixtures/make-xfmodel-goldens.sh`, with standard
//! output captured through a pipe.  `golden/<case>.rc` is the exit status,
//! `.stdout` the standard output (the deviation tables, the header listing of
//! an `-image` file and the file-open lines are all byte-identical, so the
//! whole stream is compared) and `.out` the model or transform file native
//! wrote, when it wrote one.  Model files are compared byte for byte: none of
//! these cases reaches the uninitialised `Imod.name`/`MINX` regions.
//! Inputs: the seeded files in `fixtures/xfmodel` plus `BBa_erase.fid`,
//! `BBa.xf` and `BBb.xf` from the vendored `IMOD/Etomo/uitestData/BB`.
//! Pruned 2026-09-26: 80 of 128 rows kept (dropped back/line/chunk repeats of kept options, extra find-mode and image/piece/warp permutations, same-message error variants and most interactive EOF prompts); the rest stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).

mod common;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROGRAM: &str = "xfmodel";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/xfmodel")
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file()
            && path
                .file_name()
                .is_some_and(|n| n != "cases.tsv" && n != "golden.manifest")
        {
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
    }
    let vendored = Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD/Etomo/uitestData/BB");
    for name in ["BBa_erase.fid", "BBa.xf", "BBb.xf"] {
        std::fs::copy(vendored.join(name), dir.join(name)).unwrap();
    }
    dir
}

/// The `printf` escapes the golden script feeds the interactive cases.
fn unescape(text: &str) -> String {
    if text == "-" {
        return String::new();
    }
    text.replace("\\n", "\n")
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, args, stdin) = (fields[0], fields[1], fields[2]);
        let dir = scratch(name);
        let args: Vec<&str> = if args == "-" {
            Vec::new()
        } else {
            args.split_whitespace().collect()
        };
        let mut child = common::imod_cmd(PROGRAM)
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(unescape(stdin).as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        count += 1;
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}",
                output.status.code()
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
        // A case whose output a BUGS.md fix changes compares against the
        // defined output in `defined/` instead of native's golden (kept in
        // `golden/` as the record).  The `xfmodel` `+`-for-`*` grid-extension
        // fix (`xfmodel.f90:371,374`) moves model points through a smaller,
        // differently spaced grid, by up to about 1e-4 pixel, in: dist,
        // distback, distb2bin, distx, distpre, distpreback, distcenter,
        // graddist, graddistx, graddistpreback, w8img, w8imgback, w8imgpre,
        // w8gapall, w20graddist, w20graddistback, w20dist.
        let defined = fixture_dir().join("defined").join(format!("{name}.out"));
        let expected = common::golden::load(&defined)
            .or_else(|| common::golden::load(&golden.join(format!("{name}.out"))));
        let written = ["o.mod", "o.xf"]
            .iter()
            .find_map(|o| std::fs::read(dir.join(o)).ok());
        match (expected, written) {
            (None, None) => {}
            (Some(g), Some(w)) if g.matches(&w) => {}
            (g, w) => failures.push(format!(
                "{name}: output differs (native {:?}, ours {:?} bytes)",
                g.map(|b| b.describe()),
                w.map(|b| b.len())
            )),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 80, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Runs `xfmodel` in `dir` under a 60 s timeout.
fn run_timed(dir: &Path, args: &[&str], stdin: &str) -> std::process::Output {
    let mut child = std::process::Command::new("timeout")
        .arg("60")
        .arg(env!("CARGO_BIN_EXE_imod"))
        .arg(PROGRAM)
        .args(args)
        .current_dir(dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

/// BUGS.md `xfmodel`, fixed in translation: `parselist`/`rdlist` pass a
/// literal 0 as LIMLIST and the error path stores into it; native segfaults
/// (exit 139) on a bad character with its message lost.  Defined: the
/// message is printed and the program exits 1, as for a positive LIMLIST.
#[test]
fn bad_list_character_prints_the_message_and_exits_1() {
    let dir = scratch("defined_badlist");
    for (args, stdin) in [
        (
            vec![
                "-input",
                "BBa_erase.fid",
                "-output",
                "o.xf",
                "-sections",
                "1-3x",
            ],
            "",
        ),
        (
            vec![
                "-input",
                "BBa_erase.fid",
                "-output",
                "o.mod",
                "-xforms",
                "x3.xf",
                "-chunks",
                "1,a",
            ],
            "",
        ),
        (vec![], "\n275,275\nBBa_erase.fid\n0\n0\n\no.xf\n1-3x\n"),
    ] {
        let output = run_timed(&dir, &args, stdin);
        assert_eq!(output.status.code(), Some(1), "{args:?}: {output:?}");
        assert!(
            String::from_utf8_lossy(&output.stdout)
                .contains("ERROR: PARSELIST - BAD CHARACTER IN ENTRY"),
            "{args:?}: {output:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// BUGS.md `xfmodel`, fixed in translation: with every deviation NaN
/// (`BBa_erase.fid`'s coincident section-0 points under `-rottrans`)
/// `findTransform` never sets `ipntMax`, and native indexes with stack
/// residue and segfaults.  Defined: `ipntMax` starts at 1, the report names
/// the section's first point, and the run completes.
#[test]
fn all_nan_deviations_report_point_1_and_complete() {
    let dir = scratch("defined_nanmax");
    let output = run_timed(
        &dir,
        &["-input", "BBa_erase.fid", "-output", "o.xf", "-rottrans"],
        "",
    );
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("NaN"), "{stdout}");
    assert!(dir.join("o.xf").exists());
    let _ = std::fs::remove_dir_all(&dir);
}
