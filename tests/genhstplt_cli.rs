//! Native-golden coverage for `genhstplt` (`IMOD/flib/graphics/genhstplt.f90`
//! on the graphics library `IMOD/flib/subrs/graphics`, translated with a
//! Rust-native plot layer in place of Qt).
//!
//! Every row of `fixtures/genhstplt/cases.tsv` (name, input, arguments) was
//! run through the reference program under Xvfb by
//! `fixtures/make-genhstplt-goldens.sh`, with its plot calls recorded into
//! `genhstplt.calls` (the `LD_PRELOAD` shim `fixtures/plaxrec.c`).  Here the
//! same input runs `imod genhstplt` without a window (`PLAX_HEADLESS`), whose
//! `PLAX_CALL_LOG` writes the same record.  Compared exactly: the exit
//! status, standard output, the plot call log (every `plax_*` call and its
//! arguments, so the geometry, colors, symbols and text of every plot),
//! `gmeta.ps` (but for its compile-time line) and typed-out values.  Saved
//! PNG and TIFF images are compared as their size and format: Qt drew the
//! native ones, a Rust-native painter draws ours (line rasterization and
//! font rendering differ by design; the pixels drawn from the same calls
//! were checked by eye and by difference counts, TODO.md).

mod common;

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/genhstplt")
}

/// A PNG or TIFF as `IMAGE <width>x<height> ...`; anything else unchanged.
fn image_summary(bytes: &[u8]) -> Option<Vec<u8>> {
    if bytes.len() >= 26 && bytes.starts_with(b"\x89PNG\r\n\x1a\n") && &bytes[12..16] == b"IHDR" {
        let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
        let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
        return Some(
            format!(
                "PNG {width}x{height} depth {} color {}\n",
                bytes[24], bytes[25]
            )
            .into_bytes(),
        );
    }
    let little = bytes.starts_with(b"II*\0");
    if !(little || bytes.starts_with(b"MM\0*")) {
        return None;
    }
    let u16_at = |k: usize| -> u32 {
        let b = [bytes[k], bytes[k + 1]];
        (if little {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        }) as u32
    };
    let u32_at = |k: usize| -> u32 {
        let b = [bytes[k], bytes[k + 1], bytes[k + 2], bytes[k + 3]];
        if little {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        }
    };
    let ifd = u32_at(4) as usize;
    let count = u16_at(ifd) as usize;
    let mut tags = BTreeMap::new();
    for k in 0..count {
        let entry = ifd + 2 + 12 * k;
        let tag = u16_at(entry);
        let kind = u16_at(entry + 2);
        let value = if kind == 3 {
            u16_at(entry + 8)
        } else {
            u32_at(entry + 8)
        };
        if [256, 257, 277].contains(&tag) {
            tags.insert(tag, value);
        }
    }
    Some(format!("TIFF {:?}\n", tags).into_bytes())
}

/// The comparison mask of an output file (module documentation).
fn mask_output(bytes: &[u8]) -> Vec<u8> {
    if let Some(summary) = image_summary(bytes) {
        return summary;
    }
    if bytes.starts_with(b"%!PS") {
        let text = String::from_utf8_lossy(bytes);
        return text
            .lines()
            .map(|line| {
                if line.contains("Created by the BL3DEMC PS module. Compiled") {
                    "%%Created by the BL3DEMC PS module. Compiled <date>".to_owned()
                } else {
                    line.to_owned()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
            .into_bytes();
    }
    bytes.to_vec()
}

/// Runs `imod genhstplt` headless in `dir` with the input file and arguments.
fn run(dir: &Path, input: &str, args: &[&str]) -> std::process::Output {
    let stdin = std::fs::read(fixtures().join("inputs").join(format!("{input}.in"))).unwrap();
    run_input(dir, &stdin, args)
}

/// Runs `imod genhstplt` headless in `dir` with the given input.
fn run_input(dir: &Path, stdin: &[u8], args: &[&str]) -> std::process::Output {
    let mut child = common::imod_cmd("genhstplt")
        .current_dir(dir)
        .env("PLAX_HEADLESS", "1")
        .env("PLAX_CALL_LOG", "genhstplt.calls")
        .env_remove("IMOD_PS_FONT")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(stdin).unwrap();
    child.wait_with_output().unwrap()
}

/// A fresh directory holding the data files.
fn work_dir(name: &str) -> (PathBuf, BTreeMap<String, Vec<u8>>) {
    let dir = std::env::temp_dir().join(format!("imod-rs-genhstplt-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut inputs = BTreeMap::new();
    for entry in std::fs::read_dir(fixtures().join("inputs")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "txt") {
            let file = path.file_name().unwrap().to_string_lossy().into_owned();
            let bytes = std::fs::read(&path).unwrap();
            std::fs::write(dir.join(&file), &bytes).unwrap();
            inputs.insert(file, bytes);
        }
    }
    (dir, inputs)
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixtures().join("cases.tsv")).unwrap();
    let golden = fixtures().join("golden");
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, input, args) = (fields[0], fields[1], fields[2]);
        let args: Vec<&str> = if args == "-" {
            Vec::new()
        } else {
            args.split(' ').collect()
        };
        let (dir, inputs) = work_dir(name);
        let output = run(&dir, input, &args);
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
        if let Err(why) = stdout.compare(&output.stdout, common::golden::identity, false) {
            failures.push(format!("{name}: stdout differs: {why}"));
        }
        let prefix = format!("{name}.out.");
        let mut expected = BTreeMap::new();
        for file in common::golden::list(&golden) {
            if let Some(out) = file.strip_prefix(&prefix) {
                expected.insert(out.to_owned(), common::golden::expect(&golden.join(&file)));
            }
        }
        let mut written = BTreeMap::new();
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let file = path.file_name().unwrap().to_string_lossy().into_owned();
            let bytes = std::fs::read(&path).unwrap();
            if inputs.get(&file) != Some(&bytes) {
                written.insert(file, bytes);
            }
        }
        if expected.keys().ne(written.keys()) {
            failures.push(format!(
                "{name}: output files {:?}, native {:?}",
                written.keys().collect::<Vec<_>>(),
                expected.keys().collect::<Vec<_>>()
            ));
        }
        for (file, native) in &expected {
            if let Some(ours) = written.get(file)
                && let Err(why) = native.compare(ours, mask_output, false)
            {
                failures.push(format!("{name}: {file} differs from native: {why}"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 11, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `BUGS.md` (`bshst`/`bsplt` typed-out values), defined behaviour: native
/// reads the output file to its end and then cannot write it (gfortran:
/// "Sequential READ or WRITE not allowed after EOF marker", exit 2); here
/// the values are appended to what the file already held, in the format
/// they are typed to standard output (case hist3 of the golden table).
#[test]
fn typed_out_values_are_appended_to_the_file() {
    let (dir, _) = work_dir("append");
    std::fs::write(dir.join("hist.out"), "previous line\n").unwrap();
    let output = run(&dir, "hist2", &[]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = std::fs::read_to_string(dir.join("hist.out")).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "previous line");
    assert_eq!(lines.len(), 37, "{text}");
    // Format 203, `(i3,f10.3)`: the group, then the value; the values typed
    // to standard output by hist3 are the same, in the same order
    let (dir3, _) = work_dir("append3");
    let typed = run(&dir3, "hist3", &[]);
    let typed = String::from_utf8_lossy(&typed.stdout).into_owned();
    for line in &lines[1..] {
        assert_eq!(line.len(), 13, "{line:?}");
        assert!(
            typed.contains(&format!("{line}\n")),
            "{line:?} not typed by hist3"
        );
    }
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_dir_all(dir3);
}

/// `BUGS.md` (`qtplax`), defined behaviour: with `-nograph` native has no
/// widget and dereferences NULL when asked to save the plot or wait for the
/// window; here nothing is saved and waiting exits with status 0.
#[test]
fn nograph_save_and_wait_do_not_crash() {
    let (dir, _) = work_dir("nograph");
    let output = run(&dir, "onegen", &["-nograph"]);
    assert_eq!(output.status.code(), Some(0));
    assert!(!dir.join("out.png").exists());
    let output = run(&dir, "wait", &["-nograph"]);
    assert_eq!(output.status.code(), Some(0));
    let log = std::fs::read_to_string(dir.join("genhstplt.calls")).unwrap();
    assert!(log.ends_with("wait\n"), "{log}");
    let _ = std::fs::remove_dir_all(dir);
}

/// The headless window has the requested size: `-s 400,300` saves a 400 x
/// 300 image, drawn at the scale a window of that size gets.
#[test]
fn headless_size_follows_the_size_option() {
    let (dir, _) = work_dir("size");
    let output = run(&dir, "onegen", &["-s", "400,300"]);
    assert_eq!(output.status.code(), Some(0));
    let png = std::fs::read(dir.join("out.png")).unwrap();
    assert_eq!(
        image_summary(&png).unwrap(),
        b"PNG 400x300 depth 8 color 2\n"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// `BUGS.md` (`grupnt`), defined behaviour: group counts adding up to more
/// than the points (native reads index entries it never set) are cut to the
/// points that are left: here 3 of the 4 points, then the last one, then
/// none (a 0/0 average).
#[test]
fn averaging_counts_past_the_points_are_cut() {
    let (dir, _) = work_dir("grupnt");
    let input = b"-1\n0\n-1\n2\nd.txt\n1\n0,0\n0\n1\n0,0\n0\n3\n1\n3\n3,3,3\n0\n0\n8\n";
    let output = run_input(&dir, input, &[]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    // Format 101, `(3f10.4,i5)`, the first one after the prompt
    let rows: Vec<&str> = text
        .lines()
        .filter(|line| line.len() >= 35 && line.ends_with(|c: char| c.is_ascii_digit()))
        .map(|line| &line[line.len() - 35..])
        .filter(|row| row.as_bytes()[0] == b' ' && row.contains('.'))
        .collect();
    assert_eq!(rows.len(), 3, "{text}");
    assert!(rows[0].ends_with("    3"), "{rows:?}");
    assert!(rows[1].ends_with("    1"), "{rows:?}");
    assert!(
        rows[2].ends_with("    0") && rows[2].contains("NaN"),
        "{rows:?}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// `BUGS.md` (`bshst`), defined behaviour: plot numbers past 16 (native
/// reads past the six plot positions) repeat the positions, so plot 17 is
/// drawn where plot 11 is.
#[test]
fn histogram_plot_positions_repeat_past_16() {
    let plot = |number: &str| -> String {
        let (dir, _) = work_dir(&format!("pos{number}"));
        let input = format!("-1\n0\n1\n0\nh.txt\n1\n0,0\n1\n0,0,10\n100\n0\n{number}\n1\n8\n");
        let output = run_input(&dir, input.as_bytes(), &[]);
        assert_eq!(
            output.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let ps = std::fs::read(dir.join("gmeta.ps")).unwrap();
        let _ = std::fs::remove_dir_all(dir);
        String::from_utf8_lossy(&mask_output(&ps)).into_owned()
    };
    let p11 = plot("11");
    assert!(p11.contains("drawline"), "{p11}");
    assert_eq!(plot("17"), p11);
}
