//! The body shared by the case-table golden suites whose goldens come from
//! `fixtures/make-small-prog-goldens.sh`: every row of
//! `fixtures/<suite>/cases.tsv` (name, arguments or `-`, stdin as printf
//! escapes or `-`) runs `imod <program>` in a fresh directory holding copies of
//! the fixture inputs, and its exit status, standard output and every file it
//! created or changed are compared with the native goldens
//! (`golden/<name>.rc`, `.stdout`, `.out.<file>`).  MRC label stamps are masked
//! (`common::mask_stamps`), and MRC regions native writes from uninitialised
//! memory are reconciled when `reconcile` is set.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

/// Options of one suite.
pub struct Suite<'a> {
    /// The command run, `imod <program>`.
    pub program: &'a str,
    /// The fixture directory under `fixtures/`.
    pub fixtures: &'a str,
    /// The fewest cases the table must yield.
    pub min_cases: usize,
    /// Whether output files go through `reconcile_uninitialised`.
    pub reconcile: bool,
    /// Extra environment for every case.
    pub env: &'a [(&'a str, &'a str)],
    /// A mask applied to both standard outputs before comparing.
    pub stdout_mask: fn(&[u8]) -> Vec<u8>,
}

fn fixture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

/// The fixture inputs: the regular files of the fixture directory other than
/// the case table, the manifest and the generator scripts.
fn inputs(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut map = BTreeMap::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !path.is_file()
            || name == "cases.tsv"
            || name == "golden.manifest"
            || name.starts_with("make-")
        {
            continue;
        }
        map.insert(name, std::fs::read(&path).unwrap());
    }
    map
}

/// Runs every case of the suite and panics with every difference found.
pub fn run(suite: &Suite) {
    let fixtures = fixture_dir(suite.fixtures);
    let table = std::fs::read_to_string(fixtures.join("cases.tsv")).unwrap();
    let golden = fixtures.join("golden");
    let inputs = inputs(&fixtures);
    let mut failures = Vec::new();
    let mut count = 0;
    for line in super::golden::case_rows(&table) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, args, stdin) = (fields[0], fields[1], fields[2]);
        let dir = std::env::temp_dir().join(format!(
            "imod-rs-{}-{}-{}",
            suite.fixtures,
            std::process::id(),
            name
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (file, bytes) in &inputs {
            std::fs::write(dir.join(file), bytes).unwrap();
        }
        let args: Vec<&str> = if args == "-" {
            Vec::new()
        } else {
            args.split_whitespace().collect()
        };
        let stdin = if stdin == "-" {
            String::new()
        } else {
            stdin.replace("\\n", "\n")
        };
        let mut command = super::imod_cmd(suite.program);
        command
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (key, value) in suite.env {
            command.env(key, value);
        }
        let mut child = command.spawn().unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(stdin.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        count += 1;
        let rc: i32 = super::golden::read_to_string(&golden.join(format!("{name}.rc")))
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
        let stdout = super::golden::expect(&golden.join(format!("{name}.stdout")));
        if let Err(why) = stdout.compare(&output.stdout, suite.stdout_mask, false) {
            failures.push(format!(
                "{name}: stdout differs: {why}\n--- ours\n{}",
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let prefix = format!("{name}.out.");
        let mut expected = BTreeMap::new();
        for file in super::golden::list(&golden) {
            if let Some(out) = file.strip_prefix(&prefix) {
                expected.insert(out.to_owned(), super::golden::expect(&golden.join(&file)));
            }
        }
        let mut written = BTreeMap::new();
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if !path.is_file() {
                continue;
            }
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
            if let Some(ours) = written.get(file) {
                if let Err(why) = native.compare(ours, super::mask_stamps, suite.reconcile) {
                    failures.push(format!("{name}: {file} differs from native: {why}"));
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(
        count >= suite.min_cases,
        "only {count} cases read, expected {}",
        suite.min_cases
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A standard-output mask: [`super::mask_stamps`], then drops the date and
/// time after ` Version ` in a usage banner (compile metadata, `CLAUDE.md`).
pub fn mask_banner(bytes: &[u8]) -> Vec<u8> {
    let masked = super::mask_stamps(bytes);
    let text = String::from_utf8_lossy(&masked);
    text.lines()
        .map(|line| match line.find(" Version ") {
            Some(k) => line[..k + 9].to_string(),
            None => line.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
        .into_bytes()
}
