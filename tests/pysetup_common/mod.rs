//! Shared runner for the native-golden tests of the single-axis setup scripts
//! (`copytomocoms`, `makecomfile`, `splittilt`, `alignlog`, `chunksetup`,
//! `tomocleanup`) and the dual-axis combine scripts (`setupcombine`,
//! `splitcombine`, `collectmmm`, `b3dremove`, `dualvolmatch`, `matchorwarp`,
//! `autopatchfit`), all translations of `IMOD/pysrc` Python scripts.
//!
//! Each `fixtures/<script>/cases.tsv` row was run through the native Python
//! script by `fixtures/make-<script>-goldens.sh` (`make-pysetup-goldens.py`),
//! which recorded the exit status, stdout, the list of files left behind and
//! every file that is not an unchanged input.  Here the same inputs are laid
//! out in a fresh directory, `imod <script>` runs with `IMOD_DIR` at the
//! vendored `IMOD/` (whose `com/` holds the templates) and `AUTODOC_DIR` at
//! `IMOD/autodoc`, and all of that must match byte for byte.
//!
//! Only wall-clock stamps are masked (`common::mask_stamps`): `CreatedDayStamp`
//! in `align*.com`, the number of days since 1 January 2020 on the day the
//! file was made, and the date/time stamp in the labels of an MRC output.
//!
//! `chunksetup` and `setupcombine` run the external `tomopieces`; their tests put a stand-in for
//! it on `PATH` that prints what the native program printed for the case
//! (`golden/<case>.tomopieces`).
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::common;

fn copy_tree(source: &Path, target: &Path, placed: &mut BTreeMap<String, Vec<u8>>, work: &Path) {
    let mut names = std::fs::read_dir(source)
        .unwrap_or_else(|error| panic!("read {}: {error}", source.display()))
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    names.sort();
    for path in names {
        let destination = target.join(path.file_name().unwrap());
        if path.is_dir() {
            std::fs::create_dir_all(&destination).unwrap();
            copy_tree(&path, &destination, placed, work);
        } else {
            std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
            std::fs::copy(&path, &destination).unwrap();
            placed.insert(
                relative(&destination, work),
                std::fs::read(&destination).unwrap(),
            );
        }
    }
}

fn relative(path: &Path, base: &Path) -> String {
    let mut components = Vec::new();
    for component in path.strip_prefix(base).unwrap().components() {
        match component {
            std::path::Component::CurDir => {}
            other => components.push(other.as_os_str().to_string_lossy().into_owned()),
        }
    }
    components.join("/")
}

fn walk(directory: &Path, base: &Path, files: &mut Vec<String>) {
    for entry in std::fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            walk(&path, base, files);
        } else {
            files.push(relative(&path, base));
        }
    }
}

/// `common::mask_stamps`, except that a gzip file (`archiveorig`'s
/// `_xray.ext.gz`) is compared as its header with the modification time
/// zeroed followed by its masked decompressed content: the deflate stream
/// itself carries the label stamps of the image inside, so it cannot be
/// masked in place.  Identical content gives identical compressed bytes
/// (checked against native in `TODO.md`).
///
/// A PNG image (`genhstplt`'s saved plot, which Qt rendered natively and a
/// Rust-native painter renders here) is compared as its size, bit depth and
/// color type; what it shows is checked through the plot calls that drew it
/// (`genhstplt.calls`, compared byte for byte).
fn mask_file(bytes: &[u8]) -> Vec<u8> {
    use std::io::Read as _;
    if bytes.len() >= 26 && bytes.starts_with(b"\x89PNG\r\n\x1a\n") && &bytes[12..16] == b"IHDR" {
        let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
        let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
        return format!(
            "PNG {width}x{height} depth {} color {}\n",
            bytes[24], bytes[25]
        )
        .into_bytes();
    }
    if bytes.len() < 10 || !bytes.starts_with(&[0x1f, 0x8b, 8]) {
        return common::mask_stamps(bytes);
    }
    let mut header = bytes[..10].to_vec();
    header[4..8].fill(0);
    let mut content = Vec::new();
    if flate2::read::MultiGzDecoder::new(bytes)
        .read_to_end(&mut content)
        .is_err()
    {
        return common::mask_stamps(bytes);
    }
    header.extend(common::mask_stamps(&content));
    header
}

/// Runs every case of `fixtures/<script>/cases.tsv` and returns the failures.
pub fn run_cases(script: &str) -> Vec<String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let here = root.join("fixtures").join(script);
    let golden = here.join("golden");
    let cases = std::fs::read_to_string(here.join("cases.tsv")).expect("cases.tsv");
    let mut failures = Vec::new();
    let mut count = 0;
    for row in common::golden::case_rows(&cases) {
        if row.trim().is_empty() {
            continue;
        }
        let fields = row.split('\t').collect::<Vec<_>>();
        let (name, inputs, envs, args) = (fields[0], fields[1], fields[2], fields[3]);
        count += 1;
        let work =
            std::env::temp_dir().join(format!("imod-rs-{script}-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(&work).unwrap();
        let mut placed = BTreeMap::new();
        if inputs != "-" {
            for item in inputs.split(' ') {
                let (source, target) = item.split_once(':').unwrap();
                let source_path = here.join("inputs").join(source);
                if source.ends_with('/') {
                    let destination = work.join(target);
                    std::fs::create_dir_all(&destination).unwrap();
                    copy_tree(&source_path, &destination, &mut placed, &work);
                } else {
                    let destination = work.join(target);
                    std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
                    std::fs::copy(&source_path, &destination).unwrap();
                    placed.insert(
                        relative(&destination, &work),
                        std::fs::read(&destination).unwrap(),
                    );
                }
            }
        }

        let mut command = common::imod_cmd(script);
        // `genhstplt` (run by `onegenplot`) draws without a window and logs
        // its plot calls into the case directory, as the goldens were made
        // (`make-pyscript-goldens.py`).
        command
            .env("PLAX_HEADLESS", "1")
            .env("PLAX_CALL_LOG", "genhstplt.calls")
            .current_dir(&work)
            .env("IMOD_DIR", root.join("IMOD"))
            .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
            .env_remove("IMOD_OUTPUT_FORMAT")
            .env_remove("TEST_NAMING_STYLE")
            .env_remove("TEST_USE_PCM_FOR_COM")
            .env_remove("PARALLEL_BOUNDARY_SIZE")
            .env_remove("RUNCMD_VERBOSE");
        if envs != "-" {
            for pair in envs.split(' ') {
                let (key, value) = pair.split_once('=').unwrap();
                command.env(key, value);
            }
        }
        // A suite whose script runs one of our Python-script translations
        // through the shell (`runcmd` resolves it on PATH) lists it in
        // `own_commands`; links to our binary go first on PATH.
        if let Ok(own) = std::fs::read_to_string(here.join("own_commands")) {
            for name in own.split_whitespace() {
                common::imod_link(name);
            }
            command.env(
                "PATH",
                format!(
                    "{}:{}",
                    common::command_link_directory().display(),
                    std::env::var("PATH").unwrap_or_default()
                ),
            );
        }
        let bin = work.with_extension("bin");
        if let Some(stand_in) = common::golden::read_opt(&golden.join(format!("{name}.tomopieces")))
        {
            std::fs::create_dir_all(&bin).unwrap();
            let printed = bin.join("tomopieces.out");
            std::fs::write(&printed, stand_in).unwrap();
            let program = bin.join("tomopieces");
            std::fs::write(
                &program,
                format!("#!/bin/sh\ncat '{}'\n", printed.display()),
            )
            .unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            command.env(
                "PATH",
                format!(
                    "{}:{}",
                    bin.display(),
                    std::env::var("PATH").unwrap_or_default()
                ),
            );
        }
        if args != "-" {
            command.args(args.split(' '));
        }
        let output = command.output().expect("run imod");

        let expected_rc = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse::<i32>()
            .unwrap();
        if output.status.code() != Some(expected_rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {expected_rc}; stdout {}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let expected_out = common::golden::expect(&golden.join(format!("{name}.out")));
        // Wall-clock stamps a called program prints (an MRC label echoed by
        // `tilt`, ...) are masked as in the output files.
        if expected_out
            .compare(&output.stdout, common::mask_stamps, false)
            .is_err()
        {
            failures.push(format!(
                "{name}: stdout differs:\n{}\n--- native:\n{}",
                String::from_utf8_lossy(&output.stdout),
                expected_out.display()
            ));
        }

        let mut remaining = Vec::new();
        walk(&work, &work, &mut remaining);
        remaining.sort();
        let expected_files = common::golden::read_to_string(&golden.join(format!("{name}.files")));
        let expected_files = expected_files
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if remaining != expected_files {
            failures.push(format!(
                "{name}: files left {remaining:?}, native {expected_files:?}"
            ));
        }
        for file in &remaining {
            let actual = std::fs::read(work.join(file)).unwrap();
            if let Some(expected) = common::golden::load(&golden.join(name).join(file)) {
                if let Err(why) = expected.compare(&actual, mask_file, false) {
                    failures.push(format!("{name}: {file} differs from native: {why}"));
                }
            } else if placed.get(file) != Some(&actual) {
                failures.push(format!("{name}: {file} changed, native left it as input"));
            }
        }
        let _ = std::fs::remove_dir_all(&work);
        let _ = std::fs::remove_dir_all(&bin);
    }
    assert!(count > 0, "no cases in fixtures/{script}/cases.tsv");
    failures
}
