//! Native-golden coverage for `tomosnapshot` (`IMOD/pysrc/tomosnapshot`, translated in
//! `src/imod/pysrc/tomosnapshot.rs`).
//!
//! Every row of `fixtures/tomosnapshot/cases.tsv` was run through the native Python
//! script by `fixtures/make-tomosnapshot-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn tomosnapshot_matches_native_goldens() {
    let failures = pysetup_common::run_cases("tomosnapshot");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The members of a gzipped tar archive (`ustar` headers, `x` pax headers
/// skipped), as (name, typeflag, contents).
fn tar_members(path: &std::path::Path) -> Vec<(String, u8, Vec<u8>)> {
    use std::io::Read as _;
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(std::fs::File::open(path).unwrap())
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(bytes.len() % 10240, 0, "tar not padded to a record");
    let mut members = Vec::new();
    let mut offset = 0;
    while offset + 512 <= bytes.len() && bytes[offset] != 0 {
        let header = &bytes[offset..offset + 512];
        let name = String::from_utf8_lossy(&header[..100])
            .trim_end_matches('\0')
            .to_owned();
        let size_text = String::from_utf8_lossy(&header[124..135]).into_owned();
        let size = usize::from_str_radix(size_text.trim_end_matches('\0'), 8).unwrap();
        let typeflag = header[156];
        assert_eq!(&header[257..265], b"ustar\x0000");
        let data = bytes[offset + 512..offset + 512 + size].to_vec();
        if typeflag != b'x' {
            members.push((name, typeflag, data));
        }
        offset += 512 + size.div_ceil(512) * 512;
    }
    members
}

/// Runs `imod tomosnapshot` in a fresh directory holding `files`.
fn snapshot_run(
    case: &str,
    files: &[(&str, &str)],
    envs: &[(&str, &str)],
    args: &[&str],
) -> (std::path::PathBuf, std::process::Output) {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let work = std::env::temp_dir().join(format!(
        "imod-rs-tomosnapshot-defined-{case}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&work);
    for (name, text) in files {
        let path = work.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
    std::fs::create_dir_all(&work).unwrap();
    let home = work.join("home");
    std::fs::create_dir_all(&home).unwrap();
    let mut command = common::imod_cmd("tomosnapshot");
    command
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
        .env("HOME", &home)
        .env_remove("ETOMO_LOG_DIR")
        .args(args);
    for (key, value) in envs {
        command.env(key, value);
    }
    let output = command.output().unwrap();
    (work, output)
}

/// Defined behaviour (BUGS.md): a PEET `.prm` with no `reference` entry is a
/// traceback natively (`None.find`); here the snapshot is made.
#[test]
fn tomosnapshot_peet_without_reference() {
    let (work, output) = snapshot_run(
        "peet",
        &[
            ("p.epe", "Peet.RootName=p\n"),
            ("p.prm", "fnVolume = {'v.mrc'}\ninitMOTL = {'p_init.csv'}\n"),
            ("p_init.csv", "1,2\n"),
        ],
        &[],
        &[],
    );
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let names: Vec<String> = tar_members(&work.join("p-snapshot"))
        .into_iter()
        .map(|member| member.0)
        .collect();
    assert_eq!(names[..2], ["uname.out".to_owned(), "lslrt.out".to_owned()]);
    assert!(names[2].starts_with("tomosnapshot.cms.") && names[2].ends_with('/'));
    // p_init.csv twice, as native: once from the `p*.csv` glob, once as
    // the initMOTL entry
    assert_eq!(names[3..], ["p.epe", "p.prm", "p_init.csv", "p_init.csv"]);
    let _ = std::fs::remove_dir_all(&work);
}

/// Defined behaviour (BUGS.md): a join `.info` whose first word is not an
/// integer is a ValueError traceback natively; here it counts as no files.
#[test]
fn tomosnapshot_join_bad_info() {
    let (work, output) = snapshot_run(
        "join",
        &[
            ("j.ejf", "Join.RootName=j\n"),
            ("j.info", "x 2\nl2\nl3\nl4\n"),
        ],
        &[],
        &[],
    );
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let names: Vec<String> = tar_members(&work.join("j-snapshot"))
        .into_iter()
        .map(|member| member.0)
        .collect();
    assert_eq!(names[3..], ["j.ejf", "j.info"]);
    let _ = std::fs::remove_dir_all(&work);
}

/// Defined behaviour (BUGS.md): native stores `NWUSERNAME` in a misspelled
/// variable, so its value is never replaced; here it becomes XXXX in the
/// privacy copies, and the `NWUSERNAME=` line itself is deleted.
#[test]
fn tomosnapshot_nwusername_privacy() {
    let (work, output) = snapshot_run(
        "nwuser",
        &[
            ("g.edf", "Setup.DatasetName=g\nSetup.AxisType=Single Axis\n"),
            ("tilt.log", "run by nwsecretuser today\n"),
        ],
        &[("NWUSERNAME", "nwsecretuser")],
        &[],
    );
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let members = tar_members(&work.join("g-snapshot"));
    let log = members
        .iter()
        .find(|member| member.0 == "tilt.log")
        .expect("tilt.log in the snapshot");
    assert_eq!(log.2, b"run by XXXX today\n");
    let uname = members
        .iter()
        .find(|member| member.0 == "uname.out")
        .unwrap();
    assert!(!String::from_utf8_lossy(&uname.2).contains("nwsecretuser"));
    let _ = std::fs::remove_dir_all(&work);
}

/// Defined behaviour (BUGS.md): with no `naddir.<set>` directory native's
/// `os.chdir` raises FileNotFoundError (traceback, status 1); here it is an
/// error exit with a message, status 1.
#[test]
fn tomosnapshot_missing_naddir() {
    let (work, output) = snapshot_run(
        "nad",
        &[("n.epp", "AnisotropicDiffusion.RootName=n\n")],
        &[],
        &[],
    );
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: tomosnapshot - Changing to directory naddir.n\n"
    );
    let _ = std::fs::remove_dir_all(&work);
}
