//! Native-golden coverage for `alttomosetup` (`IMOD/pysrc/alttomosetup`, translated in
//! `src/imod/pysrc/alttomosetup.rs`).
//!
//! Every row of `fixtures/alttomosetup/cases.tsv` was run through the native Python
//! script by `fixtures/make-alttomosetup-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn alttomosetup_matches_native_goldens() {
    let failures = pysetup_common::run_cases("alttomosetup");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Runs `imod alttomosetup -rootname alt -trim -procs 1` on the single-axis
/// inputs with `edf` as `g.edf` and returns the trimvol command it wrote.
fn trimvol_line(edf: &str) -> String {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let inputs = root.join("fixtures/alttomosetup/inputs");
    let work = std::env::temp_dir().join(format!(
        "imod-rs-alttomosetup-defined-{edf}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    for entry in std::fs::read_dir(inputs.join("s")).unwrap() {
        let path = entry.unwrap().path();
        std::fs::copy(&path, work.join(path.file_name().unwrap())).unwrap();
    }
    for target in ["g.ali", "alt.st"] {
        std::fs::copy(inputs.join("s/g.st"), work.join(target)).unwrap();
    }
    std::fs::copy(inputs.join(edf), work.join("g.edf")).unwrap();
    let output = common::imod_cmd("alttomosetup")
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
        .args(["-rootname", "alt", "-trim", "-procs", "1"])
        .output()
        .expect("run imod");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let text = std::fs::read_to_string(work.join("alttomo-002-sync.com")).unwrap();
    let _ = std::fs::remove_dir_all(&work);
    text.lines()
        .find(|line| line.starts_with("$trimvol"))
        .expect("trimvol line")
        .to_owned()
}

/// `BUGS.md` (alttomosetup `makeTrimvolCommandFromEDF`), defined behaviour:
/// native takes every tag found anywhere in a line, so the `ScaleXMin` ...
/// lines also set `-x`/`-y` (native writes `-x 10,20 -y 5,25` for this edf);
/// here each tag is matched as the last component of the key.
#[test]
fn trimvol_tags_match_whole_keys() {
    assert_eq!(
        trimvol_line("edfscale.edf"),
        "$trimvol -f -yz -x 3,40 -y 1,30 -z 2,8 -sx 10,20 -sy 5,25 -sz 3,6 g.rec g.rec"
    );
}

/// `BUGS.md` (alttomosetup `makeTrimvolCommandFromEDF`), defined behaviour:
/// native writes `-c 10 , FixedScaleMax` (the literal tag name, and blanks
/// trimvol's integer pair cannot parse); here `-c 10,200`.
#[test]
fn trimvol_fixed_scaling_uses_both_values() {
    assert_eq!(
        trimvol_line("edffixed.edf"),
        "$trimvol -f -yz -c 10,200 -x 3,40 -y 1,30 -z 2,8 g.rec g.rec"
    );
}
