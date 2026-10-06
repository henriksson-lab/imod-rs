//! Native-golden coverage for `xfalign` (`IMOD/pysrc/xfalign`, translated in
//! `src/imod/pysrc/xfalign.rs`).
//!
//! Every row of `fixtures/xfalign/cases.tsv` was run through the native Python
//! script by `fixtures/make-xfalign-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn xfalign_matches_native_goldens() {
    let failures = pysetup_common::run_cases("xfalign");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Fixed in translation (BUGS.md, `xfalign`): `-one` shifts a break list down
/// by one (native raises an IndexError).  Defined behaviour: `-one -break 3`
/// gives what `-break 2` gives, which is the native golden `c09`.
#[test]
fn break_list_numbered_from_one() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let inputs = root.join("fixtures/xfalign/inputs");
    let work = std::env::temp_dir().join(format!("imod-rs-xfalign-{}", std::process::id()));
    let run = |args: &str| {
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(&work).unwrap();
        std::fs::copy(inputs.join("s.mrc"), work.join("s.mrc")).unwrap();
        let output = common::imod_cmd("xfalign")
            .args(args.split(' '))
            .current_dir(&work)
            .env("IMOD_DIR", root.join("IMOD"))
            .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
            .output()
            .unwrap();
        let xf = std::fs::read(work.join("o.xf")).unwrap_or_default();
        (output.status.code(), output.stdout, xf)
    };
    let from_one = run("-one -break 3 s.mrc o.xf");
    assert_eq!(from_one.0, Some(0));
    assert_eq!(from_one, run("-break 2 s.mrc o.xf"));
    let _ = std::fs::remove_dir_all(&work);
}
