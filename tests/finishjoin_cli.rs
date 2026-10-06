//! Native-golden coverage for `finishjoin` (`IMOD/pysrc/finishjoin`, translated in
//! `src/imod/pysrc/finishjoin.rs`).
//!
//! Every row of `fixtures/finishjoin/cases.tsv` was run through the native Python
//! script by `fixtures/make-finishjoin-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn finishjoin_matches_native_goldens() {
    let failures = pysetup_common::run_cases("finishjoin");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Fixed in translation (BUGS.md, `finishjoin`): a trial interval longer
/// than a slice range gives the range's two ends (native raises a NameError
/// on the never-set `zlast`).
#[test]
fn trial_interval_longer_than_the_range_takes_both_ends() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let inputs = root.join("fixtures/finishjoin/inputs");
    let work = std::env::temp_dir().join(format!("imod-rs-finishjoin-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    for name in ["t1.rec", "t2.rec", "t3.rec", "jr.info", "jr.xf"] {
        std::fs::copy(inputs.join(name), work.join(name)).unwrap();
    }
    let output = common::imod_cmd("finishjoin")
        .args("-no -trial 20 jr 1,12 1,10 8,1".split(' '))
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let input = std::fs::read_to_string(work.join("finishjoinNewst.input")).unwrap();
    let ranges: Vec<&str> = input
        .lines()
        .filter_map(|line| line.strip_prefix("SectionsToRead "))
        .collect();
    assert_eq!(ranges, ["0,11", "0,9", "0,7"]);
    let _ = std::fs::remove_dir_all(&work);
}
