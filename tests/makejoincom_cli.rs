//! Tests for `makejoincom` (`IMOD/pysrc/makejoincom`, translated in
//! `src/imod/pysrc/makejoincom.rs`).
//!
//! The native-golden suite (20 cases on `t1/t2/t3.rec`) was deleted on
//! 2026-10-06 to keep fixtures small: the same 370 KB of tomograms were also
//! stored for `finishjoin`, which keeps them.  The defined-behaviour test
//! below reads `fixtures/finishjoin/inputs`.

mod common;

/// Fixed in translation (BUGS.md, `makejoincom`): tomograms entered with
/// `-input` are read (native raises a TypeError on the first one).  Defined
/// behaviour: the same files as the non-option form, which is the native
/// golden `two`.
#[test]
fn input_option_gives_the_non_option_result() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let inputs = root.join("fixtures/finishjoin/inputs");
    let work = std::env::temp_dir().join(format!("imod-rs-makejoincom-{}", std::process::id()));
    let run = |args: &str| {
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(&work).unwrap();
        for name in ["t1.rec", "t2.rec"] {
            std::fs::copy(inputs.join(name), work.join(name)).unwrap();
        }
        let output = common::imod_cmd("makejoincom")
            .args(args.split(' '))
            .current_dir(&work)
            .env("IMOD_DIR", root.join("IMOD"))
            .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
            .output()
            .unwrap();
        let mut files = Vec::new();
        for name in ["jr.info", "startjoin.com"] {
            files.push(std::fs::read(work.join(name)).unwrap_or_default());
        }
        (output.status.code(), output.stdout, files)
    };
    let by_option = run("-top 10,12 -input t1 -bot 1,2 -input t2 -root jr");
    let by_argument = run("-top 10,12 t1 -bot 1,2 t2 jr");
    assert_eq!(by_option.0, Some(0));
    assert_eq!(by_option, by_argument);
    let _ = std::fs::remove_dir_all(&work);
}
