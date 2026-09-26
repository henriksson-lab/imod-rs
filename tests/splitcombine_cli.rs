//! Native-golden coverage for `splitcombine` (`IMOD/pysrc/splitcombine`, translated in
//! `src/imod/pysrc/splitcombine.rs`).
//!
//! Every row of `fixtures/splitcombine/cases.tsv` was run through the native Python
//! script by `fixtures/make-splitcombine-goldens.sh`; see `tests/pysetup_common` for
//! what is compared.  The wider differential behind these goldens (with the
//! native combine programs on `PATH`) is recorded in `TODO.md` (dual-axis
//! combine scripts).

mod common;
mod pysetup_common;

#[test]
fn splitcombine_matches_native_goldens() {
    let failures = pysetup_common::run_cases("splitcombine");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// Fixed in translation (BUGS.md): native tests `optionLine3` where it means
// `optionLine4` (`splitcombine:91`), so each chunk gets the *last*
// IMOD_OUTPUT_FORMAT line before the IMOD_BRIEF_HEADER one; the translation
// copies the first, like the other three option lines.
#[test]
fn first_output_format_line_is_copied_into_chunks() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let work =
        std::env::temp_dir().join(format!("imod-rs-splitcombine-fmt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    let basic =
        std::fs::read_to_string(root.join("fixtures/splitcombine/inputs/vc_basic.com")).unwrap();
    let text = basic.replacen(
        "$setenv IMOD_BRIEF_HEADER 1\n",
        "$setenv IMOD_OUTPUT_FORMAT MRC\n$setenv IMOD_OUTPUT_FORMAT TIF\n$setenv IMOD_BRIEF_HEADER 1\n",
        1,
    );
    assert_ne!(text, basic);
    std::fs::write(work.join("volcombine.com"), text).unwrap();
    let output = common::imod_cmd("splitcombine")
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let chunk = std::fs::read_to_string(work.join("volcombine-001.com")).unwrap();
    assert!(chunk.contains("setenv IMOD_OUTPUT_FORMAT MRC"), "{chunk}");
    assert!(!chunk.contains("setenv IMOD_OUTPUT_FORMAT TIF"), "{chunk}");
    let _ = std::fs::remove_dir_all(&work);
}

// Fixed in translation (BUGS.md): `pip.py:1164-1168` finds the token end on
// the stripped line but applies it to the unstripped one, so native reads the
// indented line `  CommandFile volcombine.pcm` as option `CommandFi` with value
// `e volcombine.pcm`.  The translation splits it where the line says.
#[test]
fn indented_standard_input_line_is_split_at_the_option_end() {
    use std::io::Write as _;
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let work = std::env::temp_dir().join(format!(
        "imod-rs-splitcombine-indent-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    std::fs::copy(
        root.join("fixtures/splitcombine/inputs/vc_lowrad.com"),
        work.join("volcombine.pcm"),
    )
    .unwrap();
    let mut child = common::imod_cmd("splitcombine")
        .arg("-StandardInput")
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"  CommandFile volcombine.pcm\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert!(work.join("volcombine-001.pcm").exists());
    assert!(work.join("volcombine-finish.pcm").exists());
    let _ = std::fs::remove_dir_all(&work);
}
