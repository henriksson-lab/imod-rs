//! Native-golden coverage for `sorttiltframes` (`IMOD/pysrc/sorttiltframes`, translated in
//! `src/imod/pysrc/sorttiltframes.rs`).
//!
//! Every row of `fixtures/sorttiltframes/cases.tsv` was run through the native Python
//! script by `fixtures/make-sorttiltframes-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn sorttiltframes_matches_native_goldens() {
    let failures = pysetup_common::run_cases("sorttiltframes");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Fixed in translation (BUGS.md, `sorttiltframes`): the default `[]`
/// delimiters are matched literally, so a name with `[-3.00]` sorts at -3.
/// Defined behaviour: each run with `[]` gives exactly what the same names
/// with `_` delimiters and `-delim __` give -- and those runs are the native
/// goldens `*_us` above.
#[test]
fn bracket_delimiters_behave_like_any_other_pair() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let inputs = root.join("fixtures/sorttiltframes/inputs");
    let work = std::env::temp_dir().join(format!("imod-rs-sorttiltframes-{}", std::process::id()));
    for (brackets, underscores) in [
        ("-listin list.txt", "-listin list_us.txt -delim __"),
        ("-listin list.txt -r", "-listin list_us.txt -r -delim __"),
        (
            "-listin list.txt -fixed 2.5 -dose d.out",
            "-listin list_us.txt -fixed 2.5 -dose d.out -delim __",
        ),
        (
            "-input a[1.0].tif -input q[-2.0].tif c[1.04].tif",
            "-input a_1.0_.tif -input q_-2.0_.tif c_1.04_.tif -delim __",
        ),
    ] {
        let run = |args: &str| {
            let _ = std::fs::remove_dir_all(&work);
            std::fs::create_dir_all(&work).unwrap();
            for name in ["list.txt", "list_us.txt"] {
                std::fs::copy(inputs.join(name), work.join(name)).unwrap();
            }
            let output = common::imod_cmd("sorttiltframes")
                .args(args.split(' '))
                .current_dir(&work)
                .env("IMOD_DIR", root.join("IMOD"))
                .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
                .output()
                .unwrap();
            let dose = std::fs::read(work.join("d.out")).unwrap_or_default();
            (output.status.code(), output.stdout, dose)
        };
        let (rc_b, out_b, dose_b) = run(brackets);
        let (rc_u, out_u, dose_u) = run(underscores);
        let unbracket = |bytes: Vec<u8>| -> Vec<u8> {
            bytes
                .into_iter()
                .map(|b| if b == b'[' || b == b']' { b'_' } else { b })
                .collect()
        };
        assert_eq!(rc_b, rc_u, "{brackets}");
        assert_eq!(
            String::from_utf8_lossy(&unbracket(out_b)),
            String::from_utf8_lossy(&out_u),
            "{brackets}"
        );
        assert_eq!(dose_b, dose_u, "{brackets}");
    }
    let _ = std::fs::remove_dir_all(&work);
}
