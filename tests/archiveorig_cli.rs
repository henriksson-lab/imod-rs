//! Native-golden coverage for `archiveorig` (`IMOD/pysrc/archiveorig`, translated in
//! `src/imod/pysrc/archiveorig.rs`).
//!
//! Every row of `fixtures/archiveorig/cases.tsv` was run through the native Python
//! script by `fixtures/make-archiveorig-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn archiveorig_matches_native_goldens() {
    let failures = pysetup_common::run_cases("archiveorig");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Fixed in translation (BUGS.md, `archiveorig`): restoring two levels from
/// `st_xray.mrc.gz` and `st_xray.mrc.gz.1` finds the numbered file and
/// rebuilds the oldest original, where native's glob never finds it.  The
/// archives are made here with our own `archiveorig` from three related
/// stacks, so the restored data must equal the oldest stack's data.
#[test]
fn restore_two_levels_rebuilds_the_oldest_original() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let work = std::env::temp_dir().join(format!("imod-rs-archiveorig-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    // Byte-mode stacks (so every difference restores exactly) on the header
    // of the float fixture, with seeded content.
    let header = {
        let mut header = std::fs::read(root.join("fixtures/boxstartend/v.mrc")).unwrap()[..1024].to_vec();
        header[12..16].copy_from_slice(&0_i32.to_le_bytes());
        header
    };
    let stack = |seed: u32| {
        let mut state = seed;
        let mut bytes = header.clone();
        for _ in 0..64 * 48 * 5 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            bytes.push((state >> 16) as u8);
        }
        bytes
    };
    let newest = stack(1);
    let middle = stack(2);
    let oldest = stack(3);
    let run = |args: &[&str]| {
        let status = common::imod_cmd("archiveorig")
            .args(args)
            .current_dir(&work)
            .env("IMOD_DIR", root.join("IMOD"))
            .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
            .output()
            .unwrap();
        assert_eq!(status.status.code(), Some(0), "{args:?}: {status:?}");
    };
    // Archive middle against oldest, keep it as level 1.
    std::fs::write(work.join("m.mrc"), &middle).unwrap();
    std::fs::write(work.join("m_orig.mrc"), &oldest).unwrap();
    run(&["m.mrc"]);
    std::fs::rename(work.join("m_xray.mrc.gz"), work.join("st_xray.mrc.gz.1")).unwrap();
    // Archive newest against middle as the current level.
    std::fs::write(work.join("st.mrc"), &newest).unwrap();
    std::fs::write(work.join("st_orig.mrc"), &middle).unwrap();
    run(&["-d", "st.mrc"]);
    run(&["-r", "-n", "2", "st.mrc"]);
    let restored = std::fs::read(work.join("st_orig.mrc")).unwrap();
    assert_eq!(restored.len(), oldest.len());
    assert!(restored[1024..] == oldest[1024..], "restored data differs");
    assert!(
        !work.join("st_orig.mrc.2").exists(),
        "intermediate level kept"
    );
    assert!(work.join("st_xray.mrc.gz.1.old").exists());
    let _ = std::fs::remove_dir_all(&work);
}
