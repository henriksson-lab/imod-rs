//! Native-golden coverage for `transferfid` (`IMOD/pysrc/transferfid`, translated in
//! `src/imod/pysrc/transferfid.rs`).
//!
//! Every row of `fixtures/transferfid/cases.tsv` was run through the native Python
//! script by `fixtures/make-transferfid-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn transferfid_matches_native_goldens() {
    let failures = pysetup_common::run_cases("transferfid");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Runs `imod transferfid` in a fresh directory holding the named inputs
/// (`source:target`) plus `extra` files written from text; returns the exit
/// status, stdout and the named output files.
fn transfer_run(
    case: &str,
    inputs: &[&str],
    extra: &[(&str, String)],
    args: &str,
    outputs: &[&str],
) -> (Option<i32>, Vec<u8>, Vec<Vec<u8>>) {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("fixtures/transferfid/inputs");
    let work =
        std::env::temp_dir().join(format!("imod-rs-transferfid-{case}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    for item in inputs {
        let (from, to) = item.split_once(':').unwrap();
        std::fs::copy(source.join(from), work.join(to)).unwrap();
    }
    for (name, text) in extra {
        std::fs::write(work.join(name), text).unwrap();
    }
    let output = common::imod_cmd("transferfid")
        .args(args.split(' '))
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
        .env("IMOD_TMPDIR", &work)
        .output()
        .unwrap();
    let files = outputs
        .iter()
        .map(|name| std::fs::read(work.join(name)).unwrap_or_default())
        .collect();
    let _ = std::fs::remove_dir_all(&work);
    (output.status.code(), output.stdout, files)
}

const COMMON: [&str; 9] = [
    "seta.st:seta.st",
    "setb.st:setb.st",
    "seta.rawtlt:seta.rawtlt",
    "setb.rawtlt:setb.rawtlt",
    "seta.tlt:seta.tlt",
    "setb.tlt:setb.tlt",
    "seta.fid:seta.fid",
    "tilta.com:tilta.com",
    "tiltb.com:tiltb.com",
];

/// Fixed in translation (BUGS.md, `transferfid`): `LocalAreaTracking 1` in
/// the track file makes beadtrack track objects together, as the source's
/// comment intends.  Defined behaviour: the same seed model as a track file
/// that has `TrackObjectsTogether` itself, which is the native golden
/// `tt_plain`.
#[test]
fn local_tracking_tracks_objects_together() {
    let mut lat1 = COMMON.to_vec();
    lat1.extend(["lat1a.com:tracka.com", "lat1b.com:trackb.com"]);
    let mut together = COMMON.to_vec();
    together.extend(["tta.com:tracka.com", "ttb.com:trackb.com"]);
    let ours = transfer_run("lat1", &lat1, &[], "-s set -n 3", &["setb.seed"]);
    assert_eq!(ours.0, Some(0));
    assert_eq!(
        ours,
        transfer_run("tt", &together, &[], "-s set -n 3", &["setb.seed"])
    );
}

/// Fixed in translation (BUGS.md, `transferfid`): with no `.rawtlt` file the
/// angles come from `FirstTiltAngle` and `TiltIncrement` as numbers (native
/// builds lists and fails on them in the boundary-model path).  Defined
/// behaviour: the same result as a `.rawtlt` file holding those angles.
#[test]
fn angles_from_first_and_increment() {
    let increments = |name: &str| {
        let text = std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/transferfid/inputs")
                .join(name),
        )
        .unwrap();
        text + "FirstTiltAngle\t-30.\nTiltIncrement\t10.\n"
    };
    let no_rawtlt: Vec<&str> = COMMON
        .iter()
        .copied()
        .filter(|item| !item.contains("rawtlt"))
        .collect();
    let angles: String = (0..7).map(|i| format!("{}\n", -30 + 10 * i)).collect();
    let extra = [
        ("tracka.com", increments("t0a.com")),
        ("trackb.com", increments("t0b.com")),
    ];
    let args = "-s set -n 3 -boundary seta.fid";
    let computed = transfer_run("incr", &no_rawtlt, &extra, args, &["setb.fid"]);
    assert_eq!(
        computed.0,
        Some(0),
        "{}",
        String::from_utf8_lossy(&computed.1)
    );
    let mut with_files = extra.to_vec();
    with_files.push(("seta.rawtlt", angles.clone()));
    with_files.push(("setb.rawtlt", angles));
    let from_files = transfer_run("files", &no_rawtlt, &with_files, args, &["setb.fid"]);
    assert_eq!(computed, from_files);
}
