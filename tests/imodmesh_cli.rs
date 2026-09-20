//! `imodmesh` command-line parity fixtures (`IMOD/imodutil/imodmesh.c` plus
//! the whole of `IMOD/libmesh`).
//!
//! The models are authored by IMOD's own `wmod2imod` from WIMP text (the
//! format `imod_to_wmod` writes, `imodel_to.c:36`), so the container and every
//! contour in it is written by IMOD rather than by this crate.  With
//! `IMOD_NATIVE_IMODMESH` pointing at a native `imodmesh`, every case in the
//! matrix runs on both sides and stdout, stderr, exit status and the written
//! model bytes are compared, masking only the regions `CLAUDE.md` records as
//! unmatchable: `Imod.name[128]` past what `imodDefault` writes, and the
//! `MINX` chunk's `oscale`/`orot`.
//!
//! `IMOD_NATIVE_WMOD2IMOD` (or a native `wmod2imod` beside `IMOD_NATIVE_IMODMESH`)
//! is what builds the fixtures; without it the suite is skipped, because a
//! model this crate wrote is not an IMOD-authored fixture.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("imod-rs-imodmesh-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One WIMP contour: the display switch that selects its object, and the
/// points.
type Contour = (i32, Vec<(f32, f32, f32)>);

fn circle(cx: f32, cy: f32, r: f32, z: f32, n: i32) -> Vec<(f32, f32, f32)> {
    (0..n)
        .map(|i| {
            let a = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            (cx + r * a.cos(), cy + r * a.sin(), z)
        })
        .collect()
}

fn square(cx: f32, cy: f32, r: f32, z: f32) -> Vec<(f32, f32, f32)> {
    vec![
        (cx - r, cy - r, z),
        (cx + r, cy - r, z),
        (cx + r, cy + r, z),
        (cx - r, cy + r, z),
    ]
}

/// Write a WIMP text file in exactly the layout `imod_to_wmod`
/// (`imodel_to.c:64-101`) produces, which is what `imod_from_wmod` parses.
fn write_wimp(path: &Path, contours: &[Contour]) {
    let cnum = contours.len();
    let pnum: usize = contours.iter().map(|c| c.1.len()).sum();
    let mut text = String::new();
    text.push_str(&format!(
        " Model file name........................{}\n",
        path.file_name().unwrap().to_string_lossy()
    ));
    text.push_str(&format!(
        " max # of object....................... {:4}\n",
        2 * cnum
    ));
    text.push_str(&format!(
        " # of node............................. {:4}\n",
        2 * pnum
    ));
    text.push_str(&format!(
        " # of object........................... {:4}\n",
        cnum
    ));
    text.push_str("  Object sequence : \n");
    let mut object_count = 1;
    let mut point_count = 1;
    for (switch, points) in contours {
        text.push_str(&format!("  Object #: {object_count:11}\n"));
        object_count += 1;
        text.push_str(&format!(" # of point: {:11}\n", points.len()));
        text.push_str(&format!(" Display switch:1  {switch}\n"));
        text.push_str("     #    X       Y       Z      Mark    Label \n");
        point_count += 1;
        for (x, y, z) in points {
            text.push_str(&format!("{point_count:7} {x:7.2} {y:7.2} {z:7.2}   0\n"));
            point_count += 1;
        }
    }
    text.push_str("\n  END\n");
    std::fs::write(path, text).unwrap();
}

/// The fixture set: a plain stack of closed contours, a branching surface, a
/// nested annulus, open contours running through Z, and a mixed model.
fn fixtures() -> Vec<(&'static str, Vec<Contour>)> {
    let mut closed = Vec::new();
    for z in 0..5 {
        closed.push((247, circle(50., 50., 20., z as f32, 24)));
    }

    let mut branch = vec![(247, circle(50., 50., 25., 0., 24))];
    for z in 1..4 {
        branch.push((247, circle(35., 50., 10., z as f32, 24)));
        branch.push((247, circle(70., 50., 10., z as f32, 24)));
    }
    branch.push((247, circle(50., 50., 25., 4., 24)));

    let mut nested = Vec::new();
    for z in 0..4 {
        nested.push((247, circle(60., 60., 30., z as f32, 24)));
        nested.push((247, circle(60., 60., 14., z as f32, 24)));
    }

    let open = vec![
        (
            248,
            vec![
                (10., 10., 0.),
                (12., 14., 1.),
                (16., 18., 2.),
                (20., 25., 3.),
                (24., 30., 4.),
            ],
        ),
        (
            248,
            vec![
                (40., 10., 0.),
                (42., 15., 1.),
                (45., 20., 2.),
                (49., 26., 3.),
                (52., 31., 4.),
            ],
        ),
    ];

    let mut mixed = Vec::new();
    for z in 0..3 {
        mixed.push((247, square(30., 30., 12., z as f32)));
    }
    mixed.push((248, vec![(70., 70., 0.), (72., 74., 1.), (76., 78., 2.)]));

    vec![
        ("closed", closed),
        ("branch", branch),
        ("nested", nested),
        ("open", open),
        ("mixed", mixed),
    ]
}

/// Every case is `(name, model, argv)`; the model names index `fixtures()`
/// plus `erase`, a copy of `IMOD/Etomo/uitestData/BB/BBa_erase.fid`.
fn cases() -> Vec<(&'static str, &'static str, Vec<&'static str>)> {
    vec![
        ("plain", "closed", vec![]),
        ("cap-end", "closed", vec!["-c"]),
        ("cap-all", "closed", vec!["-C"]),
        ("cap-all-nested", "nested", vec!["-C"]),
        ("surface", "branch", vec!["-S"]),
        ("skip", "closed", vec!["-s"]),
        ("passes", "branch", vec!["-P", "2"]),
        ("zrange", "closed", vec!["-z", "1,3,1"]),
        ("zinc", "closed", vec!["-i", "2"]),
        ("objlist", "mixed", vec!["-o", "1"]),
        ("objlist2", "mixed", vec!["-o", "2"]),
        ("lowres", "closed", vec!["-l"]),
        ("timeconsuming", "closed", vec!["-T"]),
        ("force", "branch", vec!["-f"]),
        ("times", "closed", vec!["-I"]),
        ("tubes", "open", vec!["-t", "1"]),
        ("tubes-diam", "open", vec!["-t", "1", "-d", "6"]),
        ("tubes-cap", "open", vec!["-t", "1", "-d", "6", "-E"]),
        ("tubes-dome", "open", vec!["-t", "1", "-d", "6", "-H"]),
        ("open-surface", "open", vec![]),
        ("clip-x", "closed", vec!["-x", "40,60"]),
        ("clip-y", "closed", vec!["-y", "40,60"]),
        ("clip-xy", "nested", vec!["-x", "40,80", "-y", "40,80"]),
        ("overlap", "branch", vec!["-p", "10"]),
        ("flatcrit", "closed", vec!["-F", "0.5"]),
        ("tol", "closed", vec!["-R", "1.0"]),
        ("zscale", "closed", vec!["-Z", "2"]),
        ("backward", "closed", vec!["-B"]),
        ("nocap-skip", "closed", vec!["-C", "-D", "0"]),
        ("erase-flag", "closed", vec!["-e"]),
        ("append", "closed", vec!["-a"]),
        ("recompute-normals", "closed", vec!["-N"]),
        ("rescale-normals", "closed", vec!["-n", "-Z", "1.5"]),
        ("use-old", "closed", vec!["-u"]),
        ("no-cap", "closed", vec!["-noc"]),
        ("no-surf", "closed", vec!["-noS"]),
        ("fid-plain", "erase", vec![]),
        ("fid-tubes", "erase", vec!["-t", "1", "-d", "4"]),
        ("fid-cap", "erase", vec!["-c"]),
        // Error and usage paths.
        ("no-args", "", vec![]),
        ("bad-option", "closed", vec!["-Q"]),
        ("conflicting", "closed", vec!["-c", "-noc"]),
        ("missing-model", "", vec!["nosuch.mod"]),
    ]
}

/// Blank the regions a native model comparison cannot match: the model name
/// field past the 13 bytes `imodDefault` writes (native leaks heap residue
/// there) and, when present, the `MINX` chunk's uninitialised `oscale` and
/// `orot`.
fn mask(bytes: &[u8]) -> Vec<u8> {
    let mut masked = bytes.to_vec();
    if masked.len() > 136 && masked.starts_with(b"IMODV1.2") {
        masked[8 + 13..136].fill(0);
    }
    let mut index = 0;
    while index + 80 <= masked.len() {
        if &masked[index..index + 4] == b"MINX" {
            // After the 4-byte id and the 4-byte chunk size comes `IrefImage`
            // in field order: oscale, otrans, orot, cscale, ctrans, crot.
            let base = index + 8;
            masked[base..base + 12].fill(0); // oscale
            masked[base + 24..base + 36].fill(0); // orot
            index += 80;
        } else {
            index += 1;
        }
    }
    masked
}

/// Build the fixture models with a native `wmod2imod`.  Returns `None` when no
/// native binary is available, in which case the suite is skipped rather than
/// falling back on a model this crate wrote.
fn build_models(dir: &Path) -> Option<()> {
    let wmod2imod = std::env::var("IMOD_NATIVE_WMOD2IMOD").ok().or_else(|| {
        let native = std::env::var("IMOD_NATIVE_IMODMESH").ok()?;
        let beside = Path::new(&native).with_file_name("wmod2imod");
        beside
            .exists()
            .then(|| beside.to_string_lossy().into_owned())
    })?;
    for (name, contours) in fixtures() {
        let wimp = dir.join(format!("{name}.wimp"));
        write_wimp(&wimp, &contours);
        let model = dir.join(format!("{name}.mod"));
        let status = Command::new(&wmod2imod)
            .current_dir(dir)
            .arg(&wimp)
            .arg(&model)
            .status()
            .ok()?;
        if !status.success() || !model.exists() {
            return None;
        }
    }
    let fid = Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
    if fid.exists() {
        std::fs::copy(&fid, dir.join("erase.mod")).ok()?;
    }
    Some(())
}

/// Run one case on one side, into a fresh copy of the model, and return the
/// process result plus the model bytes it left behind.
fn run_case(
    dir: &Path,
    side: &str,
    name: &str,
    model: &str,
    argv: &[&str],
    native: Option<&str>,
) -> (std::process::Output, Option<Vec<u8>>) {
    let target = dir.join(format!("{side}-{name}.mod"));
    let _ = std::fs::remove_file(&target);
    let _ = std::fs::remove_file(dir.join(format!("{side}-{name}.mod~")));
    if !model.is_empty() {
        let source = dir.join(format!("{model}.mod"));
        std::fs::copy(&source, &target).unwrap();
    }
    let mut command = match native {
        Some(path) => Command::new(path),
        None => common::imod_cmd("imodmesh"),
    };
    command.current_dir(dir).args(argv);
    if !model.is_empty() {
        command.arg(target.file_name().unwrap());
    } else if name == "missing-model" {
        command.arg("nosuch.mod");
    }
    // Keep the meshing deterministic: the source's skinning loop is
    // OpenMP-parallel and concatenates one mesh array per thread, so its
    // output order is thread-count dependent.  The translation is sequential,
    // which is native at one thread.
    command.env("OMP_NUM_THREADS", "1");
    let output = command.output().unwrap();
    let written = std::fs::read(&target).ok();
    (output, written)
}

/// Without a native binary the suite still exercises every case, checking that
/// the program runs, meshes, and leaves a readable model behind.
#[test]
fn every_case_runs_and_meshes() {
    let dir = scratch();
    if build_models(&dir).is_none() {
        return;
    }
    for (name, model, argv) in cases() {
        if model == "erase" && !dir.join("erase.mod").exists() {
            continue;
        }
        let (result, written) = run_case(&dir, "rs", name, model, &argv, None);
        match name {
            "no-args" | "bad-option" | "conflicting" | "missing-model" => {
                assert!(
                    !result.status.success(),
                    "{name} should have failed: {result:?}"
                );
            }
            _ => {
                assert!(result.status.success(), "{name}: {result:?}");
                let bytes = written.expect("no model written");
                assert!(bytes.starts_with(b"IMODV1.2"), "{name}: not a model");
                if name != "erase-flag" {
                    assert!(
                        bytes.windows(4).any(|w| w == b"MESH"),
                        "{name}: no mesh chunk in the output"
                    );
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// With `IMOD_NATIVE_IMODMESH` set, every case runs on both sides and the
/// written model, stdout, stderr and exit status must agree.
#[test]
fn native_comparison_matches_every_case() {
    let Ok(native) = std::env::var("IMOD_NATIVE_IMODMESH") else {
        return;
    };
    let dir = scratch();
    if build_models(&dir).is_none() {
        return;
    }

    for (name, model, argv) in cases() {
        if model == "erase" && !dir.join("erase.mod").exists() {
            continue;
        }
        let (native_run, native_file) = run_case(&dir, "nat", name, model, &argv, Some(&native));
        let (rust_run, rust_file) = run_case(&dir, "rs", name, model, &argv, None);

        assert_eq!(
            native_run.status.code(),
            rust_run.status.code(),
            "{name} exit status"
        );
        // The program prints the model path it is working on, which differs
        // between the two sides only in the `nat-`/`rs-` prefix this harness
        // gives each copy.
        let normalise = |bytes: &[u8], side: &str| {
            String::from_utf8_lossy(bytes).replace(&format!("{side}-{name}.mod"), "MODEL")
        };
        assert_eq!(
            normalise(&native_run.stdout, "nat"),
            normalise(&rust_run.stdout, "rs"),
            "{name} stdout"
        );
        assert_eq!(
            normalise(&native_run.stderr, "nat"),
            normalise(&rust_run.stderr, "rs"),
            "{name} stderr"
        );
        match (native_file, rust_file) {
            (Some(native_bytes), Some(rust_bytes)) => {
                assert_eq!(
                    mask(&native_bytes),
                    mask(&rust_bytes),
                    "{name} model bytes differ"
                );
            }
            (None, None) => {}
            (native_bytes, rust_bytes) => panic!(
                "{name}: one side wrote a model and the other did not ({}, {})",
                native_bytes.is_some(),
                rust_bytes.is_some()
            ),
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
