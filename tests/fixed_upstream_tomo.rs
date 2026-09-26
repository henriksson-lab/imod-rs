//! Behaviour the translation defines where upstream `tilt`, `tiltalign`,
//! `beadtrack` and `tiltxcorr` have a defect (`BUGS.md`, entries marked "Fixed
//! in translation (2026-09-26)").  Native cannot be the reference for these
//! cases -- it crashes, hangs, reads unset memory or computes the wrong thing
//! -- so each test asserts the defined behaviour directly.  Cases the fixes do
//! not touch stay covered by the native goldens in `tilt_cli`, `tiltalign_cli`,
//! `beadtrack_cli` and `tiltxcorr_cli`.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

fn fixture(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(dir)
}

/// A scratch copy of every plain file in `fixtures/<dir>`.
fn scratch(dir: &str, name: &str) -> PathBuf {
    let out =
        std::env::temp_dir().join(format!("imod-rs-fixedtomo-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();
    for entry in std::fs::read_dir(fixture(dir)).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            std::fs::copy(&path, out.join(path.file_name().unwrap())).unwrap();
        }
    }
    out
}

/// Runs `command` in `dir` and kills it after `limit` (a hang is a failure).
fn run(mut command: Command, dir: &Path, limit: Duration) -> Output {
    command
        .current_dir(dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .env("OMP_NUM_THREADS", "1")
        .env("IMOD_NO_IMAGE_BACKUP", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().unwrap();
    let start = Instant::now();
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if start.elapsed() > limit {
            let _ = child.kill();
            panic!("command did not finish within {limit:?}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

const TILT_BASE: &str = "-input t.ali -output o.rec -TILTFILE t.tlt -MODE 1";

fn tilt(dir: &Path, extra: &str) -> Output {
    let mut command = common::imod_cmd("tilt");
    command.args(TILT_BASE.split_whitespace());
    command.args(extra.split_whitespace());
    run(command, dir, Duration::from_secs(120))
}

/// `setCosStretch` stepped its corner loops by `mIwidth - 1` and
/// `mIthickBP - 1`: native loops forever for `-WIDTH 1` or `-THICKNESS 1`.
/// Defined: the step is at least 1, and the run completes.
#[test]
fn tilt_width_or_thickness_one_terminates() {
    for (name, extra) in [("w1", "-THICKNESS 16 -WIDTH 1"), ("t1", "-THICKNESS 1")] {
        let dir = scratch("tilt", &format!("tilt-{name}"));
        let out = tilt(&dir, extra);
        assert_eq!(out.status.code(), Some(0), "{name}: {}", text(&out.stdout));
        assert!(dir.join("o.rec").exists(), "{name}: no output");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// `lslice % (numDo / 10)` divided by zero for fewer than 10 slices under
/// `-PARALLEL`/`-RotateBy90`: native dies with SIGFPE.  Defined: the progress
/// interval is at least 1, so every slice is reported.
#[test]
fn tilt_parallel_with_few_slices_reports_every_slice() {
    for (name, extra) in [
        ("par", "-THICKNESS 16 -PARALLEL -SLICE 4,8"),
        ("rot90", "-THICKNESS 16 -RotateBy90 -SLICE 4,8"),
    ] {
        let dir = scratch("tilt", &format!("tilt-{name}"));
        let out = tilt(&dir, extra);
        assert_eq!(out.status.code(), Some(0), "{name}: {}", text(&out.stdout));
        let stdout = text(&out.stdout);
        for slice in 1..=5 {
            assert!(
                stdout.contains(&format!("Finished slice {slice} of 5\n")),
                "{name}: slice {slice} not reported:\n{stdout}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// A missing `-LOCALFILE` reached `fgetline(NULL)` natively.  Defined: the
/// open is tested and reported (exit 1, as before).
#[test]
fn tilt_missing_local_file_reports_the_open() {
    let dir = scratch("tilt", "tilt-noloc");
    let out = tilt(&dir, "-THICKNESS 16 -LOCALFILE missing.xf");
    assert_eq!(out.status.code(), Some(1));
    let all = text(&out.stdout) + &text(&out.stderr);
    assert!(
        all.contains("Opening local alignment file missing.xf"),
        "{all}"
    );
    assert!(!all.contains("fgetline"), "{all}");
    let _ = std::fs::remove_dir_all(&dir);
}

const G1: &str = "-ModelFile g1.fid -ImageSizeXandY 1024,1024 -TiltFile g1.rawtlt \
    -OutputTransformFile g1.xf -OutputTiltFile g1.tlt -OutputResidualFile g1.resid \
    -OutputModelFile g1.3dmod -OutputFidXYZFile g1fid.xyz -RotationAngle -12 -RotOption 1 \
    -TiltOption 5 -TiltDefaultGrouping 5 -MagOption 1 -MagDefaultGrouping 4 \
    -MagReferenceView 1 -CrossValidate 0";

fn tiltalign(dir: &Path, extra: &str, autodoc: bool) -> Output {
    let mut command = common::imod_cmd("tiltalign");
    command.args(G1.split_whitespace());
    command.args(extra.split_whitespace());
    if !autodoc {
        // An empty directory: no autodoc, so the fallback option table is used.
        let empty = dir.join("noadoc");
        std::fs::create_dir_all(&empty).unwrap();
        command.env("AUTODOC_DIR", &empty);
        command.current_dir(dir);
        command
            .env("OMP_NUM_THREADS", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        return command.output().unwrap();
    }
    run(command, dir, Duration::from_secs(300))
}

/// `input_model.cpp:115` indexed the weights with `B3DMIN(numWgts, iobject)`,
/// so a single `ExtraWeights` value applied to the first listed object only
/// and the others got weight 0.  Defined: one value applies to every object,
/// which must give exactly the run with that value repeated per object.
#[test]
fn tiltalign_single_extra_weight_applies_to_every_object() {
    let one = scratch("tiltalign", "ta-w1");
    let three = scratch("tiltalign", "ta-w3");
    let a = tiltalign(&one, "-ObjectsWithExtraWeight 1-3 -ExtraWeights 2", true);
    let b = tiltalign(
        &three,
        "-ObjectsWithExtraWeight 1-3 -ExtraWeights 2,2,2",
        true,
    );
    assert_eq!(a.status.code(), Some(0), "{}", text(&a.stdout));
    assert_eq!(b.status.code(), Some(0), "{}", text(&b.stdout));
    assert_eq!(text(&a.stdout), text(&b.stdout));
    for file in ["g1.xf", "g1.tlt", "g1.resid", "g1fid.xyz"] {
        assert_eq!(
            std::fs::read(one.join(file)).unwrap(),
            std::fs::read(three.join(file)).unwrap(),
            "{file}"
        );
    }
    let _ = std::fs::remove_dir_all(&one);
    let _ = std::fs::remove_dir_all(&three);
}

/// Without an autodoc the source's fallback table lacked the extra-weight
/// options, so every run exited "Illegal option", and it typed the two
/// cross-validation coverage options boolean.  Defined: the table carries the
/// options with the types they are read as; the run matches the autodoc run.
#[test]
fn tiltalign_runs_from_its_fallback_option_table() {
    let plain = scratch("tiltalign", "ta-adoc");
    let bare = scratch("tiltalign", "ta-noadoc");
    let extra = "-SurfacesToAnalyze 2";
    let a = tiltalign(&plain, extra, true);
    let b = tiltalign(&bare, extra, false);
    assert_eq!(a.status.code(), Some(0), "{}", text(&a.stdout));
    assert_eq!(
        b.status.code(),
        Some(0),
        "{}{}",
        text(&b.stdout),
        text(&b.stderr)
    );
    assert_eq!(
        std::fs::read(plain.join("g1.xf")).unwrap(),
        std::fs::read(bare.join("g1.xf")).unwrap()
    );
    let c = tiltalign(
        &bare,
        "-CrossValidate 1 -CVMinAndMaxCoverageFactor 0.5,3 -CVCoverageTargetOrFactor 2",
        false,
    );
    assert_eq!(
        c.status.code(),
        Some(0),
        "{}{}",
        text(&c.stdout),
        text(&c.stderr)
    );
    let _ = std::fs::remove_dir_all(&plain);
    let _ = std::fs::remove_dir_all(&bare);
}

/// `FixedXYZInputFile` was checked against the number of *points* while
/// `readLinesForValues` counts values (three per point), so a file with
/// `nrealPt` to `3 * nrealPt - 1` values was accepted and the missing
/// coordinates were unset memory.  Defined: fewer than `3 * nrealPt` values
/// is the "Fewer coordinates" error.
#[test]
fn tiltalign_fixed_xyz_needs_three_values_per_point() {
    let dir = scratch("tiltalign", "ta-fixed");
    // Learn the point count from an ordinary run's XYZ output.
    let base = tiltalign(&dir, "", true);
    assert_eq!(base.status.code(), Some(0));
    let npt = std::fs::read_to_string(dir.join("g1fid.xyz"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .count();
    assert!(npt > 2);
    // npt values: enough for the source's test, a third of what is needed.
    let values: Vec<String> = (0..npt).map(|i| format!("{}.5", i % 97)).collect();
    std::fs::write(dir.join("part.xyz"), values.join("\n") + "\n").unwrap();
    let out = tiltalign(&dir, "-FixedXYZInputFile part.xyz", true);
    assert_eq!(out.status.code(), Some(1));
    let all = text(&out.stdout) + &text(&out.stderr);
    assert!(
        all.contains("Fewer coordinates in fixed XYZ input file"),
        "{all}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `readLinesForValues` returned -3 ("More coordinates ... than fiducial
/// points") only when the line that filled the array also hit end of file, so
/// native accepted a file with extra points (ignoring them) and rejected an
/// exactly fitting file whose last line had no newline.  Defined: extra
/// non-blank lines are the "More coordinates" error, and an exact fit is
/// accepted with or without a final newline.
#[test]
fn tiltalign_fixed_xyz_rejects_extra_points_only() {
    let dir = scratch("tiltalign", "ta-fixed-many");
    let base = tiltalign(&dir, "", true);
    assert_eq!(base.status.code(), Some(0));
    let npt = std::fs::read_to_string(dir.join("g1fid.xyz"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .count();
    let lines: Vec<String> = (0..npt)
        .map(|i| format!("{}.5 {}.25 {}.75", i % 97, (i * 7) % 89, i % 13))
        .collect();
    // Exact fit, no final newline: native's spurious -3.
    std::fs::write(dir.join("exact.xyz"), lines.join("\n")).unwrap();
    let out = tiltalign(&dir, "-FixedXYZInputFile exact.xyz", true);
    let all = text(&out.stdout) + &text(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{all}");
    // Exact fit plus trailing blank lines is still a fit.
    std::fs::write(dir.join("blank.xyz"), lines.join("\n") + "\n\n  \n").unwrap();
    let out = tiltalign(&dir, "-FixedXYZInputFile blank.xyz", true);
    assert_eq!(out.status.code(), Some(0));
    // Two extra points: native silently ignores them.
    let mut many = lines.clone();
    many.push("1.0 2.0 3.0".into());
    many.push("4.0 5.0 6.0".into());
    std::fs::write(dir.join("many.xyz"), many.join("\n") + "\n").unwrap();
    let out = tiltalign(&dir, "-FixedXYZInputFile many.xyz", true);
    assert_eq!(out.status.code(), Some(1));
    let all = text(&out.stdout) + &text(&out.stderr);
    assert!(
        all.contains("More coordinates in fixed XYZ input file than fiducial points"),
        "{all}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The seed-model error passed one argument to `"%s: %s"`; native prints the
/// error where the name belongs and `(null)` after it.
#[test]
fn beadtrack_seed_error_names_the_file() {
    let dir = scratch("beadtrack", "bt-seed");
    let input = "ImageFile\tt1.mrc\nInputSeedModel\tnone.seed\nOutputModel\tt1.fid\n\
                 TiltFile\tt1.rawtlt\nRotationAngle\t4\nBeadDiameter\t9\n";
    std::fs::write(dir.join("p.in"), input).unwrap();
    let mut command = common::imod_cmd("beadtrack");
    command
        .arg("-StandardInput")
        .current_dir(&dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .stdin(std::fs::File::open(dir.join("p.in")).unwrap())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let out = command.output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let all = text(&out.stdout) + &text(&out.stderr);
    assert!(all.contains("Reading seed model file none.seed: "), "{all}");
    assert!(!all.contains("(null)"), "{all}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `PipGetInOutFile("OutputFile", 2, ...)` asked for a third non-option
/// argument, so `tiltxcorr in out` failed with "No output file specified".
/// Defined: the second non-option argument is the output, and the run equals
/// the one naming both files by option.
#[test]
fn tiltxcorr_takes_the_output_file_positionally() {
    let pos = scratch("tiltxcorr", "tx-pos");
    let opt = scratch("tiltxcorr", "tx-opt");
    let mut a = common::imod_cmd("tiltxcorr");
    a.args("-tiltfile tx.tlt -rotation -6 tx.st o.xf".split_whitespace());
    let mut b = common::imod_cmd("tiltxcorr");
    b.args("-input tx.st -output o.xf -tiltfile tx.tlt -rotation -6".split_whitespace());
    let a = run(a, &pos, Duration::from_secs(120));
    let b = run(b, &opt, Duration::from_secs(120));
    assert_eq!(a.status.code(), Some(0), "{}", text(&a.stdout));
    assert_eq!(b.status.code(), Some(0), "{}", text(&b.stdout));
    assert_eq!(
        std::fs::read(pos.join("o.xf")).unwrap(),
        std::fs::read(opt.join("o.xf")).unwrap()
    );
    let _ = std::fs::remove_dir_all(&pos);
    let _ = std::fs::remove_dir_all(&opt);
}
