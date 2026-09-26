//! Native-golden coverage for `findbeads3d` (`IMOD/imodutil/findbeads3d.cpp`).
//!
//! Every case in `fixtures/findbeads3d/cases.tsv` was run through the native
//! reference `findbeads3d` by `fixtures/make-findbeads3d-goldens.sh` at
//! `OMP_NUM_THREADS=1`, stdout captured through a pipe.  The exit status is
//! `golden/<case>.rc`, stdout is `golden/<case>.out`, and every file native
//! left behind is in `golden/<case>/`.  The inputs are seeded synthetic
//! volumes with dark (float) and light (short) beads and a one-bead template,
//! all written by the native `raw2mrc`.
//!
//! Upstream defects are fixed in the translation (`BUGS.md`, 2026-09-26), so
//! a case whose output a fix changes has its expectation in `defined/`
//! instead (same layout, written from the fixed translation by
//! `fixtures/make-findbeads3d-goldens.sh defined`); `golden/` keeps the
//! native record for every case.  The fixes are asserted directly by the
//! tests after the golden loop.
//! Pruned 2026-09-26: 37 of 39 rows kept (every `defined.list` case stays; dropped `-annulus` combined with `-ylong`, each covered alone, and `e_small`, the same coordinate-range exit as `e_range`); the rest stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).

mod common;

use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/findbeads3d")
}

/// Blank the wall-clock stamps (`common::mask_stamps`).  The regions native
/// writes from uninitialised memory (`BUGS.md` §2) are reconciled first by
/// `common::reconcile_uninitialised`, which checks ours holds the defined value.
fn mask(bytes: &[u8]) -> Vec<u8> {
    common::mask_stamps(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "imod-rs-findbeads3d-{}-{}",
        std::process::id(),
        name
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            let file = path.file_name().unwrap().to_str().unwrap().to_string();
            if file.ends_with(".mrc") || file.ends_with(".tlt") || file.ends_with(".param") {
                std::fs::copy(&path, dir.join(&file)).unwrap();
            }
        }
    }
    dir
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let (name, args) = line.split_once('\t').unwrap();
        let defined = fixture_dir().join("defined");
        let golden = if common::golden::exists(&defined.join(format!("{name}.rc"))) {
            &defined
        } else {
            &golden
        };
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        let expected_out = common::golden::expect(&golden.join(format!("{name}.out")));
        let dir = scratch(name);
        let inputs: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .collect();
        let output = common::imod_cmd("findbeads3d")
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .env("OMP_NUM_THREADS", "1")
            .args(args.split_whitespace())
            .output()
            .unwrap();
        count += 1;
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}",
                output.status.code()
            ));
        }
        // The usage listing starts with a version banner carrying the build
        // date (`imodVersion`); only the lines after it are compared.
        let skip = |b: &[u8]| -> Vec<u8> {
            if b.starts_with(b"findbeads3d Version") {
                b.iter()
                    .position(|&c| c == b'\n')
                    .map_or(Vec::new(), |p| b[p + 1..].to_vec())
            } else {
                b.to_vec()
            }
        };
        if !expected_out.matches_masked(&output.stdout, skip) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                expected_out.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let expected_files: Vec<String> = common::golden::list(&golden.join(name));
        let mut produced: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .filter(|f| !inputs.contains(f))
            .collect();
        produced.sort();
        if produced != expected_files {
            failures.push(format!(
                "{name}: files {produced:?}, native {expected_files:?}"
            ));
        }
        for file in &expected_files {
            let Ok(ours) = std::fs::read(dir.join(file)) else {
                continue;
            };
            let theirs = common::golden::expect(&golden.join(name).join(file));
            if let Err(why) = theirs.compare(&ours, mask, true) {
                failures.push(format!("{name}: {file} differs from native: {why}"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 30, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Run the translation in a fresh copy of the fixtures.
fn run(name: &str, args: &[&str]) -> (Option<i32>, String, PathBuf) {
    let dir = scratch(name);
    let output = common::imod_cmd("findbeads3d")
        .current_dir(&dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .env("OMP_NUM_THREADS", "1")
        .args(args)
        .output()
        .unwrap();
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        dir,
    )
}

/// The points of a scattered-point model, through `imodinfo -a`.
fn model_points(dir: &Path, model: &str) -> Vec<[f64; 3]> {
    let output = common::imod_cmd("imodinfo")
        .current_dir(dir)
        .args(["-a", model])
        .output()
        .unwrap();
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|l| {
            let v: Vec<f64> = l
                .split_whitespace()
                .map(|t| t.parse::<f64>())
                .collect::<Result<_, _>>()
                .ok()?;
            (v.len() == 3).then(|| [v[0], v[1], v[2]])
        })
        .collect()
}

/// BUGS.md, findbeads3d: with a `-zminmax` start above 1 the first Z piece
/// was analysed as if a piece had been loaded below it, and the correlation
/// read the plane before the volume (native: SIGSEGV, exit 139).  The first
/// piece of the range is a boundary now, so the run completes.
#[test]
fn zminmax_start_above_one_completes() {
    let (rc, stdout, dir) = run(
        "fix_zminmax",
        &[
            "-in", "fb_f.mrc", "-out", "o.mod", "-size", "6", "-zminmax", "3,27",
        ],
    );
    assert_eq!(rc, Some(0), "{stdout}");
    let points = model_points(&dir, "o.mod");
    assert!(!points.is_empty(), "{stdout}");
    // Nothing outside the range (1-based Z 3..27, model Z is 0-based).
    assert!(
        points.iter().all(|p| p[2] >= 1.5 && p[2] <= 27.0),
        "{points:?}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// BUGS.md, findbeads3d: `findValueInList` returned one less than the count
/// its callers use, so `-store 1` (only the normalised top peak, 1.0, passes)
/// stored no peak natively.  It stores that one peak now.
#[test]
fn store_threshold_one_stores_the_top_peak() {
    let (rc, stdout, dir) = run(
        "fix_storeone",
        &[
            "-in", "fb_f.mrc", "-out", "o.mod", "-size", "6", "-store", "1",
        ],
    );
    assert_eq!(rc, Some(0), "{stdout}");
    assert!(
        stdout.contains("Storing 1 peaks in model above threshold of 1.0000"),
        "{stdout}"
    );
    assert_eq!(model_points(&dir, "o.mod").len(), 1);
    let _ = std::fs::remove_dir_all(dir);
}

/// BUGS.md, findbeads3d: a rejected candidate no longer moves stored peak 0
/// to its own position (`addToSortedList` returned slot 0, not -1).  On a
/// synthetic volume with 18 dark beads at known positions every bead is found.
/// (On a volume made the same way with numpy's generator the native program
/// found 16 of 18 with default settings: the overwritten slot loses beads.)
#[test]
fn every_synthetic_bead_is_found() {
    let (nx, ny, nz) = (90usize, 80usize, 40usize);
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut beads: Vec<[f64; 3]> = Vec::new();
    while beads.len() < 18 {
        let c = [
            8. + uniform() * (nx as f64 - 16.),
            8. + uniform() * (ny as f64 - 16.),
            8. + uniform() * (nz as f64 - 16.),
        ];
        if beads
            .iter()
            .all(|d| (0..3).map(|k| (c[k] - d[k]).powi(2)).sum::<f64>() > 144.)
        {
            beads.push(c);
        }
    }
    let mut vol = vec![0f64; nx * ny * nz];
    for v in vol.iter_mut() {
        let (u1, u2) = (uniform().max(1e-300), uniform());
        *v = (-2. * u1.ln()).sqrt() * (2. * std::f64::consts::PI * u2).cos();
    }
    for b in &beads {
        let amp = 3. + 3. * uniform();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let d2 = ((x as f64 - b[0]) / 3.).powi(2)
                        + ((y as f64 - b[1]) / 3.).powi(2)
                        + ((z as f64 - b[2]) / 3.9).powi(2);
                    vol[x + nx * (y + ny * z)] -= amp * (-2. * d2).exp();
                }
            }
        }
    }
    let dir = scratch("fix_truth");
    let raw: Vec<u8> = vol
        .iter()
        .flat_map(|v| ((v * 10. + 50.) as f32).to_le_bytes())
        .collect();
    std::fs::write(dir.join("gt.raw"), raw).unwrap();
    let status = common::imod_cmd("raw2mrc")
        .current_dir(&dir)
        .args([
            "-x", "90", "-y", "80", "-z", "40", "-t", "float", "gt.raw", "gt.mrc",
        ])
        .output()
        .unwrap()
        .status;
    assert!(status.success());
    let output = common::imod_cmd("findbeads3d")
        .current_dir(&dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .env("OMP_NUM_THREADS", "1")
        .args(["-in", "gt.mrc", "-out", "o.mod", "-size", "6"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let points = model_points(&dir, "o.mod");
    // Model coordinates are pixel-centred: index + 0.5.
    let missed: Vec<&[f64; 3]> = beads
        .iter()
        .filter(|b| {
            !points.iter().any(|p| {
                (0..3)
                    .map(|k| (p[k] - (b[k] + 0.5)).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    < 2.5
            })
        })
        .collect();
    assert!(missed.is_empty(), "missed {missed:?} in {points:?}");
    let _ = std::fs::remove_dir_all(dir);
}
