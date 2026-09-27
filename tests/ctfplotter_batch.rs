//! `ctfplotter` batch reimplementation (`src/imod/ctfplotter/batch`):
//! tolerance-based checks on tilt series simulated in the test with known
//! defocus, astigmatism and phase shift.  Not byte goldens: the owner's
//! acceptance criterion for this reimplementation is closeness to native and
//! to ground truth (`TODO.md`, "ctfplotter reimplementation").
//!
//! Each view is white noise filtered by the CTF (the formula ctfplotter
//! uses) with a mild envelope, scaled to about 40 counts per pixel, plus
//! Gaussian noise with Poisson variance.

#![cfg(feature = "rustfft-backend")]

mod common;

use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};

use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

const N: usize = 512;
const PIX_NM: f64 = 0.25;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "imod-rs-ctfplotter-{}-{}",
        name,
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// xorshift64* uniform in (0, 1).
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (2.0 * PI * self.uniform()).cos()
    }
}

struct Sim {
    /// Defocus of each view in microns.
    defocus: Vec<f64>,
    angles: Vec<f64>,
    /// Astigmatism in microns and its axis in degrees.
    astig: (f64, f64),
    phase_deg: f64,
}

/// Simulates the views and writes `<dir>/sim.mrc` and `<dir>/sim.tlt`.
fn simulate(dir: &Path, sim: &Sim, seed: u64) {
    let (kv, cs, amp) = (300.0f64, 2.7f64, 0.07f64);
    let wl = 1.23984 / (kv * (kv + 1022.0)).sqrt();
    let cs1 = (cs * wl).sqrt();
    let cs2 = (1.0e6 * cs / wl).sqrt().sqrt();
    let amp_phase = (-amp / (1.0 - amp * amp).sqrt()).atan();
    let mut planner = FftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(N);
    let inv = planner.plan_fft_inverse(N);
    let fft2 = |data: &mut Vec<Complex64>, f: &std::sync::Arc<dyn rustfft::Fft<f64>>| {
        for row in data.chunks_mut(N) {
            f.process(row);
        }
        let mut col = vec![Complex64::new(0.0, 0.0); N];
        for x in 0..N {
            for y in 0..N {
                col[y] = data[y * N + x];
            }
            f.process(&mut col);
            for y in 0..N {
                data[y * N + x] = col[y];
            }
        }
    };
    let mut rng = Rng(seed);
    let mut all = Vec::with_capacity(N * N * sim.defocus.len());
    for &def in &sim.defocus {
        let mut data: Vec<Complex64> = (0..N * N)
            .map(|_| Complex64::new(rng.gauss(), 0.0))
            .collect();
        fft2(&mut data, &fwd);
        for y in 0..N {
            let fy = if y <= N / 2 {
                y as f64
            } else {
                y as f64 - N as f64
            } / N as f64;
            for x in 0..N {
                let fx = if x <= N / 2 {
                    x as f64
                } else {
                    x as f64 - N as f64
                } / N as f64;
                let k = (fx * fx + fy * fy).sqrt() / PIX_NM;
                let dir_deg = fy.atan2(fx).to_degrees();
                let d =
                    def + 0.5 * sim.astig.0 * (2.0 * (dir_deg - sim.astig.1).to_radians()).cos();
                let theta = k * wl * cs2;
                let phi = 0.5 * PI * (theta.powi(4) - 2.0 * theta * theta * d / cs1) + amp_phase
                    - sim.phase_deg.to_radians();
                let env = (-0.25 * k * k).exp();
                data[y * N + x] *= -2.0 * phi.sin() * env;
            }
        }
        fft2(&mut data, &inv);
        let vals: Vec<f64> = data.iter().map(|c| c.re).collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let sd =
            (vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64).sqrt();
        for v in vals {
            let signal = 40.0 * (1.0 + 0.3 * (v - mean) / sd);
            all.push((signal + signal.max(0.0).sqrt() * rng.gauss()) as f32);
        }
    }
    write_mrc(&dir.join("sim.mrc"), &all, sim.defocus.len());
    let tlt: String = sim.angles.iter().map(|a| format!("{a:.2}\n")).collect();
    fs::write(dir.join("sim.tlt"), tlt).unwrap();
}

fn write_mrc(path: &Path, data: &[f32], nz: usize) {
    let mut h = vec![0u8; 1024];
    let put_i =
        |h: &mut Vec<u8>, w: usize, v: i32| h[w * 4..w * 4 + 4].copy_from_slice(&v.to_le_bytes());
    let put_f =
        |h: &mut Vec<u8>, w: usize, v: f32| h[w * 4..w * 4 + 4].copy_from_slice(&v.to_le_bytes());
    for (w, v) in [
        (0, N as i32),
        (1, N as i32),
        (2, nz as i32),
        (3, 2),
        (7, N as i32),
        (8, N as i32),
        (9, nz as i32),
        (16, 1),
        (17, 2),
        (18, 3),
    ] {
        put_i(&mut h, w, v);
    }
    for (w, v) in [
        (10, (N as f64 * PIX_NM * 10.0) as f32),
        (11, (N as f64 * PIX_NM * 10.0) as f32),
        (12, (nz as f64 * PIX_NM * 10.0) as f32),
        (13, 90.0),
        (14, 90.0),
        (15, 90.0),
    ] {
        put_f(&mut h, w, v);
    }
    let (mn, mx) = data
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let mean = data.iter().map(|&v| v as f64).sum::<f64>() / data.len() as f64;
    put_f(&mut h, 19, mn);
    put_f(&mut h, 20, mx);
    put_f(&mut h, 21, mean as f32);
    h[208..212].copy_from_slice(b"MAP ");
    h[212] = 0x44;
    h[213] = 0x44;
    let mut bytes = h;
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

/// Runs ctfplotter in `dir` with the common options plus `extra`.
fn run(dir: &Path, extra: &[&str]) -> std::process::Output {
    let mut cmd = common::imod_cmd("ctfplotter");
    cmd.current_dir(dir).args([
        "-input",
        "sim.mrc",
        "-angleFn",
        "sim.tlt",
        "-defFn",
        "sim.defocus",
        "-aAngle",
        "0",
        "-pixelSize",
        "0.25",
        "-volt",
        "300",
        "-cs",
        "2.7",
        "-save",
    ]);
    cmd.args(extra);
    cmd.output().expect("run ctfplotter")
}

/// Defocus file rows: (start view, end view, low angle, high angle, values...).
fn read_defocus(path: &Path) -> (Option<i32>, Vec<Vec<f64>>) {
    let text = fs::read_to_string(path).expect("defocus file written");
    let mut rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .map(|v| v.parse::<f64>().unwrap())
                .collect()
        })
        .collect();
    let mut flags = None;
    if rows[0].len() == 6 && rows[0][1] == 0.0 && rows[0][5] == 3.0 {
        flags = Some(rows[0][0] as i32);
        rows.remove(0);
    }
    (flags, rows)
}

fn five_views(defocus: f64, astig: (f64, f64), phase_deg: f64) -> Sim {
    Sim {
        defocus: (0..5).map(|i| defocus + 0.04 * (i as f64 - 2.0)).collect(),
        angles: (0..5).map(|i| -6.0 + 3.0 * i as f64).collect(),
        astig,
        phase_deg,
    }
}

#[test]
fn autofit_single_views_recovers_defocus_and_writes_version_2() {
    let dir = scratch("plain");
    let sim = five_views(2.5, (0.0, 0.0), 0.0);
    simulate(&dir, &sim, 11);
    let out = run(&dir, &["-expDef", "2800", "-autoFit", "0,0"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let text = fs::read_to_string(dir.join("sim.defocus")).unwrap();
    let first = text.lines().next().unwrap();
    assert!(
        first.ends_with("   2"),
        "version 2 marker on first line: {first:?}"
    );
    let (flags, rows) = read_defocus(&dir.join("sim.defocus"));
    assert_eq!(flags, None);
    assert_eq!(rows.len(), 5);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row[0] as usize, i + 1);
        assert_eq!(row[1] as usize, i + 1);
        assert!((row[2] - sim.angles[i]).abs() < 0.01);
        let truth = 1000.0 * sim.defocus[i];
        // 1.5%: native ctfplotter and this program are both within about
        // 0.3% of the truth on such data
        assert!(
            (row[4] - truth).abs() < 0.015 * truth,
            "view {}: {} vs truth {truth}",
            i + 1,
            row[4]
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn scan_finds_defocus_without_expected_value() {
    let dir = scratch("scan");
    let sim = five_views(2.0, (0.0, 0.0), 0.0);
    simulate(&dir, &sim, 12);
    let out = run(&dir, &["-scan", "1000,6000", "-autoFit", "0,0"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("Best focus with"));
    let (_, rows) = read_defocus(&dir.join("sim.defocus"));
    for (i, row) in rows.iter().enumerate() {
        let truth = 1000.0 * sim.defocus[i];
        assert!(
            (row[4] - truth).abs() < 0.015 * truth,
            "view {}: {} vs truth {truth}",
            i + 1,
            row[4]
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn astigmatism_is_found_and_written_as_version_3() {
    let dir = scratch("astig");
    let sim = five_views(2.5, (0.4, 30.0), 0.0);
    simulate(&dir, &sim, 13);
    let out = run(
        &dir,
        &[
            "-expDef",
            "2700",
            "-autoFit",
            "0,0",
            "-find",
            "1,0,0",
            "-minViews",
            "5,1",
        ],
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let (flags, rows) = read_defocus(&dir.join("sim.defocus"));
    assert_eq!(flags, Some(1), "astigmatism flag");
    assert_eq!(rows.len(), 5);
    for (i, row) in rows.iter().enumerate() {
        let (d1, d2, ang) = (row[4], row[5], row[6]);
        let truth = 1000.0 * sim.defocus[i];
        assert!(
            ((d1 + d2) / 2.0 - truth).abs() < 0.015 * truth,
            "view {}: mean {} vs {truth}",
            i + 1,
            (d1 + d2) / 2.0
        );
        assert!(
            (d1 - d2 - 400.0).abs() < 60.0,
            "view {}: astigmatism {} vs 400",
            i + 1,
            d1 - d2
        );
        let dang = ((ang - 30.0 + 90.0).rem_euclid(180.0)) - 90.0;
        assert!(dang.abs() < 5.0, "view {}: axis {ang} vs 30", i + 1);
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn phase_shift_is_found() {
    let dir = scratch("phase");
    let sim = five_views(1.5, (0.0, 0.0), 90.0);
    simulate(&dir, &sim, 14);
    let out = run(
        &dir,
        &[
            "-expDef",
            "1500",
            "-autoFit",
            "0,0",
            "-find",
            "0,1,0",
            "-degPhase",
            "70",
        ],
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let (flags, rows) = read_defocus(&dir.join("sim.defocus"));
    assert_eq!(flags, Some(4), "phase flag");
    for (i, row) in rows.iter().enumerate() {
        let truth = 1000.0 * sim.defocus[i];
        assert!(
            (row[4] - truth).abs() < 0.02 * truth,
            "view {}: {} vs truth {truth}",
            i + 1,
            row[4]
        );
        assert!(
            (row[5] - 90.0).abs() < 5.0,
            "view {}: phase {} vs 90",
            i + 1,
            row[5]
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn missing_required_entries_fail_with_status_1() {
    let dir = scratch("errors");
    let sim = Sim {
        defocus: vec![2.0],
        angles: vec![0.0],
        astig: (0.0, 0.0),
        phase_deg: 0.0,
    };
    simulate(&dir, &sim, 15);
    let out = common::imod_cmd("ctfplotter")
        .current_dir(&dir)
        .args([
            "-input",
            "sim.mrc",
            "-defFn",
            "x.defocus",
            "-cs",
            "2.7",
            "-expDef",
            "2000",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stdout).contains("Voltage is not specified"));
    let out = common::imod_cmd("ctfplotter")
        .current_dir(&dir)
        .args([
            "-input",
            "sim.mrc",
            "-defFn",
            "x.defocus",
            "-volt",
            "300",
            "-cs",
            "2.7",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&out.stdout)
            .contains("Neither expected defocus nor a range to scan")
    );
    assert!(!dir.join("x.defocus").exists());
    let _ = fs::remove_dir_all(&dir);
}
