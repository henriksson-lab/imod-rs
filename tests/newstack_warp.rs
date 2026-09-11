//! `newstack -distort` and `-gradient`: the distortion-field and mag-gradient
//! correction that `warpInterp` applies in place of `cubinterp`.
//!
//! A warping file given to `-xform` takes the same interpolator with
//! `linFirst` set.
//!
//! `-distort` reads the field with `readCheckWarpFile`, then fetches a
//! size-adjusted grid per section (`newstack.f90:2077-2080`); `-gradient`
//! builds a grid from the tilt/mag/rotation table
//! (`newstack.f90:2094-2110`); with both, the gradient field is *added* to the
//! distortion field.  `-subarea` shifts the grid start
//! (`newstack.f90:2424-2426`).
//!
//! `fixtures/newstack-warp-input.mrc` and `fixtures/newstack-warp.{idf,mgt}` are authored by
//! `fixtures/make-newstack-warp-inputs.py`; the `.mrc` goldens are the
//! reference `newstack`'s own output with the label block zeroed, and the
//! `.txt` goldens its stdout.  Output mode 2 keeps every interpolated value
//! rather than rounding it into a byte.

use std::process::Command;

const AUTODOC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc");

#[test]
fn distortion_and_gradient_corrections_match_the_reference_output() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("imod-rs-nswarp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(
        root.join("fixtures/newstack-warp-input.mrc"),
        dir.join("in.mrc"),
    )
    .unwrap();
    for name in ["newstack-warp.idf", "newstack-warp.mgt", "newstack-warp.xf"] {
        std::fs::copy(root.join("fixtures").join(name), dir.join(name)).unwrap();
    }

    for (name, args) in [
        ("distort", vec!["-distort", "newstack-warp.idf"]),
        ("gradient", vec!["-gradient", "newstack-warp.mgt"]),
        (
            "both",
            vec![
                "-distort",
                "newstack-warp.idf",
                "-gradient",
                "newstack-warp.mgt",
            ],
        ),
        (
            "subarea",
            vec!["-distort", "newstack-warp.idf", "-subarea", "4,-3"],
        ),
        ("linear", vec!["-gradient", "newstack-warp.mgt", "-linear"]),
        ("nearest", vec!["-distort", "newstack-warp.idf", "-nearest"]),
        // A *warping* file given to `-xform`: the linear part of each warping
        // becomes the section transform and the grid comes from
        // `getSizeAdjustedGrid` with `linFirst` set, so `warpInterp` warps
        // *after* the transform (`newstack.f90:784-797, 2064-2076`).
        ("xfwarp", vec!["-xform", "newstack-warp.xf"]),
        (
            "xfwarpoff",
            vec!["-xform", "newstack-warp.xf", "-offset", "3,-2"],
        ),
        // `-shrink` divides both the offsets and the transform shifts by
        // `readReduction` (`newstack.f90:1290-1300`), which is the shrink
        // factor here and not the binning factor.
        (
            "xfwarpshrink",
            vec!["-xform", "newstack-warp.xf", "-shrink", "2"],
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_newstack"))
            .current_dir(&dir)
            .env("AUTODOC_DIR", AUTODOC)
            .args(&args)
            .args(["-mode", "2", "-input", "in.mrc", "-output", "o.mrc"])
            .output()
            .expect("newstack executable must start");
        assert!(
            output.status.success(),
            "newstack {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let want_text =
            std::fs::read_to_string(root.join(format!("fixtures/newstack-warp-{name}.txt")))
                .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            want_text,
            "{name}: stdout must match the reference run"
        );
        let mut got = std::fs::read(dir.join("o.mrc")).unwrap();
        let want = std::fs::read(root.join(format!("fixtures/newstack-warp-{name}.mrc"))).unwrap();
        for byte in &mut got[224..1024] {
            *byte = 0;
        }
        assert_eq!(
            got.len(),
            want.len(),
            "{name}: output size must match the reference"
        );
        let differing = (0..want.len()).filter(|i| want[*i] != got[*i]).count();
        assert_eq!(
            differing, 0,
            "{name}: {differing} bytes differ from the reference output"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
