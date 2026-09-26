//! JPEG input through `iijpeg.rs`: sub-areas that exclude the bottom row.
//!
//! Upstream `jpegReadSectionAny` (`iijpeg.c:233-237`) reads non-volatile
//! locals after `longjmp`, so the reference binary fails every sub-area that
//! stops before JPEG row 0 with "Application transferred too few scanlines"
//! (`BUGS.md`, JPEG input).  Fixed in translation: such a read succeeds, and
//! the defined result is exactly the pixels a read of the whole image gives.
//! Each case here therefore compares the JPEG sub-area against the same
//! operation on an MRC copy of the fully decoded image.

mod common;

use std::path::{Path, PathBuf};

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("imod-rs-jpeg-input-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // 50 x 40 deterministic grey image.
    let mut img = image::GrayImage::new(50, 40);
    let mut state: u32 = 12345;
    for pixel in img.pixels_mut() {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
        pixel.0[0] = (state >> 16) as u8;
    }
    img.save(dir.join("x.jpg")).unwrap();
    dir
}

fn run(dir: &Path, args: &[&str]) -> std::process::Output {
    let mut cmd = common::imod_cmd(args[0]);
    let output = cmd.args(&args[1..]).current_dir(dir).output().unwrap();
    assert!(output.status.success(), "{args:?}: {output:?}");
    output
}

/// (nx, ny, byte pixels) of a byte-mode MRC file.
fn read_byte_mrc(path: &Path) -> (usize, usize, Vec<u8>) {
    let b = std::fs::read(path).unwrap();
    let int = |at: usize| i32::from_le_bytes(b[at..at + 4].try_into().unwrap());
    assert_eq!(int(12), 0, "mode");
    let (nx, ny) = (int(0) as usize, int(4) as usize);
    let start = 1024 + int(92) as usize;
    (nx, ny, b[start..start + nx * ny].to_vec())
}

/// The section row of `clip stats` without the interpolated position of the
/// maximum: that is a parabolic fit whose last digit can move with the
/// signed-byte offset of the MRC copy.
fn stats_fields(out: &[u8]) -> Vec<String> {
    let text = String::from_utf8_lossy(out).into_owned();
    let row = text.lines().find(|l| l.starts_with("   0")).unwrap();
    let tokens: Vec<&str> = row.split_whitespace().collect();
    [1, 3, 4, 5, 9, 10]
        .iter()
        .map(|&i| tokens[i].to_owned())
        .collect()
}

#[test]
fn jpeg_sub_areas_above_row_zero_read_the_decoded_pixels() {
    let dir = scratch();
    run(&dir, &["newstack", "x.jpg", "full.mrc"]);
    let (nx, ny, full) = read_byte_mrc(&dir.join("full.mrc"));
    assert_eq!((nx, ny), (50, 40));

    // Output 30 x 20 centred at (25 + 3, 20 - 2): input columns 13..43, rows 8..28.
    run(
        &dir,
        &[
            "newstack", "-size", "30,20", "-offset", "3,-2", "x.jpg", "sub.mrc",
        ],
    );
    let (sx, sy, sub) = read_byte_mrc(&dir.join("sub.mrc"));
    assert_eq!((sx, sy), (30, 20));
    for y in 0..20 {
        assert_eq!(
            &sub[y * 30..y * 30 + 30],
            &full[(y + 8) * 50 + 13..(y + 8) * 50 + 43]
        );
    }

    run(&dir, &["newstack", "-bin", "2", "x.jpg", "bin.mrc"]);
    run(&dir, &["newstack", "-bin", "2", "full.mrc", "binfull.mrc"]);
    assert_eq!(
        read_byte_mrc(&dir.join("bin.mrc")),
        read_byte_mrc(&dir.join("binfull.mrc"))
    );

    let jpg = run(&dir, &["clip", "stats", "-y", "3,30", "x.jpg"]);
    let mrc = run(&dir, &["clip", "stats", "-y", "3,30", "full.mrc"]);
    assert_eq!(stats_fields(&jpg.stdout), stats_fields(&mrc.stdout));

    let _ = std::fs::remove_dir_all(&dir);
    common::remove_command_links();
}
