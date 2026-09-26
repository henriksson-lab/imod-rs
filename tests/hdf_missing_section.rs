//! Reading a Z value that has no dataset in an HDF image stack.
//!
//! `hdf_imageio.c:133-139` means to fill the section with zeros "as an MRC
//! file would do", but advances the output pointer by the padding alone, so
//! it zeroes the first line over and over and leaves every later line holding
//! whatever the buffer held -- in `newstack`, the previous section.  Fixed in
//! translation (`BUGS.md`): the whole missing section reads as zeros.  Not
//! comparable with the reference build, which is configured `NO_HDF_LIB`.

mod common;

use std::ffi::CString;

type HidT = i64;

#[link(name = "hdf5_serial")]
unsafe extern "C" {
    fn H5Fopen(filename: *const std::ffi::c_char, flags: u32, access: HidT) -> HidT;
    fn H5Ldelete(loc: HidT, name: *const std::ffi::c_char, lapl: HidT) -> i32;
    fn H5Fclose(file: HidT) -> i32;
}

#[test]
fn newstack_reads_a_missing_hdf_section_as_zeros() {
    let dir = std::env::temp_dir().join(format!("imod-rs-hdf-missing-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // 8 x 6 x 3 float MRC with pixel values 1..=144 (no zeros anywhere).
    let mut header = vec![0u8; 1024];
    for (at, v) in [
        (0, 8i32),
        (4, 6),
        (8, 3),
        (12, 2),
        (28, 8),
        (32, 6),
        (36, 3),
    ] {
        header[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (at, v) in [
        (40, 8f32),
        (44, 6.),
        (48, 3.),
        (52, 90.),
        (56, 90.),
        (60, 90.),
    ] {
        header[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (at, v) in [(64, 1i32), (68, 2), (72, 3)] {
        header[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (at, v) in [(76, 1f32), (80, 144.), (84, 72.5)] {
        header[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    header[208..212].copy_from_slice(b"MAP ");
    header[212..216].copy_from_slice(&[0x44, 0x44, 0, 0]);
    for v in 1..=144u32 {
        header.extend_from_slice(&(v as f32).to_le_bytes());
    }
    std::fs::write(dir.join("s.mrc"), &header).unwrap();

    let run = |args: &[&str]| {
        let output = common::imod_cmd("newstack")
            .args(args)
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(output.status.success(), "{args:?}: {output:?}");
    };
    run(&["-format", "hdf", "s.mrc", "s.hdf"]);

    // Remove the dataset of Z = 1 from the stack.
    unsafe {
        let name = CString::new(dir.join("s.hdf").to_str().unwrap()).unwrap();
        let file = H5Fopen(name.as_ptr(), 1, 0);
        assert!(file >= 0);
        assert!(H5Ldelete(file, c"/MDF/images/1".as_ptr(), 0) >= 0);
        assert!(H5Fclose(file) >= 0);
    }

    run(&["-mode", "2", "s.hdf", "out.mrc"]);
    let b = std::fs::read(dir.join("out.mrc")).unwrap();
    let int = |at: usize| i32::from_le_bytes(b[at..at + 4].try_into().unwrap());
    let (nx, ny, nz) = (int(0) as usize, int(4) as usize, int(8) as usize);
    assert_eq!((nx, ny), (8, 6));
    assert!(nz >= 2);
    let data: Vec<f32> = b[1024 + int(92) as usize..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // Section 0 is the real data; section 1, which has no dataset, is all zeros
    // (native leaves section 0's lines 1..5 there).
    let expect0: Vec<f32> = (1..=48).map(|v| v as f32).collect();
    assert_eq!(&data[..48], &expect0[..]);
    assert!(
        data[48..96].iter().all(|&v| v == 0.0),
        "{:?}",
        &data[48..96]
    );

    let _ = std::fs::remove_dir_all(&dir);
    common::remove_command_links();
}
