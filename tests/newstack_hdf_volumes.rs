//! `newstack`'s HDF volume options: `-3d` (`Store3DVolumes`), `-chunk`
//! (`ChunkSizesInXYZ`), `-compression` (`HDFCompressionIndex`) and `-volumes`
//! (`VolumesToRead`), translated from `IMOD/flib/image/newstack.f90:386-399`,
//! `:1766-1812` and `:3157-3174`.  Every expectation here was taken from an
//! HDF-enabled native `newstack` first.

mod common;

use imod_rs::imod::libiimod::iihdf::ii_hdf_open_new;
use imod_rs::imod::libiimod::iimage::{
    IIFILE_DEFAULT, IIFILE_HDF, ii_allow_multi_volume, ii_close, ii_delete, ii_open, ii_open_new,
    ii_read_section_float, ii_sync_from_mrc_header, ii_write_section_float,
};
use imod_rs::imod::libiimod::mrcfiles::{MRC_MODE_FLOAT, MrcHeader, mrc_head_new, mrc_head_write};
use std::ffi::CString;

/// Every PIP-driven invocation needs an autodoc directory, exactly as a real
/// IMOD install provides one through `AUTODOC_DIR` or `IMOD_DIR`.
const AUTODOC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc");

/// Writes an 8x6x4 float MRC whose section `z` holds `z * 100 + index`, so a
/// section read back out of an output file identifies itself.
fn write_input_mrc(tag: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "imod-rs-newstack-hdf-{tag}-{}.mrc",
        std::process::id()
    ));
    let path_c = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(path_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 8, 6, 4, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for z in 0..4 {
            let mut section: Vec<f32> = (0..48).map(|i| (z * 100 + i) as f32).collect();
            assert_eq!(
                ii_write_section_float(file, section.as_mut_ptr().cast(), z),
                0
            );
        }
        ii_close(file);
    }
    path
}

/// Reads one section of an image file back as floats.
fn read_section(path: &std::path::Path, section: i32, count: usize) -> Vec<f32> {
    let path_c = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    unsafe {
        let file = ii_open(path_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null(), "opening {}", path.display());
        let mut pixels = vec![0.0_f32; count];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), section),
            0
        );
        ii_close(file);
        pixels
    }
}

/// `-3d 1` calls `overrideOutputType(5)` (`newstack.f90:396`) and then
/// `iiuAltChunkSizes` (`newstack.f90:1806`), so the output is an HDF file
/// holding one 3-D dataset rather than a stack of 2-D ones.
#[test]
fn newstack_3d_one_writes_a_single_hdf_volume() {
    let input = write_input_mrc("3d");
    let output = std::env::temp_dir().join(format!(
        "imod-rs-newstack-hdf-3d-{}.out",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&output);
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-3d", "1", "-input"])
        .arg(&input)
        .arg("-output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout={}",
        String::from_utf8_lossy(&result.stdout)
    );
    // The extension is `.out`, so only `overrideOutputType(5)` can have made
    // this an HDF file.
    let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        assert_eq!((*file).file, IIFILE_HDF);
        assert_eq!((*file).nz, 4);
        // A stack of 2-D datasets reports no Z chunking at all.
        assert!((*file).z_chunk_size > 0, "output is not a 3-D volume");
        ii_close(file);
    }
    assert_eq!(read_section(&output, 2, 48)[..3], [200., 201., 202.]);
    std::fs::remove_file(input).unwrap();
    std::fs::remove_file(output).unwrap();
}

/// `-3d -1` forbids volume output (`newstack.adoc`), so the output type is not
/// overridden and the file stays MRC.
#[test]
fn newstack_3d_minus_one_leaves_the_default_output_type() {
    let input = write_input_mrc("3dminus");
    let output = std::env::temp_dir().join(format!(
        "imod-rs-newstack-hdf-3dminus-{}.out",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&output);
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-3d", "-1", "-input"])
        .arg(&input)
        .arg("-output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(result.status.success());
    let mut signature = [0_u8; 4];
    use std::io::Read;
    std::fs::File::open(&output)
        .unwrap()
        .read_exact(&mut signature)
        .unwrap();
    assert_ne!(signature, [0x89, b'H', b'D', b'F']);
    std::fs::remove_file(input).unwrap();
    std::fs::remove_file(output).unwrap();
}

/// `-chunk` sets `ifChunkIn` and therefore `if3dVolumes = 1`
/// (`newstack.f90:391-392`), and `iiBestTileSize` (`newstack.f90:1803-1805`)
/// rounds each target to a size that tiles the image; the report at
/// `newstack.f90:1809` names the result.  Native prints exactly this line for
/// an 8x6x4 input with `-chunk 4,4,2`.
#[test]
fn newstack_chunk_reports_the_actual_tile_size() {
    let input = write_input_mrc("chunk");
    let output = std::env::temp_dir().join(format!(
        "imod-rs-newstack-hdf-chunk-{}.out",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&output);
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-chunk", "4,4,2", "-input"])
        .arg(&input)
        .arg("-output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout).into_owned();
    assert!(
        stdout.contains("Actual chunk size:       4 by      3 by   2"),
        "stdout={stdout}"
    );
    let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        assert_eq!((*file).file, IIFILE_HDF);
        assert_eq!(
            (
                (*file).tile_size_x,
                (*file).tile_size_y,
                (*file).z_chunk_size
            ),
            (4, 3, 2)
        );
        ii_close(file);
    }
    assert_eq!(read_section(&output, 3, 48)[..3], [300., 301., 302.]);
    std::fs::remove_file(input).unwrap();
    std::fs::remove_file(output).unwrap();
}

/// A target that already divides the image is used as entered, and then the
/// report at `newstack.f90:1808-1810` is not printed at all.
#[test]
fn newstack_chunk_matching_the_image_prints_no_report() {
    let input = write_input_mrc("chunkexact");
    let output = std::env::temp_dir().join(format!(
        "imod-rs-newstack-hdf-chunkexact-{}.out",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&output);
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-chunk", "8,6,4", "-input"])
        .arg(&input)
        .arg("-output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(result.status.success());
    assert!(!String::from_utf8_lossy(&result.stdout).contains("Actual chunk size"));
    std::fs::remove_file(input).unwrap();
    std::fs::remove_file(output).unwrap();
}

/// `newstack.f90:394-395`.
#[test]
fn newstack_chunk_with_3d_minus_one_is_rejected() {
    let input = write_input_mrc("chunkbad");
    let output = std::env::temp_dir().join(format!(
        "imod-rs-newstack-hdf-chunkbad-{}.out",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&output);
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-chunk", "4,4,2", "-3d", "-1", "-input"])
        .arg(&input)
        .arg("-output")
        .arg(&output)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    // `exitError` writes through `PipSetError`, which is stdout here
    // (`parse_params.c:2049`).
    assert!(
        String::from_utf8_lossy(&result.stdout).contains(
            "ERROR: NEWSTACK - You cannot enter chunk sizes and forbid volume output with -3d -1"
        ),
        "stdout={}",
        String::from_utf8_lossy(&result.stdout)
    );
    std::fs::remove_file(input).unwrap();
}

/// `-compression` reaches `iiuSetHDFCompression` only for an HDF output
/// (`newstack.f90:1794-1796`).  Level 0 stores the dataset uncompressed and
/// level 9 deflates it, so the same pixels come back from a smaller file.
#[test]
fn newstack_compression_index_changes_only_the_stored_size() {
    let input = write_input_mrc("comp");
    let mut sizes = Vec::new();
    for level in ["0", "9"] {
        let output = std::env::temp_dir().join(format!(
            "imod-rs-newstack-hdf-comp{level}-{}.out",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&output);
        let result = common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args(["-3d", "1", "-compression", level, "-input"])
            .arg(&input)
            .arg("-output")
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "stdout={}",
            String::from_utf8_lossy(&result.stdout)
        );
        assert_eq!(read_section(&output, 1, 48)[..3], [100., 101., 102.]);
        sizes.push(std::fs::metadata(&output).unwrap().len());
        std::fs::remove_file(&output).unwrap();
    }
    assert!(sizes[1] < sizes[0], "sizes={sizes:?}");
    std::fs::remove_file(input).unwrap();
}

/// Builds a two-volume HDF file whose volumes hold different pixels, the way
/// `iiFOpenNewVolume` does for a real multi-volume file.
fn write_two_volume_hdf(tag: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "imod-rs-newstack-hdf-{tag}-{}.hdf",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let path_c = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    unsafe {
        let image = ii_open_new(path_c.as_ptr(), c"wb".as_ptr(), IIFILE_HDF);
        assert!(!image.is_null());
        let header = (*image).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 8, 6, 2, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(image, header);
        (*image).z_chunk_size = 1;
        assert_eq!((*image).write_header.unwrap()(image), 0);
        for z in 0..2 {
            let mut section = vec![(10 + z) as f32; 48];
            assert_eq!(
                ii_write_section_float(image, section.as_mut_ptr().cast(), z),
                0
            );
        }
        assert_eq!(ii_hdf_open_new(image, c"wb".as_ptr()), 0);
        let second = *(*image).ii_volumes.add(1);
        let second_header = (*second).header.cast::<MrcHeader>();
        assert_eq!(
            mrc_head_new(&mut *second_header, 8, 6, 2, MRC_MODE_FLOAT),
            0
        );
        ii_sync_from_mrc_header(second, second_header);
        (*second).z_chunk_size = 1;
        assert_eq!((*second).write_header.unwrap()(second), 0);
        for z in 0..2 {
            let mut section = vec![(20 + z) as f32; 48];
            assert_eq!(
                ii_write_section_float(second, section.as_mut_ptr().cast(), z),
                0
            );
        }
        ii_delete(second);
        ii_delete(image);
    }
    path
}

/// `openInputFile` (`newstack.f90:3157-3174`) opens the file on unit 11 and
/// the requested volume on unit 1 whenever `listVolumes(indInFile) > 1`.
#[test]
fn newstack_volumes_reads_the_requested_volume() {
    let input = write_two_volume_hdf("multi");
    for (volume, expected) in [("1", 10.0_f32), ("2", 20.0)] {
        let output = std::env::temp_dir().join(format!(
            "imod-rs-newstack-hdf-vol{volume}-{}.mrc",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&output);
        let result = common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args(["-volumes", volume, "-input"])
            .arg(&input)
            .arg("-output")
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "stdout={}",
            String::from_utf8_lossy(&result.stdout)
        );
        assert_eq!(read_section(&output, 0, 48)[0], expected);
        std::fs::remove_file(&output).unwrap();
    }
    std::fs::remove_file(input).unwrap();
}

/// Without `-volumes` the source's `iiAllowMultiVolume(0)` leaves `iiOpen`
/// refusing the file (`iimage.c:299-302`), and a volume number past the end
/// fails inside `iiFOpenVolume` (`iimage.c:906`).  Both messages are the
/// library's, not `newstack`'s.
#[test]
fn newstack_volumes_out_of_range_reports_the_library_error() {
    let input = write_two_volume_hdf("multibad");
    let output = std::env::temp_dir().join(format!(
        "imod-rs-newstack-hdf-volbad-{}.mrc",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&output);
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-volumes", "3", "-input"])
        .arg(&input)
        .arg("-output")
        .arg(&output)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&result.stdout)
            .contains("ERROR: iiFOpenVolume - Requested volume index 2 out of range"),
        "stdout={}",
        String::from_utf8_lossy(&result.stdout)
    );
    let no_option = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .arg("-input")
        .arg(&input)
        .arg("-output")
        .arg(&output)
        .output()
        .unwrap();
    assert_eq!(no_option.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&no_option.stdout).contains(
            "is an HDF file with multiple volumes and cannot be opened by this program or with current options to the program"
        ),
        "stdout={}",
        String::from_utf8_lossy(&no_option.stdout)
    );
    std::fs::remove_file(input).unwrap();
}

/// `-3d 2` opens the existing output on unit 12 and creates a new volume in it
/// (`newstack.f90:1766-1772`), so the file gains a volume instead of being
/// replaced.
#[test]
fn newstack_3d_two_adds_a_volume_to_an_existing_file() {
    let input = write_input_mrc("append");
    let output = write_two_volume_hdf("append");
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-3d", "2", "-input"])
        .arg(&input)
        .arg("-output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout={}",
        String::from_utf8_lossy(&result.stdout)
    );
    let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
    unsafe {
        // `iiOpen` refuses a multi-volume HDF file unless the caller has said
        // it can handle one (`iimage.c:299-302`).
        ii_allow_multi_volume(1);
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        ii_allow_multi_volume(0);
        assert!(!file.is_null());
        assert_eq!((*file).num_volumes, 3);
        ii_close(file);
    }
    std::fs::remove_file(input).unwrap();
    std::fs::remove_file(output).unwrap();
}
