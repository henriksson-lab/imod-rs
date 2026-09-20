//! Executable support from `IMOD/librgctf/testctffind.cpp`.
//!
//! The source's final fitting call is `ctffind(...)`, implemented in the
//! separate, not-yet-translated `ctffind.cpp` unit.  Everything here is the
//! executable logic that does not depend on that missing algorithm or its
//! callback writer: source defaults, box sizing, section validation, and the
//! precise result reports.

use crate::imod::libcfshr::islice::{Islice, MrcData, slice_init};
use crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT;
use crate::imod::libiimod::mrcslice::slice_write_mrcfile;

/// `writeSlice` (`testctffind.cpp:7`).
///
/// `sliceInit` borrows the C caller's float pointer.  The Rust `Islice` owns
/// its floats, so the caller's array is copied into the temporary slice's `f`
/// member before `sliceWriteMRCfile` writes its one-section MRC file.
pub fn write_slice(filename: &str, data: &[f32], xsize: i32, ysize: i32) -> i32 {
    let Some(pixels) = usize::try_from(xsize)
        .ok()
        .and_then(|x| usize::try_from(ysize).ok().and_then(|y| x.checked_mul(y)))
    else {
        return -1;
    };
    if data.len() != pixels {
        return -1;
    }
    let mut slice = Islice {
        data: MrcData::default(),
        xsize: 0,
        ysize: 0,
        mode: 0,
        csize: 0,
        dsize: 0,
        min: 0.,
        max: 0.,
        mean: 0.,
        index: 0,
        cval: [0.; 4],
    };
    if slice_init(
        &mut slice,
        xsize,
        ysize,
        MRC_MODE_FLOAT,
        MrcData::F(data.to_vec()),
    ) != 0
    {
        return -1;
    }
    slice_write_mrcfile(filename, &mut slice)
}

#[derive(Clone, Debug, PartialEq)]
pub struct TestCtffindParameters {
    pub acceleration_voltage: f32,
    pub spherical_aberration: f32,
    pub box_size: i32,
    pub minimum_resolution: f32,
    pub maximum_resolution: f32,
    pub minimum_defocus: f32,
    pub maximum_defocus: f32,
    pub defocus_search_step: f32,
    pub astigmatism_tolerance: f32,
    pub additional_phase_shift_search_step: f32,
    pub known_astigmatism: f32,
    pub known_astigmatism_angle: f32,
    pub minimum_additional_phase_shift: f32,
    pub maximum_additional_phase_shift: f32,
    pub noisy_input_image: bool,
    pub slower_search: bool,
    pub astigmatism_is_known: bool,
    pub find_additional_phase_shift: bool,
    pub compute_extra_stats: bool,
    pub pixel_size_of_input_image: f32,
}

impl Default for TestCtffindParameters {
    fn default() -> Self {
        Self {
            acceleration_voltage: 200.,
            spherical_aberration: 2.,
            box_size: 256,
            minimum_resolution: 50.,
            maximum_resolution: 10.,
            minimum_defocus: 5_000.,
            maximum_defocus: 80_000.,
            defocus_search_step: 500.,
            astigmatism_tolerance: -100.,
            additional_phase_shift_search_step: 0.1,
            known_astigmatism: 0.,
            known_astigmatism_angle: 0.,
            minimum_additional_phase_shift: 0.,
            maximum_additional_phase_shift: 0.,
            noisy_input_image: false,
            slower_search: false,
            astigmatism_is_known: false,
            find_additional_phase_shift: false,
            compute_extra_stats: false,
            pixel_size_of_input_image: 0.,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_slice_emits_a_one_section_float_mrc_file() {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-testctffind-write-slice-{}-{}.mrc",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        assert_eq!(
            write_slice(path.to_str().unwrap(), &[1., 2., 3., 4.], 2, 2),
            0
        );
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 1024 + 4 * std::mem::size_of::<f32>());
        assert_eq!(&bytes[1024..1028], &1_f32.to_ne_bytes());
        assert_eq!(&bytes[1036..1040], &4_f32.to_ne_bytes());
        let _ = std::fs::remove_file(path);
    }
}
