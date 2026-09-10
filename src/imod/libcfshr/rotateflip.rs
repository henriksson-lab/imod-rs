//! Translation of `IMOD/libcfshr/rotateflip.c`.
#![allow(dead_code)]

/// C `rotateFlipImage` (`rotateflip.c:36`).
///
/// The C implementation copies eight input rows at a time for OpenMP
/// throughput.  This direct source-coordinate implementation preserves its
/// operation maps, element representations, contrast inversion, and output
/// ordering without changing the result.
pub unsafe fn rotate_flip_image(
    array: *mut core::ffi::c_void,
    mode: i32,
    nx: i32,
    ny: i32,
    mut operation: i32,
    left_handed: i32,
    invert_after: i32,
    invert_con: i32,
    brray: *mut core::ffi::c_void,
    nxout: *mut i32,
    nyout: *mut i32,
    _num_threads: i32,
) -> i32 {
    unsafe {
        if !matches!(mode, 0 | 1 | 6 | 2) {
            return 1;
        }
        if invert_con != 0 && !matches!(mode, 1 | 6) {
            return 2;
        }
        if !(0..=11).contains(&operation) {
            return 3;
        }
        let inverse_after = [6, 5, 4, 7, 2, 1, 0, 3];
        let right_handed = [0, 3, 2, 1, 4, 7, 6, 5, 4, 5, 6, 7];
        if left_handed == 0 {
            operation = right_handed[operation as usize];
        }
        if operation < 8 && invert_after != 0 {
            operation = inverse_after[operation as usize];
        }
        let mut rotation = operation % 4;
        let flip = if operation / 4 != 0 { -1 } else { 1 };
        *nxout = nx;
        *nyout = ny;
        if rotation % 2 != 0 {
            *nxout = ny;
            *nyout = nx;
            if flip < 0 && operation & 4 != 0 {
                rotation = 4 - rotation;
            }
        }
        let mut xstart = if rotation > 1 { *nxout - 1 } else { 0 };
        let ystart = if rotation == 1 || rotation == 2 {
            *nyout - 1
        } else {
            0
        };
        if flip < 0 {
            xstart = *nxout - 1 - xstart;
        }
        let xalong = [1, 0, -1, 0];
        let yalong = [0, -1, 0, 1];
        let xinter = [0, 1, 0, -1];
        let yinter = [1, 0, -1, 0];
        let dalong = flip * xalong[rotation as usize] + *nxout * yalong[rotation as usize];
        let dinter = flip * xinter[rotation as usize] + *nxout * yinter[rotation as usize];
        let mut dmin = 1_000_000_i32;
        let mut dmax = -1_000_000_i32;
        if invert_con != 0 {
            for index in 0..nx * ny {
                let value = if mode == 1 {
                    *array.cast::<i16>().add(index as usize) as i32
                } else {
                    *array.cast::<u16>().add(index as usize) as i32
                };
                dmin = dmin.min(value);
                dmax = dmax.max(value);
            }
        }
        for iy in 0..ny {
            for ix in 0..nx {
                let target = xstart + *nxout * ystart + iy * dinter + ix * dalong;
                match mode {
                    0 => {
                        *brray.cast::<u8>().add(target as usize) =
                            *array.cast::<u8>().add((iy * nx + ix) as usize)
                    }
                    2 => {
                        *brray.cast::<f32>().add(target as usize) =
                            *array.cast::<f32>().add((iy * nx + ix) as usize)
                    }
                    1 => {
                        let value = *array.cast::<i16>().add((iy * nx + ix) as usize) as i32;
                        *brray.cast::<i16>().add(target as usize) = if invert_con != 0 {
                            (dmax + dmin - value) as i16
                        } else {
                            value as i16
                        };
                    }
                    _ => {
                        let value = *array.cast::<u16>().add((iy * nx + ix) as usize) as i32;
                        *brray.cast::<u16>().add(target as usize) = if invert_con != 0 {
                            (dmax + dmin - value) as u16
                        } else {
                            value as u16
                        };
                    }
                }
            }
        }
        0
    }
}

/// C Fortran wrapper `rotateflipimage` (`rotateflip.c:421`).
pub unsafe fn rotate_flip_image_fortran(
    array: *mut f32,
    nx: *mut i32,
    ny: *mut i32,
    operation: *mut i32,
    brray: *mut f32,
    nxout: *mut i32,
    nyout: *mut i32,
    num_threads: *mut i32,
) -> i32 {
    unsafe {
        rotate_flip_image(
            array.cast(),
            2,
            *nx,
            *ny,
            *operation,
            0,
            0,
            0,
            brray.cast(),
            nxout,
            nyout,
            *num_threads,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{rotate_flip_image, rotate_flip_image_fortran};
    #[test]
    fn rotates_float_with_source_right_handed_operation_map() {
        let mut input = [1_f32, 2., 3., 4., 5., 6.];
        let mut output = [0_f32; 6];
        let (mut nx, mut ny) = (0, 0);
        assert_eq!(
            unsafe {
                rotate_flip_image(
                    input.as_mut_ptr().cast(),
                    2,
                    3,
                    2,
                    1,
                    0,
                    0,
                    0,
                    output.as_mut_ptr().cast(),
                    &mut nx,
                    &mut ny,
                    1,
                )
            },
            0
        );
        assert_eq!((nx, ny), (2, 3));
        assert_eq!(output, [4., 1., 5., 2., 6., 3.]);
    }

    #[test]
    fn left_handed_rotation_and_flip_coordinate_maps_match_source() {
        let mut input = [1_u8, 2, 3, 4, 5, 6];
        let mut output = [0_u8; 6];
        let (mut nxout, mut nyout) = (0, 0);
        assert_eq!(
            unsafe {
                rotate_flip_image(
                    input.as_mut_ptr().cast(),
                    0,
                    3,
                    2,
                    1,
                    1,
                    0,
                    0,
                    output.as_mut_ptr().cast(),
                    &mut nxout,
                    &mut nyout,
                    0,
                )
            },
            0
        );
        assert_eq!((nxout, nyout), (2, 3));
        assert_eq!(output, [3, 6, 2, 5, 1, 4]);
        assert_eq!(
            unsafe {
                rotate_flip_image(
                    input.as_mut_ptr().cast(),
                    0,
                    3,
                    2,
                    4,
                    1,
                    0,
                    0,
                    output.as_mut_ptr().cast(),
                    &mut nxout,
                    &mut nyout,
                    0,
                )
            },
            0
        );
        assert_eq!((nxout, nyout), (3, 2));
        assert_eq!(output, [3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn contrast_inversion_and_fortran_wrapper_follow_source_contract() {
        let mut short_input = [-3_i16, 7, 2, -1];
        let mut short_output = [0_i16; 4];
        let (mut nxout, mut nyout) = (0, 0);
        assert_eq!(
            unsafe {
                rotate_flip_image(
                    short_input.as_mut_ptr().cast(),
                    1,
                    2,
                    2,
                    0,
                    1,
                    0,
                    1,
                    short_output.as_mut_ptr().cast(),
                    &mut nxout,
                    &mut nyout,
                    0,
                )
            },
            0
        );
        assert_eq!(short_output, [7, -3, 2, 5]);
        let mut ushort_input = [4_u16, 10, 7, 8];
        let mut ushort_output = [0_u16; 4];
        assert_eq!(
            unsafe {
                rotate_flip_image(
                    ushort_input.as_mut_ptr().cast(),
                    6,
                    2,
                    2,
                    0,
                    1,
                    0,
                    1,
                    ushort_output.as_mut_ptr().cast(),
                    &mut nxout,
                    &mut nyout,
                    0,
                )
            },
            0
        );
        assert_eq!(ushort_output, [10, 4, 7, 6]);
        let mut floats = [1_f32, 2., 3., 4.];
        let mut float_out = [0_f32; 4];
        let mut nx = 2;
        let mut ny = 2;
        let mut operation = 0;
        let mut threads = 0;
        assert_eq!(
            unsafe {
                rotate_flip_image_fortran(
                    floats.as_mut_ptr(),
                    &mut nx,
                    &mut ny,
                    &mut operation,
                    float_out.as_mut_ptr(),
                    &mut nxout,
                    &mut nyout,
                    &mut threads,
                )
            },
            0
        );
        assert_eq!(float_out, floats);
        assert_eq!(
            unsafe {
                rotate_flip_image(
                    floats.as_mut_ptr().cast(),
                    2,
                    2,
                    2,
                    0,
                    1,
                    0,
                    1,
                    float_out.as_mut_ptr().cast(),
                    &mut nxout,
                    &mut nyout,
                    0,
                )
            },
            2
        );
        assert_eq!(
            unsafe {
                rotate_flip_image(
                    floats.as_mut_ptr().cast(),
                    99,
                    2,
                    2,
                    0,
                    1,
                    0,
                    0,
                    float_out.as_mut_ptr().cast(),
                    &mut nxout,
                    &mut nyout,
                    0,
                )
            },
            1
        );
    }
}
