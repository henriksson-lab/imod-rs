//! Translation of `IMOD/libcfshr/rotateflip.c`.

/// The image buffers of `rotateFlipImage`.  The C signature pairs an untyped
/// `void *array` / `void *brray` with an MRC `mode` tag; Rust spells that pair
/// as one typed value.  The source's `return 1` for an unsupported mode has no
/// representable input here and is enforced by the type instead; the source
/// never aliases `array` and `brray` (it writes `brray` while reading `array`
/// with a different index map), so they are a shared and an exclusive borrow.
pub enum RotateFlipData<'a, 'b> {
    /// `MRC_MODE_BYTE`.
    Byte {
        array: &'a [u8],
        brray: &'b mut [u8],
    },
    /// `MRC_MODE_SHORT`.
    Short {
        array: &'a [i16],
        brray: &'b mut [i16],
    },
    /// `MRC_MODE_USHORT`.
    UShort {
        array: &'a [u16],
        brray: &'b mut [u16],
    },
    /// `MRC_MODE_FLOAT`.
    Float {
        array: &'a [f32],
        brray: &'b mut [f32],
    },
}

/// C `rotateFlipImage` (`rotateflip.c:36`).
///
/// The C implementation copies eight input rows at a time for OpenMP
/// throughput.  This direct source-coordinate implementation preserves its
/// operation maps, element representations, contrast inversion, and output
/// ordering without changing the result.
pub fn rotate_flip_image(
    mut data: RotateFlipData<'_, '_>,
    nx: i32,
    ny: i32,
    mut operation: i32,
    left_handed: i32,
    invert_after: i32,
    invert_con: i32,
    nxout: &mut i32,
    nyout: &mut i32,
    _num_threads: i32,
) -> i32 {
    if invert_con != 0
        && !matches!(
            data,
            RotateFlipData::Short { .. } | RotateFlipData::UShort { .. }
        )
    {
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
            let value = match &data {
                RotateFlipData::Short { array, .. } => array[index as usize] as i32,
                RotateFlipData::UShort { array, .. } => array[index as usize] as i32,
                _ => unreachable!(),
            };
            dmin = dmin.min(value);
            dmax = dmax.max(value);
        }
    }
    /* Do the copy: `rotateflip.c:155-406`, eight input lines per strip with
    eight running output cursors, then the remaining lines one at a time.
    The cursors are signed because `dalong`/`dinter` run backwards for the
    flipped and rotated operations. */
    match &mut data {
        RotateFlipData::Float { array, brray } => {
            let (dalong, dinter) = (dalong as isize, dinter as isize);
            let nx_out = *nxout as isize;
            let nxu = nx as usize;
            let num_strips = ny / 8;
            for strip in 0..num_strips as usize {
                let bstart =
                    xstart as isize + nx_out * ystart as isize + 8 * strip as isize * dinter;
                let astart = 8 * strip * nxu;
                let mut b = [0_isize; 8];
                for line in 0..8 {
                    b[line] = bstart + line as isize * dinter;
                }
                let mut a = [0_usize; 8];
                for line in 0..8 {
                    a[line] = astart + line * nxu;
                }
                for _ix in 0..nxu {
                    for line in 0..8 {
                        let v = array[a[line]];
                        a[line] += 1;
                        brray[b[line] as usize] = v;
                        b[line] += dalong;
                    }
                }
            }
            /* Finish up last rows */
            let mut bstart =
                xstart as isize + nx_out * ystart as isize + 8 * num_strips as isize * dinter;
            let mut astart = 8 * num_strips as usize * nxu;
            for _iy in 8 * num_strips..ny {
                let mut bp = bstart;
                for _ix in 0..nxu {
                    let v = array[astart];
                    astart += 1;
                    brray[bp as usize] = v;
                    bp += dalong;
                }
                bstart += dinter;
            }
        }
        RotateFlipData::Byte { array, brray } => {
            let (dalong, dinter) = (dalong as isize, dinter as isize);
            let nx_out = *nxout as isize;
            let nxu = nx as usize;
            let num_strips = ny / 8;
            for strip in 0..num_strips as usize {
                let bstart =
                    xstart as isize + nx_out * ystart as isize + 8 * strip as isize * dinter;
                let astart = 8 * strip * nxu;
                let mut b = [0_isize; 8];
                for line in 0..8 {
                    b[line] = bstart + line as isize * dinter;
                }
                let mut a = [0_usize; 8];
                for line in 0..8 {
                    a[line] = astart + line * nxu;
                }
                for _ix in 0..nxu {
                    for line in 0..8 {
                        let v = array[a[line]];
                        a[line] += 1;
                        brray[b[line] as usize] = v;
                        b[line] += dalong;
                    }
                }
            }
            /* Finish up last rows */
            let mut bstart =
                xstart as isize + nx_out * ystart as isize + 8 * num_strips as isize * dinter;
            let mut astart = 8 * num_strips as usize * nxu;
            for _iy in 8 * num_strips..ny {
                let mut bp = bstart;
                for _ix in 0..nxu {
                    let v = array[astart];
                    astart += 1;
                    brray[bp as usize] = v;
                    bp += dalong;
                }
                bstart += dinter;
            }
        }
        RotateFlipData::Short { array, brray } => {
            if invert_con != 0 {
                let (dalong, dinter) = (dalong as isize, dinter as isize);
                let nx_out = *nxout as isize;
                let nxu = nx as usize;
                let num_strips = ny / 8;
                for strip in 0..num_strips as usize {
                    let bstart =
                        xstart as isize + nx_out * ystart as isize + 8 * strip as isize * dinter;
                    let astart = 8 * strip * nxu;
                    let mut b = [0_isize; 8];
                    for line in 0..8 {
                        b[line] = bstart + line as isize * dinter;
                    }
                    let mut a = [0_usize; 8];
                    for line in 0..8 {
                        a[line] = astart + line * nxu;
                    }
                    for _ix in 0..nxu {
                        for line in 0..8 {
                            let v = array[a[line]];
                            a[line] += 1;
                            brray[b[line] as usize] = (dmax + dmin - v as i32) as i16;
                            b[line] += dalong;
                        }
                    }
                }
                /* Finish up last rows */
                let mut bstart =
                    xstart as isize + nx_out * ystart as isize + 8 * num_strips as isize * dinter;
                let mut astart = 8 * num_strips as usize * nxu;
                for _iy in 8 * num_strips..ny {
                    let mut bp = bstart;
                    for _ix in 0..nxu {
                        let v = array[astart];
                        astart += 1;
                        brray[bp as usize] = (dmax + dmin - v as i32) as i16;
                        bp += dalong;
                    }
                    bstart += dinter;
                }
            } else {
                let (dalong, dinter) = (dalong as isize, dinter as isize);
                let nx_out = *nxout as isize;
                let nxu = nx as usize;
                let num_strips = ny / 8;
                for strip in 0..num_strips as usize {
                    let bstart =
                        xstart as isize + nx_out * ystart as isize + 8 * strip as isize * dinter;
                    let astart = 8 * strip * nxu;
                    let mut b = [0_isize; 8];
                    for line in 0..8 {
                        b[line] = bstart + line as isize * dinter;
                    }
                    let mut a = [0_usize; 8];
                    for line in 0..8 {
                        a[line] = astart + line * nxu;
                    }
                    for _ix in 0..nxu {
                        for line in 0..8 {
                            let v = array[a[line]];
                            a[line] += 1;
                            brray[b[line] as usize] = v;
                            b[line] += dalong;
                        }
                    }
                }
                /* Finish up last rows */
                let mut bstart =
                    xstart as isize + nx_out * ystart as isize + 8 * num_strips as isize * dinter;
                let mut astart = 8 * num_strips as usize * nxu;
                for _iy in 8 * num_strips..ny {
                    let mut bp = bstart;
                    for _ix in 0..nxu {
                        let v = array[astart];
                        astart += 1;
                        brray[bp as usize] = v;
                        bp += dalong;
                    }
                    bstart += dinter;
                }
            }
        }
        RotateFlipData::UShort { array, brray } => {
            if invert_con != 0 {
                let (dalong, dinter) = (dalong as isize, dinter as isize);
                let nx_out = *nxout as isize;
                let nxu = nx as usize;
                let num_strips = ny / 8;
                for strip in 0..num_strips as usize {
                    let bstart =
                        xstart as isize + nx_out * ystart as isize + 8 * strip as isize * dinter;
                    let astart = 8 * strip * nxu;
                    let mut b = [0_isize; 8];
                    for line in 0..8 {
                        b[line] = bstart + line as isize * dinter;
                    }
                    let mut a = [0_usize; 8];
                    for line in 0..8 {
                        a[line] = astart + line * nxu;
                    }
                    for _ix in 0..nxu {
                        for line in 0..8 {
                            let v = array[a[line]];
                            a[line] += 1;
                            brray[b[line] as usize] = (dmax + dmin - v as i32) as u16;
                            b[line] += dalong;
                        }
                    }
                }
                /* Finish up last rows */
                let mut bstart =
                    xstart as isize + nx_out * ystart as isize + 8 * num_strips as isize * dinter;
                let mut astart = 8 * num_strips as usize * nxu;
                for _iy in 8 * num_strips..ny {
                    let mut bp = bstart;
                    for _ix in 0..nxu {
                        let v = array[astart];
                        astart += 1;
                        brray[bp as usize] = (dmax + dmin - v as i32) as u16;
                        bp += dalong;
                    }
                    bstart += dinter;
                }
            } else {
                let (dalong, dinter) = (dalong as isize, dinter as isize);
                let nx_out = *nxout as isize;
                let nxu = nx as usize;
                let num_strips = ny / 8;
                for strip in 0..num_strips as usize {
                    let bstart =
                        xstart as isize + nx_out * ystart as isize + 8 * strip as isize * dinter;
                    let astart = 8 * strip * nxu;
                    let mut b = [0_isize; 8];
                    for line in 0..8 {
                        b[line] = bstart + line as isize * dinter;
                    }
                    let mut a = [0_usize; 8];
                    for line in 0..8 {
                        a[line] = astart + line * nxu;
                    }
                    for _ix in 0..nxu {
                        for line in 0..8 {
                            let v = array[a[line]];
                            a[line] += 1;
                            brray[b[line] as usize] = v;
                            b[line] += dalong;
                        }
                    }
                }
                /* Finish up last rows */
                let mut bstart =
                    xstart as isize + nx_out * ystart as isize + 8 * num_strips as isize * dinter;
                let mut astart = 8 * num_strips as usize * nxu;
                for _iy in 8 * num_strips..ny {
                    let mut bp = bstart;
                    for _ix in 0..nxu {
                        let v = array[astart];
                        astart += 1;
                        brray[bp as usize] = v;
                        bp += dalong;
                    }
                    bstart += dinter;
                }
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::{RotateFlipData, rotate_flip_image};
    #[test]
    fn rotates_float_with_source_right_handed_operation_map() {
        let input = [1_f32, 2., 3., 4., 5., 6.];
        let mut output = [0_f32; 6];
        let (mut nx, mut ny) = (0, 0);
        assert_eq!(
            rotate_flip_image(
                RotateFlipData::Float {
                    array: &input,
                    brray: &mut output,
                },
                3,
                2,
                1,
                0,
                0,
                0,
                &mut nx,
                &mut ny,
                1,
            ),
            0
        );
        assert_eq!((nx, ny), (2, 3));
        assert_eq!(output, [4., 1., 5., 2., 6., 3.]);
    }

    #[test]
    fn left_handed_rotation_and_flip_coordinate_maps_match_source() {
        let input = [1_u8, 2, 3, 4, 5, 6];
        let mut output = [0_u8; 6];
        let (mut nxout, mut nyout) = (0, 0);
        assert_eq!(
            rotate_flip_image(
                RotateFlipData::Byte {
                    array: &input,
                    brray: &mut output,
                },
                3,
                2,
                1,
                1,
                0,
                0,
                &mut nxout,
                &mut nyout,
                0,
            ),
            0
        );
        assert_eq!((nxout, nyout), (2, 3));
        assert_eq!(output, [3, 6, 2, 5, 1, 4]);
        assert_eq!(
            rotate_flip_image(
                RotateFlipData::Byte {
                    array: &input,
                    brray: &mut output,
                },
                3,
                2,
                4,
                1,
                0,
                0,
                &mut nxout,
                &mut nyout,
                0,
            ),
            0
        );
        assert_eq!((nxout, nyout), (3, 2));
        assert_eq!(output, [3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn contrast_inversion_follows_source_contract() {
        let short_input = [-3_i16, 7, 2, -1];
        let mut short_output = [0_i16; 4];
        let (mut nxout, mut nyout) = (0, 0);
        assert_eq!(
            rotate_flip_image(
                RotateFlipData::Short {
                    array: &short_input,
                    brray: &mut short_output,
                },
                2,
                2,
                0,
                1,
                0,
                1,
                &mut nxout,
                &mut nyout,
                0,
            ),
            0
        );
        assert_eq!(short_output, [7, -3, 2, 5]);
        let ushort_input = [4_u16, 10, 7, 8];
        let mut ushort_output = [0_u16; 4];
        assert_eq!(
            rotate_flip_image(
                RotateFlipData::UShort {
                    array: &ushort_input,
                    brray: &mut ushort_output,
                },
                2,
                2,
                0,
                1,
                0,
                1,
                &mut nxout,
                &mut nyout,
                0,
            ),
            0
        );
        assert_eq!(ushort_output, [10, 4, 7, 6]);
        let floats = [1_f32, 2., 3., 4.];
        let mut float_out = [0_f32; 4];
        assert_eq!(
            rotate_flip_image(
                RotateFlipData::Float {
                    array: &floats,
                    brray: &mut float_out,
                },
                2,
                2,
                0,
                1,
                0,
                1,
                &mut nxout,
                &mut nyout,
                0,
            ),
            2
        );
    }
}
