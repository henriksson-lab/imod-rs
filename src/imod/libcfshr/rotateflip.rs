//! Translation of `IMOD/libcfshr/rotateflip.c`.

use crate::imod::libcfshr::b3dutil::num_omp_threads;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

/// Guards the one attempt this unit makes to size rayon's global pool.
static RAYON_POOL: std::sync::Once = std::sync::Once::new();

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
/// The C implementation copies eight input rows at a time, OpenMP-parallel
/// over those strips; this translation copies eight *output* rows at a time,
/// parallel over groups of output rows (see the comment at the copy).  It
/// preserves the operation maps, element representations and contrast
/// inversion; the copy is pure data movement, so the result is unchanged.
pub fn rotate_flip_image(
    data: RotateFlipData<'_, '_>,
    nx: i32,
    ny: i32,
    mut operation: i32,
    left_handed: i32,
    invert_after: i32,
    invert_con: i32,
    nxout: &mut i32,
    nyout: &mut i32,
    num_threads: i32,
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
        // `rotateflip.c:118-130` tests the mode once and runs one flat
        // `nx * ny` loop per type.  Integer min/max, so `Ord::min`/`max` and
        // `B3DMIN`/`B3DMAX` agree on every input.
        let count = (nx * ny).max(0) as usize;
        match &data {
            RotateFlipData::Short { array, .. } => {
                for &value in &array[..count] {
                    dmax = (value as i32).max(dmax);
                    dmin = (value as i32).min(dmin);
                }
            }
            RotateFlipData::UShort { array, .. } => {
                for &value in &array[..count] {
                    dmax = (value as i32).max(dmax);
                    dmin = (value as i32).min(dmin);
                }
            }
            _ => unreachable!(),
        }
    }
    /* Do the copy: `rotateflip.c:132-406`.  The thread count comes first,
    exactly as the source chooses it (`rotateflip.c:140-148`). */
    let mut num_threads = num_threads;
    if num_threads <= 0 {
        num_threads = if nx.wrapping_mul(ny) < 1500 * 1500 {
            1
        } else if nx.wrapping_mul(ny) < 3000 * 3000 || rotation % 2 == 0 {
            2
        } else {
            3
        };
    }
    let num_threads = num_omp_threads(num_threads);

    /* The source's `#pragma omp parallel for` loops (`rotateflip.c:151,209,
    296,358`) run over *input* strips of eight lines and scatter each strip
    into the output with the cursors `xstart + nxOut * ystart + line * dinter
    + ix * dalong`, so one strip's writes are strided across the whole output
    and cannot be handed to a worker as a contiguous `&mut` slice.  This is
    pure data movement — every output element receives exactly one input
    element, unchanged, or `dmax + dmin - value` with `dmin`/`dmax` fixed
    before the copy — so the value an element receives cannot depend on the
    order in which elements are copied.  The copy is therefore partitioned by
    *output* rows instead: each worker owns a contiguous group of output rows
    and gathers into them through the inverse of the source's map.

    The forward map sends input `(ix, iy)` to output
    `xo = xstart + flip * (xalong * ix + xinter * iy)`,
    `yo = ystart + yalong * ix + yinter * iy`, whose matrix is a signed
    permutation, so its inverse is its transpose (and `1 / flip == flip`):
    `ix = yalong * (yo - ystart) + flip * xalong * (xo - xstart)`,
    `iy = yinter * (yo - ystart) + flip * xinter * (xo - xstart)`.
    Hence the source index `ix + nx * iy` steps by `srcPerRow` per output row
    and `srcPerCol` per output column from `srcOrigin` at output (0, 0).
    The map is a bijection of the `nx * ny` input onto the `nxOut * nyOut`
    output, so the gather writes every element the scatter writes, with the
    same value, and nothing else.  Output rows are taken eight at a time, so
    each step still reads eight neighbouring input elements and writes eight
    sequential streams, the mirror image of the source's strips.  The same
    closure runs on one thread and on many, and the groups are disjoint, so
    neither the schedule nor the thread count can change a byte. */
    let rot = rotation as usize;
    let nx_out = *nxout;
    let ny_out = *nyout;
    let src_per_row = (yalong[rot] + nx * yinter[rot]) as isize;
    let src_per_col = (flip * (xalong[rot] + nx * xinter[rot])) as isize;
    let src_origin = -(src_per_row * ystart as isize) - src_per_col * xstart as isize;
    let nxo = nx_out.max(0) as usize;
    let count = nxo * ny_out.max(0) as usize;
    if count == 0 || nx <= 0 || ny <= 0 {
        return 0;
    }
    let rows_per_group = if num_threads > 1 {
        (ny_out as usize).div_ceil(num_threads as usize).max(1)
    } else {
        ny_out as usize
    };
    if num_threads > 1 {
        // Native's OpenMP runtime creates at most `omp_get_num_procs()`
        // workers and `numOMPthreads` never asks for more than the
        // physical-core count (or whatever `OMP_NUM_THREADS` /
        // `IMOD_FORCE_OMP_THREADS` allow); rayon's default global pool is
        // sized from logical processors, so it is bounded to the same count.
        // `build_global` succeeds for whichever translated unit reaches it
        // first and is a harmless `Err` afterwards: every unit asks for the
        // same size.
        // The `Once` makes that attempt happen once per process rather
        // than on every call.
        RAYON_POOL.call_once(|| {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(num_omp_threads(i32::MAX) as usize)
                .build_global();
        });
    }
    macro_rules! gather_output_rows {
        ($t:ty, $array:expr, $brray:expr, |$v:ident| $conv:expr) => {{
            let array = $array;
            let run = |(g, brows): (usize, &mut [$t])| {
                let row0 = g * rows_per_group;
                let nrows = brows.len() / nxo;
                let nstrips = nrows / 8;
                for strip in 0..nstrips {
                    let strip_rows = &mut brows[strip * 8 * nxo..(strip + 1) * 8 * nxo];
                    let mut src = [0_isize; 8];
                    for line in 0..8 {
                        src[line] = src_origin + (row0 + 8 * strip + line) as isize * src_per_row;
                    }
                    for xo in 0..nxo {
                        for line in 0..8 {
                            let $v = array[src[line] as usize];
                            strip_rows[line * nxo + xo] = $conv;
                            src[line] += src_per_col;
                        }
                    }
                }
                /* Finish up last rows */
                for iy in 8 * nstrips..nrows {
                    let mut sp = src_origin + (row0 + iy) as isize * src_per_row;
                    for b in brows[iy * nxo..(iy + 1) * nxo].iter_mut() {
                        let $v = array[sp as usize];
                        *b = $conv;
                        sp += src_per_col;
                    }
                }
            };
            if num_threads > 1 {
                $brray[..count]
                    .par_chunks_mut(rows_per_group * nxo)
                    .enumerate()
                    .for_each(run);
            } else {
                $brray[..count]
                    .chunks_mut(rows_per_group * nxo)
                    .enumerate()
                    .for_each(run);
            }
        }};
    }
    match data {
        RotateFlipData::Float { array, brray } => gather_output_rows!(f32, array, brray, |v| v),
        RotateFlipData::Byte { array, brray } => gather_output_rows!(u8, array, brray, |v| v),
        RotateFlipData::Short { array, brray } => {
            if invert_con != 0 {
                gather_output_rows!(i16, array, brray, |v| (dmax + dmin - v as i32) as i16)
            } else {
                gather_output_rows!(i16, array, brray, |v| v)
            }
        }
        RotateFlipData::UShort { array, brray } => {
            if invert_con != 0 {
                gather_output_rows!(u16, array, brray, |v| (dmax + dmin - v as i32) as u16)
            } else {
                gather_output_rows!(u16, array, brray, |v| v)
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
