//! Translation of `IMOD/flib/image/tomopieces.f90`.
//!
//! TOMOPIECES figures out how to chop up a tomogram into pieces so that the
//! Fourier transforms of each piece can be done in memory.
//!
//! The main program maps to [`tomopieces`]; the subroutines `getRanges`,
//! `rangeOut` and `rangeAdd` and the function `padNiceIfFFT` map to
//! [`get_ranges`], [`range_out`], [`range_add`] and [`pad_nice_if_fft`].
//! Their 1-based Fortran arrays are 0-based slices.

use crate::imod::flib::subrs::hvem::get_nxyz::get_nxyz;
use crate::imod::flib::subrs::hvem::int_iwrite::int_iwrite;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::libcfshr::filtxcorr::nice_frame;
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_enable_entry_output, pip_get_float, pip_get_integer,
};
use crate::imod::libfft::odfft::nice_fft_limit;

/// `parameter (LIMPIECES = 5000)` (`tomopieces.f90:12-13`).
const LIMPIECES: usize = 5000;
/// `parameter (numOptions = 13)` (`tomopieces.f90:39-40`).
const NUM_OPTIONS: i32 = 13;
/// Fallback PIP table, the `options(1)` string (`tomopieces.f90:42-47`).
const OPTIONS: &str = "tomogram:TomogramOrSizeXYZ:FN:@megavox:MegaVoxels:F:@xpad:XPadding:I:@\
ypad:YPadding:I:@zpad:ZPadding:I:@xmaxpiece:XMaximumPieces:I:@\
ymaxpiece:YMaximumPieces:I:@zmaxpiece:ZMaximumPieces:I:@\
minoverlap:MinimumOverlap:I:@hdf:ChunksForHDF:B:@nofft:NoFFTSizes:B:@\
param:ParameterFile:PF:@help:usage:B:";

/// Original program `tomopieces` (`tomopieces.f90:10`).
pub fn tomopieces() {
    let mut nxyz = [0_i32; 3];
    let mut mega_vox: f32 = 80.; //MAXIMUM MEGAVOXELS
    let mut too_big: bool;
    let mut no_fft: bool;
    let mut chunked_hdf: bool;
    let (mut nx_chunk, mut ny_chunk, mut nz_chunk) = (0_i32, 0_i32, 0_i32);
    let (mut num_ypiece_min, mut num_zpiece_min) = (0_i32, 0_i32);
    //
    let mut min_overlap = 4; //MINIMUM OVERLAP TO OUTPUT
    let mut nx_pad = 8; //PADDING / TAPER EXTENT IN X
    let mut nz_pad = 8; //PADDING / TAPER EXTENT IN Z
    let mut ny_pad = 4; //PADDING IN Y
    let mut max_pieces_x = 0; //MAX PIECES IN X
    let mut max_pieces_y = 0; //MAX PIECES IN X
    let mut max_pieces_z = -1; //MAX PIECES IN X
    let max_layer_pieces = 100;
    no_fft = false;
    chunked_hdf = false;
    //
    // Pip startup: set error, parse options, do help output, get the
    // one obligatory argument to get size
    // But turn off the entry printing first!
    pip_enable_entry_output(0);
    let (mut num_opt_arg, mut num_non_opt_arg) = (0, 0);
    pip_read_or_parse_options(
        &[OPTIONS],
        NUM_OPTIONS,
        "tomopieces",
        "ERROR: TOMOPIECES - ",
        false,
        1,
        1,
        0,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    get_nxyz(true, "TomogramOrSizeXYZ", "TOMOPIECES", 1, &mut nxyz);

    let _ = pip_get_float(b"MegaVoxels", &mut mega_vox);
    let _ = pip_get_integer(b"MinimumOverlap", &mut min_overlap);
    let _ = pip_get_integer(b"XPadding", &mut nx_pad);
    let _ = pip_get_integer(b"YPadding", &mut ny_pad);
    let _ = pip_get_integer(b"ZPadding", &mut nz_pad);
    let _ = pip_get_integer(b"XMaximumPieces", &mut max_pieces_x);
    let _ = pip_get_integer(b"YMaximumPieces", &mut max_pieces_y);
    let _ = pip_get_integer(b"ZMaximumPieces", &mut max_pieces_z);
    let _ = pip_get_logical("NoFFTSizes", &mut no_fft);
    let _ = pip_get_logical("ChunksForHDF", &mut chunked_hdf);
    pip_done();
    //
    let nx = nxyz[0];
    let ny = nxyz[1];
    let nz = nxyz[2];
    //
    // Set defaults very big if no maxima entered - but if X and Y maxes
    // are still 0, set up for 19 pieces on a layer
    // Also, if one is zero and one is 1, set up for the 19-piece limit
    //
    if max_pieces_x == 0 && (max_pieces_y == 0 || max_pieces_y == 1) {
        max_pieces_x = max_layer_pieces;
    } else if max_pieces_x == 1 && max_pieces_y == 0 {
        max_pieces_y = max_layer_pieces;
    } else {
        if max_pieces_x <= 0 {
            max_pieces_x = nx / 2 - 1;
        }
        if max_pieces_y <= 0 {
            max_pieces_y = ny / 2 - 1;
        }
    }
    if max_pieces_z < 0 {
        max_pieces_z = nz / 2 - 1;
    }
    //
    // loop on the possible pieces in X; for each one, get the padded size
    //
    let mut perim_min: f32 = (10. * nx as f32) * ny as f32 * nz as f32;
    let mut piece_perim_min: f32 = perim_min;
    let mut num_xpiece_min = 0;
    // `nzOut` is a program variable: when the Z loop does not run it keeps
    // its last value (the piece is then too big and not used).
    let mut nz_out = 0;
    for num_xpieces in 1..=max_pieces_x {
        let nx_extra = (nx + (num_xpieces - 1) * min_overlap + num_xpieces - 1) / num_xpieces;
        let nx_out = pad_nice_if_fft(nx_extra, nx_pad, no_fft);
        //
        // loop on the possible pieces in Y
        //
        let mut limit_ypieces = max_pieces_y;
        if limit_ypieces == 0 {
            limit_ypieces = max_layer_pieces / num_xpieces;
        }

        for num_ypieces in 1..=limit_ypieces {
            let ny_extra = (ny + (num_ypieces - 1) * min_overlap + num_ypieces - 1) / num_ypieces;
            let ny_out = pad_nice_if_fft(ny_extra, ny_pad, no_fft);
            let mut num_zpieces = 1;
            too_big = true;
            //
            // then loop on pieces in Z, get padded size and compute padded
            // volume size - until it is no longer too big for maxmem
            //
            while num_zpieces <= max_pieces_z && too_big {
                let nz_extra =
                    (nz + (num_zpieces - 1) * min_overlap + num_zpieces - 1) / num_zpieces;
                nz_out = pad_nice_if_fft(nz_extra, nz_pad, no_fft);
                if nx_out.wrapping_mul(ny_out) as f32 * nz_out as f32 > mega_vox * 1.0e6_f32 {
                    num_zpieces += 1;
                } else {
                    too_big = false;
                }
            }
            //
            // compute perimeter of pieces and keep track of minimum
            // If total perimeter is equal, favor one with minimum individual
            // perimeter (at the expense of more pieces)
            // Of the equivalent sets when nx = nz, this favors the ones with
            // fewer X pieces
            //
            let perim: f32 = nx.wrapping_mul(ny) as f32 * num_zpieces as f32
                + nx.wrapping_mul(nz) as f32 * num_ypieces as f32
                + ny.wrapping_mul(nz) as f32 * num_xpieces as f32;
            let piece_perim: f32 = nx_out
                .wrapping_mul(ny_out)
                .wrapping_add(nx_out.wrapping_mul(nz_out))
                .wrapping_add(ny_out.wrapping_mul(nz_out))
                as f32;
            // print *,nxp, nyp, nzp, perim, piecePerim, nxout, nyout, nzout
            if !too_big
                && (perim < perim_min || (perim == perim_min && piece_perim < piece_perim_min))
            {
                perim_min = perim;
                piece_perim_min = piece_perim;
                num_xpiece_min = num_xpieces;
                num_ypiece_min = num_ypieces;
                num_zpiece_min = num_zpieces;
            }
        }
    }

    if num_xpiece_min == 0 {
        exit_error("Pieces are all too large with given maximum numbers");
    }
    let num_xpieces = num_xpiece_min;
    let num_zpieces = num_zpiece_min;
    let num_ypieces = num_ypiece_min;
    //
    // The arrays are `LIMPIECES` long in the source; a count beyond that
    // would overrun them there, and is given room here instead.
    let limx = LIMPIECES.max(num_xpieces as usize);
    let limy = LIMPIECES.max(num_ypieces as usize);
    let limz = LIMPIECES.max(num_zpieces as usize);
    let (mut ix_out_start, mut ix_out_end) = (vec![0_i32; limx], vec![0_i32; limx]);
    let (mut ix_back_start, mut ix_back_end) = (vec![0_i32; limx], vec![0_i32; limx]);
    let (mut iy_out_start, mut iy_out_end) = (vec![0_i32; limy], vec![0_i32; limy]);
    let (mut iy_back_start, mut iy_back_end) = (vec![0_i32; limy], vec![0_i32; limy]);
    let (mut iz_out_start, mut iz_out_end) = (vec![0_i32; limz], vec![0_i32; limz]);
    let (mut iz_back_start, mut iz_back_end) = (vec![0_i32; limz], vec![0_i32; limz]);
    //
    // get the starting and ending limits to extract and coordinates
    // for getting back from the padded volume
    get_ranges(
        nx,
        num_xpieces,
        min_overlap,
        nx_pad,
        &mut ix_out_start,
        &mut ix_out_end,
        &mut ix_back_start,
        &mut ix_back_end,
        no_fft,
        chunked_hdf,
        &mut nx_chunk,
    );
    get_ranges(
        ny,
        num_ypieces,
        min_overlap,
        ny_pad,
        &mut iy_out_start,
        &mut iy_out_end,
        &mut iy_back_start,
        &mut iy_back_end,
        no_fft,
        chunked_hdf,
        &mut ny_chunk,
    );
    get_ranges(
        nz,
        num_zpieces,
        min_overlap,
        nz_pad,
        &mut iz_out_start,
        &mut iz_out_end,
        &mut iz_back_start,
        &mut iz_back_end,
        no_fft,
        chunked_hdf,
        &mut nz_chunk,
    );
    let axis =
        |n: i32, start: &[i32], end: &[i32], back_start: &[i32], back_end: &[i32]| TomopiecesAxis {
            out_start: start[..n as usize].to_vec(),
            out_end: end[..n as usize].to_vec(),
            back_start: back_start[..n as usize].to_vec(),
            back_end: back_end[..n as usize].to_vec(),
        };
    let result = TomopiecesResult {
        axes: [
            axis(
                num_xpieces,
                &ix_out_start,
                &ix_out_end,
                &ix_back_start,
                &ix_back_end,
            ),
            axis(
                num_ypieces,
                &iy_out_start,
                &iy_out_end,
                &iy_back_start,
                &iy_back_end,
            ),
            axis(
                num_zpieces,
                &iz_out_start,
                &iz_out_end,
                &iz_back_start,
                &iz_back_end,
            ),
        ],
        chunk_sizes: chunked_hdf.then_some([nx_chunk, ny_chunk, nz_chunk]),
    };
    // write(*,102) outputFile(1:numOut)   102 format(a)
    for line in tomopieces_output_lines(&result) {
        println!("{line}");
    }
    RESULT_SINK.with_borrow(|slot| {
        if let Some(sink) = slot {
            *sink.lock().expect("tomopieces result sink") = Some(result);
        }
    });
    //
    exit(0);
}

/// Rust-only: one axis of the division `tomopieces` finds: for each piece
/// the range to extract (`ix/iy/izOutStart`, `...OutEnd`) and the range to
/// take back out of the padded piece (`...BackStart`, `...BackEnd`).
#[derive(Clone, Debug, Default)]
pub struct TomopiecesAxis {
    pub out_start: Vec<i32>,
    pub out_end: Vec<i32>,
    pub back_start: Vec<i32>,
    pub back_end: Vec<i32>,
}

/// Rust-only: what `tomopieces` computes, as a direct-call result (`CLAUDE.md`,
/// "Wherever we control both sides, use a direct function call now"): the
/// ranges on X, Y and Z, and the HDF chunk sizes with `-hdf`.
#[derive(Clone, Debug, Default)]
pub struct TomopiecesResult {
    pub axes: [TomopiecesAxis; 3],
    pub chunk_sizes: Option<[i32; 3]>,
}

thread_local! {
    /// Where [`tomopieces`] records its [`TomopiecesResult`] on this thread,
    /// when a direct caller set it through [`tomopieces_recording`].
    static RESULT_SINK: std::cell::RefCell<Option<std::sync::Arc<std::sync::Mutex<Option<TomopiecesResult>>>>> =
        const { std::cell::RefCell::new(None) };
}

/// Rust-only: runs program [`tomopieces`] (its options from the in-process
/// runner's `argv`) with its result recorded into `sink`.  The program ends
/// through `exit`; run it under `commands::call_in_process`.
pub fn tomopieces_recording(sink: std::sync::Arc<std::sync::Mutex<Option<TomopiecesResult>>>) {
    RESULT_SINK.with_borrow_mut(|slot| *slot = Some(sink));
    tomopieces();
}

/// Rust-only: the lines `tomopieces` prints for `result`
/// (`tomopieces.f90:217-258`, FORMATs 101, `3i7` and 102), without endings:
/// the piece counts, the chunk sizes with `-hdf`, one range line per piece
/// (two with `-hdf`), and without `-hdf` the ranges to take back on each
/// axis.  The scripts that run it (`setupcombine`, `chunksetup`) read this
/// list line by line and copy the range strings into command files.
pub fn tomopieces_output_lines(result: &TomopiecesResult) -> Vec<String> {
    // `character*80 outputFile`
    let mut output_file = [b' '; 80];
    let mut num_out = 0_i32;
    let mut lines = Vec::new();
    let [xa, ya, za] = &result.axes;
    let (num_xpieces, num_ypieces, num_zpieces) =
        (xa.out_start.len(), ya.out_start.len(), za.out_start.len());
    // write(*,101) ...  101 format(3i4)
    lines.push(format!(
        "{}{}{}",
        fortran_i(num_xpieces as i32, 4),
        fortran_i(num_ypieces as i32, 4),
        fortran_i(num_zpieces as i32, 4)
    ));
    if let Some([nx_chunk, ny_chunk, nz_chunk]) = result.chunk_sizes {
        // write(*, '(3i7)')
        lines.push(format!(
            "{}{}{}",
            fortran_i(nx_chunk, 7),
            fortran_i(ny_chunk, 7),
            fortran_i(nz_chunk, 7)
        ));
    }
    let write_out = |lines: &mut Vec<String>, buf: &[u8], num_out: i32| {
        lines.push(String::from_utf8_lossy(&buf[..num_out as usize]).into_owned());
    };
    for iz in 1..=num_zpieces {
        for iy in 1..=num_ypieces {
            for ix in 1..=num_xpieces {
                range_out(
                    xa.out_start[ix - 1],
                    xa.out_end[ix - 1],
                    b',',
                    &mut output_file,
                    &mut num_out,
                );
                range_add(ya.out_start[iy - 1], &mut output_file, &mut num_out);
                range_add(ya.out_end[iy - 1], &mut output_file, &mut num_out);
                range_add(za.out_start[iz - 1], &mut output_file, &mut num_out);
                range_add(za.out_end[iz - 1], &mut output_file, &mut num_out);
                write_out(&mut lines, &output_file, num_out);
                if let Some([nx_chunk, ny_chunk, nz_chunk]) = result.chunk_sizes {
                    range_out(
                        xa.back_start[ix - 1],
                        xa.back_end[ix - 1],
                        b',',
                        &mut output_file,
                        &mut num_out,
                    );
                    range_add(ya.back_start[iy - 1], &mut output_file, &mut num_out);
                    range_add(ya.back_end[iy - 1], &mut output_file, &mut num_out);
                    range_add(za.back_start[iz - 1], &mut output_file, &mut num_out);
                    range_add(za.back_end[iz - 1], &mut output_file, &mut num_out);
                    range_add(nx_chunk * (ix as i32 - 1), &mut output_file, &mut num_out);
                    range_add(ny_chunk * (iy as i32 - 1), &mut output_file, &mut num_out);
                    range_add(nz_chunk * (iz as i32 - 1), &mut output_file, &mut num_out);
                    write_out(&mut lines, &output_file, num_out);
                }
            }
        }
    }
    if result.chunk_sizes.is_none() {
        for axis in [xa, ya, za] {
            for i in 0..axis.back_start.len() {
                range_out(
                    axis.back_start[i],
                    axis.back_end[i],
                    b',',
                    &mut output_file,
                    &mut num_out,
                );
                write_out(&mut lines, &output_file, num_out);
            }
        }
    }
    lines
}

/// Fortran `Iw` output of an `integer*4`: right-justified in `w` columns,
/// or `w` asterisks when it does not fit.
fn fortran_i(value: i32, w: usize) -> String {
    let text = value.to_string();
    if text.len() > w {
        "*".repeat(w)
    } else {
        format!("{text:>w$}")
    }
}

/// Original subroutine `getRanges` (`tomopieces.f90:199`).
///
/// get size to extract, and size of padded output, and offset for
/// getting back from the padded volume
#[allow(clippy::too_many_arguments)]
pub fn get_ranges(
    nx: i32,
    num_xpieces: i32,
    min_overlap: i32,
    nx_pad: i32,
    ix_out_start: &mut [i32],
    ix_out_end: &mut [i32],
    ix_back_start: &mut [i32],
    ix_back_end: &mut [i32],
    no_fft: bool,
    chunked_hdf: bool,
    nx_chunk: &mut i32,
) {
    //
    if !chunked_hdf {
        let nx_extra = (nx + (num_xpieces - 1) * min_overlap + num_xpieces - 1) / num_xpieces;
        let nx_out = pad_nice_if_fft(nx_extra, nx_pad, no_fft);
        let nx_back_offset = (nx_out - nx_extra) / 2;
        //
        // divide the total overlap into nearly equal parts
        //
        let lap_total = nx_extra * num_xpieces - nx;
        let lap_base = lap_total / 1.max(num_xpieces - 1);
        let lap_extra = lap_total % 1.max(num_xpieces - 1);
        ix_out_start[0] = 0;
        ix_back_start[0] = nx_back_offset;
        //
        // get coordinates to extract, and coordinates for reassembly
        //
        for ip in 1..=num_xpieces {
            let i = (ip - 1) as usize;
            ix_out_end[i] = ix_out_start[i] + nx_extra - 1;
            if ip < num_xpieces {
                let mut lap = lap_base;
                if ip <= lap_extra {
                    lap += 1;
                }
                let lap_top = lap / 2;
                let lap_bottom = lap - lap_top;
                ix_out_start[i + 1] = ix_out_start[i] + nx_extra - lap;
                ix_back_end[i] = nx_back_offset + nx_extra - 1 - lap_top;
                ix_back_start[i + 1] = nx_back_offset + lap_bottom;
            } else {
                ix_back_end[i] = nx_back_offset + nx_extra - 1;
            }
        }
    } else {
        //
        // For HDF chunking, figure out the chunk size first, constrain to even
        if num_xpieces == 1 {
            *nx_chunk = nx;
        } else {
            *nx_chunk = 2 * (((nx + 1) / 2 + num_xpieces - 1) / num_xpieces);
        }
        //
        // For each piece, the coordinates to extract are determined by the chunking
        // Then overlap is added except at the ends
        // Then this variable size is padded, and that is the size being computed
        // Then get the coordinates to extract from that by taking off the padding and overlap
        for ip in 1..=num_xpieces {
            let i = (ip - 1) as usize;
            ix_out_start[i] = (ip - 1) * *nx_chunk;
            ix_out_end[i] = nx.min(ip * *nx_chunk) - 1;
            let nx_back = ix_out_end[i] + 1 - ix_out_start[i];
            let mut lap_top = 0;
            let mut lap_bottom = 0;
            if ip > 1 {
                lap_bottom = min_overlap - min_overlap / 2;
            }
            ix_out_start[i] -= lap_bottom;
            if ip < num_xpieces {
                lap_top = min_overlap / 2;
            }
            ix_out_end[i] += lap_top;
            let nx_extra = ix_out_end[i] + 1 - ix_out_start[i];
            let nx_out = pad_nice_if_fft(nx_extra, nx_pad, no_fft);
            ix_back_start[i] = (nx_out - nx_extra) / 2 + lap_bottom;
            ix_back_end[i] = ix_back_start[i] + nx_back - 1;
        }
    }
}

/// Original function `padNiceIfFFT` (`tomopieces.f90:275`).
pub fn pad_nice_if_fft(nx_extra: i32, nx_pad: i32, no_fft: bool) -> i32 {
    if no_fft {
        nx_extra + 2 * nx_pad
    } else {
        nice_frame(2 * ((nx_extra + 1) / 2 + nx_pad), 2, nice_fft_limit())
    }
}

/// Original subroutine `rangeOut` (`tomopieces.f90:288`).
///
/// `buf` is the caller's `character*80`.  The substring `buf(numOut + 2:)`
/// handed to `int_iwrite` is taken as whatever is left of it.
pub fn range_out(iz_start: i32, iz_end: i32, link: u8, buf: &mut [u8], num_out: &mut i32) {
    let mut num_add = 0;
    int_iwrite(buf, iz_start, num_out);
    buf[*num_out as usize] = link;
    let from = (*num_out as usize + 1).min(buf.len());
    int_iwrite(&mut buf[from..], iz_end, &mut num_add);
    *num_out = *num_out + num_add + 1;
}

/// Original subroutine `rangeAdd` (`tomopieces.f90:298`).
pub fn range_add(iz_end: i32, buf: &mut [u8], num_out: &mut i32) {
    let mut num_add = 0;
    buf[*num_out as usize] = b',';
    let from = (*num_out as usize + 1).min(buf.len());
    int_iwrite(&mut buf[from..], iz_end, &mut num_add);
    *num_out = *num_out + num_add + 1;
}
