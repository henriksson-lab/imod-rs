//! Translation of `IMOD/flib/blend/bsubs.f90`, the subroutines used only by
//! `blendmont` (and, through `find_best_shifts`, by `solvescaling.f90`).
//!
//! One Rust function per Fortran program unit, internal (`contains`)
//! procedures included, each naming the original in its doc comment.  Every
//! unit whose source has `use blendvars` takes the module struct first — `&BlendVars`
//! where the unit only reads module variables (`oneintrp`, `fastInterp`,
//! `positionInPiece`, `findNearestPiece`, `dxydgrinterp`, `findEdgeToUse`,
//! `getExtraIndents`), `&mut BlendVars` otherwise (see the design note in
//! [`super::blendvars`]).  Units without `use blendvars` (`read_list`,
//! `joint_to_rotrans`, `edge_to_rotrans`, `solve_rotrans`,
//! `lincom_rotrans`, `recen_rotrans`, `crossvalue`, `iwrBinned`) take none.
//!
//! **Fortran units.**  `readEdgeFunc` and `doedge` read and write the
//! direct-access edge-function and edge-density files on units
//! `iunEdge(ixy)` and `iunDens(ixy)`, and `doedge` writes the patch dumps on
//! the formatted units 10 and 11; `blendmont` connects all of them
//! (`blendmont.f90:793-797`, `:1107-1172`, `:1306-1321`).  The gfortran unit
//! table has no Rust analogue (`hvem/dopen.rs` sets the precedent: the opener
//! owns the connection), so the connections are the [`BlendUnits`] value that
//! `blendmont` owns and passes to the three routines that use them.  It is not
//! part of [`BlendVars`], which holds exactly the module variables; the unit
//! *numbers* stay there (`iun_edge`, `iun_dens`).  [`DirectUnit`] reproduces
//! gfortran's direct-access unformatted record semantics, measured against
//! gfortran 11 (`/big/henriksson/realbench/wave4-bsubs/probe/da.f90`): record
//! `n` starts at byte `(n - 1) * recl`; a write shorter than `recl` is padded
//! with zeros to `recl`; a record in a hole reads as zeros; a record past the
//! end of the file is a runtime error.  `recl` is in bytes (`recl_bytes.inc`,
//! written by `IMOD/setup2:429`, has `nbytes_recl_item = 1`).
//!
//! **gfortran intrinsics.**  `cosd`/`sind` are not inlined by gfortran: they
//! enter `_gfortran_cosd_r4`/`_gfortran_sind_r4` in the linked
//! `libgfortran.so.5` (`nm bsubs.o`), whose argument reduction, exact special
//! values and hi/lo `fmaf` degree-to-radian conversion are reproduced by
//! [`gfortran_cosd_r4`] and [`gfortran_sind_r4`] (disassembled from that
//! library; they live in [`crate::imod::flib::subrs::compat::gfortran_rt`],
//! shared with every translated Fortran unit, and are re-exported here).  `atand` *is* inlined, as `atanf(x) * 57.29578f`
//! (`solve_rotrans`, bit pattern `0x42652ee0` read from `bsubs.o`).
//! `nint` is `lroundf` (Rust `round`, half away from zero); real-to-integer
//! assignment truncates (`as i32`).  `MAX`/`MIN` of reals become gcc
//! `MAX_EXPR`/`MIN_EXPR` (`maxss`/`minss`), whose operand order for a NaN or a
//! signed-zero tie is gcc's choice (it even reassociates the five-argument
//! `max` in `oneintrp`); they are written here as `if a > b { a } else { b }`
//! in source order, which is exact for every other input.
//!
//! **Aliasing at call sites.**  `blendmont` calls `lincom_rotrans` and
//! `recen_rotrans` with the output array aliasing an input
//! (`blendmont.f90:2753,2767-2777`); both bodies read each input element
//! before the output element that would alias it is stored, so a caller
//! passes a copy of the aliased input.  `countedges` and `blendmont` pass
//! module elements (`xInPiece(1)`) as `positionInPiece`'s outputs; the
//! routine writes them only, so the translation writes through locals.
//!
//! No `!$OMP` directives in the unit.  The commented-out SVD solve with
//! `dgelss` (`bsubs.f90:2288`) stays commented out.

use super::blendvars::{BlendVars, LIM_EDG_BF, MAX_DIST_NEAR, MAX_IN_PC};
use super::edgesubs::{findedgefunc, setgridchars};
use super::shuffler::shuffler;
use super::smoothgrid::smoothgrid;
pub use crate::imod::flib::subrs::compat::gfortran_rt::{gfortran_cosd_r4, gfortran_sind_r4};
use crate::imod::flib::subrs::hvem::b3dxor::b3dxor;
use crate::imod::flib::subrs::hvem::frefor::frefor;
use crate::imod::flib::subrs::hvem::inside::inside;
use crate::imod::flib::subrs::hvem::parse_input_params::exit_error;
use crate::imod::flib::subrs::imsubs::convert_vms::{convert_floats, convert_longs};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::model::readw_or_imod::readw_or_imod;
use crate::imod::flib::subrs::model::scale_model::scale_model_to_image;
use crate::imod::libcfshr::amoeba::{amoebafwrap, amoebainitfwrap};
use crate::imod::libcfshr::b3dutil::{ImodFile, walltime};
use crate::imod::libcfshr::find_piece_shifts::{
    findpiecescalings, findpieceshifts, pickalternativeshifts,
};
use crate::imod::libcfshr::gaussj::gaussjfw;
use crate::imod::libcfshr::linearxforms::{xfcopy, xfinvert, xfunit};
use crate::imod::libcfshr::montagexcorr::{
    mont_xc_get_last_trimmed_max_sd, montxcbasicsizes, montxcgetlastrunnersup, montxcindsandctf,
    montxcorredge,
};
use crate::imod::libcfshr::parse_params::pip_get_string;
use crate::imod::libcfshr::reduce_by_binning::extractwithbinning;
use crate::imod::libcfshr::sdsearch::montbigsearch;
use crate::imod::libcfshr::simplestat::array_min_max_mean_fortran;
use crate::imod::libcfshr::taperatfill::taperatfill;
use crate::imod::libfft::odfft::nice_fft_limit;
use crate::imod::libfft::todfft::todfft;
use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, MRC_NLABELS};
use crate::imod::libiimod::parallelwrite::par_wrt_lin;
use crate::imod::libiimod::unit_fileio::{iiu_set_position, iiu_write_lines};
use crate::imod::libiimod::unit_header::{
    iiu_alt_size_samp_cell, iiu_create_header, iiu_write_header,
};
use crate::imod::libwarp::maggradfield::mag_gradient_shift;
use crate::imod::libwarp::warputils::interpolate_grid;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::os::unix::fs::FileExt;

// ---------------------------------------------------------------------------
// gfortran runtime boundary: Fortran unit connections and degree intrinsics.
// ---------------------------------------------------------------------------

/// A Fortran unit connected with `form = 'unformatted', access = 'direct'`
/// (`blendmont.f90:1107,1161,1306,1320`).  See the module note for the record
/// semantics this reproduces.
#[derive(Debug)]
pub struct DirectUnit {
    /// The Fortran unit number (`iunEdge(ixy)` or `iunDens(ixy)`).
    pub iunit: i32,
    /// The connected file.
    pub file: File,
    /// Its name, for gfortran's runtime-error message.
    pub name: String,
    /// `recl`, in bytes.
    pub recl: usize,
}

impl DirectUnit {
    /// `open(iunit, file = name, status = 'new' | 'old', form = 'unformatted',
    /// access = 'direct', recl = recl)`.  gfortran opens a direct-access
    /// file for reading and writing; `status = 'new'` fails if it exists.
    pub fn open(iunit: i32, name: &str, status_new: bool, recl: i32) -> std::io::Result<Self> {
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true);
        if status_new {
            options.create_new(true);
        }
        Ok(DirectUnit {
            iunit,
            file: options.open(name)?,
            name: name.to_string(),
            recl: recl as usize,
        })
    }

    /// `read(iunit, rec = rec) ...`: the whole record, `recl` bytes.  A
    /// record past the end of the file is an error.
    pub fn read_record(&self, rec: i32) -> std::io::Result<Vec<u8>> {
        let mut buf = vec![0u8; self.recl];
        self.file
            .read_exact_at(&mut buf, (rec as u64 - 1) * self.recl as u64)?;
        Ok(buf)
    }

    /// `write(iunit, rec = rec) ...`: the list's bytes, zero-padded to `recl`.
    pub fn write_record(&self, rec: i32, data: &[u8]) -> std::io::Result<()> {
        if data.len() > self.recl {
            return Err(std::io::Error::other("End of record"));
        }
        let mut buf = data.to_vec();
        buf.resize(self.recl, 0);
        self.file
            .write_all_at(&buf, (rec as u64 - 1) * self.recl as u64)
    }

    /// A transfer with no `err=`/`iostat=` that fails is a gfortran runtime
    /// error: the message goes to stderr and the process exits with status 2.
    pub fn runtime_error(&self, err: std::io::Error) -> ! {
        eprintln!(
            "At line 0 of file bsubs.f90 (unit = {}, file = '{}')\nFortran runtime error: {}",
            self.iunit, self.name, err
        );
        crate::imod::libcfshr::b3dutil::exit(2);
    }
}

/// The Fortran unit connections `bsubs` uses and `blendmont` makes (module
/// note).  `edge[ixy - 1]` is unit `iunEdge(ixy)`, `dens[ixy - 1]` is unit
/// `iunDens(ixy)`; `unit10`/`unit11` are the patch-dump units `blendmont`
/// opens with `dopen` when `izUnsmoothedPatch`/`izSmoothedPatch` is set
/// (`blendmont.f90:793-797`).
#[derive(Debug, Default)]
pub struct BlendUnits {
    pub edge: [Option<DirectUnit>; 2],
    pub dens: [Option<DirectUnit>; 2],
    pub unit10: Option<BufWriter<File>>,
    pub unit11: Option<BufWriter<File>>,
}

/// `(180/pi)` as gfortran folds it for the inline `atand` (`atanf(x) * c`):
/// the `.rodata.cst4` constant `0x42652ee0` in `bsubs.o`.
const ATAND_FACTOR: f32 = f32::from_bits(0x42652ee0);
// ---------------------------------------------------------------------------
// bsubs.f90
// ---------------------------------------------------------------------------

/// Original: `subroutine read_list(ixPcList, iyPcList, izPcList, negList,
/// multiNeg, numPieceList, minZpiece, maxZpiece, anyNeg, pipInput)`
/// (`bsubs.f90:49`).
///
/// Gets the name of a piece list file, reads the file and returns the number
/// of pieces, the min and max Z, the piece coordinates, the optional negative
/// number of each piece, and whether each section is multinegative.
/// Unit 3 is a local `BufReader`; `close(3)` is its drop.
pub fn read_list(
    ix_pc_list: &mut [i32],
    iy_pc_list: &mut [i32],
    iz_pc_list: &mut [i32],
    neg_list: &mut [i32],
    multi_neg: &mut [bool],
    num_piece_list: &mut i32,
    min_zpiece: &mut i32,
    max_zpiece: &mut i32,
    any_neg: &mut bool,
    pip_input: bool,
) {
    let mut free_input = [0.0f32; 10];
    // `character*32 errMess(4)`: written untrimmed, so each is blank-padded
    // to 32.
    let err_mess: [&str; 4] = [
        "error opening file",
        "error reading file, line",
        "bad number of values on line",
        "bad multinegative specification",
    ];
    let mut num_input = 0i32;
    let mut list_first = 0i32;
    let mut ierr: usize;
    //
    // get file name and open file
    //
    // `character*320 fileName`: at most 320 characters; gfortran drops the
    // trailing blanks of a file name.
    let mut file_name: String;
    if pip_input {
        let mut value = Vec::new();
        if pip_get_string(b"PieceListInput", &mut value) != 0 {
            exit_error("No input piece list file specified");
        }
        value.truncate(320);
        file_name = String::from_utf8_lossy(&value).into_owned();
    } else {
        // `write(*,'(1x,a,$)')` then `read(5, '(a)') fileName`.
        let _ = ImodFile::Stdout.write_all(b" name of input piece list file: ");
        let _ = ImodFile::Stdout.flush();
        let mut line = Vec::new();
        let _ = std::io::stdin().lock().read_until(b'\n', &mut line);
        if line.last() == Some(&b'\n') {
            line.pop();
        }
        line.truncate(320);
        file_name = String::from_utf8_lossy(&line).into_owned();
    }
    file_name = file_name.trim_end_matches(' ').to_string();
    ierr = 1;
    *num_piece_list = 0;
    *any_neg = false;
    // 7/14/00 CER: remove carriagecontrol for LINUX
    'error: {
        let Ok(file) = File::open(&file_name) else {
            break 'error;
        };
        let mut unit3 = BufReader::new(file);
        *min_zpiece = 100000;
        *max_zpiece = -100000;
        //
        // read each line in turn, get numbers with free format input
        //
        loop {
            // 12
            ierr = 2;
            let mut record = Vec::new();
            match unit3.read_until(b'\n', &mut record) {
                Err(_) => break 'error,
                Ok(0) => break, // end = 14
                Ok(_) => {}
            }
            if record.last() == Some(&b'\n') {
                record.pop();
                if record.last() == Some(&b'\r') {
                    record.pop();
                }
            }
            // `character*120 dummy`: the record truncated or blank-padded
            // to 120.
            record.truncate(120);
            record.resize(120, b' ');
            let mut len_act = record.len();
            let mut blank = false;
            while record[len_act - 1] == b' ' || record[len_act - 1] == 0 {
                len_act -= 1;
                if len_act == 0 {
                    blank = true;
                    break;
                }
            }
            if blank {
                continue; // go to 12
            }
            let dummy = String::from_utf8_lossy(&record).into_owned();
            frefor(&dummy, &mut free_input, &mut num_input);
            ierr = 3;
            if num_input < 3 || num_input > 4 {
                break 'error; //error if fewer than 3 numbers
            }
            *num_piece_list += 1;
            let n = (*num_piece_list - 1) as usize;
            ix_pc_list[n] = free_input[0] as i32;
            iy_pc_list[n] = free_input[1] as i32;
            iz_pc_list[n] = free_input[2] as i32;
            neg_list[n] = 0; //if 4th number, its a neg #
            if num_input == 4 {
                neg_list[n] = free_input[3] as i32;
            }
            *min_zpiece = (*min_zpiece).min(iz_pc_list[n]);
            *max_zpiece = (*max_zpiece).max(iz_pc_list[n]);
        }
        //
        // now check for multinegative montaging: all pieces in a section must
        // be labeled by negative if any are
        //
        // 14
        ierr = 4;
        for iz in *min_zpiece..=*max_zpiece {
            let ineg = (iz + 1 - *min_zpiece - 1) as usize;
            multi_neg[ineg] = false;
            let mut any_zero = false;
            let mut got_first = false;
            for ipc in 1..=*num_piece_list {
                let k = (ipc - 1) as usize;
                if iz_pc_list[k] == iz {
                    any_zero = any_zero || (neg_list[k] == 0);
                    if got_first {
                        multi_neg[ineg] = multi_neg[ineg] || (neg_list[k] != list_first);
                    } else {
                        list_first = neg_list[k];
                        got_first = true;
                    }
                }
            }
            if multi_neg[ineg] && any_zero {
                break 'error;
            }
            *any_neg = *any_neg || multi_neg[ineg];
        }
        return;
    }
    // 20
    let _ = writeln!(
        ImodFile::Stdout,
        " ERROR: BLENDMONT - {} - {:<32}{:6}",
        file_name,
        err_mess[ierr - 1],
        *num_piece_list + 1
    );
    crate::imod::libcfshr::b3dutil::exit(1);
}

/// Original: `real*4 function oneintrp(arrIn, nx, ny, x, y, izPiece)`
/// (`bsubs.f90:149`).
///
/// Interpolates in `arrIn` (dimensions `nx`, `ny`) at (`x`, `y`), coordinates
/// running from 0 to `nx - 1`; returns `dfill` outside.  `izPiece` is the
/// piece number, numbered from 1; the order is `interpOrder`.
pub fn oneintrp(
    bv: &BlendVars,
    arr_in: &[f32],
    nx: i32,
    ny: i32,
    x: f32,
    y: f32,
    iz_piece: i32,
) -> f32 {
    let a = |i: i32, j: i32| arr_in[((i - 1) + nx * (j - 1)) as usize];
    let mut oneintrp = bv.dfill;
    let mut xp = x + 1.;
    let mut yp = y + 1.;
    let mut dx = 0.0f32;
    let mut dy = 0.0f32;
    if bv.do_fields {
        let mem = (bv.mem_index[(iz_piece - 1) as usize] - 1) as usize;
        interpolate_grid(
            xp,
            yp,
            &bv.field_dx[bv.field_dx_ext[0] * bv.field_dx_ext[1] * mem..],
            &bv.field_dy[bv.field_dy_ext[0] * bv.field_dy_ext[1] * mem..],
            bv.lm_field,
            bv.nx_field,
            bv.ny_field,
            bv.x_field_strt,
            bv.y_field_strt,
            bv.x_field_intrv,
            bv.y_field_intrv,
            &mut dx,
            &mut dy,
        );
        xp += dx;
        yp += dy;
    }

    if bv.interp_order <= 1 {
        //
        // Linear interpolation
        //
        let ixp = xp as i32;
        let iyp = yp as i32;
        if ixp >= 1 && ixp <= nx && iyp >= 1 && iyp <= ny {
            dx = xp - ixp as f32;
            dy = yp - iyp as f32;
            let ixp_p1 = nx.min(ixp + 1);
            let iyp_p1 = ny.min(iyp + 1);
            oneintrp = (1. - dy) * ((1. - dx) * a(ixp, iyp) + dx * a(ixp_p1, iyp))
                + dy * ((1. - dx) * a(ixp, iyp_p1) + dx * a(ixp_p1, iyp_p1));
        }
    } else if bv.interp_order == 2 {
        //
        // Old quadratic interpolation
        //
        let ixp = xp.round() as i32;
        let iyp = yp.round() as i32;
        if ixp < 1 || ixp > nx || iyp < 1 || iyp > ny {
            return oneintrp;
        }
        dx = xp - ixp as f32;
        dy = yp - iyp as f32;
        // but if on an integer boundary already, done
        if dx == 0. && dy == 0. {
            return a(ixp, iyp);
        }
        //
        let ixp_p1 = nx.min(ixp + 1);
        let ixp_m1 = 1.max(ixp - 1);
        let iyp_p1 = ny.min(iyp + 1);
        let iyp_m1 = 1.max(iyp - 1);
        //
        // Set up terms for quadratic interpolation
        //
        let v2 = a(ixp, iyp_m1);
        let v4 = a(ixp_m1, iyp);
        let v5 = a(ixp, iyp);
        let v6 = a(ixp_p1, iyp);
        let v8 = a(ixp, iyp_p1);
        // find min and max of all 5 points
        let mut vmax = v2;
        for v in [v4, v5, v6, v8] {
            vmax = if vmax > v { vmax } else { v };
        }
        let mut vmin = v2;
        for v in [v4, v5, v6, v8] {
            vmin = if vmin < v { vmin } else { v };
        }
        //
        let a2 = (v6 + v4) * 0.5 - v5;
        let b = (v8 + v2) * 0.5 - v5;
        let c = (v6 - v4) * 0.5;
        let d = (v8 - v2) * 0.5;
        //
        // limit the new density to between the min and max of original points
        let val = a2 * dx * dx + b * dy * dy + c * dx + d * dy + v5;
        let inner = if vmax < val { vmax } else { val };
        oneintrp = if vmin > inner { vmin } else { inner };
    } else {
        //
        // cubic interpolation
        //
        let ixp = xp as i32;
        let iyp = yp as i32;
        if ixp >= 1 && ixp <= nx && iyp >= 1 && iyp <= ny {
            dx = xp - ixp as f32;
            dy = yp - iyp as f32;
            let ixp_p1 = nx.min(ixp + 1);
            let ixp_m1 = 1.max(ixp - 1);
            let iyp_p1 = ny.min(iyp + 1);
            let iyp_m1 = 1.max(iyp - 1);
            let ixp_p2 = nx.min(ixp + 2);
            let iyp_p2 = ny.min(iyp + 2);

            let dx_m1 = dx - 1.;
            let dx_dx_m1 = dx * dx_m1;
            let fx1 = -dx_m1 * dx_dx_m1;
            let fx4 = dx * dx_dx_m1;
            let fx2 = 1. + dx * dx * (dx - 2.);
            let fx3 = dx * (1. - dx_dx_m1);

            let dy_m1 = dy - 1.;
            let dy_dy_m1 = dy * dy_m1;

            let v1 = fx1 * a(ixp_m1, iyp_m1)
                + fx2 * a(ixp, iyp_m1)
                + fx3 * a(ixp_p1, iyp_m1)
                + fx4 * a(ixp_p2, iyp_m1);
            let v2 = fx1 * a(ixp_m1, iyp)
                + fx2 * a(ixp, iyp)
                + fx3 * a(ixp_p1, iyp)
                + fx4 * a(ixp_p2, iyp);
            let v3 = fx1 * a(ixp_m1, iyp_p1)
                + fx2 * a(ixp, iyp_p1)
                + fx3 * a(ixp_p1, iyp_p1)
                + fx4 * a(ixp_p2, iyp_p1);
            let v4 = fx1 * a(ixp_m1, iyp_p2)
                + fx2 * a(ixp, iyp_p2)
                + fx3 * a(ixp_p1, iyp_p2)
                + fx4 * a(ixp_p2, iyp_p2);
            oneintrp = -dy_m1 * dy_dy_m1 * v1
                + (1. + dy * dy * (dy - 2.)) * v2
                + dy * (1. - dy_dy_m1) * v3
                + dy * dy_dy_m1 * v4;
        }
    }
    oneintrp
}

/// Original: `subroutine fastInterp(arrOut, nxOutput, nyOutput, arrIn,
/// nxInput, nyInput, indXlow, inXhigh, indYlow, indYhigh, newPcXlowLeft,
/// amat, fdx, fdy, izPiece)` (`bsubs.f90:284`).
///
/// Fills the part of `arrOut` from `indXlow` to `inXhigh` and `indYlow` to
/// `indYhigh` by interpolation in `arrIn` under the transformation `amat`,
/// `fdx`, `fdy`; `newPcXlowLeft` is the X coordinate of the output array's
/// lower left corner.  `real*4 amat(2,2)` is dimension-reversed:
/// `amat(i, j)` is `amat[j - 1][i - 1]`.
pub fn fast_interp(
    bv: &BlendVars,
    arr_out: &mut [f32],
    nx_output: i32,
    _ny_output: i32,
    arr_in: &[f32],
    nx_input: i32,
    ny_input: i32,
    ind_xlow: i32,
    in_xhigh: i32,
    ind_ylow: i32,
    ind_yhigh: i32,
    new_pc_xlow_left: i32,
    amat: &[[f32; 2]; 2],
    fdx: f32,
    fdy: f32,
    iz_piece: i32,
) {
    let a = |i: i32, j: i32| arr_in[((i - 1) + nx_input * (j - 1)) as usize];
    let out = |i: i32, j: i32| ((i - 1) + nx_output * (j - 1)) as usize;
    let (mut dx, mut dy) = (0.0f32, 0.0f32);
    let field_off =
        |ext: &[usize; 3]| ext[0] * ext[1] * (bv.mem_index[(iz_piece - 1) as usize] - 1) as usize;
    //
    for ind_y in ind_ylow..=ind_yhigh {
        let iy_out = ind_y + 1 - ind_ylow;
        let mut ry = ind_y as f32;
        let mut xbase = amat[1][0] * ry + fdx;
        let mut ybase = amat[1][1] * ry + fdy;
        if bv.interp_order <= 1 {
            //
            // Linear interpolation
            //
            for ind_x in ind_xlow..=in_xhigh {
                let ix_out = ind_x + 1 - new_pc_xlow_left;
                let mut pix_val = bv.dfill;
                let mut rx = ind_x as f32;
                if bv.sec_has_warp {
                    interpolate_grid(
                        rx - 0.5,
                        ind_y as f32 - 0.5,
                        &bv.warp_dx,
                        &bv.warp_dy,
                        bv.lm_warp_x,
                        bv.nx_warp,
                        bv.ny_warp,
                        bv.x_warp_strt,
                        bv.y_warp_strt,
                        bv.x_warp_intrv,
                        bv.y_warp_intrv,
                        &mut dx,
                        &mut dy,
                    );
                    rx += dx;
                    ry = ind_y as f32 + dy;
                    xbase = amat[1][0] * ry + fdx;
                    ybase = amat[1][1] * ry + fdy;
                }
                let mut xp = amat[0][0] * rx + xbase;
                let mut yp = amat[0][1] * rx + ybase;
                if bv.do_fields {
                    interpolate_grid(
                        xp,
                        yp,
                        &bv.field_dx[field_off(&bv.field_dx_ext)..],
                        &bv.field_dy[field_off(&bv.field_dy_ext)..],
                        bv.lm_field,
                        bv.nx_field,
                        bv.ny_field,
                        bv.x_field_strt,
                        bv.y_field_strt,
                        bv.x_field_intrv,
                        bv.y_field_intrv,
                        &mut dx,
                        &mut dy,
                    );
                    xp += dx;
                    yp += dy;
                }
                let ixp = xp as i32;
                let iyp = yp as i32;
                if ixp >= 1 && ixp < nx_input && iyp >= 1 && iyp < ny_input {
                    dx = xp - ixp as f32;
                    dy = yp - iyp as f32;
                    let ixp_p1 = ixp + 1;
                    let iyp_p1 = iyp + 1;
                    pix_val = (1. - dy) * ((1. - dx) * a(ixp, iyp) + dx * a(ixp_p1, iyp))
                        + dy * ((1. - dx) * a(ixp, iyp_p1) + dx * a(ixp_p1, iyp_p1));
                }
                arr_out[out(ix_out, iy_out)] = pix_val;
            }
            //
        } else if bv.interp_order == 2 {
            //
            // Old quadratic interpolation
            //
            for ind_x in ind_xlow..=in_xhigh {
                let ix_out = ind_x + 1 - new_pc_xlow_left;
                let mut pix_val = bv.dfill;
                let mut rx = ind_x as f32;
                if bv.sec_has_warp {
                    interpolate_grid(
                        rx - 0.5,
                        ind_y as f32 - 0.5,
                        &bv.warp_dx,
                        &bv.warp_dy,
                        bv.lm_warp_x,
                        bv.nx_warp,
                        bv.ny_warp,
                        bv.x_warp_strt,
                        bv.y_warp_strt,
                        bv.x_warp_intrv,
                        bv.y_warp_intrv,
                        &mut dx,
                        &mut dy,
                    );
                    rx += dx;
                    ry = ind_y as f32 + dy;
                    xbase = amat[1][0] * ry + fdx;
                    ybase = amat[1][1] * ry + fdy;
                }
                let mut xp = amat[0][0] * rx + xbase;
                let mut yp = amat[0][1] * rx + ybase;
                if bv.do_fields {
                    interpolate_grid(
                        xp,
                        yp,
                        &bv.field_dx[field_off(&bv.field_dx_ext)..],
                        &bv.field_dy[field_off(&bv.field_dy_ext)..],
                        bv.lm_field,
                        bv.nx_field,
                        bv.ny_field,
                        bv.x_field_strt,
                        bv.y_field_strt,
                        bv.x_field_intrv,
                        bv.y_field_intrv,
                        &mut dx,
                        &mut dy,
                    );
                    xp += dx;
                    yp += dy;
                }
                'l80: {
                    let ixp = xp.round() as i32;
                    let iyp = yp.round() as i32;
                    if ixp < 1 || ixp > nx_input || iyp < 1 || iyp > ny_input {
                        break 'l80;
                    }
                    //
                    // Do quadratic interpolation
                    //
                    dx = xp - ixp as f32;
                    dy = yp - iyp as f32;
                    let v5 = a(ixp, iyp);
                    //
                    // but if on an integer boundary already, done
                    //
                    if dx == 0. && dy == 0. {
                        pix_val = v5;
                        break 'l80;
                    }
                    //
                    let ixp_p1 = nx_input.min(ixp + 1);
                    let ixp_m1 = 1.max(ixp - 1);
                    let iyp_p1 = ny_input.min(iyp + 1);
                    let iyp_m1 = 1.max(iyp - 1);
                    //
                    // Set up terms for quadratic interpolation
                    //
                    let v2 = a(ixp, iyp_m1);
                    let v4 = a(ixp_m1, iyp);
                    let v6 = a(ixp_p1, iyp);
                    let v8 = a(ixp, iyp_p1);
                    //
                    // find min and max of all 5 points
                    //
                    let mut vmax = v2;
                    for v in [v4, v5, v6, v8] {
                        vmax = if vmax > v { vmax } else { v };
                    }
                    let mut vmin = v2;
                    for v in [v4, v5, v6, v8] {
                        vmin = if vmin < v { vmin } else { v };
                    }
                    //
                    let a2 = (v6 + v4) * 0.5 - v5;
                    let b = (v8 + v2) * 0.5 - v5;
                    let c = (v6 - v4) * 0.5;
                    let d = (v8 - v2) * 0.5;
                    //
                    // limit the new density to between min and max of original points
                    //
                    let val = a2 * dx * dx + b * dy * dy + c * dx + d * dy + v5;
                    let inner = if vmax < val { vmax } else { val };
                    pix_val = if vmin > inner { vmin } else { inner };
                }
                // 80
                arr_out[out(ix_out, iy_out)] = pix_val;
            }
        } else {
            //
            // cubic interpolation
            //
            for ind_x in ind_xlow..=in_xhigh {
                let ix_out = ind_x + 1 - new_pc_xlow_left;
                let mut pix_val = bv.dfill;
                let mut rx = ind_x as f32;
                if bv.sec_has_warp {
                    interpolate_grid(
                        rx - 0.5,
                        ind_y as f32 - 0.5,
                        &bv.warp_dx,
                        &bv.warp_dy,
                        bv.lm_warp_x,
                        bv.nx_warp,
                        bv.ny_warp,
                        bv.x_warp_strt,
                        bv.y_warp_strt,
                        bv.x_warp_intrv,
                        bv.y_warp_intrv,
                        &mut dx,
                        &mut dy,
                    );
                    rx += dx;
                    ry = ind_y as f32 + dy;
                    xbase = amat[1][0] * ry + fdx;
                    ybase = amat[1][1] * ry + fdy;
                }
                let mut xp = amat[0][0] * rx + xbase;
                let mut yp = amat[0][1] * rx + ybase;
                if bv.do_fields {
                    interpolate_grid(
                        xp,
                        yp,
                        &bv.field_dx[field_off(&bv.field_dx_ext)..],
                        &bv.field_dy[field_off(&bv.field_dy_ext)..],
                        bv.lm_field,
                        bv.nx_field,
                        bv.ny_field,
                        bv.x_field_strt,
                        bv.y_field_strt,
                        bv.x_field_intrv,
                        bv.y_field_intrv,
                        &mut dx,
                        &mut dy,
                    );
                    xp += dx;
                    yp += dy;
                }
                let ixp = xp as i32;
                let iyp = yp as i32;
                if ixp >= 2 && ixp < nx_input - 1 && iyp >= 2 && iyp < ny_input - 1 {
                    dx = xp - ixp as f32;
                    dy = yp - iyp as f32;
                    let ixp_p1 = ixp + 1;
                    let ixp_m1 = ixp - 1;
                    let iyp_p1 = iyp + 1;
                    let iyp_m1 = iyp - 1;
                    let ixp_p2 = ixp + 2;
                    let iyp_p2 = iyp + 2;

                    let dx_m1 = dx - 1.;
                    let dx_dx_m1 = dx * dx_m1;
                    let fx1 = -dx_m1 * dx_dx_m1;
                    let fx4 = dx * dx_dx_m1;
                    let fx2 = 1. + dx * dx * (dx - 2.);
                    let fx3 = dx * (1. - dx_dx_m1);

                    let dy_m1 = dy - 1.;
                    let dy_dy_m1 = dy * dy_m1;

                    // The 4 x 4 neighbourhood as four row slices (one bounds
                    // check per row instead of per element); the same
                    // elements `a(ixp_m1..=ixp_p2, iyp_m1..=iyp_p2)` in the
                    // same expressions.
                    let _ = (ixp_p1, iyp_p1, iyp_p2);
                    let row = |j: i32| {
                        let start = ((ixp_m1 - 1) + nx_input * (j - 1)) as usize;
                        &arr_in[start..start + 4]
                    };
                    let (r1, r2, r3, r4) = (row(iyp_m1), row(iyp), row(iyp + 1), row(iyp + 2));
                    let v1 = fx1 * r1[0] + fx2 * r1[1] + fx3 * r1[2] + fx4 * r1[3];
                    let v2 = fx1 * r2[0] + fx2 * r2[1] + fx3 * r2[2] + fx4 * r2[3];
                    let v3 = fx1 * r3[0] + fx2 * r3[1] + fx3 * r3[2] + fx4 * r3[3];
                    let v4 = fx1 * r4[0] + fx2 * r4[1] + fx3 * r4[2] + fx4 * r4[3];
                    pix_val = -dy_m1 * dy_dy_m1 * v1
                        + (1. + dy * dy * (dy - 2.)) * v2
                        + dy * (1. - dy_dy_m1) * v3
                        + dy * dy_dy_m1 * v4;
                    //
                }
                arr_out[out(ix_out, iy_out)] = pix_val;
            }
        }
    }
}

/// Original: `subroutine joint_to_rotrans(dxgrid, dygrid, ixdim, iydim,
/// nxgrid, nygrid, intxgrid, intygrid, ixpclo, iypclo, ixpchi, iypchi,
/// nedge, r)` (`bsubs.f90:498`).
///
/// Derives a rotation/translation around the center of a joint from the
/// `nedge` edge functions in `dxgrid`/`dygrid` (`ixdim` x `iydim` x `*`);
/// `r` is the `/rotrans/` structure `theta, dx, dy, xcen, ycen`.  No
/// `implicit none`: `x1sum` etc. are real, the rest integer.
pub fn joint_to_rotrans(
    dxgrid: &[f32],
    dygrid: &[f32],
    ixdim: i32,
    iydim: i32,
    nxgrid: &[i32],
    nygrid: &[i32],
    intxgrid: i32,
    intygrid: i32,
    ixpclo: &[i32],
    iypclo: &[i32],
    ixpchi: &[i32],
    iypchi: &[i32],
    nedge: i32,
    r: &mut [f32],
) {
    const NPNTS: usize = 5000;
    let mut x1 = [0.0f32; NPNTS];
    let mut y1 = [0.0f32; NPNTS];
    let mut x2 = [0.0f32; NPNTS];
    let mut y2 = [0.0f32; NPNTS];
    let g =
        |ix: i32, iy: i32, ied: i32| ((ix - 1) + ixdim * ((iy - 1) + iydim * (ied - 1))) as usize;
    //
    // make set of coordinate pairs in each grid, add up sums
    let mut x1sum = 0.0f32;
    let mut y1sum = 0.0f32;
    let mut nn = 0i32;
    for ied in 1..=nedge {
        let e = (ied - 1) as usize;
        for ix in 1..=nxgrid[e] {
            for iy in 1..=nygrid[e] {
                nn += 1;
                let k = (nn - 1) as usize;
                x1[k] = ((ix - 1) * intxgrid + ixpclo[e]) as f32;
                y1[k] = ((iy - 1) * intygrid + iypclo[e]) as f32;
                x2[k] = ((ix - 1) * intxgrid + ixpchi[e]) as f32 + dxgrid[g(ix, iy, ied)];
                y2[k] = ((iy - 1) * intygrid + iypchi[e]) as f32 + dygrid[g(ix, iy, ied)];
                x1sum += x1[k];
                y1sum += y1[k];
            }
        }
    }
    //
    // get center in lower image and shift points to center around there
    //
    r[3] = x1sum / nn as f32;
    r[4] = y1sum / nn as f32;
    for i in 0..nn as usize {
        x1[i] -= r[3];
        x2[i] -= r[3];
        y1[i] -= r[4];
        y2[i] -= r[4];
    }
    let (mut theta, mut dx, mut dy) = (r[0], r[1], r[2]);
    solve_rotrans(&x1, &y1, &x2, &y2, nn, &mut theta, &mut dx, &mut dy);
    r[0] = theta;
    r[1] = dx;
    r[2] = dy;
}

/// Original: `subroutine edge_to_rotrans(dxgrid, dygrid, ixdim, iydim,
/// nxgrid, nygrid, intxgrid, intygrid, thetamin, dxmin, dymin)`
/// (`bsubs.f90:557`).
///
/// Derives a rotation/translation around the center of one edge from its
/// edge function.  No `implicit none`.
pub fn edge_to_rotrans(
    dxgrid: &[f32],
    dygrid: &[f32],
    ixdim: i32,
    _iydim: i32,
    nxgrid: i32,
    nygrid: i32,
    intxgrid: i32,
    intygrid: i32,
    thetamin: &mut f32,
    dxmin: &mut f32,
    dymin: &mut f32,
) {
    const NPNTS: usize = 1000;
    let mut x1 = [0.0f32; NPNTS];
    let mut y1 = [0.0f32; NPNTS];
    let mut x2 = [0.0f32; NPNTS];
    let mut y2 = [0.0f32; NPNTS];
    //
    // make a set of coordinate pairs centered around center of grid
    //
    let xcen = (nxgrid - 1) as f32 / 2. + 1.;
    let ycen = (nygrid - 1) as f32 / 2. + 1.;
    let mut nn = 0i32;
    for ix in 1..=nxgrid {
        for iy in 1..=nygrid {
            nn += 1;
            let k = (nn - 1) as usize;
            let gi = ((ix - 1) + ixdim * (iy - 1)) as usize;
            x1[k] = (ix as f32 - xcen) * intxgrid as f32;
            y1[k] = (iy as f32 - ycen) * intygrid as f32;
            x2[k] = x1[k] + dxgrid[gi];
            y2[k] = y1[k] + dygrid[gi];
        }
    }
    solve_rotrans(&x1, &y1, &x2, &y2, nn, thetamin, dxmin, dymin);
}

/// Original: `subroutine solve_rotrans(x1, y1, x2, y2, nn, thetamin, dxmin,
/// dymin)` (`bsubs.f90:588`).
///
/// Finds the rotation about the origin and translation that best
/// superimposes the `nn` points (`x1`, `y1`) on (`x2`, `y2`).  The search
/// version below the solution is commented out in the source.
pub fn solve_rotrans(
    x1: &[f32],
    y1: &[f32],
    x2: &[f32],
    y2: &[f32],
    nn: i32,
    thetamin: &mut f32,
    dxmin: &mut f32,
    dymin: &mut f32,
) {
    //
    // the real way to do it
    //
    let mut x1s = 0.0f32;
    let mut x2s = 0.0f32;
    let mut y1s = 0.0f32;
    let mut y2s = 0.0f32;
    for i in 0..nn as usize {
        x1s += x1[i];
        x2s += x2[i];
        y1s += y1[i];
        y2s += y2[i];
    }
    x1s /= nn as f32;
    x2s /= nn as f32;
    y1s /= nn as f32;
    y2s /= nn as f32;
    let mut ssd23 = 0.0f32;
    let mut ssd14 = 0.0f32;
    let mut ssd13 = 0.0f32;
    let mut ssd24 = 0.0f32;
    for i in 0..nn as usize {
        ssd23 += (y1[i] - y1s) * (x2[i] - x2s);
        ssd14 += (x1[i] - x1s) * (y2[i] - y2s);
        ssd13 += (x1[i] - x1s) * (x2[i] - x2s);
        ssd24 += (y1[i] - y1s) * (y2[i] - y2s);
    }
    *thetamin = 0.;
    if (ssd13 + ssd24).abs() > 1.0e-20 {
        // `atand` is inlined by gfortran as `atanf(x) * (180/pi)`.
        *thetamin = (-(ssd23 - ssd14) / (ssd13 + ssd24)).atan() * ATAND_FACTOR;
    }
    let costh = gfortran_cosd_r4(*thetamin);
    let sinth = gfortran_sind_r4(*thetamin);
    *dxmin = x2s - x1s * costh + y1s * sinth;
    *dymin = y2s - x1s * sinth - y1s * costh;
}

/// Original: `subroutine edgeswap(iedge, ixy, indbuf)` (`bsubs.f90:699`).
///
/// Manages the edge function buffers: returns in `indbuf` the buffer holding
/// edge `iedge` of direction `ixy`, reading it into the least recently used
/// buffer if it is not present.
pub fn edgeswap(
    bv: &mut BlendVars,
    units: &mut BlendUnits,
    iedge: i32,
    ixy: i32,
    indbuf: &mut i32,
) {
    let ibe =
        |ied: i32, ixy: i32, ext: &[usize; 2]| (ied - 1) as usize + ext[0] * (ixy - 1) as usize;
    //
    *indbuf = bv.ibuf_edge[ibe(iedge, ixy, &bv.ibuf_edge_ext)];
    if *indbuf == 0 {
        //
        // find oldest
        //
        let mut minused = bv.jus_edg_ct + 1;
        // `ioldest` is undefined in the source if no buffer qualifies, which
        // cannot happen: every `lasEdgUse` is at most `jusEdgCt`.
        let mut ioldest = 0i32;
        for i in 1..=LIM_EDG_BF {
            if minused > bv.las_edg_use[(i - 1) as usize] {
                minused = bv.las_edg_use[(i - 1) as usize];
                ioldest = i;
            }
        }
        let io = (ioldest - 1) as usize;
        //
        // mark oldest as no longer present, mark this as present
        //
        if bv.iedg_bf_list[io] > 0 {
            let k = ibe(bv.iedg_bf_list[io], bv.ixy_bf_list[io], &bv.ibuf_edge_ext);
            bv.ibuf_edge[k] = 0;
        }
        bv.iedg_bf_list[io] = iedge;
        bv.ixy_bf_list[io] = ixy;
        *indbuf = ioldest;
        let k = ibe(iedge, ixy, &bv.ibuf_edge_ext);
        bv.ibuf_edge[k] = *indbuf;
        //
        // read edge into  buffer
        //
        read_edge_func(bv, units, iedge, ixy, ioldest);
    }
    //
    // mark this one as used most recently
    //
    bv.jus_edg_ct += 1;
    bv.las_edg_use[(*indbuf - 1) as usize] = bv.jus_edg_ct;
}

/// Original: `subroutine readEdgeFunc(iedge, ixy, indbuf)`
/// (`bsubs.f90:744`).
///
/// Reads edge function `iedge`, direction `ixy`, into buffer `indbuf`,
/// swapping bytes as necessary, and adjusts it for the edge density deltas
/// when density scaling is applied.
///
/// For a skipped edge the source zeroes the scratch grids `dxgrid`,
/// `dygrid`, `ddengrid` — not the buffer just read (`bsubs.f90:777-781`);
/// reproduced.
pub fn read_edge_func(
    bv: &mut BlendVars,
    units: &mut BlendUnits,
    iedge: i32,
    ixy: i32,
    indbuf: i32,
) {
    let ib = (indbuf - 1) as usize;
    let gbf = |ix: i32, iy: i32, ext: &[usize; 3]| {
        (ix - 1) as usize + ext[0] * ((iy - 1) as usize + ext[1] * ib)
    };
    let unit = units.edge[(ixy - 1) as usize]
        .as_ref()
        .expect("blendmont connects unit iunEdge(ixy) before any edge function is read");
    let word = |rec: &[u8], k: usize| -> [u8; 4] { rec[4 * k..4 * k + 4].try_into().unwrap() };
    let nxgr: i32;
    let nygr: i32;
    if bv.need_byte_swap == 0 {
        let rec = unit
            .read_record(1 + iedge)
            .unwrap_or_else(|e| unit.runtime_error(e));
        nxgr = i32::from_ne_bytes(word(&rec, 0));
        nygr = i32::from_ne_bytes(word(&rec, 1));
        bv.ix_grd_st_bf[ib] = i32::from_ne_bytes(word(&rec, 2));
        bv.ix_ofs_bf[ib] = i32::from_ne_bytes(word(&rec, 3));
        bv.iy_grd_st_bf[ib] = i32::from_ne_bytes(word(&rec, 4));
        bv.iy_ofs_bf[ib] = i32::from_ne_bytes(word(&rec, 5));
        let mut k = 6usize;
        for iy in 1..=nygr {
            for ix in 1..=nxgr {
                bv.dx_gr_bf[gbf(ix, iy, &bv.dx_gr_bf_ext)] = f32::from_ne_bytes(word(&rec, k));
                bv.dy_gr_bf[gbf(ix, iy, &bv.dy_gr_bf_ext)] = f32::from_ne_bytes(word(&rec, k + 1));
                bv.dden_gr_bf[gbf(ix, iy, &bv.dden_gr_bf_ext)] =
                    f32::from_ne_bytes(word(&rec, k + 2));
                k += 3;
            }
        }
    } else {
        let rec = unit
            .read_record(1 + iedge)
            .unwrap_or_else(|e| unit.runtime_error(e));
        let mut b = word(&rec, 0);
        convert_longs(&mut b, 1);
        nxgr = i32::from_ne_bytes(b);
        let mut b = word(&rec, 1);
        convert_longs(&mut b, 1);
        nygr = i32::from_ne_bytes(b);
        let rec = unit
            .read_record(1 + iedge)
            .unwrap_or_else(|e| unit.runtime_error(e));
        bv.ix_grd_st_bf[ib] = i32::from_ne_bytes(word(&rec, 2));
        bv.ix_ofs_bf[ib] = i32::from_ne_bytes(word(&rec, 3));
        bv.iy_grd_st_bf[ib] = i32::from_ne_bytes(word(&rec, 4));
        bv.iy_ofs_bf[ib] = i32::from_ne_bytes(word(&rec, 5));
        let mut k = 6usize;
        for iy in 1..=nygr {
            for ix in 1..=nxgr {
                bv.dx_gr_bf[gbf(ix, iy, &bv.dx_gr_bf_ext)] = f32::from_ne_bytes(word(&rec, k));
                bv.dy_gr_bf[gbf(ix, iy, &bv.dy_gr_bf_ext)] = f32::from_ne_bytes(word(&rec, k + 1));
                bv.dden_gr_bf[gbf(ix, iy, &bv.dden_gr_bf_ext)] =
                    f32::from_ne_bytes(word(&rec, k + 2));
                k += 3;
            }
        }
        for v in [
            &mut bv.ix_grd_st_bf[ib],
            &mut bv.ix_ofs_bf[ib],
            &mut bv.iy_grd_st_bf[ib],
            &mut bv.iy_ofs_bf[ib],
        ] {
            let mut b = v.to_ne_bytes();
            convert_longs(&mut b, 1);
            *v = i32::from_ne_bytes(b);
        }
        for iy in 1..=nygr {
            for (grid, ext) in [
                (&mut bv.dx_gr_bf, bv.dx_gr_bf_ext),
                (&mut bv.dy_gr_bf, bv.dy_gr_bf_ext),
                (&mut bv.dden_gr_bf, bv.dden_gr_bf_ext),
            ] {
                let start = gbf(1, iy, &ext);
                let row = &mut grid[start..start + nxgr as usize];
                // `convert_floats(dxgrbf(1, iy, indbuf), nxgr)`: the C takes
                // the bytes of the caller's array.
                // SAFETY: `u8` has alignment 1 and every bit pattern is valid
                // for both `u8` and `f32`; the byte view covers exactly `row`.
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(row.as_mut_ptr().cast::<u8>(), row.len() * 4)
                };
                convert_floats(bytes, nxgr);
            }
        }
    }
    if bv.if_skip_edge[(iedge - 1) as usize + bv.if_skip_edge_ext[0] * (ixy - 1) as usize] > 0 {
        for grid_ext in [
            (&mut bv.dx_grid, bv.dx_grid_ext),
            (&mut bv.dy_grid, bv.dy_grid_ext),
            (&mut bv.dden_grid, bv.dden_grid_ext),
        ] {
            let (grid, ext) = grid_ext;
            for j in 1..=nygr {
                for i in 1..=nxgr {
                    grid[(i - 1) as usize + ext[0] * (j - 1) as usize] = 0.;
                }
            }
        }
    }
    bv.nx_gr_bf[ib] = nxgr;
    bv.ny_gr_bf[ib] = nygr;
    bv.int_xgr_bf[ib] = bv.int_gr_copy[(ixy - 1) as usize];
    bv.int_ygr_bf[ib] = bv.int_gr_copy[(3 - ixy - 1) as usize];

    //
    // If we have a density scaling, now adjust the edge function for the delta values
    // in the corresponding edge density function
    if bv.i_apply_dens_scaling > 0
        && bv.if_skip_edge[(iedge - 1) as usize + bv.if_skip_edge_ext[0] * (ixy - 1) as usize] == 0
    {
        let ind_den = (ixy - 1) * bv.ix_dim_den_buf + iedge - bv.iedge_cur_base[(ixy - 1) as usize];
        let idn = (ind_den - 1) as usize;
        let nx_den = bv.nx_den_buf[idn];
        let dd = |ind_val: i32, ext: &[usize; 2]| (ind_val - 1) as usize + ext[0] * idn;
        for ixgr in 1..=nxgr {
            let ix = bv.ix_grd_st_bf[ib] + (ixgr - 1) * bv.int_xgr_bf[ib];
            let mut x_in_den = (ix - bv.ix_den_start[idn]) as f32
                / bv.interval_den[(ixy - 1) as usize] as f32
                + 1.;
            let lim = nx_den as f32 - 0.01;
            let inner = if lim < x_in_den { lim } else { x_in_den };
            x_in_den = if 1. > inner { 1. } else { inner };
            let ix_den = x_in_den as i32;
            let fx_den = x_in_den - ix_den as f32;
            for iygr in 1..=nygr {
                let iy = bv.iy_grd_st_bf[ib] + (iygr - 1) * bv.int_ygr_bf[ib];
                let mut y_in_den = (iy - bv.iy_den_start[idn]) as f32
                    / bv.interval_den[(3 - ixy - 1) as usize] as f32
                    + 1.;
                let lim = bv.ny_den_buf[idn] as f32 - 0.01;
                let inner = if lim < y_in_den { lim } else { y_in_den };
                y_in_den = if 1. > inner { 1. } else { inner };
                let iy_den = y_in_den as i32;
                let fy_den = y_in_den - iy_den as f32;
                let ind_val = (iy_den - 1) * nx_den + ix_den;
                let e = bv.delta_den_buf_ext;
                let del = &bv.delta_den_buf;
                // With a one-row (or one-column) density grid the clamps give
                // `fyDen = 0` (`fxDen = 0`), and the neighbour the source still
                // reads, weighted by 0, lies past this edge's samples: in the
                // next column of `deltaDenBuf`, or for the last edge past the
                // whole array (heap bytes; `BUGS.md`).  An element outside the
                // array is read as 0 here.
                let at = |k: usize| del.get(k).copied().unwrap_or(0.0);
                let k = gbf(ixgr, iygr, &bv.dden_gr_bf_ext);
                bv.dden_gr_bf[k] = bv.dden_gr_bf[k]
                    + (1. - fy_den)
                        * ((1. - fx_den) * at(dd(ind_val, &e)) + fx_den * at(dd(ind_val + 1, &e)))
                    + fy_den
                        * ((1. - fx_den) * at(dd(ind_val + nx_den, &e))
                            + fx_den * at(dd(ind_val + 1 + nx_den, &e)));
            }
        }
    }
}

/// Original: `subroutine doedge(iedge, ixy, edgedone, sdcrit, devcrit, nfit,
/// norder, nskip, docross, xcreadin, xclegacy, useExpected, edgedispx,
/// edgedispy, idimedge)` (`bsubs.f90:832`).
///
/// "Does" edge `iedge` of direction `ixy`: if the edge is on a joint between
/// negatives, it composes the list of all edges along that joint and finds
/// their edge functions from the center outward; each function is found,
/// smoothed, and written to the edge file.  `edgedone`, `edgedispx`,
/// `edgedispy` are `(idimedge, 2)` column-major.
///
/// With `needByteSwap` set, the source swaps `dxgrid`/`dygrid`/`ddengrid`
/// in place to write them (`bsubs.f90:1208-1212`), so the next edge of a
/// joint derives its rotation from byte-swapped grids; reproduced.
pub fn doedge(
    bv: &mut BlendVars,
    units: &mut BlendUnits,
    iedge: i32,
    ixy: i32,
    edgedone: &mut [bool],
    sdcrit: f32,
    devcrit: f32,
    nfit: &[i32],
    norder: i32,
    nskip: &[i32],
    docross: bool,
    xcreadin: bool,
    xclegacy: bool,
    use_expected: bool,
    edgedispx: &mut [f32],
    edgedispy: &mut [f32],
    idimedge: i32,
) {
    const LIMPNEG: usize = 20;
    let mut multcoord = [0i32; LIMPNEG];
    let mut multedge = [0i32; LIMPNEG];
    let mut multmp = [0i32; LIMPNEG];
    let mut mcotmp = [0i32; LIMPNEG];
    let mut igrstr = [0i32; 2];
    let mut igrofs = [0i32; 2];
    let ed = |jedge: i32, ixy: i32| ((jedge - 1) + idimedge * (ixy - 1)) as usize;
    let e2 = |i: i32, ixy: i32, ext: &[usize; 2]| (i - 1) as usize + ext[0] * (ixy - 1) as usize;
    let ixyu = (ixy - 1) as usize;
    let iyxu = (3 - ixy - 1) as usize;
    // Locals the source reads only after setting them on every path that
    // reaches the read (`thetamid`/`dxmid`/`dymid` at `imult == middone + 1`
    // need `middone >= 2` there, which holds whenever `nmult >= 2`).
    let (mut ixdisp, mut iydisp) = (0i32, 0i32);
    let (mut ixdispmid, mut iydispmid) = (0i32, 0i32);
    let (mut lastedge, mut lastxdisp, mut lastydisp) = (0i32, 0i32, 0i32);
    let (mut nxgr, mut nygr) = (0i32, 0i32);
    let (mut xdisp, mut ydisp) = (0.0f32, 0.0f32);
    let (mut theta, mut edgedx, mut edgedy) = (0.0f32, 0.0f32, 0.0f32);
    let (mut thetamid, mut dxmid, mut dymid) = (0.0f32, 0.0f32, 0.0f32);
    let mut del_indent = [0.0f32; 2];
    let mut indent_use = [0i32; 2];
    let mut middone = 0i32;
    let (mut dmin, mut dsum) = (0.0f32, 0.0f32);
    // gfortran `Fw.d` editing for the patch dumps: a value too wide for the
    // field is `w` asterisks (as in `xfsubs/xfwrite.rs`).
    let f_edit = |value: f32, w: usize, d: usize| -> String {
        let text = format!("{value:>w$.d$}");
        if text.len() > w { "*".repeat(w) } else { text }
    };
    //
    // make list of edges to be done
    //
    let intxgrid = bv.int_grid[ixyu];
    let intygrid = bv.int_grid[iyxu];
    let mut nmult = 1i32;
    multedge[0] = iedge;
    let mut intscan = 6i32;
    let ipclo = bv.ipiece_lower[e2(iedge, ixy, &bv.ipiece_lower_ext)];
    let ipcup = bv.ipiece_upper[e2(iedge, ixy, &bv.ipiece_upper_ext)];
    bv.ipc_below_edge = ipclo;
    if bv.neg_list[(ipclo - 1) as usize] != bv.neg_list[(ipcup - 1) as usize] {
        //
        // if edge is across a negative boundary, need to look for all such
        // edges and add them to list
        //
        nmult = 0;
        intscan = 9;
        for i in 1..=bv.nedge[ixyu] {
            let ipc = bv.ipiece_lower[e2(i, ixy, &bv.ipiece_lower_ext)];
            if bv.iz_pc_list[(ipclo - 1) as usize] == bv.iz_pc_list[(ipc - 1) as usize]
                && bv.neg_list[(ipclo - 1) as usize] == bv.neg_list[(ipc - 1) as usize]
                && bv.neg_list[(ipcup - 1) as usize]
                    == bv.neg_list[(bv.ipiece_upper[e2(i, ixy, &bv.ipiece_upper_ext)] - 1) as usize]
            {
                nmult += 1;
                multedge[(nmult - 1) as usize] = i;
                // get coordinate of edge in ortho direction
                let mltco = if ixy == 1 {
                    bv.iy_pc_list[(ipc - 1) as usize]
                } else {
                    bv.ix_pc_list[(ipc - 1) as usize]
                };
                multcoord[(nmult - 1) as usize] = mltco;
            }
        }
        //
        // order list to go out from center of edge.  GROSS.. first order it
        //
        for i in 1..=nmult - 1 {
            for j in i..=nmult {
                let (iu, ju) = ((i - 1) as usize, (j - 1) as usize);
                if multcoord[iu] > multcoord[ju] {
                    multcoord.swap(iu, ju);
                    multedge.swap(iu, ju);
                }
            }
        }
        let midcoord = (multcoord[(nmult - 1) as usize] + multcoord[0]) / 2;
        //
        // find element closest to center
        //
        let mut mindiff = 100000i32;
        let mut imid = 0i32;
        for i in 1..=nmult {
            let idiff = (multcoord[(i - 1) as usize] - midcoord).abs();
            if idiff < mindiff {
                mindiff = idiff;
                imid = i;
            }
        }
        //
        // set up order from there to top then back from middle to bottom
        //
        let mut imult = 0i32;
        for i in imid..=nmult {
            imult += 1;
            multmp[(imult - 1) as usize] = multedge[(i - 1) as usize];
            mcotmp[(imult - 1) as usize] = multcoord[(i - 1) as usize];
        }
        middone = imult;
        for i in (1..=imid - 1).rev() {
            imult += 1;
            multmp[(imult - 1) as usize] = multedge[(i - 1) as usize];
            mcotmp[(imult - 1) as usize] = multcoord[(i - 1) as usize];
        }
        for i in 0..nmult as usize {
            multedge[i] = multmp[i];
            multcoord[i] = mcotmp[i];
        }
    }
    //
    // finally ready to set up to get edge
    //
    for imult in 1..=nmult {
        let jedge = multedge[(imult - 1) as usize];
        let jlow = bv.ipiece_lower[e2(jedge, ixy, &bv.ipiece_lower_ext)];
        let jup = bv.ipiece_upper[e2(jedge, ixy, &bv.ipiece_upper_ext)];
        let jskip = e2(jedge, ixy, &bv.if_skip_edge_ext);
        //
        let mut indlow = 0i32;
        let mut indup = 0i32;
        shuffler(bv, jlow, &mut indlow);
        shuffler(bv, jup, &mut indup);
        //
        if imult == 1 {
            //
            // for first time, set these parameters to
            // their default values with 0 offsets
            //
            ixdisp = 0;
            iydisp = 0;
            if docross {
                if xcreadin || use_expected {
                    xdisp = edgedispx[ed(jedge, ixy)];
                    ydisp = edgedispy[ed(jedge, ixy)];
                }
                if !xcreadin {
                    get_extra_indents(bv, jlow, jup, ixy, &mut del_indent);
                    let mut indent_xcorr = 2i32;
                    if del_indent[ixyu] > 0. && bv.ifill_treatment == 1 {
                        indent_xcorr = del_indent[ixyu] as i32 + 2;
                    }
                    if bv.if_skip_edge[jskip] > 0 {
                        xdisp = 0.;
                        ydisp = 0.;
                    } else {
                        // `xcorrEdge(array(indlow), array(indup), ...)`:
                        // `xcorrEdge` does not touch `array`, so it is lent
                        // out of the module for the call.
                        let array = std::mem::take(&mut bv.array);
                        xcorr_edge(
                            bv,
                            &array,
                            indlow,
                            indup,
                            ixy,
                            &mut xdisp,
                            &mut ydisp,
                            xclegacy,
                            use_expected,
                            indent_xcorr,
                        );
                        bv.array = array;

                        // Get the new items available: maxSD values all into one array for
                        // outlier analysis, and second and third peak locations
                        let k = bv.num_max_sds as usize;
                        bv.trimmed_max_sds[k] = mont_xc_get_last_trimmed_max_sd() as f32;
                        if bv.trimmed_max_sds[k] > 0. {
                            bv.num_max_sds += 1;
                            let n = (bv.num_max_sds - 1) as usize;
                            bv.max_sd_to_edge_num[n] = jedge;
                            bv.max_sd_to_ixy_of_edge[n] = ixy;
                        }
                        if bv.num_xcorr_peaks > 1 {
                            let ix = (jedge - 1) * 4 + 4 * bv.lim_edge * (ixy - 1) + 1;
                            montxcgetlastrunnersup(&mut bv.altern_disps[(ix - 1) as usize..], &2);
                        }
                    }
                    edgedispx[ed(jedge, ixy)] = xdisp;
                    edgedispy[ed(jedge, ixy)] = ydisp;
                }
                ixdisp = xdisp.round() as i32;
                iydisp = ydisp.round() as i32;
            }
            ixdispmid = ixdisp;
            iydispmid = iydisp;
            //
        } else {
            if imult == middone + 1 {
                //
                theta = thetamid; //at midway point, restore
                edgedx = dxmid; //values from first (middle)
                edgedy = dymid; //edge
                lastedge = multedge[0];
                lastxdisp = ixdispmid;
                lastydisp = iydispmid;
            } else {
                edge_to_rotrans(
                    &bv.dx_grid,
                    &bv.dy_grid,
                    bv.ixg_dim,
                    bv.iyg_dim,
                    nxgr,
                    nygr,
                    intxgrid,
                    intygrid,
                    &mut theta,
                    &mut edgedx,
                    &mut edgedy,
                );
                if imult == 2 {
                    thetamid = theta; //if that was first edge, save
                    dxmid = edgedx; //the value
                    dymid = edgedy;
                }
                lastedge = multedge[(imult - 1 - 1) as usize];
            }
            //
            // find displacement of center of next edge relative to center of
            // last edge.  First get x/y displacements between edges
            //
            let up_j = (jup - 1) as usize;
            let up_last = (bv.ipiece_upper[e2(lastedge, ixy, &bv.ipiece_upper_ext)] - 1) as usize;
            let xdispl = (bv.ix_pc_list[up_j] - bv.ix_pc_list[up_last]) as f32;
            let ydispl = (bv.iy_pc_list[up_j] - bv.iy_pc_list[up_last]) as f32;
            let costh = gfortran_cosd_r4(theta);
            let sinth = gfortran_sind_r4(theta);
            // rotate vector by theta and displace by dx, dy; the movement in
            // the tip of the displacement vector is the expected relative
            // displacement between this frame and the last
            let xrel = xdispl * costh - ydispl * sinth + edgedx - xdispl;
            let yrel = xdispl * sinth + ydispl * costh + edgedy - ydispl;
            // add pixel displacement of last frame to get total expected pixel
            // displacement of this frame
            ixdisp = xrel.round() as i32 + lastxdisp;
            iydisp = yrel.round() as i32 + lastydisp;
        }
        //
        // Determine extra indentation if distortion corrections
        //
        get_extra_indents(bv, jlow, jup, ixy, &mut del_indent);
        //
        indent_use[0] = bv.indent[0] + del_indent[0].round() as i32;
        indent_use[1] = bv.indent[1] + del_indent[1].round() as i32;
        //
        // Determine data limits for edge in long dimension if flag set
        let mut limit_lo = 0i32;
        let mut limit_hi = 0i32;
        if bv.limit_data {
            let (mut limit_lo2, mut limit_hi2) = (0i32, 0i32);
            get_data_limits(bv, jlow, 3 - ixy, 2, &mut limit_lo, &mut limit_hi);
            get_data_limits(bv, jup, 3 - ixy, 1, &mut limit_lo2, &mut limit_hi2);
            limit_lo = limit_lo.max(limit_lo2);
            limit_hi = limit_hi.min(limit_hi2);
        }
        //
        setgridchars(
            &bv.nxyz_in,
            &bv.n_overlap,
            &bv.ibox_siz,
            &indent_use,
            &bv.int_grid,
            ixy,
            ixdisp,
            iydisp,
            limit_lo,
            limit_hi,
            &mut nxgr,
            &mut nygr,
            &mut igrstr,
            &mut igrofs,
        );
        if nxgr > bv.nx_grid[ixyu] || nygr > bv.ny_grid[ixyu] {
            exit_error("One grid has more points than originally expected");
        }
        lastxdisp = ixdisp;
        lastydisp = iydisp;
        //
        let (ixg_dim, iyg_dim) = (bv.ixg_dim, bv.iyg_dim);
        let g = |ix: i32, iy: i32| ((ix - 1) + ixg_dim * (iy - 1)) as usize;
        if bv.if_skip_edge[jskip] == 0 {
            let _wallstart = walltime();
            findedgefunc(
                &bv.array[(indlow - 1) as usize..],
                &bv.array[(indup - 1) as usize..],
                bv.nxyz_in[0],
                bv.nxyz_in[1],
                igrstr[0],
                igrstr[1],
                igrofs[0],
                igrofs[1],
                &mut nxgr,
                &mut nygr,
                intxgrid,
                intygrid,
                bv.ibox_siz[ixyu],
                bv.ibox_siz[iyxu],
                intscan,
                &mut bv.dx_grid,
                &mut bv.dy_grid,
                &mut bv.sd_grid,
                &mut bv.dden_grid,
                ixg_dim,
                iyg_dim,
            );
            //
            if bv.iz_unsmoothed_patch >= 0 {
                let unit10 = units
                    .unit10
                    .as_mut()
                    .expect("blendmont connects unit 10 when izUnsmoothedPatch >= 0");
                for iy in 1..=nygr {
                    for ix in 1..=nxgr {
                        let _ = writeln!(
                            unit10,
                            "{:6}{:6}{:6}{}{}{}{}",
                            igrstr[0] + (ix - 1) * intxgrid,
                            igrstr[1] + (iy - 1) * intygrid,
                            bv.iz_unsmoothed_patch,
                            f_edit(bv.dx_grid[g(ix, iy)], 9, 2),
                            f_edit(bv.dy_grid[g(ix, iy)], 9, 2),
                            f_edit(0., 9, 2),
                            f_edit(bv.sd_grid[g(ix, iy)], 12, 4)
                        );
                    }
                }
                bv.iz_unsmoothed_patch += 1;
            }
            smoothgrid(
                &mut bv.dx_grid,
                &mut bv.dy_grid,
                &bv.sd_grid,
                &mut bv.dden_grid,
                ixg_dim,
                iyg_dim,
                nxgr,
                nygr,
                sdcrit,
                devcrit,
                nfit[ixyu],
                nfit[iyxu],
                norder,
                nskip[ixyu],
                nskip[iyxu],
            );
            //write(*,'(a,f10.6)') 'Edge function time', walltime() -wallstart
            if bv.iz_smoothed_patch >= 0 {
                let unit11 = units
                    .unit11
                    .as_mut()
                    .expect("blendmont connects unit 11 when izSmoothedPatch >= 0");
                for iy in 1..=nygr {
                    for ix in 1..=nxgr {
                        let _ = writeln!(
                            unit11,
                            "{:6}{:6}{:6}{}{}{}{}",
                            igrstr[0] + (ix - 1) * intxgrid,
                            igrstr[1] + (iy - 1) * intygrid,
                            bv.iz_smoothed_patch,
                            f_edit(bv.dx_grid[g(ix, iy)], 9, 2),
                            f_edit(bv.dy_grid[g(ix, iy)], 9, 2),
                            f_edit(0., 9, 2),
                            f_edit(bv.sd_grid[g(ix, iy)], 12, 4)
                        );
                    }
                }
                bv.iz_smoothed_patch += 1;
            }
            //
            // Compute density means
            if bv.i_den_sample > 0 {
                // Set up all the parameters
                let ind_buf = (ixy - 1) * bv.ix_dim_den_buf + jedge - bv.iedge_cur_base[ixyu];
                let ibu = (ind_buf - 1) as usize;
                bv.nx_den_buf[ibu] = 1.max(nxgr / bv.i_den_sample);
                bv.ny_den_buf[ibu] = 1.max(nygr / bv.i_den_sample);
                let num_x_sample = nxgr.min(bv.i_den_sample);
                let num_y_sample = nygr.min(bv.i_den_sample);
                let nx_skip = (nxgr % num_x_sample) / 2;
                let ny_skip = (nygr % num_y_sample) / 2;
                let nx_box = bv.ibox_siz[ixyu] + (num_x_sample - 1) * intxgrid;
                let ny_box = bv.ibox_siz[iyxu] + (num_y_sample - 1) * intygrid;
                bv.ix_den_start[ibu] = (igrstr[0] as f32
                    + ((num_x_sample - 1) as f32 / 2. + nx_skip as f32) * intxgrid as f32)
                    as i32;
                bv.iy_den_start[ibu] = (igrstr[1] as f32
                    + ((num_y_sample - 1) as f32 / 2. + ny_skip as f32) * intygrid as f32)
                    as i32;
                // These are actuals offsets to get from one coordinate to another, thus differences
                bv.ix_den_offset[ibu] = igrofs[0] - igrstr[0];
                bv.iy_den_offset[ibu] = igrofs[1] - igrstr[1];
                let ea = bv.den_abuf_ext[0];
                let eb = bv.den_bbuf_ext[0];
                //
                // Loop on the boxes, fill the buffers
                for iy_den in 1..=bv.ny_den_buf[ibu] {
                    for ix_den in 1..=bv.nx_den_buf[ibu] {
                        //
                        // First get the mean in the box in A
                        let ix = bv.ix_den_start[ibu] + (ix_den - 1) * bv.interval_den[ixyu]
                            - nx_box / 2;
                        let iy = bv.iy_den_start[ibu] + (iy_den - 1) * bv.interval_den[iyxu]
                            - ny_box / 2;
                        let ind_val = (iy_den - 1) * bv.nx_den_buf[ibu] + ix_den;
                        let ka = (ind_val - 1) as usize + ea * ibu;
                        array_min_max_mean_fortran(
                            &bv.array[(indlow - 1) as usize..],
                            &bv.nxyz_in[0],
                            &bv.nxyz_in[1],
                            &ix,
                            &(ix + nx_box - 1),
                            &iy,
                            &(iy + ny_box - 1),
                            &mut dmin,
                            &mut dsum,
                            &mut bv.den_abuf[ka],
                        );
                        //
                        // Now average the corresponding ddengrid values to get implied mean in B
                        let ix = nx_skip + (ix_den - 1) * bv.i_den_sample;
                        let iy = ny_skip + (iy_den - 1) * bv.i_den_sample;
                        dsum = 0.;
                        for j in iy + 1..=iy + num_y_sample {
                            for i in ix + 1..=ix + num_x_sample {
                                dsum += bv.dden_grid[g(i, j)];
                            }
                        }
                        bv.den_bbuf[(ind_val - 1) as usize + eb * ibu] =
                            bv.den_abuf[ka] + dsum / (num_x_sample * num_y_sample) as f32;
                    }
                }
                //
                // Write the edge densities
                let n = (bv.nx_den_buf[ibu] * bv.ny_den_buf[ibu]) as usize;
                let mut record = Vec::with_capacity(24 + 8 * n);
                for v in [
                    bv.nx_den_buf[ibu],
                    bv.ny_den_buf[ibu],
                    bv.ix_den_start[ibu],
                    bv.iy_den_start[ibu],
                    bv.ix_den_offset[ibu],
                    bv.iy_den_offset[ibu],
                ] {
                    record.extend_from_slice(&v.to_ne_bytes());
                }
                for i in 0..n {
                    record.extend_from_slice(&bv.den_abuf[i + ea * ibu].to_ne_bytes());
                    record.extend_from_slice(&bv.den_bbuf[i + eb * ibu].to_ne_bytes());
                }
                let unit = units.dens[ixyu]
                    .as_ref()
                    .expect("blendmont connects unit iunDens(ixy) when iDenSample > 0");
                unit.write_record(jedge + 1, &record)
                    .unwrap_or_else(|e| unit.runtime_error(e));
            }
        } else {
            for iy in 1..=nygr {
                for ix in 1..=nxgr {
                    bv.dx_grid[g(ix, iy)] = 0.;
                    bv.dy_grid[g(ix, iy)] = 0.;
                    bv.dden_grid[g(ix, iy)] = 0.;
                }
            }
        }
        //
        let unit = units.edge[ixyu]
            .as_ref()
            .expect("blendmont connects unit iunEdge(ixy) before edges are done");
        let mut record = Vec::with_capacity(24 + 12 * (nxgr * nygr).max(0) as usize);
        if bv.need_byte_swap == 0 {
            for v in [nxgr, nygr, igrstr[0], igrofs[0], igrstr[1], igrofs[1]] {
                record.extend_from_slice(&v.to_ne_bytes());
            }
        } else {
            //
            // In case there were incomplete edges, be able to write swapped
            ixdisp = nxgr;
            iydisp = nygr;
            let mut b = ixdisp.to_ne_bytes();
            convert_longs(&mut b, 1);
            ixdisp = i32::from_ne_bytes(b);
            let mut b = iydisp.to_ne_bytes();
            convert_longs(&mut b, 1);
            iydisp = i32::from_ne_bytes(b);
            for arr in [&mut igrstr, &mut igrofs] {
                let mut b = [0u8; 8];
                b[..4].copy_from_slice(&arr[0].to_ne_bytes());
                b[4..].copy_from_slice(&arr[1].to_ne_bytes());
                convert_longs(&mut b, 2);
                arr[0] = i32::from_ne_bytes(b[..4].try_into().unwrap());
                arr[1] = i32::from_ne_bytes(b[4..].try_into().unwrap());
            }
            for iy in 1..=nygr {
                for grid in [&mut bv.dx_grid, &mut bv.dy_grid, &mut bv.dden_grid] {
                    let start = g(1, iy);
                    let row = &mut grid[start..start + nxgr as usize];
                    // SAFETY: `u8` has alignment 1 and every bit pattern is
                    // valid for both `u8` and `f32`; the view covers `row`.
                    let bytes = unsafe {
                        std::slice::from_raw_parts_mut(row.as_mut_ptr().cast::<u8>(), row.len() * 4)
                    };
                    convert_floats(bytes, nxgr);
                }
            }
            for v in [ixdisp, iydisp, igrstr[0], igrofs[0], igrstr[1], igrofs[1]] {
                record.extend_from_slice(&v.to_ne_bytes());
            }
        }
        for iy in 1..=nygr {
            for ix in 1..=nxgr {
                record.extend_from_slice(&bv.dx_grid[g(ix, iy)].to_ne_bytes());
                record.extend_from_slice(&bv.dy_grid[g(ix, iy)].to_ne_bytes());
                record.extend_from_slice(&bv.dden_grid[g(ix, iy)].to_ne_bytes());
            }
        }
        unit.write_record(jedge + 1, &record)
            .unwrap_or_else(|e| unit.runtime_error(e));
        //
        edgedone[ed(jedge, ixy)] = true;
        //
        // Write records for any edges not done yet
        ixdisp = -1;
        if bv.need_byte_swap != 0 {
            let mut b = ixdisp.to_ne_bytes();
            convert_longs(&mut b, 1);
            ixdisp = i32::from_ne_bytes(b);
        }
        for ix in bv.last_written[ixyu] + 1..=jedge - 1 {
            if !edgedone[ed(ix, ixy)] {
                let mut record = [0u8; 8];
                record[..4].copy_from_slice(&ixdisp.to_ne_bytes());
                record[4..].copy_from_slice(&ixdisp.to_ne_bytes());
                unit.write_record(ix + 1, &record)
                    .unwrap_or_else(|e| unit.runtime_error(e));
            }
        }
        bv.last_written[ixyu] = jedge;
    }
}

/// Original: `subroutine lincom_rotrans(r1, w1, r2, w2, s)`
/// (`bsubs.f90:1238`).
///
/// Forms the linear combination of rotation/translations `r1` and `r2` with
/// weights `w1`, `w2`, shifted to the center of `r2`, into `s`.  A caller
/// whose `s` is the same array as `r2` passes a copy of `r2` (module note).
pub fn lincom_rotrans(r1: &[f32], w1: f32, r2: &[f32], w2: f32, s: &mut [f32]) {
    let mut rt1 = [0.0f32; 6];
    let mut rt2 = [0.0f32; 6];
    //
    recen_rotrans(r1, r2[3], r2[4], &mut rt1);
    xfcopy(r2, &mut rt2);
    rt2[0] = w1 * rt1[0] + w2 * r2[0];
    rt2[1] = w1 * rt1[1] + w2 * r2[1];
    rt2[2] = w1 * rt1[2] + w2 * r2[2];
    xfcopy(&rt2, s);
}

/// Original: `subroutine recen_rotrans(r, xcenew, ycenew, s)`
/// (`bsubs.f90:1259`).
///
/// Shifts rotation/translation `r` to the new center (`xcenew`, `ycenew`)
/// into `s`.  No `implicit none`: `sinth`, `cosm1` are real.
pub fn recen_rotrans(r: &[f32], xcenew: f32, ycenew: f32, s: &mut [f32]) {
    let sinth = gfortran_sind_r4(r[0]);
    let cosm1 = gfortran_cosd_r4(r[0]) - 1.;
    s[0] = r[0];
    s[1] = r[1] + cosm1 * (xcenew - r[3]) - sinth * (ycenew - r[4]);
    s[2] = r[2] + cosm1 * (ycenew - r[4]) + sinth * (xcenew - r[3]);
    s[3] = xcenew;
    s[4] = ycenew;
}

/// Original: `subroutine countedges(indx, indy, xg, yg, useEdges)`
/// (`bsubs.f90:1293`).
///
/// Converts output coordinate (`indx`, `indy`) to (`xg`, `yg`) with the
/// inverse of the optional g transform, and analyzes the pieces and edges
/// the point is in or near, into `numPieces`, `inPiece`, `numEdges`,
/// `inEdge` and related module arrays.
///
/// The `maxInside` of the y branch of the most-interior search is computed
/// with `nyin - 1 - xinpiece(i)` (`bsubs.f90:1595`); reproduced.
pub fn countedges(
    bv: &mut BlendVars,
    indx: i32,
    indy: i32,
    xg: &mut f32,
    yg: &mut f32,
    use_edges: bool,
) {
    let mut needcheck = [[false; MAX_IN_PC as usize]; 2];
    let mut xycur = [0.0f32; 2];
    let mut in_limit = [false; 2];
    let mut moved_piece = [0i32; 4];
    let mut ipc_cross = 0i32;
    let (mut xpc_cross, mut ypc_cross) = (0.0f32, 0.0f32);
    let (mut xtmp, mut ytmp) = (0.0f32, 0.0f32);
    let (mut xbak, mut ybak) = (0.0f32, 0.0f32);
    let (mut limit_lo, mut limit_hi) = (0i32, 0i32);
    let mut newpiece = 0i32;
    let mut axisin = 0i32;
    let nxin = bv.nxyz_in[0];
    let nyin = bv.nxyz_in[1];
    let n_xoverlap = bv.n_overlap[0];
    let ny_overlap = bv.n_overlap[1];
    let e2 = |i: i32, ixy: i32, ext: &[usize; 2]| (i - 1) as usize + ext[0] * (ixy - 1) as usize;
    let mp = |bv: &BlendVars, ix: i32, iy: i32| {
        bv.map_piece[(ix - 1) as usize + bv.map_piece_ext[0] * (iy - 1) as usize]
    };
    //
    bv.num_pieces = 0;
    bv.num_edges[0] = 0;
    bv.num_edges[1] = 0;
    let mut id_search = 1i32;
    //
    // get frame # that it is nominally in: actual one or nearby valid frame
    //
    *xg = indx as f32;
    *yg = indy as f32;
    if bv.sec_has_warp {
        interpolate_grid(
            *xg - 0.5,
            *yg - 0.5,
            &bv.warp_dx,
            &bv.warp_dy,
            bv.lm_warp_x,
            bv.nx_warp,
            bv.ny_warp,
            bv.x_warp_strt,
            bv.y_warp_strt,
            bv.x_warp_intrv,
            bv.y_warp_intrv,
            &mut xtmp,
            &mut ytmp,
        );
        *xg += xtmp;
        *yg += ytmp;
    }
    if bv.do_gxforms {
        xtmp = *xg;
        *xg = bv.ginv[0][0] * xtmp + bv.ginv[1][0] * *yg + bv.ginv[2][0];
        *yg = bv.ginv[0][1] * xtmp + bv.ginv[1][1] * *yg + bv.ginv[2][1];
    }
    let xframe =
        (*xg - bv.min_xpiece as f32 - (n_xoverlap / 2) as f32) / (nxin - n_xoverlap) as f32;
    let yframe =
        (*yg - bv.min_ypiece as f32 - (ny_overlap / 2) as f32) / (nyin - ny_overlap) as f32;
    let mut ixframe = (xframe + 1.) as i32; //truncation gives proper frame
    let mut iyframe = (yframe + 1.) as i32;
    let mut ngframe = ixframe < 1
        || ixframe > bv.nx_pieces
        || iyframe < 1
        || iyframe > bv.ny_pieces
        || mp(
            bv,
            bv.nx_pieces.min(1.max(ixframe)),
            bv.ny_pieces.min(1.max(iyframe)),
        ) == 0;
    if bv.multng {
        //
        // if there are multineg h's (including piece shifting!), need to make
        // sure point is actually in frame, but if frame no good, switch to
        // nearest frame first
        if ngframe {
            find_nearest_piece(bv, &mut ixframe, &mut iyframe);
        }
        let ipc = mp(bv, ixframe, iyframe);
        position_in_piece(bv, *xg, *yg, ipc, &mut xbak, &mut ybak);
        ngframe = xbak < 0. || xbak > (nxin - 1) as f32 || ybak < 0. || ybak > (nyin - 1) as f32;

        if ngframe {
            //
            // Use the error to switch to a nearby frame as center of search
            //
            xtmp = xbak + bv.ix_pc_list[(ipc - 1) as usize] as f32;
            ytmp = ybak + bv.iy_pc_list[(ipc - 1) as usize] as f32;
            ixframe = ((xtmp - bv.min_xpiece as f32 - (n_xoverlap / 2) as f32)
                / (nxin - n_xoverlap) as f32
                + 1.) as i32;
            iyframe = ((ytmp - bv.min_ypiece as f32 - (ny_overlap / 2) as f32)
                / (nyin - ny_overlap) as f32
                + 1.) as i32;
            //
            // If still not in a frame, start in the nearest and expand the search
            //
            if ixframe < 1
                || ixframe > bv.nx_pieces
                || iyframe < 1
                || iyframe > bv.ny_pieces
                || mp(
                    bv,
                    bv.nx_pieces.min(1.max(ixframe)),
                    bv.ny_pieces.min(1.max(iyframe)),
                ) == 0
            {
                find_nearest_piece(bv, &mut ixframe, &mut iyframe);
                id_search = 2;
            }
        } else {
            //
            // but even if frame is good, if in a corner, switch to the 9-piece
            // search to start with most interior point
            //
            ngframe = (xbak < bv.edge_lo_near[0] && ybak < bv.edge_lo_near[1])
                || (xbak < bv.edge_lo_near[0] && ybak > bv.edge_hi_near[1])
                || (xbak > bv.edge_hi_near[0] && ybak < bv.edge_lo_near[1])
                || (xbak > bv.edge_hi_near[0] && ybak > bv.edge_hi_near[1]);
        }
    }
    //
    // if not a good frame, look in square of nine potential pieces,
    // switch to the one that the point is a minimum distance from
    //
    if ngframe {
        let mut distmin = 1.0e10f32;
        //
        // continue loop to find most piece where point is most interior,
        // not just to find the first one.  This should be run rarely so
        // it is not a big drain
        //
        let mut ixfrm = 1.max(bv.nx_pieces.min(ixframe - id_search));
        while ixfrm <= bv.nx_pieces.min(1.max(ixframe + id_search)) {
            let mut iyfrm = 1.max(bv.ny_pieces.min(iyframe - id_search));
            while iyfrm <= bv.ny_pieces.min(1.max(iyframe + id_search)) {
                let ipc = mp(bv, ixfrm, iyfrm);
                if ipc != 0 {
                    //
                    // get real coordinate in piece, adjusting for h if present
                    //
                    position_in_piece(bv, *xg, *yg, ipc, &mut xtmp, &mut ytmp);
                    //
                    // distance is negative for a piece that point is actually in;
                    // it is negative of distance from nearest edge
                    //
                    let mut dist = -xtmp;
                    for v in [xtmp - (nxin - 1) as f32, -ytmp, ytmp - (nyin - 1) as f32] {
                        dist = if dist > v { dist } else { v };
                    }
                    if dist < distmin {
                        distmin = dist;
                        bv.min_xframe = ixfrm;
                        bv.min_yframe = iyfrm;
                    }
                }
                iyfrm += 1;
            }
            ixfrm += 1;
        }
        //
        // return if no pieces in this loop.  This seems odd but there are
        // weird edge effects if it returns just because it is not inside any
        if distmin == 1.0e10 {
            return;
        }
        ixframe = bv.min_xframe;
        iyframe = bv.min_yframe;
    }
    //
    // initialize list of pieces with this piece # on it, then start looping
    // over the pieces present in the list
    // Keep track of min and max frame numbers on list
    bv.num_pieces = 1;
    bv.in_piece[1] = mp(bv, ixframe, iyframe);
    let mut indinp = 1i32;
    needcheck[0][0] = true;
    needcheck[1][0] = true;
    bv.min_xframe = ixframe;
    bv.max_xframe = ixframe;
    bv.min_yframe = iyframe;
    bv.max_yframe = iyframe;
    bv.inp_xframe[0] = ixframe;
    bv.inp_yframe[0] = iyframe;
    position_in_piece(bv, *xg, *yg, bv.in_piece[1], &mut xtmp, &mut ytmp);
    bv.x_in_piece[0] = xtmp;
    bv.y_in_piece[0] = ytmp;
    //
    while indinp <= bv.num_pieces {
        //
        // come into this loop looking at a piece onlist; use true
        // coordinates in piece to see if point is near/in an edge to another
        //
        let iu = (indinp - 1) as usize;
        let ipc = bv.in_piece[indinp as usize];
        xycur[0] = bv.x_in_piece[iu];
        xycur[1] = bv.y_in_piece[iu];
        //
        // check the x and y directions to see if point is near an edge
        //
        for ixy in 1..=2i32 {
            let ixyu = (ixy - 1) as usize;
            for iflo in 0..=1i32 {
                if needcheck[ixyu][iu] {
                    let mut newedge = 0i32;
                    let ilo = e2(ipc, ixy, &bv.iedge_lower_ext);
                    let iup = e2(ipc, ixy, &bv.iedge_upper_ext);
                    if iflo == 1 && xycur[ixyu] < bv.edge_lo_near[ixyu] && bv.iedge_lower[ilo] > 0 {
                        newedge = bv.iedge_lower[ilo];
                        newpiece = bv.ipiece_lower[e2(newedge, ixy, &bv.ipiece_lower_ext)];
                    }
                    if iflo == 0 && xycur[ixyu] > bv.edge_hi_near[ixyu] && bv.iedge_upper[iup] > 0 {
                        newedge = bv.iedge_upper[iup];
                        newpiece = bv.ipiece_upper[e2(newedge, ixy, &bv.ipiece_upper_ext)];
                    }
                    //
                    // But if there are 4 pieces, make sure this picks up an edge
                    // between this piece and an existing piece
                    if newedge == 0 && bv.num_pieces >= 4 {
                        let lookxfr = bv.inp_xframe[iu] + (2 - ixy) * (1 - 2 * iflo);
                        let lookyfr = bv.inp_yframe[iu] + (ixy - 1) * (1 - 2 * iflo);
                        for i in 1..=bv.num_pieces {
                            let k = (i - 1) as usize;
                            if bv.inp_xframe[k] == lookxfr && bv.inp_yframe[k] == lookyfr {
                                newpiece = bv.in_piece[i as usize];
                                if iflo == 0 {
                                    newedge = bv.iedge_upper[iup];
                                }
                                if iflo == 1 {
                                    newedge = bv.iedge_lower[ilo];
                                }
                                break;
                            }
                        }
                    }
                    //
                    // if check picked up a new edge, see if edge is on list already
                    //
                    if newedge != 0 {
                        let mut edgeonlist = false;
                        for i in 1..=bv.num_edges[ixyu] {
                            edgeonlist =
                                edgeonlist || (bv.in_edge[ixyu][(i - 1) as usize] == newedge);
                        }
                        //
                        // if not, add it, and the implied piece, to list
                        //
                        if !edgeonlist {
                            let mut listno = 0i32;
                            for i in 1..=bv.num_pieces {
                                if newpiece == bv.in_piece[i as usize] {
                                    listno = i;
                                }
                            }
                            if listno == 0 {
                                //
                                // but if adding a new piece, check for point actually in
                                // piece first
                                //
                                position_in_piece(bv, *xg, *yg, newpiece, &mut xbak, &mut ybak);
                                if xbak >= 0.
                                    && xbak <= (nxin - 1) as f32
                                    && ybak >= 0.
                                    && ybak <= (nyin - 1) as f32
                                {
                                    bv.num_pieces += 1;
                                    let np = bv.num_pieces as usize;
                                    bv.in_piece[np] = newpiece;
                                    bv.x_in_piece[np - 1] = xbak;
                                    bv.y_in_piece[np - 1] = ybak;
                                    //
                                    // Get new frame numbers and maintain min/max
                                    if ixy == 1 {
                                        bv.inp_xframe[np - 1] = bv.inp_xframe[iu] + 1 - 2 * iflo;
                                        bv.inp_yframe[np - 1] = bv.inp_yframe[iu];
                                    } else {
                                        bv.inp_xframe[np - 1] = bv.inp_xframe[iu];
                                        bv.inp_yframe[np - 1] = bv.inp_yframe[iu] + 1 - 2 * iflo;
                                    }
                                    bv.min_xframe = bv.min_xframe.min(bv.inp_xframe[np - 1]);
                                    bv.min_yframe = bv.min_yframe.min(bv.inp_yframe[np - 1]);
                                    bv.max_xframe = bv.max_xframe.max(bv.inp_xframe[np - 1]);
                                    bv.max_yframe = bv.max_yframe.max(bv.inp_yframe[np - 1]);
                                    //
                                    // If there are crossed limits to edges, then still need
                                    // to check this axis for this new piece
                                    needcheck[ixyu][np - 1] =
                                        bv.edge_lo_near[ixyu] >= bv.edge_hi_near[ixyu];
                                    needcheck[(3 - ixy - 1) as usize][np - 1] = true;
                                    listno = bv.num_pieces;
                                } else if bv.num_pieces == 1
                                    && bv.any_disjoint[(bv.inp_xframe[iu] - 1) as usize
                                        + bv.any_disjoint_ext[0] * (bv.inp_yframe[iu] - 1) as usize]
                                {
                                    //
                                    // If the point was NOT in this overlapping piece, and
                                    // any corners are disjoint, find cross-corner piece
                                    // Look across upper and lower edges on the other axis
                                    // from the rejected piece to see if point in other piece
                                    newedge =
                                        bv.iedge_lower[e2(newpiece, 3 - ixy, &bv.iedge_lower_ext)];
                                    if newedge != 0 {
                                        let p = bv.ipiece_lower
                                            [e2(newedge, 3 - ixy, &bv.ipiece_lower_ext)];
                                        position_in_piece(bv, *xg, *yg, p, &mut xbak, &mut ybak);
                                        if xbak >= 0.
                                            && xbak <= (nxin - 1) as f32
                                            && ybak >= 0.
                                            && ybak <= (nyin - 1) as f32
                                        {
                                            ipc_cross = p;
                                            xpc_cross = xbak;
                                            ypc_cross = ybak;
                                        }
                                    }
                                    //
                                    newedge =
                                        bv.iedge_upper[e2(newpiece, 3 - ixy, &bv.iedge_upper_ext)];
                                    if newedge != 0 {
                                        let p = bv.ipiece_upper
                                            [e2(newedge, 3 - ixy, &bv.ipiece_upper_ext)];
                                        position_in_piece(bv, *xg, *yg, p, &mut xbak, &mut ybak);
                                        if xbak >= 0.
                                            && xbak <= (nxin - 1) as f32
                                            && ybak >= 0.
                                            && ybak <= (nyin - 1) as f32
                                        {
                                            ipc_cross = p;
                                            xpc_cross = xbak;
                                            ypc_cross = ybak;
                                        }
                                    }
                                }
                            }
                            //
                            // add edge to list only if legal piece found
                            //
                            if listno > 0 {
                                bv.num_edges[ixyu] += 1;
                                let ne = (bv.num_edges[ixyu] - 1) as usize;
                                bv.in_edge[ixyu][ne] = newedge;
                                if iflo == 0 {
                                    bv.in_ed_upper[ixyu][ne] = listno;
                                    bv.in_ed_lower[ixyu][ne] = indinp;
                                } else {
                                    bv.in_ed_upper[ixyu][ne] = indinp;
                                    bv.in_ed_lower[ixyu][ne] = listno;
                                }
                            }
                        }
                    }
                }
            }
        }
        indinp += 1;
    }
    //
    // If the pieces extend too far in either direction, we need to reduce the
    // list to the ones where the point is most interior
    if bv.max_xframe > bv.min_xframe + 1 || bv.max_yframe > bv.min_yframe + 1 {
        //
        // first find axis where point is most interior
        // `maxInside` is `integer*4`: each assignment truncates the real.
        let mut max_inside = -10000000i32;
        for i in 1..=bv.num_pieces {
            let k = (i - 1) as usize;
            let (xi, yi) = (bv.x_in_piece[k], bv.y_in_piece[k]);
            let vx = (nxin - 1) as f32 - xi;
            let mx = if xi < vx { xi } else { vx };
            if mx > max_inside as f32 {
                max_inside = mx as i32;
                axisin = 1;
            }
            let vy = (nyin - 1) as f32 - yi;
            let my = if yi < vy { yi } else { vy };
            if my > max_inside as f32 {
                let vyx = (nyin - 1) as f32 - xi;
                max_inside = (if yi < vyx { yi } else { vyx }) as i32;
                axisin = 2;
            }
        }
        //
        // do the most interior axis first and find best set of frames for it
        // Then do the other axis, with new constraint on first axis
        for ixy in 1..=2i32 {
            if axisin == ixy {
                most_interior_frames(bv, 1, &bv.x_in_piece, nxin, &mut ixframe, &mut iyframe);
                bv.min_xframe = ixframe;
                bv.max_xframe = (ixframe + 1).min(bv.max_xframe);
                if ixy == 1 {
                    constrain_frames(
                        bv.num_pieces,
                        bv.min_xframe,
                        bv.max_xframe,
                        &mut bv.min_yframe,
                        &mut bv.max_yframe,
                        &bv.inp_xframe,
                        &bv.inp_yframe,
                    );
                }
            } else {
                most_interior_frames(bv, 2, &bv.y_in_piece, nyin, &mut ixframe, &mut iyframe);
                bv.min_yframe = iyframe;
                bv.max_yframe = (iyframe + 1).min(bv.max_yframe);
                if ixy == 1 {
                    constrain_frames(
                        bv.num_pieces,
                        bv.min_yframe,
                        bv.max_yframe,
                        &mut bv.min_xframe,
                        &mut bv.max_xframe,
                        &bv.inp_yframe,
                        &bv.inp_xframe,
                    );
                }
            }
        }
        //
        // Repack the frames, retaining only the ones within range and keeping
        // track of former numbers
        let mut j = 0i32;
        for i in 1..=bv.num_pieces {
            let k = (i - 1) as usize;
            if bv.inp_xframe[k] >= bv.min_xframe
                && bv.inp_xframe[k] <= bv.max_xframe
                && bv.inp_yframe[k] >= bv.min_yframe
                && bv.inp_yframe[k] <= bv.max_yframe
            {
                j += 1;
                let jk = (j - 1) as usize;
                bv.in_piece[j as usize] = bv.in_piece[i as usize];
                bv.x_in_piece[jk] = bv.x_in_piece[k];
                bv.y_in_piece[jk] = bv.y_in_piece[k];
                bv.inp_xframe[jk] = bv.inp_xframe[k];
                bv.inp_yframe[jk] = bv.inp_yframe[k];
                moved_piece[jk] = i;
            }
        }
        bv.num_pieces = j;
        //
        // Repack edges too
        for ixyu in 0..2usize {
            let mut j = 0i32;
            for i in 1..=bv.num_edges[ixyu] {
                let k = (i - 1) as usize;
                let mut ixfrm = 0i32;
                let mut iyfrm = 0i32;
                //
                // Find the pieces that the lower and upper pieces became; if both
                // exist, retain the edge and reassign the numbers
                for kk in 1..=bv.num_pieces {
                    if bv.in_ed_lower[ixyu][k] == moved_piece[(kk - 1) as usize] {
                        ixfrm = kk;
                    }
                    if bv.in_ed_upper[ixyu][k] == moved_piece[(kk - 1) as usize] {
                        iyfrm = kk;
                    }
                }
                if ixfrm > 0 && iyfrm > 0 {
                    j += 1;
                    let jk = (j - 1) as usize;
                    bv.in_ed_lower[ixyu][jk] = ixfrm;
                    bv.in_ed_upper[ixyu][jk] = iyfrm;
                    bv.in_edge[ixyu][jk] = bv.in_edge[ixyu][k];
                }
            }
            bv.num_edges[ixyu] = j;
        }
    }
    //
    // If there is one piece and a potential cross-piece, consider switching
    if bv.num_pieces == 1 && ipc_cross != 0 {
        ixframe =
            (bv.ix_pc_list[(ipc_cross - 1) as usize] - bv.min_xpiece) / (nxin - n_xoverlap) + 1;
        iyframe =
            (bv.iy_pc_list[(ipc_cross - 1) as usize] - bv.min_ypiece) / (nyin - ny_overlap) + 1;
        let ixy = bv.map_disjoint[(ixframe.min(bv.inp_xframe[0]) - 1) as usize
            + bv.map_disjoint_ext[0] * (iyframe.min(bv.inp_yframe[0]) - 1) as usize];
        //
        // Use cross piece if point is more interior along the other axis from
        // the disjoint edges (ixy is 1, 2 for X, 3, 4 for Y)
        if ixy > 0 {
            let fmin = |a: f32, b: f32| if a < b { a } else { b };
            if (ixy <= 2
                && fmin(ypc_cross, (nyin - 1) as f32 - ypc_cross)
                    > fmin(bv.y_in_piece[0], (nyin - 1) as f32 - bv.y_in_piece[0]))
                || (ixy > 2
                    && fmin(xpc_cross, (nxin - 1) as f32 - xpc_cross)
                        > fmin(bv.x_in_piece[0], (nxin - 1) as f32 - bv.x_in_piece[0]))
            {
                bv.in_piece[1] = ipc_cross;
                bv.x_in_piece[0] = xpc_cross;
                bv.y_in_piece[0] = ypc_cross;
                bv.inp_xframe[0] = ixframe;
                bv.inp_yframe[0] = iyframe;
            }
        }
    }
    //
    // If there are two pieces and the limit flag is set, find out if one
    // should be thrown away
    if bv.limit_data && bv.num_pieces == 2 {
        let mut ixy = 1i32;
        if bv.num_edges[1] > 0 {
            ixy = 2;
        }
        for i in 1..=2i32 {
            let mut iflo = 1i32;
            //
            // if the piece is lower, get the limits on upper side
            if i == bv.in_ed_lower[(ixy - 1) as usize][0] {
                iflo = 2;
            }
            let p = bv.in_piece[i as usize];
            get_data_limits(bv, p, 3 - ixy, iflo, &mut limit_lo, &mut limit_hi);
            let k = (i - 1) as usize;
            in_limit[k] = (ixy == 1
                && bv.y_in_piece[k] >= limit_lo as f32
                && bv.y_in_piece[k] <= limit_hi as f32)
                || (ixy == 2
                    && bv.x_in_piece[k] >= limit_lo as f32
                    && bv.x_in_piece[k] <= limit_hi as f32);
        }
        if b3dxor(in_limit[0], in_limit[1]) {
            bv.num_pieces = 1;
            bv.num_edges[(ixy - 1) as usize] = 0;
            if in_limit[1] {
                bv.x_in_piece[0] = bv.x_in_piece[1];
                bv.y_in_piece[0] = bv.y_in_piece[1];
                bv.in_piece[1] = bv.in_piece[2];
            }
        }
    }
    //
    // Replace edges with ones to use if called for
    if use_edges {
        for ixyu in 0..2usize {
            for i in 0..bv.num_edges[ixyu] as usize {
                let mut newedge = 0i32;
                find_edge_to_use(bv, bv.in_edge[ixyu][i], ixyu as i32 + 1, &mut newedge);
                if newedge != 0 {
                    bv.in_edge[ixyu][i] = newedge;
                }
            }
        }
    }
}

/// Original: `subroutine mostInteriorFrames(ixy, xyinpiece, nxyin, ixbest,
/// iybest)`, internal to `countedges` (`bsubs.f90:1724`).
///
/// Finds which set of pieces has the point most interior in direction
/// `ixy`.  The host's `maxInside`, `ixfrm`, `iyfrm` and `i` that it uses are
/// dead in the host afterwards, so they are locals; `maxInside` and
/// `maxOneCol` are `integer*4`, so each assignment of the real `closest`
/// truncates, and the comparisons are made in real.
pub fn most_interior_frames(
    bv: &BlendVars,
    ixy: i32,
    xyinpiece: &[f32],
    nxyin: i32,
    ixbest: &mut i32,
    iybest: &mut i32,
) {
    let mut distin = [[0.0f32; 2]; 2];
    let mut max_inside = -100000i32;
    let mut max_one_col = -100000i32;
    *ixbest = bv.min_xframe;
    *iybest = bv.min_yframe;
    let mut ix_one_col = bv.min_xframe;
    let mut iy_one_col = bv.min_yframe;
    for iyfrm in bv.min_yframe..=bv.min_yframe.max(bv.max_yframe - 1) {
        for ixfrm in bv.min_xframe..=bv.min_xframe.max(bv.max_xframe - 1) {
            //
            // Find the minimum distance to edge for frames that fit in this range
            // `distin(ix, iy)` is `distin[iy - 1][ix - 1]`.
            distin = [[1000000.0f32; 2]; 2];
            for i in 1..=bv.num_pieces {
                let k = (i - 1) as usize;
                let ix = bv.inp_xframe[k] + 1 - ixfrm;
                let iy = bv.inp_yframe[k] + 1 - iyfrm;
                if (ix + 1) / 2 == 1 && (iy + 1) / 2 == 1 {
                    let v = nxyin as f32 - xyinpiece[k];
                    distin[(iy - 1) as usize][(ix - 1) as usize] =
                        if xyinpiece[k] < v { xyinpiece[k] } else { v };
                }
            }
            let fmin = |a: f32, b: f32| if a < b { a } else { b };
            let (dist1, dist2) = if ixy == 1 {
                (
                    fmin(distin[0][0], distin[1][0]),
                    fmin(distin[0][1], distin[1][1]),
                )
            } else {
                (
                    fmin(distin[0][0], distin[0][1]),
                    fmin(distin[1][0], distin[1][1]),
                )
            };
            //
            // Keep track of which range maximizes this distance separately for
            // ones with one column and two
            if dist1 < 999999. && dist2 < 999999. {
                let closest = (dist1 + dist2) / 2.;
                if closest <= nxyin as f32 && closest > max_inside as f32 {
                    max_inside = closest as i32;
                    *ixbest = ixfrm;
                    *iybest = iyfrm;
                }
            } else {
                let closest = fmin(dist1, dist2);
                if closest <= nxyin as f32 && closest > max_one_col as f32 {
                    max_one_col = closest as i32;
                    ix_one_col = ixfrm;
                    iy_one_col = iyfrm;
                }
            }
        }
    }
    if max_inside < 0 && max_one_col > 0 {
        *ixbest = ix_one_col;
        *iybest = iy_one_col;
    }
}

/// Original: `subroutine constrainFrames(minAxis, maxAxis, minOther,
/// maxOther, inpfAxis, inpfOther)`, internal to `countedges`
/// (`bsubs.f90:1779`).  `numPieces` is the module variable it reads through
/// host association; its `i` is a host scratch variable, dead afterwards.
pub fn constrain_frames(
    num_pieces: i32,
    min_axis: i32,
    max_axis: i32,
    min_other: &mut i32,
    max_other: &mut i32,
    inpf_axis: &[i32],
    inpf_other: &[i32],
) {
    let i = *min_other;
    *min_other = *max_other;
    *max_other = i;
    for i in 0..num_pieces as usize {
        if inpf_axis[i] >= min_axis && inpf_axis[i] <= max_axis {
            *min_other = (*min_other).min(inpf_other[i]);
            *max_other = (*max_other).max(inpf_other[i]);
        }
    }
}

/// Original: `subroutine positionInPiece(xg, yg, ipc, xinpc, yinpc)`
/// (`bsubs.f90:1800`).
///
/// Position of global point (`xg`, `yg`) (after g transforms) in piece
/// `ipc`.
pub fn position_in_piece(
    bv: &BlendVars,
    xg: f32,
    yg: f32,
    ipc: i32,
    xinpc: &mut f32,
    yinpc: &mut f32,
) {
    *xinpc = xg - bv.ix_pc_list[(ipc - 1) as usize] as f32;
    *yinpc = yg - bv.iy_pc_list[(ipc - 1) as usize] as f32;
    if bv.multng {
        let h = |i: usize, j: usize| {
            bv.hinv[(i - 1) + bv.hinv_ext[0] * ((j - 1) + bv.hinv_ext[1] * (ipc - 1) as usize)]
        };
        let xtmp = *xinpc;
        *xinpc = h(1, 1) * xtmp + h(1, 2) * *yinpc + h(1, 3);
        *yinpc = h(2, 1) * xtmp + h(2, 2) * *yinpc + h(2, 3);
    }
}

/// Original: `subroutine initNearList()` (`bsubs.f90:1819`).
///
/// Sets up an ordered list of dx, dy values to nearby pieces up to
/// `maxDistNear`, sorting on squared distance held in `array`.
pub fn init_near_list(bv: &mut BlendVars) {
    let limx = (bv.nx_pieces - 1).min(MAX_DIST_NEAR);
    let limy = (bv.ny_pieces - 1).min(MAX_DIST_NEAR);
    bv.num_pc_near = 0;
    for idx in -limx..=limx {
        for idy in -limy..=limy {
            if idx != 0 || idy != 0 {
                bv.num_pc_near += 1;
                let n = (bv.num_pc_near - 1) as usize;
                bv.idx_pc_near[n] = idx;
                bv.idy_pc_near[n] = idy;
                bv.array[n] = (idx * idx + idy * idy) as f32;
            }
        }
    }
    for i in 1..=bv.num_pc_near - 1 {
        for j in i + 1..=bv.num_pc_near {
            let (iu, ju) = ((i - 1) as usize, (j - 1) as usize);
            if bv.array[iu] > bv.array[ju] {
                bv.array.swap(iu, ju);
                bv.idx_pc_near.swap(iu, ju);
                bv.idy_pc_near.swap(iu, ju);
            }
        }
    }
}

/// Original: `subroutine findNearestPiece(ixFrame, iyFrame)`
/// (`bsubs.f90:1861`).
///
/// Clamps the frame numbers to the montage, then, if that frame has no
/// piece, switches to the nearest frame that does.
pub fn find_nearest_piece(bv: &BlendVars, ix_frame: &mut i32, iy_frame: &mut i32) {
    let mp = |ix: i32, iy: i32| {
        bv.map_piece[(ix - 1) as usize + bv.map_piece_ext[0] * (iy - 1) as usize]
    };
    *ix_frame = 1.max(bv.nx_pieces.min(*ix_frame));
    *iy_frame = 1.max(bv.ny_pieces.min(*iy_frame));
    if mp(*ix_frame, *iy_frame) != 0 {
        return;
    }
    for i in 0..bv.num_pc_near as usize {
        let ixnew = *ix_frame + bv.idx_pc_near[i];
        let iynew = *iy_frame + bv.idy_pc_near[i];
        if ixnew >= 1 && ixnew <= bv.nx_pieces && iynew >= 1 && iynew <= bv.ny_pieces {
            if mp(ixnew, iynew) != 0 {
                *ix_frame = ixnew;
                *iy_frame = iynew;
                return;
            }
        }
    }
}

/// Original: `subroutine dxydgrinterp(x1, y1, indedg, x2, y2, dden)`
/// (`bsubs.f90:1889`).
///
/// Takes a coordinate in the lower piece of the edge in buffer `indedg`,
/// interpolates the edge function bilinearly, and returns the coordinate in
/// the upper piece and the density difference.
pub fn dxydgrinterp(
    bv: &BlendVars,
    x1: f32,
    y1: f32,
    indedg: i32,
    x2: &mut f32,
    y2: &mut f32,
    dden: &mut f32,
) {
    let ie = (indedg - 1) as usize;
    let gbf = |grid: &[f32], ext: &[usize; 3], ix: i32, iy: i32| {
        grid[(ix - 1) as usize + ext[0] * ((iy - 1) as usize + ext[1] * ie)]
    };
    //
    // find fractional coordinate within edge grid
    //
    let xingrid = x1 - bv.ix_grd_st_bf[ie] as f32;
    let yingrid = y1 - bv.iy_grd_st_bf[ie] as f32;
    let xgrid = 1. + xingrid / bv.int_xgr_bf[ie] as f32;
    let ygrid = 1. + yingrid / bv.int_ygr_bf[ie] as f32;
    //
    // find all fractions and indices needed for bilinear interpolation
    //
    let mut ixg = xgrid as i32;
    ixg = 1.max((bv.nx_gr_bf[ie] - 1).min(ixg));
    let mut iyg = ygrid as i32;
    iyg = 1.max((bv.ny_gr_bf[ie] - 1).min(iyg));
    let d = xgrid - ixg as f32;
    let inner = if 1. < d { 1. } else { d };
    let fx1 = if 0. > inner { 0. } else { inner }; //NO EXTRAPOLATIONS ALLOWED
    let fx = 1. - fx1;
    let ixg1 = ixg + 1;
    let d = ygrid - iyg as f32;
    let inner = if 1. < d { 1. } else { d };
    let fy1 = if 0. > inner { 0. } else { inner };
    let fy = 1. - fy1;
    let iyg1 = iyg + 1;
    let c00 = fx * fy;
    let c10 = fx1 * fy;
    let c01 = fx * fy1;
    let c11 = fx1 * fy1;
    //
    // interpolate
    //
    let (gx, ex) = (&bv.dx_gr_bf, &bv.dx_gr_bf_ext);
    let dxinterp = c00 * gbf(gx, ex, ixg, iyg)
        + c10 * gbf(gx, ex, ixg1, iyg)
        + c01 * gbf(gx, ex, ixg, iyg1)
        + c11 * gbf(gx, ex, ixg1, iyg1);
    let (gy, ey) = (&bv.dy_gr_bf, &bv.dy_gr_bf_ext);
    let dyinterp = c00 * gbf(gy, ey, ixg, iyg)
        + c10 * gbf(gy, ey, ixg1, iyg)
        + c01 * gbf(gy, ey, ixg, iyg1)
        + c11 * gbf(gy, ey, ixg1, iyg1);
    let (gd, edn) = (&bv.dden_gr_bf, &bv.dden_gr_bf_ext);
    *dden = c00 * gbf(gd, edn, ixg, iyg)
        + c10 * gbf(gd, edn, ixg1, iyg)
        + c01 * gbf(gd, edn, ixg, iyg1)
        + c11 * gbf(gd, edn, ixg1, iyg1);
    //
    *x2 = xingrid + dxinterp + bv.ix_ofs_bf[ie] as f32;
    *y2 = yingrid + dyinterp + bv.iy_ofs_bf[ie] as f32;
}

/// Original: `subroutine crossvalue(xinlong, nxpieces, nypieces, nshort,
/// nlong)` (`bsubs.f90:1938`).  No `implicit none`: all integer but the
/// logical.
pub fn crossvalue(xinlong: bool, nxpieces: i32, nypieces: i32, nshort: &mut i32, nlong: &mut i32) {
    if xinlong {
        *nshort = nypieces;
        *nlong = nxpieces;
    } else {
        *nshort = nxpieces;
        *nlong = nypieces;
    }
}

/// Original: `subroutine xcorrEdge(arrLower, arrUpper, ixy, xDisplace,
/// yDisplace, legacy, useExpected, indentXC)` (`bsubs.f90:1953`).
///
/// Cross-correlates the overlap zone of edge direction `ixy` between the
/// lower and upper pieces, then (unless `legacy`) refines the displacement
/// with the SD search.  `arrLower`/`arrUpper` are the pieces' starts in the
/// caller's `array` (`array(indLow)`, `array(indUpper)`): the caller lends the
/// whole `array` out of the module and passes the 1-based starts
/// `ind_lower`/`ind_upper`.  The whole array is needed because the boxes the
/// source extracts can start before the piece (expected shifts larger than
/// the overlap), and `extractWithBinning` then reads the preceding memory of
/// `array` -- the previous piece in the cache.  Only a read before `array`
/// itself (from the first cache slot) is out of bounds in the source too.
///
/// Both `taperAtFill` calls taper the *lower* box, `brray`
/// (`bsubs.f90:2019-2020`, `:2037-2038`): the upper box at
/// `brray(maxbsiz / 2 + 1)` is never tapered and the lower one is tapered
/// twice.  Reproduced.
pub fn xcorr_edge(
    bv: &mut BlendVars,
    array: &[f32],
    ind_lower: i32,
    ind_upper: i32,
    ixy: i32,
    x_displace: &mut f32,
    y_displace: &mut f32,
    legacy: bool,
    use_expected: bool,
    indent_xc: i32,
) {
    let mut nxy_box = [0i32; 2];
    let mut ind0 = [0i32; 2];
    let mut ind1 = [0i32; 2];
    let mut i_displace = [0i32; 2];
    let mut ctf = [0.0f32; 8193];
    let mut r_displace = [0.0f32; 2];
    let mut num_extra = [0i32; 2];
    let mut ind0_upper = [0i32; 2];
    let mut ind1_upper = [0i32; 2];
    let mut n_expected = [0i32; 2];
    let (mut delta, mut sd_min, mut del_den_min, mut ccc) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut nx_pad, mut ny_pad, mut indent_use) = (0i32, 0i32, 0i32);
    let (mut nx_smooth, mut ny_smooth, mut max_long_shift) = (0i32, 0i32, 0i32);
    let (mut i, mut idum) = (0i32, 0i32);
    let mut ierr: i32;

    let indent_sd = 5i32; //indent for sdsearch
    let over_frac = 0.9f32; //fraction of overlap to use
    let num_iter = 4i32; //iterations for sdsearch
    let lim_step = 10i32; //limiting distance
    let nbin = bv.nbin_xcorr;
    let num_smooth = 6i32;
    let _wall_start = walltime();
    let ixyu = (ixy - 1) as usize;
    //
    // find size and limits of box in overlap zone to cut out
    //
    let mut iyx = 3 - ixy;
    let iyxu = (iyx - 1) as usize;
    let mut if_legacy = 0i32;
    if legacy {
        if_legacy = 1;
    }
    let mut if_eval_ccc = 0i32;
    if bv.num_xcorr_peaks > 1 && !legacy {
        if_eval_ccc = 1;
    }
    let mut n_overlap_use = bv.n_overlap;
    let mut ixy_plus = ixy;
    let mut if_weight = 0i32;

    // If using expected shifts, set up arrays with the integer expected shifts and with the
    // modified overlap to pass in long direction, setup for weighting by distance
    if use_expected {
        n_expected[0] = x_displace.round() as i32;
        n_expected[1] = y_displace.round() as i32;
        n_overlap_use[ixyu] += 0.max(-n_expected[ixyu]);
        n_overlap_use[iyxu] = n_expected[iyxu];
        ixy_plus = ixy + 2;
        if_weight = 1;
    }

    montxcbasicsizes(
        &ixy_plus,
        &nbin,
        &indent_xc,
        &bv.nxyz_in,
        &n_overlap_use,
        &bv.aspect_max,
        &bv.extra_width,
        &bv.pad_frac,
        &nice_fft_limit(),
        &mut indent_use,
        &mut nxy_box,
        &mut num_extra,
        &mut nx_pad,
        &mut ny_pad,
        &mut max_long_shift,
    );
    montxcindsandctf(
        &ixy_plus,
        &bv.nxyz_in,
        &n_overlap_use,
        &nxy_box,
        &nbin,
        &indent_use,
        &num_extra,
        &nx_pad,
        &ny_pad,
        &num_smooth,
        &bv.sigma1,
        &bv.sigma2,
        &bv.radius1,
        &bv.radius2,
        &if_eval_ccc,
        &mut ind0,
        &mut ind1,
        &mut ind0_upper,
        &mut ind1_upper,
        &mut nx_smooth,
        &mut ny_smooth,
        &mut ctf,
        &mut delta,
    );

    if nxy_box[0] * nxy_box[1] * nbin * nbin > bv.max_bsiz / 2 || nx_pad * ny_pad > bv.idimc {
        exit_error("Correlation arrays were not made large enough");
    }
    let half = (bv.max_bsiz / 2) as usize;
    let nbox = (nxy_box[0] * nxy_box[1]) as usize;
    // `extractWithBinning` takes the image as `void *`; the Fortran wrapper
    // hands it the `real` arrays, so the byte view is passed here.  The view
    // is the whole of `array`, and the piece's start (`arrLower` =
    // `array(indLow)`) is folded into the X range, which moves the
    // extraction's base address by exactly that many elements and changes
    // nothing else it computes.
    // SAFETY (all views): `u8` has alignment 1 and every bit pattern is
    // valid for both `u8` and `f32`; each view covers exactly its slice.
    let arr_lower = &array[(ind_lower - 1) as usize..];
    let arr_upper = &array[(ind_upper - 1) as usize..];
    let array_bytes =
        unsafe { std::slice::from_raw_parts(array.as_ptr().cast::<u8>(), array.len() * 4) };
    let (lower_bytes, upper_bytes) = (array_bytes, array_bytes);
    let (lower_ofs, upper_ofs) = (ind_lower - 1, ind_upper - 1);
    //
    // get the first image, lower piece
    {
        let b = &mut bv.brray[..];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr().cast::<u8>(), b.len() * 4) };
        ierr = extractwithbinning(
            lower_bytes,
            &bv.nxyz_in[0],
            &(ind0[0] + lower_ofs),
            &(ind1[0] + lower_ofs),
            &ind0[1],
            &ind1[1],
            &nbin,
            bytes,
            &1,
            &mut i,
            &mut idum,
        );
    }
    if bv.ifill_treatment == 2 {
        ierr = taperatfill(&mut bv.brray[..nbox], &nxy_box[0], &nxy_box[1], &64, &0);
    }
    //
    // get the second image, upper piece
    // If expected sizes, indexes are for X then Y, if not, the one index is for edge
    // direction
    if use_expected {
        ind0[ixyu] = ind0_upper[ixyu];
        ind1[ixyu] = ind1_upper[ixyu];
        ind0[iyxu] = ind0_upper[iyxu];
        ind1[iyxu] = ind1_upper[iyxu];
    } else {
        ind0[ixyu] = ind0_upper[0];
        ind1[ixyu] = ind1_upper[0];
    }
    {
        let b = &mut bv.brray[half..];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr().cast::<u8>(), b.len() * 4) };
        ierr = extractwithbinning(
            upper_bytes,
            &bv.nxyz_in[0],
            &(ind0[0] + upper_ofs),
            &(ind1[0] + upper_ofs),
            &ind0[1],
            &ind1[1],
            &nbin,
            bytes,
            &1,
            &mut i,
            &mut idum,
        );
    }
    if bv.ifill_treatment == 2 {
        ierr = taperatfill(&mut bv.brray[..nbox], &nxy_box[0], &nxy_box[1], &64, &0);
    }
    let _ = ierr;

    // 12/3/22: stop weighting by distance from expected if there is extra width, it should
    // not apply to very sloppy per se, only if weighting option is given
    //
    // Do the correlation
    // `dumpEdge` uses the module, so the correlation arrays are lent out of
    // it for the call and the callback borrows the rest.
    let brray = std::mem::take(&mut bv.brray);
    let mut xcray = std::mem::take(&mut bv.xcray);
    let mut xdray = std::mem::take(&mut bv.xdray);
    let mut xeray = std::mem::take(&mut bv.xeray);
    let nxyz_in = bv.nxyz_in;
    let n_overlap = bv.n_overlap;
    let num_xcorr_peaks = bv.num_xcorr_peaks;
    {
        let mut two_d_fft = |array: &mut [f32], nx: &mut i32, ny: &mut i32, idir: &mut i32| {
            todfft(array, *nx, *ny, *idir)
        };
        let mut dump_edge = |crray: &mut [f32],
                             nxdim: &mut i32,
                             nxpad: &mut i32,
                             nypad: &mut i32,
                             ixy: &mut i32,
                             ifcorr: &mut i32| {
            dumpedge(bv, crray, *nxdim, *nxpad, *nypad, *ixy, *ifcorr)
        };
        montxcorredge(
            &brray,
            &brray[half..],
            &nxy_box,
            &nxyz_in,
            &n_overlap,
            &nx_smooth,
            &ny_smooth,
            &nx_pad,
            &ny_pad,
            &mut xcray,
            &mut xdray,
            Some(&mut xeray),
            &num_xcorr_peaks,
            &if_legacy,
            &ctf,
            &delta,
            &num_extra,
            &nbin,
            &ixy,
            &max_long_shift,
            &if_weight,
            x_displace,
            y_displace,
            &mut ccc,
            &mut two_d_fft,
            Some(&mut dump_edge),
            &0,
        );
    }
    bv.brray = brray;
    bv.xcray = xcray;
    bv.xdray = xdray;
    bv.xeray = xeray;

    if legacy {
        return;
    }
    //
    // the following is adopted largely from setgridchars
    //
    let nxin = bv.nxyz_in[0];
    let nyin = bv.nxyz_in[1];
    let ix_displace = x_displace.round() as i32;
    let iy_displace = y_displace.round() as i32;
    if ixy == 1 {
        i_displace[0] = nxin - bv.n_overlap[0] + ix_displace;
        i_displace[1] = iy_displace;
    } else {
        i_displace[0] = ix_displace;
        i_displace[1] = nyin - bv.n_overlap[1] + iy_displace;
    }
    iyx = 3 - ixy;
    let iyxu = (iyx - 1) as usize;
    //
    // get size of box, limit to size of overlap zone
    //
    nxy_box[ixyu] = bv.n_overlap[ixyu].min(bv.nxyz_in[ixyu] - i_displace[ixyu]);
    nxy_box[ixyu] =
        (nxy_box[ixyu] - indent_sd * 2).min((over_frac * nxy_box[ixyu] as f32).round() as i32);
    nxy_box[iyxu] = bv.nxyz_in[iyxu] - i_displace[iyxu].abs();
    nxy_box[iyxu] =
        (nxy_box[iyxu] - indent_sd * 2).min((bv.aspect_max * nxy_box[ixyu] as f32).round() as i32);
    for i in 0..2usize {
        ind0[i] = (bv.nxyz_in[i] + i_displace[i] - nxy_box[i]) / 2;
        ind1[i] = ind0[i] + nxy_box[i];
        r_displace[i] = -i_displace[i] as f32;
    }
    //
    // integer scan is not needed (commented out in the source)
    //
    let (rd0, rd1) = r_displace.split_at_mut(1);
    montbigsearch(
        arr_lower,
        arr_upper,
        nxin,
        nyin,
        ind0[0],
        ind0[1],
        ind1[0],
        ind1[1],
        &mut rd0[0],
        &mut rd1[0],
        &mut sd_min,
        &mut del_den_min,
        num_iter,
        lim_step,
    );
    if ixy == 1 {
        *x_displace = -r_displace[0] - (nxin - bv.n_overlap[0]) as f32;
        *y_displace = -r_displace[1];
    } else {
        *x_displace = -r_displace[0];
        *y_displace = -r_displace[1] - (nyin - bv.n_overlap[1]) as f32;
    }
}

/// Original: `subroutine find_best_shifts(dxgridmean, dygridmean, idimedge,
/// idirIn, izsect, h, nsum, bavg, bmax, aavg, amax, tryAlts)`
/// (`bsubs.f90:2105`).
///
/// Solves for the shifts of all pieces in section `izsect` from the
/// displacements across their edges (`dxgridmean`, `dygridmean`, dimensioned
/// `(idimedge, 2)`), or for piece density offsets into `denSolution` when
/// `|idirIn| > 1`.  `h` is `real*4 h(2,3,*)`, flat column-major.  Returns the
/// mean and max edge displacement before (`bavg`, `bmax`) and after (`aavg`,
/// `amax`) over `nsum` edges.
pub fn find_best_shifts(
    bv: &mut BlendVars,
    dxgridmean: &mut [f32],
    dygridmean: &mut [f32],
    idimedge: i32,
    idir_in: i32,
    izsect: i32,
    h: &mut [f32],
    nsum: &mut i32,
    bavg: &mut f32,
    bmax: &mut f32,
    aavg: &mut f32,
    amax: &mut f32,
    try_alts: bool,
) {
    //
    // Set maxvar higher to get comparisons
    const MAX_GAUSSJ: i32 = 10;
    const MAXVAR: i32 = MAX_GAUSSJ;
    // `real*4 a(maxvar,maxvar)`, flat column-major.
    let mut a = [0.0f32; (MAXVAR * MAXVAR) as usize];
    let dm = |iedge: i32, ixy: i32| ((iedge - 1) + idimedge * (ixy - 1)) as usize;
    let hx = |i: i32, j: i32, k: i32| ((i - 1) + 2 * ((j - 1) + 3 * (k - 1))) as usize;
    let e2 = |i: i32, ixy: i32, ext: &[usize; 2]| (i - 1) as usize + ext[0] * (ixy - 1) as usize;
    let mut iedge = 0i32;
    let mut num_groups = 0i32;
    let mut i = 0i32;
    let (mut w_err_mean, mut w_err_max) = (0.0f32, 0.0f32);
    let (mut wall_adj, mut wall_gaussj) = (0.0f64, 0.0f64);

    let do_densities = idir_in.abs() > 1;
    let idir = if idir_in >= 0 { 1 } else { -1 };
    let mut num_col = 2i32;
    bv.num_alt_fixed = 0;
    bv.num_low_weight = 0;
    if do_densities {
        num_col = 1;
    }
    //
    // The data coming in are the displacements of upper piece from
    // being in alignment with the lower piece if idir = 1, or the
    // shift needed to align upper piece with lower if idir = -1
    //
    // build list of variables: ALL means all pieces that have an edge
    //
    let mut nallvar = 0i32;
    for ipc in 1..=bv.npc_list {
        let ip = (ipc - 1) as usize;
        if bv.iz_pc_list[ip] == izsect {
            if do_densities {
                bv.den_solution[ip] = 0.;
            } else {
                xfunit(&mut h[hx(1, 1, ipc)..], 1.);
                let off = bv.hinv_ext[0] * bv.hinv_ext[1] * ip;
                xfunit(&mut bv.hinv[off..], 1.);
            }
            bv.ind_var[ip] = 0;
            if bv.iedge_lower[e2(ipc, 1, &bv.iedge_lower_ext)] > 0
                || bv.iedge_lower[e2(ipc, 2, &bv.iedge_lower_ext)] > 0
                || bv.iedge_upper[e2(ipc, 1, &bv.iedge_upper_ext)] > 0
                || bv.iedge_upper[e2(ipc, 2, &bv.iedge_upper_ext)] > 0
            {
                nallvar += 1;
                if nallvar > bv.lim_var {
                    let _ = writeln!(
                        ImodFile::Stdout,
                        " nallvar, limvar{:12}{:12}",
                        nallvar,
                        bv.lim_var
                    );
                    exit_error("Arrays were not made large enough for find_best_shifts");
                }
                bv.iall_var_pc[(nallvar - 1) as usize] = ipc;
                bv.ivar_group[(nallvar - 1) as usize] = 0;
                bv.ind_var[ip] = nallvar;
            }
        }
    }
    //
    // Classify pieces into separate groups if any by following connections
    // between them
    sort_vars_into_groups(bv, nallvar, &mut num_groups);
    //
    *nsum = 0;
    let mut bsum = 0.0f32;
    *bmax = 0.;
    let mut asum = 0.0f32;
    *amax = 0.;
    *bavg = 0.;
    *aavg = 0.;
    let mut num_prev = 0i32;
    if nallvar == 0 {
        xfunit(&mut h[hx(1, 1, 1)..], 1.0);
    }
    if nallvar < 2 {
        return;
    }
    //
    // Loop on groups, set up to do fit for each group
    for igroup in 1..=num_groups {
        let mut nvar = 0i32;
        for ivar in 1..=nallvar {
            if bv.ivar_group[(ivar - 1) as usize] == igroup {
                nvar += 1;
                bv.ivar_pc[(nvar - 1) as usize] = bv.iall_var_pc[(ivar - 1) as usize];
                bv.ind_var[(bv.ivar_pc[(nvar - 1) as usize] - 1) as usize] = nvar;
            }
        }
        //
        // Try to replace bad shifts with second or third best peaks
        if try_alts && nvar > 3 {
            if pickalternativeshifts(
                &bv.ivar_pc,
                &nvar,
                &bv.ind_var,
                dxgridmean,
                dygridmean,
                &bv.ipiece_lower,
                &bv.ipiece_upper,
                &bv.if_skip_edge,
                &bv.lim_edge,
                &bv.iedge_lower,
                &bv.iedge_upper,
                &bv.lim_npc,
                &1,
                &mut bv.altern_disps,
                &2,
                &(4 * bv.lim_edge),
                &3.,
                &0.33,
                &15.,
                Some(&mut bv.iedge_alt_fixed),
                &mut bv.num_alt_fixed,
            ) != 0
            {
                exit_error("Error calling pickAlternativeShifts");
            }
        }
        if nvar > 1 {
            //
            // build matrix of simultaneous equations for minimization solution
            // by matrix inversion or SVD
            let bbx = |j: i32, ivar: i32, ext: &[usize; 2]| {
                (j - 1) as usize + ext[0] * (ivar - 1) as usize
            };
            for ivar in 1..=nvar - 1 {
                let ipc = bv.ivar_pc[(ivar - 1) as usize];
                for m in 1..=nvar - 1 {
                    bv.row_tmp[(m - 1) as usize] = 0.;
                    let k1 = bbx(1, ivar, &bv.bb_ext);
                    let k2 = bbx(2, ivar, &bv.bb_ext);
                    bv.bb[k1] = 0.;
                    bv.bb[k2] = 0.;
                }
                //
                for ixy in 1..=2 {
                    if include_edge(bv, 1, ipc, ixy, &mut iedge) {
                        bv.row_tmp[(ivar - 1) as usize] += 1.;
                        let neighpc = bv.ipiece_lower[e2(iedge, ixy, &bv.ipiece_lower_ext)];
                        let neighvar = bv.ind_var[(neighpc - 1) as usize];
                        //
                        // for a regular neighbor, enter a -1 in its term; but for the
                        // last variable being eliminated, enter a +1 for ALL other
                        // variables instead
                        //
                        if neighvar != nvar {
                            bv.row_tmp[(neighvar - 1) as usize] -= 1.;
                        } else {
                            for m in 1..=nvar - 1 {
                                bv.row_tmp[(m - 1) as usize] += 1.;
                            }
                        }
                        //
                        // when this piece is an upper piece, subtract displacements
                        // from constant term
                        //
                        let k1 = bbx(1, ivar, &bv.bb_ext);
                        bv.bb[k1] -= idir as f32 * dxgridmean[dm(iedge, ixy)];
                        if !do_densities {
                            let k2 = bbx(2, ivar, &bv.bb_ext);
                            bv.bb[k2] -= idir as f32 * dygridmean[dm(iedge, ixy)];
                        }
                    }
                    //
                    if include_edge(bv, 2, ipc, ixy, &mut iedge) {
                        bv.row_tmp[(ivar - 1) as usize] += 1.;
                        let neighpc = bv.ipiece_upper[e2(iedge, ixy, &bv.ipiece_upper_ext)];
                        let neighvar = bv.ind_var[(neighpc - 1) as usize];
                        if neighvar != nvar {
                            bv.row_tmp[(neighvar - 1) as usize] -= 1.;
                        } else {
                            for m in 1..=nvar - 1 {
                                bv.row_tmp[(m - 1) as usize] += 1.;
                            }
                        }
                        //
                        // when a lower piece, add displacements to constant terms
                        //
                        let k1 = bbx(1, ivar, &bv.bb_ext);
                        bv.bb[k1] += idir as f32 * dxgridmean[dm(iedge, ixy)];
                        if !do_densities {
                            let k2 = bbx(2, ivar, &bv.bb_ext);
                            bv.bb[k2] += idir as f32 * dygridmean[dm(iedge, ixy)];
                        }
                    }
                }
                //
                // Load the row data in a if below maxvar
                if nvar <= MAXVAR {
                    for m in 1..=nvar - 1 {
                        a[((m - 1) + MAXVAR * (ivar - 1)) as usize] = bv.row_tmp[(m - 1) as usize];
                    }
                }
            }
            //
            // Solve by iteration first
            if nvar > MAX_GAUSSJ {
                let wallstart = walltime();
                let num_avg_for_test = 10i32;
                let interval_for_test = 100i32;
                let maxiter = 100 + nvar * 10;
                if do_densities {
                    let crit_move_diff = 1.0e-8f32;
                    let crit_max_move = 1.0e-6f32;
                    if findpiecescalings(
                        &bv.ivar_pc,
                        &nvar,
                        &bv.ind_var,
                        &bv.ix_pc_list,
                        &bv.iy_pc_list,
                        dxgridmean,
                        &idir,
                        &bv.ipiece_lower,
                        &bv.ipiece_upper,
                        &bv.if_skip_edge,
                        &bv.lim_edge,
                        &mut bv.dxy_var,
                        &bv.iedge_lower,
                        &bv.iedge_upper,
                        &bv.lim_npc,
                        &mut bv.fps_work,
                        &1,
                        &0,
                        &2,
                        &crit_max_move,
                        &crit_move_diff,
                        &maxiter,
                        &num_avg_for_test,
                        &interval_for_test,
                        &mut i,
                        &mut w_err_mean,
                        &mut w_err_max,
                    ) != 0
                    {
                        exit_error("Calling findPieceScalings");
                    }
                } else {
                    let crit_move_diff = 1.0e-6f32;
                    let crit_max_move = 1.0e-4f32;
                    if findpieceshifts(
                        &bv.ivar_pc,
                        &nvar,
                        &bv.ind_var,
                        &bv.ix_pc_list,
                        &bv.iy_pc_list,
                        dxgridmean,
                        dygridmean,
                        &idir,
                        &bv.ipiece_lower,
                        &bv.ipiece_upper,
                        &bv.if_skip_edge,
                        &bv.lim_edge,
                        &mut bv.dxy_var,
                        &bv.lim_var,
                        &bv.iedge_lower,
                        &bv.iedge_upper,
                        &bv.lim_npc,
                        &mut bv.fps_work,
                        &1,
                        &0,
                        &2,
                        &bv.robust_crit,
                        &crit_max_move,
                        &crit_move_diff,
                        &maxiter,
                        &num_avg_for_test,
                        &interval_for_test,
                        &mut i,
                        &mut w_err_mean,
                        &mut w_err_max,
                    ) != 0
                    {
                        exit_error("Calling findPieceShifts");
                    }
                    //
                    // Look at the list of edge weights, find ones below threshold and put them on
                    // list to fix
                    for ivar in 1..=2 * nvar {
                        let ixy = (ivar - 1) % 2 + 1;
                        let ipc = bv.ivar_pc[((ivar - 1) / 2 + 1 - 1) as usize];
                        let iedge = bv.iedge_upper[e2(ipc, ixy, &bv.iedge_upper_ext)];
                        let w = bv.fps_work[(ivar - 1) as usize];
                        if w < 0.26 && w > -0.5 && iedge > 0 {
                            if bv.if_skip_edge[e2(iedge, ixy, &bv.if_skip_edge_ext)] == 0 {
                                bv.num_low_weight += 1;
                                bv.iedge_low_weight[(bv.num_low_weight - 1) as usize] =
                                    iedge + (ixy - 1) * bv.lim_edge;
                            }
                        }
                    }
                }
                wall_adj = walltime() - wallstart;
            }
            //
            // solve the equations with gaussj if within range
            if nvar <= MAXVAR {
                let wallstart = walltime();
                i = gaussjfw(&mut a, &(nvar - 1), &MAXVAR, &mut bv.bb, &num_col, &2);
                if i > 0 {
                    exit_error("Singular matrix when solving linear equations in gaussj");
                }
                if i < 0 {
                    exit_error("Too many variables to solve linear equations with gaussj");
                }
                wall_gaussj = walltime() - wallstart;
            }
            //
            // Use the iteration solution, do comparisons if both were done
            if nvar > MAX_GAUSSJ {
                let f_edit = |value: f32, w: usize, d: usize| -> String {
                    let text = format!("{value:>w$.d$}");
                    if text.len() > w { "*".repeat(w) } else { text }
                };
                for j in 1..=num_col {
                    let mut xsum = 0.0f32;
                    for i in 1..=nvar - 1 {
                        let kb = bbx(j, i, &bv.bb_ext);
                        let kd = (i - 1) as usize + bv.dxy_var_ext[0] * (j - 1) as usize;
                        if nvar <= MAXVAR {
                            let d = (bv.bb[kb] - bv.dxy_var[kd]).abs();
                            xsum = if xsum > d { xsum } else { d };
                        }
                        bv.bb[kb] = bv.dxy_var[kd];
                    }
                    if nvar <= MAXVAR {
                        let _ = writeln!(
                            ImodFile::Stdout,
                            "axis{:2} max shift adj - gaussj difference{}",
                            j,
                            f_edit(xsum, 15, 7)
                        );
                    }
                }
                if nvar <= MAXVAR {
                    let _ = writeln!(
                        ImodFile::Stdout,
                        "gaussj time{}   shift adj time{}",
                        f_edit(wall_gaussj as f32, 12, 6),
                        f_edit(wall_adj as f32, 12, 6)
                    );
                }
            }
        }
        //
        // take the b values as dx and dy; compute the
        // sum to get the shift for the final piece
        let mut xsum = 0.0f32;
        let mut ysum = 0.0f32;
        for i in 1..=nvar - 1 {
            let k1 = (bv.bb_ext[0] * (i - 1) as usize) as usize;
            let k2 = 1 + k1;
            let ipc = bv.ivar_pc[(i - 1) as usize];
            xsum += bv.bb[k1];
            if do_densities {
                bv.den_solution[(ipc - 1) as usize] = bv.bb[k1];
            } else {
                h[hx(1, 3, ipc)] = bv.bb[k1];
                h[hx(2, 3, ipc)] = bv.bb[k2];
                ysum += bv.bb[k2];
            }
        }
        let ipc_last = bv.ivar_pc[(nvar - 1) as usize];
        if do_densities {
            bv.den_solution[(ipc_last - 1) as usize] = -xsum;
        } else {
            h[hx(1, 3, ipc_last)] = -xsum;
            h[hx(2, 3, ipc_last)] = -ysum;
        }
        //
        // For multiple groups, find the mean displacement between this group
        // and all previous ones and adjust shifts to make that mean be zero
        // but to keep the overall mean zero
        if igroup > 1 {
            let mut dxgroup = 0.0f32;
            let mut dygroup = 0.0f32;
            let mut ndxy = 0i32;
            //
            // rebuild index to variables
            for ivar in 1..=nallvar {
                bv.ind_var[(bv.iall_var_pc[(ivar - 1) as usize] - 1) as usize] = ivar;
            }
            //
            // Loop on pieces in this group, and for each edge to a piece in a
            // lower group, add up the displacement across the edge
            for ivar in 1..=nvar {
                let ipc = bv.ivar_pc[(ivar - 1) as usize];
                for lowup in 1..=2 {
                    for ixy in 1..=2 {
                        if !include_edge(bv, lowup, ipc, ixy, &mut iedge) {
                            if iedge > 0 {
                                let ipclo = if lowup == 1 {
                                    bv.ipiece_lower[e2(iedge, ixy, &bv.ipiece_lower_ext)]
                                } else {
                                    bv.ipiece_upper[e2(iedge, ixy, &bv.ipiece_upper_ext)]
                                };
                                if bv.ivar_group[(bv.ind_var[(ipclo - 1) as usize] - 1) as usize]
                                    < igroup
                                {
                                    if do_densities {
                                        dxgroup = dxgroup + bv.den_solution[(ipc - 1) as usize]
                                            - bv.den_solution[(ipclo - 1) as usize];
                                    } else {
                                        dxgroup = dxgroup + h[hx(1, 3, ipc)] - h[hx(1, 3, ipclo)];
                                        dygroup = dygroup + h[hx(2, 3, ipc)] - h[hx(2, 3, ipclo)];
                                    }
                                    ndxy += 1;
                                }
                            }
                        }
                    }
                }
            }
            //
            // Adjust positions by a weighted fraction of the mean displacement
            if ndxy > 0 {
                dxgroup /= ndxy as f32;
                dygroup /= ndxy as f32;
                for ivar in 1..=nallvar {
                    let ipc = bv.iall_var_pc[(ivar - 1) as usize];
                    let grp = bv.ivar_group[(ivar - 1) as usize];
                    let ip = (ipc - 1) as usize;
                    if do_densities {
                        if grp < igroup {
                            bv.den_solution[ip] +=
                                (dxgroup * nvar as f32) / (nvar + num_prev) as f32;
                        } else if grp == igroup {
                            bv.den_solution[ip] -=
                                (dxgroup * num_prev as f32) / (nvar + num_prev) as f32;
                        }
                    } else if grp < igroup {
                        h[hx(1, 3, ipc)] += (dxgroup * nvar as f32) / (nvar + num_prev) as f32;
                        h[hx(2, 3, ipc)] += (dygroup * nvar as f32) / (nvar + num_prev) as f32;
                    } else if grp == igroup {
                        h[hx(1, 3, ipc)] -= (dxgroup * num_prev as f32) / (nvar + num_prev) as f32;
                        h[hx(2, 3, ipc)] -= (dygroup * num_prev as f32) / (nvar + num_prev) as f32;
                    }
                }
            }
        }
        num_prev += nvar;
    }
    //
    // compute and return the results
    //
    for ivar in 1..=nallvar {
        let ipc = bv.iall_var_pc[(ivar - 1) as usize];
        if !do_densities {
            let off = bv.hinv_ext[0] * bv.hinv_ext[1] * (ipc - 1) as usize;
            xfinvert(&h[hx(1, 1, ipc)..], &mut bv.hinv[off..]);
        }
        for ixy in 1..=2 {
            if include_edge(bv, 1, ipc, ixy, &mut iedge) {
                let ipclo = bv.ipiece_lower[e2(iedge, ixy, &bv.ipiece_lower_ext)];
                let (dxm, dym) = (dxgridmean[dm(iedge, ixy)], dygridmean[dm(iedge, ixy)]);
                let bdist = (dxm * dxm + dym * dym).sqrt();
                bsum += bdist;
                let adist = if do_densities {
                    (dxm + bv.den_solution[(ipc - 1) as usize]
                        - bv.den_solution[(ipclo - 1) as usize])
                        .abs()
                } else {
                    let ax = idir as f32 * dxm + h[hx(1, 3, ipc)] - h[hx(1, 3, ipclo)];
                    let ay = idir as f32 * dym + h[hx(2, 3, ipc)] - h[hx(2, 3, ipclo)];
                    (ax * ax + ay * ay).sqrt()
                };
                asum += adist;
                *amax = if *amax > adist { *amax } else { adist };
                *bmax = if *bmax > bdist { *bmax } else { bdist };
                *nsum += 1;
            }
        }
    }
    *bavg = bsum / *nsum as f32;
    *aavg = asum / *nsum as f32;
}

/// Original: `logical function includeEdge(lowup, kpc, kxy, ked)`, internal
/// to `find_best_shifts` (`bsubs.f90:2532`).
///
/// Whether an edge is included: `lowup` 1/2 for lower/upper edge of piece
/// `kpc` in direction `kxy`; the edge number, if any, is returned in `ked`.
pub fn include_edge(bv: &BlendVars, lowup: i32, kpc: i32, kxy: i32, ked: &mut i32) -> bool {
    if lowup == 1 {
        *ked = bv.iedge_lower[(kpc - 1) as usize + bv.iedge_lower_ext[0] * (kxy - 1) as usize];
    } else {
        *ked = bv.iedge_upper[(kpc - 1) as usize + bv.iedge_upper_ext[0] * (kxy - 1) as usize];
    }
    if *ked == 0 {
        return false;
    }
    if bv.if_skip_edge[(*ked - 1) as usize + bv.if_skip_edge_ext[0] * (kxy - 1) as usize] > 1 {
        return false;
    }
    true
}

/// Original: `subroutine checkGroup(kpc)`, internal to `find_best_shifts`
/// (`bsubs.f90:2549`).  Adds piece `kpc` to the current group
/// (`numGroups`) and the check list if it is not in a group yet; `numGroups`
/// and `numToCheck` are the host variables.
pub fn check_group(bv: &mut BlendVars, kpc: i32, num_groups: i32, num_to_check: &mut i32) {
    let k = (bv.ind_var[(kpc - 1) as usize] - 1) as usize;
    if bv.ivar_group[k] > 0 {
        return;
    }
    bv.ivar_group[k] = num_groups;
    *num_to_check += 1;
    bv.list_check[(*num_to_check - 1) as usize] = kpc;
}

/// Original: `subroutine sortVarsIntoGroups()`, internal to
/// `find_best_shifts` (`bsubs.f90:2563`).
///
/// Classifies pieces into separate groups by following connections between
/// them, then orders the groups so each is connected to one done before it.
/// Of the host variables it uses, only `nallvar` (read) and `numGroups`
/// (written) are read by the host afterwards; the rest are scratch and are
/// locals here.
pub fn sort_vars_into_groups(bv: &mut BlendVars, nallvar: i32, num_groups: &mut i32) {
    let mut iedge = 0i32;
    let e2 = |i: i32, ixy: i32, ext: &[usize; 2]| (i - 1) as usize + ext[0] * (ixy - 1) as usize;
    *num_groups = 0;
    let mut nextvar = 1i32;
    while nextvar <= nallvar {
        //
        // Look for next piece that is unclassified
        while nextvar <= nallvar {
            if bv.ivar_group[(nextvar - 1) as usize] == 0 {
                break;
            }
            nextvar += 1;
        }
        if nextvar > nallvar {
            break;
        }
        //
        // Start a new group and initialize check list
        *num_groups += 1;
        bv.ivar_group[(nextvar - 1) as usize] = *num_groups;
        let mut next_check = 1i32;
        let mut num_to_check = 1i32;
        bv.list_check[0] = bv.iall_var_pc[(nextvar - 1) as usize];
        //
        // Loop on the check list, for next piece, check its 4 edges and
        // assign and add to check list other pieces not in group yet
        while next_check <= num_to_check {
            let ipc = bv.list_check[(next_check - 1) as usize];
            for ixy in 1..=2 {
                if include_edge(bv, 1, ipc, ixy, &mut iedge) {
                    let p = bv.ipiece_lower[e2(iedge, ixy, &bv.ipiece_lower_ext)];
                    check_group(bv, p, *num_groups, &mut num_to_check);
                }
                if include_edge(bv, 2, ipc, ixy, &mut iedge) {
                    let p = bv.ipiece_upper[e2(iedge, ixy, &bv.ipiece_upper_ext)];
                    check_group(bv, p, *num_groups, &mut num_to_check);
                }
            }
            next_check += 1;
        }
        nextvar += 1;
    }
    //
    // Have to make sure the groups are going to be done in a connected order
    // if there are more than 2
    for igroup in 1..=*num_groups - 2 {
        let mut new_group = 0i32;
        for ivar in 1..=nallvar {
            //
            // Look at pieces in all the groups already done
            if bv.ivar_group[(ivar - 1) as usize] <= igroup {
                let ipc = bv.iall_var_pc[(ivar - 1) as usize];
                //
                // Look for an excluded edge to a piece in a higher group
                for ixy in 1..=2 {
                    if !include_edge(bv, 1, ipc, ixy, &mut iedge) {
                        if iedge > 0 {
                            let ipclo = bv.ipiece_lower[e2(iedge, ixy, &bv.ipiece_lower_ext)];
                            let m = bv.ivar_group[(bv.ind_var[(ipclo - 1) as usize] - 1) as usize];
                            if m > igroup {
                                new_group = m;
                            }
                        }
                    }
                    if !include_edge(bv, 2, ipc, ixy, &mut iedge) {
                        if iedge > 0 {
                            let ipclo = bv.ipiece_upper[e2(iedge, ixy, &bv.ipiece_upper_ext)];
                            let m = bv.ivar_group[(bv.ind_var[(ipclo - 1) as usize] - 1) as usize];
                            if m > igroup {
                                new_group = m;
                            }
                        }
                    }
                }
                //
                // Swap the group numbers of that group and the next one
                if new_group > 0 {
                    for i in 0..nallvar as usize {
                        if bv.ivar_group[i] == igroup + 1 {
                            bv.ivar_group[i] = new_group;
                        } else if bv.ivar_group[i] == new_group {
                            bv.ivar_group[i] = igroup + 1;
                        }
                    }
                    break;
                }
            }
        }
    }
}

/// Original: `common /funccom/ nedg, ifTrace, nTrial, izedge, errMin`
/// (`bsubs.f90:2667`, `:2778`), shared by `findBestGradient` and `gradfunc`.
/// `findBestGradient` sets every member before `gradfunc` reads it, so its
/// owner can hold it for the one search.
#[derive(Debug, Default, Clone, Copy)]
pub struct FuncCom {
    pub nedg: i32,
    pub if_trace: i32,
    pub n_trial: i32,
    pub izedge: i32,
    pub err_min: f32,
}

/// Original: `subroutine findBestGradient(dxgridmean, dygridmean, idimEdge,
/// idir, izsect, gradnew, rotnew)` (`bsubs.f90:2647`).
///
/// Finds the incremental magnification gradient and rotation that minimize
/// the mean edge error after shift solving, by simplex search over
/// `gradfunc`.
pub fn find_best_gradient(
    bv: &mut BlendVars,
    dxgridmean: &[f32],
    dygridmean: &[f32],
    idim_edge: i32,
    idir: i32,
    izsect: i32,
    gradnew: &mut f32,
    rotnew: &mut f32,
) {
    //
    // Stuff for amoeba: ftol2 and ptol2 are used the FIRST time
    //
    const NVAR: usize = 2;
    let mut pp = [0.0f32; (NVAR + 1) * (NVAR + 1)];
    let mut yy = [0.0f32; NVAR + 1];
    let mut ptol = [0.0f32; NVAR];
    let da: [f32; NVAR] = [0.5, 0.2];
    let mut var = [0.0f32; NVAR];
    let mut iter = 0i32;
    let mut jmin = 0usize;
    let mut fc = FuncCom::default();
    let dm = |iedge: i32, ixy: i32| ((iedge - 1) + idim_edge * (ixy - 1)) as usize;
    let e2 = |i: i32, ixy: i32, ext: &[usize; 2]| (i - 1) as usize + ext[0] * (ixy - 1) as usize;
    let nxin = bv.nxyz_in[0];
    let nyin = bv.nxyz_in[1];
    let n_xoverlap = bv.n_overlap[0];
    let ny_overlap = bv.n_overlap[1];

    fc.if_trace = 0;
    let ptol1 = 1.0e-5f32;
    let ftol1 = 1.0e-5f32;
    let ptol2 = 1.0e-3f32;
    let ftol2 = 1.0e-3f32;
    let delfac = 2.0f32;

    fc.izedge = izsect;
    fc.nedg = 0;
    for ipc in 1..=bv.npc_list {
        if bv.iz_pc_list[(ipc - 1) as usize] == izsect
            && (bv.iedge_lower[e2(ipc, 1, &bv.iedge_lower_ext)] > 0
                || bv.iedge_lower[e2(ipc, 2, &bv.iedge_lower_ext)] > 0)
        {
            for ixy in 1..=2 {
                let iedge = bv.iedge_lower[e2(ipc, ixy, &bv.iedge_lower_ext)];
                if iedge > 0 {
                    fc.nedg += 1;
                    let n = (fc.nedg - 1) as usize;
                    let ipclo = bv.ipiece_lower[e2(iedge, ixy, &bv.ipiece_lower_ext)];
                    //
                    // Get the residual at the edge: the amount the upper piece
                    // is still displaced away from alignment with lower
                    //
                    let kx = e2(iedge, ixy, &bv.dx_edge_ext);
                    bv.dx_edge[kx] = idir as f32 * dxgridmean[dm(iedge, ixy)];
                    let ky = e2(iedge, ixy, &bv.dy_edge_ext);
                    bv.dy_edge[ky] = idir as f32 * dygridmean[dm(iedge, ixy)];
                    //
                    // Get center for gradient in each piece
                    //
                    if bv.focus_adjusted {
                        bv.grad_xcen_lo[n] = nxin as f32 / 2.;
                        bv.grad_ycen_lo[n] = nyin as f32 / 2.;
                        bv.grad_xcen_hi[n] = nxin as f32 / 2.;
                        bv.grad_ycen_hi[n] = nyin as f32 / 2.;
                    } else {
                        let xc = (bv.min_xpiece + bv.nx_pieces * (nxin - n_xoverlap) + n_xoverlap)
                            as f32
                            / 2.;
                        let yc = (bv.min_ypiece + bv.ny_pieces * (nyin - ny_overlap) + ny_overlap)
                            as f32
                            / 2.;
                        bv.grad_xcen_lo[n] = xc - bv.ix_pc_list[(ipclo - 1) as usize] as f32;
                        bv.grad_ycen_lo[n] = yc - bv.iy_pc_list[(ipclo - 1) as usize] as f32;
                        bv.grad_xcen_hi[n] = xc - bv.ix_pc_list[(ipc - 1) as usize] as f32;
                        bv.grad_ycen_hi[n] = yc - bv.iy_pc_list[(ipc - 1) as usize] as f32;
                    }
                    //
                    // Get center point of overlap zone in each piece
                    //
                    if ixy == 1 {
                        bv.over_xcen_lo[n] = (nxin - n_xoverlap / 2) as f32;
                        bv.over_xcen_hi[n] = (n_xoverlap / 2) as f32;
                        bv.over_ycen_lo[n] = (nyin / 2) as f32;
                        bv.over_ycen_hi[n] = (nyin / 2) as f32;
                    } else {
                        bv.over_ycen_lo[n] = (nyin - ny_overlap / 2) as f32;
                        bv.over_ycen_hi[n] = (ny_overlap / 2) as f32;
                        bv.over_xcen_lo[n] = (nxin / 2) as f32;
                        bv.over_xcen_hi[n] = (nxin / 2) as f32;
                    }
                }
            }
        }
    }

    //
    // set up for minimization
    //
    fc.err_min = 1.0e30;
    fc.n_trial = 0;
    var[0] = 0.;
    var[1] = 0.;
    {
        let mut func = |p: &[f32]| -> f32 {
            let mut err = 0.0f32;
            gradfunc(bv, &mut fc, p, &mut err);
            err
        };
        amoebainitfwrap(
            &mut pp,
            &mut yy,
            NVAR + 1,
            NVAR,
            delfac,
            ptol2,
            &var,
            &da,
            &mut func,
            &mut ptol,
        );
        amoebafwrap(
            &mut pp,
            &mut yy,
            NVAR + 1,
            NVAR,
            ftol2,
            &mut func,
            &mut iter,
            &ptol,
            &mut jmin,
        );
        //
        // per Press et al. recommendation, just restart at current location
        //
        for i in 0..NVAR {
            var[i] = pp[(jmin - 1) + (NVAR + 1) * i];
        }
        amoebainitfwrap(
            &mut pp,
            &mut yy,
            NVAR + 1,
            NVAR,
            delfac / 4.,
            ptol1,
            &var,
            &da,
            &mut func,
            &mut ptol,
        );
        amoebafwrap(
            &mut pp,
            &mut yy,
            NVAR + 1,
            NVAR,
            ftol1,
            &mut func,
            &mut iter,
            &ptol,
            &mut jmin,
        );
    }
    //
    // recover result
    //
    for i in 0..NVAR {
        var[i] = pp[(jmin - 1) + (NVAR + 1) * i];
    }
    // `call gradfunc(var, errMin)`: the output is the common-block member
    // itself; `gradfunc` reads `errMin` only under `ifTrace > 0`, which is 0
    // here, so writing it back after the call is the same.
    let mut err_min = fc.err_min;
    gradfunc(bv, &mut fc, &var, &mut err_min);
    fc.err_min = err_min;
    // `73 format(' Implied incremental gradient:',2f9.4,'  mean error:',f10.4)`
    let f_edit = |value: f32, w: usize, d: usize| -> String {
        let text = format!("{value:>w$.d$}");
        if text.len() > w { "*".repeat(w) } else { text }
    };
    let _ = writeln!(
        ImodFile::Stdout,
        " Implied incremental gradient:{}{}  mean error:{}",
        f_edit(var[0], 9, 4),
        f_edit(var[1], 9, 4),
        f_edit(fc.err_min, 10, 4)
    );
    let _ = ImodFile::Stdout.flush();
    *gradnew = var[0];
    *rotnew = var[1];
}

/// Original: `subroutine gradfunc(p, funcErr)` (`bsubs.f90:2771`).
///
/// The error function for `findBestGradient`: adjusts each edge
/// displacement for the incremental gradient `p(1)` and rotation `p(2)`,
/// solves for shifts, and returns the mean residual in `funcErr`.  `fc` is
/// the `/funccom/` common block.
pub fn gradfunc(bv: &mut BlendVars, fc: &mut FuncCom, p: &[f32], func_err: &mut f32) {
    let (mut dxlo, mut dylo, mut dxhi, mut dyhi) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut bmean, mut bmax, mut aftmean, mut aftmax) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let e2 = |i: i32, ixy: i32, ext: &[usize; 2]| (i - 1) as usize + ext[0] * (ixy - 1) as usize;
    let nxin = bv.nxyz_in[0];
    let nyin = bv.nxyz_in[1];

    // `tiltAngles(min(ilistz, numAngles))`: with no tilt angles (blendmont
    // `-test` without `-gradient`/`-tiltfile`) this is element 0, the 4 bytes
    // in front of the allocation -- the high half of the malloc size word,
    // 0 for any allocation under 4 GB.  The translation uses 0 there.
    let iang = bv.ilistz.min(bv.num_angles);
    let tiltang = if iang >= 1 {
        bv.tilt_angles[(iang - 1) as usize]
    } else {
        0.0
    };
    fc.n_trial += 1;

    let mut ied = 0i32;
    for ipc in 1..=bv.npc_list {
        if bv.iz_pc_list[(ipc - 1) as usize] == fc.izedge
            && (bv.iedge_lower[e2(ipc, 1, &bv.iedge_lower_ext)] > 0
                || bv.iedge_lower[e2(ipc, 2, &bv.iedge_lower_ext)] > 0)
        {
            for ixy in 1..=2 {
                let iedge = bv.iedge_lower[e2(ipc, ixy, &bv.iedge_lower_ext)];
                if iedge > 0 {
                    ied += 1;
                    let n = (ied - 1) as usize;
                    mag_gradient_shift(
                        bv.over_xcen_lo[n],
                        bv.over_ycen_lo[n],
                        nxin,
                        nyin,
                        bv.grad_xcen_lo[n],
                        bv.grad_ycen_lo[n],
                        bv.pixel_mag_grad,
                        bv.axis_rot,
                        tiltang,
                        p[0],
                        p[1],
                        &mut dxlo,
                        &mut dylo,
                    );
                    mag_gradient_shift(
                        bv.over_xcen_hi[n],
                        bv.over_ycen_hi[n],
                        nxin,
                        nyin,
                        bv.grad_xcen_hi[n],
                        bv.grad_ycen_hi[n],
                        bv.pixel_mag_grad,
                        bv.axis_rot,
                        tiltang,
                        p[0],
                        p[1],
                        &mut dxhi,
                        &mut dyhi,
                    );
                    //
                    // Point moves by negative of shift to get to undistorted image
                    // Displacement changes by negative of upper shift and positive
                    // of  lower shift
                    //
                    let ka = e2(iedge, ixy, &bv.dx_adj_ext);
                    bv.dx_adj[ka] = bv.dx_edge[e2(iedge, ixy, &bv.dx_edge_ext)] + dxlo - dxhi;
                    let kb = e2(iedge, ixy, &bv.dy_adj_ext);
                    bv.dy_adj[kb] = bv.dy_edge[e2(iedge, ixy, &bv.dy_edge_ext)] + dylo - dyhi;
                }
            }
        }
    }

    // `find_best_shifts(dxadj, dyadj, limedge, 1, izedge, htmp, iedge, ...)`:
    // the three module arrays are lent out of the module for the call
    // (`find_best_shifts` does not touch them through the module).
    let mut dx_adj = std::mem::take(&mut bv.dx_adj);
    let mut dy_adj = std::mem::take(&mut bv.dy_adj);
    let mut htmp = std::mem::take(&mut bv.htmp);
    let mut iedge = 0i32;
    let lim_edge = bv.lim_edge;
    find_best_shifts(
        bv,
        &mut dx_adj,
        &mut dy_adj,
        lim_edge,
        1,
        fc.izedge,
        &mut htmp,
        &mut iedge,
        &mut bmean,
        &mut bmax,
        &mut aftmean,
        &mut aftmax,
        false,
    );
    bv.dx_adj = dx_adj;
    bv.dy_adj = dy_adj;
    bv.htmp = htmp;

    *func_err = aftmean;
    if fc.if_trace > 0 {
        let mut starout = ' ';
        if *func_err < fc.err_min {
            starout = '*';
            fc.err_min = *func_err;
        }
        if fc.if_trace > 1 || starout == '*' {
            // `72 format(1x,a1,i4,f15.5, 2f9.4)`
            let f_edit = |value: f32, w: usize, d: usize| -> String {
                let text = format!("{value:>w$.d$}");
                if text.len() > w { "*".repeat(w) } else { text }
            };
            let _ = writeln!(
                ImodFile::Stdout,
                " {}{:4}{}{}{}",
                starout,
                fc.n_trial,
                f_edit(*func_err, 15, 5),
                f_edit(p[0], 9, 4),
                f_edit(p[1], 9, 4)
            );
        }
        let _ = ImodFile::Stdout.flush();
    }
    //
    // Amoeba flails on hard zeros so add something (no longer, 6/17/06)
    //
}

/// Original: `subroutine findEdgeToUse(iedge, ixy, iuse)` (`bsubs.f90:2844`).
///
/// Looks up an edge to use in place of `iedge` when Z limits are set on the
/// edges to use: the original edge when no limits apply or no substitute is
/// needed, 0 when a different Z should be used but has no edge.
pub fn find_edge_to_use(bv: &BlendVars, iedge: i32, ixy: i32, iuse: &mut i32) {
    let nxin = bv.nxyz_in[0];
    let nyin = bv.nxyz_in[1];
    *iuse = iedge;
    if bv.num_use_edge == 0 && bv.iz_use_def_low < 0 {
        return;
    }
    //
    // Find frame number of piece then look up frame in the use list
    let ipclo = bv.ipiece_lower[(iedge - 1) as usize + bv.ipiece_lower_ext[0] * (ixy - 1) as usize];
    let ip = (ipclo - 1) as usize;
    let ixfrm = 1 + (bv.ix_pc_list[ip] - bv.min_xpiece) / (nxin - bv.n_overlap[0]);
    let iyfrm = 1 + (bv.iy_pc_list[ip] - bv.min_ypiece) / (nyin - bv.n_overlap[1]);
    let mut izlow = -1i32;
    let mut izhigh = 0i32;
    for i in 0..bv.num_use_edge as usize {
        if bv.ix_frm_use_edge[i] == ixfrm
            && bv.iy_frm_use_edge[i] == iyfrm
            && ixy == bv.ixy_use_edge[i]
        {
            izlow = bv.iz_low_use[i];
            izhigh = bv.iz_high_use[i];
        }
    }
    //
    // If frame not found, use the default or skip out if none
    // Then find out if there is another Z to use or not
    if izlow < 0 {
        if bv.iz_use_def_low < 0 {
            return;
        }
        izlow = bv.iz_use_def_low;
        izhigh = bv.iz_use_def_high;
    }
    let izuse = if bv.iz_pc_list[ip] < izlow {
        izlow
    } else if bv.iz_pc_list[ip] > izhigh {
        izhigh
    } else {
        return;
    };
    //
    // Seek these coordinates on that Z with an edge above
    for i in 1..=bv.npc_list {
        let k = (i - 1) as usize;
        let up = bv.iedge_upper[k + bv.iedge_upper_ext[0] * (ixy - 1) as usize];
        if bv.iz_pc_list[k] == izuse
            && bv.ix_pc_list[k] == bv.ix_pc_list[ip]
            && bv.iy_pc_list[k] == bv.iy_pc_list[ip]
            && up > 0
        {
            *iuse = up;
            return;
        }
    }
    *iuse = 0;
}

/// Original: `subroutine getDataLimits(ipc, ixy, lohi, limitLo, limitHi)`
/// (`bsubs.f90:2900`).
///
/// Returns the limits of real (not gray fill) data in dimension `ixy` for
/// an edge on side `lohi` (1 lower, 2 upper) of piece `ipc`, numbered from
/// 0, from the `limDataLo`/`limDataHi` cache or by scanning the piece.
pub fn get_data_limits(
    bv: &mut BlendVars,
    ipc: i32,
    ixy: i32,
    lohi: i32,
    limit_lo: &mut i32,
    limit_hi: &mut i32,
) {
    let nxin = bv.nxyz_in[0];
    let nyin = bv.nxyz_in[1];
    let lim_ind = bv.lim_data_ind[(ipc - 1) as usize];
    if lim_ind <= 0 {
        *limit_lo = 0;
        *limit_hi = nxin - 1;
        if ixy == 2 {
            *limit_hi = nyin - 1;
        }
        return;
    }
    let ext = bv.lim_data_lo_ext;
    let kl = (lim_ind - 1) as usize + ext[0] * ((ixy - 1) as usize + ext[1] * (lohi - 1) as usize);
    let ext = bv.lim_data_hi_ext;
    let kh = (lim_ind - 1) as usize + ext[0] * ((ixy - 1) as usize + ext[1] * (lohi - 1) as usize);
    if bv.lim_data_lo[kl] >= 0 && bv.lim_data_hi[kh] >= 0 {
        *limit_lo = bv.lim_data_lo[kl];
        *limit_hi = bv.lim_data_hi[kh];
        return;
    }
    let mut ind = 0i32;
    shuffler(bv, ipc, &mut ind);
    let arr = |k: i32| bv.array[(k - 1) as usize];

    let mut line_end = 0i32;
    let inc_pix: i32;
    let inc_line: i32;
    let num_pix: i32;
    let mut i_pix_end: i32;
    let mut line_start: i32;
    if ixy == 1 {
        inc_pix = nxin;
        inc_line = 1;
        num_pix = nyin.min((3 * bv.n_overlap[1]) / 2);
        i_pix_end = nyin - 1;
        line_start = nxin - 1;
    } else {
        inc_pix = 1;
        inc_line = nxin;
        num_pix = nxin.min((3 * bv.n_overlap[0]) / 2);
        i_pix_end = nxin - 1;
        line_start = nyin - 1;
    }
    let i_pix_str: i32;
    if lohi == 1 {
        i_pix_str = 0;
        i_pix_end = num_pix - 1;
    } else {
        i_pix_str = i_pix_end + 1 - num_pix;
    }
    let max_same = num_pix / 4;
    for idir in [-1i32, 1] {
        //
        // First find out if first line has a common value
        let mut i = i_pix_str + 1;
        let mut line_base = ind + line_start * inc_line;
        let value = arr(line_base + i_pix_str * inc_pix);
        while i <= i_pix_end {
            if arr(line_base + i * inc_pix) != value {
                break;
            }
            i += 1;
        }
        let mut line_good: i32;
        if i <= i_pix_end {
            line_good = line_start;
        } else {
            //
            // If it made it to the end, next search for line with less than
            // maximum number of pixels at this value
            line_good = -1;
            let mut line = line_start + idir;
            while line_good < 0 && idir * (line - line_end) < 0 {
                i = i_pix_str;
                let mut num_same = 0i32;
                line_base = ind + line * inc_line;
                while i <= i_pix_end && num_same < max_same {
                    if arr(line_base + i * inc_pix) == value {
                        num_same += 1;
                    }
                    i += 1;
                }
                if num_same < max_same {
                    line_good = line;
                }
                line += idir;
            }
        }
        //
        // If go to end, set it to next to last line, then save limit
        if line_good < 0 {
            line_good = line_end - idir;
        }

        if idir == -1 {
            *limit_hi = line_good;
        } else {
            *limit_lo = line_good;
        }
        line_end = line_start;
        line_start = 0;
    }
    bv.lim_data_lo[kl] = *limit_lo;
    bv.lim_data_hi[kh] = *limit_hi;
}

/// Original: `subroutine iwrBinned(iunit, array, brray, nx, nxout, ixst, ny,
/// nyout, iyst, iBinning, dmin, dmax, dsum8)` (`bsubs.f90:3000`).
///
/// Writes the `ny` lines of length `nx` in `array` to unit `iunit` with
/// binning `iBinning`, `nyout` lines of `nxout`, using `brray` as a scratch
/// line; `ixst`/`iyst` are starting pixels (negative for non-existent
/// pixels).  Maintains `dmin`, `dmax` and the sum `dsum8`.
pub fn iwr_binned(
    iunit: i32,
    array: &mut [f32],
    brray: &mut [f32],
    nx: i32,
    nxout: i32,
    ixst: i32,
    ny: i32,
    nyout: i32,
    iyst: i32,
    i_binning: i32,
    dmin: &mut f32,
    dmax: &mut f32,
    dsum8: &mut f64,
) {
    let a = |i: i32, j: i32| ((i - 1) + nx * (j - 1)) as usize;
    //
    if i_binning == 1 {
        for iy in 1..=nyout {
            for ix in 1..=nxout {
                let v = array[a(ix, iy)];
                *dmin = if *dmin < v { *dmin } else { v };
                *dmax = if *dmax > v { *dmax } else { v };
                *dsum8 += v as f64;
            }
            unsafe { par_wrt_lin(iunit, array[a(1, iy)..].as_mut_ptr().cast()) };
        }
        return;
    }

    for iy in 1..=nyout {
        let jybase = (iy - 1) * i_binning + iyst;
        for ix in 1..=nxout {
            let jxbase = (ix - 1) * i_binning + ixst;
            let mut sum = 0.0f32;
            let k = (ix - 1) as usize;
            if ix == 1 || ix == nxout || iy == 1 || iy == nyout {
                let mut nsum = 0i32;
                for jy in 1.max(jybase + 1)..=ny.min(jybase + i_binning) {
                    for jx in 1.max(jxbase + 1)..=nx.min(jxbase + i_binning) {
                        sum += array[a(jx, jy)];
                        nsum += 1;
                    }
                }
                brray[k] = sum / nsum as f32;
            } else {
                for jy in jybase + 1..=jybase + i_binning {
                    for jx in jxbase + 1..=jxbase + i_binning {
                        sum += array[a(jx, jy)];
                    }
                }
                brray[k] = sum / (i_binning * i_binning) as f32;
            }
            *dmin = if *dmin < brray[k] { *dmin } else { brray[k] };
            *dmax = if *dmax > brray[k] { *dmax } else { brray[k] };
            *dsum8 += brray[k] as f64;
        }
        unsafe { par_wrt_lin(iunit, brray.as_mut_ptr().cast()) };
    }
}

/// Original: `subroutine getExtraIndents(ipclow, ipcup, ixy, delIndent)`
/// (`bsubs.f90:3060`).
///
/// Computes the extra indent for correlations and edge functions when there
/// are distortion corrections, from the largest distortion vector along the
/// edges of pieces `ipclow` and `ipcup`.
pub fn get_extra_indents(
    bv: &BlendVars,
    ipclow: i32,
    ipcup: i32,
    ixy: i32,
    del_indent: &mut [f32; 2],
) {
    //
    del_indent[0] = 0.;
    del_indent[1] = 0.;
    if !bv.do_fields {
        return;
    }
    let memlow = bv.mem_index[(ipclow - 1) as usize];
    let memup = bv.mem_index[(ipcup - 1) as usize];
    let fx = |i: i32, j: i32, k: i32| {
        let e = &bv.field_dx_ext;
        bv.field_dx[(i - 1) as usize + e[0] * ((j - 1) as usize + e[1] * (k - 1) as usize)]
    };
    let fy = |i: i32, j: i32, k: i32| {
        let e = &bv.field_dy_ext;
        bv.field_dy[(i - 1) as usize + e[0] * ((j - 1) as usize + e[1] * (k - 1) as usize)]
    };
    let fmax = |vals: &[f32]| {
        let mut m = vals[0];
        for &v in &vals[1..] {
            m = if m > v { m } else { v };
        }
        m
    };
    let (nxf, nyf) = (bv.nx_field, bv.ny_field);
    //
    // The undistorted image moves in the direction opposite to the
    // field vector, so positive vectors at the right edge of the lower
    // piece move the border in to left and require more indent in
    // short direction.
    //
    if ixy == 1 {
        for iy in 1..=nyf {
            del_indent[0] = fmax(&[del_indent[0], fx(nxf, iy, memlow), -fx(1, iy, memup)]);
        }
        del_indent[1] = fmax(&[
            0.,
            -fy(nxf, 1, memlow),
            -fy(1, 1, memup),
            fy(nxf, nyf, memlow),
            fy(1, nyf, memup),
        ]);
    } else {
        for ix in 1..=nxf {
            del_indent[0] = fmax(&[del_indent[0], fy(ix, nyf, memlow), -fy(ix, 1, memup)]);
        }
        del_indent[1] = fmax(&[
            0.,
            -fx(1, nyf, memlow),
            -fx(1, 1, memup),
            fx(nxf, nyf, memlow),
            fx(nxf, 1, memup),
        ]);
    }
}

/// Original: `subroutine readExclusionModel(filnam, edgedispx, edgedispy,
/// idimedge, ifUseAdjusted, mapAllPc, nxmap, nymap, minzpc, numSkip)`
/// (`bsubs.f90:3105`).
///
/// Reads a model of edges to exclude and marks each edge with a model point
/// in the quadrangle nearest it in `ifSkipEdge`; `numSkip` returns the number
/// of edges skipped.  `mapAllPc` is `(nxmap, nymap, *)` column-major; `fm` is
/// the `fortmodel` module.
pub fn read_exclusion_model(
    bv: &mut BlendVars,
    fm: &mut FortModel,
    filnam: &str,
    edgedispx: &[f32],
    edgedispy: &[f32],
    idimedge: i32,
    if_use_adjusted: i32,
    map_all_pc: &[i32],
    nxmap: i32,
    nymap: i32,
    minzpc: i32,
    num_skip: &mut i32,
) {
    // `real*4 vertex(4,2)`, dimension-reversed: `vertex(i, j)` is
    // `vertex[j - 1][i - 1]`, so `vertex(1,1)`/`vertex(1,2)` are the X and Y
    // columns `inside` takes.
    let mut vertex = [[0.0f32; 4]; 2];
    let map = |ix: i32, iy: i32, iz: i32| {
        map_all_pc[((ix - 1) + nxmap * ((iy - 1) + nymap * (iz - 1))) as usize]
    };
    let nxin = bv.nxyz_in[0];
    let nyin = bv.nxyz_in[1];
    let n_xoverlap = bv.n_overlap[0];
    let ny_overlap = bv.n_overlap[1];
    //
    fm.fm_mod_size_type = 2;
    let exist = readw_or_imod(filnam, fm);
    if !exist {
        exit_error("Reading edge exclusion model file");
    }
    scale_model_to_image(1, 0, fm);
    let mut num_efonly = 0i32;
    *num_skip = 0;
    let mut num_near = 0i32;
    //
    // Loop on the edges, make a quadrangle for area nearest to each
    for ixy in 1..=2 {
        for ied in 1..=bv.nedge[(ixy - 1) as usize] {
            let ipc =
                bv.ipiece_upper[(ied - 1) as usize + bv.ipiece_upper_ext[0] * (ixy - 1) as usize];
            let ip = (ipc - 1) as usize;
            let mut ixpc = bv.ix_pc_list[ip];
            let mut iypc = bv.iy_pc_list[ip];
            let izpc = bv.iz_pc_list[ip] + 1 - minzpc;
            let mut lenx = nxin - n_xoverlap;
            let mut leny = nyin - ny_overlap;
            let ixframe = 1 + (ixpc - bv.min_xpiece) / lenx;
            let iyframe = 1 + (iypc - bv.min_ypiece) / leny;
            let mut ixright = ixpc + lenx;
            let mut iytop = iypc + leny;
            //
            // WARNING: This code tracks how 3dmod lays out data when displaying
            // a montage: it copies each entire piece into the buffer in Z order
            // A piece extends farther in one direction if it is either the last
            // piece or it is laid down after the next in that direction
            if ixframe == bv.nx_pieces {
                ixright = ixpc + nxin;
            } else if map(ixframe + 1, iyframe, izpc) < ipc {
                ixright = ixpc + nxin;
            }
            if iyframe == bv.ny_pieces {
                iytop = iypc + nyin;
            } else if map(ixframe, iyframe + 1, izpc) < ipc {
                iytop = iypc + nyin;
            }
            //
            // A piece starts farther in one direction if the piece before it
            // is laid down after this piece
            if ixframe > 1 {
                if map(ixframe - 1, iyframe, izpc) > ipc {
                    ixpc += n_xoverlap;
                }
            }
            if iyframe > 1 {
                if map(ixframe, iyframe - 1, izpc) > ipc {
                    iypc += ny_overlap;
                }
            }
            lenx = ixright - ixpc;
            leny = iytop - iypc;
            //
            // This puts the vertices on the edge at its visible ends and one
            // vertex in the middle of this visible piece; the other vertex is
            // just a fixed distance into the piece below
            vertex[0][0] = ixpc as f32 + lenx as f32 / 2.;
            vertex[1][0] = iypc as f32 + leny as f32 / 2.;
            vertex[0][1] = ixpc as f32;
            vertex[1][1] = iypc as f32;
            if ixy == 1 {
                vertex[0][2] = ixpc as f32 - (nxin - n_xoverlap) as f32 / 2.;
                vertex[1][2] = vertex[1][0];
                vertex[0][3] = vertex[0][1];
                vertex[1][3] = vertex[1][1] + leny as f32;
            } else {
                vertex[0][2] = vertex[0][0];
                vertex[1][2] = iypc as f32 - (nyin - ny_overlap) as f32 / 2.;
                vertex[0][3] = vertex[0][1] + lenx as f32;
                vertex[1][3] = vertex[1][1];
            }
            //
            // test each model point for being inside this quadrangle
            let ks = (ied - 1) as usize + bv.if_skip_edge_ext[0] * (ixy - 1) as usize;
            let kd = ((ied - 1) + idimedge * (ixy - 1)) as usize;
            for iobj in 1..=fm.max_mod_obj {
                let io = (iobj - 1) as usize;
                for ipt in 1..=fm.npt_in_obj[io] {
                    let ipnt = fm.object[(ipt + fm.ibase_obj[io] - 1) as usize].abs();
                    let pc = fm.p_coord[(ipnt - 1) as usize];
                    if pc[2].round() as i32 == bv.iz_pc_list[ip] {
                        if inside(&vertex[0], &vertex[1], 4, pc[0], pc[1])
                            && bv.if_skip_edge[ks] == 0
                        {
                            num_near += 1;
                            bv.if_skip_edge[ks] = 2;
                            if (edgedispx[kd] != 0. || edgedispy[kd] != 0.) && if_use_adjusted > 0 {
                                bv.if_skip_edge[ks] = 1;
                                if if_use_adjusted > 1 {
                                    bv.if_skip_edge[ks] = 0;
                                }
                                num_efonly += 1;
                            }
                            if bv.if_skip_edge[ks] > 0 {
                                *num_skip += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    if num_near > 0 {
        if num_efonly == 0 {
            let _ = writeln!(
                ImodFile::Stdout,
                "\n{:7} edges will be given zero edge functions and\n   excluded when solving for shifts",
                *num_skip
            );
        } else if if_use_adjusted > 1 {
            let _ = writeln!(
                ImodFile::Stdout,
                "\n{:7} edges will be given zero edge functions but{:7} others have\n  non-zero displacements and edge functions will be found for them",
                *num_skip,
                num_efonly
            );
        } else {
            let _ = writeln!(
                ImodFile::Stdout,
                "\n{:7} edges will be given zero edge functions but{:7} of them have\n  non-zero displacements and will be included when solving for shifts",
                *num_skip,
                num_efonly
            );
        }
    }
    if fm.n_point > num_near {
        let _ = writeln!(
            ImodFile::Stdout,
            "\nWARNING: only{:7} of the {:7} model points were near an edge",
            *num_skip,
            fm.n_point
        );
    }
}

/// Original: `subroutine dumpedge(crray, nxdim, nxpad, nypad, ixy, ifcorr)`
/// (`bsubs.f90:3239`).
///
/// Writes a padded image or correlation (`ifcorr` nonzero) of an `ixy`
/// edge, scaled to bytes, as the next section of the dump file on unit
/// `2 + ixy`.  `crray` is `(nxdim, nypad)`.  The header label `title` is
/// never initialized in the source and is written with 0 labels and
/// `labFlag = -1`, so its content is never used.
pub fn dumpedge(
    bv: &mut BlendVars,
    crray: &[f32],
    nxdim: i32,
    nxpad: i32,
    nypad: i32,
    ixy: i32,
    ifcorr: i32,
) {
    const MAXLINE: i32 = 4096;
    let title_labels = [[0u8; MRC_LABEL_SIZE]; MRC_NLABELS];
    let title = [0u8; MRC_LABEL_SIZE];
    let mut bline = [0.0f32; MAXLINE as usize];
    let mut kxyz = [0i32; 3];
    let (mut dmin, mut dmax, mut dmt) = (0.0f32, 0.0f32, 0.0f32);
    let ixyu = (ixy - 1) as usize;
    let c = |i: i32, j: i32| crray[((i - 1) + nxdim * (j - 1)) as usize];
    //
    if bv.if_dump_xy[ixyu] < 0 || nxpad > MAXLINE {
        return;
    }
    if bv.if_dump_xy[ixyu] == 0 {
        bv.nz_out_xy[ixyu] = 0;
        bv.nx_out_xy[ixyu] = nxpad;
        bv.ny_out_xy[ixyu] = nypad;
        kxyz[0] = nxpad;
        kxyz[1] = nypad;
        kxyz[2] = 0;
        iiu_create_header(2 + ixy, &kxyz, &kxyz, 2, &title_labels, 0);
        bv.if_dump_xy[ixyu] = 1;
    }
    unsafe { iiu_set_position(2 + ixy, bv.nz_out_xy[ixyu], 0) };
    bv.nz_out_xy[ixyu] += 1;
    iiu_alt_size_samp_cell(
        2 + ixy,
        bv.nx_out_xy[ixyu],
        bv.ny_out_xy[ixyu],
        bv.nz_out_xy[ixyu],
    );

    array_min_max_mean_fortran(
        crray, &nxdim, &nypad, &1, &nxpad, &1, &nypad, &mut dmin, &mut dmax, &mut dmt,
    );
    let scale = 255. / (dmax - dmin);
    for iy in 1..=bv.ny_out_xy[ixyu] {
        if iy <= nypad {
            if ifcorr == 0 {
                for ix in 1..=bv.nx_out_xy[ixyu] {
                    bline[(ix - 1) as usize] = scale * (c(ix.min(nxpad), iy) - dmin);
                }
            } else {
                for ix in 1..=bv.nx_out_xy[ixyu] {
                    bline[(ix - 1) as usize] = scale
                        * (c(
                            ((ix + nxpad / 2 - 1) % nxpad + 1).min(nxpad),
                            ((iy + nypad / 2 - 1) % nypad + 1).min(nypad),
                        ) - dmin);
                }
            }
        }
        unsafe { iiu_write_lines(2 + ixy, bline.as_mut_ptr().cast(), 1) };
    }
    if ifcorr != 0 {
        let ipc = (bv.ipc_below_edge - 1) as usize;
        let mut ix = (bv.ix_pc_list[ipc] - bv.min_xpiece) / (bv.nxyz_in[0] - bv.n_overlap[0]);
        let iy = (bv.iy_pc_list[ipc] - bv.min_ypiece) / (bv.nxyz_in[1] - bv.n_overlap[1]);
        if ixy == 1 {
            ix = iy * (bv.nx_pieces - 1) + ix + 1;
        }
        if ixy == 2 {
            ix = ix * (bv.ny_pieces - 1) + iy + 1;
        }
        let _ = writeln!(
            ImodFile::Stdout,
            " {} edge{:4} corr at Z{:5}",
            (ixy as u8 + b'W') as char,
            ix,
            bv.nz_out_xy[ixyu]
        );
        let _ = ImodFile::Stdout.flush();
    }
    //
    iiu_write_header(2 + ixy, &title, -1, 0., 255., 128.);
}
