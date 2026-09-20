//! Direct Rust translation of `IMOD/libiimod/parallelwrite.c`.
#![allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code,
    unused_variables
)]
use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libiimod::mrcfiles::MrcHeader;
use std::cell::{Cell, RefCell};
pub type fortStrLen_t = i32;
/// C `BoundInfo` (`parallelwrite.c:28`).  `regions` was a `malloc`ed array of
/// `numFiles` entries; it owns its storage here, which is why neither this type
/// nor [`BoundRegion`] is `Copy` any more.
#[derive(Clone, Default)]
pub struct BoundInfo {
    pub regions: Vec<BoundRegion>,
    pub nx: i32,
    pub ny: i32,
    pub num_bound_lines: i32,
    pub num_files: i32,
    pub every_sec: i32,
    pub hdf_index: i32,
}
/// C `BoundRegion` (`parallelwrite.c:21`).
#[derive(Clone)]
pub struct BoundRegion {
    /// Boundary-file name, read from a line of the boundary-info file.
    pub file: String,
    pub section: [i32; 2],
    pub start_line: [i32; 2],
}
pub type ImodImageFile = crate::imod::libiimod::iimage::ImodImageFile;
#[derive(Copy, Clone)]
pub struct BufSegment {
    /// Offset of this segment in the owned shared HDF buffer.
    pub buf_offset: usize,
    pub num_bytes: i32,
    pub section: i32,
    pub start_line: i32,
    pub chunk_lines: i32,
    pub ind_x0: i32,
    pub ind_x1: i32,
    pub nxdim_pad_right: i32,
    pub ix_start: i32,
    pub iy_start: i32,
    pub iy_end: i32,
}
pub const SEEK_SET: i32 = 0 as i32;
pub const MAX_INFOS: i32 = 5 as i32;
pub const MAXLINE: i32 = 1024 as i32;
pub const DEFAULT_HDF_BUF_MB: i32 = 20 as i32;
pub const HDF_BUFFER_ENV_VAR: &str = "PARALLEL_HDF_BUF_SIZE";
struct ParallelWriteState {
    infos: [BoundInfo; MAX_INFOS as usize],
    num_infos: i32,
    cur_info: i32,
    hdf_bufs: [Vec<u8>; MAX_INFOS as usize],
    segments: [Vec<BufSegment>; MAX_INFOS as usize],
    buf_index: [i32; MAX_INFOS as usize],
    nx_full: [i32; MAX_INFOS as usize],
    ny_full: [i32; MAX_INFOS as usize],
    iz_cur: [i32; MAX_INFOS as usize],
    iy_cur: [i32; MAX_INFOS as usize],
    lines_bound: [i32; MAX_INFOS as usize],
    iunit_bound: [i32; MAX_INFOS as usize],
    nz_full: [i32; MAX_INFOS as usize],
    iy_bound: [[i32; 2]; MAX_INFOS as usize],
    if_open: [i32; MAX_INFOS as usize],
    iz_bound: [[i32; 2]; MAX_INFOS as usize],
    if_all_sec: [i32; MAX_INFOS as usize],
}

impl Default for ParallelWriteState {
    fn default() -> Self {
        Self {
            infos: std::array::from_fn(|_| BoundInfo::default()),
            cur_info: -1,
            hdf_bufs: std::array::from_fn(|_| Vec::new()),
            segments: std::array::from_fn(|_| Vec::new()),
            num_infos: 0,
            buf_index: [0; MAX_INFOS as usize],
            nx_full: [0; MAX_INFOS as usize],
            ny_full: [0; MAX_INFOS as usize],
            iz_cur: [0; MAX_INFOS as usize],
            iy_cur: [0; MAX_INFOS as usize],
            lines_bound: [0; MAX_INFOS as usize],
            iunit_bound: [0; MAX_INFOS as usize],
            nz_full: [0; MAX_INFOS as usize],
            iy_bound: [[0; 2]; MAX_INFOS as usize],
            if_open: [0; MAX_INFOS as usize],
            iz_bound: [[0; 2]; MAX_INFOS as usize],
            if_all_sec: [0; MAX_INFOS as usize],
        }
    }
}

thread_local! {
    static S_PARALLEL_WRITE: RefCell<ParallelWriteState> = RefCell::new(ParallelWriteState::default());
    static S_BUF_SIZE: Cell<i32> = const { Cell::new(DEFAULT_HDF_BUF_MB * 1_048_576) };
    static S_BOUNDARY_WRITE: RefCell<BoundaryWriteState> = RefCell::new(BoundaryWriteState::default());
    static S_RECLOSE_WALL_SUM: Cell<f64> = const { Cell::new(0.0) };
    static S_PREPARE_WALL_SUM: Cell<f64> = const { Cell::new(0.0) };
}
#[derive(Default)]
struct BoundaryWriteState {
    hbound: Option<MrcHeader>,
    dsize: i32,
    csize: i32,
    lines_bound: i32,
    sections: [i32; 2],
    start_lines: [i32; 2],
    fp_bound: Option<ImodFile>,
}
pub fn par_wrt_initialize(filename: &str, nxin: i32, nyin: i32) -> i32 {
    S_PARALLEL_WRITE.with(|state| {
        let mut state = state.borrow_mut();
        if state.num_infos >= MAX_INFOS {
            return 5;
        }
        let info_index = state.num_infos as usize;
        let bi = &mut state.infos[info_index];
        bi.hdf_index = -1;
        if filename.is_empty() {
            bi.regions.clear();
            state.cur_info = state.num_infos;
            state.num_infos += 1;
            return 0;
        }
        if nxin < 0 {
            bi.hdf_index = crate::imod::libcfshr::b3dutil::b3d_open_lock_file(filename);
            if bi.hdf_index < 0 {
                return 6;
            }
            bi.nx = -nxin;
            bi.ny = nyin;
            // `parallelwrite.c:129-133`.  `atoi` of a string with no leading
            // number is 0, which the failing parse stands in for.
            if let Ok(env_value) = std::env::var(HDF_BUFFER_ENV_VAR) {
                let buffer_mb = env_value.trim_start().parse::<i32>().unwrap_or(0);
                if buffer_mb > 0 {
                    S_BUF_SIZE.with(|size| size.set(buffer_mb * 1_048_576));
                }
            }
            let Ok(buffer_size) = usize::try_from(S_BUF_SIZE.with(Cell::get)) else {
                return 7;
            };
            let mut hdf_buffer = Vec::new();
            if hdf_buffer.try_reserve_exact(buffer_size).is_err() {
                return 7;
            }
            hdf_buffer.resize(buffer_size, 0);
            state.hdf_bufs[info_index] = hdf_buffer;
            state.segments[info_index].clear();
            state.buf_index[info_index] = 0;
            state.cur_info = state.num_infos;
            state.num_infos += 1;
            return 0;
        }
        let fp = crate::imod::libcfshr::b3dutil::ImodFile::open(filename, "r");
        let Some(mut fp) = fp else {
            return 1;
        };
        let mut line = [0u8; MAXLINE as usize];
        if crate::imod::libcfshr::b3dutil::fgetline(&mut fp, &mut line, MAXLINE) <= 0 {
            return 2;
        }
        // `parallelwrite.c:148` is `sscanf(line, "%d %d %d %d %d", …)`: each `%d`
        // skips leading whitespace, takes an optional sign and digits, and stops at
        // the first character that cannot extend the number; a field that does not
        // convert leaves its variable alone and ends the scan.
        let mut version = 0;
        {
            let text =
                String::from_utf8_lossy(&line[..line.iter().position(|b| *b == 0).unwrap_or(0)])
                    .into_owned();
            let mut fields = text.split_ascii_whitespace();
            for target in [
                &mut version,
                &mut bi.every_sec,
                &mut bi.nx,
                &mut bi.num_bound_lines,
                &mut bi.num_files,
            ] {
                match fields.next().and_then(|f| f.parse::<i32>().ok()) {
                    Some(value) => *target = value,
                    None => break,
                }
            }
        }
        if bi.nx != nxin || bi.num_bound_lines <= 0 || bi.num_files <= 0 {
            return 3;
        }
        bi.regions.clear();
        for _i in 0..bi.num_files {
            if crate::imod::libcfshr::b3dutil::fgetline(&mut fp, &mut line, MAXLINE) <= 0 {
                return 2;
            }
            let mut region = BoundRegion {
                file: String::from_utf8_lossy(
                    &line[..line.iter().position(|b| *b == 0).unwrap_or(0)],
                )
                .into_owned(),
                section: [0; 2],
                start_line: [0; 2],
            };
            if crate::imod::libcfshr::b3dutil::fgetline(&mut fp, &mut line, MAXLINE) <= 0 {
                return 2;
            }
            // `parallelwrite.c:167`, the same `%d` scan as above.
            {
                let text = String::from_utf8_lossy(
                    &line[..line.iter().position(|b| *b == 0).unwrap_or(0)],
                )
                .into_owned();
                for (index, field) in text.split_ascii_whitespace().take(4).enumerate() {
                    let Ok(value) = field.parse::<i32>() else {
                        break;
                    };
                    match index {
                        0 => region.section[0] = value,
                        1 => region.start_line[0] = value,
                        2 => region.section[1] = value,
                        3 => region.start_line[1] = value,
                        _ => unreachable!(),
                    }
                }
            }
            if region.section[1] >= 0 && region.start_line[1] < 0 {
                region.start_line[1] = nyin - bi.num_bound_lines;
            }
            bi.regions.push(region);
        }
        // `fclose(fp)`: the handle closes when it leaves scope.
        drop(fp);
        bi.ny = nyin;
        state.cur_info = state.num_infos;
        state.num_infos += 1;
        0
    })
}
pub fn par_wrt_properties(all_sec: &mut i32, lines_bound: &mut i32, nfiles: &mut i32) -> i32 {
    S_PARALLEL_WRITE.with(|state| {
        let state = state.borrow();
        if state.num_infos == 0 || state.cur_info < 0 {
            return 1;
        }
        *lines_bound = 0;
        *all_sec = 0;
        *nfiles = 0;
        let bi = &state.infos[state.cur_info as usize];
        if bi.hdf_index >= 0 {
            *lines_bound = bi.nx;
            *nfiles = bi.ny;
            *all_sec = bi.hdf_index;
            return -1;
        }
        if bi.regions.is_empty() {
            return 0;
        }
        *lines_bound = bi.num_bound_lines;
        *all_sec = bi.every_sec;
        *nfiles = bi.num_files;
        0
    })
}
pub fn par_wrt_set_current(index: i32) -> i32 {
    S_PARALLEL_WRITE.with(|state| {
        let mut state = state.borrow_mut();
        if index < 0 || index >= state.num_infos {
            return 1;
        }
        state.cur_info = index;
        0
    })
}
pub fn par_wrt_close() {
    S_PARALLEL_WRITE.with(|state| {
        let state = state.borrow();
        for info in state.infos.iter().take(state.num_infos as usize) {
            if info.hdf_index >= 0 {
                unsafe { crate::imod::libcfshr::b3dutil::b3d_close_lock_file(info.hdf_index) };
            }
        }
    })
}
pub unsafe fn parallel_write_slice(
    buf: *mut ::core::ffi::c_void,
    fout: &mut ImodFile,
    hdata: *mut MrcHeader,
    slice: i32,
) -> i32 {
    S_BOUNDARY_WRITE.with(|boundary| unsafe {
        parallel_write_slice_state(buf, fout, hdata, slice, &mut boundary.borrow_mut())
    })
}

unsafe fn parallel_write_slice_state(
    buf: *mut ::core::ffi::c_void,
    fout: &mut ImodFile,
    hdata: *mut MrcHeader,
    slice: i32,
    state: &mut BoundaryWriteState,
) -> i32 {
    // `parallelwrite.c:245` `static MrcHeader hbound;`.  A file-scope C struct
    // is zero-initialised once and keeps its address; `Option` in a `static`
    // gives the same lifetime and address without `mem::zeroed`, which is
    // undefined for a type holding an `ImodFile` (NATIVE.md 4b).
    let mut allsec = 0;
    let mut nfiles = 0;
    let mut filename = String::new();
    // `parallelwrite.c:251` `(ImodImageFile *)fout` — the reverse of the HDF
    // identity token: for an HDF file `iiFile->fp` *is* `(FILE *)iiFile`, so
    // casting back recovers the struct.
    let ii_file = match fout {
        ImodFile::Token(address) => *address as *mut ImodImageFile,
        _ => ::core::ptr::null_mut(),
    };
    let hbound = state.hbound.get_or_insert_with(MrcHeader::default);
    let mut fout = fout.clone();
    let parallel_hdf = S_PARALLEL_WRITE.with(|parallel| {
        let parallel = parallel.borrow();
        parallel.cur_info >= 0 && parallel.infos[parallel.cur_info as usize].hdf_index >= 0
    });
    if parallel_hdf {
        crate::imod::libiimod::mrcfiles::mrc_getdcsize(
            (*hdata).mode,
            &mut state.dsize,
            &mut state.csize,
        );
        let err = add_segment(
            buf.cast(),
            state.dsize * state.csize * (*hdata).nx * (*hdata).ny,
            slice,
            -2,
            ii_file,
            hdata,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        if err >= 0 {
            return err;
        }
        fout = (*ii_file).fp.clone().unwrap();
    }
    let image_bytes = (state.dsize * state.csize * (*hdata).nx * (*hdata).ny) as usize;
    let mut err = crate::imod::libiimod::mrcfiles::mrc_write_slice(
        core::slice::from_raw_parts(buf.cast(), image_bytes),
        &mut fout,
        &mut *hdata,
        slice,
        b'Z',
    );
    if parallel_hdf && par_wrt_reclose_hdf(ii_file, hdata) != 0 {
        return 1;
    }
    if err != 0 || parallel_hdf {
        return err;
    }
    if state.lines_bound < 0 {
        if par_wrt_properties(&mut allsec, &mut state.lines_bound, &mut nfiles) != 0 {
            state.lines_bound = 0;
        }
        if state.lines_bound == 0 {
            return 0;
        }
        crate::imod::libiimod::mrcfiles::mrc_head_new(
            hbound,
            (*hdata).nx,
            state.lines_bound,
            2,
            (*hdata).mode,
        );
        err = par_wrt_find_region(
            slice,
            0,
            (*hdata).ny,
            &mut filename,
            &mut state.sections,
            &mut state.start_lines,
        );
        if err != 0 {
            crate::imod::libcfshr::b3dutil::b3d_error(
                Some(&mut ImodFile::Stdout),
                format_args!(
                    "ERROR: sliceWriteParallel - finding parallel writing region for slice {} (err {})\n",
                    slice, err
                ),
            );
            return err;
        }
        if crate::imod::libiimod::mrcfiles::mrc_getdcsize(
            (*hdata).mode,
            &mut state.dsize,
            &mut state.csize,
        ) != 0
        {
            crate::imod::libcfshr::b3dutil::b3d_error(
                Some(&mut ImodFile::Stdout),
                format_args!("ERROR: sliceWriteParallel - unknown mode.\n"),
            );
            return 1;
        }
        crate::imod::libcfshr::b3dutil::imod_backup_file(&filename);
        state.fp_bound = ImodFile::open(&filename, "wb");
        if state.fp_bound.is_none() {
            crate::imod::libcfshr::b3dutil::b3d_error(
                Some(&mut ImodFile::Stdout),
                format_args!(
                    "ERROR: sliceWriteParallel - opening boundary file {}\n",
                    filename
                ),
            );
            return 1;
        }
        if crate::imod::libiimod::mrcfiles::mrc_head_write(state.fp_bound.as_mut().unwrap(), hbound)
            != 0
        {
            return 1;
        }
    }
    if state.lines_bound == 0 {
        return 0;
    }
    for ib in 0..2usize {
        if state.sections[ib] >= 0 && slice == state.sections[ib] {
            crate::imod::libcfshr::b3dutil::b3d_fseek(
                state.fp_bound.as_mut().unwrap(),
                hbound.header_size
                    + ib as i32 * hbound.nx * state.lines_bound * state.csize * state.dsize,
                SEEK_SET,
            );
            let offset = (hbound.nx * state.start_lines[ib] * state.csize * state.dsize) as usize;
            let data = core::slice::from_raw_parts(
                buf.cast::<u8>().add(offset),
                (hbound.nx * state.lines_bound * state.csize * state.dsize) as usize,
            );
            err = crate::imod::libiimod::mrcfiles::mrc_write_slice(
                data,
                state.fp_bound.as_mut().unwrap(),
                hbound,
                ib as i32,
                b'Z',
            );
            if err != 0 {
                return err;
            }
        }
    }
    0
}
pub unsafe fn par_wrt_reclose_hdf(ii_file: *mut ImodImageFile, hdata: *mut MrcHeader) -> i32 {
    let wall_start = crate::imod::libcfshr::b3dutil::wall_time();
    if !hdata.is_null()
        && crate::imod::libiimod::mrcfiles::mrc_head_write(
            &mut (*ii_file).fp.clone().unwrap(),
            &mut *hdata,
        ) != 0
    {
        crate::imod::libcfshr::b3dutil::b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!("ERROR:parWrtRecloseHDF  - Rewriting header of HDF file\n"),
        );
        return 1;
    }
    crate::imod::libiimod::iimage::ii_close(ii_file);
    let hdf_index = S_PARALLEL_WRITE.with(|state| {
        let state = state.borrow();
        state.infos[state.cur_info as usize].hdf_index
    });
    let err = crate::imod::libcfshr::b3dutil::b3d_unlock_file(hdf_index);
    if err != 0 {
        crate::imod::libcfshr::b3dutil::b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "ERROR: parWrtRecloseHDF - Releasing file lock for HDF file (err {})\n",
                err
            ),
        );
        return 1;
    }
    S_RECLOSE_WALL_SUM
        .with(|sum| sum.set(sum.get() + crate::imod::libcfshr::b3dutil::wall_time() - wall_start));
    0
}
pub unsafe fn par_wrt_flush_buffers(ii_file: *mut ImodImageFile, hdata: *mut MrcHeader) -> i32 {
    if par_wrt_prepare_hdf(ii_file) != 0 {
        return 1;
    }
    let ierr = write_segments(ii_file, hdata, 0);
    if par_wrt_reclose_hdf(ii_file, hdata) != 0 || ierr != 0 {
        return 1;
    }
    0
}
pub fn iiu_par_wrt_initialize(
    filename: &str,
    iunit_bound: i32,
    nx_in: i32,
    ny_in: i32,
    nz_in: i32,
) -> i32 {
    let setup_error = S_PARALLEL_WRITE.with(|state| {
        let mut state = state.borrow_mut();
        if state.num_infos >= MAX_INFOS {
            return true;
        }
        let index = state.num_infos as usize;
        state.iunit_bound[index] = iunit_bound;
        state.nx_full[index] = nx_in.abs();
        state.ny_full[index] = ny_in;
        state.nz_full[index] = nz_in;
        state.iz_cur[index] = 0;
        state.iy_cur[index] = 0;
        state.lines_bound[index] = 0;
        state.if_open[index] = 0;
        false
    });
    if setup_error {
        return 5;
    }
    let retval = par_wrt_initialize(filename, nx_in, ny_in);
    if retval == 0 && !filename.is_empty() {
        let mut num_files = 0;
        let mut all_sec = 0;
        let mut lines_bound = 0;
        par_wrt_properties(&mut all_sec, &mut lines_bound, &mut num_files);
        S_PARALLEL_WRITE.with(|state| {
            let mut state = state.borrow_mut();
            let index = state.cur_info as usize;
            state.if_all_sec[index] = all_sec;
            state.lines_bound[index] = lines_bound;
        });
    }
    retval
}
pub unsafe fn par_wrt_posn(iunit: i32, iz: i32, iy: i32) {
    crate::imod::libiimod::unit_fileio::iiu_set_position(iunit, iz, iy);
    S_PARALLEL_WRITE.with(|state| {
        let mut state = state.borrow_mut();
        if state.num_infos > 0 && state.cur_info >= 0 {
            let index = state.cur_info as usize;
            state.iz_cur[index] = iz;
            state.iy_cur[index] = iy;
        }
    });
}
pub unsafe fn par_wrt_sec(iunit: i32, array: *mut ::core::ffi::c_void) -> i32 {
    let barray = array.cast::<u8>();
    let state_values = S_PARALLEL_WRITE.with(|state| {
        let state = state.borrow();
        (state.cur_info >= 0).then(|| {
            let index = state.cur_info as usize;
            (
                state.infos[index].hdf_index >= 0,
                state.nx_full[index],
                state.ny_full[index],
                state.iz_cur[index],
            )
        })
    });
    if let Some((true, nx_full, ny_full, iz_cur)) = state_values {
        let ii_file = crate::imod::libiimod::unit_fileio::iiu_get_ii_file(iunit);
        let num_bytes =
            crate::imod::libiimod::unit_fileio::iiu_buf_bytes_per_pixel(iunit) * nx_full * ny_full;
        let ierr = add_segment(
            barray,
            num_bytes,
            iz_cur,
            -1,
            ii_file,
            ::core::ptr::null_mut::<MrcHeader>(),
            iunit,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        if ierr > 0 {
            return 1;
        }
        if ierr == 0 {
            advance_section();
            return 0;
        }
    }
    let mut ierr = crate::imod::libiimod::unit_fileio::iiu_write_section(iunit, array);
    if ierr != 0 {
        return ierr;
    }
    if iiu_par_wrt_reclose_hdf(iunit, 1) != 0 {
        advance_section();
        return 0;
    }
    let Some((iz_cur, iz_bound, iunit_bound, nx_full, ny_full, lines_bound)) = S_PARALLEL_WRITE
        .with(|state| {
            let state = state.borrow();
            if state.num_infos <= 0 || state.cur_info < 0 {
                return None;
            }
            let index = state.cur_info as usize;
            (state.lines_bound[index] != 0).then(|| {
                (
                    state.iz_cur[index],
                    state.iz_bound[index],
                    state.iunit_bound[index],
                    state.nx_full[index],
                    state.ny_full[index],
                    state.lines_bound[index],
                )
            })
        })
    else {
        return 0;
    };
    pw_open_if_needed(iz_cur, 0, ny_full, &raw mut ierr);
    if ierr != 0 {
        crate::imod::libcfshr::b3dutil::b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "\nERROR: parWrtSec - Finding parallel write boundary region sec {} err {}\n",
                iz_cur, ierr
            ),
        );
        std::process::exit(1);
    }
    if iz_cur == iz_bound[0] {
        crate::imod::libiimod::unit_fileio::iiu_set_position(iunit_bound, 0, 0);
        crate::imod::libiimod::unit_fileio::iiu_write_section(iunit_bound, array);
    }
    if iz_cur == iz_bound[1] {
        crate::imod::libiimod::unit_fileio::iiu_set_position(iunit_bound, 1, 0);
        ierr = crate::imod::libiimod::unit_fileio::iiu_write_section(
            iunit_bound,
            barray
                .add(
                    (crate::imod::libiimod::unit_fileio::iiu_buf_bytes_per_pixel(iunit)
                        * nx_full
                        * (ny_full - lines_bound)) as usize,
                )
                .cast(),
        );
        if ierr != 0 {
            return ierr;
        }
    }
    advance_section();
    0
}
pub unsafe fn par_wrt_lin(iunit: i32, array: *mut ::core::ffi::c_void) -> i32 {
    let state_values = S_PARALLEL_WRITE.with(|state| {
        let state = state.borrow();
        (state.cur_info >= 0).then(|| {
            let index = state.cur_info as usize;
            (
                state.infos[index].hdf_index >= 0,
                state.nx_full[index],
                state.iz_cur[index],
                state.iy_cur[index],
            )
        })
    });
    if let Some((true, nx_full, iz_cur, iy_cur)) = state_values {
        let ii_file = crate::imod::libiimod::unit_fileio::iiu_get_ii_file(iunit);
        let num_bytes =
            crate::imod::libiimod::unit_fileio::iiu_buf_bytes_per_pixel(iunit) * nx_full;
        let ierr = add_segment(
            array.cast(),
            num_bytes,
            iz_cur,
            iy_cur,
            ii_file,
            ::core::ptr::null_mut::<MrcHeader>(),
            iunit,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        if ierr > 0 {
            return 1;
        }
        if ierr == 0 {
            advance_line();
            return 0;
        }
    }
    let mut ierr = crate::imod::libiimod::unit_fileio::iiu_write_lines(iunit, array, 1);
    if ierr != 0 {
        return ierr;
    }
    if iiu_par_wrt_reclose_hdf(iunit, 1) != 0 {
        advance_line();
        return 0;
    }
    let Some((iz_cur, iy_cur, iy_bound, iz_bound, if_all_sec, lines_bound, iunit_bound, if_open)) =
        S_PARALLEL_WRITE.with(|state| {
            let state = state.borrow();
            if state.num_infos <= 0 || state.cur_info < 0 {
                return None;
            }
            let index = state.cur_info as usize;
            (state.lines_bound[index] != 0).then(|| {
                (
                    state.iz_cur[index],
                    state.iy_cur[index],
                    state.iy_bound[index],
                    state.iz_bound[index],
                    state.if_all_sec[index],
                    state.lines_bound[index],
                    state.iunit_bound[index],
                    state.if_open[index],
                )
            })
        })
    else {
        return 0;
    };
    if if_open == 0 {
        pw_open_if_needed(iz_cur, iy_cur, 1, &raw mut ierr);
        if ierr != 0 {
            crate::imod::libcfshr::b3dutil::b3d_error(
                Some(&mut ImodFile::Stdout),
                format_args!(
                    "\nERROR: parWrtSec - Finding parallel write boundary region at {}, {} err {}\n",
                    iz_cur, iy_cur, ierr
                ),
            );
            std::process::exit(1);
        }
    }
    if if_all_sec != 0 {
        if iy_cur < iy_bound[0] + lines_bound {
            crate::imod::libiimod::unit_fileio::iiu_set_position(
                iunit_bound,
                2 * iz_cur,
                iy_cur - iy_bound[0],
            );
            ierr = crate::imod::libiimod::unit_fileio::iiu_write_lines(iunit_bound, array, 1);
        }
        if iy_cur >= iy_bound[1] {
            crate::imod::libiimod::unit_fileio::iiu_set_position(
                iunit_bound,
                2 * iz_cur + 1,
                iy_cur - iy_bound[1],
            );
            ierr = crate::imod::libiimod::unit_fileio::iiu_write_lines(iunit_bound, array, 1);
        }
    } else {
        if iz_cur == iz_bound[0] && iy_cur < iy_bound[0] + lines_bound {
            crate::imod::libiimod::unit_fileio::iiu_set_position(
                iunit_bound,
                0,
                iy_cur - iy_bound[0],
            );
            ierr = crate::imod::libiimod::unit_fileio::iiu_write_lines(iunit_bound, array, 1);
        }
        if iz_cur == iz_bound[1] && iy_cur >= iy_bound[1] {
            crate::imod::libiimod::unit_fileio::iiu_set_position(
                iunit_bound,
                1,
                iy_cur - iy_bound[1],
            );
            ierr = crate::imod::libiimod::unit_fileio::iiu_write_lines(iunit_bound, array, 1);
        }
    }
    if ierr != 0 {
        return ierr;
    }
    advance_line();
    0
}
pub unsafe fn iiu_par_wrt_sec_part(
    iunit: i32,
    array: *mut ::core::ffi::c_void,
    nxdim: i32,
    ix_start: i32,
    ind_x0: i32,
    ind_x1: i32,
    iy_start: i32,
    iy_end: i32,
) -> i32 {
    let mut ierr: i32;
    let num_bytes: i32;
    let num_trim: i32;
    let ii_file: *mut ImodImageFile;
    let state_values = S_PARALLEL_WRITE.with(|state| {
        let state = state.borrow();
        (state.cur_info >= 0).then(|| {
            let index = state.cur_info as usize;
            (
                state.infos[index].hdf_index >= 0,
                state.iz_cur[index],
                state.iy_cur[index],
            )
        })
    });
    if let Some((true, iz_cur, iy_cur)) = state_values {
        ii_file = crate::imod::libiimod::unit_fileio::iiu_get_ii_file(iunit);
        num_bytes = crate::imod::libiimod::unit_fileio::iiu_buf_bytes_per_pixel(iunit)
            * nxdim
            * (iy_end + 1 - iy_start);
        num_trim =
            crate::imod::libiimod::unit_fileio::iiu_buf_bytes_per_pixel(iunit) * nxdim * iy_start;
        ierr = add_segment(
            array.cast::<u8>().add(num_trim as usize),
            num_bytes,
            iz_cur,
            iy_cur,
            ii_file,
            ::core::ptr::null_mut::<MrcHeader>(),
            iunit,
            1,
            nxdim,
            ix_start,
            ind_x0,
            ind_x1,
            0,
            iy_end - iy_start,
        );
        if ierr > 0 {
            return 1;
        }
        if ierr == 0 {
            return 0;
        }
    }
    ierr = crate::imod::libiimod::unit_fileio::iiu_write_sec_part(
        iunit, array, nxdim, ix_start, ind_x0, ind_x1, iy_start, iy_end,
    );
    if ierr != 0 {
        return ierr;
    }
    iiu_par_wrt_reclose_hdf(iunit, 1);
    0
}
pub unsafe fn iiu_par_wrt_reclose_hdf(iunit: i32, write_header: i32) -> i32 {
    let ii_file: *mut ImodImageFile;
    let hdata: *mut MrcHeader;
    let parallel_hdf = S_PARALLEL_WRITE.with(|state| {
        let state = state.borrow();
        state.num_infos > 0
            && state.cur_info >= 0
            && state.infos[state.cur_info as usize].hdf_index >= 0
    });
    if !parallel_hdf {
        return 0;
    }
    ii_file = crate::imod::libiimod::unit_fileio::iiu_get_ii_file(iunit);
    hdata =
        crate::imod::libiimod::unit_fileio::iiu_mrc_header(iunit, "iiu_par_wrt_reclose_hdf", 1, 0);
    if par_wrt_reclose_hdf(
        ii_file,
        if write_header != 0 {
            hdata
        } else {
            ::core::ptr::null_mut::<MrcHeader>()
        },
    ) != 0
    {
        std::process::exit(1);
    }
    1
}
pub unsafe fn iiu_write_dummy_sec_to_hdf(iunit: i32) {
    let mut buf = [0u8; 32];
    crate::imod::libiimod::unit_fileio::iiu_sync_with_mrc_header(iunit);
    let ii_file = crate::imod::libiimod::unit_fileio::iiu_get_ii_file(iunit);
    if crate::imod::libiimod::iihdf::hdf_write_dummy_section(ii_file, buf.as_mut_ptr(), 0) != 0 {
        std::process::exit(1);
    }
}
pub unsafe fn iiu_par_wrt_flush_buffers(iunit: i32) -> i32 {
    let ii_file = crate::imod::libiimod::unit_fileio::iiu_get_ii_file(iunit);
    if iiu_prepare_write_hdf(iunit) != 0 {
        return 1;
    }
    let ierr = write_segments(ii_file, ::core::ptr::null_mut::<MrcHeader>(), iunit);
    iiu_par_wrt_reclose_hdf(iunit, 1);
    ierr
}
fn par_wrt_find_region(
    sec_num: i32,
    line_num: i32,
    nl_write: i32,
    filename: &mut String,
    sections: &mut [i32; 2],
    start_lines: &mut [i32; 2],
) -> i32 {
    S_PARALLEL_WRITE.with(|state| {
        let state = state.borrow();
        if state.num_infos == 0
            || state.cur_info < 0
            || state.infos[state.cur_info as usize].regions.is_empty()
        {
            return 1;
        }
        let bi = &state.infos[state.cur_info as usize];
        for i in 0..bi.num_files {
            let region = &bi.regions[i as usize];
            let mut past_start = true;
            let mut before_end = true;
            if bi.every_sec != 0 {
                if region.start_line[0] >= 0 && region.start_line[0] > line_num + nl_write - 1 {
                    past_start = false;
                }
                if region.start_line[1] >= 0
                    && region.start_line[1] + bi.num_bound_lines - 1 < line_num
                {
                    before_end = false;
                }
            } else {
                if region.section[0] >= 0
                    && (region.section[0] > sec_num
                        || (region.section[0] == sec_num
                            && region.start_line[0] > line_num + nl_write - 1))
                {
                    past_start = false;
                }
                if region.section[1] >= 0
                    && (region.section[1] < sec_num
                        || (region.section[1] == sec_num
                            && region.start_line[1] + bi.num_bound_lines - 1 < line_num))
                {
                    before_end = false;
                }
            }
            if past_start && before_end {
                // The caller takes ownership of the selected boundary-file name.
                *filename = region.file.clone();
                sections[0] = region.section[0];
                start_lines[0] = region.start_line[0];
                sections[1] = region.section[1];
                start_lines[1] = region.start_line[1];
                if bi.every_sec != 0 {
                    if start_lines[0] < 0 {
                        start_lines[0] -= bi.num_bound_lines;
                    }
                    if start_lines[1] < 0 {
                        start_lines[1] = bi.ny + 1;
                    }
                }
                return 0;
            }
        }
        2
    })
}
unsafe fn par_wrt_prepare_hdf(ii_file: *mut ImodImageFile) -> i32 {
    let wall_start = crate::imod::libcfshr::b3dutil::wall_time();
    let hdf_index = S_PARALLEL_WRITE.with(|state| {
        let state = state.borrow();
        state.infos[state.cur_info as usize].hdf_index
    });
    let err = crate::imod::libcfshr::b3dutil::b3d_lock_file(hdf_index);
    if err != 0 {
        crate::imod::libcfshr::b3dutil::b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "\nERROR: parWrtPrepareHDF - Obtaining file lock for HDF file (err {})\n",
                err
            ),
        );
        return 1;
    }
    (*ii_file).state = IISTATE_NOTINIT;
    let err = crate::imod::libiimod::iimage::ii_reopen(&mut *ii_file);
    if err != 0 {
        crate::imod::libcfshr::b3dutil::b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "\nERROR: parWrtPrepareHDF - Reopening HDF file (err {})\n",
                err
            ),
        );
        return 1;
    }
    S_PREPARE_WALL_SUM
        .with(|sum| sum.set(sum.get() + crate::imod::libcfshr::b3dutil::wall_time() - wall_start));
    0
}
unsafe fn add_segment(
    buf: *mut u8,
    num_bytes: i32,
    section: i32,
    start_line: i32,
    ii_file: *mut ImodImageFile,
    hdata: *mut MrcHeader,
    iunit: i32,
    chunk: i32,
    nxdim: i32,
    ix_start: i32,
    x0: i32,
    x1: i32,
    iy_start: i32,
    iy_end: i32,
) -> i32 {
    let buffer_full = S_PARALLEL_WRITE.with(|state| {
        let state = state.borrow();
        state.buf_index[state.cur_info as usize] + num_bytes > S_BUF_SIZE.with(Cell::get)
    });
    if buffer_full {
        let err = if start_line < -1 {
            par_wrt_prepare_hdf(ii_file)
        } else {
            iiu_prepare_write_hdf(iunit)
        };
        if err != 0 {
            return 1;
        }
        let err = write_segments(ii_file, hdata, iunit);
        let still_full = S_PARALLEL_WRITE.with(|state| {
            let state = state.borrow();
            state.buf_index[state.cur_info as usize] + num_bytes > S_BUF_SIZE.with(Cell::get)
        });
        if err == 0 && (start_line < 0 || still_full) {
            return -1;
        }
        if start_line < -1 {
            par_wrt_reclose_hdf(ii_file, hdata);
        } else {
            iiu_par_wrt_reclose_hdf(iunit, 1);
        }
        if err != 0 {
            return 1;
        }
    }
    let Ok(num_bytes_usize) = usize::try_from(num_bytes) else {
        return 1;
    };
    S_PARALLEL_WRITE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let info_index = state.cur_info as usize;
        if state.segments[info_index].len() == state.segments[info_index].capacity() {
            let increment = if start_line >= 0 { 128 } else { 8 };
            if state.segments[info_index]
                .try_reserve_exact(increment)
                .is_err()
            {
                return 1;
            }
        }
        let Ok(buf_offset) = usize::try_from(state.buf_index[info_index]) else {
            return 1;
        };
        let Some(end_offset) = buf_offset.checked_add(num_bytes_usize) else {
            return 1;
        };
        let Some(destination) = state.hdf_bufs[info_index].get_mut(buf_offset..end_offset) else {
            return 1;
        };
        if num_bytes_usize != 0 {
            let source = core::slice::from_raw_parts(buf.cast_const(), num_bytes_usize);
            destination.copy_from_slice(source);
        }
        state.segments[info_index].push(BufSegment {
            buf_offset,
            num_bytes,
            section,
            start_line,
            chunk_lines: chunk,
            ind_x0: x0,
            ind_x1: x1,
            nxdim_pad_right: nxdim,
            ix_start,
            iy_start,
            iy_end,
        });
        state.buf_index[info_index] += num_bytes;
        0
    })
}
unsafe fn write_segments(ii_file: *mut ImodImageFile, _hdata: *mut MrcHeader, iunit: i32) -> i32 {
    S_PARALLEL_WRITE.with(|state| unsafe {
        let mut state = state.borrow_mut();
        let info_index = state.cur_info as usize;
        if state.segments[info_index].is_empty() {
            return 0;
        }
        let mut num_lines = 0;
        let mut start_line = 0;
        let mut line_section = 0;
        let mut lines_buf = ::core::ptr::null_mut::<u8>();
        for ind in 0..state.segments[info_index].len() {
            let seg = state.segments[info_index][ind];
            let segment_buffer = state.hdf_bufs[info_index].as_mut_ptr().add(seg.buf_offset);
            if seg.start_line < 0 {
                if write_buffered_lines(
                    iunit,
                    line_section,
                    start_line,
                    lines_buf,
                    &raw mut num_lines,
                ) != 0
                {
                    return 1;
                }
                let err = if seg.start_line == -1 {
                    crate::imod::libiimod::unit_fileio::iiu_set_position(iunit, seg.section, 0);
                    crate::imod::libiimod::unit_fileio::iiu_write_section(
                        iunit,
                        segment_buffer.cast(),
                    )
                } else {
                    crate::imod::libiimod::iimage::ii_write_section(
                        &mut *ii_file,
                        core::slice::from_raw_parts_mut(segment_buffer, seg.num_bytes as usize),
                        seg.section,
                    )
                };
                if err != 0 {
                    return 1;
                }
            } else if seg.chunk_lines == 1 {
                crate::imod::libiimod::unit_fileio::iiu_set_position(
                    iunit,
                    seg.section,
                    seg.start_line,
                );
                if crate::imod::libiimod::unit_fileio::iiu_write_sec_part(
                    iunit,
                    segment_buffer.cast(),
                    seg.nxdim_pad_right,
                    seg.ix_start,
                    seg.ind_x0,
                    seg.ind_x1,
                    seg.iy_start,
                    seg.iy_end,
                ) != 0
                {
                    return 1;
                }
            } else if seg.chunk_lines == 2 {
                (*ii_file).llx = seg.ind_x0;
                (*ii_file).urx = seg.ind_x1;
                (*ii_file).lly = seg.iy_start;
                (*ii_file).ury = seg.iy_end;
                (*ii_file).pad_left = seg.ix_start;
                (*ii_file).pad_right = seg.nxdim_pad_right;
                if crate::imod::libiimod::iimage::ii_write_section(
                    &mut *ii_file,
                    core::slice::from_raw_parts_mut(segment_buffer, seg.num_bytes as usize),
                    seg.section,
                ) != 0
                {
                    return 1;
                }
            } else {
                if num_lines != 0
                    && (start_line + num_lines != seg.start_line || line_section != seg.section)
                {
                    if write_buffered_lines(
                        iunit,
                        line_section,
                        start_line,
                        lines_buf,
                        &raw mut num_lines,
                    ) != 0
                    {
                        return 1;
                    }
                }
                if num_lines == 0 {
                    lines_buf = segment_buffer;
                    start_line = seg.start_line;
                    line_section = seg.section;
                }
                num_lines += 1;
            }
        }
        if write_buffered_lines(
            iunit,
            line_section,
            start_line,
            lines_buf,
            &raw mut num_lines,
        ) != 0
        {
            return 1;
        }
        state.buf_index[info_index] = 0;
        state.segments[info_index].clear();
        0
    })
}
unsafe fn clear_segments(info_ind: i32) {
    S_PARALLEL_WRITE.with(|state| {
        let mut state = state.borrow_mut();
        state.buf_index[info_ind as usize] = 0;
        state.segments[info_ind as usize].clear();
    });
}
unsafe fn pw_open_if_needed(iz_sec: i32, iy_line: i32, nlines_write: i32, ierr: *mut i32) {
    let mut nxyz = [0; 3];
    let mut filename = String::new();
    let title = "parallel_write: boundary lines";
    let mut cell = [0.0, 0.0, 0.0, 90.0, 90.0, 90.0];
    *ierr = 0;
    let Some((info_index, nx_full, lines_bound, nz_full, if_all_sec, iunit_bound)) =
        S_PARALLEL_WRITE.with(|state| {
            let state = state.borrow();
            if state.num_infos <= 0 || state.cur_info < 0 {
                return None;
            }
            let index = state.cur_info as usize;
            if state.if_open[index] != 0 || state.lines_bound[index] == 0 {
                return None;
            }
            Some((
                index,
                state.nx_full[index],
                state.lines_bound[index],
                state.nz_full[index],
                state.if_all_sec[index],
                state.iunit_bound[index],
            ))
        })
    else {
        return;
    };
    let mut iz_bound = [0; 2];
    let mut iy_bound = [0; 2];
    *ierr = par_wrt_find_region(
        iz_sec,
        iy_line,
        nlines_write,
        &mut filename,
        &mut iz_bound,
        &mut iy_bound,
    );
    if *ierr != 0 {
        return;
    }
    S_PARALLEL_WRITE.with(|state| {
        let mut state = state.borrow_mut();
        state.iz_bound[info_index] = iz_bound;
        state.iy_bound[info_index] = iy_bound;
    });
    nxyz[0] = nx_full;
    nxyz[1] = lines_bound;
    nxyz[2] = 2;
    if if_all_sec != 0 {
        nxyz[2] = 2 * nz_full;
    }
    *ierr = crate::imod::libiimod::unit_fileio::iiu_open(iunit_bound, &filename, "NEW");
    if *ierr != 0 {
        return;
    }
    crate::imod::libiimod::unit_header::iiu_create_header(
        iunit_bound,
        &nxyz,
        &nxyz,
        2,
        &[[0; crate::imod::libiimod::mrcfiles::MRC_LABEL_SIZE];
            crate::imod::libiimod::mrcfiles::MRC_NLABELS],
        0,
    );
    cell[0] = nxyz[0] as f32;
    cell[1] = nxyz[1] as f32;
    cell[2] = nxyz[2] as f32;
    crate::imod::libiimod::unit_header::iiu_alt_cell(iunit_bound, &cell);
    *ierr = crate::imod::libiimod::unit_header::iiu_write_header_str(
        iunit_bound,
        title,
        0,
        -32000.0,
        32000.0f32,
        0.0f32,
    );
    S_PARALLEL_WRITE.with(|state| state.borrow_mut().if_open[info_index] = 1);
}
unsafe fn iiu_prepare_write_hdf(iunit: i32) -> i32 {
    let ii_file = crate::imod::libiimod::unit_fileio::iiu_get_ii_file(iunit);
    if par_wrt_prepare_hdf(ii_file) != 0 {
        return 1;
    }
    crate::imod::libiimod::unit_fileio::iiu_reassign_header_ptr(iunit);
    crate::imod::libiimod::unit_fileio::iiu_sync_with_mrc_header(iunit);
    0
}
unsafe fn advance_section() {
    S_PARALLEL_WRITE.with(|state| {
        let mut state = state.borrow_mut();
        let index = state.cur_info as usize;
        state.iz_cur[index] += 1;
        state.iy_cur[index] = 0;
    });
}
unsafe fn advance_line() {
    S_PARALLEL_WRITE.with(|state| {
        let mut state = state.borrow_mut();
        let index = state.cur_info as usize;
        state.iy_cur[index] += 1;
        if state.iy_cur[index] >= state.ny_full[index] {
            state.iy_cur[index] = 0;
            state.iz_cur[index] += 1;
        }
    });
}
unsafe fn write_buffered_lines(
    iunit: i32,
    section: i32,
    start_line: i32,
    lines_buf: *mut u8,
    num_lines: *mut i32,
) -> i32 {
    if *num_lines != 0 {
        crate::imod::libiimod::unit_fileio::iiu_set_position(iunit, section, start_line);
        if crate::imod::libiimod::unit_fileio::iiu_write_lines(iunit, lines_buf.cast(), *num_lines)
            != 0
        {
            return 1;
        }
        *num_lines = 0;
    }
    0
}
pub const IISTATE_NOTINIT: i32 = 0 as i32;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libiimod::mrcfiles::{MRC_MODE_BYTE, mrc_head_new, mrc_head_write};

    #[test]
    fn parallel_write_slice_writes_a_real_mrc_z_slice_without_parallel_setup() {
        unsafe {
            S_PARALLEL_WRITE.with(|state| *state.borrow_mut() = ParallelWriteState::default());
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 3, 2, 1, MRC_MODE_BYTE), 0);
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            let mut pixels = [3u8, 1, 4, 1, 5, 9];
            assert_eq!(
                parallel_write_slice(pixels.as_mut_ptr().cast(), &mut file, &mut header, 0),
                0
            );
            {
                use std::io::Write;
                file.flush().unwrap();
            }
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    header.header_size as i32,
                    SEEK_SET
                ),
                0
            );
            let mut written = [0u8; 6];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(&mut written, 1, 6, &mut file),
                written.len()
            );
            // `mrc_head_new` inherits IMOD's signed-byte output convention.
            // `mrc_write_z`, reached directly by `parallel_write_slice`, stores
            // signed byte samples with the source's +128 disk offset.
            assert_eq!(written, [131, 129, 132, 129, 133, 137]);
            drop(file);
        }
    }
}
