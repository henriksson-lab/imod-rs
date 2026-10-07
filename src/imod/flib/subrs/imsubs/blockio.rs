//! Translation of `IMOD/flib/subrs/imsubs/blockio.c`: the unit-numbered block
//! I/O layer (`qopen`/`qread`/`qwrite`/`qseek`/...) that Fortran code calls.
//!
//! The file-static `units[MAX_UNIT]` table and `find_unit`'s static
//! `firstTime` are held in one thread-local record, as `imodel_fwrap.rs` does
//! for its file statics; a unit's `FILE *` is an [`ImodFile`], which is the
//! crate's `FILE *` (a buffered stream with C's seek and flush rules).  The
//! Fortran calling convention is gone (`NATIVE.md`): a unit number read by a
//! routine is passed by value, one it returns is `&mut i32`, an input
//! `CHARACTER*(*)` is a `&str` (its hidden length is the slice length), the
//! `qinquire` output name is a fixed-width blank-padded `&mut [u8]`, and the
//! `char *array` data buffers are byte slices.
//!
//! Not translated: `strDesc_s` and the `F77STRING` signatures of `qopen` and
//! `qinquire`, which are the VMS descriptor convention and are not compiled
//! on this platform (`F77STRING` is not defined by
//! `/tmp/imod-reference-build/include/imodconfig.h`); the `F77FUNCAP` name
//! mangling; and the `_WIN32` `_stat64` arm of `qinquire`.
//!
//! `errno` after a failed read or write: the C prints `strerror(errno)` only
//! when `errno` is non-zero, which for a short `fread` means only when the
//! short count came from an error rather than end of file.  [`b3d_fread`] and
//! [`b3d_fwrite`] return the item count as the C calls do and keep no error,
//! so the system line is not printed for those two; every realistic failure
//! here is end of file, where C prints no system line either.
use crate::imod::libcfshr::b3dutil::OsStrExt;
use std::cell::RefCell;

use crate::imod::libcfshr::b3dutil::{
    ImodFile, SEEK_CUR, SEEK_SET, b3d_fread, b3d_fseek, b3d_fwrite, mrc_big_seek, mrc_huge_seek,
};

/// Original: `MAX_MODE` (`blockio.c:70`), unused since the modes were
/// eliminated (`blockio.c:103`) but still declared.
pub const MAX_MODE: i32 = 17; /* JRK: max mode from 5 to 17 */
/// Original: `MAX_UNIT` (`blockio.c:71`).
pub const MAX_UNIT: i32 = 20; /* JRK: max unit from 5 to 10; DNM to 20 */

/* JRK: added these for the new attribute in Unit struct. */
/// Original: `UNIT_ATBUT_RO` (`blockio.c:74`).
pub const UNIT_ATBUT_RO: i32 = 1;
/// Original: `UNIT_ATBUT_NEW` (`blockio.c:75`).
pub const UNIT_ATBUT_NEW: i32 = 2;
/// Original: `UNIT_ATBUT_OLD` (`blockio.c:76`).
pub const UNIT_ATBUT_OLD: i32 = 3;
/// Original: `UNIT_ATBUT_SCRATCH` (`blockio.c:77`).
pub const UNIT_ATBUT_SCRATCH: i32 = 4;

/// `sizeof u->fname` (`char fname[326]`, `blockio.c:108`).
const FNAME_SIZE: i32 = 326;
/// `sizeof matstr` (`char matstr[16]`, `blockio.c:163`).
const MATSTR_SIZE: i32 = 16;

/// Original: `Unit` (`blockio.c:105-115`).
///
/// `fname` holds the C string's bytes without its terminator; `tail_name` is
/// the `char *tailName` pointer into it, as an offset.
#[derive(Default)]
struct Unit {
    being_used: i32,
    fname: Vec<u8>,
    fp: Option<ImodFile>,
    read_only: i32,
    write_only: i32,
    pos: u32,
    attribute: i32,   /* JRK: keep track of attibutes for files. */
    tail_name: usize, /* Pointer to filename only in fname */
}

/// The translation unit's file statics: `units` (`blockio.c:132`) and
/// `find_unit`'s `static int firstTime = 1` (`blockio.c:510`).
struct BlockIoState {
    units: Vec<Unit>,
    first_time: i32,
}

thread_local! {
    static BLOCKIO_STATE: RefCell<BlockIoState> = RefCell::new(BlockIoState {
        units: (0..MAX_UNIT).map(|_| Unit::default()).collect(),
        first_time: 1,
    });
}

/// The unit's file name as the C `%s` prints it.
fn fname_str(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Original: `qopen` (`blockio.c:144`), the non-`F77STRING` signature.
pub fn qopen(iunit: &mut i32, name: &str, attribute: &str) {
    BLOCKIO_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let unit = find_unit(&mut state);
        /* JRK: Style, declarations should be up here. */
        let mut mode: Option<usize> = None;
        let modes: [&str; 4] = ["rb", "rb+", "wb", "wb+"];

        if unit >= 0 {
            let u = &mut state.units[unit as usize];
            u.being_used = 1;
            u.fname = get_fstr(name.as_bytes(), FNAME_SIZE);
            u.write_only = 0;
            u.read_only = 0;
            u.pos = 0;
            let matstr = get_fstr(attribute.as_bytes(), MATSTR_SIZE);

            // `strncmp(matstr, "RO", 1) == 0` compares the first character
            // only, and an empty `matstr` compares its terminator.
            let first = matstr.first().copied().unwrap_or(0);
            if first == b'R' {
                mode = Some(0);
                u.attribute = UNIT_ATBUT_RO;
                u.read_only = 1;
            }
            if first == b'N' {
                let mut oldfilename = get_fstr(name.as_bytes(), FNAME_SIZE);

                /* DNM 10/20/03: check for existence of file before making backup,
                and delete old backup first */
                if std::env::var_os("IMOD_NO_IMAGE_BACKUP").is_none()
                    && std::fs::metadata(std::ffi::OsStr::from_bytes(&oldfilename)).is_ok()
                {
                    oldfilename.push(b'~');
                    let _ = std::fs::remove_file(std::ffi::OsStr::from_bytes(&oldfilename));
                    if let Err(error) = std::fs::rename(
                        std::ffi::OsStr::from_bytes(&u.fname),
                        std::ffi::OsStr::from_bytes(&oldfilename),
                    ) {
                        print!(
                            "\nWARNING: qopen - Could not rename '{}' to '{}'\n",
                            fname_str(&u.fname),
                            fname_str(&oldfilename)
                        );
                        if error.raw_os_error().is_some_and(|code| code != 0) {
                            print!("WARNING: from system - {}\n", error);
                        }
                    }
                }
                mode = Some(3);
                u.attribute = UNIT_ATBUT_NEW;
            }
            if first == b'O' {
                mode = Some(1);
                u.attribute = UNIT_ATBUT_OLD;
            }
            if first == b'S' {
                mode = Some(3);
                u.attribute = UNIT_ATBUT_SCRATCH;
            }

            // Source-level UB: an attribute starting with none of R/N/O/S
            // leaves `mode` uninitialised and indexes `modes` with it.  No
            // caller passes one (the only one is `readw_or_imod`'s 'RO').
            let mode = mode.expect("qopen: attribute leaves mode uninitialised (blockio.c:190)");
            let path = std::ffi::OsStr::from_bytes(&u.fname);
            u.fp = ImodFile::open(path, modes[mode]);
            if u.fp.is_none() {
                print!(
                    "\nERROR: qopen - Could not open '{}'\n",
                    fname_str(&u.fname)
                );
                // `errSave = errno`: the failed `open(2)` under `fopen` set it.
                let err_save = std::io::Error::last_os_error();
                if err_save.raw_os_error().is_some_and(|code| code != 0) {
                    print!("ERROR: from system - {}\n", err_save);
                }
                crate::imod::libcfshr::b3dutil::exit(3);
            }
            *iunit = unit + 1;

            /* Get the tail of the filename for other error messages */
            let tail_name = u.fname.iter().rposition(|&c| c == b'/');
            let tailback = u.fname.iter().rposition(|&c| c == b'\\');
            // `tailback > u->tailName` with a NULL `tailName` is true for any
            // non-NULL `tailback`.
            let tail_name = match (tail_name, tailback) {
                (None, back) => back,
                (Some(slash), Some(back)) if back > slash => Some(back),
                (slash, _) => slash,
            };
            u.tail_name = match tail_name {
                None => 0,
                Some(index) => index + 1,
            };
        } else {
            *iunit = -1;
        }
    });
}

/// Original: `qclose` (`blockio.c:224`).
pub fn qclose(iunit: i32) {
    BLOCKIO_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let unit = iunit - 1;
        if unit >= 0 && unit < MAX_UNIT {
            let u = &mut state.units[unit as usize];

            if u.being_used != 0 {
                u.being_used = 0;
                // `fclose(u->fp)`: dropping the stream flushes and closes it.
                u.fp = None;

                /* JRK: Delete scratch files */
                /* DNM 2/10/05: switch from unlink to remove to avoid unistd.h */
                if u.attribute == UNIT_ATBUT_SCRATCH {
                    let _ = std::fs::remove_file(std::ffi::OsStr::from_bytes(&u.fname));
                }
            }
        }
    });
}

/// Original: `qread` (`blockio.c:242`).
pub fn qread(iunit: i32, array: &mut [u8], nitems: i32, ier: &mut i32) {
    BLOCKIO_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let unit = iunit - 1;
        let u = check_unit(&mut state, unit, "qread", 0);
        if let Some(u) = u {
            let bc = nitems;
            if u.write_only != 0 {
                print!(
                    "\nERROR: qread - '{}' is write only.\n",
                    fname_str(&u.fname[u.tail_name..])
                );
                crate::imod::libcfshr::b3dutil::exit(3);
            }
            if b3d_fread(array, 1, bc as usize, u.fp.as_mut().unwrap()) != bc as usize {
                print!(
                    "\nERROR: qread - reading '{}'\n",
                    fname_str(&u.fname[u.tail_name..])
                );
                crate::imod::libcfshr::b3dutil::exit(3);
            }
            u.pos = u.pos.wrapping_add(bc as u32);
            *ier = 0;
        } else {
            *ier = -1;
        }
    });
}

/// Original: `qwrite` (`blockio.c:268`).
pub fn qwrite(iunit: i32, array: &[u8], nitems: i32) {
    BLOCKIO_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let unit = iunit - 1;

        let u = check_unit(&mut state, unit, "qwrite", 1).unwrap();
        let bc = nitems;
        if u.read_only != 0 {
            print!(
                "\nERROR: qwrite - '{}' is read only.\n",
                fname_str(&u.fname[u.tail_name..])
            );
            crate::imod::libcfshr::b3dutil::exit(3);
        }

        if b3d_fwrite(array, 1, bc as usize, u.fp.as_mut().unwrap()) != bc as usize {
            print!(
                "\nERROR: qwrite - writing '{}'\n",
                fname_str(&u.fname[u.tail_name..])
            );
            crate::imod::libcfshr::b3dutil::exit(3);
        }
        u.pos = u.pos.wrapping_add(bc as u32);
    });
}

/* DNM 10/23/00: switch from using system-dependent "seek_name" to call a
big_seek function that seeks in chunks less than 2 GB; also change test
for error to test for -1 returned rather than a negative number.
Change to test for nonzero when switch to fseek */
/// Original: `qseek` (`blockio.c:295`).
pub fn qseek(iunit: i32, base: i32, line: i32, section: i32, nxbytes: i32, nylines: i32) {
    BLOCKIO_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let unit = iunit - 1;
        let u = check_unit(&mut state, unit, "qseek", 1).unwrap();
        u.pos = (nxbytes as u32)
            .wrapping_mul(
                ((line - 1) as u32)
                    .wrapping_add((nylines as u32).wrapping_mul((section - 1) as u32)),
            )
            .wrapping_add((base - 1) as u32);

        /*  if (lseek(u->fp, u->pos = pos, 0) < 0) */
        if mrc_huge_seek(
            u.fp.as_mut().unwrap(),
            base - 1,
            0,
            line - 1,
            section - 1,
            nxbytes,
            nylines,
            1,
            SEEK_SET,
        ) != 0
        {
            print!(
                "\nERROR: qseek - Doing mrcHugeSeek in '{}'\n",
                fname_str(&u.fname[u.tail_name..])
            );
            crate::imod::libcfshr::b3dutil::exit(3);
        }
    });
}

/* qback is used only for small movements, within a section, so
it doesn't need to call big_seek.  However, change the test for error to
test for = -1 instead of < 0; then to test for !=0 when switch to fseek */
/// Original: `qback` (`blockio.c:321`).
pub fn qback(iunit: i32, ireclength: i32) {
    BLOCKIO_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let unit = iunit - 1;
        let u = check_unit(&mut state, unit, "qback", 1).unwrap();
        let amt = ireclength.wrapping_neg();
        u.pos = u.pos.wrapping_add(amt as u32);
        if b3d_fseek(u.fp.as_mut().unwrap(), amt, SEEK_CUR) != 0 {
            print!(
                "\nERROR: qback - Doing seek in '{}'\n",
                fname_str(&u.fname[u.tail_name..])
            );
            crate::imod::libcfshr::b3dutil::exit(3);
        }
    });
}

/* qskip needs to move by large amounts within sections so it takes two numbers
and moves by the product */
/// Original: `qskip` (`blockio.c:339`).
pub fn qskip(iunit: i32, ireclength: i32, nrecords: i32) {
    BLOCKIO_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let unit = iunit - 1;
        let u = check_unit(&mut state, unit, "qskip", 1).unwrap();
        u.pos = u.pos.wrapping_add(ireclength.wrapping_mul(nrecords) as u32);
        if mrc_big_seek(u.fp.as_mut().unwrap(), 0, ireclength, nrecords, SEEK_CUR) != 0 {
            print!(
                "\nERROR: qskip - Doing seek in '{}'\n",
                fname_str(&u.fname[u.tail_name..])
            );
            crate::imod::libcfshr::b3dutil::exit(3);
        }
    });
}

/// Original: `qinquire` (`blockio.c:362`), the non-`F77STRING`, non-`_WIN32`
/// arm.
///
/// `stat`'s return is not tested in the source, so a failed `stat` leaves
/// `buf` uninitialised and the size is garbage; the translation reports 0 for
/// that case.
pub fn qinquire(iunit: i32, filename: &mut [u8], flen: &mut i32) {
    BLOCKIO_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let unit = iunit - 1;
        let u = check_unit(&mut state, unit, "qinquire", 0);
        if let Some(u) = u {
            /* generic stat is not good enough on Windows */
            let st_size = std::fs::metadata(std::ffi::OsStr::from_bytes(&u.fname))
                .map(|buf| buf.len() as i64)
                .unwrap_or(0);
            set_fstr(filename, &u.fname);
            *flen = (st_size as f64 / 1024.) as i32;
        } else {
            /* DNM: HUH? */
            /* set_fstr(filename, filename_l, ""); */
            *flen = -1;
        }
    });
}

/* This will fail for large files, but try to keep pos good up to 4 GB by
making it an unsigned int */
/// Original: `qlocate` (`blockio.c:398`).
pub fn qlocate(iunit: i32, location: &mut i32) {
    BLOCKIO_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let unit = iunit - 1;
        let u = check_unit(&mut state, unit, "qlocate", 1).unwrap();
        *location = u.pos.wrapping_add(1) as i32;
    });
}

/// Original: `fill` (`blockio.c:405`).
///
/// `b` is a Fortran `CHARACTER` argument of hidden length `lb`, and
/// `strlen(b)` scans it for a terminator a Fortran string does not carry; the
/// translation takes the scan to end at the slice's end when no NUL is in it.
pub fn fill(a: &mut [u8], b: &[u8], np: i32, mut lb: i32) {
    let mut n = np;
    let mut ai = 0usize;

    let strlen = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    if strlen == 1
    /* MWT :: temporary fix 20.AUG.91 */
    {
        lb = np;
        n = 1;
    }

    while n > 0 {
        n -= 1;
        let mut l = lb;
        let mut c = 0usize;
        while l > 0 {
            l -= 1;
            a[ai] = b[c];
            ai += 1;
            c += 1;
        }
    }
}

/************************************************************************
 ***************************    static FUNCTIONS    ********************
 ************************************************************************/

/* Compares a C and Fortran string */

/************************************************************************/
/* Creates a c string from a fortran one */
/// Original: `get_fstr` (`blockio.c:461`).
///
/// Returns the bytes of the C string the source leaves in `str`: at most
/// `l - 1` characters, ending after the last character that is neither blank
/// nor NUL.  A NUL earlier than that would end the C string there for every
/// consumer (`fopen`, `%s`, `strlen`), so the result is cut at it too.
fn get_fstr(fstr: &[u8], l: i32) -> Vec<u8> {
    let mut lfstr = fstr.len() as i32;
    let mut l = l;
    let mut i = 0usize;
    let mut lnblnk: i32 = -1;
    let mut str_ = Vec::new();

    /* Keep track of last non blank, non null character and put null after it */
    while lfstr > 0 && l > 1 {
        if fstr[i] != b' ' && fstr[i] != 0 {
            lnblnk = i as i32;
        }
        str_.push(fstr[i]);
        i += 1;
        lfstr -= 1;
        l -= 1;
    }
    str_.truncate((lnblnk + 1) as usize);
    if let Some(end) = str_.iter().position(|&c| c == 0) {
        str_.truncate(end);
    }
    str_
} /*get_fstr*/

/************************************************************************/
/* Creates a fortran string from a c one */
/// Original: `set_fstr` (`blockio.c:479`).
fn set_fstr(fstr: &mut [u8], str_: &[u8]) {
    let mut lfstr = fstr.len();
    let mut fi = 0usize;
    let mut si = 0usize;
    while lfstr > 0 && si < str_.len() && str_[si] != 0 {
        fstr[fi] = str_[si];
        fi += 1;
        si += 1;
        lfstr -= 1;
    }
    while lfstr > 0 {
        fstr[fi] = b' ';
        fi += 1;
        lfstr -= 1;
    }
}

/// Original: `find_unit` (`blockio.c:494`).
fn find_unit(state: &mut BlockIoState) -> i32 {
    if state.first_time != 0 {
        for i in 0..MAX_UNIT as usize {
            state.units[i].being_used = 0;
        }
    }
    state.first_time = 0;

    for i in 0..MAX_UNIT as usize {
        if state.units[i].being_used == 0 {
            return i as i32;
        }
    }
    -1
}

/* Checks for legal unit number and whether unit is open, gives error
message with function name and unit number, exits if doExit set */
/// Original: `check_unit` (`blockio.c:527`).
fn check_unit<'a>(
    state: &'a mut BlockIoState,
    unit: i32,
    function: &str,
    do_exit: i32,
) -> Option<&'a mut Unit> {
    if unit < 0 || unit >= MAX_UNIT {
        print!(
            "\nERROR: {} - {} is not a legal unit number.\n",
            function,
            unit + 1
        );
        if do_exit != 0 {
            crate::imod::libcfshr::b3dutil::exit(3);
        }
        return None;
    }
    let u = &mut state.units[unit as usize];
    if u.being_used == 0 {
        print!("\nERROR: {} - unit {} is not open.\n", function, unit + 1);
        if do_exit != 0 {
            crate::imod::libcfshr::b3dutil::exit(3);
        }
        return None;
    }
    Some(u)
}
