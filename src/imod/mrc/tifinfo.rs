//! Translation of `IMOD/mrc/tifinfo.c` -- print the IFD entries of TIFF
//! files (marked `UNSUPPORTED` upstream, and not built by
//! `IMOD/mrc/Makefile`; its native reference is compiled from the vendored
//! source, see `fixtures/tifinfo/make-goldens.sh`).
//!
//! **The source assumes a big-endian host.**  It reads every field in the
//! host's byte order and swaps whenever the file's order word is *not*
//! `0x4d4d` (`tifinfo.c:78`), and it shifts a short `value` down by 16 bits
//! only when it did not swap (`:124-125`).  On a little-endian host both are
//! backwards, and the native binary cannot read any real TIFF (an endless
//! `Reading 0 entries` loop for `MM`, a segfault on a garbage type index for
//! `II`; `BUGS.md`).
//!
//! **Fixed in translation (BUGS.md):** the translation behaves on every
//! host as the source does on the big-endian host it was written for: it
//! swaps when the file's byte order differs from the host's, and applies
//! the short-value shift to `MM` files.  Beyond that it also swaps the
//! `-v` short and long data values (the source never does), prints an
//! entry type past the six-name table as its number instead of reading past
//! the table, and stops walking the IFD chain when a read fails or an
//! offset repeats instead of looping forever.
//!
//! Every C stream read is `fread` through [`b3d_fread`], which leaves the
//! destination unchanged past what the file held, as `fread` does.  Locals
//! the source leaves uninitialised start at zero here -- see the notes at
//! each one.

use crate::imod::libcfshr::b3dutil::OsStrExt as _;
use std::ffi::OsString;
use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::sync::atomic::{AtomicI32, Ordering};

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, b3d_fread, b3d_rewind, c_format_bytes, exit};

/// `int Verbose = 0;` (`tifinfo.c:30`).
static VERBOSE: AtomicI32 = AtomicI32::new(0);

/// `char *typeStrings[]` (`tifinfo.c:32`).
static TYPE_STRINGS: [&str; 6] = ["NULL ", "BYTE ", "ASCII", "SHORT", "LONG ", "RATIONAL"];

/// C `swap` (`tifinfo.c:36`): reverses the first `size` bytes of `ptr`,
/// rounded down to an even count.
fn swap(ptr: &mut [u8], mut size: u32) {
    let mut begin: usize;
    let mut end: usize;
    let mut tmp: u8;

    if (size % 2) != 0 {
        size -= 1;
    }

    begin = 0;
    end = size.wrapping_sub(1) as usize;

    let mut i: i32 = 0;
    while (i as u32) < (size / 2) {
        tmp = ptr[begin];
        ptr[begin] = ptr[end];
        ptr[end] = tmp;

        begin += 1;
        end = end.wrapping_sub(1);
        i += 1;
    }
}

/// C `tiff_print_info` (`tifinfo.c:61`).
fn tiff_print_info(fp: &mut ImodFile) {
    // `unsigned short byteOrder, versionNumber, IFDentries;` and
    // `unsigned int IFDoffset;` are uninitialised in the source, so a failed
    // `fread` leaves stack residue in them; zero here.  For `IFDentries`
    // that is also what the native build shows: an `IFDoffset` past the end
    // of the file reads nothing and it prints `Reading 0 entries`.
    let mut byte_order = [0u8; 2];
    let mut version_number = [0u8; 2];
    let mut ifd_entries = [0u8; 2];
    let mut ifdi: i32 = 1;
    let mut i: u32;
    let mut ifd_offset = [0u8; 4];
    let mut data_index: i32;
    let mut dsize: i32;
    let mut swap_data: i16 = 0;
    // The loop body's `tag`, `type`, `length` and `value` are uninitialised
    // block locals; a failed read leaves the previous entry's value in the
    // same stack slot, which keeping them here reproduces.
    let mut entry = [0u8; 12];
    let mut ferror = false;
    let mut out = ImodFile::Stdout;
    b3d_rewind(fp);

    b3d_fread(&mut byte_order, 2, 1, fp);
    let byte_order_v = u16::from_ne_bytes(byte_order);

    if (byte_order_v != 0x4949) && (byte_order_v != 0x4d4d) {
        let _ = out.write_all(&c_format_bytes(
            "Not a tif file: first word must be 0x4949 or 0x4d4d,  not %x\n",
            &[CArg::Uint(byte_order_v as u64)],
        ));
        return;
    }
    // BUGS.md, fixed in translation: swap when the file's order differs
    // from the host's (the source's `byteOrder != 0x4d4d` is that test on a
    // big-endian host only).
    let file_big_endian = byte_order == *b"MM";
    if file_big_endian != cfg!(target_endian = "big") {
        swap_data = 1;
    }
    let _ = out.write_all(&c_format_bytes(
        "TIFF: %x",
        &[CArg::Uint(byte_order_v as u64)],
    ));

    b3d_fread(&mut version_number, 2, 1, fp);

    if swap_data != 0 {
        swap(&mut version_number, 2);
    }
    let version_number_v = u16::from_ne_bytes(version_number);

    if (version_number_v != 0x002a) && (version_number_v != 0x2a00) {
        let _ = out.write_all(&c_format_bytes(
            "\nBad TIFF Version number. Must be 42, not %d\n",
            &[CArg::Int(version_number_v as i64)],
        ));
        return;
    }
    let _ = out.write_all(&c_format_bytes(
        " %x\n",
        &[CArg::Uint(version_number_v as u64)],
    ));

    b3d_fread(&mut ifd_offset, 4, 1, fp);
    if swap_data != 0 {
        swap(&mut ifd_offset, 4);
    }

    let mut visited: Vec<u32> = Vec::new();
    while u32::from_ne_bytes(ifd_offset) != 0 {
        let offset = u32::from_ne_bytes(ifd_offset);
        // BUGS.md, fixed in translation: a repeated offset would walk the
        // same records forever.
        if visited.contains(&offset) {
            return;
        }
        visited.push(offset);
        let _ = fp.seek(SeekFrom::Start(offset as u64));
        // A failed read leaves the source's `IFDentries` holding its old
        // value; BUGS.md, fixed in translation: no entries are read then.
        if b3d_fread(&mut ifd_entries, 2, 1, fp) != 1 {
            ifd_entries = [0, 0];
        }
        if swap_data != 0 {
            swap(&mut ifd_entries, 2);
        }
        let entries = u16::from_ne_bytes(ifd_entries);
        let _ = out.write_all(&c_format_bytes(
            "Reading %d entries in record %d at %d:\n",
            &[
                CArg::Int(entries as i64),
                CArg::Int(ifdi as i64),
                CArg::Int(offset as i32 as i64),
            ],
        ));
        ifdi += 1;
        i = 0;
        while i < entries as u32 {
            // `fread(&tag, 2, 1, fp); fread(&type, 2, 1, fp);
            // fread(&length, 4, 1, fp); fread(&value, 4, 1, fp);` -- four
            // consecutive reads of adjacent fields, each of which keeps what
            // it got at end of file; one byte-wise read of the twelve bytes
            // leaves the same bytes behind.  An `Err` is what sets the
            // stream's error indicator, which `ferror` then reports.
            let mut done = 0usize;
            while done < 12 {
                match fp.read(&mut entry[done..12]) {
                    Ok(0) => break,
                    Ok(n) => done += n,
                    Err(_) => {
                        ferror = true;
                        break;
                    }
                }
            }

            if ferror {
                // `perror("tifinfo")`.
                let message = std::io::Error::last_os_error().to_string();
                let message = message
                    .split(" (os error ")
                    .next()
                    .unwrap_or(message.as_str());
                let _ = ImodFile::Stderr
                    .write_all(&c_format_bytes("tifinfo: %s\n", &[CArg::Str(message)]));
                return;
            }

            if swap_data != 0 {
                swap(&mut entry[0..2], 2);
                swap(&mut entry[2..4], 2);
                swap(&mut entry[4..8], 4);
                swap(&mut entry[8..12], 4);
            }
            let tag = u16::from_ne_bytes([entry[0], entry[1]]);
            let type_ = u16::from_ne_bytes([entry[2], entry[3]]);
            let length = u32::from_ne_bytes([entry[4], entry[5], entry[6], entry[7]]);
            let mut value = u32::from_ne_bytes([entry[8], entry[9], entry[10], entry[11]]);
            // The source's `else if` is the no-swap branch on a big-endian
            // host, i.e. an `MM` file.
            if file_big_endian && (type_ == 3) && (length < 3) {
                value >>= 16;
                entry[8..12].copy_from_slice(&value.to_ne_bytes());
            }

            // `typeStrings[type]` with `type` past the six-entry table
            // (`tifinfo.c:127-128`) reads beyond the array -- undefined
            // behaviour.  BUGS.md, fixed in translation: such a type (a valid
            // TIFF 6 type such as UNDEFINED or DOUBLE) is printed as its number.
            let unknown_type;
            let type_string = match TYPE_STRINGS.get(type_ as usize) {
                Some(name) => *name,
                None => {
                    unknown_type = format!("({type_})");
                    unknown_type.as_str()
                }
            };
            let _ = out.write_all(&c_format_bytes(
                "\t%d %s %d %d ",
                &[
                    CArg::Int(tag as i64),
                    CArg::Str(type_string),
                    CArg::Int(length as i32 as i64),
                    CArg::Int(value as i32 as i64),
                ],
            ));

            dsize = match type_ {
                1 | 2 => length as i32,
                3 => length.wrapping_mul(2) as i32,
                4 => length.wrapping_mul(4) as i32,
                _ => 0,
            };

            if VERBOSE.load(Ordering::Relaxed) != 0 && dsize > 4 {
                // `int pos = ftell(fp);`
                let pos = fp.tell() as i32;
                // `char bbuf; short sbuf; int lbuf;` -- uninitialised.
                let mut bbuf = [0u8; 1];
                let mut sbuf = [0u8; 2];
                let mut lbuf = [0u8; 4];
                let _ = fp.seek(SeekFrom::Start(value as u64));
                let _ = out.write_all(b"\t : ");

                data_index = 0;
                while (data_index as u32) < length {
                    match type_ {
                        1 => {
                            // byte
                            b3d_fread(&mut bbuf, 1, 1, fp);
                            let _ = out.write_all(&c_format_bytes(
                                "%d ",
                                &[CArg::Int(bbuf[0] as i8 as i64)],
                            ));
                        }
                        2 => {
                            // ascii
                            b3d_fread(&mut bbuf, 1, 1, fp);
                            let _ = out.write_all(&c_format_bytes("%c", &[CArg::Chr(bbuf[0])]));
                        }
                        3 => {
                            // short
                            b3d_fread(&mut sbuf, 2, 1, fp);
                            // BUGS.md, fixed in translation: the source never
                            // swaps the data values.
                            if swap_data != 0 {
                                swap(&mut sbuf, 2);
                            }
                            let _ = out.write_all(&c_format_bytes(
                                "%d ",
                                &[CArg::Int(i16::from_ne_bytes(sbuf) as i64)],
                            ));
                        }
                        4 => {
                            // long
                            b3d_fread(&mut lbuf, 4, 1, fp);
                            if swap_data != 0 {
                                swap(&mut lbuf, 4);
                            }
                            let _ = out.write_all(&c_format_bytes(
                                "%d ",
                                &[CArg::Int(i32::from_ne_bytes(lbuf) as i64)],
                            ));
                        }
                        // rational, default
                        _ => {}
                    }
                    data_index = data_index.wrapping_add(1);
                }
                // `fseek(fp, pos, SEEK_SET)`: a negative `pos` fails.
                if pos >= 0 {
                    let _ = fp.seek(SeekFrom::Start(pos as u64));
                }
            }

            let _ = out.write_all(b"\n");
            i += 1;
        }
        // `(IFDoffset + 2) + (IFDentries * 12)`: unsigned int arithmetic.
        let next = offset
            .wrapping_add(2)
            .wrapping_add((entries as i32 * 12) as u32);
        let _ = fp.seek(SeekFrom::Start(next as u64));
        // BUGS.md, fixed in translation: a failed read leaves the source's
        // `IFDoffset` unchanged and it re-reads the same record forever; the
        // walk ends there instead.
        if b3d_fread(&mut ifd_offset, 4, 1, fp) != 1 {
            return;
        }
        if swap_data != 0 {
            swap(&mut ifd_offset, 4);
        }
    }
}

/// C `main` in `tifinfo.c` (`tifinfo.c:199`).
pub fn tifinfo(argv: &[OsString]) -> ! {
    let argc = argv.len();
    let mut first: usize = 1;

    // The static initialiser of `Verbose`, which a process gets once; an
    // in-process run gets it here.
    VERBOSE.store(0, Ordering::Relaxed);

    if argc > 2 {
        let a1 = argv[1].as_bytes();
        if a1.first() == Some(&b'-') {
            if a1.get(1) == Some(&b'v') {
                VERBOSE.store(1, Ordering::Relaxed);
                first += 1;
            }
        }
    }

    let mut i = first;
    while i < argc {
        let Some(mut fin) = ImodFile::open(&argv[i], "rb") else {
            exit(1);
        };
        tiff_print_info(&mut fin);
        drop(fin);
        i += 1;
    }

    exit(0);
}
