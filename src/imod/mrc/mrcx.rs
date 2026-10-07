//! Translation of `IMOD/mrc/mrcx.c` -- convert an MRC file between big- and
//! little-endian IEEE (and, on a big-endian host, VMS) byte order.
//!
//! **Configuration.**  The generated `imodconfig.h`
//! (`/tmp/imod-reference-build/include/imodconfig.h`, written by
//! `IMOD/setup`) defines both `B3D_LITTLE_ENDIAN` and `SWAP_IEEE_FLOATS` on
//! this platform, so:
//!
//! * `swapieee` is TRUE (`mrcx.c:93-94`) and `floatFunc` is always
//!   `convertLongs`; `convertFloats` -- the VAX conversion -- is compiled and
//!   selected only on the `swapieee == FALSE` arm, which nothing on this
//!   platform reaches.  It is translated because it is part of the unit and
//!   that arm still names it.
//! * the argument parser is the `#else` arm of `#ifndef B3D_LITTLE_ENDIAN`
//!   (`:132-138`): no `-vms`/`-ieee` options, and any `-` argument is a usage
//!   error.  The `#ifndef` arm (`:100-130`) is not what the compiler sees here
//!   and is not translated.
//! * the direction message is the `#ifdef B3D_LITTLE_ENDIAN` arm (`:258-261`).
//!
//! **Byte images.**  The source reads and writes `struct MRCheader` through
//! its first `56 * 4` bytes (`fread(&hdata, 56, 4, fin)`,
//! `fwrite(&hconv, 56, 4, fout)`), relying on the struct having no padding
//! there, and `convertHeader` fills that struct field by field through
//! `(b3dUByte *)&header->field` pointers.  The fields tile bytes 0-223
//! exactly, so the translation holds those 224 bytes as the array the C
//! treats them as, and each `&header->field` becomes the field's byte offset
//! (named at each call).  The Rust [`MrcHeader`] has no C layout and is used
//! only for the fields `main` inspects.
//!
//! Every `fread`/`fwrite` is [`b3d_fread`]/[`b3d_fwrite`], the tree's stdio
//! translation: a short read leaves the rest of the buffer as it was, as
//! `fread` does.  In place, the source opens the file twice (`"rb"` and
//! `"rb+"`) and writes each piece after reading it at the same offset, so the
//! reading stream always stays ahead of the writing one; two [`ImodFile`]s
//! reproduce that, and the file keeps any bytes past the image data.

use crate::imod::libcfshr::b3dutil::OsStrExt as _;
use std::ffi::OsString;
use std::io::{Seek as _, SeekFrom, Write as _};

use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_fread, b3d_fwrite, b3d_rewind, c_format_bytes, exit, imod_prog_name,
};
use crate::imod::libiimod::mrcfiles::{
    MRC_LABEL_SIZE, MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT,
    MRC_MODE_RGB, MRC_MODE_SHORT, MRC_NLABELS, MrcHeader, mrc_swap_longs, mrc_swap_shorts,
    mrc_test_size,
};

/// `void (*convertFunc)()` / `void (*floatFunc)()`: every conversion routine
/// has the signature `(FILE *fp, b3dUByte *buffer, int n, int direction)`.
type ConvertFunc = fn(&mut ImodFile, &mut [u8], i32, i32);

/// C `main` in `mrcx.c` (`mrcx.c:74`).
pub fn mrcx(argv: &[OsString]) -> ! {
    let convert_func: ConvertFunc;
    let float_func: ConvertFunc;
    // `struct MRCheader hdata`: its first `56 * 4` bytes as read, and the
    // fields `main` works with.
    let mut hbuf = [0u8; 56 * 4];
    let mut hdata = MrcHeader::default();
    // `struct MRCheader hconv`, filled by `convertHeader`.
    let mut hconv = [0u8; 56 * 4];
    let mut hconv_labels = [[0u8; MRC_LABEL_SIZE + 1]; MRC_NLABELS];
    let mut hconv_symops: Option<Vec<u8>> = None;
    let factor: i32;
    let size: i32;
    let tonative: i32;
    let swapieee: bool;
    let mut i: usize;
    let inplace: bool;
    let filesize: i32;
    let mut datasize: i32;
    let argc = argv.len();
    let progname = imod_prog_name(
        &argv
            .first()
            .map_or(String::new(), |a| a.to_string_lossy().into_owned()),
    );
    let progname = progname.as_str();

    // `#ifdef SWAP_IEEE_FLOATS`: defined on this platform.
    swapieee = true;

    // `#else` arm of `#ifndef B3D_LITTLE_ENDIAN` (`mrcx.c:132-138`).
    if argc < 2 || argc > 3 || argv[1].as_bytes().first() == Some(&b'-') {
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "%s: Usage [filename] [opt filename]\n",
            &[CArg::Str(progname)],
        ));
        let _ = ImodFile::Stderr
            .write_all(b"There is no option to convert VMS data on this machine.\n");
        exit(1);
    }
    i = 1;

    let Some(mut fin) = ImodFile::open(&argv[i], "rb") else {
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "%s: Couldn't open %s.\n",
            &[CArg::Str(progname), CArg::Bytes(argv[i].as_bytes())],
        ));
        exit(10);
    };
    i += 1;

    let fout = if i >= argc {
        inplace = true;
        ImodFile::open(&argv[i - 1], "rb+")
    } else {
        inplace = false;
        ImodFile::open(&argv[i], "wb")
    };

    let Some(mut fout) = fout else {
        // In place, `argv[i]` is `argv[argc]`, the terminating NULL, which
        // glibc's `printf` prints as `(null)`.
        let name: &[u8] = if i < argc {
            argv[i].as_bytes()
        } else {
            b"(null)"
        };
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "%s: Couldn't open %s.\n",
            &[CArg::Str(progname), CArg::Bytes(name)],
        ));
        exit(10);
    };

    if swapieee {
        float_func = convert_longs;
    } else {
        float_func = convert_floats;
    }

    if b3d_fread(&mut hbuf, 56, 4, &mut fin) == 0 {
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "%s: Error Reading header.\n",
            &[CArg::Str(progname)],
        ));
        exit(3);
    }
    {
        let long_at = |o: usize| i32::from_ne_bytes(hbuf[o..o + 4].try_into().unwrap());
        let short_at = |o: usize| i16::from_ne_bytes(hbuf[o..o + 2].try_into().unwrap());
        hdata.nx = long_at(0);
        hdata.ny = long_at(4);
        hdata.nz = long_at(8);
        hdata.mode = long_at(12);
        hdata.nxstart = long_at(16);
        hdata.nystart = long_at(20);
        hdata.nzstart = long_at(24);
        hdata.mx = long_at(28);
        hdata.my = long_at(32);
        hdata.mz = long_at(36);
        hdata.mapc = long_at(64);
        hdata.mapr = long_at(68);
        hdata.maps = long_at(72);
        hdata.next = long_at(92);
        hdata.nint = short_at(128);
        hdata.nreal = short_at(130);
        hdata.sub = short_at(132);
        hdata.zfac = short_at(134);
        hdata.cmap.copy_from_slice(&hbuf[208..212]);
        hdata.nlabl = long_at(220);
    }

    hdata.swapped = 0;

    // Test for byte-swapped data with image size and the map numbers
    if mrc_test_size(&hdata) != 0 {
        // Mark data as swapped and do the swaps of critical data
        hdata.swapped = 1;
        let mut longs = [
            hdata.nx,
            hdata.ny,
            hdata.nz,
            hdata.mode,
            hdata.nxstart,
            hdata.nystart,
            hdata.nzstart,
            hdata.mx,
            hdata.my,
            hdata.mz,
        ];
        mrc_swap_longs(&mut longs, 10);
        [
            hdata.nx,
            hdata.ny,
            hdata.nz,
            hdata.mode,
            hdata.nxstart,
            hdata.nystart,
            hdata.nzstart,
            hdata.mx,
            hdata.my,
            hdata.mz,
        ] = longs;
        let mut map = [hdata.mapc, hdata.mapr, hdata.maps];
        mrc_swap_longs(&mut map, 3);
        [hdata.mapc, hdata.mapr, hdata.maps] = map;
        let mut next = [hdata.next];
        mrc_swap_longs(&mut next, 1);
        hdata.next = next[0];
        let mut shorts = [hdata.nint, hdata.nreal, hdata.sub, hdata.zfac];
        mrc_swap_shorts(&mut shorts, 4);
        [hdata.nint, hdata.nreal, hdata.sub, hdata.zfac] = shorts;
        let mut nlabl = [hdata.nlabl];
        mrc_swap_longs(&mut nlabl, 1);
        hdata.nlabl = nlabl[0];
    }
    if mrc_test_size(&hdata) != 0 || hdata.mode > 16 || hdata.mode < 0 {
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "%s: This is not an MRC file, even after swapping bytes in header.\n",
            &[CArg::Str(progname)],
        ));
        exit(3);
    }

    hdata.header_size = 1024;
    hdata.header_size = hdata.header_size.wrapping_add(hdata.next);
    hdata.section_skip = 0;
    tonative = hdata.swapped;
    datasize = hdata.nx.wrapping_mul(hdata.ny).wrapping_mul(hdata.nz);
    // `fseek(fin, 0, SEEK_END); filesize = ftell(fin);` into an `int`.
    let _ = fin.seek(SeekFrom::End(0));
    filesize = fin.tell() as i32;
    match hdata.mode {
        MRC_MODE_BYTE => {
            convert_func = convert_bytes;
            size = 1;
            factor = 1;
        }

        MRC_MODE_SHORT => {
            datasize = datasize.wrapping_mul(2);
            convert_func = convert_shorts;
            size = 2;
            factor = 1;
        }

        MRC_MODE_FLOAT => {
            datasize = datasize.wrapping_mul(4);
            convert_func = float_func;
            size = 4;
            factor = 1;
        }

        MRC_MODE_COMPLEX_SHORT => {
            datasize = datasize.wrapping_mul(2 * 2);
            convert_func = convert_shorts;
            size = 2;
            factor = 2;
        }

        MRC_MODE_COMPLEX_FLOAT => {
            datasize = datasize.wrapping_mul(2 * 4);
            convert_func = float_func;
            size = 4;
            factor = 2;
        }

        MRC_MODE_RGB => {
            datasize = datasize.wrapping_mul(3);
            convert_func = convert_bytes;
            size = 1;
            factor = 3;
        }

        _ => {
            let _ = ImodFile::Stderr.write_all(&c_format_bytes(
                "%s: data type %d unsupported.\n",
                &[CArg::Str(progname), CArg::Int(hdata.mode as i64)],
            ));
            exit(3);
        }
    }

    datasize = datasize.wrapping_add(hdata.header_size);
    if filesize < datasize || filesize > datasize.wrapping_add(511) {
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "mrcx: Warning input file is %d bytes, expected size is %d.\n",
            &[CArg::Int(filesize as i64), CArg::Int(datasize as i64)],
        ));
    }
    b3d_rewind(&mut fin);

    // `#ifdef B3D_LITTLE_ENDIAN`
    if tonative != 0 {
        let _ = ImodFile::Stdout.write_all(b"Converting big-endian IEEE to little-endian IEEE.\n");
    } else {
        let _ = ImodFile::Stdout.write_all(b"Converting little-endian IEEE to big-endian IEEE.\n");
    }

    convert_header(
        float_func,
        hdata.next,
        &mut hconv,
        &mut hconv_labels,
        &mut hconv_symops,
        &mut fin,
        &mut fout,
        tonative,
        &hdata.cmap,
    );

    b3d_fwrite(&hconv, 56, 4, &mut fout);

    i = 0;
    while i < MRC_NLABELS {
        b3d_fwrite(&hconv_labels[i], MRC_LABEL_SIZE, 1, &mut fout);
        i += 1;
    }

    if hdata.next != 0 {
        // A negative `next` leaves `hconv.symops` an uninitialised pointer
        // (`convertHeader` allocates only for `nextra > 0`) and passes the
        // negative size to `fwrite` as a huge `size_t` -- undefined
        // behaviour with nothing to reproduce.  BUGS.md, fixed in
        // translation: nothing is written then.
        if let Some(symops) = hconv_symops.as_ref() {
            b3d_fwrite(symops, hdata.next as usize, 1, &mut fout);
        }
    }

    if !inplace || !std::ptr::fn_addr_eq(convert_func, convert_bytes as ConvertFunc) {
        convert_body(
            convert_func,
            &hdata,
            size,
            factor,
            tonative,
            &mut fin,
            &mut fout,
        );
    }

    drop(fin);
    // In place, `fout` is not closed; `exit` flushes it.
    if !inplace {
        drop(fout);
    }
    exit(0);
}

/// C `convertBody` (`mrcx.c:305`): sets up the image data conversion and
/// writes the converted data to the output file.
fn convert_body(
    convert_func: ConvertFunc,
    header: &MrcHeader,
    mut size: i32,
    factor: i32,
    direction: i32,
    infp: &mut ImodFile,
    outfp: &mut ImodFile,
) {
    let mut lcv: u32;
    let mut scan_line: Vec<u8>; // converted scanline
    let scans: i32;
    let mut pix = [0u8; 1];
    let mut i: i32;

    if size == 1 {
        // BUGS.md, fixed in translation: `mrcx.c:324` copies `nx * ny * nz`
        // bytes whatever `factor` is, so native writes only a third of an RGB
        // (mode 16, `factor` 3) body to a new file.  Every byte is copied here.
        size = header
            .nx
            .wrapping_mul(header.ny)
            .wrapping_mul(header.nz)
            .wrapping_mul(factor);
        i = 0;
        while i < size {
            if b3d_fread(&mut pix, 1, 1, infp) == 0 {
                // `perror("Error copying data ")`.
                let message = std::io::Error::last_os_error().to_string();
                let message = message
                    .split(" (os error ")
                    .next()
                    .unwrap_or(message.as_str());
                let _ = ImodFile::Stderr.write_all(&c_format_bytes(
                    "Error copying data : %s\n",
                    &[CArg::Str(message)],
                ));
                return;
            }
            b3d_fwrite(&pix, 1, 1, outfp);
            i += 1;
        }
        return;
    }

    scan_line = vec![0u8; header.nx.wrapping_mul(size).wrapping_mul(factor) as usize];

    scans = header.ny.wrapping_mul(header.nz);

    lcv = 0;
    while lcv < scans as u32 {
        convert_func(
            infp,
            &mut scan_line,
            header.nx.wrapping_mul(factor),
            direction,
        );
        b3d_fwrite(
            &scan_line,
            size as usize,
            header.nx.wrapping_mul(factor) as usize,
            outfp,
        );
        lcv += 1;
    }
}

/// C `convertBytes` (`mrcx.c:352`): converts byte data.
fn convert_bytes(fp: &mut ImodFile, buffer: &mut [u8], no_bytes: i32, _direction: i32) {
    // Read in data
    if b3d_fread(buffer, 1, no_bytes as usize, fp) != no_bytes as usize {
        let _ = ImodFile::Stderr.write_all(b"Read Failed While Reading Byte Quantities\n");
    }
}

/// C `convertFloats` (`mrcx.c:372`): converts a floating point number in VAX
/// representation to IEEE and vice versa.  Selected only when `swapieee` is
/// FALSE, which this platform's configuration never makes it.
fn convert_floats(fp: &mut ImodFile, buffer: &mut [u8], no_floats: i32, direction: i32) {
    let mut exp: u8;
    let mut lcv: u32;
    let no_bytes: u32;
    let mut temp: u8; // temporary place holder

    no_bytes = (no_floats as u32).wrapping_mul(4);

    if b3d_fread(buffer, 1, no_bytes as usize, fp) != no_bytes as usize {
        let _ = ImodFile::Stderr.write_all(b"Read Failed While Reading Float Quantities\n");
    }

    lcv = 0;
    while lcv < no_bytes {
        let l = lcv as usize;
        if direction != 0 {
            // FROM_VAX
            // `(buffer[lcv+1] << 1) | ...` is an int, stored into the
            // `b3dUByte exp` before the comparison.
            exp = ((buffer[l + 1] as i32) << 1 | ((buffer[l] as i32) >> 7 & 0x01)) as u8;
            if exp > 3 && exp != 0 {
                buffer[l + 1] = buffer[l + 1].wrapping_sub(1);
            } else if exp <= 3 && exp != 0 {
                // must zero out the mantissa
                // we want manitssa 0 & exponent 1
                buffer[l] = 0x80;
                buffer[l + 1] &= 0x80;
                buffer[l + 3] = 0;
                buffer[l + 2] = 0;
            }

            temp = buffer[l];
            buffer[l] = buffer[l + 1];
            buffer[l + 1] = temp;
        } else {
            // TO_VAX
            exp = ((buffer[l] as i32) << 1 | ((buffer[l + 1] as i32) >> 7 & 0x01)) as u8;
            if exp < 253 && exp != 0 {
                buffer[l] = buffer[l].wrapping_add(1);
            } else if exp >= 253 {
                // must also max out the exp & mantissa
                // we want manitssa all 1 & exponent 255
                buffer[l] |= 0x7F;
                buffer[l + 1] = 0xFF;
                buffer[l + 3] = 0xFF;
                buffer[l + 2] = 0xFF;
            }

            temp = buffer[l];
            buffer[l] = buffer[l + 1];
            buffer[l + 1] = temp;
        }

        if ((direction != 0 && exp > 3) || (direction == 0 && exp < 253)) && exp != 0 {
            temp = buffer[l + 2];
            buffer[l + 2] = buffer[l + 3];
            buffer[l + 3] = temp;
        }
        lcv += 4;
    }
}

/// C `convertHeader` (`mrcx.c:460`).  `header` is the first `56 * 4` bytes
/// of the C `struct MRCheader`, with `labels` and `symops` beside it; each
/// `(b3dUByte *)&header->field` is the field's byte offset.  The source's
/// closing `header->fp = outfp` stores into a field nothing reads again.
#[allow(clippy::too_many_arguments)]
fn convert_header(
    float_func: ConvertFunc,
    nextra: i32,
    header: &mut [u8; 56 * 4],
    labels: &mut [[u8; MRC_LABEL_SIZE + 1]; MRC_NLABELS],
    symops: &mut Option<Vec<u8>>,
    infp: &mut ImodFile,
    _outfp: &mut ImodFile,
    direction: i32,
    cmap: &[u8; 4],
) {
    let mut lcv: usize;

    convert_longs(infp, &mut header[0..4], 1, direction); // nx
    convert_longs(infp, &mut header[4..8], 1, direction); // ny
    convert_longs(infp, &mut header[8..12], 1, direction); // nz

    convert_longs(infp, &mut header[12..16], 1, direction); // mode

    convert_longs(infp, &mut header[16..20], 1, direction); // nxstart
    convert_longs(infp, &mut header[20..24], 1, direction); // nystart
    convert_longs(infp, &mut header[24..28], 1, direction); // nzstart
    convert_longs(infp, &mut header[28..32], 1, direction); // mx
    convert_longs(infp, &mut header[32..36], 1, direction); // my
    convert_longs(infp, &mut header[36..40], 1, direction); // mz
    float_func(infp, &mut header[40..44], 1, direction); // xlen
    float_func(infp, &mut header[44..48], 1, direction); // ylen
    float_func(infp, &mut header[48..52], 1, direction); // zlen
    float_func(infp, &mut header[52..56], 1, direction); // alpha
    float_func(infp, &mut header[56..60], 1, direction); // beta
    float_func(infp, &mut header[60..64], 1, direction); // gamma
    convert_longs(infp, &mut header[64..68], 1, direction); // mapc
    convert_longs(infp, &mut header[68..72], 1, direction); // mapr
    convert_longs(infp, &mut header[72..76], 1, direction); // maps
    float_func(infp, &mut header[76..80], 1, direction); // amin
    float_func(infp, &mut header[80..84], 1, direction); // amax
    float_func(infp, &mut header[84..88], 1, direction); // amean
    convert_longs(infp, &mut header[88..92], 1, direction); // ispg
    //     convertLongs(infp, (b3dUByte *)header->extra, 16, direction);
    convert_longs(infp, &mut header[92..96], 1, direction); // next
    convert_shorts(infp, &mut header[96..98], 1, direction); // creatid
    convert_bytes(infp, &mut header[98..128], 30, direction); // blank
    convert_shorts(infp, &mut header[128..136], 4, direction); // nint
    float_func(infp, &mut header[136..160], 6, direction); // min2

    convert_shorts(infp, &mut header[160..162], 1, direction); // idtype
    convert_shorts(infp, &mut header[162..164], 1, direction); // lens

    convert_shorts(infp, &mut header[164..166], 1, direction); // nd1
    convert_shorts(infp, &mut header[166..168], 1, direction); // nd2
    convert_shorts(infp, &mut header[168..170], 1, direction); // vd1
    convert_shorts(infp, &mut header[170..172], 1, direction); // vd2

    float_func(infp, &mut header[172..196], 6, direction); // tiltangles

    if cmap[0] != b'M' || cmap[1] != b'A' || cmap[2] != b'P' {
        // Old-style header, swap wavelength then origin data
        convert_shorts(infp, &mut header[196..208], 6, direction); // xorg
        float_func(infp, &mut header[208..220], 3, direction); // cmap[0]
        let _ = ImodFile::Stdout.write_all(b"Preserving old-style MRC header.\n");
    } else {
        // new style, swap floats and retain cmap; flip stamp
        float_func(infp, &mut header[196..200], 1, direction); // xorg
        float_func(infp, &mut header[200..204], 1, direction); // yorg
        float_func(infp, &mut header[204..208], 1, direction); // zorg
        convert_bytes(infp, &mut header[208..216], 8, direction); // cmap[0]
        float_func(infp, &mut header[216..220], 1, direction); // rms
        // stamp[0]
        header[212] = if header[212] == 17 { 68 } else { 17 };
    }
    convert_longs(infp, &mut header[220..224], 1, direction); // nlabl

    lcv = 0;
    while lcv < MRC_NLABELS {
        convert_bytes(
            infp,
            &mut labels[lcv][..MRC_LABEL_SIZE],
            MRC_LABEL_SIZE as i32,
            direction,
        );
        labels[lcv][MRC_LABEL_SIZE] = b'\0';
        lcv += 1;
    }

    // THIS WILL NOT WORK WITH DATA CONSISTING OF INTEGERS AND REALS
    if nextra > 0 {
        let buffer = symops.insert(vec![0u8; nextra as usize]);
        convert_shorts(infp, &mut buffer[..], nextra / 2, direction);
        if nextra % 2 != 0 {
            convert_bytes(infp, &mut buffer[(nextra - 1) as usize..], 1, direction);
        }
    }
}

/// C `convertLongs` (`mrcx.c:552`): converts a VAX longword to a "unix"
/// longword -- in practice, reverses each 4-byte word.
fn convert_longs(fp: &mut ImodFile, buffer: &mut [u8], no_longs: i32, _direction: i32) {
    let no_bytes: u32; // no bytes to read

    // Read in data
    no_bytes = (no_longs as u32).wrapping_mul(4);

    if b3d_fread(buffer, 1, no_bytes as usize, fp) != no_bytes as usize {
        let _ = ImodFile::Stderr.write_all(b"mrcx: Read Failed While Reading Long Quantities\n");
    }

    // DNM 3/16/01: discovered that the code here did nothing on a
    // little-endian machine!  Switch to call library routine
    // `mrc_swap_longs((int *)buffer, noLongs)`.
    let mut longs: Vec<i32> = buffer[..no_bytes as usize]
        .chunks_exact(4)
        .map(|b| i32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    mrc_swap_longs(&mut longs, no_longs as usize);
    for (dst, value) in buffer.chunks_exact_mut(4).zip(longs.iter()) {
        dst.copy_from_slice(&value.to_ne_bytes());
    }
}

/// C `convertShorts` (`mrcx.c:580`): converts a VAX shortword to a "unix"
/// shortword -- in practice, reverses each 2-byte word.
fn convert_shorts(fp: &mut ImodFile, buffer: &mut [u8], no_shorts: i32, _direction: i32) {
    let no_bytes: u32; // no bytes to read from the input file

    // Read in data
    no_bytes = (no_shorts as u32).wrapping_mul(2);

    if b3d_fread(buffer, 1, no_bytes as usize, fp) != no_bytes as usize {
        let _ = ImodFile::Stderr.write_all(b"mrcx: Read Failed While Reading Short Quantities\n");
    }

    // DNM 3/16/01: `mrc_swap_shorts((b3dInt16 *)buffer, noShorts)`.
    let mut shorts: Vec<i16> = buffer[..no_bytes as usize]
        .chunks_exact(2)
        .map(|b| i16::from_ne_bytes([b[0], b[1]]))
        .collect();
    mrc_swap_shorts(&mut shorts, no_shorts as usize);
    for (dst, value) in buffer.chunks_exact_mut(2).zip(shorts.iter()) {
        dst.copy_from_slice(&value.to_ne_bytes());
    }
}
