//! Translation of `IMOD/pysrc/archiveorig`: makes a compressed difference
//! file for archiving an original stack, or restores the original from it.
//!
//! The script's top level is [`archiveorig`]; its two functions are
//! [`cleanup`] and [`spool_gzip`].  `subimage` is our own program and runs in
//! process through `imodpy::run_cmd`.
//!
//! Python's `gzip.open(name, 'wb')` writes a gzip member whose deflate data
//! comes from zlib at level 9 (`zlib.compressobj(9, DEFLATED, -15, 8, 0)`).
//! That stream is reproduced byte for byte by calling the system zlib
//! (`libz`, a foreign library like libtiff) -- the stream is the same for any
//! way the input is split into `compress` calls and was checked identical
//! between zlib 1.2.11 and 1.3.1.  The header is written as
//! `GzipFile._write_gzip_header` (Python 3.12) writes it: method 8, `FNAME`
//! with the base name minus `.gz`, the current time, XFL 2, OS 255.  Reading
//! (`gzip.open(name, 'rb')`, every member of the file) uses `flate2`'s
//! multi-member decoder, whose output does not depend on the backend.

use super::imodpy::{
    add_imod_bin_ignore_sighup, cleanup_files, exit_from_imod_error, fmtstr, glob_glob,
    make_backup_file, os_path_splitext, pass_on_key_interrupt, print_pid, prnstr, run_cmd,
};
use super::pip::{
    exit_error, pip_exit_on_error, pip_get_boolean, pip_get_integer, pip_get_non_option_arg,
    pip_parse_input, pip_print_help,
};
use std::ffi::{OsString, c_char, c_int, c_uint, c_ulong, c_void};
use std::io::{Read as _, Write as _};
use std::path::Path;

/// zlib's `z_stream` (`zlib.h`), for the deflate calls below.
#[repr(C)]
struct ZStream {
    next_in: *const u8,
    avail_in: c_uint,
    total_in: c_ulong,
    next_out: *mut u8,
    avail_out: c_uint,
    total_out: c_ulong,
    msg: *const c_char,
    state: *mut c_void,
    zalloc: *const c_void,
    zfree: *const c_void,
    opaque: *mut c_void,
    data_type: c_int,
    adler: c_ulong,
    reserved: c_ulong,
}

#[link(name = "z")]
unsafe extern "C" {
    fn zlibVersion() -> *const c_char;
    fn deflateInit2_(
        strm: *mut ZStream,
        level: c_int,
        method: c_int,
        window_bits: c_int,
        mem_level: c_int,
        strategy: c_int,
        version: *const c_char,
        stream_size: c_int,
    ) -> c_int;
    fn deflate(strm: *mut ZStream, flush: c_int) -> c_int;
    fn deflateEnd(strm: *mut ZStream) -> c_int;
    fn crc32(crc: c_ulong, buf: *const u8, len: c_uint) -> c_ulong;
}

/// `def cleanup(which)` (`archiveorig:17`): cleanup after error or
/// interrupt: call with 1 for restore or 2 for archive.
fn cleanup(which: i32, xrayname: &str, origname: &str) {
    let _ = (|| -> std::io::Result<()> {
        if !xrayname.is_empty() {
            std::fs::remove_file(xrayname)?;
        }
        if which == 1 && !origname.is_empty() {
            std::fs::remove_file(origname)?;
        }
        Ok(())
    })();
}

/// `def spoolGzip(which)` (`archiveorig:26`): 1 uncompresses `compname` to
/// `xrayname`, 2 compresses `xrayname` to `compname`.
fn spool_gzip(which: i32, compname: &str, xrayname: &str, origname: &str) {
    let chunksize = 10000000_usize;
    let errstr;
    let inname;
    if which == 1 {
        errstr = format!("Uncompressing {compname}");
        inname = format!("{compname}.old");
    } else {
        errstr = format!("Compressing {xrayname}");
        inname = xrayname.to_owned();
    }
    let spooled: Result<(), ()> = (|| {
        if which == 1 {
            let infile = std::fs::File::open(compname).map_err(|_| ())?;
            let mut infile = flate2::read::MultiGzDecoder::new(std::io::BufReader::new(infile));
            let mut outfile = std::fs::File::create(xrayname).map_err(|_| ())?;
            prnstr(&format!("{errstr} ..."), "\n", false);
            let mut data = vec![0_u8; chunksize];
            loop {
                // `infile.read(chunksize)`: a buffered read fills the request
                // unless the stream ends.
                let mut lendata = 0;
                while lendata < chunksize {
                    let got = infile.read(&mut data[lendata..]).map_err(|_| ())?;
                    if got == 0 {
                        break;
                    }
                    lendata += got;
                }
                if lendata == 0 {
                    break;
                }
                outfile.write_all(&data[..lendata]).map_err(|_| ())?;
                if lendata < chunksize {
                    break;
                }
            }
            return Ok(());
        }

        // `gzip.open(compname, 'wb')` writes the header when it opens
        let mut outfile = std::fs::File::create(compname).map_err(|_| ())?;
        let base = Path::new(compname)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        // RFC 1952 requires the FNAME field to be Latin-1; a name that is not
        // representable that way is left out.
        let mut fname: Vec<u8> = Vec::new();
        for ch in base.chars() {
            if (ch as u32) < 256 {
                fname.push(ch as u32 as u8);
            } else {
                fname.clear();
                break;
            }
        }
        if fname.ends_with(b".gz") {
            fname.truncate(fname.len() - 3);
        }
        let mtime = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0) as u32;
        let mut header = vec![0o37_u8, 0o213, 0o10, if fname.is_empty() { 0 } else { 8 }];
        header.extend_from_slice(&mtime.to_le_bytes());
        header.extend_from_slice(&[2, 0o377]);
        if !fname.is_empty() {
            header.extend_from_slice(&fname);
            header.push(0);
        }
        outfile.write_all(&header).map_err(|_| ())?;
        let mut infile = std::fs::File::open(xrayname).map_err(|_| ())?;
        prnstr(&format!("{errstr} ..."), "\n", false);

        // SAFETY: a zeroed `z_stream` with null allocators is what
        // `deflateInit2` expects; the buffers outlive each call.
        let mut stream: ZStream = unsafe { std::mem::zeroed() };
        let status = unsafe {
            deflateInit2_(
                &mut stream,
                9,
                8,
                -15,
                8,
                0,
                zlibVersion(),
                std::mem::size_of::<ZStream>() as c_int,
            )
        };
        if status != 0 {
            return Err(());
        }
        let mut crc: c_ulong = 0;
        let mut size: u32 = 0;
        let mut out = vec![0_u8; 1 << 17];
        let mut run = |stream: &mut ZStream,
                       input: &[u8],
                       flush: c_int,
                       outfile: &mut std::fs::File|
         -> Result<(), ()> {
            stream.next_in = input.as_ptr();
            stream.avail_in = input.len() as c_uint;
            loop {
                stream.next_out = out.as_mut_ptr();
                stream.avail_out = out.len() as c_uint;
                let status = unsafe { deflate(stream, flush) };
                if status < 0 && status != -5 {
                    return Err(());
                }
                let produced = out.len() - stream.avail_out as usize;
                outfile.write_all(&out[..produced]).map_err(|_| ())?;
                if flush == 4 {
                    if status == 1 {
                        return Ok(());
                    }
                } else if stream.avail_in == 0 && stream.avail_out != 0 {
                    return Ok(());
                }
            }
        };
        let mut data = vec![0_u8; chunksize];
        let result = (|| -> Result<(), ()> {
            loop {
                let mut lendata = 0;
                while lendata < chunksize {
                    let got = infile.read(&mut data[lendata..]).map_err(|_| ())?;
                    if got == 0 {
                        break;
                    }
                    lendata += got;
                }
                if lendata == 0 {
                    break;
                }
                crc = unsafe { crc32(crc, data.as_ptr(), lendata as c_uint) };
                size = size.wrapping_add(lendata as u32);
                run(&mut stream, &data[..lendata], 0, &mut outfile)?;
                if lendata < chunksize {
                    break;
                }
            }
            // `outfile.close()`: finish the stream, then the CRC and size
            run(&mut stream, &[], 4, &mut outfile)?;
            outfile
                .write_all(&(crc as u32).to_le_bytes())
                .map_err(|_| ())?;
            outfile.write_all(&size.to_le_bytes()).map_err(|_| ())?;
            Ok(())
        })();
        unsafe { deflateEnd(&mut stream) };
        result
    })();
    if spooled.is_err() {
        cleanup(which, xrayname, origname);
        exit_error(&errstr);
    }

    let renamed: std::io::Result<()> = if which == 1 {
        make_backup_file(&inname);
        std::fs::rename(compname, &inname)
    } else {
        std::fs::remove_file(&inname)
    };
    if renamed.is_err() {
        if which == 1 {
            prnstr(
                &format!("WARNING: archiveorig - Failed to rename {compname} to {inname}"),
                "\n",
                false,
            );
        } else {
            prnstr(
                &format!("WARNING: archiveorig - Failed to remove {inname}"),
                "\n",
                false,
            );
        }
    }
}

/// The script's top level (`archiveorig:73-220`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn archiveorig(arguments: &[OsString]) -> i32 {
    let progname = "archiveorig";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let done = |status: i32| {
        let _ = std::io::stdout().flush();
        status
    };

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return done(1);
    }
    // SAFETY: the script sets its own environment before running anything.
    unsafe { std::env::set_var("PIP_PRINT_ENTRIES", "0") };

    let options: Vec<String> = [
        "r::B:Restore setname_orig.ext from setname.st and setname_xray.ext.gz",
        "d::B:Delete setname_orig.ext after computing difference",
        "n::I:Number of levels of archives to restore",
        ":PID:B:Print process ID",
        "h::B:Print usage and exit",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    pip_exit_on_error(0, &prefix);
    let (_num_opts, num_non_opts) = pip_parse_input(&argv, &options).unwrap_or_else(|_| {
        super::pip::python_uncaught("TypeError: cannot unpack non-iterable NoneType object")
    });

    let do_pid = pip_get_boolean("PID", 0).unwrap_or(0);
    print_pid(do_pid != 0);

    if num_non_opts < 1 || pip_get_boolean("h", 0).unwrap_or(0) != 0 {
        pip_print_help(progname, 0, 1, 0);
        return done(0);
    }

    let restore = pip_get_boolean("r", 0).unwrap_or(0);
    let delete = pip_get_boolean("d", 0).unwrap_or(0);
    let mut num_levels = pip_get_integer("n", 1).unwrap_or(1);
    pass_on_key_interrupt(true);

    let mut stack = pip_get_non_option_arg(0).unwrap_or_default();
    if !Path::new(&stack).exists() {
        exit_error(&format!("File {stack} does not exist"));
    }

    let (setname, stack_ext) = os_path_splitext(&stack);
    let mut origname = format!("{setname}_orig{stack_ext}");
    let xrayname = format!("{setname}_xray{stack_ext}");
    let mut compname = format!("{xrayname}.gz");

    // For Restores
    if restore != 0 {
        if !Path::new(&compname).exists() {
            exit_error(&format!("File {compname} does not exist"));
        }

        // Get the collection of existing files (Python 2.3 does not have sort(reverse))
        // Fixed in translation (BUGS.md, `archiveorig`): the source globs
        // `compname + '[0-9][0-9]'` without the `.` of the `_xray.ext.gz.N`
        // names it restores from below, so no numbered file is ever found
        // (and one named `...gzN` would crash the `int()` conversion).
        let mut comp_list = vec![compname.clone()];
        for suffix in [".[0-9][0-9]", ".[0-9]"] {
            let mut comp_suff = glob_glob(&format!("{compname}{suffix}"));
            comp_suff.sort();
            comp_suff.reverse();
            comp_list.extend(comp_suff);
        }

        num_levels = num_levels.max(1);
        if num_levels as usize > comp_list.len() {
            exit_error(&format!(
                "There are not enough xray{stack_ext}.gz files to restore {num_levels} levels"
            ));
        }

        let mut next_level: i64 = 1;
        if num_levels > 1 {
            let mut num_list: Vec<i64> = Vec::new();
            for name in &comp_list[1..] {
                let (_, ext) = os_path_splitext(name);
                num_list.push(ext[1..].parse::<i64>().unwrap_or_else(|_| {
                    super::pip::python_uncaught(&format!(
                        "ValueError: invalid literal for int() with base 10: '{}'",
                        &ext[1..]
                    ))
                }));
            }
            next_level = num_list[0] + 1;
            for ind in 1..(num_levels - 1) as usize {
                if num_list[ind - 1] - num_list[ind] != 1 {
                    exit_error(&format!(
                        "There is not a complete sequence of _xray{stack_ext}.gz.N files; N = {} is missing",
                        num_list[ind] - 1
                    ));
                }
            }
        }

        // Loop on the levels, create an _origN for each _xrayN
        let real_stack = stack.clone();
        for ind in 0..num_levels {
            if ind != 0 {
                compname = format!("{setname}_xray{stack_ext}.gz.{next_level}");
            }
            origname = format!("{setname}_orig{stack_ext}");
            if ind < num_levels - 1 {
                origname += &format!(".{next_level}");
            }

            prnstr(&format!("Restoring {origname} ..."), "\n", false);
            spool_gzip(1, &compname, &xrayname, &origname);
            if run_cmd(
                &fmtstr(
                    "subimage \"{}\" \"{}\" \"{}\"",
                    &[stack.clone(), xrayname.clone(), origname.clone()],
                ),
                None,
                None,
                None,
                &[],
            )
            .is_err()
            {
                cleanup(1, &xrayname, &origname);
                exit_from_imod_error(progname);
            }

            // Clean up the difference file, and the previous orig file after the first round,
            // and prepare for the next round where this orig file is stack
            cleanup_files(&[xrayname.clone()]);
            if ind != 0 && real_stack != stack {
                cleanup_files(&[stack.clone()]);
            }
            stack = origname.clone();
            next_level -= 1;
        }

        prnstr("DONE", "\n", false);
        return done(0);
    }

    // Archiving
    if !Path::new(&origname).exists() {
        exit_error(&format!("File {origname} does not exist"));
    }

    if Path::new(&xrayname).exists() && std::fs::remove_file(&xrayname).is_err() {
        prnstr(
            &format!("WARNING: {progname} - Could not remove existing {xrayname}"),
            "\n",
            false,
        );
    }

    make_backup_file(&compname);
    prnstr("Getting difference image ...", "\n", false);
    if run_cmd(
        &fmtstr(
            "subimage -mode 2 \"{}\" \"{}\" \"{}\"",
            &[stack.clone(), origname.clone(), xrayname.clone()],
        ),
        None,
        None,
        None,
        &[],
    )
    .is_err()
    {
        cleanup(2, &xrayname, &origname);
        exit_from_imod_error(progname);
    }

    spool_gzip(2, &compname, &xrayname, &origname);

    let st_size = std::fs::metadata(&compname)
        .map(|meta| meta.len())
        .unwrap_or(0);
    prnstr(
        &format!(
            "Compressed difference file  {compname}  has size {:.2} MB",
            st_size as f64 / (1024. * 1024.)
        ),
        "\n",
        false,
    );
    if delete != 0 {
        prnstr(&format!("Deleting {origname} ..."), "\n", false);
        if std::fs::remove_file(&origname).is_err() {
            prnstr(
                &format!("WARNING: {progname} - Could not remove {origname}"),
                "\n",
                false,
            );
        }
    } else {
        prnstr(&format!("It is now safe to delete {origname}"), "\n", false);
    }
    prnstr(
        &format!("To restore it, enter:   {progname} -r {stack}"),
        "\n",
        false,
    );
    done(0)
}
