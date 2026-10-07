//! Translation of `IMOD/flib/image/vmstocsh.f`.
//!
//! VMSTOCSH takes a VMS-style command file from standard input and converts
//! it to text suitable for piping to a C shell, on standard output.  The
//! Fortran main program maps to [`vmstocsh`]; the source has no other program
//! units.
//!
//! The source is compiled with `-fbackslash`, so its `'\\'` literals are one
//! backslash and `'\\$'` is the two characters `\$`.  The text is handled as
//! bytes, as a Fortran `character` variable holds it.
//!
//! This converter is kept so existing users (`submfg -s`, command lines that
//! pipe to `tcsh`) keep working; the crate itself runs command files in
//! process through [`crate::imod::comrun`] instead of `vmstocsh | tcsh`.

use crate::imod::libcfshr::b3dutil::{exit, program_args_os};
use std::io::{Read as _, Write as _};

/// `character*10240 linein, linecom, logfile` (`vmstocsh.f:14`).
const LINE_LEN: usize = 10240;

/// Original program `vmstocsh` (`vmstocsh.f:1-121`).
///
/// A `character*10240` variable is a byte vector of exactly [`LINE_LEN`]
/// bytes, blank-padded; `len_trim` is the length without trailing blanks.
pub fn vmstocsh() {
    // Fortran assignment to a character*10240 variable: truncate or blank-pad
    let assign = |text: &[u8]| -> Vec<u8> {
        let mut out = vec![b' '; LINE_LEN];
        let n = text.len().min(LINE_LEN);
        out[..n].copy_from_slice(&text[..n]);
        out
    };
    let len_trim = |text: &[u8]| -> usize {
        let mut n = text.len();
        while n > 0 && text[n - 1] == b' ' {
            n -= 1;
        }
        n
    };

    // Standard input, read directly from descriptor 0 so that nothing a
    // previous in-process command left in a shared buffer is seen.
    let mut input: Vec<u8> = Vec::new();
    #[cfg(unix)]
    {
        use std::os::fd::FromRawFd as _;
        let mut stdin = std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(0) });
        let _ = stdin.read_to_end(&mut input);
    }
    // The handle behind CRT descriptor 0 (`_get_osfhandle`), borrowed.
    #[cfg(windows)]
    {
        use std::os::windows::io::FromRawHandle as _;
        let handle = unsafe { libc::get_osfhandle(0) };
        if handle != -1 {
            let mut stdin = std::mem::ManuallyDrop::new(unsafe {
                std::fs::File::from_raw_handle(handle as std::os::windows::io::RawHandle)
            });
            let _ = stdin.read_to_end(&mut input);
        }
    }
    // Records of the formatted sequential input: each ends at a newline; a
    // last line without one is still a record.
    let mut records: Vec<&[u8]> = input.split(|&b| b == b'\n').collect();
    if input.is_empty() || input.ends_with(b"\n") {
        records.pop();
    }
    let mut records = records.into_iter();

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    // write(6,101) with format (a): the text and a record end
    let mut write = |text: &[u8]| {
        let _ = out.write_all(text);
        let _ = out.write_all(b"\n");
    };

    let herestring: &[u8] = b"HERESTRING";
    let mut logfile = assign(b" ");
    let mut lenlog = 1usize;
    let indarrow = 2usize;
    write(b"nohup");
    let arguments = program_args_os();
    if arguments.len() > 1 {
        // call getarg(1,logfile)
        use crate::imod::libcfshr::b3dutil::OsStrExt as _;
        logfile = assign(arguments[1].as_bytes());
        lenlog = len_trim(&logfile);
        let name = logfile[..lenlog].to_vec();
        let mut line = b"if (-e \"".to_vec();
        line.extend_from_slice(&name);
        line.extend_from_slice(b"\") \\mv -f \"");
        line.extend_from_slice(&name);
        line.extend_from_slice(b"\" \"");
        line.extend_from_slice(&name);
        line.extend_from_slice(b"~\"");
        write(&line);
        let mut text = b"  > \"".to_vec();
        text.extend_from_slice(&name);
        text.push(b'"');
        logfile = assign(&text);
        lenlog = len_trim(&logfile);
    }

    write(b"if ($?IMOD_DIR) then");
    write(b"    setenv PATH \"$IMOD_DIR/bin:$PATH\"");
    write(b"endif");
    write(b"if ($?IMOD_QTLIBDIR && $?LD_LIBRARY_PATH) then");
    write(b"    setenv LD_LIBRARY_PATH \"${IMOD_QTLIBDIR}:$LD_LIBRARY_PATH\"");
    write(b"endif");
    write(b"setenv PIP_PRINT_ENTRIES 1");
    write(b"echo2 Shell PID: $$");

    let mut iffirst: i32 = -1;
    // `linecom` is never read before a command line assigns it: the only
    // earlier test is guarded by `iffirst.eq.0`.
    let mut linecom = assign(b" ");
    let mut lencom = 2usize;
    let mut reading = true;
    while reading {
        reading = false;
        let mut linein = assign(b"$ ");
        if let Some(record) = records.next() {
            linein = assign(record);
            reading = true;
        }
        // 10
        let mut lenin = len_trim(&linein);
        //
        // For Cygwin/windows, if the line is not properly stripped of
        // Return, replace it now
        //
        if lenin > 0 && linein[lenin - 1] == 13 {
            linein[lenin - 1] = b' ';
            lenin -= 1;
        }
        if linein[0] != b'#' && &linein[..2] != b"$!" {
            // A dumped `\\` line followed by a blank one leaves `lencom` 0, and
            // the source then reads `linecom(0:0)`, before the variable; that
            // out-of-bounds byte is taken as not a backslash here.
            if iffirst == 0 && lencom > 0 && linecom[lencom - 1] == b'\\' {
                let before = lencom.saturating_sub(1).max(1);
                if lencom > 1 && linecom[before - 1] == b'\\' {
                    //
                    // if last line needs to be continued in the output
                    // dump the last line, replace with current line
                    //
                    write(&linecom[1..lencom - 1]);
                    let mut text = b" ".to_vec();
                    text.extend_from_slice(&linein);
                    linecom = assign(&text);
                } else {
                    //
                    // otherwise, a continuation line of a command line: add it on
                    //
                    let mut text = linecom[..lencom - 1].to_vec();
                    text.push(b' ');
                    text.extend_from_slice(&linein[..lenin]);
                    linecom = assign(&text);
                }
                lencom = len_trim(&linecom);
                //
            } else if linein[0] == b'$' || linein[0] == b'%' {
                //
                // a new command line: if the last line was not an entry line
                // to a previous command, it was a command itself and needs to
                // be passed through now; if it was an entry line, put out the
                // herestring to terminate entries
                //
                if iffirst >= 0 {
                    if iffirst == 0 {
                        let mut text = linecom.get(1..lencom).unwrap_or(&[]).to_vec();
                        text.extend_from_slice(&logfile[..lenlog]);
                        write(&text);
                        logfile[indarrow - 1] = b'>';
                    } else {
                        write(herestring);
                    }
                }
                linecom = linein.clone();
                lencom = lenin;
                iffirst = 0;
            } else {
                //
                // not a command line: if it is the first entry line, dump the
                // command line with the << herestring on the end
                // in any case, pass the line through
                //
                if iffirst >= 0 {
                    if iffirst == 0 {
                        let mut text = linecom.get(1..lencom).unwrap_or(&[]).to_vec();
                        text.extend_from_slice(b" << ");
                        text.extend_from_slice(herestring);
                        text.extend_from_slice(&logfile[..lenlog]);
                        write(&text);
                        logfile[indarrow - 1] = b'>';
                        iffirst = 1;
                    }
                    //
                    // Remove escape of leading $ so variables can be passed in
                    //
                    if &linein[..2] == b"\\$" {
                        write(&linein[1..lenin.max(1)]);
                    } else {
                        write(&linein[..lenin]);
                    }
                }
            }
        }
    }
    let mut text = b"echo SUCCESSFULLY COMPLETED".to_vec();
    text.extend_from_slice(&logfile[..lenlog]);
    write(&text);
    drop(write);
    let _ = out.flush();
    drop(out);
    exit(0);
}
