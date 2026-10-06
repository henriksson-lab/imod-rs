//! Translation of `IMOD/pysrc/tomosnapshot`: collects small files and other
//! information about one form of eTomo processing into a gzipped tar file.
//!
//! The script's module globals (the file lists, `tempdir`, `copylogs`,
//! `tarlist`, `thumbnails`, `headerCopies`, `privSed`, `doThumbnails`,
//! `stdTypeExts`, `stackExts`) live in [`Snapshot`]; its functions are its
//! methods, and its top level is [`tomosnapshot`].
//!
//! Commands: `newstack`, `header`, `imodinfo` are our own programs and
//! `copyheader` a Python-script translation; all go through
//! `imodpy::run_cmd`, which runs them in process or as a child of our own
//! binary.  `getmrcsize` is the direct `imodpy::get_mrc_size`.  `uname -a`,
//! `ls -lrt` and `ls -ld .` are system tools and stay processes (through
//! `run_cmd`'s `sh -c`).  The Windows-only `cygcheck` runs, `dir /OD` and
//! the `sys.getwindowsversion()` report are not translated (the
//! `win32`/`cygwin` platform tests are false here).
//!
//! **The tar file.**  Python's `tarfile.open(outFile, 'w|gz')` is written in
//! process, as the Python 3.12 `tarfile` module writes it: the default PAX
//! format with a `././@PaxHeader` block carrying the float `mtime` before
//! every member, `ustar` headers, directories recursed in sorted order, two
//! zero blocks and padding to a 10240-byte record; the gzip stream carries
//! `FNAME` and the current time.  The deflate encoder is `flate2`'s at level
//! 9, not zlib's, so the compressed bytes differ from native while the tar
//! stream inside is laid out the same.  (The archive could never be byte
//! identical: it holds file times, the temporary directory's name and the
//! time of the run.)
//!
//! **`uname.out`.**  It records `"Python version: " + sys.version`; there is
//! no Python interpreter here, so that line names the translation instead.
//!
//! **`tempfile.mkdtemp`.**  Made here with the same name shape (prefix plus
//! eight characters from `[a-z0-9_]`), mode 0700, and returned as an
//! absolute path, as Python 3.12 does.
//!
//! Upstream bugs fixed in translation (each in BUGS.md): the `NWUSERNAME`
//! privacy substitution is never added because the value is stored in a
//! misspelled variable; a PEET `.prm` without a `reference` entry, and a
//! join `.info` file whose first word is not an integer, crash the script
//! with a traceback; `os.chdir` into a missing `naddir.<set>` crashes it.

use super::imodpy::{
    ImodpyError, allowed_raw_stack_extensions, dataset_filename, exit_from_imod_error,
    find_root_axis_and_extensions, get_mrc_size, glob_glob, os_path_splitext, prnstr, py_int,
    py_round, py_str_float, read_text_file, run_cmd, set_root_and_extension,
    standard_type_extensions, write_text_file,
};
use super::pip::{
    exit_error, pip_exit_on_error, pip_get_boolean, pip_get_in_out_file, pip_get_string,
    pip_parse_input, pip_print_help,
};
use super::pysed::{PysedSrc, pysed};
use std::ffi::{CString, OsString};
use std::io::Write as _;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
use std::path::Path;

const PROGNAME: &str = "tomosnapshot";

// Constants for the types
const TOMO: usize = 0;
const JOIN: usize = 1;
const PEET: usize = 2;
const NAD: usize = 3;
const PARALLEL: usize = 4;
const SERIAL: usize = 5;
const BATCH: usize = 6;

/// `os.path.basename`
fn basename(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_owned()
}

/// `os.path.dirname`
fn dirname(path: &str) -> String {
    match path.rfind('/') {
        Some(index) => {
            let head = &path[..index + 1];
            if head.chars().all(|c| c == '/') {
                head.to_owned()
            } else {
                head.trim_end_matches('/').to_owned()
            }
        }
        None => String::new(),
    }
}

/// `os.path.join` of two components
fn join(a: &str, b: &str) -> String {
    if b.starts_with('/') {
        return b.to_owned();
    }
    let mut joined = a.to_owned();
    if !joined.is_empty() && !joined.ends_with('/') {
        joined.push('/');
    }
    joined.push_str(b);
    joined
}

/// The module globals of the script.
struct Snapshot {
    tempdir: String,
    copylogs: Vec<String>,
    tarlist: Vec<String>,
    thumbnails: Vec<String>,
    header_copies: Vec<String>,
    priv_sed: Vec<String>,
    do_thumbnails: bool,
    std_type_exts: Vec<String>,
    stack_exts: Vec<String>,
}

/// Matches `getKeyValue` (`tomosnapshot:91`): the value of `key` in an etomo
/// or prm file, `None` when there is none.
fn get_key_value(etomo_lines: &[String], key: &str) -> Option<String> {
    for i in 0..etomo_lines.len() {
        let mut line = etomo_lines[i].clone();
        if line.starts_with(key) {
            if line.contains('{') && !line.contains('}') {
                for j in i + 1..etomo_lines.len() {
                    line += &format!(" {}", etomo_lines[j]);
                    if etomo_lines[j].contains('}') {
                        break;
                    }
                }
            }
            let lsplit: Vec<&str> = line.split('=').collect();
            if lsplit.len() > 1 {
                return Some(lsplit[1].trim().trim_matches(['{', '}']).trim().to_owned());
            }
        }
    }
    None
}

/// Matches `prmUniqueEntries` (`tomosnapshot:107`).
fn prm_unique_entries(prm_lines: &[String], key: &str) -> Option<Vec<String>> {
    let value = get_key_value(prm_lines, key)?;
    if !value.contains('\'') {
        return None;
    }
    let mut entries: Vec<String> = Vec::new();
    for entry in value.split(',') {
        let entry = entry.trim().trim_matches('\'').to_owned();
        if !entries.contains(&entry) {
            entries.push(entry);
        }
    }
    Some(entries)
}

/// `shutil.copystat`: times (to the nanosecond), then permission bits.
fn copystat(from_file: &str, to_file: &str) -> std::io::Result<()> {
    let meta = std::fs::metadata(from_file)?;
    let times = std::fs::FileTimes::new()
        .set_accessed(meta.accessed()?)
        .set_modified(meta.modified()?);
    std::fs::OpenOptions::new()
        .write(true)
        .open(to_file)?
        .set_times(times)?;
    std::fs::set_permissions(
        to_file,
        std::fs::Permissions::from_mode(meta.mode() & 0o7777),
    )
}

/// `warning` (`tomosnapshot:216`)
fn warning(message: &str) {
    prnstr(&format!("WARNING: {PROGNAME} - {message}"), "\n", false);
}

/// Matches `stripAndWrite` (`tomosnapshot:221`).
fn strip_and_write(outfile: &str, lines: &mut [String]) {
    for line in lines.iter_mut() {
        *line = line.trim_end_matches(['\r', '\n']).to_owned();
    }
    let _ = write_text_file(outfile, lines, false);
}

/// Matches `removeWarn` (`tomosnapshot:228`).
fn remove_warn(rfile: &str) {
    if std::fs::remove_file(rfile).is_err() {
        warning(&format!("Failed to remove {rfile}"));
    }
}

impl Snapshot {
    /// Matches `privacyCopy` (`tomosnapshot:121`).
    ///
    /// `Err` is the `SystemExit` that `pysed`'s `psReportErr` raises when the
    /// file cannot be read (it prints the message first): the source's
    /// `except Exception` does not catch it, so it leaves this function and
    /// the copystat is not done; each caller then does what its own
    /// `except` does with a `SystemExit`.
    fn privacy_copy(&self, from_file: &str, to_file: &str) -> Result<(), ()> {
        if pysed(
            &self.priv_sed,
            PysedSrc::File(from_file),
            Some(to_file),
            false,
            '|',
            true,
        )
        .is_err()
        {
            return Err(());
        }
        if copystat(from_file, to_file).is_err() {
            prnstr("copystat failed", "\n", false);
        }
        Ok(())
    }

    /// Matches `copyErrLog` (`tomosnapshot:133`).
    fn copy_err_log(&mut self, errfile: &str) {
        let errfile = errfile.replace('\\', "/");
        let base = basename(&errfile);
        if !Path::new(&errfile).exists() || self.copylogs.contains(&base) {
            return;
        }
        let copied_file = join(&self.tempdir, &base);
        // The bare `except:` catches the `SystemExit` of a failed copy
        if self.privacy_copy(&errfile, &copied_file).is_err() {
            warning(&format!("Failed to copy {errfile}"));
            return;
        }
        self.copylogs.push(base.clone());

        // Add to tar list if it is other directory, or if it is not a log file;
        // otherwise the globs of logs will take care of it
        if self.tempdir != "." || !base.ends_with(".log") {
            self.tarlist.push(base);
        }
        if std::fs::set_permissions(&copied_file, std::fs::Permissions::from_mode(0o644)).is_err() {
            warning(&format!("Failed to change mode of {copied_file}"));
        }
    }

    /// Matches `makeThumbnail` (`tomosnapshot:159`).
    fn make_thumbnail(&mut self, imfile: &str) {
        if !self.do_thumbnails || !Path::new(imfile).exists() {
            return;
        }
        let mut shrink = 4.0_f64;
        let max_size = 512. * 512.;
        let result: Result<(), ImodpyError> = (|| {
            let (nx, ny, nz) = get_mrc_size(imfile)?;
            let nxy = nx as i64 * ny as i64;
            if nxy as f64 > max_size * shrink.powi(2) {
                shrink = (nxy as f64 / max_size).sqrt();
            }
            let base = basename(&format!("{imfile}.tn"));
            let outfile = join(&self.tempdir, &base);
            let newstcom = vec![
                format!("InputFile {imfile}"),
                format!("OutputFile {outfile}"),
                "ModeToOutput 0".to_owned(),
                "FloatDensities 1".to_owned(),
                format!("SectionsToRead {}", nz.div_euclid(2)),
                format!("ShrinkByFactor {}", py_str_float(shrink)),
            ];
            run_cmd("newstack -StandardInput", Some(&newstcom), None, None, &[])?;
            self.thumbnails.push(outfile);
            self.tarlist.push(base);
            Ok(())
        })();
        if result.is_err() {
            warning(&format!("Error making thumbnail of {imfile}"));
        }
    }

    /// Matches `copyHeader` (`tomosnapshot:183`).
    fn copy_header(&mut self, imfile: &str) {
        if !Path::new(imfile).exists() {
            return;
        }
        let base = basename(&format!("{imfile}.head"));
        let outfile = join(&self.tempdir, &base);
        match run_cmd(
            &format!("copyheader \"{imfile}\" \"{outfile}\""),
            None,
            None,
            None,
            &[],
        ) {
            Ok(_) => {
                self.header_copies.push(outfile);
                self.tarlist.push(base);
            }
            Err(_) => warning(&format!("Error copying header of {imfile}")),
        }
    }

    /// Matches `expandFileList` (`tomosnapshot:199`).
    fn expand_file_list(&self, file_list: &mut Vec<String>) {
        let mut to_add: Vec<String> = Vec::new();
        for ending in file_list.iter() {
            if ending.ends_with(".st") {
                for ext in &self.stack_exts[1..] {
                    to_add.push(ending.replace(".st", &format!(".{ext}")));
                }
            } else {
                for ext in &self.std_type_exts[1..] {
                    to_add.push(dataset_filename(ending, None, Some(ext)));
                }
            }
        }
        file_list.extend(to_add);
    }
}

/// `tempfile.mkdtemp(prefix = prefix, dir = dir)` of Python 3.12: a new
/// directory, mode 0700, named prefix plus eight random characters, returned
/// as an absolute path.
fn mkdtemp(prefix: &str, dir: &str) -> std::io::Result<String> {
    let characters = b"abcdefghijklmnopqrstuvwxyz0123456789_";
    let mut seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0)
        ^ ((std::process::id() as u64) << 32);
    for _ in 0..10000 {
        let mut name = prefix.to_owned();
        for _ in 0..8 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            name.push(characters[(seed % characters.len() as u64) as usize] as char);
        }
        let file = join(dir, &name);
        match std::fs::DirBuilder::new().mode(0o700).create(&file) {
            Ok(()) => return Ok(super::imodpy::os_path_abspath(&file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists))
}

/// The writer of `tarfile.open(name, 'w|gz')` (Python 3.12, `PAX_FORMAT`,
/// `_Stream` with gzip at compression level 9).
struct TarStream {
    encoder: flate2::write::DeflateEncoder<std::fs::File>,
    crc: flate2::Crc,
    offset: u64,
}

impl TarStream {
    /// `_Stream.__init__` / `_init_write_gz`: the gzip header with `FNAME`
    fn open(name: &str) -> std::io::Result<TarStream> {
        let mut file = std::fs::File::create(name)?;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs() as u32)
            .unwrap_or(0);
        let mut header = vec![0x1f, 0x8b, 8, 8];
        header.extend_from_slice(&timestamp.to_le_bytes());
        header.extend_from_slice(&[2, 0xff]);
        let mut gzname = name.to_owned();
        if gzname.ends_with(".gz") {
            gzname.truncate(gzname.len() - 3);
        }
        // "iso-8859-1" with "replace"
        for c in basename(&gzname).chars() {
            header.push(if (c as u32) < 256 { c as u8 } else { b'?' });
        }
        header.push(0);
        file.write_all(&header)?;
        Ok(TarStream {
            encoder: flate2::write::DeflateEncoder::new(file, flate2::Compression::new(9)),
            crc: flate2::Crc::new(),
            offset: 0,
        })
    }

    fn write(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.crc.update(bytes);
        self.offset += bytes.len() as u64;
        self.encoder.write_all(bytes)
    }

    /// `TarInfo._create_header`: one 512-byte header block
    fn header_block(
        name: &str,
        mode: u32,
        uid: u64,
        gid: u64,
        size: u64,
        mtime: u64,
        typeflag: u8,
        uname: &str,
        gname: &str,
    ) -> Vec<u8> {
        // `stn`: encode as ASCII with "replace", truncate, NUL-pad
        let stn = |text: &str, length: usize| -> Vec<u8> {
            let mut bytes: Vec<u8> = text
                .chars()
                .map(|c| if c.is_ascii() { c as u8 } else { b'?' })
                .collect();
            bytes.truncate(length);
            bytes.resize(length, 0);
            bytes
        };
        // `itn` for the POSIX formats: `"%0*o" % (digits - 1, n) + NUL`
        let itn = |value: u64, digits: usize| -> Vec<u8> {
            let mut bytes = format!("{value:0width$o}", width = digits - 1).into_bytes();
            bytes.push(0);
            bytes
        };
        let mut buf = Vec::with_capacity(512);
        buf.extend(stn(name, 100));
        buf.extend(itn(mode as u64 & 0o7777, 8));
        buf.extend(itn(uid, 8));
        buf.extend(itn(gid, 8));
        buf.extend(itn(size, 12));
        buf.extend(itn(mtime, 12));
        buf.extend_from_slice(b"        ");
        buf.push(typeflag);
        buf.extend(stn("", 100));
        buf.extend_from_slice(b"ustar\x00");
        buf.extend_from_slice(b"00");
        buf.extend(stn(uname, 32));
        buf.extend(stn(gname, 32));
        // devmajor and devminor are blank for anything but a device
        buf.extend(stn("", 8));
        buf.extend(stn("", 8));
        buf.extend(stn("", 155));
        buf.resize(512, 0);
        let chksum: u32 = buf.iter().map(|byte| *byte as u32).sum();
        let text = format!("{chksum:06o}\0");
        buf[148..155].copy_from_slice(text.as_bytes());
        buf
    }

    /// `_create_payload`: data padded to a block
    fn payload(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.write(bytes)?;
        let remainder = bytes.len() % 512;
        if remainder > 0 {
            self.write(&vec![0u8; 512 - remainder])?;
        }
        Ok(())
    }

    /// `TarFile.add(name)`, recursive for a directory.
    fn add(&mut self, name: &str) -> std::io::Result<()> {
        let meta = std::fs::symlink_metadata(name)?;
        let arcname = name.trim_start_matches('/');
        let (typeflag, size, arcname) = if meta.is_dir() {
            let mut dirname = arcname.to_owned();
            if !dirname.ends_with('/') {
                dirname.push('/');
            }
            (b'5', 0, dirname)
        } else if meta.is_file() {
            (b'0', meta.len(), arcname.to_owned())
        } else {
            // Not reached: only regular files and the com directory are added
            return Err(std::io::Error::from(std::io::ErrorKind::Unsupported));
        };
        let uname = user_name(meta.uid());
        let gname = group_name(meta.gid());

        // `create_pax_header`: a string field that is too long or not ASCII,
        // and the float mtime, go into a pax header
        let mut records: Vec<(String, String)> = Vec::new();
        for (value, hname, length) in [
            (arcname.as_str(), "path", 100),
            (uname.as_str(), "uname", 32),
            (gname.as_str(), "gname", 32),
        ] {
            if !value.is_ascii() || value.chars().count() > length {
                records.push((hname.to_owned(), value.to_owned()));
            }
        }
        let mut uid = meta.uid() as u64;
        let mut gid = meta.gid() as u64;
        let mut size_field = size;
        for (field, value, digits, hname) in [
            (&mut uid, meta.uid() as u64, 8u32, "uid"),
            (&mut gid, meta.gid() as u64, 8, "gid"),
            (&mut size_field, size, 12, "size"),
        ] {
            if value >= 8u64.pow(digits - 1) {
                *field = 0;
                records.push((hname.to_owned(), value.to_string()));
            }
        }
        // `st_mtime` is `sec + nsec * 1e-9`, a float, so it is always recorded
        let mtime_float = meta.mtime() as f64 + meta.mtime_nsec() as f64 * 1e-9;
        let mut mtime = py_round(mtime_float) as i64;
        if !(0..8i64.pow(11)).contains(&mtime) {
            mtime = 0;
        }
        records.push(("mtime".to_owned(), py_str_float(mtime_float)));

        // `_create_pax_generic_header`
        let mut text = Vec::new();
        for (keyword, value) in &records {
            let body = format!(" {keyword}={value}\n");
            let l = body.len();
            let mut n = 0;
            let mut p = 0;
            loop {
                n = l + p.to_string().len();
                if n == p {
                    break;
                }
                p = n;
            }
            text.extend_from_slice(format!("{p}{body}").as_bytes());
        }
        let pax = Self::header_block(
            "././@PaxHeader",
            0,
            0,
            0,
            text.len() as u64,
            0,
            b'x',
            "",
            "",
        );
        self.write(&pax)?;
        self.payload(&text)?;
        let header = Self::header_block(
            &arcname,
            meta.mode(),
            uid,
            gid,
            size_field,
            mtime as u64,
            typeflag,
            &uname,
            &gname,
        );

        if meta.is_dir() {
            self.write(&header)?;
            let mut entries: Vec<String> = std::fs::read_dir(name)?
                .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
                .collect::<std::io::Result<_>>()?;
            entries.sort();
            for entry in entries {
                self.add(&join(name, &entry))?;
            }
        } else {
            let data = std::fs::read(name)?;
            self.write(&header)?;
            self.payload(&data)?;
        }
        Ok(())
    }

    /// `TarFile.close` and `_Stream.close`
    fn close(mut self) -> std::io::Result<()> {
        self.write(&[0u8; 1024])?;
        let remainder = self.offset % 10240;
        if remainder > 0 {
            self.write(&vec![0u8; (10240 - remainder) as usize])?;
        }
        let crc = self.crc.sum();
        let length = self.crc.amount();
        let mut file = self.encoder.finish()?;
        file.write_all(&crc.to_le_bytes())?;
        file.write_all(&length.to_le_bytes())?;
        Ok(())
    }
}

/// `pwd.getpwuid(uid)[0]`, or '' when there is none
fn user_name(uid: u32) -> String {
    // SAFETY: getpwuid returns a pointer into static storage or null; the
    // name is copied out at once.
    unsafe {
        let entry = libc::getpwuid(uid);
        if entry.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr((*entry).pw_name)
            .to_string_lossy()
            .into_owned()
    }
}

/// `grp.getgrgid(gid)[0]`, or '' when there is none
fn group_name(gid: u32) -> String {
    // SAFETY: as for `user_name`
    unsafe {
        let entry = libc::getgrgid(gid);
        if entry.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr((*entry).gr_name)
            .to_string_lossy()
            .into_owned()
    }
}

/// The script's top level (`tomosnapshot:239-873`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn tomosnapshot(arguments: &[OsString]) -> i32 {
    let prefix = format!("ERROR: {PROGNAME} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let done = |status: i32| {
        let _ = std::io::stdout().flush();
        status
    };

    // The lists of different kinds of files for each kind of etomo data set
    let strs = |list: &[&str]| -> Vec<String> { list.iter().map(|s| (*s).to_owned()).collect() };
    let edf_set_files = strs(&[
        ".rawtlt",
        "_peak.mod",
        ".prexf",
        ".prexg",
        ".seed",
        ".fid",
        ".3dmod",
        ".tlt",
        "fid.xyz",
        ".resid",
        ".resmod",
        ".tltxf",
        "_fid.xf",
        "local.xf",
        ".matmod",
        ".pl",
        ".ecd",
        "_orig.seed",
        ".xtilt",
        ".maggrad",
        ".zfac",
        ".defocus",
        "_erase.fid",
        "_3dfind.mod",
        "_flat.mod",
        "_flat.mod",
        "_fid.tlt",
        ".erase",
        "_rawbound.mod",
        "_afsbound.mod",
        "_ptbound.mod",
        "_pt.fid",
        ".xf",
    ]);
    let edf_axis_headers = strs(&[
        "_orig.st",
        ".st",
        ".preali",
        ".ali",
        ".rec",
        ".mat",
        "_full.rec",
    ]);
    let edf_single_files = strs(&[
        "solvezero.xf",
        "solve.xf",
        "refine.xf",
        "warp.xf",
        "inverse.xf",
        "patch.out",
        "patch_vector.mod",
        "patch_vector_ccc.mod",
        "patch_region.mod",
        "processchunks.out",
        "savework",
        "processchunksa.out",
        "processchunksb.out",
        "processchunks.csh",
        "processchunksa.csh",
        "processchunksb.csh",
        "transferfid.coord",
        "combine.out",
        "volcombine.csh",
        "tomopitcha.mod",
        "tomopitchb.mod",
        "tomopitch.mod",
        "rotation.xf",
    ]);
    let edf_single_headers: Vec<String> = Vec::new();
    let edf_axis_copy_heads = strs(&[".st"]);
    let edf_axis_thumbs = strs(&[".st", ".rec", ".mat", "_full.rec"]);
    let edf_single_thumbs = strs(&[
        "mid.rec", "mida.rec", "midb.rec", "bot.rec", "bota.rec", "botb.rec", "top.rec",
        "topa.rec", "topb.rec", "sum.rec",
    ]);
    let ejf_set_files = strs(&[
        ".info",
        ".tomoxf",
        ".tomoxg",
        ".xf",
        "_auto.xcxf",
        "_auto.xf",
        "_empty.xf",
        "_midas.xf",
        ".sqzxf",
        ".xpndxf",
        "_refine.mod",
        "_join.mod",
        "_refine.alimod",
        "_refine.xf",
        "_refine.xg",
        "_refinejoin.xg",
    ]);
    let epe_set_files = strs(&[".prm"]);
    let epe_single_files = strs(&["processchunks.out", "processchunks.csh"]);
    let epp_single_files = strs(&["processchunks.out", "processchunks.csh"]);
    let ess_set_files = strs(&[
        ".xf",
        ".xg",
        "_auto.xcxf",
        "_auto.xf",
        "_auto.linxf",
        "_midas.xf",
        "_bound.mod",
    ]);
    let ess_axis_headers = strs(&["_preblend.mrc", "_ali.mrc"]);
    let ess_axis_thumbs = strs(&["_preblend.mrc", "_ali.mrc"]);
    let ebt_set_files = strs(&["_project.log", ".adoc"]);

    let all_set_files = [
        edf_set_files,
        ejf_set_files,
        epe_set_files,
        Vec::new(),
        Vec::new(),
        ess_set_files,
        ebt_set_files,
    ];
    let mut all_axis_headers = [
        edf_axis_headers,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        ess_axis_headers,
        Vec::new(),
    ];
    let mut all_single_files = [
        edf_single_files,
        Vec::new(),
        epe_single_files,
        Vec::new(),
        epp_single_files,
        Vec::new(),
        Vec::new(),
    ];
    let mut all_single_headers = [
        edf_single_headers,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    let mut all_axis_thumbs = [
        edf_axis_thumbs,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        ess_axis_thumbs,
        Vec::new(),
    ];
    let mut all_single_thumbs = [
        edf_single_thumbs,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    let mut all_axis_copy_heads = [
        edf_axis_copy_heads,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];

    // The extensions of file types
    let ext_list = ["edf", "ejf", "epe", "epp", "ess", "ebt"];

    // Keys for different types of files in each file type
    let all_set_keys: [&[&str]; 6] = [
        &["Setup.DatasetName"],
        &["Join.RootName"],
        &["Peet.RootName"],
        &["AnisotropicDiffusion.RootName", "Parallel.RootName"],
        &["SerialSections.RootName"],
        &["meta.RootName"],
    ];

    let mut snap = Snapshot {
        tempdir: ".".to_owned(),
        copylogs: Vec::new(),
        tarlist: Vec::new(),
        thumbnails: Vec::new(),
        header_copies: Vec::new(),
        priv_sed: Vec::new(),
        do_thumbnails: false,
        std_type_exts: Vec::new(),
        stack_exts: Vec::new(),
    };

    //
    // Setup runtime environment
    let imod_dir = match std::env::var("IMOD_DIR") {
        Ok(dir) => {
            super::imodpy::add_imod_bin_ignore_sighup();
            dir
        }
        Err(_) => {
            print!("{prefix} IMOD_DIR is not defined!\n");
            return done(1);
        }
    };

    // Startup
    let uname_file = "uname.out";
    let lslrt_file = "lslrt.out";

    let options = strs(&[
        "e:EtomoFile:FN:Etomo data file to define type of snapshot (can be non-option argument)",
        "o:OutputFile:FN:Name of tarred output file (default rootname-snapshot)",
        "w:WriteableDirectory:FN:Location at which to place output and temporary files",
        "t:Thumbnails:B:Make thumbnails of middle sections of some image files",
        "s:SkipCygcheck:B:Skip running cygcheck on Windows",
        "help:usage:B:Print usage output",
    ]);
    pip_exit_on_error(0, &prefix);
    // A `None` return is unpacked into a tuple: a TypeError traceback
    if pip_parse_input(&argv, &options).is_err() {
        let _ = std::io::stdout().flush();
        eprintln!("Traceback (most recent call last):");
        eprintln!("TypeError: cannot unpack non-iterable NoneType object");
        crate::imod::libcfshr::b3dutil::exit(1);
    }

    // Print help
    if pip_get_boolean("help", 0).unwrap_or(0) != 0 {
        prnstr(
            "   tomosnapshot collects small files and other information about
   one form of Etomo processing.
   If no etomo data file is specified, it searches for files in this order:
   tomogram generation (.edj), joining (.ejf), PEET (.epe), NAD (.epe),
   generic parallel processing (.epe), serial sections (.ess), batch (.ebt)
   It must be run from the data directory, but you do not need to have write
   permission in that directory if you specify an alternate writeable directory.
   Run \"imodhelp tomosnapshot\" to see privacy practices implemented for these
   snapshots.",
            "\n",
            false,
        );

        pip_print_help(PROGNAME, 0, 0, 0);
        return done(0);
    }

    // Process options
    let mut etomo_file: Option<String> = pip_get_in_out_file("EtomoFile", 0).ok().flatten();
    let mut input_dir = String::new();
    if let Some(file) = etomo_file.as_mut().filter(|file| !file.is_empty()) {
        *file = file.replace('\\', "/");
        input_dir = dirname(file);
        if input_dir.is_empty() {
            input_dir = ".".to_owned();
        }
        input_dir.push('/');
    }

    snap.do_thumbnails = pip_get_boolean("Thumbnails", 0).unwrap_or(0) != 0;
    let writeable_dir = pip_get_string("WriteableDirectory", ".").unwrap_or_default();
    let _skip_cygcheck = pip_get_boolean("SkipCygcheck", 0).unwrap_or(0);

    if !Path::new(&writeable_dir).is_dir() {
        exit_error(&format!("{writeable_dir} is not a directory"));
    }
    let writable = CString::new(writeable_dir.as_str())
        // SAFETY: a NUL-terminated path for access(2)
        .map(|path| unsafe { libc::access(path.as_ptr(), libc::W_OK) } == 0)
        .unwrap_or(false);
    if !writable {
        exit_error(&format!(
            "You do not have permission to write in the directory {writeable_dir}"
        ));
    }

    for ext in ext_list {
        if etomo_file.is_none() {
            let files = glob_glob(&format!("*.{ext}"));
            if !files.is_empty() {
                etomo_file = Some(files[0].clone());
            }
        }
    }

    // If there is an etomo file at all, search it for the setname of appropriate type
    let mut type_ind = 0usize;
    let mut setname: Option<String> = None;
    let mut axis_type: Option<String> = None;
    let mut view_type: Option<String> = None;
    let mut copylist: Vec<String> = Vec::new();
    snap.tarlist = vec![uname_file.to_owned(), lslrt_file.to_owned()];
    let mut etomo_lines: Vec<String> = Vec::new();
    if let Some(file) = etomo_file.clone().filter(|file| !file.is_empty()) {
        copylist.push(file.clone());
        etomo_lines = read_text_file(&file, None, false, None).unwrap_or_default();
        let mut matched = false;
        for ext_ind in 0..ext_list.len() {
            if file.ends_with(ext_list[ext_ind]) {
                matched = true;

                // If the extension matches, check for the keys that go with that extension
                let mut found = false;
                for key in all_set_keys[ext_ind] {
                    setname = get_key_value(&etomo_lines, key);
                    if setname.as_deref().is_some_and(|name| !name.is_empty()) {
                        found = true;
                        break;
                    }
                    type_ind += 1;
                }

                // If no key is found, it is an error; except fall back to xcorr for an edf
                if !found {
                    if type_ind > 1 {
                        exit_error("Data set root name not found in etomo data file");
                    }
                    warning("Data set root name not found in etomo data file");
                    etomo_file = None;
                    break;
                }
            }

            // done looking if got setname or had to fall back; otherwise increment type
            if setname.as_deref().is_some_and(|name| !name.is_empty()) || etomo_file.is_none() {
                break;
            }
            type_ind += all_set_keys[ext_ind].len();
        }
        if !matched {
            exit_error(&format!(
                "Specified etomo file {file} does not have an appropriate extension"
            ));
        }

        // For edf, look up the axis type; fall back to looking at xcorr.com
        if etomo_file.is_some() && type_ind == TOMO {
            axis_type = get_key_value(&etomo_lines, "Setup.AxisType");
            if axis_type.as_deref().is_none_or(str::is_empty) {
                warning("Cannot find axis type in edf file");
                etomo_file = None;
            }
            view_type = get_key_value(&etomo_lines, "Setup.ViewType");
        }
    }

    // If there was no etomo file found or specified or failure occurred, now do the fallback
    if etomo_file.is_none() {
        type_ind = TOMO;
        let (_com_ext, dual_num, root, _type_ext, _stack_ext) =
            find_root_axis_and_extensions(0, None);
        if dual_num == 2 {
            axis_type = Some("Dual Axis".to_owned());
        } else if dual_num >= 0 {
            axis_type = Some("Single Axis".to_owned());
        } else if setname.as_deref().is_some_and(|name| !name.is_empty()) {
            exit_error("Cannot determine axis type from com/pcm files in the input directory");
        } else {
            exit_error(
                "Cannot find an etomo file or sufficient com/pcm files in the input directory to proceed",
            );
        }

        // Get the setname from the analysis if it is still needed
        if setname.as_deref().is_none_or(str::is_empty) {
            if root.is_empty() {
                exit_error(
                    "Cannot find dataset name from analysis of files; if running from Etomo, do File - Save and try again",
                );
            }
            setname = Some(root);
        }
    }
    let setname = setname.unwrap_or_default();

    // Now we know the data type, get file lists
    let one_axis_set_files = all_set_files[type_ind].clone();
    let mut one_axis_headers = std::mem::take(&mut all_axis_headers[type_ind]);
    let mut single_files = std::mem::take(&mut all_single_files[type_ind]);
    let mut single_headers = std::mem::take(&mut all_single_headers[type_ind]);
    let mut one_axis_copy_heads = std::mem::take(&mut all_axis_copy_heads[type_ind]);
    let mut one_axis_thumbs = std::mem::take(&mut all_axis_thumbs[type_ind]);
    let mut single_thumbs = std::mem::take(&mut all_single_thumbs[type_ind]);

    // Expand some of the file lists for possible stack extensions and naming styles
    set_root_and_extension("", "");
    snap.std_type_exts = standard_type_extensions();
    snap.stack_exts = allowed_raw_stack_extensions();
    for file_list in [
        &mut one_axis_headers,
        &mut single_headers,
        &mut one_axis_copy_heads,
        &mut one_axis_thumbs,
        &mut single_thumbs,
    ] {
        snap.expand_file_list(file_list);
    }

    let mut naxis = 1;
    let mut axislet = "";
    if axis_type.as_deref() == Some("Dual Axis") {
        naxis = 2;
        axislet = "a";
    }

    let output_file =
        pip_get_string("OutputFile", &format!("{setname}-snapshot")).unwrap_or_default();

    // Get a temporary dir for privacy copies, and the com file dir
    snap.tempdir = match mkdtemp("tomosnaptmp-", &writeable_dir) {
        Ok(dir) => dir,
        Err(_) => exit_error("Creating temporary directory to copy files into"),
    };
    let comdir = match mkdtemp("tomosnapshot.cms.", &snap.tempdir) {
        Ok(dir) => dir.replace('\\', "/"),
        Err(_) => exit_error("Creating temporary directory for com files"),
    };

    // Get the directory listing and the uname output
    // Find out if there is no cygwin: i.e., it is windows python and cygcheck or uname fails
    // 9/16/21: Reinhard Rechel had a machine where uname ran, so do cygcheck first
    let mut uname_out: Vec<String> = Vec::new();
    if let Ok(lines) = run_cmd("uname -a", None, None, Some("stdout"), &[]) {
        uname_out = lines.unwrap_or_default();
        // `unameOut[0]` of an empty output is an IndexError, caught by the
        // same `except Exception`
        if let Some(first) = uname_out.first() {
            let usplit: Vec<&str> = first.split(' ').collect();
            if usplit.len() > 2 {
                let mut utrim = vec![usplit[0]];
                utrim.extend_from_slice(&usplit[2..]);
                uname_out = vec![utrim.join(" ")];
            }
        }
    }

    // Get the Python version
    uname_out.push(
        "Python version: none - this is the imod-rs Rust translation of tomosnapshot".to_owned(),
    );
    // Not caught in the source: a failure is a traceback and status 1
    let infolines = match run_cmd("imodinfo", None, None, None, &[]) {
        Ok(lines) => lines.unwrap_or_default(),
        Err(_) => exit_from_imod_error(PROGNAME),
    };
    if !infolines.is_empty() {
        uname_out.push(infolines[0].clone());
    }

    // Get the names of CUDA libs so we know what package this is
    let mut cuda_path = join(&imod_dir, "qtlib");
    if !Path::new(&cuda_path).exists() {
        cuda_path = join(&imod_dir, "bin");
    }
    let cuda_list = glob_glob(&join(&cuda_path, "*cuda*"));
    if !cuda_list.is_empty() {
        uname_out.extend(cuda_list);
    }

    let mut user_name = String::new();
    let mut nwuser_name = String::new();
    let mut dir_name = String::new();
    for (k, v) in std::env::vars_os() {
        let k = k.to_string_lossy().into_owned();
        let v = v.to_string_lossy().into_owned();
        if (k == "USER" || k == "USERNAME") && user_name.is_empty() {
            user_name = v.clone();
        } else if k == "NWUSERNAME" && nwuser_name.is_empty() {
            // Fixed in translation (BUGS.md): native assigns `nwUserName`, a
            // different name, so the substitution below is never added
            nwuser_name = v.clone();
        } else if (k == "HOME" || k == "HOMEPATH") && dir_name.is_empty() {
            let home = v.replace('\\', "/");
            dir_name = home.rsplit('/').next().unwrap_or("").to_owned();
        }

        uname_out.push(format!("{k}={v}"));
    }

    // Compose privacy sed command
    if !dir_name.is_empty() && user_name.is_empty() {
        user_name = dir_name;
    }

    snap.priv_sed = strs(&[
        "|COMPUTERNAME *=|d",
        "|EMAILADDR *=|d",
        "|MAIL *=|d",
        "|HOME *=|d",
        "|HOMEPATH *=|d",
        "|HOST *=|d",
        "|HOSTNAME *=|d",
        "|LOGNAME *=|d",
        "|LOGONSERVER *=|d",
        "|MACADDRESS *=|d",
        "|MAIL *=|d",
        "|NWUSERNAME *=|d",
        "|SSH_CLIENT *=|d",
        "|SSH_CONNECTION *=|d",
        "|UNC[^ ]* *=|d",
        "|EmailAddress |d",
        "|USER *=|d",
        "|USERDNSDOMAIN *=|d",
        "|USERDOMAIN *=|d",
        "|USERDOMAIN_ROAMINGPROFILE *=|d",
        "|USERNAME *=|d",
    ]);
    if !user_name.is_empty() {
        snap.priv_sed.push(format!("|{user_name}|s||XXXX|g"));
    }
    if !nwuser_name.is_empty() {
        snap.priv_sed.push(format!("|{nwuser_name}|s||XXXX|g"));
    }

    // (The registry dump with `cygcheck -s -v -r` is for Cygwin and Windows only)

    // Type-specific additions to lists here:
    //
    // TOMOGRAM
    if type_ind == TOMO {
        single_files.extend(glob_glob("*bound.info"));
        single_files.extend(glob_glob("autofidseed*.info"));
        let mut adocs = glob_glob("batch*.adoc");
        adocs.extend(glob_glob("*emplate*.adoc"));
        single_files.extend(adocs);
    }

    //
    // JOIN
    if type_ind == JOIN {
        let infofile = format!("{setname}.info");
        if Path::new(&infofile).exists() {
            let infolines = read_text_file(&infofile, None, false, None).unwrap_or_default();
            let nlines = infolines.len();
            if nlines > 3 {
                let lsplit: Vec<&str> = infolines[0].split_whitespace().collect();
                let mut numfiles = 0i64;
                if lsplit.len() > 1 {
                    // Fixed in translation (BUGS.md): a first word that is
                    // not an integer is a ValueError traceback natively; it
                    // counts as no files here
                    numfiles = py_int(lsplit[0]).unwrap_or(0);
                }
                if numfiles > 0 {
                    let start = 3.max(nlines as i64 - numfiles) as usize;
                    for i in start..nlines {
                        single_headers.push(infolines[i].clone());
                        single_thumbs.push(infolines[i].clone());
                    }
                }
            }
        }

        let mut added = strs(&[
            ".sampavg",
            ".sample",
            ".join",
            "_modeled.join",
            "_trial.join",
        ]);
        snap.expand_file_list(&mut added);
        one_axis_headers.extend(added);
    }

    //
    // PEET
    if type_ind == PEET {
        single_files.extend(glob_glob(&format!("{setname}*.csv")));
        let refs = glob_glob(&format!("*{setname}*_Ref*.mrc"));
        single_headers.extend(refs.iter().cloned());
        single_thumbs.extend(refs);
        let avgs = glob_glob(&format!("*{setname}*_AvgVol*.mrc"));
        single_headers.extend(avgs.iter().cloned());
        single_thumbs.extend(avgs);
        let prm_file = format!("{setname}.prm");
        if Path::new(&prm_file).exists() {
            let prm_lines = read_text_file(&prm_file, None, false, None).unwrap_or_default();
            if let Some(volumes) = prm_unique_entries(&prm_lines, "fnVolume") {
                if !volumes.is_empty() {
                    single_headers.extend(volumes.iter().cloned());
                    single_thumbs.extend(volumes);
                }
            }
            // Fixed in translation (BUGS.md): with no `reference` entry native
            // calls `find` on None, an AttributeError traceback
            if let Some(reference) = get_key_value(&prm_lines, "reference") {
                if reference.contains('\'') {
                    single_headers.push(reference.trim_matches('\'').to_owned());
                }
            }
            for key in ["fnModParticle", "initMOTL"] {
                if let Some(entries) = prm_unique_entries(&prm_lines, key) {
                    for entry in entries {
                        // If a file with the base name is here, use it by adding to list;
                        // if it is not, copy it here and have it deleted at end
                        let entry = entry.replace('\\', "/");
                        let base = basename(&entry);
                        if Path::new(&base).exists() {
                            single_files.push(base);
                        } else {
                            snap.copy_err_log(&entry);
                        }
                    }
                }
            }
        }
    }

    //
    // NAD
    let mut naddir: Option<String> = None;
    if type_ind == NAD {
        let dir = format!("naddir.{setname}");
        let mut added = vec![setname.clone(), join(&dir, "test.input")];
        snap.expand_file_list(&mut added);
        single_headers.extend(added);
        single_files.push(format!("{dir}-pc.csh"));
        single_files.push(join(&dir, "processchunks.out"));
        single_headers.extend(glob_glob(&join(&dir, "test.K*[0-9]*[^~]")));
        naddir = Some(dir);
    }

    //
    // Serial sections
    if type_ind == SERIAL && etomo_file.is_some() {
        if let Some(stack) =
            get_key_value(&etomo_lines, "SerialSections.Stack").filter(|s| !s.is_empty())
        {
            single_headers.push(stack.clone());
            single_thumbs.push(stack);
        }
    }

    //
    // Batch tomograms
    if type_ind == BATCH {
        // Add the stacks if they are in the table
        let mut key_num = 1;
        loop {
            let stack = match get_key_value(&etomo_lines, &format!("meta.ref.ebt{key_num}")) {
                Some(stack) if !stack.is_empty() => stack,
                _ => break,
            };
            if get_key_value(&etomo_lines, &format!("meta.row.ebt{key_num}.RowNumber"))
                .is_some_and(|value| !value.is_empty())
            {
                single_headers.push(stack.clone());

                // Strip a or b from dataset name if it is dual and compose adoc name
                let dual = get_key_value(&etomo_lines, &format!("meta.row.ebt{key_num}.dual"))
                    .as_deref()
                    == Some("true");
                let (mut stack_root, _ext) = os_path_splitext(&basename(&stack));
                if dual && (stack_root.ends_with('a') || stack_root.ends_with('b')) {
                    stack_root.pop();
                }
                let set_adoc = format!("{}_{stack_root}.adoc", join(&dirname(&stack), &setname));

                // Add to copy list if it exists
                if Path::new(&set_adoc).exists() {
                    copylist.push(set_adoc);
                } else {
                    // If not, find original stack location and check there
                    if let Some(orig_stack) =
                        get_key_value(&etomo_lines, &format!("meta.row.ebt{key_num}.OrigStack"))
                            .filter(|s| !s.is_empty())
                    {
                        let set_adoc = format!(
                            "{}_{stack_root}.adoc",
                            join(&dirname(&orig_stack), &setname)
                        );
                        if Path::new(&set_adoc).exists() {
                            copylist.push(set_adoc);
                        }
                    }
                }
            }

            key_num += 1;
        }
    }

    // Run the file listing now that we know if we need a subdir
    let cur_dir = std::env::current_dir().unwrap_or_default();
    let list_dir = naddir.clone().or_else(|| {
        if input_dir.is_empty() {
            None
        } else {
            Some(input_dir.clone())
        }
    });
    if let Some(dir) = &list_dir {
        // Fixed in translation (BUGS.md): a missing directory is an uncaught
        // FileNotFoundError natively
        if std::env::set_current_dir(dir).is_err() {
            exit_error(&format!("Changing to directory {dir}"));
        }
    }
    // Not caught in the source: a failure is a traceback and status 1
    let mut lslrt = match run_cmd("ls -lrt", None, None, None, &[]) {
        Ok(lines) => lines.unwrap_or_default(),
        Err(_) => exit_from_imod_error(PROGNAME),
    };
    match run_cmd("ls -ld .", None, None, None, &[]) {
        Ok(lines) => lslrt.extend(lines.unwrap_or_default()),
        Err(_) => exit_from_imod_error(PROGNAME),
    }
    let lsplit: Vec<String> = lslrt
        .last()
        .map(|line| line.split_whitespace().map(str::to_owned).collect())
        .unwrap_or_default();
    let owner = lsplit.get(2).cloned().unwrap_or_default();
    snap.priv_sed.push(format!("|{owner}|s||YYYY|g"));
    if list_dir.is_some() {
        let _ = std::env::set_current_dir(&cur_dir);
    }

    let mut priv_lines = pysed(
        &snap.priv_sed,
        PysedSrc::Lines(&lslrt),
        None,
        false,
        '|',
        false,
    )
    .ok()
    .flatten()
    .unwrap_or_default();
    strip_and_write(&join(&snap.tempdir, lslrt_file), &mut priv_lines);

    // Get error logs from elsewhere
    if Path::new("etomo_err.log").exists() {
        let errlines = read_text_file("etomo_err.log", None, false, None).unwrap_or_default();
        for line in errlines {
            if line.starts_with("Error log") {
                let nameind = line.find("is in ").map_or(5, |index| index + 6);
                if nameind > 10 {
                    snap.copy_err_log(&line[nameind..]);
                }
            }
        }
    }

    // Get last 10 from the log directory (may be redundant)
    // NOTE csh tomosnapshot has a bug, should be ls -rt
    let mut log_dir = std::env::var("ETOMO_LOG_DIR").ok();
    if log_dir.is_none() {
        if let Some(home) = std::env::var("HOME").ok().filter(|home| !home.is_empty()) {
            log_dir = Some(join(&home, ".etomologs"));
        }
    }
    if let Some(log_dir) = log_dir.filter(|dir| !dir.is_empty()) {
        let readable = CString::new(log_dir.as_str())
            // SAFETY: a NUL-terminated path for access(2)
            .map(|path| unsafe { libc::access(path.as_ptr(), libc::R_OK) } == 0)
            .unwrap_or(false);
        if Path::new(&log_dir).exists() && Path::new(&log_dir).is_dir() && readable {
            let errlogs = glob_glob(&join(&log_dir, "etomo_err*.log"));
            let numlogs = errlogs.len();
            if numlogs > 0 {
                // sort the logs by newest first (higher mtime)
                let mut sort_ind: Vec<usize> = (0..numlogs).collect();
                let mut mtimes: Vec<f64> = Vec::new();
                for i in 0..numlogs {
                    // An OSError here is uncaught natively
                    let meta = match std::fs::metadata(&errlogs[i]) {
                        Ok(meta) => meta,
                        Err(_) => exit_error(&format!("Getting time of {}", errlogs[i])),
                    };
                    mtimes.push(meta.mtime() as f64 + meta.mtime_nsec() as f64 * 1e-9);
                }
                if numlogs > 1 {
                    for i in 0..numlogs - 1 {
                        for j in i + 1..numlogs {
                            if mtimes[sort_ind[i]] < mtimes[sort_ind[j]] {
                                sort_ind.swap(i, j);
                            }
                        }
                    }
                }
                for i in 0..10.min(numlogs) {
                    snap.copy_err_log(&errlogs[sort_ind[i]]);
                }
            }
        }
    }

    // Get all com and log files and backups except the ta*.log
    // Look in naddir for NAD, or confine to setname-* for parallel processing
    let mut globpref = input_dir.clone();
    if let Some(dir) = &naddir {
        globpref += &format!("{dir}/");
    } else if type_ind == PARALLEL {
        globpref += &format!("{setname}-");
    }

    if type_ind == BATCH {
        globpref = format!("{input_dir}{setname}");
    } else {
        globpref.push('*');
    }
    let mut comlist = glob_glob(&format!("{globpref}.com"));
    comlist.extend(glob_glob(&format!("{globpref}.com~")));
    let mut comlistb = glob_glob(&format!("{globpref}.pcm"));
    comlistb.extend(glob_glob(&format!("{globpref}.pcm~")));
    let mut loglist = glob_glob(&format!("{globpref}.log"));
    let mut loglistb = glob_glob(&format!("{globpref}.log~"));
    for logs in [&mut loglist, &mut loglistb] {
        for i in (0..logs.len()).rev() {
            if basename(&logs[i]).starts_with("ta") {
                logs.remove(i);
            }
        }
    }

    // Copy the com files to the com dir with .com stripped
    for comfile in &comlist {
        let (mut base, ext) = os_path_splitext(&basename(&comfile.replace('\\', "/")));
        if ext.ends_with('~') {
            base.push('~');
        }
        // The bare `except:` catches the `SystemExit` of a failed copy
        if snap.privacy_copy(comfile, &join(&comdir, &base)).is_err() {
            warning(&format!("Failed to copy {comfile} to {comdir}"));
        }
    }

    snap.tarlist.push(basename(&comdir));

    // Put the logs and backup coms on a copy list
    copylist.extend(comlistb);
    copylist.extend(loglist);
    copylist.extend(loglistb);

    // Loop on axis files, adding files to the copy list if they exist, getting headers
    for _axnum in 0..naxis {
        for ext in &one_axis_set_files {
            let sfile = format!("{input_dir}{setname}{axislet}{ext}");
            if Path::new(&sfile).exists() {
                copylist.push(sfile);
            }
        }

        for ext in &one_axis_headers {
            let sfile = format!("{input_dir}{setname}{axislet}{ext}");
            if Path::new(&sfile).exists() {
                match run_cmd(&format!("header {sfile}"), None, None, None, &[]) {
                    Ok(lines) => uname_out.extend(lines.unwrap_or_default()),
                    Err(_) => warning(&format!("Failed to run header on {sfile}")),
                }
            }
        }

        for ext in &one_axis_thumbs {
            let mut use_ext = ext.clone();
            let thumb_root = format!("{input_dir}{setname}{axislet}");

            // For a montage, get a shot of the preali or ali if possible
            if type_ind == TOMO
                && snap
                    .stack_exts
                    .iter()
                    .any(|stack| stack.as_str() == &ext[1..])
                && view_type.as_deref() == Some("Montage")
            {
                for type_ext in &snap.std_type_exts {
                    let preali = dataset_filename(".preali", None, Some(type_ext));
                    let ali = dataset_filename(".ali", None, Some(type_ext));
                    if Path::new(&format!("{thumb_root}{preali}")).exists() {
                        use_ext = preali;
                        break;
                    } else if Path::new(&format!("{thumb_root}{ali}")).exists() {
                        use_ext = ali;
                        break;
                    }
                }
            }

            let sfile = format!("{thumb_root}{use_ext}");
            snap.make_thumbnail(&sfile);
        }

        for ext in &one_axis_copy_heads {
            let sfile = format!("{input_dir}{setname}{axislet}{ext}");
            snap.copy_header(&sfile);
        }

        axislet = "b";
    }

    // Add single files to copy list if they exist
    for sfile in &single_files {
        let sfile = format!("{input_dir}{sfile}");
        if Path::new(&sfile).exists() {
            copylist.push(sfile);
        }
    }

    // Get headers of single files if they exist.
    // They may have spaces in path so run header with standard input
    for sfile in &single_headers {
        let mut sfile = sfile.clone();
        if type_ind != BATCH {
            sfile = format!("{input_dir}{sfile}");
        }
        if Path::new(&sfile).exists() {
            match run_cmd(
                "header -StandardInput",
                Some(&[format!("InputFile {sfile}")]),
                None,
                None,
                &[],
            ) {
                Ok(lines) => uname_out.extend(lines.unwrap_or_default()),
                Err(_) => warning(&format!("Failed to run header on {sfile}")),
            }
        }
    }

    // Get thumbnails, without adding input directory since join files can have abs paths
    for sfile in &single_thumbs {
        snap.make_thumbnail(sfile);
    }

    // Add them all to the tar list, or copy them to other directory if needed and add to
    // tar list only if copy succeeds
    if snap.tempdir == "." {
        snap.tarlist.extend(copylist);
    } else {
        for sfile in &copylist {
            // (`sfile.replace('\\', '/')` discards its result in the source)
            let (_fileroot, ext) = os_path_splitext(sfile);
            let copied: std::io::Result<()> =
                if ext.contains("mod") || ext.contains("fid") || ext.contains("seed") {
                    // "You need to use copy2, not copy then copystat, to get dates right"
                    // Hopefully that was with older Python: copystat seems to work in privacy
                    // copies
                    let target = join(&snap.tempdir, &basename(sfile));
                    std::fs::copy(sfile, &target).and_then(|_| copystat(sfile, &target))
                } else {
                    // The `SystemExit` of a failed privacy copy is not an
                    // `Exception`, so it ends the script here
                    if snap
                        .privacy_copy(sfile, &join(&snap.tempdir, &basename(sfile)))
                        .is_err()
                    {
                        crate::imod::libcfshr::b3dutil::exit(1);
                    }
                    Ok(())
                };
            match copied {
                Ok(()) => snap.tarlist.push(basename(sfile)),
                Err(_) => warning(&format!("Error copying {sfile} to {}", snap.tempdir)),
            }
        }
    }

    // Write uname.out at last
    let mut priv_lines = pysed(
        &snap.priv_sed,
        PysedSrc::Lines(&uname_out),
        None,
        false,
        '|',
        false,
    )
    .ok()
    .flatten()
    .unwrap_or_default();
    strip_and_write(&join(&snap.tempdir, uname_file), &mut priv_lines);

    // Tar the files
    let mut out_file = output_file.clone();
    if snap.tempdir != "." {
        if std::env::set_current_dir(&snap.tempdir).is_err() {
            exit_error(&format!("Changing to directory {}", snap.tempdir));
        }
        out_file = join("..", &output_file);
    }

    let tarred: std::io::Result<()> = (|| {
        let mut tar_file = TarStream::open(&out_file)?;
        for sfile in &snap.tarlist {
            if tar_file.add(sfile).is_err() {
                warning(&format!("Error adding {sfile} to the tar file"));
            }
        }
        tar_file.close()
    })();
    if tarred.is_err() {
        exit_error("Opening or writing the tar file");
    }

    // Clean up the com dir
    if snap.tempdir != "." {
        prnstr(
            "Run \"imodhelp tomosnapshot\" to see privacy practices followed for snapshots",
            "\n",
            false,
        );
        if writeable_dir != "." {
            prnstr(
                &format!(
                    "Snapshot done, placed in {}",
                    join(&writeable_dir, &output_file)
                ),
                "\n",
                false,
            );
        } else {
            prnstr(
                &format!("Snapshot done, placed in {output_file}"),
                "\n",
                false,
            );
        }
        let _ = std::env::set_current_dir("..");
        if std::fs::remove_dir_all(basename(&snap.tempdir.replace('\\', "/"))).is_err() {
            warning(&format!(
                "Failed to remove temporary directory {}",
                snap.tempdir
            ));
        }
    } else {
        prnstr(
            &format!("Snapshot done, placed in {output_file}"),
            "\n",
            false,
        );
        if std::fs::remove_dir_all(basename(&comdir)).is_err() {
            warning(&format!(
                "Failed to remove temporary com file directory {comdir}"
            ));
        }
        remove_warn(uname_file);
        remove_warn(lslrt_file);
        for log in &snap.copylogs {
            remove_warn(log);
        }
        for thumb in &snap.thumbnails {
            remove_warn(thumb);
        }
        for heads in &snap.header_copies {
            remove_warn(heads);
        }
    }

    done(0)
}
