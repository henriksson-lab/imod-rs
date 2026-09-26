//! Shared test helpers.
//!
//! The crate builds a single `imod` binary that dispatches to every translated
//! command (see `src/bin/imod.rs`), so there is no longer a
//! `CARGO_BIN_EXE_<command>` per command.  These helpers give the tests the two
//! invocation forms the launcher supports.
#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;

pub mod golden;

/// A `Command` running `<command>` through the `imod <command> …` subcommand
/// form.  The launcher re-execs itself with `argv[0]` rewritten, so the
/// translated program observes exactly the `argv` it would have as its own
/// executable.
pub fn imod_cmd(command: &str) -> Command {
    let mut assembled = Command::new(env!("CARGO_BIN_EXE_imod"));
    assembled.arg(command);
    // Backend choice is deliberately a child-process concern in integration
    // tests.  A developer's shell selection must not turn an ordinary parity
    // fixture into a different test; tests for a Rust backend set it explicitly
    // on the command they are exercising.
    assembled.env_remove("IMOD_RS_TIFF_BACKEND");
    assembled.env_remove("IMOD_RS_MRC2TIF_ENCODER");
    assembled.env_remove("IMOD_RS_FFT_BACKEND");
    assembled
}

/// Directory holding the per-command symlinks, qualified by process id the way
/// the rest of this suite's fixtures are, so two test binaries never share one.
pub fn command_link_directory() -> PathBuf {
    std::env::temp_dir().join(format!("imod-rs-command-links-{}", std::process::id()))
}

/// Path to a symlink named `<command>` pointing at the `imod` binary — the
/// second invocation form, and the one an IMOD-style install uses.
///
/// Creation tolerates the directory and the link already existing, so repeated
/// calls within a process are safe; `remove_command_links` cleans the tree up.
pub fn imod_link(command: &str) -> PathBuf {
    let binary = PathBuf::from(env!("CARGO_BIN_EXE_imod"));
    let directory = command_link_directory();
    match std::fs::create_dir_all(&directory) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => panic!("create {}: {error}", directory.display()),
    }
    let link = directory.join(command);
    match std::os::unix::fs::symlink(&binary, &link) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => panic!("link {}: {error}", link.display()),
    }
    link
}

/// A `Command` running `<command>` through its symlink.
pub fn imod_link_cmd(command: &str) -> Command {
    let mut assembled = Command::new(imod_link(command));
    assembled.env_remove("IMOD_RS_TIFF_BACKEND");
    assembled.env_remove("IMOD_RS_MRC2TIF_ENCODER");
    assembled.env_remove("IMOD_RS_FFT_BACKEND");
    assembled
}

/// Removes the symlink directory this process created.
pub fn remove_command_links() {
    let _ = std::fs::remove_dir_all(command_link_directory());
}

// ---------------------------------------------------------------------------
// Date/time stamps.
//
// Native IMOD stamps the wall clock into what it writes, so a golden captured
// on one day can only be compared with today's output once the stamp bytes are
// masked — the *date* as well as the time, or the suite fails on every day but
// the capture date (`CLAUDE.md`, "Timestamps").  Everything here masks the
// stamp bytes and nothing else: a mask that absorbs more hides real defects
// (the TIFF `DateTime` month off-by-one of `BUGS.md` §7 was hidden exactly that
// way).  Check a new suite with `/big/henriksson/realbench/tools/dateshift_goldens.sh`,
// which reruns the suites with the clock shifted by 400 days.
// ---------------------------------------------------------------------------

/// What a masked `dd-Mmm-yy  HH:MM:SS` stamp is replaced by (19 bytes, same length).
pub const STAMP_MASK: &[u8; 19] = b"##-###-##  ##:##:##";

const MONTHS: [&[u8; 3]; 12] = [
    b"Jan", b"Feb", b"Mar", b"Apr", b"May", b"Jun", b"Jul", b"Aug", b"Sep", b"Oct", b"Nov", b"Dec",
];

/// Whether `w` (19 bytes) is an MRC label stamp, `dd-Mmm-yy  HH:MM:SS`, as
/// `mrcfiles.c`'s `mrc_head_label` formats it (`" %d-%b-%y  %H:%M:%S"`).
fn is_label_stamp(w: &[u8]) -> bool {
    let d = |k: usize| w[k].is_ascii_digit();
    d(0) && d(1)
        && w[2] == b'-'
        && MONTHS.iter().any(|m| &w[3..6] == &m[..])
        && w[6] == b'-'
        && d(7)
        && d(8)
        && w[9] == b' '
        && w[10] == b' '
        && d(11)
        && d(12)
        && w[13] == b':'
        && d(14)
        && d(15)
        && w[16] == b':'
        && d(17)
        && d(18)
}

/// Replace every `dd-Mmm-yy  HH:MM:SS` label stamp in `bytes` with
/// [`STAMP_MASK`].  For text: stdout of anything that lists MRC labels
/// (`header`, the `-verbose` header dumps), logs, and label text in general.
pub fn mask_label_stamps(bytes: &mut [u8]) {
    let mut i = 0;
    while i + 19 <= bytes.len() {
        if is_label_stamp(&bytes[i..i + 19]) {
            bytes[i..i + 19].copy_from_slice(STAMP_MASK);
            i += 19;
        } else {
            i += 1;
        }
    }
}

/// Whether `bytes` is an MRC file as IMOD writes one (`MAP ` at byte 208).
pub fn is_mrc(bytes: &[u8]) -> bool {
    bytes.len() >= 1024 && &bytes[208..212] == b"MAP "
}

/// Mask the label stamps of an MRC file: only stamps, only inside the label
/// area (bytes 224..1024).  Does nothing to a file that is not MRC.
pub fn mask_mrc_stamps(bytes: &mut [u8]) {
    if is_mrc(bytes) {
        mask_label_stamps(&mut bytes[224..1024]);
    }
}

/// Reconcile, in a copy of `native`, the regions native IMOD writes from
/// uninitialised memory (`BUGS.md` §2), after checking that `ours` holds the
/// defined content there.  Wherever the two differ inside such a region, `ours`
/// must hold the defined value — otherwise this panics — and native's bytes are
/// replaced by ours; everywhere else, including those regions when the two
/// already agree (a header or model copied from its input), nothing changes.
/// So our bytes are always compared: against native where native is
/// meaningful, against the defined value where it is residue.
///
/// - MRC: label slots at and past `nlabl` (`mrc_head_new` never clears
///   `labels`): defined as zero.
/// - Binary model: `Imod.name` past its terminator (`imodNew` `malloc`s and
///   `imodDefault` writes 13 bytes): defined as zero.
/// - Binary model `MINX` chunk: `oscale` and `orot` of an `IrefImage` the
///   source `malloc`s and never sets (`imodjoin.c:181`, `imodtrans.c:93`):
///   defined as the identity, `oscale` (1,1,1) and `orot` (0,0,0).
pub fn reconcile_uninitialised(ours: &[u8], native: &[u8]) -> Vec<u8> {
    fn take(
        ours: &[u8],
        out: &mut [u8],
        range: std::ops::Range<usize>,
        defined: &[u8],
        what: &str,
    ) {
        if range.end > ours.len()
            || range.end > out.len()
            || ours[range.clone()] == out[range.clone()]
        {
            return;
        }
        assert_eq!(
            &ours[range.clone()],
            defined,
            "{what} (bytes {range:?}) differs from native and is not the defined value"
        );
        out[range.clone()].copy_from_slice(&ours[range]);
    }
    let mut out = native.to_vec();
    if is_mrc(ours) && is_mrc(native) {
        let nlabl = i32::from_le_bytes(ours[220..224].try_into().unwrap()).clamp(0, 10) as usize;
        let start = 224 + 80 * nlabl;
        take(
            ours,
            &mut out,
            start..1024,
            &vec![0; 1024 - start],
            "MRC unused label slots",
        );
    } else if ours.len() > 136 && ours.starts_with(b"IMODV1.2") && native.starts_with(b"IMODV1.2") {
        let end = ours[8..136]
            .iter()
            .position(|&b| b == 0)
            .map_or(136, |p| 8 + p + 1);
        take(
            ours,
            &mut out,
            end..136,
            &vec![0; 136 - end],
            "model name tail",
        );
        let one = 1f32.to_be_bytes();
        let unit: Vec<u8> = [one, one, one].concat();
        let mut index = 0;
        while index + 80 <= ours.len() {
            if &ours[index..index + 4] == b"MINX" && out.get(index..index + 4) == Some(&b"MINX"[..])
            {
                // After id and size: oscale, otrans, orot, cscale, ctrans, crot.
                let base = index + 8;
                take(ours, &mut out, base..base + 12, &unit, "MINX oscale");
                take(ours, &mut out, base + 24..base + 36, &[0; 12], "MINX orot");
                index += 80;
            } else {
                index += 1;
            }
        }
    }
    out
}

/// Whether `bytes` starts with a classic or big TIFF header.
pub fn is_tiff(bytes: &[u8]) -> bool {
    bytes.len() >= 16
        && (bytes.starts_with(b"II*\0")
            || bytes.starts_with(b"MM\0*")
            || bytes.starts_with(b"II+\0")
            || bytes.starts_with(b"MM\0+"))
}

/// Mask the value of every TIFF `DateTime` tag (306, `YYYY:MM:DD HH:MM:SS`)
/// found by walking the IFD chain; nothing else in the file is touched.
///
/// The whole value is masked, date included, because a golden's date is its
/// capture date.  This therefore also absorbs native's month off-by-one
/// (`iitif.c:2643`, `BUGS.md` §7): a live native-vs-Rust differential that is
/// meant to see that must compare the month itself.
pub fn mask_tiff_datetime(bytes: &mut [u8]) {
    if !is_tiff(bytes) {
        return;
    }
    let little = bytes.starts_with(b"II");
    let rd = |b: &[u8], at: usize, n: usize| -> u64 {
        let mut v = 0u64;
        for k in 0..n {
            let byte = b[at + if little { n - 1 - k } else { k }] as u64;
            v = (v << 8) | byte;
        }
        v
    };
    let big = match rd(bytes, 2, 2) {
        42 => false,
        43 => true,
        _ => return,
    };
    let (count_len, entry_len, value_len) = if big { (8, 20, 8) } else { (2, 12, 4) };
    let mut ifd = if big {
        rd(bytes, 8, 8)
    } else {
        rd(bytes, 4, 4)
    } as usize;
    let mut seen = 0;
    while ifd != 0 && ifd + count_len <= bytes.len() && seen < 100_000 {
        seen += 1;
        let entries = rd(bytes, ifd, count_len) as usize;
        let first = ifd + count_len;
        if first + entries * entry_len + value_len > bytes.len() {
            return;
        }
        for e in 0..entries {
            let at = first + e * entry_len;
            if rd(bytes, at, 2) != 306 || rd(bytes, at + 2, 2) != 2 {
                continue;
            }
            let count = rd(bytes, at + 4, value_len) as usize;
            let field = at + 4 + value_len;
            let start = if count <= value_len {
                field
            } else {
                rd(bytes, field, value_len) as usize
            };
            // 19 characters and the NUL; mask the characters only.
            let end = (start + count.min(19)).min(bytes.len());
            if start < end {
                bytes[start..end].fill(b'#');
            }
        }
        ifd = rd(bytes, first + entries * entry_len, value_len) as usize;
    }
}

/// Mask the number on a `CreatedDayStamp` line of a com/pcm file (the day
/// count `copytomocoms`/`makecomfile` stamp when they create the file).  Only
/// the value is replaced; the line and everything else stay.
pub fn mask_created_day_stamp(bytes: &mut Vec<u8>) {
    const KEY: &[u8] = b"CreatedDayStamp";
    let mut out = Vec::with_capacity(bytes.len());
    for line in bytes.split_inclusive(|&b| b == b'\n') {
        if line.starts_with(KEY) {
            let rest = &line[KEY.len()..];
            let blanks = rest
                .iter()
                .take_while(|&&b| b == b' ' || b == b'\t')
                .count();
            let digits = rest[blanks..]
                .iter()
                .take_while(|b| b.is_ascii_digit())
                .count();
            if blanks > 0 && digits > 0 {
                out.extend_from_slice(&line[..KEY.len() + blanks]);
                out.push(b'#');
                out.extend_from_slice(&rest[blanks + digits..]);
                continue;
            }
        }
        out.extend_from_slice(line);
    }
    *bytes = out;
}

/// Mask every wall-clock stamp a program output can carry, choosing by
/// content: an MRC file's label stamps (label area only), a TIFF's `DateTime`
/// tags, and otherwise — text such as stdout, logs or com files — every
/// `dd-Mmm-yy  HH:MM:SS` stamp and every `CreatedDayStamp` value.  Returns the
/// masked copy.  Regions native writes from uninitialised memory are not
/// masked: see [`reconcile_uninitialised`].
pub fn mask_stamps(bytes: &[u8]) -> Vec<u8> {
    let mut masked = bytes.to_vec();
    if is_mrc(&masked) {
        mask_mrc_stamps(&mut masked);
    } else if is_tiff(&masked) {
        mask_tiff_datetime(&mut masked);
    } else if !masked.starts_with(b"IMOD") {
        mask_label_stamps(&mut masked);
        mask_created_day_stamp(&mut masked);
    }
    masked
}
