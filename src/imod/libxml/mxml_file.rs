//! Translation of `IMOD/libxml/mxml-file.c`.
//!
//! The C hands its `getc`/`putc` callbacks a `void *` that is a
//! `_mxml_fdbuf_t *`, a `FILE *` or a `const char **` depending on which entry
//! point was called.  Those three are [`MxmlSource`] and [`MxmlSink`] here: the
//! callback set is unchanged, and each callback matches the one arm its own
//! entry point built, exactly as the C casts the `void *` it knows it was
//! given.
#![allow(dead_code)]

use super::*;
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format, c_format_bytes};
use core::any::Any;
use std::io::{Read, Write};

/// Matches C `ENCODE_UTF8` (`mxml-file.c:31`).
pub const ENCODE_UTF8: i32 = 0;
/// Matches C `ENCODE_UTF16BE` (`mxml-file.c:32`).
pub const ENCODE_UTF16BE: i32 = 1;
/// Matches C `ENCODE_UTF16LE` (`mxml-file.c:33`).
pub const ENCODE_UTF16LE: i32 = 2;

/// C `EOF` from `<stdio.h>`.
const EOF: i32 = -1;

/// Matches C `_mxml_fdbuf_t` (`mxml-file.c:47`).
///
/// The C keeps the raw descriptor and calls `read`/`write` on it; the caller of
/// `mxmlLoadFd`/`mxmlSaveFd` still owns that descriptor, so it is duplicated
/// into an owned `File` here — a dup shares the file offset with the original,
/// which is the behaviour the C's plain `read`/`write` on the caller's
/// descriptor has.  `current` and `end` are offsets into `buffer` rather than
/// pointers into it.
pub struct MxmlFdbuf {
    pub fd: std::fs::File,
    pub current: usize,
    pub end: usize,
    pub buffer: [u8; 8192],
}

/// The `void *p` the C hands to a `_mxml_getc_cb_t` (`mxml-file.c:44`).
pub enum MxmlSource<'a> {
    /// `mxmlLoadFd`/`mxmlSAXLoadFd`: a `_mxml_fdbuf_t *`.
    Fd(MxmlFdbuf),
    /// `mxmlLoadFile`/`mxmlSAXLoadFile`: a `FILE *`.
    File(&'a mut ImodFile),
    /// `mxmlLoadString`/`mxmlSAXLoadString`: a `const char **` walking a
    /// NUL-terminated string.  Reading at or past `pos == s.len()` is the C
    /// reading the terminating NUL.
    String { s: &'a [u8], pos: usize },
}

/// The `void *p` the C hands to a `_mxml_putc_cb_t` (`mxml-file.c:45`).
pub enum MxmlSink<'a> {
    /// `mxmlSaveFd`: a `_mxml_fdbuf_t *`.
    Fd(MxmlFdbuf),
    /// `mxmlSaveFile`: a `FILE *`.
    File(&'a mut dyn Write),
    /// `mxmlSaveString`: the C's two-element `char *ptr[2]`, a write cursor and
    /// the end of the buffer.  `ptr` counts past `end` without writing, which
    /// is how `mxmlSaveString` reports the length a short buffer needed.
    String {
        buffer: &'a mut [u8],
        ptr: usize,
        end: usize,
    },
}

/// Matches C `_mxml_getc_cb_t` (`mxml-file.c:44`).
pub type MxmlGetcCb = Option<fn(&mut MxmlSource, &mut i32) -> i32>;
/// Matches C `_mxml_putc_cb_t` (`mxml-file.c:45`).
pub type MxmlPutcCb = Option<fn(i32, &mut MxmlSink) -> i32>;

/// Matches C `MXML_NO_CALLBACK` where a SAX callback is expected
/// (`mxml-file.c:97`).
const MXML_NO_CALLBACK_SAX: MxmlSaxCb = None;

/// Matches C `mxmlLoadFd` (`mxml-file.c:80`).
pub fn mxml_load_fd(
    arena: &mut MxmlArena,
    top: Option<usize>,
    fd: std::os::fd::BorrowedFd,
    cb: MxmlLoadCb,
) -> Option<usize> {
    /*
     * Initialize the file descriptor buffer...
     */

    let mut buf: MxmlSource = MxmlSource::Fd(MxmlFdbuf {
        fd: std::fs::File::from(fd.try_clone_to_owned().ok()?),
        current: 0,
        end: 0,
        buffer: [0; 8192],
    });

    /*
     * Read the XML data...
     */

    mxml_load_data(
        arena,
        top,
        &mut buf,
        cb,
        Some(mxml_fd_getc),
        MXML_NO_CALLBACK_SAX,
        &mut (),
    )
}

/// Matches C `mxmlLoadFile` (`mxml-file.c:117`).
pub fn mxml_load_file(
    arena: &mut MxmlArena,
    top: Option<usize>,
    fp: &mut ImodFile,
    cb: MxmlLoadCb,
) -> Option<usize> {
    /*
     * Read the XML data...
     */

    let mut p: MxmlSource = MxmlSource::File(fp);

    mxml_load_data(
        arena,
        top,
        &mut p,
        cb,
        Some(mxml_file_getc),
        MXML_NO_CALLBACK_SAX,
        &mut (),
    )
}

/// Matches C `mxmlLoadString` (`mxml-file.c:145`).
pub fn mxml_load_string(
    arena: &mut MxmlArena,
    top: Option<usize>,
    s: &[u8],
    cb: MxmlLoadCb,
) -> Option<usize> {
    let mut sp: MxmlSource = MxmlSource::String { s, pos: 0 };

    /*
     * Read the XML data...
     */

    mxml_load_data(
        arena,
        top,
        &mut sp,
        cb,
        Some(mxml_string_getc),
        MXML_NO_CALLBACK_SAX,
        &mut (),
    )
}

/// Matches C `mxmlSaveAllocString` (`mxml-file.c:170`).
pub fn mxml_save_alloc_string(
    arena: &MxmlArena,
    node: Option<usize>,
    cb: MxmlSaveCb,
) -> Option<Vec<u8>> {
    let bytes: i32;
    let mut buffer: [u8; 8192] = [0; 8192];
    let mut s: Vec<u8>;

    /*
     * Write the node to the temporary buffer...
     */

    bytes = mxml_save_string(arena, node, &mut buffer, 8192, cb);

    if bytes <= 0 {
        return None;
    }

    if bytes < (8192 - 1) as i32 {
        /*
         * Node fit inside the buffer, so just duplicate that string and
         * return...
         */

        return Some(buffer[..bytes as usize].to_vec());
    }

    /*
     * Allocate a buffer of the required size and save the node to the
     * new buffer...
     */

    s = vec![0u8; (bytes + 1) as usize];

    mxml_save_string(arena, node, &mut s, bytes + 1, cb);

    /*
     * Return the allocated string...  The C returns a NUL-terminated buffer of
     * `bytes + 1` characters; the owned Vec carries the string itself.
     */

    s.truncate(bytes as usize);
    Some(s)
}

/// Matches C `mxmlSaveFd` (`mxml-file.c:218`).
pub fn mxml_save_fd(
    arena: &MxmlArena,
    node: Option<usize>,
    fd: std::os::fd::BorrowedFd,
    cb: MxmlSaveCb,
) -> i32 {
    let col: i32;

    /*
     * Initialize the file descriptor buffer...
     */

    let Ok(owned) = fd.try_clone_to_owned() else {
        return -1;
    };
    let mut buf: MxmlSink = MxmlSink::Fd(MxmlFdbuf {
        fd: std::fs::File::from(owned),
        current: 0,
        end: 8192,
        buffer: [0; 8192],
    });

    /*
     * Write the node...
     */

    col = mxml_global().with_borrow(|global| {
        mxml_write_node(arena, node, &mut buf, cb, 0, Some(mxml_fd_putc), global)
    });
    if col < 0 {
        return -1;
    }

    if col > 0 && mxml_fd_putc(b'\n' as i32, &mut buf) < 0 {
        return -1;
    }

    /*
     * Flush and return...
     */

    let MxmlSink::Fd(fdbuf) = &mut buf else {
        return -1;
    };
    mxml_fd_write(fdbuf)
}

/// Matches C `mxmlSaveFile` (`mxml-file.c:262`).
pub fn mxml_save_file(
    arena: &MxmlArena,
    node: Option<usize>,
    fp: &mut dyn Write,
    cb: MxmlSaveCb,
) -> i32 {
    let col: i32;

    /*
     * Write the node...
     */

    let mut p: MxmlSink = MxmlSink::File(fp);

    col = mxml_global().with_borrow(|global| {
        mxml_write_node(arena, node, &mut p, cb, 0, Some(mxml_file_putc), global)
    });
    if col < 0 {
        return -1;
    }

    /* The C calls putc('\n', fp) here, which is mxml_file_putc's body. */
    if col > 0 && mxml_file_putc(b'\n' as i32, &mut p) < 0 {
        return -1;
    }

    /*
     * Return 0 (success)...
     */

    0
}

/// Matches C `mxmlSaveString` (`mxml-file.c:299`).
pub fn mxml_save_string(
    arena: &MxmlArena,
    node: Option<usize>,
    buffer: &mut [u8],
    bufsize: i32,
    cb: MxmlSaveCb,
) -> i32 {
    let col: i32;

    /*
     * Write the node...  ptr[0] is the write cursor and ptr[1] the end of the
     * buffer.
     */

    let mut ptr: MxmlSink = MxmlSink::String {
        buffer,
        ptr: 0,
        end: bufsize as usize,
    };

    col = mxml_global().with_borrow(|global| {
        mxml_write_node(arena, node, &mut ptr, cb, 0, Some(mxml_string_putc), global)
    });
    if col < 0 {
        return -1;
    }

    if col > 0 {
        mxml_string_putc(b'\n' as i32, &mut ptr);
    }

    /*
     * Nul-terminate the buffer...
     */

    let MxmlSink::String {
        buffer,
        ptr,
        end: _,
    } = &mut ptr
    else {
        return -1;
    };

    if *ptr >= bufsize as usize {
        buffer[(bufsize - 1) as usize] = 0;
    } else {
        buffer[*ptr] = 0;
    }

    /*
     * Return the number of characters...
     */

    *ptr as i32
}

/// Matches C `mxmlSAXLoadFd` (`mxml-file.c:349`).
pub fn mxml_sax_load_fd(
    arena: &mut MxmlArena,
    top: Option<usize>,
    fd: std::os::fd::BorrowedFd,
    cb: MxmlLoadCb,
    sax_cb: MxmlSaxCb,
    sax_data: &mut dyn Any,
) -> Option<usize> {
    /*
     * Initialize the file descriptor buffer...
     */

    let mut buf: MxmlSource = MxmlSource::Fd(MxmlFdbuf {
        fd: std::fs::File::from(fd.try_clone_to_owned().ok()?),
        current: 0,
        end: 0,
        buffer: [0; 8192],
    });

    /*
     * Read the XML data...
     */

    mxml_load_data(
        arena,
        top,
        &mut buf,
        cb,
        Some(mxml_fd_getc),
        sax_cb,
        sax_data,
    )
}

/// Matches C `mxmlSAXLoadFile` (`mxml-file.c:391`).
pub fn mxml_sax_load_file(
    arena: &mut MxmlArena,
    top: Option<usize>,
    fp: &mut ImodFile,
    cb: MxmlLoadCb,
    sax_cb: MxmlSaxCb,
    sax_data: &mut dyn Any,
) -> Option<usize> {
    /*
     * Read the XML data...
     */

    let mut p: MxmlSource = MxmlSource::File(fp);

    mxml_load_data(
        arena,
        top,
        &mut p,
        cb,
        Some(mxml_file_getc),
        sax_cb,
        sax_data,
    )
}

/// Matches C `mxmlSAXLoadString` (`mxml-file.c:429`).
pub fn mxml_sax_load_string(
    arena: &mut MxmlArena,
    top: Option<usize>,
    s: &[u8],
    cb: MxmlLoadCb,
    sax_cb: MxmlSaxCb,
    sax_data: &mut dyn Any,
) -> Option<usize> {
    let mut sp: MxmlSource = MxmlSource::String { s, pos: 0 };

    /*
     * Read the XML data...
     */

    mxml_load_data(
        arena,
        top,
        &mut sp,
        cb,
        Some(mxml_string_getc),
        sax_cb,
        sax_data,
    )
}

/// Matches C `mxmlSetCustomHandlers` (`mxml-file.c:453`).
pub fn mxml_set_custom_handlers(load: MxmlCustomLoadCb, save: MxmlCustomSaveCb) {
    mxml_global().with_borrow_mut(|global| {
        global.custom_load_cb = load;
        global.custom_save_cb = save;
    });
}

/// Matches C `mxmlSetErrorCallback` (`mxml-file.c:469`).
pub fn mxml_set_error_callback(cb: MxmlErrorCb) {
    mxml_global().with_borrow_mut(|global| {
        global.error_cb = cb;
    });
}

/// Matches C `mxmlSetWrapMargin` (`mxml-file.c:484`).
pub fn mxml_set_wrap_margin(column: i32) {
    mxml_global().with_borrow_mut(|global| {
        global.wrap = column;
    });
}

/// Matches C static `mxml_add_char` (`mxml-file.c:496`).
///
/// The C grows a `realloc`ed buffer and advances a cursor into it; the owned
/// `Vec` is both, so the `bufptr`/`buffer`/`bufsize` triple collapses to one
/// argument and the "unable to expand string buffer" arm cannot be reached.
pub fn mxml_add_char(ch: i32, bufptr: &mut Vec<u8>) -> i32 {
    /*
     * Nul-terminate the buffer as needed...
     */

    if ch < 0x80 {
        /*
         * Single byte ASCII...
         */

        bufptr.push(ch as u8);
    } else if ch < 0x800 {
        /*
         * Two-byte UTF-8...
         */

        bufptr.push((0xc0 | (ch >> 6)) as u8);
        bufptr.push((0x80 | (ch & 0x3f)) as u8);
    } else if ch < 0x10000 {
        /*
         * Three-byte UTF-8...
         */

        bufptr.push((0xe0 | (ch >> 12)) as u8);
        bufptr.push((0x80 | ((ch >> 6) & 0x3f)) as u8);
        bufptr.push((0x80 | (ch & 0x3f)) as u8);
    } else {
        /*
         * Four-byte UTF-8...
         */

        bufptr.push((0xf0 | (ch >> 18)) as u8);
        bufptr.push((0x80 | ((ch >> 12) & 0x3f)) as u8);
        bufptr.push((0x80 | ((ch >> 6) & 0x3f)) as u8);
        bufptr.push((0x80 | (ch & 0x3f)) as u8);
    }

    0
}

/// Matches C static `mxml_fd_getc` (`mxml-file.c:568`).
pub fn mxml_fd_getc(p: &mut MxmlSource, encoding: &mut i32) -> i32 {
    let mut ch: i32;
    let mut temp: i32;

    /*
     * Get the next character...
     */

    let MxmlSource::Fd(buf) = p else {
        return EOF;
    };
    if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
        return EOF;
    }

    ch = buf.buffer[buf.current] as i32;
    buf.current += 1;

    match *encoding {
        ENCODE_UTF8 => {
            /*
             * Got a UTF-8 character; convert UTF-8 to Unicode and return...
             */

            if (ch & 0x80) == 0 {
                /*
                 * ASCII
                 */

                if ch < b' ' as i32
                    && ch != b'\n' as i32
                    && ch != b'\r' as i32
                    && ch != b'\t' as i32
                {
                    mxml_error(
                        c_format(
                            "Bad control character 0x%02x not allowed by XML standard!",
                            &[CArg::Uint(ch as u32 as u64)],
                        )
                        .as_bytes(),
                    );
                    return EOF;
                }

                return ch;
            } else if ch == 0xfe {
                /*
                 * UTF-16 big-endian BOM?
                 */

                if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                ch = buf.buffer[buf.current] as i32;
                buf.current += 1;

                if ch != 0xff {
                    return EOF;
                }

                *encoding = ENCODE_UTF16BE;

                return mxml_fd_getc(p, encoding);
            } else if ch == 0xff {
                /*
                 * UTF-16 little-endian BOM?
                 */

                if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                ch = buf.buffer[buf.current] as i32;
                buf.current += 1;

                if ch != 0xfe {
                    return EOF;
                }

                *encoding = ENCODE_UTF16LE;

                return mxml_fd_getc(p, encoding);
            } else if (ch & 0xe0) == 0xc0 {
                /*
                 * Two-byte value...
                 */

                if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = buf.buffer[buf.current] as i32;
                buf.current += 1;

                if (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = ((ch & 0x1f) << 6) | (temp & 0x3f);

                if ch < 0x80 {
                    mxml_error(
                        c_format(
                            "Invalid UTF-8 sequence for character 0x%04x!",
                            &[CArg::Uint(ch as u32 as u64)],
                        )
                        .as_bytes(),
                    );
                    return EOF;
                }
            } else if (ch & 0xf0) == 0xe0 {
                /*
                 * Three-byte value...
                 */

                if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = buf.buffer[buf.current] as i32;
                buf.current += 1;

                if (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = ((ch & 0x0f) << 6) | (temp & 0x3f);

                if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = buf.buffer[buf.current] as i32;
                buf.current += 1;

                if (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = (ch << 6) | (temp & 0x3f);

                if ch < 0x800 {
                    mxml_error(
                        c_format(
                            "Invalid UTF-8 sequence for character 0x%04x!",
                            &[CArg::Uint(ch as u32 as u64)],
                        )
                        .as_bytes(),
                    );
                    return EOF;
                }

                /*
                 * Ignore (strip) Byte Order Mark (BOM)...
                 */

                if ch == 0xfeff {
                    return mxml_fd_getc(p, encoding);
                }
            } else if (ch & 0xf8) == 0xf0 {
                /*
                 * Four-byte value...
                 */

                if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = buf.buffer[buf.current] as i32;
                buf.current += 1;

                if (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = ((ch & 0x07) << 6) | (temp & 0x3f);

                if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = buf.buffer[buf.current] as i32;
                buf.current += 1;

                if (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = (ch << 6) | (temp & 0x3f);

                if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = buf.buffer[buf.current] as i32;
                buf.current += 1;

                if (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = (ch << 6) | (temp & 0x3f);

                if ch < 0x10000 {
                    mxml_error(
                        c_format(
                            "Invalid UTF-8 sequence for character 0x%04x!",
                            &[CArg::Uint(ch as u32 as u64)],
                        )
                        .as_bytes(),
                    );
                    return EOF;
                }
            } else {
                return EOF;
            }
        }

        ENCODE_UTF16BE => {
            /*
             * Read UTF-16 big-endian char...
             */

            if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
                return EOF;
            }

            temp = buf.buffer[buf.current] as i32;
            buf.current += 1;

            ch = (ch << 8) | temp;

            if ch < b' ' as i32 && ch != b'\n' as i32 && ch != b'\r' as i32 && ch != b'\t' as i32 {
                mxml_error(
                    c_format(
                        "Bad control character 0x%02x not allowed by XML standard!",
                        &[CArg::Uint(ch as u32 as u64)],
                    )
                    .as_bytes(),
                );
                return EOF;
            } else if ch >= 0xd800 && ch <= 0xdbff {
                /*
                 * Multi-word UTF-16 char...
                 */

                let mut lch: i32;

                if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                lch = buf.buffer[buf.current] as i32;
                buf.current += 1;

                if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = buf.buffer[buf.current] as i32;
                buf.current += 1;

                lch = (lch << 8) | temp;

                if !(0xdc00..0xdfff).contains(&lch) {
                    return EOF;
                }

                ch = (((ch & 0x3ff) << 10) | (lch & 0x3ff)) + 0x10000;
            }
        }

        ENCODE_UTF16LE => {
            /*
             * Read UTF-16 little-endian char...
             */

            if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
                return EOF;
            }

            temp = buf.buffer[buf.current] as i32;
            buf.current += 1;

            ch |= temp << 8;

            if ch < b' ' as i32 && ch != b'\n' as i32 && ch != b'\r' as i32 && ch != b'\t' as i32 {
                mxml_error(
                    c_format(
                        "Bad control character 0x%02x not allowed by XML standard!",
                        &[CArg::Uint(ch as u32 as u64)],
                    )
                    .as_bytes(),
                );
                return EOF;
            } else if ch >= 0xd800 && ch <= 0xdbff {
                /*
                 * Multi-word UTF-16 char...
                 */

                let mut lch: i32;

                if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                lch = buf.buffer[buf.current] as i32;
                buf.current += 1;

                if buf.current >= buf.end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = buf.buffer[buf.current] as i32;
                buf.current += 1;

                lch |= temp << 8;

                if !(0xdc00..0xdfff).contains(&lch) {
                    return EOF;
                }

                ch = (((ch & 0x3ff) << 10) | (lch & 0x3ff)) + 0x10000;
            }
        }

        _ => {}
    }

    ch
}

/// Matches C static `mxml_fd_putc` (`mxml-file.c:807`).
pub fn mxml_fd_putc(ch: i32, p: &mut MxmlSink) -> i32 {
    /*
     * Flush the write buffer as needed - note above that "end" still indicates
     * the end of the buffer...
     */

    let MxmlSink::Fd(buf) = p else {
        return -1;
    };
    if buf.current >= buf.end && mxml_fd_write(buf) < 0 {
        return -1;
    }

    buf.buffer[buf.current] = ch as u8;
    buf.current += 1;

    /*
     * Return successfully...
     */

    0
}

/// Matches C static `mxml_fd_read` (`mxml-file.c:838`).
///
/// The C loops over `EAGAIN`/`EINTR`; `std::io::Read` reports the latter as
/// `ErrorKind::Interrupted` and the former as `WouldBlock`, and the loop is
/// the same.
pub fn mxml_fd_read(buf: &mut MxmlFdbuf) -> i32 {
    let bytes: usize;

    /*
     * Read from the file descriptor...
     */

    loop {
        match buf.fd.read(&mut buf.buffer) {
            Ok(n) => {
                bytes = n;
                break;
            }
            Err(e) => {
                if e.kind() != std::io::ErrorKind::WouldBlock
                    && e.kind() != std::io::ErrorKind::Interrupted
                {
                    return -1;
                }
            }
        }
    }

    if bytes == 0 {
        return -1;
    }

    /*
     * Update the pointers and return success...
     */

    buf.current = 0;
    buf.end = bytes;

    0
}

/// Matches C static `mxml_fd_write` (`mxml-file.c:879`).
pub fn mxml_fd_write(buf: &mut MxmlFdbuf) -> i32 {
    let mut bytes: usize;
    let mut ptr: usize;

    /*
     * Return 0 if there is nothing to write...
     */

    if buf.current == 0 {
        return 0;
    }

    /*
     * Loop until we have written everything...
     */

    ptr = 0;
    while ptr < buf.current {
        match buf.fd.write(&buf.buffer[ptr..buf.current]) {
            Ok(n) => bytes = n,
            Err(_) => return -1,
        }
        ptr += bytes;
    }

    /*
     * All done, reset pointers and return success...
     */

    buf.current = 0;

    0
}

/// Matches C static `mxml_file_getc` (`mxml-file.c:920`).
///
/// The C reads through `getc` on a buffered `FILE *`; `ImodFile` is unbuffered,
/// so each character is one `read`.  The bytes are the same.
pub fn mxml_file_getc(p: &mut MxmlSource, encoding: &mut i32) -> i32 {
    let mut ch: i32;
    let mut temp: i32;

    /*
     * Read a character from the file and see if it is EOF or ASCII...
     */

    let MxmlSource::File(fp) = p else {
        return EOF;
    };
    let mut one: [u8; 1] = [0];
    ch = match fp.read(&mut one) {
        Ok(1) => one[0] as i32,
        _ => EOF,
    };

    if ch == EOF {
        return EOF;
    }

    match *encoding {
        ENCODE_UTF8 => {
            /*
             * Got a UTF-8 character; convert UTF-8 to Unicode and return...
             */

            if (ch & 0x80) == 0 {
                /*
                 * ASCII
                 */

                if ch < b' ' as i32
                    && ch != b'\n' as i32
                    && ch != b'\r' as i32
                    && ch != b'\t' as i32
                {
                    mxml_error(
                        c_format(
                            "Bad control character 0x%02x not allowed by XML standard!",
                            &[CArg::Uint(ch as u32 as u64)],
                        )
                        .as_bytes(),
                    );
                    return EOF;
                }

                return ch;
            } else if ch == 0xfe {
                /*
                 * UTF-16 big-endian BOM?
                 */

                ch = match fp.read(&mut one) {
                    Ok(1) => one[0] as i32,
                    _ => EOF,
                };
                if ch != 0xff {
                    return EOF;
                }

                *encoding = ENCODE_UTF16BE;

                return mxml_file_getc(p, encoding);
            } else if ch == 0xff {
                /*
                 * UTF-16 little-endian BOM?
                 */

                ch = match fp.read(&mut one) {
                    Ok(1) => one[0] as i32,
                    _ => EOF,
                };
                if ch != 0xfe {
                    return EOF;
                }

                *encoding = ENCODE_UTF16LE;

                return mxml_file_getc(p, encoding);
            } else if (ch & 0xe0) == 0xc0 {
                /*
                 * Two-byte value...
                 */

                temp = match fp.read(&mut one) {
                    Ok(1) => one[0] as i32,
                    _ => EOF,
                };
                if temp == EOF || (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = ((ch & 0x1f) << 6) | (temp & 0x3f);

                if ch < 0x80 {
                    mxml_error(
                        c_format(
                            "Invalid UTF-8 sequence for character 0x%04x!",
                            &[CArg::Uint(ch as u32 as u64)],
                        )
                        .as_bytes(),
                    );
                    return EOF;
                }
            } else if (ch & 0xf0) == 0xe0 {
                /*
                 * Three-byte value...
                 */

                temp = match fp.read(&mut one) {
                    Ok(1) => one[0] as i32,
                    _ => EOF,
                };
                if temp == EOF || (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = ((ch & 0x0f) << 6) | (temp & 0x3f);

                temp = match fp.read(&mut one) {
                    Ok(1) => one[0] as i32,
                    _ => EOF,
                };
                if temp == EOF || (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = (ch << 6) | (temp & 0x3f);

                if ch < 0x800 {
                    mxml_error(
                        c_format(
                            "Invalid UTF-8 sequence for character 0x%04x!",
                            &[CArg::Uint(ch as u32 as u64)],
                        )
                        .as_bytes(),
                    );
                    return EOF;
                }

                /*
                 * Ignore (strip) Byte Order Mark (BOM)...
                 */

                if ch == 0xfeff {
                    return mxml_file_getc(p, encoding);
                }
            } else if (ch & 0xf8) == 0xf0 {
                /*
                 * Four-byte value...
                 */

                temp = match fp.read(&mut one) {
                    Ok(1) => one[0] as i32,
                    _ => EOF,
                };
                if temp == EOF || (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = ((ch & 0x07) << 6) | (temp & 0x3f);

                temp = match fp.read(&mut one) {
                    Ok(1) => one[0] as i32,
                    _ => EOF,
                };
                if temp == EOF || (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = (ch << 6) | (temp & 0x3f);

                temp = match fp.read(&mut one) {
                    Ok(1) => one[0] as i32,
                    _ => EOF,
                };
                if temp == EOF || (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = (ch << 6) | (temp & 0x3f);

                if ch < 0x10000 {
                    mxml_error(
                        c_format(
                            "Invalid UTF-8 sequence for character 0x%04x!",
                            &[CArg::Uint(ch as u32 as u64)],
                        )
                        .as_bytes(),
                    );
                    return EOF;
                }
            } else {
                return EOF;
            }
        }

        ENCODE_UTF16BE => {
            /*
             * Read UTF-16 big-endian char...
             */

            ch = (ch << 8)
                | match fp.read(&mut one) {
                    Ok(1) => one[0] as i32,
                    _ => EOF,
                };

            if ch < b' ' as i32 && ch != b'\n' as i32 && ch != b'\r' as i32 && ch != b'\t' as i32 {
                mxml_error(
                    c_format(
                        "Bad control character 0x%02x not allowed by XML standard!",
                        &[CArg::Uint(ch as u32 as u64)],
                    )
                    .as_bytes(),
                );
                return EOF;
            } else if ch >= 0xd800 && ch <= 0xdbff {
                /*
                 * Multi-word UTF-16 char...
                 */

                let mut lch: i32 = match fp.read(&mut one) {
                    Ok(1) => one[0] as i32,
                    _ => EOF,
                };
                lch = (lch << 8)
                    | match fp.read(&mut one) {
                        Ok(1) => one[0] as i32,
                        _ => EOF,
                    };

                if !(0xdc00..0xdfff).contains(&lch) {
                    return EOF;
                }

                ch = (((ch & 0x3ff) << 10) | (lch & 0x3ff)) + 0x10000;
            }
        }

        ENCODE_UTF16LE => {
            /*
             * Read UTF-16 little-endian char...
             */

            ch |= match fp.read(&mut one) {
                Ok(1) => one[0] as i32,
                _ => EOF,
            } << 8;

            if ch < b' ' as i32 && ch != b'\n' as i32 && ch != b'\r' as i32 && ch != b'\t' as i32 {
                mxml_error(
                    c_format(
                        "Bad control character 0x%02x not allowed by XML standard!",
                        &[CArg::Uint(ch as u32 as u64)],
                    )
                    .as_bytes(),
                );
                return EOF;
            } else if ch >= 0xd800 && ch <= 0xdbff {
                /*
                 * Multi-word UTF-16 char...
                 */

                let mut lch: i32 = match fp.read(&mut one) {
                    Ok(1) => one[0] as i32,
                    _ => EOF,
                };
                lch |= match fp.read(&mut one) {
                    Ok(1) => one[0] as i32,
                    _ => EOF,
                } << 8;

                if !(0xdc00..0xdfff).contains(&lch) {
                    return EOF;
                }

                ch = (((ch & 0x3ff) << 10) | (lch & 0x3ff)) + 0x10000;
            }
        }

        _ => {}
    }

    ch
}

/// Matches C static `mxml_file_putc` (`mxml-file.c:1102`).
pub fn mxml_file_putc(ch: i32, p: &mut MxmlSink) -> i32 {
    let MxmlSink::File(fp) = p else {
        return -1;
    };
    if fp.write_all(&[ch as u8]).is_err() {
        -1
    } else {
        0
    }
}

/// Matches C static `mxml_get_entity` (`mxml-file.c:1114`).
pub fn mxml_get_entity(
    arena: &MxmlArena,
    parent: Option<usize>,
    p: &mut MxmlSource,
    encoding: &mut i32,
    getc_cb: MxmlGetcCb,
) -> i32 {
    let mut ch: i32;
    let mut entity: [u8; 64] = [0; 64];
    let mut entptr: usize;

    /* `parent ? parent->value.element.name : "null"` in every message below. */
    let pname: &[u8] = match parent {
        Some(parent) => match &arena.node(parent).value {
            MxmlValue::Element(element) => element.name.as_deref().unwrap_or(b"(null)"),
            _ => b"(null)",
        },
        None => b"null",
    };

    entptr = 0;

    loop {
        ch = getc_cb.unwrap()(p, encoding);
        if ch == EOF {
            break;
        }
        if ch > 126 || (!(ch as u8).is_ascii_alphanumeric() && ch != b'#' as i32) {
            break;
        } else if entptr < 64 - 1 {
            entity[entptr] = ch as u8;
            entptr += 1;
        } else {
            let mut msg: Vec<u8> = Vec::new();
            msg.extend_from_slice(b"Entity name too long under parent <");
            msg.extend_from_slice(pname);
            msg.extend_from_slice(b">!");
            mxml_error(&msg);
            break;
        }
    }

    let entity: &[u8] = &entity[..entptr];

    if ch != b';' as i32 {
        let mut msg: Vec<u8> = Vec::new();
        msg.extend_from_slice(b"Character entity \"");
        msg.extend_from_slice(entity);
        msg.extend_from_slice(b"\" not terminated under parent <");
        msg.extend_from_slice(pname);
        msg.extend_from_slice(b">!");
        mxml_error(&msg);
        return EOF;
    }

    if !entity.is_empty() && entity[0] == b'#' {
        if entity.len() > 1 && entity[1] == b'x' {
            /* strtol(entity + 2, NULL, 16) */
            let digits: &[u8] = &entity[2..];
            let mut value: i64 = 0;
            let mut k = 0;
            while k < digits.len() && digits[k].is_ascii_hexdigit() {
                value = value
                    .saturating_mul(16)
                    .saturating_add((digits[k] as char).to_digit(16).unwrap() as i64);
                k += 1;
            }
            ch = value as i32;
        } else {
            /* strtol(entity + 1, NULL, 10) */
            let digits: &[u8] = &entity[1..];
            let mut value: i64 = 0;
            let mut k = 0;
            while k < digits.len() && digits[k].is_ascii_digit() {
                value = value
                    .saturating_mul(10)
                    .saturating_add((digits[k] - b'0') as i64);
                k += 1;
            }
            ch = value as i32;
        }
    } else {
        ch = mxml_entity_get_value(entity);
        if ch < 0 {
            let mut msg: Vec<u8> = Vec::new();
            msg.extend_from_slice(b"Entity name \"");
            msg.extend_from_slice(entity);
            msg.extend_from_slice(b";\" not supported under parent <");
            msg.extend_from_slice(pname);
            msg.extend_from_slice(b">!");
            mxml_error(&msg);
        }
    }

    if ch < b' ' as i32 && ch != b'\n' as i32 && ch != b'\r' as i32 && ch != b'\t' as i32 {
        let mut msg: Vec<u8> = Vec::new();
        msg.extend_from_slice(
            c_format(
                "Bad control character 0x%02x under parent <",
                &[CArg::Uint(ch as u32 as u64)],
            )
            .as_bytes(),
        );
        msg.extend_from_slice(pname);
        msg.extend_from_slice(b"> not allowed by XML standard!");
        mxml_error(&msg);
        return EOF;
    }

    ch
}

/// Matches C static inline `mxml_isspace` (`mxml-file.c:60`).
pub fn mxml_isspace(ch: i32) -> i32 {
    (ch == b' ' as i32 || ch == b'\t' as i32 || ch == b'\r' as i32 || ch == b'\n' as i32) as i32
}

/// Matches C static `mxml_load_data` (`mxml-file.c:1191`).
pub fn mxml_load_data(
    arena: &mut MxmlArena,
    top: Option<usize>,
    p: &mut MxmlSource,
    cb: MxmlLoadCb,
    getc_cb: MxmlGetcCb,
    sax_cb: MxmlSaxCb,
    sax_data: &mut dyn Any,
) -> Option<usize> {
    let mut node: Option<usize>;
    let mut first: Option<usize>;
    let mut parent: Option<usize>;
    let mut ch: i32;
    let mut whitespace: i32;
    let mut buffer: Vec<u8>;
    let mut type_: MxmlType;
    let mut encoding: i32;
    static TYPES: [&[u8]; 6] = [
        b"MXML_ELEMENT",
        b"MXML_INTEGER",
        b"MXML_OPAQUE",
        b"MXML_REAL",
        b"MXML_TEXT",
        b"MXML_CUSTOM",
    ];

    /*
     * Read elements and other nodes from the file...  The C mallocs a 64-byte
     * buffer and reports "Unable to allocate string buffer!" when that fails.
     */

    buffer = Vec::with_capacity(64);
    parent = top;
    first = None;
    whitespace = 0;
    encoding = ENCODE_UTF8;

    if cb.is_some() && parent.is_some() {
        type_ = cb.unwrap()(arena, parent);
    } else if parent.is_some() {
        type_ = MXML_TEXT;
    } else {
        type_ = MXML_IGNORE;
    }

    'error: {
        loop {
            ch = getc_cb.unwrap()(p, &mut encoding);
            if ch == EOF {
                break;
            }

            if (ch == b'<' as i32
                || (mxml_isspace(ch) != 0 && type_ != MXML_OPAQUE && type_ != MXML_CUSTOM))
                && !buffer.is_empty()
            {
                /*
                 * Add a new value node...
                 */

                /*
                 * `bufend` is where the C's `bufptr` ends up: the whole buffer
                 * unless strtol/strtod moved it, and `bufend < buffer.len()`
                 * is the C's `*bufptr` test for a trailing unparsed character.
                 */
                let mut bufend: usize = buffer.len();

                match type_ {
                    MXML_INTEGER => {
                        /* strtol(buffer, &bufptr, 0) */
                        let b: &[u8] = &buffer;
                        let mut k = 0;
                        while k < b.len()
                            && (b[k] == b' '
                                || b[k] == b'\t'
                                || b[k] == b'\n'
                                || b[k] == 0x0b
                                || b[k] == 0x0c
                                || b[k] == b'\r')
                        {
                            k += 1;
                        }
                        let negative = if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
                            let negative = b[k] == b'-';
                            k += 1;
                            negative
                        } else {
                            false
                        };
                        let mut base: u32 = 10;
                        if k + 1 < b.len()
                            && b[k] == b'0'
                            && (b[k + 1] | 32) == b'x'
                            && b.get(k + 2).is_some_and(|c| c.is_ascii_hexdigit())
                        {
                            base = 16;
                            k += 2;
                        } else if k < b.len() && b[k] == b'0' {
                            base = 8;
                        }
                        let start = k;
                        let mut value: i64 = 0;
                        while k < b.len() {
                            let Some(digit) = (b[k] as char).to_digit(base) else {
                                break;
                            };
                            value = value
                                .saturating_mul(base as i64)
                                .saturating_add(digit as i64);
                            k += 1;
                        }
                        if k == start {
                            /* No conversion: strtol leaves endptr at nptr. */
                            k = 0;
                        }
                        bufend = k;
                        node = mxml_new_integer(
                            arena,
                            parent,
                            if negative {
                                value.saturating_neg()
                            } else {
                                value
                            } as i32,
                        );
                    }

                    MXML_OPAQUE => {
                        node = mxml_new_opaque(arena, parent, Some(&buffer));
                    }

                    MXML_REAL => {
                        /*
                         * strtod(buffer, &bufptr): the longest prefix that is a
                         * decimal floating literal, parsed with Rust's
                         * correctly-rounded reader.  The C library also accepts
                         * `inf`, `nan` and C99 hexadecimal floats; those
                         * spellings are not accepted here and read as a bad
                         * real value instead.
                         */
                        let b: &[u8] = &buffer;
                        let mut k = 0;
                        while k < b.len()
                            && (b[k] == b' '
                                || b[k] == b'\t'
                                || b[k] == b'\n'
                                || b[k] == 0x0b
                                || b[k] == 0x0c
                                || b[k] == b'\r')
                        {
                            k += 1;
                        }
                        let numstart = k;
                        if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
                            k += 1;
                        }
                        let digits_before = k;
                        while k < b.len() && b[k].is_ascii_digit() {
                            k += 1;
                        }
                        let mut any_digits = k > digits_before;
                        if k < b.len() && b[k] == b'.' {
                            k += 1;
                            let digits_after = k;
                            while k < b.len() && b[k].is_ascii_digit() {
                                k += 1;
                            }
                            any_digits = any_digits || k > digits_after;
                        }
                        if !any_digits {
                            k = 0;
                        } else {
                            let mantissa_end = k;
                            if k < b.len() && (b[k] | 32) == b'e' {
                                let mut e = k + 1;
                                if e < b.len() && (b[e] == b'+' || b[e] == b'-') {
                                    e += 1;
                                }
                                let expstart = e;
                                while e < b.len() && b[e].is_ascii_digit() {
                                    e += 1;
                                }
                                if e > expstart {
                                    k = e;
                                } else {
                                    k = mantissa_end;
                                }
                            }
                        }
                        let real: f64 = if k == 0 {
                            0.0
                        } else {
                            std::str::from_utf8(&b[numstart..k])
                                .ok()
                                .and_then(|s| s.parse::<f64>().ok())
                                .unwrap_or(0.0)
                        };
                        bufend = k;
                        node = mxml_new_real(arena, parent, real);
                    }

                    MXML_TEXT => {
                        node = mxml_new_text(arena, parent, whitespace, Some(&buffer));
                    }

                    MXML_CUSTOM
                        if mxml_global()
                            .with_borrow(|global| global.custom_load_cb)
                            .is_some() =>
                    {
                        /*
                         * Use the callback to fill in the custom data...
                         */

                        node = mxml_new_custom(arena, parent, None, None);

                        let custom_load_cb =
                            mxml_global().with_borrow(|global| global.custom_load_cb);
                        if custom_load_cb.unwrap()(arena, node.unwrap(), &buffer) != 0 {
                            let mut msg: Vec<u8> = Vec::new();
                            msg.extend_from_slice(b"Bad custom value '");
                            msg.extend_from_slice(&buffer);
                            msg.extend_from_slice(b"' in parent <");
                            msg.extend_from_slice(match parent {
                                Some(parent) => match &arena.node(parent).value {
                                    MxmlValue::Element(element) => {
                                        element.name.as_deref().unwrap_or(b"(null)")
                                    }
                                    _ => b"(null)",
                                },
                                None => b"null",
                            });
                            msg.extend_from_slice(b">!");
                            mxml_error(&msg);
                            mxml_delete(arena, node);
                            node = None;
                        }
                    }

                    _ => {
                        node = None;
                    }
                }

                if bufend < buffer.len() {
                    /*
                     * Bad integer/real number value...
                     */

                    let mut msg: Vec<u8> = Vec::new();
                    msg.extend_from_slice(b"Bad ");
                    msg.extend_from_slice(if type_ == MXML_INTEGER {
                        b"integer".as_slice()
                    } else {
                        b"real".as_slice()
                    });
                    msg.extend_from_slice(b" value '");
                    msg.extend_from_slice(&buffer);
                    msg.extend_from_slice(b"' in parent <");
                    msg.extend_from_slice(match parent {
                        Some(parent) => match &arena.node(parent).value {
                            MxmlValue::Element(element) => {
                                element.name.as_deref().unwrap_or(b"(null)")
                            }
                            _ => b"(null)",
                        },
                        None => b"null",
                    });
                    msg.extend_from_slice(b">!");
                    mxml_error(&msg);
                    break;
                }

                buffer.clear();
                whitespace = (mxml_isspace(ch) != 0 && type_ == MXML_TEXT) as i32;

                if node.is_none() && type_ != MXML_IGNORE {
                    /*
                     * Print error and return...
                     */

                    let mut msg: Vec<u8> = Vec::new();
                    msg.extend_from_slice(b"Unable to add value node of type ");
                    msg.extend_from_slice(TYPES[type_ as usize]);
                    msg.extend_from_slice(b" to parent <");
                    msg.extend_from_slice(match parent {
                        Some(parent) => match &arena.node(parent).value {
                            MxmlValue::Element(element) => {
                                element.name.as_deref().unwrap_or(b"(null)")
                            }
                            _ => b"(null)",
                        },
                        None => b"null",
                    });
                    msg.extend_from_slice(b">!");
                    mxml_error(&msg);
                    break 'error;
                }

                if let Some(sax) = sax_cb {
                    sax(arena, node, MXML_SAX_DATA, sax_data);

                    if mxml_release(arena, node) == 0 {
                        node = None;
                    }
                }

                if first.is_none() && node.is_some() {
                    first = node;
                }
            } else if mxml_isspace(ch) != 0 && type_ == MXML_TEXT {
                whitespace = 1;
            }

            /*
             * Add lone whitespace node if we have an element and existing
             * whitespace...
             */

            if ch == b'<' as i32 && whitespace != 0 && type_ == MXML_TEXT {
                if parent.is_some() {
                    node = mxml_new_text(arena, parent, whitespace, Some(b""));

                    if let Some(sax) = sax_cb {
                        sax(arena, node, MXML_SAX_DATA, sax_data);

                        if mxml_release(arena, node) == 0 {
                            node = None;
                        }
                    }

                    if first.is_none() && node.is_some() {
                        first = node;
                    }
                }

                whitespace = 0;
            }

            if ch == b'<' as i32 {
                /*
                 * Start of open/close tag...
                 */

                buffer.clear();

                loop {
                    ch = getc_cb.unwrap()(p, &mut encoding);
                    if ch == EOF {
                        break;
                    }

                    if mxml_isspace(ch) != 0
                        || ch == b'>' as i32
                        || (ch == b'/' as i32 && !buffer.is_empty())
                    {
                        break;
                    } else if ch == b'<' as i32 {
                        mxml_error(b"Bare < in element!");
                        break 'error;
                    } else if ch == b'&' as i32 {
                        ch = mxml_get_entity(arena, parent, p, &mut encoding, getc_cb);
                        if ch == EOF {
                            break 'error;
                        }

                        if mxml_add_char(ch, &mut buffer) != 0 {
                            break 'error;
                        }
                    } else if ch < b'0' as i32
                        && ch != b'!' as i32
                        && ch != b'-' as i32
                        && ch != b'.' as i32
                        && ch != b'/' as i32
                    {
                        break 'error;
                    } else if mxml_add_char(ch, &mut buffer) != 0 {
                        break 'error;
                    } else if (buffer.len() == 1 && buffer[0] == b'?')
                        || (buffer.len() == 3 && buffer.starts_with(b"!--"))
                        || (buffer.len() == 8 && buffer.starts_with(b"![CDATA["))
                    {
                        break;
                    }
                }

                if buffer == b"!--" {
                    /*
                     * Gather rest of comment...
                     */

                    loop {
                        ch = getc_cb.unwrap()(p, &mut encoding);
                        if ch == EOF {
                            break;
                        }

                        if ch == b'>' as i32
                            && buffer.len() > 4
                            && buffer[buffer.len() - 3] != b'-'
                            && buffer[buffer.len() - 2] == b'-'
                            && buffer[buffer.len() - 1] == b'-'
                        {
                            break;
                        } else if mxml_add_char(ch, &mut buffer) != 0 {
                            break 'error;
                        }
                    }

                    /*
                     * Error out if we didn't get the whole comment...
                     */

                    if ch != b'>' as i32 {
                        /*
                         * Print error and return...
                         */

                        mxml_error(b"Early EOF in comment node!");
                        break 'error;
                    }

                    /*
                     * Otherwise add this as an element under the current parent...
                     */

                    if parent.is_none() && first.is_some() {
                        let mut msg: Vec<u8> = Vec::new();
                        msg.push(b'<');
                        msg.extend_from_slice(&buffer);
                        msg.extend_from_slice(b"> cannot be a second root node after <");
                        msg.extend_from_slice(match &arena.node(first.unwrap()).value {
                            MxmlValue::Element(element) => {
                                element.name.as_deref().unwrap_or(b"(null)")
                            }
                            _ => b"(null)",
                        });
                        msg.push(b'>');
                        mxml_error(&msg);
                        break 'error;
                    }

                    node = mxml_new_element(arena, parent, Some(&buffer));
                    if node.is_none() {
                        /*
                         * Just print error for now...
                         */

                        let mut msg: Vec<u8> = Vec::new();
                        msg.extend_from_slice(b"Unable to add comment node to parent <");
                        msg.extend_from_slice(match parent {
                            Some(parent) => match &arena.node(parent).value {
                                MxmlValue::Element(element) => {
                                    element.name.as_deref().unwrap_or(b"(null)")
                                }
                                _ => b"(null)",
                            },
                            None => b"null",
                        });
                        msg.extend_from_slice(b">!");
                        mxml_error(&msg);
                        break;
                    }

                    if let Some(sax) = sax_cb {
                        sax(arena, node, MXML_SAX_COMMENT, sax_data);

                        if mxml_release(arena, node) == 0 {
                            node = None;
                        }
                    }

                    if node.is_some() && first.is_none() {
                        first = node;
                    }
                } else if buffer == b"![CDATA[" {
                    /*
                     * Gather CDATA section...
                     */

                    loop {
                        ch = getc_cb.unwrap()(p, &mut encoding);
                        if ch == EOF {
                            break;
                        }

                        if ch == b'>' as i32
                            && buffer.len() >= 2
                            && buffer[buffer.len() - 2..] == *b"]]"
                        {
                            break;
                        } else if mxml_add_char(ch, &mut buffer) != 0 {
                            break 'error;
                        }
                    }

                    /*
                     * Error out if we didn't get the whole comment...
                     */

                    if ch != b'>' as i32 {
                        /*
                         * Print error and return...
                         */

                        mxml_error(b"Early EOF in CDATA node!");
                        break 'error;
                    }

                    /*
                     * Otherwise add this as an element under the current parent...
                     */

                    if parent.is_none() && first.is_some() {
                        let mut msg: Vec<u8> = Vec::new();
                        msg.push(b'<');
                        msg.extend_from_slice(&buffer);
                        msg.extend_from_slice(b"> cannot be a second root node after <");
                        msg.extend_from_slice(match &arena.node(first.unwrap()).value {
                            MxmlValue::Element(element) => {
                                element.name.as_deref().unwrap_or(b"(null)")
                            }
                            _ => b"(null)",
                        });
                        msg.push(b'>');
                        mxml_error(&msg);
                        break 'error;
                    }

                    node = mxml_new_element(arena, parent, Some(&buffer));
                    if node.is_none() {
                        /*
                         * Print error and return...
                         */

                        let mut msg: Vec<u8> = Vec::new();
                        msg.extend_from_slice(b"Unable to add CDATA node to parent <");
                        msg.extend_from_slice(match parent {
                            Some(parent) => match &arena.node(parent).value {
                                MxmlValue::Element(element) => {
                                    element.name.as_deref().unwrap_or(b"(null)")
                                }
                                _ => b"(null)",
                            },
                            None => b"null",
                        });
                        msg.extend_from_slice(b">!");
                        mxml_error(&msg);
                        break 'error;
                    }

                    if let Some(sax) = sax_cb {
                        sax(arena, node, MXML_SAX_CDATA, sax_data);

                        if mxml_release(arena, node) == 0 {
                            node = None;
                        }
                    }

                    if node.is_some() && first.is_none() {
                        first = node;
                    }
                } else if !buffer.is_empty() && buffer[0] == b'?' {
                    /*
                     * Gather rest of processing instruction...
                     */

                    loop {
                        ch = getc_cb.unwrap()(p, &mut encoding);
                        if ch == EOF {
                            break;
                        }

                        if ch == b'>' as i32
                            && !buffer.is_empty()
                            && buffer[buffer.len() - 1] == b'?'
                        {
                            break;
                        } else if mxml_add_char(ch, &mut buffer) != 0 {
                            break 'error;
                        }
                    }

                    /*
                     * Error out if we didn't get the whole processing instruction...
                     */

                    if ch != b'>' as i32 {
                        /*
                         * Print error and return...
                         */

                        mxml_error(b"Early EOF in processing instruction node!");
                        break 'error;
                    }

                    /*
                     * Otherwise add this as an element under the current parent...
                     */

                    if parent.is_none() && first.is_some() {
                        let mut msg: Vec<u8> = Vec::new();
                        msg.push(b'<');
                        msg.extend_from_slice(&buffer);
                        msg.extend_from_slice(b"> cannot be a second root node after <");
                        msg.extend_from_slice(match &arena.node(first.unwrap()).value {
                            MxmlValue::Element(element) => {
                                element.name.as_deref().unwrap_or(b"(null)")
                            }
                            _ => b"(null)",
                        });
                        msg.push(b'>');
                        mxml_error(&msg);
                        break 'error;
                    }

                    node = mxml_new_element(arena, parent, Some(&buffer));
                    if node.is_none() {
                        /*
                         * Print error and return...
                         */

                        let mut msg: Vec<u8> = Vec::new();
                        msg.extend_from_slice(
                            b"Unable to add processing instruction node to parent <",
                        );
                        msg.extend_from_slice(match parent {
                            Some(parent) => match &arena.node(parent).value {
                                MxmlValue::Element(element) => {
                                    element.name.as_deref().unwrap_or(b"(null)")
                                }
                                _ => b"(null)",
                            },
                            None => b"null",
                        });
                        msg.extend_from_slice(b">!");
                        mxml_error(&msg);
                        break 'error;
                    }

                    if let Some(sax) = sax_cb {
                        sax(arena, node, MXML_SAX_DIRECTIVE, sax_data);

                        if mxml_release(arena, node) == 0 {
                            node = None;
                        }
                    }

                    if node.is_some() {
                        if first.is_none() {
                            first = node;
                        }

                        if parent.is_none() {
                            /*
                             * Got the XML declaration, so add it as the root node...
                             */

                            parent = node;

                            if cb.is_some() {
                                type_ = cb.unwrap()(arena, parent);
                            } else {
                                type_ = MXML_TEXT;
                            }
                        }
                    }
                } else if !buffer.is_empty() && buffer[0] == b'!' {
                    /*
                     * Gather rest of declaration...
                     */

                    loop {
                        if ch == b'>' as i32 {
                            break;
                        } else {
                            if ch == b'&' as i32 {
                                ch = mxml_get_entity(arena, parent, p, &mut encoding, getc_cb);
                                if ch == EOF {
                                    break 'error;
                                }
                            }

                            if mxml_add_char(ch, &mut buffer) != 0 {
                                break 'error;
                            }
                        }

                        ch = getc_cb.unwrap()(p, &mut encoding);
                        if ch == EOF {
                            break;
                        }
                    }

                    /*
                     * Error out if we didn't get the whole declaration...
                     */

                    if ch != b'>' as i32 {
                        /*
                         * Print error and return...
                         */

                        mxml_error(b"Early EOF in declaration node!");
                        break 'error;
                    }

                    /*
                     * Otherwise add this as an element under the current parent...
                     */

                    if parent.is_none() && first.is_some() {
                        let mut msg: Vec<u8> = Vec::new();
                        msg.push(b'<');
                        msg.extend_from_slice(&buffer);
                        msg.extend_from_slice(b"> cannot be a second root node after <");
                        msg.extend_from_slice(match &arena.node(first.unwrap()).value {
                            MxmlValue::Element(element) => {
                                element.name.as_deref().unwrap_or(b"(null)")
                            }
                            _ => b"(null)",
                        });
                        msg.push(b'>');
                        mxml_error(&msg);
                        break 'error;
                    }

                    node = mxml_new_element(arena, parent, Some(&buffer));
                    if node.is_none() {
                        /*
                         * Print error and return...
                         */

                        let mut msg: Vec<u8> = Vec::new();
                        msg.extend_from_slice(b"Unable to add declaration node to parent <");
                        msg.extend_from_slice(match parent {
                            Some(parent) => match &arena.node(parent).value {
                                MxmlValue::Element(element) => {
                                    element.name.as_deref().unwrap_or(b"(null)")
                                }
                                _ => b"(null)",
                            },
                            None => b"null",
                        });
                        msg.extend_from_slice(b">!");
                        mxml_error(&msg);
                        break 'error;
                    }

                    if let Some(sax) = sax_cb {
                        sax(arena, node, MXML_SAX_DIRECTIVE, sax_data);

                        if mxml_release(arena, node) == 0 {
                            node = None;
                        }
                    }

                    if node.is_some() {
                        if first.is_none() {
                            first = node;
                        }

                        if parent.is_none() {
                            /*
                             * Got the XML declaration, so add it as the root node...
                             */

                            parent = node;

                            if cb.is_some() {
                                type_ = cb.unwrap()(arena, parent);
                            } else {
                                type_ = MXML_TEXT;
                            }
                        }
                    }
                } else if !buffer.is_empty() && buffer[0] == b'/' {
                    /*
                     * Handle close tag...
                     */

                    let matches = match parent {
                        Some(parent) => match &arena.node(parent).value {
                            MxmlValue::Element(element) => {
                                element.name.as_deref() == Some(&buffer[1..])
                            }
                            _ => false,
                        },
                        None => false,
                    };

                    if !matches {
                        /*
                         * Close tag doesn't match tree; print an error for now...
                         */

                        let mut msg: Vec<u8> = Vec::new();
                        msg.extend_from_slice(b"Mismatched close tag <");
                        msg.extend_from_slice(&buffer);
                        msg.extend_from_slice(b"> under parent <");
                        msg.extend_from_slice(match parent {
                            Some(parent) => match &arena.node(parent).value {
                                MxmlValue::Element(element) => {
                                    element.name.as_deref().unwrap_or(b"(null)")
                                }
                                _ => b"(null)",
                            },
                            None => b"(null)",
                        });
                        msg.extend_from_slice(b">!");
                        mxml_error(&msg);
                        break 'error;
                    }

                    /*
                     * Keep reading until we see >...
                     */

                    while ch != b'>' as i32 && ch != EOF {
                        ch = getc_cb.unwrap()(p, &mut encoding);
                    }

                    node = parent;
                    parent = arena.node(parent.unwrap()).parent;

                    if let Some(sax) = sax_cb {
                        sax(arena, node, MXML_SAX_ELEMENT_CLOSE, sax_data);

                        if mxml_release(arena, node) == 0 && first == node {
                            first = None;
                        }
                    }

                    /*
                     * Ascend into the parent and set the value type as needed...
                     */

                    if cb.is_some() && parent.is_some() {
                        type_ = cb.unwrap()(arena, parent);
                    }
                } else {
                    /*
                     * Handle open tag...
                     */

                    if parent.is_none() && first.is_some() {
                        let mut msg: Vec<u8> = Vec::new();
                        msg.push(b'<');
                        msg.extend_from_slice(&buffer);
                        msg.extend_from_slice(b"> cannot be a second root node after <");
                        msg.extend_from_slice(match &arena.node(first.unwrap()).value {
                            MxmlValue::Element(element) => {
                                element.name.as_deref().unwrap_or(b"(null)")
                            }
                            _ => b"(null)",
                        });
                        msg.push(b'>');
                        mxml_error(&msg);
                        break 'error;
                    }

                    node = mxml_new_element(arena, parent, Some(&buffer));
                    if node.is_none() {
                        /*
                         * Just print error for now...
                         */

                        let mut msg: Vec<u8> = Vec::new();
                        msg.extend_from_slice(b"Unable to add element node to parent <");
                        msg.extend_from_slice(match parent {
                            Some(parent) => match &arena.node(parent).value {
                                MxmlValue::Element(element) => {
                                    element.name.as_deref().unwrap_or(b"(null)")
                                }
                                _ => b"(null)",
                            },
                            None => b"null",
                        });
                        msg.extend_from_slice(b">!");
                        mxml_error(&msg);
                        break 'error;
                    }

                    if mxml_isspace(ch) != 0 {
                        ch = mxml_parse_element(arena, node.unwrap(), p, &mut encoding, getc_cb);
                        if ch == EOF {
                            break 'error;
                        }
                    } else if ch == b'/' as i32 {
                        ch = getc_cb.unwrap()(p, &mut encoding);
                        if ch != b'>' as i32 {
                            let mut msg: Vec<u8> = Vec::new();
                            msg.extend_from_slice(
                                c_format(
                                    "Expected > but got '%c' instead for element <",
                                    &[CArg::Chr(ch as u8)],
                                )
                                .as_bytes(),
                            );
                            msg.extend_from_slice(&buffer);
                            msg.extend_from_slice(b"/>!");
                            mxml_error(&msg);
                            mxml_delete(arena, node);
                            break 'error;
                        }

                        ch = b'/' as i32;
                    }

                    if let Some(sax) = sax_cb {
                        sax(arena, node, MXML_SAX_ELEMENT_OPEN, sax_data);
                    }

                    if first.is_none() {
                        first = node;
                    }

                    if ch == EOF {
                        break;
                    }

                    if ch != b'/' as i32 {
                        /*
                         * Descend into this node, setting the value type as needed...
                         */

                        parent = node;

                        if cb.is_some() && parent.is_some() {
                            type_ = cb.unwrap()(arena, parent);
                        } else {
                            type_ = MXML_TEXT;
                        }
                    } else if let Some(sax) = sax_cb {
                        sax(arena, node, MXML_SAX_ELEMENT_CLOSE, sax_data);

                        if mxml_release(arena, node) == 0 && first == node {
                            first = None;
                        }
                    }
                }

                buffer.clear();
            } else if ch == b'&' as i32 {
                /*
                 * Add character entity to current buffer...
                 */

                ch = mxml_get_entity(arena, parent, p, &mut encoding, getc_cb);
                if ch == EOF {
                    break 'error;
                }

                if mxml_add_char(ch, &mut buffer) != 0 {
                    break 'error;
                }
            } else if type_ == MXML_OPAQUE || type_ == MXML_CUSTOM || mxml_isspace(ch) == 0 {
                /*
                 * Add character to current buffer...
                 */

                if mxml_add_char(ch, &mut buffer) != 0 {
                    break 'error;
                }
            }
        }

        /*
         * Free the string buffer - we don't need it anymore...
         */

        drop(buffer);

        /*
         * Find the top element and return it...
         */

        if parent.is_some() {
            node = parent;

            while parent != top && arena.node(parent.unwrap()).parent.is_some() {
                parent = arena.node(parent.unwrap()).parent;
            }

            if node != parent {
                let mut msg: Vec<u8> = Vec::new();
                msg.extend_from_slice(b"Missing close tag </");
                msg.extend_from_slice(match &arena.node(node.unwrap()).value {
                    MxmlValue::Element(element) => element.name.as_deref().unwrap_or(b"(null)"),
                    _ => b"(null)",
                });
                msg.extend_from_slice(b"> under parent <");
                msg.extend_from_slice(match arena.node(node.unwrap()).parent {
                    Some(parent) => match &arena.node(parent).value {
                        MxmlValue::Element(element) => element.name.as_deref().unwrap_or(b"(null)"),
                        _ => b"(null)",
                    },
                    None => b"(null)",
                });
                msg.extend_from_slice(b">!");
                mxml_error(&msg);

                mxml_delete(arena, first);

                return None;
            }
        }

        if parent.is_some() {
            return parent;
        } else {
            return first;
        }
    }

    /*
     * Common error return...
     */

    mxml_delete(arena, first);

    None
}

/// Matches C static `mxml_parse_element` (`mxml-file.c:1758`).
pub fn mxml_parse_element(
    arena: &mut MxmlArena,
    node: usize,
    p: &mut MxmlSource,
    encoding: &mut i32,
    getc_cb: MxmlGetcCb,
) -> i32 {
    let mut ch: i32;
    let mut quote: i32;
    let mut name: Vec<u8>;
    let mut value: Vec<u8>;

    /*
     * Initialize the name and value buffers...  The C mallocs 64 bytes for
     * each and reports "Unable to allocate memory for name!"/"...value!" when
     * that fails.
     */

    name = Vec::with_capacity(64);
    value = Vec::with_capacity(64);

    /*
     * Loop until we hit a >, /, ?, or EOF...
     */

    'error: {
        loop {
            ch = getc_cb.unwrap()(p, encoding);
            if ch == EOF {
                break;
            }

            /*
             * Skip leading whitespace...
             */

            if mxml_isspace(ch) != 0 {
                continue;
            }

            /*
             * Stop at /, ?, or >...
             */

            if ch == b'/' as i32 || ch == b'?' as i32 {
                /*
                 * Grab the > character and print an error if it isn't there...
                 */

                quote = getc_cb.unwrap()(p, encoding);

                if quote != b'>' as i32 {
                    let mut msg: Vec<u8> = Vec::new();
                    msg.extend_from_slice(
                        c_format(
                            "Expected '>' after '%c' for element ",
                            &[CArg::Chr(ch as u8)],
                        )
                        .as_bytes(),
                    );
                    msg.extend_from_slice(match &arena.node(node).value {
                        MxmlValue::Element(element) => element.name.as_deref().unwrap_or(b"(null)"),
                        _ => b"(null)",
                    });
                    msg.extend_from_slice(
                        c_format(", but got '%c'!", &[CArg::Chr(quote as u8)]).as_bytes(),
                    );
                    mxml_error(&msg);
                    break 'error;
                }

                break;
            } else if ch == b'<' as i32 {
                let mut msg: Vec<u8> = Vec::new();
                msg.extend_from_slice(b"Bare < in element ");
                msg.extend_from_slice(match &arena.node(node).value {
                    MxmlValue::Element(element) => element.name.as_deref().unwrap_or(b"(null)"),
                    _ => b"(null)",
                });
                msg.push(b'!');
                mxml_error(&msg);
                break 'error;
            } else if ch == b'>' as i32 {
                break;
            }

            /*
             * Read the attribute name...
             */

            name.clear();
            name.push(ch as u8);

            if ch == b'\"' as i32 || ch == b'\'' as i32 {
                /*
                 * Name is in quotes, so get a quoted string...
                 */

                quote = ch;

                loop {
                    ch = getc_cb.unwrap()(p, encoding);
                    if ch == EOF {
                        break;
                    }

                    if ch == b'&' as i32 {
                        ch = mxml_get_entity(arena, Some(node), p, encoding, getc_cb);
                        if ch == EOF {
                            break 'error;
                        }
                    }

                    if mxml_add_char(ch, &mut name) != 0 {
                        break 'error;
                    }

                    if ch == quote {
                        break;
                    }
                }
            } else {
                /*
                 * Grab an normal, non-quoted name...
                 */

                loop {
                    ch = getc_cb.unwrap()(p, encoding);
                    if ch == EOF {
                        break;
                    }

                    if mxml_isspace(ch) != 0
                        || ch == b'=' as i32
                        || ch == b'/' as i32
                        || ch == b'>' as i32
                        || ch == b'?' as i32
                    {
                        break;
                    } else {
                        if ch == b'&' as i32 {
                            ch = mxml_get_entity(arena, Some(node), p, encoding, getc_cb);
                            if ch == EOF {
                                break 'error;
                            }
                        }

                        if mxml_add_char(ch, &mut name) != 0 {
                            break 'error;
                        }
                    }
                }
            }

            if mxml_element_get_attr(arena, Some(node), Some(&name)).is_some() {
                break 'error;
            }

            while ch != EOF && mxml_isspace(ch) != 0 {
                ch = getc_cb.unwrap()(p, encoding);
            }

            if ch == b'=' as i32 {
                /*
                 * Read the attribute value...
                 */

                loop {
                    ch = getc_cb.unwrap()(p, encoding);
                    if ch == EOF || mxml_isspace(ch) == 0 {
                        break;
                    }
                }

                if ch == EOF {
                    let mut msg: Vec<u8> = Vec::new();
                    msg.extend_from_slice(b"Missing value for attribute '");
                    msg.extend_from_slice(&name);
                    msg.extend_from_slice(b"' in element ");
                    msg.extend_from_slice(match &arena.node(node).value {
                        MxmlValue::Element(element) => element.name.as_deref().unwrap_or(b"(null)"),
                        _ => b"(null)",
                    });
                    msg.push(b'!');
                    mxml_error(&msg);
                    break 'error;
                }

                if ch == b'\'' as i32 || ch == b'\"' as i32 {
                    /*
                     * Read quoted value...
                     */

                    quote = ch;
                    value.clear();

                    loop {
                        ch = getc_cb.unwrap()(p, encoding);
                        if ch == EOF {
                            break;
                        }

                        if ch == quote {
                            break;
                        } else {
                            if ch == b'&' as i32 {
                                ch = mxml_get_entity(arena, Some(node), p, encoding, getc_cb);
                                if ch == EOF {
                                    break 'error;
                                }
                            }

                            if mxml_add_char(ch, &mut value) != 0 {
                                break 'error;
                            }
                        }
                    }
                } else {
                    /*
                     * Read unquoted value...
                     */

                    value.clear();
                    value.push(ch as u8);

                    loop {
                        ch = getc_cb.unwrap()(p, encoding);
                        if ch == EOF {
                            break;
                        }

                        if mxml_isspace(ch) != 0
                            || ch == b'=' as i32
                            || ch == b'/' as i32
                            || ch == b'>' as i32
                        {
                            break;
                        } else {
                            if ch == b'&' as i32 {
                                ch = mxml_get_entity(arena, Some(node), p, encoding, getc_cb);
                                if ch == EOF {
                                    break 'error;
                                }
                            }

                            if mxml_add_char(ch, &mut value) != 0 {
                                break 'error;
                            }
                        }
                    }
                }

                /*
                 * Set the attribute with the given string value...
                 */

                mxml_element_set_attr(arena, Some(node), Some(&name), Some(&value));
            } else {
                let mut msg: Vec<u8> = Vec::new();
                msg.extend_from_slice(b"Missing value for attribute '");
                msg.extend_from_slice(&name);
                msg.extend_from_slice(b"' in element ");
                msg.extend_from_slice(match &arena.node(node).value {
                    MxmlValue::Element(element) => element.name.as_deref().unwrap_or(b"(null)"),
                    _ => b"(null)",
                });
                msg.push(b'!');
                mxml_error(&msg);
                break 'error;
            }

            /*
             * Check the end character...
             */

            if ch == b'/' as i32 || ch == b'?' as i32 {
                /*
                 * Grab the > character and print an error if it isn't there...
                 */

                quote = getc_cb.unwrap()(p, encoding);

                if quote != b'>' as i32 {
                    let mut msg: Vec<u8> = Vec::new();
                    msg.extend_from_slice(
                        c_format(
                            "Expected '>' after '%c' for element ",
                            &[CArg::Chr(ch as u8)],
                        )
                        .as_bytes(),
                    );
                    msg.extend_from_slice(match &arena.node(node).value {
                        MxmlValue::Element(element) => element.name.as_deref().unwrap_or(b"(null)"),
                        _ => b"(null)",
                    });
                    msg.extend_from_slice(
                        c_format(", but got '%c'!", &[CArg::Chr(quote as u8)]).as_bytes(),
                    );
                    mxml_error(&msg);
                    ch = EOF;
                }

                break;
            } else if ch == b'>' as i32 {
                break;
            }
        }

        /*
         * Free the name and value buffers and return...
         */

        return ch;
    }

    /*
     * Common error return point...
     */

    EOF
}

/// Matches C static `mxml_string_getc` (`mxml-file.c:2049`).
///
/// The C walks a NUL-terminated string; reading at or past the end of the byte
/// slice is the C reading that terminating NUL, so every dereference below is
/// `s.get(pos).unwrap_or(&0)`.
pub fn mxml_string_getc(p: &mut MxmlSource, encoding: &mut i32) -> i32 {
    let mut ch: i32;

    let MxmlSource::String { s, pos } = p else {
        return EOF;
    };

    ch = (*s.get(*pos).unwrap_or(&0) as i32) & 255;
    if ch != 0 || *encoding == ENCODE_UTF16LE {
        *pos += 1;

        match *encoding {
            ENCODE_UTF8 => {
                /*
                 * Got a UTF-8 character; convert UTF-8 to Unicode and return...
                 */

                if (ch & 0x80) == 0 {
                    /*
                     * ASCII
                     */

                    if ch < b' ' as i32
                        && ch != b'\n' as i32
                        && ch != b'\r' as i32
                        && ch != b'\t' as i32
                    {
                        mxml_error(
                            c_format(
                                "Bad control character 0x%02x not allowed by XML standard!",
                                &[CArg::Uint(ch as u32 as u64)],
                            )
                            .as_bytes(),
                        );
                        return EOF;
                    }

                    return ch;
                } else if ch == 0xfe {
                    /*
                     * UTF-16 big-endian BOM?
                     */

                    if ((*s.get(*pos).unwrap_or(&0) as i32) & 255) != 0xff {
                        return EOF;
                    }

                    *encoding = ENCODE_UTF16BE;
                    *pos += 1;

                    return mxml_string_getc(p, encoding);
                } else if ch == 0xff {
                    /*
                     * UTF-16 little-endian BOM?
                     */

                    if ((*s.get(*pos).unwrap_or(&0) as i32) & 255) != 0xfe {
                        return EOF;
                    }

                    *encoding = ENCODE_UTF16LE;
                    *pos += 1;

                    return mxml_string_getc(p, encoding);
                } else if (ch & 0xe0) == 0xc0 {
                    /*
                     * Two-byte value...
                     */

                    if ((*s.get(*pos).unwrap_or(&0) as i32) & 0xc0) != 0x80 {
                        return EOF;
                    }

                    ch = ((ch & 0x1f) << 6) | ((*s.get(*pos).unwrap_or(&0) as i32) & 0x3f);

                    *pos += 1;

                    if ch < 0x80 {
                        mxml_error(
                            c_format(
                                "Invalid UTF-8 sequence for character 0x%04x!",
                                &[CArg::Uint(ch as u32 as u64)],
                            )
                            .as_bytes(),
                        );
                        return EOF;
                    }

                    return ch;
                } else if (ch & 0xf0) == 0xe0 {
                    /*
                     * Three-byte value...
                     */

                    if ((*s.get(*pos).unwrap_or(&0) as i32) & 0xc0) != 0x80
                        || ((*s.get(*pos + 1).unwrap_or(&0) as i32) & 0xc0) != 0x80
                    {
                        return EOF;
                    }

                    ch = ((((ch & 0x0f) << 6) | ((*s.get(*pos).unwrap_or(&0) as i32) & 0x3f)) << 6)
                        | ((*s.get(*pos + 1).unwrap_or(&0) as i32) & 0x3f);

                    *pos += 2;

                    if ch < 0x800 {
                        mxml_error(
                            c_format(
                                "Invalid UTF-8 sequence for character 0x%04x!",
                                &[CArg::Uint(ch as u32 as u64)],
                            )
                            .as_bytes(),
                        );
                        return EOF;
                    }

                    /*
                     * Ignore (strip) Byte Order Mark (BOM)...
                     */

                    if ch == 0xfeff {
                        return mxml_string_getc(p, encoding);
                    }

                    return ch;
                } else if (ch & 0xf8) == 0xf0 {
                    /*
                     * Four-byte value...
                     */

                    if ((*s.get(*pos).unwrap_or(&0) as i32) & 0xc0) != 0x80
                        || ((*s.get(*pos + 1).unwrap_or(&0) as i32) & 0xc0) != 0x80
                        || ((*s.get(*pos + 2).unwrap_or(&0) as i32) & 0xc0) != 0x80
                    {
                        return EOF;
                    }

                    ch = ((((((ch & 0x07) << 6) | ((*s.get(*pos).unwrap_or(&0) as i32) & 0x3f))
                        << 6)
                        | ((*s.get(*pos + 1).unwrap_or(&0) as i32) & 0x3f))
                        << 6)
                        | ((*s.get(*pos + 2).unwrap_or(&0) as i32) & 0x3f);

                    *pos += 3;

                    if ch < 0x10000 {
                        mxml_error(
                            c_format(
                                "Invalid UTF-8 sequence for character 0x%04x!",
                                &[CArg::Uint(ch as u32 as u64)],
                            )
                            .as_bytes(),
                        );
                        return EOF;
                    }

                    return ch;
                } else {
                    return EOF;
                }
            }

            ENCODE_UTF16BE => {
                /*
                 * Read UTF-16 big-endian char...
                 */

                ch = (ch << 8) | ((*s.get(*pos).unwrap_or(&0) as i32) & 255);
                *pos += 1;

                if ch < b' ' as i32
                    && ch != b'\n' as i32
                    && ch != b'\r' as i32
                    && ch != b'\t' as i32
                {
                    mxml_error(
                        c_format(
                            "Bad control character 0x%02x not allowed by XML standard!",
                            &[CArg::Uint(ch as u32 as u64)],
                        )
                        .as_bytes(),
                    );
                    return EOF;
                } else if ch >= 0xd800 && ch <= 0xdbff {
                    /*
                     * Multi-word UTF-16 char...
                     */

                    let lch: i32;

                    if *s.get(*pos).unwrap_or(&0) == 0 {
                        return EOF;
                    }

                    lch = (((*s.get(*pos).unwrap_or(&0) as i32) & 255) << 8)
                        | ((*s.get(*pos + 1).unwrap_or(&0) as i32) & 255);
                    *pos += 2;

                    if !(0xdc00..0xdfff).contains(&lch) {
                        return EOF;
                    }

                    ch = (((ch & 0x3ff) << 10) | (lch & 0x3ff)) + 0x10000;
                }

                return ch;
            }

            ENCODE_UTF16LE => {
                /*
                 * Read UTF-16 little-endian char...
                 */

                ch |= ((*s.get(*pos).unwrap_or(&0) as i32) & 255) << 8;

                if ch == 0 {
                    *pos -= 1;
                    return EOF;
                }

                *pos += 1;

                if ch < b' ' as i32
                    && ch != b'\n' as i32
                    && ch != b'\r' as i32
                    && ch != b'\t' as i32
                {
                    mxml_error(
                        c_format(
                            "Bad control character 0x%02x not allowed by XML standard!",
                            &[CArg::Uint(ch as u32 as u64)],
                        )
                        .as_bytes(),
                    );
                    return EOF;
                } else if ch >= 0xd800 && ch <= 0xdbff {
                    /*
                     * Multi-word UTF-16 char...
                     */

                    let lch: i32;

                    if *s.get(*pos + 1).unwrap_or(&0) == 0 {
                        return EOF;
                    }

                    lch = (((*s.get(*pos + 1).unwrap_or(&0) as i32) & 255) << 8)
                        | ((*s.get(*pos).unwrap_or(&0) as i32) & 255);
                    *pos += 2;

                    if !(0xdc00..0xdfff).contains(&lch) {
                        return EOF;
                    }

                    ch = (((ch & 0x3ff) << 10) | (lch & 0x3ff)) + 0x10000;
                }

                return ch;
            }

            _ => {}
        }
    }

    EOF
}

/// Matches C static `mxml_string_putc` (`mxml-file.c:2258`).
pub fn mxml_string_putc(ch: i32, p: &mut MxmlSink) -> i32 {
    let MxmlSink::String { buffer, ptr, end } = p else {
        return -1;
    };

    if *ptr < *end {
        buffer[*ptr] = ch as u8;
    }

    *ptr += 1;

    0
}

/// Matches C static `mxml_write_name` (`mxml-file.c:2280`).
///
/// The C indexes a `char` array, so `mxmlEntityGetName` is given the *signed*
/// character value and never matches a byte with the high bit set; the cast
/// through `i8` below keeps that.
pub fn mxml_write_name(s: &[u8], p: &mut MxmlSink, putc_cb: MxmlPutcCb) -> i32 {
    let quote: u8;
    let mut i: usize = 0;

    if !s.is_empty() && (s[0] == b'\"' || s[0] == b'\'') {
        /*
         * Write a quoted name string...
         */

        if putc_cb.unwrap()(s[0] as i8 as i32, p) < 0 {
            return -1;
        }

        quote = s[0];
        i += 1;

        while i < s.len() && s[i] != quote {
            /*
             * Escape special characters...
             */

            let name = mxml_entity_get_name(s[i] as i8 as i32);
            if let Some(name) = name {
                if putc_cb.unwrap()(b'&' as i32, p) < 0 {
                    return -1;
                }

                for c in name {
                    if putc_cb.unwrap()(*c as i8 as i32, p) < 0 {
                        return -1;
                    }
                }

                if putc_cb.unwrap()(b';' as i32, p) < 0 {
                    return -1;
                }
            } else if putc_cb.unwrap()(s[i] as i8 as i32, p) < 0 {
                return -1;
            }

            i += 1;
        }

        /*
         * Write the end quote...
         */

        if putc_cb.unwrap()(quote as i8 as i32, p) < 0 {
            return -1;
        }
    } else {
        /*
         * Write a non-quoted name string...
         */

        while i < s.len() {
            if putc_cb.unwrap()(s[i] as i8 as i32, p) < 0 {
                return -1;
            }

            i += 1;
        }
    }

    0
}

/// Matches C static `mxml_write_string` (`mxml-file.c:2618`).
pub fn mxml_write_string(s: &[u8], p: &mut MxmlSink, putc_cb: MxmlPutcCb) -> i32 {
    let mut i: usize = 0;

    while i < s.len() {
        /*
         * Escape special characters...
         */

        let name = mxml_entity_get_name(s[i] as i8 as i32);
        if let Some(name) = name {
            if putc_cb.unwrap()(b'&' as i32, p) < 0 {
                return -1;
            }

            for c in name {
                if putc_cb.unwrap()(*c as i8 as i32, p) < 0 {
                    return -1;
                }
            }

            if putc_cb.unwrap()(b';' as i32, p) < 0 {
                return -1;
            }
        } else if putc_cb.unwrap()(s[i] as i8 as i32, p) < 0 {
            return -1;
        }

        i += 1;
    }

    0
}

/// Matches C static `mxml_write_ws` (`mxml-file.c:2660`).
pub fn mxml_write_ws(
    arena: &MxmlArena,
    node: usize,
    p: &mut MxmlSink,
    cb: MxmlSaveCb,
    ws: i32,
    mut col: i32,
    putc_cb: MxmlPutcCb,
) -> i32 {
    if let Some(f) = cb {
        let s = f(arena, node, ws);
        if let Some(s) = s {
            for c in &s {
                if putc_cb.unwrap()(*c as i8 as i32, p) < 0 {
                    return -1;
                } else if *c == b'\n' {
                    col = 0;
                } else if *c == b'\t' {
                    col += MXML_TAB;
                    col -= col % MXML_TAB;
                } else {
                    col += 1;
                }
            }
        }
    }

    col
}

/// Matches C static `mxml_write_node` (`mxml-file.c:2352`).
pub fn mxml_write_node(
    arena: &MxmlArena,
    node: Option<usize>,
    p: &mut MxmlSink,
    cb: MxmlSaveCb,
    mut col: i32,
    putc_cb: MxmlPutcCb,
    global: &MxmlGlobal,
) -> i32 {
    let mut current: Option<usize>;
    let mut next: Option<usize>;
    let mut i: i32;
    let mut width: i32;
    let mut attr: usize;

    current = node;
    while let Some(cur) = current {
        /*
         * Print the node value...
         */

        match arena.node(cur).type_ {
            MXML_ELEMENT => {
                col = mxml_write_ws(arena, cur, p, cb, MXML_WS_BEFORE_OPEN, col, putc_cb);

                if putc_cb.unwrap()(b'<' as i32, p) < 0 {
                    return -1;
                }

                let MxmlValue::Element(element) = &arena.node(cur).value else {
                    return -1;
                };
                let name: &[u8] = element.name.as_deref().unwrap_or(b"");

                if (!name.is_empty() && name[0] == b'?')
                    || name.starts_with(b"!--")
                    || name.starts_with(b"![CDATA[")
                {
                    /*
                     * Comments, CDATA, and processing instructions do not
                     * use character entities, but XML declarations are
                     * written as-is...
                     */

                    for c in name {
                        if putc_cb.unwrap()(*c as i8 as i32, p) < 0 {
                            return -1;
                        }
                    }
                } else if mxml_write_name(name, p, putc_cb) < 0 {
                    return -1;
                }

                col += name.len() as i32 + 1;

                for attribute in &element.attrs {
                    width = attribute.name.len() as i32;

                    if let Some(value) = attribute.value.as_deref() {
                        width += value.len() as i32 + 3;
                    }

                    if global.wrap > 0 && (col + width) > global.wrap {
                        if putc_cb.unwrap()(b'\n' as i32, p) < 0 {
                            return -1;
                        }

                        col = 0;
                    } else {
                        if putc_cb.unwrap()(b' ' as i32, p) < 0 {
                            return -1;
                        }

                        col += 1;
                    }

                    if mxml_write_name(&attribute.name, p, putc_cb) < 0 {
                        return -1;
                    }

                    if let Some(value) = attribute.value.as_deref() {
                        if putc_cb.unwrap()(b'=' as i32, p) < 0 {
                            return -1;
                        }
                        if putc_cb.unwrap()(b'\"' as i32, p) < 0 {
                            return -1;
                        }
                        if mxml_write_string(value, p, putc_cb) < 0 {
                            return -1;
                        }
                        if putc_cb.unwrap()(b'\"' as i32, p) < 0 {
                            return -1;
                        }
                    }

                    col += width;
                }

                if arena.node(cur).child.is_some() {
                    /*
                     * Write children...
                     */

                    if putc_cb.unwrap()(b'>' as i32, p) < 0 {
                        return -1;
                    } else {
                        col += 1;
                    }

                    col = mxml_write_ws(arena, cur, p, cb, MXML_WS_AFTER_OPEN, col, putc_cb);
                } else if !name.is_empty() && (name[0] == b'!' || name[0] == b'?') {
                    /*
                     * The ?xml declaration and !DOCTYPE nodes are written
                     * without a closing tag...
                     */

                    if putc_cb.unwrap()(b'>' as i32, p) < 0 {
                        return -1;
                    } else {
                        col += 1;
                    }

                    col = mxml_write_ws(arena, cur, p, cb, MXML_WS_AFTER_OPEN, col, putc_cb);
                } else {
                    if putc_cb.unwrap()(b' ' as i32, p) < 0 {
                        return -1;
                    }
                    if putc_cb.unwrap()(b'/' as i32, p) < 0 {
                        return -1;
                    }
                    if putc_cb.unwrap()(b'>' as i32, p) < 0 {
                        return -1;
                    }

                    col += 3;

                    col = mxml_write_ws(arena, cur, p, cb, MXML_WS_AFTER_OPEN, col, putc_cb);
                }
            }

            MXML_INTEGER => {
                if arena.node(cur).prev.is_some() {
                    if global.wrap > 0 && col > global.wrap {
                        if putc_cb.unwrap()(b'\n' as i32, p) < 0 {
                            return -1;
                        }

                        col = 0;
                    } else if putc_cb.unwrap()(b' ' as i32, p) < 0 {
                        return -1;
                    } else {
                        col += 1;
                    }
                }

                let MxmlValue::Integer(integer) = arena.node(cur).value else {
                    return -1;
                };
                let s: String = c_format("%d", &[CArg::Int(integer as i64)]);
                if mxml_write_string(s.as_bytes(), p, putc_cb) < 0 {
                    return -1;
                }

                col += s.len() as i32;
            }

            MXML_OPAQUE => {
                let MxmlValue::Opaque(opaque) = &arena.node(cur).value else {
                    return -1;
                };
                let opaque: &[u8] = opaque.as_deref().unwrap_or(b"");
                if mxml_write_string(opaque, p, putc_cb) < 0 {
                    return -1;
                }

                col += opaque.len() as i32;
            }

            MXML_REAL => {
                if arena.node(cur).prev.is_some() {
                    if global.wrap > 0 && col > global.wrap {
                        if putc_cb.unwrap()(b'\n' as i32, p) < 0 {
                            return -1;
                        }

                        col = 0;
                    } else if putc_cb.unwrap()(b' ' as i32, p) < 0 {
                        return -1;
                    } else {
                        col += 1;
                    }
                }

                let MxmlValue::Real(real) = arena.node(cur).value else {
                    return -1;
                };
                let s: String = c_format("%f", &[CArg::Dbl(real)]);
                if mxml_write_string(s.as_bytes(), p, putc_cb) < 0 {
                    return -1;
                }

                col += s.len() as i32;
            }

            MXML_TEXT => {
                let MxmlValue::Text(text) = &arena.node(cur).value else {
                    return -1;
                };
                if text.whitespace != 0 && col > 0 {
                    if global.wrap > 0 && col > global.wrap {
                        if putc_cb.unwrap()(b'\n' as i32, p) < 0 {
                            return -1;
                        }

                        col = 0;
                    } else if putc_cb.unwrap()(b' ' as i32, p) < 0 {
                        return -1;
                    } else {
                        col += 1;
                    }
                }

                let MxmlValue::Text(text) = &arena.node(cur).value else {
                    return -1;
                };
                let string: &[u8] = text.string.as_deref().unwrap_or(b"");
                if mxml_write_string(string, p, putc_cb) < 0 {
                    return -1;
                }

                col += string.len() as i32;
            }

            MXML_CUSTOM if global.custom_save_cb.is_some() => {
                /* The C passes `node`, not `current`, to the save callback. */
                let Some(data) = global.custom_save_cb.unwrap()(arena, node.unwrap()) else {
                    return -1;
                };

                if mxml_write_string(&data, p, putc_cb) < 0 {
                    return -1;
                }

                match data.iter().rposition(|c| *c == b'\n') {
                    None => col += data.len() as i32,
                    Some(newline) => col = (data.len() - newline) as i32,
                }
            }

            _ => {
                return -1;
            }
        }

        /*
         * Figure out the next node...
         */

        next = arena.node(cur).child;
        if next.is_none() {
            let mut cur = cur;
            loop {
                next = arena.node(cur).next;
                if next.is_some() {
                    break;
                }

                if Some(cur) == node {
                    break;
                }

                cur = arena.node(cur).parent.unwrap();

                let name: &[u8] = match &arena.node(cur).value {
                    MxmlValue::Element(element) => element.name.as_deref().unwrap_or(b""),
                    _ => b"",
                };

                if name.is_empty() || (name[0] != b'!' && name[0] != b'?') {
                    col = mxml_write_ws(arena, cur, p, cb, MXML_WS_BEFORE_CLOSE, col, putc_cb);

                    if putc_cb.unwrap()(b'<' as i32, p) < 0 {
                        return -1;
                    }
                    if putc_cb.unwrap()(b'/' as i32, p) < 0 {
                        return -1;
                    }
                    if mxml_write_string(name, p, putc_cb) < 0 {
                        return -1;
                    }
                    if putc_cb.unwrap()(b'>' as i32, p) < 0 {
                        return -1;
                    }

                    col += name.len() as i32 + 3;

                    col = mxml_write_ws(arena, cur, p, cb, MXML_WS_AFTER_CLOSE, col, putc_cb);
                }
            }
        }

        current = next;
    }

    col
}

#[cfg(test)]
mod tests {
    //! The `xml_probe` test is a differential driver: it reproduces, byte for
    //! byte, the output of the C program in
    //! `scratchpad/xmldiff/probe.c` linked against the reference
    //! `libimxml.so`.  It only runs when `IMOD_XML_PROBE_OUT` is set.
    //!
    //! One line of that report cannot be reproduced any more: the C prints
    //! `getUserData=%p`, a heap address, and the user data is an owned `Box`
    //! here rather than an integer cast to a pointer.  It prints `(set)` /
    //! `(nil)` instead.
    use super::*;
    use std::cell::Cell;

    thread_local! {
        static S_LAST_LEVEL: Cell<i32> = const { Cell::new(-1) };
    }

    /// Copy of `ixmlWhitespace_cb` from `IMOD/libcfshr/mxmlwrap.c`.
    fn ws_cb(arena: &MxmlArena, node: usize, where_: i32) -> Option<Vec<u8>> {
        let mut level: i32 = -1;
        let mut parent = arena.node(node).parent;
        let spaces: [u8; 32] = [b' '; 32];
        if where_ != MXML_WS_BEFORE_OPEN && where_ != MXML_WS_BEFORE_CLOSE {
            return None;
        }
        while let Some(pp) = parent {
            level += 1;
            parent = arena.node(pp).parent;
        }
        if level > 16 {
            level = 16;
        } else if level < 0 {
            level = 0;
        }
        let last = S_LAST_LEVEL.get();
        if last < 0 {
            S_LAST_LEVEL.set(level);
            return None;
        }
        if level == last && where_ == MXML_WS_BEFORE_CLOSE {
            return None;
        }
        S_LAST_LEVEL.set(level);
        let mut out: Vec<u8> = vec![b'\n'];
        out.extend_from_slice(&spaces[..(2 * level) as usize]);
        Some(out)
    }

    fn node_index(nodes: &[usize], n: Option<usize>) -> i32 {
        let Some(n) = n else {
            return -1;
        };
        for (i, p) in nodes.iter().enumerate() {
            if *p == n {
                return i as i32;
            }
        }
        -2
    }

    fn dump_tree(
        out: &mut dyn Write,
        arena: &MxmlArena,
        tree: Option<usize>,
        tag: &str,
    ) -> Vec<usize> {
        let mut ws: i32 = 0;
        let mut nodes: Vec<usize> = Vec::new();
        let mut n = tree;
        while let Some(cur) = n {
            nodes.push(cur);
            n = mxml_walk_next(arena, n, tree, MXML_DESCEND);
        }
        let num = nodes.len() as i32;
        let _ = out.write_all(
            c_format("%s nodes=%d\n", &[CArg::Str(tag), CArg::Int(num as i64)]).as_bytes(),
        );
        for i in 0..num {
            let n = nodes[i as usize];
            let _ = out.write_all(
                c_format(
                    "  [%d] type=%d parent=%d child=%d last=%d prev=%d next=%d ref=%d",
                    &[
                        CArg::Int(i as i64),
                        CArg::Int(mxml_get_type(arena, Some(n)) as i64),
                        CArg::Int(node_index(&nodes, arena.node(n).parent) as i64),
                        CArg::Int(node_index(&nodes, arena.node(n).child) as i64),
                        CArg::Int(node_index(&nodes, arena.node(n).last_child) as i64),
                        CArg::Int(node_index(&nodes, arena.node(n).prev) as i64),
                        CArg::Int(node_index(&nodes, arena.node(n).next) as i64),
                        CArg::Int(mxml_get_ref_count(arena, Some(n)) as i64),
                    ],
                )
                .as_bytes(),
            );
            match &arena.node(n).value {
                MxmlValue::Element(element) => {
                    let _ = out.write_all(&c_format_bytes(
                        " elem=<%s> nattr=%d",
                        &[
                            CArg::Bytes(element.name.as_deref().unwrap_or(b"(null)")),
                            CArg::Int(element.attrs.len() as i64),
                        ],
                    ));
                    for (k, a) in element.attrs.iter().enumerate() {
                        let _ = out.write_all(&c_format_bytes(
                            " attr[%d]=%s=\"%s\"",
                            &[
                                CArg::Int(k as i64),
                                CArg::Bytes(&a.name),
                                CArg::Bytes(a.value.as_deref().unwrap_or(b"(nil)")),
                            ],
                        ));
                    }
                    if let Some(s) = mxml_get_cdata(arena, Some(n)) {
                        let _ = out.write_all(&c_format_bytes(" cdata=<%s>", &[CArg::Bytes(s)]));
                    }
                }
                MxmlValue::Opaque(opaque) => {
                    let _ = out.write_all(&c_format_bytes(
                        " opaque=<%s>",
                        &[CArg::Bytes(opaque.as_deref().unwrap_or(b"(null)"))],
                    ));
                }
                MxmlValue::Text(text) => {
                    let _ = out.write_all(&c_format_bytes(
                        " text=<%s> ws=%d",
                        &[
                            CArg::Bytes(text.string.as_deref().unwrap_or(b"(null)")),
                            CArg::Int(text.whitespace as i64),
                        ],
                    ));
                }
                MxmlValue::Integer(integer) => {
                    let _ = out.write_all(
                        c_format(" integer=%d", &[CArg::Int(*integer as i64)]).as_bytes(),
                    );
                }
                MxmlValue::Real(real) => {
                    let _ = out.write_all(c_format(" real=%f", &[CArg::Dbl(*real)]).as_bytes());
                }
                _ => {}
            }
            let _ = out.write_all(&c_format_bytes(
                " getElem=%s getOpaque=%s getInt=%d getReal=%f getText=%s",
                &[
                    CArg::Bytes(mxml_get_element(arena, Some(n)).unwrap_or(b"(nil)")),
                    CArg::Bytes(mxml_get_opaque(arena, Some(n)).unwrap_or(b"(nil)")),
                    CArg::Int(mxml_get_integer(arena, Some(n)) as i64),
                    CArg::Dbl(mxml_get_real(arena, Some(n))),
                    CArg::Bytes(mxml_get_text(arena, Some(n), Some(&mut ws)).unwrap_or(b"(nil)")),
                ],
            ));
            let _ = out.write_all(b"\n");
        }
        nodes
    }

    fn save_report(
        out: &mut dyn Write,
        arena: &MxmlArena,
        tree: Option<usize>,
        tag: &str,
        cb: MxmlSaveCb,
        wrap: i32,
    ) {
        let mut buf: Vec<u8> = vec![0; 400000];
        mxml_set_wrap_margin(wrap);
        S_LAST_LEVEL.set(-1);
        let n = mxml_save_string(arena, tree, &mut buf, 400000, cb);
        let end = buf.iter().position(|c| *c == 0).unwrap_or(buf.len());
        let _ = out.write_all(&c_format_bytes(
            "%s saveString=%d\n[%s]\n",
            &[
                CArg::Str(tag),
                CArg::Int(n as i64),
                CArg::Bytes(&buf[..end]),
            ],
        ));
        S_LAST_LEVEL.set(-1);
        let alloc = mxml_save_alloc_string(arena, tree, cb);
        let _ = out.write_all(
            c_format(
                "%s saveAlloc=%s\n",
                &[
                    CArg::Str(tag),
                    CArg::Str(if alloc.is_some() { "ok" } else { "(nil)" }),
                ],
            )
            .as_bytes(),
        );
        if let Some(alloc) = alloc {
            let _ = out.write_all(&c_format_bytes("[%s]\n", &[CArg::Bytes(&alloc)]));
        }
        S_LAST_LEVEL.set(-1);
        let mut small: [u8; 40] = [0; 40];
        let m = mxml_save_string(arena, tree, &mut small, 40, cb);
        let end = small.iter().position(|c| *c == 0).unwrap_or(small.len());
        let _ = out.write_all(&c_format_bytes(
            "%s saveSmall=%d [%s]\n",
            &[
                CArg::Str(tag),
                CArg::Int(m as i64),
                CArg::Bytes(&small[..end]),
            ],
        ));
        mxml_set_wrap_margin(72);
    }

    fn probe_file(out: &mut dyn Write, path: &str, base: &str, savedir: &str) {
        let _ = out.write_all(c_format("=== FILE %s\n", &[CArg::Str(base)]).as_bytes());
        let Some(mut fp) = ImodFile::open(path, "r") else {
            let _ = out.write_all(b"  no open\n");
            return;
        };
        let arena = &mut MxmlArena::new();
        let tree = mxml_load_file(arena, MXML_NO_PARENT, &mut fp, Some(mxml_opaque_cb));
        drop(fp);
        if tree.is_none() {
            let _ = out.write_all(b"  load NULL\n");
            return;
        }
        let nodes = dump_tree(out, arena, tree, "  TREE");

        let _ = out.write_all(
            c_format(
                "  walkNext(tree,tree,DESCEND) idx=%d\n",
                &[CArg::Int(
                    node_index(&nodes, mxml_walk_next(arena, tree, tree, MXML_DESCEND)) as i64,
                )],
            )
            .as_bytes(),
        );
        let _ = out.write_all(
            c_format(
                "  walkNext(tree,tree,NO_DESCEND) idx=%d\n",
                &[CArg::Int(
                    node_index(&nodes, mxml_walk_next(arena, tree, tree, MXML_NO_DESCEND)) as i64,
                )],
            )
            .as_bytes(),
        );
        let mut n = tree;
        while mxml_walk_next(arena, n, tree, MXML_DESCEND).is_some() {
            n = mxml_walk_next(arena, n, tree, MXML_DESCEND);
        }
        let _ = out.write_all(
            c_format(
                "  lastNode=%d walkPrev=%d walkPrevNoDesc=%d\n",
                &[
                    CArg::Int(node_index(&nodes, n) as i64),
                    CArg::Int(
                        node_index(&nodes, mxml_walk_prev(arena, n, tree, MXML_DESCEND)) as i64,
                    ),
                    CArg::Int(
                        node_index(&nodes, mxml_walk_prev(arena, n, tree, MXML_NO_DESCEND)) as i64,
                    ),
                ],
            )
            .as_bytes(),
        );

        let mut count: i32 = 0;
        let mut n = mxml_find_element(arena, tree, tree, Some(b"Field"), None, None, MXML_DESCEND);
        while n.is_some() {
            if count < 5 {
                let a = mxml_element_get_attr(arena, n, Some(b"name"));
                let _ = out.write_all(&c_format_bytes(
                    "  findField[%d]=%d attr=%s\n",
                    &[
                        CArg::Int(count as i64),
                        CArg::Int(node_index(&nodes, n) as i64),
                        CArg::Bytes(a.unwrap_or(b"(nil)")),
                    ],
                ));
            }
            count += 1;
            n = mxml_find_element(arena, n, tree, Some(b"Field"), None, None, MXML_DESCEND);
        }
        let _ =
            out.write_all(c_format("  findFieldCount=%d\n", &[CArg::Int(count as i64)]).as_bytes());

        for (label, path) in [
            (
                "  findPath(autodoc/PreData/Version)=%d\n",
                b"autodoc/PreData/Version".as_slice(),
            ),
            ("  findPath(*/short)=%d\n", b"*/short".as_slice()),
            ("  findPath(nosuch)=%d\n", b"nosuch".as_slice()),
        ] {
            let _ = out.write_all(
                c_format(
                    label,
                    &[CArg::Int(
                        node_index(&nodes, mxml_find_path(arena, tree, path)) as i64,
                    )],
                )
                .as_bytes(),
            );
        }

        let ind = mxml_index_new(arena, tree, Some(b"Field"), Some(b"name"));
        if ind.is_none() {
            let _ = out.write_all(b"  index NULL\n");
        } else {
            let mut ind = ind.unwrap();
            let _ = out.write_all(
                c_format(
                    "  indexCount=%d alloc=%d\n",
                    &[
                        CArg::Int(mxml_index_get_count(Some(&ind)) as i64),
                        CArg::Int(ind.nodes.capacity() as i64),
                    ],
                )
                .as_bytes(),
            );
            let mut count: i32 = 0;
            let mut n = mxml_index_reset(Some(&mut ind));
            while n.is_some() {
                if count < 8 {
                    let a = mxml_element_get_attr(arena, n, Some(b"name"));
                    let _ = out.write_all(&c_format_bytes(
                        "  indexEnum[%d]=%d %s\n",
                        &[
                            CArg::Int(count as i64),
                            CArg::Int(node_index(&nodes, n) as i64),
                            CArg::Bytes(a.unwrap_or(b"(nil)")),
                        ],
                    ));
                }
                count += 1;
                n = mxml_index_enum(Some(&mut ind));
            }
            let _ = out.write_all(
                c_format("  indexEnumCount=%d\n", &[CArg::Int(count as i64)]).as_bytes(),
            );
            mxml_index_reset(Some(&mut ind));
            let n = mxml_index_find(arena, Some(&mut ind), Some(b"Field"), Some(b"InputFile"));
            let _ = out.write_all(
                c_format(
                    "  indexFind(Field,InputFile)=%d\n",
                    &[CArg::Int(node_index(&nodes, n) as i64)],
                )
                .as_bytes(),
            );
            mxml_index_reset(Some(&mut ind));
            let n = mxml_index_find(arena, Some(&mut ind), Some(b"Field"), Some(b"zzz-none"));
            let _ = out.write_all(
                c_format(
                    "  indexFind(Field,zzz-none)=%d\n",
                    &[CArg::Int(node_index(&nodes, n) as i64)],
                )
                .as_bytes(),
            );
            mxml_index_delete(Some(ind));
        }

        let ind = mxml_index_new(arena, tree, None, None);
        let _ = out.write_all(
            c_format(
                "  indexAllCount=%d\n",
                &[CArg::Int(match &ind {
                    Some(ind) => mxml_index_get_count(Some(ind)) as i64,
                    None => -1,
                })],
            )
            .as_bytes(),
        );
        mxml_index_delete(ind);

        save_report(out, arena, tree, "  SAVE-nocb", None, 72);
        save_report(out, arena, tree, "  SAVE-ws", Some(ws_cb), 0);
        save_report(out, arena, tree, "  SAVE-wrap20", None, 20);

        let savepath = format!("{}/{}.save", savedir, base);
        if let Some(mut fp) = ImodFile::open(&savepath, "w") {
            mxml_set_wrap_margin(0);
            S_LAST_LEVEL.set(-1);
            let _ = out.write_all(
                c_format(
                    "  saveFile=%d\n",
                    &[CArg::Int(
                        mxml_save_file(arena, tree, &mut fp, Some(ws_cb)) as i64
                    )],
                )
                .as_bytes(),
            );
            drop(fp);
            mxml_set_wrap_margin(72);
        }

        mxml_delete(arena, tree);
    }

    const CASES: &[&[u8]] = &[
        b"<a>b</a>",
        b"<?xml version=\"1.0\"?><root><x>1</x></root>",
        b"<!DOCTYPE foo><root/>",
        b"<root><!-- a comment --><x/></root>",
        b"<root><![CDATA[some <raw> data]]></root>",
        b"<root a='1' b=\"two\" c=unquoted>text</root>",
        b"<root>&amp;&lt;&gt;&quot;&apos;&#65;&#x42;&nbsp;</root>",
        b"<root>&bogus;</root>",
        b"<root><a></b></root>",
        b"<root",
        b"",
        b"<root>unclosed",
        b"<a/><b/>",
        b"<root>   spaced   text   </root>",
        b"<root><child x=\"1\"/><child x=\"2\"/><child x=\"3\"/></root>",
    ];

    fn probe_strings(out: &mut dyn Write) {
        let mut buf: [u8; 8192] = [0; 8192];
        for (i, case) in CASES.iter().enumerate() {
            let _ = out.write_all(&c_format_bytes(
                "=== STRING %d [%s]\n",
                &[CArg::Int(i as i64), CArg::Bytes(case)],
            ));
            let _ = out.flush();
            let arena = &mut MxmlArena::new();
            let tree = mxml_load_string(arena, MXML_NO_PARENT, case, Some(mxml_opaque_cb));
            if tree.is_none() {
                let _ = out.write_all(b"  load NULL\n");
                continue;
            }
            dump_tree(out, arena, tree, "  TREE");
            S_LAST_LEVEL.set(-1);
            let n = mxml_save_string(arena, tree, &mut buf, 8192, None);
            let end = buf.iter().position(|c| *c == 0).unwrap_or(buf.len());
            let _ = out.write_all(&c_format_bytes(
                "  save=%d [%s]\n",
                &[CArg::Int(n as i64), CArg::Bytes(&buf[..end])],
            ));
            mxml_delete(arena, tree);
        }
        for (tag, cb) in [
            ("=== STRING-TEXTCB %d\n", None as MxmlLoadCb),
            ("=== STRING-INTCB %d\n", Some(mxml_integer_cb as _)),
            ("=== STRING-REALCB %d\n", Some(mxml_real_cb as _)),
            ("=== STRING-IGNORECB %d\n", Some(mxml_ignore_cb as _)),
        ] {
            for (i, case) in CASES.iter().enumerate() {
                let _ = out.write_all(c_format(tag, &[CArg::Int(i as i64)]).as_bytes());
                let _ = out.flush();
                let arena = &mut MxmlArena::new();
                let tree = mxml_load_string(arena, MXML_NO_PARENT, case, cb);
                if tree.is_none() {
                    let _ = out.write_all(b"  load NULL\n");
                    continue;
                }
                dump_tree(out, arena, tree, "  TREE");
                mxml_delete(arena, tree);
            }
        }
    }

    fn probe_build(out: &mut dyn Write) {
        let mut buf: [u8; 8192] = [0; 8192];
        let mut i: i32 = 0;
        let arena = &mut MxmlArena::new();

        let _ = out.write_all(b"=== BUILD\n");
        let xml = mxml_new_xml(arena, Some(b"1.0"));
        dump_tree(out, arena, xml, "  XMLONLY");
        let top = mxml_new_element(arena, xml, Some(b"root"));
        let elem = mxml_new_element(arena, top, Some(b"child"));
        mxml_element_set_attr(arena, elem, Some(b"one"), Some(b"1"));
        mxml_element_set_attr(arena, elem, Some(b"two"), Some(b"2"));
        mxml_element_set_attr(arena, elem, Some(b"one"), Some(b"1b"));
        mxml_element_set_attr(arena, elem, Some(b"nil"), None);
        mxml_element_set_attrf(arena, elem, Some(b"fmt"), Some(b"%d"), b"42");
        mxml_new_text(arena, elem, 0, Some(b"hello & <world>"));
        mxml_new_text(arena, top, 1, Some(b"spaced"));
        mxml_new_integer(arena, top, 17);
        mxml_new_real(arena, top, 3.5);
        mxml_new_opaque(arena, top, Some(b"opaque \"text\""));
        mxml_new_cdata(arena, top, Some(b"cdata & stuff"));
        dump_tree(out, arena, xml, "  BUILT");
        let _ = out.write_all(&c_format_bytes(
            "  getAttr(one)=%s\n",
            &[CArg::Bytes(
                mxml_element_get_attr(arena, elem, Some(b"one")).unwrap_or(b"(null)"),
            )],
        ));
        for (label, name) in [
            ("  getAttr(nil)=%s\n", b"nil".as_slice()),
            ("  getAttr(zzz)=%s\n", b"zzz".as_slice()),
        ] {
            let _ = out.write_all(
                c_format(
                    label,
                    &[CArg::Str(
                        if mxml_element_get_attr(arena, elem, Some(name)).is_some() {
                            "nonnull"
                        } else {
                            "(nil)"
                        },
                    )],
                )
                .as_bytes(),
            );
        }
        mxml_element_delete_attr(arena, elem, Some(b"two"));
        mxml_element_delete_attr(arena, elem, Some(b"zzz"));
        dump_tree(out, arena, xml, "  AFTERDEL");
        save_report(out, arena, xml, "  BUILDSAVE", None, 72);
        save_report(out, arena, xml, "  BUILDSAVEWS", Some(ws_cb), 0);

        let t = mxml_new_element(arena, MXML_NO_PARENT, Some(b"orphan"));
        let r1 = mxml_retain(arena, t);
        let r2 = mxml_retain(arena, t);
        let r3 = mxml_release(arena, t);
        let r4 = mxml_release(arena, t);
        let r5 = mxml_release(arena, t);
        let _ = out.write_all(
            c_format(
                "  retain=%d retain=%d release=%d release=%d releaseFinal=%d\n",
                &[
                    CArg::Int(r1 as i64),
                    CArg::Int(r2 as i64),
                    CArg::Int(r3 as i64),
                    CArg::Int(r4 as i64),
                    CArg::Int(r5 as i64),
                ],
            )
            .as_bytes(),
        );
        let _ = out.write_all(
            c_format(
                "  retainNull=%d releaseNull=%d\n",
                &[
                    CArg::Int(mxml_retain(arena, None) as i64),
                    CArg::Int(mxml_release(arena, None) as i64),
                ],
            )
            .as_bytes(),
        );

        let t = mxml_new_element(arena, MXML_NO_PARENT, Some(b"setme"));
        let _ = out.write_all(
            c_format(
                "  setElement=%d\n",
                &[CArg::Int(
                    mxml_set_element(arena, t, Some(b"renamed")) as i64
                )],
            )
            .as_bytes(),
        );
        let _ = out.write_all(&c_format_bytes(
            "  name=%s\n",
            &[CArg::Bytes(mxml_get_element(arena, t).unwrap_or(b"(null)"))],
        ));
        let _ = out.write_all(
            c_format(
                "  setCDATAbad=%d\n",
                &[CArg::Int(mxml_set_cdata(arena, t, Some(b"x")) as i64)],
            )
            .as_bytes(),
        );
        let _ = out.write_all(
            c_format(
                "  setUserData=%d\n",
                &[CArg::Int(
                    mxml_set_user_data(arena, t, Some(Box::new(1usize))) as i64,
                )],
            )
            .as_bytes(),
        );
        /* The C prints the pointer with %p; there is no address to print. */
        let _ = out.write_all(
            c_format(
                "  getUserData=%s\n",
                &[CArg::Str(if mxml_get_user_data(arena, t).is_some() {
                    "(set)"
                } else {
                    "(nil)"
                })],
            )
            .as_bytes(),
        );
        mxml_delete(arena, t);

        let t = mxml_new_integer(arena, MXML_NO_PARENT, 5);
        let a = mxml_set_integer(arena, t, 9);
        let _ = out.write_all(
            c_format(
                "  setInteger=%d %d\n",
                &[
                    CArg::Int(a as i64),
                    CArg::Int(mxml_get_integer(arena, t) as i64),
                ],
            )
            .as_bytes(),
        );
        let _ = out.write_all(
            c_format(
                "  setReal=%d\n",
                &[CArg::Int(mxml_set_real(arena, t, 1.0) as i64)],
            )
            .as_bytes(),
        );
        mxml_delete(arena, t);
        let t = mxml_new_real(arena, MXML_NO_PARENT, 5.0);
        let a = mxml_set_real(arena, t, 9.5);
        let _ = out.write_all(
            c_format(
                "  setReal=%d %f\n",
                &[CArg::Int(a as i64), CArg::Dbl(mxml_get_real(arena, t))],
            )
            .as_bytes(),
        );
        mxml_delete(arena, t);
        let t = mxml_new_text(arena, MXML_NO_PARENT, 0, Some(b"a"));
        let _ = out.write_all(
            c_format(
                "  setText=%d\n",
                &[CArg::Int(mxml_set_text(arena, t, 1, Some(b"bcd")) as i64)],
            )
            .as_bytes(),
        );
        let _ = out.write_all(&c_format_bytes(
            "  getText=%s\n",
            &[CArg::Bytes(
                mxml_get_text(arena, t, Some(&mut i)).unwrap_or(b"(null)"),
            )],
        ));
        mxml_delete(arena, t);
        let t = mxml_new_opaque(arena, MXML_NO_PARENT, Some(b"a"));
        let a = mxml_set_opaque(arena, t, Some(b"xyz"));
        let _ = out.write_all(&c_format_bytes(
            "  setOpaque=%d %s\n",
            &[
                CArg::Int(a as i64),
                CArg::Bytes(mxml_get_opaque(arena, t).unwrap_or(b"(null)")),
            ],
        ));
        mxml_delete(arena, t);
        let t = mxml_new_cdata(arena, MXML_NO_PARENT, Some(b"abc"));
        let c0 = mxml_get_cdata(arena, t).unwrap_or(b"(null)").to_vec();
        let sc = mxml_set_cdata(arena, t, Some(b"def"));
        let _ = out.write_all(&c_format_bytes(
            "  cdata=%s setCDATA=%d then=%s\n",
            &[
                CArg::Bytes(&c0),
                CArg::Int(sc as i64),
                CArg::Bytes(mxml_get_cdata(arena, t).unwrap_or(b"(null)")),
            ],
        ));
        S_LAST_LEVEL.set(-1);
        let n = mxml_save_string(arena, t, &mut buf, 8192, None);
        let end = buf.iter().position(|c| *c == 0).unwrap_or(buf.len());
        let _ = out.write_all(&c_format_bytes(
            "  cdataSave=%d [%s]\n",
            &[CArg::Int(n as i64), CArg::Bytes(&buf[..end])],
        ));
        mxml_delete(arena, t);

        mxml_delete(arena, xml);
    }

    fn probe_entities(out: &mut dyn Write) {
        let names: &[&[u8]] = &[
            b"amp", b"lt", b"gt", b"quot", b"apos", b"nbsp", b"AElig", b"zwnj", b"Alpha", b"euro",
            b"bogus", b"", b"a", b"zzzz",
        ];
        let _ = out.write_all(b"=== ENTITIES\n");
        for i in 0..300 {
            if let Some(n) = mxml_entity_get_name(i) {
                let _ = out.write_all(&c_format_bytes(
                    "  getName(%d)=%s\n",
                    &[CArg::Int(i as i64), CArg::Bytes(n)],
                ));
            }
        }
        for name in names {
            let _ = out.write_all(&c_format_bytes(
                "  getValue(%s)=%d\n",
                &[
                    CArg::Bytes(name),
                    CArg::Int(mxml_entity_get_value(name) as i64),
                ],
            ));
        }
    }

    /// A processing instruction read at top level becomes the parent of
    /// everything that follows (`mxml-file.c:1797-1804`), so the document
    /// element hangs off the `?xml?` node.  Expected values produced by the
    /// reference `libimxml.so`:
    ///     PI root elem=?xml version="1.0"? child=root walkNext=root
    #[test]
    fn processing_instruction_becomes_the_parent_of_the_document() {
        let arena = &mut MxmlArena::new();
        let tree = mxml_load_string(
            arena,
            MXML_NO_PARENT,
            b"<?xml version=\"1.0\"?><root><x>1</x></root>",
            Some(mxml_opaque_cb),
        );
        assert!(tree.is_some());
        assert_eq!(
            mxml_get_element(arena, tree),
            Some(b"?xml version=\"1.0\"?".as_slice())
        );
        let child = mxml_get_first_child(arena, tree);
        assert!(
            child.is_some(),
            "the ?xml node must own the document element"
        );
        assert_eq!(mxml_get_element(arena, child), Some(b"root".as_slice()));
        let walk = mxml_walk_next(arena, tree, tree, MXML_DESCEND);
        assert!(
            walk.is_some(),
            "mxmlWalkNext must descend into the ?xml node"
        );
        assert_eq!(mxml_get_element(arena, walk), Some(b"root".as_slice()));
        mxml_delete(arena, tree);
    }

    /// `mxmlSaveString`/`mxmlSaveFile` must render every child, including the
    /// children of `?`/`!` nodes, and must honour the `mxml_save_cb_t`
    /// whitespace callback.  Byte strings and return values below come from the
    /// reference `libimxml.so`.
    #[test]
    fn save_string_and_save_file_match_native_bytes() {
        let mut buf: [u8; 8192] = [0; 8192];
        let arena = &mut MxmlArena::new();
        let tree = mxml_load_string(
            arena,
            MXML_NO_PARENT,
            b"<root><child x=\"1\"/><child x=\"2\"/><child x=\"3\"/></root>",
            Some(mxml_opaque_cb),
        );
        assert!(tree.is_some());

        let n = mxml_save_string(arena, tree, &mut buf, 8192, None);
        assert_eq!(n, 59);
        assert_eq!(
            &buf[..n as usize],
            b"<root><child x=\"1\" /><child x=\"2\" /><child x=\"3\" /></root>\n"
        );

        /* With the indenting callback and no wrap margin. */
        mxml_set_wrap_margin(0);
        S_LAST_LEVEL.set(-1);
        let n = mxml_save_string(arena, tree, &mut buf, 8192, Some(ws_cb));
        assert_eq!(n, 62);
        assert_eq!(
            &buf[..n as usize],
            b"<root>\n<child x=\"1\" />\n<child x=\"2\" />\n<child x=\"3\" /></root>\n"
        );

        /* mxmlSaveFile writes the same bytes through a file. */
        let path =
            std::env::temp_dir().join(format!("imodrs_mxml_savefile-{}.xml", std::process::id()));
        let mut fp = ImodFile::open(path.to_str().unwrap(), "w").unwrap();
        S_LAST_LEVEL.set(-1);
        assert_eq!(mxml_save_file(arena, tree, &mut fp, Some(ws_cb)), 0);
        drop(fp);
        let written = std::fs::read(&path).unwrap();
        assert_eq!(
            written,
            b"<root>\n<child x=\"1\" />\n<child x=\"2\" />\n<child x=\"3\" /></root>\n"
        );
        let _ = std::fs::remove_file(&path);

        /*
         * A tree whose root is a processing instruction still renders all
         * of its children; the native library writes 55 bytes here.
         */
        let xml = mxml_load_string(
            arena,
            MXML_NO_PARENT,
            b"<?xml version=\"1.0\"?><root><child x=\"1\"/></root>",
            Some(mxml_opaque_cb),
        );
        assert!(xml.is_some());
        S_LAST_LEVEL.set(-1);
        let n = mxml_save_string(arena, xml, &mut buf, 8192, Some(ws_cb));
        assert_eq!(n, 55);
        assert_eq!(
            &buf[..n as usize],
            b"<?xml version=\"1.0\"?>\n<root>\n  <child x=\"1\" />\n</root>\n"
        );
        mxml_delete(arena, xml);

        mxml_set_wrap_margin(72);

        /* A short buffer truncates but still returns the full length. */
        let m = mxml_save_string(arena, tree, &mut buf, 20, None);
        assert_eq!(m, 59);
        assert_eq!(&buf[..19], b"<root><child x=\"1\" ");
        assert_eq!(buf[19], 0);

        mxml_delete(arena, tree);
    }

    /// `_mxml_global()` starts `wrap` at 72 (`mxml-private.c:167`), so a long
    /// attribute list breaks at column 72 with no explicit
    /// `mxmlSetWrapMargin` call.  Native bytes below.
    #[test]
    fn default_wrap_margin_is_72_columns() {
        let mut buf: [u8; 8192] = [0; 8192];
        let arena = &mut MxmlArena::new();
        let xml = mxml_new_xml(arena, Some(b"1.0"));
        let top = mxml_new_element(arena, xml, Some(b"root"));
        let e = mxml_new_element(arena, top, Some(b"e"));
        mxml_element_set_attr(arena, e, Some(b"aaaaaaaaaa"), Some(b"1111111111"));
        mxml_element_set_attr(arena, e, Some(b"bbbbbbbbbb"), Some(b"2222222222"));
        mxml_element_set_attr(arena, e, Some(b"cccccccccc"), Some(b"3333333333"));
        let n = mxml_save_string(arena, xml, &mut buf, 8192, None);
        assert_eq!(n, 129);
        assert_eq!(
            &buf[..n as usize],
            b"<?xml version=\"1.0\" encoding=\"utf-8\"?><root><e aaaaaaaaaa=\"1111111111\"\nbbbbbbbbbb=\"2222222222\" cccccccccc=\"3333333333\" /></root>\n".as_slice()
        );
        mxml_delete(arena, xml);
    }

    /// Character entities resolve through the full upstream table and are
    /// re-encoded as UTF-8 by `mxml_add_char`.  Native byte sequence for
    /// `&AElig;&#65;&nbsp;` is `c3 86 41 c2 a0`.
    #[test]
    fn named_and_numeric_entities_decode_to_utf8() {
        let arena = &mut MxmlArena::new();
        let tree = mxml_load_string(
            arena,
            MXML_NO_PARENT,
            b"<root>&AElig;&#65;&nbsp;</root>",
            Some(mxml_opaque_cb),
        );
        assert!(tree.is_some());
        let o = mxml_get_opaque(arena, tree);
        assert_eq!(o, Some([0xc3u8, 0x86, 0x41, 0xc2, 0xa0].as_slice()));
        mxml_delete(arena, tree);
    }

    /// autodoc round-trip driver mirroring `scratchpad/xmldiff/adocrt.c`.
    ///
    #[test]
    fn xml_autodoc_roundtrip() {
        use crate::imod::libcfshr::autodoc::{
            adoc_clear, adoc_get_write_as_xml, adoc_get_xml_root_element, adoc_read,
            adoc_set_write_as_xml, adoc_write,
        };
        let Ok(reppath) = std::env::var("IMOD_XML_RT_REPORT") else {
            return;
        };
        let outdir = std::env::var("IMOD_XML_RT_OUTDIR").unwrap();
        let asxml: i32 = std::env::var("IMOD_XML_RT_ASXML").unwrap().parse().unwrap();
        let list = std::env::var("IMOD_XML_RT_FILES").unwrap();
        let files = std::fs::read_to_string(&list).unwrap();
        let mut rep = ImodFile::open(&reppath, "w").unwrap();
        for line in files.lines() {
            if line.is_empty() {
                continue;
            }
            let base = line.rsplit('/').next().unwrap();
            let ind = adoc_read(line.as_bytes());
            let _ = rep.write_all(
                c_format("%s read=%d", &[CArg::Str(base), CArg::Int(ind as i64)]).as_bytes(),
            );
            if ind < 0 {
                let _ = rep.write_all(b"\n");
                continue;
            }
            let mut root: Option<Vec<u8>> = None;
            let err = adoc_get_xml_root_element(&mut root);
            let rootbytes: Vec<u8> = match &root {
                None => b"(nil)".to_vec(),
                Some(root) => root.clone(),
            };
            let _ = rep.write_all(&c_format_bytes(
                " rootErr=%d root=%s",
                &[CArg::Int(err as i64), CArg::Bytes(&rootbytes)],
            ));
            let _ = rep.write_all(
                c_format(" xmlRead=%d", &[CArg::Int(adoc_get_write_as_xml() as i64)]).as_bytes(),
            );
            adoc_set_write_as_xml(asxml);
            let out = format!("{}/{}.out", outdir, base);
            let _ = rep.write_all(
                c_format(
                    " write=%d\n",
                    &[CArg::Int(adoc_write(out.as_bytes()) as i64)],
                )
                .as_bytes(),
            );
            adoc_clear(ind);
            adoc_set_write_as_xml(0);
        }
    }

    #[test]
    fn xml_probe() {
        let Ok(outpath) = std::env::var("IMOD_XML_PROBE_OUT") else {
            return;
        };
        let savedir = std::env::var("IMOD_XML_PROBE_SAVEDIR").unwrap();
        let list = std::env::var("IMOD_XML_PROBE_FILES").unwrap();
        let files = std::fs::read_to_string(&list).unwrap();
        let mut out = ImodFile::open(&outpath, "w").unwrap();
        probe_entities(&mut out);
        probe_strings(&mut out);
        probe_build(&mut out);
        for line in files.lines() {
            if line.is_empty() {
                continue;
            }
            let base = line.rsplit('/').next().unwrap();
            probe_file(&mut out, line, base, &savedir);
        }
    }
}
