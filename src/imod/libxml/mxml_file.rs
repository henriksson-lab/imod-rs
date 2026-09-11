//! Translation of `IMOD/libxml/mxml-file.c`.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use super::*;
use core::ffi::{c_char, c_int, c_uchar, c_void};

/// Matches C `ENCODE_UTF8` (`mxml-file.c:31`).
pub const ENCODE_UTF8: c_int = 0;
/// Matches C `ENCODE_UTF16BE` (`mxml-file.c:32`).
pub const ENCODE_UTF16BE: c_int = 1;
/// Matches C `ENCODE_UTF16LE` (`mxml-file.c:33`).
pub const ENCODE_UTF16LE: c_int = 2;

/// C `EOF` from `<stdio.h>`.
const EOF: c_int = -1;

/// Matches C `_mxml_getc_cb_t` (`mxml-file.c:44`).
pub type MxmlGetcCb = Option<unsafe extern "C" fn(*mut c_void, *mut c_int) -> c_int>;
/// Matches C `_mxml_putc_cb_t` (`mxml-file.c:45`).
pub type MxmlPutcCb = Option<unsafe extern "C" fn(c_int, *mut c_void) -> c_int>;

/// Matches C `_mxml_fdbuf_t` (`mxml-file.c:47`).
#[repr(C)]
pub struct MxmlFdbuf {
    pub fd: c_int,
    pub current: *mut c_uchar,
    pub end: *mut c_uchar,
    pub buffer: [c_uchar; 8192],
}

/// Matches C `mxmlLoadFd` (`mxml-file.c:80`).
pub unsafe fn mxml_load_fd(top: *mut MxmlNode, fd: c_int, cb: MxmlLoadCb) -> *mut MxmlNode {
    let mut buf: MxmlFdbuf = core::mem::zeroed();

    /*
     * Initialize the file descriptor buffer...
     */

    buf.fd = fd;
    buf.current = buf.buffer.as_mut_ptr();
    buf.end = buf.buffer.as_mut_ptr();

    /*
     * Read the XML data...
     */

    mxml_load_data(
        top,
        &raw mut buf as *mut c_void,
        cb,
        Some(mxml_fd_getc),
        MXML_NO_CALLBACK_SAX,
        core::ptr::null_mut(),
    )
}

/// Matches C `MXML_NO_CALLBACK` where a SAX callback is expected
/// (`mxml-file.c:97`).
const MXML_NO_CALLBACK_SAX: MxmlSaxCb = None;

/// Matches C `mxmlLoadFile` (`mxml-file.c:117`).
pub unsafe fn mxml_load_file(
    top: *mut MxmlNode,
    fp: *mut libc::FILE,
    cb: MxmlLoadCb,
) -> *mut MxmlNode {
    /*
     * Read the XML data...
     */

    mxml_load_data(
        top,
        fp as *mut c_void,
        cb,
        Some(mxml_file_getc),
        MXML_NO_CALLBACK_SAX,
        core::ptr::null_mut(),
    )
}

/// Matches C `mxmlLoadString` (`mxml-file.c:145`).
pub unsafe fn mxml_load_string(
    top: *mut MxmlNode,
    s: *const c_char,
    cb: MxmlLoadCb,
) -> *mut MxmlNode {
    let mut sp: *const c_char = s;

    /*
     * Read the XML data...
     */

    mxml_load_data(
        top,
        &raw mut sp as *mut c_void,
        cb,
        Some(mxml_string_getc),
        MXML_NO_CALLBACK_SAX,
        core::ptr::null_mut(),
    )
}

/// Matches C `mxmlSaveAllocString` (`mxml-file.c:170`).
pub unsafe fn mxml_save_alloc_string(node: *mut MxmlNode, cb: MxmlSaveCb) -> *mut c_char {
    let bytes: c_int;
    let mut buffer: [c_char; 8192] = [0; 8192];
    let s: *mut c_char;

    /*
     * Write the node to the temporary buffer...
     */

    bytes = mxml_save_string(
        node,
        buffer.as_mut_ptr(),
        core::mem::size_of::<[c_char; 8192]>() as c_int,
        cb,
    );

    if bytes <= 0 {
        return core::ptr::null_mut();
    }

    if bytes < (core::mem::size_of::<[c_char; 8192]>() - 1) as c_int {
        /*
         * Node fit inside the buffer, so just duplicate that string and
         * return...
         */

        return libc::strdup(buffer.as_ptr());
    }

    /*
     * Allocate a buffer of the required size and save the node to the
     * new buffer...
     */

    s = libc::malloc((bytes + 1) as usize) as *mut c_char;
    if s.is_null() {
        return core::ptr::null_mut();
    }

    mxml_save_string(node, s, bytes + 1, cb);

    /*
     * Return the allocated string...
     */

    s
}

/// Matches C `mxmlSaveFd` (`mxml-file.c:218`).
pub unsafe fn mxml_save_fd(node: *mut MxmlNode, fd: c_int, cb: MxmlSaveCb) -> c_int {
    let col: c_int;
    let mut buf: MxmlFdbuf = core::mem::zeroed();
    let global: *mut MxmlGlobal = mxml_global();

    /*
     * Initialize the file descriptor buffer...
     */

    buf.fd = fd;
    buf.current = buf.buffer.as_mut_ptr();
    buf.end = buf
        .buffer
        .as_mut_ptr()
        .add(core::mem::size_of::<[c_uchar; 8192]>());

    /*
     * Write the node...
     */

    col = mxml_write_node(
        node,
        &raw mut buf as *mut c_void,
        cb,
        0,
        Some(mxml_fd_putc),
        global,
    );
    if col < 0 {
        return -1;
    }

    if col > 0 && mxml_fd_putc('\n' as c_int, &raw mut buf as *mut c_void) < 0 {
        return -1;
    }

    /*
     * Flush and return...
     */

    mxml_fd_write(&raw mut buf)
}

/// Matches C `mxmlSaveFile` (`mxml-file.c:262`).
pub unsafe fn mxml_save_file(node: *mut MxmlNode, fp: *mut libc::FILE, cb: MxmlSaveCb) -> c_int {
    let col: c_int;
    let global: *mut MxmlGlobal = mxml_global();

    /*
     * Write the node...
     */

    col = mxml_write_node(node, fp as *mut c_void, cb, 0, Some(mxml_file_putc), global);
    if col < 0 {
        return -1;
    }

    if col > 0 && libc::fputc('\n' as c_int, fp) < 0 {
        return -1;
    }

    /*
     * Return 0 (success)...
     */

    0
}

/// Matches C `mxmlSaveString` (`mxml-file.c:299`).
pub unsafe fn mxml_save_string(
    node: *mut MxmlNode,
    buffer: *mut c_char,
    bufsize: c_int,
    cb: MxmlSaveCb,
) -> c_int {
    let col: c_int;
    let mut ptr: [*mut c_char; 2] = [core::ptr::null_mut(); 2];
    let global: *mut MxmlGlobal = mxml_global();

    /*
     * Write the node...
     */

    ptr[0] = buffer;
    ptr[1] = buffer.add(bufsize as usize);

    col = mxml_write_node(
        node,
        ptr.as_mut_ptr() as *mut c_void,
        cb,
        0,
        Some(mxml_string_putc),
        global,
    );
    if col < 0 {
        return -1;
    }

    if col > 0 {
        mxml_string_putc('\n' as c_int, ptr.as_mut_ptr() as *mut c_void);
    }

    /*
     * Nul-terminate the buffer...
     */

    if ptr[0] >= ptr[1] {
        *buffer.add((bufsize - 1) as usize) = 0;
    } else {
        *ptr[0] = 0;
    }

    /*
     * Return the number of characters...
     */

    (ptr[0] as usize - buffer as usize) as c_int
}

/// Matches C `mxmlSAXLoadFd` (`mxml-file.c:349`).
pub unsafe fn mxml_sax_load_fd(
    top: *mut MxmlNode,
    fd: c_int,
    cb: MxmlLoadCb,
    sax_cb: MxmlSaxCb,
    sax_data: *mut c_void,
) -> *mut MxmlNode {
    let mut buf: MxmlFdbuf = core::mem::zeroed();

    /*
     * Initialize the file descriptor buffer...
     */

    buf.fd = fd;
    buf.current = buf.buffer.as_mut_ptr();
    buf.end = buf.buffer.as_mut_ptr();

    /*
     * Read the XML data...
     */

    mxml_load_data(
        top,
        &raw mut buf as *mut c_void,
        cb,
        Some(mxml_fd_getc),
        sax_cb,
        sax_data,
    )
}

/// Matches C `mxmlSAXLoadFile` (`mxml-file.c:391`).
pub unsafe fn mxml_sax_load_file(
    top: *mut MxmlNode,
    fp: *mut libc::FILE,
    cb: MxmlLoadCb,
    sax_cb: MxmlSaxCb,
    sax_data: *mut c_void,
) -> *mut MxmlNode {
    /*
     * Read the XML data...
     */

    mxml_load_data(
        top,
        fp as *mut c_void,
        cb,
        Some(mxml_file_getc),
        sax_cb,
        sax_data,
    )
}

/// Matches C `mxmlSAXLoadString` (`mxml-file.c:429`).
pub unsafe fn mxml_sax_load_string(
    top: *mut MxmlNode,
    s: *const c_char,
    cb: MxmlLoadCb,
    sax_cb: MxmlSaxCb,
    sax_data: *mut c_void,
) -> *mut MxmlNode {
    let mut sp: *const c_char = s;

    /*
     * Read the XML data...
     */

    mxml_load_data(
        top,
        &raw mut sp as *mut c_void,
        cb,
        Some(mxml_string_getc),
        sax_cb,
        sax_data,
    )
}

/// Matches C `mxmlSetCustomHandlers` (`mxml-file.c:453`).
pub unsafe fn mxml_set_custom_handlers(load: MxmlCustomLoadCb, save: MxmlCustomSaveCb) {
    let global: *mut MxmlGlobal = mxml_global();

    (*global).custom_load_cb = load;
    (*global).custom_save_cb = save;
}

/// Matches C `mxmlSetErrorCallback` (`mxml-file.c:469`).
pub unsafe fn mxml_set_error_callback(cb: MxmlErrorCb) {
    let global: *mut MxmlGlobal = mxml_global();

    (*global).error_cb = cb;
}

/// Matches C `mxmlSetWrapMargin` (`mxml-file.c:484`).
pub unsafe fn mxml_set_wrap_margin(column: c_int) {
    let global: *mut MxmlGlobal = mxml_global();

    (*global).wrap = column;
}

/// Matches C static `mxml_add_char` (`mxml-file.c:496`).
pub unsafe fn mxml_add_char(
    ch: c_int,
    bufptr: *mut *mut c_char,
    buffer: *mut *mut c_char,
    bufsize: *mut c_int,
) -> c_int {
    let newbuffer: *mut c_char;

    if *bufptr >= (*buffer).add((*bufsize - 4) as usize) {
        /*
         * Increase the size of the buffer...
         */

        if *bufsize < 1024 {
            *bufsize *= 2;
        } else {
            *bufsize += 1024;
        }

        newbuffer = libc::realloc(*buffer as *mut c_void, *bufsize as usize) as *mut c_char;
        if newbuffer.is_null() {
            libc::free(*buffer as *mut c_void);

            let mut msg: [c_char; 1024] = [0; 1024];
            libc::snprintf(
                msg.as_mut_ptr(),
                core::mem::size_of::<[c_char; 1024]>(),
                c"Unable to expand string buffer to %d bytes!".as_ptr(),
                *bufsize,
            );
            mxml_error(msg.as_ptr());

            return -1;
        }

        *bufptr = newbuffer.add(*bufptr as usize - *buffer as usize);
        *buffer = newbuffer;
    }

    /*
     * Nul-terminate the buffer as needed...
     */

    if ch < 0x80 {
        /*
         * Single byte ASCII...
         */

        **bufptr = ch as c_char;
        *bufptr = (*bufptr).add(1);
    } else if ch < 0x800 {
        /*
         * Two-byte UTF-8...
         */

        **bufptr = (0xc0 | (ch >> 6)) as c_char;
        *bufptr = (*bufptr).add(1);
        **bufptr = (0x80 | (ch & 0x3f)) as c_char;
        *bufptr = (*bufptr).add(1);
    } else if ch < 0x10000 {
        /*
         * Three-byte UTF-8...
         */

        **bufptr = (0xe0 | (ch >> 12)) as c_char;
        *bufptr = (*bufptr).add(1);
        **bufptr = (0x80 | ((ch >> 6) & 0x3f)) as c_char;
        *bufptr = (*bufptr).add(1);
        **bufptr = (0x80 | (ch & 0x3f)) as c_char;
        *bufptr = (*bufptr).add(1);
    } else {
        /*
         * Four-byte UTF-8...
         */

        **bufptr = (0xf0 | (ch >> 18)) as c_char;
        *bufptr = (*bufptr).add(1);
        **bufptr = (0x80 | ((ch >> 12) & 0x3f)) as c_char;
        *bufptr = (*bufptr).add(1);
        **bufptr = (0x80 | ((ch >> 6) & 0x3f)) as c_char;
        *bufptr = (*bufptr).add(1);
        **bufptr = (0x80 | (ch & 0x3f)) as c_char;
        *bufptr = (*bufptr).add(1);
    }

    0
}

/// Matches C static `mxml_fd_getc` (`mxml-file.c:568`).
pub unsafe extern "C" fn mxml_fd_getc(p: *mut c_void, encoding: *mut c_int) -> c_int {
    let buf: *mut MxmlFdbuf;
    let mut ch: c_int;
    let mut temp: c_int;

    /*
     * Get the next character...
     */

    buf = p as *mut MxmlFdbuf;
    if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
        return EOF;
    }

    ch = *(*buf).current as c_int;
    (*buf).current = (*buf).current.add(1);

    match *encoding {
        ENCODE_UTF8 => {
            /*
             * Got a UTF-8 character; convert UTF-8 to Unicode and return...
             */

            if (ch & 0x80) == 0 {
                /*
                 * ASCII
                 */

                if ch < b' ' as c_int
                    && ch != b'\n' as c_int
                    && ch != b'\r' as c_int
                    && ch != b'\t' as c_int
                {
                    let mut msg: [c_char; 1024] = [0; 1024];
                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Bad control character 0x%02x not allowed by XML standard!".as_ptr(),
                        ch,
                    );
                    mxml_error(msg.as_ptr());
                    return EOF;
                }

                return ch;
            } else if ch == 0xfe {
                /*
                 * UTF-16 big-endian BOM?
                 */

                if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                ch = *(*buf).current as c_int;
                (*buf).current = (*buf).current.add(1);

                if ch != 0xff {
                    return EOF;
                }

                *encoding = ENCODE_UTF16BE;

                return mxml_fd_getc(p, encoding);
            } else if ch == 0xff {
                /*
                 * UTF-16 little-endian BOM?
                 */

                if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                ch = *(*buf).current as c_int;
                (*buf).current = (*buf).current.add(1);

                if ch != 0xfe {
                    return EOF;
                }

                *encoding = ENCODE_UTF16LE;

                return mxml_fd_getc(p, encoding);
            } else if (ch & 0xe0) == 0xc0 {
                /*
                 * Two-byte value...
                 */

                if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = *(*buf).current as c_int;
                (*buf).current = (*buf).current.add(1);

                if (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = ((ch & 0x1f) << 6) | (temp & 0x3f);

                if ch < 0x80 {
                    let mut msg: [c_char; 1024] = [0; 1024];
                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Invalid UTF-8 sequence for character 0x%04x!".as_ptr(),
                        ch,
                    );
                    mxml_error(msg.as_ptr());
                    return EOF;
                }
            } else if (ch & 0xf0) == 0xe0 {
                /*
                 * Three-byte value...
                 */

                if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = *(*buf).current as c_int;
                (*buf).current = (*buf).current.add(1);

                if (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = ((ch & 0x0f) << 6) | (temp & 0x3f);

                if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = *(*buf).current as c_int;
                (*buf).current = (*buf).current.add(1);

                if (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = (ch << 6) | (temp & 0x3f);

                if ch < 0x800 {
                    let mut msg: [c_char; 1024] = [0; 1024];
                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Invalid UTF-8 sequence for character 0x%04x!".as_ptr(),
                        ch,
                    );
                    mxml_error(msg.as_ptr());
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

                if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = *(*buf).current as c_int;
                (*buf).current = (*buf).current.add(1);

                if (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = ((ch & 0x07) << 6) | (temp & 0x3f);

                if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = *(*buf).current as c_int;
                (*buf).current = (*buf).current.add(1);

                if (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = (ch << 6) | (temp & 0x3f);

                if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = *(*buf).current as c_int;
                (*buf).current = (*buf).current.add(1);

                if (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = (ch << 6) | (temp & 0x3f);

                if ch < 0x10000 {
                    let mut msg: [c_char; 1024] = [0; 1024];
                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Invalid UTF-8 sequence for character 0x%04x!".as_ptr(),
                        ch,
                    );
                    mxml_error(msg.as_ptr());
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

            if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
                return EOF;
            }

            temp = *(*buf).current as c_int;
            (*buf).current = (*buf).current.add(1);

            ch = (ch << 8) | temp;

            if ch < b' ' as c_int
                && ch != b'\n' as c_int
                && ch != b'\r' as c_int
                && ch != b'\t' as c_int
            {
                let mut msg: [c_char; 1024] = [0; 1024];
                libc::snprintf(
                    msg.as_mut_ptr(),
                    core::mem::size_of::<[c_char; 1024]>(),
                    c"Bad control character 0x%02x not allowed by XML standard!".as_ptr(),
                    ch,
                );
                mxml_error(msg.as_ptr());
                return EOF;
            } else if ch >= 0xd800 && ch <= 0xdbff {
                /*
                 * Multi-word UTF-16 char...
                 */

                let mut lch: c_int;

                if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                lch = *(*buf).current as c_int;
                (*buf).current = (*buf).current.add(1);

                if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = *(*buf).current as c_int;
                (*buf).current = (*buf).current.add(1);

                lch = (lch << 8) | temp;

                if lch < 0xdc00 || lch >= 0xdfff {
                    return EOF;
                }

                ch = (((ch & 0x3ff) << 10) | (lch & 0x3ff)) + 0x10000;
            }
        }

        ENCODE_UTF16LE => {
            /*
             * Read UTF-16 little-endian char...
             */

            if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
                return EOF;
            }

            temp = *(*buf).current as c_int;
            (*buf).current = (*buf).current.add(1);

            ch |= temp << 8;

            if ch < b' ' as c_int
                && ch != b'\n' as c_int
                && ch != b'\r' as c_int
                && ch != b'\t' as c_int
            {
                let mut msg: [c_char; 1024] = [0; 1024];
                libc::snprintf(
                    msg.as_mut_ptr(),
                    core::mem::size_of::<[c_char; 1024]>(),
                    c"Bad control character 0x%02x not allowed by XML standard!".as_ptr(),
                    ch,
                );
                mxml_error(msg.as_ptr());
                return EOF;
            } else if ch >= 0xd800 && ch <= 0xdbff {
                /*
                 * Multi-word UTF-16 char...
                 */

                let mut lch: c_int;

                if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                lch = *(*buf).current as c_int;
                (*buf).current = (*buf).current.add(1);

                if (*buf).current >= (*buf).end && mxml_fd_read(buf) < 0 {
                    return EOF;
                }

                temp = *(*buf).current as c_int;
                (*buf).current = (*buf).current.add(1);

                lch |= temp << 8;

                if lch < 0xdc00 || lch >= 0xdfff {
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
pub unsafe extern "C" fn mxml_fd_putc(ch: c_int, p: *mut c_void) -> c_int {
    let buf: *mut MxmlFdbuf;

    /*
     * Flush the write buffer as needed - note above that "end" still indicates
     * the end of the buffer...
     */

    buf = p as *mut MxmlFdbuf;
    if (*buf).current >= (*buf).end && mxml_fd_write(buf) < 0 {
        return -1;
    }

    *(*buf).current = ch as c_uchar;
    (*buf).current = (*buf).current.add(1);

    /*
     * Return successfully...
     */

    0
}

/// Matches C static `mxml_fd_read` (`mxml-file.c:838`).
pub unsafe fn mxml_fd_read(buf: *mut MxmlFdbuf) -> c_int {
    let mut bytes: isize;

    /*
     * Range check input...
     */

    if buf.is_null() {
        return -1;
    }

    /*
     * Read from the file descriptor...
     */

    loop {
        bytes = libc::read(
            (*buf).fd,
            (*buf).buffer.as_mut_ptr() as *mut c_void,
            core::mem::size_of::<[c_uchar; 8192]>(),
        );
        if bytes >= 0 {
            break;
        }
        let err = *libc::__errno_location();
        if err != libc::EAGAIN && err != libc::EINTR {
            return -1;
        }
    }

    if bytes == 0 {
        return -1;
    }

    /*
     * Update the pointers and return success...
     */

    (*buf).current = (*buf).buffer.as_mut_ptr();
    (*buf).end = (*buf).buffer.as_mut_ptr().add(bytes as usize);

    0
}

/// Matches C static `mxml_fd_write` (`mxml-file.c:879`).
pub unsafe fn mxml_fd_write(buf: *mut MxmlFdbuf) -> c_int {
    let mut bytes: isize;
    let mut ptr: *mut c_uchar;

    /*
     * Range check...
     */

    if buf.is_null() {
        return -1;
    }

    /*
     * Return 0 if there is nothing to write...
     */

    if (*buf).current == (*buf).buffer.as_mut_ptr() {
        return 0;
    }

    /*
     * Loop until we have written everything...
     */

    ptr = (*buf).buffer.as_mut_ptr();
    while ptr < (*buf).current {
        bytes = libc::write(
            (*buf).fd,
            ptr as *const c_void,
            (*buf).current as usize - ptr as usize,
        );
        if bytes < 0 {
            return -1;
        }
        ptr = ptr.add(bytes as usize);
    }

    /*
     * All done, reset pointers and return success...
     */

    (*buf).current = (*buf).buffer.as_mut_ptr();

    0
}

/// Matches C static `mxml_file_getc` (`mxml-file.c:920`).
pub unsafe extern "C" fn mxml_file_getc(p: *mut c_void, encoding: *mut c_int) -> c_int {
    let mut ch: c_int;
    let mut temp: c_int;
    let fp: *mut libc::FILE;

    /*
     * Read a character from the file and see if it is EOF or ASCII...
     */

    fp = p as *mut libc::FILE;
    ch = libc::fgetc(fp);

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

                if ch < b' ' as c_int
                    && ch != b'\n' as c_int
                    && ch != b'\r' as c_int
                    && ch != b'\t' as c_int
                {
                    let mut msg: [c_char; 1024] = [0; 1024];
                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Bad control character 0x%02x not allowed by XML standard!".as_ptr(),
                        ch,
                    );
                    mxml_error(msg.as_ptr());
                    return EOF;
                }

                return ch;
            } else if ch == 0xfe {
                /*
                 * UTF-16 big-endian BOM?
                 */

                ch = libc::fgetc(fp);
                if ch != 0xff {
                    return EOF;
                }

                *encoding = ENCODE_UTF16BE;

                return mxml_file_getc(p, encoding);
            } else if ch == 0xff {
                /*
                 * UTF-16 little-endian BOM?
                 */

                ch = libc::fgetc(fp);
                if ch != 0xfe {
                    return EOF;
                }

                *encoding = ENCODE_UTF16LE;

                return mxml_file_getc(p, encoding);
            } else if (ch & 0xe0) == 0xc0 {
                /*
                 * Two-byte value...
                 */

                temp = libc::fgetc(fp);
                if temp == EOF || (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = ((ch & 0x1f) << 6) | (temp & 0x3f);

                if ch < 0x80 {
                    let mut msg: [c_char; 1024] = [0; 1024];
                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Invalid UTF-8 sequence for character 0x%04x!".as_ptr(),
                        ch,
                    );
                    mxml_error(msg.as_ptr());
                    return EOF;
                }
            } else if (ch & 0xf0) == 0xe0 {
                /*
                 * Three-byte value...
                 */

                temp = libc::fgetc(fp);
                if temp == EOF || (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = ((ch & 0x0f) << 6) | (temp & 0x3f);

                temp = libc::fgetc(fp);
                if temp == EOF || (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = (ch << 6) | (temp & 0x3f);

                if ch < 0x800 {
                    let mut msg: [c_char; 1024] = [0; 1024];
                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Invalid UTF-8 sequence for character 0x%04x!".as_ptr(),
                        ch,
                    );
                    mxml_error(msg.as_ptr());
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

                temp = libc::fgetc(fp);
                if temp == EOF || (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = ((ch & 0x07) << 6) | (temp & 0x3f);

                temp = libc::fgetc(fp);
                if temp == EOF || (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = (ch << 6) | (temp & 0x3f);

                temp = libc::fgetc(fp);
                if temp == EOF || (temp & 0xc0) != 0x80 {
                    return EOF;
                }

                ch = (ch << 6) | (temp & 0x3f);

                if ch < 0x10000 {
                    let mut msg: [c_char; 1024] = [0; 1024];
                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Invalid UTF-8 sequence for character 0x%04x!".as_ptr(),
                        ch,
                    );
                    mxml_error(msg.as_ptr());
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

            ch = (ch << 8) | libc::fgetc(fp);

            if ch < b' ' as c_int
                && ch != b'\n' as c_int
                && ch != b'\r' as c_int
                && ch != b'\t' as c_int
            {
                let mut msg: [c_char; 1024] = [0; 1024];
                libc::snprintf(
                    msg.as_mut_ptr(),
                    core::mem::size_of::<[c_char; 1024]>(),
                    c"Bad control character 0x%02x not allowed by XML standard!".as_ptr(),
                    ch,
                );
                mxml_error(msg.as_ptr());
                return EOF;
            } else if ch >= 0xd800 && ch <= 0xdbff {
                /*
                 * Multi-word UTF-16 char...
                 */

                let mut lch: c_int = libc::fgetc(fp);
                lch = (lch << 8) | libc::fgetc(fp);

                if lch < 0xdc00 || lch >= 0xdfff {
                    return EOF;
                }

                ch = (((ch & 0x3ff) << 10) | (lch & 0x3ff)) + 0x10000;
            }
        }

        ENCODE_UTF16LE => {
            /*
             * Read UTF-16 little-endian char...
             */

            ch |= libc::fgetc(fp) << 8;

            if ch < b' ' as c_int
                && ch != b'\n' as c_int
                && ch != b'\r' as c_int
                && ch != b'\t' as c_int
            {
                let mut msg: [c_char; 1024] = [0; 1024];
                libc::snprintf(
                    msg.as_mut_ptr(),
                    core::mem::size_of::<[c_char; 1024]>(),
                    c"Bad control character 0x%02x not allowed by XML standard!".as_ptr(),
                    ch,
                );
                mxml_error(msg.as_ptr());
                return EOF;
            } else if ch >= 0xd800 && ch <= 0xdbff {
                /*
                 * Multi-word UTF-16 char...
                 */

                let mut lch: c_int = libc::fgetc(fp);
                lch |= libc::fgetc(fp) << 8;

                if lch < 0xdc00 || lch >= 0xdfff {
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
pub unsafe extern "C" fn mxml_file_putc(ch: c_int, p: *mut c_void) -> c_int {
    if libc::fputc(ch, p as *mut libc::FILE) == EOF {
        -1
    } else {
        0
    }
}

/// Matches C static `mxml_get_entity` (`mxml-file.c:1114`).
pub unsafe fn mxml_get_entity(
    parent: *mut MxmlNode,
    p: *mut c_void,
    encoding: *mut c_int,
    getc_cb: MxmlGetcCb,
) -> c_int {
    let mut ch: c_int;
    let mut entity: [c_char; 64] = [0; 64];
    let mut entptr: *mut c_char;

    entptr = entity.as_mut_ptr();

    loop {
        ch = getc_cb.unwrap()(p, encoding);
        if ch == EOF {
            break;
        }
        if ch > 126 || (libc::isalnum(ch) == 0 && ch != b'#' as c_int) {
            break;
        } else if entptr
            < entity
                .as_mut_ptr()
                .add(core::mem::size_of::<[c_char; 64]>() - 1)
        {
            *entptr = ch as c_char;
            entptr = entptr.add(1);
        } else {
            let mut msg: [c_char; 1024] = [0; 1024];
            libc::snprintf(
                msg.as_mut_ptr(),
                core::mem::size_of::<[c_char; 1024]>(),
                c"Entity name too long under parent <%s>!".as_ptr(),
                if !parent.is_null() {
                    (*parent).value.element.name as *const c_char
                } else {
                    c"null".as_ptr()
                },
            );
            mxml_error(msg.as_ptr());
            break;
        }
    }

    *entptr = 0;

    if ch != b';' as c_int {
        let mut msg: [c_char; 1024] = [0; 1024];
        libc::snprintf(
            msg.as_mut_ptr(),
            core::mem::size_of::<[c_char; 1024]>(),
            c"Character entity \"%s\" not terminated under parent <%s>!".as_ptr(),
            entity.as_ptr(),
            if !parent.is_null() {
                (*parent).value.element.name as *const c_char
            } else {
                c"null".as_ptr()
            },
        );
        mxml_error(msg.as_ptr());
        return EOF;
    }

    if entity[0] == b'#' as c_char {
        if entity[1] == b'x' as c_char {
            ch = libc::strtol(entity.as_ptr().add(2), core::ptr::null_mut(), 16) as c_int;
        } else {
            ch = libc::strtol(entity.as_ptr().add(1), core::ptr::null_mut(), 10) as c_int;
        }
    } else {
        ch = mxml_entity_get_value(entity.as_ptr());
        if ch < 0 {
            let mut msg: [c_char; 1024] = [0; 1024];
            libc::snprintf(
                msg.as_mut_ptr(),
                core::mem::size_of::<[c_char; 1024]>(),
                c"Entity name \"%s;\" not supported under parent <%s>!".as_ptr(),
                entity.as_ptr(),
                if !parent.is_null() {
                    (*parent).value.element.name as *const c_char
                } else {
                    c"null".as_ptr()
                },
            );
            mxml_error(msg.as_ptr());
        }
    }

    if ch < b' ' as c_int && ch != b'\n' as c_int && ch != b'\r' as c_int && ch != b'\t' as c_int {
        let mut msg: [c_char; 1024] = [0; 1024];
        libc::snprintf(
            msg.as_mut_ptr(),
            core::mem::size_of::<[c_char; 1024]>(),
            c"Bad control character 0x%02x under parent <%s> not allowed by XML standard!".as_ptr(),
            ch,
            if !parent.is_null() {
                (*parent).value.element.name as *const c_char
            } else {
                c"null".as_ptr()
            },
        );
        mxml_error(msg.as_ptr());
        return EOF;
    }

    ch
}

/// Matches C static inline `mxml_isspace` (`mxml-file.c:60`).
pub unsafe fn mxml_isspace(ch: c_int) -> c_int {
    (ch == b' ' as c_int || ch == b'\t' as c_int || ch == b'\r' as c_int || ch == b'\n' as c_int)
        as c_int
}

/// Matches C static `mxml_load_data` (`mxml-file.c:1191`).
pub unsafe fn mxml_load_data(
    top: *mut MxmlNode,
    p: *mut c_void,
    cb: MxmlLoadCb,
    getc_cb: MxmlGetcCb,
    sax_cb: MxmlSaxCb,
    sax_data: *mut c_void,
) -> *mut MxmlNode {
    let mut node: *mut MxmlNode;
    let mut first: *mut MxmlNode;
    let mut parent: *mut MxmlNode;
    let mut ch: c_int;
    let mut whitespace: c_int;
    let mut buffer: *mut c_char;
    let mut bufptr: *mut c_char;
    let mut bufsize: c_int;
    let mut type_: MxmlType;
    let mut encoding: c_int;
    let global: *mut MxmlGlobal = mxml_global();
    static TYPES: [&core::ffi::CStr; 6] = [
        c"MXML_ELEMENT",
        c"MXML_INTEGER",
        c"MXML_OPAQUE",
        c"MXML_REAL",
        c"MXML_TEXT",
        c"MXML_CUSTOM",
    ];
    let mut msg: [c_char; 1024] = [0; 1024];

    /*
     * Read elements and other nodes from the file...
     */

    buffer = libc::malloc(64) as *mut c_char;
    if buffer.is_null() {
        mxml_error(c"Unable to allocate string buffer!".as_ptr());
        return core::ptr::null_mut();
    }

    bufsize = 64;
    bufptr = buffer;
    parent = top;
    first = core::ptr::null_mut();
    whitespace = 0;
    encoding = ENCODE_UTF8;

    if cb.is_some() && !parent.is_null() {
        type_ = cb.unwrap()(parent);
    } else if !parent.is_null() {
        type_ = MXML_TEXT;
    } else {
        type_ = MXML_IGNORE;
    }

    'error: {
        loop {
            ch = getc_cb.unwrap()(p, &raw mut encoding);
            if ch == EOF {
                break;
            }

            if (ch == b'<' as c_int
                || (mxml_isspace(ch) != 0 && type_ != MXML_OPAQUE && type_ != MXML_CUSTOM))
                && bufptr > buffer
            {
                /*
                 * Add a new value node...
                 */

                *bufptr = 0;

                match type_ {
                    MXML_INTEGER => {
                        node = mxml_new_integer(
                            parent,
                            libc::strtol(buffer, &raw mut bufptr, 0) as c_int,
                        );
                    }

                    MXML_OPAQUE => {
                        node = mxml_new_opaque(parent, buffer);
                    }

                    MXML_REAL => {
                        node = mxml_new_real(parent, libc::strtod(buffer, &raw mut bufptr));
                    }

                    MXML_TEXT => {
                        node = mxml_new_text(parent, whitespace, buffer);
                    }

                    MXML_CUSTOM if (*global).custom_load_cb.is_some() => {
                        /*
                         * Use the callback to fill in the custom data...
                         */

                        node = mxml_new_custom(parent, core::ptr::null_mut(), None);

                        if (*global).custom_load_cb.unwrap()(node, buffer) != 0 {
                            libc::snprintf(
                                msg.as_mut_ptr(),
                                core::mem::size_of::<[c_char; 1024]>(),
                                c"Bad custom value '%s' in parent <%s>!".as_ptr(),
                                buffer,
                                if !parent.is_null() {
                                    (*parent).value.element.name as *const c_char
                                } else {
                                    c"null".as_ptr()
                                },
                            );
                            mxml_error(msg.as_ptr());
                            mxml_delete(node);
                            node = core::ptr::null_mut();
                        }
                    }

                    _ => {
                        node = core::ptr::null_mut();
                    }
                }

                if *bufptr != 0 {
                    /*
                     * Bad integer/real number value...
                     */

                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Bad %s value '%s' in parent <%s>!".as_ptr(),
                        if type_ == MXML_INTEGER {
                            c"integer".as_ptr()
                        } else {
                            c"real".as_ptr()
                        },
                        buffer,
                        if !parent.is_null() {
                            (*parent).value.element.name as *const c_char
                        } else {
                            c"null".as_ptr()
                        },
                    );
                    mxml_error(msg.as_ptr());
                    break;
                }

                bufptr = buffer;
                whitespace = (mxml_isspace(ch) != 0 && type_ == MXML_TEXT) as c_int;

                if node.is_null() && type_ != MXML_IGNORE {
                    /*
                     * Print error and return...
                     */

                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Unable to add value node of type %s to parent <%s>!".as_ptr(),
                        TYPES[type_ as usize].as_ptr(),
                        if !parent.is_null() {
                            (*parent).value.element.name as *const c_char
                        } else {
                            c"null".as_ptr()
                        },
                    );
                    mxml_error(msg.as_ptr());
                    break 'error;
                }

                if let Some(sax) = sax_cb {
                    sax(node, MXML_SAX_DATA, sax_data);

                    if mxml_release(node) == 0 {
                        node = core::ptr::null_mut();
                    }
                }

                if first.is_null() && !node.is_null() {
                    first = node;
                }
            } else if mxml_isspace(ch) != 0 && type_ == MXML_TEXT {
                whitespace = 1;
            }

            /*
             * Add lone whitespace node if we have an element and existing
             * whitespace...
             */

            if ch == b'<' as c_int && whitespace != 0 && type_ == MXML_TEXT {
                if !parent.is_null() {
                    node = mxml_new_text(parent, whitespace, c"".as_ptr());

                    if let Some(sax) = sax_cb {
                        sax(node, MXML_SAX_DATA, sax_data);

                        if mxml_release(node) == 0 {
                            node = core::ptr::null_mut();
                        }
                    }

                    if first.is_null() && !node.is_null() {
                        first = node;
                    }
                }

                whitespace = 0;
            }

            if ch == b'<' as c_int {
                /*
                 * Start of open/close tag...
                 */

                bufptr = buffer;

                loop {
                    ch = getc_cb.unwrap()(p, &raw mut encoding);
                    if ch == EOF {
                        break;
                    }

                    if mxml_isspace(ch) != 0
                        || ch == b'>' as c_int
                        || (ch == b'/' as c_int && bufptr > buffer)
                    {
                        break;
                    } else if ch == b'<' as c_int {
                        mxml_error(c"Bare < in element!".as_ptr());
                        break 'error;
                    } else if ch == b'&' as c_int {
                        ch = mxml_get_entity(parent, p, &raw mut encoding, getc_cb);
                        if ch == EOF {
                            break 'error;
                        }

                        if mxml_add_char(ch, &raw mut bufptr, &raw mut buffer, &raw mut bufsize)
                            != 0
                        {
                            break 'error;
                        }
                    } else if ch < b'0' as c_int
                        && ch != b'!' as c_int
                        && ch != b'-' as c_int
                        && ch != b'.' as c_int
                        && ch != b'/' as c_int
                    {
                        break 'error;
                    } else if mxml_add_char(ch, &raw mut bufptr, &raw mut buffer, &raw mut bufsize)
                        != 0
                    {
                        break 'error;
                    } else if ((bufptr as usize - buffer as usize) == 1
                        && *buffer == b'?' as c_char)
                        || ((bufptr as usize - buffer as usize) == 3
                            && libc::strncmp(buffer, c"!--".as_ptr(), 3) == 0)
                        || ((bufptr as usize - buffer as usize) == 8
                            && libc::strncmp(buffer, c"![CDATA[".as_ptr(), 8) == 0)
                    {
                        break;
                    }
                }

                *bufptr = 0;

                if libc::strcmp(buffer, c"!--".as_ptr()) == 0 {
                    /*
                     * Gather rest of comment...
                     */

                    loop {
                        ch = getc_cb.unwrap()(p, &raw mut encoding);
                        if ch == EOF {
                            break;
                        }

                        if ch == b'>' as c_int
                            && bufptr > buffer.add(4)
                            && *bufptr.offset(-3) != b'-' as c_char
                            && *bufptr.offset(-2) == b'-' as c_char
                            && *bufptr.offset(-1) == b'-' as c_char
                        {
                            break;
                        } else if mxml_add_char(
                            ch,
                            &raw mut bufptr,
                            &raw mut buffer,
                            &raw mut bufsize,
                        ) != 0
                        {
                            break 'error;
                        }
                    }

                    /*
                     * Error out if we didn't get the whole comment...
                     */

                    if ch != b'>' as c_int {
                        /*
                         * Print error and return...
                         */

                        mxml_error(c"Early EOF in comment node!".as_ptr());
                        break 'error;
                    }

                    /*
                     * Otherwise add this as an element under the current parent...
                     */

                    *bufptr = 0;

                    if parent.is_null() && !first.is_null() {
                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"<%s> cannot be a second root node after <%s>".as_ptr(),
                            buffer,
                            (*first).value.element.name,
                        );
                        mxml_error(msg.as_ptr());
                        break 'error;
                    }

                    node = mxml_new_element(parent, buffer);
                    if node.is_null() {
                        /*
                         * Just print error for now...
                         */

                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"Unable to add comment node to parent <%s>!".as_ptr(),
                            if !parent.is_null() {
                                (*parent).value.element.name as *const c_char
                            } else {
                                c"null".as_ptr()
                            },
                        );
                        mxml_error(msg.as_ptr());
                        break;
                    }

                    if let Some(sax) = sax_cb {
                        sax(node, MXML_SAX_COMMENT, sax_data);

                        if mxml_release(node) == 0 {
                            node = core::ptr::null_mut();
                        }
                    }

                    if !node.is_null() && first.is_null() {
                        first = node;
                    }
                } else if libc::strcmp(buffer, c"![CDATA[".as_ptr()) == 0 {
                    /*
                     * Gather CDATA section...
                     */

                    loop {
                        ch = getc_cb.unwrap()(p, &raw mut encoding);
                        if ch == EOF {
                            break;
                        }

                        if ch == b'>' as c_int
                            && libc::strncmp(bufptr.offset(-2), c"]]".as_ptr(), 2) == 0
                        {
                            break;
                        } else if mxml_add_char(
                            ch,
                            &raw mut bufptr,
                            &raw mut buffer,
                            &raw mut bufsize,
                        ) != 0
                        {
                            break 'error;
                        }
                    }

                    /*
                     * Error out if we didn't get the whole comment...
                     */

                    if ch != b'>' as c_int {
                        /*
                         * Print error and return...
                         */

                        mxml_error(c"Early EOF in CDATA node!".as_ptr());
                        break 'error;
                    }

                    /*
                     * Otherwise add this as an element under the current parent...
                     */

                    *bufptr = 0;

                    if parent.is_null() && !first.is_null() {
                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"<%s> cannot be a second root node after <%s>".as_ptr(),
                            buffer,
                            (*first).value.element.name,
                        );
                        mxml_error(msg.as_ptr());
                        break 'error;
                    }

                    node = mxml_new_element(parent, buffer);
                    if node.is_null() {
                        /*
                         * Print error and return...
                         */

                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"Unable to add CDATA node to parent <%s>!".as_ptr(),
                            if !parent.is_null() {
                                (*parent).value.element.name as *const c_char
                            } else {
                                c"null".as_ptr()
                            },
                        );
                        mxml_error(msg.as_ptr());
                        break 'error;
                    }

                    if let Some(sax) = sax_cb {
                        sax(node, MXML_SAX_CDATA, sax_data);

                        if mxml_release(node) == 0 {
                            node = core::ptr::null_mut();
                        }
                    }

                    if !node.is_null() && first.is_null() {
                        first = node;
                    }
                } else if *buffer == b'?' as c_char {
                    /*
                     * Gather rest of processing instruction...
                     */

                    loop {
                        ch = getc_cb.unwrap()(p, &raw mut encoding);
                        if ch == EOF {
                            break;
                        }

                        if ch == b'>' as c_int
                            && bufptr > buffer
                            && *bufptr.offset(-1) == b'?' as c_char
                        {
                            break;
                        } else if mxml_add_char(
                            ch,
                            &raw mut bufptr,
                            &raw mut buffer,
                            &raw mut bufsize,
                        ) != 0
                        {
                            break 'error;
                        }
                    }

                    /*
                     * Error out if we didn't get the whole processing instruction...
                     */

                    if ch != b'>' as c_int {
                        /*
                         * Print error and return...
                         */

                        mxml_error(c"Early EOF in processing instruction node!".as_ptr());
                        break 'error;
                    }

                    /*
                     * Otherwise add this as an element under the current parent...
                     */

                    *bufptr = 0;

                    if parent.is_null() && !first.is_null() {
                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"<%s> cannot be a second root node after <%s>".as_ptr(),
                            buffer,
                            (*first).value.element.name,
                        );
                        mxml_error(msg.as_ptr());
                        break 'error;
                    }

                    node = mxml_new_element(parent, buffer);
                    if node.is_null() {
                        /*
                         * Print error and return...
                         */

                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"Unable to add processing instruction node to parent <%s>!".as_ptr(),
                            if !parent.is_null() {
                                (*parent).value.element.name as *const c_char
                            } else {
                                c"null".as_ptr()
                            },
                        );
                        mxml_error(msg.as_ptr());
                        break 'error;
                    }

                    if let Some(sax) = sax_cb {
                        sax(node, MXML_SAX_DIRECTIVE, sax_data);

                        if mxml_release(node) == 0 {
                            node = core::ptr::null_mut();
                        }
                    }

                    if !node.is_null() {
                        if first.is_null() {
                            first = node;
                        }

                        if parent.is_null() {
                            /*
                             * Got the XML declaration, so add it as the root node...
                             */

                            parent = node;

                            if cb.is_some() {
                                type_ = cb.unwrap()(parent);
                            } else {
                                type_ = MXML_TEXT;
                            }
                        }
                    }
                } else if *buffer == b'!' as c_char {
                    /*
                     * Gather rest of declaration...
                     */

                    loop {
                        if ch == b'>' as c_int {
                            break;
                        } else {
                            if ch == b'&' as c_int {
                                ch = mxml_get_entity(parent, p, &raw mut encoding, getc_cb);
                                if ch == EOF {
                                    break 'error;
                                }
                            }

                            if mxml_add_char(ch, &raw mut bufptr, &raw mut buffer, &raw mut bufsize)
                                != 0
                            {
                                break 'error;
                            }
                        }

                        ch = getc_cb.unwrap()(p, &raw mut encoding);
                        if ch == EOF {
                            break;
                        }
                    }

                    /*
                     * Error out if we didn't get the whole declaration...
                     */

                    if ch != b'>' as c_int {
                        /*
                         * Print error and return...
                         */

                        mxml_error(c"Early EOF in declaration node!".as_ptr());
                        break 'error;
                    }

                    /*
                     * Otherwise add this as an element under the current parent...
                     */

                    *bufptr = 0;

                    if parent.is_null() && !first.is_null() {
                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"<%s> cannot be a second root node after <%s>".as_ptr(),
                            buffer,
                            (*first).value.element.name,
                        );
                        mxml_error(msg.as_ptr());
                        break 'error;
                    }

                    node = mxml_new_element(parent, buffer);
                    if node.is_null() {
                        /*
                         * Print error and return...
                         */

                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"Unable to add declaration node to parent <%s>!".as_ptr(),
                            if !parent.is_null() {
                                (*parent).value.element.name as *const c_char
                            } else {
                                c"null".as_ptr()
                            },
                        );
                        mxml_error(msg.as_ptr());
                        break 'error;
                    }

                    if let Some(sax) = sax_cb {
                        sax(node, MXML_SAX_DIRECTIVE, sax_data);

                        if mxml_release(node) == 0 {
                            node = core::ptr::null_mut();
                        }
                    }

                    if !node.is_null() {
                        if first.is_null() {
                            first = node;
                        }

                        if parent.is_null() {
                            /*
                             * Got the XML declaration, so add it as the root node...
                             */

                            parent = node;

                            if cb.is_some() {
                                type_ = cb.unwrap()(parent);
                            } else {
                                type_ = MXML_TEXT;
                            }
                        }
                    }
                } else if *buffer == b'/' as c_char {
                    /*
                     * Handle close tag...
                     */

                    if parent.is_null()
                        || libc::strcmp(buffer.add(1), (*parent).value.element.name) != 0
                    {
                        /*
                         * Close tag doesn't match tree; print an error for now...
                         */

                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"Mismatched close tag <%s> under parent <%s>!".as_ptr(),
                            buffer,
                            if !parent.is_null() {
                                (*parent).value.element.name as *const c_char
                            } else {
                                c"(null)".as_ptr()
                            },
                        );
                        mxml_error(msg.as_ptr());
                        break 'error;
                    }

                    /*
                     * Keep reading until we see >...
                     */

                    while ch != b'>' as c_int && ch != EOF {
                        ch = getc_cb.unwrap()(p, &raw mut encoding);
                    }

                    node = parent;
                    parent = (*parent).parent;

                    if let Some(sax) = sax_cb {
                        sax(node, MXML_SAX_ELEMENT_CLOSE, sax_data);

                        if mxml_release(node) == 0 && first == node {
                            first = core::ptr::null_mut();
                        }
                    }

                    /*
                     * Ascend into the parent and set the value type as needed...
                     */

                    if cb.is_some() && !parent.is_null() {
                        type_ = cb.unwrap()(parent);
                    }
                } else {
                    /*
                     * Handle open tag...
                     */

                    if parent.is_null() && !first.is_null() {
                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"<%s> cannot be a second root node after <%s>".as_ptr(),
                            buffer,
                            (*first).value.element.name,
                        );
                        mxml_error(msg.as_ptr());
                        break 'error;
                    }

                    node = mxml_new_element(parent, buffer);
                    if node.is_null() {
                        /*
                         * Just print error for now...
                         */

                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"Unable to add element node to parent <%s>!".as_ptr(),
                            if !parent.is_null() {
                                (*parent).value.element.name as *const c_char
                            } else {
                                c"null".as_ptr()
                            },
                        );
                        mxml_error(msg.as_ptr());
                        break 'error;
                    }

                    if mxml_isspace(ch) != 0 {
                        ch = mxml_parse_element(node, p, &raw mut encoding, getc_cb);
                        if ch == EOF {
                            break 'error;
                        }
                    } else if ch == b'/' as c_int {
                        ch = getc_cb.unwrap()(p, &raw mut encoding);
                        if ch != b'>' as c_int {
                            libc::snprintf(
                                msg.as_mut_ptr(),
                                core::mem::size_of::<[c_char; 1024]>(),
                                c"Expected > but got '%c' instead for element <%s/>!".as_ptr(),
                                ch,
                                buffer,
                            );
                            mxml_error(msg.as_ptr());
                            mxml_delete(node);
                            break 'error;
                        }

                        ch = b'/' as c_int;
                    }

                    if let Some(sax) = sax_cb {
                        sax(node, MXML_SAX_ELEMENT_OPEN, sax_data);
                    }

                    if first.is_null() {
                        first = node;
                    }

                    if ch == EOF {
                        break;
                    }

                    if ch != b'/' as c_int {
                        /*
                         * Descend into this node, setting the value type as needed...
                         */

                        parent = node;

                        if cb.is_some() && !parent.is_null() {
                            type_ = cb.unwrap()(parent);
                        } else {
                            type_ = MXML_TEXT;
                        }
                    } else if let Some(sax) = sax_cb {
                        sax(node, MXML_SAX_ELEMENT_CLOSE, sax_data);

                        if mxml_release(node) == 0 && first == node {
                            first = core::ptr::null_mut();
                        }
                    }
                }

                bufptr = buffer;
            } else if ch == b'&' as c_int {
                /*
                 * Add character entity to current buffer...
                 */

                ch = mxml_get_entity(parent, p, &raw mut encoding, getc_cb);
                if ch == EOF {
                    break 'error;
                }

                if mxml_add_char(ch, &raw mut bufptr, &raw mut buffer, &raw mut bufsize) != 0 {
                    break 'error;
                }
            } else if type_ == MXML_OPAQUE || type_ == MXML_CUSTOM || mxml_isspace(ch) == 0 {
                /*
                 * Add character to current buffer...
                 */

                if mxml_add_char(ch, &raw mut bufptr, &raw mut buffer, &raw mut bufsize) != 0 {
                    break 'error;
                }
            }
        }

        /*
         * Free the string buffer - we don't need it anymore...
         */

        libc::free(buffer as *mut c_void);

        /*
         * Find the top element and return it...
         */

        if !parent.is_null() {
            node = parent;

            while parent != top && !(*parent).parent.is_null() {
                parent = (*parent).parent;
            }

            if node != parent {
                libc::snprintf(
                    msg.as_mut_ptr(),
                    core::mem::size_of::<[c_char; 1024]>(),
                    c"Missing close tag </%s> under parent <%s>!".as_ptr(),
                    (*node).value.element.name,
                    if !(*node).parent.is_null() {
                        (*(*node).parent).value.element.name as *const c_char
                    } else {
                        c"(null)".as_ptr()
                    },
                );
                mxml_error(msg.as_ptr());

                mxml_delete(first);

                return core::ptr::null_mut();
            }
        }

        if !parent.is_null() {
            return parent;
        } else {
            return first;
        }
    }

    /*
     * Common error return...
     */

    mxml_delete(first);

    libc::free(buffer as *mut c_void);

    core::ptr::null_mut()
}

/// Matches C static `mxml_parse_element` (`mxml-file.c:1758`).
pub unsafe fn mxml_parse_element(
    node: *mut MxmlNode,
    p: *mut c_void,
    encoding: *mut c_int,
    getc_cb: MxmlGetcCb,
) -> c_int {
    let mut ch: c_int;
    let mut quote: c_int;
    let mut name: *mut c_char;
    let mut value: *mut c_char;
    let mut ptr: *mut c_char;
    let mut namesize: c_int;
    let mut valsize: c_int;
    let mut msg: [c_char; 1024] = [0; 1024];

    /*
     * Initialize the name and value buffers...
     */

    name = libc::malloc(64) as *mut c_char;
    if name.is_null() {
        mxml_error(c"Unable to allocate memory for name!".as_ptr());
        return EOF;
    }

    namesize = 64;

    value = libc::malloc(64) as *mut c_char;
    if value.is_null() {
        libc::free(name as *mut c_void);
        mxml_error(c"Unable to allocate memory for value!".as_ptr());
        return EOF;
    }

    valsize = 64;

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

            if ch == b'/' as c_int || ch == b'?' as c_int {
                /*
                 * Grab the > character and print an error if it isn't there...
                 */

                quote = getc_cb.unwrap()(p, encoding);

                if quote != b'>' as c_int {
                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Expected '>' after '%c' for element %s, but got '%c'!".as_ptr(),
                        ch,
                        (*node).value.element.name,
                        quote,
                    );
                    mxml_error(msg.as_ptr());
                    break 'error;
                }

                break;
            } else if ch == b'<' as c_int {
                libc::snprintf(
                    msg.as_mut_ptr(),
                    core::mem::size_of::<[c_char; 1024]>(),
                    c"Bare < in element %s!".as_ptr(),
                    (*node).value.element.name,
                );
                mxml_error(msg.as_ptr());
                break 'error;
            } else if ch == b'>' as c_int {
                break;
            }

            /*
             * Read the attribute name...
             */

            *name = ch as c_char;
            ptr = name.add(1);

            if ch == b'\"' as c_int || ch == b'\'' as c_int {
                /*
                 * Name is in quotes, so get a quoted string...
                 */

                quote = ch;

                loop {
                    ch = getc_cb.unwrap()(p, encoding);
                    if ch == EOF {
                        break;
                    }

                    if ch == b'&' as c_int {
                        ch = mxml_get_entity(node, p, encoding, getc_cb);
                        if ch == EOF {
                            break 'error;
                        }
                    }

                    if mxml_add_char(ch, &raw mut ptr, &raw mut name, &raw mut namesize) != 0 {
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
                        || ch == b'=' as c_int
                        || ch == b'/' as c_int
                        || ch == b'>' as c_int
                        || ch == b'?' as c_int
                    {
                        break;
                    } else {
                        if ch == b'&' as c_int {
                            ch = mxml_get_entity(node, p, encoding, getc_cb);
                            if ch == EOF {
                                break 'error;
                            }
                        }

                        if mxml_add_char(ch, &raw mut ptr, &raw mut name, &raw mut namesize) != 0 {
                            break 'error;
                        }
                    }
                }
            }

            *ptr = 0;

            if !mxml_element_get_attr(node, name).is_null() {
                break 'error;
            }

            while ch != EOF && mxml_isspace(ch) != 0 {
                ch = getc_cb.unwrap()(p, encoding);
            }

            if ch == b'=' as c_int {
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
                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Missing value for attribute '%s' in element %s!".as_ptr(),
                        name,
                        (*node).value.element.name,
                    );
                    mxml_error(msg.as_ptr());
                    break 'error;
                }

                if ch == b'\'' as c_int || ch == b'\"' as c_int {
                    /*
                     * Read quoted value...
                     */

                    quote = ch;
                    ptr = value;

                    loop {
                        ch = getc_cb.unwrap()(p, encoding);
                        if ch == EOF {
                            break;
                        }

                        if ch == quote {
                            break;
                        } else {
                            if ch == b'&' as c_int {
                                ch = mxml_get_entity(node, p, encoding, getc_cb);
                                if ch == EOF {
                                    break 'error;
                                }
                            }

                            if mxml_add_char(ch, &raw mut ptr, &raw mut value, &raw mut valsize)
                                != 0
                            {
                                break 'error;
                            }
                        }
                    }

                    *ptr = 0;
                } else {
                    /*
                     * Read unquoted value...
                     */

                    *value = ch as c_char;
                    ptr = value.add(1);

                    loop {
                        ch = getc_cb.unwrap()(p, encoding);
                        if ch == EOF {
                            break;
                        }

                        if mxml_isspace(ch) != 0
                            || ch == b'=' as c_int
                            || ch == b'/' as c_int
                            || ch == b'>' as c_int
                        {
                            break;
                        } else {
                            if ch == b'&' as c_int {
                                ch = mxml_get_entity(node, p, encoding, getc_cb);
                                if ch == EOF {
                                    break 'error;
                                }
                            }

                            if mxml_add_char(ch, &raw mut ptr, &raw mut value, &raw mut valsize)
                                != 0
                            {
                                break 'error;
                            }
                        }
                    }

                    *ptr = 0;
                }

                /*
                 * Set the attribute with the given string value...
                 */

                mxml_element_set_attr(node, name, value);
            } else {
                libc::snprintf(
                    msg.as_mut_ptr(),
                    core::mem::size_of::<[c_char; 1024]>(),
                    c"Missing value for attribute '%s' in element %s!".as_ptr(),
                    name,
                    (*node).value.element.name,
                );
                mxml_error(msg.as_ptr());
                break 'error;
            }

            /*
             * Check the end character...
             */

            if ch == b'/' as c_int || ch == b'?' as c_int {
                /*
                 * Grab the > character and print an error if it isn't there...
                 */

                quote = getc_cb.unwrap()(p, encoding);

                if quote != b'>' as c_int {
                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Expected '>' after '%c' for element %s, but got '%c'!".as_ptr(),
                        ch,
                        (*node).value.element.name,
                        quote,
                    );
                    mxml_error(msg.as_ptr());
                    ch = EOF;
                }

                break;
            } else if ch == b'>' as c_int {
                break;
            }
        }

        /*
         * Free the name and value buffers and return...
         */

        libc::free(name as *mut c_void);
        libc::free(value as *mut c_void);

        return ch;
    }

    /*
     * Common error return point...
     */

    libc::free(name as *mut c_void);
    libc::free(value as *mut c_void);

    EOF
}

/// Matches C static `mxml_string_getc` (`mxml-file.c:2049`).
pub unsafe extern "C" fn mxml_string_getc(p: *mut c_void, encoding: *mut c_int) -> c_int {
    let mut ch: c_int;
    let s: *mut *const c_char;

    s = p as *mut *const c_char;
    ch = (**s as c_int) & 255;
    if ch != 0 || *encoding == ENCODE_UTF16LE {
        *s = (*s).add(1);

        match *encoding {
            ENCODE_UTF8 => {
                /*
                 * Got a UTF-8 character; convert UTF-8 to Unicode and return...
                 */

                if (ch & 0x80) == 0 {
                    /*
                     * ASCII
                     */

                    if ch < b' ' as c_int
                        && ch != b'\n' as c_int
                        && ch != b'\r' as c_int
                        && ch != b'\t' as c_int
                    {
                        let mut msg: [c_char; 1024] = [0; 1024];
                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"Bad control character 0x%02x not allowed by XML standard!".as_ptr(),
                            ch,
                        );
                        mxml_error(msg.as_ptr());
                        return EOF;
                    }

                    return ch;
                } else if ch == 0xfe {
                    /*
                     * UTF-16 big-endian BOM?
                     */

                    if ((**s as c_int) & 255) != 0xff {
                        return EOF;
                    }

                    *encoding = ENCODE_UTF16BE;
                    *s = (*s).add(1);

                    return mxml_string_getc(p, encoding);
                } else if ch == 0xff {
                    /*
                     * UTF-16 little-endian BOM?
                     */

                    if ((**s as c_int) & 255) != 0xfe {
                        return EOF;
                    }

                    *encoding = ENCODE_UTF16LE;
                    *s = (*s).add(1);

                    return mxml_string_getc(p, encoding);
                } else if (ch & 0xe0) == 0xc0 {
                    /*
                     * Two-byte value...
                     */

                    if ((**s as c_int) & 0xc0) != 0x80 {
                        return EOF;
                    }

                    ch = ((ch & 0x1f) << 6) | ((**s as c_int) & 0x3f);

                    *s = (*s).add(1);

                    if ch < 0x80 {
                        let mut msg: [c_char; 1024] = [0; 1024];
                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"Invalid UTF-8 sequence for character 0x%04x!".as_ptr(),
                            ch,
                        );
                        mxml_error(msg.as_ptr());
                        return EOF;
                    }

                    return ch;
                } else if (ch & 0xf0) == 0xe0 {
                    /*
                     * Three-byte value...
                     */

                    if ((**s as c_int) & 0xc0) != 0x80 || ((*(*s).add(1) as c_int) & 0xc0) != 0x80 {
                        return EOF;
                    }

                    ch = ((((ch & 0x0f) << 6) | ((**s as c_int) & 0x3f)) << 6)
                        | ((*(*s).add(1) as c_int) & 0x3f);

                    *s = (*s).add(2);

                    if ch < 0x800 {
                        let mut msg: [c_char; 1024] = [0; 1024];
                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"Invalid UTF-8 sequence for character 0x%04x!".as_ptr(),
                            ch,
                        );
                        mxml_error(msg.as_ptr());
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

                    if ((**s as c_int) & 0xc0) != 0x80
                        || ((*(*s).add(1) as c_int) & 0xc0) != 0x80
                        || ((*(*s).add(2) as c_int) & 0xc0) != 0x80
                    {
                        return EOF;
                    }

                    ch = ((((((ch & 0x07) << 6) | ((**s as c_int) & 0x3f)) << 6)
                        | ((*(*s).add(1) as c_int) & 0x3f))
                        << 6)
                        | ((*(*s).add(2) as c_int) & 0x3f);

                    *s = (*s).add(3);

                    if ch < 0x10000 {
                        let mut msg: [c_char; 1024] = [0; 1024];
                        libc::snprintf(
                            msg.as_mut_ptr(),
                            core::mem::size_of::<[c_char; 1024]>(),
                            c"Invalid UTF-8 sequence for character 0x%04x!".as_ptr(),
                            ch,
                        );
                        mxml_error(msg.as_ptr());
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

                ch = (ch << 8) | ((**s as c_int) & 255);
                *s = (*s).add(1);

                if ch < b' ' as c_int
                    && ch != b'\n' as c_int
                    && ch != b'\r' as c_int
                    && ch != b'\t' as c_int
                {
                    let mut msg: [c_char; 1024] = [0; 1024];
                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Bad control character 0x%02x not allowed by XML standard!".as_ptr(),
                        ch,
                    );
                    mxml_error(msg.as_ptr());
                    return EOF;
                } else if ch >= 0xd800 && ch <= 0xdbff {
                    /*
                     * Multi-word UTF-16 char...
                     */

                    let lch: c_int;

                    if **s == 0 {
                        return EOF;
                    }

                    lch = (((**s as c_int) & 255) << 8) | ((*(*s).add(1) as c_int) & 255);
                    *s = (*s).add(2);

                    if lch < 0xdc00 || lch >= 0xdfff {
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

                ch = ch | (((**s as c_int) & 255) << 8);

                if ch == 0 {
                    *s = (*s).offset(-1);
                    return EOF;
                }

                *s = (*s).add(1);

                if ch < b' ' as c_int
                    && ch != b'\n' as c_int
                    && ch != b'\r' as c_int
                    && ch != b'\t' as c_int
                {
                    let mut msg: [c_char; 1024] = [0; 1024];
                    libc::snprintf(
                        msg.as_mut_ptr(),
                        core::mem::size_of::<[c_char; 1024]>(),
                        c"Bad control character 0x%02x not allowed by XML standard!".as_ptr(),
                        ch,
                    );
                    mxml_error(msg.as_ptr());
                    return EOF;
                } else if ch >= 0xd800 && ch <= 0xdbff {
                    /*
                     * Multi-word UTF-16 char...
                     */

                    let lch: c_int;

                    if *(*s).add(1) == 0 {
                        return EOF;
                    }

                    lch = (((*(*s).add(1) as c_int) & 255) << 8) | ((**s as c_int) & 255);
                    *s = (*s).add(2);

                    if lch < 0xdc00 || lch >= 0xdfff {
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
pub unsafe extern "C" fn mxml_string_putc(ch: c_int, p: *mut c_void) -> c_int {
    let pp: *mut *mut c_char;

    pp = p as *mut *mut c_char;

    if *pp < *pp.add(1) {
        **pp = ch as c_char;
    }

    *pp = (*pp).add(1);

    0
}

/// Matches C static `mxml_write_name` (`mxml-file.c:2280`).
pub unsafe fn mxml_write_name(mut s: *const c_char, p: *mut c_void, putc_cb: MxmlPutcCb) -> c_int {
    let quote: c_char;
    let mut name: *const c_char;

    if *s == b'\"' as c_char || *s == b'\'' as c_char {
        /*
         * Write a quoted name string...
         */

        if putc_cb.unwrap()(*s as c_int, p) < 0 {
            return -1;
        }

        quote = *s;
        s = s.add(1);

        while *s != 0 && *s != quote {
            /*
             * Escape special characters...
             */

            name = mxml_entity_get_name(*s as c_int);
            if !name.is_null() {
                if putc_cb.unwrap()(b'&' as c_int, p) < 0 {
                    return -1;
                }

                while *name != 0 {
                    if putc_cb.unwrap()(*name as c_int, p) < 0 {
                        return -1;
                    }

                    name = name.add(1);
                }

                if putc_cb.unwrap()(b';' as c_int, p) < 0 {
                    return -1;
                }
            } else if putc_cb.unwrap()(*s as c_int, p) < 0 {
                return -1;
            }

            s = s.add(1);
        }

        /*
         * Write the end quote...
         */

        if putc_cb.unwrap()(quote as c_int, p) < 0 {
            return -1;
        }
    } else {
        /*
         * Write a non-quoted name string...
         */

        while *s != 0 {
            if putc_cb.unwrap()(*s as c_int, p) < 0 {
                return -1;
            }

            s = s.add(1);
        }
    }

    0
}

/// Matches C static `mxml_write_node` (`mxml-file.c:2352`).
pub unsafe fn mxml_write_node(
    node: *mut MxmlNode,
    p: *mut c_void,
    cb: MxmlSaveCb,
    mut col: c_int,
    putc_cb: MxmlPutcCb,
    global: *mut MxmlGlobal,
) -> c_int {
    let mut current: *mut MxmlNode;
    let mut next: *mut MxmlNode;
    let mut i: c_int;
    let mut width: c_int;
    let mut attr: *mut MxmlAttr;
    let mut s: [c_char; 255] = [0; 255];

    current = node;
    while !current.is_null() {
        /*
         * Print the node value...
         */

        match (*current).type_ {
            MXML_ELEMENT => {
                col = mxml_write_ws(current, p, cb, MXML_WS_BEFORE_OPEN, col, putc_cb);

                if putc_cb.unwrap()(b'<' as c_int, p) < 0 {
                    return -1;
                }

                if *(*current).value.element.name == b'?' as c_char
                    || libc::strncmp((*current).value.element.name, c"!--".as_ptr(), 3) == 0
                    || libc::strncmp((*current).value.element.name, c"![CDATA[".as_ptr(), 8) == 0
                {
                    /*
                     * Comments, CDATA, and processing instructions do not
                     * use character entities, but XML declarations are
                     * written as-is...
                     */

                    let mut ptr: *const c_char;

                    ptr = (*current).value.element.name;
                    while *ptr != 0 {
                        if putc_cb.unwrap()(*ptr as c_int, p) < 0 {
                            return -1;
                        }
                        ptr = ptr.add(1);
                    }
                } else if mxml_write_name((*current).value.element.name, p, putc_cb) < 0 {
                    return -1;
                }

                col += libc::strlen((*current).value.element.name) as c_int + 1;

                i = (*current).value.element.num_attrs;
                attr = (*current).value.element.attrs;
                while i > 0 {
                    width = libc::strlen((*attr).name) as c_int;

                    if !(*attr).value.is_null() {
                        width += libc::strlen((*attr).value) as c_int + 3;
                    }

                    if (*global).wrap > 0 && (col + width) > (*global).wrap {
                        if putc_cb.unwrap()(b'\n' as c_int, p) < 0 {
                            return -1;
                        }

                        col = 0;
                    } else {
                        if putc_cb.unwrap()(b' ' as c_int, p) < 0 {
                            return -1;
                        }

                        col += 1;
                    }

                    if mxml_write_name((*attr).name, p, putc_cb) < 0 {
                        return -1;
                    }

                    if !(*attr).value.is_null() {
                        if putc_cb.unwrap()(b'=' as c_int, p) < 0 {
                            return -1;
                        }
                        if putc_cb.unwrap()(b'\"' as c_int, p) < 0 {
                            return -1;
                        }
                        if mxml_write_string((*attr).value, p, putc_cb) < 0 {
                            return -1;
                        }
                        if putc_cb.unwrap()(b'\"' as c_int, p) < 0 {
                            return -1;
                        }
                    }

                    col += width;

                    i -= 1;
                    attr = attr.add(1);
                }

                if !(*current).child.is_null() {
                    /*
                     * Write children...
                     */

                    if putc_cb.unwrap()(b'>' as c_int, p) < 0 {
                        return -1;
                    } else {
                        col += 1;
                    }

                    col = mxml_write_ws(current, p, cb, MXML_WS_AFTER_OPEN, col, putc_cb);
                } else if *(*current).value.element.name == b'!' as c_char
                    || *(*current).value.element.name == b'?' as c_char
                {
                    /*
                     * The ?xml declaration and !DOCTYPE nodes are written
                     * without a closing tag...
                     */

                    if putc_cb.unwrap()(b'>' as c_int, p) < 0 {
                        return -1;
                    } else {
                        col += 1;
                    }

                    col = mxml_write_ws(current, p, cb, MXML_WS_AFTER_OPEN, col, putc_cb);
                } else {
                    if putc_cb.unwrap()(b' ' as c_int, p) < 0 {
                        return -1;
                    }
                    if putc_cb.unwrap()(b'/' as c_int, p) < 0 {
                        return -1;
                    }
                    if putc_cb.unwrap()(b'>' as c_int, p) < 0 {
                        return -1;
                    }

                    col += 3;

                    col = mxml_write_ws(current, p, cb, MXML_WS_AFTER_OPEN, col, putc_cb);
                }
            }

            MXML_INTEGER => {
                if !(*current).prev.is_null() {
                    if (*global).wrap > 0 && col > (*global).wrap {
                        if putc_cb.unwrap()(b'\n' as c_int, p) < 0 {
                            return -1;
                        }

                        col = 0;
                    } else if putc_cb.unwrap()(b' ' as c_int, p) < 0 {
                        return -1;
                    } else {
                        col += 1;
                    }
                }

                libc::sprintf(s.as_mut_ptr(), c"%d".as_ptr(), (*current).value.integer);
                if mxml_write_string(s.as_ptr(), p, putc_cb) < 0 {
                    return -1;
                }

                col += libc::strlen(s.as_ptr()) as c_int;
            }

            MXML_OPAQUE => {
                if mxml_write_string((*current).value.opaque, p, putc_cb) < 0 {
                    return -1;
                }

                col += libc::strlen((*current).value.opaque) as c_int;
            }

            MXML_REAL => {
                if !(*current).prev.is_null() {
                    if (*global).wrap > 0 && col > (*global).wrap {
                        if putc_cb.unwrap()(b'\n' as c_int, p) < 0 {
                            return -1;
                        }

                        col = 0;
                    } else if putc_cb.unwrap()(b' ' as c_int, p) < 0 {
                        return -1;
                    } else {
                        col += 1;
                    }
                }

                libc::sprintf(s.as_mut_ptr(), c"%f".as_ptr(), (*current).value.real);
                if mxml_write_string(s.as_ptr(), p, putc_cb) < 0 {
                    return -1;
                }

                col += libc::strlen(s.as_ptr()) as c_int;
            }

            MXML_TEXT => {
                if (*current).value.text.whitespace != 0 && col > 0 {
                    if (*global).wrap > 0 && col > (*global).wrap {
                        if putc_cb.unwrap()(b'\n' as c_int, p) < 0 {
                            return -1;
                        }

                        col = 0;
                    } else if putc_cb.unwrap()(b' ' as c_int, p) < 0 {
                        return -1;
                    } else {
                        col += 1;
                    }
                }

                if mxml_write_string((*current).value.text.string, p, putc_cb) < 0 {
                    return -1;
                }

                col += libc::strlen((*current).value.text.string) as c_int;
            }

            MXML_CUSTOM if (*global).custom_save_cb.is_some() => {
                let data: *mut c_char;
                let newline: *const c_char;

                data = (*global).custom_save_cb.unwrap()(node);
                if data.is_null() {
                    return -1;
                }

                if mxml_write_string(data, p, putc_cb) < 0 {
                    return -1;
                }

                newline = libc::strrchr(data, '\n' as c_int);
                if newline.is_null() {
                    col += libc::strlen(data) as c_int;
                } else {
                    col = libc::strlen(newline) as c_int;
                }

                libc::free(data as *mut c_void);
            }

            _ => {
                return -1;
            }
        }

        /*
         * Figure out the next node...
         */

        next = (*current).child;
        if next.is_null() {
            loop {
                next = (*current).next;
                if !next.is_null() {
                    break;
                }

                if current == node {
                    break;
                }

                current = (*current).parent;

                if *(*current).value.element.name != b'!' as c_char
                    && *(*current).value.element.name != b'?' as c_char
                {
                    col = mxml_write_ws(current, p, cb, MXML_WS_BEFORE_CLOSE, col, putc_cb);

                    if putc_cb.unwrap()(b'<' as c_int, p) < 0 {
                        return -1;
                    }
                    if putc_cb.unwrap()(b'/' as c_int, p) < 0 {
                        return -1;
                    }
                    if mxml_write_string((*current).value.element.name, p, putc_cb) < 0 {
                        return -1;
                    }
                    if putc_cb.unwrap()(b'>' as c_int, p) < 0 {
                        return -1;
                    }

                    col += libc::strlen((*current).value.element.name) as c_int + 3;

                    col = mxml_write_ws(current, p, cb, MXML_WS_AFTER_CLOSE, col, putc_cb);
                }
            }
        }

        current = next;
    }

    col
}

/// Matches C static `mxml_write_string` (`mxml-file.c:2618`).
pub unsafe fn mxml_write_string(
    mut s: *const c_char,
    p: *mut c_void,
    putc_cb: MxmlPutcCb,
) -> c_int {
    let mut name: *const c_char;

    while *s != 0 {
        /*
         * Escape special characters...
         */

        name = mxml_entity_get_name(*s as c_int);
        if !name.is_null() {
            if putc_cb.unwrap()(b'&' as c_int, p) < 0 {
                return -1;
            }

            while *name != 0 {
                if putc_cb.unwrap()(*name as c_int, p) < 0 {
                    return -1;
                }

                name = name.add(1);
            }

            if putc_cb.unwrap()(b';' as c_int, p) < 0 {
                return -1;
            }
        } else if putc_cb.unwrap()(*s as c_int, p) < 0 {
            return -1;
        }

        s = s.add(1);
    }

    0
}

/// Matches C static `mxml_write_ws` (`mxml-file.c:2660`).
pub unsafe fn mxml_write_ws(
    node: *mut MxmlNode,
    p: *mut c_void,
    cb: MxmlSaveCb,
    ws: c_int,
    mut col: c_int,
    putc_cb: MxmlPutcCb,
) -> c_int {
    let mut s: *const c_char;

    if let Some(f) = cb {
        s = f(node, ws);
        if !s.is_null() {
            while *s != 0 {
                if putc_cb.unwrap()(*s as c_int, p) < 0 {
                    return -1;
                } else if *s == b'\n' as c_char {
                    col = 0;
                } else if *s == b'\t' as c_char {
                    col += MXML_TAB;
                    col = col - (col % MXML_TAB);
                } else {
                    col += 1;
                }

                s = s.add(1);
            }
        }
    }

    col
}

#[cfg(test)]
mod tests {
    //! The `xml_probe` test is a differential driver: it reproduces, byte for
    //! byte, the output of the C program in
    //! `scratchpad/xmldiff/probe.c` linked against the reference
    //! `libimxml.so`.  It only runs when `IMOD_XML_PROBE_OUT` is set.
    use super::*;
    use std::ffi::CString;

    unsafe extern "C" {
        static mut stderr: *mut libc::FILE;
    }

    static mut OUT: *mut libc::FILE = core::ptr::null_mut();
    static mut S_LAST_LEVEL: c_int = -1;
    static mut S_WS_BUFFER: [c_char; 36] = [0; 36];
    static mut NODES: Vec<*mut MxmlNode> = Vec::new();

    /// Copy of `ixmlWhitespace_cb` from `IMOD/libcfshr/mxmlwrap.c`.
    unsafe extern "C" fn ws_cb(node: *mut MxmlNode, where_: c_int) -> *const c_char {
        let mut level: c_int = -1;
        let mut parent = (*node).parent;
        let mut spaces: [c_char; 33] = [b' ' as c_char; 33];
        if where_ != MXML_WS_BEFORE_OPEN && where_ != MXML_WS_BEFORE_CLOSE {
            return core::ptr::null();
        }
        while !parent.is_null() {
            level += 1;
            parent = (*parent).parent;
        }
        if level > 16 {
            level = 16;
        } else if level < 0 {
            level = 0;
        }
        if S_LAST_LEVEL < 0 {
            S_LAST_LEVEL = level;
            return core::ptr::null();
        }
        if level == S_LAST_LEVEL && where_ == MXML_WS_BEFORE_CLOSE {
            return core::ptr::null();
        }
        S_LAST_LEVEL = level;
        spaces[32] = 0;
        libc::snprintf(
            (&raw mut S_WS_BUFFER).cast::<c_char>(),
            36,
            c"\n%s".as_ptr(),
            spaces.as_ptr().add((32 - 2 * level) as usize),
        );
        (&raw const S_WS_BUFFER).cast::<c_char>()
    }

    unsafe fn node_index(n: *mut MxmlNode) -> c_int {
        if n.is_null() {
            return -1;
        }
        let nodes: &Vec<*mut MxmlNode> = &*(&raw const NODES);
        for (i, p) in nodes.iter().enumerate() {
            if *p == n {
                return i as c_int;
            }
        }
        -2
    }

    unsafe fn dump_tree(tree: *mut MxmlNode, tag: &core::ffi::CStr) {
        let mut ws: c_int = 0;
        let nodes: &mut Vec<*mut MxmlNode> = &mut *(&raw mut NODES);
        nodes.clear();
        let mut n = tree;
        while !n.is_null() {
            nodes.push(n);
            n = mxml_walk_next(n, tree, MXML_DESCEND);
        }
        let num = nodes.len() as c_int;
        libc::fprintf(OUT, c"%s nodes=%d\n".as_ptr(), tag.as_ptr(), num);
        for i in 0..num {
            let n = (&*(&raw const NODES))[i as usize];
            libc::fprintf(
                OUT,
                c"  [%d] type=%d parent=%d child=%d last=%d prev=%d next=%d ref=%d".as_ptr(),
                i,
                mxml_get_type(n),
                node_index((*n).parent),
                node_index((*n).child),
                node_index((*n).last_child),
                node_index((*n).prev),
                node_index((*n).next),
                mxml_get_ref_count(n),
            );
            match (*n).type_ {
                MXML_ELEMENT => {
                    libc::fprintf(
                        OUT,
                        c" elem=<%s> nattr=%d".as_ptr(),
                        (*n).value.element.name,
                        (*n).value.element.num_attrs,
                    );
                    for k in 0..(*n).value.element.num_attrs {
                        let a = (*n).value.element.attrs.add(k as usize);
                        libc::fprintf(
                            OUT,
                            c" attr[%d]=%s=\"%s\"".as_ptr(),
                            k,
                            (*a).name,
                            if !(*a).value.is_null() {
                                (*a).value as *const c_char
                            } else {
                                c"(nil)".as_ptr()
                            },
                        );
                    }
                    let s = mxml_get_cdata(n);
                    if !s.is_null() {
                        libc::fprintf(OUT, c" cdata=<%s>".as_ptr(), s);
                    }
                }
                MXML_OPAQUE => {
                    libc::fprintf(OUT, c" opaque=<%s>".as_ptr(), (*n).value.opaque);
                }
                MXML_TEXT => {
                    libc::fprintf(
                        OUT,
                        c" text=<%s> ws=%d".as_ptr(),
                        (*n).value.text.string,
                        (*n).value.text.whitespace,
                    );
                }
                MXML_INTEGER => {
                    libc::fprintf(OUT, c" integer=%d".as_ptr(), (*n).value.integer);
                }
                MXML_REAL => {
                    libc::fprintf(OUT, c" real=%f".as_ptr(), (*n).value.real);
                }
                _ => {}
            }
            let s = mxml_get_element(n);
            libc::fprintf(
                OUT,
                c" getElem=%s getOpaque=%s getInt=%d getReal=%f getText=%s".as_ptr(),
                if !s.is_null() { s } else { c"(nil)".as_ptr() },
                if !mxml_get_opaque(n).is_null() {
                    mxml_get_opaque(n)
                } else {
                    c"(nil)".as_ptr()
                },
                mxml_get_integer(n),
                mxml_get_real(n),
                if !mxml_get_text(n, &raw mut ws).is_null() {
                    mxml_get_text(n, &raw mut ws)
                } else {
                    c"(nil)".as_ptr()
                },
            );
            libc::fprintf(OUT, c"\n".as_ptr());
        }
    }

    unsafe fn save_report(tree: *mut MxmlNode, tag: &core::ffi::CStr, cb: MxmlSaveCb, wrap: c_int) {
        let mut buf: Vec<c_char> = vec![0; 400000];
        mxml_set_wrap_margin(wrap);
        S_LAST_LEVEL = -1;
        let n = mxml_save_string(tree, buf.as_mut_ptr(), 400000, cb);
        libc::fprintf(
            OUT,
            c"%s saveString=%d\n[%s]\n".as_ptr(),
            tag.as_ptr(),
            n,
            buf.as_ptr(),
        );
        S_LAST_LEVEL = -1;
        let alloc = mxml_save_alloc_string(tree, cb);
        libc::fprintf(
            OUT,
            c"%s saveAlloc=%s\n".as_ptr(),
            tag.as_ptr(),
            if !alloc.is_null() {
                c"ok".as_ptr()
            } else {
                c"(nil)".as_ptr()
            },
        );
        if !alloc.is_null() {
            libc::fprintf(OUT, c"[%s]\n".as_ptr(), alloc);
            libc::free(alloc as *mut c_void);
        }
        S_LAST_LEVEL = -1;
        let mut small: [c_char; 40] = [0; 40];
        let m = mxml_save_string(tree, small.as_mut_ptr(), 40, cb);
        libc::fprintf(
            OUT,
            c"%s saveSmall=%d [%s]\n".as_ptr(),
            tag.as_ptr(),
            m,
            small.as_ptr(),
        );
        mxml_set_wrap_margin(72);
    }

    unsafe fn probe_file(path: &str, base: &str, savedir: &str) {
        let cpath = CString::new(path).unwrap();
        let cbase = CString::new(base).unwrap();
        libc::fprintf(OUT, c"=== FILE %s\n".as_ptr(), cbase.as_ptr());
        let fp = libc::fopen(cpath.as_ptr(), c"r".as_ptr());
        if fp.is_null() {
            libc::fprintf(OUT, c"  no open\n".as_ptr());
            return;
        }
        let tree = mxml_load_file(core::ptr::null_mut(), fp, Some(mxml_opaque_cb));
        libc::fclose(fp);
        if tree.is_null() {
            libc::fprintf(OUT, c"  load NULL\n".as_ptr());
            return;
        }
        dump_tree(tree, c"  TREE");

        libc::fprintf(
            OUT,
            c"  walkNext(tree,tree,DESCEND) idx=%d\n".as_ptr(),
            node_index(mxml_walk_next(tree, tree, MXML_DESCEND)),
        );
        libc::fprintf(
            OUT,
            c"  walkNext(tree,tree,NO_DESCEND) idx=%d\n".as_ptr(),
            node_index(mxml_walk_next(tree, tree, MXML_NO_DESCEND)),
        );
        let mut n = tree;
        while !mxml_walk_next(n, tree, MXML_DESCEND).is_null() {
            n = mxml_walk_next(n, tree, MXML_DESCEND);
        }
        libc::fprintf(
            OUT,
            c"  lastNode=%d walkPrev=%d walkPrevNoDesc=%d\n".as_ptr(),
            node_index(n),
            node_index(mxml_walk_prev(n, tree, MXML_DESCEND)),
            node_index(mxml_walk_prev(n, tree, MXML_NO_DESCEND)),
        );

        let mut count: c_int = 0;
        let mut n = mxml_find_element(
            tree,
            tree,
            c"Field".as_ptr(),
            core::ptr::null(),
            core::ptr::null(),
            MXML_DESCEND,
        );
        while !n.is_null() {
            if count < 5 {
                let a = mxml_element_get_attr(n, c"name".as_ptr());
                libc::fprintf(
                    OUT,
                    c"  findField[%d]=%d attr=%s\n".as_ptr(),
                    count,
                    node_index(n),
                    if !a.is_null() { a } else { c"(nil)".as_ptr() },
                );
            }
            count += 1;
            n = mxml_find_element(
                n,
                tree,
                c"Field".as_ptr(),
                core::ptr::null(),
                core::ptr::null(),
                MXML_DESCEND,
            );
        }
        libc::fprintf(OUT, c"  findFieldCount=%d\n".as_ptr(), count);

        libc::fprintf(
            OUT,
            c"  findPath(autodoc/PreData/Version)=%d\n".as_ptr(),
            node_index(mxml_find_path(tree, c"autodoc/PreData/Version".as_ptr())),
        );
        libc::fprintf(
            OUT,
            c"  findPath(*/short)=%d\n".as_ptr(),
            node_index(mxml_find_path(tree, c"*/short".as_ptr())),
        );
        libc::fprintf(
            OUT,
            c"  findPath(nosuch)=%d\n".as_ptr(),
            node_index(mxml_find_path(tree, c"nosuch".as_ptr())),
        );

        let ind = mxml_index_new(tree, c"Field".as_ptr(), c"name".as_ptr());
        if ind.is_null() {
            libc::fprintf(OUT, c"  index NULL\n".as_ptr());
        } else {
            libc::fprintf(
                OUT,
                c"  indexCount=%d alloc=%d\n".as_ptr(),
                mxml_index_get_count(ind),
                (*ind).alloc_nodes,
            );
            let mut count: c_int = 0;
            let mut n = mxml_index_reset(ind);
            while !n.is_null() {
                if count < 8 {
                    let a = mxml_element_get_attr(n, c"name".as_ptr());
                    libc::fprintf(
                        OUT,
                        c"  indexEnum[%d]=%d %s\n".as_ptr(),
                        count,
                        node_index(n),
                        if !a.is_null() { a } else { c"(nil)".as_ptr() },
                    );
                }
                count += 1;
                n = mxml_index_enum(ind);
            }
            libc::fprintf(OUT, c"  indexEnumCount=%d\n".as_ptr(), count);
            mxml_index_reset(ind);
            let n = mxml_index_find(ind, c"Field".as_ptr(), c"InputFile".as_ptr());
            libc::fprintf(
                OUT,
                c"  indexFind(Field,InputFile)=%d\n".as_ptr(),
                node_index(n),
            );
            mxml_index_reset(ind);
            let n = mxml_index_find(ind, c"Field".as_ptr(), c"zzz-none".as_ptr());
            libc::fprintf(
                OUT,
                c"  indexFind(Field,zzz-none)=%d\n".as_ptr(),
                node_index(n),
            );
            mxml_index_delete(ind);
        }

        let ind = mxml_index_new(tree, core::ptr::null(), core::ptr::null());
        libc::fprintf(
            OUT,
            c"  indexAllCount=%d\n".as_ptr(),
            if !ind.is_null() {
                mxml_index_get_count(ind)
            } else {
                -1
            },
        );
        if !ind.is_null() {
            mxml_index_delete(ind);
        }

        save_report(tree, c"  SAVE-nocb", None, 72);
        save_report(tree, c"  SAVE-ws", Some(ws_cb), 0);
        save_report(tree, c"  SAVE-wrap20", None, 20);

        let savepath = CString::new(format!("{}/{}.save", savedir, base)).unwrap();
        let fp = libc::fopen(savepath.as_ptr(), c"w".as_ptr());
        if !fp.is_null() {
            mxml_set_wrap_margin(0);
            S_LAST_LEVEL = -1;
            libc::fprintf(
                OUT,
                c"  saveFile=%d\n".as_ptr(),
                mxml_save_file(tree, fp, Some(ws_cb)),
            );
            libc::fclose(fp);
            mxml_set_wrap_margin(72);
        }

        mxml_delete(tree);
    }

    const CASES: &[&core::ffi::CStr] = &[
        c"<a>b</a>",
        c"<?xml version=\"1.0\"?><root><x>1</x></root>",
        c"<!DOCTYPE foo><root/>",
        c"<root><!-- a comment --><x/></root>",
        c"<root><![CDATA[some <raw> data]]></root>",
        c"<root a='1' b=\"two\" c=unquoted>text</root>",
        c"<root>&amp;&lt;&gt;&quot;&apos;&#65;&#x42;&nbsp;</root>",
        c"<root>&bogus;</root>",
        c"<root><a></b></root>",
        c"<root",
        c"",
        c"<root>unclosed",
        c"<a/><b/>",
        c"<root>   spaced   text   </root>",
        c"<root><child x=\"1\"/><child x=\"2\"/><child x=\"3\"/></root>",
    ];

    unsafe fn probe_strings() {
        let mut buf: [c_char; 8192] = [0; 8192];
        for (i, case) in CASES.iter().enumerate() {
            libc::fprintf(
                OUT,
                c"=== STRING %d [%s]\n".as_ptr(),
                i as c_int,
                case.as_ptr(),
            );
            libc::fflush(OUT);
            let tree = mxml_load_string(core::ptr::null_mut(), case.as_ptr(), Some(mxml_opaque_cb));
            if tree.is_null() {
                libc::fprintf(OUT, c"  load NULL\n".as_ptr());
                continue;
            }
            dump_tree(tree, c"  TREE");
            S_LAST_LEVEL = -1;
            libc::fprintf(
                OUT,
                c"  save=%d [%s]\n".as_ptr(),
                mxml_save_string(tree, buf.as_mut_ptr(), 8192, None),
                buf.as_ptr(),
            );
            mxml_delete(tree);
        }
        for (tag, cb) in [
            (c"=== STRING-TEXTCB %d\n", None as MxmlLoadCb),
            (c"=== STRING-INTCB %d\n", Some(mxml_integer_cb as _)),
            (c"=== STRING-REALCB %d\n", Some(mxml_real_cb as _)),
            (c"=== STRING-IGNORECB %d\n", Some(mxml_ignore_cb as _)),
        ] {
            for (i, case) in CASES.iter().enumerate() {
                libc::fprintf(OUT, tag.as_ptr(), i as c_int);
                libc::fflush(OUT);
                let tree = mxml_load_string(core::ptr::null_mut(), case.as_ptr(), cb);
                if tree.is_null() {
                    libc::fprintf(OUT, c"  load NULL\n".as_ptr());
                    continue;
                }
                dump_tree(tree, c"  TREE");
                mxml_delete(tree);
            }
        }
    }

    unsafe fn probe_build() {
        let mut buf: [c_char; 8192] = [0; 8192];
        let mut i: c_int = 0;

        libc::fprintf(OUT, c"=== BUILD\n".as_ptr());
        let xml = mxml_new_xml(c"1.0".as_ptr());
        dump_tree(xml, c"  XMLONLY");
        let top = mxml_new_element(xml, c"root".as_ptr());
        let elem = mxml_new_element(top, c"child".as_ptr());
        mxml_element_set_attr(elem, c"one".as_ptr(), c"1".as_ptr());
        mxml_element_set_attr(elem, c"two".as_ptr(), c"2".as_ptr());
        mxml_element_set_attr(elem, c"one".as_ptr(), c"1b".as_ptr());
        mxml_element_set_attr(elem, c"nil".as_ptr(), core::ptr::null());
        mxml_element_set_attrf(
            elem,
            c"fmt".as_ptr(),
            c"%d".as_ptr(),
            42usize as *mut c_void,
        );
        mxml_new_text(elem, 0, c"hello & <world>".as_ptr());
        mxml_new_text(top, 1, c"spaced".as_ptr());
        mxml_new_integer(top, 17);
        mxml_new_real(top, 3.5);
        mxml_new_opaque(top, c"opaque \"text\"".as_ptr());
        mxml_new_cdata(top, c"cdata & stuff".as_ptr());
        dump_tree(xml, c"  BUILT");
        libc::fprintf(
            OUT,
            c"  getAttr(one)=%s\n".as_ptr(),
            mxml_element_get_attr(elem, c"one".as_ptr()),
        );
        libc::fprintf(
            OUT,
            c"  getAttr(nil)=%s\n".as_ptr(),
            if !mxml_element_get_attr(elem, c"nil".as_ptr()).is_null() {
                c"nonnull".as_ptr()
            } else {
                c"(nil)".as_ptr()
            },
        );
        libc::fprintf(
            OUT,
            c"  getAttr(zzz)=%s\n".as_ptr(),
            if !mxml_element_get_attr(elem, c"zzz".as_ptr()).is_null() {
                c"nonnull".as_ptr()
            } else {
                c"(nil)".as_ptr()
            },
        );
        mxml_element_delete_attr(elem, c"two".as_ptr());
        mxml_element_delete_attr(elem, c"zzz".as_ptr());
        dump_tree(xml, c"  AFTERDEL");
        save_report(xml, c"  BUILDSAVE", None, 72);
        save_report(xml, c"  BUILDSAVEWS", Some(ws_cb), 0);

        let t = mxml_new_element(core::ptr::null_mut(), c"orphan".as_ptr());
        let r1 = mxml_retain(t);
        let r2 = mxml_retain(t);
        let r3 = mxml_release(t);
        let r4 = mxml_release(t);
        let r5 = mxml_release(t);
        libc::fprintf(
            OUT,
            c"  retain=%d retain=%d release=%d release=%d releaseFinal=%d\n".as_ptr(),
            r1,
            r2,
            r3,
            r4,
            r5,
        );
        libc::fprintf(
            OUT,
            c"  retainNull=%d releaseNull=%d\n".as_ptr(),
            mxml_retain(core::ptr::null_mut()),
            mxml_release(core::ptr::null_mut()),
        );

        let t = mxml_new_element(core::ptr::null_mut(), c"setme".as_ptr());
        libc::fprintf(
            OUT,
            c"  setElement=%d\n".as_ptr(),
            mxml_set_element(t, c"renamed".as_ptr()),
        );
        libc::fprintf(OUT, c"  name=%s\n".as_ptr(), mxml_get_element(t));
        libc::fprintf(
            OUT,
            c"  setCDATAbad=%d\n".as_ptr(),
            mxml_set_cdata(t, c"x".as_ptr()),
        );
        libc::fprintf(
            OUT,
            c"  setUserData=%d\n".as_ptr(),
            mxml_set_user_data(t, 1usize as *mut c_void),
        );
        libc::fprintf(OUT, c"  getUserData=%p\n".as_ptr(), mxml_get_user_data(t));
        mxml_delete(t);

        let t = mxml_new_integer(core::ptr::null_mut(), 5);
        let a = mxml_set_integer(t, 9);
        libc::fprintf(
            OUT,
            c"  setInteger=%d %d\n".as_ptr(),
            a,
            mxml_get_integer(t),
        );
        libc::fprintf(OUT, c"  setReal=%d\n".as_ptr(), mxml_set_real(t, 1.0));
        mxml_delete(t);
        let t = mxml_new_real(core::ptr::null_mut(), 5.0);
        let a = mxml_set_real(t, 9.5);
        libc::fprintf(OUT, c"  setReal=%d %f\n".as_ptr(), a, mxml_get_real(t));
        mxml_delete(t);
        let t = mxml_new_text(core::ptr::null_mut(), 0, c"a".as_ptr());
        libc::fprintf(
            OUT,
            c"  setText=%d\n".as_ptr(),
            mxml_set_text(t, 1, c"bcd".as_ptr()),
        );
        libc::fprintf(
            OUT,
            c"  getText=%s\n".as_ptr(),
            mxml_get_text(t, &raw mut i),
        );
        mxml_delete(t);
        let t = mxml_new_opaque(core::ptr::null_mut(), c"a".as_ptr());
        let a = mxml_set_opaque(t, c"xyz".as_ptr());
        libc::fprintf(OUT, c"  setOpaque=%d %s\n".as_ptr(), a, mxml_get_opaque(t));
        mxml_delete(t);
        let t = mxml_new_cdata(core::ptr::null_mut(), c"abc".as_ptr());
        let c0 = mxml_get_cdata(t);
        let sc = mxml_set_cdata(t, c"def".as_ptr());
        libc::fprintf(
            OUT,
            c"  cdata=%s setCDATA=%d then=%s\n".as_ptr(),
            c0,
            sc,
            mxml_get_cdata(t),
        );
        S_LAST_LEVEL = -1;
        libc::fprintf(
            OUT,
            c"  cdataSave=%d [%s]\n".as_ptr(),
            mxml_save_string(t, buf.as_mut_ptr(), 8192, None),
            buf.as_ptr(),
        );
        mxml_delete(t);

        mxml_delete(xml);
    }

    unsafe fn probe_entities() {
        let names: &[&core::ffi::CStr] = &[
            c"amp", c"lt", c"gt", c"quot", c"apos", c"nbsp", c"AElig", c"zwnj", c"Alpha", c"euro",
            c"bogus", c"", c"a", c"zzzz",
        ];
        libc::fprintf(OUT, c"=== ENTITIES\n".as_ptr());
        for i in 0..300 {
            let n = mxml_entity_get_name(i);
            if !n.is_null() {
                libc::fprintf(OUT, c"  getName(%d)=%s\n".as_ptr(), i, n);
            }
        }
        for name in names {
            libc::fprintf(
                OUT,
                c"  getValue(%s)=%d\n".as_ptr(),
                name.as_ptr(),
                mxml_entity_get_value(name.as_ptr()),
            );
        }
    }

    /// A processing instruction read at top level becomes the parent of
    /// everything that follows (`mxml-file.c:1797-1804`), so the document
    /// element hangs off the `?xml?` node.  Expected values produced by the
    /// reference `libimxml.so`:
    ///     PI root elem=?xml version="1.0"? child=root walkNext=root
    #[test]
    fn processing_instruction_becomes_the_parent_of_the_document() {
        unsafe {
            let tree = mxml_load_string(
                core::ptr::null_mut(),
                c"<?xml version=\"1.0\"?><root><x>1</x></root>".as_ptr(),
                Some(mxml_opaque_cb),
            );
            assert!(!tree.is_null());
            assert_eq!(
                core::ffi::CStr::from_ptr(mxml_get_element(tree)).to_bytes(),
                b"?xml version=\"1.0\"?"
            );
            let child = mxml_get_first_child(tree);
            assert!(
                !child.is_null(),
                "the ?xml node must own the document element"
            );
            assert_eq!(
                core::ffi::CStr::from_ptr(mxml_get_element(child)).to_bytes(),
                b"root"
            );
            let walk = mxml_walk_next(tree, tree, MXML_DESCEND);
            assert!(
                !walk.is_null(),
                "mxmlWalkNext must descend into the ?xml node"
            );
            assert_eq!(
                core::ffi::CStr::from_ptr(mxml_get_element(walk)).to_bytes(),
                b"root"
            );
            mxml_delete(tree);
        }
    }

    /// `mxmlSaveString`/`mxmlSaveFile` must render every child, including the
    /// children of `?`/`!` nodes, and must honour the `mxml_save_cb_t`
    /// whitespace callback.  Byte strings and return values below come from the
    /// reference `libimxml.so`.
    #[test]
    fn save_string_and_save_file_match_native_bytes() {
        unsafe {
            let mut buf: [c_char; 8192] = [0; 8192];
            let tree = mxml_load_string(
                core::ptr::null_mut(),
                c"<root><child x=\"1\"/><child x=\"2\"/><child x=\"3\"/></root>".as_ptr(),
                Some(mxml_opaque_cb),
            );
            assert!(!tree.is_null());

            let n = mxml_save_string(tree, buf.as_mut_ptr(), 8192, None);
            assert_eq!(n, 59);
            assert_eq!(
                core::ffi::CStr::from_ptr(buf.as_ptr()).to_bytes(),
                b"<root><child x=\"1\" /><child x=\"2\" /><child x=\"3\" /></root>\n"
            );

            /* With the indenting callback and no wrap margin. */
            mxml_set_wrap_margin(0);
            S_LAST_LEVEL = -1;
            let n = mxml_save_string(tree, buf.as_mut_ptr(), 8192, Some(ws_cb));
            assert_eq!(n, 62);
            assert_eq!(
                core::ffi::CStr::from_ptr(buf.as_ptr()).to_bytes(),
                b"<root>\n<child x=\"1\" />\n<child x=\"2\" />\n<child x=\"3\" /></root>\n"
            );

            /* mxmlSaveFile writes the same bytes through a FILE *. */
            let path = std::env::temp_dir().join("imodrs_mxml_savefile.xml");
            let cpath = CString::new(path.to_str().unwrap()).unwrap();
            let fp = libc::fopen(cpath.as_ptr(), c"w".as_ptr());
            assert!(!fp.is_null());
            S_LAST_LEVEL = -1;
            assert_eq!(mxml_save_file(tree, fp, Some(ws_cb)), 0);
            libc::fclose(fp);
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
                core::ptr::null_mut(),
                c"<?xml version=\"1.0\"?><root><child x=\"1\"/></root>".as_ptr(),
                Some(mxml_opaque_cb),
            );
            assert!(!xml.is_null());
            S_LAST_LEVEL = -1;
            let n = mxml_save_string(xml, buf.as_mut_ptr(), 8192, Some(ws_cb));
            assert_eq!(n, 55);
            assert_eq!(
                core::ffi::CStr::from_ptr(buf.as_ptr()).to_bytes(),
                b"<?xml version=\"1.0\"?>\n<root>\n  <child x=\"1\" />\n</root>\n"
            );
            mxml_delete(xml);

            mxml_set_wrap_margin(72);

            /* A short buffer truncates but still returns the full length. */
            let m = mxml_save_string(tree, buf.as_mut_ptr(), 20, None);
            assert_eq!(m, 59);
            assert_eq!(
                core::ffi::CStr::from_ptr(buf.as_ptr()).to_bytes(),
                b"<root><child x=\"1\" "
            );

            mxml_delete(tree);
        }
    }

    /// `_mxml_global()` starts `wrap` at 72 (`mxml-private.c:167`), so a long
    /// attribute list breaks at column 72 with no explicit
    /// `mxmlSetWrapMargin` call.  Native bytes below.
    #[test]
    fn default_wrap_margin_is_72_columns() {
        unsafe {
            let mut buf: [c_char; 8192] = [0; 8192];
            let xml = mxml_new_xml(c"1.0".as_ptr());
            let top = mxml_new_element(xml, c"root".as_ptr());
            let e = mxml_new_element(top, c"e".as_ptr());
            mxml_element_set_attr(e, c"aaaaaaaaaa".as_ptr(), c"1111111111".as_ptr());
            mxml_element_set_attr(e, c"bbbbbbbbbb".as_ptr(), c"2222222222".as_ptr());
            mxml_element_set_attr(e, c"cccccccccc".as_ptr(), c"3333333333".as_ptr());
            let n = mxml_save_string(xml, buf.as_mut_ptr(), 8192, None);
            assert_eq!(n, 129);
            assert_eq!(
                core::ffi::CStr::from_ptr(buf.as_ptr()).to_bytes(),
                b"<?xml version=\"1.0\" encoding=\"utf-8\"?><root><e aaaaaaaaaa=\"1111111111\"\nbbbbbbbbbb=\"2222222222\" cccccccccc=\"3333333333\" /></root>\n"
            );
            mxml_delete(xml);
        }
    }

    /// Character entities resolve through the full upstream table and are
    /// re-encoded as UTF-8 by `mxml_add_char`.  Native byte sequence for
    /// `&AElig;&#65;&nbsp;` is `c3 86 41 c2 a0`.
    #[test]
    fn named_and_numeric_entities_decode_to_utf8() {
        unsafe {
            let tree = mxml_load_string(
                core::ptr::null_mut(),
                c"<root>&AElig;&#65;&nbsp;</root>".as_ptr(),
                Some(mxml_opaque_cb),
            );
            assert!(!tree.is_null());
            let o = mxml_get_opaque(tree);
            assert!(!o.is_null());
            assert_eq!(
                core::ffi::CStr::from_ptr(o).to_bytes(),
                &[0xc3u8, 0x86, 0x41, 0xc2, 0xa0]
            );
            mxml_delete(tree);
        }
    }

    /// autodoc round-trip driver mirroring `scratchpad/xmldiff/adocrt.c`.
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
        let asxml: c_int = std::env::var("IMOD_XML_RT_ASXML").unwrap().parse().unwrap();
        let list = std::env::var("IMOD_XML_RT_FILES").unwrap();
        let files = std::fs::read_to_string(&list).unwrap();
        unsafe {
            let crep = CString::new(reppath).unwrap();
            let rep = libc::fopen(crep.as_ptr(), c"w".as_ptr());
            assert!(!rep.is_null());
            for line in files.lines() {
                if line.is_empty() {
                    continue;
                }
                let base = line.rsplit('/').next().unwrap();
                let cline = CString::new(line).unwrap();
                let cbase = CString::new(base).unwrap();
                let ind = adoc_read(cline.as_ptr());
                libc::fprintf(rep, c"%s read=%d".as_ptr(), cbase.as_ptr(), ind);
                if ind < 0 {
                    libc::fprintf(rep, c"\n".as_ptr());
                    continue;
                }
                let mut root: *mut c_char = core::ptr::null_mut();
                let err = adoc_get_xml_root_element(&raw mut root);
                libc::fprintf(
                    rep,
                    c" rootErr=%d root=%s".as_ptr(),
                    err,
                    if !root.is_null() {
                        root as *const c_char
                    } else {
                        c"(nil)".as_ptr()
                    },
                );
                if !root.is_null() {
                    libc::free(root as *mut c_void);
                }
                libc::fprintf(rep, c" xmlRead=%d".as_ptr(), adoc_get_write_as_xml());
                adoc_set_write_as_xml(asxml);
                let out = CString::new(format!("{}/{}.out", outdir, base)).unwrap();
                libc::fprintf(rep, c" write=%d\n".as_ptr(), adoc_write(out.as_ptr()));
                adoc_clear(ind);
                adoc_set_write_as_xml(0);
            }
            libc::fclose(rep);
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
        unsafe {
            let cout = CString::new(outpath).unwrap();
            OUT = libc::fopen(cout.as_ptr(), c"w".as_ptr());
            assert!(!OUT.is_null());
            probe_entities();
            probe_strings();
            probe_build();
            for line in files.lines() {
                if line.is_empty() {
                    continue;
                }
                let base = line.rsplit('/').next().unwrap();
                probe_file(line, base, &savedir);
            }
            libc::fclose(OUT);
            libc::fflush(stderr);
        }
    }
}
