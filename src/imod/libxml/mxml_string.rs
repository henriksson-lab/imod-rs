//! Translation of `IMOD/libxml/mxml-string.c`.
//!
//! `config.h` defines `HAVE_SNPRINTF`, `HAVE_STRDUP` and `HAVE_VSNPRINTF` on
//! every non-Windows build, so `_mxml_snprintf`, `_mxml_strdup` and
//! `_mxml_vsnprintf` are preprocessed out of this compilation unit and are not
//! part of the vendored library.  Only `_mxml_strdupf` and `_mxml_vstrdupf`
//! are compiled, and both are translated below.
//!
//! Stable Rust cannot *define* a C-variadic function, so both take their single
//! variable argument explicitly, as an already-converted byte string.  That
//! covers every call in the vendored library: `_mxml_strdupf("![CDATA[%s]]",
//! data)` in `mxml-node.c` and `mxml-set.c`, and the `va_list` that
//! `mxmlNewTextf` and `mxmlSetTextf` forward.  A caller with a numeric
//! conversion (`mxmlwrap.c:375` passes `"%d"`) must convert its argument
//! through the shared C-format writer first; reproducing `vsnprintf`'s
//! specifier set here would duplicate that writer, which NATIVE.md keeps in
//! `b3dutil` as one verified boundary translation.

/// Matches C `_mxml_strdupf` (`mxml-string.c:89`).
pub fn _mxml_strdupf(format: &[u8], arg: &[u8]) -> Vec<u8> {
    /*
     * Get a pointer to the additional arguments, format the string,
     * and return it...
     */

    _mxml_vstrdupf(format, arg)
}

/// Matches C `_mxml_vstrdupf` (`mxml-string.c:425`).
///
/// The C function formats into a 256-byte stack buffer, `strdup`s it when it
/// fitted and otherwise `calloc`s `bytes + 1` and formats again; both arms
/// produce the same string, and the owned `Vec` replaces both allocations.
pub fn _mxml_vstrdupf(format: &[u8], arg: &[u8]) -> Vec<u8> {
    let mut buffer: Vec<u8> = Vec::new();
    let mut used = false;
    let mut i = 0;

    while i < format.len() {
        if format[i] != b'%' {
            buffer.push(format[i]);
            i += 1;
            continue;
        }

        /*
         * A conversion: "%%" is a literal percent, anything else consumes the
         * one variable argument.
         */

        i += 1;
        if i < format.len() && format[i] == b'%' {
            buffer.push(b'%');
            i += 1;
            continue;
        }

        while i < format.len() && !format[i].is_ascii_alphabetic() {
            i += 1;
        }
        if i < format.len() {
            i += 1;
        }

        if !used {
            buffer.extend_from_slice(arg);
            used = true;
        }
    }

    buffer
}
