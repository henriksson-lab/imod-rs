use super::*;
use core::ffi::{c_char, c_int, c_void};
use std::ffi::{CStr, CString};

unsafe fn parse_xml(top: *mut MxmlNode, source: &str, cb: MxmlLoadCb) -> *mut MxmlNode {
    unsafe {
        let mut root = top;
        let mut current = top;
        let mut position = 0usize;
        let bytes = source.as_bytes();
        while position < bytes.len() {
            if bytes[position] == b'<' {
                let Some(offset) = source[position..].find('>') else {
                    break;
                };
                let raw = &source[position + 1..position + offset];
                position += offset + 1;
                if raw.starts_with('/') {
                    if !current.is_null() {
                        current = (*current).parent;
                    }
                    continue;
                }
                if raw.starts_with('!') || raw.starts_with('?') {
                    let x = CString::new(raw).unwrap();
                    let n = mxml_new_element(current, x.as_ptr());
                    if root.is_null() {
                        root = n;
                    }
                    continue;
                }
                let close = raw.ends_with('/');
                let body = raw.trim_end_matches('/').trim();
                let mut words = body.split_whitespace();
                let Some(name) = words.next() else { continue };
                let key = CString::new(name).unwrap();
                let node = mxml_new_element(current, key.as_ptr());
                if root.is_null() {
                    root = node;
                }
                let mut attributes = body[name.len()..].trim();
                while !attributes.is_empty() {
                    let Some(eq) = attributes.find('=') else {
                        break;
                    };
                    let an = attributes[..eq].trim();
                    attributes = attributes[eq + 1..].trim_start();
                    let quote = attributes.as_bytes().first().copied().unwrap_or(b' ');
                    let (av, rest) = if quote == b'\'' || quote == b'\"' {
                        match attributes[1..].find(quote as char) {
                            Some(end) => (&attributes[1..end + 1], &attributes[end + 2..]),
                            None => (attributes, ""),
                        }
                    } else {
                        let end = attributes
                            .find(char::is_whitespace)
                            .unwrap_or(attributes.len());
                        (&attributes[..end], &attributes[end..])
                    };
                    let an = CString::new(an).unwrap();
                    let av = CString::new(av).unwrap();
                    mxml_element_set_attr(node, an.as_ptr(), av.as_ptr());
                    attributes = rest.trim_start();
                }
                if !close {
                    current = node;
                }
            } else {
                let end = source[position..]
                    .find('<')
                    .map(|x| position + x)
                    .unwrap_or(bytes.len());
                let text = source[position..end].trim();
                if !text.is_empty() && !current.is_null() {
                    let z = CString::new(text).unwrap();
                    match cb.map(|f| f(current)).unwrap_or(MXML_OPAQUE) {
                        MXML_TEXT => {
                            mxml_new_text(current, 0, z.as_ptr());
                        }
                        MXML_INTEGER => {
                            mxml_new_integer(current, text.parse().unwrap_or(0));
                        }
                        MXML_REAL => {
                            mxml_new_real(current, text.parse().unwrap_or(0.0));
                        }
                        MXML_IGNORE => {}
                        _ => {
                            mxml_new_opaque(current, z.as_ptr());
                        }
                    }
                }
                position = end;
            }
        }
        root
    }
}
pub unsafe fn mxml_load_string(
    top: *mut MxmlNode,
    s: *const c_char,
    cb: MxmlLoadCb,
) -> *mut MxmlNode {
    unsafe {
        if s.is_null() {
            core::ptr::null_mut()
        } else {
            parse_xml(top, &CStr::from_ptr(s).to_string_lossy(), cb)
        }
    }
}
pub unsafe fn mxml_load_file(
    top: *mut MxmlNode,
    fp: *mut libc::FILE,
    cb: MxmlLoadCb,
) -> *mut MxmlNode {
    unsafe {
        if fp.is_null() {
            return core::ptr::null_mut();
        }
        let mut data = Vec::new();
        let mut ch = libc::fgetc(fp);
        while ch != libc::EOF {
            data.push(ch as u8);
            ch = libc::fgetc(fp);
        }
        parse_xml(top, &String::from_utf8_lossy(&data), cb)
    }
}
pub unsafe fn mxml_load_fd(top: *mut MxmlNode, fd: c_int, cb: MxmlLoadCb) -> *mut MxmlNode {
    unsafe {
        let copy = libc::dup(fd);
        if copy < 0 {
            return core::ptr::null_mut();
        }
        let fp = libc::fdopen(copy, c"r".as_ptr());
        if fp.is_null() {
            libc::close(copy);
            return core::ptr::null_mut();
        }
        let node = mxml_load_file(top, fp, cb);
        libc::fclose(fp);
        node
    }
}
unsafe fn render(n: *mut MxmlNode, out: &mut String) {
    unsafe {
        match (*n).type_ {
            MXML_ELEMENT => {
                let name = CStr::from_ptr((*n).value.element.name).to_string_lossy();
                if name.starts_with('?') || name.starts_with('!') {
                    out.push('<');
                    out.push_str(&name);
                    out.push('>');
                    return;
                }
                out.push('<');
                out.push_str(&name);
                let e = (*n).value.element;
                for i in 0..e.num_attrs {
                    let a = e.attrs.add(i as usize);
                    out.push(' ');
                    out.push_str(&CStr::from_ptr((*a).name).to_string_lossy());
                    out.push_str("=\"");
                    if !(*a).value.is_null() {
                        out.push_str(&CStr::from_ptr((*a).value).to_string_lossy())
                    };
                    out.push('\"');
                }
                if (*n).child.is_null() {
                    out.push_str("/>")
                } else {
                    out.push('>');
                    let mut child = (*n).child;
                    while !child.is_null() {
                        render(child, out);
                        child = (*child).next
                    }
                    out.push_str("</");
                    out.push_str(&name);
                    out.push('>');
                }
            }
            MXML_OPAQUE => out.push_str(&CStr::from_ptr((*n).value.opaque).to_string_lossy()),
            MXML_TEXT => out.push_str(&CStr::from_ptr((*n).value.text.string).to_string_lossy()),
            MXML_INTEGER => out.push_str(&(*n).value.integer.to_string()),
            MXML_REAL => out.push_str(&(*n).value.real.to_string()),
            _ => {}
        }
    }
}
pub unsafe fn mxml_save_string(
    node: *mut MxmlNode,
    buffer: *mut c_char,
    size: c_int,
    _: MxmlSaveCb,
) -> c_int {
    unsafe {
        if node.is_null() || buffer.is_null() || size <= 0 {
            return -1;
        }
        let mut out = String::new();
        render(node, &mut out);
        let n = out.len().min(size as usize - 1);
        core::ptr::copy_nonoverlapping(out.as_ptr().cast(), buffer, n);
        *buffer.add(n) = 0;
        out.len() as c_int
    }
}
pub unsafe fn mxml_save_file(node: *mut MxmlNode, fp: *mut libc::FILE, _: MxmlSaveCb) -> c_int {
    unsafe {
        if node.is_null() || fp.is_null() {
            return -1;
        }
        let mut out = String::new();
        render(node, &mut out);
        if libc::fwrite(out.as_ptr().cast(), 1, out.len(), fp) != out.len() {
            -1
        } else if libc::fputc(b'\n' as c_int, fp) < 0 {
            -1
        } else {
            0
        }
    }
}
pub unsafe fn mxml_save_alloc_string(node: *mut MxmlNode, _: MxmlSaveCb) -> *mut c_char {
    unsafe {
        if node.is_null() {
            return core::ptr::null_mut();
        }
        let mut out = String::new();
        render(node, &mut out);
        CString::new(out).unwrap().into_raw()
    }
}
pub unsafe fn mxml_save_fd(node: *mut MxmlNode, fd: c_int, cb: MxmlSaveCb) -> c_int {
    unsafe {
        let copy = libc::dup(fd);
        if copy < 0 {
            return -1;
        }
        let fp = libc::fdopen(copy, c"w".as_ptr());
        if fp.is_null() {
            libc::close(copy);
            return -1;
        }
        let result = mxml_save_file(node, fp, cb);
        libc::fclose(fp);
        result
    }
}
pub unsafe fn mxml_sax_load_string(
    top: *mut MxmlNode,
    s: *const c_char,
    cb: MxmlLoadCb,
    _: MxmlSaxCb,
    _: *mut c_void,
) -> *mut MxmlNode {
    unsafe { mxml_load_string(top, s, cb) }
}
pub unsafe fn mxml_sax_load_file(
    top: *mut MxmlNode,
    fp: *mut libc::FILE,
    cb: MxmlLoadCb,
    _: MxmlSaxCb,
    _: *mut c_void,
) -> *mut MxmlNode {
    unsafe { mxml_load_file(top, fp, cb) }
}
pub unsafe fn mxml_sax_load_fd(
    top: *mut MxmlNode,
    fd: c_int,
    cb: MxmlLoadCb,
    _: MxmlSaxCb,
    _: *mut c_void,
) -> *mut MxmlNode {
    unsafe { mxml_load_fd(top, fd, cb) }
}
pub unsafe fn mxml_set_custom_handlers(load: MxmlCustomLoadCb, save: MxmlCustomSaveCb) {
    let mut g = mxml_global().lock().unwrap();
    g.custom_load_cb = load;
    g.custom_save_cb = save
}
pub unsafe fn mxml_set_error_callback(cb: MxmlErrorCb) {
    mxml_global().lock().unwrap().error_cb = cb
}
pub unsafe fn mxml_set_wrap_margin(column: c_int) {
    mxml_global().lock().unwrap().wrap = column
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loads_vendored_ant_build_file() {
        let path = CString::new("IMOD/Etomo/build.xml").unwrap();
        unsafe {
            let file = libc::fopen(path.as_ptr(), c"r".as_ptr());
            assert!(!file.is_null());
            let root = mxml_load_file(core::ptr::null_mut(), file, Some(mxml_opaque_cb));
            libc::fclose(file);
            assert!(!root.is_null());
            assert!(!mxml_get_element(root).is_null());
            mxml_delete(root);
        }
    }
}
