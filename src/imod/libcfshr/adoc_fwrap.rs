//! Translation of `IMOD/libcfshr/adoc_fwrap.c`.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::imod::libcfshr::autodoc::*;
use crate::imod::libcfshr::b3dutil::{c2f_string, f2c_string};
use crate::imod::libcfshr::parse_params::pip_set_error;
use core::ffi::{CStr, c_char, c_void};

pub type FortStrLenT = i32;

/// Matches C static `adocf2cstr`.
///
/// `f2c_string` is the Fortran half of the bridge and keeps its `c_char`
/// buffer (NATIVE.md §7); the trimmed copy it returns is handed on as owned
/// bytes, because `autodoc.c`'s own entry points take bytes now.
pub unsafe fn adocf2cstr(string: *const c_char, string_size: FortStrLenT) -> Option<Vec<u8>> {
    let new_str = f2c_string(string, string_size);
    if new_str.is_null() {
        pip_set_error(b"Memory error converting string from Fortran to C");
        return None;
    }
    let owned = CStr::from_ptr(new_str).to_bytes().to_vec();
    Some(owned)
}

/// Matches C static `twof2cstr`.
pub unsafe fn twof2cstr(
    coll_name: *const c_char,
    key: *const c_char,
    coll_size: FortStrLenT,
    key_size: FortStrLenT,
    c_str: &mut Vec<u8>,
    k_str: &mut Vec<u8>,
) -> i32 {
    let Some(coll) = adocf2cstr(coll_name, coll_size) else {
        return -1;
    };
    *c_str = coll;
    let Some(k) = adocf2cstr(key, key_size) else {
        return -1;
    };
    *k_str = k;
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocread_(filename: *mut c_char, name_size: FortStrLenT) -> i32 {
    let Some(c_str) = adocf2cstr(filename, name_size) else {
        return -1;
    };
    let err = adoc_read(&c_str);
    if err >= 0 { err + 1 } else { err }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocxmlreadstatus_(
    a: *mut i32,
    b: *mut i32,
    c: *mut i32,
    d: *mut i32,
    e: *mut i32,
    f: *mut i32,
) -> i32 {
    adoc_xml_read_status(&mut *a, &mut *b, &mut *c, &mut *d, &mut *e, &mut *f)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocopenimagemetadata_(
    filename: *mut c_char,
    add_mdoc: *mut i32,
    montage: *mut i32,
    num_sect: *mut i32,
    sect_type: *mut i32,
    name_size: FortStrLenT,
) -> i32 {
    let Some(c_str) = adocf2cstr(filename, name_size) else {
        return -1;
    };
    let err = adoc_open_image_metadata(
        &c_str,
        *add_mdoc,
        &mut *montage,
        &mut *num_sect,
        &mut *sect_type,
    );
    if err >= 0 { err + 1 } else { err }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetimagemetainfo_(
    montage: *mut i32,
    num_sect: *mut i32,
    sect_type: *mut i32,
) -> i32 {
    adoc_get_image_meta_info(&mut *montage, &mut *num_sect, &mut *sect_type)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocnew_() -> i32 {
    adoc_new()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocsetcurrent_(index: *mut i32) -> i32 {
    adoc_set_current(*index - 1)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocclear_(index: *mut i32) {
    adoc_clear(*index - 1)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocdone_() {
    adoc_done()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocwrite_(filename: *mut c_char, name_size: FortStrLenT) -> i32 {
    let Some(c_str) = adocf2cstr(filename, name_size) else {
        return -1;
    };
    let err = adoc_write(&c_str);
    err
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocsetwriteasxml_(as_xml: *mut i32) {
    adoc_set_write_as_xml(*as_xml)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetwriteasxml_() -> i32 {
    adoc_get_write_as_xml()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetxmlrootelement_(
    element: *mut c_char,
    elem_size: FortStrLenT,
) -> i32 {
    let mut string: Option<Vec<u8>> = None;
    let mut err = adoc_get_xml_root_element(&mut string);
    // `c2f_string` is the Fortran half of the bridge and takes a C string.
    if err == 0 && string.is_none() {
        c2f_string(c" ".as_ptr(), element, elem_size);
    } else if err == 0 {
        let mut nul = string.clone().unwrap();
        nul.push(0);
        if c2f_string(nul.as_ptr().cast::<c_char>(), element, elem_size) != 0 {
            pip_set_error(b"In AdocGetXmlRootElement, string is too long for character variable");
            err = -1;
        }
    }
    err
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocsetxmlrootelement_(
    element: *mut c_char,
    elem_size: FortStrLenT,
) -> i32 {
    let Some(c_str) = adocf2cstr(element, elem_size) else {
        return -1;
    };
    let err = adoc_set_xml_root_element(&c_str);
    err
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocaddsection_(
    coll_name: *mut c_char,
    name: *mut c_char,
    coll_size: FortStrLenT,
    name_size: FortStrLenT,
) -> i32 {
    let (mut c_str, mut k_str) = (Vec::new(), Vec::new());
    if twof2cstr(
        coll_name, name, coll_size, name_size, &mut c_str, &mut k_str,
    ) != 0
    {
        return -1;
    }
    let err = adoc_add_section(&c_str, &k_str);
    if err >= 0 { err + 1 } else { err }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocorderwritebyvalue_(
    coll_name: *mut c_char,
    coll_size: FortStrLenT,
) -> i32 {
    let Some(c_str) = adocf2cstr(coll_name, coll_size) else {
        return -1;
    };
    let err = adoc_order_write_by_value(Some(&c_str));
    err
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocsetkeyvalue_(
    coll_name: *mut c_char,
    sect_ind: *mut i32,
    key: *mut c_char,
    value: *mut c_char,
    coll_size: i32,
    key_size: FortStrLenT,
    val_size: FortStrLenT,
) -> i32 {
    let Some(v_str) = adocf2cstr(value, val_size) else {
        return -1;
    };
    let (mut c_str, mut k_str) = (Vec::new(), Vec::new());
    if twof2cstr(coll_name, key, coll_size, key_size, &mut c_str, &mut k_str) != 0 {
        return -1;
    }
    let err = if !v_str.is_empty() {
        adoc_set_key_value(&c_str, *sect_ind - 1, &k_str, Some(&v_str))
    } else {
        adoc_set_key_value(&c_str, *sect_ind - 1, &k_str, None)
    };
    err
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocsetinteger_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    val: *mut i32,
    cs: i32,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_set_integer(&c, *si - 1, &k, *val);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocsettwointegers_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    a: *mut i32,
    b: *mut i32,
    cs: FortStrLenT,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_set_two_integers(&c, *si - 1, &k, *a, *b);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocsetthreeintegers_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    a: *mut i32,
    b: *mut i32,
    d: *mut i32,
    cs: FortStrLenT,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_set_three_integers(&c, *si - 1, &k, *a, *b, *d);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocsetintegerarray_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    vals: *mut i32,
    num: *mut i32,
    cs: FortStrLenT,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_set_integer_array(
        &c,
        *si - 1,
        &k,
        core::slice::from_raw_parts(vals, (*num).max(0) as usize),
        *num,
    );
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocsetfloat_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    val: *mut f32,
    cs: i32,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_set_float(&c, *si - 1, &k, *val);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocsetdouble_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    val: *mut f64,
    cs: i32,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_set_double(&c, *si - 1, &k, *val);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocsettwofloats_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    a: *mut f32,
    b: *mut f32,
    cs: FortStrLenT,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_set_two_floats(&c, *si - 1, &k, *a, *b);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocsetthreefloats_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    a: *mut f32,
    b: *mut f32,
    d: *mut f32,
    cs: FortStrLenT,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_set_three_floats(&c, *si - 1, &k, *a, *b, *d);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocsetfloatarray_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    vals: *mut f32,
    num: *mut i32,
    cs: FortStrLenT,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_set_float_array(
        &c,
        *si - 1,
        &k,
        core::slice::from_raw_parts(vals, (*num).max(0) as usize),
        *num,
    );
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocdeletekeyvalue_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    cs: FortStrLenT,
    ks: i32,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_delete_key_value(&c, *si - 1, &k);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetsectionname_(
    coll: *mut c_char,
    si: *mut i32,
    string: *mut c_char,
    cs: i32,
    ss: FortStrLenT,
) -> i32 {
    let Some(c) = adocf2cstr(coll, cs) else {
        return -1;
    };
    let mut p: Vec<u8> = Vec::new();
    let mut e = adoc_get_section_name(&c, *si - 1, &mut p);
    p.push(0);
    if e == 0 && c2f_string(p.as_ptr().cast::<c_char>(), string, ss) != 0 {
        pip_set_error(b"In AdocGetSectionName, string is too long for character variable");
        e = -1;
    }
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetnumberofsections_(coll: *mut c_char, cs: FortStrLenT) -> i32 {
    let Some(c) = adocf2cstr(coll, cs) else {
        return -1;
    };
    let e = adoc_get_number_of_sections(&c);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adoclookupsection_(
    typ: *mut c_char,
    name: *mut c_char,
    ts: FortStrLenT,
    ns: FortStrLenT,
) -> i32 {
    let (mut t, mut n) = (Vec::new(), Vec::new());
    if twof2cstr(typ, name, ts, ns, &mut t, &mut n) != 0 {
        return -1;
    }
    let e = adoc_lookup_section(&t, &n);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adoclookupbynamevalue_(
    typ: *mut c_char,
    value: *mut i32,
    ts: FortStrLenT,
) -> i32 {
    let Some(t) = adocf2cstr(typ, ts) else {
        return -1;
    };
    let e = adoc_lookup_by_name_value(&t, *value);
    if e >= 0 { e + 1 } else { e }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocfindinsertindex_(
    typ: *mut c_char,
    value: *mut i32,
    ts: FortStrLenT,
) -> i32 {
    let Some(t) = adocf2cstr(typ, ts) else {
        return -1;
    };
    let e = adoc_find_insert_index(&t, *value);
    if e >= 0 { e + 1 } else { e }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocinsertsection_(
    typ: *const c_char,
    si: *mut i32,
    name: *const c_char,
    ts: FortStrLenT,
    ns: i32,
) -> i32 {
    let (mut t, mut n) = (Vec::new(), Vec::new());
    if twof2cstr(typ, name, ts, ns, &mut t, &mut n) != 0 {
        return -1;
    }
    let e = adoc_insert_section(&t, *si - 1, &n);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetnumcollections_() -> i32 {
    adoc_get_num_collections()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetcollectionname_(
    ci: *mut i32,
    string: *mut c_char,
    ss: FortStrLenT,
) -> i32 {
    let mut p: Vec<u8> = Vec::new();
    let mut e = adoc_get_collection_name(*ci - 1, &mut p);
    p.push(0);
    if e == 0 && c2f_string(p.as_ptr().cast::<c_char>(), string, ss) != 0 {
        pip_set_error(b"In AdocGetCollectionName, string is too long for character variable");
        e = -1;
    }
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adoctransfersection_(
    typ: *mut c_char,
    si: *mut i32,
    to: *mut i32,
    name: *mut c_char,
    by: *mut i32,
    ts: FortStrLenT,
    ns: FortStrLenT,
) -> i32 {
    let (mut t, mut n) = (Vec::new(), Vec::new());
    if twof2cstr(typ, name, ts, ns, &mut t, &mut n) != 0 {
        return -1;
    }
    let e = adoc_transfer_section(&t, *si - 1, *to - 1, Some(&n), *by);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adoctransfertonewtype_(
    typ: *mut c_char,
    si: *mut i32,
    to: *mut i32,
    new_typ: *mut c_char,
    new_name: *mut c_char,
    by: *mut i32,
    ts: FortStrLenT,
    new_size: FortStrLenT,
    ns: FortStrLenT,
) -> i32 {
    let Some(new_str) = adocf2cstr(new_typ, new_size) else {
        return -1;
    };
    let (mut t, mut n) = (Vec::new(), Vec::new());
    if twof2cstr(typ, new_name, ts, ns, &mut t, &mut n) != 0 {
        return -1;
    }
    let e = adoc_transfer_to_new_type(&t, *si - 1, *to - 1, &new_str, Some(&n), *by);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetstring_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    string: *mut c_char,
    cs: i32,
    ks: FortStrLenT,
    ss: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let mut p: Vec<u8> = Vec::new();
    let mut e = adoc_get_string(&c, *si - 1, &k, &mut p);
    p.push(0);
    if e == 0 && c2f_string(p.as_ptr().cast::<c_char>(), string, ss) != 0 {
        pip_set_error(b"In AdocGetString, string is too long for character variable");
        e = -1;
    }
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetinteger_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    val: *mut i32,
    cs: i32,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_get_integer(&c, *si - 1, &k, &mut *val);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgettwointegers_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    a: *mut i32,
    b: *mut i32,
    cs: FortStrLenT,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_get_two_integers(&c, *si - 1, &k, &mut *a, &mut *b);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetthreeintegers_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    a: *mut i32,
    b: *mut i32,
    d: *mut i32,
    cs: FortStrLenT,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_get_three_integers(&c, *si - 1, &k, &mut *a, &mut *b, &mut *d);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetintegerarray_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    array: *mut i32,
    num: *mut i32,
    size: *mut i32,
    cs: FortStrLenT,
    ks: i32,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_get_integer_array(
        &c,
        *si - 1,
        &k,
        core::slice::from_raw_parts_mut(array, (*size).max(0) as usize),
        &mut *num,
        *size,
    );
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetfloat_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    val: *mut f32,
    cs: i32,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_get_float(&c, *si - 1, &k, &mut *val);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetdouble_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    val: *mut f64,
    cs: i32,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_get_double(&c, *si - 1, &k, &mut *val);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgettwofloats_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    a: *mut f32,
    b: *mut f32,
    cs: FortStrLenT,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_get_two_floats(&c, *si - 1, &k, &mut *a, &mut *b);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetthreefloats_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    a: *mut f32,
    b: *mut f32,
    d: *mut f32,
    cs: FortStrLenT,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_get_three_floats(&c, *si - 1, &k, &mut *a, &mut *b, &mut *d);
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetfloatarray_(
    coll: *mut c_char,
    si: *mut i32,
    key: *mut c_char,
    array: *mut f32,
    num: *mut i32,
    size: *mut i32,
    cs: FortStrLenT,
    ks: FortStrLenT,
) -> i32 {
    let (mut c, mut k) = (Vec::new(), Vec::new());
    if twof2cstr(coll, key, cs, ks, &mut c, &mut k) != 0 {
        return -1;
    }
    let e = adoc_get_float_array(
        &c,
        *si - 1,
        &k,
        core::slice::from_raw_parts_mut(array, (*size).max(0) as usize),
        &mut *num,
        *size,
    );
    e
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adocgetstandardnames_(
    global: *mut c_char,
    zvalue: *mut c_char,
    gs: FortStrLenT,
    zs: i32,
) -> i32 {
    let mut e = 0;
    if c2f_string(c"PreData".as_ptr(), global, gs) != 0 {
        pip_set_error(b"In AdocGetStandardNames, global name is too long for character variable");
        e = -1;
    }
    if e == 0 && c2f_string(c"ZValue".as_ptr(), zvalue, zs) != 0 {
        pip_set_error(b"In AdocGetSectionName, zvalue name is too long for character variable");
        e = -1;
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Shared with `autodoc::tests`: both drive the one process-global
    /// autodoc collection.
    use crate::imod::libcfshr::autodoc::tests::TEST_LOCK;

    #[test]
    fn fortran_wrappers_convert_trim_pad_and_one_base_indices() {
        let _lock = TEST_LOCK.lock().unwrap();
        unsafe {
            adocdone_();
            assert!(adocnew_() >= 0);
            let mut collection = *b"PreData  ";
            let mut key = *b"Count ";
            let mut section = 1;
            let mut value = 37;
            assert_eq!(
                adocsetinteger_(
                    collection.as_mut_ptr().cast(),
                    &mut section,
                    key.as_mut_ptr().cast(),
                    &mut value,
                    collection.len() as i32,
                    key.len() as i32,
                ),
                0
            );
            value = 0;
            assert_eq!(
                adocgetinteger_(
                    collection.as_mut_ptr().cast(),
                    &mut section,
                    key.as_mut_ptr().cast(),
                    &mut value,
                    collection.len() as i32,
                    key.len() as i32,
                ),
                0
            );
            assert_eq!(value, 37);
            let mut global = [0i8; 8];
            let mut zvalue = [0i8; 8];
            assert_eq!(
                adocgetstandardnames_(global.as_mut_ptr(), zvalue.as_mut_ptr(), 8, 8),
                0
            );
            assert_eq!(
                &global,
                &[
                    b'P' as i8, b'r' as i8, b'e' as i8, b'D' as i8, b'a' as i8, b't' as i8,
                    b'a' as i8, b' ' as i8
                ]
            );
            assert_eq!(
                &zvalue[..6],
                &[
                    b'Z' as i8, b'V' as i8, b'a' as i8, b'l' as i8, b'u' as i8, b'e' as i8
                ]
            );
        }
    }

    #[test]
    fn wrapper_add_section_returns_fortran_one_base_index() {
        let _lock = TEST_LOCK.lock().unwrap();
        unsafe {
            adocdone_();
            adocnew_();
            let mut collection = *b"ZValue";
            let mut name = *b"42  ";
            assert_eq!(
                adocaddsection_(
                    collection.as_mut_ptr().cast(),
                    name.as_mut_ptr().cast(),
                    collection.len() as i32,
                    name.len() as i32,
                ),
                1
            );
            assert_eq!(
                adocgetnumberofsections_(collection.as_mut_ptr().cast(), collection.len() as i32),
                1
            );
        }
    }

    /// An all-blank Fortran value makes `adocsetkeyvalue` pass NULL to
    /// `AdocSetKeyValue`, and `setKeyValueType` rejects a NULL value with -1
    /// (`autodoc.c:1125`); nothing is stored, so `adocgetstring` then returns 1
    /// and leaves the caller's buffer untouched (`adoc_fwrap.c:533`).
    /// Verified against the reference `libcfshr.so`: `adocsetkeyvalue_` -> -1,
    /// `adocgetstring_` -> 1.
    #[test]
    fn empty_fortran_value_is_rejected() {
        let _lock = TEST_LOCK.lock().unwrap();
        unsafe {
            adocdone_();
            adocnew_();
            let mut collection = *b"PreData";
            let mut key = *b"Empty";
            let mut value = *b"    ";
            let mut section = 1;
            assert_eq!(
                adocsetkeyvalue_(
                    collection.as_mut_ptr().cast(),
                    &mut section,
                    key.as_mut_ptr().cast(),
                    value.as_mut_ptr().cast(),
                    collection.len() as i32,
                    key.len() as i32,
                    value.len() as i32,
                ),
                -1
            );
            let mut output = [0i8; 8];
            assert_eq!(
                adocgetstring_(
                    collection.as_mut_ptr().cast(),
                    &mut section,
                    key.as_mut_ptr().cast(),
                    output.as_mut_ptr(),
                    collection.len() as i32,
                    key.len() as i32,
                    output.len() as i32,
                ),
                1
            );
            assert_eq!(output, [0i8; 8]);
        }
    }
}
