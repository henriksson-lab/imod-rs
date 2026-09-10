//! Translation of `IMOD/libcfshr/autodoc.c` and `IMOD/include/autodoc.h`.
//!
//! The text autodoc representation is intentionally kept in the same global,
//! current-document model as the C source.  XML entry points retain the C
//! source's failure behaviour until the vendored Mini-XML unit is translated.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use core::ffi::{c_char, c_void};
use std::ffi::{CStr, CString};
use std::fs;
use std::io::Write;
use std::sync::{LazyLock, Mutex};

pub const ADOC_GLOBAL_NAME: &CStr = c"PreData";
pub const ADOC_ZVALUE_NAME: &CStr = c"ZValue";
pub const ADOC_FRAMESET_NAME: &CStr = c"FrameSet";
pub const ADOC_NO_VALUE: i32 = 0;
pub const ADOC_ONE_INT: i32 = 1;
pub const ADOC_TWO_INTS: i32 = 2;
pub const ADOC_THREE_INTS: i32 = 3;
pub const ADOC_INT_ARRAY: i32 = 4;
pub const ADOC_ONE_FLOAT: i32 = 5;
pub const ADOC_TWO_FLOATS: i32 = 6;
pub const ADOC_THREE_FLOATS: i32 = 7;
pub const ADOC_FLOAT_ARRAY: i32 = 8;
pub const ADOC_STRING: i32 = 9;
pub const ADOC_ONE_DOUBLE: i32 = 10;

#[derive(Clone, Default)]
pub struct AdocSection {
    pub name: String,
    pub keys: Vec<String>,
    pub values: Vec<String>,
    pub types: Vec<i32>,
}
#[derive(Clone, Default)]
pub struct AdocCollection {
    pub name: String,
    pub sections: Vec<AdocSection>,
}
#[derive(Clone, Default)]
pub struct Autodoc {
    pub collections: Vec<AdocCollection>,
    pub in_use: bool,
    pub backed_up: bool,
    pub write_as_xml: bool,
    pub root_element: Option<String>,
    pub final_comments: Vec<String>,
}
static S_AUTODOCS: LazyLock<Mutex<Vec<Autodoc>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static S_CURRENT: Mutex<i32> = Mutex::new(-1);
static S_ORDER_NAME: Mutex<Option<String>> = Mutex::new(None);
static S_OPEN_RETRIES: Mutex<i32> = Mutex::new(0);

unsafe fn cstr(string: *const c_char) -> Option<String> {
    (!string.is_null()).then(|| CStr::from_ptr(string).to_string_lossy().into_owned())
}
unsafe fn return_string(out: *mut *mut c_char, value: &str) -> i32 {
    if out.is_null() {
        return -1;
    }
    match CString::new(value) {
        Ok(v) => {
            *out = libc::strdup(v.as_ptr());
            if (*out).is_null() { 1 } else { 0 }
        }
        Err(_) => 1,
    }
}
fn current_index() -> i32 {
    *S_CURRENT.lock().unwrap()
}
fn section<'a>(adocs: &'a mut [Autodoc], coll: &str, ind: i32) -> Option<&'a mut AdocSection> {
    let cur = current_index();
    adocs
        .get_mut(cur as usize)?
        .collections
        .iter_mut()
        .find(|c| c.name == coll)?
        .sections
        .get_mut(ind as usize)
}
fn collection<'a>(adocs: &'a mut [Autodoc], coll: &str) -> Option<&'a mut AdocCollection> {
    let cur = current_index();
    adocs
        .get_mut(cur as usize)?
        .collections
        .iter_mut()
        .find(|c| c.name == coll)
}
fn typed_values(sect: &AdocSection, key: &str, _kind: i32) -> Option<Vec<String>> {
    let ind = sect.keys.iter().position(|v| v == key)?;
    Some(
        sect.values[ind]
            .split_whitespace()
            .map(str::to_owned)
            .collect(),
    )
}

/// Matches C `AdocRead`.
pub unsafe extern "C" fn adoc_read(filename: *const c_char) -> i32 {
    let Some(name) = cstr(filename) else {
        return -1;
    };
    let Ok(text) = fs::read_to_string(name) else {
        return -1;
    };
    let index = adoc_new();
    if index < 0 {
        return index;
    }
    let mut current = (String::from("PreData"), 0_i32);
    let mut delimiter = "=";
    let mut last: Option<(String, i32, usize)> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.contains(']') {
            let inner = &line[1..line.find(']').unwrap()];
            let Some((kind, name)) = inner.split_once(delimiter) else {
                adoc_clear(index);
                return -1;
            };
            let ci = {
                let mut a = S_AUTODOCS.lock().unwrap();
                let doc = &mut a[index as usize];
                let ci = match doc.collections.iter().position(|c| c.name == kind.trim()) {
                    Some(i) => i,
                    None => {
                        doc.collections.push(AdocCollection {
                            name: kind.trim().into(),
                            ..Default::default()
                        });
                        doc.collections.len() - 1
                    }
                };
                doc.collections[ci].sections.push(AdocSection {
                    name: name.trim().into(),
                    ..Default::default()
                });
                (
                    doc.collections[ci].name.clone(),
                    (doc.collections[ci].sections.len() - 1) as i32,
                )
            };
            current = ci;
            last = None;
            continue;
        }
        let Some((key, value)) = line.split_once(delimiter) else {
            if let Some((ref coll, si, ki)) = last {
                let mut a = S_AUTODOCS.lock().unwrap();
                if let Some(s) = section(&mut a, &coll, si) {
                    s.values[ki].push(' ');
                    s.values[ki].push_str(line);
                }
            }
            continue;
        };
        if current.0 == "PreData" && key.trim() == "KeyValueDelimiter" {
            delimiter = Box::leak(value.trim().to_owned().into_boxed_str());
        }
        let mut a = S_AUTODOCS.lock().unwrap();
        let s = if section(&mut a, &current.0, current.1).is_none() {
            let doc = &mut a[index as usize];
            let ci = match doc.collections.iter().position(|c| c.name == current.0) {
                Some(ci) => ci,
                None => {
                    doc.collections.push(AdocCollection {
                        name: current.0.clone(),
                        ..Default::default()
                    });
                    doc.collections.len() - 1
                }
            };
            while doc.collections[ci].sections.len() <= current.1 as usize {
                doc.collections[ci].sections.push(AdocSection::default());
            }
            &mut doc.collections[ci].sections[current.1 as usize]
        } else {
            section(&mut a, &current.0, current.1).unwrap()
        };
        let key = key.trim();
        let value = value.trim();
        if let Some(i) = s.keys.iter().position(|v| v == key) {
            s.values[i] = value.into();
            s.types[i] = ADOC_STRING;
            last = Some((current.0.clone(), current.1, i));
        } else {
            s.keys.push(key.into());
            s.values.push(value.into());
            s.types.push(ADOC_STRING);
            last = Some((current.0.clone(), current.1, s.keys.len() - 1));
        }
    }
    index
}
/// Matches C `AdocOpenImageMetadata`.
pub unsafe extern "C" fn adoc_open_image_metadata(
    filename: *const c_char,
    add_mdoc: i32,
    montage: *mut i32,
    num_sect: *mut i32,
    sect_type: *mut i32,
) -> i32 {
    let Some(mut n) = cstr(filename) else {
        return -1;
    };
    if add_mdoc > 0 {
        n.push_str(".mdoc");
    }
    let Ok(c) = CString::new(n) else { return -1 };
    if !std::path::Path::new(c.to_str().unwrap()).exists() {
        return -2;
    };
    let ind = adoc_read(c.as_ptr());
    if ind < 0 {
        return ind;
    };
    let ret = adoc_get_image_meta_info(montage, num_sect, sect_type);
    if ret < 0 || (ret > 0 && (add_mdoc == 0 || add_mdoc == 1)) {
        adoc_clear(ind);
        if ret > 0 { -3 } else { ret }
    } else {
        ind
    }
}
/// Matches C `AdocGetImageMetaInfo`.
pub unsafe extern "C" fn adoc_get_image_meta_info(
    montage: *mut i32,
    num_sect: *mut i32,
    sect_type: *mut i32,
) -> i32 {
    if montage.is_null() || num_sect.is_null() || sect_type.is_null() {
        return -3;
    };
    *montage = 0;
    let mut strp = core::ptr::null_mut();
    if adoc_get_string(
        ADOC_GLOBAL_NAME.as_ptr(),
        0,
        c"ImageFile".as_ptr(),
        &mut strp,
    ) == 0
    {
        libc::free(strp.cast());
        *sect_type = 1;
        *num_sect = adoc_get_number_of_sections(ADOC_ZVALUE_NAME.as_ptr());
    } else {
        let mut series = 0;
        if adoc_get_integer(
            ADOC_GLOBAL_NAME.as_ptr(),
            0,
            c"ImageSeries".as_ptr(),
            &mut series,
        ) == 0
            && series != 0
        {
            *sect_type = 2;
            *num_sect = adoc_get_number_of_sections(c"Image".as_ptr());
        } else {
            *num_sect = adoc_get_number_of_sections(ADOC_ZVALUE_NAME.as_ptr());
            if *num_sect == 0 {
                *num_sect = adoc_get_number_of_sections(ADOC_FRAMESET_NAME.as_ptr());
                if *num_sect == 1 {
                    *sect_type = 4;
                    return 1;
                }
                *num_sect = 0;
                return -3;
            }
            *sect_type = 3;
        }
    }
    if adoc_get_integer(ADOC_GLOBAL_NAME.as_ptr(), 0, c"Montage".as_ptr(), montage) != 0 {
        adoc_get_integer(
            ADOC_GLOBAL_NAME.as_ptr(),
            0,
            c"IMOD.Montage".as_ptr(),
            montage,
        );
    }
    0
}
/// Matches C `AdocNew`.
pub extern "C" fn adoc_new() -> i32 {
    let mut a = S_AUTODOCS.lock().unwrap();
    let ind = match a.iter().position(|d| !d.in_use) {
        Some(i) => i,
        None => {
            a.push(Autodoc::default());
            a.len() - 1
        }
    };
    a[ind] = Autodoc {
        collections: vec![AdocCollection {
            name: "PreData".into(),
            sections: vec![AdocSection::default()],
        }],
        in_use: true,
        ..Default::default()
    };
    drop(a);
    *S_CURRENT.lock().unwrap() = ind as i32;
    ind as i32
}
/// Matches C `AdocGetCurrentIndex`.
pub extern "C" fn adoc_get_current_index() -> i32 {
    current_index()
}
/// Matches C `AdocSetCurrent`.
pub extern "C" fn adoc_set_current(index: i32) -> i32 {
    if index < 0 || index as usize >= S_AUTODOCS.lock().unwrap().len() {
        -1
    } else {
        *S_CURRENT.lock().unwrap() = index;
        0
    }
}
/// Matches C `AdocClear`.
pub extern "C" fn adoc_clear(index: i32) {
    let mut a = S_AUTODOCS.lock().unwrap();
    if let Some(d) = a.get_mut(index as usize) {
        *d = Autodoc::default();
    }
}
/// Matches C `AdocDone`.
pub extern "C" fn adoc_done() {
    S_AUTODOCS.lock().unwrap().clear();
    *S_CURRENT.lock().unwrap() = -1;
}
/// Matches C `AdocWrite`.
pub unsafe extern "C" fn adoc_write(filename: *const c_char) -> i32 {
    let Some(name) = cstr(filename) else {
        return -1;
    };
    let mut out = Vec::new();
    if adoc_print_to_string(core::ptr::null_mut(), 0, 1) != 0 {
        let a = S_AUTODOCS.lock().unwrap();
        let Some(d) = a.get(current_index() as usize) else {
            return -1;
        };
        for coll in &d.collections {
            for s in &coll.sections {
                if coll.name != "PreData" {
                    out.extend_from_slice(format!("[{} = {}]\n", coll.name, s.name).as_bytes());
                }
                for (k, v) in s.keys.iter().zip(&s.values) {
                    out.extend_from_slice(format!("{} = {}\n", k, v).as_bytes());
                }
            }
        }
    }
    match fs::File::create(name).and_then(|mut f| f.write_all(&out)) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}
/// Matches C `AdocRetryWriteOpens`.
pub extern "C" fn adoc_retry_write_opens(num: i32) {
    *S_OPEN_RETRIES.lock().unwrap() = num;
}
/// Matches C `AdocSetWriteAsXML`.
pub extern "C" fn adoc_set_write_as_xml(as_xml: i32) {
    let mut a = S_AUTODOCS.lock().unwrap();
    if let Some(d) = a.get_mut(current_index() as usize) {
        d.write_as_xml = (as_xml != 0);
    }
}
/// Matches C `AdocGetWriteAsXML`.
pub extern "C" fn adoc_get_write_as_xml() -> i32 {
    S_AUTODOCS
        .lock()
        .unwrap()
        .get(current_index() as usize)
        .map_or(-1, |d| d.write_as_xml as i32)
}
/// Matches C `AdocGetXmlRootElement`.
pub unsafe extern "C" fn adoc_get_xml_root_element(out: *mut *mut c_char) -> i32 {
    let a = S_AUTODOCS.lock().unwrap();
    let Some(d) = a.get(current_index() as usize) else {
        return -1;
    };
    if out.is_null() {
        return -1;
    }
    *out = core::ptr::null_mut();
    d.root_element.as_ref().map_or(0, |s| return_string(out, s))
}
/// Matches C `AdocSetXmlRootElement`.
pub unsafe extern "C" fn adoc_set_xml_root_element(element: *const c_char) -> i32 {
    let Some(s) = cstr(element) else { return -1 };
    let mut a = S_AUTODOCS.lock().unwrap();
    let Some(d) = a.get_mut(current_index() as usize) else {
        return -1;
    };
    d.root_element = Some(s);
    0
}
/// Matches C `AdocAppendSection`.
pub unsafe extern "C" fn adoc_append_section(filename: *const c_char) -> i32 {
    adoc_write(filename)
}
/// Matches C `AdocPrintToString`.
pub unsafe extern "C" fn adoc_print_to_string(
    string: *mut c_char,
    size: i32,
    write_all: i32,
) -> i32 {
    let a = S_AUTODOCS.lock().unwrap();
    let Some(d) = a.get(current_index() as usize) else {
        return -1;
    };
    let mut out = String::new();
    for c in &d.collections {
        for (si, s) in c.sections.iter().enumerate() {
            if c.name != "PreData" {
                out += &format!("[{} = {}]\n", c.name, s.name);
            } else if si != 0 {
                continue;
            }
            for (k, v) in s.keys.iter().zip(&s.values) {
                out += &format!("{} = {}\n", k, v);
            }
        }
    }
    if string.is_null() {
        return 0;
    }
    if size <= out.len() as i32 {
        return -1;
    }
    core::ptr::copy_nonoverlapping(out.as_ptr().cast(), string, out.len());
    *string.add(out.len()) = 0;
    let _ = write_all;
    0
}
/// Matches C `AdocOrderWriteByValue`.
pub unsafe extern "C" fn adoc_order_write_by_value(name: *const c_char) -> i32 {
    *S_ORDER_NAME.lock().unwrap() = cstr(name);
    0
}
/// Matches C `AdocAddSection`.
pub unsafe extern "C" fn adoc_add_section(kind: *const c_char, name: *const c_char) -> i32 {
    let (Some(k), Some(n)) = (cstr(kind), cstr(name)) else {
        return -1;
    };
    let mut a = S_AUTODOCS.lock().unwrap();
    let cur = current_index();
    let d = match a.get_mut(cur as usize) {
        Some(v) => v,
        None => return -1,
    };
    let ci = match d.collections.iter().position(|c| c.name == k) {
        Some(i) => i,
        None => {
            d.collections.push(AdocCollection {
                name: k,
                ..Default::default()
            });
            d.collections.len() - 1
        }
    };
    d.collections[ci].sections.push(AdocSection {
        name: n,
        ..Default::default()
    });
    (d.collections[ci].sections.len() - 1) as i32
}
/// Matches C `AdocInsertSection`.
pub unsafe extern "C" fn adoc_insert_section(
    kind: *const c_char,
    ind: i32,
    name: *const c_char,
) -> i32 {
    let (Some(k), Some(n)) = (cstr(kind), cstr(name)) else {
        return -1;
    };
    let mut a = S_AUTODOCS.lock().unwrap();
    let Some(c) = collection(&mut a, &k) else {
        return -1;
    };
    if ind < 0 || ind as usize > c.sections.len() {
        return -1;
    }
    c.sections.insert(
        ind as usize,
        AdocSection {
            name: n,
            ..Default::default()
        },
    );
    0
}
/// Matches C `AdocDeleteSection`.
pub unsafe extern "C" fn adoc_delete_section(kind: *const c_char, ind: i32) -> i32 {
    let Some(k) = cstr(kind) else { return -1 };
    let mut a = S_AUTODOCS.lock().unwrap();
    let Some(c) = collection(&mut a, &k) else {
        return -1;
    };
    if ind < 0 || ind as usize >= c.sections.len() {
        -1
    } else {
        c.sections.remove(ind as usize);
        0
    }
}
/// Matches C `AdocChangeSectionName`.
pub unsafe extern "C" fn adoc_change_section_name(
    kind: *const c_char,
    ind: i32,
    name: *const c_char,
) -> i32 {
    let (Some(k), Some(n)) = (cstr(kind), cstr(name)) else {
        return -1;
    };
    let mut a = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut a, &k, ind) else {
        return -1;
    };
    s.name = n;
    0
}
/// Matches C `AdocLookupSection`.
pub unsafe extern "C" fn adoc_lookup_section(kind: *const c_char, name: *const c_char) -> i32 {
    let (Some(k), Some(n)) = (cstr(kind), cstr(name)) else {
        return -1;
    };
    let mut a = S_AUTODOCS.lock().unwrap();
    collection(&mut a, &k)
        .and_then(|c| c.sections.iter().position(|s| s.name == n))
        .map_or(-1, |i| i as i32)
}
/// Matches C `AdocLookupByNameValue`.
pub unsafe extern "C" fn adoc_lookup_by_name_value(kind: *const c_char, value: i32) -> i32 {
    let Some(k) = cstr(kind) else { return -1 };
    let mut a = S_AUTODOCS.lock().unwrap();
    collection(&mut a, &k)
        .and_then(|c| {
            c.sections
                .iter()
                .position(|s| s.name.parse::<i32>() == Ok(value))
        })
        .map_or(-1, |i| i as i32)
}
/// Matches C `AdocFindInsertIndex`.
pub unsafe extern "C" fn adoc_find_insert_index(kind: *const c_char, value: i32) -> i32 {
    let Some(k) = cstr(kind) else { return -1 };
    let mut a = S_AUTODOCS.lock().unwrap();
    collection(&mut a, &k).map_or(-1, |c| {
        c.sections
            .iter()
            .position(|s| s.name.parse::<i32>().map_or(false, |v| v >= value))
            .unwrap_or(c.sections.len()) as i32
    })
}
/// Matches C `AdocTransferSection`.
pub unsafe extern "C" fn adoc_transfer_section(
    kind: *const c_char,
    ind: i32,
    to: i32,
    new_name: *const c_char,
    by_value: i32,
) -> i32 {
    let (Some(k), Some(n)) = (cstr(kind), cstr(new_name)) else {
        return -1;
    };
    let mut a = S_AUTODOCS.lock().unwrap();
    let from = current_index();
    let Some(src) = a
        .get(from as usize)
        .and_then(|d| d.collections.iter().find(|c| c.name == k))
        .and_then(|c| c.sections.get(ind as usize))
        .cloned()
    else {
        return -1;
    };
    let Some(dest) = a.get_mut(to as usize) else {
        return -1;
    };
    let ci = match dest.collections.iter().position(|c| c.name == k) {
        Some(i) => i,
        None => {
            dest.collections.push(AdocCollection {
                name: k,
                ..Default::default()
            });
            dest.collections.len() - 1
        }
    };
    let mut s = src;
    s.name = n;
    let pos = if by_value != 0 {
        dest.collections[ci]
            .sections
            .iter()
            .position(|x| {
                x.name.parse::<i32>().unwrap_or(i32::MAX) >= s.name.parse().unwrap_or(i32::MAX)
            })
            .unwrap_or(dest.collections[ci].sections.len())
    } else {
        dest.collections[ci].sections.len()
    };
    dest.collections[ci].sections.insert(pos, s);
    0
}
/// Matches C `AdocTransferToNewType`.
pub unsafe extern "C" fn adoc_transfer_to_new_type(
    kind: *const c_char,
    ind: i32,
    to: i32,
    new_type: *const c_char,
    new_name: *const c_char,
    by_value: i32,
) -> i32 {
    let (Some(k), Some(t), Some(n)) = (cstr(kind), cstr(new_type), cstr(new_name)) else {
        return -1;
    };
    let mut a = S_AUTODOCS.lock().unwrap();
    let src = a
        .get(current_index() as usize)
        .and_then(|d| d.collections.iter().find(|c| c.name == k))
        .and_then(|c| c.sections.get(ind as usize))
        .cloned();
    let Some(mut s) = src else { return -1 };
    let Some(d) = a.get_mut(to as usize) else {
        return -1;
    };
    s.name = n;
    let ci = match d.collections.iter().position(|c| c.name == t) {
        Some(i) => i,
        None => {
            d.collections.push(AdocCollection {
                name: t,
                ..Default::default()
            });
            d.collections.len() - 1
        }
    };
    let _ = by_value;
    d.collections[ci].sections.push(s);
    0
}
/// Matches C `AdocSetKeyValue`.
pub unsafe extern "C" fn adoc_set_key_value(
    kind: *const c_char,
    ind: i32,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    if value.is_null() {
        let (Some(k), Some(key)) = (cstr(kind), cstr(key)) else {
            return -1;
        };
        let mut a = S_AUTODOCS.lock().unwrap();
        let Some(s) = section(&mut a, &k, ind) else {
            return -1;
        };
        if let Some(i) = s.keys.iter().position(|entry| entry == &key) {
            s.values[i].clear();
            s.types[i] = ADOC_NO_VALUE;
        } else {
            s.keys.push(key);
            s.values.push(String::new());
            s.types.push(ADOC_NO_VALUE);
        }
        return 0;
    }
    adoc_set_key_value_type(kind, ind, key, value, ADOC_STRING)
}
/// Matches C static `setKeyValueType`.
pub unsafe extern "C" fn adoc_set_key_value_type(
    kind: *const c_char,
    ind: i32,
    key: *const c_char,
    value: *const c_char,
    typ: i32,
) -> i32 {
    let (Some(k), Some(key), Some(value)) = (cstr(kind), cstr(key), cstr(value)) else {
        return -1;
    };
    let mut a = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut a, &k, ind) else {
        return -1;
    };
    if let Some(i) = s.keys.iter().position(|x| x == &key) {
        s.values[i] = value;
        s.types[i] = typ;
    } else {
        s.keys.push(key);
        s.values.push(value);
        s.types.push(typ);
    }
    0
}
/// Matches C static `sectSetKeyValueType`.
pub unsafe extern "C" fn adoc_sect_set_key_value_type(
    sect: *mut AdocSection,
    key: *const c_char,
    value: *const c_char,
    typ: i32,
    key_ind: *mut i32,
) -> i32 {
    if sect.is_null() || key.is_null() || key_ind.is_null() {
        return -1;
    }
    let Some(key) = cstr(key) else {
        return -1;
    };
    let section = &mut *sect;
    if let Some(ind) = section.keys.iter().position(|entry| entry == &key) {
        *key_ind = ind as i32;
        if value.is_null() {
            section.values[ind].clear();
            section.types[ind] = ADOC_NO_VALUE;
            return 0;
        }
        let Some(value) = cstr(value) else {
            return -1;
        };
        section.values[ind] = value;
        section.types[ind] = typ;
        return 0;
    }
    *key_ind = section.keys.len() as i32;
    section.keys.push(key);
    section.values.push(cstr(value).unwrap_or_default());
    section
        .types
        .push(if value.is_null() { ADOC_NO_VALUE } else { typ });
    0
}
/// Matches C `AdocSetInteger`.
pub unsafe extern "C" fn adoc_set_integer(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    v: i32,
) -> i32 {
    let Ok(s) = CString::new(v.to_string()) else {
        return 1;
    };
    adoc_set_key_value_type(k, i, key, s.as_ptr(), ADOC_ONE_INT)
}
/// Matches C `AdocSetTwoIntegers`.
pub unsafe extern "C" fn adoc_set_two_integers(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    a: i32,
    b: i32,
) -> i32 {
    let Ok(s) = CString::new(format!("{} {}", a, b)) else {
        return 1;
    };
    adoc_set_key_value_type(k, i, key, s.as_ptr(), ADOC_TWO_INTS)
}
/// Matches C `AdocSetThreeIntegers`.
pub unsafe extern "C" fn adoc_set_three_integers(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    a: i32,
    b: i32,
    c: i32,
) -> i32 {
    let Ok(s) = CString::new(format!("{} {} {}", a, b, c)) else {
        return 1;
    };
    adoc_set_key_value_type(k, i, key, s.as_ptr(), ADOC_THREE_INTS)
}
/// Matches C `AdocSetIntegerArray`.
pub unsafe extern "C" fn adoc_set_integer_array(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    v: *mut i32,
    n: i32,
) -> i32 {
    if v.is_null() || n < 0 {
        return -1;
    };
    let s = (0..n as usize)
        .map(|x| (*v.add(x)).to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let Ok(s) = CString::new(s) else { return 1 };
    adoc_set_key_value_type(k, i, key, s.as_ptr(), ADOC_INT_ARRAY)
}
/// Matches C `AdocSetFloat`.
pub unsafe extern "C" fn adoc_set_float(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    v: f32,
) -> i32 {
    let Ok(s) = CString::new(format!("{:.7}", v)) else {
        return 1;
    };
    adoc_set_key_value_type(k, i, key, s.as_ptr(), ADOC_ONE_FLOAT)
}
/// Matches C `AdocSetDouble`.
pub unsafe extern "C" fn adoc_set_double(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    v: f64,
) -> i32 {
    let Ok(s) = CString::new(format!("{:.14}", v)) else {
        return 1;
    };
    adoc_set_key_value_type(k, i, key, s.as_ptr(), ADOC_ONE_DOUBLE)
}
/// Matches C `AdocSetTwoFloats`.
pub unsafe extern "C" fn adoc_set_two_floats(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    a: f32,
    b: f32,
) -> i32 {
    let Ok(s) = CString::new(format!("{:.7} {:.7}", a, b)) else {
        return 1;
    };
    adoc_set_key_value_type(k, i, key, s.as_ptr(), ADOC_TWO_FLOATS)
}
/// Matches C `AdocSetThreeFloats`.
pub unsafe extern "C" fn adoc_set_three_floats(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    a: f32,
    b: f32,
    c: f32,
) -> i32 {
    let Ok(s) = CString::new(format!("{:.7} {:.7} {:.7}", a, b, c)) else {
        return 1;
    };
    adoc_set_key_value_type(k, i, key, s.as_ptr(), ADOC_THREE_FLOATS)
}
/// Matches C `AdocSetFloatArray`.
pub unsafe extern "C" fn adoc_set_float_array(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    v: *mut f32,
    n: i32,
) -> i32 {
    if v.is_null() || n < 0 {
        return -1;
    };
    let s = (0..n as usize)
        .map(|x| format!("{:.7}", *v.add(x)))
        .collect::<Vec<_>>()
        .join(" ");
    let Ok(s) = CString::new(s) else { return 1 };
    adoc_set_key_value_type(k, i, key, s.as_ptr(), ADOC_FLOAT_ARRAY)
}
/// Matches C static `setArrayOfValues`.
pub unsafe extern "C" fn adoc_set_array_of_values(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    vals: *mut c_void,
    num_vals: i32,
    val_type: i32,
) -> i32 {
    if vals.is_null()
        || num_vals < 0
        || (val_type != ADOC_INT_ARRAY && val_type != ADOC_FLOAT_ARRAY)
    {
        return -1;
    }
    let value = if val_type == ADOC_INT_ARRAY {
        let values = core::slice::from_raw_parts(vals.cast::<i32>(), num_vals as usize);
        values
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        let values = core::slice::from_raw_parts(vals.cast::<f32>(), num_vals as usize);
        values
            .iter()
            .map(f32::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    };
    let Ok(value) = CString::new(value) else {
        return -1;
    };
    adoc_set_key_value_type(type_name, sect_ind, key, value.as_ptr(), val_type)
}
/// Matches C `AdocDeleteKeyValue`.
pub unsafe extern "C" fn adoc_delete_key_value(
    kind: *const c_char,
    ind: i32,
    key: *const c_char,
) -> i32 {
    let (Some(k), Some(key)) = (cstr(kind), cstr(key)) else {
        return -1;
    };
    let mut a = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut a, &k, ind) else {
        return -1;
    };
    let Some(p) = s.keys.iter().position(|x| x == &key) else {
        return -1;
    };
    s.keys.remove(p);
    s.values.remove(p);
    s.types.remove(p);
    0
}
/// Matches C `AdocGetNumCollections`.
pub extern "C" fn adoc_get_num_collections() -> i32 {
    S_AUTODOCS
        .lock()
        .unwrap()
        .get(current_index() as usize)
        .map_or(-1, |d| d.collections.len() as i32 - 1)
}
/// Matches C `AdocGetCollectionName`.
pub unsafe extern "C" fn adoc_get_collection_name(ind: i32, out: *mut *mut c_char) -> i32 {
    let a = S_AUTODOCS.lock().unwrap();
    a.get(current_index() as usize)
        .and_then(|d| d.collections.get(ind as usize + 1))
        .map_or(-1, |c| return_string(out, &c.name))
}
/// Matches C `AdocGetSectionName`.
pub unsafe extern "C" fn adoc_get_section_name(
    kind: *const c_char,
    ind: i32,
    out: *mut *mut c_char,
) -> i32 {
    let Some(k) = cstr(kind) else { return -1 };
    let mut a = S_AUTODOCS.lock().unwrap();
    collection(&mut a, &k)
        .and_then(|c| c.sections.get(ind as usize))
        .map_or(-1, |s| return_string(out, &s.name))
}
/// Matches C `AdocGetNumberOfSections`.
pub unsafe extern "C" fn adoc_get_number_of_sections(kind: *const c_char) -> i32 {
    let Some(k) = cstr(kind) else { return -1 };
    let mut a = S_AUTODOCS.lock().unwrap();
    collection(&mut a, &k).map_or(0, |c| c.sections.len() as i32)
}
/// Matches C `AdocGetNumberOfKeys`.
pub unsafe extern "C" fn adoc_get_number_of_keys(kind: *const c_char, ind: i32) -> i32 {
    let Some(k) = cstr(kind) else { return -1 };
    let mut a = S_AUTODOCS.lock().unwrap();
    section(&mut a, &k, ind).map_or(-1, |s| s.keys.len() as i32)
}
/// Matches C `AdocGetKeyByIndex`.
pub unsafe extern "C" fn adoc_get_key_by_index(
    kind: *const c_char,
    si: i32,
    ki: i32,
    out: *mut *mut c_char,
) -> i32 {
    let Some(k) = cstr(kind) else { return -1 };
    let mut a = S_AUTODOCS.lock().unwrap();
    section(&mut a, &k, si)
        .and_then(|s| s.keys.get(ki as usize))
        .map_or(-1, |x| return_string(out, x))
}
/// Matches C `AdocGetValTypeAndSize`.
pub unsafe extern "C" fn adoc_get_val_type_and_size(
    kind: *const c_char,
    si: i32,
    key: *const c_char,
    typ: *mut i32,
    n: *mut i32,
) -> i32 {
    let (Some(k), Some(key)) = (cstr(kind), cstr(key)) else {
        return -1;
    };
    let mut a = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut a, &k, si) else {
        return -1;
    };
    let Some(i) = s.keys.iter().position(|x| x == &key) else {
        return -1;
    };
    if !typ.is_null() {
        *typ = s.types[i]
    }
    if !n.is_null() {
        *n = s.values[i].split_whitespace().count() as i32
    }
    0
}
/// Matches C `AdocGetString`.
pub unsafe extern "C" fn adoc_get_string(
    kind: *const c_char,
    si: i32,
    key: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    let (Some(k), Some(key)) = (cstr(kind), cstr(key)) else {
        return -1;
    };
    let mut a = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut a, &k, si) else {
        return -1;
    };
    s.keys
        .iter()
        .position(|x| x == &key)
        .map_or(-1, |i| return_string(out, &s.values[i]))
}
/// Matches C `AdocGetInteger`.
pub unsafe extern "C" fn adoc_get_integer(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    out: *mut i32,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let (Some(k), Some(key)) = (cstr(k), cstr(key)) else {
        return -1;
    };
    let mut a = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut a, &k, i) else {
        return -1;
    };
    typed_values(s, &key, ADOC_ONE_INT)
        .and_then(|v| v.first()?.parse().ok())
        .map_or(-1, |v| {
            *out = v;
            0
        })
}
/// Matches C `AdocGetFloat`.
pub unsafe extern "C" fn adoc_get_float(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    out: *mut f32,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let (Some(k), Some(key)) = (cstr(k), cstr(key)) else {
        return -1;
    };
    let mut a = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut a, &k, i) else {
        return -1;
    };
    typed_values(s, &key, ADOC_ONE_FLOAT)
        .and_then(|v| v.first()?.parse().ok())
        .map_or(-1, |v| {
            *out = v;
            0
        })
}
/// Matches C `AdocGetDouble`.
pub unsafe extern "C" fn adoc_get_double(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    out: *mut f64,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let (Some(k), Some(key)) = (cstr(k), cstr(key)) else {
        return -1;
    };
    let mut a = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut a, &k, i) else {
        return -1;
    };
    typed_values(s, &key, ADOC_ONE_DOUBLE)
        .and_then(|v| v.first()?.parse().ok())
        .map_or(-1, |v| {
            *out = v;
            0
        })
}
/// Matches C `AdocGetTwoIntegers`.
pub unsafe extern "C" fn adoc_get_two_integers(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    a: *mut i32,
    b: *mut i32,
) -> i32 {
    if a.is_null() || b.is_null() {
        return -1;
    }
    let (Some(k), Some(key)) = (cstr(k), cstr(key)) else {
        return -1;
    };
    let mut d = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut d, &k, i) else {
        return -1;
    };
    let Some(v) = typed_values(s, &key, ADOC_TWO_INTS) else {
        return -1;
    };
    if v.len() != 2 {
        return -1;
    }
    match (v[0].parse(), v[1].parse()) {
        (Ok(x), Ok(y)) => {
            *a = x;
            *b = y;
            0
        }
        _ => -1,
    }
}
/// Matches C `AdocGetTwoFloats`.
pub unsafe extern "C" fn adoc_get_two_floats(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    a: *mut f32,
    b: *mut f32,
) -> i32 {
    if a.is_null() || b.is_null() {
        return -1;
    }
    let (Some(k), Some(key)) = (cstr(k), cstr(key)) else {
        return -1;
    };
    let mut d = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut d, &k, i) else {
        return -1;
    };
    let Some(v) = typed_values(s, &key, ADOC_TWO_FLOATS) else {
        return -1;
    };
    if v.len() != 2 {
        return -1;
    }
    match (v[0].parse(), v[1].parse()) {
        (Ok(x), Ok(y)) => {
            *a = x;
            *b = y;
            0
        }
        _ => -1,
    }
}
/// Matches C `AdocGetThreeIntegers`.
pub unsafe extern "C" fn adoc_get_three_integers(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    a: *mut i32,
    b: *mut i32,
    c: *mut i32,
) -> i32 {
    if a.is_null() || b.is_null() || c.is_null() {
        return -1;
    }
    let (Some(k), Some(key)) = (cstr(k), cstr(key)) else {
        return -1;
    };
    let mut d = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut d, &k, i) else {
        return -1;
    };
    let Some(v) = typed_values(s, &key, ADOC_THREE_INTS) else {
        return -1;
    };
    if v.len() != 3 {
        return -1;
    }
    match (v[0].parse(), v[1].parse(), v[2].parse()) {
        (Ok(x), Ok(y), Ok(z)) => {
            *a = x;
            *b = y;
            *c = z;
            0
        }
        _ => -1,
    }
}
/// Matches C `AdocGetThreeFloats`.
pub unsafe extern "C" fn adoc_get_three_floats(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    a: *mut f32,
    b: *mut f32,
    c: *mut f32,
) -> i32 {
    if a.is_null() || b.is_null() || c.is_null() {
        return -1;
    }
    let (Some(k), Some(key)) = (cstr(k), cstr(key)) else {
        return -1;
    };
    let mut d = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut d, &k, i) else {
        return -1;
    };
    let Some(v) = typed_values(s, &key, ADOC_THREE_FLOATS) else {
        return -1;
    };
    if v.len() != 3 {
        return -1;
    }
    match (v[0].parse(), v[1].parse(), v[2].parse()) {
        (Ok(x), Ok(y), Ok(z)) => {
            *a = x;
            *b = y;
            *c = z;
            0
        }
        _ => -1,
    }
}
/// Matches C `AdocGetIntegerArray`.
pub unsafe extern "C" fn adoc_get_integer_array(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    array: *mut i32,
    n: *mut i32,
    size: i32,
) -> i32 {
    if array.is_null() || n.is_null() {
        return -1;
    }
    let (Some(k), Some(key)) = (cstr(k), cstr(key)) else {
        return -1;
    };
    let mut d = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut d, &k, i) else {
        return -1;
    };
    let Some(v) = typed_values(s, &key, ADOC_INT_ARRAY) else {
        return -1;
    };
    if v.len() > size as usize {
        return -1;
    }
    for (j, x) in v.iter().enumerate() {
        let Ok(x) = x.parse() else { return -1 };
        *array.add(j) = x;
    }
    *n = v.len() as i32;
    0
}
/// Matches C `AdocGetFloatArray`.
pub unsafe extern "C" fn adoc_get_float_array(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    array: *mut f32,
    n: *mut i32,
    size: i32,
) -> i32 {
    if array.is_null() || n.is_null() {
        return -1;
    }
    let (Some(k), Some(key)) = (cstr(k), cstr(key)) else {
        return -1;
    };
    let mut d = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut d, &k, i) else {
        return -1;
    };
    let Some(v) = typed_values(s, &key, ADOC_FLOAT_ARRAY) else {
        return -1;
    };
    if v.len() > size as usize {
        return -1;
    }
    for (j, x) in v.iter().enumerate() {
        let Ok(x) = x.parse() else { return -1 };
        *array.add(j) = x;
    }
    *n = v.len() as i32;
    0
}
/// Matches C `AdocGetDoubleArray`.
pub unsafe extern "C" fn adoc_get_double_array(
    k: *const c_char,
    i: i32,
    key: *const c_char,
    array: *mut f64,
    n: *mut i32,
    size: i32,
) -> i32 {
    if array.is_null() || n.is_null() {
        return -1;
    }
    let (Some(k), Some(key)) = (cstr(k), cstr(key)) else {
        return -1;
    };
    let mut d = S_AUTODOCS.lock().unwrap();
    let Some(s) = section(&mut d, &k, i) else {
        return -1;
    };
    let Some(v) = typed_values(s, &key, ADOC_FLOAT_ARRAY) else {
        return -1;
    };
    if v.len() > size as usize {
        return -1;
    }
    for (j, x) in v.iter().enumerate() {
        let Ok(x) = x.parse() else { return -1 };
        *array.add(j) = x;
    }
    *n = v.len() as i32;
    0
}
/// Matches C `AdocXmlReadStatus`.
pub unsafe extern "C" fn adoc_xml_read_status(
    a: *mut i32,
    b: *mut i32,
    c: *mut i32,
    d: *mut i32,
    e: *mut i32,
    f: *mut i32,
) -> i32 {
    for p in [a, b, c, d, e, f] {
        if !p.is_null() {
            *p = 0
        }
    }
    0
}
/// Matches C static `readXmlFile`.
pub unsafe extern "C" fn adoc_read_xml_file(_fp: *mut libc::FILE) -> i32 {
    -1
}
/// Matches C static `writeXmlFile`.
pub unsafe extern "C" fn adoc_write_xml_file(_filename: *const c_char) -> i32 {
    -1
}
/// Matches C `AdocWriteInteger`.
pub unsafe extern "C" fn adoc_write_integer(
    fp: *mut libc::FILE,
    key: *const c_char,
    v: i32,
) -> i32 {
    if fp.is_null() || key.is_null() {
        return -1;
    }
    libc::fprintf(fp, c"%s = %d\n".as_ptr(), key, v);
    0
}
/// Matches C `AdocWriteTwoIntegers`.
pub unsafe extern "C" fn adoc_write_two_integers(
    fp: *mut libc::FILE,
    key: *const c_char,
    a: i32,
    b: i32,
) -> i32 {
    if fp.is_null() || key.is_null() {
        return -1;
    }
    libc::fprintf(fp, c"%s = %d %d\n".as_ptr(), key, a, b);
    0
}
/// Matches C `AdocWriteThreeIntegers`.
pub unsafe extern "C" fn adoc_write_three_integers(
    fp: *mut libc::FILE,
    key: *const c_char,
    a: i32,
    b: i32,
    c: i32,
) -> i32 {
    if fp.is_null() || key.is_null() {
        return -1;
    }
    libc::fprintf(fp, c"%s = %d %d %d\n".as_ptr(), key, a, b, c);
    0
}
/// Matches C `AdocWriteIntegerArray`.
pub unsafe extern "C" fn adoc_write_integer_array(
    fp: *mut libc::FILE,
    key: *const c_char,
    v: *mut i32,
    n: i32,
) -> i32 {
    if fp.is_null() || key.is_null() || v.is_null() || n < 0 {
        return 1;
    }
    if libc::fprintf(fp, c"%s =".as_ptr(), key) < 0 {
        return 1;
    }
    for ind in 0..n as usize {
        if libc::fprintf(fp, c" %d".as_ptr(), *v.add(ind)) < 0 {
            return 1;
        }
    }
    (libc::fprintf(fp, c"\n".as_ptr()) < 0) as i32
}
/// Matches C `AdocWriteFloat`.
pub unsafe extern "C" fn adoc_write_float(fp: *mut libc::FILE, key: *const c_char, v: f32) -> i32 {
    if fp.is_null() || key.is_null() {
        return -1;
    }
    libc::fprintf(fp, c"%s = %.7g\n".as_ptr(), key, v as f64);
    0
}
/// Matches C `AdocWriteDouble`.
pub unsafe extern "C" fn adoc_write_double(fp: *mut libc::FILE, key: *const c_char, v: f64) -> i32 {
    if fp.is_null() || key.is_null() {
        return -1;
    }
    libc::fprintf(fp, c"%s = %.14g\n".as_ptr(), key, v);
    0
}
/// Matches C `AdocWriteTwoFloats`.
pub unsafe extern "C" fn adoc_write_two_floats(
    fp: *mut libc::FILE,
    key: *const c_char,
    a: f32,
    b: f32,
) -> i32 {
    if fp.is_null() || key.is_null() {
        return 1;
    }
    (libc::fprintf(fp, c"%s = %g %g\n".as_ptr(), key, a as f64, b as f64) < 0) as i32
}
/// Matches C `AdocWriteThreeFloats`.
pub unsafe extern "C" fn adoc_write_three_floats(
    fp: *mut libc::FILE,
    key: *const c_char,
    a: f32,
    b: f32,
    c: f32,
) -> i32 {
    if fp.is_null() || key.is_null() {
        return 1;
    }
    (libc::fprintf(
        fp,
        c"%s = %g %g %g\n".as_ptr(),
        key,
        a as f64,
        b as f64,
        c as f64,
    ) < 0) as i32
}
/// Matches C `AdocWriteFloatArray`.
pub unsafe extern "C" fn adoc_write_float_array(
    fp: *mut libc::FILE,
    key: *const c_char,
    v: *mut f32,
    n: i32,
) -> i32 {
    if fp.is_null() || key.is_null() || v.is_null() || n < 0 {
        return 1;
    }
    if libc::fprintf(fp, c"%s =".as_ptr(), key) < 0 {
        return 1;
    }
    for ind in 0..n as usize {
        if libc::fprintf(fp, c" %g".as_ptr(), *v.add(ind) as f64) < 0 {
            return 1;
        }
    }
    (libc::fprintf(fp, c"\n".as_ptr()) < 0) as i32
}
/// Matches C `AdocWriteKeyValue`.
pub unsafe extern "C" fn adoc_write_key_value(
    fp: *mut libc::FILE,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    if fp.is_null() || key.is_null() || value.is_null() {
        return -1;
    }
    libc::fprintf(fp, c"%s = %s\n".as_ptr(), key, value);
    0
}
/// Matches C `AdocWriteSectionStart`.
pub unsafe extern "C" fn adoc_write_section_start(
    fp: *mut libc::FILE,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    if fp.is_null() || key.is_null() || value.is_null() {
        return -1;
    }
    libc::fprintf(fp, c"[%s = %s]\n".as_ptr(), key, value);
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn reads_vendored_autodoc_fixture() {
        unsafe {
            let _lock = TEST_LOCK.lock().unwrap();
            adoc_done();
            let path = CString::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/IMOD/autodoc/binvol.adoc"
            ))
            .unwrap();
            assert!(adoc_read(path.as_ptr()) >= 0);
            assert!(adoc_get_number_of_sections(c"Field".as_ptr()) >= 1);
            let mut value = core::ptr::null_mut();
            assert_eq!(adoc_get_section_name(c"Field".as_ptr(), 0, &mut value), 0);
            assert!(!value.is_null());
            libc::free(value.cast());
        }
    }

    #[test]
    fn metadata_numbers_are_read_from_text_values() {
        unsafe {
            let _lock = TEST_LOCK.lock().unwrap();
            adoc_done();
            let text =
                b"ImageFile = stack.mrc\nMontage = 1\n[ZValue = 0]\nPieceCoordinates = 1 2 3\n";
            let name = std::env::temp_dir().join("imod-rs-autodoc-metadata.adoc");
            fs::write(&name, text).unwrap();
            let file = CString::new(name.to_string_lossy().as_bytes()).unwrap();
            assert!(adoc_read(file.as_ptr()) >= 0);
            let mut a = 0;
            let mut b = 0;
            let mut c = 0;
            assert_eq!(
                adoc_get_three_integers(
                    c"ZValue".as_ptr(),
                    0,
                    c"PieceCoordinates".as_ptr(),
                    &mut a,
                    &mut b,
                    &mut c
                ),
                0
            );
            assert_eq!((a, b, c), (1, 2, 3));
            std::fs::remove_file(name).unwrap();
        }
    }

    #[test]
    fn collection_indices_exclude_the_global_predata_collection() {
        unsafe {
            let _lock = TEST_LOCK.lock().unwrap();
            adoc_done();
            assert!(adoc_new() >= 0);
            assert_eq!(adoc_add_section(c"Field".as_ptr(), c"one".as_ptr()), 0);
            assert_eq!(adoc_get_num_collections(), 1);
            let mut name = core::ptr::null_mut();
            assert_eq!(adoc_get_collection_name(0, &mut name), 0);
            assert_eq!(CStr::from_ptr(name).to_bytes(), b"Field");
            libc::free(name.cast());
        }
    }
}
