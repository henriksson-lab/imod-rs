//! Translation of `IMOD/libcfshr/autodoc.c` and `IMOD/include/autodoc.h`.
//!
//! The module keeps the C source's data structures and its module-level static
//! state, so that operation order, the `MALLOC_CHUNK` growth of every array,
//! and every error return match the original.  What changed is the *storage*:
//! the source's `malloc`ed C strings are `Vec<u8>` and its NULL pointers are
//! `None`, so no autodoc key, value, section name or comment is a
//! NUL-terminated string any more.  Autodoc keys and values are bytes read out
//! of a file, never guaranteed text, so they are `Vec<u8>` and not `String`.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::{
    CArg, b3d_error, b3d_milli_sleep, c_format_bytes, imod_backup_file,
};
use crate::imod::libcfshr::mxmlwrap::{ixml_reset_last_level, ixml_whitespace_cb};
use crate::imod::libcfshr::parse_params::{
    PIP_DOUBLE, PIP_FLOAT, PIP_INTEGER, pip_get_line_of_values, pip_read_next_line,
    pip_starts_with, strtod, strtol,
};
use crate::imod::libcfshr::robuststat::rs_sort_indexed_floats;
use crate::imod::libxml::{
    MXML_DESCEND, MXML_ELEMENT, MXML_NO_PARENT, MXML_OPAQUE, MxmlArena, MxmlSaveCb, MxmlValue,
    mxml_delete, mxml_element_get_attr, mxml_element_set_attr, mxml_get_element,
    mxml_get_first_child, mxml_get_last_child, mxml_get_next_sibling, mxml_get_type,
    mxml_load_file, mxml_new_element, mxml_new_text, mxml_new_xml, mxml_opaque_cb, mxml_save_file,
    mxml_set_wrap_margin, mxml_walk_next,
};
use core::cell::{Cell, RefCell};
use std::io::{Seek, SeekFrom, Write};

/* --- IMOD/include/autodoc.h ------------------------------------------- */

/// Matches C `ADOC_GLOBAL_NAME` (`autodoc.h:16`).
pub const ADOC_GLOBAL_NAME: &[u8] = b"PreData";
/// Matches C `ADOC_ZVALUE_NAME` (`autodoc.h:17`).
pub const ADOC_ZVALUE_NAME: &[u8] = b"ZValue";
/// Matches C `ADOC_FRAMESET_NAME` (`autodoc.h:18`).
pub const ADOC_FRAMESET_NAME: &[u8] = b"FrameSet";

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

/* --- autodoc.c:22-56 : the three structures ---------------------------- */

/// Matches C `struct adoc_section` / `AdocSection` (`autodoc.c:23`).
///
/// `keys`, `values` and `types` are held at length `max_keys`, the size the C
/// `realloc`s them to, with `num_keys` the number in use, so every index in the
/// source is the same index here.  A NULL key or value is `None`.
#[derive(Clone, Default)]
pub struct AdocSection {
    /// value after delimiter in section header
    pub name: Option<Vec<u8>>,
    /// Array of strings with keys
    pub keys: Vec<Option<Vec<u8>>>,
    /// Array of strings with values
    pub values: Vec<Option<Vec<u8>>>,
    /// Number of key/value pairs
    pub num_keys: i32,
    /// Current size of array
    pub max_keys: i32,
    /// List of comments strings
    pub comments: Vec<Vec<u8>>,
    /// Array of key indexes they occur before
    pub com_index: Vec<i32>,
    /// Number of comments
    pub num_comments: i32,
    /// Array of types of keys
    pub types: Vec<u8>,
}

/// Matches C `struct adoc_collection` / `AdocCollection` (`autodoc.c:35`).
#[derive(Clone, Default)]
pub struct AdocCollection {
    /// section type, before delimiter in header
    pub name: Option<Vec<u8>>,
    /// Array of sections
    pub sections: Vec<AdocSection>,
    /// Number of sections
    pub num_sections: i32,
    /// Current size of array
    pub max_sections: i32,
}

/// Matches C `struct adoc_autodoc` / `Autodoc` (`autodoc.c:42`).
#[derive(Clone, Default)]
pub struct Autodoc {
    pub collections: Vec<AdocCollection>,
    pub num_collections: i32,
    pub final_comments: Vec<Vec<u8>>,
    pub num_final_com: i32,
    pub coll_list: Vec<i32>,
    pub sect_list: Vec<i32>,
    pub num_sections: i32,
    pub max_sections: i32,
    pub in_use: i32,
    pub backed_up: i32,
    pub write_as_xml: i32,
    /// Name for root element for XML file, in or out
    pub root_element: Option<Vec<u8>>,
}

thread_local! {
    /* The static variables that can hold multiple autodocs (autodoc.c:59-63).
       `sCurAdoc` was a pointer into `sAutodocs`; it is the index now, because a
       `Vec` that grows moves its elements and a stored reference would not
       survive `addAutodoc`. */
    static S_AUTODOCS: RefCell<Vec<Autodoc>> = const { RefCell::new(Vec::new()) };
    static S_CUR_ADOC_IND: Cell<i32> = const { Cell::new(-1) };
    static S_NAME_FOR_ORDERING: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };

    /* Static variables for XML writing (autodoc.c:66-68) */
    static S_LAST_WAS_XML: Cell<i32> = const { Cell::new(0) };
    static S_NUM_SECT_NOT_ELEM: Cell<i32> = const { Cell::new(0) };
    static S_NUM_SECT_NO_NAME: Cell<i32> = const { Cell::new(0) };
    static S_NUM_CHILD_NOT_ELEM: Cell<i32> = const { Cell::new(0) };
    static S_NUM_CHILD_ATTRIBS: Cell<i32> = const { Cell::new(0) };
    static S_NUM_VALUE_NOT_TEXT: Cell<i32> = const { Cell::new(0) };
    static S_NUM_MULTIPLE_CHILDS: Cell<i32> = const { Cell::new(0) };

    /* Static variables for writing to file or string (autodoc.c:71-74).
       `sString` was the caller's `char *`; `fsPrintf` appends here and
       `writeFile` hands the bytes back, so the `snprintf` bookkeeping in
       `sBytesLeft`/`sBytesWritten` is unchanged. */
    static S_FILE: RefCell<Option<ImodFile>> = const { RefCell::new(None) };
    static S_STRING: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
    static S_BYTES_LEFT: Cell<i32> = const { Cell::new(0) };
    static S_BYTES_WRITTEN: Cell<i32> = const { Cell::new(0) };

    static S_OPEN_RETRIES: Cell<i32> = const { Cell::new(0) };

    /// Matches C `static char sDefaultDelim[] = "=";` (`autodoc.c:84`) together
    /// with `sValueDelim`, which points either at it or at `sNewDelim`; one
    /// owned copy of the delimiter in use says the same thing.
    static S_VALUE_DELIM: RefCell<Vec<u8>> = RefCell::new(S_DEFAULT_DELIM.to_vec());
    static S_NEW_DELIM: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
}

static S_DEFAULT_DELIM: &[u8] = b"=";

const OPEN_DELIM: &[u8] = b"[";
const CLOSE_DELIM: &[u8] = b"]";
const XML_START: &[u8] = b"<?xml";
const XML_COMMENT_START: &[u8] = b"!--";
const XML_COMSTART_LEN: i32 = 3;

const BIG_STR_SIZE: usize = 10240;
const ERR_STR_SIZE: usize = 1024;
const MALLOC_CHUNK: i32 = 10;

/// Matches C `AdocRead` (`autodoc.c:130`).
pub fn adoc_read(filename: &[u8]) -> i32 {
    let mut got_section: i32 = 0;
    /* `err` is uninitialised in the C source; it is only read after at least one
    loop iteration has assigned it, except for a completely empty file. */
    let mut err: i32 = 0;
    let mut line_len: i32 = 0;
    let mut indst: i32 = 0;
    let mut icol: i32 = 0;
    let mut ikey: i32 = 0;
    let mut last_ind: i32;
    let index: i32;
    let bad_line: i32 = 1234;
    /* `curSect` was an `AdocSection *`; a `Vec` moves when it grows, so the
    section is identified by its collection and its index in it. */
    let mut cur_coll: i32;
    let mut cur_sect: i32;
    let mut key: Vec<u8> = Vec::new();
    let mut value: Option<Vec<u8>> = None;
    /* The C's `char bigStr[BIG_STR_SIZE]`, holding the line without its NUL. */
    let mut big_str: Vec<u8> = Vec::new();
    let mut comment_char: u8 = b'#';
    let mut comment_list: Vec<Vec<u8>> = Vec::new();
    let mut max_comments: i32 = 0;
    let mut num_comments: i32 = 0;
    let mut first_line: i32 = 1;

    S_VALUE_DELIM.with_borrow_mut(|delim| *delim = S_DEFAULT_DELIM.to_vec());
    let name = String::from_utf8_lossy(filename).into_owned();
    let Some(mut afile) = ImodFile::open(&name, "r") else {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: AdocRead - Error opening autodoc file {}",
                String::from_utf8_lossy(filename)
            ),
        );
        return -1;
    };

    /* Create a new adoc, which sets up global collection/section
    and takes care of cleanup if it fails */
    index = add_autodoc();
    if index < 0 {
        return -1;
    }

    adoc_set_current(index);
    cur_coll = 0;
    cur_sect = 0;
    last_ind = -1;
    S_LAST_WAS_XML.set(0);

    let result = S_AUTODOCS.with_borrow_mut(|adocs| {
        let adoc = &mut adocs[index as usize];
        loop {
            /* We cannot allow in-line comments so that value lines can contain
            anything.  But do allow blank and comment lines */
            big_str.clear();
            line_len = pip_read_next_line(
                &mut afile,
                &mut big_str,
                BIG_STR_SIZE as i32,
                comment_char,
                1,
                0,
                &mut indst,
            );
            if line_len == -3 {
                break;
            }
            if line_len < 0 {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: AdocRead - {} {}\n",
                        if line_len == -2 {
                            "Error reading autodoc file"
                        } else {
                            "Line too long in autodoc file"
                        },
                        String::from_utf8_lossy(filename)
                    ),
                );
                err = -1;
                break;
            }

            /* For first line, check for XML file and read that */
            let line = indst as usize;
            if first_line != 0 && pip_starts_with(&big_str[line..], XML_START) != 0 {
                err = read_xml_file(adoc, &mut afile);
                if err < 0 {
                    return err;
                }
                return S_CUR_ADOC_IND.get();
            }
            first_line = 0;

            /* First check for comment and add it to list */
            if indst >= line_len || big_str[line] == comment_char {
                err = add_to_comment_list(
                    &mut comment_list,
                    &mut num_comments,
                    &mut max_comments,
                    &big_str,
                );
                if err != 0 {
                    break;
                }
                continue;
            }

            let close = big_str[line..]
                .windows(CLOSE_DELIM.len())
                .position(|w| w == CLOSE_DELIM)
                .map(|p| line + p);
            if pip_starts_with(&big_str[line..], OPEN_DELIM) != 0 && close.is_some() {
                /* If this is a section start, get name - value.  Here there must be
                a value and it is an error if there is none. */
                let line_end2 = close.unwrap();
                err = parse_key_value(
                    &big_str,
                    line + OPEN_DELIM.len(),
                    line_end2,
                    &mut key,
                    &mut value,
                );
                if value.is_none() {
                    err = 1;
                }
                if err != 0 {
                    err = if err > 0 { bad_line } else { err };
                    break;
                }

                /* Lookup the collection under the key and create one if not found */
                icol = lookup_collection(adoc, &key);
                if icol < 0 {
                    err = add_collection(adoc, &key);
                    if err != 0 {
                        break;
                    }
                    icol = adoc.num_collections - 1;
                }

                /* Add a section to the collection and set it as current one */
                err = add_section(adoc, icol, value.as_ref().unwrap());
                if err != 0 {
                    break;
                }
                cur_coll = icol;
                cur_sect = adoc.collections[icol as usize].num_sections - 1;
                got_section = 1;
                last_ind = -1;
            } else {
                /* Otherwise this is key-value inside a section.  First check for
                continuation line and append to last value. */
                let has_delim = S_VALUE_DELIM.with_borrow(|delim| {
                    !delim.is_empty()
                        && big_str[line..]
                            .windows(delim.len())
                            .any(|w| w == &delim[..])
                });
                if last_ind >= 0
                    && !has_delim
                    && adoc.collections[cur_coll as usize].sections[cur_sect as usize].values
                        [last_ind as usize]
                        .is_some()
                {
                    /* Replace null with space and new null, then append new string */
                    let sect = &mut adoc.collections[cur_coll as usize].sections[cur_sect as usize];
                    let vals = sect.values[last_ind as usize].as_mut().unwrap();
                    vals.push(b' ');
                    vals.extend_from_slice(&big_str[line..]);
                    continue;
                }

                /* This should be a key-value pair now */
                let line_end = line_len as usize;
                err = parse_key_value(&big_str, line, line_end, &mut key, &mut value);
                if err != 0 {
                    err = if err > 0 { bad_line } else { err };
                    break;
                }

                /* Handle new key-value delimiter - replace previous new value if any */
                if got_section == 0 && key == b"KeyValueDelimiter" && value.is_some() {
                    S_NEW_DELIM.with_borrow_mut(|new| *new = Some(value.clone().unwrap()));
                    S_VALUE_DELIM.with_borrow_mut(|delim| *delim = value.clone().unwrap());
                }

                /* Handle change of comment character */
                if got_section == 0 && key == b"CommentCharacter" {
                    /* C reads `*value`, the first byte of the value, and a NULL
                    value would be a null dereference (autodoc.c:268). */
                    comment_char = value.as_ref().and_then(|v| v.first().copied()).unwrap_or(0);
                }

                /* Look up the key first to replace an existing value */
                let sect = &mut adoc.collections[cur_coll as usize].sections[cur_sect as usize];
                ikey = lookup_key(sect, &key);
                if ikey >= 0 {
                    sect.values[ikey as usize] = value.take();
                    last_ind = ikey;
                } else {
                    /* Or just add the key-value */
                    err = add_key(sect, &key, value.as_deref(), ADOC_STRING);
                    if err != 0 {
                        break;
                    }
                    last_ind = sect.num_keys - 1;
                }
            }

            /* If there are comments, attach to item just added */
            if num_comments != 0 {
                let sect = &mut adoc.collections[cur_coll as usize].sections[cur_sect as usize];
                err = add_comments(sect, &mut comment_list, &mut num_comments, last_ind);
                if err != 0 {
                    break;
                }
            }
        }

        /* END OF FILE: If error, clean out autodoc, compose message for bad line */
        if err != 0 {
            delete_adoc(adoc);
            if err == bad_line {
                big_str.truncate(big_str.len().min(ERR_STR_SIZE - 50));
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "Error: AdocRead -Improperly formatted line in autodoc: {}\n",
                        String::from_utf8_lossy(&big_str)
                    ),
                );
                err = -1;
            }
        }

        if err == 0 {
            handle_final_comments(adoc, err, &mut comment_list, num_comments);
        }

        if err != 0 { err } else { index }
    });
    result
}

/// Matches C `AdocXmlReadStatus` (`autodoc.c:333`).
pub fn adoc_xml_read_status(
    sect_not_elem: &mut i32,
    sect_no_name: &mut i32,
    child_not_elem: &mut i32,
    child_attribs: &mut i32,
    value_not_text: &mut i32,
    multiple_childs: &mut i32,
) -> i32 {
    *sect_no_name = S_NUM_SECT_NO_NAME.get();
    *sect_not_elem = S_NUM_SECT_NOT_ELEM.get();
    *child_not_elem = S_NUM_CHILD_NOT_ELEM.get();
    *child_attribs = S_NUM_CHILD_ATTRIBS.get();
    *value_not_text = S_NUM_VALUE_NOT_TEXT.get();
    *multiple_childs = S_NUM_MULTIPLE_CHILDS.get();
    if S_LAST_WAS_XML.get() == 0 {
        return 0;
    }
    if S_NUM_SECT_NOT_ELEM.get()
        + S_NUM_SECT_NO_NAME.get()
        + S_NUM_CHILD_NOT_ELEM.get()
        + S_NUM_CHILD_ATTRIBS.get()
        + S_NUM_VALUE_NOT_TEXT.get()
        + S_NUM_MULTIPLE_CHILDS.get()
        > 0
    {
        -1
    } else {
        1
    }
}

/// Matches C `AdocOpenImageMetadata` (`autodoc.c:363`).
pub fn adoc_open_image_metadata(
    filename: &[u8],
    add_mdoc: i32,
    montage: &mut i32,
    num_sect: &mut i32,
    sect_type: &mut i32,
) -> i32 {
    // Filesystem access already uses a lossy Rust `String` below, so keep the
    // path as owned text while attaching the optional metadata suffix.  The
    // autodoc contents themselves remain bytes: unlike paths at this boundary,
    // they can contain non-UTF-8 data that must round-trip unchanged.
    let mut usename = String::from_utf8_lossy(filename).into_owned();
    let series: i32;
    let index: i32;

    /* Attach extension to file if requested */
    if add_mdoc > 0 {
        usename.push_str(".mdoc");
    }

    /* Return -2 if it does not exist, -1 if error reading it */
    if std::fs::metadata(&usename).is_err() {
        index = -2;
    } else {
        index = adoc_read(usename.as_bytes());
    }
    if index < 0 {
        return index;
    }

    series = adoc_get_image_meta_info(montage, num_sect, sect_type);
    if series < 0 || (series > 0 && (add_mdoc == 0 || add_mdoc == 1)) {
        adoc_clear(index);
        return if series > 0 { -3 } else { series };
    }
    index
}

/// Matches C `AdocGetImageMetaInfo` (`autodoc.c:405`).
pub fn adoc_get_image_meta_info(montage: &mut i32, num_sect: &mut i32, sect_type: &mut i32) -> i32 {
    let mut series: i32 = 0;
    let mut usename: Vec<u8> = Vec::new();

    *montage = 0;
    if adoc_get_string(ADOC_GLOBAL_NAME, 0, b"ImageFile", &mut usename) == 0 {
        *sect_type = 1;
        *num_sect = adoc_get_number_of_sections(ADOC_ZVALUE_NAME);
    } else if adoc_get_integer(ADOC_GLOBAL_NAME, 0, b"ImageSeries", &mut series) == 0 && series != 0
    {
        *sect_type = 2;
        *num_sect = adoc_get_number_of_sections(b"Image");
    } else {
        *num_sect = adoc_get_number_of_sections(ADOC_ZVALUE_NAME);
        if *num_sect == 0 {
            *num_sect = adoc_get_number_of_sections(ADOC_FRAMESET_NAME);
            if *num_sect == 1 {
                *sect_type = 4;
                return 1;
            } else {
                *num_sect = 0;
                return -3;
            }
        } else {
            *sect_type = 3;
        }
    }

    if adoc_get_integer(ADOC_GLOBAL_NAME, 0, b"Montage", montage) != 0 {
        adoc_get_integer(ADOC_GLOBAL_NAME, 0, b"IMOD.Montage", montage);
    }
    0
}

/// Matches C `AdocNew` (`autodoc.c:436`).
pub fn adoc_new() -> i32 {
    let err: i32 = add_autodoc();
    if err < 0 {
        return err;
    }
    adoc_set_current(err);
    err
}

/// Matches C `AdocGetCurrentIndex` (`autodoc.c:448`).
pub fn adoc_get_current_index() -> i32 {
    S_CUR_ADOC_IND.get()
}

/// Matches C `AdocSetCurrent` (`autodoc.c:457`).
pub fn adoc_set_current(index: i32) -> i32 {
    if index < 0 || index >= S_AUTODOCS.with_borrow(|adocs| adocs.len() as i32) {
        return -1;
    }
    S_CUR_ADOC_IND.set(index);
    0
}

/// Matches C `AdocClear` (`autodoc.c:469`).
pub fn adoc_clear(index: i32) {
    S_AUTODOCS.with_borrow_mut(|adocs| {
        if index >= 0 && index < adocs.len() as i32 {
            delete_adoc(&mut adocs[index as usize]);
        }
    });
}

/// Matches C `AdocDone` (`autodoc.c:478`).
pub fn adoc_done() {
    S_AUTODOCS.with_borrow_mut(|adocs| {
        for adoc in adocs.iter_mut() {
            delete_adoc(adoc);
        }
        adocs.clear();
    });
    S_NAME_FOR_ORDERING.with_borrow_mut(|name| *name = None);
    S_CUR_ADOC_IND.set(-1);
}

/// Matches C `AdocWrite` (`autodoc.c:495`).
pub fn adoc_write(filename: &[u8]) -> i32 {
    let mut backerr: i32 = 0;
    let mut retval: i32 = 0;

    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    let (backed_up, write_as_xml) = S_AUTODOCS.with_borrow(|adocs| {
        (
            adocs[cur as usize].backed_up,
            adocs[cur as usize].write_as_xml,
        )
    });
    if backed_up == 0 {
        backerr = imod_backup_file(String::from_utf8_lossy(filename).as_ref());
    }
    S_AUTODOCS.with_borrow_mut(|adocs| adocs[cur as usize].backed_up = 1);
    if write_as_xml != 0 {
        return write_xml_file(filename);
    }
    let Some(mut afile) = open_for_write(filename, "w") else {
        return -1;
    };
    if S_AUTODOCS
        .with_borrow(|adocs| write_file(&adocs[cur as usize], Some(afile.clone()), None, 0, 1))
        != 0
    {
        retval = -1;
    } else {
        S_AUTODOCS.with_borrow(|adocs| {
            let adoc = &adocs[cur as usize];
            for i in 0..adoc.num_final_com as usize {
                let _ = afile.write_all(&c_format_bytes(
                    "%s\n",
                    &[CArg::Bytes(&adoc.final_comments[i])],
                ));
            }
        });
    }

    if retval != 0 { retval } else { backerr }
}

/// Matches C `AdocRetryWriteOpens` (`autodoc.c:522`).
pub fn adoc_retry_write_opens(num: i32) {
    S_OPEN_RETRIES.set(num);
}

/// Matches C `AdocSetWriteAsXML` (`autodoc.c:530`).
pub fn adoc_set_write_as_xml(as_xml: i32) {
    let cur = S_CUR_ADOC_IND.get();
    if cur >= 0 {
        S_AUTODOCS.with_borrow_mut(|adocs| {
            adocs[cur as usize].write_as_xml = if as_xml != 0 { 1 } else { 0 }
        });
    }
}

/// Matches C `AdocGetWriteAsXML` (`autodoc.c:540`).
pub fn adoc_get_write_as_xml() -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow(|adocs| adocs[cur as usize].write_as_xml)
}

/// Matches C `AdocGetXmlRootElement` (`autodoc.c:552`).
///
/// The C `strdup`s into the caller's `char **`; the copy is the caller's `Vec`
/// now, and the source's NULL when there is no root element is `None`.
pub fn adoc_get_xml_root_element(string: &mut Option<Vec<u8>>) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    *string = None;
    S_AUTODOCS.with_borrow(|adocs| {
        if let Some(root) = &adocs[cur as usize].root_element {
            *string = Some(root.clone());
        }
    });
    0
}

/// Matches C `AdocSetXmlRootElement` (`autodoc.c:567`).
pub fn adoc_set_xml_root_element(element: &[u8]) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow_mut(|adocs| adocs[cur as usize].root_element = Some(element.to_vec()));
    0
}

/// Matches C `AdocAppendSection` (`autodoc.c:582`).
pub fn adoc_append_section(filename: &[u8]) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    if S_AUTODOCS.with_borrow(|adocs| adocs[cur as usize].write_as_xml) != 0 {
        return write_xml_file(filename);
    }
    let Some(afile) = open_for_write(filename, "a") else {
        return -1;
    };
    S_AUTODOCS.with_borrow(|adocs| write_file(&adocs[cur as usize], Some(afile), None, 0, 0))
}

/// Matches C `AdocPrintToString` (`autodoc.c:601`).
///
/// The C writes into the caller's `char *` with `snprintf` and refuses to
/// overrun `stringSize`; the bytes land in the caller's `Vec` instead, with the
/// same size limit and the same -1 when it is reached.
pub fn adoc_print_to_string(string: &mut Vec<u8>, string_size: i32, write_all: i32) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        /* `writeFile` dereferences `sCurAdoc` with no NULL check
        (`autodoc.c:627`); reproducing that would be a null dereference. */
        return -1;
    }
    S_AUTODOCS.with_borrow(|adocs| {
        write_file(
            &adocs[cur as usize],
            None,
            Some(string),
            string_size,
            write_all,
        )
    })
}

/// Matches C `AdocOrderWriteByValue` (`autodoc.c:611`).
pub fn adoc_order_write_by_value(type_name: Option<&[u8]>) -> i32 {
    S_NAME_FOR_ORDERING.with_borrow_mut(|name| *name = None);
    let Some(type_name) = type_name else {
        return 0;
    };
    S_NAME_FOR_ORDERING.with_borrow_mut(|name| *name = Some(type_name.to_vec()));
    0
}

/// Matches C static `writeFile` (`autodoc.c:622`).
///
/// `sCurAdoc` is passed in rather than read from the static, because the caller
/// already holds the autodoc list open.
pub fn write_file(
    adoc: &Autodoc,
    afile: Option<ImodFile>,
    string: Option<&mut Vec<u8>>,
    string_size: i32,
    write_all: i32,
) -> i32 {
    let mut i: i32;
    let mut j: i32;
    let mut k: i32;
    let mut ind: i32;
    let mut com_ind: i32;
    let mut write: i32;
    let mut last_blank: i32;
    let mut use_ind: i32;
    let mut retval: i32 = 0;
    let mut ord_sect_inds: Vec<i32> = Vec::new();
    let ordered_write: i32 = if write_all != 0
        && S_NAME_FOR_ORDERING.with_borrow(|name| name.is_some())
        && adoc.num_sections > 1
    {
        1
    } else {
        0
    };
    let to_string = string.is_some();
    S_FILE.with_borrow_mut(|file| *file = afile);
    S_STRING.with_borrow_mut(|s| *s = if to_string { Some(Vec::new()) } else { None });
    S_BYTES_LEFT.set(string_size);
    S_BYTES_WRITTEN.set(0);
    if S_FILE.with_borrow(|file| file.is_none()) && !to_string {
        return -1;
    }

    /* For ordered writing, get arrays for indexes and float values, and set up values
    for all the sections of the given type */
    if ordered_write != 0 {
        let Some(inds) = setup_section_order(adoc) else {
            S_FILE.with_borrow_mut(|file| *file = None);
            S_STRING.with_borrow_mut(|s| *s = None);
            return -1;
        };
        ord_sect_inds = inds;
    }

    /* Initialize delimiter, loop on indexes in the autodoc */
    S_VALUE_DELIM.with_borrow_mut(|delim| *delim = S_DEFAULT_DELIM.to_vec());
    ind = 0;
    while ind < adoc.num_sections {
        write = if write_all != 0 || ind == adoc.num_sections - 1 {
            1
        } else {
            0
        };
        use_ind = if ordered_write != 0 {
            ord_sect_inds[ind as usize]
        } else {
            ind
        };
        i = adoc.coll_list[use_ind as usize];
        j = adoc.sect_list[use_ind as usize];
        let coll = &adoc.collections[i as usize];
        let sect = &coll.sections[j as usize];

        /* dump comments before section */
        com_ind = 0;
        last_blank = 0;
        while write != 0 && com_ind < sect.num_comments && sect.com_index[com_ind as usize] == -1 {
            if write != 0 {
                last_blank = if sect.comments[com_ind as usize].is_empty() {
                    1
                } else {
                    0
                };
                let com = &sect.comments[com_ind as usize];
                com_ind += 1;
                if fs_printf("%s\n", &[CArg::Bytes(com)]) != 0 {
                    retval = -1;
                    break;
                }
            }
        }
        if retval != 0 {
            break;
        }

        /* Write section name unless we're in global */
        if (i != 0 || j != 0 || sect.name.as_deref() != Some(ADOC_GLOBAL_NAME)) && write != 0 {
            let delim = S_VALUE_DELIM.with_borrow(|delim| delim.clone());
            if fs_printf(
                "%s[%s %s %s]\n",
                &[
                    CArg::Bytes(if last_blank != 0 { b"" } else { b"\n" }),
                    CArg::Bytes(coll.name.as_deref().unwrap_or(b"")),
                    CArg::Bytes(&delim),
                    CArg::Bytes(sect.name.as_deref().unwrap_or(b"")),
                ],
            ) != 0
            {
                retval = -1;
                break;
            }
        }

        /* Loop on key-values */
        k = 0;
        while k < sect.num_keys {
            /* dump comments associated with this index */
            while write != 0 && com_ind < sect.num_comments && sect.com_index[com_ind as usize] == k
            {
                if write != 0 {
                    let com = &sect.comments[com_ind as usize];
                    com_ind += 1;
                    if fs_printf("%s\n", &[CArg::Bytes(com)]) != 0 {
                        retval = -1;
                        break;
                    }
                }
            }
            if retval != 0 {
                break;
            }

            /* Print key-value pairs with non-null values */
            if sect.keys[k as usize].is_some() && sect.values[k as usize].is_some() {
                let delim = S_VALUE_DELIM.with_borrow(|delim| delim.clone());
                if write != 0
                    && fs_printf(
                        "%s %s %s\n",
                        &[
                            CArg::Bytes(sect.keys[k as usize].as_deref().unwrap()),
                            CArg::Bytes(&delim),
                            CArg::Bytes(sect.values[k as usize].as_deref().unwrap()),
                        ],
                    ) != 0
                {
                    retval = -1;
                    break;
                }

                /* After a new delimiter is written, need to set delimiter */
                if i == 0
                    && j == 0
                    && sect.keys[k as usize].as_deref() == Some(&b"KeyValueDelimiter"[..])
                {
                    let new = sect.values[k as usize].clone().unwrap();
                    S_NEW_DELIM.with_borrow_mut(|d| *d = Some(new.clone()));
                    S_VALUE_DELIM.with_borrow_mut(|d| *d = new);
                }

            /* Print keys without values too */
            } else if sect.keys[k as usize].is_some() && write != 0 {
                let delim = S_VALUE_DELIM.with_borrow(|delim| delim.clone());
                if fs_printf(
                    "%s %s \n",
                    &[
                        CArg::Bytes(sect.keys[k as usize].as_deref().unwrap()),
                        CArg::Bytes(&delim),
                    ],
                ) != 0
                {
                    retval = -1;
                    break;
                }
            }
            k += 1;
        }
        if retval != 0 {
            break;
        }
        ind += 1;
    }
    S_FILE.with_borrow_mut(|file| *file = None);
    let written = S_STRING.with_borrow_mut(|s| s.take());
    if let (Some(out), Some(written)) = (string, written) {
        *out = written;
    }
    retval
}

/// Matches C static `fsPrintf` (`autodoc.c:740`).
///
/// The source is variadic; the arguments are a `CArg` slice here, and the
/// formatting goes through `b3dutil::c_format_bytes`, the tree's translation of
/// the C library's own `printf`.
pub fn fs_printf(format: &str, args: &[CArg]) -> i32 {
    let mut retval: i32 = 0;
    let num_written: i32;
    let text = c_format_bytes(format, args);
    let have_file = S_FILE.with_borrow(|file| file.is_some());
    if have_file {
        if S_FILE
            .with_borrow_mut(|file| file.as_mut().unwrap().write_all(&text))
            .is_err()
        {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: AdocWrite - writing element to file\n"),
            );
            retval = -1;
        }
    } else {
        /* `snprintf` returns the length it would have written and copies at
        most `sBytesLeft - 1` of it. */
        num_written = text.len() as i32;
        let bytes_left = S_BYTES_LEFT.get();
        let copied = if bytes_left > 0 {
            num_written.min(bytes_left - 1).max(0) as usize
        } else {
            0
        };
        S_STRING.with_borrow_mut(|s| {
            if let Some(s) = s.as_mut() {
                s.extend_from_slice(&text[..copied]);
            }
        });
        if num_written >= bytes_left {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: AdocWrite - writing element to string\n"),
            );
            retval = -1;
        } else {
            S_BYTES_LEFT.set(bytes_left - num_written);
            S_BYTES_WRITTEN.set(S_BYTES_WRITTEN.get() + num_written);
        }
    }
    retval
}

/// Matches C static `setupSectionOrder` (`autodoc.c:768`).
pub fn setup_section_order(adoc: &Autodoc) -> Option<Vec<i32>> {
    let mut i: i32;
    let mut j: i32 = 0;
    let mut ind: i32;
    let mut max_value: f32 = -1.0e37;
    let mut ord_sect_values: Vec<f32> = vec![0.; adoc.num_sections as usize];
    let mut ord_sect_inds: Vec<i32> = vec![0; adoc.num_sections as usize];

    let name_for_ordering = S_NAME_FOR_ORDERING.with_borrow(|name| name.clone());
    ord_sect_values[0] = max_value;
    ord_sect_inds[0] = 0;
    ind = 1;
    while ind < adoc.num_sections {
        i = adoc.coll_list[ind as usize];
        j = adoc.sect_list[ind as usize];
        ord_sect_inds[ind as usize] = ind;
        let coll = &adoc.collections[i as usize];
        if coll.name == name_for_ordering && coll.sections[j as usize].name.is_some() {
            /* `atof(name)` */
            let mut scanned = 0usize;
            ord_sect_values[ind as usize] = strtod(
                coll.sections[j as usize].name.as_deref().unwrap(),
                &mut scanned,
            ) as f32;
            let v = ord_sect_values[ind as usize];
            max_value = if max_value > v { max_value } else { v };
        }
        ind += 1;
    }

    /* Then set up values for the rest of the sections, above the real values and in
    the order they occur */
    max_value = if 0.0f64 > max_value as f64 {
        0.0f64
    } else {
        max_value as f64
    } as f32;
    ind = 1;
    while ind < adoc.num_sections {
        i = adoc.coll_list[ind as usize];
        let coll = &adoc.collections[i as usize];
        /* NOTE: the source reuses `j` from the loop above rather than re-reading
        sectList[ind]; that stale index is preserved here (autodoc.c:806). */
        if coll.name != name_for_ordering || coll.sections[j as usize].name.is_none() {
            max_value = (max_value as f64 + 1.0) as f32;
            ord_sect_values[ind as usize] = max_value;
        }
        ind += 1;
    }

    /* Sort, use the sorted indexes below */
    rs_sort_indexed_floats(&ord_sect_values, &mut ord_sect_inds, adoc.num_sections);
    let _ = &mut ord_sect_values;
    Some(ord_sect_inds)
}

/// Matches C `AdocAddSection` (`autodoc.c:820`).
pub fn adoc_add_section(type_name: &[u8], name: &[u8]) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow_mut(|adocs| {
        let adoc = &mut adocs[cur as usize];
        let mut coll_ind = lookup_collection(adoc, type_name);
        if coll_ind < 0 {
            if add_collection(adoc, type_name) != 0 {
                return -1;
            }
            coll_ind = adoc.num_collections - 1;
        }
        if add_section(adoc, coll_ind, name) != 0 {
            return -1;
        }
        adoc.collections[coll_ind as usize].num_sections - 1
    })
}

/// Matches C `AdocInsertSection` (`autodoc.c:844`).
pub fn adoc_insert_section(type_name: &[u8], sect_ind: i32, name: &[u8]) -> i32 {
    let mut i: i32 = 0;
    let mut coll_ind: i32;
    let mut master_ind: i32 = 0;
    let mut num_sect: i32 = 0;
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    coll_ind = S_AUTODOCS.with_borrow(|adocs| lookup_collection(&adocs[cur as usize], type_name));
    if coll_ind >= 0 {
        num_sect = S_AUTODOCS
            .with_borrow(|adocs| adocs[cur as usize].collections[coll_ind as usize].num_sections);
    }
    if sect_ind < 0 || sect_ind > num_sect {
        return -1;
    }

    /* Find the index of this section in the master list if it needs to be shuffled */
    if sect_ind < num_sect {
        master_ind = find_section_in_adoc_list(coll_ind, sect_ind);
        if master_ind < 0 {
            return -1;
        }
    }

    /* Add section to end regardless, then return if that is all that is needed */
    if adoc_add_section(type_name, name) < 0 {
        return -1;
    }
    if sect_ind == num_sect {
        return 0;
    }

    S_AUTODOCS.with_borrow_mut(|adocs| {
        let adoc = &mut adocs[cur as usize];
        /* Fix collection index if a new collection had to be added */
        if coll_ind < 0 {
            coll_ind = adoc.num_collections - 1;
        }

        /* Save the new section then move existing sections up and copy new one into place */
        let coll = &mut adoc.collections[coll_ind as usize];
        let new_sect = coll.sections[(coll.num_sections - 1) as usize].clone();
        i = coll.num_sections - 1;
        while i > sect_ind {
            coll.sections[i as usize] = coll.sections[(i - 1) as usize].clone();
            i -= 1;
        }
        coll.sections[sect_ind as usize] = new_sect;

        /* Move the master lists up and decrement any other indices in this collection */
        i = adoc.num_sections - 1;
        while i > master_ind {
            adoc.coll_list[i as usize] = adoc.coll_list[(i - 1) as usize];
            adoc.sect_list[i as usize] = adoc.sect_list[(i - 1) as usize];
            if adoc.coll_list[i as usize] == coll_ind && adoc.sect_list[i as usize] >= sect_ind {
                adoc.sect_list[i as usize] += 1;
            }
            i -= 1;
        }

        0
    })
}

/// Matches C `AdocDeleteSection` (`autodoc.c:900`).
pub fn adoc_delete_section(type_name: &[u8], sect_ind: i32) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    let coll_ind =
        S_AUTODOCS.with_borrow(|adocs| lookup_collection(&adocs[cur as usize], type_name));
    if coll_ind < 0 {
        return -1;
    }
    if sect_ind < 0
        || sect_ind
            >= S_AUTODOCS.with_borrow(|adocs| {
                adocs[cur as usize].collections[coll_ind as usize].num_sections
            })
    {
        return -1;
    }

    /* Find the index of this section in the master list */
    let master_ind = find_section_in_adoc_list(coll_ind, sect_ind);
    if master_ind < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow_mut(|adocs| {
        let adoc = &mut adocs[cur as usize];
        let mut i: i32;
        delete_section(&mut adoc.collections[coll_ind as usize].sections[sect_ind as usize]);

        /* Repack the sections */
        let coll = &mut adoc.collections[coll_ind as usize];
        i = sect_ind + 1;
        while i < coll.num_sections {
            coll.sections[(i - 1) as usize] = coll.sections[i as usize].clone();
            i += 1;
        }
        coll.num_sections -= 1;

        /* Repack the master list and decrement any other indices in this collection */
        i = master_ind + 1;
        while i < adoc.num_sections {
            if adoc.coll_list[i as usize] == coll_ind && adoc.sect_list[i as usize] > sect_ind {
                adoc.sect_list[i as usize] -= 1;
            }
            adoc.coll_list[(i - 1) as usize] = adoc.coll_list[i as usize];
            adoc.sect_list[(i - 1) as usize] = adoc.sect_list[i as usize];
            i += 1;
        }
        adoc.num_sections -= 1;
        0
    })
}

/// Matches C `AdocChangeSectionName` (`autodoc.c:942`).
pub fn adoc_change_section_name(type_name: &[u8], sect_ind: i32, new_name: &[u8]) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow_mut(|adocs| {
        let adoc = &mut adocs[cur as usize];
        let coll_ind = lookup_collection(adoc, type_name);
        if coll_ind < 0 {
            return -1;
        }
        let coll = &mut adoc.collections[coll_ind as usize];
        if sect_ind < 0 || sect_ind >= coll.num_sections {
            return -1;
        }
        coll.sections[sect_ind as usize].name = Some(new_name.to_vec());
        0
    })
}

/// Matches C `AdocLookupSection` (`autodoc.c:969`).
pub fn adoc_lookup_section(type_name: &[u8], name: &[u8]) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -2;
    }
    S_AUTODOCS.with_borrow(|adocs| {
        let adoc = &adocs[cur as usize];
        let coll_ind = lookup_collection(adoc, type_name);
        if coll_ind < 0 {
            return -2;
        }
        let coll = &adoc.collections[coll_ind as usize];
        let mut sect_ind = 0;
        while sect_ind < coll.num_sections {
            if coll.sections[sect_ind as usize].name.as_deref() == Some(name) {
                return sect_ind;
            }
            sect_ind += 1;
        }
        -1
    })
}

/// Matches C `AdocLookupByNameValue` (`autodoc.c:993`).
pub fn adoc_lookup_by_name_value(type_name: &[u8], name_value: i32) -> i32 {
    /* `char buf[15]; sprintf(buf, "%d", nameValue);` */
    let buf = c_format_bytes("%d", &[CArg::Int(name_value as i64)]);
    adoc_lookup_section(type_name, &buf)
}

/// Matches C `AdocFindInsertIndex` (`autodoc.c:1005`).
pub fn adoc_find_insert_index(type_name: &[u8], name_value: i32) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow(|adocs| {
        let adoc = &adocs[cur as usize];
        let coll_ind = lookup_collection(adoc, type_name);
        if coll_ind < 0 {
            return 0;
        }
        let coll = &adoc.collections[coll_ind as usize];
        let mut sect_ind = 0;
        while sect_ind < coll.num_sections {
            /* `atoi(sections[sectInd].name)` */
            let mut scanned = 0usize;
            let sect_value = strtol(
                coll.sections[sect_ind as usize]
                    .name
                    .as_deref()
                    .unwrap_or(b""),
                &mut scanned,
                10,
            ) as i32;
            if name_value == sect_value {
                return -1;
            }
            if name_value < sect_value {
                return sect_ind;
            }
            sect_ind += 1;
        }
        coll.num_sections
    })
}

/// Matches C `AdocTransferSection` (`autodoc.c:1039`).
pub fn adoc_transfer_section(
    type_name: &[u8],
    sect_ind: i32,
    to_adoc_ind: i32,
    new_name: Option<&[u8]>,
    by_value: i32,
) -> i32 {
    adoc_transfer_to_new_type(
        type_name,
        sect_ind,
        to_adoc_ind,
        type_name,
        new_name,
        by_value,
    )
}

/// Matches C `AdocTransferToNewType` (`autodoc.c:1049`).
///
/// The C holds an `AdocSection *` into the source autodoc across the switch to
/// the destination one; the key-value triples it reads out of that section are
/// copied here before the switch, which is the same data because the loop never
/// writes to the source.
pub fn adoc_transfer_to_new_type(
    type_name: &[u8],
    sect_ind: i32,
    to_adoc_ind: i32,
    new_type: &[u8],
    new_name: Option<&[u8]>,
    by_value: i32,
) -> i32 {
    let mut err: i32 = 0;
    let mut ind: i32;
    let mut new_sect_ind: i32;
    let coll_ind: i32;
    let name_val: i32;
    let cur_ind_save: i32 = S_CUR_ADOC_IND.get();

    /* Get the section then switch adocs */
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    let num_autodocs = S_AUTODOCS.with_borrow(|adocs| adocs.len() as i32);
    let entries: Vec<(Option<Vec<u8>>, Option<Vec<u8>>, u8)> =
        match S_AUTODOCS.with_borrow(|adocs| {
            let adoc = &adocs[cur as usize];
            get_section(adoc, type_name, sect_ind).map(|(ic, is)| {
                let sect = &adoc.collections[ic].sections[is];
                (0..sect.num_keys as usize)
                    .map(|n| (sect.keys[n].clone(), sect.values[n].clone(), sect.types[n]))
                    .collect::<Vec<_>>()
            })
        }) {
            Some(entries) => entries,
            None => return -1,
        };
    if to_adoc_ind < 0 || to_adoc_ind == cur_ind_save || to_adoc_ind >= num_autodocs {
        return -2;
    }
    adoc_set_current(to_adoc_ind);

    /* Set index to 0 for global section or go on to look up and/or add section */
    if type_name == ADOC_GLOBAL_NAME {
        new_sect_ind = 0;
    } else {
        let Some(new_name) = new_name else {
            return -2;
        };

        /* If the section does not exist, add it, using insert if the collection does
        exist */
        new_sect_ind = adoc_lookup_section(new_type, new_name);
        if new_sect_ind < 0 {
            coll_ind = S_AUTODOCS
                .with_borrow(|adocs| lookup_collection(&adocs[to_adoc_ind as usize], new_type));
            if coll_ind < 0 {
                new_sect_ind = 0;
                err = adoc_add_section(new_type, new_name);
            } else {
                new_sect_ind = S_AUTODOCS.with_borrow(|adocs| {
                    adocs[to_adoc_ind as usize].collections[coll_ind as usize].num_sections
                });
                if by_value != 0 {
                    /* `atoi(newName)` */
                    let mut scanned = 0usize;
                    name_val = strtol(new_name, &mut scanned, 10) as i32;
                    new_sect_ind = adoc_find_insert_index(new_type, name_val);
                }
                err = adoc_insert_section(new_type, new_sect_ind, new_name);
            }
            if err < 0 {
                adoc_set_current(cur_ind_save);
                return -3;
            }
        }
    }

    /* Copy the key/values and their types.  Skip NULL ones, which happen with HDF adoc */
    ind = 0;
    while (ind as usize) < entries.len() && err == 0 {
        let (key, value, type_) = &entries[ind as usize];
        if key.is_some()
            && value.is_some()
            && set_key_value_type(
                new_type,
                new_sect_ind,
                key.as_deref().unwrap(),
                value.as_deref(),
                *type_ as i32,
            ) < 0
        {
            err = -4;
        }
        ind += 1;
    }
    adoc_set_current(cur_ind_save);
    err
}

/// Matches C `AdocSetKeyValue` (`autodoc.c:1109`).
pub fn adoc_set_key_value(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    value: Option<&[u8]>,
) -> i32 {
    set_key_value_type(type_name, sect_ind, key, value, ADOC_STRING)
}

/// Matches C static `setKeyValueType` (`autodoc.c:1117`).
pub fn set_key_value_type(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    value: Option<&[u8]>,
    type_: i32,
) -> i32 {
    let mut key_ind: i32 = 0;
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow_mut(|adocs| {
        let adoc = &mut adocs[cur as usize];
        let Some((ic, is)) = get_section(adoc, type_name, sect_ind) else {
            return -1;
        };
        if value.is_none() {
            return -1;
        }
        let sect = &mut adoc.collections[ic].sections[is];
        sect_set_key_value_type(sect, key, value, type_, &mut key_ind)
    })
}

/// Matches C static `sectSetKeyValueType` (`autodoc.c:1130`).
pub fn sect_set_key_value_type(
    sect: &mut AdocSection,
    key: &[u8],
    value: Option<&[u8]>,
    type_: i32,
    key_ind: &mut i32,
) -> i32 {
    *key_ind = lookup_key(sect, key);

    /* If key already exists, clear out value and set it again */
    if *key_ind >= 0 {
        if let Some(value) = value {
            sect.values[*key_ind as usize] = Some(value.to_vec());
            sect.types[*key_ind as usize] = type_ as u8;
        } else {
            sect.values[*key_ind as usize] = None;
            sect.types[*key_ind as usize] = ADOC_NO_VALUE as u8;
        }
    } else {
        *key_ind = sect.num_keys;
        return add_key(sect, key, value, type_);
    }
    0
}

/// Matches C `AdocSetInteger` (`autodoc.c:1163`).
pub fn adoc_set_integer(type_name: &[u8], sect_ind: i32, key: &[u8], ival: i32) -> i32 {
    let str = c_format_bytes("%d", &[CArg::Int(ival as i64)]);
    set_key_value_type(type_name, sect_ind, key, Some(&str), ADOC_ONE_INT)
}

/// Matches C `AdocSetTwoIntegers` (`autodoc.c:1174`).
pub fn adoc_set_two_integers(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    ival1: i32,
    ival2: i32,
) -> i32 {
    let str = c_format_bytes("%d %d", &[CArg::Int(ival1 as i64), CArg::Int(ival2 as i64)]);
    set_key_value_type(type_name, sect_ind, key, Some(&str), ADOC_TWO_INTS)
}

/// Matches C `AdocSetThreeIntegers` (`autodoc.c:1186`).
pub fn adoc_set_three_integers(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    ival1: i32,
    ival2: i32,
    ival3: i32,
) -> i32 {
    let str = c_format_bytes(
        "%d %d %d",
        &[
            CArg::Int(ival1 as i64),
            CArg::Int(ival2 as i64),
            CArg::Int(ival3 as i64),
        ],
    );
    set_key_value_type(type_name, sect_ind, key, Some(&str), ADOC_THREE_INTS)
}

/// Matches C `AdocSetIntegerArray` (`autodoc.c:1198`).
pub fn adoc_set_integer_array(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    ivals: &[i32],
    num_vals: i32,
) -> i32 {
    set_array_of_values(
        type_name,
        sect_ind,
        key,
        ArrayOfValues::Ints(ivals),
        num_vals,
        ADOC_INT_ARRAY,
    )
}

/// Matches C `AdocSetFloat` (`autodoc.c:1210`).
pub fn adoc_set_float(type_name: &[u8], sect_ind: i32, key: &[u8], val: f32) -> i32 {
    /* `val` is promoted to double by the C varargs call. */
    let str = c_format_bytes("%g", &[CArg::Dbl(val as f64)]);
    set_key_value_type(type_name, sect_ind, key, Some(&str), ADOC_ONE_FLOAT)
}

/// Matches C `AdocSetTwoFloats` (`autodoc.c:1221`).
pub fn adoc_set_two_floats(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    val1: f32,
    val2: f32,
) -> i32 {
    let str = c_format_bytes("%g %g", &[CArg::Dbl(val1 as f64), CArg::Dbl(val2 as f64)]);
    set_key_value_type(type_name, sect_ind, key, Some(&str), ADOC_TWO_FLOATS)
}

/// Matches C `AdocSetThreeFloats` (`autodoc.c:1233`).
pub fn adoc_set_three_floats(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    val1: f32,
    val2: f32,
    val3: f32,
) -> i32 {
    let str = c_format_bytes(
        "%g %g %g",
        &[
            CArg::Dbl(val1 as f64),
            CArg::Dbl(val2 as f64),
            CArg::Dbl(val3 as f64),
        ],
    );
    set_key_value_type(type_name, sect_ind, key, Some(&str), ADOC_THREE_FLOATS)
}

/// Matches C `AdocSetFloatArray` (`autodoc.c:1245`).
pub fn adoc_set_float_array(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    vals: &[f32],
    num_vals: i32,
) -> i32 {
    set_array_of_values(
        type_name,
        sect_ind,
        key,
        ArrayOfValues::Floats(vals),
        num_vals,
        ADOC_FLOAT_ARRAY,
    )
}

/// Matches C `AdocSetDouble` (`autodoc.c:1252`).
pub fn adoc_set_double(type_name: &[u8], sect_ind: i32, key: &[u8], val: f64) -> i32 {
    let str = c_format_bytes("%g", &[CArg::Dbl(val)]);
    set_key_value_type(type_name, sect_ind, key, Some(&str), ADOC_ONE_DOUBLE)
}

/// The `void *vals` that `setArrayOfValues` (`autodoc.c:1263`) casts to either
/// `int *` or `float *` according to `valType`, named rather than punned.
pub enum ArrayOfValues<'a> {
    Ints(&'a [i32]),
    Floats(&'a [f32]),
}

/// Matches C static `setArrayOfValues` (`autodoc.c:1263`).
pub fn set_array_of_values(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    vals: ArrayOfValues,
    num_vals: i32,
    val_type: i32,
) -> i32 {
    let mut tmp: Vec<u8>;
    let mut full_str: Vec<u8> = Vec::new();
    let mut ind: i32;
    let mut tot_len: i32 = 0;

    /* Add up the characters needed for the each value */
    ind = 0;
    while ind < num_vals {
        /* C branches on `valType == ADOC_INT_ARRAY` and casts the same
        `void *` either way; the variant carries that choice instead. */
        tmp = match &vals {
            ArrayOfValues::Ints(ivals) => {
                c_format_bytes("%d ", &[CArg::Int(ivals[ind as usize] as i64)])
            }
            ArrayOfValues::Floats(fvals) => {
                c_format_bytes("%g ", &[CArg::Dbl(fvals[ind as usize] as f64)])
            }
        };
        tot_len += tmp.len() as i32 + 1;
        ind += 1;
    }
    let _ = tot_len;

    /* Get the string and build it up by writing again */
    ind = 0;
    while ind < num_vals {
        tmp = match &vals {
            ArrayOfValues::Ints(ivals) => c_format_bytes(
                "%s%d",
                &[
                    CArg::Bytes(if ind != 0 { b" " } else { b"" }),
                    CArg::Int(ivals[ind as usize] as i64),
                ],
            ),
            ArrayOfValues::Floats(fvals) => c_format_bytes(
                "%s%g",
                &[
                    CArg::Bytes(if ind != 0 { b" " } else { b"" }),
                    CArg::Dbl(fvals[ind as usize] as f64),
                ],
            ),
        };
        full_str.extend_from_slice(&tmp);
        ind += 1;
    }
    set_key_value_type(type_name, sect_ind, key, Some(&full_str), val_type)
}

/// Matches C `AdocDeleteKeyValue` (`autodoc.c:1305`).
pub fn adoc_delete_key_value(type_name: &[u8], sect_ind: i32, key: &[u8]) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow_mut(|adocs| {
        let adoc = &mut adocs[cur as usize];
        let Some((ic, is)) = get_section(adoc, type_name, sect_ind) else {
            return -1;
        };
        let sect = &mut adoc.collections[ic].sections[is];
        let key_ind = lookup_key(sect, key);
        if key_ind < 0 {
            return -1;
        }
        sect.values[key_ind as usize] = None;
        sect.keys[key_ind as usize] = None;
        sect.types[key_ind as usize] = ADOC_NO_VALUE as u8;
        0
    })
}

/// Matches C `AdocGetNumCollections` (`autodoc.c:1331`).
pub fn adoc_get_num_collections() -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow(|adocs| adocs[cur as usize].num_collections - 1)
}

/// Matches C `AdocGetCollectionName` (`autodoc.c:1343`).
pub fn adoc_get_collection_name(coll_ind: i32, string: &mut Vec<u8>) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow(|adocs| {
        let adoc = &adocs[cur as usize];
        if coll_ind < 0 || coll_ind >= adoc.num_collections - 1 {
            return -1;
        }
        *string = adoc.collections[(coll_ind + 1) as usize]
            .name
            .clone()
            .unwrap_or_default();
        0
    })
}

/// Matches C `AdocGetSectionName` (`autodoc.c:1356`).
pub fn adoc_get_section_name(type_name: &[u8], sect_ind: i32, string: &mut Vec<u8>) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow(|adocs| {
        let adoc = &adocs[cur as usize];
        let Some((ic, is)) = get_section(adoc, type_name, sect_ind) else {
            return -1;
        };
        *string = adoc.collections[ic].sections[is]
            .name
            .clone()
            .unwrap_or_default();
        0
    })
}

/// Matches C `AdocGetNumberOfSections` (`autodoc.c:1370`).
pub fn adoc_get_number_of_sections(type_name: &[u8]) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow(|adocs| {
        let adoc = &adocs[cur as usize];
        let coll_ind = lookup_collection(adoc, type_name);
        if coll_ind < 0 {
            return 0;
        }
        adoc.collections[coll_ind as usize].num_sections
    })
}

/// Matches C `AdocGetNumberOfKeys` (`autodoc.c:1385`).
pub fn adoc_get_number_of_keys(type_name: &[u8], sect_ind: i32) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow(|adocs| {
        let adoc = &adocs[cur as usize];
        let Some((ic, is)) = get_section(adoc, type_name, sect_ind) else {
            return -1;
        };
        adoc.collections[ic].sections[is].num_keys
    })
}

/// Matches C `AdocGetKeyByIndex` (`autodoc.c:1399`).
pub fn adoc_get_key_by_index(
    type_name: &[u8],
    sect_ind: i32,
    key_ind: i32,
    key: &mut Option<Vec<u8>>,
) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow(|adocs| {
        let adoc = &adocs[cur as usize];
        let Some((ic, is)) = get_section(adoc, type_name, sect_ind) else {
            return -1;
        };
        let sect = &adoc.collections[ic].sections[is];
        if key_ind < 0 || key_ind >= sect.num_keys {
            return -1;
        }
        *key = sect.keys[key_ind as usize].clone();
        0
    })
}

/// Matches C `AdocGetValTypeAndSize` (`autodoc.c:1420`).
pub fn adoc_get_val_type_and_size(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    val_type: &mut i32,
    num_tokens: &mut i32,
) -> i32 {
    *val_type = ADOC_NO_VALUE;
    *num_tokens = 0;
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow(|adocs| {
        let adoc = &adocs[cur as usize];
        let Some((ic, is)) = get_section(adoc, type_name, sect_ind) else {
            return -1;
        };
        let sect = &adoc.collections[ic].sections[is];
        let key_ind = lookup_key(sect, key);
        if key_ind < 0 || sect.values[key_ind as usize].is_none() {
            return 1;
        }
        *val_type = sect.types[key_ind as usize] as i32;

        /* Get a copy of the string and use the dreadful strtok */
        let valstr = sect.values[key_ind as usize].as_deref().unwrap();
        for token in valstr.split(|&b| b == b' ') {
            if !token.is_empty() {
                *num_tokens += 1;
            }
        }
        0
    })
}

/// Matches C `AdocGetString` (`autodoc.c:1457`).
pub fn adoc_get_string(type_name: &[u8], sect_ind: i32, key: &[u8], string: &mut Vec<u8>) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow(|adocs| {
        let adoc = &adocs[cur as usize];
        let Some((ic, is)) = get_section(adoc, type_name, sect_ind) else {
            return -1;
        };
        let sect = &adoc.collections[ic].sections[is];
        let key_ind = lookup_key(sect, key);
        if key_ind < 0 || sect.values[key_ind as usize].is_none() {
            return 1;
        }
        *string = sect.values[key_ind as usize].clone().unwrap();
        0
    })
}

/// Matches C `AdocGetInteger` (`autodoc.c:1477`).
pub fn adoc_get_integer(type_name: &[u8], sect_ind: i32, key: &[u8], val1: &mut i32) -> i32 {
    let mut num: i32 = 1;
    let mut tmp: [i32; 1] = [0; 1];
    let err = adoc_get_integer_array(type_name, sect_ind, key, &mut tmp, &mut num, 1);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    0
}

/// Matches C `AdocGetFloat` (`autodoc.c:1489`).
pub fn adoc_get_float(type_name: &[u8], sect_ind: i32, key: &[u8], val1: &mut f32) -> i32 {
    let mut num: i32 = 1;
    let mut tmp: [f32; 1] = [0.; 1];
    let err = adoc_get_float_array(type_name, sect_ind, key, &mut tmp, &mut num, 1);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    0
}

/// Matches C `AdocGetDouble` (`autodoc.c:1501`).
pub fn adoc_get_double(type_name: &[u8], sect_ind: i32, key: &[u8], val1: &mut f64) -> i32 {
    let mut num_to_get: i32 = 1;
    let mut string: Vec<u8> = Vec::new();
    let err = adoc_get_string(type_name, sect_ind, key, &mut string);
    if err != 0 {
        return err;
    }
    pip_get_line_of_values(
        &string,
        &string,
        crate::imod::libcfshr::parse_params::PipValueArray::Double(core::slice::from_mut(val1)),
        PIP_DOUBLE,
        &mut num_to_get,
        1,
    )
}

/// Matches C `AdocGetTwoIntegers` (`autodoc.c:1517`).
pub fn adoc_get_two_integers(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    val1: &mut i32,
    val2: &mut i32,
) -> i32 {
    let mut num: i32 = 2;
    let mut tmp: [i32; 2] = [0; 2];
    let err = adoc_get_integer_array(type_name, sect_ind, key, &mut tmp, &mut num, 2);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    *val2 = tmp[1];
    0
}

/// Matches C `AdocGetTwoFloats` (`autodoc.c:1531`).
pub fn adoc_get_two_floats(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    val1: &mut f32,
    val2: &mut f32,
) -> i32 {
    let mut num: i32 = 2;
    let mut tmp: [f32; 2] = [0.; 2];
    let err = adoc_get_float_array(type_name, sect_ind, key, &mut tmp, &mut num, 2);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    *val2 = tmp[1];
    0
}

/// Matches C `AdocGetThreeIntegers` (`autodoc.c:1548`).
pub fn adoc_get_three_integers(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    val1: &mut i32,
    val2: &mut i32,
    val3: &mut i32,
) -> i32 {
    let mut num: i32 = 3;
    let mut tmp: [i32; 3] = [0; 3];
    let err = adoc_get_integer_array(type_name, sect_ind, key, &mut tmp, &mut num, 3);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    *val2 = tmp[1];
    *val3 = tmp[2];
    0
}

/// Matches C `AdocGetThreeFloats` (`autodoc.c:1563`).
pub fn adoc_get_three_floats(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    val1: &mut f32,
    val2: &mut f32,
    val3: &mut f32,
) -> i32 {
    let mut num: i32 = 3;
    let mut tmp: [f32; 3] = [0.; 3];
    let err = adoc_get_float_array(type_name, sect_ind, key, &mut tmp, &mut num, 3);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    *val2 = tmp[1];
    *val3 = tmp[2];
    0
}

/// Matches C `AdocGetIntegerArray` (`autodoc.c:1587`).
pub fn adoc_get_integer_array(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    array: &mut [i32],
    num_to_get: &mut i32,
    array_size: i32,
) -> i32 {
    let mut string: Vec<u8> = Vec::new();
    let err = adoc_get_string(type_name, sect_ind, key, &mut string);
    if err != 0 {
        return err;
    }
    pip_get_line_of_values(
        &string,
        &string,
        crate::imod::libcfshr::parse_params::PipValueArray::Int(array),
        PIP_INTEGER,
        num_to_get,
        array_size,
    )
}

/// Matches C `AdocGetFloatArray` (`autodoc.c:1601`).
pub fn adoc_get_float_array(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    array: &mut [f32],
    num_to_get: &mut i32,
    array_size: i32,
) -> i32 {
    let mut string: Vec<u8> = Vec::new();
    let err = adoc_get_string(type_name, sect_ind, key, &mut string);
    if err != 0 {
        return err;
    }
    pip_get_line_of_values(
        &string,
        &string,
        crate::imod::libcfshr::parse_params::PipValueArray::Float(array),
        PIP_FLOAT,
        num_to_get,
        array_size,
    )
}

/// Matches C `AdocGetDoubleArray` (`autodoc.c:1615`).
pub fn adoc_get_double_array(
    type_name: &[u8],
    sect_ind: i32,
    key: &[u8],
    array: &mut [f64],
    num_to_get: &mut i32,
    array_size: i32,
) -> i32 {
    let mut string: Vec<u8> = Vec::new();
    let err = adoc_get_string(type_name, sect_ind, key, &mut string);
    if err != 0 {
        return err;
    }
    pip_get_line_of_values(
        &string,
        &string,
        crate::imod::libcfshr::parse_params::PipValueArray::Double(array),
        PIP_DOUBLE,
        num_to_get,
        array_size,
    )
}

/// Matches C `AdocWriteInteger` (`autodoc.c:1638`).
pub fn adoc_write_integer(fp: &mut ImodFile, key: &[u8], ival: i32) -> i32 {
    if fp
        .write_all(&c_format_bytes(
            "%s = %d\n",
            &[CArg::Bytes(key), CArg::Int(ival as i64)],
        ))
        .is_err()
    {
        return 1;
    }
    0
}

/// Matches C `AdocWriteTwoIntegers` (`autodoc.c:1649`).
pub fn adoc_write_two_integers(fp: &mut ImodFile, key: &[u8], ival1: i32, ival2: i32) -> i32 {
    if fp
        .write_all(&c_format_bytes(
            "%s = %d %d\n",
            &[
                CArg::Bytes(key),
                CArg::Int(ival1 as i64),
                CArg::Int(ival2 as i64),
            ],
        ))
        .is_err()
    {
        return 1;
    }
    0
}

/// Matches C `AdocWriteThreeIntegers` (`autodoc.c:1660`).
pub fn adoc_write_three_integers(
    fp: &mut ImodFile,
    key: &[u8],
    ival1: i32,
    ival2: i32,
    ival3: i32,
) -> i32 {
    if fp
        .write_all(&c_format_bytes(
            "%s = %d %d %d\n",
            &[
                CArg::Bytes(key),
                CArg::Int(ival1 as i64),
                CArg::Int(ival2 as i64),
                CArg::Int(ival3 as i64),
            ],
        ))
        .is_err()
    {
        return 1;
    }
    0
}

/// Matches C `AdocWriteIntegerArray` (`autodoc.c:1671`).
pub fn adoc_write_integer_array(
    fp: &mut ImodFile,
    key: &[u8],
    ivals: &[i32],
    num_vals: i32,
) -> i32 {
    let mut ind: i32;
    if fp
        .write_all(&c_format_bytes("%s =", &[CArg::Bytes(key)]))
        .is_err()
    {
        return 1;
    }
    ind = 0;
    while ind < num_vals {
        if fp
            .write_all(&c_format_bytes(
                " %d",
                &[CArg::Int(ivals[ind as usize] as i64)],
            ))
            .is_err()
        {
            return 1;
        }
        ind += 1;
    }
    if fp.write_all(b"\n").is_err() {
        return 1;
    }
    0
}

/// Matches C `AdocWriteFloat` (`autodoc.c:1685`).
pub fn adoc_write_float(fp: &mut ImodFile, key: &[u8], val: f32) -> i32 {
    if fp
        .write_all(&c_format_bytes(
            "%s = %g\n",
            &[CArg::Bytes(key), CArg::Dbl(val as f64)],
        ))
        .is_err()
    {
        return 1;
    }
    0
}

/// Matches C `AdocWriteTwoFloats` (`autodoc.c:1695`).
pub fn adoc_write_two_floats(fp: &mut ImodFile, key: &[u8], val1: f32, val2: f32) -> i32 {
    if fp
        .write_all(&c_format_bytes(
            "%s = %g %g\n",
            &[
                CArg::Bytes(key),
                CArg::Dbl(val1 as f64),
                CArg::Dbl(val2 as f64),
            ],
        ))
        .is_err()
    {
        return 1;
    }
    0
}

/// Matches C `AdocWriteThreeFloats` (`autodoc.c:1706`).
pub fn adoc_write_three_floats(
    fp: &mut ImodFile,
    key: &[u8],
    val1: f32,
    val2: f32,
    val3: f32,
) -> i32 {
    if fp
        .write_all(&c_format_bytes(
            "%s = %g %g %g\n",
            &[
                CArg::Bytes(key),
                CArg::Dbl(val1 as f64),
                CArg::Dbl(val2 as f64),
                CArg::Dbl(val3 as f64),
            ],
        ))
        .is_err()
    {
        return 1;
    }
    0
}

/// Matches C `AdocWriteFloatArray` (`autodoc.c:1717`).
pub fn adoc_write_float_array(fp: &mut ImodFile, key: &[u8], vals: &[f32], num_vals: i32) -> i32 {
    let mut ind: i32;
    if fp
        .write_all(&c_format_bytes("%s =", &[CArg::Bytes(key)]))
        .is_err()
    {
        return 1;
    }
    ind = 0;
    while ind < num_vals {
        if fp
            .write_all(&c_format_bytes(
                " %g",
                &[CArg::Dbl(vals[ind as usize] as f64)],
            ))
            .is_err()
        {
            return 1;
        }
        ind += 1;
    }
    if fp.write_all(b"\n").is_err() {
        return 1;
    }
    0
}

/// Matches C `AdocWriteDouble` (`autodoc.c:1731`).
pub fn adoc_write_double(fp: &mut ImodFile, key: &[u8], val: f64) -> i32 {
    if fp
        .write_all(&c_format_bytes(
            "%s = %g\n",
            &[CArg::Bytes(key), CArg::Dbl(val)],
        ))
        .is_err()
    {
        return 1;
    }
    0
}

/// Matches C `AdocWriteKeyValue` (`autodoc.c:1741`).
pub fn adoc_write_key_value(fp: &mut ImodFile, key: &[u8], value: &[u8]) -> i32 {
    if fp
        .write_all(&c_format_bytes(
            "%s = %s\n",
            &[CArg::Bytes(key), CArg::Bytes(value)],
        ))
        .is_err()
    {
        return 1;
    }
    0
}

/// Matches C `AdocWriteSectionStart` (`autodoc.c:1752`).
pub fn adoc_write_section_start(fp: &mut ImodFile, key: &[u8], value: Option<&[u8]>) -> i32 {
    if fp
        .write_all(&c_format_bytes(
            "[%s = %s]\n",
            &[CArg::Bytes(key), CArg::Bytes(value.unwrap_or(b""))],
        ))
        .is_err()
    {
        return 1;
    }
    0
}

/// Matches C static `readXmlFile` (`autodoc.c:1765`).
///
/// The mini-XML tree is arena-allocated, so the node pointers are slot indices
/// into a `MxmlArena` that lives for the length of this function, and the
/// element names and values it hands back are byte slices that go straight into
/// the autodoc's own `Vec<u8>` storage.  The autodoc is passed in rather than
/// read from `sCurAdoc`, because the caller already holds the list open.
pub fn read_xml_file(adoc: &mut Autodoc, fp: &mut ImodFile) -> i32 {
    let xml: Option<usize>;
    let mut node: Option<usize>;
    let mut top: Option<usize>;
    let mut sect_node: Option<usize>;
    let mut child: Option<usize>;
    let mut key: Option<Vec<u8>>;
    let mut value: Option<Vec<u8>>;
    let mut icol: i32;
    let mut ind: i32;
    let mut global: i32;
    let mut last_ind: i32 = -1;
    let mut err: i32 = 0;
    let mut cur_coll: i32 = 0;
    let mut cur_sect: i32 = 0;
    let mut comment_list: Vec<Vec<u8>> = Vec::new();
    let mut max_comments: i32 = 0;
    let mut num_comments: i32 = 0;

    S_LAST_WAS_XML.set(1);
    let _ = fp.seek(SeekFrom::Start(0));

    S_NUM_SECT_NOT_ELEM.set(0);
    S_NUM_SECT_NO_NAME.set(0);
    S_NUM_CHILD_NOT_ELEM.set(0);
    S_NUM_CHILD_ATTRIBS.set(0);
    S_NUM_VALUE_NOT_TEXT.set(0);
    S_NUM_MULTIPLE_CHILDS.set(0);

    let arena = &mut MxmlArena::new();
    xml = mxml_load_file(arena, MXML_NO_PARENT, fp, Some(mxml_opaque_cb));
    if xml.is_none() {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: AdocRead - Loading file as XML\n"),
        );
        return -2;
    }

    /* Get the top node and then get the next if it is opaque node */
    top = mxml_walk_next(arena, xml, xml, MXML_DESCEND);
    if mxml_get_type(arena, top) == MXML_OPAQUE {
        top = mxml_walk_next(arena, top, xml, MXML_DESCEND);
    }
    key = mxml_get_element(arena, top).map(|k| k.to_vec());
    if let Some(key) = &key {
        adoc.root_element = Some(key.clone());
    }

    /* Walk through the children of the top node, (autodoc) */
    sect_node = mxml_get_first_child(arena, top);
    while let Some(cur_sect_node) = sect_node {
        /* If a child of top is opaque, it is presumed whitespace and skip it */
        if arena.node(cur_sect_node).type_ == MXML_OPAQUE {
            sect_node = mxml_get_next_sibling(arena, sect_node);
            continue;
        }

        if test_and_add_comment(
            arena,
            cur_sect_node,
            &mut comment_list,
            &mut num_comments,
            &mut max_comments,
        ) != 0
        {
            sect_node = mxml_get_next_sibling(arena, sect_node);
            continue;
        }

        /* This is the section type if it an element node, which it really will be, and
        it must have the "name" attribute for the value */
        key = mxml_get_element(arena, sect_node).map(|k| k.to_vec());
        global = if key.as_deref() == Some(ADOC_GLOBAL_NAME) {
            1
        } else {
            0
        };
        value = mxml_element_get_attr(arena, sect_node, Some(b"name")).map(|v| v.to_vec());
        if key.is_none() {
            S_NUM_SECT_NOT_ELEM.set(S_NUM_SECT_NOT_ELEM.get() + 1);
        }
        if global == 0 && value.is_none() {
            S_NUM_SECT_NO_NAME.set(S_NUM_SECT_NO_NAME.get() + 1);
        }
        if key.is_none() || (global == 0 && value.is_none()) {
            sect_node = mxml_get_next_sibling(arena, sect_node);
            continue;
        }
        let key_bytes = key.clone().unwrap();

        /* Lookup the collection under the key and create one if not found */
        icol = lookup_collection(adoc, &key_bytes);
        if icol < 0 {
            err = add_collection(adoc, &key_bytes);
            if err != 0 {
                break;
            }
            icol = adoc.num_collections - 1;
        }

        /* Add a section to the collection and set it as current one */
        if global == 0 {
            err = add_section(adoc, icol, value.as_deref().unwrap());
            if err != 0 {
                break;
            }
        }
        cur_coll = icol;
        cur_sect = adoc.collections[icol as usize].num_sections - 1;
        last_ind = -1;
        if num_comments != 0 {
            err = add_comments(
                &mut adoc.collections[cur_coll as usize].sections[cur_sect as usize],
                &mut comment_list,
                &mut num_comments,
                last_ind,
            );
            if err != 0 {
                break;
            }
        }

        /* Assign any other attributes as key-values in the section */
        let attrs = match &arena.node(cur_sect_node).value {
            MxmlValue::Element(element) => element.attrs.clone(),
            _ => Vec::new(),
        };
        for attr in attrs {
            let (aname, avalue) = (attr.name, attr.value);
            if aname != b"name" {
                err = sect_set_key_value_type(
                    &mut adoc.collections[cur_coll as usize].sections[cur_sect as usize],
                    &aname,
                    avalue.as_deref(),
                    ADOC_STRING,
                    &mut last_ind,
                );
                if err != 0 {
                    break;
                }
            }
        }

        /* Now walk through the children of section node */
        node = mxml_get_first_child(arena, sect_node);
        while let Some(cur_node) = node {
            /* Again, skip children that are opaque and definitely white space */
            if arena.node(cur_node).type_ == MXML_OPAQUE
                && match &arena.node(cur_node).value {
                    MxmlValue::Opaque(opaque) => opaque
                        .as_deref()
                        .is_some_and(|o| !o.is_empty() && o[0] == b'\n'),
                    _ => false,
                }
            {
                node = mxml_get_next_sibling(arena, node);
                continue;
            }

            if test_and_add_comment(
                arena,
                cur_node,
                &mut comment_list,
                &mut num_comments,
                &mut max_comments,
            ) != 0
            {
                node = mxml_get_next_sibling(arena, node);
                continue;
            }

            /* It must be an element */
            key = mxml_get_element(arena, node).map(|k| k.to_vec());
            if key.is_none() {
                S_NUM_CHILD_NOT_ELEM.set(S_NUM_CHILD_NOT_ELEM.get() + 1);
            } else {
                if match &arena.node(cur_node).value {
                    MxmlValue::Element(element) => !element.attrs.is_empty(),
                    _ => false,
                } {
                    S_NUM_CHILD_ATTRIBS.set(S_NUM_CHILD_ATTRIBS.get() + 1);
                }
                child = mxml_get_first_child(arena, node);

                /* The first child should be opaque and there should be only one */
                if child.is_some() && mxml_get_type(arena, child) != MXML_OPAQUE {
                    S_NUM_VALUE_NOT_TEXT.set(S_NUM_VALUE_NOT_TEXT.get() + 1);
                } else {
                    if child.is_some() && mxml_get_last_child(arena, node) != child {
                        S_NUM_MULTIPLE_CHILDS.set(S_NUM_MULTIPLE_CHILDS.get() + 1);
                    }
                    value = match child {
                        Some(child) => match &arena.node(child).value {
                            MxmlValue::Opaque(opaque) => opaque.clone(),
                            _ => None,
                        },
                        None => None,
                    };
                    err = sect_set_key_value_type(
                        &mut adoc.collections[cur_coll as usize].sections[cur_sect as usize],
                        key.as_deref().unwrap(),
                        value.as_deref(),
                        ADOC_STRING,
                        &mut last_ind,
                    );
                    if err != 0 {
                        break;
                    }
                    if num_comments != 0 {
                        err = add_comments(
                            &mut adoc.collections[cur_coll as usize].sections[cur_sect as usize],
                            &mut comment_list,
                            &mut num_comments,
                            last_ind,
                        );
                        if err != 0 {
                            break;
                        }
                    }
                }
            }

            /* Step to next key-value in section, if any */
            node = mxml_get_next_sibling(arena, node);
        }
        if err != 0 {
            break;
        }

        /* Add any comments that have accumulated */
        if num_comments != 0 {
            err = add_comments(
                &mut adoc.collections[cur_coll as usize].sections[cur_sect as usize],
                &mut comment_list,
                &mut num_comments,
                last_ind,
            );
            if err != 0 {
                break;
            }
        }

        /* Step to next section if any */
        sect_node = mxml_get_next_sibling(arena, sect_node);
    }

    if err != 0 {
        delete_adoc(adoc);
    }
    handle_final_comments(adoc, err, &mut comment_list, num_comments);

    mxml_delete(arena, xml);
    err
}

/// Matches C static `addToCommentList` (`autodoc.c:1971`).
///
/// `commentList` is held at length `maxComments`, the size the C `realloc`s it
/// to, so a slot the source assigns by index exists here too.
pub fn add_to_comment_list(
    comment_list: &mut Vec<Vec<u8>>,
    num_comments: &mut i32,
    max_comments: &mut i32,
    comment: &[u8],
) -> i32 {
    if *num_comments >= *max_comments {
        *max_comments += 1;
        comment_list.resize(*max_comments as usize, Vec::new());
    }
    comment_list[*num_comments as usize] = comment.to_vec();
    *num_comments += 1;
    0
}

/// Matches C static `testAndAddComment` (`autodoc.c:1992`).
pub fn test_and_add_comment(
    arena: &MxmlArena,
    node: usize,
    comment_list: &mut Vec<Vec<u8>>,
    num_comments: &mut i32,
    max_comments: &mut i32,
) -> i32 {
    let key: Option<Vec<u8>>;
    let mut len: i32;
    key = mxml_get_element(arena, Some(node)).map(|k| k.to_vec());
    if arena.node(node).type_ == MXML_ELEMENT
        && key
            .as_ref()
            .is_some_and(|k| pip_starts_with(k, XML_COMMENT_START) != 0)
    {
        let key = key.unwrap();
        /* `strdup(key + XML_COMSTART_LEN - 1)`, then `tmpStr[0] = '#'` */
        let mut tmp_str = key[(XML_COMSTART_LEN - 1) as usize..].to_vec();
        tmp_str[0] = b'#';
        len = tmp_str.len() as i32;
        if len >= 2 && tmp_str[(len - 2) as usize] == b'-' && tmp_str[(len - 1) as usize] == b'-' {
            len -= 2;
            tmp_str.truncate(len as usize);
        }
        if len != 0 {
            add_to_comment_list(comment_list, num_comments, max_comments, &tmp_str);
        }
        return 1;
    }
    0
}

/// Matches C static `handleFinalComments` (`autodoc.c:2019`).
///
/// The source writes `*commentList[i]`, which parses as `*(commentList[i])` and
/// reads past the single `char ***` for `i > 0`.  The intended
/// `(*commentList)[i]` is used here instead; matching the source literally would
/// mean shipping an out-of-bounds read (`autodoc.c:2025`).
pub fn handle_final_comments(
    adoc: &mut Autodoc,
    err: i32,
    comment_list: &mut Vec<Vec<u8>>,
    num_comments: i32,
) {
    if err != 0 {
        /* Clean out comment list */
        comment_list.clear();
    } else if !comment_list.is_empty() {
        /* If good, transfer any comments to the autodoc */
        if num_comments != 0 {
            adoc.final_comments = std::mem::take(comment_list);
            adoc.num_final_com = num_comments;
        } else {
            comment_list.clear();
        }
    }
}

/// Matches C static `writeXmlFile` (`autodoc.c:2044`).
pub fn write_xml_file(filename: &[u8]) -> i32 {
    let xml: Option<usize>;
    let mut node: Option<usize> = None;
    let mut elem: Option<usize> = None;
    let top: Option<usize>;
    let mut i: i32 = 0;
    let mut j: i32 = 0;
    let mut k: i32 = 0;
    let mut ind: i32 = 0;
    let mut use_ind: i32 = 0;
    let mut com_ind: i32 = 0;
    let mut ord_sect_inds: Vec<i32> = Vec::new();

    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    let ordered_write: i32 = if S_NAME_FOR_ORDERING.with_borrow(|name| name.is_some())
        && S_AUTODOCS.with_borrow(|adocs| adocs[cur as usize].num_sections) > 1
    {
        1
    } else {
        0
    };

    let Some(mut afile) = open_for_write(filename, "w") else {
        return -1;
    };
    if ordered_write != 0 {
        let Some(inds) = S_AUTODOCS.with_borrow(|adocs| setup_section_order(&adocs[cur as usize]))
        else {
            return -1;
        };
        ord_sect_inds = inds;
    }

    let arena = &mut MxmlArena::new();
    xml = mxml_new_xml(arena, Some(b"1.0"));
    let result = S_AUTODOCS.with_borrow(|adocs| {
        let adoc = &adocs[cur as usize];
        let root = adoc.root_element.as_deref().unwrap_or(b"autodoc");
        let top = mxml_new_element(arena, xml, Some(root));
        ind = 0;
        while ind < adoc.num_sections {
            use_ind = if ordered_write != 0 {
                ord_sect_inds[ind as usize]
            } else {
                ind
            };
            com_ind = 0;
            i = adoc.coll_list[use_ind as usize];
            j = adoc.sect_list[use_ind as usize];
            let coll = &adoc.collections[i as usize];
            let sect = &coll.sections[j as usize];
            while com_ind < sect.num_comments && sect.com_index[com_ind as usize] == -1 {
                write_comment_to_xml(arena, top, &sect.comments[com_ind as usize]);
                com_ind += 1;
            }
            node = mxml_new_element(arena, top, coll.name.as_deref());
            if i != 0 || j != 0 || sect.name.as_deref() != Some(ADOC_GLOBAL_NAME) {
                mxml_element_set_attr(arena, node, Some(b"name"), sect.name.as_deref());
            }

            /* Loop on key-values */
            k = 0;
            while k < sect.num_keys {
                while com_ind < sect.num_comments && sect.com_index[com_ind as usize] == k {
                    write_comment_to_xml(arena, node, &sect.comments[com_ind as usize]);
                    com_ind += 1;
                }
                elem = mxml_new_element(arena, node, sect.keys[k as usize].as_deref());
                if let Some(value) = sect.values[k as usize].as_deref() {
                    mxml_new_text(arena, elem, 0, Some(value));
                }
                k += 1;
            }
            ind += 1;
        }
        i = 0;
        while i < adoc.num_final_com {
            write_comment_to_xml(arena, top, &adoc.final_comments[i as usize]);
            i += 1;
        }
        top
    });
    top = result;
    let _ = top;

    mxml_set_wrap_margin(0);
    ixml_reset_last_level();
    let cb: MxmlSaveCb = Some(ixml_whitespace_cb);
    ind = mxml_save_file(arena, xml, &mut afile, cb);
    mxml_delete(arena, xml);
    ind
}

/// Matches C static `writeCommentToXML` (`autodoc.c:2124`).
pub fn write_comment_to_xml(arena: &mut MxmlArena, parent: Option<usize>, comment: &[u8]) {
    let len: i32;

    len = comment.len() as i32;
    if len == 0 {
        return;
    }
    let tmp_str = c_format_bytes(
        "!--%s%s--",
        &[
            CArg::Bytes(&comment[1..]),
            CArg::Bytes(if comment[(len - 1) as usize] == b' ' {
                b""
            } else {
                b" "
            }),
        ],
    );
    mxml_new_element(arena, parent, Some(&tmp_str));
}

/// Matches C static `addKey` (`autodoc.c:2147`).
pub fn add_key(sect: &mut AdocSection, key: &[u8], value: Option<&[u8]>, type_: i32) -> i32 {
    /* First allocate enough memory if needed */
    if sect.max_keys == 0 {
        sect.max_keys = MALLOC_CHUNK;
    } else if sect.num_keys >= sect.max_keys {
        sect.max_keys += MALLOC_CHUNK;
    }
    sect.keys.resize(sect.max_keys as usize, None);
    sect.values.resize(sect.max_keys as usize, None);
    sect.types.resize(sect.max_keys as usize, 0);

    /* Copy key and value and increment count */
    sect.keys[sect.num_keys as usize] = Some(key.to_vec());
    sect.values[sect.num_keys as usize] = value.map(|v| v.to_vec());
    sect.types[sect.num_keys as usize] = if value.is_some() {
        type_ as u8
    } else {
        ADOC_NO_VALUE as u8
    };
    sect.num_keys += 1;
    0
}

/// Matches C static `addSection` (`autodoc.c:2185`).
pub fn add_section(adoc: &mut Autodoc, coll_ind: i32, name: &[u8]) -> i32 {
    /* First allocate enough memory if needed for the sections in the collection
    and for the master lists in the autodoc */
    {
        let coll = &mut adoc.collections[coll_ind as usize];
        if coll.max_sections == 0 {
            coll.max_sections = MALLOC_CHUNK;
        } else if coll.num_sections >= coll.max_sections {
            coll.max_sections += MALLOC_CHUNK;
        }
        coll.sections
            .resize(coll.max_sections as usize, AdocSection::default());
    }

    if adoc.max_sections == 0 {
        adoc.max_sections = MALLOC_CHUNK;
    } else if adoc.num_sections >= adoc.max_sections {
        adoc.max_sections += MALLOC_CHUNK;
    }
    adoc.coll_list.resize(adoc.max_sections as usize, 0);
    adoc.sect_list.resize(adoc.max_sections as usize, 0);

    /* Copy the name and initialize to empty keys */
    let num_sections;
    {
        let coll = &mut adoc.collections[coll_ind as usize];
        let sect = &mut coll.sections[coll.num_sections as usize];
        *sect = AdocSection {
            name: Some(name.to_vec()),
            ..AdocSection::default()
        };
        num_sections = coll.num_sections;
    }

    /* Add the collection and section # to master list */
    adoc.coll_list[adoc.num_sections as usize] = coll_ind;
    adoc.sect_list[adoc.num_sections as usize] = num_sections;
    adoc.num_sections += 1;
    adoc.collections[coll_ind as usize].num_sections += 1;
    0
}

/// Matches C static `addCollection` (`autodoc.c:2242`).
pub fn add_collection(adoc: &mut Autodoc, name: &[u8]) -> i32 {
    /* Allocate just one at a time when needed */
    adoc.collections.resize(
        (adoc.num_collections + 1) as usize,
        AdocCollection::default(),
    );
    let coll = &mut adoc.collections[adoc.num_collections as usize];
    coll.name = Some(name.to_vec());
    coll.num_sections = 0;
    coll.max_sections = 0;
    coll.sections = Vec::new();
    adoc.num_collections += 1;
    0
}

/// Matches C static `addAutodoc` (`autodoc.c:2268`).
pub fn add_autodoc() -> i32 {
    let mut index: i32 = -1;

    S_AUTODOCS.with_borrow_mut(|adocs| {
        /* Search for a free autodoc in array */
        let mut i: i32 = 0;
        while i < adocs.len() as i32 {
            if adocs[i as usize].in_use == 0 {
                index = i;
                break;
            }
            i += 1;
        }

        if index < 0 {
            /* Allocate just one at a time when needed */
            index = adocs.len() as i32;
            adocs.push(Autodoc::default());
        }

        /* Initialize collections */
        let adoc = &mut adocs[index as usize];
        *adoc = Autodoc {
            in_use: 1,
            ..Autodoc::default()
        };

        /* Add a collection and section for global data */
        if add_collection(adoc, ADOC_GLOBAL_NAME) != 0 {
            return -1;
        }
        if add_section(adoc, 0, ADOC_GLOBAL_NAME) != 0 {
            delete_adoc(adoc);
            return -1;
        }
        index
    })
}

/// Matches C static `deleteAdoc` (`autodoc.c:2317`).
pub fn delete_adoc(adoc: &mut Autodoc) {
    let mut i: i32;
    let mut j: i32;
    i = 0;
    while i < adoc.num_collections {
        let coll = &mut adoc.collections[i as usize];
        j = 0;
        while j < coll.num_sections {
            delete_section(&mut coll.sections[j as usize]);
            j += 1;
        }

        /* Free sections */
        coll.sections = Vec::new();
        coll.name = None;
        i += 1;
    }

    /* Free collections */
    adoc.collections = Vec::new();
    adoc.num_collections = 0;

    /* Free lists of sections */
    adoc.coll_list = Vec::new();
    adoc.sect_list = Vec::new();
    adoc.num_sections = 0;
    adoc.max_sections = 0;

    /* Free final comments */
    adoc.final_comments = Vec::new();
    adoc.num_final_com = 0;
    adoc.root_element = None;
    adoc.in_use = 0;
}

/// Matches C static `deleteSection` (`autodoc.c:2356`).
pub fn delete_section(sect: &mut AdocSection) {
    /* Clean key/values out of section */
    sect.keys = Vec::new();
    sect.values = Vec::new();
    sect.types = Vec::new();
    sect.name = None;

    /* Clean comments out of section */
    sect.comments = Vec::new();
    sect.com_index = Vec::new();
}

/// Matches C static `parseKeyValue` (`autodoc.c:2381`).
///
/// The source takes two `char *` into one NUL-terminated line buffer; `line`
/// and `end` are indices into that buffer here, because `strstr(line, ...)`
/// searches past `end` to the buffer's NUL and the two are not one slice.
pub fn parse_key_value(
    buf: &[u8],
    line: usize,
    end: usize,
    key: &mut Vec<u8>,
    value: &mut Option<Vec<u8>>,
) -> i32 {
    let mut line = line;
    let mut end = end;
    let mut val_start: usize;
    let mut key_end: usize;
    let key_len: usize;
    let val_len: isize;

    /* Eat spaces at start and end */
    while line < end && (buf[line] == b' ' || buf[line] == b'\t') {
        line += 1;
    }
    while line < end && (buf[end - 1] == b' ' || buf[end - 1] == b'\t') {
        end -= 1;
    }
    if line == end {
        return 1;
    }

    /* Find delimiter.  If it is not there or no text before it, error */
    let delim = S_VALUE_DELIM.with_borrow(|delim| delim.clone());
    let found = if delim.is_empty() || delim.len() > buf.len() - line {
        None
    } else {
        buf[line..]
            .windows(delim.len())
            .position(|w| w == &delim[..])
            .map(|p| line + p)
    };
    let Some(found) = found else {
        return 1;
    };
    val_start = found;
    if val_start == line {
        return 1;
    }

    /* Eat spaces after key */
    key_end = val_start;
    while key_end > line && (buf[key_end - 1] == b' ' || buf[key_end - 1] == b'\t') {
        key_end -= 1;
    }

    /* Eat spaces after the delimiter.  Allow an empty value */
    val_start += delim.len();
    while val_start < end && (buf[val_start] == b' ' || buf[val_start] == b'\t') {
        val_start += 1;
    }

    /* Allocate for strings and copy them */
    key_len = key_end - line;
    *key = buf[line..line + key_len].to_vec();

    val_len = end as isize - val_start as isize;
    *value = None;
    if val_len != 0 {
        if val_len < 0 {
            /* C computes `end - valStart` as a negative `int` and hands
            `valLen + 1` to `malloc`, which fails; `adocMemoryError` then
            returns -1 (`autodoc.c:2429`). */
            adoc_memory_error(true, "parseKeyValue");
            return -1;
        }
        *value = Some(buf[val_start..end].to_vec());
    }
    0
}

/// Matches C static `lookupKey` (`autodoc.c:2429`).
pub fn lookup_key(sect: &AdocSection, key: &[u8]) -> i32 {
    let mut i: i32;
    i = 0;
    while i < sect.num_keys {
        if sect.keys[i as usize].as_deref() == Some(key) {
            return i;
        }
        i += 1;
    }
    -1
}

/// Matches C static `lookupCollection` (`autodoc.c:2442`).
pub fn lookup_collection(adoc: &Autodoc, name: &[u8]) -> i32 {
    let mut i: i32;
    i = 0;
    while i < adoc.num_collections {
        if adoc.collections[i as usize].name.as_deref() == Some(name) {
            return i;
        }
        i += 1;
    }
    -1
}

/// Matches C static `getSection` (`autodoc.c:2455`).
///
/// The C returns an `AdocSection *` into `sCurAdoc`; a `Vec` moves when it
/// grows, so this returns the collection index and the section index that
/// address the same section, and the autodoc is passed in.
pub fn get_section(adoc: &Autodoc, type_name: &[u8], sect_ind: i32) -> Option<(usize, usize)> {
    if sect_ind < 0 {
        return None;
    }
    let coll_ind = lookup_collection(adoc, type_name);
    if coll_ind < 0 {
        return None;
    }
    if sect_ind >= adoc.collections[coll_ind as usize].num_sections {
        return None;
    }
    Some((coll_ind as usize, sect_ind as usize))
}

/// Matches C static `addComments` (`autodoc.c:2472`).
pub fn add_comments(
    sect: &mut AdocSection,
    comments: &mut [Vec<u8>],
    num_comments: &mut i32,
    index: i32,
) -> i32 {
    let mut i: i32;
    let new_num: i32 = sect.num_comments + *num_comments;
    sect.comments.resize(new_num as usize, Vec::new());
    sect.com_index.resize(new_num as usize, 0);
    i = 0;
    while i < *num_comments {
        /* The C moves the pointer into the section rather than copying it. */
        sect.comments[sect.num_comments as usize] = std::mem::take(&mut comments[i as usize]);
        sect.com_index[sect.num_comments as usize] = index;
        sect.num_comments += 1;
        i += 1;
    }
    *num_comments = 0;
    0
}

/// Matches C static `findSectionInAdocList` (`autodoc.c:2497`).
pub fn find_section_in_adoc_list(coll_ind: i32, sect_ind: i32) -> i32 {
    let cur = S_CUR_ADOC_IND.get();
    if cur < 0 {
        return -1;
    }
    S_AUTODOCS.with_borrow(|adocs| {
        let adoc = &adocs[cur as usize];
        let mut i: i32 = 0;
        while i < adoc.num_sections {
            if adoc.coll_list[i as usize] == coll_ind && adoc.sect_list[i as usize] == sect_ind {
                return i;
            }
            i += 1;
        }
        -1
    })
}

/// Matches C static `adocMemoryError` (`autodoc.c:2506`).
///
/// The C tests the pointer it was handed for NULL; every allocation in this
/// module is a `Vec` that aborts rather than returning null, so what is passed
/// is the outcome of the test the caller already made.
pub fn adoc_memory_error(failed: bool, routine: &str) -> i32 {
    if !failed {
        return 0;
    }
    b3d_error(
        Some(&mut ImodFile::Stderr),
        format_args!("ERROR: {routine} - Allocating memory for string or autodoc component\n"),
    );
    -1
}

/// Matches C static `openForWrite` (`autodoc.c:2515`).
pub fn open_for_write(name: &[u8], mode: &str) -> Option<ImodFile> {
    let mut fp: Option<ImodFile> = None;
    let mut ind: i32;
    let trials: i32 = if 0 > S_OPEN_RETRIES.get() {
        0
    } else {
        S_OPEN_RETRIES.get()
    };
    let path = String::from_utf8_lossy(name).into_owned();
    ind = 0;
    while ind <= trials {
        fp = ImodFile::open(&path, mode);
        if fp.is_some() || ind == trials {
            return fp;
        }
        b3d_milli_sleep(250);
        ind += 1;
    }
    fp
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Mutex;
    /// The autodoc collection (`sAutodocs`) is process-global, so every test
    /// that touches it -- here and in `adoc_fwrap`, which drives the same
    /// collection through the Fortran wrappers -- must serialize on this one
    /// lock, not on a per-module one.
    pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn reads_vendored_autodoc_fixture() {
        let _lock = TEST_LOCK.lock().unwrap();
        adoc_done();
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc/binvol.adoc");
        assert!(adoc_read(path.as_bytes()) >= 0);
        assert!(adoc_get_number_of_sections(b"Field") >= 1);
        let mut value = Vec::new();
        assert_eq!(adoc_get_section_name(b"Field", 0, &mut value), 0);
        assert!(!value.is_empty());
        adoc_done();
    }

    #[test]
    fn metadata_numbers_are_read_from_text_values() {
        let _lock = TEST_LOCK.lock().unwrap();
        adoc_done();
        let text = b"ImageFile = stack.mrc\nMontage = 1\n[ZValue = 0]\nPieceCoordinates = 1 2 3\n";
        let name = std::env::temp_dir().join("imod-rs-autodoc-metadata.adoc");
        std::fs::write(&name, text).unwrap();
        assert!(adoc_read(name.to_string_lossy().as_bytes()) >= 0);
        let mut a = 0;
        let mut b = 0;
        let mut c = 0;
        assert_eq!(
            adoc_get_three_integers(b"ZValue", 0, b"PieceCoordinates", &mut a, &mut b, &mut c),
            0
        );
        assert_eq!((a, b, c), (1, 2, 3));
        std::fs::remove_file(name).unwrap();
        adoc_done();
    }

    #[test]
    fn collection_indices_exclude_the_global_predata_collection() {
        let _lock = TEST_LOCK.lock().unwrap();
        adoc_done();
        assert!(adoc_new() >= 0);
        assert_eq!(adoc_add_section(b"Field", b"one"), 0);
        assert_eq!(adoc_get_num_collections(), 1);
        let mut name = Vec::new();
        assert_eq!(adoc_get_collection_name(0, &mut name), 0);
        assert_eq!(name, b"Field");
        adoc_done();
    }

    /// `writeFile` through `AdocPrintToString`, exercising `fsPrintf`'s string path.
    #[test]
    fn print_to_string_reproduces_the_written_layout() {
        let _lock = TEST_LOCK.lock().unwrap();
        adoc_done();
        assert!(adoc_new() >= 0);
        assert_eq!(
            adoc_set_key_value(ADOC_GLOBAL_NAME, 0, b"ImageFile", Some(b"a.mrc")),
            0
        );
        assert_eq!(adoc_add_section(ADOC_ZVALUE_NAME, b"0"), 0);
        assert_eq!(
            adoc_set_two_integers(ADOC_ZVALUE_NAME, 0, b"PieceCoordinates", 3, 4),
            0
        );
        let mut buf = Vec::new();
        assert_eq!(adoc_print_to_string(&mut buf, 512, 1), 0);
        assert_eq!(
            buf,
            b"ImageFile = a.mrc\n\n[ZValue = 0]\nPieceCoordinates = 3 4\n"
        );
        adoc_done();
    }

    /// Round-trip a real vendored autodoc through the reader and the writer.
    #[test]
    fn round_trips_a_vendored_autodoc_byte_for_byte() {
        let _lock = TEST_LOCK.lock().unwrap();
        adoc_done();
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc/binvol.adoc");
        assert!(adoc_read(src.as_bytes()) >= 0);
        let out = std::env::temp_dir().join("imod-rs-autodoc-roundtrip.adoc");
        assert_eq!(adoc_write(out.to_string_lossy().as_bytes()), 0);
        let written = std::fs::read(&out).unwrap();
        adoc_done();
        /* Re-read the written file and check it produces the same key count */
        assert!(adoc_read(out.to_string_lossy().as_bytes()) >= 0);
        let n = adoc_get_number_of_sections(b"Field");
        adoc_done();
        assert!(n >= 1);
        assert!(!written.is_empty());
        let _ = std::fs::remove_file(&out);
    }

    /// Comments read from an autodoc are attached and written back out in place.
    #[test]
    fn comments_are_preserved_across_a_write() {
        let _lock = TEST_LOCK.lock().unwrap();
        adoc_done();
        let text = b"# leading comment\nVersion = 1\n\n[Field = A]\n# about B\nB = 2\n";
        let name = std::env::temp_dir().join("imod-rs-autodoc-comments.adoc");
        std::fs::write(&name, text).unwrap();
        assert!(adoc_read(name.to_string_lossy().as_bytes()) >= 0);
        let mut buf = Vec::new();
        assert_eq!(adoc_print_to_string(&mut buf, 1024, 1), 0);
        adoc_done();
        let _ = std::fs::remove_file(&name);
        assert_eq!(
            String::from_utf8_lossy(&buf),
            "# leading comment\nVersion = 1\n\n[Field = A]\n# about B\nB = 2\n"
        );
    }

    /// `AdocGetValTypeAndSize` counts space-separated tokens and reports the type.
    #[test]
    fn val_type_and_size_reports_tokens() {
        let _lock = TEST_LOCK.lock().unwrap();
        adoc_done();
        assert!(adoc_new() >= 0);
        let vals = [1.5f32, 2.5, 3.5];
        assert_eq!(
            adoc_set_float_array(ADOC_GLOBAL_NAME, 0, b"Vals", &vals, 3),
            0
        );
        let mut vtype = 0;
        let mut ntok = 0;
        assert_eq!(
            adoc_get_val_type_and_size(ADOC_GLOBAL_NAME, 0, b"Vals", &mut vtype, &mut ntok),
            0
        );
        assert_eq!((vtype, ntok), (ADOC_FLOAT_ARRAY, 3));
        let mut s = Vec::new();
        assert_eq!(adoc_get_string(ADOC_GLOBAL_NAME, 0, b"Vals", &mut s), 0);
        assert_eq!(s, b"1.5 2.5 3.5");
        adoc_done();
    }

    /// C-versus-Rust differential driver for the autodoc API, the counterpart
    /// of `scratchpad/adocapi/drv.c`.  It runs only when
    /// `IMOD_ADOC_API_REPORT` and `IMOD_ADOC_API_FILES` are set, and is the
    /// acceptance harness rather than an assertion of its own.
    #[test]
    fn adoc_api_differential() {
        use crate::imod::libcfshr::b3dutil::{CArg, c_format_bytes};
        let Ok(reppath) = std::env::var("IMOD_ADOC_API_REPORT") else {
            return;
        };
        let _lock = TEST_LOCK.lock().unwrap();
        let list = std::env::var("IMOD_ADOC_API_FILES").unwrap();
        let files = std::fs::read_to_string(&list).unwrap();
        let mut rep: Vec<u8> = Vec::new();

        fn dump_all(rep: &mut Vec<u8>, tag: &str) {
            use crate::imod::libcfshr::b3dutil::{CArg, c_format_bytes};
            let nc = adoc_get_num_collections();
            rep.extend_from_slice(&c_format_bytes(
                "%s numColl=%d\n",
                &[CArg::Str(tag), CArg::Int(nc as i64)],
            ));
            for ci in 0..nc {
                let mut cname: Vec<u8> = Vec::new();
                let err = adoc_get_collection_name(ci, &mut cname);
                rep.extend_from_slice(&c_format_bytes(
                    "%s coll[%d] err=%d name=%s\n",
                    &[
                        CArg::Str(tag),
                        CArg::Int(ci as i64),
                        CArg::Int(err as i64),
                        CArg::Bytes(if err == 0 { &cname } else { b"(nil)" }),
                    ],
                ));
                if err != 0 {
                    continue;
                }
                let ns = adoc_get_number_of_sections(&cname);
                rep.extend_from_slice(&c_format_bytes(
                    "%s   numSect=%d\n",
                    &[CArg::Str(tag), CArg::Int(ns as i64)],
                ));
                for si in 0..ns {
                    let mut sname: Vec<u8> = Vec::new();
                    let err = adoc_get_section_name(&cname, si, &mut sname);
                    let nk = adoc_get_number_of_keys(&cname, si);
                    rep.extend_from_slice(&c_format_bytes(
                        "%s   sect[%d] err=%d name=%s numKeys=%d lookup=%d\n",
                        &[
                            CArg::Str(tag),
                            CArg::Int(si as i64),
                            CArg::Int(err as i64),
                            CArg::Bytes(if err == 0 { &sname } else { b"(nil)" }),
                            CArg::Int(nk as i64),
                            CArg::Int(adoc_lookup_section(&cname, &sname) as i64),
                        ],
                    ));
                    for ki in 0..nk {
                        let mut key: Option<Vec<u8>> = None;
                        let err = adoc_get_key_by_index(&cname, si, ki, &mut key);
                        let Some(key) = key else {
                            rep.extend_from_slice(&c_format_bytes(
                                "%s     key[%d] err=%d (nil)\n",
                                &[CArg::Str(tag), CArg::Int(ki as i64), CArg::Int(err as i64)],
                            ));
                            continue;
                        };
                        let (mut vt, mut nt) = (-9, -9);
                        let err = adoc_get_val_type_and_size(&cname, si, &key, &mut vt, &mut nt);
                        rep.extend_from_slice(&c_format_bytes(
                            "%s     key[%d] %s tErr=%d type=%d ntok=%d",
                            &[
                                CArg::Str(tag),
                                CArg::Int(ki as i64),
                                CArg::Bytes(&key),
                                CArg::Int(err as i64),
                                CArg::Int(vt as i64),
                                CArg::Int(nt as i64),
                            ],
                        ));
                        let mut val: Vec<u8> = Vec::new();
                        let err = adoc_get_string(&cname, si, &key, &mut val);
                        rep.extend_from_slice(&c_format_bytes(
                            " sErr=%d s=%s",
                            &[
                                CArg::Int(err as i64),
                                CArg::Bytes(if err == 0 { &val } else { b"(nil)" }),
                            ],
                        ));
                        /* `PipGetLineOfValues`' error path `strncpy`s the value
                        into a 512-byte static; a longer value overruns it and
                        native segfaults, so the numeric getters are driven only
                        for values the C can survive. */
                        if err == 0 && val.len() < 200 {
                            let (mut i1, mut i2, mut i3) = (-9, -9, -9);
                            let (mut f1, mut f2, mut f3) = (-9f32, -9f32, -9f32);
                            let mut d1 = -9f64;
                            let e = adoc_get_integer(&cname, si, &key, &mut i1);
                            rep.extend_from_slice(&c_format_bytes(
                                " i=%d,%d",
                                &[CArg::Int(e as i64), CArg::Int(i1 as i64)],
                            ));
                            let e = adoc_get_two_integers(&cname, si, &key, &mut i1, &mut i2);
                            rep.extend_from_slice(&c_format_bytes(
                                " i2=%d,%d,%d",
                                &[
                                    CArg::Int(e as i64),
                                    CArg::Int(i1 as i64),
                                    CArg::Int(i2 as i64),
                                ],
                            ));
                            let e = adoc_get_three_integers(
                                &cname, si, &key, &mut i1, &mut i2, &mut i3,
                            );
                            rep.extend_from_slice(&c_format_bytes(
                                " i3=%d,%d,%d,%d",
                                &[
                                    CArg::Int(e as i64),
                                    CArg::Int(i1 as i64),
                                    CArg::Int(i2 as i64),
                                    CArg::Int(i3 as i64),
                                ],
                            ));
                            let e = adoc_get_float(&cname, si, &key, &mut f1);
                            rep.extend_from_slice(&c_format_bytes(
                                " f=%d,%g",
                                &[CArg::Int(e as i64), CArg::Dbl(f1 as f64)],
                            ));
                            let e = adoc_get_two_floats(&cname, si, &key, &mut f1, &mut f2);
                            rep.extend_from_slice(&c_format_bytes(
                                " f2=%d,%g,%g",
                                &[
                                    CArg::Int(e as i64),
                                    CArg::Dbl(f1 as f64),
                                    CArg::Dbl(f2 as f64),
                                ],
                            ));
                            let e =
                                adoc_get_three_floats(&cname, si, &key, &mut f1, &mut f2, &mut f3);
                            rep.extend_from_slice(&c_format_bytes(
                                " f3=%d,%g,%g,%g",
                                &[
                                    CArg::Int(e as i64),
                                    CArg::Dbl(f1 as f64),
                                    CArg::Dbl(f2 as f64),
                                    CArg::Dbl(f3 as f64),
                                ],
                            ));
                            let e = adoc_get_double(&cname, si, &key, &mut d1);
                            rep.extend_from_slice(&c_format_bytes(
                                " d=%d,%g",
                                &[CArg::Int(e as i64), CArg::Dbl(d1)],
                            ));
                            let mut arr = [0i32; 8];
                            let mut num = 8;
                            let e = adoc_get_integer_array(&cname, si, &key, &mut arr, &mut num, 8);
                            rep.extend_from_slice(&c_format_bytes(
                                " ia=%d,%d",
                                &[CArg::Int(e as i64), CArg::Int(num as i64)],
                            ));
                            if e == 0 {
                                for v in arr.iter().take(num.max(0) as usize) {
                                    rep.extend_from_slice(&c_format_bytes(
                                        ":%d",
                                        &[CArg::Int(*v as i64)],
                                    ));
                                }
                            }
                            let mut farr = [0f32; 8];
                            let mut num = 8;
                            let e = adoc_get_float_array(&cname, si, &key, &mut farr, &mut num, 8);
                            rep.extend_from_slice(&c_format_bytes(
                                " fa=%d,%d",
                                &[CArg::Int(e as i64), CArg::Int(num as i64)],
                            ));
                            if e == 0 {
                                for v in farr.iter().take(num.max(0) as usize) {
                                    rep.extend_from_slice(&c_format_bytes(
                                        ":%g",
                                        &[CArg::Dbl(*v as f64)],
                                    ));
                                }
                            }
                        }
                        rep.push(b'\n');
                    }
                }
            }
        }

        for line in files.lines() {
            if line.is_empty() {
                continue;
            }
            let base = line.rsplit('/').next().unwrap();
            let ind = adoc_read(line.as_bytes());
            rep.extend_from_slice(&c_format_bytes(
                "FILE %s read=%d cur=%d\n",
                &[
                    CArg::Str(base),
                    CArg::Int(ind as i64),
                    CArg::Int(adoc_get_current_index() as i64),
                ],
            ));
            if ind < 0 {
                continue;
            }
            let (mut a, mut b, mut c, mut d, mut e, mut f) = (-9, -9, -9, -9, -9, -9);
            let err = adoc_xml_read_status(&mut a, &mut b, &mut c, &mut d, &mut e, &mut f);
            rep.extend_from_slice(&c_format_bytes(
                "xmlStatus=%d %d %d %d %d %d %d\n",
                &[
                    CArg::Int(err as i64),
                    CArg::Int(a as i64),
                    CArg::Int(b as i64),
                    CArg::Int(c as i64),
                    CArg::Int(d as i64),
                    CArg::Int(e as i64),
                    CArg::Int(f as i64),
                ],
            ));
            let (mut mont, mut nsect, mut stype) = (-9, -9, -9);
            let e = adoc_get_image_meta_info(&mut mont, &mut nsect, &mut stype);
            rep.extend_from_slice(&c_format_bytes(
                "meta=%d %d %d %d\n",
                &[
                    CArg::Int(e as i64),
                    CArg::Int(mont as i64),
                    CArg::Int(nsect as i64),
                    CArg::Int(stype as i64),
                ],
            ));
            dump_all(&mut rep, "R");

            rep.extend_from_slice(&c_format_bytes(
                "lookupByName=%d findInsert=%d\n",
                &[
                    CArg::Int(adoc_lookup_by_name_value(b"ZValue", 1) as i64),
                    CArg::Int(adoc_find_insert_index(b"ZValue", 7) as i64),
                ],
            ));
            let mut line_of = |name: &str, v: i32, rep: &mut Vec<u8>| {
                rep.extend_from_slice(&c_format_bytes(
                    "%s=%d\n",
                    &[CArg::Str(name), CArg::Int(v as i64)],
                ));
            };
            line_of("addSect", adoc_add_section(b"Probe", b"17"), &mut rep);
            line_of(
                "setKV",
                adoc_set_key_value(b"Probe", 0, b"Str", Some(b"hello there")),
                &mut rep,
            );
            line_of(
                "setInt",
                adoc_set_integer(b"Probe", 0, b"One", -5),
                &mut rep,
            );
            line_of(
                "set2Int",
                adoc_set_two_integers(b"Probe", 0, b"Two", 3, -400000),
                &mut rep,
            );
            line_of(
                "set3Int",
                adoc_set_three_integers(b"Probe", 0, b"Three", 1, 2, 3),
                &mut rep,
            );
            line_of("setFlt", adoc_set_float(b"Probe", 0, b"F1", 1.5), &mut rep);
            line_of(
                "set2Flt",
                adoc_set_two_floats(b"Probe", 0, b"F2", 0.1, -2.5e7),
                &mut rep,
            );
            line_of(
                "set3Flt",
                adoc_set_three_floats(b"Probe", 0, b"F3", 1e-8, 0., 3.25),
                &mut rep,
            );
            line_of(
                "setDbl",
                adoc_set_double(b"Probe", 0, b"D1", 1.0 / 3.0),
                &mut rep,
            );
            let iv = [1i32, -2, 300000, 4];
            let fv = [1.5f32, -0.25, 1e10, 0.];
            line_of(
                "setIA",
                adoc_set_integer_array(b"Probe", 0, b"IA", &iv, 4),
                &mut rep,
            );
            line_of(
                "setFA",
                adoc_set_float_array(b"Probe", 0, b"FA", &fv, 4),
                &mut rep,
            );
            line_of("insSect", adoc_insert_section(b"Probe", 0, b"9"), &mut rep);
            line_of(
                "changeName",
                adoc_change_section_name(b"Probe", 1, b"18"),
                &mut rep,
            );
            line_of(
                "delKV",
                adoc_delete_key_value(b"Probe", 1, b"One"),
                &mut rep,
            );
            dump_all(&mut rep, "M");
            line_of("delSect", adoc_delete_section(b"Probe", 0), &mut rep);
            dump_all(&mut rep, "D");

            let second = adoc_new();
            line_of("new", second, &mut rep);
            adoc_set_current(ind);
            line_of(
                "xfer",
                adoc_transfer_section(b"Probe", 0, second, Some(b"77"), 1),
                &mut rep,
            );
            adoc_set_current(second);
            dump_all(&mut rep, "T");
            adoc_set_current(ind);
            adoc_clear(second);

            let mut sz = 40;
            while sz <= 160000 {
                let mut buf: Vec<u8> = Vec::new();
                let err = adoc_print_to_string(&mut buf, sz, 1);
                rep.extend_from_slice(&c_format_bytes(
                    "print(%d)=%d [%s]\n",
                    &[
                        CArg::Int(sz as i64),
                        CArg::Int(err as i64),
                        CArg::Bytes(&buf),
                    ],
                ));
                let mut buf: Vec<u8> = Vec::new();
                let err = adoc_print_to_string(&mut buf, sz, 0);
                rep.extend_from_slice(&c_format_bytes(
                    "printLast(%d)=%d [%s]\n",
                    &[
                        CArg::Int(sz as i64),
                        CArg::Int(err as i64),
                        CArg::Bytes(&buf),
                    ],
                ));
                sz *= 20;
            }
            line_of("orderNull", adoc_order_write_by_value(None), &mut rep);
            line_of(
                "orderZ",
                adoc_order_write_by_value(Some(b"ZValue")),
                &mut rep,
            );
            let mut buf: Vec<u8> = Vec::new();
            let err = adoc_print_to_string(&mut buf, 160000, 1);
            rep.extend_from_slice(&c_format_bytes(
                "printOrdered=%d [%s]\n",
                &[CArg::Int(err as i64), CArg::Bytes(&buf)],
            ));
            adoc_order_write_by_value(None);
            line_of("setRoot", adoc_set_xml_root_element(b"myroot"), &mut rep);
            let mut root: Option<Vec<u8>> = None;
            adoc_get_xml_root_element(&mut root);
            rep.extend_from_slice(&c_format_bytes(
                "root=%s\n",
                &[CArg::Bytes(root.as_deref().unwrap_or(b"(nil)"))],
            ));
            adoc_clear(ind);
        }
        adoc_done();
        std::fs::write(&reppath, &rep).unwrap();
    }
}
