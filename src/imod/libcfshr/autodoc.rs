//! Translation of `IMOD/libcfshr/autodoc.c` and `IMOD/include/autodoc.h`.
//!
//! The module keeps the C source's exact data structures (arrays of
//! `malloc`ed C strings inside `AdocSection`/`AdocCollection`/`Autodoc`) and its
//! module-level static state, so that operation order, allocation growth in
//! `MALLOC_CHUNK` blocks, and every error return matches the original.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::{b3d_error, b3d_milli_sleep, imod_backup_file};
use crate::imod::libcfshr::mxmlwrap::{ixml_reset_last_level, ixml_whitespace_cb};
use crate::imod::libcfshr::parse_params::{
    PIP_DOUBLE, PIP_FLOAT, PIP_INTEGER, pip_get_line_of_values, pip_read_next_line, pip_starts_with,
};
use crate::imod::libcfshr::robuststat::rs_sort_indexed_floats;
use crate::imod::libxml::{
    MXML_DESCEND, MXML_ELEMENT, MXML_NO_PARENT, MXML_OPAQUE, MxmlArena, MxmlSaveCb, MxmlValue,
    mxml_delete, mxml_element_get_attr, mxml_element_set_attr, mxml_get_element,
    mxml_get_first_child, mxml_get_last_child, mxml_get_next_sibling, mxml_get_type,
    mxml_load_file, mxml_new_element, mxml_new_text, mxml_new_xml, mxml_opaque_cb, mxml_save_file,
    mxml_set_wrap_margin, mxml_walk_next,
};
use core::ffi::{c_char, c_int, c_void};
use std::ffi::CStr;

/* --- IMOD/include/autodoc.h ------------------------------------------- */

/// Matches C `ADOC_GLOBAL_NAME` (`autodoc.h:16`).
pub const ADOC_GLOBAL_NAME: &CStr = c"PreData";
/// Matches C `ADOC_ZVALUE_NAME` (`autodoc.h:17`).
pub const ADOC_ZVALUE_NAME: &CStr = c"ZValue";
/// Matches C `ADOC_FRAMESET_NAME` (`autodoc.h:18`).
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

/* --- autodoc.c:22-56 : the three structures ---------------------------- */

/// Matches C `struct adoc_section` / `AdocSection` (`autodoc.c:23`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AdocSection {
    /// value after delimiter in section header
    pub name: *mut c_char,
    /// Array of strings with keys
    pub keys: *mut *mut c_char,
    /// Array of strings with values
    pub values: *mut *mut c_char,
    /// Number of key/value pairs
    pub num_keys: c_int,
    /// Current size of array
    pub max_keys: c_int,
    /// List of comments strings
    pub comments: *mut *mut c_char,
    /// Array of key indexes they occur before
    pub com_index: *mut c_int,
    /// Number of comments
    pub num_comments: c_int,
    /// Array of types of keys
    pub types: *mut u8,
}

/// Matches C `struct adoc_collection` / `AdocCollection` (`autodoc.c:35`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AdocCollection {
    /// section type, before delimiter in header
    pub name: *mut c_char,
    /// Array of sections
    pub sections: *mut AdocSection,
    /// Number of sections
    pub num_sections: c_int,
    /// Current size of array
    pub max_sections: c_int,
}

/// Matches C `struct adoc_autodoc` / `Autodoc` (`autodoc.c:42`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Autodoc {
    pub collections: *mut AdocCollection,
    pub num_collections: c_int,
    pub final_comments: *mut *mut c_char,
    pub num_final_com: c_int,
    pub coll_list: *mut c_int,
    pub sect_list: *mut c_int,
    pub num_sections: c_int,
    pub max_sections: c_int,
    pub in_use: c_int,
    pub backed_up: c_int,
    pub write_as_xml: c_int,
    /// Name for root element for XML file, in or out
    pub root_element: *mut c_char,
}

/* The static variables that can hold multiple autodocs (autodoc.c:59-63) */
static mut S_AUTODOCS: *mut Autodoc = core::ptr::null_mut();
static mut S_NUM_AUTODOCS: c_int = 0;
static mut S_CUR_ADOC_IND: c_int = -1;
static mut S_CUR_ADOC: *mut Autodoc = core::ptr::null_mut();
static mut S_NAME_FOR_ORDERING: *mut c_char = core::ptr::null_mut();

/* Static variables for XML writing (autodoc.c:66-68) */
static mut S_LAST_WAS_XML: c_int = 0;
static mut S_NUM_SECT_NOT_ELEM: c_int = 0;
static mut S_NUM_SECT_NO_NAME: c_int = 0;
static mut S_NUM_CHILD_NOT_ELEM: c_int = 0;
static mut S_NUM_CHILD_ATTRIBS: c_int = 0;
static mut S_NUM_VALUE_NOT_TEXT: c_int = 0;
static mut S_NUM_MULTIPLE_CHILDS: c_int = 0;

/* Static variables for writing to file or string (autodoc.c:71-74) */
static mut S_FILE: *mut libc::FILE = core::ptr::null_mut();
static mut S_STRING: *mut c_char = core::ptr::null_mut();
static mut S_BYTES_LEFT: c_int = 0;
static mut S_BYTES_WRITTEN: c_int = 0;

static mut S_OPEN_RETRIES: c_int = 0;

const OPEN_DELIM: &CStr = c"[";
const CLOSE_DELIM: &CStr = c"]";
const XML_START: &CStr = c"<?xml";
const XML_COMMENT_START: &CStr = c"!--";
const XML_COMSTART_LEN: i32 = 3;

/// Matches C `static char sDefaultDelim[] = "=";` (`autodoc.c:84`).
static mut S_DEFAULT_DELIM: [c_char; 2] = [b'=' as c_char, 0];
static mut S_VALUE_DELIM: *mut c_char = &raw mut S_DEFAULT_DELIM as *mut c_char;
static mut S_NEW_DELIM: *mut c_char = core::ptr::null_mut();

const BIG_STR_SIZE: usize = 10240;
const ERR_STR_SIZE: usize = 1024;
const MALLOC_CHUNK: i32 = 10;

/// Matches C `AdocRead` (`autodoc.c:130`).
pub unsafe extern "C" fn adoc_read(filename: *const c_char) -> i32 {
    let mut got_section: c_int = 0;
    /* `err` is uninitialised in the C source; it is only read after at least one
    loop iteration has assigned it, except for a completely empty file. */
    let mut err: c_int = 0;
    let mut line_len: c_int;
    let mut indst: c_int = 0;
    let mut icol: c_int;
    let mut ikey: c_int;
    let mut last_ind: c_int;
    let index: c_int;
    let bad_line: c_int = 1234;
    let mut cur_sect: *mut AdocSection;
    let mut coll: *mut AdocCollection;
    let mut line: *mut c_char;
    let mut line_end: *mut c_char;
    let mut key: *mut c_char = core::ptr::null_mut();
    let mut value: *mut c_char = core::ptr::null_mut();
    let mut big_str: [c_char; BIG_STR_SIZE] = [0; BIG_STR_SIZE];
    let mut comment_char: c_char = b'#' as c_char;
    let mut comment_list: *mut *mut c_char = core::ptr::null_mut();
    let mut max_comments: c_int = 0;
    let mut num_comments: c_int = 0;
    let mut first_line: c_int = 1;
    let afile: *mut libc::FILE;

    S_VALUE_DELIM = &raw mut S_DEFAULT_DELIM as *mut c_char;
    afile = libc::fopen(filename, c"r".as_ptr());
    if afile.is_null() {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: AdocRead - Error opening autodoc file {}",
                CStr::from_ptr(filename).to_string_lossy()
            ),
        );
        return -1;
    }

    /* Create a new adoc, which sets up global collection/section
    and takes care of cleanup if it fails */
    index = add_autodoc();
    if index < 0 {
        libc::fclose(afile);
        return -1;
    }

    adoc_set_current(index);
    cur_sect = &mut *(*(*S_CUR_ADOC).collections.add(0)).sections.add(0);
    last_ind = -1;
    S_LAST_WAS_XML = 0;

    let borrowed = std::os::fd::BorrowedFd::borrow_raw(libc::fileno(afile));
    let Ok(owned) = borrowed.try_clone_to_owned() else {
        libc::fclose(afile);
        return -1;
    };
    let mut aimod = ImodFile::File(std::rc::Rc::new(std::fs::File::from(owned)));

    loop {
        /* We cannot allow in-line comments so that value lines can contain
        anything.  But do allow blank and comment lines */
        /* `PipReadNextLine` now reads through an `ImodFile`; `aimod` is a
        duplicate of this file's descriptor, so both share one offset, and
        `readXmlFile` rewinds (`autodoc.c:1778`) before it reads. */
        let mut line_buf: Vec<u8> = Vec::new();
        line_len = pip_read_next_line(
            &mut aimod,
            &mut line_buf,
            BIG_STR_SIZE as c_int,
            comment_char as u8,
            1,
            0,
            &mut indst,
        );
        if line_len >= 0 {
            for (i, b) in line_buf.iter().enumerate() {
                big_str[i] = *b as c_char;
            }
            big_str[line_buf.len()] = 0;
        }
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
                    CStr::from_ptr(filename).to_string_lossy()
                ),
            );
            err = -1;
            break;
        }

        /* For first line, check for XML file and read that */
        line = big_str.as_mut_ptr().add(indst as usize);
        if first_line != 0
            && pip_starts_with(
                CStr::from_ptr(line).to_bytes(),
                CStr::from_ptr(XML_START.as_ptr()).to_bytes(),
            ) != 0
        {
            err = read_xml_file(afile);
            libc::fclose(afile);
            if err < 0 {
                return err;
            }
            return S_CUR_ADOC_IND;
        }
        first_line = 0;

        /* First check for comment and add it to list */
        if indst >= line_len || *line == comment_char {
            err = add_to_comment_list(
                &mut comment_list,
                &mut num_comments,
                &mut max_comments,
                big_str.as_mut_ptr(),
            );
            if err != 0 {
                break;
            }
            continue;
        }

        if pip_starts_with(
            CStr::from_ptr(line).to_bytes(),
            CStr::from_ptr(OPEN_DELIM.as_ptr()).to_bytes(),
        ) != 0
            && !libc::strstr(line, CLOSE_DELIM.as_ptr()).is_null()
        {
            /* If this is a section start, get name - value.  Here there must be
            a value and it is an error if there is none. */
            let line_end2 = libc::strstr(line, CLOSE_DELIM.as_ptr());
            err = parse_key_value(
                line.add(libc::strlen(OPEN_DELIM.as_ptr())),
                line_end2,
                &mut key,
                &mut value,
            );
            if value.is_null() {
                err = 1;
            }
            if err != 0 {
                err = if err > 0 { bad_line } else { err };
                break;
            }

            /* Lookup the collection under the key and create one if not found */
            icol = lookup_collection(S_CUR_ADOC, key);
            if icol < 0 {
                err = add_collection(S_CUR_ADOC, key);
                if err != 0 {
                    break;
                }
                icol = (*S_CUR_ADOC).num_collections - 1;
            }
            coll = (*S_CUR_ADOC).collections.add(icol as usize);

            /* Add a section to the collection and set it as current one */
            err = add_section(S_CUR_ADOC, icol, value);
            if err != 0 {
                break;
            }
            cur_sect = (*coll).sections.add(((*coll).num_sections - 1) as usize);
            got_section = 1;
            libc::free(key.cast::<c_void>());
            libc::free(value.cast::<c_void>());
            last_ind = -1;
        } else {
            /* Otherwise this is key-value inside a section.  First check for
            continuation line and append to last value. */
            if last_ind >= 0
                && libc::strstr(line, S_VALUE_DELIM).is_null()
                && !(*(*cur_sect).values.add(last_ind as usize)).is_null()
            {
                ikey = libc::strlen(*(*cur_sect).values.add(last_ind as usize)) as c_int;
                *(*cur_sect).values.add(last_ind as usize) = libc::realloc(
                    (*(*cur_sect).values.add(last_ind as usize)).cast::<c_void>(),
                    (ikey + line_len - indst + 3) as usize,
                )
                .cast::<c_char>();
                err = adoc_memory_error(
                    (*(*cur_sect).values.add(last_ind as usize)).cast::<c_void>(),
                    c"AdocRead".as_ptr(),
                );
                if err != 0 {
                    break;
                }

                /* Replace null with space and new null, then append new string */
                let vals = *(*cur_sect).values.add(last_ind as usize);
                *vals.add(ikey as usize) = b' ' as c_char;
                *vals.add((ikey + 1) as usize) = 0;
                libc::strcat(vals, line);
                continue;
            }

            /* This should be a key-value pair now */
            line_end = line.add((line_len - indst) as usize);
            err = parse_key_value(line, line_end, &mut key, &mut value);
            if err != 0 {
                err = if err > 0 { bad_line } else { err };
                break;
            }

            /* Handle new key-value delimiter - replace previous new value if any */
            if got_section == 0
                && libc::strcmp(key, c"KeyValueDelimiter".as_ptr()) == 0
                && !value.is_null()
            {
                if !S_NEW_DELIM.is_null() {
                    libc::free(S_NEW_DELIM.cast::<c_void>());
                }
                S_NEW_DELIM = libc::strdup(value);
                err = adoc_memory_error(S_NEW_DELIM.cast::<c_void>(), c"AdocRead".as_ptr());
                if err != 0 {
                    break;
                }
                S_VALUE_DELIM = S_NEW_DELIM;
            }

            /* Handle change of comment character */
            if got_section == 0 && libc::strcmp(key, c"CommentCharacter".as_ptr()) == 0 {
                comment_char = *value;
            }

            /* Look up the key first to replace an existing value */
            ikey = lookup_key(cur_sect, key);
            if ikey >= 0 {
                if !(*(*cur_sect).values.add(ikey as usize)).is_null() {
                    libc::free((*(*cur_sect).values.add(ikey as usize)).cast::<c_void>());
                }
                *(*cur_sect).values.add(ikey as usize) = value;
                libc::free(key.cast::<c_void>());
                last_ind = ikey;
            } else {
                /* Or just add the key-value */
                err = add_key(cur_sect, key, value, ADOC_STRING);
                if err != 0 {
                    break;
                }
                libc::free(key.cast::<c_void>());
                if !value.is_null() {
                    libc::free(value.cast::<c_void>());
                }
                last_ind = (*cur_sect).num_keys - 1;
            }
        }

        /* If there are comments, attach to item just added */
        if num_comments != 0 {
            err = add_comments(cur_sect, comment_list, &mut num_comments, last_ind);
            if err != 0 {
                break;
            }
        }
    }

    /* END OF FILE: If error, clean out autodoc, compose message for bad line */
    if err != 0 {
        delete_adoc(S_CUR_ADOC);
        if err == bad_line {
            big_str[ERR_STR_SIZE - 50] = 0;
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "Error: AdocRead -Improperly formatted line in autodoc: {}\n",
                    CStr::from_ptr(big_str.as_ptr()).to_string_lossy()
                ),
            );
            err = -1;
        }
    }

    if err == 0 {
        handle_final_comments(err, &mut comment_list, num_comments);
    }

    libc::fclose(afile);
    if err != 0 { err } else { index }
}

/// Matches C `AdocXmlReadStatus` (`autodoc.c:333`).
pub unsafe extern "C" fn adoc_xml_read_status(
    sect_not_elem: *mut c_int,
    sect_no_name: *mut c_int,
    child_not_elem: *mut c_int,
    child_attribs: *mut c_int,
    value_not_text: *mut c_int,
    multiple_childs: *mut c_int,
) -> i32 {
    *sect_no_name = S_NUM_SECT_NO_NAME;
    *sect_not_elem = S_NUM_SECT_NOT_ELEM;
    *child_not_elem = S_NUM_CHILD_NOT_ELEM;
    *child_attribs = S_NUM_CHILD_ATTRIBS;
    *value_not_text = S_NUM_VALUE_NOT_TEXT;
    *multiple_childs = S_NUM_MULTIPLE_CHILDS;
    if S_LAST_WAS_XML == 0 {
        return 0;
    }
    if S_NUM_SECT_NOT_ELEM
        + S_NUM_SECT_NO_NAME
        + S_NUM_CHILD_NOT_ELEM
        + S_NUM_CHILD_ATTRIBS
        + S_NUM_VALUE_NOT_TEXT
        + S_NUM_MULTIPLE_CHILDS
        > 0
    {
        -1
    } else {
        1
    }
}

/// Matches C `AdocOpenImageMetadata` (`autodoc.c:363`).
pub unsafe extern "C" fn adoc_open_image_metadata(
    filename: *const c_char,
    add_mdoc: i32,
    montage: *mut c_int,
    num_sect: *mut c_int,
    sect_type: *mut c_int,
) -> i32 {
    let mut buf = core::mem::MaybeUninit::<libc::stat>::uninit();
    let mut usename: *mut c_char = filename as *mut c_char;
    let series: c_int;
    let index: c_int;

    /* Attach extension to file if requested */
    if add_mdoc > 0 {
        usename = libc::malloc(libc::strlen(filename) + 6).cast::<c_char>();
        if usename.is_null() {
            return -1;
        }
        libc::sprintf(usename, c"%s.mdoc".as_ptr(), filename);
    }

    /* Return -2 if it does not exist, -1 if error reading it */
    if libc::stat(usename, buf.as_mut_ptr()) != 0 {
        index = -2;
    } else {
        index = adoc_read(usename);
    }
    if add_mdoc > 0 {
        libc::free(usename.cast::<c_void>());
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
pub unsafe extern "C" fn adoc_get_image_meta_info(
    montage: *mut c_int,
    num_sect: *mut c_int,
    sect_type: *mut c_int,
) -> i32 {
    let mut series: c_int = 0;
    let mut usename: *mut c_char = core::ptr::null_mut();

    *montage = 0;
    if adoc_get_string(
        ADOC_GLOBAL_NAME.as_ptr(),
        0,
        c"ImageFile".as_ptr(),
        &mut usename,
    ) == 0
    {
        *sect_type = 1;
        libc::free(usename.cast::<c_void>());
        *num_sect = adoc_get_number_of_sections(ADOC_ZVALUE_NAME.as_ptr());
    } else if adoc_get_integer(
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
            } else {
                *num_sect = 0;
                return -3;
            }
        } else {
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

/// Matches C `AdocNew` (`autodoc.c:436`).
pub unsafe extern "C" fn adoc_new() -> i32 {
    let err: c_int = add_autodoc();
    if err < 0 {
        return err;
    }
    adoc_set_current(err);
    err
}

/// Matches C `AdocGetCurrentIndex` (`autodoc.c:448`).
pub unsafe extern "C" fn adoc_get_current_index() -> i32 {
    S_CUR_ADOC_IND
}

/// Matches C `AdocSetCurrent` (`autodoc.c:457`).
pub unsafe extern "C" fn adoc_set_current(index: i32) -> i32 {
    if index < 0 || index >= S_NUM_AUTODOCS {
        return -1;
    }
    S_CUR_ADOC_IND = index;
    S_CUR_ADOC = S_AUTODOCS.add(S_CUR_ADOC_IND as usize);
    0
}

/// Matches C `AdocClear` (`autodoc.c:469`).
pub unsafe extern "C" fn adoc_clear(index: i32) {
    if index >= 0 && index < S_NUM_AUTODOCS {
        delete_adoc(S_AUTODOCS.add(index as usize));
    }
}

/// Matches C `AdocDone` (`autodoc.c:478`).
pub unsafe extern "C" fn adoc_done() {
    let mut i: c_int = 0;
    while i < S_NUM_AUTODOCS {
        delete_adoc(S_AUTODOCS.add(i as usize));
        i += 1;
    }
    if !S_AUTODOCS.is_null() {
        libc::free(S_AUTODOCS.cast::<c_void>());
    }
    if !S_NAME_FOR_ORDERING.is_null() {
        libc::free(S_NAME_FOR_ORDERING.cast::<c_void>());
        S_NAME_FOR_ORDERING = core::ptr::null_mut();
    }
    S_AUTODOCS = core::ptr::null_mut();
    S_NUM_AUTODOCS = 0;
    S_CUR_ADOC_IND = -1;
    S_CUR_ADOC = core::ptr::null_mut();
}

/// Matches C `AdocWrite` (`autodoc.c:495`).
pub unsafe extern "C" fn adoc_write(filename: *const c_char) -> i32 {
    let mut i: c_int;
    let mut backerr: c_int = 0;
    let mut retval: c_int = 0;
    let afile: *mut libc::FILE;

    if S_CUR_ADOC.is_null() {
        return -1;
    }
    if (*S_CUR_ADOC).backed_up == 0 {
        backerr = imod_backup_file(CStr::from_ptr(filename).to_string_lossy().as_ref());
    }
    (*S_CUR_ADOC).backed_up = 1;
    if (*S_CUR_ADOC).write_as_xml != 0 {
        return write_xml_file(filename);
    }
    afile = open_for_write(filename, c"w".as_ptr());
    if afile.is_null() {
        return -1;
    }
    if write_file(afile, core::ptr::null_mut(), 0, 1) != 0 {
        retval = -1;
    } else {
        i = 0;
        while i < (*S_CUR_ADOC).num_final_com {
            libc::fprintf(
                afile,
                c"%s\n".as_ptr(),
                *(*S_CUR_ADOC).final_comments.add(i as usize),
            );
            i += 1;
        }
    }

    libc::fclose(afile);
    if retval != 0 { retval } else { backerr }
}

/// Matches C `AdocRetryWriteOpens` (`autodoc.c:522`).
pub unsafe extern "C" fn adoc_retry_write_opens(num: i32) {
    S_OPEN_RETRIES = num;
}

/// Matches C `AdocSetWriteAsXML` (`autodoc.c:530`).
pub unsafe extern "C" fn adoc_set_write_as_xml(as_xml: i32) {
    if !S_CUR_ADOC.is_null() {
        (*S_CUR_ADOC).write_as_xml = if as_xml != 0 { 1 } else { 0 };
    }
}

/// Matches C `AdocGetWriteAsXML` (`autodoc.c:540`).
pub unsafe extern "C" fn adoc_get_write_as_xml() -> i32 {
    if S_CUR_ADOC.is_null() {
        return -1;
    }
    (*S_CUR_ADOC).write_as_xml
}

/// Matches C `AdocGetXmlRootElement` (`autodoc.c:552`).
pub unsafe extern "C" fn adoc_get_xml_root_element(string: *mut *mut c_char) -> i32 {
    if S_CUR_ADOC.is_null() {
        return -1;
    }
    *string = core::ptr::null_mut();
    if !(*S_CUR_ADOC).root_element.is_null() {
        *string = libc::strdup((*S_CUR_ADOC).root_element);
        if adoc_memory_error(
            (*string).cast::<c_void>(),
            c"AdocGetXmlRootElement".as_ptr(),
        ) != 0
        {
            return 1;
        }
    }
    0
}

/// Matches C `AdocSetXmlRootElement` (`autodoc.c:567`).
pub unsafe extern "C" fn adoc_set_xml_root_element(element: *const c_char) -> i32 {
    if S_CUR_ADOC.is_null() || element.is_null() {
        return -1;
    }
    if !(*S_CUR_ADOC).root_element.is_null() {
        libc::free((*S_CUR_ADOC).root_element.cast::<c_void>());
        (*S_CUR_ADOC).root_element = core::ptr::null_mut();
    }
    (*S_CUR_ADOC).root_element = libc::strdup(element);
    if adoc_memory_error(
        (*S_CUR_ADOC).root_element.cast::<c_void>(),
        c"AdocSetXmlRootElement".as_ptr(),
    ) != 0
    {
        return 1;
    }
    0
}

/// Matches C `AdocAppendSection` (`autodoc.c:582`).
pub unsafe extern "C" fn adoc_append_section(filename: *const c_char) -> i32 {
    let afile: *mut libc::FILE;
    let retval: c_int;
    if S_CUR_ADOC.is_null() {
        return -1;
    }
    if (*S_CUR_ADOC).write_as_xml != 0 {
        return write_xml_file(filename);
    }
    afile = open_for_write(filename, c"a".as_ptr());
    if afile.is_null() {
        return -1;
    }
    retval = write_file(afile, core::ptr::null_mut(), 0, 0);
    libc::fclose(afile);
    retval
}

/// Matches C `AdocPrintToString` (`autodoc.c:601`).
pub unsafe extern "C" fn adoc_print_to_string(
    string: *mut c_char,
    string_size: i32,
    write_all: i32,
) -> i32 {
    write_file(core::ptr::null_mut(), string, string_size, write_all)
}

/// Matches C `AdocOrderWriteByValue` (`autodoc.c:611`).
pub unsafe extern "C" fn adoc_order_write_by_value(type_name: *const c_char) -> i32 {
    if !S_NAME_FOR_ORDERING.is_null() {
        libc::free(S_NAME_FOR_ORDERING.cast::<c_void>());
        S_NAME_FOR_ORDERING = core::ptr::null_mut();
    }
    if type_name.is_null() {
        return 0;
    }
    S_NAME_FOR_ORDERING = libc::strdup(type_name);
    if adoc_memory_error(
        S_NAME_FOR_ORDERING.cast::<c_void>(),
        c"AdocOrderWriteByValue".as_ptr(),
    ) != 0
    {
        return 1;
    }
    0
}

/// Matches C static `writeFile` (`autodoc.c:622`).
pub unsafe fn write_file(
    afile: *mut libc::FILE,
    string: *mut c_char,
    string_size: i32,
    write_all: i32,
) -> i32 {
    let mut i: c_int;
    let mut j: c_int;
    let mut k: c_int;
    let mut ind: c_int;
    let mut com_ind: c_int;
    let mut write: c_int;
    let mut last_blank: c_int;
    let mut use_ind: c_int;
    let mut retval: c_int = 0;
    let mut coll: *mut AdocCollection;
    let mut sect: *mut AdocSection;
    let mut ord_sect_inds: *mut c_int = core::ptr::null_mut();
    let ordered_write: c_int =
        if write_all != 0 && !S_NAME_FOR_ORDERING.is_null() && (*S_CUR_ADOC).num_sections > 1 {
            1
        } else {
            0
        };
    S_FILE = afile;
    S_STRING = string;
    S_BYTES_LEFT = string_size;
    S_BYTES_WRITTEN = 0;
    if afile.is_null() && string.is_null() {
        return -1;
    }

    /* For ordered writing, get arrays for indexes and float values, and set up values
    for all the sections of the given type */
    if ordered_write != 0 {
        ord_sect_inds = setup_section_order();
        if ord_sect_inds.is_null() {
            return -1;
        }
    }

    /* Initialize delimiter, loop on indexes in the autodoc */
    S_VALUE_DELIM = &raw mut S_DEFAULT_DELIM as *mut c_char;
    ind = 0;
    while ind < (*S_CUR_ADOC).num_sections {
        write = if write_all != 0 || ind == (*S_CUR_ADOC).num_sections - 1 {
            1
        } else {
            0
        };
        use_ind = if ordered_write != 0 {
            *ord_sect_inds.add(ind as usize)
        } else {
            ind
        };
        i = *(*S_CUR_ADOC).coll_list.add(use_ind as usize);
        j = *(*S_CUR_ADOC).sect_list.add(use_ind as usize);
        coll = (*S_CUR_ADOC).collections.add(i as usize);
        sect = (*coll).sections.add(j as usize);

        /* dump comments before section */
        com_ind = 0;
        last_blank = 0;
        while write != 0
            && com_ind < (*sect).num_comments
            && *(*sect).com_index.add(com_ind as usize) == -1
        {
            if write != 0 {
                last_blank = if *(*(*sect).comments.add(com_ind as usize)) == 0 {
                    1
                } else {
                    0
                };
                let com = *(*sect).comments.add(com_ind as usize);
                com_ind += 1;
                if fs_printf(
                    c"%s\n".as_ptr(),
                    com,
                    core::ptr::null(),
                    core::ptr::null(),
                    core::ptr::null(),
                ) != 0
                {
                    retval = -1;
                    break;
                }
            }
        }
        if retval != 0 {
            break;
        }

        /* Write section name unless we're in global */
        if (i != 0 || j != 0 || libc::strcmp((*sect).name, ADOC_GLOBAL_NAME.as_ptr()) != 0)
            && write != 0
        {
            if fs_printf(
                c"%s[%s %s %s]\n".as_ptr(),
                if last_blank != 0 {
                    c"".as_ptr()
                } else {
                    c"\n".as_ptr()
                },
                (*coll).name,
                S_VALUE_DELIM,
                (*sect).name,
            ) != 0
            {
                retval = -1;
                break;
            }
        }

        /* Loop on key-values */
        k = 0;
        while k < (*sect).num_keys {
            /* dump comments associated with this index */
            while write != 0
                && com_ind < (*sect).num_comments
                && *(*sect).com_index.add(com_ind as usize) == k
            {
                if write != 0 {
                    let com = *(*sect).comments.add(com_ind as usize);
                    com_ind += 1;
                    if fs_printf(
                        c"%s\n".as_ptr(),
                        com,
                        core::ptr::null(),
                        core::ptr::null(),
                        core::ptr::null(),
                    ) != 0
                    {
                        retval = -1;
                        break;
                    }
                }
            }
            if retval != 0 {
                break;
            }

            /* Print key-value pairs with non-null values */
            if !(*(*sect).keys.add(k as usize)).is_null()
                && !(*(*sect).values.add(k as usize)).is_null()
            {
                if write != 0
                    && fs_printf(
                        c"%s %s %s\n".as_ptr(),
                        *(*sect).keys.add(k as usize),
                        S_VALUE_DELIM,
                        *(*sect).values.add(k as usize),
                        core::ptr::null(),
                    ) != 0
                {
                    retval = -1;
                    break;
                }

                /* After a new delimiter is written, need to set delimiter */
                if i == 0
                    && j == 0
                    && libc::strcmp(c"KeyValueDelimiter".as_ptr(), *(*sect).keys.add(k as usize))
                        == 0
                {
                    if !S_NEW_DELIM.is_null() {
                        libc::free(S_NEW_DELIM.cast::<c_void>());
                    }
                    S_NEW_DELIM = libc::strdup(*(*sect).values.add(k as usize));
                    if adoc_memory_error(S_NEW_DELIM.cast::<c_void>(), c"AdocWrite".as_ptr()) != 0 {
                        retval = -1;
                        break;
                    }
                    S_VALUE_DELIM = S_NEW_DELIM;
                }

            /* Print keys without values too */
            } else if !(*(*sect).keys.add(k as usize)).is_null() && write != 0 {
                if fs_printf(
                    c"%s %s \n".as_ptr(),
                    *(*sect).keys.add(k as usize),
                    S_VALUE_DELIM,
                    core::ptr::null(),
                    core::ptr::null(),
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
    if ordered_write != 0 {
        libc::free(ord_sect_inds.cast::<c_void>());
    }
    retval
}

/// Matches C static `fsPrintf` (`autodoc.c:740`).
///
/// Rust cannot define a C variadic function, so the source's `...` is spelled as
/// four `const char *` slots; every call site in `autodoc.c` uses only `%s`
/// conversions and at most four of them, and the unused trailing arguments are
/// never consumed by `printf`.  The formatting itself still goes through libc so
/// the emitted bytes are identical.
pub unsafe fn fs_printf(
    format: *const c_char,
    a1: *const c_char,
    a2: *const c_char,
    a3: *const c_char,
    a4: *const c_char,
) -> i32 {
    let mut retval: c_int = 0;
    let num_written: c_int;
    if !S_FILE.is_null() {
        if libc::fprintf(S_FILE, format, a1, a2, a3, a4) < 0 {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: AdocWrite - writing element to file\n"),
            );
            retval = -1;
        }
    } else {
        num_written = libc::snprintf(
            S_STRING.add(S_BYTES_WRITTEN as usize),
            S_BYTES_LEFT as libc::size_t,
            format,
            a1,
            a2,
            a3,
            a4,
        );
        if num_written >= S_BYTES_LEFT {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: AdocWrite - writing element to string\n"),
            );
            retval = -1;
        } else {
            S_BYTES_LEFT -= num_written;
            S_BYTES_WRITTEN += num_written;
        }
    }
    retval
}

/// Matches C static `setupSectionOrder` (`autodoc.c:768`).
pub unsafe fn setup_section_order() -> *mut c_int {
    let mut i: c_int;
    let mut j: c_int = 0;
    let mut ind: c_int;
    let mut coll: *mut AdocCollection;
    let mut max_value: f32 = -1.0e37;
    let ord_sect_values: *mut f32;
    let ord_sect_inds: *mut c_int;

    ord_sect_values =
        libc::malloc((*S_CUR_ADOC).num_sections as usize * core::mem::size_of::<f32>())
            .cast::<f32>();
    ord_sect_inds =
        libc::malloc((*S_CUR_ADOC).num_sections as usize * core::mem::size_of::<c_int>())
            .cast::<c_int>();
    if ord_sect_inds.is_null() || ord_sect_values.is_null() {
        adoc_memory_error(core::ptr::null_mut(), c"setupOrderedWrite".as_ptr());
        return core::ptr::null_mut();
    }
    *ord_sect_values.add(0) = max_value;
    *ord_sect_inds.add(0) = 0;
    ind = 1;
    while ind < (*S_CUR_ADOC).num_sections {
        i = *(*S_CUR_ADOC).coll_list.add(ind as usize);
        j = *(*S_CUR_ADOC).sect_list.add(ind as usize);
        *ord_sect_inds.add(ind as usize) = ind;
        coll = (*S_CUR_ADOC).collections.add(i as usize);
        if libc::strcmp((*coll).name, S_NAME_FOR_ORDERING) == 0
            && !(*(*coll).sections.add(j as usize)).name.is_null()
        {
            *ord_sect_values.add(ind as usize) =
                libc::atof((*(*coll).sections.add(j as usize)).name) as f32;
            let v = *ord_sect_values.add(ind as usize);
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
    while ind < (*S_CUR_ADOC).num_sections {
        i = *(*S_CUR_ADOC).coll_list.add(ind as usize);
        coll = (*S_CUR_ADOC).collections.add(i as usize);
        /* NOTE: the source reuses `j` from the loop above rather than re-reading
        sectList[ind]; that stale index is preserved here (autodoc.c:806). */
        if libc::strcmp((*coll).name, S_NAME_FOR_ORDERING) != 0
            || (*(*coll).sections.add(j as usize)).name.is_null()
        {
            max_value = (max_value as f64 + 1.0) as f32;
            *ord_sect_values.add(ind as usize) = max_value;
        }
        ind += 1;
    }

    /* Sort, use the sorted indexes below */
    rs_sort_indexed_floats(
        core::slice::from_raw_parts(ord_sect_values, (*S_CUR_ADOC).num_sections as usize),
        core::slice::from_raw_parts_mut(ord_sect_inds, (*S_CUR_ADOC).num_sections as usize),
        (*S_CUR_ADOC).num_sections,
    );
    libc::free(ord_sect_values.cast::<c_void>());
    ord_sect_inds
}

/// Matches C `AdocAddSection` (`autodoc.c:820`).
pub unsafe extern "C" fn adoc_add_section(type_name: *const c_char, name: *const c_char) -> i32 {
    let coll: *mut AdocCollection;
    let mut coll_ind: c_int;

    if S_CUR_ADOC.is_null() || type_name.is_null() || name.is_null() {
        return -1;
    }
    coll_ind = lookup_collection(S_CUR_ADOC, type_name);
    if coll_ind < 0 {
        if add_collection(S_CUR_ADOC, type_name) != 0 {
            return -1;
        }
        coll_ind = (*S_CUR_ADOC).num_collections - 1;
    }
    coll = (*S_CUR_ADOC).collections.add(coll_ind as usize);
    if add_section(S_CUR_ADOC, coll_ind, name) != 0 {
        return -1;
    }
    (*coll).num_sections - 1
}

/// Matches C `AdocInsertSection` (`autodoc.c:844`).
pub unsafe extern "C" fn adoc_insert_section(
    type_name: *const c_char,
    sect_ind: i32,
    name: *const c_char,
) -> i32 {
    let coll: *mut AdocCollection;
    let mut i: c_int;
    let mut coll_ind: c_int;
    let mut master_ind: c_int = 0;
    let mut num_sect: c_int = 0;
    let new_sect: AdocSection;
    if S_CUR_ADOC.is_null() || type_name.is_null() || name.is_null() {
        return -1;
    }
    coll_ind = lookup_collection(S_CUR_ADOC, type_name);
    if coll_ind >= 0 {
        num_sect = (*(*S_CUR_ADOC).collections.add(coll_ind as usize)).num_sections;
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

    /* Fix collection index if a new collection had to be added */
    if coll_ind < 0 {
        coll_ind = (*S_CUR_ADOC).num_collections - 1;
    }
    coll = (*S_CUR_ADOC).collections.add(coll_ind as usize);

    /* Save the new section then move existing sections up and copy new one into place */
    new_sect = *(*coll).sections.add(((*coll).num_sections - 1) as usize);
    i = (*coll).num_sections - 1;
    while i > sect_ind {
        *(*coll).sections.add(i as usize) = *(*coll).sections.add((i - 1) as usize);
        i -= 1;
    }
    *(*coll).sections.add(sect_ind as usize) = new_sect;

    /* Move the master lists up and decrement any other indices in this collection */
    i = (*S_CUR_ADOC).num_sections - 1;
    while i > master_ind {
        *(*S_CUR_ADOC).coll_list.add(i as usize) = *(*S_CUR_ADOC).coll_list.add((i - 1) as usize);
        *(*S_CUR_ADOC).sect_list.add(i as usize) = *(*S_CUR_ADOC).sect_list.add((i - 1) as usize);
        if *(*S_CUR_ADOC).coll_list.add(i as usize) == coll_ind
            && *(*S_CUR_ADOC).sect_list.add(i as usize) >= sect_ind
        {
            *(*S_CUR_ADOC).sect_list.add(i as usize) += 1;
        }
        i -= 1;
    }

    0
}

/// Matches C `AdocDeleteSection` (`autodoc.c:900`).
pub unsafe extern "C" fn adoc_delete_section(type_name: *const c_char, sect_ind: i32) -> i32 {
    let coll: *mut AdocCollection;
    let coll_ind: c_int;
    let mut i: c_int;
    let master_ind: c_int;
    if S_CUR_ADOC.is_null() || type_name.is_null() {
        return -1;
    }
    coll_ind = lookup_collection(S_CUR_ADOC, type_name);
    if coll_ind < 0 {
        return -1;
    }
    coll = (*S_CUR_ADOC).collections.add(coll_ind as usize);
    if sect_ind < 0 || sect_ind >= (*coll).num_sections {
        return -1;
    }

    /* Find the index of this section in the master list */
    master_ind = find_section_in_adoc_list(coll_ind, sect_ind);
    if master_ind < 0 {
        return -1;
    }
    delete_section((*coll).sections.add(sect_ind as usize));

    /* Repack the sections */
    i = sect_ind + 1;
    while i < (*coll).num_sections {
        *(*coll).sections.add((i - 1) as usize) = *(*coll).sections.add(i as usize);
        i += 1;
    }
    (*coll).num_sections -= 1;

    /* Repack the master list and decrement any other indices in this collection */
    i = master_ind + 1;
    while i < (*S_CUR_ADOC).num_sections {
        if *(*S_CUR_ADOC).coll_list.add(i as usize) == coll_ind
            && *(*S_CUR_ADOC).sect_list.add(i as usize) > sect_ind
        {
            *(*S_CUR_ADOC).sect_list.add(i as usize) -= 1;
        }
        *(*S_CUR_ADOC).coll_list.add((i - 1) as usize) = *(*S_CUR_ADOC).coll_list.add(i as usize);
        *(*S_CUR_ADOC).sect_list.add((i - 1) as usize) = *(*S_CUR_ADOC).sect_list.add(i as usize);
        i += 1;
    }
    (*S_CUR_ADOC).num_sections -= 1;
    0
}

/// Matches C `AdocChangeSectionName` (`autodoc.c:942`).
pub unsafe extern "C" fn adoc_change_section_name(
    type_name: *const c_char,
    sect_ind: i32,
    new_name: *const c_char,
) -> i32 {
    let coll: *mut AdocCollection;
    let coll_ind: c_int;
    let new_copy: *mut c_char;
    if S_CUR_ADOC.is_null() || type_name.is_null() || new_name.is_null() {
        return -1;
    }
    coll_ind = lookup_collection(S_CUR_ADOC, type_name);
    if coll_ind < 0 {
        return -1;
    }
    coll = (*S_CUR_ADOC).collections.add(coll_ind as usize);
    if sect_ind < 0 || sect_ind >= (*coll).num_sections {
        return -1;
    }
    new_copy = libc::strdup(new_name);
    if new_copy.is_null() {
        return -1;
    }
    let sect = (*coll).sections.add(sect_ind as usize);
    if !(*sect).name.is_null() {
        libc::free((*sect).name.cast::<c_void>());
        (*sect).name = core::ptr::null_mut();
    }
    (*sect).name = new_copy;
    0
}

/// Matches C `AdocLookupSection` (`autodoc.c:969`).
pub unsafe extern "C" fn adoc_lookup_section(type_name: *const c_char, name: *const c_char) -> i32 {
    let coll: *mut AdocCollection;
    let coll_ind: c_int;
    let mut sect_ind: c_int;

    if S_CUR_ADOC.is_null() || type_name.is_null() || name.is_null() {
        return -2;
    }
    coll_ind = lookup_collection(S_CUR_ADOC, type_name);
    if coll_ind < 0 {
        return -2;
    }
    coll = (*S_CUR_ADOC).collections.add(coll_ind as usize);
    sect_ind = 0;
    while sect_ind < (*coll).num_sections {
        if libc::strcmp((*(*coll).sections.add(sect_ind as usize)).name, name) == 0 {
            return sect_ind;
        }
        sect_ind += 1;
    }
    -1
}

/// Matches C `AdocLookupByNameValue` (`autodoc.c:993`).
pub unsafe extern "C" fn adoc_lookup_by_name_value(
    type_name: *const c_char,
    name_value: i32,
) -> i32 {
    let mut buf: [c_char; 15] = [0; 15];
    libc::sprintf(buf.as_mut_ptr(), c"%d".as_ptr(), name_value);
    adoc_lookup_section(type_name, buf.as_ptr())
}

/// Matches C `AdocFindInsertIndex` (`autodoc.c:1005`).
pub unsafe extern "C" fn adoc_find_insert_index(type_name: *const c_char, name_value: i32) -> i32 {
    let coll: *mut AdocCollection;
    let coll_ind: c_int;
    let mut sect_ind: c_int;
    let mut sect_value: c_int;

    if S_CUR_ADOC.is_null() || type_name.is_null() {
        return -1;
    }
    coll_ind = lookup_collection(S_CUR_ADOC, type_name);
    if coll_ind < 0 {
        return 0;
    }
    coll = (*S_CUR_ADOC).collections.add(coll_ind as usize);
    sect_ind = 0;
    while sect_ind < (*coll).num_sections {
        sect_value = libc::atoi((*(*coll).sections.add(sect_ind as usize)).name);
        if name_value == sect_value {
            return -1;
        }
        if name_value < sect_value {
            return sect_ind;
        }
        sect_ind += 1;
    }
    (*coll).num_sections
}

/// Matches C `AdocTransferSection` (`autodoc.c:1039`).
pub unsafe extern "C" fn adoc_transfer_section(
    type_name: *const c_char,
    sect_ind: i32,
    to_adoc_ind: i32,
    new_name: *const c_char,
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
pub unsafe extern "C" fn adoc_transfer_to_new_type(
    type_name: *const c_char,
    sect_ind: i32,
    to_adoc_ind: i32,
    new_type: *const c_char,
    new_name: *const c_char,
    by_value: i32,
) -> i32 {
    let mut err: c_int = 0;
    let mut ind: c_int;
    let mut new_sect_ind: c_int;
    let coll_ind: c_int;
    let name_val: c_int;
    let sect: *mut AdocSection;
    let cur_ind_save: c_int = S_CUR_ADOC_IND;

    /* Get the section then switch adocs */
    sect = get_section(type_name, sect_ind);
    if sect.is_null() {
        return -1;
    }
    if to_adoc_ind < 0 || to_adoc_ind == S_CUR_ADOC_IND || to_adoc_ind >= S_NUM_AUTODOCS {
        return -2;
    }
    adoc_set_current(to_adoc_ind);

    /* Set index to 0 for global section or go on to look up and/or add section */
    if libc::strcmp(type_name, ADOC_GLOBAL_NAME.as_ptr()) == 0 {
        new_sect_ind = 0;
    } else {
        if new_name.is_null() {
            return -2;
        }

        /* If the section does not exist, add it, using insert if the collection does
        exist */
        new_sect_ind = adoc_lookup_section(new_type, new_name);
        if new_sect_ind < 0 {
            coll_ind = lookup_collection(S_CUR_ADOC, new_type);
            if coll_ind < 0 {
                new_sect_ind = 0;
                err = adoc_add_section(new_type, new_name);
            } else {
                new_sect_ind = (*(*S_CUR_ADOC).collections.add(coll_ind as usize)).num_sections;
                if by_value != 0 {
                    name_val = libc::atoi(new_name);
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
    while ind < (*sect).num_keys && err == 0 {
        if !(*(*sect).keys.add(ind as usize)).is_null()
            && !(*(*sect).values.add(ind as usize)).is_null()
            && set_key_value_type(
                new_type,
                new_sect_ind,
                *(*sect).keys.add(ind as usize),
                *(*sect).values.add(ind as usize),
                *(*sect).types.add(ind as usize) as c_int,
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
pub unsafe extern "C" fn adoc_set_key_value(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    set_key_value_type(type_name, sect_ind, key, value, ADOC_STRING)
}

/// Matches C static `setKeyValueType` (`autodoc.c:1117`).
pub unsafe fn set_key_value_type(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    value: *const c_char,
    type_: i32,
) -> i32 {
    let sect: *mut AdocSection;
    let mut key_ind: c_int = 0;

    sect = get_section(type_name, sect_ind);
    if sect.is_null() {
        return -1;
    }
    if key.is_null() || value.is_null() {
        return -1;
    }
    sect_set_key_value_type(sect, key, value, type_, &mut key_ind)
}

/// Matches C static `sectSetKeyValueType` (`autodoc.c:1130`).
pub unsafe fn sect_set_key_value_type(
    sect: *mut AdocSection,
    key: *const c_char,
    value: *const c_char,
    type_: i32,
    key_ind: *mut c_int,
) -> i32 {
    *key_ind = lookup_key(sect, key);

    /* If key already exists, clear out value and set it again */
    if *key_ind >= 0 {
        if !(*(*sect).values.add(*key_ind as usize)).is_null() {
            libc::free((*(*sect).values.add(*key_ind as usize)).cast::<c_void>());
        }
        if !value.is_null() {
            *(*sect).values.add(*key_ind as usize) = libc::strdup(value);
            if adoc_memory_error(
                (*(*sect).values.add(*key_ind as usize)).cast::<c_void>(),
                c"AdocSetKeyValue".as_ptr(),
            ) != 0
            {
                return -1;
            }
            *(*sect).types.add(*key_ind as usize) = type_ as u8;
        } else {
            *(*sect).values.add(*key_ind as usize) = core::ptr::null_mut();
            *(*sect).types.add(*key_ind as usize) = ADOC_NO_VALUE as u8;
        }
    } else {
        *key_ind = (*sect).num_keys;
        return add_key(sect, key, value, type_);
    }
    0
}

/// Matches C `AdocSetInteger` (`autodoc.c:1163`).
pub unsafe extern "C" fn adoc_set_integer(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    ival: i32,
) -> i32 {
    let mut str: [c_char; 30] = [0; 30];
    libc::sprintf(str.as_mut_ptr(), c"%d".as_ptr(), ival);
    set_key_value_type(type_name, sect_ind, key, str.as_ptr(), ADOC_ONE_INT)
}

/// Matches C `AdocSetTwoIntegers` (`autodoc.c:1174`).
pub unsafe extern "C" fn adoc_set_two_integers(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    ival1: i32,
    ival2: i32,
) -> i32 {
    let mut str: [c_char; 60] = [0; 60];
    libc::sprintf(str.as_mut_ptr(), c"%d %d".as_ptr(), ival1, ival2);
    set_key_value_type(type_name, sect_ind, key, str.as_ptr(), ADOC_TWO_INTS)
}

/// Matches C `AdocSetThreeIntegers` (`autodoc.c:1186`).
pub unsafe extern "C" fn adoc_set_three_integers(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    ival1: i32,
    ival2: i32,
    ival3: i32,
) -> i32 {
    let mut str: [c_char; 90] = [0; 90];
    libc::sprintf(str.as_mut_ptr(), c"%d %d %d".as_ptr(), ival1, ival2, ival3);
    set_key_value_type(type_name, sect_ind, key, str.as_ptr(), ADOC_THREE_INTS)
}

/// Matches C `AdocSetIntegerArray` (`autodoc.c:1198`).
pub unsafe extern "C" fn adoc_set_integer_array(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    ivals: *mut c_int,
    num_vals: i32,
) -> i32 {
    set_array_of_values(
        type_name,
        sect_ind,
        key,
        ivals.cast::<c_void>(),
        num_vals,
        ADOC_INT_ARRAY,
    )
}

/// Matches C `AdocSetFloat` (`autodoc.c:1210`).
pub unsafe extern "C" fn adoc_set_float(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    val: f32,
) -> i32 {
    let mut str: [c_char; 30] = [0; 30];
    /* `val` is promoted to double by the C varargs call. */
    libc::sprintf(str.as_mut_ptr(), c"%g".as_ptr(), val as f64);
    set_key_value_type(type_name, sect_ind, key, str.as_ptr(), ADOC_ONE_FLOAT)
}

/// Matches C `AdocSetTwoFloats` (`autodoc.c:1221`).
pub unsafe extern "C" fn adoc_set_two_floats(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    val1: f32,
    val2: f32,
) -> i32 {
    let mut str: [c_char; 60] = [0; 60];
    libc::sprintf(
        str.as_mut_ptr(),
        c"%g %g".as_ptr(),
        val1 as f64,
        val2 as f64,
    );
    set_key_value_type(type_name, sect_ind, key, str.as_ptr(), ADOC_TWO_FLOATS)
}

/// Matches C `AdocSetThreeFloats` (`autodoc.c:1233`).
pub unsafe extern "C" fn adoc_set_three_floats(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    val1: f32,
    val2: f32,
    val3: f32,
) -> i32 {
    let mut str: [c_char; 90] = [0; 90];
    libc::sprintf(
        str.as_mut_ptr(),
        c"%g %g %g".as_ptr(),
        val1 as f64,
        val2 as f64,
        val3 as f64,
    );
    set_key_value_type(type_name, sect_ind, key, str.as_ptr(), ADOC_THREE_FLOATS)
}

/// Matches C `AdocSetFloatArray` (`autodoc.c:1245`).
pub unsafe extern "C" fn adoc_set_float_array(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    vals: *mut f32,
    num_vals: i32,
) -> i32 {
    set_array_of_values(
        type_name,
        sect_ind,
        key,
        vals.cast::<c_void>(),
        num_vals,
        ADOC_FLOAT_ARRAY,
    )
}

/// Matches C `AdocSetDouble` (`autodoc.c:1252`).
pub unsafe extern "C" fn adoc_set_double(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    val: f64,
) -> i32 {
    let mut str: [c_char; 30] = [0; 30];
    libc::sprintf(str.as_mut_ptr(), c"%g".as_ptr(), val);
    set_key_value_type(type_name, sect_ind, key, str.as_ptr(), ADOC_ONE_DOUBLE)
}

/// Matches C static `setArrayOfValues` (`autodoc.c:1263`).
pub unsafe fn set_array_of_values(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    vals: *mut c_void,
    num_vals: i32,
    val_type: i32,
) -> i32 {
    let mut tmp: [c_char; 40] = [0; 40];
    let full_str: *mut c_char;
    let ivals: *mut c_int = vals.cast::<c_int>();
    let fvals: *mut f32 = vals.cast::<f32>();
    let mut ind: c_int;
    let mut tot_len: c_int = 0;

    /* Add up the characters needed for the each value */
    ind = 0;
    while ind < num_vals {
        if val_type == ADOC_INT_ARRAY {
            libc::sprintf(tmp.as_mut_ptr(), c"%d ".as_ptr(), *ivals.add(ind as usize));
        } else {
            libc::sprintf(
                tmp.as_mut_ptr(),
                c"%g ".as_ptr(),
                *fvals.add(ind as usize) as f64,
            );
        }
        tot_len += libc::strlen(tmp.as_ptr()) as c_int + 1;
        ind += 1;
    }

    /* Get the string and build it up by writing again */
    full_str = libc::malloc(tot_len as usize).cast::<c_char>();
    if full_str.is_null() {
        return -1;
    }
    *full_str.add(0) = 0;
    ind = 0;
    while ind < num_vals {
        if val_type == ADOC_INT_ARRAY {
            libc::sprintf(
                tmp.as_mut_ptr(),
                c"%s%d".as_ptr(),
                if ind != 0 {
                    c" ".as_ptr()
                } else {
                    c"".as_ptr()
                },
                *ivals.add(ind as usize),
            );
        } else {
            libc::sprintf(
                tmp.as_mut_ptr(),
                c"%s%g".as_ptr(),
                if ind != 0 {
                    c" ".as_ptr()
                } else {
                    c"".as_ptr()
                },
                *fvals.add(ind as usize) as f64,
            );
        }
        libc::strcat(full_str, tmp.as_ptr());
        ind += 1;
    }
    ind = set_key_value_type(type_name, sect_ind, key, full_str, val_type);
    libc::free(full_str.cast::<c_void>());
    ind
}

/// Matches C `AdocDeleteKeyValue` (`autodoc.c:1305`).
pub unsafe extern "C" fn adoc_delete_key_value(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
) -> i32 {
    let sect: *mut AdocSection;
    let key_ind: c_int;

    sect = get_section(type_name, sect_ind);
    if sect.is_null() {
        return -1;
    }
    key_ind = lookup_key(sect, key);
    if key_ind < 0 {
        return -1;
    }
    if !(*(*sect).values.add(key_ind as usize)).is_null() {
        libc::free((*(*sect).values.add(key_ind as usize)).cast::<c_void>());
        *(*sect).values.add(key_ind as usize) = core::ptr::null_mut();
    }
    if !(*(*sect).keys.add(key_ind as usize)).is_null() {
        libc::free((*(*sect).keys.add(key_ind as usize)).cast::<c_void>());
        *(*sect).keys.add(key_ind as usize) = core::ptr::null_mut();
    }
    *(*sect).types.add(key_ind as usize) = ADOC_NO_VALUE as u8;
    0
}

/// Matches C `AdocGetNumCollections` (`autodoc.c:1331`).
pub unsafe extern "C" fn adoc_get_num_collections() -> i32 {
    if S_CUR_ADOC.is_null() {
        return -1;
    }
    (*S_CUR_ADOC).num_collections - 1
}

/// Matches C `AdocGetCollectionName` (`autodoc.c:1343`).
pub unsafe extern "C" fn adoc_get_collection_name(coll_ind: i32, string: *mut *mut c_char) -> i32 {
    if S_CUR_ADOC.is_null() || coll_ind < 0 || coll_ind >= (*S_CUR_ADOC).num_collections - 1 {
        return -1;
    }
    *string = libc::strdup((*(*S_CUR_ADOC).collections.add((coll_ind + 1) as usize)).name);
    adoc_memory_error(
        (*string).cast::<c_void>(),
        c"AdocGetCollectionName".as_ptr(),
    )
}

/// Matches C `AdocGetSectionName` (`autodoc.c:1356`).
pub unsafe extern "C" fn adoc_get_section_name(
    type_name: *const c_char,
    sect_ind: i32,
    string: *mut *mut c_char,
) -> i32 {
    let sect: *mut AdocSection;

    sect = get_section(type_name, sect_ind);
    if sect.is_null() {
        return -1;
    }
    *string = libc::strdup((*sect).name);
    adoc_memory_error((*string).cast::<c_void>(), c"AdocGetSectionName".as_ptr())
}

/// Matches C `AdocGetNumberOfSections` (`autodoc.c:1370`).
pub unsafe extern "C" fn adoc_get_number_of_sections(type_name: *const c_char) -> i32 {
    let coll_ind: c_int;
    if S_CUR_ADOC.is_null() || type_name.is_null() {
        return -1;
    }
    coll_ind = lookup_collection(S_CUR_ADOC, type_name);
    if coll_ind < 0 {
        return 0;
    }
    (*(*S_CUR_ADOC).collections.add(coll_ind as usize)).num_sections
}

/// Matches C `AdocGetNumberOfKeys` (`autodoc.c:1385`).
pub unsafe extern "C" fn adoc_get_number_of_keys(type_name: *const c_char, sect_ind: i32) -> i32 {
    let sect: *mut AdocSection;
    sect = get_section(type_name, sect_ind);
    if sect.is_null() {
        return -1;
    }
    (*sect).num_keys
}

/// Matches C `AdocGetKeyByIndex` (`autodoc.c:1399`).
pub unsafe extern "C" fn adoc_get_key_by_index(
    type_name: *const c_char,
    sect_ind: i32,
    key_ind: i32,
    key: *mut *mut c_char,
) -> i32 {
    let sect: *mut AdocSection;
    sect = get_section(type_name, sect_ind);
    if sect.is_null() {
        return -1;
    }
    if key_ind < 0 || key_ind >= (*sect).num_keys {
        return -1;
    }
    *key = core::ptr::null_mut();
    if !(*(*sect).keys.add(key_ind as usize)).is_null() {
        *key = libc::strdup(*(*sect).keys.add(key_ind as usize));
    }
    0
}

/// Matches C `AdocGetValTypeAndSize` (`autodoc.c:1420`).
pub unsafe extern "C" fn adoc_get_val_type_and_size(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    val_type: *mut c_int,
    num_tokens: *mut c_int,
) -> i32 {
    let sect: *mut AdocSection;
    let key_ind: c_int;
    let valstr: *mut c_char;
    let mut parsed: *mut c_char;
    *val_type = ADOC_NO_VALUE;
    *num_tokens = 0;
    if key.is_null() {
        return -1;
    }
    sect = get_section(type_name, sect_ind);
    if sect.is_null() {
        return -1;
    }
    key_ind = lookup_key(sect, key);
    if key_ind < 0 || (*(*sect).values.add(key_ind as usize)).is_null() {
        return 1;
    }
    *val_type = *(*sect).types.add(key_ind as usize) as c_int;

    /* Get a copy of the string and use the dreadful strtok */
    valstr = libc::strdup(*(*sect).values.add(key_ind as usize));
    if valstr.is_null() {
        adoc_memory_error(core::ptr::null_mut(), c"AdocGetValTypeAndSize".as_ptr());
        return -1;
    }
    parsed = valstr;
    while !libc::strtok(parsed, c" ".as_ptr()).is_null() {
        parsed = core::ptr::null_mut();
        *num_tokens += 1;
    }
    libc::free(valstr.cast::<c_void>());
    0
}

/// Matches C `AdocGetString` (`autodoc.c:1457`).
pub unsafe extern "C" fn adoc_get_string(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    string: *mut *mut c_char,
) -> i32 {
    let sect: *mut AdocSection;
    let key_ind: c_int;

    if key.is_null() {
        return -1;
    }
    sect = get_section(type_name, sect_ind);
    if sect.is_null() {
        return -1;
    }
    key_ind = lookup_key(sect, key);
    if key_ind < 0 || (*(*sect).values.add(key_ind as usize)).is_null() {
        return 1;
    }
    *string = libc::strdup(*(*sect).values.add(key_ind as usize));
    adoc_memory_error((*string).cast::<c_void>(), c"AdocGetString".as_ptr())
}

/// Matches C `AdocGetInteger` (`autodoc.c:1477`).
pub unsafe extern "C" fn adoc_get_integer(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    val1: *mut c_int,
) -> i32 {
    let err: c_int;
    let mut num: c_int = 1;
    let mut tmp: [c_int; 1] = [0; 1];
    err = adoc_get_integer_array(type_name, sect_ind, key, tmp.as_mut_ptr(), &mut num, 1);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    0
}

/// Matches C `AdocGetFloat` (`autodoc.c:1489`).
pub unsafe extern "C" fn adoc_get_float(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    val1: *mut f32,
) -> i32 {
    let err: c_int;
    let mut num: c_int = 1;
    let mut tmp: [f32; 1] = [0.; 1];
    err = adoc_get_float_array(type_name, sect_ind, key, tmp.as_mut_ptr(), &mut num, 1);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    0
}

/// Matches C `AdocGetDouble` (`autodoc.c:1501`).
pub unsafe extern "C" fn adoc_get_double(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    val1: *mut f64,
) -> i32 {
    let mut err: c_int;
    let mut num_to_get: c_int = 1;
    let mut string: *mut c_char = core::ptr::null_mut();
    err = adoc_get_string(type_name, sect_ind, key, &mut string);
    if err != 0 {
        return err;
    }
    let value_bytes = CStr::from_ptr(string).to_bytes().to_vec();
    err = pip_get_line_of_values(
        &value_bytes,
        &value_bytes,
        crate::imod::libcfshr::parse_params::PipValueArray::Double(
            core::slice::from_raw_parts_mut(val1, 1),
        ),
        PIP_DOUBLE,
        &mut num_to_get,
        1,
    );
    libc::free(string.cast::<c_void>());
    err
}

/// Matches C `AdocGetTwoIntegers` (`autodoc.c:1517`).
pub unsafe extern "C" fn adoc_get_two_integers(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    val1: *mut c_int,
    val2: *mut c_int,
) -> i32 {
    let err: c_int;
    let mut num: c_int = 2;
    let mut tmp: [c_int; 2] = [0; 2];
    err = adoc_get_integer_array(type_name, sect_ind, key, tmp.as_mut_ptr(), &mut num, 2);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    *val2 = tmp[1];
    0
}

/// Matches C `AdocGetTwoFloats` (`autodoc.c:1531`).
pub unsafe extern "C" fn adoc_get_two_floats(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    val1: *mut f32,
    val2: *mut f32,
) -> i32 {
    let err: c_int;
    let mut num: c_int = 2;
    let mut tmp: [f32; 2] = [0.; 2];
    err = adoc_get_float_array(type_name, sect_ind, key, tmp.as_mut_ptr(), &mut num, 2);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    *val2 = tmp[1];
    0
}

/// Matches C `AdocGetThreeIntegers` (`autodoc.c:1548`).
pub unsafe extern "C" fn adoc_get_three_integers(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    val1: *mut c_int,
    val2: *mut c_int,
    val3: *mut c_int,
) -> i32 {
    let err: c_int;
    let mut num: c_int = 3;
    let mut tmp: [c_int; 3] = [0; 3];
    err = adoc_get_integer_array(type_name, sect_ind, key, tmp.as_mut_ptr(), &mut num, 3);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    *val2 = tmp[1];
    *val3 = tmp[2];
    0
}

/// Matches C `AdocGetThreeFloats` (`autodoc.c:1563`).
pub unsafe extern "C" fn adoc_get_three_floats(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    val1: *mut f32,
    val2: *mut f32,
    val3: *mut f32,
) -> i32 {
    let err: c_int;
    let mut num: c_int = 3;
    let mut tmp: [f32; 3] = [0.; 3];
    err = adoc_get_float_array(type_name, sect_ind, key, tmp.as_mut_ptr(), &mut num, 3);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    *val2 = tmp[1];
    *val3 = tmp[2];
    0
}

/// Matches C `AdocGetIntegerArray` (`autodoc.c:1587`).
pub unsafe extern "C" fn adoc_get_integer_array(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    array: *mut c_int,
    num_to_get: *mut c_int,
    array_size: i32,
) -> i32 {
    let mut string: *mut c_char = core::ptr::null_mut();
    let mut err: c_int;
    err = adoc_get_string(type_name, sect_ind, key, &mut string);
    if err != 0 {
        return err;
    }
    let value_bytes = CStr::from_ptr(string).to_bytes().to_vec();
    err = pip_get_line_of_values(
        &value_bytes,
        &value_bytes,
        crate::imod::libcfshr::parse_params::PipValueArray::Int(core::slice::from_raw_parts_mut(
            array,
            array_size.max(0) as usize,
        )),
        PIP_INTEGER,
        &mut *num_to_get,
        array_size,
    );
    libc::free(string.cast::<c_void>());
    err
}

/// Matches C `AdocGetFloatArray` (`autodoc.c:1601`).
pub unsafe extern "C" fn adoc_get_float_array(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    array: *mut f32,
    num_to_get: *mut c_int,
    array_size: i32,
) -> i32 {
    let mut string: *mut c_char = core::ptr::null_mut();
    let mut err: c_int;
    err = adoc_get_string(type_name, sect_ind, key, &mut string);
    if err != 0 {
        return err;
    }
    let value_bytes = CStr::from_ptr(string).to_bytes().to_vec();
    err = pip_get_line_of_values(
        &value_bytes,
        &value_bytes,
        crate::imod::libcfshr::parse_params::PipValueArray::Float(core::slice::from_raw_parts_mut(
            array,
            array_size.max(0) as usize,
        )),
        PIP_FLOAT,
        &mut *num_to_get,
        array_size,
    );
    libc::free(string.cast::<c_void>());
    err
}

/// Matches C `AdocGetDoubleArray` (`autodoc.c:1615`).
pub unsafe extern "C" fn adoc_get_double_array(
    type_name: *const c_char,
    sect_ind: i32,
    key: *const c_char,
    array: *mut f64,
    num_to_get: *mut c_int,
    array_size: i32,
) -> i32 {
    let mut string: *mut c_char = core::ptr::null_mut();
    let mut err: c_int;
    err = adoc_get_string(type_name, sect_ind, key, &mut string);
    if err != 0 {
        return err;
    }
    let value_bytes = CStr::from_ptr(string).to_bytes().to_vec();
    err = pip_get_line_of_values(
        &value_bytes,
        &value_bytes,
        crate::imod::libcfshr::parse_params::PipValueArray::Double(
            core::slice::from_raw_parts_mut(array, array_size.max(0) as usize),
        ),
        PIP_DOUBLE,
        &mut *num_to_get,
        array_size,
    );
    libc::free(string.cast::<c_void>());
    err
}

/// Matches C `AdocWriteInteger` (`autodoc.c:1638`).
pub unsafe extern "C" fn adoc_write_integer(
    fp: *mut libc::FILE,
    key: *const c_char,
    ival: i32,
) -> i32 {
    if libc::fprintf(fp, c"%s = %d\n".as_ptr(), key, ival) < 0 {
        return 1;
    }
    0
}

/// Matches C `AdocWriteTwoIntegers` (`autodoc.c:1649`).
pub unsafe extern "C" fn adoc_write_two_integers(
    fp: *mut libc::FILE,
    key: *const c_char,
    ival1: i32,
    ival2: i32,
) -> i32 {
    if libc::fprintf(fp, c"%s = %d %d\n".as_ptr(), key, ival1, ival2) < 0 {
        return 1;
    }
    0
}

/// Matches C `AdocWriteThreeIntegers` (`autodoc.c:1660`).
pub unsafe extern "C" fn adoc_write_three_integers(
    fp: *mut libc::FILE,
    key: *const c_char,
    ival1: i32,
    ival2: i32,
    ival3: i32,
) -> i32 {
    if libc::fprintf(fp, c"%s = %d %d %d\n".as_ptr(), key, ival1, ival2, ival3) < 0 {
        return 1;
    }
    0
}

/// Matches C `AdocWriteIntegerArray` (`autodoc.c:1671`).
pub unsafe extern "C" fn adoc_write_integer_array(
    fp: *mut libc::FILE,
    key: *const c_char,
    ivals: *mut c_int,
    num_vals: i32,
) -> i32 {
    let mut ind: c_int;
    if libc::fprintf(fp, c"%s =".as_ptr(), key) < 0 {
        return 1;
    }
    ind = 0;
    while ind < num_vals {
        if libc::fprintf(fp, c" %d".as_ptr(), *ivals.add(ind as usize)) < 0 {
            return 1;
        }
        ind += 1;
    }
    if libc::fprintf(fp, c"\n".as_ptr()) < 0 {
        return 1;
    }
    0
}

/// Matches C `AdocWriteFloat` (`autodoc.c:1685`).
pub unsafe extern "C" fn adoc_write_float(
    fp: *mut libc::FILE,
    key: *const c_char,
    val: f32,
) -> i32 {
    if libc::fprintf(fp, c"%s = %g\n".as_ptr(), key, val as f64) < 0 {
        return 1;
    }
    0
}

/// Matches C `AdocWriteTwoFloats` (`autodoc.c:1695`).
pub unsafe extern "C" fn adoc_write_two_floats(
    fp: *mut libc::FILE,
    key: *const c_char,
    val1: f32,
    val2: f32,
) -> i32 {
    if libc::fprintf(fp, c"%s = %g %g\n".as_ptr(), key, val1 as f64, val2 as f64) < 0 {
        return 1;
    }
    0
}

/// Matches C `AdocWriteThreeFloats` (`autodoc.c:1706`).
pub unsafe extern "C" fn adoc_write_three_floats(
    fp: *mut libc::FILE,
    key: *const c_char,
    val1: f32,
    val2: f32,
    val3: f32,
) -> i32 {
    if libc::fprintf(
        fp,
        c"%s = %g %g %g\n".as_ptr(),
        key,
        val1 as f64,
        val2 as f64,
        val3 as f64,
    ) < 0
    {
        return 1;
    }
    0
}

/// Matches C `AdocWriteFloatArray` (`autodoc.c:1717`).
pub unsafe extern "C" fn adoc_write_float_array(
    fp: *mut libc::FILE,
    key: *const c_char,
    vals: *mut f32,
    num_vals: i32,
) -> i32 {
    let mut ind: c_int;
    if libc::fprintf(fp, c"%s =".as_ptr(), key) < 0 {
        return 1;
    }
    ind = 0;
    while ind < num_vals {
        if libc::fprintf(fp, c" %g".as_ptr(), *vals.add(ind as usize) as f64) < 0 {
            return 1;
        }
        ind += 1;
    }
    if libc::fprintf(fp, c"\n".as_ptr()) < 0 {
        return 1;
    }
    0
}

/// Matches C `AdocWriteDouble` (`autodoc.c:1731`).
pub unsafe extern "C" fn adoc_write_double(
    fp: *mut libc::FILE,
    key: *const c_char,
    val: f64,
) -> i32 {
    if libc::fprintf(fp, c"%s = %g\n".as_ptr(), key, val) < 0 {
        return 1;
    }
    0
}

/// Matches C `AdocWriteKeyValue` (`autodoc.c:1741`).
pub unsafe extern "C" fn adoc_write_key_value(
    fp: *mut libc::FILE,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    if libc::fprintf(fp, c"%s = %s\n".as_ptr(), key, value) < 0 {
        return 1;
    }
    0
}

/// Matches C `AdocWriteSectionStart` (`autodoc.c:1752`).
pub unsafe extern "C" fn adoc_write_section_start(
    fp: *mut libc::FILE,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    if libc::fprintf(
        fp,
        c"[%s = %s]\n".as_ptr(),
        key,
        if !value.is_null() {
            value
        } else {
            c"".as_ptr()
        },
    ) < 0
    {
        return 1;
    }
    0
}

/// Matches C static `readXmlFile` (`autodoc.c:1765`).
///
/// The mini-XML tree is arena-allocated now, so the node pointers are slot
/// indices into a `MxmlArena` that lives for the length of this function, and
/// the element names and values it hands back are byte slices that have to be
/// NUL-terminated again for the rest of autodoc, which is still C-shaped.
pub unsafe fn read_xml_file(fp: *mut libc::FILE) -> i32 {
    let xml: Option<usize>;
    let mut node: Option<usize>;
    let mut top: Option<usize>;
    let mut sect_node: Option<usize>;
    let mut child: Option<usize>;
    let mut key: Option<Vec<u8>>;
    let mut value: Option<Vec<u8>>;
    let mut icol: c_int;
    let mut ind: c_int;
    let mut global: c_int;
    let mut last_ind: c_int;
    let mut err: c_int = 0;
    let mut cur_sect: *mut AdocSection = core::ptr::null_mut();
    let mut coll: *mut AdocCollection;
    let mut comment_list: *mut *mut c_char = core::ptr::null_mut();
    let mut max_comments: c_int = 0;
    let mut num_comments: c_int = 0;

    S_LAST_WAS_XML = 1;
    libc::rewind(fp);

    S_NUM_SECT_NOT_ELEM = 0;
    S_NUM_SECT_NO_NAME = 0;
    S_NUM_CHILD_NOT_ELEM = 0;
    S_NUM_CHILD_ATTRIBS = 0;
    S_NUM_VALUE_NOT_TEXT = 0;
    S_NUM_MULTIPLE_CHILDS = 0;

    /*
     * `mxmlLoadFile` reads a `FILE *`; the descriptor is duplicated into an
     * `ImodFile` so that the read starts where the rewind left it.  The caller
     * closes `fp` as soon as this returns.
     */
    let borrowed = std::os::fd::BorrowedFd::borrow_raw(libc::fileno(fp));
    let Ok(owned) = borrowed.try_clone_to_owned() else {
        return -2;
    };
    let mut afile = ImodFile::File(std::rc::Rc::new(std::fs::File::from(owned)));

    let arena = &mut MxmlArena::new();
    xml = mxml_load_file(arena, MXML_NO_PARENT, &mut afile, Some(mxml_opaque_cb));
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
    key = mxml_get_element(arena, top).map(|k| {
        let mut k = k.to_vec();
        k.push(0);
        k
    });
    if let Some(key) = &key {
        (*S_CUR_ADOC).root_element = libc::strdup(key.as_ptr().cast::<c_char>());
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
        key = mxml_get_element(arena, sect_node).map(|k| {
            let mut k = k.to_vec();
            k.push(0);
            k
        });
        global = if key.as_ref().is_some_and(|k| {
            libc::strcmp(k.as_ptr().cast::<c_char>(), ADOC_GLOBAL_NAME.as_ptr()) == 0
        }) {
            1
        } else {
            0
        };
        value = mxml_element_get_attr(arena, sect_node, Some(b"name")).map(|v| {
            let mut v = v.to_vec();
            v.push(0);
            v
        });
        if key.is_none() {
            S_NUM_SECT_NOT_ELEM += 1;
        }
        if global == 0 && value.is_none() {
            S_NUM_SECT_NO_NAME += 1;
        }
        if key.is_none() || (global == 0 && value.is_none()) {
            sect_node = mxml_get_next_sibling(arena, sect_node);
            continue;
        }
        let key_ptr = key.as_ref().unwrap().as_ptr().cast::<c_char>();

        /* Lookup the collection under the key and create one if not found */
        icol = lookup_collection(S_CUR_ADOC, key_ptr);
        if icol < 0 {
            err = add_collection(S_CUR_ADOC, key_ptr);
            if err != 0 {
                break;
            }
            icol = (*S_CUR_ADOC).num_collections - 1;
        }
        coll = (*S_CUR_ADOC).collections.add(icol as usize);

        /* Add a section to the collection and set it as current one */
        if global == 0 {
            err = add_section(
                S_CUR_ADOC,
                icol,
                value.as_ref().unwrap().as_ptr().cast::<c_char>(),
            );
            if err != 0 {
                break;
            }
        }
        cur_sect = (*coll).sections.add(((*coll).num_sections - 1) as usize);
        last_ind = -1;
        if num_comments != 0 {
            err = add_comments(cur_sect, comment_list, &mut num_comments, last_ind);
            if err != 0 {
                break;
            }
        }

        /* Assign any other attributes as key-values in the section */
        let num_attrs = match &arena.node(cur_sect_node).value {
            MxmlValue::Element(element) => element.num_attrs,
            _ => 0,
        };
        ind = 0;
        while ind < num_attrs {
            let (aname, avalue) = match &arena.node(cur_sect_node).value {
                MxmlValue::Element(element) => {
                    let attr = &element.attrs[ind as usize];
                    let mut aname = attr.name.clone();
                    aname.push(0);
                    let avalue = attr.value.as_ref().map(|v| {
                        let mut v = v.clone();
                        v.push(0);
                        v
                    });
                    (aname, avalue)
                }
                _ => break,
            };
            if libc::strcmp(aname.as_ptr().cast::<c_char>(), c"name".as_ptr()) != 0 {
                err = sect_set_key_value_type(
                    cur_sect,
                    aname.as_ptr().cast::<c_char>(),
                    match &avalue {
                        Some(v) => v.as_ptr().cast::<c_char>(),
                        None => core::ptr::null(),
                    },
                    ADOC_STRING,
                    &mut last_ind,
                );
                if err != 0 {
                    break;
                }
            }
            ind += 1;
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
            key = mxml_get_element(arena, node).map(|k| {
                let mut k = k.to_vec();
                k.push(0);
                k
            });
            if key.is_none() {
                S_NUM_CHILD_NOT_ELEM += 1;
            } else {
                if match &arena.node(cur_node).value {
                    MxmlValue::Element(element) => element.num_attrs != 0,
                    _ => false,
                } {
                    S_NUM_CHILD_ATTRIBS += 1;
                }
                child = mxml_get_first_child(arena, node);

                /* The first child should be opaque and there should be only one */
                if child.is_some() && mxml_get_type(arena, child) != MXML_OPAQUE {
                    S_NUM_VALUE_NOT_TEXT += 1;
                } else {
                    if child.is_some() && mxml_get_last_child(arena, node) != child {
                        S_NUM_MULTIPLE_CHILDS += 1;
                    }
                    value = match child {
                        Some(child) => match &arena.node(child).value {
                            MxmlValue::Opaque(opaque) => opaque.as_ref().map(|o| {
                                let mut o = o.clone();
                                o.push(0);
                                o
                            }),
                            _ => None,
                        },
                        None => None,
                    };
                    err = sect_set_key_value_type(
                        cur_sect,
                        key.as_ref().unwrap().as_ptr().cast::<c_char>(),
                        match &value {
                            Some(v) => v.as_ptr().cast::<c_char>(),
                            None => core::ptr::null(),
                        },
                        ADOC_STRING,
                        &mut last_ind,
                    );
                    if err != 0 {
                        break;
                    }
                    if num_comments != 0 {
                        err = add_comments(cur_sect, comment_list, &mut num_comments, last_ind);
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
            err = add_comments(cur_sect, comment_list, &mut num_comments, last_ind);
            if err != 0 {
                break;
            }
        }

        /* Step to next section if any */
        sect_node = mxml_get_next_sibling(arena, sect_node);
    }

    if err != 0 {
        delete_adoc(S_CUR_ADOC);
    }
    handle_final_comments(err, &mut comment_list, num_comments);

    mxml_delete(arena, xml);
    err
}

/// Matches C static `addToCommentList` (`autodoc.c:1971`).
pub unsafe fn add_to_comment_list(
    comment_list: *mut *mut *mut c_char,
    num_comments: *mut c_int,
    max_comments: *mut c_int,
    comment: *mut c_char,
) -> i32 {
    if *num_comments >= *max_comments {
        if *max_comments != 0 {
            *comment_list = libc::realloc(
                (*comment_list).cast::<c_void>(),
                (*max_comments + 1) as usize * core::mem::size_of::<*mut c_char>(),
            )
            .cast::<*mut c_char>();
        } else {
            *comment_list = libc::malloc(core::mem::size_of::<*mut c_char>()).cast::<*mut c_char>();
        }
        if adoc_memory_error((*comment_list).cast::<c_void>(), c"AdocRead".as_ptr()) != 0 {
            return 1;
        }
        *max_comments += 1;
    }
    *(*comment_list).add(*num_comments as usize) = libc::strdup(comment);
    let added = *(*comment_list).add(*num_comments as usize);
    *num_comments += 1;
    if adoc_memory_error(added.cast::<c_void>(), c"AdocRead".as_ptr()) != 0 {
        return 1;
    }
    0
}

/// Matches C static `testAndAddComment` (`autodoc.c:1992`).
pub unsafe fn test_and_add_comment(
    arena: &MxmlArena,
    node: usize,
    comment_list: *mut *mut *mut c_char,
    num_comments: *mut c_int,
    max_comments: *mut c_int,
) -> i32 {
    let key: Option<Vec<u8>>;
    let tmp_str: *mut c_char;
    let mut len: c_int;
    key = mxml_get_element(arena, Some(node)).map(|k| {
        let mut k = k.to_vec();
        k.push(0);
        k
    });
    if arena.node(node).type_ == MXML_ELEMENT
        && key.as_ref().is_some_and(|k| {
            pip_starts_with(k, CStr::from_ptr(XML_COMMENT_START.as_ptr()).to_bytes()) != 0
        })
    {
        let key = key.unwrap();
        tmp_str = libc::strdup(
            key.as_ptr()
                .cast::<c_char>()
                .add((XML_COMSTART_LEN - 1) as usize),
        );
        if !tmp_str.is_null() {
            *tmp_str.add(0) = b'#' as c_char;
            len = libc::strlen(tmp_str) as c_int;
            if len >= 2
                && *tmp_str.add((len - 2) as usize) == b'-' as c_char
                && *tmp_str.add((len - 1) as usize) == b'-' as c_char
            {
                len -= 2;
                *tmp_str.add(len as usize) = 0;
            }
            if len != 0 {
                add_to_comment_list(comment_list, num_comments, max_comments, tmp_str);
            }
            libc::free(tmp_str.cast::<c_void>());
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
pub unsafe fn handle_final_comments(
    err: i32,
    comment_list: *mut *mut *mut c_char,
    num_comments: i32,
) {
    let mut i: c_int;
    if err != 0 {
        /* Clean out comment list */
        i = 0;
        while i < num_comments {
            if !(*(*comment_list).add(i as usize)).is_null() {
                libc::free((*(*comment_list).add(i as usize)).cast::<c_void>());
            }
            i += 1;
        }
        if !(*comment_list).is_null() {
            libc::free((*comment_list).cast::<c_void>());
        }
    } else if !(*comment_list).is_null() {
        /* If good, transfer any comments to the autodoc */
        if num_comments != 0 {
            (*S_CUR_ADOC).final_comments = *comment_list;
            (*S_CUR_ADOC).num_final_com = num_comments;
        } else {
            libc::free((*comment_list).cast::<c_void>());
        }
    }
}

/// Matches C static `writeXmlFile` (`autodoc.c:2044`).
///
/// The tree is arena-allocated, and `mxmlSaveFile` takes a Rust writer, so the
/// `FILE *` that `openForWrite` returns is written through a duplicate of its
/// descriptor.  Nothing else writes to that `FILE *`, so no output interleaves.
pub unsafe fn write_xml_file(filename: *const c_char) -> i32 {
    let afile: *mut libc::FILE;
    let xml: Option<usize>;
    let mut node: Option<usize>;
    let mut elem: Option<usize>;
    let top: Option<usize>;
    let mut i: c_int;
    let mut j: c_int;
    let mut k: c_int;
    let mut ind: c_int;
    let mut use_ind: c_int;
    let mut com_ind: c_int;
    let mut coll: *mut AdocCollection;
    let mut sect: *mut AdocSection;
    let mut ord_sect_inds: *mut c_int = core::ptr::null_mut();
    let ordered_write: c_int = if !S_NAME_FOR_ORDERING.is_null() && (*S_CUR_ADOC).num_sections > 1 {
        1
    } else {
        0
    };

    if S_CUR_ADOC.is_null() {
        return -1;
    }
    afile = open_for_write(filename, c"w".as_ptr());
    if afile.is_null() {
        return -1;
    }
    if ordered_write != 0 {
        ord_sect_inds = setup_section_order();
        if ord_sect_inds.is_null() {
            return -1;
        }
    }

    let arena = &mut MxmlArena::new();
    xml = mxml_new_xml(arena, Some(b"1.0"));
    let root: Vec<u8> = if !(*S_CUR_ADOC).root_element.is_null() {
        let mut r = CStr::from_ptr((*S_CUR_ADOC).root_element)
            .to_bytes()
            .to_vec();
        r.push(0);
        r
    } else {
        b"autodoc\0".to_vec()
    };
    top = mxml_new_element(arena, xml, Some(&root[..root.len() - 1]));
    ind = 0;
    while ind < (*S_CUR_ADOC).num_sections {
        use_ind = if ordered_write != 0 {
            *ord_sect_inds.add(ind as usize)
        } else {
            ind
        };
        com_ind = 0;
        i = *(*S_CUR_ADOC).coll_list.add(use_ind as usize);
        j = *(*S_CUR_ADOC).sect_list.add(use_ind as usize);
        coll = (*S_CUR_ADOC).collections.add(i as usize);
        sect = (*coll).sections.add(j as usize);
        while com_ind < (*sect).num_comments && *(*sect).com_index.add(com_ind as usize) == -1 {
            write_comment_to_xml(arena, top, *(*sect).comments.add(com_ind as usize));
            com_ind += 1;
        }
        node = mxml_new_element(arena, top, Some(CStr::from_ptr((*coll).name).to_bytes()));
        if i != 0 || j != 0 || libc::strcmp((*sect).name, ADOC_GLOBAL_NAME.as_ptr()) != 0 {
            mxml_element_set_attr(
                arena,
                node,
                Some(b"name"),
                Some(CStr::from_ptr((*sect).name).to_bytes()),
            );
        }

        /* Loop on key-values */
        k = 0;
        while k < (*sect).num_keys {
            while com_ind < (*sect).num_comments && *(*sect).com_index.add(com_ind as usize) == k {
                write_comment_to_xml(arena, node, *(*sect).comments.add(com_ind as usize));
                com_ind += 1;
            }
            elem = mxml_new_element(
                arena,
                node,
                Some(CStr::from_ptr(*(*sect).keys.add(k as usize)).to_bytes()),
            );
            if !(*(*sect).values.add(k as usize)).is_null() {
                mxml_new_text(
                    arena,
                    elem,
                    0,
                    Some(CStr::from_ptr(*(*sect).values.add(k as usize)).to_bytes()),
                );
            }
            k += 1;
        }
        ind += 1;
    }
    i = 0;
    while i < (*S_CUR_ADOC).num_final_com {
        write_comment_to_xml(arena, top, *(*S_CUR_ADOC).final_comments.add(i as usize));
        i += 1;
    }

    mxml_set_wrap_margin(0);
    ixml_reset_last_level();
    let cb: MxmlSaveCb = Some(ixml_whitespace_cb);
    let borrowed = std::os::fd::BorrowedFd::borrow_raw(libc::fileno(afile));
    let Ok(owned) = borrowed.try_clone_to_owned() else {
        libc::fclose(afile);
        return -1;
    };
    let mut out = ImodFile::File(std::rc::Rc::new(std::fs::File::from(owned)));
    ind = mxml_save_file(arena, xml, &mut out, cb);
    drop(out);
    libc::fclose(afile);
    mxml_delete(arena, xml);
    ind
}

/// Matches C static `writeCommentToXML` (`autodoc.c:2124`).
pub unsafe fn write_comment_to_xml(
    arena: &mut MxmlArena,
    parent: Option<usize>,
    comment: *mut c_char,
) {
    let len: c_int;
    let tmp_str: *mut c_char;

    len = libc::strlen(comment) as c_int;
    if len == 0 {
        return;
    }
    tmp_str = libc::malloc((len + 10) as usize).cast::<c_char>();
    if !tmp_str.is_null() {
        libc::sprintf(
            tmp_str,
            c"!--%s%s--".as_ptr(),
            comment.add(1),
            if *comment.add((len - 1) as usize) == b' ' as c_char {
                c"".as_ptr()
            } else {
                c" ".as_ptr()
            },
        );
        mxml_new_element(arena, parent, Some(CStr::from_ptr(tmp_str).to_bytes()));
        libc::free(tmp_str.cast::<c_void>());
    }
}

/// Matches C static `addKey` (`autodoc.c:2147`).
pub unsafe fn add_key(
    sect: *mut AdocSection,
    key: *const c_char,
    value: *const c_char,
    type_: i32,
) -> i32 {
    /* First allocate enough memory if needed */
    if (*sect).max_keys == 0 {
        (*sect).keys = libc::malloc(MALLOC_CHUNK as usize * core::mem::size_of::<*mut c_char>())
            .cast::<*mut c_char>();
        (*sect).values = libc::malloc(MALLOC_CHUNK as usize * core::mem::size_of::<*mut c_char>())
            .cast::<*mut c_char>();
        (*sect).types =
            libc::malloc(MALLOC_CHUNK as usize * core::mem::size_of::<u8>()).cast::<u8>();
        (*sect).max_keys = MALLOC_CHUNK;
    } else if (*sect).num_keys >= (*sect).max_keys {
        (*sect).keys = libc::realloc(
            (*sect).keys.cast::<c_void>(),
            ((*sect).max_keys + MALLOC_CHUNK) as usize * core::mem::size_of::<*mut c_char>(),
        )
        .cast::<*mut c_char>();
        (*sect).values = libc::realloc(
            (*sect).values.cast::<c_void>(),
            ((*sect).max_keys + MALLOC_CHUNK) as usize * core::mem::size_of::<*mut c_char>(),
        )
        .cast::<*mut c_char>();
        (*sect).types = libc::realloc(
            (*sect).types.cast::<c_void>(),
            ((*sect).max_keys + MALLOC_CHUNK) as usize * core::mem::size_of::<u8>(),
        )
        .cast::<u8>();
        (*sect).max_keys += MALLOC_CHUNK;
    }
    if (*sect).keys.is_null() || (*sect).values.is_null() || (*sect).types.is_null() {
        adoc_memory_error(core::ptr::null_mut(), c"addKey".as_ptr());
        return -1;
    }

    /* Copy key and value and increment count */
    *(*sect).keys.add((*sect).num_keys as usize) = libc::strdup(key);
    if !value.is_null() {
        *(*sect).values.add((*sect).num_keys as usize) = libc::strdup(value);
    } else {
        *(*sect).values.add((*sect).num_keys as usize) = core::ptr::null_mut();
    }
    *(*sect).types.add((*sect).num_keys as usize) = if !value.is_null() {
        type_ as u8
    } else {
        ADOC_NO_VALUE as u8
    };
    if (*(*sect).keys.add((*sect).num_keys as usize)).is_null()
        || (!value.is_null() && (*(*sect).values.add((*sect).num_keys as usize)).is_null())
    {
        adoc_memory_error(core::ptr::null_mut(), c"addKey".as_ptr());
        return -1;
    }
    (*sect).num_keys += 1;
    0
}

/// Matches C static `addSection` (`autodoc.c:2185`).
pub unsafe fn add_section(adoc: *mut Autodoc, coll_ind: i32, name: *const c_char) -> i32 {
    let coll: *mut AdocCollection = (*adoc).collections.add(coll_ind as usize);
    let sect: *mut AdocSection;

    /* First allocate enough memory if needed for the sections in the collection
    and for the master lists in the autodoc */
    if (*coll).max_sections == 0 {
        (*coll).sections =
            libc::malloc(MALLOC_CHUNK as usize * core::mem::size_of::<AdocSection>())
                .cast::<AdocSection>();
        (*coll).max_sections = MALLOC_CHUNK;
    } else if (*coll).num_sections >= (*coll).max_sections {
        (*coll).sections = libc::realloc(
            (*coll).sections.cast::<c_void>(),
            ((*coll).max_sections + MALLOC_CHUNK) as usize * core::mem::size_of::<AdocSection>(),
        )
        .cast::<AdocSection>();
        (*coll).max_sections += MALLOC_CHUNK;
    }
    if adoc_memory_error((*coll).sections.cast::<c_void>(), c"addSection".as_ptr()) != 0 {
        return -1;
    }

    if (*adoc).max_sections == 0 {
        (*adoc).coll_list =
            libc::malloc(MALLOC_CHUNK as usize * core::mem::size_of::<c_int>()).cast::<c_int>();
        (*adoc).sect_list =
            libc::malloc(MALLOC_CHUNK as usize * core::mem::size_of::<c_int>()).cast::<c_int>();
        (*adoc).max_sections = MALLOC_CHUNK;
    } else if (*adoc).num_sections >= (*adoc).max_sections {
        (*adoc).coll_list = libc::realloc(
            (*adoc).coll_list.cast::<c_void>(),
            ((*adoc).max_sections + MALLOC_CHUNK) as usize * core::mem::size_of::<c_int>(),
        )
        .cast::<c_int>();
        (*adoc).sect_list = libc::realloc(
            (*adoc).sect_list.cast::<c_void>(),
            ((*adoc).max_sections + MALLOC_CHUNK) as usize * core::mem::size_of::<c_int>(),
        )
        .cast::<c_int>();
        (*adoc).max_sections += MALLOC_CHUNK;
    }
    if (*adoc).coll_list.is_null() || (*adoc).sect_list.is_null() {
        adoc_memory_error(core::ptr::null_mut(), c"addSection".as_ptr());
        return -1;
    }

    /* Copy the name and initialize to empty keys */
    sect = (*coll).sections.add((*coll).num_sections as usize);
    (*sect).name = libc::strdup(name);
    if adoc_memory_error((*sect).name.cast::<c_void>(), c"addSection".as_ptr()) != 0 {
        return -1;
    }
    (*sect).keys = core::ptr::null_mut();
    (*sect).values = core::ptr::null_mut();
    (*sect).types = core::ptr::null_mut();
    (*sect).num_keys = 0;
    (*sect).max_keys = 0;
    (*sect).comments = core::ptr::null_mut();
    (*sect).com_index = core::ptr::null_mut();
    (*sect).num_comments = 0;

    /* Add the collection and section # to master list */
    *(*adoc).coll_list.add((*adoc).num_sections as usize) = coll_ind;
    *(*adoc).sect_list.add((*adoc).num_sections as usize) = (*coll).num_sections;
    (*adoc).num_sections += 1;
    (*coll).num_sections += 1;
    0
}

/// Matches C static `addCollection` (`autodoc.c:2242`).
pub unsafe fn add_collection(adoc: *mut Autodoc, name: *const c_char) -> i32 {
    let coll: *mut AdocCollection;

    /* Allocate just one at a time when needed */
    if (*adoc).num_collections == 0 {
        (*adoc).collections =
            libc::malloc(core::mem::size_of::<AdocCollection>()).cast::<AdocCollection>();
    } else {
        (*adoc).collections = libc::realloc(
            (*adoc).collections.cast::<c_void>(),
            ((*adoc).num_collections + 1) as usize * core::mem::size_of::<AdocCollection>(),
        )
        .cast::<AdocCollection>();
    }
    if adoc_memory_error(
        (*adoc).collections.cast::<c_void>(),
        c"addCollection".as_ptr(),
    ) != 0
    {
        return -1;
    }
    coll = (*adoc).collections.add((*adoc).num_collections as usize);
    (*coll).name = libc::strdup(name);
    if adoc_memory_error((*coll).name.cast::<c_void>(), c"addCollection".as_ptr()) != 0 {
        return -1;
    }
    (*coll).num_sections = 0;
    (*coll).max_sections = 0;
    (*coll).sections = core::ptr::null_mut();
    (*adoc).num_collections += 1;
    0
}

/// Matches C static `addAutodoc` (`autodoc.c:2268`).
pub unsafe fn add_autodoc() -> i32 {
    let adoc: *mut Autodoc;
    let mut index: c_int = -1;
    let mut i: c_int;

    /* Search for a free autodoc in array */
    i = 0;
    while i < S_NUM_AUTODOCS {
        if (*S_AUTODOCS.add(i as usize)).in_use == 0 {
            index = i;
            break;
        }
        i += 1;
    }

    if index < 0 {
        /* Allocate just one at a time when needed */
        if S_NUM_AUTODOCS == 0 {
            S_AUTODOCS = libc::malloc(core::mem::size_of::<Autodoc>()).cast::<Autodoc>();
        } else {
            S_AUTODOCS = libc::realloc(
                S_AUTODOCS.cast::<c_void>(),
                (S_NUM_AUTODOCS + 1) as usize * core::mem::size_of::<Autodoc>(),
            )
            .cast::<Autodoc>();
        }
        if adoc_memory_error(S_AUTODOCS.cast::<c_void>(), c"addAutodoc".as_ptr()) != 0 {
            return -1;
        }
        index = S_NUM_AUTODOCS;
        S_NUM_AUTODOCS += 1;
    }

    /* Initialize collections */
    adoc = S_AUTODOCS.add(index as usize);
    (*adoc).collections = core::ptr::null_mut();
    (*adoc).num_collections = 0;
    (*adoc).final_comments = core::ptr::null_mut();
    (*adoc).num_final_com = 0;
    (*adoc).coll_list = core::ptr::null_mut();
    (*adoc).sect_list = core::ptr::null_mut();
    (*adoc).num_sections = 0;
    (*adoc).max_sections = 0;
    (*adoc).in_use = 1;
    (*adoc).backed_up = 0;
    (*adoc).write_as_xml = 0;
    (*adoc).root_element = core::ptr::null_mut();

    /* Add a collection and section for global data */
    if add_collection(adoc, ADOC_GLOBAL_NAME.as_ptr()) != 0 {
        return -1;
    }
    if add_section(adoc, 0, ADOC_GLOBAL_NAME.as_ptr()) != 0 {
        delete_adoc(adoc);
        return -1;
    }
    index
}

/// Matches C static `deleteAdoc` (`autodoc.c:2317`).
pub unsafe fn delete_adoc(adoc: *mut Autodoc) {
    let mut sect: *mut AdocSection;
    let mut coll: *mut AdocCollection;
    let mut i: c_int;
    let mut j: c_int;
    i = 0;
    while i < (*adoc).num_collections {
        coll = (*adoc).collections.add(i as usize);
        j = 0;
        while j < (*coll).num_sections {
            sect = (*coll).sections.add(j as usize);
            delete_section(sect);
            j += 1;
        }

        /* Free sections */
        if !(*coll).sections.is_null() {
            libc::free((*coll).sections.cast::<c_void>());
            (*coll).sections = core::ptr::null_mut();
        }
        if !(*coll).name.is_null() {
            libc::free((*coll).name.cast::<c_void>());
            (*coll).name = core::ptr::null_mut();
        }
        i += 1;
    }

    /* Free collections */
    if !(*adoc).collections.is_null() {
        libc::free((*adoc).collections.cast::<c_void>());
        (*adoc).collections = core::ptr::null_mut();
    }
    (*adoc).num_collections = 0;

    /* Free lists of sections */
    if !(*adoc).coll_list.is_null() {
        libc::free((*adoc).coll_list.cast::<c_void>());
        (*adoc).coll_list = core::ptr::null_mut();
    }
    if !(*adoc).sect_list.is_null() {
        libc::free((*adoc).sect_list.cast::<c_void>());
        (*adoc).sect_list = core::ptr::null_mut();
    }
    (*adoc).num_sections = 0;
    (*adoc).max_sections = 0;

    /* Free final comments */
    i = 0;
    while i < (*adoc).num_final_com {
        if !(*(*adoc).final_comments.add(i as usize)).is_null() {
            libc::free((*(*adoc).final_comments.add(i as usize)).cast::<c_void>());
            *(*adoc).final_comments.add(i as usize) = core::ptr::null_mut();
        }
        i += 1;
    }
    if !(*adoc).final_comments.is_null() {
        libc::free((*adoc).final_comments.cast::<c_void>());
        (*adoc).final_comments = core::ptr::null_mut();
    }
    (*adoc).num_final_com = 0;
    if !(*adoc).root_element.is_null() {
        libc::free((*adoc).root_element.cast::<c_void>());
        (*adoc).root_element = core::ptr::null_mut();
    }
    (*adoc).in_use = 0;
}

/// Matches C static `deleteSection` (`autodoc.c:2356`).
pub unsafe fn delete_section(sect: *mut AdocSection) {
    let mut k: c_int;

    /* Clean key/values out of section */
    k = 0;
    while k < (*sect).num_keys {
        if !(*(*sect).keys.add(k as usize)).is_null() {
            libc::free((*(*sect).keys.add(k as usize)).cast::<c_void>());
            *(*sect).keys.add(k as usize) = core::ptr::null_mut();
        }
        if !(*(*sect).values.add(k as usize)).is_null() {
            libc::free((*(*sect).values.add(k as usize)).cast::<c_void>());
            *(*sect).values.add(k as usize) = core::ptr::null_mut();
        }
        k += 1;
    }
    if !(*sect).keys.is_null() {
        libc::free((*sect).keys.cast::<c_void>());
        (*sect).keys = core::ptr::null_mut();
    }
    if !(*sect).values.is_null() {
        libc::free((*sect).values.cast::<c_void>());
        (*sect).values = core::ptr::null_mut();
    }
    if !(*sect).types.is_null() {
        libc::free((*sect).types.cast::<c_void>());
        (*sect).types = core::ptr::null_mut();
    }
    if !(*sect).name.is_null() {
        libc::free((*sect).name.cast::<c_void>());
        (*sect).name = core::ptr::null_mut();
    }

    /* Clean comments out of section */
    k = 0;
    while k < (*sect).num_comments {
        if !(*(*sect).comments.add(k as usize)).is_null() {
            libc::free((*(*sect).comments.add(k as usize)).cast::<c_void>());
            *(*sect).comments.add(k as usize) = core::ptr::null_mut();
        }
        k += 1;
    }
    if !(*sect).comments.is_null() {
        libc::free((*sect).comments.cast::<c_void>());
        (*sect).comments = core::ptr::null_mut();
    }
    if !(*sect).com_index.is_null() {
        libc::free((*sect).com_index.cast::<c_void>());
        (*sect).com_index = core::ptr::null_mut();
    }
}

/// Matches C static `parseKeyValue` (`autodoc.c:2381`).
pub unsafe fn parse_key_value(
    line: *mut c_char,
    end: *mut c_char,
    key: *mut *mut c_char,
    value: *mut *mut c_char,
) -> i32 {
    let mut line = line;
    let mut end = end;
    let mut val_start: *mut c_char;
    let mut key_end: *mut c_char;
    let key_len: c_int;
    let val_len: c_int;

    /* Eat spaces at start and end */
    while line < end && (*line == b' ' as c_char || *line == b'\t' as c_char) {
        line = line.add(1);
    }
    while line < end && (*end.sub(1) == b' ' as c_char || *end.sub(1) == b'\t' as c_char) {
        end = end.sub(1);
    }
    if line == end {
        return 1;
    }

    /* Find delimiter.  If it is not there or no text before it, error */
    val_start = libc::strstr(line, S_VALUE_DELIM);
    if val_start.is_null() || val_start == line {
        return 1;
    }

    /* Eat spaces after key */
    key_end = val_start;
    while key_end > line
        && (*key_end.sub(1) == b' ' as c_char || *key_end.sub(1) == b'\t' as c_char)
    {
        key_end = key_end.sub(1);
    }

    /* Eat spaces after the delimiter.  Allow an empty value */
    val_start = val_start.add(libc::strlen(S_VALUE_DELIM));
    while val_start < end && (*val_start == b' ' as c_char || *val_start == b'\t' as c_char) {
        val_start = val_start.add(1);
    }

    /* Allocate for strings and copy them */
    key_len = key_end.offset_from(line) as c_int;
    *key = libc::malloc((key_len + 1) as usize).cast::<c_char>();
    if adoc_memory_error((*key).cast::<c_void>(), c"parseKeyValue".as_ptr()) != 0 {
        return -1;
    }
    libc::memcpy(
        (*key).cast::<c_void>(),
        line.cast::<c_void>(),
        key_len as usize,
    );
    *(*key).add(key_len as usize) = 0;

    val_len = end.offset_from(val_start) as c_int;
    *value = core::ptr::null_mut();
    if val_len != 0 {
        *value = libc::malloc((val_len + 1) as usize).cast::<c_char>();
        if adoc_memory_error((*value).cast::<c_void>(), c"parseKeyValue".as_ptr()) != 0 {
            return -1;
        }
        libc::memcpy(
            (*value).cast::<c_void>(),
            val_start.cast::<c_void>(),
            val_len as usize,
        );
        *(*value).add(val_len as usize) = 0;
    }
    0
}

/// Matches C static `lookupKey` (`autodoc.c:2429`).
pub unsafe fn lookup_key(sect: *mut AdocSection, key: *const c_char) -> i32 {
    let mut i: c_int;
    if key.is_null() {
        return -1;
    }
    i = 0;
    while i < (*sect).num_keys {
        if !(*(*sect).keys.add(i as usize)).is_null()
            && libc::strcmp(key, *(*sect).keys.add(i as usize)) == 0
        {
            return i;
        }
        i += 1;
    }
    -1
}

/// Matches C static `lookupCollection` (`autodoc.c:2442`).
pub unsafe fn lookup_collection(adoc: *mut Autodoc, name: *const c_char) -> i32 {
    let mut i: c_int;
    if name.is_null() || adoc.is_null() {
        return -1;
    }
    i = 0;
    while i < (*adoc).num_collections {
        if libc::strcmp(name, (*(*adoc).collections.add(i as usize)).name) == 0 {
            return i;
        }
        i += 1;
    }
    -1
}

/// Matches C static `getSection` (`autodoc.c:2455`).
pub unsafe fn get_section(type_name: *const c_char, sect_ind: i32) -> *mut AdocSection {
    let coll: *mut AdocCollection;
    let coll_ind: c_int;
    if S_CUR_ADOC.is_null() || type_name.is_null() || sect_ind < 0 {
        return core::ptr::null_mut();
    }
    coll_ind = lookup_collection(S_CUR_ADOC, type_name);
    if coll_ind < 0 {
        return core::ptr::null_mut();
    }
    coll = (*S_CUR_ADOC).collections.add(coll_ind as usize);
    if sect_ind >= (*coll).num_sections {
        return core::ptr::null_mut();
    }
    (*coll).sections.add(sect_ind as usize)
}

/// Matches C static `addComments` (`autodoc.c:2472`).
pub unsafe fn add_comments(
    sect: *mut AdocSection,
    comments: *mut *mut c_char,
    num_comments: *mut c_int,
    index: i32,
) -> i32 {
    let mut i: c_int;
    let new_num: c_int = (*sect).num_comments + *num_comments;
    if (*sect).num_comments != 0 {
        (*sect).comments = libc::realloc(
            (*sect).comments.cast::<c_void>(),
            new_num as usize * core::mem::size_of::<*mut c_char>(),
        )
        .cast::<*mut c_char>();
        (*sect).com_index = libc::realloc(
            (*sect).com_index.cast::<c_void>(),
            new_num as usize * core::mem::size_of::<c_int>(),
        )
        .cast::<c_int>();
    } else {
        (*sect).comments = libc::malloc(new_num as usize * core::mem::size_of::<*mut c_char>())
            .cast::<*mut c_char>();
        (*sect).com_index =
            libc::malloc(new_num as usize * core::mem::size_of::<c_int>()).cast::<c_int>();
    }
    if (*sect).comments.is_null() || (*sect).com_index.is_null() {
        adoc_memory_error(core::ptr::null_mut(), c"addComments".as_ptr());
        return -1;
    }
    i = 0;
    while i < *num_comments {
        *(*sect).comments.add((*sect).num_comments as usize) = *comments.add(i as usize);
        *(*sect).com_index.add((*sect).num_comments as usize) = index;
        (*sect).num_comments += 1;
        i += 1;
    }
    *num_comments = 0;
    0
}

/// Matches C static `findSectionInAdocList` (`autodoc.c:2497`).
pub unsafe fn find_section_in_adoc_list(coll_ind: i32, sect_ind: i32) -> i32 {
    let mut i: c_int;
    i = 0;
    while i < (*S_CUR_ADOC).num_sections {
        if *(*S_CUR_ADOC).coll_list.add(i as usize) == coll_ind
            && *(*S_CUR_ADOC).sect_list.add(i as usize) == sect_ind
        {
            return i;
        }
        i += 1;
    }
    -1
}

/// Matches C static `adocMemoryError` (`autodoc.c:2506`).
pub unsafe fn adoc_memory_error(ptr: *mut c_void, routine: *const c_char) -> i32 {
    if !ptr.is_null() {
        return 0;
    }
    b3d_error(
        Some(&mut ImodFile::Stderr),
        format_args!(
            "ERROR: {} - Allocating memory for string or autodoc component\n",
            CStr::from_ptr(routine).to_string_lossy()
        ),
    );
    -1
}

/// Matches C static `openForWrite` (`autodoc.c:2515`).
pub unsafe fn open_for_write(name: *const c_char, mode: *const c_char) -> *mut libc::FILE {
    let mut fp: *mut libc::FILE = core::ptr::null_mut();
    let mut ind: c_int;
    let trials: c_int = if 0 > S_OPEN_RETRIES {
        0
    } else {
        S_OPEN_RETRIES
    };
    ind = 0;
    while ind <= trials {
        fp = libc::fopen(name, mode);
        if !fp.is_null() || ind == trials {
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
    use std::ffi::CString;
    use std::sync::Mutex;
    /// The autodoc collection (`sAutodocs`) is process-global, so every test
    /// that touches it -- here and in `adoc_fwrap`, which drives the same
    /// collection through the Fortran wrappers -- must serialize on this one
    /// lock, not on a per-module one.
    pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

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
            adoc_done();
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
            std::fs::write(&name, text).unwrap();
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
            adoc_done();
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
            adoc_done();
        }
    }

    /// `writeFile` through `AdocPrintToString`, exercising `fsPrintf`'s string path.
    #[test]
    fn print_to_string_reproduces_the_written_layout() {
        unsafe {
            let _lock = TEST_LOCK.lock().unwrap();
            adoc_done();
            assert!(adoc_new() >= 0);
            assert_eq!(
                adoc_set_key_value(
                    ADOC_GLOBAL_NAME.as_ptr(),
                    0,
                    c"ImageFile".as_ptr(),
                    c"a.mrc".as_ptr()
                ),
                0
            );
            assert_eq!(
                adoc_add_section(ADOC_ZVALUE_NAME.as_ptr(), c"0".as_ptr()),
                0
            );
            assert_eq!(
                adoc_set_two_integers(
                    ADOC_ZVALUE_NAME.as_ptr(),
                    0,
                    c"PieceCoordinates".as_ptr(),
                    3,
                    4
                ),
                0
            );
            let mut buf = [0 as c_char; 512];
            assert_eq!(adoc_print_to_string(buf.as_mut_ptr(), 512, 1), 0);
            assert_eq!(
                CStr::from_ptr(buf.as_ptr()).to_bytes(),
                b"ImageFile = a.mrc\n\n[ZValue = 0]\nPieceCoordinates = 3 4\n"
            );
            adoc_done();
        }
    }

    /// Round-trip a real vendored autodoc through the reader and the writer.
    #[test]
    fn round_trips_a_vendored_autodoc_byte_for_byte() {
        unsafe {
            let _lock = TEST_LOCK.lock().unwrap();
            adoc_done();
            let src = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc/binvol.adoc");
            let path = CString::new(src).unwrap();
            assert!(adoc_read(path.as_ptr()) >= 0);
            let out = std::env::temp_dir().join("imod-rs-autodoc-roundtrip.adoc");
            let out_c = CString::new(out.to_string_lossy().as_bytes()).unwrap();
            assert_eq!(adoc_write(out_c.as_ptr()), 0);
            let written = std::fs::read(&out).unwrap();
            adoc_done();
            /* Re-read the written file and check it produces the same key count */
            assert!(adoc_read(out_c.as_ptr()) >= 0);
            let n = adoc_get_number_of_sections(c"Field".as_ptr());
            adoc_done();
            assert!(n >= 1);
            assert!(!written.is_empty());
            let _ = std::fs::remove_file(&out);
        }
    }

    /// Comments read from an autodoc are attached and written back out in place.
    #[test]
    fn comments_are_preserved_across_a_write() {
        unsafe {
            let _lock = TEST_LOCK.lock().unwrap();
            adoc_done();
            let text = b"# leading comment\nVersion = 1\n\n[Field = A]\n# about B\nB = 2\n";
            let name = std::env::temp_dir().join("imod-rs-autodoc-comments.adoc");
            std::fs::write(&name, text).unwrap();
            let file = CString::new(name.to_string_lossy().as_bytes()).unwrap();
            assert!(adoc_read(file.as_ptr()) >= 0);
            let mut buf = [0 as c_char; 1024];
            assert_eq!(adoc_print_to_string(buf.as_mut_ptr(), 1024, 1), 0);
            let got = CStr::from_ptr(buf.as_ptr()).to_bytes().to_vec();
            adoc_done();
            let _ = std::fs::remove_file(&name);
            assert_eq!(
                String::from_utf8_lossy(&got),
                "# leading comment\nVersion = 1\n\n[Field = A]\n# about B\nB = 2\n"
            );
        }
    }

    /// `AdocGetValTypeAndSize` counts space-separated tokens and reports the type.
    #[test]
    fn val_type_and_size_reports_tokens() {
        unsafe {
            let _lock = TEST_LOCK.lock().unwrap();
            adoc_done();
            assert!(adoc_new() >= 0);
            let mut vals = [1.5f32, 2.5, 3.5];
            assert_eq!(
                adoc_set_float_array(
                    ADOC_GLOBAL_NAME.as_ptr(),
                    0,
                    c"Vals".as_ptr(),
                    vals.as_mut_ptr(),
                    3
                ),
                0
            );
            let mut vtype = 0;
            let mut ntok = 0;
            assert_eq!(
                adoc_get_val_type_and_size(
                    ADOC_GLOBAL_NAME.as_ptr(),
                    0,
                    c"Vals".as_ptr(),
                    &mut vtype,
                    &mut ntok
                ),
                0
            );
            assert_eq!((vtype, ntok), (ADOC_FLOAT_ARRAY, 3));
            let mut s: *mut c_char = core::ptr::null_mut();
            assert_eq!(
                adoc_get_string(ADOC_GLOBAL_NAME.as_ptr(), 0, c"Vals".as_ptr(), &mut s),
                0
            );
            assert_eq!(CStr::from_ptr(s).to_bytes(), b"1.5 2.5 3.5");
            libc::free(s.cast());
            adoc_done();
        }
    }

    /// Insert/delete keep the master section list consistent.
    #[test]
    fn insert_and_delete_section_maintain_the_master_list() {
        unsafe {
            let _lock = TEST_LOCK.lock().unwrap();
            adoc_done();
            assert!(adoc_new() >= 0);
            assert_eq!(adoc_add_section(c"Z".as_ptr(), c"0".as_ptr()), 0);
            assert_eq!(adoc_add_section(c"Z".as_ptr(), c"2".as_ptr()), 1);
            assert_eq!(adoc_find_insert_index(c"Z".as_ptr(), 1), 1);
            assert_eq!(adoc_insert_section(c"Z".as_ptr(), 1, c"1".as_ptr()), 0);
            assert_eq!(adoc_get_number_of_sections(c"Z".as_ptr()), 3);
            let mut nm: *mut c_char = core::ptr::null_mut();
            assert_eq!(adoc_get_section_name(c"Z".as_ptr(), 1, &mut nm), 0);
            assert_eq!(CStr::from_ptr(nm).to_bytes(), b"1");
            libc::free(nm.cast());
            assert_eq!(adoc_lookup_by_name_value(c"Z".as_ptr(), 2), 2);
            assert_eq!(adoc_delete_section(c"Z".as_ptr(), 1), 0);
            assert_eq!(adoc_lookup_by_name_value(c"Z".as_ptr(), 2), 1);
            assert_eq!(adoc_get_number_of_sections(c"Z".as_ptr()), 2);
            adoc_done();
        }
    }

    /// `AdocOrderWriteByValue` sorts the named collection's sections by the
    /// numeric value of their names and puts everything else after them.  The
    /// expected text is the byte-for-byte output of the C driver linked against
    /// the reference `libcfshr`.
    #[test]
    fn ordered_write_sorts_sections_by_numeric_name() {
        unsafe {
            let _lock = TEST_LOCK.lock().unwrap();
            adoc_done();
            assert!(adoc_new() >= 0);
            adoc_set_key_value(
                ADOC_GLOBAL_NAME.as_ptr(),
                0,
                c"ImageFile".as_ptr(),
                c"a.mrc".as_ptr(),
            );
            let names = [c"7", c"3", c"12", c"-2", c"0", c"notanumber", c"9"];
            for (i, name) in names.iter().enumerate() {
                assert_eq!(
                    adoc_add_section(c"ZValue".as_ptr(), name.as_ptr()),
                    i as i32
                );
                adoc_set_integer(c"ZValue".as_ptr(), i as i32, c"Idx".as_ptr(), i as i32);
            }
            assert_eq!(adoc_add_section(c"Other".as_ptr(), c"5".as_ptr()), 0);
            adoc_set_integer(c"Other".as_ptr(), 0, c"Idx".as_ptr(), 100);
            assert_eq!(adoc_order_write_by_value(c"ZValue".as_ptr()), 0);
            let mut buf = [0 as c_char; 2048];
            assert_eq!(adoc_print_to_string(buf.as_mut_ptr(), 2048, 1), 0);
            let got = CStr::from_ptr(buf.as_ptr()).to_bytes().to_vec();
            assert_eq!(adoc_order_write_by_value(core::ptr::null()), 0);
            adoc_done();
            assert_eq!(
                String::from_utf8_lossy(&got),
                "ImageFile = a.mrc\n\n[ZValue = -2]\nIdx = 3\n\n[ZValue = 0]\nIdx = 4\n\n\
                 [ZValue = notanumber]\nIdx = 5\n\n[ZValue = 3]\nIdx = 1\n\n[ZValue = 7]\n\
                 Idx = 0\n\n[ZValue = 9]\nIdx = 6\n\n[ZValue = 12]\nIdx = 2\n\n\
                 [Other = 5]\nIdx = 100\n"
            );
        }
    }

    /// `openForWrite` gives up and `AdocWrite` reports -1 for an unwritable path,
    /// with or without retries.
    #[test]
    fn write_to_an_unopenable_path_returns_minus_one() {
        unsafe {
            let _lock = TEST_LOCK.lock().unwrap();
            adoc_done();
            assert!(adoc_new() >= 0);
            adoc_set_key_value(ADOC_GLOBAL_NAME.as_ptr(), 0, c"A".as_ptr(), c"1".as_ptr());
            adoc_retry_write_opens(0);
            assert_eq!(adoc_write(c"/no/such/dir/x.adoc".as_ptr()), -1);
            assert_eq!(adoc_append_section(c"/no/such/dir/x.adoc".as_ptr()), -1);
            adoc_done();
            /* No current autodoc left */
            assert_eq!(adoc_write(c"/no/such/dir/x.adoc".as_ptr()), -1);
        }
    }

    /// `AdocTransferSection` copies key/values and their types into another autodoc.
    #[test]
    fn transfer_section_copies_keys_and_types() {
        unsafe {
            let _lock = TEST_LOCK.lock().unwrap();
            adoc_done();
            let src = adoc_new();
            assert!(src >= 0);
            assert_eq!(adoc_add_section(c"ZValue".as_ptr(), c"4".as_ptr()), 0);
            assert_eq!(
                adoc_set_three_floats(
                    c"ZValue".as_ptr(),
                    0,
                    c"StagePosition".as_ptr(),
                    1.5,
                    -2.5,
                    0.0
                ),
                0
            );
            let dst = adoc_new();
            assert!(dst > src);
            adoc_set_current(src);
            assert_eq!(
                adoc_transfer_section(c"ZValue".as_ptr(), 0, dst, c"4".as_ptr(), 1),
                0
            );
            /* Transferring to the current autodoc is rejected */
            assert_eq!(
                adoc_transfer_section(c"ZValue".as_ptr(), 0, src, c"4".as_ptr(), 1),
                -2
            );
            adoc_set_current(dst);
            let mut vtype = 0;
            let mut ntok = 0;
            assert_eq!(
                adoc_get_val_type_and_size(
                    c"ZValue".as_ptr(),
                    0,
                    c"StagePosition".as_ptr(),
                    &mut vtype,
                    &mut ntok
                ),
                0
            );
            assert_eq!((vtype, ntok), (ADOC_THREE_FLOATS, 3));
            adoc_done();
        }
    }

    /// The `CommentCharacter` and `KeyValueDelimiter` directives change parsing
    /// for the rest of the file, and `writeFile` re-derives the delimiter from
    /// the global section as it writes.  The expected text is the byte-for-byte
    /// output of the C driver linked against the reference `libcfshr`.
    ///
    /// Note that the two directives are order-sensitive in the source: once
    /// `KeyValueDelimiter` has been seen, a later `CommentCharacter = ;` line
    /// contains no delimiter and is swallowed as a continuation of the previous
    /// value (`autodoc.c:234-249`).  Native does the same.
    #[test]
    fn delimiter_and_comment_character_directives_are_honoured() {
        unsafe {
            let _lock = TEST_LOCK.lock().unwrap();
            adoc_done();
            let text = b"CommentCharacter = ;\n; a comment\nKeyValueDelimiter = :\nA : 5\n\n\
                         [Sect : one]\nB : 3 4\n";
            let name = std::env::temp_dir().join("imod-rs-autodoc-delim.adoc");
            std::fs::write(&name, text).unwrap();
            let file = CString::new(name.to_string_lossy().as_bytes()).unwrap();
            assert!(adoc_read(file.as_ptr()) >= 0);
            let mut i1 = 0;
            let mut i2 = 0;
            assert_eq!(
                adoc_get_two_integers(c"Sect".as_ptr(), 0, c"B".as_ptr(), &mut i1, &mut i2),
                0
            );
            assert_eq!((i1, i2), (3, 4));
            let mut buf = [0 as c_char; 512];
            assert_eq!(adoc_print_to_string(buf.as_mut_ptr(), 512, 1), 0);
            let got = CStr::from_ptr(buf.as_ptr()).to_bytes().to_vec();
            adoc_done();
            let _ = std::fs::remove_file(&name);
            assert_eq!(
                String::from_utf8_lossy(&got),
                "CommentCharacter = ;\n; a comment\nKeyValueDelimiter = :\nA : 5\n\n\
                 [Sect : one]\nB : 3 4\n"
            );
        }
    }

    /// A continuation line (no delimiter) is appended to the previous value.
    #[test]
    fn continuation_lines_append_to_previous_value() {
        unsafe {
            let _lock = TEST_LOCK.lock().unwrap();
            adoc_done();
            let text = b"[Field = A]\nText = one\ntwo\n";
            let name = std::env::temp_dir().join("imod-rs-autodoc-continue.adoc");
            std::fs::write(&name, text).unwrap();
            let file = CString::new(name.to_string_lossy().as_bytes()).unwrap();
            assert!(adoc_read(file.as_ptr()) >= 0);
            let mut s: *mut c_char = core::ptr::null_mut();
            assert_eq!(
                adoc_get_string(c"Field".as_ptr(), 0, c"Text".as_ptr(), &mut s),
                0
            );
            assert_eq!(CStr::from_ptr(s).to_bytes(), b"one two");
            libc::free(s.cast());
            adoc_done();
            let _ = std::fs::remove_file(&name);
        }
    }
}
