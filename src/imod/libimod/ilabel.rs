//! Labels for points and contours, from `IMOD/libimod/ilabel.c` and the
//! `Ilabel`/`IlabelItem` declarations of `IMOD/include/imodel.h:308-325`.
//!
//! The C structure carries a `b3dByte *name` that is either NULL or a malloc'd
//! buffer of `len` bytes, and a malloc'd `IlabelItem *label` array of `nl`
//! entries.  This translation keeps `len` (it is load-bearing: the reallocation
//! test in `imodLabelName`/`imodLabelItemAdd` reads it, and `imodLabelRead`
//! takes it from the file) and keeps the NULL/non-NULL distinction of `name` as
//! an `Option`, while following the crate convention of representing the
//! malloc'd `label` array plus its `nl` count as a `Vec`.

use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes};

use super::imodel::{IMOD_ERROR_MEMORY, IMOD_ERROR_WRITE};
use super::imodel_files::{imod_get_bytes, imod_get_int, imod_put_bytes, imod_put_int};

/// Original: `IlabelItem` (`imodel.h:317`).
///
/// `name` is the malloc'd NUL-terminated buffer, `len` its allocated size.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IlabelItem {
    pub name: Option<Vec<u8>>,
    pub len: i32,
    pub index: i32,
}

/// Original: `Ilabel` (`imodel.h:325`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ilabel {
    pub name: Option<Vec<u8>>,
    pub len: i32,
    pub label: Vec<IlabelItem>,
}

/// Original: `imodLabelNew` (`ilabel.c:17`).
pub fn imod_label_new() -> Ilabel {
    let mut label = Ilabel::default();
    label.label.clear();
    label.name = None;
    label.len = 0;
    label
}

/// Original: `imodLabelDup` (`ilabel.c:51`).
pub fn imod_label_dup(label: Option<&Ilabel>) -> Option<Ilabel> {
    let label = label?;

    let mut new_label = imod_label_new();

    if let Some(name) = &label.name {
        /* `strlen(label->name)`, the string length, not the buffer size. */
        let len = name.iter().position(|&b| b == 0).unwrap_or(name.len());

        if label.len != 0 {
            let mut copy = vec![0_u8; len + 1];
            copy[..len + 1].copy_from_slice(&name[..len + 1]);
            new_label.name = Some(copy);
            new_label.len = len as i32;
        }
    }

    for i in 0..label.label.len() {
        let name = label.label[i].name.clone();
        imod_label_item_add(&mut new_label, name.as_deref(), label.label[i].index);
    }
    Some(new_label)
}

/// Original: `imodLabelName` (`ilabel.c:87`).
///
/// `val` is the C `const char *`, taken here as the NUL-terminated bytes
/// without the terminator.
pub fn imod_label_name(label: Option<&mut Ilabel>, val: Option<&[u8]>) -> i32 {
    let label = match label {
        Some(label) => label,
        None => return 1,
    };
    let val = match val {
        Some(val) => val,
        None => return 1,
    };
    let len = val.iter().position(|&b| b == 0).unwrap_or(val.len());

    if label.name.is_none() {
        label.name = Some(vec![0_u8; len + 1]);
        label.len = len as i32 + 1;
    } else if label.len < (len as i32 + 1) {
        /* B3DFREE then malloc: the old buffer's contents are discarded. */
        label.name = Some(vec![0_u8; len + 1]);
        label.len = len as i32 + 1;
    }
    let name = label.name.as_mut().unwrap();
    name[..len].copy_from_slice(&val[..len]);
    name[len] = 0;
    0
}

/// Original: `imodLabelItemAdd` (`ilabel.c:120`).
pub fn imod_label_item_add(label: &mut Ilabel, val: Option<&[u8]>, index: i32) {
    let val = match val {
        Some(val) => val,
        None => return,
    };

    let len = val.iter().position(|&b| b == 0).unwrap_or(val.len());

    /* Look for an existing label with this index and replace it if found */
    for i in 0..label.label.len() {
        if index == label.label[i].index {
            if label.label[i].len < (len as i32 + 1) {
                label.label[i].name = Some(vec![0_u8; len + 1]);
                label.label[i].len = len as i32 + 1;
            }
            let name = label.label[i].name.as_mut().unwrap();
            name[..len].copy_from_slice(&val[..len]);
            name[len] = 0;
            return;
        }
    }

    /* Now forget it if it is a zero-length label */
    if len == 0 {
        return;
    }

    /* Otherwise make a new label item and copy the label into it */
    let mut name = vec![0_u8; len + 1];
    name[..len].copy_from_slice(&val[..len]);
    name[len] = 0;
    label.label.push(IlabelItem {
        name: Some(name),
        len: len as i32 + 1,
        index,
    });
}

/// Original: `imodLabelItemMove` (`ilabel.c:169`).
pub fn imod_label_item_move(label: Option<&mut Ilabel>, to_index: i32, from_index: i32) {
    let label = match label {
        Some(label) => label,
        None => return,
    };

    for i in 0..label.label.len() {
        if from_index == label.label[i].index {
            label.label[i].index = to_index;
            break;
        }
    }
}

/// Original: `imodLabelItemDelete` (`ilabel.c:185`).
pub fn imod_label_item_delete(label: Option<&mut Ilabel>, index: i32) {
    let label = match label {
        Some(label) => label,
        None => return,
    };

    let mut deli: i32 = -1;
    for i in 0..label.label.len() {
        if index == label.label[i].index {
            deli = i as i32;
            break;
        }
    }
    if deli < 0 {
        return;
    }
    label.label.remove(deli as usize);
}

/// Original: `imodLabelNameGet` (`ilabel.c:210`).
pub fn imod_label_name_get(label: Option<&Ilabel>) -> Option<&[u8]> {
    let label = label?;
    label.name.as_deref()
}

/// Original: `imodLabelItemGet` (`ilabel.c:222`).
pub fn imod_label_item_get(label: Option<&Ilabel>, index: i32) -> Option<&[u8]> {
    let label = label?;
    if label.label.is_empty() {
        return None;
    }
    for i in 0..label.label.len() {
        if index == label.label[i].index {
            return label.label[i].name.as_deref();
        }
    }
    None
}

/// Original: `imodLabelPrint` (`ilabel.c:237`).
pub fn imod_label_print(lab: Option<&Ilabel>, fout: &mut dyn Write) {
    let lab = match lab {
        Some(lab) => lab,
        None => return,
    };
    if let Some(name) = &lab.name {
        let end = name
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(name.len());
        let _ = fout.write_all(&c_format_bytes(
            "contour label : \"%s\"\n",
            &[CArg::Bytes(&name[..end])],
        ));
    }
    if !lab.label.is_empty() {
        for i in 0..lab.label.len() {
            // glibc prints `(null)` for a NULL `%s`; `IlabelItem::name` is the
            // C `b3dByte *name`, which `imodLabelRead` can leave NULL.
            let text: &[u8] = match &lab.label[i].name {
                Some(name) => {
                    &name[..name
                        .iter()
                        .position(|&byte| byte == 0)
                        .unwrap_or(name.len())]
                }
                None => b"(null)",
            };
            let _ = fout.write_all(&c_format_bytes(
                "\t%3d : \"%s\"\n",
                &[CArg::Int(lab.label[i].index as i64), CArg::Bytes(text)],
            ));
        }
    }
}

/* MATCHING STUFF, UNUSED 8/21/07 */
/// Original: `imodLabelMatch` (`ilabel.c:255`).
pub fn imod_label_match(label: Option<&Ilabel>, tstr: Option<&[u8]>) -> i32 {
    let (label, tstr) = match (label, tstr) {
        (Some(label), Some(tstr)) => (label, tstr),
        _ => return 0,
    };

    ilabel_match_reg(
        tstr,
        match &label.name {
            Some(name) => name,
            None => &[],
        },
    )
}

/// Original: `imodLabelItemMatch` (`ilabel.c:264`).
pub fn imod_label_item_match(label: Option<&Ilabel>, tstr: Option<&[u8]>, index: i32) -> i32 {
    let (label, tstr) = match (label, tstr) {
        (Some(label), Some(tstr)) => (label, tstr),
        _ => return 0,
    };

    let lstr = imod_label_item_get(Some(label), index);
    let lstr = match lstr {
        Some(lstr) => lstr,
        None => return 0,
    };

    ilabel_match_reg(tstr, lstr)
}

/// Original: `ilabelMatchReg` (`ilabel.c:277`).
///
/// The C walks two NUL-terminated `const char *`.  Both are byte slices here,
/// with the same NUL sentinel: `exp` is indexed from a moving base exactly as
/// the source advances its pointer, and `exp[0] == 0` -- the source's
/// end-of-expression test -- becomes "at or past the slice's NUL".
pub fn ilabel_match_reg(exp: &[u8], str_: &[u8]) -> i32 {
    let len: i32;
    let mut n: i32;

    /* The C's `(!exp) || (!str)` NULL test; an empty slice has no NUL to read
    and stands for the same absence. */
    if exp.is_empty() || str_.is_empty() {
        return 0;
    }

    len = str_
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(str_.len()) as i32;

    /* The C's moving `exp` pointer, as an offset from the start of the slice. */
    let mut base = 0usize;
    let at =
        |slice: &[u8], index: usize| -> u8 { if index < slice.len() { slice[index] } else { 0 } };

    let mut i: i32 = 0;
    while i < len {
        if at(exp, base) == 0 {
            return 0;
        }

        if at(exp, base) == b'\\' {
            base += 1;
        } else {
            if at(exp, base) == b'?' {
                base += 1;
                i += 1;
                continue;
            }
            if at(exp, base) == b'*' {
                n = at(exp, base + 1) as i32;
                if n == 0 {
                    return 1;
                }
                if n == at(str_, i as usize) as i32
                    && ilabel_match_reg(&exp[(base + 1).min(exp.len())..], &str_[i as usize..]) != 0
                {
                    return 1;
                }
                i += 1;
                continue;
            }
        }

        if at(exp, base) != at(str_, i as usize) {
            return 0;
        }
        base += 1;
        i += 1;
    }
    if at(exp, base) != 0 {
        return 0;
    }
    1
}

/*****************************************************************************/
/* file io */
/* format

bytes data
----------
4     'LABL'
4     length of entire label data structures

4     # of label items
4     size of contour label
size  contour label string padded to 4 byte chunks.

for each label item:

4     size
size  data.

*/

/// Original: `getpadlen` (`ilabel.c:369`), the static padded-length helper.
fn getpadlen(string: Option<&[u8]>) -> i32 {
    let mut len: i32;
    let pad: i32;
    let string = match string {
        Some(string) => string,
        None => return 0,
    };
    len = string.iter().position(|&b| b == 0).unwrap_or(string.len()) as i32 + 1;
    if len == 0 {
        return 0;
    }
    pad = len % 4;
    len /= 4;
    len *= 4;
    if pad != 0 {
        len += 4;
    }
    len
}

/// Original: `imodLabelWrite` (`ilabel.c:384`).
pub fn imod_label_write(lab: Option<&Ilabel>, tag: u32, fout: &mut ImodFile) -> i32 {
    let mut id: u32;
    let mut len: i32;
    let mut pad: i32;
    let mut lpad: i32;

    let lab = match lab {
        Some(lab) => lab,
        None => return -1,
    };

    /* `bgnpos = ftell(fout)` is dead in the source; the value is never used. */

    if imod_put_int(fout, tag as i32).is_err() {
        return IMOD_ERROR_WRITE;
    }

    /* Calculate lenth of data to be written. Put out 4 nulls for an empty name */
    id = 8;
    len = getpadlen(lab.name.as_deref());
    if len == 0 {
        len = 4;
    }
    id = id.wrapping_add(len as u32);
    for l in 0..lab.label.len() {
        id = id.wrapping_add(8);
        id = id.wrapping_add(getpadlen(lab.label[l].name.as_deref()) as u32);
    }
    if imod_put_int(fout, id as i32).is_err() {
        return IMOD_ERROR_WRITE;
    }

    /* write the number of labels. */
    if imod_put_int(fout, lab.label.len() as i32).is_err() {
        return IMOD_ERROR_WRITE;
    }
    lpad = getpadlen(lab.name.as_deref());
    if lpad == 0 {
        lpad = 4;
    }
    if imod_put_int(fout, lpad).is_err() {
        return IMOD_ERROR_WRITE;
    }

    pad = lpad;
    if let Some(name) = &lab.name {
        len = name.iter().position(|&b| b == 0).unwrap_or(name.len()) as i32;
        if imod_put_bytes(fout, name, len).is_err() {
            return IMOD_ERROR_WRITE;
        }
        pad = lpad - len;
    }
    if pad > 0 {
        let zeros = [0_u8; 4];
        if imod_put_bytes(fout, &zeros, pad).is_err() {
            return IMOD_ERROR_WRITE;
        }
    }

    for l in 0..lab.label.len() {
        if imod_put_int(fout, lab.label[l].index).is_err() {
            return IMOD_ERROR_WRITE;
        }
        lpad = getpadlen(lab.label[l].name.as_deref());
        let name = lab.label[l].name.as_deref().unwrap_or(&[]);
        len = name.iter().position(|&b| b == 0).unwrap_or(name.len()) as i32;
        pad = lpad - len;

        if imod_put_int(fout, lpad).is_err() {
            return IMOD_ERROR_WRITE;
        }
        if imod_put_bytes(fout, name, len).is_err() {
            return IMOD_ERROR_WRITE;
        }
        if pad > 0 {
            let zeros = [0_u8; 4];
            if imod_put_bytes(fout, &zeros, pad).is_err() {
                return IMOD_ERROR_WRITE;
            }
        }
    }

    0
}

/// Original: `imodLabelRead` (`ilabel.c:449`).
pub fn imod_label_read(fin: &mut ImodFile, err: &mut i32) -> Option<Ilabel> {
    let retcode = 0;
    let mut lab = imod_label_new();
    let ml: i32;
    *err = IMOD_ERROR_MEMORY;

    /* size of data chunk. */
    let _ml = imod_get_int(fin).ok()?;

    /* number of labels. */
    ml = imod_get_int(fin).ok()?;

    /* The name of the label list. */
    lab.len = imod_get_int(fin).ok()?;
    let mut name = vec![0_u8; lab.len.max(0) as usize];
    imod_get_bytes(fin, &mut name, lab.len).ok()?;
    lab.name = Some(name);

    /* The label list. */
    for _l in 0..ml {
        let index = imod_get_int(fin).ok()?;
        let len = imod_get_int(fin).ok()?;
        let mut name = vec![0_u8; len.max(0) as usize];
        imod_get_bytes(fin, &mut name, len).ok()?;
        lab.label.push(IlabelItem {
            name: Some(name),
            len,
            index,
        });
    }

    *err = retcode;
    Some(lab)
}
