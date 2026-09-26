//! `IMOD/libimod/objgroup.c` and `IMOD/include/objgroup.h`.
//!
//! `Ilist *` of `IobjGroup` is a `Vec<Iobj_group>`, and a group's `objList`
//! (an `Ilist` of `b3dInt32`) is a `Vec<i32>`; `ilistSize(NULL)` is 0, which an
//! empty `Vec` gives too.  Memory-error returns cannot arise and are dropped.

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::robuststat::rs_sort_ints;

use super::imodel::Iobj_group;
use super::imodel::{IMOD_ERROR_FORMAT, IMOD_ERROR_READ, IMOD_ERROR_WRITE};

/// `OBJGRP_STRSIZE` (`objgroup.h:19`).
pub const OBJGRP_STRSIZE: usize = 32;

/// `sizeof(IobjGroup)` on the LP64 platform the reference is built for: an
/// `Ilist *` plus `char name[32]` (`objgroup.h:21-24`).  `objGroupListBytes`
/// reports the C allocation sizes, not the Rust ones.
const SIZEOF_IOBJGROUP: usize = 40;
/// `sizeof(Ilist)` on LP64: `void *` plus five `int` (`ilist.h:35-43`), padded to 8.
const SIZEOF_ILIST: usize = 32;

/// Original: `objGroupNew` (`objgroup.c:18`).
pub fn obj_group_new() -> Iobj_group {
    // The C mallocs, sets `objList = NULL` and `name[0] = 0`; the rest of the
    // name is uninitialised there and zero here.
    Iobj_group::default()
}
/// Original: `objGroupClear` (`objgroup.c:37`).
pub fn obj_group_clear(group: &mut Iobj_group) {
    group.obj_list = Vec::new();
    group.name[0] = 0x00;
}
/// Original: `objGroupAppend` (`objgroup.c:61`).
pub fn obj_group_append(group: &mut Iobj_group, ob: i32) -> i32 {
    group.obj_list.push(ob);
    0
}
/// Original: `objGroupLookup` (`objgroup.c:73`).
pub fn obj_group_lookup(group: &Iobj_group, ob: i32) -> i32 {
    for i in 0..group.obj_list.len() {
        let j = &group.obj_list[i];
        if ob == *j {
            return i as i32;
        }
    }
    -1
}
/// Original: `objGroupListExpand` (`objgroup.c:88`).
pub fn obj_group_list_expand(grplist: &mut Vec<Iobj_group>) -> &mut Iobj_group {
    let group = obj_group_new();
    grplist.push(group);
    let last = grplist.len() - 1;
    &mut grplist[last]
}
/// Original: `objGroupListDup` (`objgroup.c:109`).
pub fn obj_group_list_dup(grplist: &[Iobj_group]) -> Vec<Iobj_group> {
    let mut newlist = Vec::with_capacity(grplist.len());
    for i in 0..grplist.len() {
        let ogroup = &grplist[i];
        newlist.push(Iobj_group {
            obj_list: ogroup.obj_list.clone(),
            name: ogroup.name,
        });
    }
    newlist
}
/// Original: `objGroupListDelete` (`objgroup.c:135`).
pub fn obj_group_list_delete(grplist: &mut Vec<Iobj_group>) {
    for i in 0..grplist.len() {
        obj_group_clear(&mut grplist[i]);
    }
    grplist.clear();
}
/// Original: `objGroupListRemove` (`objgroup.c:145`).
pub fn obj_group_list_remove(grplist: &mut Vec<Iobj_group>, item: i32) -> i32 {
    // `ilistItem` returns NULL for an index below 0 or at/after the size.
    if item < 0 || item as usize >= grplist.len() {
        return 1;
    }
    obj_group_clear(&mut grplist[item as usize]);
    grplist.remove(item as usize);
    0
}
/// Original: `objGroupListBytes` (`objgroup.c:157`).
pub fn obj_group_list_bytes(grplist: &[Iobj_group]) -> i32 {
    let mut count = (grplist.len() * SIZEOF_IOBJGROUP) as i32;
    for i in 0..grplist.len() {
        let group = &grplist[i];
        count += (group.obj_list.len() * 4 + SIZEOF_ILIST) as i32;
    }
    count
}
/// Original: `objGroupListChecksum` (`objgroup.c:170`).
pub fn obj_group_list_checksum(grplist: &[Iobj_group]) -> f64 {
    let mut count = grplist.len() as f64;
    for i in 0..grplist.len() {
        let group = &grplist[i];
        let mut j = 0;
        while j < OBJGRP_STRSIZE && group.name[j] != 0 {
            // `(int)group->name[j]`: `char` is signed on this platform.
            count += (group.name[j] as i8 as i32) as f64;
            j += 1;
        }
        count += group.obj_list.len() as f64;
        let objs = &group.obj_list;
        for j in 0..objs.len() {
            count += objs[j] as f64;
        }
    }
    count
}
/// Original: `objGroupListWrite` (`objgroup.c:193`).
pub fn obj_group_list_write(grplist: &[Iobj_group], fout: &mut ImodFile) -> Result<(), i32> {
    for i in 0..grplist.len() {
        let group = &grplist[i];
        let id = super::imodel_files::ID_OGRP as i32;
        fout.imod_put_int(id).map_err(|_| IMOD_ERROR_WRITE)?;
        let num = group.obj_list.len() as i32;
        let size = OBJGRP_STRSIZE as i32 + 4 * num;
        fout.imod_put_int(size).map_err(|_| IMOD_ERROR_WRITE)?;
        fout.imod_put_bytes(&group.name, OBJGRP_STRSIZE as i32)
            .map_err(|_| IMOD_ERROR_WRITE)?;
        if num != 0 {
            let objs = &group.obj_list;
            if objs.is_empty() {
                return Err(IMOD_ERROR_FORMAT);
            }
            fout.imod_put_ints(objs, num)
                .map_err(|_| IMOD_ERROR_WRITE)?;
        }
    }
    Ok(())
}
/// Original: `objGroupRead` (`objgroup.c:223`).
///
/// A short read maps to `IMOD_ERROR_READ` as everywhere else in the
/// translated reader; the C's `ferror` test does not see end-of-file, which
/// its caller's chunk loop then meets.
pub fn obj_group_read(grplistp: &mut Vec<Iobj_group>, fin: &mut ImodFile) -> Result<(), i32> {
    let size = fin.imod_get_int().map_err(|_| IMOD_ERROR_READ)?;
    let num = (size - OBJGRP_STRSIZE as i32) / 4;
    let group = obj_group_list_expand(grplistp);
    fin.imod_get_bytes(&mut group.name, OBJGRP_STRSIZE as i32)
        .map_err(|_| IMOD_ERROR_READ)?;
    group.obj_list = Vec::with_capacity(num.max(16) as usize);

    /* Read the data directly into the list structure and set size directly */
    // The source tests `if (num)`: a chunk size below 32 gives a negative
    // `num`, which `imodGetInts` hands to `fread` as a huge `size_t`, and the
    // FORTIFY build aborts with "buffer overflow detected" (rc 134, measured
    // with native `imodtrans`/`imodinfo`; BUGS.md).  Not reproduced: a
    // negative count reads no objects, as the size-0 case does.
    if num > 0 {
        group.obj_list.resize(num as usize, 0);
        fin.imod_get_ints(&mut group.obj_list, num)
            .map_err(|_| IMOD_ERROR_READ)?;
    }
    Ok(())
}
/// Original: `objGroupListToObjList` (`objgroup.c:259`).
///
/// `grpNumList`/`numInList` are the `Vec`'s contents and length.
pub fn obj_group_list_to_obj_list(
    group_list: &[Iobj_group],
    grp_num_list: &mut Vec<i32>,
    new_group_list: Option<&mut Vec<Iobj_group>>,
    new_obj_base: i32,
) -> i32 {
    let in_list = &*grp_num_list;
    let num_in = in_list.len();
    let mut num_tot_obj = 0usize;
    let mut num_trimmed = 0usize;

    // Count total objects listed in groups including duplicates
    for ind in 0..num_in {
        if in_list[ind] < 0 || in_list[ind] as usize >= group_list.len() {
            return 1;
        }
        let group = &group_list[in_list[ind] as usize];
        num_tot_obj += group.obj_list.len();
    }

    // Get oversized list
    let mut new_list = vec![0i32; num_tot_obj];

    // Fill new list with all objects
    num_tot_obj = 0;
    for ind in 0..num_in {
        let group = &group_list[in_list[ind] as usize];
        for ob in 0..group.obj_list.len() {
            new_list[num_tot_obj] = group.obj_list[ob];
            num_tot_obj += 1;
        }
    }

    // Sort and trim it to unique objects
    rs_sort_ints(&mut new_list, num_tot_obj as i32);
    for ind in 0..num_tot_obj {
        if ind == 0 || new_list[ind] != new_list[num_trimmed - 1] {
            new_list[num_trimmed] = new_list[ind];
            num_trimmed += 1;
        }
    }

    let in_list = std::mem::take(grp_num_list);
    new_list.truncate(num_trimmed);
    *grp_num_list = new_list;
    let new_list = &*grp_num_list;
    let Some(new_group_list) = new_group_list else {
        return 0;
    };

    // For each existing group, add an object group to output list
    for ind in 0..num_in {
        let group = &group_list[in_list[ind] as usize];
        let new_group = obj_group_list_expand(new_group_list);

        // Copy name: `strncpy` zero-pads past the terminator, then byte 31 is
        // forced to 0.
        let len = group
            .name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(OBJGRP_STRSIZE);
        if len != 0 {
            new_group.name[..len].copy_from_slice(&group.name[..len]);
            new_group.name[len..OBJGRP_STRSIZE].fill(0);
            new_group.name[OBJGRP_STRSIZE - 1] = 0x00;
        }

        // Look up new object number of each member of group and add to new group
        for ob in 0..group.obj_list.len() {
            let ob_num = group.obj_list[ob];
            for jnd in 0..num_trimmed {
                if new_list[jnd] == ob_num {
                    let new_num = jnd as i32 + new_obj_base;
                    if obj_group_append(new_group, new_num) != 0 {
                        return 2;
                    }
                    break;
                }
            }
        }
    }

    // free and replace input list, return new object group list
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BUGS.md` "objGroupRead", fixed in translation: an `OGRP` chunk size
    /// below 32 gives a negative object count, which native hands to `fread`
    /// (and aborts); here it reads a group with no objects.
    #[test]
    fn a_negative_object_count_reads_no_objects() {
        let path =
            std::env::temp_dir().join(format!("imod-rs-objgroup-negative-{}", std::process::id()));
        let mut bytes = 16i32.to_be_bytes().to_vec();
        bytes.extend_from_slice(&[b'g'; OBJGRP_STRSIZE]);
        std::fs::write(&path, &bytes).unwrap();
        let mut fin = ImodFile::open(&path, "rb").unwrap();
        let mut groups = Vec::new();
        assert!(obj_group_read(&mut groups, &mut fin).is_ok());
        assert_eq!(groups.len(), 1);
        assert!(groups[0].obj_list.is_empty());
        drop(fin);
        let _ = std::fs::remove_file(&path);
    }
}
