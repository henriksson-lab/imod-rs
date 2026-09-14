//! `IMOD/libimod/objgroup.c` and `IMOD/include/objgroup.h`.

use crate::imod::libcfshr::b3dutil::ImodFile;
use std::fs::File;
use std::io::{Read, Write};

use super::imodel::Iobj_group;

/// Original: `objGroupNew` (`objgroup.c:18`).
pub fn obj_group_new() -> Iobj_group {
    Iobj_group::default()
}
/// Original: `objGroupDelete` (`objgroup.c:29`).
pub fn obj_group_delete(group: &mut Iobj_group) {
    obj_group_clear(group);
}
/// Original: `objGroupClear` (`objgroup.c:37`).
pub fn obj_group_clear(group: &mut Iobj_group) {
    group.obj_list.clear();
    group.name = [0; 32];
}
/// Original: `objGroupDup` (`objgroup.c:47`).
pub fn obj_group_dup(old_group: &Iobj_group) -> Iobj_group {
    old_group.clone()
}
/// Original: `objGroupAppend` (`objgroup.c:61`).
pub fn obj_group_append(group: &mut Iobj_group, ob: i32) -> i32 {
    group.obj_list.push(ob);
    0
}
/// Original: `objGroupLookup` (`objgroup.c:73`).
pub fn obj_group_lookup(group: &Iobj_group, ob: i32) -> i32 {
    group
        .obj_list
        .iter()
        .position(|&item| item == ob)
        .map(|item| item as i32)
        .unwrap_or(-1)
}
/// Original: `objGroupListExpand` (`objgroup.c:88`).
pub fn obj_group_list_expand(group_list: &mut Vec<Iobj_group>) -> &mut Iobj_group {
    group_list.push(obj_group_new());
    group_list.last_mut().unwrap()
}
/// Original: `objGroupListDup` (`objgroup.c:109`).
pub fn obj_group_list_dup(group_list: &[Iobj_group]) -> Vec<Iobj_group> {
    group_list.to_vec()
}
/// Original: `objGroupListDelete` (`objgroup.c:135`).
pub fn obj_group_list_delete(group_list: &mut Vec<Iobj_group>) {
    for group in group_list.iter_mut() {
        obj_group_clear(group);
    }
    group_list.clear();
}
/// Original: `objGroupListRemove` (`objgroup.c:145`).
pub fn obj_group_list_remove(group_list: &mut Vec<Iobj_group>, item: i32) -> i32 {
    let Some(group) = group_list.get_mut(item.max(0) as usize) else {
        return 1;
    };
    obj_group_clear(group);
    group_list.remove(item as usize);
    0
}
/// Original: `objGroupListBytes` (`objgroup.c:157`).
pub fn obj_group_list_bytes(group_list: &[Iobj_group]) -> i32 {
    (group_list.len() * std::mem::size_of::<Iobj_group>()
        + group_list
            .iter()
            .map(|group| group.obj_list.len() * 4)
            .sum::<usize>()) as i32
}
/// Original: `objGroupListChecksum` (`objgroup.c:170`).
pub fn obj_group_list_checksum(group_list: &[Iobj_group]) -> f64 {
    let mut count = group_list.len() as f64;
    for group in group_list {
        count += group
            .name
            .iter()
            .take_while(|&&byte| byte != 0)
            .map(|&byte| byte as f64)
            .sum::<f64>();
        count += group.obj_list.len() as f64;
        count += group
            .obj_list
            .iter()
            .map(|&object| object as f64)
            .sum::<f64>();
    }
    count
}
/// Original: `objGroupListWrite` (`objgroup.c:193`).
pub fn obj_group_list_write(group_list: &[Iobj_group], file: &mut ImodFile) -> Result<(), i32> {
    for group in group_list {
        file.write_all(b"OGRP").map_err(|_| 11)?;
        file.write_all(&(32_i32 + 4 * group.obj_list.len() as i32).to_be_bytes())
            .map_err(|_| 11)?;
        file.write_all(&group.name).map_err(|_| 11)?;
        for object in &group.obj_list {
            file.write_all(&object.to_be_bytes()).map_err(|_| 11)?;
        }
    }
    Ok(())
}
/// Original: `objGroupRead` (`objgroup.c:223`).
pub fn obj_group_read(group_list: &mut Vec<Iobj_group>, file: &mut ImodFile) -> Result<(), i32> {
    let mut size = [0; 4];
    file.read_exact(&mut size).map_err(|_| 10)?;
    let count = (i32::from_be_bytes(size) - 32) / 4;
    if count < 0 {
        return Err(21);
    }
    let group = obj_group_list_expand(group_list);
    file.read_exact(&mut group.name).map_err(|_| 10)?;
    for _ in 0..count {
        let mut object = [0; 4];
        file.read_exact(&mut object).map_err(|_| 10)?;
        group.obj_list.push(i32::from_be_bytes(object));
    }
    Ok(())
}
/// Original: `objGroupListToObjList` (`objgroup.c:259`).
pub fn obj_group_list_to_obj_list(
    group_list: &[Iobj_group],
    group_numbers: &mut Vec<i32>,
    mut new_group_list: Option<&mut Vec<Iobj_group>>,
    new_obj_base: i32,
) -> i32 {
    let mut objects = Vec::new();
    for &number in group_numbers.iter() {
        let Some(group) = group_list.get(number as usize) else {
            return 1;
        };
        objects.extend_from_slice(&group.obj_list);
    }
    objects.sort_unstable();
    objects.dedup();
    if let Some(new_groups) = new_group_list.as_mut() {
        for &number in group_numbers.iter() {
            let group = &group_list[number as usize];
            let new_group = obj_group_list_expand(new_groups);
            new_group.name = group.name;
            for &object in &group.obj_list {
                if let Some(index) = objects.iter().position(|&entry| entry == object) {
                    new_group.obj_list.push(index as i32 + new_obj_base);
                }
            }
        }
    }
    *group_numbers = objects;
    0
}
