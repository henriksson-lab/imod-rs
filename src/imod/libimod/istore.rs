//! Scaffold for the complete `IMOD/libimod/istore.c` source unit.
#![allow(dead_code, unused_variables)]
use crate::imod::libimod::imodel::{Icont, Iobj};
use crate::imod::libimod::imodel_files::{
    imod_get_float, imod_get_int, imod_get_short, imod_put_float, imod_put_int, imod_put_short,
};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
#[repr(C)]
#[derive(Clone, Copy)]
pub union StoreUnion {
    pub i: i32,
    pub f: f32,
    pub us: [u16; 2],
    pub s: [i16; 2],
    pub b: [u8; 4],
}
impl Default for StoreUnion {
    fn default() -> Self {
        Self { i: 0 }
    }
}
impl core::fmt::Debug for StoreUnion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        unsafe { self.i.fmt(f) }
    }
}
impl PartialEq for StoreUnion {
    fn eq(&self, other: &Self) -> bool {
        unsafe { self.i == other.i }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Istore {
    pub type_: i16,
    pub flags: u16,
    pub index: StoreUnion,
    pub value: StoreUnion,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct DrawProps {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    pub fill_red: f32,
    pub fill_green: f32,
    pub fill_blue: f32,
    pub trans: i32,
    pub connect: i32,
    pub gap: i32,
    pub linewidth: i32,
    pub linewidth2: i32,
    pub symtype: i32,
    pub symflags: i32,
    pub symsize: i32,
    pub value1: f32,
    pub valskip: i32,
    pub no_cap: i32,
}
pub fn imod_write_store(list: &[Istore], id: i32, file: &mut File) -> i32 {
    if list.is_empty() {
        return 0;
    }
    if imod_put_int(file, id).is_err() || imod_put_int(file, (list.len() * 12) as i32).is_err() {
        return 1;
    }
    for store in list {
        if imod_put_short(file, store.type_).is_err()
            || imod_put_short(file, store.flags as i16).is_err()
        {
            return 1;
        }
        let mut dtype = store.flags & 3;
        for item in [store.index, store.value] {
            let error = match dtype {
                0 => imod_put_int(file, unsafe { item.i }).is_err(),
                1 => imod_put_float(file, unsafe { item.f }).is_err(),
                2 => {
                    let shorts = unsafe { item.s };
                    imod_put_short(file, shorts[0]).is_err()
                        || imod_put_short(file, shorts[1]).is_err()
                }
                _ => file.write_all(&unsafe { item.b }).is_err(),
            };
            if error {
                return 1;
            }
            dtype = (store.flags >> 2) & 3;
        }
    }
    0
}
/// Original: `imodReadStore` (`istore.c`).
pub fn imod_read_store(file: &mut File, error: &mut i32) -> Option<Vec<Istore>> {
    let nread = match imod_get_int(file) {
        Ok(bytes) if bytes > 0 => bytes / 12,
        _ => {
            *error = 1;
            return None;
        }
    };
    *error = 0;
    let mut list = Vec::with_capacity(nread as usize);
    let mut need_sort = false;
    let mut last_index = 0;
    for entry in 0..nread {
        let type_ = match imod_get_short(file) {
            Ok(value) => value,
            Err(_) => {
                *error = 1;
                return None;
            }
        };
        let flags = match imod_get_short(file) {
            Ok(value) => value as u16,
            Err(_) => {
                *error = 1;
                return None;
            }
        };
        let mut items = [StoreUnion::default(), StoreUnion::default()];
        let mut dtype = flags & 3;
        for item in &mut items {
            *item = match dtype {
                0 => match imod_get_int(file) {
                    Ok(value) => StoreUnion { i: value },
                    Err(_) => {
                        *error = 1;
                        return None;
                    }
                },
                1 => match imod_get_float(file) {
                    Ok(value) => StoreUnion { f: value },
                    Err(_) => {
                        *error = 1;
                        return None;
                    }
                },
                2 => match (imod_get_short(file), imod_get_short(file)) {
                    (Ok(first), Ok(second)) => StoreUnion { s: [first, second] },
                    _ => {
                        *error = 1;
                        return None;
                    }
                },
                _ => {
                    let mut bytes = [0; 4];
                    if file.read_exact(&mut bytes).is_err() {
                        *error = 1;
                        return None;
                    }
                    StoreUnion { b: bytes }
                }
            };
            dtype = (flags >> 2) & 3;
        }
        let store = Istore {
            type_,
            flags,
            index: items[0],
            value: items[1],
        };
        let index = if flags & ((1 << 4) | 3) == 0 {
            unsafe { store.index.i }
        } else {
            i32::MAX
        };
        if entry != 0 && index < last_index {
            need_sort = true;
        }
        last_index = index;
        list.push(store);
    }
    if need_sort {
        istore_sort(&mut list);
    }
    Some(list)
}
/// Original: `storeCompare` (`istore.c`).
pub fn store_compare(one: &Istore, two: &Istore) -> core::cmp::Ordering {
    let first = if one.flags & ((1 << 4) | 3) == 0 {
        unsafe { one.index.i }
    } else {
        i32::MAX
    };
    let second = if two.flags & ((1 << 4) | 3) == 0 {
        unsafe { two.index.i }
    } else {
        i32::MAX
    };
    first.cmp(&second)
}
/// Original: `istoreNextObjItem` (`istore.c`).
pub fn istore_next_obj_item<'a>(
    list: &'a [Istore],
    co: i32,
    surf: i32,
    first: i32,
    cursor: &mut usize,
) -> Option<&'a Istore> {
    if first != 0 {
        *cursor = 0;
    }
    while *cursor < list.len() {
        let store = &list[*cursor];
        *cursor += 1;
        if store.flags & ((1 << 4) | 3) != 0 {
            return None;
        }
        let index = unsafe { store.index.i };
        if (store.flags & (1 << 6) == 0 && index == co)
            || (store.flags & (1 << 6) != 0 && index == surf)
        {
            return Some(store);
        }
    }
    None
}
pub fn istore_sort(list: &mut Vec<Istore>) {
    list.sort_by(store_compare);
}
pub fn istore_insert(list: &mut Vec<Istore>, store: Istore) -> i32 {
    let (_, mut after) = istore_lookup(list, unsafe { store.index.i });
    if store.flags & (1 << 4) != 0 {
        after = list.len();
    }
    let (lookup, _) = istore_lookup(list, unsafe { store.index.i });
    if store.flags & (1 << 5) != 0 {
        if let Some(index) = lookup {
            after = index;
        }
    }
    list.insert(after, store);
    0
}
pub fn istore_lookup(list: &[Istore], index: i32) -> (Option<usize>, usize) {
    let mut first = None;
    let mut after = list.len();
    for (item, store) in list.iter().enumerate() {
        if store.flags & ((1 << 4) | 3) != 0 || unsafe { store.index.i } > index {
            after = item;
            break;
        }
        if unsafe { store.index.i } == index {
            first.get_or_insert(item);
            after = item + 1;
        }
    }
    if let Some(found) = first {
        while after < list.len()
            && list[after].flags & ((1 << 4) | 3) == 0
            && unsafe { list[after].index.i } == index
        {
            after += 1;
        }
        (Some(found), after)
    } else {
        (None, after)
    }
}
pub fn istore_dump(list: &[Istore]) {
    let types = [
        "COLOR",
        "FCOLOR",
        "TRANS",
        "GAP",
        "CONNECT",
        "3DWIDTH",
        "2DWIDTH",
        "SYMTYPE",
        "SYMSIZE",
        "VALUE1",
        "MINMAX1",
        "VALUE2",
        "MINMAX2",
        "VALUE3",
        "MINMAX3",
        "VALUE4",
        "MINMAX4",
        "VALUE5",
        "MINMAX5",
        "VALUE6",
        "MINMAX6",
        "ISOPARAM",
        "ISOTHRESH",
    ];
    println!(" {} items in list:", list.len());
    for store in list {
        print!("{:6}-", store.type_);
        if store.type_ > 0 && store.type_ as usize <= types.len() {
            print!("{}", types[store.type_ as usize - 1]);
        }
        print!("  {:6o}-", store.flags);
        let mut dtype = store.flags & 3;
        for item in [store.index, store.value] {
            match dtype {
                0 => print!(" {:11}", unsafe { item.i }),
                1 => print!(" {:12.6}", unsafe { item.f }),
                2 => print!(" {:6} {:6}", unsafe { item.s[0] }, unsafe { item.s[1] }),
                _ => print!(
                    " {:3} {:3} {:3} {:3}",
                    unsafe { item.b[0] },
                    unsafe { item.b[1] },
                    unsafe { item.b[2] },
                    unsafe { item.b[3] }
                ),
            }
            dtype = (store.flags >> 2) & 3;
        }
        println!();
    }
}
pub fn istore_checksum(list: &[Istore]) -> f64 {
    let mut sum = 0.;
    for store in list {
        sum += (store.flags as i16 + store.type_) as f64;
        let mut dtype = store.flags & 3;
        for item in [store.index, store.value] {
            sum += match dtype {
                0 => unsafe { item.i as f64 },
                1 => unsafe { item.f as f64 },
                2 => unsafe { (item.s[0] + item.s[1]) as f64 },
                _ => unsafe {
                    (item.b[0] as u32 + item.b[1] as u32 + item.b[2] as u32 + item.b[3] as u32)
                        as f64
                },
            };
            dtype = (store.flags >> 2) & 3;
        }
    }
    sum
}
pub fn istore_count_items(list: &[Istore], type_: i16, stop: i32) -> i32 {
    let mut count = 0;
    for store in list {
        if store.type_ == type_ {
            count += 1;
            if stop != 0 {
                return 1;
            }
        }
    }
    count
}
pub fn istore_count_object_items(
    obj: &Iobj,
    type_: i16,
    do_cont: i32,
    do_mesh: i32,
    stop: i32,
) -> i32 {
    let mut count = istore_count_items(&obj.store, type_, stop);
    if count != 0 && stop != 0 {
        return count;
    }
    if do_cont != 0 {
        for cont in &obj.cont {
            count += istore_count_items(&cont.store, type_, stop);
            if count != 0 && stop != 0 {
                return count;
            }
        }
    }
    if do_mesh != 0 {
        for mesh in &obj.mesh {
            count += istore_count_items(&mesh.store, type_, stop);
            if count != 0 && stop != 0 {
                return count;
            }
        }
    }
    count
}
pub fn istore_count_cont_surf_items(list: &[Istore], index: i32, surf_flag: i32) -> i32 {
    list.iter()
        .take_while(|store| store.flags & ((1 << 4) | 3) == 0)
        .filter(|store| {
            (store.flags & (1 << 6) != 0) == (surf_flag != 0) && unsafe { store.index.i } == index
        })
        .count() as i32
}
pub fn istore_point_is_gap(list: &[Istore], index: i32) -> i32 {
    let (lookup, after) = istore_lookup(list, index);
    let Some(lookup) = lookup else {
        return 0;
    };
    for store in &list[lookup..after] {
        if store.type_ == 4 {
            return 1;
        }
    }
    0
}
pub fn istore_connect_number(list: &[Istore], index: i32) -> i32 {
    let (lookup, after) = istore_lookup(list, index);
    let Some(lookup) = lookup else {
        return -1;
    };
    for store in &list[lookup..after] {
        if store.type_ == 5 {
            return unsafe { store.value.i };
        }
    }
    -1
}
pub fn istore_add_min_max(list: &mut Vec<Istore>, type_: i16, min: f32, max: f32) -> i32 {
    if type_ < 11 || type_ > 21 || (type_ - 11) % 2 != 0 {
        return 1;
    }
    for store in list.iter_mut().rev() {
        if store.flags & ((1 << 4) | 3) == 0 {
            break;
        }
        if store.type_ == type_ && store.flags == ((1 << 4) | (1 << 2) | 1) {
            store.index = StoreUnion { f: min };
            store.value = StoreUnion { f: max };
            return 0;
        }
    }
    istore_insert(
        list,
        Istore {
            type_,
            flags: (1 << 4) | (1 << 2) | 1,
            index: StoreUnion { f: min },
            value: StoreUnion { f: max },
        },
    )
}
pub fn istore_find_add_min_max1(obj: &mut Iobj) -> i32 {
    istore_find_add_min_max(obj, 10)
}
pub fn istore_find_add_min_max(obj: &mut Iobj, type_: i16) -> i32 {
    if type_ < 10 || type_ > 20 || (type_ - 10) % 2 != 0 {
        return 2;
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for store in obj
        .store
        .iter()
        .chain(obj.cont.iter().flat_map(|cont| cont.store.iter()))
    {
        if store.type_ == type_ && store.flags & (1 << 5) == 0 {
            let value = unsafe { store.value.f };
            min = min.min(value);
            max = max.max(value)
        }
    }
    if min > max {
        -1
    } else {
        istore_add_min_max(&mut obj.store, type_ + 1, min, max)
    }
}
pub fn istore_get_min_max(list: &[Istore], type_: i16, min: &mut f32, max: &mut f32) -> i32 {
    for store in list.iter().rev() {
        if store.flags & ((1 << 4) | 3) == 0 {
            return 0;
        }
        if store.type_ == type_ && store.flags & 3 == 1 {
            *min = unsafe { store.index.f };
            *max = unsafe { store.value.f };
            return 1;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_store_order_lookup_and_revert() {
        let mut list = Vec::new();
        istore_insert(
            &mut list,
            Istore {
                index: StoreUnion { i: 3 },
                ..Istore::default()
            },
        );
        istore_insert(
            &mut list,
            Istore {
                index: StoreUnion { i: 2 },
                ..Istore::default()
            },
        );
        istore_insert(
            &mut list,
            Istore {
                index: StoreUnion { i: 2 },
                flags: 1 << 5,
                ..Istore::default()
            },
        );
        istore_insert(
            &mut list,
            Istore {
                index: StoreUnion { i: 0 },
                flags: 1 << 4,
                ..Istore::default()
            },
        );
        assert_eq!(unsafe { list[0].index.i }, 2);
        assert_ne!(list[0].flags & (1 << 5), 0);
        assert_eq!(unsafe { list[1].index.i }, 2);
        assert_eq!(istore_lookup(&list, 2), (Some(0), 2));
        assert_ne!(list.last().unwrap().flags & (1 << 4), 0);
    }
    #[test]
    fn point_gap_and_connect_number_follow_source_index_lookup() {
        let list = [
            Istore {
                type_: 4,
                index: StoreUnion { i: 2 },
                ..Default::default()
            },
            Istore {
                type_: 5,
                index: StoreUnion { i: 2 },
                value: StoreUnion { i: 12 },
                ..Default::default()
            },
        ];
        assert_eq!(istore_point_is_gap(&list, 2), 1);
        assert_eq!(istore_point_is_gap(&list, 3), 0);
        assert_eq!(istore_connect_number(&list, 2), 12);
        assert_eq!(istore_connect_number(&list, 3), -1);
    }
    #[test]
    fn source_counts_and_minmax_span_object_contour_mesh_stores() {
        let value = Istore {
            type_: 10,
            value: StoreUnion { f: 3. },
            ..Istore::default()
        };
        let mut object = Iobj {
            store: vec![value],
            cont: vec![crate::imod::libimod::imodel::Icont {
                store: vec![Istore {
                    type_: 10,
                    value: StoreUnion { f: 7. },
                    ..Istore::default()
                }],
                ..Default::default()
            }],
            mesh: vec![crate::imod::libimod::imodel::Imesh {
                store: vec![Istore {
                    type_: 10,
                    ..Istore::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(istore_count_object_items(&object, 10, 1, 1, 0), 3);
        assert_eq!(istore_find_add_min_max1(&mut object), 0);
        let mut min = 0.;
        let mut max = 0.;
        assert_eq!(istore_get_min_max(&object.store, 11, &mut min, &mut max), 1);
        assert_eq!((min, max), (3., 7.));
        let surface = Istore {
            index: StoreUnion { i: 4 },
            flags: 1 << 6,
            ..Istore::default()
        };
        assert_eq!(istore_count_cont_surf_items(&[surface], 4, 1), 1);
    }
    #[test]
    fn first_skip_and_trans_state_match_source_rules() {
        let list = [
            Istore {
                index: StoreUnion { i: 2 },
                ..Default::default()
            },
            Istore {
                index: StoreUnion { i: 5 },
                ..Default::default()
            },
            Istore {
                flags: 1 << 4,
                ..Default::default()
            },
        ];
        assert_eq!(istore_first_change_index(&list), 2);
        assert_eq!(istore_skip_to_index(&list, 3), 5);
        assert_eq!(istore_skip_to_index(&list, 6), -1);
        let trans = [
            Istore {
                type_: 3,
                value: StoreUnion { i: 2 },
                ..Default::default()
            },
            Istore {
                type_: 3,
                flags: 1 << 5,
                value: StoreUnion { i: 0 },
                ..Default::default()
            },
        ];
        assert_eq!(istore_trans_state_matches(&trans, 1), 1);
        assert_eq!(istore_trans_state_matches(&trans, 0), 0);
        assert_eq!(istore_trans_state_matches(&trans, 2), 0);
    }
    #[test]
    fn next_change_groups_index_and_tracks_revert_state() {
        let list = [
            Istore {
                type_: 3,
                index: StoreUnion { i: 2 },
                value: StoreUnion { i: 4 },
                ..Default::default()
            },
            Istore {
                type_: 3,
                index: StoreUnion { i: 3 },
                flags: 1 << 5,
                ..Default::default()
            },
            Istore {
                flags: 1 << 4,
                ..Default::default()
            },
        ];
        let def = DrawProps {
            trans: 1,
            ..Default::default()
        };
        let mut props = def;
        let (mut cursor, mut state, mut changes) = (0, 0, 0);
        assert_eq!(
            istore_next_change(
                &list,
                &mut cursor,
                &def,
                &mut props,
                &mut state,
                &mut changes
            ),
            3
        );
        assert_eq!((props.trans, state, changes), (4, 4, 4));
        assert_eq!(
            istore_next_change(
                &list,
                &mut cursor,
                &def,
                &mut props,
                &mut state,
                &mut changes
            ),
            -1
        );
        assert_eq!(props.trans, 1);
    }
    #[test]
    fn list_point_props_resets_transient_properties_off_change_point() {
        let list = [
            Istore {
                type_: 4,
                index: StoreUnion { i: 2 },
                ..Default::default()
            },
            Istore {
                type_: 5,
                index: StoreUnion { i: 2 },
                value: StoreUnion { i: 9 },
                ..Default::default()
            },
            Istore {
                type_: 10,
                index: StoreUnion { i: 2 },
                value: StoreUnion { f: 3. },
                ..Default::default()
            },
        ];
        let cont = DrawProps::default();
        let mut point = DrawProps::default();
        assert_ne!(istore_list_point_props(&list, &cont, &mut point, 2) & 8, 0);
        assert_eq!((point.gap, point.connect, point.value1), (1, 9, 3.));
        let mut later = DrawProps::default();
        assert_eq!(
            istore_list_point_props(&list, &cont, &mut later, 3) & (8 | 16 | 512),
            0
        );
        assert_eq!((later.gap, later.connect, later.value1), (0, 0, 0.));
    }
    #[test]
    fn insert_end_and_clear_change_follow_source_change_sequence() {
        let mut list = Vec::new();
        let start = Istore {
            type_: 3,
            index: StoreUnion { i: 2 },
            value: StoreUnion { i: 7 },
            ..Default::default()
        };
        assert_eq!(istore_insert_change(&mut list, start), 0);
        assert_eq!(istore_end_change(&mut list, 3, 5), 0);
        assert_eq!(list.len(), 2);
        assert_eq!(unsafe { list[0].index.i }, 2);
        assert_ne!(list[1].flags & (1 << 5), 0);
        assert_eq!(unsafe { list[1].index.i }, 5);
        assert_eq!(istore_clear_change(&mut list, 3, 3), 0);
        assert!(list.is_empty());
        assert_eq!(istore_clear_change(&mut list, 3, 3), 1);
    }

    #[test]
    fn clear_range_and_one_index_items_follow_source_matching_rules() {
        let mut changes = vec![
            Istore {
                type_: 6,
                index: StoreUnion { i: 1 },
                value: StoreUnion { i: 4 },
                ..Default::default()
            },
            Istore {
                type_: 6,
                index: StoreUnion { i: 4 },
                flags: 1 << 5,
                ..Default::default()
            },
            Istore {
                type_: 6,
                index: StoreUnion { i: 7 },
                value: StoreUnion { i: 8 },
                ..Default::default()
            },
        ];
        istore_clear_range(&mut changes, 6, 2, 4);
        assert_eq!(changes.len(), 1);
        assert_eq!(unsafe { changes[0].index.i }, 7);

        let mut single = Vec::new();
        let point = Istore {
            type_: 9,
            flags: 1 << 7,
            index: StoreUnion { i: 4 },
            value: StoreUnion { i: 2 },
        };
        assert_eq!(istore_add_one_index_item(&mut single, point), 0);
        assert_eq!(
            istore_add_one_index_item(
                &mut single,
                Istore {
                    value: StoreUnion { i: 9 },
                    ..point
                },
            ),
            0
        );
        assert_eq!(single.len(), 1);
        assert_eq!(unsafe { single[0].value.i }, 9);
        let surface = Istore {
            flags: (1 << 7) | (1 << 6),
            value: StoreUnion { i: 3 },
            ..point
        };
        assert_eq!(istore_add_one_index_item(&mut single, surface), 0);
        assert_eq!(single.len(), 2);
        assert_eq!(istore_clear_one_index_item(&mut single, 9, 4, 0), 0);
        assert_eq!(single.len(), 1);
        assert_eq!(istore_clear_one_index_item(&mut single, 9, 4, 1), 0);
        assert!(single.is_empty());
        assert_eq!(istore_clear_one_index_item(&mut single, 9, 4, 0), 1);
    }

    #[test]
    fn generate_items_and_point_items_use_source_property_masks() {
        let props = DrawProps {
            red: 1.,
            green: 0.5,
            blue: 0.,
            fill_red: 0.,
            fill_green: 1.,
            fill_blue: 0.25,
            trans: 7,
            linewidth: 3,
            value1: 1.25,
            ..Default::default()
        };
        let all = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 5) | (1 << 9);
        let mut list = Vec::new();
        assert_eq!(istore_generate_items(&mut list, &props, all, 8, all), 0);
        assert_eq!(list.len(), 5);
        assert_eq!(
            list.iter().map(|item| item.type_).collect::<Vec<_>>(),
            [1, 2, 3, 6, 10]
        );
        assert_eq!(unsafe { list[0].value.b }, [255, 127, 0, 0]);
        assert_eq!(unsafe { list[4].value.f }, 1.25);

        let contour_list = [Istore {
            type_: 3,
            index: StoreUnion { i: 2 },
            value: StoreUnion { i: 9 },
            ..Default::default()
        }];
        let mut mesh_list = Vec::new();
        assert_eq!(
            istore_gen_point_items(&contour_list, &props, 0, 2, &mut mesh_list, 4, 1 << 2,),
            0
        );
        assert_eq!(mesh_list.len(), 1);
        assert_eq!(unsafe { mesh_list[0].index.i }, 4);
        assert_eq!(unsafe { mesh_list[0].value.i }, 9);
    }

    #[test]
    fn break_find_shift_and_delete_follow_source_index_rules() {
        let mut changes = vec![Istore {
            type_: 3,
            index: StoreUnion { i: 0 },
            value: StoreUnion { i: 4 },
            ..Default::default()
        }];
        assert_eq!(istore_break_changes(&mut changes, 2, 5), 0);
        assert_eq!(changes.len(), 3);
        assert_eq!(unsafe { changes[1].index.i }, 2);
        assert_ne!(changes[1].flags & (1 << 5), 0);
        assert_eq!(unsafe { changes[2].index.i }, 2);
        assert_eq!(istore_find_break(&changes, 2), 2);

        istore_shift_index(&mut changes, 2, -1, 1);
        assert_eq!(unsafe { changes[1].index.i }, 3);
        assert_eq!(unsafe { changes[2].index.i }, 3);
        assert_eq!(istore_delete_point(&mut changes, 3, 6), 0);
        assert_eq!(changes.len(), 2);
        assert_eq!(unsafe { changes[1].index.i }, 3);
        assert_ne!(changes[1].flags & (1 << 5), 0);

        let mut cont_surf = vec![
            Istore {
                type_: 9,
                index: StoreUnion { i: 1 },
                ..Default::default()
            },
            Istore {
                type_: 9,
                flags: 1 << 6,
                index: StoreUnion { i: 1 },
                ..Default::default()
            },
            Istore {
                type_: 9,
                index: StoreUnion { i: 2 },
                ..Default::default()
            },
        ];
        istore_delete_cont_surf(&mut cont_surf, 1, 1);
        assert_eq!(cont_surf.len(), 2);
        istore_delete_cont_surf(&mut cont_surf, 1, 0);
        assert_eq!(cont_surf.len(), 1);
        assert_eq!(unsafe { cont_surf[0].index.i }, 1);
    }

    #[test]
    fn extract_copy_clean_and_invert_follow_source_store_ranges() {
        let changes = vec![
            Istore {
                type_: 3,
                index: StoreUnion { i: 0 },
                value: StoreUnion { i: 5 },
                ..Default::default()
            },
            Istore {
                type_: 3,
                flags: 1 << 5,
                index: StoreUnion { i: 5 },
                ..Default::default()
            },
            Istore {
                type_: 11,
                flags: 1 << 4,
                ..Default::default()
            },
        ];
        let mut extracted = Vec::new();
        assert_eq!(
            istore_extract_changes(&changes, &mut extracted, 2, 4, 0, 6),
            0
        );
        assert_eq!(extracted.len(), 2);
        assert_eq!(unsafe { extracted[0].index.i }, 0);
        assert_eq!(unsafe { extracted[1].index.i }, 3);

        let mut copied = Vec::new();
        assert_eq!(istore_copy_non_index(&changes, &mut copied), 0);
        assert_eq!(copied.len(), 1);
        let mut point_and_surface = vec![
            Istore {
                type_: 9,
                index: StoreUnion { i: 2 },
                ..Default::default()
            },
            Istore {
                type_: 9,
                flags: 1 << 6,
                index: StoreUnion { i: 2 },
                ..Default::default()
            },
        ];
        assert_eq!(
            istore_copy_cont_surf_items(&point_and_surface, &mut copied, 2, 7, 1),
            0
        );
        assert!(
            copied
                .iter()
                .any(|store| { store.flags & (1 << 6) != 0 && unsafe { store.index.i } == 7 })
        );
        istore_clean_ends(&mut point_and_surface);
        assert_eq!(point_and_surface.len(), 2);

        let mut gap = vec![Istore {
            type_: 4,
            flags: 1 << 7,
            index: StoreUnion { i: 0 },
            ..Default::default()
        }];
        assert_eq!(istore_invert(&mut gap, 4), 0);
        assert_eq!(unsafe { gap[0].index.i }, 2);
    }

    #[test]
    fn break_contour_splits_store_at_requested_point_range() {
        let mut contour = Icont {
            pts: vec![
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
            ],
            store: vec![
                Istore {
                    type_: 3,
                    index: StoreUnion { i: 0 },
                    value: StoreUnion { i: 2 },
                    ..Default::default()
                },
                Istore {
                    type_: 3,
                    flags: 1 << 5,
                    index: StoreUnion { i: 4 },
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut new_contour = Icont {
            pts: vec![Default::default(), Default::default()],
            ..Default::default()
        };
        assert_eq!(
            istore_break_contour(&mut contour, &mut new_contour, 1, 2),
            0
        );
        assert_eq!(new_contour.store.len(), 2);
        assert_eq!(unsafe { new_contour.store[0].index.i }, 0);
        assert_eq!(unsafe { new_contour.store[1].index.i }, 2);
        assert_eq!(contour.store.len(), 4);
    }

    #[test]
    fn binary_store_roundtrip_preserves_source_twelve_byte_records() {
        let path = std::env::temp_dir().join(format!("imod-rs-istore-{}.bin", std::process::id()));
        let mut file = File::create(&path).unwrap();
        let source = vec![
            Istore {
                type_: 1,
                flags: 1 | (3 << 2),
                index: StoreUnion { f: 1.5 },
                value: StoreUnion { b: [1, 2, 3, 4] },
            },
            Istore {
                type_: 10,
                flags: 2,
                index: StoreUnion { s: [-2, 9] },
                value: StoreUnion { i: 17 },
            },
        ];
        assert_eq!(imod_write_store(&source, 0x5354_4f52, &mut file), 0);
        assert_eq!(file.metadata().unwrap().len(), 32);
        drop(file);
        let mut file = File::open(&path).unwrap();
        file.seek(SeekFrom::Start(4)).unwrap();
        let mut error = -1;
        let read = imod_read_store(&mut file, &mut error);
        assert_eq!(error, 0);
        let read = read.unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].flags, source[0].flags);
        assert_eq!(unsafe { read[0].index.f }, 1.5);
        assert_eq!(unsafe { read[0].value.b }, [1, 2, 3, 4]);
        assert_eq!(unsafe { read[1].index.s }, [-2, 9]);
        assert_eq!(unsafe { read[1].value.i }, 17);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn checksum_retain_and_object_iteration_follow_source_flags() {
        let list = vec![
            Istore {
                type_: 3,
                index: StoreUnion { i: 2 },
                value: StoreUnion { i: 4 },
                ..Default::default()
            },
            Istore {
                type_: 4,
                flags: 1 << 7,
                index: StoreUnion { i: 3 },
                ..Default::default()
            },
            Istore {
                type_: 3,
                flags: 1 << 5,
                index: StoreUnion { i: 5 },
                ..Default::default()
            },
            Istore {
                type_: 9,
                flags: 1 << 6,
                index: StoreUnion { i: 7 },
                ..Default::default()
            },
        ];
        assert_eq!(istore_checksum(&list[..1]), 9.);
        assert_eq!(istore_retain_point(&list, 2), 1);
        assert_eq!(istore_retain_point(&list, 4), 1);
        assert_eq!(istore_retain_point(&list, 0), 0);
        let mut cursor = 0;
        assert_eq!(
            istore_next_obj_item(&list, 2, 7, 1, &mut cursor).map(|store| store.type_),
            Some(3)
        );
        assert_eq!(
            istore_next_obj_item(&list, 2, 7, 0, &mut cursor).map(|store| store.type_),
            Some(9)
        );
    }

    #[test]
    fn draw_property_paths_apply_surface_contour_point_and_revert_items() {
        let mut object = Iobj {
            red: 0.1,
            green: 0.2,
            blue: 0.3,
            symbol: 4,
            store: vec![
                Istore {
                    type_: 1,
                    flags: 1 << 6,
                    index: StoreUnion { i: 3 },
                    value: StoreUnion {
                        b: [128, 64, 32, 0],
                    },
                },
                Istore {
                    type_: 8,
                    index: StoreUnion { i: 0 },
                    value: StoreUnion { i: -3 },
                    ..Default::default()
                },
            ],
            cont: vec![Icont {
                surf: 3,
                store: vec![
                    Istore {
                        type_: 2,
                        index: StoreUnion { i: 1 },
                        value: StoreUnion {
                            b: [0, 255, 128, 0],
                        },
                        ..Default::default()
                    },
                    Istore {
                        type_: 2,
                        flags: 1 << 5,
                        index: StoreUnion { i: 2 },
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        istore_sort(&mut object.store);
        let mut cont = DrawProps::default();
        let mut point = DrawProps::default();
        assert_eq!(
            istore_point_draw_props(&object, &mut cont, &mut point, 0, 1),
            2
        );
        assert_eq!(
            (cont.red, cont.green, cont.blue),
            (128. / 255., 64. / 255., 32. / 255.)
        );
        assert_eq!((cont.symtype, cont.symflags & 1), (2, 1));
        assert_eq!(
            (point.fill_red, point.fill_green, point.fill_blue),
            (0., 1., 128. / 255.)
        );
        assert_eq!(
            istore_point_draw_props(&object, &mut cont, &mut point, 0, 2),
            0
        );
        assert_eq!(
            (point.fill_red, point.fill_green, point.fill_blue),
            (0., 0., 0.)
        );
    }
}
pub fn istore_retain_point(list: &[Istore], index: i32) -> i32 {
    if list.is_empty() {
        return 0;
    }
    let (lookup, after) = istore_lookup(list, index);
    if lookup.is_some() {
        return 1;
    }
    for store in list.iter().skip(after) {
        if store.flags & ((1 << 4) | 3) != 0 || unsafe { store.index.i } > index + 1 {
            break;
        }
        if store.flags & (1 << 5) != 0 {
            return 1;
        }
    }
    for store in list[..after].iter().rev() {
        if unsafe { store.index.i } < index - 1 {
            break;
        }
        if store.type_ == 4 {
            return 1;
        }
    }
    0
}
pub fn istore_insert_change(list: &mut Vec<Istore>, store: Istore) -> i32 {
    let (found, after) = istore_lookup(list, unsafe { store.index.i });
    if let Some(start) = found {
        let mut i = start;
        while i < after && i < list.len() {
            if list[i].type_ == store.type_ {
                list.remove(i);
            } else {
                i += 1
            }
        }
    }
    let before = list.iter().rev().find(
        |item| unsafe { item.index.i } < unsafe { store.index.i } && item.type_ == store.type_,
    );
    if store.flags & (1 << 7) == 0
        && before.is_some_and(|item| {
            item.flags & (1 << 5) == 0 && unsafe { item.value.i } == unsafe { store.value.i }
        })
    {
        return 0;
    }
    istore_insert(list, store)
}
pub fn istore_end_change(list: &mut Vec<Istore>, type_: i16, index: i32) -> i32 {
    if list.is_empty() {
        return 1;
    }
    let (lookup, mut after) = istore_lookup(list, index);
    let mut i = after;
    while i < list.len() {
        if list[i].flags & ((1 << 4) | 3) != 0 {
            break;
        }
        if list[i].type_ == type_ {
            if list[i].flags & (1 << 5) == 0 {
                break;
            }
            list.remove(i);
            continue;
        }
        i += 1;
    }
    let mut need_end = true;
    if let Some(lookup) = lookup {
        let mut i = lookup;
        while i < after {
            if list[i].type_ == type_ {
                need_end = false;
                if list[i].flags & (1 << 5) == 0 {
                    for previous in (0..i).rev() {
                        if list[previous].type_ == type_ {
                            if list[previous].flags & (1 << 5) == 0 {
                                need_end = true;
                            }
                            break;
                        }
                    }
                    list.remove(i);
                    after -= 1;
                    continue;
                }
            }
            i += 1;
        }
    }
    if need_end {
        istore_insert(
            list,
            Istore {
                type_,
                flags: 1 << 5,
                index: StoreUnion { i: index },
                ..Default::default()
            },
        )
    } else {
        0
    }
}
pub fn istore_clear_change(list: &mut Vec<Istore>, type_: i16, index: i32) -> i32 {
    if list.is_empty() {
        return 1;
    }
    let (_, after) = istore_lookup(list, index - 1);
    let mut i = after;
    while i < list.len() {
        if list[i].flags & ((1 << 4) | 3) != 0 {
            break;
        }
        if list[i].type_ == type_ {
            let flags = list[i].flags;
            list.remove(i);
            if flags & (1 << 5) != 0 {
                break;
            }
            continue;
        }
        i += 1;
    }
    for i in (0..after.min(list.len())).rev() {
        if list[i].type_ == type_ {
            if list[i].flags & (1 << 5) != 0 {
                break;
            }
            list.remove(i);
        }
    }
    0
}
pub fn istore_clear_range(list: &mut Vec<Istore>, type_: i16, start: i32, end: i32) {
    if list.is_empty() {
        return;
    }
    let (_, after) = istore_lookup(list, start);
    let mut has_change = false;
    for store in list.iter().skip(after) {
        if store.flags & ((1 << 4) | 3) != 0 || unsafe { store.index.i } > end {
            break;
        }
        if store.type_ == type_ {
            has_change = true;
            break;
        }
    }
    if !has_change {
        for store in list[..after.min(list.len())].iter().rev() {
            if store.type_ == type_ {
                has_change = store.flags & (1 << 5) == 0;
                break;
            }
        }
    }
    if !has_change {
        return;
    }
    let mut ended = false;
    let mut i = after;
    while i < list.len() {
        if list[i].flags & ((1 << 4) | 3) != 0 {
            break;
        }
        if list[i].type_ == type_ {
            let flags = list[i].flags;
            let index = unsafe { list[i].index.i };
            if ended && index > end {
                break;
            }
            list.remove(i);
            ended = flags & (1 << 5) != 0;
            if ended && index >= end {
                break;
            }
            continue;
        }
        i += 1;
    }
    for i in (0..after.min(list.len())).rev() {
        if list[i].type_ == type_ {
            if list[i].flags & (1 << 5) != 0 {
                break;
            }
            list.remove(i);
        }
    }
}
pub fn istore_add_one_index_item(list: &mut Vec<Istore>, store: Istore) -> i32 {
    let (lookup, after) = istore_lookup(list, unsafe { store.index.i });
    if let Some(lookup) = lookup {
        for item in &mut list[lookup..after] {
            if item.type_ == store.type_ && (item.flags & (1 << 6)) == (store.flags & (1 << 6)) {
                *item = store;
                return 0;
            }
        }
    }
    istore_insert(list, store)
}
pub fn istore_clear_one_index_item(
    list: &mut Vec<Istore>,
    type_: i16,
    index: i32,
    surf_flag: i32,
) -> i32 {
    if list.is_empty() {
        return 1;
    }
    let (lookup, after) = istore_lookup(list, index);
    let Some(lookup) = lookup else {
        return -1;
    };
    let surf_flag = if surf_flag != 0 { 1 << 6 } else { 0 };
    for item in lookup..after {
        if list[item].type_ == type_ && (list[item].flags & (1 << 6)) == surf_flag {
            list.remove(item);
            return 0;
        }
    }
    -1
}
pub fn istore_generate_items(
    list: &mut Vec<Istore>,
    props: &DrawProps,
    state_flags: i32,
    index: i32,
    gen_flags: i32,
) -> i32 {
    if gen_flags & state_flags & (1 << 0) != 0
        && istore_insert(
            list,
            Istore {
                type_: 1,
                flags: 1 << 3,
                index: StoreUnion { i: index },
                value: StoreUnion {
                    b: [
                        (255. * props.red) as u8,
                        (255. * props.green) as u8,
                        (255. * props.blue) as u8,
                        0,
                    ],
                },
            },
        ) != 0
    {
        return 1;
    }
    if gen_flags & state_flags & (1 << 1) != 0
        && istore_insert(
            list,
            Istore {
                type_: 2,
                flags: 1 << 3,
                index: StoreUnion { i: index },
                value: StoreUnion {
                    b: [
                        (255. * props.fill_red) as u8,
                        (255. * props.fill_green) as u8,
                        (255. * props.fill_blue) as u8,
                        0,
                    ],
                },
            },
        ) != 0
    {
        return 1;
    }
    if gen_flags & state_flags & (1 << 2) != 0
        && istore_insert(
            list,
            Istore {
                type_: 3,
                index: StoreUnion { i: index },
                value: StoreUnion { i: props.trans },
                ..Default::default()
            },
        ) != 0
    {
        return 1;
    }
    if gen_flags & state_flags & (1 << 5) != 0
        && istore_insert(
            list,
            Istore {
                type_: 6,
                index: StoreUnion { i: index },
                value: StoreUnion { i: props.linewidth },
                ..Default::default()
            },
        ) != 0
    {
        return 1;
    }
    if gen_flags & state_flags & (1 << 9) != 0
        && istore_insert(
            list,
            Istore {
                type_: 10,
                flags: 1 << 2,
                index: StoreUnion { i: index },
                value: StoreUnion { f: props.value1 },
            },
        ) != 0
    {
        return 1;
    }
    0
}
pub fn istore_gen_point_items(
    contour_list: &[Istore],
    contour_props: &DrawProps,
    contour_state: i32,
    point_index: i32,
    mesh_list: &mut Vec<Istore>,
    mesh_index: i32,
    gen_flags: i32,
) -> i32 {
    let mut point_props = DrawProps::default();
    let state_flags =
        istore_list_point_props(contour_list, contour_props, &mut point_props, point_index);
    istore_generate_items(
        mesh_list,
        &point_props,
        state_flags | contour_state,
        mesh_index,
        gen_flags,
    )
}
pub fn istore_break_changes(list: &mut Vec<Istore>, index: i32, psize: i32) -> i32 {
    if list.is_empty() {
        return 0;
    }
    let mut cur_start = 0;
    while cur_start < list.len() {
        let current = list[cur_start];
        if current.flags & ((1 << 4) | 3) != 0 || unsafe { current.index.i } >= index {
            break;
        }
        if current.flags & ((1 << 7) | (1 << 5)) != 0 {
            cur_start += 1;
            continue;
        }
        let mut store = current;
        let mut point_revert = index + 1;
        let mut need_restart = index < psize;
        for next in list.iter().skip(cur_start + 1) {
            if next.flags & ((1 << 4) | 3) != 0 || unsafe { next.index.i } > index {
                break;
            }
            if store.type_ == next.type_ {
                if next.flags & (1 << 5) != 0 {
                    point_revert = unsafe { next.index.i };
                    break;
                }
                if unsafe { next.index.i } == index {
                    need_restart = false;
                } else {
                    store = *next;
                }
            }
        }
        if point_revert > index {
            let mut store_end = store;
            store_end.flags |= 1 << 5;
            store_end.index = StoreUnion { i: index };
            if istore_insert(list, store_end) != 0 {
                return 1;
            }
            if need_restart {
                store.index = StoreUnion { i: index };
                if istore_insert(list, store) != 0 {
                    return 1;
                }
            }
        }
        cur_start += 1;
    }
    0
}
pub fn istore_find_break(list: &[Istore], index: i32) -> i32 {
    let (lookup, after) = istore_lookup(list, index);
    let Some(lookup) = lookup else {
        return after as i32;
    };
    for (item_index, item) in list[lookup..after].iter().enumerate() {
        if item.flags & (1 << 5) == 0 {
            return (lookup + item_index) as i32;
        }
    }
    after as i32
}
pub fn istore_shift_index(list: &mut Vec<Istore>, point_index: i32, start_scan: i32, amount: i32) {
    if list.is_empty() {
        return;
    }
    let start_scan = if start_scan < 0 {
        let (lookup, after) = istore_lookup(list, point_index);
        lookup.unwrap_or(after)
    } else {
        start_scan as usize
    };
    for store in list.iter_mut().skip(start_scan) {
        if store.flags & ((1 << 4) | 3) != 0 {
            break;
        }
        if store.flags & (1 << 6) == 0 && unsafe { store.index.i } >= point_index {
            store.index = StoreUnion {
                i: unsafe { store.index.i } + amount,
            };
        }
    }
    istore_sort(list);
}
pub fn istore_delete_point(list: &mut Vec<Istore>, index: i32, psize: i32) -> i32 {
    if list.is_empty() {
        return 0;
    }
    let (lookup, after) = istore_lookup(list, index);
    let Some(lookup) = lookup else {
        istore_shift_index(list, index + 1, after as i32, -1);
        return 0;
    };
    if index < psize - 1 {
        let current = list[lookup..after].to_vec();
        for store in current {
            if store.flags & (1 << 7) != 0 {
                continue;
            }
            let mut need_move = true;
            let mut remove_end = None;
            for (offset, next) in list.iter().enumerate().skip(after) {
                if next.flags & ((1 << 4) | 3) != 0 || unsafe { next.index.i } > index + 1 {
                    break;
                }
                if next.type_ == store.type_ {
                    need_move = false;
                    if next.flags & (1 << 5) != 0 {
                        let mut delete_end = true;
                        for previous in list[..lookup].iter().rev() {
                            if previous.type_ == store.type_ {
                                if previous.flags & (1 << 5) == 0 {
                                    delete_end = false;
                                }
                                break;
                            }
                        }
                        if delete_end {
                            remove_end = Some(offset);
                        }
                    }
                    break;
                }
            }
            if let Some(remove_end) = remove_end {
                list.remove(remove_end);
            }
            if need_move {
                let mut moved = store;
                moved.index = StoreUnion {
                    i: unsafe { moved.index.i } + 1,
                };
                if istore_insert(list, moved) != 0 {
                    return 1;
                }
            }
        }
    }
    let (_, after) = istore_lookup(list, index);
    let (lookup, _) = istore_lookup(list, index);
    if let Some(lookup) = lookup {
        list.drain(lookup..after);
        istore_shift_index(list, index + 1, lookup as i32, -1);
    }
    0
}
pub fn istore_delete_cont_surf(list: &mut Vec<Istore>, index: i32, surf_flag: i32) {
    if list.is_empty() {
        return;
    }
    let surf_flag = if surf_flag != 0 { 1 << 6 } else { 0 };
    let (lookup, mut after) = istore_lookup(list, index);
    if let Some(mut lookup) = lookup {
        while lookup < after {
            if list[lookup].flags & (1 << 6) == surf_flag {
                list.remove(lookup);
                after -= 1;
            } else {
                lookup += 1;
            }
        }
    }
    if surf_flag == 0 {
        istore_shift_index(list, index + 1, after as i32, -1);
    }
}
pub fn istore_break_contour(cont: &mut Icont, ncont: &mut Icont, p1: i32, mut p2: i32) -> i32 {
    if cont.store.is_empty() {
        return 0;
    }
    let psize = cont.pts.len() as i32;
    if p2 < 0 {
        p2 = psize - 1;
    }
    if psize == 0 || p1 < 0 || p1 >= psize || p2 >= psize || p2 < p1 {
        return 1;
    }
    let mut first = Vec::new();
    if istore_extract_changes(&cont.store, &mut first, 0, p1 - 1, 0, psize) != 0 {
        return 1;
    }
    let mut new_store = Vec::new();
    if istore_extract_changes(&cont.store, &mut new_store, p1, p2, 0, psize) != 0 {
        return 1;
    }
    ncont.store = new_store;
    if istore_extract_changes(&cont.store, &mut first, p2 + 1, psize - 1, p1, psize) != 0 {
        return 1;
    }
    cont.store = first;
    0
}
pub fn istore_invert(list: &mut Vec<Istore>, psize: i32) -> i32 {
    if psize < 2 || list.is_empty() {
        return 0;
    }
    if istore_break_changes(list, psize, psize) != 0 {
        return 1;
    }
    let source = list.clone();
    let mut inverted = Vec::new();
    let mut cur_start = 0;
    while cur_start < source.len() {
        let current = source[cur_start];
        if current.flags & ((1 << 4) | 3) != 0 {
            if istore_insert(&mut inverted, current) != 0 {
                return 1;
            }
            cur_start += 1;
            continue;
        }
        if current.flags & (1 << 7) != 0 {
            let mut store = current;
            store.index = StoreUnion {
                i: psize - 1 - unsafe { store.index.i },
            };
            if store.type_ == 4 {
                store.index = StoreUnion {
                    i: (unsafe { store.index.i } - 1 + psize) % psize,
                };
            }
            if istore_insert(&mut inverted, store) != 0 {
                return 1;
            }
            cur_start += 1;
            continue;
        }
        let mut store = current;
        store.index = StoreUnion {
            i: psize - unsafe { store.index.i },
        };
        store.flags |= 1 << 5;
        if istore_insert(&mut inverted, store) != 0 {
            return 1;
        }
        let mut next = cur_start + 1;
        while next < source.len() {
            let successor = source[next];
            if current.type_ == successor.type_ {
                let mut start = current;
                start.index = StoreUnion {
                    i: psize - unsafe { successor.index.i },
                };
                if istore_insert(&mut inverted, start) != 0 {
                    return 1;
                }
                if successor.flags & (1 << 5) != 0 {
                    break;
                }
            }
            next += 1;
        }
        cur_start += 1;
    }
    *list = inverted;
    0
}
pub fn istore_clean_ends(list: &mut Vec<Istore>) {
    let mut i = 0;
    while i < list.len() {
        if list[i].flags & ((1 << 4) | 3) != 0 {
            return;
        }
        if list[i].flags & (1 << 5) != 0 {
            let mut remove = false;
            for next in list.iter().skip(i + 1) {
                if next.flags & ((1 << 4) | 3) != 0
                    || unsafe { next.index.i } != unsafe { list[i].index.i }
                {
                    break;
                }
                if next.type_ == list[i].type_ {
                    remove = true;
                    break;
                }
            }
            if remove {
                list.remove(i);
                continue;
            }
        }
        i += 1;
    }
}
pub fn istore_extract_changes(
    old_list: &[Istore],
    new_list: &mut Vec<Istore>,
    index_start: i32,
    index_end: i32,
    new_start: i32,
    psize: i32,
) -> i32 {
    if old_list.is_empty() || index_start > index_end {
        return 0;
    }
    let mut temporary = old_list.to_vec();
    if index_start != 0 && istore_break_changes(&mut temporary, index_start, psize) != 0 {
        return 1;
    }
    if istore_break_changes(&mut temporary, index_end + 1, psize) != 0 {
        return 1;
    }
    let after_first = istore_find_break(&temporary, index_start) as usize;
    let after_last = istore_find_break(&temporary, index_end + 1) as usize;
    for store in &temporary[after_first..after_last] {
        let mut store = *store;
        store.index = StoreUnion {
            i: unsafe { store.index.i } + new_start - index_start,
        };
        if istore_insert(new_list, store) != 0 {
            return 1;
        }
    }
    0
}
pub fn istore_copy_non_index(old_list: &[Istore], new_list: &mut Vec<Istore>) -> i32 {
    let (_, after) = istore_lookup(old_list, i32::MAX);
    for store in old_list.iter().skip(after) {
        if istore_insert(new_list, *store) != 0 {
            return 1;
        }
    }
    0
}
pub fn istore_copy_cont_surf_items(
    old_list: &[Istore],
    new_list: &mut Vec<Istore>,
    index_from: i32,
    index_to: i32,
    surf_flag: i32,
) -> i32 {
    let surf_flag = if surf_flag != 0 { 1 << 6 } else { 0 };
    let (lookup, after) = istore_lookup(old_list, index_from);
    if let Some(lookup) = lookup {
        for store in &old_list[lookup..after] {
            if store.flags & (1 << 6) == surf_flag {
                let mut store = *store;
                store.index = StoreUnion { i: index_to };
                if istore_insert(new_list, store) != 0 {
                    return 1;
                }
            }
        }
    }
    0
}
pub fn istore_default_draw_props(obj: &Iobj, props: &mut DrawProps) {
    props.red = obj.red;
    props.green = obj.green;
    props.blue = obj.blue;
    props.fill_red = obj.fillred as f32 / 255.;
    props.fill_green = obj.fillgreen as f32 / 255.;
    props.fill_blue = obj.fillblue as f32 / 255.;
    props.trans = obj.trans as i32;
    props.linewidth = obj.linewidth as i32;
    props.linewidth2 = obj.linewidth2 as i32;
    props.symtype = obj.symbol as i32;
    props.symflags = obj.symflags as i32;
    props.symsize = obj.symsize as i32;
    props.connect = 0;
    props.gap = 0;
    props.value1 = 0.;
    props.no_cap = 0;
}
pub fn istore_cont_surf_draw_props(
    list: &[Istore],
    def_props: &DrawProps,
    cont_props: &mut DrawProps,
    co: i32,
    surf: i32,
    cont_state: &mut i32,
    surf_state: &mut i32,
) -> i32 {
    *cont_props = *def_props;
    *cont_state = 0;
    *surf_state = 0;
    let surface_state = surf_state as *mut i32;
    let contour_state = cont_state as *mut i32;
    for (which, surface, out) in [(surf, true, surface_state), (co, false, contour_state)] {
        if which < 0 {
            continue;
        }
        let (found, after) = istore_lookup(list, which);
        if let Some(start) = found {
            for store in &list[start..after] {
                if (store.flags & (1 << 6) != 0) != surface {
                    continue;
                }
                match store.type_ {
                    1 => {
                        unsafe {
                            *out |= 1;
                        }
                        unsafe {
                            cont_props.red = store.value.b[0] as f32 / 255.;
                            cont_props.green = store.value.b[1] as f32 / 255.;
                            cont_props.blue = store.value.b[2] as f32 / 255.;
                        }
                    }
                    2 => {
                        unsafe {
                            *out |= 2;
                        }
                        unsafe {
                            cont_props.fill_red = store.value.b[0] as f32 / 255.;
                            cont_props.fill_green = store.value.b[1] as f32 / 255.;
                            cont_props.fill_blue = store.value.b[2] as f32 / 255.;
                        }
                    }
                    3 => {
                        unsafe {
                            *out |= 4;
                        }
                        cont_props.trans = unsafe { store.value.i };
                    }
                    4 => {
                        unsafe {
                            *out |= 8;
                        }
                        cont_props.gap = 1;
                    }
                    5 => {
                        unsafe {
                            *out |= 16;
                        }
                        cont_props.connect = unsafe { store.value.i };
                    }
                    6 => {
                        unsafe {
                            *out |= 32;
                        }
                        cont_props.linewidth = unsafe { store.value.i };
                    }
                    7 => {
                        unsafe {
                            *out |= 64;
                        }
                        cont_props.linewidth2 = unsafe { store.value.i };
                    }
                    9 => {
                        unsafe {
                            *out |= 256;
                        }
                        cont_props.symsize = unsafe { store.value.i };
                    }
                    8 => {
                        unsafe {
                            *out |= 128;
                        }
                        cont_props.symflags &= !1;
                        cont_props.symtype = unsafe { store.value.i };
                        if cont_props.symtype < 0 {
                            cont_props.symtype = -1 - cont_props.symtype;
                            cont_props.symflags |= 1;
                        }
                    }
                    10 => {
                        unsafe {
                            *out |= 512;
                        }
                        cont_props.value1 = unsafe { store.value.f };
                    }
                    24 => cont_props.no_cap = 1,
                    _ => {}
                }
            }
        }
    }
    *cont_state | *surf_state
}
pub fn istore_first_change_index(list: &[Istore]) -> i32 {
    if list.is_empty() || list[0].flags & ((1 << 4) | 3) != 0 {
        return -1;
    }
    unsafe { list[0].index.i }
}
/// Original: `istoreNextChange` (`istore.c:1738`).
pub fn istore_next_change(
    list: &[Istore],
    cursor: &mut usize,
    def: &DrawProps,
    props: &mut DrawProps,
    state: &mut i32,
    changes: &mut i32,
) -> i32 {
    *changes = 0;
    *state &= !(8 | 16);
    props.gap = 0;
    props.connect = 0;
    let mut index = -1;
    while *cursor < list.len() {
        let store = &list[*cursor];
        if store.flags & ((1 << 4) | 3) != 0 {
            return -1;
        }
        let item = unsafe { store.index.i };
        if index < 0 {
            index = item
        } else if item != index {
            return item;
        }
        *cursor += 1;
        let ending = store.flags & (1 << 5) != 0;
        match store.type_ {
            1 => {
                *changes |= 1;
                if ending {
                    props.red = def.red;
                    props.green = def.green;
                    props.blue = def.blue;
                    *state &= !1;
                } else {
                    props.red = unsafe { store.value.b[0] } as f32 / 255.;
                    props.green = unsafe { store.value.b[1] } as f32 / 255.;
                    props.blue = unsafe { store.value.b[2] } as f32 / 255.;
                    *state |= 1;
                }
            }
            2 => {
                *changes |= 2;
                if ending {
                    props.fill_red = def.fill_red;
                    props.fill_green = def.fill_green;
                    props.fill_blue = def.fill_blue;
                    *state &= !2;
                } else {
                    props.fill_red = unsafe { store.value.b[0] } as f32 / 255.;
                    props.fill_green = unsafe { store.value.b[1] } as f32 / 255.;
                    props.fill_blue = unsafe { store.value.b[2] } as f32 / 255.;
                    *state |= 2;
                }
            }
            3 => {
                *changes |= 4;
                if ending {
                    props.trans = def.trans;
                    *state &= !4
                } else {
                    props.trans = unsafe { store.value.i };
                    *state |= 4
                }
            }
            4 => {
                *changes |= 8;
                *state |= 8;
                props.gap = 1
            }
            5 => {
                *changes |= 16;
                *state |= 16;
                props.connect = unsafe { store.value.i }
            }
            6 => {
                *changes |= 32;
                if ending {
                    props.linewidth = def.linewidth;
                    *state &= !32
                } else {
                    props.linewidth = unsafe { store.value.i };
                    *state |= 32
                }
            }
            7 => {
                *changes |= 64;
                if ending {
                    props.linewidth2 = def.linewidth2;
                    *state &= !64
                } else {
                    props.linewidth2 = unsafe { store.value.i };
                    *state |= 64
                }
            }
            9 => {
                *changes |= 256;
                if ending {
                    props.symsize = def.symsize;
                    *state &= !256
                } else {
                    props.symsize = unsafe { store.value.i };
                    *state |= 256
                }
            }
            8 => {
                *changes |= 128;
                if ending {
                    props.symflags = def.symflags;
                    props.symtype = def.symtype;
                    *state &= !128;
                } else {
                    props.symflags &= !1;
                    props.symtype = unsafe { store.value.i };
                    if props.symtype < 0 {
                        props.symtype = -1 - props.symtype;
                        props.symflags |= 1;
                    }
                    *state |= 128;
                }
            }
            10 => {
                *changes |= 512;
                if ending {
                    props.value1 = def.value1;
                    *state &= !512
                } else {
                    props.value1 = unsafe { store.value.f };
                    *state |= 512
                }
            }
            _ => {}
        }
    }
    -1
}
pub fn istore_point_draw_props(
    obj: &Iobj,
    cont_props: &mut DrawProps,
    point_props: &mut DrawProps,
    co: usize,
    pt: i32,
) -> i32 {
    if co >= obj.cont.len() {
        return 0;
    }
    let mut default_props = DrawProps::default();
    let mut cont_state = 0;
    let mut surf_state = 0;
    istore_default_draw_props(obj, &mut default_props);
    istore_cont_surf_draw_props(
        &obj.store,
        &default_props,
        cont_props,
        co as i32,
        obj.cont[co].surf,
        &mut cont_state,
        &mut surf_state,
    );
    istore_list_point_props(&obj.cont[co].store, cont_props, point_props, pt)
}
pub fn istore_list_point_props(
    list: &[Istore],
    cont: &DrawProps,
    point: &mut DrawProps,
    pt: i32,
) -> i32 {
    *point = *cont;
    let (mut cursor, mut state, mut changes) = (0, 0, 0);
    let mut next = istore_first_change_index(list);
    let mut last = -1;
    while next >= 0 && next <= pt {
        last = next;
        next = istore_next_change(list, &mut cursor, cont, point, &mut state, &mut changes);
    }
    if last != pt {
        point.gap = 0;
        point.connect = 0;
        point.value1 = 0.;
        state &= !(8 | 16 | 512)
    }
    state
}
pub fn istore_skip_to_index(list: &[Istore], index: i32) -> i32 {
    let (lookup, after) = istore_lookup(list, index);
    let item = lookup.unwrap_or(after);
    if item >= list.len() || list[item].flags & ((1 << 4) | 3) != 0 {
        return -1;
    }
    unsafe { list[item].index.i }
}
pub fn istore_trans_state_matches(list: &[Istore], state: i32) -> i32 {
    list.iter()
        .any(|store| {
            store.type_ == 3
                && store.flags & (1 << 5) == 0
                && (if unsafe { store.value.i } != 0 { 1 } else { 0 }) == state
        })
        .then_some(1)
        .unwrap_or(0)
}
