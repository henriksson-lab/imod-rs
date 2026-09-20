//! Scaffold for the complete `IMOD/libimod/istore.c` source unit.
#![allow(unused_variables)]
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libimod::imodel::{IMOD_ERROR_READ, Icont, Iobj};
use crate::imod::libimod::imodel_files::{
    imod_get_float, imod_get_int, imod_get_short, imod_put_float, imod_put_int, imod_put_short,
};
use std::io::{Read, Write};

/// Original `GEN_STORE_*` scalar encodings (`include/istore.h:23-26`).
pub const GEN_STORE_INT: u16 = 0;
pub const GEN_STORE_FLOAT: u16 = 1;
pub const GEN_STORE_SHORT: u16 = 2;
pub const GEN_STORE_BYTE: u16 = 3;
/// Original `GEN_STORE_NOINDEX` (`include/istore.h:29`).
pub const GEN_STORE_NOINDEX: u16 = 1 << 4;
/// Original `GEN_STORE_REVERT` (`include/istore.h:30`).
pub const GEN_STORE_REVERT: u16 = 1 << 5;
/// Original `GEN_STORE_SURFACE` (`include/istore.h:31`).
pub const GEN_STORE_SURFACE: u16 = 1 << 6;
/// Original `GEN_STORE_ONEPOINT` (`include/istore.h:32`).
pub const GEN_STORE_ONEPOINT: u16 = 1 << 7;
/// Original `GEN_STORE_GAP` / `GEN_STORE_CONNECT` (`include/istore.h:41-42`).
pub const GEN_STORE_GAP: i16 = 4;
pub const GEN_STORE_CONNECT: i16 = 5;
/// Original general-storage change types (`include/istore.h:38-61`).
pub const GEN_STORE_COLOR: i16 = 1;
pub const GEN_STORE_FCOLOR: i16 = 2;
pub const GEN_STORE_TRANS: i16 = 3;
pub const GEN_STORE_3DWIDTH: i16 = 6;
pub const GEN_STORE_2DWIDTH: i16 = 7;
pub const GEN_STORE_SYMTYPE: i16 = 8;
pub const GEN_STORE_SYMSIZE: i16 = 9;
pub const GEN_STORE_VALUE1: i16 = 10;
pub const GEN_STORE_MINMAX1: i16 = 11;
pub const GEN_STORE_VALUE2: i16 = 12;
pub const GEN_STORE_MINMAX2: i16 = 13;
pub const GEN_STORE_VALUE3: i16 = 14;
pub const GEN_STORE_MINMAX3: i16 = 15;
pub const GEN_STORE_VALUE4: i16 = 16;
pub const GEN_STORE_MINMAX4: i16 = 17;
pub const GEN_STORE_VALUE5: i16 = 18;
pub const GEN_STORE_MINMAX5: i16 = 19;
pub const GEN_STORE_VALUE6: i16 = 20;
pub const GEN_STORE_MINMAX6: i16 = 21;
pub const GEN_STORE_NO_CAP: i16 = 24;
/// Original: `CHANGED_COLOR` (`istore.h:64`).
pub const CHANGED_COLOR: i32 = 1 << 0;
/// Original: `CHANGED_FCOLOR` (`istore.h:65`).
pub const CHANGED_FCOLOR: i32 = 1 << 1;
/// Original: `CHANGED_TRANS` (`istore.h:66`).
pub const CHANGED_TRANS: i32 = 1 << 2;
/// Original: `CHANGED_GAP` (`istore.h:67`).
pub const CHANGED_GAP: i32 = 1 << 3;
/// Original: `CHANGED_CONNECT` (`istore.h:68`).
pub const CHANGED_CONNECT: i32 = 1 << 4;
/// Original: `CHANGED_3DWIDTH` (`istore.h:69`).
pub const CHANGED_3DWIDTH: i32 = 1 << 5;
/// Original: `CHANGED_2DWIDTH` (`istore.h:70`).
pub const CHANGED_2DWIDTH: i32 = 1 << 6;
/// Original: `CHANGED_SYMTYPE` (`istore.h:71`).
pub const CHANGED_SYMTYPE: i32 = 1 << 7;
/// Original: `CHANGED_SYMSIZE` (`istore.h:72`).
pub const CHANGED_SYMSIZE: i32 = 1 << 8;
/// Original: `CHANGED_VALUE1` (`istore.h:73`).
pub const CHANGED_VALUE1: i32 = 1 << 9;
/// Original: `StoreUnion` / `union store_type` (`istore.h:90`).
///
/// A genuine C type-punning union: `istore.c` writes one member and reads
/// another -- `istoreGenerateItems` stores an `int` and `istoreExtractChanges`
/// reads the same four bytes as `b[4]` -- and `imodel_files.c` writes them to
/// disk as one 32-bit word either way.  So the four bytes, not any one member,
/// are the value.
///
/// Rust has no *safe* union: every read of a `union` field is `unsafe` because
/// the compiler cannot know which member was last written.  The four bytes are
/// therefore held directly, in the machine's own order, and the five members of
/// the C union are accessor pairs over them -- one method per member of
/// `istore.h:91-95`, with a setter and a constructor each, and nothing else.
/// `to_ne_bytes`/`from_ne_bytes` is a bit-for-bit reinterpretation, exactly what
/// the union does, so no byte moves and no value changes.
#[derive(Clone, Copy, Default)]
pub struct StoreUnion {
    /// The union's four bytes, in the machine's byte order.
    pub bytes: [u8; 4],
}
impl StoreUnion {
    /// `StoreUnion::i` (`istore.h:91`), read.
    pub fn i(self) -> i32 {
        i32::from_ne_bytes(self.bytes)
    }
    /// `StoreUnion::i` (`istore.h:91`), written.
    pub fn set_i(&mut self, value: i32) {
        self.bytes = value.to_ne_bytes();
    }
    /// `StoreUnion::i` (`istore.h:91`), as an initialiser.
    pub fn from_i(value: i32) -> Self {
        Self {
            bytes: value.to_ne_bytes(),
        }
    }
    /// `StoreUnion::f` (`istore.h:92`), read.
    pub fn f(self) -> f32 {
        f32::from_ne_bytes(self.bytes)
    }
    /// `StoreUnion::f` (`istore.h:92`), written.
    pub fn set_f(&mut self, value: f32) {
        self.bytes = value.to_ne_bytes();
    }
    /// `StoreUnion::f` (`istore.h:92`), as an initialiser.
    pub fn from_f(value: f32) -> Self {
        Self {
            bytes: value.to_ne_bytes(),
        }
    }
    /// `StoreUnion::us[2]` (`istore.h:93`), read.
    pub fn us(self) -> [u16; 2] {
        [
            u16::from_ne_bytes([self.bytes[0], self.bytes[1]]),
            u16::from_ne_bytes([self.bytes[2], self.bytes[3]]),
        ]
    }
    /// `StoreUnion::us[2]` (`istore.h:93`), written.
    pub fn set_us(&mut self, value: [u16; 2]) {
        let (low, high) = (value[0].to_ne_bytes(), value[1].to_ne_bytes());
        self.bytes = [low[0], low[1], high[0], high[1]];
    }
    /// `StoreUnion::us[2]` (`istore.h:93`), as an initialiser.
    pub fn from_us(value: [u16; 2]) -> Self {
        let mut union = Self::default();
        union.set_us(value);
        union
    }
    /// `StoreUnion::s[2]` (`istore.h:94`), read.
    pub fn s(self) -> [i16; 2] {
        [
            i16::from_ne_bytes([self.bytes[0], self.bytes[1]]),
            i16::from_ne_bytes([self.bytes[2], self.bytes[3]]),
        ]
    }
    /// `StoreUnion::s[2]` (`istore.h:94`), written.
    pub fn set_s(&mut self, value: [i16; 2]) {
        let (low, high) = (value[0].to_ne_bytes(), value[1].to_ne_bytes());
        self.bytes = [low[0], low[1], high[0], high[1]];
    }
    /// `StoreUnion::s[2]` (`istore.h:94`), as an initialiser.
    pub fn from_s(value: [i16; 2]) -> Self {
        let mut union = Self::default();
        union.set_s(value);
        union
    }
    /// `StoreUnion::b[4]` (`istore.h:95`), read.
    pub fn b(self) -> [u8; 4] {
        self.bytes
    }
    /// `StoreUnion::b[4]` (`istore.h:95`), written.
    pub fn set_b(&mut self, value: [u8; 4]) {
        self.bytes = value;
    }
    /// `StoreUnion::b[4]` (`istore.h:95`), as an initialiser.
    pub fn from_b(value: [u8; 4]) -> Self {
        Self { bytes: value }
    }
}
impl core::fmt::Debug for StoreUnion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.i().fmt(f)
    }
}
impl PartialEq for StoreUnion {
    fn eq(&self, other: &Self) -> bool {
        self.i() == other.i()
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Istore {
    pub type_: i16,
    pub flags: u16,
    pub index: StoreUnion,
    pub value: StoreUnion,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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
/// Original: `imodWriteStore` (`istore.c:23`).
pub fn imod_write_store(list: &[Istore], id: i32, file: &mut ImodFile) -> i32 {
    if list.is_empty() {
        return 0;
    }
    let _ = imod_put_int(file, id);
    let i = list.len() as i32 * 12;
    let _ = imod_put_int(file, i);
    for store in list {
        let _ = imod_put_short(file, store.type_);
        let _ = imod_put_short(file, store.flags as i16);
        // Set up to write index
        let mut dtype = store.flags & 3;
        for item in [store.index, store.value] {
            // `imodWriteStore` only propagates the error from `imodPutBytes`;
            // the int/float/short writes drop their `ferror` return.
            match dtype {
                0 => {
                    let _ = imod_put_int(file, (item.i()));
                }
                1 => {
                    let _ = imod_put_float(file, (item.f()));
                }
                2 => {
                    let shorts = (item.s());
                    let _ = imod_put_short(file, shorts[0]);
                    let _ = imod_put_short(file, shorts[1]);
                }
                _ => {
                    if file.write_all(&(item.b())).is_err() {
                        return 1;
                    }
                }
            }
            // For second time through, set up to write value
            dtype = (store.flags >> 2) & 3;
        }
    }
    0
}
/// Original: `imodReadStore` (`istore.c`).
pub fn imod_read_store(file: &mut ImodFile, error: &mut i32) -> Option<Vec<Istore>> {
    let nread = match imod_get_int(file) {
        Ok(bytes) => bytes / 12,
        Err(_) => {
            *error = IMOD_ERROR_READ;
            return None;
        }
    };
    *error = 0;
    if nread <= 0 {
        *error = IMOD_ERROR_READ;
        return None;
    }
    let mut list = Vec::with_capacity(nread as usize);
    let mut need_sort = false;
    let mut last_index = 0;
    for entry in 0..nread {
        let type_ = match imod_get_short(file) {
            Ok(value) => value,
            Err(_) => {
                *error = IMOD_ERROR_READ;
                return None;
            }
        };
        let flags = match imod_get_short(file) {
            Ok(value) => value as u16,
            Err(_) => {
                *error = IMOD_ERROR_READ;
                return None;
            }
        };
        let mut items = [StoreUnion::default(), StoreUnion::default()];
        let mut dtype = flags & 3;
        for item in &mut items {
            *item = match dtype {
                0 => match imod_get_int(file) {
                    Ok(value) => StoreUnion::from_i(value),
                    Err(_) => {
                        *error = IMOD_ERROR_READ;
                        return None;
                    }
                },
                1 => match imod_get_float(file) {
                    Ok(value) => StoreUnion::from_f(value),
                    Err(_) => {
                        *error = IMOD_ERROR_READ;
                        return None;
                    }
                },
                2 => match (imod_get_short(file), imod_get_short(file)) {
                    (Ok(first), Ok(second)) => StoreUnion::from_s([first, second]),
                    _ => {
                        *error = IMOD_ERROR_READ;
                        return None;
                    }
                },
                _ => {
                    let mut bytes = [0; 4];
                    if file.read_exact(&mut bytes).is_err() {
                        *error = IMOD_ERROR_READ;
                        return None;
                    }
                    StoreUnion::from_b(bytes)
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
            (store.index.i())
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
        (one.index.i())
    } else {
        i32::MAX
    };
    let second = if two.flags & ((1 << 4) | 3) == 0 {
        (two.index.i())
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
        let index = (store.index.i());
        if (store.flags & (1 << 6) == 0 && index == co)
            || (store.flags & (1 << 6) != 0 && index == surf)
        {
            return Some(store);
        }
    }
    None
}
/// Original: `istoreSort` (`istore.c:152`).
pub fn istore_sort(list: &mut Vec<Istore>) {
    list.sort_by(store_compare);
}
/// Original: `istoreInsert` (`istore.c:184`).
pub fn istore_insert(list: &mut Vec<Istore>, store: Istore) -> i32 {
    // `lookup` stays -1 for a GEN_STORE_NOINDEX item because the source never
    // calls istoreLookup on that path, so GEN_STORE_REVERT cannot take effect.
    let mut lookup = None;
    let after;
    if store.flags & (1 << 4) != 0 {
        after = list.len();
    } else {
        let found = istore_lookup(list, (store.index.i()));
        lookup = found.0;
        after = found.1;
    }
    if store.flags & (1 << 5) != 0 {
        if let Some(index) = lookup {
            list.insert(index, store);
            return 0;
        }
    }
    list.insert(after, store);
    0
}
/// Original: `istoreLookup` (`istore.c:207`).
pub fn istore_lookup(list: &[Istore], index: i32) -> (Option<usize>, usize) {
    let noindex: u16 = (1 << 4) | 3;
    let mut matched: i32 = -1;
    if list.is_empty() {
        return (None, 0);
    }
    let mut below: i32 = 0;
    let mut above: i32 = list.len() as i32 - 1;

    // test that first element is below item  - if above, done
    let store = &list[0];
    if store.flags & noindex != 0 || (store.index.i()) > index {
        return (None, 0);
    } else if (store.index.i()) == index {
        matched = 0;
    }

    // test that last element is above item - if below, set after to list end
    // and return
    let store = &list[above as usize];
    if matched < 0 && store.flags & noindex == 0 {
        if (store.index.i()) < index {
            return (None, list.len());
        } else if (store.index.i()) == index {
            matched = above;
        }
    }

    // Look at element midway between below and above and replace either the
    // below or the above element
    while matched < 0 && above - below > 1 {
        let mid = (above + below) / 2;
        let store = &list[mid as usize];
        if store.flags & noindex != 0 || (store.index.i()) > index {
            above = mid;
        } else if (store.index.i()) == index {
            matched = mid;
        } else {
            below = mid;
        }
    }

    // If there is still no match, then set after to the one above
    if matched < 0 {
        return (None, above as usize);
    }

    // If there is a match, find first one after the matching index
    let mut mid = matched + 1;
    while mid < list.len() as i32 {
        let store = &list[mid as usize];
        if store.flags & noindex != 0 || (store.index.i()) > index {
            break;
        }
        mid += 1;
    }
    let after = mid as usize;

    // Then find first one before the match, and return first match
    let mut mid = matched - 1;
    while mid >= 0 {
        if (list[mid as usize].index.i()) < index {
            break;
        }
        mid -= 1;
    }
    (Some((mid + 1) as usize), after)
}
/// Original: `istoreDump` (`istore.c:281`).
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
    let mut out = ImodFile::Stdout;
    let _ = out.write_all(format!(" {} items in list:\n", list.len()).as_bytes());
    for store in list {
        let _ = out.write_all(format!("{:>6}-", store.type_).as_bytes());
        if store.type_ > 0 && store.type_ as usize <= types.len() {
            let _ = out.write_all(types[store.type_ as usize - 1].as_bytes());
        }
        let _ = out.write_all(format!("  {:>6o}-", store.flags).as_bytes());
        let mut dtype = 0;
        let j = if store.type_ == 23 || store.type_ == 22 {
            1
        } else {
            0
        };
        if store.flags & (1 << 4) != 0 {
            let _ = out.write_all(b"NOIND");
            dtype = 1;
        }
        if j == 0 && store.flags & (1 << 5) != 0 {
            let _ = out.write_all(if dtype != 0 { b"|REVERT" } else { b"REVERT" });
            dtype = 1;
        }
        if j == 0 && store.flags & (1 << 6) != 0 {
            let _ = out.write_all(if dtype != 0 { b"|SURF" } else { b"SURF" });
            dtype = 1;
        }
        if j == 0 && store.flags & (1 << 7) != 0 {
            let _ = out.write_all(if dtype != 0 { b"|ONEPT" } else { b"ONEPT" });
            dtype = 1;
        }
        if j != 0 && store.flags & (1 << 5) != 0 {
            let _ = out.write_all(if dtype != 0 { b"|CAP" } else { b"CAP" });
            dtype = 1;
        }
        if j != 0 && store.flags & (1 << 6) != 0 {
            let _ = out.write_all(if dtype != 0 { b"|DEL" } else { b"DEL" });
            dtype = 1;
        }
        if j != 0 && store.flags & (1 << 7) != 0 {
            let _ = out.write_all(if dtype != 0 { b"|OUTER" } else { b"OUTER" });
        }
        let mut dtype = store.flags & 3;
        for item in [store.index, store.value] {
            match dtype {
                0 => {
                    let _ = out.write_all(format!(" {:>11}", item.i()).as_bytes());
                }
                1 => {
                    let _ = out
                        .write_all(c_format(" %12.6g", &[CArg::Dbl(item.f() as f64)]).as_bytes());
                }
                2 => {
                    let [first, second] = item.s();
                    let _ = out.write_all(format!(" {:>6} {:>6}", first, second).as_bytes());
                }
                _ => {
                    let [first, second, third, fourth] = item.b();
                    let _ = out.write_all(
                        format!(" {:>3} {:>3} {:>3} {:>3}", first, second, third, fourth)
                            .as_bytes(),
                    );
                }
            }
            dtype = (store.flags >> 2) & 3;
        }
        let _ = out.write_all(b"\n");
    }
}
/// Original: `istoreChecksum` (`istore.c:357`).
pub fn istore_checksum(list: &[Istore]) -> f64 {
    let mut sum = 0.;
    for store in list {
        // `store->flags` is b3dUInt16 and `store->type` b3dInt16; both promote
        // to int before the addition.
        sum += (store.flags as i32 + store.type_ as i32) as f64;
        let mut dtype = store.flags & 3;
        for item in [store.index, store.value] {
            sum += match dtype {
                0 => (item.i() as f64),
                1 => (item.f() as f64),
                2 => ((item.s()[0] as i32 + item.s()[1] as i32) as f64),
                _ => {
                    ((item.b()[0] as u32
                        + item.b()[1] as u32
                        + item.b()[2] as u32
                        + item.b()[3] as u32) as f64)
                }
            };
            dtype = (store.flags >> 2) & 3;
        }
    }
    sum
}
/// Original: `istoreCountItems` (`istore.c:397`).
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
/// Original: `istoreCountObjectItems` (`istore.c:422`).
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
/// Original: `istoreCountContSurfItems` (`istore.c:450`).
pub fn istore_count_cont_surf_items(list: &[Istore], index: i32, surf_flag: i32) -> i32 {
    let mut count = 0;
    if list.is_empty() {
        return 0;
    }
    let surf_flag = if surf_flag != 0 { 1 << 6 } else { 0 };
    let mut index = index;
    for store in list {
        if store.flags & ((1 << 4) | 3) != 0 {
            break;
        }
        // `istore.c:461` overwrites the caller's index with the item's own,
        // so the `index == stp->index.i` test at :462 is always true.
        index = (store.index.i());
        if (store.flags & (1 << 6)) == surf_flag && index == (store.index.i()) {
            count += 1;
        }
    }
    count
}
/// Original: `istorePointIsGap` (`istore.c:472`).
pub fn istore_point_is_gap(list: &[Istore], index: i32) -> i32 {
    let (lookup, after) = istore_lookup(list, index);
    let Some(lookup) = lookup else {
        return 0;
    };
    for store in &list[lookup..after] {
        if store.type_ == GEN_STORE_GAP {
            return 1;
        }
    }
    0
}
/// Original: `istoreConnectNumber` (`istore.c:491`).
pub fn istore_connect_number(list: &[Istore], index: i32) -> i32 {
    let (lookup, after) = istore_lookup(list, index);
    let Some(lookup) = lookup else {
        return -1;
    };
    for store in &list[lookup..after] {
        if store.type_ == GEN_STORE_CONNECT {
            return (store.value.i());
        }
    }
    -1
}
/// Original: `istoreAddMinMax` (`istore.c:511`).
pub fn istore_add_min_max(list: &mut Vec<Istore>, type_: i16, min: f32, max: f32) -> i32 {
    if type_ < 11 || type_ > 21 || (type_ - 11) % 2 != 0 {
        return 1;
    }
    for store in list.iter_mut().rev() {
        if store.flags & ((1 << 4) | 3) == 0 {
            break;
        }
        if store.type_ == type_ && store.flags == ((1 << 4) | (1 << 2) | 1) {
            store.index = StoreUnion::from_f(min);
            store.value = StoreUnion::from_f(max);
            return 0;
        }
    }
    istore_insert(
        list,
        Istore {
            type_,
            flags: (1 << 4) | (1 << 2) | 1,
            index: StoreUnion::from_f(min),
            value: StoreUnion::from_f(max),
        },
    )
}
/// Original: `istoreFindAddMinMax1` (`istore.c:546`).
pub fn istore_find_add_min_max1(obj: &mut Iobj) -> i32 {
    istore_find_add_min_max(obj, 10)
}
/// Original: `istoreFindAddMinMax` (`istore.c:558`).
pub fn istore_find_add_min_max(obj: &mut Iobj, type_: i16) -> i32 {
    if type_ < 10 || type_ > 20 || (type_ - 10) % 2 != 0 {
        return 2;
    }
    let mut min = 1.0e37_f32;
    let mut max = -1.0e37_f32;
    for store in obj
        .store
        .iter()
        .chain(obj.cont.iter().flat_map(|cont| cont.store.iter()))
    {
        if store.type_ == type_ && store.flags & (1 << 5) == 0 {
            let value = (store.value.f());
            // B3DMIN/B3DMAX are plain ternaries, not IEEE minNum.
            min = if min < value { min } else { value };
            max = if max > value { max } else { value };
        }
    }
    if min > max {
        -1
    } else {
        istore_add_min_max(&mut obj.store, type_ + 1, min, max)
    }
}
/// Original: `istoreGetMinMax` (`istore.c:593`).
///
/// `size` is declared by `istore.h:190` and documented as currently unused; the
/// body (`istore.c:595-608`) never reads it.
pub fn istore_get_min_max(
    list: &[Istore],
    size: i32,
    type_: i16,
    min: &mut f32,
    max: &mut f32,
) -> i32 {
    for store in list.iter().rev() {
        if store.flags & ((1 << 4) | 3) == 0 {
            return 0;
        }
        if store.type_ == type_ && store.flags & 3 == 1 {
            *min = (store.index.f());
            *max = (store.value.f());
            return 1;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};

    /// `imodReadStore` divides the chunk length by SIZE_STOR before the
    /// `nread <= 0` test (`istore.c:80-86`), so a chunk shorter than one
    /// record is an IMOD_ERROR_READ, not an empty list.
    #[test]
    fn read_store_rejects_chunk_shorter_than_one_record() {
        for (bytes, records) in [(-12i32, 0usize), (0, 0), (8, 0), (12, 1), (24, 2)] {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-istore-chunk-{}-{}.bin",
                std::process::id(),
                bytes
            ));
            let mut file = ImodFile::open(path.to_str().unwrap(), "wb").unwrap();
            file.write_all(&bytes.to_be_bytes()).unwrap();
            for _ in 0..records {
                file.write_all(&[0u8; 12]).unwrap();
            }
            drop(file);
            let mut file = ImodFile::open(path.to_str().unwrap(), "rb").unwrap();
            let mut error = -1;
            let read = imod_read_store(&mut file, &mut error);
            if records == 0 {
                assert!(read.is_none(), "chunk of {} bytes must fail", bytes);
                assert_eq!(error, IMOD_ERROR_READ);
            } else {
                assert_eq!(error, 0);
                assert_eq!(read.unwrap().len(), records);
            }
            std::fs::remove_file(&path).unwrap();
        }
    }
    /// Verbatim output of a driver linked against the pinned
    /// `IMOD/libimod/istore.c` and `IMOD/libcfshr/ilist.c`.
    const ISTORE_C_DRIVER: &str = concat!(
        "lookup(-1) = -1 after 0\n",
        "lookup(0) = 0 after 1\n",
        "lookup(1) = -1 after 1\n",
        "lookup(2) = 1 after 3\n",
        "lookup(3) = -1 after 3\n",
        "lookup(4) = -1 after 3\n",
        "lookup(5) = 3 after 4\n",
        "lookup(6) = -1 after 4\n",
        "lookup(null) = -1 after 0\n",
        "lookup(empty) = -1 after 0\n",
        "insert revert ret=0\n",
        "insert.revert size=4 [3,32,2,77] [3,0,2,10] [3,0,2,11] [3,0,5,12]\n",
        "insert noindex ret=0\n",
        "insert.noindex size=5 [3,32,2,77] [3,0,2,10] [3,0,2,11] [3,0,5,12] [9,16,2,88]\n",
        "insert noindex|revert ret=0\n",
        "insert.noindex_revert size=6 [3,32,2,77] [3,0,2,10] [3,0,2,11] [3,0,5,12] [9,16,2,88] [9,48,2,99]\n",
        "insert into null ret=0\n",
        "insert.null size=1 [3,0,4,5]\n",
        "insertChange ret=0\n",
        "insertChange.range size=4 [6,0,2,20] [7,0,2,30] [3,0,2,55] [3,0,3,40]\n",
        "insertChange dup ret=0\n",
        "insertChange.dup size=2 [3,0,0,7] [6,0,1,3]\n",
        "insertChange onepoint ret=0\n",
        "insertChange.onepoint size=3 [3,0,0,7] [6,0,1,3] [3,128,6,7]\n",
        "insertChange empty ret=0\n",
        "insertChange.empty size=1 [3,0,1,2]\n",
        "endChange ret=0\n",
        "endChange.multi size=3 [3,0,1,5] [3,32,2,0] [6,0,2,9]\n",
        "endChange single ret=0\n",
        "endChange.single size=0\n",
        "endChange again ret=1\n",
        "endChange.again size=0\n",
        "endChange empty ret=1\n",
        "clearChange ret=0\n",
        "clearChange size=4 [3,0,1,5] [3,0,3,6] [3,32,6,0] [6,0,2,1]\n",
        "clearRange size=1 [6,0,7,8]\n",
        "clearRange.spanning size=0\n",
        "addOne ret=0\n",
        "addOne replace ret=0\n",
        "addOne surf ret=0\n",
        "addOne size=2 [9,128,4,9] [9,192,4,3]\n",
        "clearOne cont ret=0\n",
        "clearOne surf ret=0\n",
        "clearOne none ret=1\n",
        "clearOne size=0\n",
        "breakChanges ret=0\n",
        "breakChanges size=3 [3,0,0,4] [3,32,2,4] [3,0,2,4]\n",
        "findBreak(2) = 2\n",
        "shiftIndex size=3 [3,0,0,4] [3,32,3,4] [3,0,3,4]\n",
        "breakChanges@psize ret=0\n",
        "breakChanges.psize size=4 [3,0,0,4] [6,0,1,2] [6,32,5,2] [3,32,5,4]\n",
        "shiftIndex.surface size=4 [9,64,2,1] [3,0,5,4] [3,0,9,5] [11,21,0,0]\n",
        "deletePoint mid ret=0\n",
        "deletePoint.mid size=2 [3,0,1,4] [3,32,2,0]\n",
        "deletePoint end-follows ret=0\n",
        "deletePoint.endfollows size=0\n",
        "deletePoint dup ret=0\n",
        "deletePoint.dup size=3 [3,0,2,5] [6,0,2,7] [3,0,3,9]\n",
        "deletePoint first ret=0\n",
        "deletePoint.first size=2 [3,0,0,4] [6,0,2,1]\n",
        "deletePoint last ret=0\n",
        "deletePoint.last size=1 [3,0,0,4]\n",
        "deletePoint nomatch ret=0\n",
        "deletePoint.nomatch size=1 [3,0,3,4]\n",
        "deletePoint empty ret=0\n",
        "deleteContSurf.surf size=2 [9,0,1,1] [9,0,2,3]\n",
        "deleteContSurf.cont size=1 [9,0,1,3]\n",
        "deleteContSurf.dupfirst size=1 [9,0,0,3]\n",
        "cleanEnds size=3 [3,0,2,5] [6,32,4,0] [11,21,0,0]\n",
        "invert gap ret=0\n",
        "invert.gap size=1 [4,128,2,0]\n",
        "invert range ret=0\n",
        "invert.range size=2 [3,0,2,7] [3,32,5,7]\n",
        "invert chain ret=0\n",
        "invert.chain size=3 [3,0,3,8] [3,0,5,7] [3,32,7,7]\n",
        "invert open ret=0\n",
        "invert.open size=3 [3,0,0,7] [3,32,4,7] [11,21,0,0]\n",
        "extract ret=0\n",
        "extract size=2 [3,0,0,5] [3,32,3,0]\n",
        "copyNonIndex ret=0\n",
        "copyNonIndex size=1 [11,21,0,0]\n",
        "copyContSurf ret=0\n",
        "copyContSurf.surf size=1 [9,64,7,2]\n",
        "copyContSurf cont ret=0\n",
        "copyContSurf.cont size=1 [9,0,7,1]\n",
        "breakContour ret=0\n",
        "breakContour.new size=2 [3,0,0,2] [3,32,2,2]\n",
        "breakContour.old size=4 [3,0,0,2] [3,32,1,2] [3,0,1,2] [3,32,2,0]\n",
        "countContSurf(1,0) = 2\n",
        "countContSurf(1,1) = 1\n",
        "countContSurf(99,0) = 2\n",
        "countItems(9) = 3\n",
        "countItems(9,stop) = 1\n",
        "skipToIndex(0) = 1\n",
        "skipToIndex(2) = 2\n",
        "skipToIndex(3) = -1\n",
        "firstChangeIndex = 1\n",
        "pointIsGap(2) = 1\n",
        "pointIsGap(3) = 0\n",
        "connectNumber(2) = 12\n",
        "connectNumber(3) = -1\n",
        "retainPoint(0) = 0\n",
        "retainPoint(1) = 0\n",
        "retainPoint(2) = 1\n",
        "retainPoint(3) = 1\n",
        "retainPoint(4) = 1\n",
        "retainPoint(5) = 1\n",
        "retainPoint(6) = 0\n",
        "transState(1) = 1\n",
        "transState(0) = 0\n",
        "checksum = 201.0\n",
        "nextObjItem first type=3\n",
        "nextObjItem next type=9\n",
        "nextObjItem end type=-1\n",
        "default r=0.100000 g=0.200000 b=0.300000 fr=0.000000 fg=0.000000 fb=0.000000 tr=0 con=0 gap=0 lw=0 lw2=0 sy=4 sf=0 ss=0 v1=0.000000 nc=0\n",
        "contSurf ret=129 contState=128 surfState=1\n",
        "contSurf r=0.501961 g=0.250980 b=0.125490 fr=0.000000 fg=0.000000 fb=0.000000 tr=0 con=0 gap=0 lw=0 lw2=0 sy=2 sf=1 ss=0 v1=0.000000 nc=0\n",
        "contSurf negsurf ret=0 contState=0 surfState=0\n",
        "contSurf.negsurf r=0.100000 g=0.200000 b=0.300000 fr=0.000000 fg=0.000000 fb=0.000000 tr=0 con=0 gap=0 lw=0 lw2=0 sy=4 sf=0 ss=0 v1=0.000000 nc=0\n",
        "contSurf negco ret=1 contState=0 surfState=1\n",
        "contSurf.negco r=0.501961 g=0.250980 b=0.125490 fr=0.000000 fg=0.000000 fb=0.000000 tr=0 con=0 gap=0 lw=0 lw2=0 sy=4 sf=0 ss=0 v1=0.000000 nc=0\n",
        "listPointProps(0) = 0 trans=1 gap=0 con=0 v1=0.000000\n",
        "listPointProps(1) = 0 trans=1 gap=0 con=0 v1=0.000000\n",
        "listPointProps(2) = 4 trans=4 gap=0 con=0 v1=0.000000\n",
        "listPointProps(3) = 0 trans=1 gap=0 con=0 v1=0.000000\n",
        "listPointProps(4) = 0 trans=1 gap=0 con=0 v1=0.000000\n",
        "findAddMinMax1 ret=0\n",
        "minmax.store size=2 [10,4,0,1077936128] [11,21,1077936128,1088421888]\n",
        "getMinMax ret=1 min=3.000000 max=7.000000\n",
        "getMinMax missing ret=0\n",
        "findAddMinMax bad ret=2\n",
        "addMinMax bad ret=1\n",
        "generateItems ret=0\n",
        "gen[0] type=1 flags=12 index=8 bytes=255,127,0 value.i=32767 value.f=0.000000\n",
        "gen[1] type=2 flags=12 index=8 bytes=0,255,63 value.i=4194048 value.f=0.000000\n",
        "gen[2] type=3 flags=0 index=8 bytes=7,0,0 value.i=7 value.f=0.000000\n",
        "gen[3] type=6 flags=0 index=8 bytes=3,0,0 value.i=3 value.f=0.000000\n",
        "gen[4] type=10 flags=4 index=8 bytes=0,0,160 value.i=1067450368 value.f=1.250000\n",
        "p2 endChange endfirst ret=0\n",
        "p2.endChange.endfirst size=3 [3,0,0,5] [3,32,2,0] [3,32,2,0]\n",
        "p2.cleanEnds.noindex size=1 [3,16,0,0]\n",
        "p2.cleanEnds.empty size=0\n",
        "p2 checksum wide = 1966210004.0\n",
        "p2 breakChanges ret=0\n",
        "p2.breakChanges.endedbefore size=6 [3,0,0,4] [3,32,1,0] [6,0,0,2] [6,32,2,2] [6,0,2,2] [6,32,3,0]\n",
        "p2 deletePoint onepoint ret=0\n",
        "p2.deletePoint.onepoint size=1 [3,0,1,5]\n",
        "p2 deletePoint samestart ret=0\n",
        "p2.deletePoint.samestart size=2 [3,0,1,5] [3,32,3,0]\n",
        "p2.shiftIndex.start size=3 [3,0,1,1] [3,0,7,2] [3,0,8,3]\n",
        "p2 extract inverted ret=0\n",
        "p2.extract.inverted size=0\n",
        "p2 extract whole ret=0\n",
        "p2.extract.whole size=2 [3,0,0,5] [3,32,5,0]\n",
        "p2 extract empty ret=0\n",
        "p2.extract.emptysrc size=0\n",
        "p2 firstChangeIndex = 1\n",
        "p2 nextChange[0] = 3 state=9 changes=9\n",
        "p2.nextChange r=0.039216 g=0.078431 b=0.117647 fr=0.000000 fg=0.000000 fb=0.000000 tr=2 con=0 gap=1 lw=4 lw2=0 sy=0 sf=0 ss=0 v1=0.000000 nc=0\n",
        "p2 nextChange[1] = 4 state=5 changes=4\n",
        "p2.nextChange r=0.039216 g=0.078431 b=0.117647 fr=0.000000 fg=0.000000 fb=0.000000 tr=6 con=0 gap=0 lw=4 lw2=0 sy=0 sf=0 ss=0 v1=0.000000 nc=0\n",
        "p2 nextChange[2] = -1 state=4 changes=1\n",
        "p2.nextChange r=0.500000 g=0.000000 b=0.000000 fr=0.000000 fg=0.000000 fb=0.000000 tr=6 con=0 gap=0 lw=4 lw2=0 sy=0 sf=0 ss=0 v1=0.000000 nc=0\n",
        "p2 genPointItems ret=0\n",
        "p2.genPointItems size=1 [1,12,9,1971210]\n",
        "p2 countObjectItems all = 3\n",
        "p2 countObjectItems cont = 2\n",
        "p2 countObjectItems stop = 1\n",
        "p2 countObjectItems none = 0\n",
        "p2 pointDrawProps = 4\n",
        "p2.pointDrawProps r=0.000000 g=0.000000 b=0.000000 fr=0.000000 fg=0.000000 fb=0.000000 tr=9 con=0 gap=0 lw=0 lw2=0 sy=0 sf=0 ss=0 v1=0.000000 nc=0\n",
        "p2 retainPoint empty = 0\n",
        "p2 skipToIndex empty = -1\n",
        "p2 firstChangeIndex empty = -1\n",
        "p2 pointIsGap empty = 0\n",
        "p2 connectNumber empty = -1\n",
        "p2 countContSurf empty = 0\n",
        "p2 transState empty = 0\n",
        "p2 checksum empty = 0.0\n",
        "p2 breakChanges empty = 0\n",
        "p2 invert empty = 0\n",
        "p3 generateItems ret=0\n",
        "p3 gen[0] type=1 bytes=126,231,254\n",
        "p3 gen[1] type=2 bytes=254,127,255\n",
        "p3 addMinMax update ret=0\n",
        "p3 minmax[0] type=3 flags=0 index.f=0.000 value.f=0.000\n",
        "p3 minmax[1] type=11 flags=21 index.f=4.000 value.f=8.000\n",
        "p3 minmax[2] type=13 flags=21 index.f=0.000 value.f=0.000\n",
        "p3 addMinMax new ret=0\n",
        "p3 size after new = 4\n",
        "p3 lookup(-1) = -1 after 0\n",
        "p3 lookup(0) = 0 after 1\n",
        "p3 lookup(1) = 1 after 4\n",
        "p3 lookup(2) = -1 after 4\n",
        "p3 lookup(3) = -1 after 4\n",
        "p3 lookup(4) = 4 after 6\n",
        "p3 lookup(5) = -1 after 6\n",
        "p3 lookup(6) = -1 after 6\n",
        "p3 lookup(7) = 6 after 7\n",
        "p3 lookup(8) = -1 after 7\n",
        "p3 lookup(9) = 7 after 8\n",
        "p3 lookup(10) = -1 after 8\n",
        "p3 lookup noindexonly = -1 after 0\n",
        "p3 insert into noindexonly ret=0\n",
        "p3.insert.noindexonly size=3 [3,0,5,1] [11,21,0,0] [13,21,0,0]\n",
        "p3.clearRange.nochange size=1 [6,0,1,4]\n",
        "p3.clearRange.startbefore size=0\n",
        "p3.deleteContSurf.surfonly size=1 [9,64,2,2]\n",
        "p3.deleteContSurf.shiftskipssurf size=2 [9,64,1,2] [9,0,1,3]\n",
    );
    /// Differential harness against a driver linked directly to
    /// `IMOD/libimod/istore.c`; the expected text is that driver's output.
    #[test]
    fn source_c_driver_differential_dump() {
        fn mk(spec: &[[i32; 4]]) -> Vec<Istore> {
            spec.iter()
                .map(|s| Istore {
                    type_: s[0] as i16,
                    flags: s[1] as u16,
                    index: StoreUnion::from_i(s[2]),
                    value: StoreUnion::from_i(s[3]),
                })
                .collect()
        }
        fn dump(out: &mut String, tag: &str, list: &[Istore]) {
            out.push_str(&format!("{} size={}", tag, list.len()));
            for store in list {
                out.push_str(&format!(
                    " [{},{},{},{}]",
                    store.type_,
                    store.flags,
                    (store.index.i()),
                    (store.value.i())
                ));
            }
            out.push('\n');
        }
        fn dump_props(out: &mut String, tag: &str, p: &DrawProps) {
            out.push_str(&format!(
                "{} r={:.6} g={:.6} b={:.6} fr={:.6} fg={:.6} fb={:.6} tr={} con={} gap={} lw={} lw2={} sy={} sf={} ss={} v1={:.6} nc={}\n",
                tag, p.red, p.green, p.blue, p.fill_red, p.fill_green, p.fill_blue,
                p.trans, p.connect, p.gap, p.linewidth, p.linewidth2, p.symtype,
                p.symflags, p.symsize, p.value1, p.no_cap
            ));
        }
        const GAPF: u16 = 1 << 7;
        const REV: u16 = 1 << 5;
        const NOIND: u16 = 1 << 4;
        const SURF: u16 = 1 << 6;
        let mut o = String::new();

        // ---- istoreLookup ----
        let list = mk(&[
            [3, 0, 0, 10],
            [3, 0, 2, 11],
            [3, 0, 2, 12],
            [3, 0, 5, 13],
            [11, 21, 0, 0],
        ]);
        for i in -1..=6 {
            let (lookup, after) = istore_lookup(&list, i);
            o.push_str(&format!(
                "lookup({}) = {} after {}\n",
                i,
                lookup.map_or(-1, |v| v as i32),
                after
            ));
        }
        let (lookup, after) = istore_lookup(&[], 2);
        o.push_str(&format!(
            "lookup(null) = {} after {}\n",
            lookup.map_or(-1, |v| v as i32),
            after
        ));
        let (lookup, after) = istore_lookup(&[], 2);
        o.push_str(&format!(
            "lookup(empty) = {} after {}\n",
            lookup.map_or(-1, |v| v as i32),
            after
        ));

        // ---- istoreInsert ----
        let mut list = mk(&[[3, 0, 2, 10], [3, 0, 2, 11], [3, 0, 5, 12]]);
        o.push_str(&format!(
            "insert revert ret={}\n",
            istore_insert(
                &mut list,
                Istore {
                    type_: 3,
                    flags: REV,
                    index: StoreUnion::from_i(2),
                    value: StoreUnion::from_i(77),
                }
            )
        ));
        dump(&mut o, "insert.revert", &list);
        o.push_str(&format!(
            "insert noindex ret={}\n",
            istore_insert(
                &mut list,
                Istore {
                    type_: 9,
                    flags: NOIND,
                    index: StoreUnion::from_i(2),
                    value: StoreUnion::from_i(88),
                }
            )
        ));
        dump(&mut o, "insert.noindex", &list);
        o.push_str(&format!(
            "insert noindex|revert ret={}\n",
            istore_insert(
                &mut list,
                Istore {
                    type_: 9,
                    flags: NOIND | REV,
                    index: StoreUnion::from_i(2),
                    value: StoreUnion::from_i(99),
                }
            )
        ));
        dump(&mut o, "insert.noindex_revert", &list);
        let mut list = Vec::new();
        o.push_str(&format!(
            "insert into null ret={}\n",
            istore_insert(
                &mut list,
                Istore {
                    type_: 3,
                    flags: 0,
                    index: StoreUnion::from_i(4),
                    value: StoreUnion::from_i(5),
                }
            )
        ));
        dump(&mut o, "insert.null", &list);

        // ---- istoreInsertChange ----
        let mut list = mk(&[[3, 0, 2, 10], [6, 0, 2, 20], [7, 0, 2, 30], [3, 0, 3, 40]]);
        o.push_str(&format!(
            "insertChange ret={}\n",
            istore_insert_change(
                &mut list,
                Istore {
                    type_: 3,
                    flags: 0,
                    index: StoreUnion::from_i(2),
                    value: StoreUnion::from_i(55),
                }
            )
        ));
        dump(&mut o, "insertChange.range", &list);
        let mut list = mk(&[[3, 0, 0, 7], [6, 0, 1, 3]]);
        o.push_str(&format!(
            "insertChange dup ret={}\n",
            istore_insert_change(
                &mut list,
                Istore {
                    type_: 3,
                    flags: 0,
                    index: StoreUnion::from_i(4),
                    value: StoreUnion::from_i(7),
                }
            )
        ));
        dump(&mut o, "insertChange.dup", &list);
        o.push_str(&format!(
            "insertChange onepoint ret={}\n",
            istore_insert_change(
                &mut list,
                Istore {
                    type_: 3,
                    flags: GAPF,
                    index: StoreUnion::from_i(6),
                    value: StoreUnion::from_i(7),
                }
            )
        ));
        dump(&mut o, "insertChange.onepoint", &list);
        let mut list = Vec::new();
        o.push_str(&format!(
            "insertChange empty ret={}\n",
            istore_insert_change(
                &mut list,
                Istore {
                    type_: 3,
                    flags: 0,
                    index: StoreUnion::from_i(1),
                    value: StoreUnion::from_i(2),
                }
            )
        ));
        dump(&mut o, "insertChange.empty", &list);

        // ---- istoreEndChange ----
        let mut list = mk(&[
            [3, 0, 1, 5],
            [3, 0, 2, 6],
            [6, 0, 2, 9],
            [3, 0, 2, 7],
            [3, REV as i32, 4, 0],
        ]);
        o.push_str(&format!(
            "endChange ret={}\n",
            istore_end_change(&mut list, 3, 2)
        ));
        dump(&mut o, "endChange.multi", &list);
        let mut list = mk(&[[3, 0, 2, 6]]);
        o.push_str(&format!(
            "endChange single ret={}\n",
            istore_end_change(&mut list, 3, 2)
        ));
        dump(&mut o, "endChange.single", &list);
        o.push_str(&format!(
            "endChange again ret={}\n",
            istore_end_change(&mut list, 3, 5)
        ));
        dump(&mut o, "endChange.again", &list);
        let mut list = Vec::new();
        o.push_str(&format!(
            "endChange empty ret={}\n",
            istore_end_change(&mut list, 3, 5)
        ));

        // ---- istoreClearChange / istoreClearRange ----
        let mut list = mk(&[
            [3, 0, 1, 5],
            [3, 0, 3, 6],
            [3, REV as i32, 6, 0],
            [6, 0, 2, 1],
        ]);
        o.push_str(&format!(
            "clearChange ret={}\n",
            istore_clear_change(&mut list, 3, 4)
        ));
        dump(&mut o, "clearChange", &list);
        let mut list = mk(&[[6, 0, 1, 4], [6, REV as i32, 4, 0], [6, 0, 7, 8]]);
        istore_clear_range(&mut list, 6, 2, 4);
        dump(&mut o, "clearRange", &list);
        let mut list = mk(&[[6, 0, 1, 4], [6, REV as i32, 9, 0]]);
        istore_clear_range(&mut list, 6, 2, 4);
        dump(&mut o, "clearRange.spanning", &list);

        // ---- one-index items ----
        let mut list = Vec::new();
        let one = Istore {
            type_: 9,
            flags: GAPF,
            index: StoreUnion::from_i(4),
            value: StoreUnion::from_i(2),
        };
        o.push_str(&format!(
            "addOne ret={}\n",
            istore_add_one_index_item(&mut list, one)
        ));
        o.push_str(&format!(
            "addOne replace ret={}\n",
            istore_add_one_index_item(
                &mut list,
                Istore {
                    value: StoreUnion::from_i(9),
                    ..one
                }
            )
        ));
        o.push_str(&format!(
            "addOne surf ret={}\n",
            istore_add_one_index_item(
                &mut list,
                Istore {
                    flags: GAPF | SURF,
                    value: StoreUnion::from_i(3),
                    ..one
                }
            )
        ));
        dump(&mut o, "addOne", &list);
        o.push_str(&format!(
            "clearOne cont ret={}\n",
            istore_clear_one_index_item(&mut list, 9, 4, 0)
        ));
        o.push_str(&format!(
            "clearOne surf ret={}\n",
            istore_clear_one_index_item(&mut list, 9, 4, 1)
        ));
        o.push_str(&format!(
            "clearOne none ret={}\n",
            istore_clear_one_index_item(&mut list, 9, 4, 0)
        ));
        dump(&mut o, "clearOne", &list);

        // ---- break / find / shift ----
        let mut list = mk(&[[3, 0, 0, 4]]);
        o.push_str(&format!(
            "breakChanges ret={}\n",
            istore_break_changes(&mut list, 2, 5)
        ));
        dump(&mut o, "breakChanges", &list);
        o.push_str(&format!("findBreak(2) = {}\n", istore_find_break(&list, 2)));
        istore_shift_index(&mut list, 2, -1, 1);
        dump(&mut o, "shiftIndex", &list);
        let mut list = mk(&[[3, 0, 0, 4], [6, 0, 1, 2]]);
        o.push_str(&format!(
            "breakChanges@psize ret={}\n",
            istore_break_changes(&mut list, 5, 5)
        ));
        dump(&mut o, "breakChanges.psize", &list);
        let mut list = mk(&[
            [9, SURF as i32, 2, 1],
            [3, 0, 2, 4],
            [3, 0, 6, 5],
            [11, 21, 0, 0],
        ]);
        istore_shift_index(&mut list, 2, -1, 3);
        dump(&mut o, "shiftIndex.surface", &list);

        // ---- istoreDeletePoint ----
        let mut list = mk(&[[3, 0, 1, 4], [3, REV as i32, 3, 0]]);
        o.push_str(&format!(
            "deletePoint mid ret={}\n",
            istore_delete_point(&mut list, 1, 6)
        ));
        dump(&mut o, "deletePoint.mid", &list);
        let mut list = mk(&[[3, 0, 1, 4], [3, REV as i32, 2, 0]]);
        o.push_str(&format!(
            "deletePoint end-follows ret={}\n",
            istore_delete_point(&mut list, 1, 6)
        ));
        dump(&mut o, "deletePoint.endfollows", &list);
        let mut list = mk(&[
            [4, GAPF as i32, 2, 0],
            [3, 0, 2, 5],
            [6, 0, 2, 7],
            [3, 0, 4, 9],
        ]);
        o.push_str(&format!(
            "deletePoint dup ret={}\n",
            istore_delete_point(&mut list, 2, 6)
        ));
        dump(&mut o, "deletePoint.dup", &list);
        let mut list = mk(&[[3, 0, 0, 4], [6, 0, 3, 1]]);
        o.push_str(&format!(
            "deletePoint first ret={}\n",
            istore_delete_point(&mut list, 0, 6)
        ));
        dump(&mut o, "deletePoint.first", &list);
        let mut list = mk(&[[3, 0, 0, 4], [6, 0, 5, 1]]);
        o.push_str(&format!(
            "deletePoint last ret={}\n",
            istore_delete_point(&mut list, 5, 6)
        ));
        dump(&mut o, "deletePoint.last", &list);
        let mut list = mk(&[[3, 0, 4, 4]]);
        o.push_str(&format!(
            "deletePoint nomatch ret={}\n",
            istore_delete_point(&mut list, 1, 6)
        ));
        dump(&mut o, "deletePoint.nomatch", &list);
        let mut list = Vec::new();
        o.push_str(&format!(
            "deletePoint empty ret={}\n",
            istore_delete_point(&mut list, 1, 6)
        ));

        // ---- istoreDeleteContSurf ----
        let mut list = mk(&[[9, 0, 1, 1], [9, SURF as i32, 1, 2], [9, 0, 2, 3]]);
        istore_delete_cont_surf(&mut list, 1, 1);
        dump(&mut o, "deleteContSurf.surf", &list);
        istore_delete_cont_surf(&mut list, 1, 0);
        dump(&mut o, "deleteContSurf.cont", &list);
        let mut list = mk(&[[9, 0, 0, 1], [9, 0, 0, 2], [9, 0, 1, 3]]);
        istore_delete_cont_surf(&mut list, 0, 0);
        dump(&mut o, "deleteContSurf.dupfirst", &list);

        // ---- istoreCleanEnds ----
        let mut list = mk(&[
            [3, REV as i32, 2, 0],
            [3, 0, 2, 5],
            [6, REV as i32, 4, 0],
            [11, 21, 0, 0],
        ]);
        istore_clean_ends(&mut list);
        dump(&mut o, "cleanEnds", &list);

        // ---- istoreInvert ----
        let mut list = mk(&[[4, GAPF as i32, 0, 0]]);
        o.push_str(&format!("invert gap ret={}\n", istore_invert(&mut list, 4)));
        dump(&mut o, "invert.gap", &list);
        let mut list = mk(&[[3, 0, 1, 7], [3, REV as i32, 4, 0]]);
        o.push_str(&format!(
            "invert range ret={}\n",
            istore_invert(&mut list, 6)
        ));
        dump(&mut o, "invert.range", &list);
        let mut list = mk(&[[3, 0, 1, 7], [3, 0, 3, 8], [3, REV as i32, 5, 0]]);
        o.push_str(&format!(
            "invert chain ret={}\n",
            istore_invert(&mut list, 8)
        ));
        dump(&mut o, "invert.chain", &list);
        let mut list = mk(&[[3, 0, 1, 7], [11, 21, 0, 0]]);
        o.push_str(&format!(
            "invert open ret={}\n",
            istore_invert(&mut list, 5)
        ));
        dump(&mut o, "invert.open", &list);

        // ---- extract / copy ----
        let list = mk(&[[3, 0, 0, 5], [3, REV as i32, 5, 0], [11, 21, 0, 0]]);
        let mut nlist = Vec::new();
        o.push_str(&format!(
            "extract ret={}\n",
            istore_extract_changes(&list, &mut nlist, 2, 4, 0, 6)
        ));
        dump(&mut o, "extract", &nlist);
        let mut nlist = Vec::new();
        o.push_str(&format!(
            "copyNonIndex ret={}\n",
            istore_copy_non_index(&list, &mut nlist)
        ));
        dump(&mut o, "copyNonIndex", &nlist);
        let list = mk(&[[9, 0, 2, 1], [9, SURF as i32, 2, 2]]);
        let mut nlist = Vec::new();
        o.push_str(&format!(
            "copyContSurf ret={}\n",
            istore_copy_cont_surf_items(&list, &mut nlist, 2, 7, 1)
        ));
        dump(&mut o, "copyContSurf.surf", &nlist);
        let mut nlist = Vec::new();
        o.push_str(&format!(
            "copyContSurf cont ret={}\n",
            istore_copy_cont_surf_items(&list, &mut nlist, 2, 7, 0)
        ));
        dump(&mut o, "copyContSurf.cont", &nlist);

        // ---- istoreBreakContour ----
        let mut cont = Icont {
            pts: vec![Default::default(); 4],
            store: mk(&[[3, 0, 0, 2], [3, REV as i32, 4, 0]]),
            ..Default::default()
        };
        let mut ncont = Icont {
            pts: vec![Default::default(); 2],
            ..Default::default()
        };
        o.push_str(&format!(
            "breakContour ret={}\n",
            istore_break_contour(&mut cont, &mut ncont, 1, 2)
        ));
        dump(&mut o, "breakContour.new", &ncont.store);
        dump(&mut o, "breakContour.old", &cont.store);

        // ---- counting / lookup helpers ----
        let list = mk(&[
            [9, 0, 1, 1],
            [9, SURF as i32, 1, 2],
            [9, 0, 2, 3],
            [11, 21, 0, 0],
        ]);
        o.push_str(&format!(
            "countContSurf(1,0) = {}\n",
            istore_count_cont_surf_items(&list, 1, 0)
        ));
        o.push_str(&format!(
            "countContSurf(1,1) = {}\n",
            istore_count_cont_surf_items(&list, 1, 1)
        ));
        o.push_str(&format!(
            "countContSurf(99,0) = {}\n",
            istore_count_cont_surf_items(&list, 99, 0)
        ));
        o.push_str(&format!(
            "countItems(9) = {}\n",
            istore_count_items(&list, 9, 0)
        ));
        o.push_str(&format!(
            "countItems(9,stop) = {}\n",
            istore_count_items(&list, 9, 1)
        ));
        o.push_str(&format!(
            "skipToIndex(0) = {}\n",
            istore_skip_to_index(&list, 0)
        ));
        o.push_str(&format!(
            "skipToIndex(2) = {}\n",
            istore_skip_to_index(&list, 2)
        ));
        o.push_str(&format!(
            "skipToIndex(3) = {}\n",
            istore_skip_to_index(&list, 3)
        ));
        o.push_str(&format!(
            "firstChangeIndex = {}\n",
            istore_first_change_index(&list)
        ));
        let list = mk(&[
            [4, GAPF as i32, 2, 0],
            [5, 0, 2, 12],
            [3, 0, 4, 1],
            [3, REV as i32, 5, 0],
        ]);
        o.push_str(&format!(
            "pointIsGap(2) = {}\n",
            istore_point_is_gap(&list, 2)
        ));
        o.push_str(&format!(
            "pointIsGap(3) = {}\n",
            istore_point_is_gap(&list, 3)
        ));
        o.push_str(&format!(
            "connectNumber(2) = {}\n",
            istore_connect_number(&list, 2)
        ));
        o.push_str(&format!(
            "connectNumber(3) = {}\n",
            istore_connect_number(&list, 3)
        ));
        for i in 0..=6 {
            o.push_str(&format!(
                "retainPoint({}) = {}\n",
                i,
                istore_retain_point(&list, i)
            ));
        }
        o.push_str(&format!(
            "transState(1) = {}\n",
            istore_trans_state_matches(&list, 1)
        ));
        o.push_str(&format!(
            "transState(0) = {}\n",
            istore_trans_state_matches(&list, 0)
        ));
        o.push_str(&format!("checksum = {:.1}\n", istore_checksum(&list)));

        // ---- istoreNextObjItem ----
        let list = mk(&[
            [3, 0, 2, 4],
            [4, GAPF as i32, 3, 0],
            [3, REV as i32, 5, 0],
            [9, SURF as i32, 7, 0],
            [11, 21, 0, 0],
        ]);
        let mut cursor = 0;
        o.push_str(&format!(
            "nextObjItem first type={}\n",
            istore_next_obj_item(&list, 2, 7, 1, &mut cursor).map_or(-1, |s| s.type_)
        ));
        o.push_str(&format!(
            "nextObjItem next type={}\n",
            istore_next_obj_item(&list, 2, 7, 0, &mut cursor).map_or(-1, |s| s.type_)
        ));
        o.push_str(&format!(
            "nextObjItem end type={}\n",
            istore_next_obj_item(&list, 2, 7, 0, &mut cursor).map_or(-1, |s| s.type_)
        ));

        // ---- draw property paths ----
        let mut obj = Iobj {
            red: 0.1,
            green: 0.2,
            blue: 0.3,
            symbol: 4,
            store: mk(&[[1, (SURF | (3 << 2)) as i32, 3, 0], [8, 0, 0, -3]]),
            ..Default::default()
        };
        obj.store[0].value = StoreUnion::from_b([128, 64, 32, 0]);
        istore_sort(&mut obj.store);
        let mut def = DrawProps::default();
        istore_default_draw_props(&obj, &mut def);
        dump_props(&mut o, "default", &def);
        let mut cont_props = DrawProps::default();
        let (mut cont_state, mut surf_state) = (0, 0);
        let ret = istore_cont_surf_draw_props(
            &obj.store,
            &def,
            &mut cont_props,
            0,
            3,
            &mut cont_state,
            &mut surf_state,
        );
        o.push_str(&format!(
            "contSurf ret={} contState={} surfState={}\n",
            ret, cont_state, surf_state
        ));
        dump_props(&mut o, "contSurf", &cont_props);
        let ret = istore_cont_surf_draw_props(
            &obj.store,
            &def,
            &mut cont_props,
            0,
            -1,
            &mut cont_state,
            &mut surf_state,
        );
        o.push_str(&format!(
            "contSurf negsurf ret={} contState={} surfState={}\n",
            ret, cont_state, surf_state
        ));
        dump_props(&mut o, "contSurf.negsurf", &cont_props);
        let ret = istore_cont_surf_draw_props(
            &obj.store,
            &def,
            &mut cont_props,
            -1,
            3,
            &mut cont_state,
            &mut surf_state,
        );
        o.push_str(&format!(
            "contSurf negco ret={} contState={} surfState={}\n",
            ret, cont_state, surf_state
        ));
        dump_props(&mut o, "contSurf.negco", &cont_props);
        let list = mk(&[[3, 0, 2, 4], [3, REV as i32, 3, 0], [11, 21, 0, 0]]);
        let def = DrawProps {
            trans: 1,
            ..Default::default()
        };
        for i in 0..=4 {
            let mut pt = DrawProps::default();
            let ret = istore_list_point_props(&list, &def, &mut pt, i);
            o.push_str(&format!(
                "listPointProps({}) = {} trans={} gap={} con={} v1={:.6}\n",
                i, ret, pt.trans, pt.gap, pt.connect, pt.value1
            ));
        }

        // ---- min/max ----
        let mut obj = Iobj {
            store: mk(&[[10, 4, 0, 0]]),
            cont: vec![Icont {
                store: mk(&[[10, 4, 0, 0]]),
                ..Default::default()
            }],
            ..Default::default()
        };
        obj.store[0].value = StoreUnion::from_f(3.);
        obj.cont[0].store[0].value = StoreUnion::from_f(7.);
        o.push_str(&format!(
            "findAddMinMax1 ret={}\n",
            istore_find_add_min_max1(&mut obj)
        ));
        dump(&mut o, "minmax.store", &obj.store);
        let (mut mn, mut mx) = (-7., -7.);
        let ret = istore_get_min_max(&obj.store, obj.cont.len() as i32, 11, &mut mn, &mut mx);
        o.push_str(&format!(
            "getMinMax ret={} min={:.6} max={:.6}\n",
            ret, mn, mx
        ));
        o.push_str(&format!(
            "getMinMax missing ret={}\n",
            istore_get_min_max(&obj.store, obj.cont.len() as i32, 13, &mut mn, &mut mx)
        ));
        o.push_str(&format!(
            "findAddMinMax bad ret={}\n",
            istore_find_add_min_max(&mut obj, 11)
        ));
        o.push_str(&format!(
            "addMinMax bad ret={}\n",
            istore_add_min_max(&mut obj.store, 12, 0., 1.)
        ));

        // ---- istoreGenerateItems ----
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
        o.push_str(&format!(
            "generateItems ret={}\n",
            istore_generate_items(&mut list, &props, all, 8, all)
        ));
        for (i, store) in list.iter().enumerate() {
            o.push_str(&format!(
                "gen[{}] type={} flags={} index={} bytes={},{},{} value.i={} value.f={:.6}\n",
                i,
                store.type_,
                store.flags,
                (store.index.i()),
                (store.value.b()[0]),
                (store.value.b()[1]),
                (store.value.b()[2]),
                (store.value.i()),
                (store.value.f())
            ));
        }

        // ================= PART 2: fix-discriminating cases =================
        let mut list = mk(&[[3, 0, 0, 5], [3, REV as i32, 2, 0], [3, 0, 2, 7]]);
        o.push_str(&format!(
            "p2 endChange endfirst ret={}\n",
            istore_end_change(&mut list, 3, 2)
        ));
        dump(&mut o, "p2.endChange.endfirst", &list);

        let mut list = mk(&[[3, REV as i32, 0, 0], [3, NOIND as i32, 0, 0]]);
        istore_clean_ends(&mut list);
        dump(&mut o, "p2.cleanEnds.noindex", &list);
        let mut list = Vec::new();
        istore_clean_ends(&mut list);
        dump(&mut o, "p2.cleanEnds.empty", &list);

        let mut list = mk(&[[-5, 40000, 0, 0], [7, 2, 0, 0]]);
        list[1].index = StoreUnion::from_s([30000, 30000]);
        list[1].value = StoreUnion::from_s([30000, 30000]);
        o.push_str(&format!(
            "p2 checksum wide = {:.1}\n",
            istore_checksum(&list)
        ));

        let mut list = mk(&[
            [3, 0, 0, 4],
            [3, REV as i32, 1, 0],
            [6, 0, 0, 2],
            [6, REV as i32, 3, 0],
        ]);
        o.push_str(&format!(
            "p2 breakChanges ret={}\n",
            istore_break_changes(&mut list, 2, 5)
        ));
        dump(&mut o, "p2.breakChanges.endedbefore", &list);

        let mut list = mk(&[[4, GAPF as i32, 1, 0], [4, GAPF as i32, 1, 0], [3, 0, 2, 5]]);
        o.push_str(&format!(
            "p2 deletePoint onepoint ret={}\n",
            istore_delete_point(&mut list, 1, 6)
        ));
        dump(&mut o, "p2.deletePoint.onepoint", &list);
        let mut list = mk(&[[3, 0, 1, 5], [3, 0, 2, 5], [3, REV as i32, 4, 0]]);
        o.push_str(&format!(
            "p2 deletePoint samestart ret={}\n",
            istore_delete_point(&mut list, 1, 6)
        ));
        dump(&mut o, "p2.deletePoint.samestart", &list);

        let mut list = mk(&[[3, 0, 1, 1], [3, 0, 2, 2], [3, 0, 3, 3]]);
        istore_shift_index(&mut list, 2, 1, 5);
        dump(&mut o, "p2.shiftIndex.start", &list);

        let list = mk(&[[3, 0, 0, 5], [3, REV as i32, 5, 0]]);
        let mut nlist = Vec::new();
        o.push_str(&format!(
            "p2 extract inverted ret={}\n",
            istore_extract_changes(&list, &mut nlist, 4, 2, 0, 6)
        ));
        dump(&mut o, "p2.extract.inverted", &nlist);
        let mut nlist = Vec::new();
        o.push_str(&format!(
            "p2 extract whole ret={}\n",
            istore_extract_changes(&list, &mut nlist, 0, 5, 0, 6)
        ));
        dump(&mut o, "p2.extract.whole", &nlist);
        let mut nlist = Vec::new();
        o.push_str(&format!(
            "p2 extract empty ret={}\n",
            istore_extract_changes(&[], &mut nlist, 0, 5, 0, 6)
        ));
        dump(&mut o, "p2.extract.emptysrc", &nlist);

        let mut list = mk(&[
            [1, 3 << 2, 1, 0],
            [4, GAPF as i32, 1, 0],
            [3, 0, 3, 6],
            [1, (REV | (3 << 2)) as i32, 4, 0],
            [11, 21, 0, 0],
        ]);
        list[0].value = StoreUnion::from_b([10, 20, 30, 0]);
        let def = DrawProps {
            red: 0.5,
            trans: 2,
            linewidth: 4,
            ..Default::default()
        };
        let mut pt = def;
        let (mut state, mut changes, mut cursor) = (0, 0, 0);
        o.push_str(&format!(
            "p2 firstChangeIndex = {}\n",
            istore_first_change_index(&list)
        ));
        for i in 0..5 {
            let next =
                istore_next_change(&list, &mut cursor, &def, &mut pt, &mut state, &mut changes);
            o.push_str(&format!(
                "p2 nextChange[{}] = {} state={} changes={}\n",
                i, next, state, changes
            ));
            dump_props(&mut o, "p2.nextChange", &pt);
            if next < 0 {
                break;
            }
        }
        let mut nlist = Vec::new();
        o.push_str(&format!(
            "p2 genPointItems ret={}\n",
            istore_gen_point_items(&list, &def, 0, 1, &mut nlist, 9, 1 | 4)
        ));
        dump(&mut o, "p2.genPointItems", &nlist);

        let mut obj = Iobj {
            store: mk(&[[10, 4, 0, 0]]),
            cont: vec![Icont {
                store: mk(&[[10, 4, 0, 0]]),
                ..Default::default()
            }],
            mesh: vec![crate::imod::libimod::imodel::Imesh {
                store: mk(&[[10, 4, 0, 0]]),
                ..Default::default()
            }],
            ..Default::default()
        };
        o.push_str(&format!(
            "p2 countObjectItems all = {}\n",
            istore_count_object_items(&obj, 10, 1, 1, 0)
        ));
        o.push_str(&format!(
            "p2 countObjectItems cont = {}\n",
            istore_count_object_items(&obj, 10, 1, 0, 0)
        ));
        o.push_str(&format!(
            "p2 countObjectItems stop = {}\n",
            istore_count_object_items(&obj, 10, 1, 1, 1)
        ));
        o.push_str(&format!(
            "p2 countObjectItems none = {}\n",
            istore_count_object_items(&obj, 12, 1, 1, 0)
        ));
        obj.cont[0].pts = vec![Default::default(); 3];
        obj.cont[0].surf = 0;
        obj.cont[0].store = mk(&[[3, 0, 0, 9]]);
        let mut cont_props = DrawProps::default();
        let mut pt_props = DrawProps::default();
        o.push_str(&format!(
            "p2 pointDrawProps = {}\n",
            istore_point_draw_props(&obj, &mut cont_props, &mut pt_props, 0, 0)
        ));
        dump_props(&mut o, "p2.pointDrawProps", &pt_props);

        o.push_str(&format!(
            "p2 retainPoint empty = {}\n",
            istore_retain_point(&[], 1)
        ));
        o.push_str(&format!(
            "p2 skipToIndex empty = {}\n",
            istore_skip_to_index(&[], 1)
        ));
        o.push_str(&format!(
            "p2 firstChangeIndex empty = {}\n",
            istore_first_change_index(&[])
        ));
        o.push_str(&format!(
            "p2 pointIsGap empty = {}\n",
            istore_point_is_gap(&[], 1)
        ));
        o.push_str(&format!(
            "p2 connectNumber empty = {}\n",
            istore_connect_number(&[], 1)
        ));
        o.push_str(&format!(
            "p2 countContSurf empty = {}\n",
            istore_count_cont_surf_items(&[], 1, 0)
        ));
        o.push_str(&format!(
            "p2 transState empty = {}\n",
            istore_trans_state_matches(&[], 0)
        ));
        o.push_str(&format!(
            "p2 checksum empty = {:.1}\n",
            istore_checksum(&[])
        ));
        let mut empty = Vec::new();
        o.push_str(&format!(
            "p2 breakChanges empty = {}\n",
            istore_break_changes(&mut empty, 1, 4)
        ));
        let mut empty = Vec::new();
        o.push_str(&format!(
            "p2 invert empty = {}\n",
            istore_invert(&mut empty, 4)
        ));

        // ================= PART 3 =================
        let props = DrawProps {
            red: 1.5,
            green: -0.1,
            blue: 0.999,
            fill_red: 2.,
            fill_green: 0.5,
            fill_blue: 1.,
            ..Default::default()
        };
        let mut list = Vec::new();
        o.push_str(&format!(
            "p3 generateItems ret={}\n",
            istore_generate_items(&mut list, &props, 1 | 2, 3, 1 | 2)
        ));
        for (i, store) in list.iter().enumerate() {
            o.push_str(&format!(
                "p3 gen[{}] type={} bytes={},{},{}\n",
                i,
                store.type_,
                (store.value.b()[0]),
                (store.value.b()[1]),
                (store.value.b()[2])
            ));
        }

        let mut list = mk(&[[3, 0, 1, 1], [11, 21, 0, 0], [13, 21, 0, 0]]);
        list[1].index = StoreUnion::from_f(1.);
        list[1].value = StoreUnion::from_f(2.);
        o.push_str(&format!(
            "p3 addMinMax update ret={}\n",
            istore_add_min_max(&mut list, 11, 4., 8.)
        ));
        for (i, store) in list.iter().enumerate() {
            o.push_str(&format!(
                "p3 minmax[{}] type={} flags={} index.f={:.3} value.f={:.3}\n",
                i,
                store.type_,
                store.flags,
                (store.index.f()),
                (store.value.f())
            ));
        }
        o.push_str(&format!(
            "p3 addMinMax new ret={}\n",
            istore_add_min_max(&mut list, 15, 1., 3.)
        ));
        o.push_str(&format!("p3 size after new = {}\n", list.len()));

        let list = mk(&[
            [3, 0, 0, 0],
            [3, 0, 1, 0],
            [3, 0, 1, 0],
            [3, 0, 1, 0],
            [3, 0, 4, 0],
            [3, 0, 4, 0],
            [3, 0, 7, 0],
            [3, 0, 9, 0],
            [11, 21, 0, 0],
        ]);
        for i in -1..=10 {
            let (lookup, after) = istore_lookup(&list, i);
            o.push_str(&format!(
                "p3 lookup({}) = {} after {}\n",
                i,
                lookup.map_or(-1, |v| v as i32),
                after
            ));
        }
        let mut list = mk(&[[11, 21, 0, 0], [13, 21, 0, 0]]);
        let (lookup, after) = istore_lookup(&list, 0);
        o.push_str(&format!(
            "p3 lookup noindexonly = {} after {}\n",
            lookup.map_or(-1, |v| v as i32),
            after
        ));
        o.push_str(&format!(
            "p3 insert into noindexonly ret={}\n",
            istore_insert(
                &mut list,
                Istore {
                    type_: 3,
                    flags: 0,
                    index: StoreUnion::from_i(5),
                    value: StoreUnion::from_i(1),
                }
            )
        ));
        dump(&mut o, "p3.insert.noindexonly", &list);

        let mut list = mk(&[[6, 0, 1, 4]]);
        istore_clear_range(&mut list, 3, 0, 4);
        dump(&mut o, "p3.clearRange.nochange", &list);
        istore_clear_range(&mut list, 6, 3, 4);
        dump(&mut o, "p3.clearRange.startbefore", &list);

        let mut list = mk(&[[9, SURF as i32, 1, 1], [9, SURF as i32, 2, 2]]);
        istore_delete_cont_surf(&mut list, 1, 1);
        dump(&mut o, "p3.deleteContSurf.surfonly", &list);
        let mut list = mk(&[[9, 0, 0, 1], [9, SURF as i32, 1, 2], [9, 0, 2, 3]]);
        istore_delete_cont_surf(&mut list, 0, 0);
        dump(&mut o, "p3.deleteContSurf.shiftskipssurf", &list);

        let expected = ISTORE_C_DRIVER;
        for (line, (got, want)) in o.lines().zip(expected.lines()).enumerate() {
            assert_eq!(got, want, "line {} differs from istore.c driver", line + 1);
        }
        assert_eq!(o.lines().count(), expected.lines().count());
    }
    #[test]
    fn source_store_order_lookup_and_revert() {
        let mut list = Vec::new();
        istore_insert(
            &mut list,
            Istore {
                index: StoreUnion::from_i(3),
                ..Istore::default()
            },
        );
        istore_insert(
            &mut list,
            Istore {
                index: StoreUnion::from_i(2),
                ..Istore::default()
            },
        );
        istore_insert(
            &mut list,
            Istore {
                index: StoreUnion::from_i(2),
                flags: 1 << 5,
                ..Istore::default()
            },
        );
        istore_insert(
            &mut list,
            Istore {
                index: StoreUnion::from_i(0),
                flags: 1 << 4,
                ..Istore::default()
            },
        );
        assert_eq!((list[0].index.i()), 2);
        assert_ne!(list[0].flags & (1 << 5), 0);
        assert_eq!((list[1].index.i()), 2);
        assert_eq!(istore_lookup(&list, 2), (Some(0), 2));
        assert_ne!(list.last().unwrap().flags & (1 << 4), 0);
    }
    #[test]
    fn point_gap_and_connect_number_follow_source_index_lookup() {
        let list = [
            Istore {
                type_: 4,
                index: StoreUnion::from_i(2),
                ..Default::default()
            },
            Istore {
                type_: 5,
                index: StoreUnion::from_i(2),
                value: StoreUnion::from_i(12),
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
            value: StoreUnion::from_f(3.),
            ..Istore::default()
        };
        let mut object = Iobj {
            store: vec![value],
            cont: vec![crate::imod::libimod::imodel::Icont {
                store: vec![Istore {
                    type_: 10,
                    value: StoreUnion::from_f(7.),
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
        assert_eq!(
            istore_get_min_max(
                &object.store,
                object.cont.len() as i32,
                11,
                &mut min,
                &mut max
            ),
            1
        );
        assert_eq!((min, max), (3., 7.));
        let surface = Istore {
            index: StoreUnion::from_i(4),
            flags: 1 << 6,
            ..Istore::default()
        };
        assert_eq!(istore_count_cont_surf_items(&[surface], 4, 1), 1);
    }
    #[test]
    fn first_skip_and_trans_state_match_source_rules() {
        let list = [
            Istore {
                index: StoreUnion::from_i(2),
                ..Default::default()
            },
            Istore {
                index: StoreUnion::from_i(5),
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
                value: StoreUnion::from_i(2),
                ..Default::default()
            },
            Istore {
                type_: 3,
                flags: 1 << 5,
                value: StoreUnion::from_i(0),
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
                index: StoreUnion::from_i(2),
                value: StoreUnion::from_i(4),
                ..Default::default()
            },
            Istore {
                type_: 3,
                index: StoreUnion::from_i(3),
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
                index: StoreUnion::from_i(2),
                ..Default::default()
            },
            Istore {
                type_: 5,
                index: StoreUnion::from_i(2),
                value: StoreUnion::from_i(9),
                ..Default::default()
            },
            Istore {
                type_: 10,
                index: StoreUnion::from_i(2),
                value: StoreUnion::from_f(3.),
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
            index: StoreUnion::from_i(2),
            value: StoreUnion::from_i(7),
            ..Default::default()
        };
        assert_eq!(istore_insert_change(&mut list, start), 0);
        assert_eq!(istore_end_change(&mut list, 3, 5), 0);
        assert_eq!(list.len(), 2);
        assert_eq!((list[0].index.i()), 2);
        assert_ne!(list[1].flags & (1 << 5), 0);
        assert_eq!((list[1].index.i()), 5);
        assert_eq!(istore_clear_change(&mut list, 3, 3), 0);
        assert!(list.is_empty());
        assert_eq!(istore_clear_change(&mut list, 3, 3), 1);
    }

    #[test]
    fn clear_range_and_one_index_items_follow_source_matching_rules() {
        let mut changes = vec![
            Istore {
                type_: 6,
                index: StoreUnion::from_i(1),
                value: StoreUnion::from_i(4),
                ..Default::default()
            },
            Istore {
                type_: 6,
                index: StoreUnion::from_i(4),
                flags: 1 << 5,
                ..Default::default()
            },
            Istore {
                type_: 6,
                index: StoreUnion::from_i(7),
                value: StoreUnion::from_i(8),
                ..Default::default()
            },
        ];
        istore_clear_range(&mut changes, 6, 2, 4);
        assert_eq!(changes.len(), 1);
        assert_eq!((changes[0].index.i()), 7);

        let mut single = Vec::new();
        let point = Istore {
            type_: 9,
            flags: 1 << 7,
            index: StoreUnion::from_i(4),
            value: StoreUnion::from_i(2),
        };
        assert_eq!(istore_add_one_index_item(&mut single, point), 0);
        assert_eq!(
            istore_add_one_index_item(
                &mut single,
                Istore {
                    value: StoreUnion::from_i(9),
                    ..point
                },
            ),
            0
        );
        assert_eq!(single.len(), 1);
        assert_eq!((single[0].value.i()), 9);
        let surface = Istore {
            flags: (1 << 7) | (1 << 6),
            value: StoreUnion::from_i(3),
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
        assert_eq!(list[0].flags, 3 << 2);
        assert_eq!(list[1].flags, 3 << 2);
        assert_eq!((list[0].value.b()), [255, 127, 0, 0]);
        assert_eq!((list[4].value.f()), 1.25);

        // `imodWriteStore` must retain source byte order for generated colour
        // payloads: treating this as GEN_STORE_SHORT reverses each pair.
        let path = std::env::temp_dir().join(format!(
            "imod-rs-istore-generated-colour-{}.bin",
            std::process::id()
        ));
        let mut file = ImodFile::open(path.to_str().unwrap(), "wb").unwrap();
        assert_eq!(imod_write_store(&list[..1], 0x5354_4f52, &mut file), 0);
        drop(file);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[16..20], &[255, 127, 0, 0]);
        std::fs::remove_file(path).unwrap();

        let contour_list = [Istore {
            type_: 3,
            index: StoreUnion::from_i(2),
            value: StoreUnion::from_i(9),
            ..Default::default()
        }];
        let mut mesh_list = Vec::new();
        assert_eq!(
            istore_gen_point_items(&contour_list, &props, 0, 2, &mut mesh_list, 4, 1 << 2,),
            0
        );
        assert_eq!(mesh_list.len(), 1);
        assert_eq!((mesh_list[0].index.i()), 4);
        assert_eq!((mesh_list[0].value.i()), 9);
    }

    #[test]
    fn break_find_shift_and_delete_follow_source_index_rules() {
        let mut changes = vec![Istore {
            type_: 3,
            index: StoreUnion::from_i(0),
            value: StoreUnion::from_i(4),
            ..Default::default()
        }];
        assert_eq!(istore_break_changes(&mut changes, 2, 5), 0);
        assert_eq!(changes.len(), 3);
        assert_eq!((changes[1].index.i()), 2);
        assert_ne!(changes[1].flags & (1 << 5), 0);
        assert_eq!((changes[2].index.i()), 2);
        assert_eq!(istore_find_break(&changes, 2), 2);

        istore_shift_index(&mut changes, 2, -1, 1);
        assert_eq!((changes[1].index.i()), 3);
        assert_eq!((changes[2].index.i()), 3);
        assert_eq!(istore_delete_point(&mut changes, 3, 6), 0);
        assert_eq!(changes.len(), 2);
        assert_eq!((changes[1].index.i()), 3);
        assert_ne!(changes[1].flags & (1 << 5), 0);

        let mut cont_surf = vec![
            Istore {
                type_: 9,
                index: StoreUnion::from_i(1),
                ..Default::default()
            },
            Istore {
                type_: 9,
                flags: 1 << 6,
                index: StoreUnion::from_i(1),
                ..Default::default()
            },
            Istore {
                type_: 9,
                index: StoreUnion::from_i(2),
                ..Default::default()
            },
        ];
        istore_delete_cont_surf(&mut cont_surf, 1, 1);
        assert_eq!(cont_surf.len(), 2);
        istore_delete_cont_surf(&mut cont_surf, 1, 0);
        assert_eq!(cont_surf.len(), 1);
        assert_eq!((cont_surf[0].index.i()), 1);
    }

    #[test]
    fn extract_copy_clean_and_invert_follow_source_store_ranges() {
        let changes = vec![
            Istore {
                type_: 3,
                index: StoreUnion::from_i(0),
                value: StoreUnion::from_i(5),
                ..Default::default()
            },
            Istore {
                type_: 3,
                flags: 1 << 5,
                index: StoreUnion::from_i(5),
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
        assert_eq!((extracted[0].index.i()), 0);
        assert_eq!((extracted[1].index.i()), 3);

        let mut copied = Vec::new();
        assert_eq!(istore_copy_non_index(&changes, &mut copied), 0);
        assert_eq!(copied.len(), 1);
        let mut point_and_surface = vec![
            Istore {
                type_: 9,
                index: StoreUnion::from_i(2),
                ..Default::default()
            },
            Istore {
                type_: 9,
                flags: 1 << 6,
                index: StoreUnion::from_i(2),
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
                .any(|store| { store.flags & (1 << 6) != 0 && (store.index.i()) == 7 })
        );
        istore_clean_ends(&mut point_and_surface);
        assert_eq!(point_and_surface.len(), 2);

        let mut gap = vec![Istore {
            type_: 4,
            flags: 1 << 7,
            index: StoreUnion::from_i(0),
            ..Default::default()
        }];
        assert_eq!(istore_invert(&mut gap, 4), 0);
        assert_eq!((gap[0].index.i()), 2);

        // `istoreInvert` consumes the matching successor while translating a
        // multi-point change; otherwise that old end is translated a second
        // time as an independent start.
        let mut range = vec![
            Istore {
                type_: 3,
                index: StoreUnion::from_i(1),
                value: StoreUnion::from_i(7),
                ..Default::default()
            },
            Istore {
                type_: 3,
                flags: 1 << 5,
                index: StoreUnion::from_i(4),
                ..Default::default()
            },
        ];
        assert_eq!(istore_invert(&mut range, 6), 0);
        assert_eq!(range.len(), 2);
        assert_eq!((range[0].index.i()), 2);
        assert_eq!(range[0].flags & (1 << 5), 0);
        assert_eq!((range[1].index.i()), 5);
        assert_ne!(range[1].flags & (1 << 5), 0);
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
                    index: StoreUnion::from_i(0),
                    value: StoreUnion::from_i(2),
                    ..Default::default()
                },
                Istore {
                    type_: 3,
                    flags: 1 << 5,
                    index: StoreUnion::from_i(4),
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
        assert_eq!((new_contour.store[0].index.i()), 0);
        assert_eq!((new_contour.store[1].index.i()), 2);
        assert_eq!(contour.store.len(), 4);
    }

    #[test]
    fn binary_store_roundtrip_preserves_source_twelve_byte_records() {
        let path = std::env::temp_dir().join(format!("imod-rs-istore-{}.bin", std::process::id()));
        let mut file = ImodFile::open(path.to_str().unwrap(), "wb").unwrap();
        let source = vec![
            Istore {
                type_: 1,
                flags: 1 | (3 << 2),
                index: StoreUnion::from_f(1.5),
                value: StoreUnion::from_b([1, 2, 3, 4]),
            },
            Istore {
                type_: 10,
                flags: 2,
                index: StoreUnion::from_s([-2, 9]),
                value: StoreUnion::from_i(17),
            },
        ];
        assert_eq!(imod_write_store(&source, 0x5354_4f52, &mut file), 0);
        // Two 12-byte records plus the 4-byte id and 4-byte length.
        assert_eq!(file.seek(SeekFrom::End(0)).unwrap(), 32);
        drop(file);
        let mut file = ImodFile::open(path.to_str().unwrap(), "rb").unwrap();
        file.seek(SeekFrom::Start(4)).unwrap();
        let mut error = -1;
        let read = imod_read_store(&mut file, &mut error);
        assert_eq!(error, 0);
        let read = read.unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].flags, source[0].flags);
        assert_eq!((read[0].index.f()), 1.5);
        assert_eq!((read[0].value.b()), [1, 2, 3, 4]);
        assert_eq!((read[1].index.s()), [-2, 9]);
        assert_eq!((read[1].value.i()), 17);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn checksum_retain_and_object_iteration_follow_source_flags() {
        let list = vec![
            Istore {
                type_: 3,
                index: StoreUnion::from_i(2),
                value: StoreUnion::from_i(4),
                ..Default::default()
            },
            Istore {
                type_: 4,
                flags: 1 << 7,
                index: StoreUnion::from_i(3),
                ..Default::default()
            },
            Istore {
                type_: 3,
                flags: 1 << 5,
                index: StoreUnion::from_i(5),
                ..Default::default()
            },
            Istore {
                type_: 9,
                flags: 1 << 6,
                index: StoreUnion::from_i(7),
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
                    index: StoreUnion::from_i(3),
                    value: StoreUnion::from_b([128, 64, 32, 0]),
                },
                Istore {
                    type_: 8,
                    index: StoreUnion::from_i(0),
                    value: StoreUnion::from_i(-3),
                    ..Default::default()
                },
            ],
            cont: vec![Icont {
                surf: 3,
                store: vec![
                    Istore {
                        type_: 2,
                        index: StoreUnion::from_i(1),
                        value: StoreUnion::from_b([0, 255, 128, 0]),
                        ..Default::default()
                    },
                    Istore {
                        type_: 2,
                        flags: 1 << 5,
                        index: StoreUnion::from_i(2),
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
/// Original: `istoreRetainPoint` (`istore.c:614`).
pub fn istore_retain_point(list: &[Istore], index: i32) -> i32 {
    if list.is_empty() {
        return 0;
    }
    let (lookup, after) = istore_lookup(list, index);
    if lookup.is_some() {
        return 1;
    }
    for store in list.iter().skip(after) {
        if store.flags & ((1 << 4) | 3) != 0 || (store.index.i()) > index + 1 {
            break;
        }
        if store.flags & (1 << 5) != 0 {
            return 1;
        }
    }
    for store in list[..after].iter().rev() {
        if (store.index.i()) < index - 1 {
            break;
        }
        if store.type_ == 4 {
            return 1;
        }
    }
    0
}
/// Original: `istoreInsertChange` (`istore.c:657`).
pub fn istore_insert_change(list: &mut Vec<Istore>, store: Istore) -> i32 {
    let (found, mut after) = istore_lookup(list, (store.index.i()));

    // If there is a match at the current index, eliminate it
    if let Some(start) = found {
        let mut i = start;
        while i < after {
            if list[i].type_ == store.type_ {
                list.remove(i);
                after -= 1;
            } else {
                i += 1;
            }
        }
    }

    // Look backwards for multiple-point items and see if there is fully
    // matching start; if not insert the new item
    let mut lookup = found.unwrap_or(after);
    let mut need_item = true;
    if store.flags & (1 << 7) != 0 {
        lookup = 0;
    }
    for i in (0..lookup).rev() {
        if list[i].type_ == store.type_ {
            if list[i].flags & (1 << 5) == 0 && (list[i].value.i()) == (store.value.i()) {
                need_item = false;
            }
            break;
        }
    }

    // Insert if still needed
    if need_item && istore_insert(list, store) != 0 {
        return 1;
    }
    0
}
/// Original: `istoreEndChange` (`istore.c:724`).
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
                    // Search back for a previous start from `lookup - 1`
                    // (`istore.c:759`), not from the current item.
                    for previous in (0..lookup).rev() {
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
                index: StoreUnion::from_i(index),
                ..Default::default()
            },
        )
    } else {
        0
    }
}
/// Original: `istoreClearChange` (`istore.c:795`).
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
/// Original: `istoreClearRange` (`istore.c:834`).
pub fn istore_clear_range(list: &mut Vec<Istore>, type_: i16, start: i32, end: i32) {
    if list.is_empty() {
        return;
    }
    let (_, after) = istore_lookup(list, start);
    let mut has_change = false;
    for store in list.iter().skip(after) {
        if store.flags & ((1 << 4) | 3) != 0 || (store.index.i()) > end {
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
            let index = (list[i].index.i());
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
/// Original: `istoreAddOneIndexItem` (`istore.c:904`).
pub fn istore_add_one_index_item(list: &mut Vec<Istore>, store: Istore) -> i32 {
    let (lookup, after) = istore_lookup(list, (store.index.i()));
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
/// Original: `istoreClearOneIndexItem` (`istore.c:936`).
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
/// Original: `istoreGenerateItems` (`istore.c:963`).
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
                flags: 3 << 2,
                index: StoreUnion::from_i(index),
                // `(int)(255. * props->red)` truncates in double and is then
                // narrowed to b3dUByte by modular conversion.
                value: StoreUnion::from_b([
                    (255. * props.red as f64) as i32 as u8,
                    (255. * props.green as f64) as i32 as u8,
                    (255. * props.blue as f64) as i32 as u8,
                    0,
                ]),
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
                flags: 3 << 2,
                index: StoreUnion::from_i(index),
                value: StoreUnion::from_b([
                    (255. * props.fill_red as f64) as i32 as u8,
                    (255. * props.fill_green as f64) as i32 as u8,
                    (255. * props.fill_blue as f64) as i32 as u8,
                    0,
                ]),
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
                index: StoreUnion::from_i(index),
                value: StoreUnion::from_i(props.trans),
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
                index: StoreUnion::from_i(index),
                value: StoreUnion::from_i(props.linewidth),
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
                index: StoreUnion::from_i(index),
                value: StoreUnion::from_f(props.value1),
            },
        ) != 0
    {
        return 1;
    }
    0
}
/// Original: `istoreGenPointItems` (`istore.c:1023`).
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
/// Original: `istoreBreakChanges` (`istore.c:1044`).
pub fn istore_break_changes(list: &mut Vec<Istore>, index: i32, psize: i32) -> i32 {
    if list.is_empty() {
        return 0;
    }
    let mut cur_start = 0;
    while cur_start < list.len() {
        let current = list[cur_start];
        if current.flags & ((1 << 4) | 3) != 0 || (current.index.i()) >= index {
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
            if next.flags & ((1 << 4) | 3) != 0 || (next.index.i()) > index {
                break;
            }
            if store.type_ == next.type_ {
                if next.flags & (1 << 5) != 0 {
                    point_revert = (next.index.i());
                    break;
                }
                if (next.index.i()) == index {
                    need_restart = false;
                } else {
                    store = *next;
                }
            }
        }
        if point_revert > index {
            let mut store_end = store;
            store_end.flags |= 1 << 5;
            store_end.index = StoreUnion::from_i(index);
            if istore_insert(list, store_end) != 0 {
                return 1;
            }
            if need_restart {
                store.index = StoreUnion::from_i(index);
                if istore_insert(list, store) != 0 {
                    return 1;
                }
            }
        }
        cur_start += 1;
    }
    0
}
/// Original: `istoreFindBreak` (`istore.c:1128`).
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
/// Original: `istoreShiftIndex` (`istore.c:1151`).
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
        if store.flags & (1 << 6) == 0 && (store.index.i()) >= point_index {
            store.index = StoreUnion::from_i((store.index.i()) + amount);
        }
    }
    istore_sort(list);
}
/// Original: `istoreDeletePoint` (`istore.c:1185`).
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
                if next.flags & ((1 << 4) | 3) != 0 || (next.index.i()) > index + 1 {
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
                moved.index = StoreUnion::from_i((moved.index.i()) + 1);
                if istore_insert(list, moved) != 0 {
                    return 1;
                }
            }
        }
    }
    // Delete the current point items and then shift indexes; the moves above
    // only touch positions at or past `after`, so the original range still
    // holds exactly the items for this index (`istore.c:1256-1258`).
    list.drain(lookup..after);
    istore_shift_index(list, index + 1, lookup as i32, -1);
    0
}
/// Original: `istoreDeleteContSurf` (`istore.c:1267`).
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
/// Original: `istoreBreakContour` (`istore.c:1295`).
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
/// Original: `istoreInvert` (`istore.c:1337`).
pub fn istore_invert(list: &mut Vec<Istore>, psize: i32) -> i32 {
    if psize < 2 || list.is_empty() {
        return 0;
    }
    if istore_break_changes(list, psize, psize) != 0 {
        return 1;
    }
    let mut inverted = Vec::new();
    let mut cur_start = 0;
    while cur_start < list.len() {
        let current = list[cur_start];
        if current.flags & ((1 << 4) | 3) != 0 {
            if istore_insert(&mut inverted, current) != 0 {
                return 1;
            }
            cur_start += 1;
            continue;
        }
        // Copy one-point type but move gap back by one
        if current.flags & (1 << 7) != 0 {
            let mut item = current;
            item.index = StoreUnion::from_i(psize - 1 - (item.index.i()));
            if item.type_ == 4 {
                let moved = (item.index.i()) - 1;
                item.index = StoreUnion::from_i(if moved < 0 { psize - 1 } else { moved });
            }
            if istore_insert(&mut inverted, item) != 0 {
                return 1;
            }
            cur_start += 1;
            continue;
        }

        // Convert the start to an end and add it with inverted index, then
        // save the starting change
        let mut item = current;
        item.index = StoreUnion::from_i(psize - (item.index.i()));
        item.flags |= 1 << 5;
        if istore_insert(&mut inverted, item) != 0 {
            return 1;
        }
        // `store` tracks the previous matching item, which is what gets
        // emitted at each successor's inverted index (`istore.c:1386,1402`).
        let mut store = current;
        let mut next = cur_start + 1;
        while next < list.len() {
            let successor = list[next];
            if store.type_ == successor.type_ {
                store.index = StoreUnion::from_i(psize - (successor.index.i()));
                if istore_insert(&mut inverted, store) != 0 {
                    return 1;
                }
                store = successor;
                list.remove(next);
                if store.flags & (1 << 5) != 0 {
                    break;
                }
                continue;
            }
            next += 1;
        }
        cur_start += 1;
    }
    *list = inverted;
    0
}
/// Original: `istoreCleanEnds` (`istore.c:1420`).
pub fn istore_clean_ends(list: &mut Vec<Istore>) {
    let mut i = 0;
    while i < list.len() {
        if list[i].flags & ((1 << 4) | 3) != 0 {
            return;
        }
        if list[i].flags & (1 << 5) != 0 {
            let mut remove = false;
            for next in list.iter().skip(i + 1) {
                // `istore.c:1431` tests `stp->flags`, not `stp2->flags`; the
                // outer item is already known to carry an index, so only the
                // index comparison can break the scan.
                if (next.index.i()) != (list[i].index.i()) {
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
/// Original: `istoreExtractChanges` (`istore.c:1451`).
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
        store.index = StoreUnion::from_i((store.index.i()) + new_start - index_start);
        if istore_insert(new_list, store) != 0 {
            return 1;
        }
    }
    0
}
/// Original: `istoreCopyNonIndex` (`istore.c:1505`).
pub fn istore_copy_non_index(old_list: &[Istore], new_list: &mut Vec<Istore>) -> i32 {
    let (_, after) = istore_lookup(old_list, i32::MAX);
    for store in old_list.iter().skip(after) {
        if istore_insert(new_list, *store) != 0 {
            return 1;
        }
    }
    0
}
/// Original: `istoreCopyContSurfItems` (`istore.c:1523`).
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
                store.index = StoreUnion::from_i(index_to);
                if istore_insert(new_list, store) != 0 {
                    return 1;
                }
            }
        }
    }
    0
}
/// Original: `istoreDefaultDrawProps` (`istore.c:1579`).
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
/// Original: `istoreContSurfDrawProps` (`istore.c:1608`).
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
    if list.is_empty() {
        return 0;
    }
    // Set up to loop on surface entries first.  `istore.c:1626` skips the
    // `which = co; surfFlag = 0;` update at :1706 when `which < 0`, so a
    // negative surface number suppresses the contour pass entirely.
    let mut which = surf;
    let mut surf_flag: u16 = 1 << 6;
    for j in 0..2 {
        let mut state = 0;
        if which < 0 {
            continue;
        }
        let (found, after) = istore_lookup(list, which);
        if let Some(start) = found {
            for store in &list[start..after] {
                if (store.flags & (1 << 6)) != surf_flag {
                    continue;
                }
                match store.type_ {
                    1 => {
                        state |= 1;
                        cont_props.red = store.value.b()[0] as f32 / 255.;
                        cont_props.green = store.value.b()[1] as f32 / 255.;
                        cont_props.blue = store.value.b()[2] as f32 / 255.;
                    }
                    2 => {
                        state |= 2;
                        cont_props.fill_red = store.value.b()[0] as f32 / 255.;
                        cont_props.fill_green = store.value.b()[1] as f32 / 255.;
                        cont_props.fill_blue = store.value.b()[2] as f32 / 255.;
                    }
                    3 => {
                        state |= 4;
                        cont_props.trans = (store.value.i());
                    }
                    4 => {
                        state |= 8;
                        cont_props.gap = 1;
                    }
                    5 => {
                        state |= 16;
                        cont_props.connect = (store.value.i());
                    }
                    6 => {
                        state |= 32;
                        cont_props.linewidth = (store.value.i());
                    }
                    7 => {
                        state |= 64;
                        cont_props.linewidth2 = (store.value.i());
                    }
                    9 => {
                        state |= 256;
                        cont_props.symsize = (store.value.i());
                    }
                    8 => {
                        state |= 128;
                        cont_props.symflags &= !1;
                        cont_props.symtype = (store.value.i());
                        if cont_props.symtype < 0 {
                            cont_props.symtype = -1 - cont_props.symtype;
                            cont_props.symflags |= 1;
                        }
                    }
                    10 => {
                        state |= 512;
                        cont_props.value1 = (store.value.f());
                    }
                    24 => cont_props.no_cap = 1,
                    _ => {}
                }
            }
        }
        if j != 0 {
            *cont_state = state;
        } else {
            *surf_state = state;
        }

        // Next pass through, loop on contour entries
        which = co;
        surf_flag = 0;
    }
    *cont_state | *surf_state
}
/// Original: `istoreFirstChangeIndex` (`istore.c:1717`).
pub fn istore_first_change_index(list: &[Istore]) -> i32 {
    if list.is_empty() || list[0].flags & ((1 << 4) | 3) != 0 {
        return -1;
    }
    (list[0].index.i())
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
        let item = (store.index.i());
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
                    props.red = (store.value.b()[0]) as f32 / 255.;
                    props.green = (store.value.b()[1]) as f32 / 255.;
                    props.blue = (store.value.b()[2]) as f32 / 255.;
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
                    props.fill_red = (store.value.b()[0]) as f32 / 255.;
                    props.fill_green = (store.value.b()[1]) as f32 / 255.;
                    props.fill_blue = (store.value.b()[2]) as f32 / 255.;
                    *state |= 2;
                }
            }
            3 => {
                *changes |= 4;
                if ending {
                    props.trans = def.trans;
                    *state &= !4
                } else {
                    props.trans = (store.value.i());
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
                props.connect = (store.value.i())
            }
            6 => {
                *changes |= 32;
                if ending {
                    props.linewidth = def.linewidth;
                    *state &= !32
                } else {
                    props.linewidth = (store.value.i());
                    *state |= 32
                }
            }
            7 => {
                *changes |= 64;
                if ending {
                    props.linewidth2 = def.linewidth2;
                    *state &= !64
                } else {
                    props.linewidth2 = (store.value.i());
                    *state |= 64
                }
            }
            9 => {
                *changes |= 256;
                if ending {
                    props.symsize = def.symsize;
                    *state &= !256
                } else {
                    props.symsize = (store.value.i());
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
                    props.symtype = (store.value.i());
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
                    props.value1 = (store.value.f());
                    *state |= 512
                }
            }
            _ => {}
        }
    }
    -1
}
/// Original: `istorePointDrawProps` (`istore.c:1897`).
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
/// Original: `istoreListPointProps` (`istore.c:1917`).
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
/// Original: `istoreSkipToIndex` (`istore.c:1949`).
pub fn istore_skip_to_index(list: &[Istore], index: i32) -> i32 {
    let (lookup, after) = istore_lookup(list, index);
    let item = lookup.unwrap_or(after);
    if item >= list.len() || list[item].flags & ((1 << 4) | 3) != 0 {
        return -1;
    }
    (list[item].index.i())
}
/// Original: `istoreTransStateMatches` (`istore.c:1971`).
pub fn istore_trans_state_matches(list: &[Istore], state: i32) -> i32 {
    list.iter()
        .any(|store| {
            store.type_ == 3
                && store.flags & (1 << 5) == 0
                && (if (store.value.i()) != 0 { 1 } else { 0 }) == state
        })
        .then_some(1)
        .unwrap_or(0)
}
