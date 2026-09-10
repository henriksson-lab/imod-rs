//! Translation of `IMOD/libwarp/hash.c` and `hash.h`.
#![allow(dead_code)]

use core::ffi::{c_char, c_void};

/// C `ht_keycp` (`hash.h`).
pub type HtKeycp = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
/// C `ht_keyeq` (`hash.h`).
pub type HtKeyeq = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
/// C `ht_key2hash` (`hash.h`).
pub type HtKey2hash = unsafe extern "C" fn(*mut c_void) -> u32;

/// C `struct ht_bucket` (`hash.c`).
#[repr(C)]
pub struct HtBucket {
    pub key: *mut c_void,
    pub data: *mut c_void,
    pub id: i32,
    pub next: *mut HtBucket,
}

/// C `struct hashtable` (`hash.c`).
#[repr(C)]
pub struct Hashtable {
    pub size: i32,
    pub n: i32,
    pub naccum: i32,
    pub nhash: i32,
    pub cp: HtKeycp,
    pub eq: HtKeyeq,
    pub hash: HtKey2hash,
    pub table: *mut *mut HtBucket,
}

/// Original `ht_create` (`hash.c:61`).
pub unsafe fn ht_create(size: i32, cp: HtKeycp, eq: HtKeyeq, hash: HtKey2hash) -> *mut Hashtable {
    unsafe {
        let table = libc::malloc(core::mem::size_of::<Hashtable>()).cast::<Hashtable>();
        assert!(!table.is_null());
        if size <= 0 {
            libc::free(table.cast());
            return core::ptr::null_mut();
        }
        (*table).size = size;
        (*table).table = libc::malloc(core::mem::size_of::<*mut HtBucket>() * size as usize).cast();
        assert!(!(*table).table.is_null());
        let bucket = (*table).table;
        if bucket.is_null() {
            libc::free(table.cast());
            return core::ptr::null_mut();
        }
        for index in 0..size {
            *bucket.add(index as usize) = core::ptr::null_mut();
        }
        (*table).n = 0;
        (*table).naccum = 0;
        (*table).nhash = 0;
        (*table).eq = eq;
        (*table).cp = cp;
        (*table).hash = hash;
        table
    }
}

/// Original `ht_destroy` (`hash.c:102`).
pub unsafe fn ht_destroy(table: *mut Hashtable) {
    unsafe {
        if table.is_null() {
            return;
        }
        for index in 0..(*table).size {
            let mut bucket = *(*table).table.add(index as usize);
            while !bucket.is_null() {
                let previous = bucket;
                libc::free((*bucket).key);
                bucket = (*bucket).next;
                libc::free(previous.cast());
            }
        }
        libc::free((*table).table.cast());
        libc::free(table.cast());
    }
}

/// Original `ht_insert` (`hash.c:133`).
pub unsafe fn ht_insert(table: *mut Hashtable, key: *mut c_void, data: *mut c_void) -> *mut c_void {
    unsafe {
        let value = ((*table).hash)(key) % (*table).size as u32;
        if (*(*table).table.add(value as usize)).is_null() {
            let bucket = libc::malloc(core::mem::size_of::<HtBucket>()).cast::<HtBucket>();
            assert!(!bucket.is_null());
            (*bucket).key = ((*table).cp)(key);
            (*bucket).next = core::ptr::null_mut();
            (*bucket).data = data;
            (*bucket).id = (*table).naccum;
            *(*table).table.add(value as usize) = bucket;
            (*table).n += 1;
            (*table).naccum += 1;
            (*table).nhash += 1;
            return core::ptr::null_mut();
        }
        let mut bucket = *(*table).table.add(value as usize);
        while !bucket.is_null() {
            if ((*table).eq)(key, (*bucket).key) == 1 {
                let old_data = (*bucket).data;
                (*bucket).data = data;
                (*bucket).id = (*table).naccum;
                (*table).naccum += 1;
                return old_data;
            }
            bucket = (*bucket).next;
        }
        let bucket = libc::malloc(core::mem::size_of::<HtBucket>()).cast::<HtBucket>();
        assert!(!bucket.is_null());
        (*bucket).key = ((*table).cp)(key);
        (*bucket).data = data;
        (*bucket).next = *(*table).table.add(value as usize);
        (*bucket).id = (*table).naccum;
        *(*table).table.add(value as usize) = bucket;
        (*table).n += 1;
        (*table).naccum += 1;
        core::ptr::null_mut()
    }
}

/// Original `ht_find` (`hash.c:204`).
pub unsafe fn ht_find(table: *mut Hashtable, key: *mut c_void) -> *mut c_void {
    unsafe {
        let value = ((*table).hash)(key) % (*table).size as u32;
        let mut bucket = *(*table).table.add(value as usize);
        if bucket.is_null() {
            return core::ptr::null_mut();
        }
        while !bucket.is_null() {
            if ((*table).eq)(key, (*bucket).key) == 1 {
                return (*bucket).data;
            }
            bucket = (*bucket).next;
        }
        core::ptr::null_mut()
    }
}

/// Original `ht_delete` (`hash.c:227`).
pub unsafe fn ht_delete(table: *mut Hashtable, key: *mut c_void) -> *mut c_void {
    unsafe {
        let value = ((*table).hash)(key) % (*table).size as u32;
        if (*(*table).table.add(value as usize)).is_null() {
            return core::ptr::null_mut();
        }
        let mut previous: *mut HtBucket = core::ptr::null_mut();
        let mut bucket = *(*table).table.add(value as usize);
        while !bucket.is_null() {
            if ((*table).eq)(key, (*bucket).key) == 1 {
                let data = (*bucket).data;
                if !previous.is_null() {
                    (*previous).next = (*bucket).next;
                } else {
                    *(*table).table.add(value as usize) = (*bucket).next;
                    (*table).nhash -= 1;
                }
                libc::free((*bucket).key);
                libc::free(bucket.cast());
                (*table).n -= 1;
                return data;
            }
            previous = bucket;
            bucket = (*bucket).next;
        }
        core::ptr::null_mut()
    }
}

/// Original `ht_process` (`hash.c:281`).
pub unsafe fn ht_process(table: *mut Hashtable, function: unsafe extern "C" fn(*mut c_void)) {
    unsafe {
        for index in 0..(*table).size {
            let mut bucket = *(*table).table.add(index as usize);
            while !bucket.is_null() {
                function((*bucket).data);
                bucket = (*bucket).next;
            }
        }
    }
}

/// Original static `strhash` (`hash.c:298`).
pub unsafe extern "C" fn strhash(key: *mut c_void) -> u32 {
    unsafe {
        let mut string = key.cast::<c_char>();
        let mut hash_value = 0_u32;
        while *string != 0 {
            hash_value ^= core::ptr::read_unaligned(string.cast::<u32>());
            hash_value = hash_value.wrapping_shl(1);
            string = string.add(1);
        }
        hash_value
    }
}

/// Original static `strcp` (`hash.c:311`).
pub unsafe extern "C" fn strcp(key: *mut c_void) -> *mut c_void {
    unsafe { libc::strdup(key.cast()).cast() }
}

/// Original static `streq` (`hash.c:316`).
pub unsafe extern "C" fn streq(key1: *mut c_void, key2: *mut c_void) -> i32 {
    unsafe { (libc::strcmp(key1.cast(), key2.cast()) == 0) as i32 }
}

/// Original static `d1hash` (`hash.c:324`).
pub unsafe extern "C" fn d1hash(key: *mut c_void) -> u32 {
    unsafe {
        let value = key.cast::<u32>();
        (*value).wrapping_add(*value.add(1))
    }
}

/// Original static `d1cp` (`hash.c:335`).
pub unsafe extern "C" fn d1cp(key: *mut c_void) -> *mut c_void {
    unsafe {
        let new_key = libc::malloc(core::mem::size_of::<f64>()).cast::<f64>();
        *new_key = *key.cast::<f64>();
        new_key.cast()
    }
}

/// Original static `d1eq` (`hash.c:344`).
pub unsafe extern "C" fn d1eq(key1: *mut c_void, key2: *mut c_void) -> i32 {
    unsafe { (*key1.cast::<f64>() == *key2.cast::<f64>()) as i32 }
}

/// Original static `d2hash` (`hash.c:354`).
pub unsafe extern "C" fn d2hash(key: *mut c_void) -> u32 {
    unsafe {
        let value = key.cast::<u32>();
        (*value)
            .wrapping_add(*value.add(1))
            .wrapping_add((*value.add(2)).wrapping_mul(3))
            .wrapping_add((*value.add(3)).wrapping_mul(7))
    }
}

/// Original static `d2cp` (`hash.c:370`).
pub unsafe extern "C" fn d2cp(key: *mut c_void) -> *mut c_void {
    unsafe {
        let new_key = libc::malloc(2 * core::mem::size_of::<f64>()).cast::<f64>();
        *new_key = *key.cast::<f64>();
        *new_key.add(1) = *key.cast::<f64>().add(1);
        new_key.cast()
    }
}

/// Original static `d2eq` (`hash.c:381`).
pub unsafe extern "C" fn d2eq(key1: *mut c_void, key2: *mut c_void) -> i32 {
    unsafe {
        (*key1.cast::<f64>() == *key2.cast::<f64>()
            && *key1.cast::<f64>().add(1) == *key2.cast::<f64>().add(1)) as i32
    }
}

/// Original static `i1hash` (`hash.c:391`).
pub unsafe extern "C" fn i1hash(key: *mut c_void) -> u32 {
    unsafe { *key.cast::<u32>() }
}

/// Original static `i1cp` (`hash.c:396`).
pub unsafe extern "C" fn i1cp(key: *mut c_void) -> *mut c_void {
    unsafe {
        let new_key = libc::malloc(core::mem::size_of::<i32>()).cast::<i32>();
        *new_key = *key.cast::<i32>();
        new_key.cast()
    }
}

/// Original static `i1eq` (`hash.c:405`).
pub unsafe extern "C" fn i1eq(key1: *mut c_void, key2: *mut c_void) -> i32 {
    unsafe { (*key1.cast::<i32>() == *key2.cast::<i32>()) as i32 }
}

/// Original static `i2hash` (`hash.c:415`).
pub unsafe extern "C" fn i2hash(key: *mut c_void) -> u32 {
    unsafe {
        let value = key.cast::<u32>();
        (*value).wrapping_add((*value.add(1)).wrapping_shl(16))
    }
}

/// Original static `i2cp` (`hash.c:426`).
pub unsafe extern "C" fn i2cp(key: *mut c_void) -> *mut c_void {
    unsafe {
        let new_key = libc::malloc(2 * core::mem::size_of::<i32>()).cast::<i32>();
        *new_key = *key.cast::<i32>();
        *new_key.add(1) = *key.cast::<i32>().add(1);
        new_key.cast()
    }
}

/// Original static `i2eq` (`hash.c:437`).
pub unsafe extern "C" fn i2eq(key1: *mut c_void, key2: *mut c_void) -> i32 {
    unsafe {
        (*key1.cast::<i32>() == *key2.cast::<i32>()
            && *key1.cast::<i32>().add(1) == *key2.cast::<i32>().add(1)) as i32
    }
}

/// Original `ht_create_d1` (`hash.c:446`).
pub unsafe fn ht_create_d1(size: i32) -> *mut Hashtable {
    assert_eq!(core::mem::size_of::<f64>(), 2 * core::mem::size_of::<i32>());
    unsafe { ht_create(size, d1cp, d1eq, d1hash) }
}

/// Original `ht_create_d2` (`hash.c:452`).
pub unsafe fn ht_create_d2(size: i32) -> *mut Hashtable {
    assert_eq!(core::mem::size_of::<f64>(), 2 * core::mem::size_of::<i32>());
    unsafe { ht_create(size, d2cp, d2eq, d2hash) }
}

/// Original `ht_create_str` (`hash.c:458`).
pub unsafe fn ht_create_str(size: i32) -> *mut Hashtable {
    unsafe { ht_create(size, strcp, streq, strhash) }
}

/// Original `ht_create_i1` (`hash.c:463`).
pub unsafe fn ht_create_i1(size: i32) -> *mut Hashtable {
    unsafe { ht_create(size, i1cp, i1eq, i1hash) }
}

/// Original `ht_create_i2` (`hash.c:468`).
pub unsafe fn ht_create_i2(size: i32) -> *mut Hashtable {
    assert_eq!(core::mem::size_of::<i32>(), 4);
    unsafe { ht_create(size, i2cp, i2eq, i2hash) }
}

/// Original `ht_getnentries` (`hash.c:474`).
pub unsafe fn ht_getnentries(table: *mut Hashtable) -> i32 {
    unsafe { (*table).n }
}

/// Original `ht_getsize` (`hash.c:479`).
pub unsafe fn ht_getsize(table: *mut Hashtable) -> i32 {
    unsafe { (*table).size }
}

/// Original `ht_getnfilled` (`hash.c:484`).
pub unsafe fn ht_getnfilled(table: *mut Hashtable) -> i32 {
    unsafe { (*table).nhash }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn increment_data(data: *mut c_void) {
        unsafe { *data.cast::<i32>() += 1 }
    }

    #[test]
    fn integer_pair_table_preserves_source_collision_replace_delete_and_process_behavior() {
        unsafe {
            let table = ht_create_i2(1);
            let mut first_key = [4_i32, 5];
            let mut second_key = [6_i32, 7];
            let mut first_data = 20_i32;
            let mut replacement_data = 30_i32;
            let mut second_data = 40_i32;
            assert!(
                ht_insert(
                    table,
                    first_key.as_mut_ptr().cast(),
                    (&mut first_data as *mut i32).cast()
                )
                .is_null()
            );
            assert!(
                ht_insert(
                    table,
                    second_key.as_mut_ptr().cast(),
                    (&mut second_data as *mut i32).cast()
                )
                .is_null()
            );
            assert_eq!(ht_getnentries(table), 2);
            assert_eq!(ht_getnfilled(table), 1);
            assert_eq!(
                ht_find(table, first_key.as_mut_ptr().cast()),
                (&mut first_data as *mut i32).cast()
            );
            assert_eq!(
                ht_insert(
                    table,
                    first_key.as_mut_ptr().cast(),
                    (&mut replacement_data as *mut i32).cast()
                ),
                (&mut first_data as *mut i32).cast()
            );
            assert_eq!(ht_getnentries(table), 2);
            ht_process(table, increment_data);
            assert_eq!((replacement_data, second_data), (31, 41));
            assert_eq!(
                ht_delete(table, first_key.as_mut_ptr().cast()),
                (&mut replacement_data as *mut i32).cast()
            );
            assert_eq!(ht_getnentries(table), 1);
            assert!(ht_delete(table, first_key.as_mut_ptr().cast()).is_null());
            ht_destroy(table);
        }
    }

    #[test]
    fn string_and_double_key_constructors_copy_and_find_source_keys() {
        unsafe {
            let string_table = ht_create_str(5);
            let key = c"source-key";
            let mut value = 7_i32;
            ht_insert(
                string_table,
                key.as_ptr().cast_mut().cast(),
                (&mut value as *mut i32).cast(),
            );
            assert_eq!(
                ht_find(string_table, c"source-key".as_ptr().cast_mut().cast()),
                (&mut value as *mut i32).cast()
            );
            ht_destroy(string_table);
            let double_table = ht_create_d2(5);
            let mut point = [1.5_f64, -3.25];
            let mut value = 9_i32;
            ht_insert(
                double_table,
                point.as_mut_ptr().cast(),
                (&mut value as *mut i32).cast(),
            );
            assert_eq!(
                ht_find(double_table, point.as_mut_ptr().cast()),
                (&mut value as *mut i32).cast()
            );
            ht_destroy(double_table);
        }
    }
}
