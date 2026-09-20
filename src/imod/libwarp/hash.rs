//! Translation of `IMOD/libwarp/hash.c` and `hash.h`.
//!
//! # Modelling the chained buckets
//!
//! The source is the classic C hash table: `ht_bucket** table` is an array of
//! `size` list heads, each bucket carries a `next` pointer, and `ht_insert`
//! pushes a new bucket onto the *head* of its list (`hash.c:170-180`, "we'll
//! add it to the head of the list at this spot"). That order is observable —
//! `ht_process` walks each list from the head, and the natural-neighbour code
//! downstream frees and accumulates through it — so it is preserved exactly
//! here: `table` is a `Vec` of `size` bucket lists, and a new bucket goes in at
//! index 0, which is where the `next`-chain would have put it. Walking a
//! `Vec<HtBucket>` front to back is walking the C list from its head.
//!
//! The C key is a `void*` copied by a `ht_keycp` callback and freed by
//! `ht_destroy`; the C data is a `void*` the table never owns. Both become type
//! parameters: `K` is the key, owned by the table and duplicated by `cp`
//! exactly as `strdup`/`malloc` duplicate it in the source, and `T` is the
//! data, still owned by the caller. Each of the source's five key flavours
//! (`char*`, `double`, `double[2]`, `int`, `int[2]`) instantiates `K`
//! concretely, so `strcp`/`d1cp`/`d2cp`/`i1cp`/`i2cp` and their `eq`/`hash`
//! partners translate one-for-one with no `void*` left.

/// C `INT_PER_DOUBLE` (`hash.c:33`).
const INT_PER_DOUBLE: usize = 2;
/// C `BYTE_PER_INT` (`hash.c:34`).
const BYTE_PER_INT: usize = 4;

/// C `ht_keycp` (`hash.h:29`).
pub type HtKeycp<K> = fn(&K) -> K;
/// C `ht_keyeq` (`hash.h:33`).
pub type HtKeyeq<K> = fn(&K, &K) -> i32;
/// C `ht_key2hash` (`hash.h:37`).
pub type HtKey2hash<K> = fn(&K) -> u32;

/// C `struct ht_bucket` (`hash.c:38`).
///
/// The source's `struct ht_bucket* next` is gone: the chain is the position of
/// the bucket in its list, see the module comment.
pub struct HtBucket<K, T> {
    pub key: K,
    pub data: T,
    /// `int id` — "unique id -- just in case" (`hash.c:41`); nothing reads it.
    pub id: i32,
}

/// C `struct hashtable` (`hash.c:48`).
pub struct Hashtable<K, T> {
    /// table size
    pub size: i32,
    /// current number of entries
    pub n: i32,
    /// number of inserted entries
    pub naccum: i32,
    /// number of used table elements
    pub nhash: i32,
    pub cp: HtKeycp<K>,
    pub eq: HtKeyeq<K>,
    pub hash: HtKey2hash<K>,
    pub table: Vec<Vec<HtBucket<K, T>>>,
}

/// Original `ht_create` (`hash.c:61`).
///
/// The source mallocs the table first and frees it again when `size <= 0`; the
/// `NULL` return becomes `None`, and the field assignments the source makes
/// after the allocation are folded into the one construction here.
pub fn ht_create<K, T>(
    size: i32,
    cp: HtKeycp<K>,
    eq: HtKeyeq<K>,
    hash: HtKey2hash<K>,
) -> Option<Hashtable<K, T>> {
    if size <= 0 {
        return None;
    }

    let mut table = Hashtable {
        size,
        n: 0,
        naccum: 0,
        nhash: 0,
        cp,
        eq,
        hash,
        table: Vec::with_capacity(size as usize),
    };

    for _ in 0..size {
        table.table.push(Vec::new());
    }

    Some(table)
}

/// Original `ht_destroy` (`hash.c:102`).
///
/// (Take care of deallocating data by `ht_process()` prior to destroying the
/// table if necessary.)
pub fn ht_destroy<K, T>(table: Option<Hashtable<K, T>>) {
    if table.is_none() {
        return;
    }

    // The source's three `free` loops — every bucket's key, every bucket, then
    // the head array and the table itself. Dropping the box does all of it.
    drop(table);
}

/// Original `ht_insert` (`hash.c:133`).
///
/// Returns the old data associated with the key, `None` if the key wasn't in
/// the table previously.
pub fn ht_insert<K, T>(table: &mut Hashtable<K, T>, key: &K, data: T) -> Option<T> {
    let val = ((table.hash)(key) % table.size as u32) as usize;

    /*
     * NULL means this bucket hasn't been used yet.  We'll simply allocate
     * space for our new bucket and put our data there, with the table
     * pointing at it.
     */
    if table.table[val].is_empty() {
        let bucket = HtBucket {
            key: (table.cp)(key),
            data,
            id: table.naccum,
        };

        table.table[val].push(bucket);
        table.n += 1;
        table.naccum += 1;
        table.nhash += 1;

        return None;
    }

    /*
     * This spot in the table is already in use.  See if the current string
     * has already been inserted, and if so, return corresponding data.
     */
    let mut i = 0;
    while i < table.table[val].len() {
        if (table.eq)(key, &table.table[val][i].key) == 1 {
            let old_data = core::mem::replace(&mut table.table[val][i].data, data);

            table.table[val][i].id = table.naccum;
            table.naccum += 1;

            return Some(old_data);
        }
        i += 1;
    }

    /*
     * This key must not be in the table yet.  We'll add it to the head of
     * the list at this spot in the hash table.
     */
    let bucket = HtBucket {
        key: (table.cp)(key),
        data,
        id: table.naccum,
    };

    table.table[val].insert(0, bucket);
    table.n += 1;
    table.naccum += 1;

    None
}

/// Original `ht_find` (`hash.c:204`).
///
/// Returns the data associated with a key, `None` if the key has not been
/// inserted in the table.
pub fn ht_find<K, T: Copy>(table: &Hashtable<K, T>, key: &K) -> Option<T> {
    let val = ((table.hash)(key) % table.size as u32) as usize;

    if table.table[val].is_empty() {
        return None;
    }

    let mut i = 0;
    while i < table.table[val].len() {
        if (table.eq)(key, &table.table[val][i].key) == 1 {
            return Some(table.table[val][i].data);
        }
        i += 1;
    }

    None
}

/// Original `ht_delete` (`hash.c:227`).
///
/// Returns the data that was associated with the key so that the calling code
/// can dispose it properly.
pub fn ht_delete<K, T>(table: &mut Hashtable<K, T>, key: &K) -> Option<T> {
    let val = ((table.hash)(key) % table.size as u32) as usize;

    if table.table[val].is_empty() {
        return None;
    }

    /*
     * Traverse the list, keeping track of the previous node in the list.
     * When we find the node to delete, we set the previous node's next
     * pointer to point to the node after ourself instead.
     */
    let mut i = 0;
    while i < table.table[val].len() {
        if (table.eq)(key, &table.table[val][i].key) == 1 {
            let bucket = table.table[val].remove(i);
            let data = bucket.data;
            if i != 0 {
                // `prev->next = bucket->next`: removing from the middle of the
                // list, the head array is untouched.
            } else {
                /*
                 * If 'prev' still equals NULL, it means that we need to
                 * delete the first node in the list.
                 *
                 * `hash.c:250` decrements nhash for every head deletion, even
                 * when the list still has further buckets; that is the
                 * source's behaviour and is kept.
                 */
                table.nhash -= 1;
            }
            table.n -= 1;

            return Some(data);
        }
        i += 1;
    }

    /*
     * If we get here, it means we didn't find the item in the table.
     */
    None
}

/// Original `ht_process` (`hash.c:281`).
pub fn ht_process<K, T: Copy>(table: &Hashtable<K, T>, func: fn(T)) {
    for i in 0..table.size {
        if !table.table[i as usize].is_empty() {
            let mut j = 0;
            while j < table.table[i as usize].len() {
                func(table.table[i as usize][j].data);
                j += 1;
            }
        }
    }
}

/*
 * functions for for string keys
 */

/// Original static `strhash` (`hash.c:298`).
///
/// `hashvalue ^= *(unsigned int*) str` reads four bytes at every position of
/// the string, so for the last three characters it reads past the `strdup`
/// allocation — an over-read that CLAUDE.md's "source-level UB" rule says to
/// document rather than reproduce. The bytes at and past the terminating NUL
/// are taken as zero here. The word is assembled little-endian, which is what
/// the cast reads on every platform this tree is built for.
pub fn strhash(key: &String) -> u32 {
    let str = key.as_bytes();
    let mut hashvalue: u32 = 0;

    let mut i = 0;
    while i < str.len() && str[i] != 0 {
        let mut word: u32 = 0;
        for j in 0..4 {
            if i + j < str.len() {
                word |= (str[i + j] as u32) << (8 * j);
            }
        }
        hashvalue ^= word;
        hashvalue <<= 1;
        i += 1;
    }

    hashvalue
}

/// Original static `strcp` (`hash.c:311`).
pub fn strcp(key: &String) -> String {
    key.clone()
}

/// Original static `streq` (`hash.c:316`).
pub fn streq(key1: &String, key2: &String) -> i32 {
    (key1 == key2) as i32
}

/* functions for for double keys */

/// Original static `d1hash` (`hash.c:324`).
///
/// `unsigned int* v = key; return v[0] + v[1]` reinterprets the double's eight
/// bytes as two words; the sum is order-independent, so the split here matches
/// on either endianness.
pub fn d1hash(key: &f64) -> u32 {
    let v = key.to_bits();

    (v as u32).wrapping_add((v >> 32) as u32)
}

/// Original static `d1cp` (`hash.c:335`).
pub fn d1cp(key: &f64) -> f64 {
    *key
}

/// Original static `d1eq` (`hash.c:344`).
pub fn d1eq(key1: &f64, key2: &f64) -> i32 {
    (*key1 == *key2) as i32
}

/*
 * functions for for double[2] keys
 */

/// Original static `d2hash` (`hash.c:354`).
///
/// `v[0] + v[1] + v[2] * 3 + v[3] * 7` over the sixteen bytes of the two
/// doubles read as four words; the multipliers make the word order matter, so
/// the little-endian split is spelled out.
pub fn d2hash(key: &[f64; 2]) -> u32 {
    let b0 = key[0].to_bits();
    let b1 = key[1].to_bits();
    let v = [b0 as u32, (b0 >> 32) as u32, b1 as u32, (b1 >> 32) as u32];

    /*
     * PS: here multiplications suppose to make (a,b) and (b,a) generate
     * different hash values
     */
    v[0].wrapping_add(v[1])
        .wrapping_add(v[2].wrapping_mul(3))
        .wrapping_add(v[3].wrapping_mul(7))
}

/// Original static `d2cp` (`hash.c:370`).
pub fn d2cp(key: &[f64; 2]) -> [f64; 2] {
    [key[0], key[1]]
}

/// Original static `d2eq` (`hash.c:381`).
pub fn d2eq(key1: &[f64; 2], key2: &[f64; 2]) -> i32 {
    ((key1[0] == key2[0]) && (key1[1] == key2[1])) as i32
}

/*
 * functions for for int[1] keys
 */

/// Original static `i1hash` (`hash.c:391`).
pub fn i1hash(key: &i32) -> u32 {
    *key as u32
}

/// Original static `i1cp` (`hash.c:396`).
pub fn i1cp(key: &i32) -> i32 {
    *key
}

/// Original static `i1eq` (`hash.c:405`).
pub fn i1eq(key1: &i32, key2: &i32) -> i32 {
    (*key1 == *key2) as i32
}

/*
 * functions for for int[2] keys
 */

/// Original static `i2hash` (`hash.c:415`).
pub fn i2hash(key: &[i32; 2]) -> u32 {
    let v = [key[0] as u32, key[1] as u32];

    v[0].wrapping_add(v[1] << 16)
}

/// Original static `i2cp` (`hash.c:426`).
pub fn i2cp(key: &[i32; 2]) -> [i32; 2] {
    [key[0], key[1]]
}

/// Original static `i2eq` (`hash.c:437`).
pub fn i2eq(key1: &[i32; 2], key2: &[i32; 2]) -> i32 {
    ((key1[0] == key2[0]) && (key1[1] == key2[1])) as i32
}

/// Original `ht_create_d1` (`hash.c:446`).
pub fn ht_create_d1<T>(size: i32) -> Option<Hashtable<f64, T>> {
    assert_eq!(
        core::mem::size_of::<f64>(),
        INT_PER_DOUBLE * core::mem::size_of::<i32>()
    );
    ht_create(size, d1cp, d1eq, d1hash)
}

/// Original `ht_create_d2` (`hash.c:452`).
pub fn ht_create_d2<T>(size: i32) -> Option<Hashtable<[f64; 2], T>> {
    assert_eq!(
        core::mem::size_of::<f64>(),
        INT_PER_DOUBLE * core::mem::size_of::<i32>()
    );
    ht_create(size, d2cp, d2eq, d2hash)
}

/// Original `ht_create_str` (`hash.c:458`).
pub fn ht_create_str<T>(size: i32) -> Option<Hashtable<String, T>> {
    ht_create(size, strcp, streq, strhash)
}

/// Original `ht_create_i1` (`hash.c:463`).
pub fn ht_create_i1<T>(size: i32) -> Option<Hashtable<i32, T>> {
    ht_create(size, i1cp, i1eq, i1hash)
}

/// Original `ht_create_i2` (`hash.c:468`).
pub fn ht_create_i2<T>(size: i32) -> Option<Hashtable<[i32; 2], T>> {
    assert_eq!(core::mem::size_of::<i32>(), BYTE_PER_INT);
    ht_create(size, i2cp, i2eq, i2hash)
}

/// Original `ht_getnentries` (`hash.c:474`).
pub fn ht_getnentries<K, T>(table: &Hashtable<K, T>) -> i32 {
    table.n
}

/// Original `ht_getsize` (`hash.c:479`).
pub fn ht_getsize<K, T>(table: &Hashtable<K, T>) -> i32 {
    table.size
}

/// Original `ht_getnfilled` (`hash.c:484`).
pub fn ht_getnfilled<K, T>(table: &Hashtable<K, T>) -> i32 {
    table.nhash
}

#[cfg(test)]
mod tests {
    use super::*;

    static PROCESSED: std::sync::Mutex<Vec<i32>> = std::sync::Mutex::new(Vec::new());

    fn record_data(data: i32) {
        PROCESSED.lock().unwrap().push(data);
    }

    #[test]
    fn integer_pair_table_preserves_source_collision_replace_delete_and_process_behavior() {
        let mut table = ht_create_i2::<i32>(1).unwrap();
        let first_key = [4_i32, 5];
        let second_key = [6_i32, 7];
        assert_eq!(ht_insert(&mut table, &first_key, 20), None);
        assert_eq!(ht_insert(&mut table, &second_key, 40), None);
        assert_eq!(ht_getnentries(&table), 2);
        assert_eq!(ht_getsize(&table), 1);
        assert_eq!(ht_getnfilled(&table), 1);
        assert_eq!(ht_find(&table, &first_key), Some(20));
        assert_eq!(ht_insert(&mut table, &first_key, 30), Some(20));
        assert_eq!(ht_getnentries(&table), 2);
        // `hash.c:174` inserts a colliding key at the *head* of the list, so
        // the second key is visited first.
        PROCESSED.lock().unwrap().clear();
        ht_process(&table, record_data);
        assert_eq!(*PROCESSED.lock().unwrap(), vec![40, 30]);
        assert_eq!(ht_delete(&mut table, &first_key), Some(30));
        assert_eq!(ht_getnentries(&table), 1);
        // Deleting a non-head bucket leaves nhash alone.
        assert_eq!(ht_getnfilled(&table), 1);
        assert_eq!(ht_delete(&mut table, &first_key), None);
        ht_destroy(Some(table));
        assert!(ht_create_i2::<i32>(0).is_none());
    }

    #[test]
    fn string_and_double_key_constructors_copy_and_find_source_keys() {
        let mut string_table = ht_create_str::<i32>(5).unwrap();
        ht_insert(&mut string_table, &"source-key".to_string(), 7);
        assert_eq!(ht_find(&string_table, &"source-key".to_string()), Some(7));
        assert_eq!(ht_find(&string_table, &"other-key".to_string()), None);
        ht_destroy(Some(string_table));

        let mut double_table = ht_create_d2::<i32>(5).unwrap();
        let point = [1.5_f64, -3.25];
        ht_insert(&mut double_table, &point, 9);
        assert_eq!(ht_find(&double_table, &point), Some(9));
        ht_destroy(Some(double_table));

        let mut single_table = ht_create_d1::<i32>(3).unwrap();
        ht_insert(&mut single_table, &2.5_f64, 11);
        assert_eq!(ht_find(&single_table, &2.5_f64), Some(11));
        ht_destroy(Some(single_table));

        let mut int_table = ht_create_i1::<i32>(3).unwrap();
        ht_insert(&mut int_table, &-4_i32, 13);
        assert_eq!(ht_find(&int_table, &-4_i32), Some(13));
        ht_destroy(Some(int_table));
    }
}
