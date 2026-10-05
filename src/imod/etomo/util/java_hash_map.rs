//! Rust-only JDK stand-in (like `etomo/jdk.rs`): `java.util.HashMap` and
//! `java.util.concurrent.ConcurrentHashMap` with Java's iteration order, and
//! `System.getenv()`'s order.  **Not a translated eTomo unit.**
//!
//! eTomo writes some maps out in hash iteration order (the options
//! `CopyTomoComs.genOptions` feeds to copytomocoms, `DirectiveFileCollection`'s
//! copyarg maps, the machine list `ProcesschunksParam.reorderComputerMapGpuFirst`
//! rebuilds, the 3dmods `BaseImodManager` quits, the environment dump at start-up),
//! so a Rust `HashMap` (random order) or `BTreeMap` (key order) changes what reaches
//! a child process or a log.  The order of a Java hash map is not random: it is a
//! function of the keys' `hashCode()`, the table length and, within one bucket, the
//! history of puts and removes.  This module reproduces it for the reference JVM
//! (OpenJDK 19, `/usr/lib/jvm/java-19-openjdk-amd64`), checked against it by
//! `tests::java_differential` (see there).
//!
//! What a key's hash is: [`JavaHashCode`].  `String.hashCode()` and
//! `Integer.hashCode()` are fixed by the Java specification, so maps keyed by them
//! iterate reproducibly.  A key whose class does not override `hashCode` (an enum
//! constant, `Thread`, any eTomo object compared by identity) hashes by the JVM's
//! identity hash, which differs from run to run of the JVM itself; such a map's order
//! is a JVM boundary and is not reproduced (each such site says so in place).
//!
//! `java.util.HashMap` (OpenJDK 8+, [`JavaHashMap`]):
//! - the key's hash is spread by `h ^ (h >>> 16)` (`HashMap.hash`);
//! - the table is allocated on the first `put`: 16 buckets for `new HashMap()`,
//!   `tableSizeFor(n)` for `new HashMap(n)`; it doubles when `++size > threshold`
//!   (load factor 0.75), and also when a bucket reaches `TREEIFY_THRESHOLD` while the
//!   table is smaller than `MIN_TREEIFY_CAPACITY` (`treeifyBin`'s resize); `clear`
//!   and `remove` never shrink it;
//! - iteration visits the buckets in index order, and within a bucket the entries in
//!   the order they were added (`putVal` appends to the bin, a `put` of an existing
//!   key replaces the value in place, `resize` splits a bin keeping relative order).
//!   Entries are kept in insertion order here; iteration sorts them stably by bucket
//!   index under the current table length, which is the same order.  A bucket that
//!   Java turns into a red-black tree (9 or more entries with 64 or more buckets) is
//!   iterated in insertion order here; eTomo's maps hold a few dozen keys, so that
//!   case is not reached.
//!
//! `java.util.concurrent.ConcurrentHashMap` ([`JavaConcurrentHashMap`]) appends to a
//! bin like `HashMap`, but its `transfer` (resize) moves the run of nodes that ends a
//! bin as a block and *prepends* the nodes before it one by one, so a resize can
//! reverse part of a bin.  The bins are therefore kept explicitly.  A bin Java turns
//! into a `TreeBin` (8 or more colliding keys with 64 or more bins) takes later puts
//! at its head; that is not reproduced (eTomo's one such map holds a few lock
//! numbers).
//!
//! `java.util.Properties` is backed by a `ConcurrentHashMap` since JDK 9, and since
//! JDK 18 `Properties.store` writes its entries sorted by key (JDK-8231640), so the
//! data files eTomo writes (`.edf`, `.ejf`, `.epp`, `.ebt`, `.etomo`) are key-sorted
//! on the reference JVM, which is what `utilities::java_util_properties_store` does.
//! On a JDK before 18 they come out in hash order.

/// `HashMap.TREEIFY_THRESHOLD` (also `ConcurrentHashMap`'s).
const TREEIFY_THRESHOLD: usize = 8;
/// `HashMap.MIN_TREEIFY_CAPACITY` (also `ConcurrentHashMap`'s).
const MIN_TREEIFY_CAPACITY: usize = 64;
/// `HashMap.DEFAULT_INITIAL_CAPACITY`, `ConcurrentHashMap.DEFAULT_CAPACITY`.
const DEFAULT_INITIAL_CAPACITY: usize = 16;
/// `HashMap.MAXIMUM_CAPACITY`.
const MAXIMUM_CAPACITY: usize = 1 << 30;

/// The value of a Java key's `hashCode()`.
pub trait JavaHashCode {
    /// Java `hashCode()`.
    fn java_hash_code(&self) -> i32;
}

/// Java `String.hashCode()`: `s[0]*31^(n-1) + ... + s[n-1]` over UTF-16 code units.
pub fn string_hash_code(key: &str) -> i32 {
    let mut h: i32 = 0;
    for unit in key.encode_utf16() {
        h = h.wrapping_mul(31).wrapping_add(unit as i32);
    }
    h
}

impl JavaHashCode for str {
    fn java_hash_code(&self) -> i32 {
        string_hash_code(self)
    }
}

impl JavaHashCode for String {
    fn java_hash_code(&self) -> i32 {
        string_hash_code(self)
    }
}

/// Java `Integer.hashCode()`: the value.
impl JavaHashCode for i32 {
    fn java_hash_code(&self) -> i32 {
        *self
    }
}

/// Java `HashMap.tableSizeFor(cap)`: the smallest power of two `>= cap` (1 for 0).
fn table_size_for(cap: usize) -> usize {
    if cap <= 1 {
        1
    } else {
        cap.next_power_of_two().min(MAXIMUM_CAPACITY)
    }
}

/// Java `HashMap.hash(Object)` for a non-null key.
fn hash_map_spread(h: i32) -> u32 {
    let h = h as u32;
    h ^ (h >> 16)
}

/// `java.util.HashMap<K, V>` with Java's iteration order.
#[derive(Clone, Debug)]
pub struct JavaHashMap<K, V> {
    /// Live entries in insertion order.
    entries: Vec<(K, V)>,
    /// Table length; 0 before the first `put` (Java's null table).
    capacity: usize,
    /// Java `threshold`: before the table exists, the initial table length (0 for
    /// the default 16); after, `capacity * 0.75`.
    threshold: usize,
}

impl<K, V> Default for JavaHashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> JavaHashMap<K, V> {
    /// Java `new HashMap<K, V>()`.
    pub fn new() -> Self {
        JavaHashMap {
            entries: Vec::new(),
            capacity: 0,
            threshold: 0,
        }
    }

    /// Java `new HashMap<K, V>(initialCapacity)`.
    pub fn with_capacity(initial_capacity: usize) -> Self {
        JavaHashMap {
            entries: Vec::new(),
            capacity: 0,
            threshold: table_size_for(initial_capacity),
        }
    }

    /// Java `size()`.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Java `clear()`.  The table keeps its length.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Java `resize()`.
    fn resize(&mut self) {
        self.capacity = if self.capacity > 0 {
            (self.capacity * 2).min(MAXIMUM_CAPACITY)
        } else if self.threshold > 0 {
            self.threshold
        } else {
            DEFAULT_INITIAL_CAPACITY
        };
        // `(int)(newCap * loadFactor)`, which is also `oldThr << 1` once the table
        // has 16 or more buckets.
        self.threshold = self.capacity * 3 / 4;
    }
}

impl<K: Eq + JavaHashCode, V> JavaHashMap<K, V> {
    fn bucket<Q: JavaHashCode + ?Sized>(&self, key: &Q) -> usize {
        (hash_map_spread(key.java_hash_code()) as usize) & (self.capacity - 1)
    }

    fn position<Q>(&self, key: &Q) -> Option<usize>
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.entries.iter().position(|(k, _)| k.borrow() == key)
    }

    /// Java `put(key, value)`; returns the previous value.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.capacity == 0 {
            self.resize();
        }
        if let Some(index) = self.position(&key) {
            return Some(std::mem::replace(&mut self.entries[index].1, value));
        }
        // `binCount >= TREEIFY_THRESHOLD - 1`: the bin already held 8 nodes.
        let bucket = self.bucket(&key);
        let bin_length = self
            .entries
            .iter()
            .filter(|(k, _)| self.bucket(k) == bucket)
            .count();
        self.entries.push((key, value));
        if bin_length >= TREEIFY_THRESHOLD && self.capacity < MIN_TREEIFY_CAPACITY {
            self.resize();
        }
        if self.entries.len() > self.threshold {
            self.resize();
        }
        None
    }

    /// Java `putAll(m)` (`putMapEntries`), `m`'s entries given in `m`'s iteration
    /// order.  Presizes the table as OpenJDK 19 does (`Math.ceil(s / loadFactor)`,
    /// JDK-8281631) and then puts each entry.
    pub fn put_all<I: IntoIterator<Item = (K, V)>>(&mut self, entries: I) {
        let entries: Vec<(K, V)> = entries.into_iter().collect();
        let s = entries.len();
        if s > 0 {
            if self.capacity == 0 {
                // `double dt = Math.ceil(s / (double)loadFactor)`.
                let t = (s as f64 / 0.75f32 as f64).ceil() as usize;
                let t = t.min(MAXIMUM_CAPACITY);
                if t > self.threshold {
                    self.threshold = table_size_for(t);
                }
            } else {
                while s > self.threshold && self.capacity < MAXIMUM_CAPACITY {
                    self.resize();
                }
            }
            for (key, value) in entries {
                self.insert(key, value);
            }
        }
    }

    /// Java `new HashMap<K, V>(m)`, `m`'s entries given in `m`'s iteration order.
    pub fn from_map<I: IntoIterator<Item = (K, V)>>(entries: I) -> Self {
        let mut map = Self::new();
        map.put_all(entries);
        map
    }

    /// Java `get(key)`.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.position(key).map(|index| &self.entries[index].1)
    }

    /// Java `get(key)` for in-place changes of the value.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.position(key).map(|index| &mut self.entries[index].1)
    }

    /// Java `containsKey(key)`.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.position(key).is_some()
    }

    /// Java `remove(key)`; returns the removed value.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let index = self.position(key)?;
        Some(self.entries.remove(index).1)
    }

    /// The entry indices in Java's iteration order.
    fn order(&self) -> Vec<usize> {
        let mut order: Vec<(usize, usize)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(index, (key, _))| (self.bucket(key), index))
            .collect();
        // Stable: entries of one bucket stay in insertion order.
        order.sort_by_key(|&(bucket, _)| bucket);
        order.into_iter().map(|(_, index)| index).collect()
    }

    /// Java `entrySet().iterator()`, in Java's order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.order().into_iter().map(move |index| {
            let (key, value) = &self.entries[index];
            (key, value)
        })
    }

    /// Java `keySet().iterator()`, in Java's order.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(key, _)| key)
    }

    /// Java `values().iterator()`, in Java's order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, value)| value)
    }

    /// Java `AbstractMap.toString()`: `{k1=v1, k2=v2}` in iteration order, each key
    /// and value rendered by the caller's `String.valueOf`.
    pub fn to_java_string(
        &self,
        key_string: impl Fn(&K) -> String,
        value_string: impl Fn(&V) -> String,
    ) -> String {
        let mut builder = String::from("{");
        for (i, (key, value)) in self.iter().enumerate() {
            if i > 0 {
                builder.push_str(", ");
            }
            builder.push_str(&key_string(key));
            builder.push('=');
            builder.push_str(&value_string(value));
        }
        builder.push('}');
        builder
    }
}

/// Java `ConcurrentHashMap.HASH_BITS`.
const HASH_BITS: u32 = 0x7fff_ffff;

/// Java `ConcurrentHashMap.spread(h)`.
fn concurrent_spread(h: i32) -> u32 {
    let h = h as u32;
    (h ^ (h >> 16)) & HASH_BITS
}

/// `java.util.concurrent.ConcurrentHashMap<K, V>` with Java's iteration order, for
/// one thread at a time (the callers hold a lock around it, so no resize is ever
/// observed half done).
#[derive(Clone, Debug)]
pub struct JavaConcurrentHashMap<K, V> {
    /// Java `table`: each bin's nodes, head first; empty before `initTable`.
    table: Vec<Vec<(K, V)>>,
    /// Java `sizeCtl` once the table exists (`0.75 * n`).
    size_ctl: usize,
    /// Java `size()`.
    size: usize,
}

impl<K, V> Default for JavaConcurrentHashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> JavaConcurrentHashMap<K, V> {
    /// Java `new ConcurrentHashMap<K, V>()`.
    pub fn new() -> Self {
        JavaConcurrentHashMap {
            table: Vec::new(),
            size_ctl: 0,
            size: 0,
        }
    }

    /// Java `size()`.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl<K: Eq + JavaHashCode, V> JavaConcurrentHashMap<K, V> {
    fn index(hash: u32, n: usize) -> usize {
        (hash as usize) & (n - 1)
    }

    /// Java `initTable()` (default capacity).
    fn init_table(&mut self) {
        let n = DEFAULT_INITIAL_CAPACITY;
        self.table = (0..n).map(|_| Vec::new()).collect();
        self.size_ctl = n - (n >> 2);
    }

    /// Java `transfer(tab, null)`: doubles the table.  Each bin's trailing run of
    /// nodes bound for the same half moves as a block; the nodes before it are
    /// prepended one by one to their new bin.
    fn transfer(&mut self) {
        let n = self.table.len();
        let old = std::mem::take(&mut self.table);
        let mut next: Vec<Vec<(K, V)>> = (0..2 * n).map(|_| Vec::new()).collect();
        for (i, bin) in old.into_iter().enumerate() {
            if bin.is_empty() {
                continue;
            }
            let bits: Vec<bool> = bin
                .iter()
                .map(|(key, _)| (concurrent_spread(key.java_hash_code()) as usize) & n != 0)
                .collect();
            // `lastRun`: the start of the trailing run with one `runBit`.
            let mut last_run = 0;
            for p in 1..bin.len() {
                if bits[p] != bits[p - 1] {
                    last_run = p;
                }
            }
            let run_bit = bits[last_run];
            let mut nodes: Vec<Option<(K, V)>> = bin.into_iter().map(Some).collect();
            let mut ln: Vec<(K, V)> = Vec::new();
            let mut hn: Vec<(K, V)> = Vec::new();
            {
                let run: Vec<(K, V)> = nodes[last_run..]
                    .iter_mut()
                    .map(|node| node.take().unwrap())
                    .collect();
                if run_bit {
                    hn = run;
                } else {
                    ln = run;
                }
            }
            for p in 0..last_run {
                let node = nodes[p].take().unwrap();
                if bits[p] {
                    hn.insert(0, node);
                } else {
                    ln.insert(0, node);
                }
            }
            next[i] = ln;
            next[i + n] = hn;
        }
        self.table = next;
        // `sizeCtl = (n << 1) - (n >>> 1)`.
        self.size_ctl = (n << 1) - (n >> 1);
    }

    /// Java `put(key, value)` (`putVal(key, value, false)`); returns the previous
    /// value.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = concurrent_spread(key.java_hash_code());
        if self.table.is_empty() {
            self.init_table();
        }
        let n = self.table.len();
        let i = Self::index(hash, n);
        let bin = &mut self.table[i];
        // `binCount`: the position (from 1) of the node replaced, or the nodes walked
        // before appending (0 for an empty bin).  Java calls `treeifyBin` for either
        // when it reaches TREEIFY_THRESHOLD.
        let (bin_count, old_value) = match bin.iter().position(|(k, _)| *k == key) {
            Some(j) => (j + 1, Some(std::mem::replace(&mut bin[j].1, value))),
            None => {
                let bin_count = bin.len();
                bin.push((key, value));
                (bin_count, None)
            }
        };
        if bin_count >= TREEIFY_THRESHOLD {
            // `treeifyBin`: below MIN_TREEIFY_CAPACITY, `tryPresize(n << 1)`, which
            // keeps doubling while `tableSizeFor(3n + 1)` exceeds `sizeCtl` (16
            // buckets become 128).  (At 64 bins or more Java makes a TreeBin, which
            // prepends later puts; not reproduced, see the module comment.)
            if n < MIN_TREEIFY_CAPACITY {
                let size = n << 1;
                let c = table_size_for(size + (size >> 1) + 1);
                while c > self.size_ctl && self.table.len() < MAXIMUM_CAPACITY {
                    self.transfer();
                }
            }
        }
        if old_value.is_some() {
            return old_value;
        }
        // `addCount(1L, binCount)`.
        self.size += 1;
        while self.size >= self.size_ctl && self.table.len() < MAXIMUM_CAPACITY {
            self.transfer();
        }
        None
    }

    fn find<Q>(&self, key: &Q) -> Option<(usize, usize)>
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + JavaHashCode + ?Sized,
    {
        if self.table.is_empty() {
            return None;
        }
        let i = Self::index(concurrent_spread(key.java_hash_code()), self.table.len());
        self.table[i]
            .iter()
            .position(|(k, _)| k.borrow() == key)
            .map(|j| (i, j))
    }

    /// Java `get(key)`.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + JavaHashCode + ?Sized,
    {
        self.find(key).map(|(i, j)| &self.table[i][j].1)
    }

    /// Java `containsKey(key)`.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + JavaHashCode + ?Sized,
    {
        self.find(key).is_some()
    }

    /// Java `remove(key)`; returns the removed value.  The table keeps its length.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + JavaHashCode + ?Sized,
    {
        let (i, j) = self.find(key)?;
        self.size -= 1;
        Some(self.table[i].remove(j).1)
    }

    /// Java `clear()`.  The table keeps its length.
    pub fn clear(&mut self) {
        for bin in self.table.iter_mut() {
            bin.clear();
        }
        self.size = 0;
    }

    /// Java `entrySet().iterator()`, in Java's order (bins in index order, each
    /// head first).
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.table
            .iter()
            .flat_map(|bin| bin.iter().map(|(key, value)| (key, value)))
    }

    /// Java `keySet().iterator()`, in Java's order.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(key, _)| key)
    }

    /// Java `values().iterator()`, in Java's order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, value)| value)
    }
}

/// A `java.lang.ProcessEnvironment.Variable`: the name's bytes, whose `hashCode()`
/// is `ExternalData.hashCode()`, `arrayHash` over the *signed* bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
struct EnvironmentVariable(Vec<u8>);

impl JavaHashCode for EnvironmentVariable {
    fn java_hash_code(&self) -> i32 {
        let mut hash: i32 = 0;
        for &byte in &self.0 {
            hash = hash.wrapping_mul(31).wrapping_add(byte as i8 as i32);
        }
        hash
    }
}

/// `System.getenv()`'s iteration order (OpenJDK `java.lang.ProcessEnvironment`,
/// Unix): the C `environ` entries that hold a `=` are split at the first one into
/// name and value bytes, then put into `new HashMap<>(count + 3)` **from the last
/// entry to the first** (`for (int i = environ.length-1; i > 0; i-=2)`), keyed by
/// `Variable`, whose hash is over the name's bytes; `getenv()` iterates that map.
/// Names and values are decoded lossily as UTF-8 here (Java decodes with
/// `sun.jnu.encoding`).  `std::env::vars_os` is the `environ` array, except that it
/// drops an entry whose only `=` is its first byte, which Java keeps with an empty
/// name; no shell produces one.
pub fn java_lang_system_getenv() -> Vec<(String, String)> {
    use std::os::unix::ffi::OsStrExt;
    process_environment_order(
        std::env::vars_os()
            .map(|(key, value)| (key.as_bytes().to_vec(), value.as_bytes().to_vec()))
            .collect(),
    )
    .into_iter()
    .map(|(key, value)| {
        (
            String::from_utf8_lossy(&key).into_owned(),
            String::from_utf8_lossy(&value).into_owned(),
        )
    })
    .collect()
}

/// `ProcessEnvironment`'s static initializer and `getenv()`'s iteration over the
/// `environ` entries (name, value), given in `environ` order.
fn process_environment_order(environ: Vec<(Vec<u8>, Vec<u8>)>) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut the_environment: JavaHashMap<EnvironmentVariable, Vec<u8>> =
        JavaHashMap::with_capacity(environ.len() + 3);
    for (key, value) in environ.into_iter().rev() {
        the_environment.insert(EnvironmentVariable(key), value);
    }
    the_environment
        .iter()
        .map(|(key, value)| (key.0.clone(), value.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_hash_code_matches_java() {
        // "hello".hashCode() == 99162322, "".hashCode() == 0,
        // "polygenelubricants".hashCode() == Integer.MIN_VALUE.
        assert_eq!(string_hash_code("hello"), 99162322);
        assert_eq!(string_hash_code(""), 0);
        assert_eq!(string_hash_code("polygenelubricants"), i32::MIN);
    }

    #[test]
    fn iteration_follows_buckets_then_insertion() {
        // In Java, new HashMap with put("b"),put("a"),put("c") iterates a, b, c
        // (buckets 1, 2, 3 of 16 for hashes 97, 98, 99).
        let mut map: JavaHashMap<String, i32> = JavaHashMap::new();
        map.insert("b".to_owned(), 1);
        map.insert("a".to_owned(), 2);
        map.insert("c".to_owned(), 3);
        let keys: Vec<&String> = map.keys().collect();
        assert_eq!(keys, ["a", "b", "c"]);
        // "q" (113) shares bucket 1 with "a" (97) at 16 buckets, after it.
        map.insert("q".to_owned(), 4);
        let keys: Vec<&String> = map.keys().collect();
        assert_eq!(keys, ["a", "q", "b", "c"]);
    }

    /// One scripted run: `new` (or `cap N`), then `put K`, `del K`, `clear`, `copy`
    /// (`new HashMap(this)`), `putall K...` (from a LinkedHashMap of those keys); keys
    /// are strings or, for `int` maps, integers.  Prints the final key order joined
    /// by `,`.  The same script is replayed by
    /// `/big/henriksson/gui2/hashorder/java/HashOrder.java` on the reference JVM.
    fn replay_string(script: &str) -> String {
        let mut map: JavaHashMap<String, ()> = JavaHashMap::new();
        for op in script.split(';') {
            let words: Vec<&str> = op.split_whitespace().collect();
            match words.as_slice() {
                ["cap", n] => map = JavaHashMap::with_capacity(n.parse().unwrap()),
                ["put", key] => {
                    map.insert((*key).to_owned(), ());
                }
                ["del", key] => {
                    map.remove(*key);
                }
                ["clear"] => map.clear(),
                ["copy"] => {
                    let entries: Vec<(String, ())> =
                        map.keys().map(|key| (key.clone(), ())).collect();
                    map = JavaHashMap::from_map(entries);
                }
                ["putall", keys @ ..] => {
                    // The Java side puts the keys into a LinkedHashMap first, which
                    // keeps the first occurrence of a repeated key.
                    let mut linked: Vec<String> = Vec::new();
                    for key in keys {
                        if !linked.iter().any(|k| k == key) {
                            linked.push((*key).to_owned());
                        }
                    }
                    map.put_all(linked.into_iter().map(|key| (key, ())));
                }
                _ => panic!("bad op {op}"),
            }
        }
        map.keys().cloned().collect::<Vec<_>>().join(",")
    }

    fn replay_concurrent(script: &str) -> String {
        let mut map: JavaConcurrentHashMap<i32, ()> = JavaConcurrentHashMap::new();
        for op in script.split(';') {
            let words: Vec<&str> = op.split_whitespace().collect();
            match words.as_slice() {
                ["put", key] => {
                    map.insert(key.parse().unwrap(), ());
                }
                ["del", key] => {
                    map.remove(&key.parse::<i32>().unwrap());
                }
                ["clear"] => map.clear(),
                _ => panic!("bad op {op}"),
            }
        }
        map.keys()
            .map(|key| key.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    #[test]
    fn small_scripts_match_java() {
        // Expected orders printed by OpenJDK 19 for the same scripts.
        // putAll of 12 string keys into a new HashMap presizes it to 16 buckets on
        // JDK 19 (32 before JDK-8281631).
        assert_eq!(
            replay_string("putall 16 1 2 3 4 5 6 7 8 9 10 11"),
            "11,1,2,3,4,16,5,6,7,8,9,10"
        );
        // A ConcurrentHashMap transfer reverses the nodes before the last run:
        // 0,16,32,48 in bucket 0 of 16 split at 32 buckets into 0,32 / 16,48.
        let mut script = String::from("put 0;put 16;put 32;put 48");
        for k in 1..12 {
            script.push_str(&format!(";put {}", 100 + k));
        }
        assert_eq!(
            replay_concurrent(&script),
            "32,0,101,102,103,104,105,106,107,108,109,110,111,16,48"
        );
    }

    /// Replays every script of `$IMOD_RS_JAVA_HASH_ORDER` (lines `kind<TAB>script<TAB>
    /// expected` written by `HashOrder.java` on the reference JVM; kind `s` is
    /// `HashMap<String,..>`, `c` is `ConcurrentHashMap<Integer,..>`, `e` is a `System.getenv()` whose
    /// `environ` entries are the script, separated by U+0001).  Skipped when
    /// the variable is unset.
    #[test]
    fn java_differential() {
        let Ok(path) = std::env::var("IMOD_RS_JAVA_HASH_ORDER") else {
            return;
        };
        let text = std::fs::read_to_string(path).unwrap();
        let mut count = 0;
        let mut failures: Vec<String> = Vec::new();
        for line in text.lines() {
            let fields: Vec<&str> = line.split('\t').collect();
            let expected = fields.get(2).copied().unwrap_or("");
            let actual = match fields[0] {
                "s" => replay_string(fields[1]),
                "c" => replay_concurrent(fields[1]),
                "e" => process_environment_order(
                    fields[1]
                        .split('\u{1}')
                        .filter(|entry| !entry.is_empty())
                        .map(|entry| {
                            let (key, value) = entry.split_once('=').unwrap();
                            (key.as_bytes().to_vec(), value.as_bytes().to_vec())
                        })
                        .collect(),
                )
                .into_iter()
                .map(|(key, _)| String::from_utf8(key).unwrap())
                .collect::<Vec<_>>()
                .join(","),
                kind => panic!("bad kind {kind}"),
            };
            if actual != expected {
                failures.push(format!(
                    "{} script {}\n  rust {}\n  java {}",
                    fields[0], fields[1], actual, expected
                ));
            }
            count += 1;
        }
        assert!(count > 0);
        assert!(
            failures.is_empty(),
            "{} of {} differ:\n{}",
            failures.len(),
            count,
            failures.join("\n")
        );
    }
}
