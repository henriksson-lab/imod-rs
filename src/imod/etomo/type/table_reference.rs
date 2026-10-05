//! `IMOD/Etomo/src/etomo/type/TableReference.java`.
//!
//! Generates and stores IDs which are unique within an instance of the class.  Each ID
//! is linked to a unique, non-null parameter string value, the instance-level
//! uniqueness of which is enforced.  This string is a file name minus the extension.
//! IDs are not modifiable and are never deleted.  The string values may be modified.
//! The IDs can be used as stable keys for serializing.  Each instance of
//! TableReference needs a unique prefix string for its IDs.
//!
//! IMPORTANT:  The constructor parameter idPrefix value is used as a key in data
//! files, so changing its value requires backwards compatibility code to be added.
//!
//! Properties example:
//! ```text
//! meta.ref.ebt1=/home/sueh/NOBACKUP/test datasets/Linux/Development4/UITests/dual-test-guiORIG/BBa.st
//! meta.ref.ebt2=/home/sueh/NOBACKUP/test datasets/Linux/Development4/UITests/dual-montage/midzone2a.st
//! meta.ref.ebt.lastID=ebt3
//! ```
//!
//! **Representation.**  One instance is shared by the batchruntomo manager, its meta
//! data, its dataset table (event dispatch thread) and the serieswatcher monitor (a
//! process thread), so the state sits behind one lock and every method takes `&self`.
//! The two Java `HashMap`s are `JavaHashMap`s, which iterate in Java's order: `store`
//! walks `idMap` (its "stored ref" diagnostics come out in that order), `idIterator`
//! decides the order the rows are loaded in, and `toString` prints both maps.

use std::collections::BTreeMap;

use crate::imod::etomo::util::java_hash_map::JavaHashMap;
use std::sync::Mutex;

use super::const_etomo_number::Type;
use super::duplicate_exception::DuplicateException;
use super::etomo_number::EtomoNumber;
use super::not_loaded_exception::NotLoadedException;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java private static final `GROUP_KEY`.
const GROUP_KEY: &str = "ref";
/// Java private static final `BASE_ID_NUM`.
const BASE_ID_NUM: &str = "0";
/// Java private static final `LAST_ID_KEY`.
const LAST_ID_KEY: &str = "lastID";

/// What `TableReference.put` throws.
#[derive(Debug)]
pub enum PutError {
    /// `DuplicateException`.
    Duplicate(DuplicateException),
    /// `NotLoadedException`.
    NotLoaded(NotLoadedException),
}

/// The fields of Java `TableReference`.
struct State {
    /// Java private final `idMap`: Map<fileKey (file-path minus extension), ID>.
    id_map: JavaHashMap<String, String>,
    /// Java private final `filePathMap`: Map<ID, file-path>.
    file_path_map: JavaHashMap<String, String>,
    /// Java private `loaded`, initially false.
    loaded: bool,
    /// Java private `lastIDNum = new EtomoNumber(EtomoNumber.Type.LONG)`.
    last_id_num: EtomoNumber,
}

/// Java `public final class TableReference`.
pub struct TableReference {
    /// Java private final `idPrefix`.
    id_prefix: String,
    state: Mutex<State>,
}

impl TableReference {
    /// Java `TableReference(String)`.
    pub fn new(id_prefix: &str) -> TableReference {
        let mut last_id_num = EtomoNumber::new_with_type(Some(Type::Long));
        last_id_num.set_string(Some(BASE_ID_NUM));
        TableReference {
            id_prefix: id_prefix.to_owned(),
            state: Mutex::new(State {
                id_map: JavaHashMap::new(),
                file_path_map: JavaHashMap::new(),
                loaded: false,
                last_id_num,
            }),
        }
    }

    /// Java `getID(String)`.  An ID if the key derived from filePath has already been
    /// loaded into this reference, otherwise null.
    pub fn get_id(&self, file_path: Option<&str>) -> Option<String> {
        let state = self.state.lock().unwrap();
        Self::make_file_key(file_path).and_then(|key| state.id_map.get(&key).cloned())
    }

    /// Java private static `makeFileKey(String)`.
    fn make_file_key(file_path: Option<&str>) -> Option<String> {
        // CleanPrint.ALLOW_DUPLICATES.printToOut("G:", "fileKey:" + fileKey, false);
        utilities::remove_extension(file_path)
    }

    /// Java `getFilePath(String)`.
    pub fn get_file_path(&self, id: Option<&str>) -> Option<String> {
        let state = self.state.lock().unwrap();
        id.and_then(|id| state.file_path_map.get(id).cloned())
    }

    /// Java `changeFilePath(String, String)`.  Retrieve the entries associated with this
    /// id and replace the filePath and fileKey.
    pub fn change_file_path(&self, id: Option<&str>, new_file_path: Option<&str>) {
        let mut state = self.state.lock().unwrap();
        let Some(id) = id else {
            return;
        };
        if let Some(old_file_path) = state.file_path_map.get(id).cloned() {
            let old_file_key = Self::make_file_key(Some(&old_file_path));
            state.file_path_map.remove(id);
            if let Some(old_file_key) = old_file_key {
                state.id_map.remove(&old_file_key);
            }
            if let Some(new_file_path) = new_file_path {
                state
                    .file_path_map
                    .insert(id.to_owned(), new_file_path.to_owned());
            }
            if let Some(new_file_key) = Self::make_file_key(new_file_path) {
                state.id_map.insert(new_file_key, id.to_owned());
            }
        }
    }

    /// Java `put(String) throws DuplicateException, NotLoadedException`.  Generates an ID
    /// and adds it as the value to idMap, with uniqueString as the key.  Add the
    /// uniqueString to unqueStringMap, with the ID as the key.
    pub fn put(&self, file_path: &str) -> Result<String, PutError> {
        let mut state = self.state.lock().unwrap();
        let file_key = Self::make_file_key(Some(file_path)).unwrap_or_default();
        if state.id_map.contains_key(&file_key) {
            return Err(PutError::Duplicate(DuplicateException::new(&format!(
                "String is already associated with an ID:  {}",
                file_path
            ))));
        }
        let id = self.next_id(&mut state).map_err(PutError::NotLoaded)?;
        state.id_map.insert(file_key, id.clone());
        state.file_path_map.insert(id.clone(), file_path.to_owned());
        Ok(id)
    }

    /// Java package-private `idIterator()`: the IDs.
    pub fn id_iterator(&self) -> Vec<String> {
        self.state
            .lock()
            .unwrap()
            .id_map
            .values()
            .cloned()
            .collect()
    }

    /// Java private `nextID() throws NotLoadedException`.  Generates an instance-level
    /// unique ID.
    fn next_id(&self, state: &mut State) -> Result<String, NotLoadedException> {
        if !state.loaded {
            return Err(NotLoadedException::new("Reference uninitialized"));
        }
        state.last_id_num.add_int(1);
        Ok(format!("{}{}", self.id_prefix, state.last_id_num))
    }

    /// Java private `createPrepend(String)`.
    fn create_prepend(prepend: Option<&str>) -> String {
        let Some(prepend) = prepend else {
            return GROUP_KEY.to_owned();
        };
        if java_lang_string_matches_whitespace(prepend) {
            return GROUP_KEY.to_owned();
        }
        let prepend =
            crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim(prepend);
        if prepend.ends_with('.') {
            return format!("{}{}", prepend, GROUP_KEY);
        }
        format!("{}.{}", prepend, GROUP_KEY)
    }

    /// Java `setNew()`.  Call when the instance is new, and doesn't have to be loaded
    /// from properties.
    pub fn set_new(&self) {
        self.state.lock().unwrap().loaded = true;
    }

    /// Java private `loadLastIDNum(String, boolean, String)`.  Load lastIdNum from
    /// lastId.  Places an error message in the _err.log if it fails.
    fn load_last_id_num(
        &self,
        state: &mut State,
        last_id: Option<&str>,
        repair_done: bool,
        prepend: &str,
    ) -> bool {
        state.loaded = false;
        state.last_id_num.reset();
        if let Some(last_id) = last_id {
            if !last_id.starts_with(&self.id_prefix) {
                // May be just the number part of the ID
                state.last_id_num.set_string(Some(last_id));
            } else {
                state
                    .last_id_num
                    .set_string(Some(&last_id[self.id_prefix.len()..]));
            }
        }
        if state.last_id_num.is_null()
            || !state.last_id_num.is_valid()
            || state.last_id_num.lt_string(Some(BASE_ID_NUM))
        {
            if !repair_done {
                eprintln!(
                    "WARNING: property {} is invalid in the dataset file.  lastIDNum:{},lastID:{},prepend{}\nAttempting to repair...",
                    self.get_last_id_key(prepend),
                    state.last_id_num,
                    last_id.unwrap_or("null"),
                    prepend
                );
                // Thread.dumpStack(): the Java thread's stack trace has no Rust
                // counterpart.
            } else {
                eprintln!(
                    "ERROR: property {} is invalid in the dataset file: {}.  Unable to load.",
                    self.get_last_id_key(prepend),
                    state.last_id_num
                );
            }
            return false;
        }
        true
    }

    /// Java private `getLastIDKey(String)`.  Assumes that prepend has already been set up.
    fn get_last_id_key(&self, prepend: &str) -> String {
        format!("{}.{}.{}", prepend, self.id_prefix, LAST_ID_KEY)
    }

    /// Java package-private `load(Properties, String)`.  Load from properties.
    pub fn load(&self, props: &BTreeMap<String, String>, prepend: Option<&str>) {
        let mut state = self.state.lock().unwrap();
        // reset
        state.loaded = false;
        state.last_id_num.reset();
        state.id_map.clear();
        state.file_path_map.clear();
        // load
        let prepend = Self::create_prepend(prepend);
        let mut last_id = props.get(&self.get_last_id_key(&prepend)).cloned();
        if !self.load_last_id_num(&mut state, last_id.as_deref(), false, &prepend) {
            // Attempt to repair lastID
            let generic_key = format!("{}.{}", prepend, self.id_prefix);
            last_id = None;
            // To repair lastId, find the highest ID in props
            for key in props.keys() {
                if key.starts_with(&generic_key) {
                    let cur_id = key[generic_key.len() - self.id_prefix.len()..].to_owned();
                    // `lastID.compareTo(curID) < 0`: Java compares UTF-16 code units,
                    // which orders like `str` comparison for every character outside
                    // the supplementary planes (and identically for these IDs).
                    if last_id.is_none() || last_id.as_deref().unwrap() < cur_id.as_str() {
                        last_id = Some(cur_id);
                    }
                }
            }
            if !self.load_last_id_num(&mut state, last_id.as_deref(), true, &prepend) {
                return;
            }
        }
        // Loading prepend properties. All properties with IDs greater then lastID will
        // not be loaded, and may be overwritten.;
        // loading prepend.ID = uniqueString
        let last = state.last_id_num.get_long();
        let mut id_num: i64 = 1;
        while id_num <= last {
            let id = format!("{}{}", self.id_prefix, id_num);
            let file_path = props.get(&format!("{}.{}", prepend, id)).cloned();
            // Since put prevents a null uniqueString from being saved, assume that a null
            // uniqueString here mean that this ID was deleted - not an error.
            if let Some(file_path) = file_path {
                if file_path.contains("/dual/") {
                    // Looking for mystery bug:
                    // meta.ref.ebt4=/home/NOBACKUP/sueh/test
                    // datasets/Development/linux/UITests/dual/dual/BBa.st
                    eprintln!("loaded ref {}", file_path);
                }
                let file_key = Self::make_file_key(Some(&file_path)).unwrap_or_default();
                if !state.id_map.contains_key(&file_key) {
                    state.id_map.insert(file_key, id.clone());
                    state.file_path_map.insert(id, file_path);
                } else {
                    eprintln!(
                        "ERROR: duplicate string, {}, under {}.{} in the dataset file.  This property will not be loaded.",
                        file_path, prepend, id
                    );
                }
            }
            id_num += 1;
        }
        state.loaded = true;
    }

    /// Java package-private `store(Properties, String)`.  Store in properties.
    pub fn store(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let state = self.state.lock().unwrap();
        if !state.loaded {
            return;
        }
        let prepend = Self::create_prepend(prepend);
        props.insert(
            self.get_last_id_key(&prepend),
            format!("{}{}", self.id_prefix, state.last_id_num),
        );
        // saving:
        // prepend.ID = filePath
        for (file_key, id) in state.id_map.iter() {
            if file_key.contains("/dual/") {
                // Looking for mystery bug:
                // meta.ref.ebt4=/home/NOBACKUP/sueh/test
                // datasets/Development/linux/UITests/dual/dual/BBa.st
                eprintln!("stored ref {}", file_key);
            }
            // Save filePath
            if let Some(file_path) = state.file_path_map.get(id) {
                props.insert(format!("{}.{}", prepend, id), file_path.clone());
            }
        }
    }
}

/// Java `toString()` (the source leaves the bracket open).
impl std::fmt::Display for TableReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap();
        write!(
            f,
            "[idMap:{},\nfilePathMap:{},\nidPrefix:{},loaded:{},lastIDNum:{}",
            state
                .id_map
                .to_java_string(|key| key.clone(), |value| value.clone()),
            state
                .file_path_map
                .to_java_string(|key| key.clone(), |value| value.clone()),
            self.id_prefix,
            state.loaded,
            state.last_id_num
        )
    }
}
