//! `IMOD/Etomo/src/etomo/util/UniqueKey.java`.

use std::hash::{Hash, Hasher};

use super::unique_hashed_array::UniqueHashedArray;

/// Java `UniqueKey`.
///
/// The name and count are immutable.  `count` distinguishes equal names in a
/// particular `UniqueHashedArray`, just as Java's package-private constructor
/// does.
#[derive(Clone, Debug, Eq)]
pub struct UniqueKey {
    name: String,
    count: i32,
}

impl UniqueKey {
    /// Java package-private `UniqueKey(String, UniqueHashedArray)`.
    pub(crate) fn new<V>(name: String, keyed_storage: &UniqueHashedArray<V>) -> Self {
        let mut temp_count = 0_i32;
        for index in 0..keyed_storage.size() {
            let Some(stored_key) = keyed_storage.get_key(index) else {
                continue;
            };
            if stored_key.name == name {
                temp_count = stored_key.count.wrapping_add(1);
            }
        }
        Self {
            name,
            count: temp_count,
        }
    }

    /// Java `equals(UniqueKey)`.
    pub fn equals(&self, that: Option<&Self>) -> bool {
        that.is_some_and(|that| self.name == that.name && self.count == that.count)
    }

    /// Java `getName()`.
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Java `hashCode()`.
    pub fn hash_code(&self) -> i32 {
        let mut name_hash_code = 0_i32;
        for character in self.name.encode_utf16() {
            name_hash_code = name_hash_code
                .wrapping_mul(31)
                .wrapping_add(i32::from(character));
        }
        name_hash_code.wrapping_add(self.count)
    }

    /// Java private `paramString()`.
    fn param_string(&self) -> String {
        format!(",name={},count ={}", self.name, self.count)
    }
}

impl PartialEq for UniqueKey {
    fn eq(&self, other: &Self) -> bool {
        self.equals(Some(other))
    }
}

impl Hash for UniqueKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_i32(self.hash_code());
    }
}

impl std::fmt::Display for UniqueKey {
    /// Java `toString()`.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "etomo.util.UniqueKey[{}]", self.param_string())
    }
}

#[cfg(test)]
mod tests {
    use super::UniqueKey;
    use crate::imod::etomo::util::unique_hashed_array::UniqueHashedArray;

    #[test]
    fn hash_code_and_to_string_match_java() {
        let mut keyed_storage = UniqueHashedArray::new();
        let first = keyed_storage.add_with_name("Aa".to_string(), ());
        let second = keyed_storage.add_with_name("Aa".to_string(), ());

        assert_eq!(first.hash_code(), 2112);
        assert_eq!(second.hash_code(), 2113);
        assert_eq!(
            second.to_string(),
            "etomo.util.UniqueKey[,name=Aa,count =1]"
        );
    }
}
