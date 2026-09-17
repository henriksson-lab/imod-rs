//! `IMOD/Etomo/src/etomo/ManagerKey.java`.
//!
//! `ManagerKey` is intentionally a separate mutable holder around `UniqueKey`.
//! In particular, two holders with null keys compare equal only when they are the
//! same holder, matching the source's reference-identity branch.
#![allow(dead_code)]

use crate::imod::etomo::util::unique_key::UniqueKey;

/// Java package-private final `ManagerKey`.
pub(crate) struct ManagerKey {
    /// Java private `uniqueKey`, initialised to null.
    unique_key: Option<UniqueKey>,
}

impl Default for ManagerKey {
    /// Java's implicit no-argument constructor.
    fn default() -> Self {
        Self { unique_key: None }
    }
}

impl ManagerKey {
    /// Java `equals(ManagerKey)`.
    ///
    /// Rust cannot overload `equals`, so the argument type is included in the
    /// systematically snake-cased name.
    pub(crate) fn equals_manager_key(&self, manager_key: Option<&ManagerKey>) -> bool {
        let Some(manager_key) = manager_key else {
            return false;
        };
        if self.unique_key.is_none() != manager_key.unique_key.is_none() {
            return false;
        }
        if self.unique_key.is_none() {
            return std::ptr::eq(self, manager_key);
        }
        self.unique_key
            .as_ref()
            .unwrap()
            .equals(manager_key.unique_key.as_ref())
    }

    /// Java `equals(UniqueKey)`.
    pub(crate) fn equals_unique_key(&self, unique_key: Option<&UniqueKey>) -> bool {
        if self.unique_key.is_none() && unique_key.is_none() {
            return true;
        }
        if self.unique_key.is_none() != unique_key.is_none() {
            return false;
        }
        self.unique_key.as_ref().unwrap().equals(unique_key)
    }

    /// Java package-private `setKey`.
    ///
    /// `UniqueKey` is immutable, so ownership transfer is equivalent to Java's
    /// reference assignment for this source type.
    pub(crate) fn set_key(&mut self, key: Option<UniqueKey>) {
        self.unique_key = key;
    }

    /// Java `getKey()`.
    pub fn get_key(&self) -> Option<&UniqueKey> {
        self.unique_key.as_ref()
    }
}

/// Java `toString`.
impl std::fmt::Display for ManagerKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.unique_key.as_ref() {
            Some(unique_key) => write!(formatter, "[{unique_key}]"),
            None => write!(formatter, "[null]"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ManagerKey;
    use crate::imod::etomo::util::unique_hashed_array::UniqueHashedArray;

    #[test]
    fn equals_manager_key_uses_java_identity_for_two_null_keys() {
        let first = ManagerKey::default();
        let second = ManagerKey::default();

        assert!(!first.equals_manager_key(None));
        assert!(first.equals_manager_key(Some(&first)));
        assert!(!first.equals_manager_key(Some(&second)));
        assert!(first.equals_unique_key(None));
        assert_eq!(first.to_string(), "[null]");
    }

    #[test]
    fn equals_manager_key_uses_the_unique_key_value() {
        let mut keyed_storage = UniqueHashedArray::new();
        let key = keyed_storage.add_with_name("manager".to_string(), ());
        let mut first = ManagerKey::default();
        let mut second = ManagerKey::default();
        first.set_key(Some(key.clone()));
        second.set_key(Some(key));

        assert!(first.equals_manager_key(Some(&second)));
        assert!(first.equals_unique_key(second.get_key()));
        assert!(!first.equals_unique_key(None));
    }
}
