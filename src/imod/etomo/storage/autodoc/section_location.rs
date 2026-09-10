//! `IMOD/Etomo/src/etomo/storage/autodoc/SectionLocation.java`.

/// Iterator state for the source Autodoc section traversal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectionLocation {
    section_type: Option<String>,
    index: i32,
}

impl SectionLocation {
    /// Java `SectionLocation(String, int)`.
    pub fn new(section_type: Option<String>, index: i32) -> Self {
        Self {
            section_type,
            index,
        }
    }
    /// Java `SectionLocation(int)`.
    pub fn new_with_index(index: i32) -> Self {
        Self {
            section_type: None,
            index,
        }
    }
    /// Java `getType()`.
    pub fn get_type(&self) -> Option<&str> {
        self.section_type.as_deref()
    }
    /// Java `getIndex()`.
    pub fn get_index(&self) -> i32 {
        self.index
    }
    /// Java `setIndex(int)`.
    pub fn set_index(&mut self, index: i32) {
        self.index = index;
    }
}
impl std::fmt::Display for SectionLocation {
    /// Java `toString()`.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "[type={},index={}]",
            self.section_type.as_deref().unwrap_or("null"),
            self.index
        )
    }
}

#[cfg(test)]
mod tests {
    use super::SectionLocation;
    #[test]
    fn source_string_and_mutation_are_preserved() {
        let mut location = SectionLocation::new(Some("Field".into()), 4);
        location.set_index(5);
        assert_eq!(location.to_string(), "[type=Field,index=5]");
    }
}
