//! `IMOD/Etomo/src/etomo/storage/autodoc/StatementLocation.java`.

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StatementLocation {
    index: usize,
    debug: bool,
}
impl StatementLocation {
    /// Java package-private `StatementLocation()`.
    pub fn new() -> Self {
        Self::default()
    }
    /// Java `getIndex()`.
    pub fn get_index(&self) -> usize {
        self.index
    }
    /// Java `setIndex(int)`.
    pub fn set_index(&mut self, index: usize) {
        self.index = index;
    }
    /// Java `increment()`.
    pub fn increment(&mut self) {
        if self.debug {
            println!("increment:index={}", self.index);
        }
        self.index += 1;
    }
    /// Java `isOutOfRange(List)`.
    pub fn is_out_of_range<T>(&self, list: Option<&[T]>) -> bool {
        if self.debug {
            println!("isOutOfRange:list.size()={}", list.map_or(0, <[T]>::len));
        }
        list.is_none_or(|list| self.index >= list.len())
    }
    /// Java `toString()`.
    pub fn to_string(&self) -> String {
        format!("index={}", self.index)
    }
    /// Java `setDebug(boolean)`.
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }
}
#[cfg(test)]
mod tests {
    use super::StatementLocation;
    #[test]
    fn tracks_source_list_location() {
        let mut location = StatementLocation::new();
        assert!(!location.is_out_of_range(Some(&[1])));
        location.increment();
        assert!(location.is_out_of_range(Some(&[1])));
        assert!(location.is_out_of_range::<i32>(None));
        assert_eq!(location.to_string(), "index=1");
    }
}
