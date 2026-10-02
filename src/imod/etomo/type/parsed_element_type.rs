//! `IMOD/Etomo/src/etomo/type/ParsedElementType.java`.
//!
//! The type changes how an empty element displays itself.  A Java class with six
//! static instances compared by identity: the instances are `pub static` items here,
//! handed around as `&'static ParsedElementType` and compared with `std::ptr::eq`.
//! Java declares no fields; `id` exists only so that each static is a distinct,
//! non-zero-sized object (zero-sized statics need not have distinct addresses).

/// Java `ParsedElementType`.
#[derive(Debug)]
pub struct ParsedElementType {
    /// Not a Java field: gives every instance its own address (see the module comment).
    id: u8,
}

/// Java `NON_MATLAB_NUMBER`.
pub static NON_MATLAB_NUMBER: ParsedElementType = ParsedElementType { id: 0 };
/// Java `MATLAB_NUMBER`.
pub static MATLAB_NUMBER: ParsedElementType = ParsedElementType { id: 1 };
/// Java package-private `NON_MATLAB_ARRAY`.
pub static NON_MATLAB_ARRAY: ParsedElementType = ParsedElementType { id: 2 };
/// Java `MATLAB_ARRAY`.
pub static MATLAB_ARRAY: ParsedElementType = ParsedElementType { id: 3 };
/// Java `MATLAB_ARRAY_DESCRIPTOR`.
pub static MATLAB_ARRAY_DESCRIPTOR: ParsedElementType = ParsedElementType { id: 4 };
/// Java package-private `STRING`.
pub static STRING: ParsedElementType = ParsedElementType { id: 5 };

impl ParsedElementType {
    /// Java package-private `isArray()`.
    pub fn is_array(&self) -> bool {
        if std::ptr::eq(self, &NON_MATLAB_ARRAY)
            || std::ptr::eq(self, &MATLAB_ARRAY)
            || std::ptr::eq(self, &MATLAB_ARRAY_DESCRIPTOR)
        {
            return true;
        }
        false
    }

    /// Java package-private `isMatlab()`.
    pub fn is_matlab(&self) -> bool {
        if std::ptr::eq(self, &MATLAB_NUMBER)
            || std::ptr::eq(self, &MATLAB_ARRAY)
            || std::ptr::eq(self, &MATLAB_ARRAY_DESCRIPTOR)
        {
            return true;
        }
        false
    }

    /// Java package-private `toArrayInstance()`.
    pub fn to_array_instance(&'static self) -> &'static ParsedElementType {
        if std::ptr::eq(self, &NON_MATLAB_ARRAY)
            || std::ptr::eq(self, &MATLAB_ARRAY)
            || std::ptr::eq(self, &MATLAB_ARRAY_DESCRIPTOR)
            || std::ptr::eq(self, &STRING)
        {
            // Already an array or has no array equivalent
            return self;
        }
        if std::ptr::eq(self, &NON_MATLAB_NUMBER) {
            return &NON_MATLAB_ARRAY;
        }
        if std::ptr::eq(self, &MATLAB_NUMBER) {
            return &MATLAB_ARRAY;
        }
        &NON_MATLAB_ARRAY
    }
}

/// Java `toString()`.
impl std::fmt::Display for ParsedElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if std::ptr::eq(self, &NON_MATLAB_NUMBER) {
            return f.write_str("Non Matlab Number");
        }
        if std::ptr::eq(self, &NON_MATLAB_ARRAY) {
            return f.write_str("Non Matlab Array");
        }
        if std::ptr::eq(self, &MATLAB_NUMBER) {
            return f.write_str("Matlab Number");
        }
        if std::ptr::eq(self, &MATLAB_ARRAY) {
            return f.write_str("Matlab Array");
        }
        if std::ptr::eq(self, &MATLAB_ARRAY_DESCRIPTOR) {
            return f.write_str("Matlab Array Descriptor");
        }
        if std::ptr::eq(self, &STRING) {
            return f.write_str("String");
        }
        f.write_str("Unknown ParsedElementType")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instances_map_to_their_array_forms() {
        assert!(std::ptr::eq(
            NON_MATLAB_NUMBER.to_array_instance(),
            &NON_MATLAB_ARRAY
        ));
        assert!(std::ptr::eq(
            MATLAB_NUMBER.to_array_instance(),
            &MATLAB_ARRAY
        ));
        assert!(MATLAB_ARRAY_DESCRIPTOR.is_array() && MATLAB_ARRAY_DESCRIPTOR.is_matlab());
        assert!(!STRING.is_array());
        assert_eq!(STRING.to_string(), "String");
    }
}
