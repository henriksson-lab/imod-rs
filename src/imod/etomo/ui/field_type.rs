//! `IMOD/Etomo/src/etomo/ui/FieldType.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final` singleton, matching the shape used by
//! `etomo::r#type::image_filename_style`.  The four `private final` fields are the
//! values the constructor was handed, so they are recovered from the variant rather
//! than stored.
#![allow(dead_code)]

use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::validation_type::ValidationType;

/// Java's nested `public static final class CollectionType`.  Contains the types of
/// collections and a splitter that can be used to divide them into numeric elements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectionType {
    /// Java `ARRAY`.  Array separators are commas and whitespace.
    Array,
    /// Java `LIST`.  List separators are commas, whitespace, and dashes.
    List,
    /// Java `MATLAB_ARRAY`.  Matlab Array separators are commas, whitespace and colons.
    MatlabArray,
}

impl CollectionType {
    /// Java field `splitter`.
    fn splitter(self) -> &'static str {
        match self {
            Self::Array => "\\s*,\\s*|\\s+",
            Self::List => "\\s*,\\s*|\\s+|\\s*\\-\\s*",
            Self::MatlabArray => "\\s*,\\s*|\\s+|\\s*\\:\\s*",
        }
    }

    /// Java `getSplitter`.
    pub fn get_splitter(self) -> &'static str {
        self.splitter()
    }
}

/// Java `FieldType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldType {
    /// Java `STRING`: `ValidationType.STRING`, 9 columns.
    String,
    /// Java `FILE`: `ValidationType.STRING`, 15 columns.
    File,
    /// Java `STRING_ARRAY`: `ValidationType.STRING`, `CollectionType.ARRAY`, 15 columns.
    StringArray,
    /// Java `INTEGER`: `ValidationType.INTEGER`, 3 columns.
    Integer,
    /// Java `FLOATING_POINT`: `ValidationType.FLOATING_POINT`, 3 columns.
    FloatingPoint,
    /// Java `INTEGER_PAIR`: `ValidationType.INTEGER`, `CollectionType.ARRAY`, required
    /// size 2, 6 columns.
    IntegerPair,
    /// Java `FLOATING_POINT_PAIR`: `ValidationType.FLOATING_POINT`,
    /// `CollectionType.ARRAY`, required size 2, 6 columns.
    FloatingPointPair,
    /// Java `INTEGER_TRIPLE`: `ValidationType.INTEGER`, `CollectionType.ARRAY`, required
    /// size 3, 9 columns.
    IntegerTriple,
    /// Java `FLOATING_POINT_ARRAY`: `ValidationType.FLOATING_POINT`,
    /// `CollectionType.ARRAY`, 9 columns.
    FloatingPointArray,
    /// Java `INTEGER_ARRAY`: `ValidationType.INTEGER`, `CollectionType.ARRAY`, 9
    /// columns.
    IntegerArray,
    /// Java `INTEGER_LIST`: `ValidationType.INTEGER`, `CollectionType.LIST`, 9 columns.
    /// An array description (may contain elements like '1 - 3').
    IntegerList,
    /// Java `MATLAB_INTEGER_ARRAY`: `ValidationType.INTEGER`,
    /// `CollectionType.MATLAB_ARRAY`, 9 columns.
    MatlabIntegerArray,
}

impl FieldType {
    /// Java public field `validationType`.
    pub fn validation_type(self) -> ValidationType {
        match self {
            Self::String | Self::File | Self::StringArray => ValidationType::String,
            Self::Integer
            | Self::IntegerPair
            | Self::IntegerTriple
            | Self::IntegerArray
            | Self::IntegerList
            | Self::MatlabIntegerArray => ValidationType::Integer,
            Self::FloatingPoint | Self::FloatingPointPair | Self::FloatingPointArray => {
                ValidationType::FloatingPoint
            }
        }
    }

    /// Java private field `collectionType`.  The two-argument constructor leaves it
    /// null.
    fn collection_type(self) -> Option<CollectionType> {
        match self {
            Self::String | Self::File | Self::Integer | Self::FloatingPoint => None,
            Self::StringArray
            | Self::IntegerPair
            | Self::FloatingPointPair
            | Self::IntegerTriple
            | Self::FloatingPointArray
            | Self::IntegerArray => Some(CollectionType::Array),
            Self::IntegerList => Some(CollectionType::List),
            Self::MatlabIntegerArray => Some(CollectionType::MatlabArray),
        }
    }

    /// Java public field `requiredSize`, for arrays with a fixed number of elements
    /// (pairs and triples).  The constructors that do not take one set it to -1.
    pub fn required_size(self) -> i32 {
        match self {
            Self::IntegerPair | Self::FloatingPointPair => 2,
            Self::IntegerTriple => 3,
            _ => -1,
        }
    }

    /// Java private field `columns`.
    fn columns(self) -> i32 {
        match self {
            Self::String
            | Self::StringArray
            | Self::FloatingPointArray
            | Self::IntegerArray
            | Self::IntegerList
            | Self::MatlabIntegerArray => match self {
                Self::StringArray => 15,
                _ => 9,
            },
            Self::File => 15,
            Self::Integer | Self::FloatingPoint => 3,
            Self::IntegerPair | Self::FloatingPointPair => 6,
            Self::IntegerTriple => 9,
        }
    }

    /// Java `getInstance(DirectiveValueType)`.
    pub fn get_instance(value_type: Option<std::convert::Infallible>) -> Option<FieldType> {
        // TODO(unit): needs etomo/storage/DirectiveValueType.java - the parameter's
        // declared type; the body is a chain of identity tests against its ten
        // singletons.
        let _ = value_type;
        None
    }

    /// Java `getNumericType`.  Return the largest possible equivalent numeric type.
    /// Returns null for strings, arrays, and lists.
    pub fn get_numeric_type(self) -> Option<Type> {
        self.validation_type().get_numeric_type()
    }

    /// Java `getCollectionType`.
    pub fn get_collection_type(self) -> Option<CollectionType> {
        self.collection_type()
    }

    /// Java `hasRequiredSize`.
    pub fn has_required_size(self) -> bool {
        self.required_size() != -1
    }

    /// Java `isCollection`.
    pub fn is_collection(self) -> bool {
        self.collection_type().is_some()
    }

    /// Java `getColumns`.
    pub fn get_columns(self) -> i32 {
        self.columns()
    }

    /// Java `getSplitter`.  The source dereferences `collectionType` without a null
    /// check, so a non-collection field type throws a NullPointerException here.
    pub fn get_splitter(self) -> &'static str {
        self.collection_type().unwrap().splitter()
    }
}

/// Java `toString`.  Note that the source's string is missing its closing bracket.
impl std::fmt::Display for FieldType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[validationType:{},collectionType:{},requiredSize:{}",
            self.validation_type(),
            match self.collection_type() {
                // `CollectionType` declares no `toString`, so Java prints
                // `Object.toString()` - `getClass().getName() + "@" +
                // Integer.toHexString(hashCode())` - whose identity hash is not
                // reproducible.
                None => "null".to_string(),
                Some(_) => "null".to_string(),
            },
            self.required_size()
        )
    }
}
