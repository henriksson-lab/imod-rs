//! Owned Rust representation of declarations in `IMOD/librgctf/f2c.h`.
//!
//! The original header is an ABI bridge for f2c-generated C.  Rust callers
//! use these values as ordinary owned descriptions; no C pointers, varargs,
//! or function-pointer aliases are retained.

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FortranComplex {
    pub real: f32,
    pub imaginary: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FortranDoubleComplex {
    pub real: f64,
    pub imaginary: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExternalIoList {
    pub error: bool,
    pub unit: i64,
    pub end: bool,
    pub format: Option<String>,
    pub record: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InternalIoList {
    pub error: bool,
    pub unit: String,
    pub end: bool,
    pub format: Option<String>,
    pub record_length: i64,
    pub record_number: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpenList {
    pub error: bool,
    pub unit: i64,
    pub file_name: Option<String>,
    pub status: Option<String>,
    pub access: Option<String>,
    pub format: Option<String>,
    pub record_length: i64,
    pub blank: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CloseList {
    pub error: bool,
    pub unit: i64,
    pub status: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PositionList {
    pub error: bool,
    pub unit: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InquireList {
    pub error: bool,
    pub unit: i64,
    pub file_name: Option<String>,
    pub exists: Option<bool>,
    pub is_open: Option<bool>,
    pub number: Option<i64>,
    pub is_named: Option<bool>,
    pub name: Option<String>,
    pub access: Option<String>,
    pub sequential: Option<String>,
    pub direct: Option<String>,
    pub format: Option<String>,
    pub formatted: Option<String>,
    pub unformatted: Option<String>,
    pub record_length: Option<i64>,
    pub next_record: Option<i64>,
    pub blank: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MultiType {
    Integer1(i8),
    Short(i16),
    Integer(i32),
    Real(f32),
    DoubleReal(f64),
    Complex(FortranComplex),
    DoubleComplex(FortranDoubleComplex),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VariableDescriptor {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub type_code: i32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NameList {
    pub name: String,
    pub variables: Vec<VariableDescriptor>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_f2c_descriptions_do_not_need_c_pointers() {
        let list = NameList {
            name: "input".into(),
            variables: vec![VariableDescriptor {
                name: "n".into(),
                dimensions: vec![2, 3],
                type_code: 1,
            }],
        };
        assert_eq!(list.variables[0].dimensions, [2, 3]);
        assert_eq!(FortranComplex::default().imaginary, 0.);
    }
}
