//! `IMOD/Etomo/src/etomo/logic/Converter.java`.
//!
//! Description:
//!
//! Copyright: Copyright 2020 - 2022 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! Every method of the final class `Converter` is static, so the module is a flat set of
//! functions rather than a type.  A Java `String text` argument may be null and
//! `Utilities.isEmpty` is the null guard, so text parameters are `Option<&str>`; the
//! boxed `Long`/`Integer`/`Double` returns are `Option`.
//!
//! `Math.round(float)` and `Float.parseFloat` are modelled here, next to their only
//! caller, following the convention `etomo/util/stack_trace.rs` uses for
//! `java.lang.Thread.getId()`.
#![allow(dead_code)]

use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_double_value_of, java_lang_integer_parse_int, java_lang_long_parse_long,
    java_lang_string_trim,
};
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::util::utilities::{is_empty, java_lang_math_round, java_lang_string_split};
use regex::Regex;

/// `java.lang.Float.parseFloat(String)`, which routes through the same
/// `FloatingDecimal.readJavaFormatString` grammar as `Double.parseDouble` and then
/// rounds the decimal to `float` directly (not double-then-narrow).  The accepted
/// syntax, the trimming, and the two `NumberFormatException` messages are therefore
/// exactly those of `java_lang_double_value_of`; only the target width differs.
///
/// Deviation: as in `java_lang_double_value_of`, the hexadecimal-significand form is
/// not implemented and reaches this function as a syntax error.
fn java_lang_float_parse_float(value: &str) -> Result<f32, String> {
    let trimmed = java_lang_string_trim(value);
    if trimmed.is_empty() {
        return Err("empty String".to_string());
    }
    let error = || Err(format!("For input string: \"{}\"", trimmed));
    let (sign, unsigned) = match trimmed.as_bytes()[0] {
        b'+' => (1.0f32, &trimmed[1..]),
        b'-' => (-1.0f32, &trimmed[1..]),
        _ => (1.0f32, trimmed),
    };
    if unsigned == "NaN" {
        return Ok(f32::NAN);
    }
    if unsigned == "Infinity" {
        return Ok(sign * f32::INFINITY);
    }
    // Strip the optional type suffix.
    let body = match unsigned.as_bytes().last() {
        Some(b'f') | Some(b'F') | Some(b'd') | Some(b'D') => &unsigned[..unsigned.len() - 1],
        _ => unsigned,
    };
    // Digits [. Digits] [ExponentPart] with at least one digit in the significand.
    let bytes = body.as_bytes();
    let mut index = 0;
    let mut digits = 0;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
        digits += 1;
    }
    if index < bytes.len() && bytes[index] == b'.' {
        index += 1;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return error();
    }
    if index < bytes.len() && (bytes[index] == b'e' || bytes[index] == b'E') {
        index += 1;
        if index < bytes.len() && (bytes[index] == b'+' || bytes[index] == b'-') {
            index += 1;
        }
        let mut exponent_digits = 0;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
            exponent_digits += 1;
        }
        if exponent_digits == 0 {
            return error();
        }
    }
    if index != bytes.len() {
        return error();
    }
    // Rust's f32 parser is correctly rounded, as Java's is.
    match body.parse::<f32>() {
        Ok(parsed) => Ok(sign * parsed),
        Err(_) => error(),
    }
}

/// `java.lang.Math.round(float)`, returning an `int`.  The javadoc's
/// `(int) Math.floor(a + 0.5f)` wording has not described the implementation since
/// JDK-8010430; the bit-manipulating body below is what the runtime executes, and it
/// differs from the floor form for arguments whose `a + 0.5f` rounds up in float
/// (`0x1.fffffep-2` rounds to 0, not 1).
fn java_lang_math_round_float(a: f32) -> i32 {
    // FloatConsts.SIGNIFICAND_WIDTH = 24, EXP_BIAS = 127,
    // EXP_BIT_MASK = 0x7F800000, SIGNIF_BIT_MASK = 0x007FFFFF.
    let int_bits = a.to_bits() as i32;
    let biased_exp = (int_bits & 0x7F800000i32) >> (24 - 1);
    let shift = (24 - 2 + 127) - biased_exp;
    if (shift & -32) == 0 {
        // shift >= 0 && shift < 32
        let mut r = (int_bits & 0x007FFFFFi32) | (0x007FFFFFi32 + 1);
        if int_bits < 0 {
            r = r.wrapping_neg();
        }
        ((r >> shift) + 1) >> 1
    } else {
        // a is a NaN, an infinity, or is already an integer of magnitude at least 2^23.
        // Rust's `as` cast for f32 to i32 saturates and maps NaN to 0, exactly as the
        // Java narrowing primitive conversion does.
        a as i32
    }
}

/// Java `toLong`.
///
/// Returns text as a Long if text is a valid number, otherwise returns null.  Rounds if
/// the text is a valid floating point number.
pub fn to_long(text: Option<&str>) -> Option<i64> {
    if is_empty(text) {
        return None;
    }
    let text = text.unwrap();
    match java_lang_long_parse_long(text) {
        Ok(value) => Some(value),
        // catch (NumberFormatException e) - `e` itself is never printed.
        Err(_) => match java_lang_double_value_of(text) {
            Ok(value) => Some(java_lang_math_round(value)),
            Err(message) => {
                // `e0.printStackTrace()`.  A Java stack trace is a property of the JVM,
                // not of the program, so only the exception's first line is written to
                // stderr where the source writes the whole trace.
                eprintln!("java.lang.NumberFormatException: {}", message);
                None
            }
        },
    }
}

/// Java `toInteger(String)`.
///
/// Returns text as a Integer if text is a valid integer, otherwise returns null.
pub fn to_integer(text: Option<&str>) -> Option<i32> {
    to_integer_with_round(text, false)
}

/// Java `toInteger(String, boolean)`.
///
/// Returns text as a Integer if text is a valid number, otherwise returns null.  Rounds
/// if the round parameter is true, and the text is a valid floating point number.
pub fn to_integer_with_round(text: Option<&str>, round: bool) -> Option<i32> {
    if is_empty(text) {
        return None;
    }
    let text = text.unwrap();
    match java_lang_integer_parse_int(text) {
        Ok(value) => Some(value),
        Err(message) => {
            if !round {
                // `e.printStackTrace()`; see `to_long`.
                eprintln!("java.lang.NumberFormatException: {}", message);
                return None;
            }
            match java_lang_float_parse_float(text) {
                Ok(value) => Some(java_lang_math_round_float(value)),
                Err(message) => {
                    // `e0.printStackTrace()`; see `to_long`.
                    eprintln!("java.lang.NumberFormatException: {}", message);
                    None
                }
            }
        }
    }
}

/// Java `toInteger(Long, boolean)`.
///
/// Converts a long to an integer.  If the long is too large, returns null and prints an
/// IllegalArgumentException message and stack trace.
pub fn to_integer_from_long(l_number: Option<i64>, suppress_err_log: bool) -> Option<i32> {
    let l_number = match l_number {
        None => return None,
        Some(l_number) => l_number,
    };
    if l_number < i32::MIN as i64 || l_number > i32::MAX as i64 {
        if !suppress_err_log {
            // `new IllegalArgumentException(...).printStackTrace()`; see `to_long`.
            eprintln!(
                "java.lang.IllegalArgumentException: {} cannot be converted to Integer without changing its value.",
                l_number
            );
        }
        return None;
    }
    // `lNumber.intValue()`, the Java narrowing primitive conversion.
    Some(l_number as i32)
}

/// Java `toDouble`.
pub fn to_double(text: Option<&str>) -> Option<f64> {
    if is_empty(text) {
        return None;
    }
    match java_lang_double_value_of(text.unwrap()) {
        Ok(value) => Some(value),
        Err(message) => {
            // `e.printStackTrace()`; see `to_long`.
            eprintln!("java.lang.NumberFormatException: {}", message);
            None
        }
    }
}

/// Java `toDoubleArray(String, FieldType)`.  A Java `Double[]` element may be null, so
/// the element type is `Option<f64>`.
pub fn to_double_array(
    text: Option<&str>,
    field_type: Option<FieldType>,
) -> Option<Vec<Option<f64>>> {
    if is_empty(text) {
        return None;
    }
    let field_type = match field_type {
        None => FieldType::FloatingPointArray,
        Some(field_type) => field_type,
    };
    // `text.split(fieldType.getSplitter())`.  The splitter is a Java regular expression
    // whose `\s` means the five ASCII characters `[ \t\n\x0B\f\r]`, while the Rust
    // regex crate's `\s` is Unicode whitespace, so that one class is rewritten.
    let string_array = java_lang_string_split(
        text.unwrap(),
        &Regex::new(
            &field_type
                .get_splitter()
                .replace("\\s", "[ \t\n\u{0B}\u{0C}\r]"),
        )
        .unwrap(),
    );
    let mut double_array: Vec<Option<f64>> = Vec::with_capacity(string_array.len());
    for item in &string_array {
        double_array.push(to_double(Some(item)));
    }
    Some(double_array)
}

/// Java `toArray(List<String>)`.  A Java `List<String>` element may be null, so the
/// element type is `Option<String>`.
pub fn to_array(list: Option<&[Option<String>]>) -> Option<Vec<Option<String>>> {
    let list = match list {
        None => return None,
        Some(list) => list,
    };
    let size = list.len();
    let mut array: Option<Vec<Option<String>>> = None;
    if size == 1 {
        array = Some(vec![list[0].clone()]);
    } else if size > 1 {
        array = Some(list.to_vec());
    }
    array
}
